"""Regularized and ensemble Stage 2 FX models with walk-forward optimization."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import DMatrix, XGBRegressor


cwd = Path.cwd()
if (cwd / "src").exists():
    project_root = cwd
elif (cwd.parent / "src").exists():
    project_root = cwd.parent
else:
    raise FileNotFoundError("Could not locate project root containing 'src'.")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


DEFAULT_PROCESSED_DIR = project_root / "data" / "processed"
DEFAULT_OUTPUT_DIR = project_root / "data" / "model_outputs"
DEFAULT_MODELS = ("ols", "ridge", "lasso", "elasticnet", "xgb", "lgbm")
LINEAR_MODELS = {"ols", "ridge", "lasso", "elasticnet"}
TREE_MODELS = {"xgb", "lgbm"}
MODEL_LABELS = {
    "ols": "OLS",
    "ridge": "RidgeCV",
    "lasso": "LassoCV",
    "elasticnet": "ElasticNetCV",
    "xgb": "XGBRegressor",
    "lgbm": "LGBMRegressor",
}
MODEL_FEATURE_SUFFIX = {
    "ols": "raw",
    "ridge": "std",
    "lasso": "std",
    "elasticnet": "std",
    "xgb": "raw",
    "lgbm": "raw",
}
LINEAR_ALPHA_GRID = np.logspace(-4, 2, 12)
ELASTICNET_L1_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
TREE_PARAM_GRID = {
    "n_estimators": (200, 400),
    "max_depth": (2, 3),
    "learning_rate": (0.03, 0.1),
    "subsample": (0.8, 1.0),
}


def _sanitize_column_name(name: str) -> str:
    """Convert a human-readable driver name into the saved snake_case form."""
    clean = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean.lower()


def _compute_days_in_signal(error_z: pd.Series, threshold: float = 2.0) -> pd.Series:
    """Count consecutive days where z-score stays above +threshold or below -threshold."""
    regime = pd.Series(0, index=error_z.index, dtype=int)
    regime = regime.where(~(error_z > threshold), 1)
    regime = regime.where(~(error_z < -threshold), -1)

    days = pd.Series(0, index=error_z.index, dtype=int)
    run = 0
    prev_regime = 0
    for dt, current_regime in regime.items():
        if current_regime == 0:
            run = 0
            prev_regime = 0
        elif current_regime == prev_regime:
            run += 1
        else:
            run = 1
            prev_regime = current_regime
        days.at[dt] = run

    return days


def _adjusted_r2(r2: float, n_obs: int, n_features: int) -> float:
    """Return adjusted R^2, or NaN when undefined."""
    if n_obs <= n_features + 1:
        return np.nan
    return 1 - (1 - r2) * ((n_obs - 1) / (n_obs - n_features - 1))


def _rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Return root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_master_training_data(
    currency: str,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """Load one saved master CSV and return the full modeling frame."""
    csv_path = Path(processed_dir) / f"{currency.lower()}_master.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} does not exist. Build the master CSVs before running Stage 2 ML models."
        )

    df = pd.read_csv(csv_path, index_col="Date", parse_dates=["Date"]).sort_index()
    df = df.loc[df.index.notna()].copy()

    required_cols = [
        "Actual_Price",
        "Log_Return",
        "driver_1_name",
        "driver_2_name",
        "driver_3_name",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required master CSV columns for {currency}: {missing}")

    return df


def _active_feature_names(
    df: pd.DataFrame,
    current_date: pd.Timestamp,
    feature_suffix: str,
) -> tuple[list[str], list[str]]:
    """Return human-readable driver names and the aligned feature columns for this date."""
    top_names = [
        df.at[current_date, "driver_1_name"],
        df.at[current_date, "driver_2_name"],
        df.at[current_date, "driver_3_name"],
    ]

    active_names: list[str] = []
    active_cols: list[str] = []
    for name in top_names:
        if pd.isna(name):
            continue
        feature_col = f"{_sanitize_column_name(str(name))}_{feature_suffix}"
        if feature_col in df.columns and pd.notna(df.at[current_date, feature_col]):
            if feature_col not in active_cols:
                active_cols.append(feature_col)
                active_names.append(str(name))

    return active_names, active_cols


def _effective_tscv(n_obs: int, requested_splits: int) -> TimeSeriesSplit | None:
    """Return a feasible TimeSeriesSplit for the sample size, or None if impossible."""
    max_splits = min(requested_splits, max(0, n_obs - 1))
    if max_splits < 2:
        return None
    return TimeSeriesSplit(n_splits=max_splits)


def _cross_validated_rmse(
    model_name: str,
    params: Dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: TimeSeriesSplit | None,
    random_state: int = 42,
    early_stopping_rounds: int = 25,
) -> float:
    """Estimate validation RMSE for the provided hyperparameters."""
    if cv is None:
        return np.nan

    fold_scores: list[float] = []
    for fold_train_idx, fold_val_idx in cv.split(x_train):
        x_fold_train = x_train.iloc[fold_train_idx]
        y_fold_train = y_train.iloc[fold_train_idx]
        x_fold_val = x_train.iloc[fold_val_idx]
        y_fold_val = y_train.iloc[fold_val_idx]

        model = _fit_final_model(
            model_name=model_name,
            x_train=x_fold_train,
            y_train=y_fold_train,
            params=params,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            eval_set=(x_fold_val, y_fold_val),
        )
        pred_val = model.predict(x_fold_val)
        fold_scores.append(_rmse(y_fold_val, pred_val))

    return float(np.mean(fold_scores)) if fold_scores else np.nan


def _build_linear_tuner(
    model_name: str,
    cv: TimeSeriesSplit | None,
    random_state: int = 42,
):
    """Create the requested linear-model CV tuner."""
    if model_name == "ridge":
        return RidgeCV(
            alphas=LINEAR_ALPHA_GRID,
            fit_intercept=True,
            scoring="neg_mean_squared_error",
            cv=cv,
        )
    if model_name == "lasso":
        return LassoCV(
            alphas=LINEAR_ALPHA_GRID,
            fit_intercept=True,
            cv=cv,
            max_iter=5000,
            random_state=random_state,
            n_jobs=1,
        )
    if model_name == "elasticnet":
        return ElasticNetCV(
            alphas=LINEAR_ALPHA_GRID,
            l1_ratio=ELASTICNET_L1_GRID,
            fit_intercept=True,
            cv=cv,
            max_iter=5000,
            random_state=random_state,
            n_jobs=1,
        )
    raise ValueError(f"{model_name} is not a regularized linear model.")


def _tune_linear_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int,
    random_state: int = 42,
) -> tuple[Dict[str, Any], float]:
    """Tune one linear model with TimeSeriesSplit."""
    cv = _effective_tscv(len(x_train), cv_splits)
    if cv is None:
        return {}, np.nan

    tuner = _build_linear_tuner(model_name=model_name, cv=cv, random_state=random_state)
    tuner.fit(x_train, y_train)

    if model_name == "ridge":
        params: Dict[str, Any] = {"alpha": float(tuner.alpha_)}
    elif model_name == "lasso":
        params = {"alpha": float(tuner.alpha_)}
    else:
        params = {
            "alpha": float(tuner.alpha_),
            "l1_ratio": float(tuner.l1_ratio_),
        }

    validation_rmse = _cross_validated_rmse(
        model_name=model_name,
        params=params,
        x_train=x_train,
        y_train=y_train,
        cv=cv,
        random_state=random_state,
    )
    return params, validation_rmse


def _iter_tree_param_grid() -> Iterable[Dict[str, Any]]:
    """Yield all tree-model hyperparameter combinations."""
    keys = list(TREE_PARAM_GRID.keys())
    for values in product(*(TREE_PARAM_GRID[key] for key in keys)):
        yield dict(zip(keys, values))


def _split_train_eval(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    min_eval_size: int = 20,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None, pd.Series | None]:
    """Split the most recent block off as the evaluation set when feasible."""
    if len(x_train) < (min_eval_size * 2):
        return x_train, y_train, None, None

    eval_size = max(min_eval_size, len(x_train) // 5)
    if eval_size >= len(x_train):
        return x_train, y_train, None, None

    split_point = len(x_train) - eval_size
    return (
        x_train.iloc[:split_point],
        y_train.iloc[:split_point],
        x_train.iloc[split_point:],
        y_train.iloc[split_point:],
    )


def _build_tree_model(
    model_name: str,
    params: Dict[str, Any],
    random_state: int = 42,
    early_stopping_rounds: int = 25,
):
    """Create the requested tree model."""
    common_params = {
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "subsample": params["subsample"],
        "random_state": random_state,
    }

    if model_name == "xgb":
        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            colsample_bytree=1.0,
            reg_lambda=1.0,
            min_child_weight=1.0,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="rmse",
            **common_params,
        )
    if model_name == "lgbm":
        return lgb.LGBMRegressor(
            objective="regression",
            num_leaves=31,
            min_child_samples=10,
            verbosity=-1,
            **common_params,
        )
    raise ValueError(f"{model_name} is not a tree model.")


def _fit_tree_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    random_state: int = 42,
    early_stopping_rounds: int = 25,
    eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
):
    """Fit one tree model, using early stopping when an eval set is available."""
    model = _build_tree_model(
        model_name=model_name,
        params=params,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )

    if eval_set is None:
        x_fit, y_fit, x_eval, y_eval = _split_train_eval(x_train, y_train)
    else:
        x_fit, y_fit = x_train, y_train
        x_eval, y_eval = eval_set

    if x_eval is None or y_eval is None:
        model.fit(x_train, y_train, verbose=False)
        return model

    if model_name == "xgb":
        model.fit(
            x_fit,
            y_fit,
            eval_set=[(x_eval, y_eval)],
            verbose=False,
        )
    else:
        model.fit(
            x_fit,
            y_fit,
            eval_set=[(x_eval, y_eval)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
    return model


def _tune_tree_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int,
    random_state: int = 42,
    early_stopping_rounds: int = 25,
) -> tuple[Dict[str, Any], float]:
    """Tune one tree model with time-series validation."""
    cv = _effective_tscv(len(x_train), cv_splits)
    if cv is None:
        params = next(iter(_iter_tree_param_grid()))
        return params, np.nan

    best_params: Dict[str, Any] | None = None
    best_score = np.inf

    for params in _iter_tree_param_grid():
        fold_scores: list[float] = []
        for fold_train_idx, fold_val_idx in cv.split(x_train):
            x_fold_train = x_train.iloc[fold_train_idx]
            y_fold_train = y_train.iloc[fold_train_idx]
            x_fold_val = x_train.iloc[fold_val_idx]
            y_fold_val = y_train.iloc[fold_val_idx]

            model = _fit_tree_model(
                model_name=model_name,
                x_train=x_fold_train,
                y_train=y_fold_train,
                params=params,
                random_state=random_state,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=(x_fold_val, y_fold_val),
            )
            pred_val = model.predict(x_fold_val)
            fold_scores.append(_rmse(y_fold_val, pred_val))

        score = float(np.mean(fold_scores)) if fold_scores else np.inf
        if score < best_score:
            best_score = score
            best_params = params

    if best_params is None:
        best_params = next(iter(_iter_tree_param_grid()))
        best_score = np.nan
    return best_params, best_score


def _fit_final_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    random_state: int = 42,
    early_stopping_rounds: int = 25,
    eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
):
    """Fit the requested model using already-selected hyperparameters."""
    if model_name == "ols":
        model = LinearRegression(fit_intercept=True)
        model.fit(x_train, y_train)
        return model
    if model_name == "ridge":
        model = Ridge(alpha=params["alpha"], fit_intercept=True, random_state=random_state)
        model.fit(x_train, y_train)
        return model
    if model_name == "lasso":
        model = Lasso(
            alpha=params["alpha"],
            fit_intercept=True,
            max_iter=5000,
            random_state=random_state,
        )
        model.fit(x_train, y_train)
        return model
    if model_name == "elasticnet":
        model = ElasticNet(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            fit_intercept=True,
            max_iter=5000,
            random_state=random_state,
        )
        model.fit(x_train, y_train)
        return model
    if model_name in TREE_MODELS:
        return _fit_tree_model(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            params=params,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            eval_set=eval_set,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def _select_model_params(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int,
    random_state: int = 42,
    early_stopping_rounds: int = 25,
) -> tuple[Dict[str, Any], float]:
    """Tune one model and return the selected hyperparameters plus validation RMSE."""
    if model_name == "ols":
        validation_rmse = _cross_validated_rmse(
            model_name=model_name,
            params={},
            x_train=x_train,
            y_train=y_train,
            cv=_effective_tscv(len(x_train), cv_splits),
            random_state=random_state,
        )
        return {}, validation_rmse
    if model_name in {"ridge", "lasso", "elasticnet"}:
        return _tune_linear_model(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            cv_splits=cv_splits,
            random_state=random_state,
        )
    return _tune_tree_model(
        model_name=model_name,
        x_train=x_train,
        y_train=y_train,
        cv_splits=cv_splits,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )


def _tree_feature_contributions(
    model_name: str,
    model,
    x_now: pd.DataFrame,
) -> Dict[str, float]:
    """Return tree-model feature contributions for the current observation."""
    if model_name == "xgb":
        contribs = model.get_booster().predict(
            DMatrix(x_now, feature_names=list(x_now.columns)),
            pred_contribs=True,
        )[0]
    elif model_name == "lgbm":
        contribs = np.asarray(model.predict(x_now, pred_contrib=True))[0]
    else:
        return {}

    feature_values = contribs[:-1]
    return {
        feature_name: float(feature_value)
        for feature_name, feature_value in zip(x_now.columns, feature_values)
    }


def _pad_driver_values(
    active_names: Sequence[str],
    values_by_feature: Dict[str, float],
    feature_suffix: str,
) -> tuple[list[Any], list[Any]]:
    """Align driver names and attribution values into three ordered slots."""
    names: list[Any] = []
    values: list[Any] = []
    for name in active_names:
        feature_col = f"{_sanitize_column_name(name)}_{feature_suffix}"
        names.append(name)
        values.append(float(values_by_feature.get(feature_col, np.nan)))

    while len(names) < 3:
        names.append(np.nan)
        values.append(np.nan)
    return names[:3], values[:3]


def _model_intercept_value(model) -> float:
    """Return a scalar intercept when the estimator exposes one."""
    if not hasattr(model, "intercept_"):
        return np.nan
    intercept = np.asarray(model.intercept_).reshape(-1)
    if intercept.size == 0:
        return np.nan
    return float(intercept[0])


def run_stage2_model(
    currency: str,
    model_name: str = "ols",
    window: int = 250,
    error_sum_window: int = 10,
    z_window: int | None = None,
    recenter_window: int = 60,
    return_scale: float = 100.0,
    cv_splits: int = 4,
    retune_frequency: int = 60,
    early_stopping_rounds: int = 25,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    start_date: str | pd.Timestamp | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run one Stage 2 model with walk-forward training and optional tuning."""
    model_name = model_name.lower()
    if model_name not in MODEL_FEATURE_SUFFIX:
        raise ValueError(f"Unsupported model_name: {model_name}")
    if window <= 0:
        raise ValueError("window must be a positive integer.")
    if recenter_window <= 0:
        raise ValueError("recenter_window must be a positive integer.")
    if retune_frequency <= 0:
        raise ValueError("retune_frequency must be a positive integer.")

    df = load_master_training_data(currency=currency, processed_dir=processed_dir)
    feature_suffix = MODEL_FEATURE_SUFFIX[model_name]
    signal_window = recenter_window if z_window is None else z_window
    if signal_window <= 0:
        raise ValueError("z_window must be a positive integer when provided.")

    results = []
    fv_anchor = float(df["Actual_Price"].iloc[0])
    last_recenter_i = 0
    last_tuned_i: int | None = None
    selected_params: Dict[str, Any] = {}
    validation_rmse = np.nan
    retuned = False

    for i in range(window, len(df)):
        current_date = df.index[i]
        active_names, active_cols = _active_feature_names(
            df=df,
            current_date=current_date,
            feature_suffix=feature_suffix,
        )
        if not active_cols:
            continue

        subset_cols = ["Log_Return", *active_cols]
        lookback_df = df.iloc[i - window : i].loc[:, subset_cols].dropna()
        if len(lookback_df) < len(active_cols) + 20:
            continue

        x_train = lookback_df[active_cols]
        y_train = lookback_df["Log_Return"]
        x_now = df.loc[[current_date], active_cols]

        if x_now.isna().any(axis=None):
            continue

        actual_log_change = df.at[current_date, "Log_Return"]
        current_price = df.at[current_date, "Actual_Price"]
        if pd.isna(actual_log_change) or pd.isna(current_price):
            continue

        if model_name != "ols":
            should_retune = last_tuned_i is None or (i - last_tuned_i) >= retune_frequency
            if should_retune:
                selected_params, validation_rmse = _select_model_params(
                    model_name=model_name,
                    x_train=x_train,
                    y_train=y_train,
                    cv_splits=cv_splits,
                    random_state=random_state,
                    early_stopping_rounds=early_stopping_rounds,
                )
                last_tuned_i = i
                retuned = True
            else:
                retuned = False
        else:
            selected_params, validation_rmse = _select_model_params(
                model_name=model_name,
                x_train=x_train,
                y_train=y_train,
                cv_splits=cv_splits,
                random_state=random_state,
                early_stopping_rounds=early_stopping_rounds,
            )
            retuned = True

        recentered = False
        if (i - last_recenter_i) >= recenter_window:
            fv_anchor = float(current_price)
            last_recenter_i = i
            recentered = True

        model = _fit_final_model(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            params=selected_params,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
        )

        pred_log_change = float(model.predict(x_now)[0])
        fv_anchor = float(fv_anchor * np.exp(pred_log_change / return_scale))
        error = float(actual_log_change - pred_log_change)

        fitted_train = model.predict(x_train)
        residuals = y_train - fitted_train
        n_obs = len(x_train)
        n_features = len(active_cols)
        r2 = float(model.score(x_train, y_train))
        adj_r2 = _adjusted_r2(r2, n_obs, n_features)
        train_rmse = _rmse(y_train, fitted_train)
        residual_std = float(pd.Series(residuals).std(ddof=1)) if len(residuals) > 1 else np.nan

        linear_values: Dict[str, float] = {}
        shap_values: Dict[str, float] = {}
        if hasattr(model, "coef_"):
            linear_values = {
                feature_name: float(beta)
                for feature_name, beta in zip(active_cols, np.asarray(model.coef_))
            }
        elif model_name in TREE_MODELS:
            shap_values = _tree_feature_contributions(model_name=model_name, model=model, x_now=x_now)

        driver_names, beta_values = _pad_driver_values(
            active_names=active_names,
            values_by_feature=linear_values,
            feature_suffix=feature_suffix,
        )
        _, shap_contribs = _pad_driver_values(
            active_names=active_names,
            values_by_feature=shap_values,
            feature_suffix=feature_suffix,
        )

        results.append(
            {
                "Date": current_date,
                "Model_Name": model_name,
                "Actual_Price": float(current_price),
                "Macro_Anchor_Price": fv_anchor,
                "Fair_Value_Price": fv_anchor,
                "Actual_Ret": float(actual_log_change),
                "Pred_Ret": pred_log_change,
                "Error": error,
                "R2": r2,
                "Adj_R2": adj_r2,
                "Train_RMSE": train_rmse,
                "Validation_RMSE": validation_rmse,
                "Residual_STD": residual_std,
                "Window_Obs": n_obs,
                "Drivers_Used_Count": n_features,
                "Drivers_Used": ", ".join(active_names),
                "Intercept": _model_intercept_value(model),
                "Driver_1_Name": driver_names[0],
                "Driver_2_Name": driver_names[1],
                "Driver_3_Name": driver_names[2],
                "Driver_1_Beta": beta_values[0],
                "Driver_2_Beta": beta_values[1],
                "Driver_3_Beta": beta_values[2],
                "Driver_1_SHAP": shap_contribs[0],
                "Driver_2_SHAP": shap_contribs[1],
                "Driver_3_SHAP": shap_contribs[2],
                "Attribution_Method": "SHAP" if model_name in TREE_MODELS else "BETA",
                "Best_Params": json.dumps(selected_params, sort_keys=True),
                "Alpha": float(selected_params["alpha"]) if "alpha" in selected_params else np.nan,
                "L1_Ratio": float(selected_params["l1_ratio"]) if "l1_ratio" in selected_params else np.nan,
                "N_Estimators": int(selected_params["n_estimators"])
                if "n_estimators" in selected_params
                else np.nan,
                "Max_Depth": int(selected_params["max_depth"])
                if "max_depth" in selected_params
                else np.nan,
                "Learning_Rate": float(selected_params["learning_rate"])
                if "learning_rate" in selected_params
                else np.nan,
                "Subsample": float(selected_params["subsample"])
                if "subsample" in selected_params
                else np.nan,
                "Retuned": retuned,
                "Recenter_Event": recentered,
            }
        )

    if not results:
        raise ValueError(f"No Stage 2 rows were generated for {currency} using {model_name}.")

    res_df = pd.DataFrame(results).set_index("Date").sort_index()
    res_df["Cum_Error"] = res_df["Error"].ewm(
        span=error_sum_window,
        adjust=False,
        min_periods=error_sum_window,
    ).mean()
    res_df["Macro_Gap"] = res_df["Actual_Price"] - res_df["Macro_Anchor_Price"]
    res_df["Signal_Z"] = (
        res_df["Macro_Gap"] - res_df["Macro_Gap"].rolling(signal_window).mean()
    ) / res_df["Macro_Gap"].rolling(signal_window).std()
    res_df["Error_Z"] = res_df["Signal_Z"]
    res_df["Signal"] = np.select(
        [res_df["Signal_Z"] > 2.0, res_df["Signal_Z"] < -2.0],
        ["SELL", "BUY"],
        default="NEUTRAL",
    )
    res_df["Days_In_Signal"] = _compute_days_in_signal(res_df["Signal_Z"], threshold=2.0)

    if start_date is not None:
        start_date = pd.Timestamp(start_date)
        res_df = res_df.loc[res_df.index >= start_date].copy()

    if res_df.empty:
        raise ValueError(
            f"No Stage 2 rows remain for {currency} using {model_name} after start_date filter."
        )

    return res_df


def run_stage2_model_suite(
    currency: str,
    models: Sequence[str] = DEFAULT_MODELS,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Run the requested model suite and return one result DataFrame per model."""
    outputs: Dict[str, pd.DataFrame] = {}
    for model_name in models:
        outputs[model_name.lower()] = run_stage2_model(
            currency=currency,
            model_name=model_name.lower(),
            **kwargs,
        )
    return outputs


def save_stage2_model_suite(
    currency: str,
    models: Sequence[str] = DEFAULT_MODELS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    **kwargs,
) -> Dict[str, Path]:
    """Run the model suite and save one CSV per model."""
    output_path = Path(output_dir) / currency.lower()
    output_path.mkdir(parents=True, exist_ok=True)

    outputs = run_stage2_model_suite(currency=currency, models=models, **kwargs)
    saved_paths: Dict[str, Path] = {}
    for model_name, df in outputs.items():
        csv_path = output_path / f"stage2_{model_name}_results.csv"
        df.to_csv(csv_path, index=True)
        saved_paths[model_name] = csv_path
    return saved_paths


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create a CLI for the Stage 2 ML model runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("currency", help="Currency code, for example eur or gbp.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Models to run.",
    )
    parser.add_argument("--window", type=int, default=250, help="Rolling training window.")
    parser.add_argument(
        "--error-sum-window",
        type=int,
        default=10,
        help="EWMA span for the cumulative error smoothing.",
    )
    parser.add_argument(
        "--z-window",
        type=int,
        default=None,
        help="Rolling window for Signal_Z. Defaults to recenter_window.",
    )
    parser.add_argument(
        "--recenter-window",
        type=int,
        default=60,
        help="Days between fair-value anchor recenters.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=4,
        help="TimeSeriesSplit folds used during tuning.",
    )
    parser.add_argument(
        "--retune-frequency",
        type=int,
        default=60,
        help="Days between hyperparameter retunes in the walk-forward loop.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=25,
        help="Early stopping rounds for the tree models.",
    )
    parser.add_argument(
        "--processed-dir",
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing the saved master CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where model CSVs should be saved.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional YYYY-MM-DD start date filter.",
    )
    return parser


def main() -> None:
    """Run the CLI and print the saved output paths."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    saved_paths = save_stage2_model_suite(
        currency=args.currency,
        models=args.models,
        window=args.window,
        error_sum_window=args.error_sum_window,
        z_window=args.z_window,
        recenter_window=args.recenter_window,
        cv_splits=args.cv_splits,
        retune_frequency=args.retune_frequency,
        early_stopping_rounds=args.early_stopping_rounds,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
    )

    print("\nSaved Stage 2 model outputs:")
    for model_name, path in saved_paths.items():
        print(f"- {MODEL_LABELS.get(model_name, model_name)}: {path}")


if __name__ == "__main__":
    main()
