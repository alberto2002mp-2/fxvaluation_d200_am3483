"""Generate a Machine Learning Performance Audit for Stage 2 fair value signals."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


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
DEFAULT_AUDIT_DIR = project_root / "data" / "audits"
DEFAULT_CURRENCIES = ("eur", "gbp", "aud", "nzd", "cad", "jpy", "chf", "nok", "sek")


from src.stage2_ml_models import DEFAULT_MODELS, MODEL_LABELS, run_stage2_model_suite


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


def _max_drawdown(equity_curve: pd.Series) -> float:
    """Return the maximum drawdown from an equity curve."""
    clean_curve = equity_curve.dropna()
    if clean_curve.empty:
        return np.nan
    running_peak = clean_curve.cummax()
    drawdown = clean_curve / running_peak - 1.0
    return float(drawdown.min())


def _annualized_sharpe_ratio(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Return the annualized Sharpe ratio of a daily return series."""
    clean = daily_returns.dropna()
    if clean.empty:
        return np.nan

    volatility = float(clean.std(ddof=1))
    if np.isclose(volatility, 0.0):
        return np.nan

    mean_return = float(clean.mean())
    return mean_return / volatility * np.sqrt(periods_per_year)


def _rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Return root mean squared error."""
    return float(np.sqrt(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))))


def _largest_beta_driver(row: pd.Series) -> pd.Series:
    """Return the driver with the largest absolute beta on this date."""
    candidates = []
    for idx in range(1, 4):
        name = row.get(f"Driver_{idx}_Name")
        beta = row.get(f"Driver_{idx}_Beta")
        if pd.isna(name) or pd.isna(beta):
            continue
        candidates.append((str(name), float(beta)))

    if not candidates:
        return pd.Series(
            {
                "Largest_Beta_Driver": np.nan,
                "Largest_Beta": np.nan,
                "Largest_Beta_Abs": np.nan,
            }
        )

    driver_name, beta = max(candidates, key=lambda item: abs(item[1]))
    return pd.Series(
        {
            "Largest_Beta_Driver": driver_name,
            "Largest_Beta": beta,
            "Largest_Beta_Abs": abs(beta),
        }
    )


def _largest_model_attribution(row: pd.Series) -> pd.Series:
    """Return the dominant attribution for either linear betas or tree SHAP values."""
    use_shap = False
    if row.get("Attribution_Method") == "SHAP":
        use_shap = True
    else:
        shap_values = [row.get("Driver_1_SHAP"), row.get("Driver_2_SHAP"), row.get("Driver_3_SHAP")]
        use_shap = any(pd.notna(value) for value in shap_values)

    value_cols = (
        ("Driver_1_SHAP", "Driver_2_SHAP", "Driver_3_SHAP")
        if use_shap
        else ("Driver_1_Beta", "Driver_2_Beta", "Driver_3_Beta")
    )
    method = "SHAP" if use_shap else "BETA"

    candidates = []
    for idx, value_col in enumerate(value_cols, start=1):
        name = row.get(f"Driver_{idx}_Name")
        value = row.get(value_col)
        if pd.isna(name) or pd.isna(value):
            continue
        candidates.append((str(name), float(value)))

    if not candidates:
        return pd.Series(
            {
                "Largest_Attribution_Driver": np.nan,
                "Largest_Attribution_Value": np.nan,
                "Largest_Attribution_Abs": np.nan,
                "Attribution_Method": method,
            }
        )

    driver_name, value = max(candidates, key=lambda item: abs(item[1]))
    return pd.Series(
        {
            "Largest_Attribution_Driver": driver_name,
            "Largest_Attribution_Value": value,
            "Largest_Attribution_Abs": abs(value),
            "Attribution_Method": method,
        }
    )


def _get_model_ready_data(
    currency: str,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """Load the saved master CSV and return the raw-feature Stage 2 inputs."""
    csv_path = Path(processed_dir) / f"{currency.lower()}_master.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} does not exist. Build the master CSVs before running the audit."
        )

    df = pd.read_csv(csv_path, index_col="Date", parse_dates=["Date"]).sort_index()
    df = df.loc[df.index.notna()].copy()

    required_cols = ["Actual_Price", "Log_Return"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required master CSV columns for {currency}: {missing}")

    feature_cols = [col for col in df.columns if col.endswith("_raw")]
    out = df.loc[:, required_cols + feature_cols].copy()

    top_aliases = {
        "driver_1_name": ["driver_1_name", "Driver 1 Name"],
        "driver_2_name": ["driver_2_name", "Driver 2 Name"],
        "driver_3_name": ["driver_3_name", "Driver 3 Name"],
        "driver_1_beta_z": ["driver_1_beta_z", "Driver 1 Beta Z"],
        "driver_2_beta_z": ["driver_2_beta_z", "Driver 2 Beta Z"],
        "driver_3_beta_z": ["driver_3_beta_z", "Driver 3 Beta Z"],
        "driver_1_normal_beta": ["driver_1_normal_beta", "Driver 1 Normal Beta"],
        "driver_2_normal_beta": ["driver_2_normal_beta", "Driver 2 Normal Beta"],
        "driver_3_normal_beta": ["driver_3_normal_beta", "Driver 3 Normal Beta"],
    }
    for alias_col, stored_candidates in top_aliases.items():
        for stored_col in stored_candidates:
            if stored_col in df.columns:
                out[alias_col] = df[stored_col]
                break

    return out


def build_stage2_audit_dataset(
    currency: str,
    window: int = 50,
    error_sum_window: int = 10,
    z_window: int | None = None,
    recenter_window: int = 60,
    forward_days: int = 10,
    threshold: float = 2.0,
    return_scale: float = 100.0,
    sharpe_window: int = 252,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Rebuild the Stage 2 rolling model and return a signal-level audit dataset.

    The signal trigger is defined as the first day in a BUY/SELL regime
    (`Days_In_Signal == 1`). This avoids double-counting multi-day signal runs.
    """
    df = _get_model_ready_data(currency=currency, processed_dir=processed_dir).copy()
    df = df.sort_index()

    required_cols = [
        "Actual_Price",
        "Log_Return",
        "driver_1_name",
        "driver_2_name",
        "driver_3_name",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for Stage 2 audit: {missing}")

    if recenter_window <= 0:
        raise ValueError("recenter_window must be a positive integer.")
    if forward_days <= 0:
        raise ValueError("forward_days must be a positive integer.")

    signal_window = recenter_window if z_window is None else z_window
    if signal_window <= 0:
        raise ValueError("z_window must be a positive integer when provided.")

    results = []
    fv_anchor = float(df["Actual_Price"].iloc[0])
    last_recenter_i = 0

    for i in range(window, len(df)):
        current_date = df.index[i]
        top_names = [
            df.at[current_date, "driver_1_name"],
            df.at[current_date, "driver_2_name"],
            df.at[current_date, "driver_3_name"],
        ]

        active_driver_names = []
        active_driver_cols = []
        for name in top_names:
            if pd.isna(name):
                continue
            raw_col = f"{_sanitize_column_name(str(name))}_raw"
            if raw_col in df.columns and pd.notna(df.at[current_date, raw_col]):
                if raw_col not in active_driver_cols:
                    active_driver_cols.append(raw_col)
                    active_driver_names.append(str(name))

        if not active_driver_cols:
            continue

        subset_cols = ["Log_Return", *active_driver_cols]
        lookback_df = df.iloc[i - window : i].loc[:, subset_cols].dropna()
        if len(lookback_df) < len(active_driver_cols) + 2:
            continue

        x_train = lookback_df[active_driver_cols]
        y_train = lookback_df["Log_Return"]
        x_now = df.loc[[current_date], active_driver_cols]

        if x_now.isna().any(axis=None):
            continue

        actual_log_change = df.at[current_date, "Log_Return"]
        current_price = df.at[current_date, "Actual_Price"]
        if pd.isna(actual_log_change) or pd.isna(current_price):
            continue

        recentered = False
        if (i - last_recenter_i) >= recenter_window:
            fv_anchor = float(current_price)
            last_recenter_i = i
            recentered = True

        model = LinearRegression(fit_intercept=True)
        model.fit(x_train, y_train)

        pred_log_change = float(model.predict(x_now)[0])
        fv_anchor = float(fv_anchor * np.exp(pred_log_change / return_scale))
        error = float(actual_log_change - pred_log_change)

        fitted_train = model.predict(x_train)
        residuals = y_train - fitted_train
        n_obs = len(lookback_df)
        n_features = len(active_driver_cols)
        r2 = float(model.score(x_train, y_train))
        adj_r2 = _adjusted_r2(r2, n_obs, n_features)
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        residual_std = float(residuals.std(ddof=1)) if len(residuals) > 1 else np.nan

        driver_name_pad = active_driver_names + [np.nan] * (3 - len(active_driver_names))
        driver_beta_pad = [float(beta) for beta in model.coef_] + [np.nan] * (
            3 - len(active_driver_names)
        )

        results.append(
            {
                "Date": current_date,
                "Actual_Price": float(current_price),
                "Macro_Anchor_Price": fv_anchor,
                "Fair_Value_Price": fv_anchor,
                "Actual_Ret": float(actual_log_change),
                "Pred_Ret": pred_log_change,
                "Error": error,
                "R2": r2,
                "Adj_R2": adj_r2,
                "RMSE": rmse,
                "Residual_STD": residual_std,
                "Window_Obs": n_obs,
                "Drivers_Used_Count": n_features,
                "Drivers_Used": ", ".join(active_driver_names),
                "Intercept": float(model.intercept_),
                "Driver_1_Name": driver_name_pad[0],
                "Driver_2_Name": driver_name_pad[1],
                "Driver_3_Name": driver_name_pad[2],
                "Driver_1_Beta": driver_beta_pad[0],
                "Driver_2_Beta": driver_beta_pad[1],
                "Driver_3_Beta": driver_beta_pad[2],
                "Recenter_Event": recentered,
            }
        )

    if not results:
        raise ValueError(f"No Stage 2 audit rows were generated for {currency}.")

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
        [res_df["Signal_Z"] > threshold, res_df["Signal_Z"] < -threshold],
        ["SELL", "BUY"],
        default="NEUTRAL",
    )
    res_df["Days_In_Signal"] = _compute_days_in_signal(res_df["Signal_Z"], threshold=threshold)
    res_df["Signal_Triggered"] = (res_df["Signal"] != "NEUTRAL") & (
        res_df["Days_In_Signal"] == 1
    )

    if start_date is not None:
        start_date = pd.Timestamp(start_date)
        res_df = res_df.loc[res_df.index >= start_date].copy()

    if res_df.empty:
        raise ValueError(f"No Stage 2 audit rows remain for {currency} after start_date filter.")

    res_df["Forward_10d_Return"] = (
        res_df["Actual_Price"].shift(-forward_days) / res_df["Actual_Price"] - 1.0
    )

    hit_conditions = [
        (res_df["Signal"] == "BUY") & (res_df["Forward_10d_Return"] > 0),
        (res_df["Signal"] == "SELL") & (res_df["Forward_10d_Return"] < 0),
    ]
    res_df["Hit"] = pd.Series(np.select(hit_conditions, [True, True], default=False), index=res_df.index)
    res_df["Hit"] = res_df["Hit"].astype(object)
    res_df.loc[res_df["Signal"] == "NEUTRAL", "Hit"] = np.nan
    res_df.loc[res_df["Forward_10d_Return"].isna(), "Hit"] = np.nan

    driver_summary = res_df.apply(_largest_beta_driver, axis=1)
    res_df = pd.concat([res_df, driver_summary], axis=1)

    daily_returns = res_df["Actual_Price"].pct_change().fillna(0.0)
    position_map = {"BUY": 1.0, "SELL": -1.0, "NEUTRAL": 0.0}
    res_df["Position"] = res_df["Signal"].map(position_map).fillna(0.0)
    res_df["Strategy_Daily_Return"] = res_df["Position"].shift(1).fillna(0.0) * daily_returns
    res_df["Strategy_Equity_Curve"] = (1.0 + res_df["Strategy_Daily_Return"]).cumprod()
    rolling_mean = res_df["Strategy_Daily_Return"].rolling(
        window=sharpe_window,
        min_periods=sharpe_window,
    ).mean()
    rolling_std = res_df["Strategy_Daily_Return"].rolling(
        window=sharpe_window,
        min_periods=sharpe_window,
    ).std()
    res_df["Rolling_Sharpe"] = (rolling_mean / rolling_std) * np.sqrt(252)
    res_df["Strategy_Drawdown"] = (
        res_df["Strategy_Equity_Curve"] / res_df["Strategy_Equity_Curve"].cummax() - 1.0
    )

    return res_df


def _build_hit_summary(res_df: pd.DataFrame) -> pd.DataFrame:
    """Return the signal-trigger subset used for hit-rate reporting."""
    signal_df = res_df.loc[res_df["Signal_Triggered"]].copy()
    signal_df = signal_df.loc[
        signal_df["Signal"].isin(["BUY", "SELL"]) & signal_df["Forward_10d_Return"].notna()
    ]
    return signal_df


def build_stage2_ml_performance_audit(
    currency: str,
    window: int = 50,
    error_sum_window: int = 10,
    z_window: int | None = None,
    recenter_window: int = 60,
    forward_days: int = 10,
    threshold: float = 2.0,
    return_scale: float = 100.0,
    sharpe_window: int = 252,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    start_date: str | pd.Timestamp | None = None,
) -> Dict[str, pd.DataFrame]:
    """Build all requested Stage 2 ML performance audit outputs."""
    res_df = build_stage2_audit_dataset(
        currency=currency,
        window=window,
        error_sum_window=error_sum_window,
        z_window=z_window,
        recenter_window=recenter_window,
        forward_days=forward_days,
        threshold=threshold,
        return_scale=return_scale,
        sharpe_window=sharpe_window,
        processed_dir=processed_dir,
        start_date=start_date,
    )

    signal_df = _build_hit_summary(res_df)
    hit_df = signal_df.loc[signal_df["Hit"] == True].copy()

    ic_df = res_df.loc[res_df["Error_Z"].notna() & res_df["Forward_10d_Return"].notna()]
    info_coefficient = (
        float(ic_df["Error_Z"].corr(ic_df["Forward_10d_Return"]))
        if len(ic_df) >= 2
        else np.nan
    )

    summary = pd.DataFrame(
        [
            {
                "Currency": currency.upper(),
                "Observations": len(res_df),
                "Signal_Triggers": len(signal_df),
                "BUY_Triggers": int((signal_df["Signal"] == "BUY").sum()),
                "SELL_Triggers": int((signal_df["Signal"] == "SELL").sum()),
                "Hit_Count": int((signal_df["Hit"] == True).sum()),
                "Miss_Count": int((signal_df["Hit"] == False).sum()),
                "Average_Hit_Rate_Pct": float(signal_df["Hit"].mean() * 100.0)
                if not signal_df.empty
                else np.nan,
                "Information_Coefficient": info_coefficient,
                "Average_Adj_R2": float(res_df["Adj_R2"].dropna().mean())
                if res_df["Adj_R2"].notna().any()
                else np.nan,
                "Average_RMSE": float(res_df["RMSE"].dropna().mean())
                if res_df["RMSE"].notna().any()
                else np.nan,
                "Average_Residual_STD": float(res_df["Residual_STD"].dropna().mean())
                if res_df["Residual_STD"].notna().any()
                else np.nan,
                "Strategy_Total_Return_Pct": float(
                    (res_df["Strategy_Equity_Curve"].iloc[-1] - 1.0) * 100.0
                )
                if not res_df.empty
                else np.nan,
                "Strategy_Sharpe_Ratio": float(
                    _annualized_sharpe_ratio(res_df["Strategy_Daily_Return"])
                )
                if not res_df.empty
                else np.nan,
                "Average_Rolling_Sharpe": float(res_df["Rolling_Sharpe"].dropna().mean())
                if res_df["Rolling_Sharpe"].notna().any()
                else np.nan,
                "Maximum_Drawdown_Pct": float(
                    _max_drawdown(res_df["Strategy_Equity_Curve"]) * 100.0
                )
                if not res_df.empty
                else np.nan,
            }
        ]
    )

    interpretability_cols = [
        "Signal",
        "Forward_10d_Return",
        "Hit",
        "Largest_Beta_Driver",
        "Largest_Beta",
        "Largest_Beta_Abs",
        "Driver_1_Name",
        "Driver_1_Beta",
        "Driver_2_Name",
        "Driver_2_Beta",
        "Driver_3_Name",
        "Driver_3_Beta",
    ]
    interpretability_df = hit_df.loc[:, interpretability_cols].copy()
    interpretability_df.index.name = "Date"

    return {
        "summary": summary,
        "audit_dataset": res_df,
        "signal_triggers": signal_df,
        "interpretability_hits": interpretability_df,
    }


def _plot_generalization_metrics(
    audit_df: pd.DataFrame,
    currency: str,
    output_path: str | Path,
) -> None:
    """Save a time-series chart of rolling Adjusted R^2 and RMSE."""
    plot_df = audit_df.loc[audit_df["Adj_R2"].notna() | audit_df["RMSE"].notna()]
    if plot_df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(plot_df.index, plot_df["Adj_R2"], color="tab:green", linewidth=1.5)
    axes[0].axhline(0.0, color="grey", linestyle=":", linewidth=1.0)
    axes[0].set_title(f"{currency.upper()} Stage 2 Rolling Adjusted R^2")
    axes[0].set_ylabel("Adj R^2")

    axes[1].plot(plot_df.index, plot_df["RMSE"], color="tab:red", linewidth=1.5)
    axes[1].set_title(f"{currency.upper()} Stage 2 Rolling RMSE")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_strategy_equity_curve(
    audit_df: pd.DataFrame,
    currency: str,
    output_path: str | Path,
) -> None:
    """Save a time-series chart of the strategy equity curve."""
    plot_df = audit_df.loc[audit_df["Strategy_Equity_Curve"].notna()]
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        plot_df.index,
        plot_df["Strategy_Equity_Curve"],
        color="tab:blue",
        linewidth=1.6,
    )
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.0)
    ax.set_title(f"{currency.upper()} Stage 2 Strategy Equity Curve")
    ax.set_ylabel("Equity Curve")
    ax.set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def prepare_stage2_strategy_audit(
    stage2_df: pd.DataFrame,
    forward_days: int = 10,
    threshold: float = 2.0,
    sharpe_window: int = 252,
) -> pd.DataFrame:
    """Add strategy and audit fields to an existing Stage 2 result frame."""
    audit_df = stage2_df.copy().sort_index()

    if "Signal_Z" not in audit_df.columns and "Error_Z" in audit_df.columns:
        audit_df["Signal_Z"] = audit_df["Error_Z"]
    if "Error_Z" not in audit_df.columns and "Signal_Z" in audit_df.columns:
        audit_df["Error_Z"] = audit_df["Signal_Z"]
    if "Signal" not in audit_df.columns:
        audit_df["Signal"] = np.select(
            [audit_df["Signal_Z"] > threshold, audit_df["Signal_Z"] < -threshold],
            ["SELL", "BUY"],
            default="NEUTRAL",
        )
    if "Days_In_Signal" not in audit_df.columns:
        audit_df["Days_In_Signal"] = _compute_days_in_signal(audit_df["Signal_Z"], threshold=threshold)

    audit_df["Signal_Triggered"] = (audit_df["Signal"] != "NEUTRAL") & (
        audit_df["Days_In_Signal"] == 1
    )
    audit_df["Forward_10d_Return"] = (
        audit_df["Actual_Price"].shift(-forward_days) / audit_df["Actual_Price"] - 1.0
    )

    hit_conditions = [
        (audit_df["Signal"] == "BUY") & (audit_df["Forward_10d_Return"] > 0),
        (audit_df["Signal"] == "SELL") & (audit_df["Forward_10d_Return"] < 0),
    ]
    audit_df["Hit"] = pd.Series(
        np.select(hit_conditions, [True, True], default=False),
        index=audit_df.index,
    ).astype(object)
    audit_df.loc[audit_df["Signal"] == "NEUTRAL", "Hit"] = np.nan
    audit_df.loc[audit_df["Forward_10d_Return"].isna(), "Hit"] = np.nan

    attribution_summary = audit_df.apply(_largest_model_attribution, axis=1)
    audit_df = pd.concat([audit_df, attribution_summary], axis=1)

    daily_returns = audit_df["Actual_Price"].pct_change().fillna(0.0)
    position_map = {"BUY": 1.0, "SELL": -1.0, "NEUTRAL": 0.0}
    audit_df["Position"] = audit_df["Signal"].map(position_map).fillna(0.0)
    audit_df["Strategy_Daily_Return"] = audit_df["Position"].shift(1).fillna(0.0) * daily_returns
    audit_df["Strategy_Equity_Curve"] = (1.0 + audit_df["Strategy_Daily_Return"]).cumprod()

    rolling_mean = audit_df["Strategy_Daily_Return"].rolling(
        window=sharpe_window,
        min_periods=sharpe_window,
    ).mean()
    rolling_std = audit_df["Strategy_Daily_Return"].rolling(
        window=sharpe_window,
        min_periods=sharpe_window,
    ).std()
    audit_df["Rolling_Sharpe"] = (rolling_mean / rolling_std) * np.sqrt(252)
    audit_df["Strategy_Drawdown"] = (
        audit_df["Strategy_Equity_Curve"] / audit_df["Strategy_Equity_Curve"].cummax() - 1.0
    )

    return audit_df


def _build_interpretability_hits(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Return hit-level attribution details for one model."""
    signal_df = _build_hit_summary(audit_df)
    hit_df = signal_df.loc[signal_df["Hit"] == True].copy()

    cols = [
        "Signal",
        "Forward_10d_Return",
        "Hit",
        "Attribution_Method",
        "Largest_Attribution_Driver",
        "Largest_Attribution_Value",
        "Largest_Attribution_Abs",
        "Driver_1_Name",
        "Driver_1_Beta",
        "Driver_1_SHAP",
        "Driver_2_Name",
        "Driver_2_Beta",
        "Driver_2_SHAP",
        "Driver_3_Name",
        "Driver_3_Beta",
        "Driver_3_SHAP",
    ]
    out = hit_df.loc[:, [col for col in cols if col in hit_df.columns]].copy()
    out.index.name = "Date"
    return out


def _best_tuning_event(audit_df: pd.DataFrame) -> Dict[str, Any]:
    """Return the best tuning event ranked by validation RMSE."""
    if "Retuned" in audit_df.columns and audit_df["Retuned"].notna().any():
        tune_df = audit_df.loc[audit_df["Retuned"] == True].copy()
    else:
        tune_df = audit_df.copy()

    if "Validation_RMSE" in tune_df.columns:
        tune_df = tune_df.loc[tune_df["Validation_RMSE"].notna()].copy()
    if tune_df.empty:
        return {}

    best_row = tune_df.sort_values(by="Validation_RMSE", ascending=True).iloc[0]
    return {
        "Best_Params": best_row.get("Best_Params", ""),
        "Best_Validation_RMSE": float(best_row["Validation_RMSE"])
        if pd.notna(best_row.get("Validation_RMSE"))
        else np.nan,
        "Best_Alpha": float(best_row["Alpha"]) if pd.notna(best_row.get("Alpha")) else np.nan,
        "Best_L1_Ratio": float(best_row["L1_Ratio"])
        if pd.notna(best_row.get("L1_Ratio"))
        else np.nan,
        "Best_N_Estimators": int(best_row["N_Estimators"])
        if pd.notna(best_row.get("N_Estimators"))
        else np.nan,
        "Best_Max_Depth": int(best_row["Max_Depth"])
        if pd.notna(best_row.get("Max_Depth"))
        else np.nan,
        "Best_Learning_Rate": float(best_row["Learning_Rate"])
        if pd.notna(best_row.get("Learning_Rate"))
        else np.nan,
        "Best_Subsample": float(best_row["Subsample"])
        if pd.notna(best_row.get("Subsample"))
        else np.nan,
    }


def _summarize_model_audit(
    model_name: str,
    currency: str,
    audit_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Return one comparison-summary row for a model."""
    signal_df = _build_hit_summary(audit_df)
    ic_df = audit_df.loc[audit_df["Error_Z"].notna() & audit_df["Forward_10d_Return"].notna()]
    info_coefficient = (
        float(ic_df["Error_Z"].corr(ic_df["Forward_10d_Return"]))
        if len(ic_df) >= 2
        else np.nan
    )

    test_rmse = _rmse(audit_df["Actual_Ret"], audit_df["Pred_Ret"])
    avg_train_rmse = float(audit_df["Train_RMSE"].dropna().mean()) if "Train_RMSE" in audit_df else np.nan
    avg_validation_rmse = (
        float(audit_df["Validation_RMSE"].dropna().mean())
        if "Validation_RMSE" in audit_df and audit_df["Validation_RMSE"].notna().any()
        else np.nan
    )
    tuning_summary = _best_tuning_event(audit_df)

    row = {
        "Currency": currency.upper(),
        "Model": MODEL_LABELS.get(model_name, model_name.upper()),
        "Observations": len(audit_df),
        "Signal_Triggers": len(signal_df),
        "Hit_Rate_Pct": float(signal_df["Hit"].mean() * 100.0) if not signal_df.empty else np.nan,
        "Information_Coefficient": info_coefficient,
        "Sharpe_Ratio": float(_annualized_sharpe_ratio(audit_df["Strategy_Daily_Return"])),
        "Maximum_Drawdown_Pct": float(_max_drawdown(audit_df["Strategy_Equity_Curve"]) * 100.0),
        "Strategy_Total_Return_Pct": float((audit_df["Strategy_Equity_Curve"].iloc[-1] - 1.0) * 100.0),
        "Average_Adj_R2": float(audit_df["Adj_R2"].dropna().mean()) if audit_df["Adj_R2"].notna().any() else np.nan,
        "Average_Train_RMSE": avg_train_rmse,
        "Average_Validation_RMSE": avg_validation_rmse,
        "Test_RMSE": test_rmse,
        "Generalization_Gap": test_rmse - avg_train_rmse if pd.notna(avg_train_rmse) else np.nan,
    }
    row.update(tuning_summary)
    return row


def _plot_multi_model_equity_curves(
    audit_map: Dict[str, pd.DataFrame],
    currency: str,
    output_path: str | Path,
) -> None:
    """Save a multi-model equity-curve comparison plot."""
    fig, ax = plt.subplots(figsize=(14, 6))
    oos_start: pd.Timestamp | None = None

    for model_name, audit_df in audit_map.items():
        plot_df = audit_df.loc[audit_df["Strategy_Equity_Curve"].notna()]
        if plot_df.empty:
            continue
        model_start = plot_df.index.min()
        if oos_start is None or model_start < oos_start:
            oos_start = model_start
        ax.plot(
            plot_df.index,
            plot_df["Strategy_Equity_Curve"],
            linewidth=1.4,
            label=MODEL_LABELS.get(model_name, model_name.upper()),
        )

    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.0)
    if oos_start is not None:
        ax.axvline(
            x=oos_start,
            color="grey",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            label="OOS Start",
        )
    ax.set_title(f"{currency.upper()} Stage 2 Multi-Model Equity Curves")
    ax.set_ylabel("Equity Curve")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_stage2_model_comparison_audit(
    currency: str,
    models: Sequence[str] = DEFAULT_MODELS,
    window: int = 250,
    error_sum_window: int = 10,
    z_window: int | None = None,
    recenter_window: int = 60,
    forward_days: int = 10,
    threshold: float = 2.0,
    return_scale: float = 100.0,
    sharpe_window: int = 252,
    cv_splits: int = 4,
    retune_frequency: int = 60,
    early_stopping_rounds: int = 25,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    start_date: str | pd.Timestamp | None = None,
) -> Dict[str, Any]:
    """Run the full model suite and build the requested comparison audit."""
    stage2_results = run_stage2_model_suite(
        currency=currency,
        models=models,
        window=window,
        error_sum_window=error_sum_window,
        z_window=z_window,
        recenter_window=recenter_window,
        return_scale=return_scale,
        cv_splits=cv_splits,
        retune_frequency=retune_frequency,
        early_stopping_rounds=early_stopping_rounds,
        processed_dir=processed_dir,
        start_date=start_date,
    )

    audit_map: Dict[str, pd.DataFrame] = {}
    summary_rows: list[Dict[str, Any]] = []
    interpretability_frames: list[pd.DataFrame] = []

    for model_name, model_df in stage2_results.items():
        audit_df = prepare_stage2_strategy_audit(
            stage2_df=model_df,
            forward_days=forward_days,
            threshold=threshold,
            sharpe_window=sharpe_window,
        )
        audit_map[model_name] = audit_df
        summary_rows.append(_summarize_model_audit(model_name=model_name, currency=currency, audit_df=audit_df))

        hits_df = _build_interpretability_hits(audit_df)
        hits_df.insert(0, "Model", MODEL_LABELS.get(model_name, model_name.upper()))
        interpretability_frames.append(hits_df)

    comparison_summary = pd.DataFrame(summary_rows).sort_values(
        by=["Generalization_Gap", "Test_RMSE"],
        ascending=[True, True],
    )
    combined_hits = (
        pd.concat(interpretability_frames, axis=0)
        if interpretability_frames
        else pd.DataFrame()
    )

    return {
        "comparison_summary": comparison_summary,
        "audit_map": audit_map,
        "combined_interpretability_hits": combined_hits,
    }


def save_stage2_model_comparison_audit(
    currency: str,
    models: Sequence[str] = DEFAULT_MODELS,
    output_dir: str | Path = DEFAULT_AUDIT_DIR,
    **kwargs,
) -> Dict[str, Path]:
    """Save the multi-model comparison audit outputs to disk."""
    audit = build_stage2_model_comparison_audit(
        currency=currency,
        models=models,
        **kwargs,
    )

    currency_dir = Path(output_dir) / currency.lower()
    currency_dir.mkdir(parents=True, exist_ok=True)

    summary_path = currency_dir / "stage2_model_comparison_summary.csv"
    equity_plot_path = currency_dir / "stage2_model_equity_curves.png"
    hits_path = currency_dir / "stage2_model_interpretability_hits.csv"

    audit["comparison_summary"].to_csv(summary_path, index=False)
    if not audit["combined_interpretability_hits"].empty:
        audit["combined_interpretability_hits"].to_csv(hits_path, index=True)
    _plot_multi_model_equity_curves(
        audit_map=audit["audit_map"],
        currency=currency,
        output_path=equity_plot_path,
    )

    saved_paths = {
        "comparison_summary_csv": summary_path,
        "equity_curve_comparison_png": equity_plot_path,
    }
    if not audit["combined_interpretability_hits"].empty:
        saved_paths["interpretability_hits_csv"] = hits_path

    for model_name, audit_df in audit["audit_map"].items():
        model_audit_path = currency_dir / f"stage2_{model_name}_audit_dataset.csv"
        model_generalization_path = currency_dir / f"stage2_{model_name}_generalization_metrics.png"
        model_equity_path = currency_dir / f"stage2_{model_name}_strategy_equity_curve.png"
        audit_df.to_csv(model_audit_path, index=True)
        _plot_generalization_metrics(
            audit_df=audit_df,
            currency=f"{currency.upper()} {MODEL_LABELS.get(model_name, model_name.upper())}",
            output_path=model_generalization_path,
        )
        _plot_strategy_equity_curve(
            audit_df=audit_df,
            currency=f"{currency.upper()} {MODEL_LABELS.get(model_name, model_name.upper())}",
            output_path=model_equity_path,
        )
        saved_paths[f"{model_name}_audit_dataset_csv"] = model_audit_path
        saved_paths[f"{model_name}_generalization_plot_png"] = model_generalization_path
        saved_paths[f"{model_name}_equity_curve_png"] = model_equity_path

    return saved_paths


def build_g10_master_comparison_table(
    comparison_summaries: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate per-currency comparison summaries into one master ranking table."""
    combined = pd.concat(comparison_summaries, axis=0, ignore_index=True)

    grouped = (
        combined.groupby("Model", as_index=False)
        .agg(
            Currencies=("Currency", "nunique"),
            Average_Information_Coefficient=("Information_Coefficient", "mean"),
            Average_Hit_Rate_Pct=("Hit_Rate_Pct", "mean"),
            Average_Sharpe_Ratio=("Sharpe_Ratio", "mean"),
            Average_Maximum_Drawdown_Pct=("Maximum_Drawdown_Pct", "mean"),
            Average_Generalization_Gap=("Generalization_Gap", "mean"),
        )
    )
    grouped["IC_Rank"] = grouped["Average_Information_Coefficient"].rank(
        ascending=False,
        method="dense",
    )
    grouped["Hit_Rank"] = grouped["Average_Hit_Rate_Pct"].rank(
        ascending=False,
        method="dense",
    )
    grouped["Combined_Rank"] = (grouped["IC_Rank"] + grouped["Hit_Rank"]) / 2.0
    grouped = grouped.sort_values(
        by=["Combined_Rank", "Average_Information_Coefficient", "Average_Hit_Rate_Pct"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return grouped


def save_stage2_g10_master_comparison(
    currencies: Sequence[str] = DEFAULT_CURRENCIES,
    models: Sequence[str] = DEFAULT_MODELS,
    output_dir: str | Path = DEFAULT_AUDIT_DIR,
    **kwargs,
) -> Dict[str, Path]:
    """Run and save the comparison audit for the full currency universe plus a master table."""
    comparison_summaries: list[pd.DataFrame] = []

    for currency in currencies:
        save_stage2_model_comparison_audit(
            currency=currency,
            models=models,
            output_dir=output_dir,
            **kwargs,
        )
        summary_path = Path(output_dir) / currency.lower() / "stage2_model_comparison_summary.csv"
        comparison_summaries.append(pd.read_csv(summary_path))

    master_table = build_g10_master_comparison_table(comparison_summaries)
    master_path = Path(output_dir) / "stage2_g10_master_model_ranking.csv"
    master_table.to_csv(master_path, index=False)
    return {"master_ranking_csv": master_path}


def save_stage2_ml_performance_audit(
    currency: str,
    output_dir: str | Path = DEFAULT_AUDIT_DIR,
    **kwargs,
) -> Dict[str, Path]:
    """Generate the audit and save CSV/PNG outputs to disk."""
    audit = build_stage2_ml_performance_audit(currency=currency, **kwargs)

    currency_dir = Path(output_dir) / currency.lower()
    currency_dir.mkdir(parents=True, exist_ok=True)

    summary_path = currency_dir / "stage2_ml_audit_summary.csv"
    audit_path = currency_dir / "stage2_ml_audit_dataset.csv"
    triggers_path = currency_dir / "stage2_signal_triggers.csv"
    hits_path = currency_dir / "stage2_hit_interpretability.csv"
    plot_path = currency_dir / "stage2_generalization_metrics.png"
    equity_plot_path = currency_dir / "stage2_strategy_equity_curve.png"

    audit["summary"].to_csv(summary_path, index=False)
    audit["audit_dataset"].to_csv(audit_path, index=True)
    audit["signal_triggers"].to_csv(triggers_path, index=True)
    audit["interpretability_hits"].to_csv(hits_path, index=True)
    _plot_generalization_metrics(audit["audit_dataset"], currency=currency, output_path=plot_path)
    _plot_strategy_equity_curve(
        audit["audit_dataset"],
        currency=currency,
        output_path=equity_plot_path,
    )

    return {
        "summary_csv": summary_path,
        "audit_dataset_csv": audit_path,
        "signal_triggers_csv": triggers_path,
        "interpretability_hits_csv": hits_path,
        "generalization_plot_png": plot_path,
        "strategy_equity_curve_png": equity_plot_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create a CLI for the Stage 2 ML performance audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("currency", help="Currency code, for example eur or gbp.")
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Run the multi-model comparison audit instead of the single OLS baseline audit.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model names to include in the multi-model comparison audit.",
    )
    parser.add_argument("--window", type=int, default=50, help="Rolling training window.")
    parser.add_argument(
        "--error-sum-window",
        type=int,
        default=10,
        help="EWMA span for cumulative error smoothing.",
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
        "--forward-days",
        type=int,
        default=10,
        help="Forward return horizon used for hit-rate and IC.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Absolute Signal_Z threshold for BUY/SELL classification.",
    )
    parser.add_argument(
        "--return-scale",
        type=float,
        default=100.0,
        help="Scaling used to compound predicted log returns into levels.",
    )
    parser.add_argument(
        "--sharpe-window",
        type=int,
        default=252,
        help="Rolling window used for the rolling Sharpe ratio series.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=4,
        help="TimeSeriesSplit folds used by the regularized and tree models.",
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
        default=DEFAULT_AUDIT_DIR,
        help="Directory where audit outputs should be saved.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional YYYY-MM-DD start date filter.",
    )
    return parser


def main() -> None:
    """Run the CLI and print the key audit outputs."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.compare_models:
        output_paths = save_stage2_model_comparison_audit(
            currency=args.currency,
            models=args.models,
            window=args.window,
            error_sum_window=args.error_sum_window,
            z_window=args.z_window,
            recenter_window=args.recenter_window,
            forward_days=args.forward_days,
            threshold=args.threshold,
            return_scale=args.return_scale,
            sharpe_window=args.sharpe_window,
            cv_splits=args.cv_splits,
            retune_frequency=args.retune_frequency,
            early_stopping_rounds=args.early_stopping_rounds,
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            start_date=args.start_date,
        )
        summary_df = pd.read_csv(output_paths["comparison_summary_csv"])
        print("\nStage 2 Multi-Model Comparison Audit")
    else:
        output_paths = save_stage2_ml_performance_audit(
            currency=args.currency,
            window=args.window,
            error_sum_window=args.error_sum_window,
            z_window=args.z_window,
            recenter_window=args.recenter_window,
            forward_days=args.forward_days,
            threshold=args.threshold,
            return_scale=args.return_scale,
            sharpe_window=args.sharpe_window,
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            start_date=args.start_date,
        )
        summary_df = pd.read_csv(output_paths["summary_csv"])
        print("\nStage 2 Machine Learning Performance Audit")

    print(summary_df.to_string(index=False))
    print("\nSaved outputs:")
    for label, path in output_paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
