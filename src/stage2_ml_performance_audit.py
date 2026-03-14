"""Generate a Machine Learning Performance Audit for Stage 2 fair value signals."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Dict

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

    audit["summary"].to_csv(summary_path, index=False)
    audit["audit_dataset"].to_csv(audit_path, index=True)
    audit["signal_triggers"].to_csv(triggers_path, index=True)
    audit["interpretability_hits"].to_csv(hits_path, index=True)
    _plot_generalization_metrics(audit["audit_dataset"], currency=currency, output_path=plot_path)

    return {
        "summary_csv": summary_path,
        "audit_dataset_csv": audit_path,
        "signal_triggers_csv": triggers_path,
        "interpretability_hits_csv": hits_path,
        "generalization_plot_png": plot_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create a CLI for the Stage 2 ML performance audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("currency", help="Currency code, for example eur or gbp.")
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

    output_paths = save_stage2_ml_performance_audit(
        currency=args.currency,
        window=args.window,
        error_sum_window=args.error_sum_window,
        z_window=args.z_window,
        recenter_window=args.recenter_window,
        forward_days=args.forward_days,
        threshold=args.threshold,
        return_scale=args.return_scale,
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
