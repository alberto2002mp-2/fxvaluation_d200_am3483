"""Convenience Stage 2 fair value runner using saved master training CSVs."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


cwd = Path.cwd()
if (cwd / "src").exists():
    project_root = cwd
elif (cwd.parent / "src").exists():
    project_root = cwd.parent
else:
    raise FileNotFoundError("Could not locate project root containing 'src'.")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.build_model_ready_data import DEFAULT_PROCESSED_DIR
from src.stage2_ml_models import run_stage2_model


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


def plot_stage2_fair_value_plotly(
    res_df: pd.DataFrame,
    currency: str,
    start_date: str | pd.Timestamp | None = None,
):
    """Plot Stage 2 fair value levels and error z-score with Plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for Stage 2 charts. Install it with `pip install plotly`."
        ) from exc

    plot_df = res_df.copy()
    if start_date is not None:
        start_date = pd.Timestamp(start_date)
        plot_df = plot_df.loc[plot_df.index >= start_date]

    if plot_df.empty:
        raise ValueError(f"No Stage 2 fair value rows available for {currency} after start_date filter.")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            f"{currency.upper()} Fair Value Level Analysis",
            "Trading Signal: Detrended Gap Z-Score",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["Actual_Price"],
            mode="lines",
            name="Actual Price",
            line=dict(color="orange", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["Macro_Anchor_Price"],
            mode="lines",
            name="Macro Anchor Price",
            line=dict(color="blue", width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["Signal_Z"],
            mode="lines",
            name="Signal Z-score",
            line=dict(color="purple", width=2),
        ),
        row=2,
        col=1,
    )

    sell_df = plot_df.copy()
    sell_df["sell_fill"] = np.where(sell_df["Signal_Z"] > 2, sell_df["Signal_Z"], np.nan)
    fig.add_trace(
        go.Scatter(
            x=sell_df.index,
            y=sell_df["sell_fill"],
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.20)",
            name="SELL Zone",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    buy_df = plot_df.copy()
    buy_df["buy_fill"] = np.where(buy_df["Signal_Z"] < -2, buy_df["Signal_Z"], np.nan)
    fig.add_trace(
        go.Scatter(
            x=buy_df.index,
            y=buy_df["buy_fill"],
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(0, 128, 0, 0.20)",
            name="BUY Zone",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    if "Recenter_Event" in plot_df.columns:
        recenter_dates = plot_df.index[plot_df["Recenter_Event"].fillna(False)]
        for dt in recenter_dates:
            fig.add_vline(
                x=dt,
                line_dash="dash",
                line_color="rgba(80, 80, 80, 0.55)",
                line_width=1,
                row=1,
                col=1,
            )
            fig.add_vline(
                x=dt,
                line_dash="dash",
                line_color="rgba(80, 80, 80, 0.55)",
                line_width=1,
                row=2,
                col=1,
            )

    fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="grey", row=2, col=1)
    fig.add_hrect(
        y0=-1,
        y1=1,
        fillcolor="rgba(128, 128, 128, 0.12)",
        line_width=0,
        layer="below",
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Price Level", row=1, col=1)
    fig.update_yaxes(title_text="Z-score", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=850,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=30, t=90, b=50),
    )

    return fig


def run_stage2_fair_value(
    currency: str,
    window: int = 50,
    error_sum_window: int = 10,
    z_window: int | None = None,
    recenter_window: int = 60,
    return_scale: float = 100.0,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    start_date: str | pd.Timestamp | None = None,
    show_plot: bool = True,
) -> pd.DataFrame:
    """
    Run Stage 2 fair value reconstruction from the saved master CSV.

    Notes:
    - Uses raw (`*_raw`) features for fitting.
    - Uses `Log_Return` as the target.
    - Builds a cumulative `Macro_Anchor_Price` from predicted log changes.
    - Applies regime-based re-centering every `recenter_window` days.
    - `return_scale=100.0` matches the project's use of `100 * log-diff`.
    """
    res_df = run_stage2_model(
        currency=currency,
        model_name="ols",
        window=window,
        error_sum_window=error_sum_window,
        z_window=z_window,
        recenter_window=recenter_window,
        return_scale=return_scale,
        processed_dir=processed_dir,
        start_date=start_date,
    )

    if show_plot:
        fig = plot_stage2_fair_value_plotly(res_df, currency=currency, start_date=None)
        fig.show()

    return res_df


def run_stage2_fair_value_ensemble(
    currency: str,
    fast_window: int = 50,
    fast_error_sum_window: int = 5,
    fast_z_window: int = 50,
    slow_window: int = 125,
    slow_error_sum_window: int = 20,
    slow_z_window: int = 125,
    recenter_window: int = 60,
    confirmation_threshold: float = -1.5,
    return_scale: float = 100.0,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    start_date: str | pd.Timestamp | None = None,
    show_plot: bool = True,
) -> pd.DataFrame:
    """
    Run fast/slow Stage 2 models and return an ensemble-confirmed signal.

    Ensemble rule:
    - STRONG BUY when both fast and slow Error_Z are below confirmation_threshold.
    """
    fast_df = run_stage2_fair_value(
        currency=currency,
        window=fast_window,
        error_sum_window=fast_error_sum_window,
        z_window=fast_z_window,
        return_scale=return_scale,
        processed_dir=processed_dir,
        start_date=start_date,
        show_plot=False,
        recenter_window=recenter_window,
    )
    slow_df = run_stage2_fair_value(
        currency=currency,
        window=slow_window,
        error_sum_window=slow_error_sum_window,
        z_window=slow_z_window,
        return_scale=return_scale,
        processed_dir=processed_dir,
        start_date=start_date,
        show_plot=False,
        recenter_window=recenter_window,
    )

    overlap = fast_df.index.intersection(slow_df.index)
    if overlap.empty:
        raise ValueError("Fast and slow model outputs do not overlap in time.")

    out = fast_df.loc[overlap].copy()
    out = out.rename(
        columns={
            "Signal_Z": "Signal_Z_Fast",
            "Error_Z": "Error_Z_Fast",
            "Macro_Gap": "Macro_Gap_Fast",
            "Signal": "Signal_Fast",
            "Days_In_Signal": "Days_In_Signal_Fast",
        }
    )

    out["Signal_Z_Slow"] = slow_df.loc[overlap, "Signal_Z"]
    out["Error_Z_Slow"] = slow_df.loc[overlap, "Error_Z"]
    out["Macro_Gap_Slow"] = slow_df.loc[overlap, "Macro_Gap"]
    out["Signal_Slow"] = slow_df.loc[overlap, "Signal"]
    out["Days_In_Signal_Slow"] = slow_df.loc[overlap, "Days_In_Signal"]

    strong_buy = (out["Error_Z_Fast"] < confirmation_threshold) & (
        out["Error_Z_Slow"] < confirmation_threshold
    )
    out["Signal"] = np.where(strong_buy, "STRONG BUY", "NEUTRAL")

    if show_plot:
        plot_df = out.copy()
        plot_df["Signal_Z"] = plot_df["Signal_Z_Fast"]
        plot_df["Error_Z"] = plot_df["Error_Z_Fast"]
        fig = plot_stage2_fair_value_plotly(plot_df, currency=currency, start_date=None)
        fig.show()

    return out


if __name__ == "__main__":
    df = run_stage2_fair_value("eur")
    print(df.tail())
