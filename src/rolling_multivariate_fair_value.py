"""Stage 2 adaptive rolling multivariate fair value engine."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots


cwd = Path.cwd()
if (cwd / "src").exists():
    project_root = cwd
elif (cwd.parent / "src").exists():
    project_root = cwd.parent
else:
    raise FileNotFoundError("Could not locate project root containing 'src'.")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from src.data.build_ultimate_df import build_ultimate_df
from src.data.create_dataframes import fx_df as raw_fx_df
from src.data.standardize_rolling_drivers import build_standardized_df_map
from src.diversified_top_drivers_history import build_diversified_top_drivers_map
from src.rolling_univariate_ols import DEFAULT_Y_COL_MAP, build_rolling_maps


RESULT_COLS = [
    "Actual_Price",
    "Fair_Value",
    "Upper_Band",
    "Lower_Band",
    "Error_Gap",
    "Adj_R2",
    "RMSE",
    "MAE",
    "Drivers_Used_Count",
]

RAW_PRICE_COL_MAP = {
    "eur": "EURUSD Curncy",
    "gbp": "GBPUSD Curncy",
    "jpy": "USDJPY Curncy",
    "chf": "USDCHF Curncy",
    "cad": "USDCAD Curncy",
    "aud": "AUDUSD Curncy",
    "nzd": "NZDUSD Curncy",
    "nok": "USDNOK Curncy",
    "sek": "USDSEK Curncy",
}


def _driver_name_cols(top_df: pd.DataFrame) -> List[str]:
    """Return ordered driver name columns such as Driver 1 Name, Driver 2 Name, ..."""
    pattern = re.compile(r"Driver (\d+) Name")
    cols: List[Tuple[int, str]] = []
    for col in top_df.columns:
        match = pattern.fullmatch(col)
        if match:
            cols.append((int(match.group(1)), col))
    return [col for _, col in sorted(cols)]


def _selected_driver_names(row: pd.Series, name_cols: Sequence[str], top_n: int) -> List[str]:
    """Return unique non-null driver names, preserving selection order."""
    names: List[str] = []
    seen = set()
    for col in name_cols[:top_n]:
        name = row.get(col)
        if not isinstance(name, str):
            continue
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _subset_window_df(
    currency_df: pd.DataFrame,
    dt,
    y_col: str,
    drivers: Sequence[str],
    window: int,
) -> pd.DataFrame:
    """Return the trailing clean window for y and the selected raw drivers."""
    subset_cols = [y_col] + list(drivers)
    return currency_df.loc[:dt, subset_cols].tail(window).dropna()


def _best_driver_subset(
    currency_df: pd.DataFrame,
    dt,
    y_col: str,
    candidate_drivers: Sequence[str],
    window: int,
    min_obs: int | None = None,
) -> Tuple[List[str], pd.DataFrame] | None:
    """
    Choose the richest usable subset of drivers for this timestamp.

    Preference order:
    1. More active drivers
    2. More clean observations after dropna
    """
    valid_candidates = [
        driver
        for driver in candidate_drivers
        if driver in currency_df.columns and pd.notna(currency_df.at[dt, driver])
    ]
    if not valid_candidates:
        return None

    best_choice: Tuple[int, int, List[str], pd.DataFrame] | None = None
    for subset_size in range(len(valid_candidates), 0, -1):
        for subset in combinations(valid_candidates, subset_size):
            window_df = _subset_window_df(currency_df, dt, y_col, subset, window)
            required_obs = max(subset_size + 2, min_obs or 0)
            if len(window_df) < required_obs:
                continue

            choice = (subset_size, len(window_df), list(subset), window_df)
            if best_choice is None or choice[:2] > best_choice[:2]:
                best_choice = choice

        if best_choice is not None:
            break

    if best_choice is None:
        return None
    return best_choice[2], best_choice[3]


def _adjusted_r2(r2: float, n_obs: int, n_features: int) -> float:
    """Return adjusted R^2, or NaN when undefined."""
    if n_obs <= n_features + 1:
        return np.nan
    return 1 - (1 - r2) * ((n_obs - 1) / (n_obs - n_features - 1))


def _rolling_zscore(series: pd.Series, window: int = 250) -> pd.Series:
    """Return rolling z-score using the trailing window mean and std."""
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    return (series - rolling_mean) / rolling_std


def _build_fair_value_anchor(
    predicted_log_change: pd.Series,
    actual_price_level: pd.Series,
    return_scale: float = 100.0,
) -> pd.Series:
    """Compound predicted log changes from the first available actual level."""
    anchor = pd.Series(index=predicted_log_change.index, dtype=float)
    first_valid_idx = actual_price_level.first_valid_index()
    if first_valid_idx is None:
        return anchor

    base_level = actual_price_level.loc[first_valid_idx]
    if pd.isna(base_level):
        return anchor

    running_level = float(base_level)
    started = False
    for dt in predicted_log_change.index:
        pred = predicted_log_change.loc[dt]
        if dt == first_valid_idx:
            anchor.loc[dt] = running_level
            started = True
            continue
        if not started or pd.isna(pred):
            anchor.loc[dt] = np.nan
            continue
        running_level = running_level * np.exp(pred / return_scale)
        anchor.loc[dt] = running_level

    return anchor


def augment_with_price_levels_and_signals(
    final_df: pd.DataFrame,
    currency: str,
    raw_price_df: pd.DataFrame | None = None,
    signal_window: int = 250,
    return_scale: float = 100.0,
) -> pd.DataFrame:
    """Add price-level fair values, residual z-scores, and trading signals."""
    if raw_price_df is None:
        raw_price_df = raw_fx_df

    price_col = RAW_PRICE_COL_MAP.get(currency)
    if price_col is None:
        raise KeyError(f"No raw FX price column configured for {currency}.")
    if price_col not in raw_price_df.columns:
        raise KeyError(f"{price_col} not found in raw_price_df.")

    out = final_df.copy()
    out["Actual_Log_Change"] = out["Actual_Price"]
    out["Predicted_Log_Change"] = out["Fair_Value"]

    actual_price_level = raw_price_df[price_col].reindex(out.index)
    out["Actual_Price_Level"] = actual_price_level
    out["Fair_Value_Level"] = actual_price_level.shift(1) * np.exp(
        out["Predicted_Log_Change"] / return_scale
    )
    out["Fair_Value_Anchor"] = _build_fair_value_anchor(
        predicted_log_change=out["Predicted_Log_Change"],
        actual_price_level=actual_price_level,
        return_scale=return_scale,
    )

    out["Residual"] = out["Actual_Log_Change"] - out["Predicted_Log_Change"]
    out["Residual_Zscore"] = _rolling_zscore(out["Residual"], window=signal_window)
    out["Signal"] = np.select(
        [out["Residual_Zscore"] > 2.0, out["Residual_Zscore"] < -2.0],
        ["SELL", "BUY"],
        default="NEUTRAL",
    )

    return out


def build_currency_stage2_fv(
    currency_df: pd.DataFrame,
    top_drivers_df: pd.DataFrame,
    y_col: str,
    window: int = 250,
    top_n: int = 3,
    min_obs: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build Stage 2 adaptive fair value outputs for one currency.

    Returns:
        final_df: requested fair value result columns plus MAE.
        metadata_df: driver names, coefficients, intercept, and window stats.
    """
    if y_col not in currency_df.columns:
        raise KeyError(f"{y_col} not found in currency_df.")

    name_cols = _driver_name_cols(top_drivers_df)
    if len(name_cols) < 1:
        raise KeyError("No driver name columns found in top_drivers_df.")

    aligned_index = currency_df.index.intersection(top_drivers_df.index)
    currency_df = currency_df.sort_index().loc[aligned_index]
    top_drivers_df = top_drivers_df.sort_index().loc[aligned_index]

    final_df = pd.DataFrame(index=aligned_index, columns=RESULT_COLS, dtype=float)
    metadata_cols = [
        "Intercept",
        "Drivers_Used",
        "Window_Obs",
        "R2",
        "Adj_R2",
        "RMSE",
        "MAE",
        "Driver 1 Name",
        "Driver 2 Name",
        "Driver 3 Name",
        "Driver 1 Beta",
        "Driver 2 Beta",
        "Driver 3 Beta",
    ]
    metadata_df = pd.DataFrame(index=aligned_index, columns=metadata_cols, dtype=object)

    for dt in aligned_index:
        actual_price = currency_df.at[dt, y_col]
        final_df.at[dt, "Actual_Price"] = actual_price

        if pd.isna(actual_price):
            continue

        selected_drivers = _selected_driver_names(top_drivers_df.loc[dt], name_cols, top_n)
        if not selected_drivers:
            continue

        best_fit = _best_driver_subset(
            currency_df=currency_df,
            dt=dt,
            y_col=y_col,
            candidate_drivers=selected_drivers,
            window=window,
            min_obs=min_obs,
        )
        if best_fit is None:
            continue

        active_drivers, window_df = best_fit
        current_X = currency_df.loc[[dt], active_drivers]
        if current_X.isna().any(axis=None):
            continue

        model = LinearRegression(fit_intercept=True)
        model.fit(window_df[active_drivers], window_df[y_col])

        fitted_window = model.predict(window_df[active_drivers])
        residuals = window_df[y_col] - fitted_window
        n_obs = len(window_df)
        n_features = len(active_drivers)
        r2 = float(model.score(window_df[active_drivers], window_df[y_col]))
        adj_r2 = _adjusted_r2(r2, n_obs, n_features)
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        mae = float(np.mean(np.abs(residuals)))

        fair_value = float(model.predict(current_X)[0])
        upper_band = fair_value + 2 * rmse
        lower_band = fair_value - 2 * rmse
        error_gap = actual_price - fair_value

        final_df.loc[dt] = [
            actual_price,
            fair_value,
            upper_band,
            lower_band,
            error_gap,
            adj_r2,
            rmse,
            mae,
            n_features,
        ]

        driver_name_pad = active_drivers + [np.nan] * (top_n - len(active_drivers))
        driver_beta_pad = [float(beta) for beta in model.coef_] + [np.nan] * (
            top_n - len(active_drivers)
        )
        metadata_df.loc[dt] = [
            float(model.intercept_),
            ", ".join(active_drivers),
            n_obs,
            r2,
            adj_r2,
            rmse,
            mae,
            driver_name_pad[0],
            driver_name_pad[1],
            driver_name_pad[2],
            driver_beta_pad[0],
            driver_beta_pad[1],
            driver_beta_pad[2],
        ]

    return final_df, metadata_df


def build_final_fv_results(
    ultimate_df: Dict[str, pd.DataFrame],
    top_mapz: Dict[str, pd.DataFrame],
    y_col_map: Dict[str, str] | None = None,
    window: int = 250,
    top_n: int = 3,
    min_obs: int | None = None,
    raw_price_df: pd.DataFrame | None = None,
    signal_window: int = 250,
    return_scale: float = 100.0,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Build the Stage 2 fair value result map and the supporting metadata map."""
    if y_col_map is None:
        y_col_map = DEFAULT_Y_COL_MAP

    final_fv_results: Dict[str, pd.DataFrame] = {}
    fv_metadata_map: Dict[str, pd.DataFrame] = {}

    for currency, top_drivers_df in top_mapz.items():
        currency_df = ultimate_df.get(currency)
        if currency_df is None:
            continue

        y_col = y_col_map.get(currency)
        if y_col is None:
            raise KeyError(f"No default y_col found for {currency}.")

        final_df, metadata_df = build_currency_stage2_fv(
            currency_df=currency_df,
            top_drivers_df=top_drivers_df,
            y_col=y_col,
            window=window,
            top_n=top_n,
            min_obs=min_obs,
        )
        final_fv_results[currency] = augment_with_price_levels_and_signals(
            final_df=final_df,
            currency=currency,
            raw_price_df=raw_price_df,
            signal_window=signal_window,
            return_scale=return_scale,
        )
        fv_metadata_map[currency] = metadata_df

    return final_fv_results, fv_metadata_map


def plot_stage2_diagnostics(
    final_fv_results: Dict[str, pd.DataFrame],
    currency: str = "eur",
) -> None:
    """Generate the requested dual-pane Stage 2 diagnostic chart."""
    result_df = final_fv_results.get(currency)
    if result_df is None:
        raise KeyError(f"{currency} not found in final_fv_results.")

    plot_df = result_df.dropna(subset=["Actual_Price", "Fair_Value"])
    if plot_df.empty:
        raise ValueError(f"No complete Actual/Fair Value rows available for {currency}.")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    axes[0].plot(plot_df.index, plot_df["Actual_Price"], label="Actual Price", linewidth=2.0)
    axes[0].plot(
        plot_df.index,
        plot_df["Fair_Value"],
        label="Fair Value",
        linewidth=2.0,
        linestyle="--",
    )
    band_df = plot_df.dropna(subset=["Lower_Band", "Upper_Band"])
    if not band_df.empty:
        axes[0].fill_between(
            band_df.index,
            band_df["Lower_Band"],
            band_df["Upper_Band"],
            color="lightblue",
            alpha=0.35,
            label="Statistical Fair Value Range",
        )
    axes[0].set_title(f"{currency.upper()} Stage 2 Adaptive Fair Value")
    axes[0].set_ylabel("Price")
    axes[0].legend()

    quality_df = plot_df.dropna(subset=["Adj_R2"])
    axes[1].plot(quality_df.index, quality_df["Adj_R2"], color="tab:green", linewidth=1.8)
    axes[1].axhline(0.0, color="grey", linewidth=1.0, linestyle=":")
    axes[1].set_title("Rolling Adjusted R^2")
    axes[1].set_ylabel("Adj R^2")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.show()


def plot_stage2_diagnostics_plotly(
    final_fv_results: Dict[str, pd.DataFrame],
    currency: str = "eur",
    start_date: str | pd.Timestamp | None = None,
):
    """Generate the Stage 2 dual-pane diagnostic chart with Plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for plot_stage2_diagnostics_plotly. Install it with `pip install plotly`."
        ) from exc

    result_df = final_fv_results.get(currency)
    if result_df is None:
        raise KeyError(f"{currency} not found in final_fv_results.")

    plot_df = result_df.copy()
    if start_date is not None:
        start_date = pd.Timestamp(start_date)
        plot_df = plot_df.loc[plot_df.index >= start_date]

    plot_df = plot_df.dropna(subset=["Actual_Price", "Fair_Value"])
    if plot_df.empty:
        raise ValueError(f"No complete Actual/Fair Value rows available for {currency}.")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=(
            f"{currency.upper()} Stage 2 Adaptive Fair Value",
            "Rolling Adjusted R^2",
        ),
    )

    band_df = plot_df.dropna(subset=["Lower_Band", "Upper_Band"])
    if not band_df.empty:
        fig.add_trace(
            go.Scatter(
                x=band_df.index,
                y=band_df["Upper_Band"],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=band_df.index,
                y=band_df["Lower_Band"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(173, 216, 230, 0.35)",
                name="Statistical Fair Value Range",
                hovertemplate="Lower: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["Actual_Price"],
            mode="lines",
            name="Actual Price",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["Fair_Value"],
            mode="lines",
            name="Fair Value",
            line=dict(width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )

    quality_df = plot_df.dropna(subset=["Adj_R2"])
    fig.add_trace(
        go.Scatter(
            x=quality_df.index,
            y=quality_df["Adj_R2"],
            mode="lines",
            name="Adj R2",
            line=dict(width=2, color="green"),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Adj R2", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    if start_date is not None:
        fig.update_xaxes(range=[start_date, plot_df.index.max()], row=1, col=1)
        fig.update_xaxes(range=[start_date, plot_df.index.max()], row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=30, t=80, b=50),
    )

    return fig


def plot_level_signal_diagnostics(
    final_fv_results: Dict[str, pd.DataFrame],
    currency: str = "eur",
) -> None:
    """Plot actual/fair-value levels and residual z-score signal bands."""
    result_df = final_fv_results.get(currency)
    if result_df is None:
        raise KeyError(f"{currency} not found in final_fv_results.")

    level_df = result_df.dropna(subset=["Actual_Price_Level", "Fair_Value_Level"])
    if level_df.empty:
        raise ValueError(f"No complete price level rows available for {currency}.")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    axes[0].plot(level_df.index, level_df["Actual_Price_Level"], label="Actual Price Level", linewidth=2.0)
    axes[0].plot(
        level_df.index,
        level_df["Fair_Value_Level"],
        label="Fair Value Level",
        linewidth=2.0,
        linestyle="--",
    )
    axes[0].set_title(f"{currency.upper()} Actual vs Reconstructed Fair Value Level")
    axes[0].set_ylabel("Price Level")
    axes[0].legend()

    signal_df = result_df.dropna(subset=["Residual_Zscore"])
    axes[1].plot(signal_df.index, signal_df["Residual_Zscore"], color="tab:purple", linewidth=1.8)
    axes[1].axhline(2.0, color="red", linestyle="--", linewidth=1.2)
    axes[1].axhline(-2.0, color="green", linestyle="--", linewidth=1.2)
    axes[1].axhline(0.0, color="grey", linestyle=":", linewidth=1.0)
    axes[1].set_title("Residual Z-score")
    axes[1].set_ylabel("Z-score")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.show()


def generate_level_report(final_fv_results: Dict[str, pd.DataFrame]) -> None:
    """Print the latest residual z-score and valuation status for each currency."""
    print("\nStage 2 Level Report")
    print("-" * 72)
    for currency in sorted(final_fv_results):
        df = final_fv_results[currency]
        latest = df.dropna(subset=["Residual_Zscore"]).tail(1)
        if latest.empty:
            print(f"{currency.upper():<6}  no z-score available yet")
            continue

        row = latest.iloc[0]
        zscore = float(row["Residual_Zscore"])
        signal = str(row["Signal"])
        price_level = row.get("Actual_Price_Level", np.nan)
        fair_level = row.get("Fair_Value_Level", np.nan)

        if zscore > 2.0:
            status = "OVERBOUGHT"
        elif zscore < -2.0:
            status = "OVERSOLD"
        else:
            status = "NEUTRAL"

        print(
            f"{currency.upper():<6}  z={zscore:>6.2f}  signal={signal:<8}  "
            f"status={status:<10}  actual={price_level:>10.4f}  fair={fair_level:>10.4f}"
        )


def build_inputs(window: int = 250) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
]:
    """Build raw inputs and Stage 1 diversified top-driver selections."""
    ultimate_df = build_ultimate_df()
    standardized_df_map = build_standardized_df_map(ultimate_df, window=window)
    betas_mapz, signif_mapz = build_rolling_maps(standardized_df_map, window=window)
    betas_raw, _ = build_rolling_maps(ultimate_df, window=window)

    top_mapz = build_diversified_top_drivers_map(
        betas_mapz,
        signif_mapz,
        min_significance=95.0,
        top_n=3,
        betas_raw_map=betas_raw,
    )
    return ultimate_df, top_mapz


if __name__ == "__main__":
    ultimate_df, top_mapz = build_inputs(window=250)
    final_fv_results, fv_metadata_map = build_final_fv_results(
        ultimate_df,
        top_mapz,
        window=250,
        top_n=3,
    )

    if "eur" in final_fv_results:
        print("\nEUR Stage 2 fair value (tail):")
        print(final_fv_results["eur"].tail())
        print("\nEUR Stage 2 metadata (tail):")
        print(fv_metadata_map["eur"].tail())

    generate_level_report(final_fv_results)
    plot_stage2_diagnostics(final_fv_results, currency="eur")
    plot_level_signal_diagnostics(final_fv_results, currency="eur")
