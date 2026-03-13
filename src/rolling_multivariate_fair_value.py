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
        final_fv_results[currency] = final_df
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

    plot_stage2_diagnostics(final_fv_results, currency="eur")
