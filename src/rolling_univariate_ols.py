"""Rolling univariate OLS betas and significance for each driver."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS


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

ultimate_df = build_ultimate_df()

DEFAULT_Y_COL_MAP = {
    "eur": "EURUSD",
    "gbp": "GBPUSD",
    "jpy": "USDJPY",
    "chf": "USDCHF",
    "cad": "USDCAD",
    "aud": "AUDUSD",
    "nzd": "NZDUSD",
    "nok": "USDNOK",
    "sek": "USDSEK",
}


def _pvalues_from_tvalues(tvalues: pd.DataFrame, df_resid) -> pd.DataFrame:
    """Compute two-sided p-values from t-values and df_resid."""
    try:
        from scipy import stats
    except Exception:
        return pd.DataFrame(index=tvalues.index, columns=tvalues.columns, dtype=float)

    df_resid_arr = df_resid
    if np.isscalar(df_resid_arr):
        df_resid_arr = np.full(tvalues.shape[0], float(df_resid_arr))

    df_resid_series = pd.Series(df_resid_arr, index=tvalues.index)
    pvals = pd.DataFrame(index=tvalues.index, columns=tvalues.columns, dtype=float)
    for col in tvalues.columns:
        tcol = tvalues[col]
        df_col = df_resid_series.reindex(tcol.index)
        pvals[col] = 2 * stats.t.sf(np.abs(tcol), df_col)
    return pvals


def _as_frame(arr, index, columns) -> pd.DataFrame:
    if isinstance(arr, pd.DataFrame):
        return arr
    return pd.DataFrame(arr, index=index, columns=columns)


def rolling_univariate_ols(
    df: pd.DataFrame,
    y_col: str,
    window: int = 250,
    min_obs: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run rolling OLS per driver with pairwise deletion and return (betas_df, significance_df).

    This avoids dropping entire rows across all drivers when a single variable is missing.
    """
    if y_col not in df.columns:
        raise KeyError(f"{y_col} not found in dataframe columns.")

    X_cols = [c for c in df.columns if c != y_col]
    if not X_cols:
        raise ValueError("No driver columns found (only y_col present).")

    if min_obs is None:
        min_obs = window

    betas = pd.DataFrame(index=df.index, columns=X_cols, dtype=float)
    signif_pct = pd.DataFrame(index=df.index, columns=X_cols, dtype=float)

    for driver in X_cols:
        data = df[[y_col, driver]].dropna()
        if len(data) < min_obs:
            continue

        exog = sm.add_constant(data[driver], has_constant="add")
        model = RollingOLS(data[y_col], exog, window=window, min_nobs=min_obs).fit()

        params = _as_frame(model.params, data.index, exog.columns).reindex(df.index)
        if hasattr(model, "pvalues") and model.pvalues is not None:
            pvals = _as_frame(model.pvalues, data.index, exog.columns).reindex(df.index)
        elif hasattr(model, "tvalues") and model.tvalues is not None:
            tvals = _as_frame(model.tvalues, data.index, exog.columns)
            pvals = _pvalues_from_tvalues(tvals, model.df_resid).reindex(df.index)
        else:
            pvals = pd.DataFrame(index=df.index, columns=exog.columns, dtype=float)

        betas[driver] = params[driver]
        signif_pct[driver] = (1 - pvals[driver]) * 100

    return betas, signif_pct


def build_rolling_maps(
    df_map: Dict[str, pd.DataFrame],
    y_col_map: Dict[str, str] | None = None,
    window: int = 250,
    min_obs: int | None = None,
    drop_bbdxy: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Build rolling betas and significance maps for each currency."""
    if y_col_map is None:
        y_col_map = DEFAULT_Y_COL_MAP

    betas_map: Dict[str, pd.DataFrame] = {}
    signif_map: Dict[str, pd.DataFrame] = {}

    for currency, df in df_map.items():
        if drop_bbdxy and "BBDXY" in df.columns:
            df = df.drop(columns=["BBDXY"])

        y_col = y_col_map.get(currency)
        if y_col is None:
            raise KeyError(f"No default y_col found for {currency}.")

        betas_df, signif_df = rolling_univariate_ols(
            df, y_col=y_col, window=window, min_obs=min_obs
        )
        betas_map[currency] = betas_df
        signif_map[currency] = signif_df

    return betas_map, signif_map


if __name__ == "__main__":
    betas_map, signif_map = build_rolling_maps(ultimate_df, window=250)
    sample = "nok"
    if sample in betas_map:
        print(f"{sample} betas (tail):")
        print(betas_map[sample].tail())
        print(f"\n{sample} significance % (tail):")
        print(signif_map[sample].tail())
