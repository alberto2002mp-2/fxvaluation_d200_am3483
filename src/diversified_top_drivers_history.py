"""Build diversified top drivers history per currency using categorical diversification."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

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


from src.data.build_ultimate_df import build_ultimate_df
from src.data.standardize_rolling_drivers import build_standardized_df_map
from src.rolling_univariate_ols import build_rolling_maps


CATEGORY_MAP: Dict[str, List[str]] = {
    "Broad USD trend": ["BBDXY", "EMFX Index", "Asia EMFX"],
    "Yield Spreads": [
        "2y yield",
        "5y yield",
        "10y yield",
        "3m1m forward OIS",
        "6m1m forward OIS",
        "1y1m forward OIS",
        "Real 2y yield",
    ],
    "Equities": [
        "Local - S&P500",
        "Local - Wilshire",
        "Local - Dow Jones",
        "MSCI World - S&P500",
        "MSCI World - Wilshire",
        "MSCI World - Dow Jones",
        "MSCI World ex-US",
        "MSCI EM",
        "Local Index",
    ],
    "Commodities": [
        "BCOM Index",
        "Gold",
        "Oil",
        "Coppper COMEX",
        "Copper LME",
        "FN1 Comdty",
        "TZT1 Comdty",
        "NG1 COMB Comdty",
        "Whole Milk",
        "Skimmed Milk",
        "SCOH6 COMB Comdty",
        "IACA COMB Comdty",
        "Crude oil WCS",
    ],
    "Global Risk": ["VIX Index", "MOVE Index", "JPMVG71M Index"],
}


def build_diversified_top_drivers_map(
    betas_map: Dict[str, pd.DataFrame],
    signif_map: Dict[str, pd.DataFrame],
    min_significance: float = 95.0,
    top_n: int = 3,
    betas_raw_map: Dict[str, pd.DataFrame] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Return diversified top drivers per date with standardized and raw betas."""
    out_map: Dict[str, pd.DataFrame] = {}

    name_cols = [f"Driver {i} Name" for i in range(1, top_n + 1)]
    beta_z_cols = [f"Driver {i} Beta Z" for i in range(1, top_n + 1)]
    beta_raw_cols = [f"Driver {i} Normal Beta" for i in range(1, top_n + 1)]
    out_cols = name_cols + beta_z_cols + beta_raw_cols

    for currency, betas_df in betas_map.items():
        signif_df = signif_map.get(currency)
        if signif_df is None:
            continue
        raw_betas_df = None if betas_raw_map is None else betas_raw_map.get(currency)

        out_df = pd.DataFrame(index=betas_df.index, columns=out_cols, dtype=object)

        for dt in betas_df.index:
            row_betas = betas_df.loc[dt]
            row_signif = signif_df.loc[dt]
            row_raw_betas = (
                raw_betas_df.loc[dt]
                if raw_betas_df is not None and dt in raw_betas_df.index
                else pd.Series(dtype=float)
            )

            sig_mask = row_signif >= min_significance
            sig_betas = row_betas[sig_mask].dropna()

            if sig_betas.empty:
                out_df.loc[dt] = [np.nan] * len(out_cols)
                continue

            category_winners = {}
            for category, drivers in CATEGORY_MAP.items():
                available = [d for d in drivers if d in sig_betas.index]
                if not available:
                    continue

                cat_betas = sig_betas.loc[available]
                winner = cat_betas.abs().sort_values(ascending=False).index[0]
                category_winners[winner] = cat_betas[winner]

            if not category_winners:
                out_df.loc[dt] = [np.nan] * len(out_cols)
                continue

            winners_series = pd.Series(category_winners)
            top_idx = winners_series.abs().sort_values(ascending=False).head(top_n).index

            top_names = list(top_idx) + [np.nan] * (top_n - len(top_idx))
            top_betas_z = [winners_series.get(name) for name in top_idx] + [
                np.nan
            ] * (top_n - len(top_idx))
            top_betas_raw = [
                row_raw_betas.get(name, np.nan) if pd.notna(name) else np.nan
                for name in top_names
            ]

            out_df.loc[dt] = top_names + top_betas_z + top_betas_raw

        out_map[currency] = out_df

    return out_map


if __name__ == "__main__":
    ultimate_df = build_ultimate_df()
    standardized_df_map = build_standardized_df_map(ultimate_df)
    betas_map, signif_map = build_rolling_maps(standardized_df_map, window=250)
    betas_raw_map, _ = build_rolling_maps(ultimate_df, window=250)

    diversified_top_drivers_map = build_diversified_top_drivers_map(
        betas_map,
        signif_map,
        min_significance=95.0,
        top_n=3,
        betas_raw_map=betas_raw_map,
    )

    for sample in ("aud", "nok"):
        if sample in diversified_top_drivers_map:
            print(f"\n{sample} diversified top drivers history (tail):")
            print(diversified_top_drivers_map[sample].tail())
        else:
            print(f"\n[warn] {sample} not found in diversified_top_drivers_map.")
