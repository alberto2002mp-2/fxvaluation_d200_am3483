"""Build top drivers history for each currency based on rolling betas/significance."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

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


from src.rolling_univariate_ols import build_rolling_maps
from src.data.build_ultimate_df import build_ultimate_df


def build_top_drivers_history_map(
    betas_map: Dict[str, pd.DataFrame],
    signif_map: Dict[str, pd.DataFrame],
    min_significance: float = 95.0,
    top_n: int = 3,
) -> Dict[str, pd.DataFrame]:
    """Return a map of DataFrames with top driver names/betas per date."""
    top_map: Dict[str, pd.DataFrame] = {}

    name_cols = [f"Driver {i} Name" for i in range(1, top_n + 1)]
    beta_cols = [f"Driver {i} Beta" for i in range(1, top_n + 1)]
    out_cols = name_cols + beta_cols

    for currency, betas_df in betas_map.items():
        signif_df = signif_map.get(currency)
        if signif_df is None:
            continue

        out_df = pd.DataFrame(index=betas_df.index, columns=out_cols, dtype=object)

        for dt in betas_df.index:
            row_betas = betas_df.loc[dt]
            row_signif = signif_df.loc[dt]

            sig_mask = row_signif >= min_significance
            sig_betas = row_betas[sig_mask].dropna()

            if sig_betas.empty:
                out_df.loc[dt] = [np.nan] * (2 * top_n)
                continue

            top_idx = sig_betas.abs().sort_values(ascending=False).head(top_n).index
            top_names = list(top_idx) + [np.nan] * (top_n - len(top_idx))
            top_betas = [sig_betas.get(name) for name in top_idx] + [
                np.nan
            ] * (top_n - len(top_idx))

            out_df.loc[dt] = top_names + top_betas

        top_map[currency] = out_df

    return top_map


if __name__ == "__main__":
    ultimate_df = build_ultimate_df()
    betas_map, signif_map = build_rolling_maps(ultimate_df, window=250)

    top_drivers_history_map = build_top_drivers_history_map(
        betas_map, signif_map, min_significance=95.0, top_n=3
    )

    for sample in ("aud", "nok"):
        if sample in top_drivers_history_map:
            print(f"\n{sample} top drivers history (tail):")
            print(top_drivers_history_map[sample].tail())
        else:
            print(f"\n[warn] {sample} not found in top_drivers_history_map.")
