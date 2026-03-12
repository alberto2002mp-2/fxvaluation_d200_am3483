"""Standardize FX driver data with rolling z-scores per currency."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

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


def build_standardized_df_map(
    df_map: Dict[str, pd.DataFrame],
    window: int = 250,
    min_periods: int = 100,
    clip_low: float = -4.0,
    clip_high: float = 4.0,
) -> Dict[str, pd.DataFrame]:
    """Return standardized DataFrames with rolling z-scored drivers and original y."""
    standardized_df_map: Dict[str, pd.DataFrame] = {}

    for currency, df in df_map.items():
        if df.shape[1] < 2:
            standardized_df_map[currency] = df.copy()
            continue

        y_col = df.columns[0]
        x_cols = df.columns[1:]

        y = df[y_col]
        X = df[x_cols]

        rolling_mean = X.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = X.rolling(window=window, min_periods=min_periods).std()

        X_z = (X - rolling_mean) / rolling_std
        X_z = X_z.clip(lower=clip_low, upper=clip_high)

        standardized_df = pd.concat([y, X_z], axis=1)
        standardized_df_map[currency] = standardized_df

    return standardized_df_map


if __name__ == "__main__":
    ultimate_df = build_ultimate_df()
    standardized_df_map = build_standardized_df_map(ultimate_df)

    # Example quick check
    for sample in ("aud", "nok"):
        if sample in standardized_df_map:
            print(f"\n{sample} standardized (tail):")
            print(standardized_df_map[sample].tail())
