"""Build ultimate_df by combining processed data with MOVE Index levels."""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, Iterable

import pandas as pd

# Ensure project root is on sys.path (src/data/build_ultimate_df.py -> project root)
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.statistical_audit import df_processed, level_series


CURRENCIES: list[str] = [
    "eur",
    "gbp",
    "jpy",
    "chf",
    "cad",
    "aud",
    "nzd",
    "nok",
    "sek",
]


def build_ultimate_df(
    currencies: Iterable[str] = CURRENCIES,
) -> Dict[str, pd.DataFrame]:
    """Create a dict of processed DataFrames with MOVE Index appended."""
    ultimate_df: Dict[str, pd.DataFrame] = {}

    for currency in currencies:
        proc_key = f"{currency}_df_processed"
        if proc_key not in df_processed:
            print(f"[warn] Missing df_processed for {currency}.")
            continue
        if currency not in level_series:
            print(f"[warn] Missing level_series for {currency}.")
            continue

        df_proc = df_processed[proc_key].copy()
        drop_cols = []
        if "USTs Volatility" in df_proc.columns:
            drop_cols.append("USTs Volatility")
        if "G10 FX Volatility" in df_proc.columns:
            drop_cols.append("G10 FX Volatility")
        if "NY Natural Gas" in df_proc.columns:
            drop_cols.append("NY Natural Gas")
        if "TTF Natural Gas" in df_proc.columns:
            drop_cols.append("TTF Natural Gas")
        if "UK Natural Gas" in df_proc.columns:
            drop_cols.append("UK Natural Gas")
        if "VIX" in df_proc.columns:
            drop_cols.append("VIX")
        if currency == "aud" and "Iron ore" in df_proc.columns:
            drop_cols.append("Iron ore")
        if currency == "aud" and "Coking coal" in df_proc.columns:
            drop_cols.append("Coking coal")
        if drop_cols:
            df_proc = df_proc.drop(columns=drop_cols)

        df_level = level_series[currency]
        required_level_cols = [
            "MOVE Index",
            "JPMVG71M Index",
            "NG1 COMB Comdty",
            "TZT1 Comdty",
            "FN1 Comdty",
            "VIX Index",
        ]
        missing_levels = [col for col in required_level_cols if col not in df_level.columns]
        if missing_levels:
            for col in missing_levels:
                print(f"[warn] Missing '{col}' in level_series for {currency}.")
            continue
        level_cols = df_level[required_level_cols]
        if currency == "aud" and "SCOH6 COMB Comdty" in df_level.columns:
            level_cols = level_cols.join(df_level[["SCOH6 COMB Comdty"]], how="left")
        if currency == "aud" and "IACA COMB Comdty" in df_level.columns:
            level_cols = level_cols.join(df_level[["IACA COMB Comdty"]], how="left")
        df_out = df_proc.join(level_cols, how="left")

        ultimate_df[currency] = df_out

    return ultimate_df



#if __name__ == "__main__":
 #   if "aud" in ultimate_df:
  #      print(ultimate_df["aud"].head())
   # else:
    #    print("[warn] AUD not built.")
