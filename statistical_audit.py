"""Statistical audit of FX drivers using ADF and Johansen tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.data.create_dataframes import df2_map, driver_specs_map, processed_df_map


def _adf_pvalue(series: pd.Series, min_obs: int = 20) -> float:
    clean = series.dropna()
    if len(clean) < min_obs:
        return np.nan
    try:
        return float(adfuller(clean, autolag="AIC")[1])
    except Exception:
        return np.nan


def _level_series(
    df_levels: pd.DataFrame, spec: Dict[str, Any]
) -> Optional[pd.Series]:
    if spec["type"] == "single":
        col = spec["col"]
        if col in df_levels.columns:
            return df_levels[col]
        return None
    if spec["type"] == "spread":
        left = spec["left"]
        right = spec["right"]
        if left in df_levels.columns and right in df_levels.columns:
            return df_levels[left] - df_levels[right]
        return None
    raise ValueError(f"Unknown driver spec type: {spec['type']}")


def _johansen_pair(y_level: pd.Series, x_level: pd.Series) -> tuple[float, float, str]:
    pair = pd.concat([y_level, x_level], axis=1).dropna()
    if len(pair) < 20:
        return np.nan, np.nan, "No"

    try:
        res = coint_johansen(pair, det_order=0, k_ar_diff=1)
        trace_stat = float(res.lr1[0])
        crit_95 = float(res.cvt[0, 1])
        is_coint = "Yes" if trace_stat > crit_95 else "No"
        return trace_stat, crit_95, is_coint
    except Exception:
        return np.nan, np.nan, "No"


def run_statistical_audit() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for df_name in sorted(processed_df_map.keys()):
        df_processed = processed_df_map[df_name]
        df2_key = df_name.replace("_df_processed", "_df2")
        if df2_key not in df2_map:
            continue

        df_levels = df2_map[df2_key]
        specs = driver_specs_map.get(df2_key, {})
        if df_processed.empty:
            continue

        y_var = df_processed.columns[0]
        y_spec = specs.get(y_var)
        y_level = _level_series(df_levels, y_spec) if y_spec else None

        currency = df_name.replace("_df_processed", "")

        for var in df_processed.columns:
            spec = specs.get(var)
            if spec is None:
                continue

            level_series = _level_series(df_levels, spec)
            if level_series is None:
                continue

            adf_level = _adf_pvalue(level_series)
            adf_diff = _adf_pvalue(df_processed[var])

            if pd.notna(adf_level) and adf_level <= 0.05:
                order = "I(0)"
            elif pd.notna(adf_level) and adf_level > 0.05 and pd.notna(adf_diff) and adf_diff <= 0.05:
                order = "I(1)"
            else:
                order = "Undetermined"

            trace_stat = np.nan
            crit_95 = np.nan
            is_coint = "No"
            if y_level is not None and var != y_var:
                trace_stat, crit_95, is_coint = _johansen_pair(y_level, level_series)

            rows.append(
                {
                    "Currency": currency,
                    "Variable": var,
                    "ADF p-val (Level)": adf_level,
                    "ADF p-val (Diff)": adf_diff,
                    "Order of Integration": order,
                    "Johansen Trace Stat": trace_stat,
                    "95% Crit Value": crit_95,
                    "Is Cointegrated": is_coint,
                }
            )

    master = pd.DataFrame(rows)

    for currency, df in master.groupby("Currency", sort=True):
        print(f"\n=== {currency.upper()} ===")
        print(df.to_string(index=False))

    out_path = Path("statistical_audit_results.csv")
    master.to_csv(out_path, index=False)
    print(f"\nSaved master summary to {out_path}")

    return master


if __name__ == "__main__":
    run_statistical_audit()

# bash command to run code:
# python statistical_audit.py
