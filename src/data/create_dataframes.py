"""Build cleaned and derived FX dataframes.

This module is import-friendly for notebooks. Importing it will:
1) Load all raw Excel sheets into `frames` (keyed as "<sheet>_df").
2) Apply light column cleanup to select base frames.
3) Build derived frames such as `sek_df2`, and `df2_map`.

If you want to re-run with a different Excel path, call `build_all(...)`.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Dict

import numpy as np
import pandas as pd

# Ensure project root is on sys.path (src/data/create_dataframes.py -> project root)
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data import load_excel_sheets

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def drop_columns(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    """Drop specified columns from a dataframe.

    Raises a ValueError if any requested columns are missing.
    """
    missing = [col for col in columns_to_drop if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataframe: {missing}")
    return df.drop(columns=columns_to_drop)


def load_frames(excel_path: str | pathlib.Path | None = None) -> Dict[str, pd.DataFrame]:
    """Load all sheets and return a dict keyed as '<sheet>_df' (lowercase)."""
    raw = load_excel_sheets.load_all_sheets(excel_path)
    return {f"{name.lower()}_df": df for name, df in raw.items()}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DROP_COLUMNS = {
    "sek_df": [
        "G0021 3M1M BLC2 Curncy",
        "G0021 6M1M BLC2 Curncy",
        "G0021 1Y1M BLC2 Curncy",
    ],
    "nok_df": [
        "G0078 3M1M BLC2 Curncy",
        "G0078 6M1M BLC2 Curncy",
        "G0078 1Y1M BLC2 Curncy",
    ],
    "aud_df": [
        "G0001 3M1M BLC2 Curncy",
        "G0001 6M1M BLC2 Curncy",
        "G0001 1Y1M BLC2 Curncy",
    ],
    "nzd_df": [
        "G0049 3M1M BLC2 Curncy",
        "G0049 6M1M BLC2 Curncy",
        "G0049 1Y1M BLC2 Curncy",
    ],
    "chf_df": [
        "G0082 3M1M BLC2 Curncy",
        "G0082 6M1M BLC2 Curncy",
        "G0082 1Y1M BLC2 Curncy",
    ],
    "jpy_df": [
        "G0018 3M1M BLC2 Curncy",
        "G0018 6M1M BLC2 Curncy",
        "G0018 1Y1M BLC2 Curncy",
    ],
}

DF2_DROP_COLUMNS = [
    "S0042FS 3M1M BLC Curncy",
    "S0042FS 6M1M BLC Curncy",
    "S0042FS 1Y1M BLC Curncy",
]
DF2_DROP_CURRENCIES = {"sek", "nok", "aud", "nzd", "chf", "jpy"}

FX_COMPONENT_COLS = [
    "EURUSD Curncy",
    "USDJPY Curncy",
    "USDCAD Curncy",
    "GBPUSD Curncy",
    "USDMXN Curncy",
    "USDCNH Curncy",
    "USDCHF Curncy",
    "AUDUSD Curncy",
    "USDKRW Curncy",
    "USDINR Curncy",
    "USDSGD Curncy",
    "USDTWD Curncy",
]
FX_COMPONENT_WEIGHTS = np.array(
    [0.2947, 0.1238, 0.1165, 0.1027, 0.0962, 0.0700, 0.0447, 0.0439, 0.0316, 0.0283, 0.0261, 0.0215]
)

CURRENCY_SPECS = {
    "eur": {"frame": "eur_df", "fx_cols": ["EURUSD Curncy", "BBDXY_ex_EUR", "EMFXDBET Index"]},
    "gbp": {"frame": "gbp_df", "fx_cols": ["GBPUSD Curncy", "BBDXY_ex_GBP", "EMFXDBET Index"]},
    "jpy": {"frame": "jpy_df", "fx_cols": ["USDJPY Curncy", "BBDXY_ex_JPY", "CTTWBUSA Index"]},
    "chf": {"frame": "chf_df", "fx_cols": ["USDCHF Curncy", "BBDXY_ex_CHF", "EMFXDBET Index"]},
    "cad": {"frame": "cad_df", "fx_cols": ["USDCAD Curncy", "BBDXY_ex_CAD", "EMFXDBET Index"]},
    "aud": {"frame": "aud_df", "fx_cols": ["AUDUSD Curncy", "BBDXY_ex_AUD", "CTTWBUSA Index"]},
    "nzd": {"frame": "nzd_df", "fx_cols": ["NZDUSD Curncy", "BBDXY Curncy", "CTTWBUSA Index"]},
    "nok": {"frame": "nok_df", "fx_cols": ["USDNOK Curncy", "BBDXY Curncy", "EMFXDBET Index"]},
    "sek": {"frame": "sek_df", "fx_cols": ["USDSEK Curncy", "BBDXY Curncy", "EMFXDBET Index"]},
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _apply_base_drops(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    cleaned = dict(frames)
    for name, cols in BASE_DROP_COLUMNS.items():
        if name in cleaned:
            cleaned[name] = drop_columns(cleaned[name], cols)
    return cleaned


def build_fx_newdf(fx_df: pd.DataFrame) -> pd.DataFrame:
    """Create fx_df with synthetic BBDXY-ex indices appended."""
    required = ["BBDXY Curncy", *FX_COMPONENT_COLS]
    missing = [c for c in required if c not in fx_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in fx_df: {missing}")

    bbdxy_index = fx_df["BBDXY Curncy"].dropna().index
    px = fx_df[FX_COMPONENT_COLS].reindex(bbdxy_index)

    log_px = np.log(px)
    signs = pd.Series([1 if c.startswith("USD") else -1 for c in FX_COMPONENT_COLS], index=FX_COMPONENT_COLS)
    r_df = log_px.diff().mul(signs, axis=1).fillna(0.0)

    w = pd.Series(FX_COMPONENT_WEIGHTS, index=FX_COMPONENT_COLS)
    exclusions = ["EURUSD Curncy", "USDJPY Curncy", "USDCHF Curncy", "GBPUSD Curncy", "AUDUSD Curncy", "USDCAD Curncy"]

    w_ex = pd.DataFrame(
        np.tile(w.values, (len(exclusions), 1)).T,
        index=FX_COMPONENT_COLS,
        columns=exclusions,
    )
    for excl in exclusions:
        w_ex.loc[excl, excl] = 0

    def _normalize_weights(weights: pd.DataFrame, cnh_active: bool) -> pd.DataFrame:
        w_norm = weights.copy()
        if not cnh_active:
            w_norm.loc["USDCNH Curncy", :] = 0
        return w_norm.div(w_norm.sum(axis=0), axis=1)

    cnh_start = pd.Timestamp("2010-08-23")
    pre_mask = r_df.index < cnh_start
    post_mask = r_df.index >= cnh_start

    w_pre = _normalize_weights(w_ex, cnh_active=False)
    w_post = _normalize_weights(w_ex, cnh_active=True)

    r_pre = r_df.loc[pre_mask].dot(w_pre)
    r_post = r_df.loc[post_mask].dot(w_post)
    r_all = pd.concat([r_pre, r_post]).sort_index()

    base_level = fx_df["BBDXY Curncy"].dropna().iloc[0]
    idx_all = base_level * np.exp(r_all.cumsum())

    idx_all.columns = [
        "BBDXY_ex_EUR",
        "BBDXY_ex_JPY",
        "BBDXY_ex_CHF",
        "BBDXY_ex_GBP",
        "BBDXY_ex_AUD",
        "BBDXY_ex_CAD",
    ]

    return pd.concat([fx_df, idx_all], axis=1)


def build_df2_map(frames: Dict[str, pd.DataFrame], fx_newdf: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build df2_map and also populate '<cur>_df2' globals."""
    if "usd_df" not in frames:
        raise KeyError("usd_df not found in frames.")

    required_fx_cols = {c for spec in CURRENCY_SPECS.values() for c in spec["fx_cols"]}
    missing_fx = sorted(required_fx_cols - set(fx_newdf.columns))
    if missing_fx:
        raise ValueError(f"Missing required columns in fx_newdf: {missing_fx}")

    df2_map: Dict[str, pd.DataFrame] = {}
    usd_df = frames["usd_df"]

    for cur, spec in CURRENCY_SPECS.items():
        base_key = spec["frame"]
        if base_key not in frames:
            raise KeyError(f"{base_key} not found in frames.")

        df2 = pd.concat([frames[base_key], usd_df], axis=1)
        df2 = pd.concat([df2, fx_newdf[spec["fx_cols"]]], axis=1)

        if cur in DF2_DROP_CURRENCIES:
            drop_cols = [c for c in DF2_DROP_COLUMNS if c in df2.columns]
            if drop_cols:
                df2 = df2.drop(columns=drop_cols)

        name = f"{cur}_df2"
        df2_map[name] = df2
        df2_map[cur] = df2
        globals()[name] = df2

    return df2_map


def build_all(excel_path: str | pathlib.Path | None = None):
    """Build and return (frames, fx_newdf, df2_map)."""
    frames = _apply_base_drops(load_frames(excel_path))
    fx_newdf = build_fx_newdf(frames["fx_df"])
    df2_map = build_df2_map(frames, fx_newdf)
    return frames, fx_newdf, df2_map


# ---------------------------------------------------------------------------
# Module-level build (import-friendly defaults)
# ---------------------------------------------------------------------------

frames, fx_newdf, df2_map = build_all()

# Convenience module-level variables (kept for backward compatibility)
fx_df = frames["fx_df"]
usd_df = frames["usd_df"]
eur_df = frames["eur_df"]
gbp_df = frames["gbp_df"]
cad_df = frames["cad_df"]
aud_df = frames["aud_df"]
nzd_df = frames["nzd_df"]
nok_df = frames["nok_df"]
sek_df = frames["sek_df"]
chf_df = frames["chf_df"]
jpy_df = frames["jpy_df"]

__all__ = [
    "frames",
    "fx_newdf",
    "df2_map",
    "fx_df",
    "usd_df",
    "eur_df",
    "gbp_df",
    "cad_df",
    "aud_df",
    "nzd_df",
    "nok_df",
    "sek_df",
    "chf_df",
    "jpy_df",
    "build_all",
    "build_fx_newdf",
    "build_df2_map",
    "load_frames",
    "drop_columns",
]
