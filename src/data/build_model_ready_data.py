"""Build master training CSVs and load model-ready slices by currency."""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Dict

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.data.build_ultimate_df import build_ultimate_df
from src.data.create_dataframes import fx_df as raw_fx_df
from src.data.standardize_rolling_drivers import build_standardized_df_map
from src.diversified_top_drivers_history import build_diversified_top_drivers_map
from src.rolling_multivariate_fair_value import RAW_PRICE_COL_MAP
from src.rolling_univariate_ols import DEFAULT_Y_COL_MAP, build_rolling_maps


DEFAULT_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


def _sanitize_column_name(name: str) -> str:
    """Convert human-readable column names into stable snake_case names."""
    clean = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean.lower()


def _suffix_feature_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Return a copy with sanitized feature names plus the requested suffix."""
    renamed = {
        col: f"{_sanitize_column_name(col)}_{suffix}"
        for col in df.columns
    }
    return df.rename(columns=renamed)


def _sanitize_top_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the top-driver map with sanitized column names."""
    return df.rename(columns={col: _sanitize_column_name(col) for col in df.columns})


def build_master_training_map(
    window: int = 250,
    min_significance: float = 95.0,
    top_n: int = 3,
) -> Dict[str, pd.DataFrame]:
    """Build one master training DataFrame per currency."""
    ultimate_df = build_ultimate_df()
    standardized_df_map = build_standardized_df_map(ultimate_df, window=window)
    betas_mapz, signif_mapz = build_rolling_maps(standardized_df_map, window=window)
    betas_raw, _ = build_rolling_maps(ultimate_df, window=window)
    top_mapz = build_diversified_top_drivers_map(
        betas_mapz,
        signif_mapz,
        min_significance=min_significance,
        top_n=top_n,
        betas_raw_map=betas_raw,
    )

    master_map: Dict[str, pd.DataFrame] = {}

    for currency, raw_df in ultimate_df.items():
        y_col = DEFAULT_Y_COL_MAP.get(currency)
        raw_price_col = RAW_PRICE_COL_MAP.get(currency)
        std_df = standardized_df_map.get(currency)
        top_df = top_mapz.get(currency)

        if y_col is None:
            raise KeyError(f"No target column configured for {currency}.")
        if raw_price_col is None:
            raise KeyError(f"No raw price column configured for {currency}.")
        if std_df is None:
            raise KeyError(f"No standardized dataframe found for {currency}.")
        if raw_price_col not in raw_fx_df.columns:
            raise KeyError(f"{raw_price_col} not found in raw FX dataframe.")

        log_return = raw_df[[y_col]].rename(columns={y_col: "Log_Return"})
        actual_price = raw_fx_df[[raw_price_col]].rename(columns={raw_price_col: "Actual_Price"})

        raw_features = _suffix_feature_columns(raw_df.drop(columns=[y_col]), "raw")
        std_features = _suffix_feature_columns(std_df.drop(columns=[y_col]), "std")
        frames = [actual_price, log_return, raw_features, std_features]

        if top_df is not None:
            frames.append(_sanitize_top_map_columns(top_df))

        master_df = pd.concat(frames, axis=1, join="outer").sort_index()
        master_df.index.name = "Date"
        master_map[currency] = master_df

    return master_map


def save_master_training_csvs(
    output_dir: str | Path = DEFAULT_PROCESSED_DIR,
    window: int = 250,
    min_significance: float = 95.0,
    top_n: int = 3,
) -> Dict[str, pd.DataFrame]:
    """Build and save one master training CSV per currency."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    master_map = build_master_training_map(
        window=window,
        min_significance=min_significance,
        top_n=top_n,
    )

    for currency, df in master_map.items():
        df.to_csv(output_path / f"{currency}_master.csv", index=True)

    return master_map


def get_model_ready_data(
    currency: str,
    model_type: str = "linear",
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """
    Load a saved master CSV and return the feature slice for the requested model type.

    `linear` -> standardized features (`_std`)
    `gbm` -> raw features (`_raw`)
    Always returns `Actual_Price` and `Log_Return`.
    """
    csv_path = Path(processed_dir) / f"{currency}_master.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} does not exist. Run `save_master_training_csvs()` first."
        )

    df = pd.read_csv(csv_path, index_col="Date", parse_dates=["Date"])
    target_cols = ["Actual_Price", "Log_Return"]

    if model_type == "linear":
        feature_cols = [col for col in df.columns if col.endswith("_std")]
    elif model_type == "gbm":
        feature_cols = [col for col in df.columns if col.endswith("_raw")]
    else:
        raise ValueError("model_type must be 'linear' or 'gbm'.")

    keep_cols = target_cols + feature_cols
    return df.loc[:, keep_cols]


if __name__ == "__main__":
    master_map = save_master_training_csvs()
    for currency in ("eur", "aud"):
        if currency in master_map:
            print(f"\nSaved {currency}_master.csv with shape {master_map[currency].shape}")
