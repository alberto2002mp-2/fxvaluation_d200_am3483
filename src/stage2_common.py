"""Shared Stage 2 helpers for data loading, feature selection, and metrics."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_OUTPUT_DIR = DATA_DIR / "model_outputs"
DEFAULT_AUDIT_DIR = DATA_DIR / "audits"
DEFAULT_CURRENCIES = ("eur", "gbp", "aud", "nzd", "cad", "jpy", "chf", "nok", "sek")
TOP_DRIVER_NAME_COLUMNS = ("driver_1_name", "driver_2_name", "driver_3_name")
TOP_DRIVER_ALIAS_MAP = {
    "driver_1_name": ("driver_1_name", "Driver 1 Name"),
    "driver_2_name": ("driver_2_name", "Driver 2 Name"),
    "driver_3_name": ("driver_3_name", "Driver 3 Name"),
    "driver_1_beta_z": ("driver_1_beta_z", "Driver 1 Beta Z"),
    "driver_2_beta_z": ("driver_2_beta_z", "Driver 2 Beta Z"),
    "driver_3_beta_z": ("driver_3_beta_z", "Driver 3 Beta Z"),
    "driver_1_normal_beta": ("driver_1_normal_beta", "Driver 1 Normal Beta"),
    "driver_2_normal_beta": ("driver_2_normal_beta", "Driver 2 Normal Beta"),
    "driver_3_normal_beta": ("driver_3_normal_beta", "Driver 3 Normal Beta"),
}


def sanitize_column_name(name: str) -> str:
    """Convert a human-readable macro driver label into a stable feature name.

    This keeps rolling-factor columns portable across CSV round-trips and model
    families that expect deterministic feature names.

    Args:
        name: Original driver label as stored in the source workbook or ranking map.

    Returns:
        A lowercase snake_case column label safe for persisted model features.
    """
    clean = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean.lower()


def compute_days_in_signal(error_z: pd.Series, threshold: float = 2.0) -> pd.Series:
    """Count consecutive days spent in the same signal regime.

    The routine segments the standardized fair-value gap into BUY, SELL, and
    NEUTRAL regimes so the audit layer can distinguish a fresh trading trigger
    from a continuing signal.

    Args:
        error_z: Rolling z-score series for the fair-value gap.
        threshold: Absolute trigger level used to define the active regime.

    Returns:
        A series aligned to ``error_z`` with the run length of the current regime.
    """
    regime = pd.Series(0, index=error_z.index, dtype=int)
    regime = regime.where(~(error_z > threshold), 1)
    regime = regime.where(~(error_z < -threshold), -1)

    days = pd.Series(0, index=error_z.index, dtype=int)
    run = 0
    prev_regime = 0
    for dt, current_regime in regime.items():
        if current_regime == 0:
            run = 0
            prev_regime = 0
        elif current_regime == prev_regime:
            run += 1
        else:
            run = 1
            prev_regime = current_regime
        days.at[dt] = run

    return days


def adjusted_r2(r2: float, n_obs: int, n_features: int) -> float:
    """Compute adjusted R-squared for a rolling regression window.

    Adjusted R-squared penalizes models that overfit by adding weak explanatory
    variables, which is useful when the active driver set changes through time.

    Args:
        r2: In-sample coefficient of determination.
        n_obs: Number of observations used to estimate the model.
        n_features: Number of active predictors in the fitted specification.

    Returns:
        The adjusted R-squared, or ``np.nan`` when the statistic is undefined.
    """
    if n_obs <= n_features + 1:
        return np.nan
    return 1 - (1 - r2) * ((n_obs - 1) / (n_obs - n_features - 1))


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Compute root mean squared error for model validation.

    RMSE is used throughout Stage 2 to compare forecast quality on both rolling
    training windows and out-of-sample validation folds.

    Args:
        y_true: Observed target values.
        y_pred: Predicted target values from a fitted learner.

    Returns:
        The scalar RMSE between the observed and predicted values.
    """
    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(np.square(true_array - pred_array))))


def load_master_training_data(
    currency: str,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    required_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load a persisted master training table for one currency.

    The master table is the canonical Stage 2 modeling input containing the FX
    level, log return target, sanitized feature matrix, and rolling driver ranks.

    Args:
        currency: ISO-like currency key used in the processed master CSV name.
        processed_dir: Directory containing the saved ``*_master.csv`` files.
        required_cols: Optional columns that must be present after loading.

    Returns:
        A date-indexed master training DataFrame sorted in chronological order.

    Raises:
        FileNotFoundError: If the master CSV is missing.
        KeyError: If any required columns are absent.
    """
    csv_path = Path(processed_dir) / f"{currency.lower()}_master.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} does not exist. Build the master CSVs before running Stage 2."
        )

    df = pd.read_csv(csv_path, index_col="Date", parse_dates=["Date"]).sort_index()
    df = df.loc[df.index.notna()].copy()

    required = tuple(required_cols or ())
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required master CSV columns for {currency}: {missing}")

    return df


def load_model_ready_data(
    currency: str,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    feature_suffix: str = "raw",
) -> pd.DataFrame:
    """Load a Stage 2-ready subset of the master table for one model family.

    This extracts the target, realized FX level, driver metadata, and the
    appropriate feature representation for either raw tree inputs or
    standardized linear-model inputs.

    Args:
        currency: ISO-like currency key used in the processed master CSV name.
        processed_dir: Directory containing the saved ``*_master.csv`` files.
        feature_suffix: Feature family suffix, typically ``"raw"`` or ``"std"``.

    Returns:
        A modeling DataFrame containing the target columns, requested features,
        and aliased top-driver metadata needed by the audit layer.
    """
    df = load_master_training_data(
        currency=currency,
        processed_dir=processed_dir,
        required_cols=("Actual_Price", "Log_Return"),
    )

    feature_cols = [col for col in df.columns if col.endswith(f"_{feature_suffix}")]
    out = df.loc[:, ["Actual_Price", "Log_Return", *feature_cols]].copy()

    for alias_col, stored_candidates in TOP_DRIVER_ALIAS_MAP.items():
        for stored_col in stored_candidates:
            if stored_col in df.columns:
                out[alias_col] = df[stored_col]
                break

    return out


def top_driver_names(
    df: pd.DataFrame,
    current_date: pd.Timestamp,
) -> list[object]:
    """Return the ranked driver labels available on a given date.

    Args:
        df: Master training or audit DataFrame containing top-driver metadata.
        current_date: Evaluation date whose active drivers should be inspected.

    Returns:
        The ordered list of stored top-driver labels for the selected date.
    """
    return [
        df.at[current_date, column]
        for column in TOP_DRIVER_NAME_COLUMNS
        if column in df.columns
    ]


def active_feature_names(
    df: pd.DataFrame,
    current_date: pd.Timestamp,
    feature_suffix: str,
) -> tuple[list[str], list[str]]:
    """Resolve the active ranked drivers into valid model feature columns.

    The Stage 2 pipeline re-estimates models with the currently dominant macro
    drivers, so this helper aligns the rolling driver ranking with the feature
    matrix required by the selected learner.

    Args:
        df: Master training or audit DataFrame containing driver metadata.
        current_date: Evaluation date in the walk-forward loop.
        feature_suffix: Feature family suffix used by the target learner.

    Returns:
        A tuple of ``(driver_names, feature_columns)`` limited to non-null,
        non-duplicated, and currently available factors.
    """
    active_names: list[str] = []
    active_cols: list[str] = []

    for name in top_driver_names(df=df, current_date=current_date):
        if pd.isna(name):
            continue

        driver_name = str(name)
        feature_col = f"{sanitize_column_name(driver_name)}_{feature_suffix}"
        if feature_col in df.columns and pd.notna(df.at[current_date, feature_col]):
            if feature_col not in active_cols:
                active_cols.append(feature_col)
                active_names.append(driver_name)

    return active_names, active_cols
