# src/data/load_excel_sheets.py
"""Utilities for loading Bloomberg-style paired-column Excel sheets.

Each sheet (except for one named "Summary") contains a collection of time
series arranged in two-column blocks: a date column with a variable name in
the first row, followed by a value column.  Different series start on different
dates and are to be merged with an outer join on the date index.

This module exposes a helper to load every sheet into a dictionary of
:class:`pandas.DataFrame` objects as well as module-level variables such as
``fx_df`` and ``usd_df`` for convenience.

The code assumes ``pandas`` is installed (listed in ``requirements.txt``) and
uses relative paths so it can be executed identically on any machine.
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Dict

import pandas as pd


def _parse_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Convert a raw dataframe read with ``header=None`` into a merged time
    series.

    Parameters
    ----------
    df : pd.DataFrame
        The output from ``pd.read_excel(..., header=None)`` for one sheet.
    sheet_name : str
        Used for warning messages.

    Returns
    -------
    pd.DataFrame
        DataFrame whose columns are the variable names and whose index is a
        ``DatetimeIndex``.  Rows are sorted and missing values are preserved.
    """
    series_frames: list[pd.DataFrame] = []
    seen_names: dict[str, int] = {}

    # iterate over columns two at a time
    ncols = df.shape[1]
    for i in range(0, ncols, 2):
        date_col = df.iloc[:, i]
        if i + 1 >= ncols:
            # odd number of columns, ignore trailing column
            break
        value_col = df.iloc[:, i + 1]

        varname = date_col.iat[0]
        if pd.isna(varname):
            # skip blank header
            continue
        varname = str(varname)

        # ensure unique column names if sheet happens to contain duplicates
        count = seen_names.get(varname, 0)
        if count:
            unique_name = f"{varname}_{count}"
        else:
            unique_name = varname
        seen_names[varname] = count + 1

        dates = date_col.iloc[1:]
        values = value_col.iloc[1:]

        # attempt to parse dates, warn if some cannot be converted
        parsed = pd.to_datetime(dates, errors="coerce")
        if parsed.isna().any():
            warnings.warn(
                f"Unable to parse some dates in sheet '{sheet_name}', "
                f"column '{varname}'; those rows will be NaT",
                UserWarning,
            )

        temp = pd.DataFrame({unique_name: values.values}, index=parsed.values)
        # drop duplicate dates within the series (keep first occurrence).
        if temp.index.duplicated().any():
            temp = temp[~temp.index.duplicated(keep="first")]
        series_frames.append(temp)

    if series_frames:
        merged = pd.concat(series_frames, axis=1, join="outer")
        merged.sort_index(inplace=True)
        # ensure DatetimeIndex and timezone naive
        merged.index = pd.DatetimeIndex(merged.index.values)
        return merged
    else:
        # empty sheet or no usable pairs
        return pd.DataFrame()


def load_all_sheets(
    excel_path: str | pathlib.Path = "data/rawdata.xlsx",
) -> Dict[str, pd.DataFrame]:
    """Load every sheet from the provided Excel file except for "Summary".

    Parameters
    ----------
    excel_path : str or pathlib.Path, optional
        Path to the workbook.  Defaults to ``data/rawdata.xlsx`` relative to the
        current working directory.  Historically the raw workbook lived under a
        top-level ``data/`` folder, but some users prefer ``raw_data/`` to
        avoid confusion with the package namespace.  If the given path does not
        exist we also attempt the same filename inside ``raw_data/`` before
        raising ``FileNotFoundError``.
    """
    """Load every sheet from the provided Excel file except for "Summary".

    Parameters
    ----------
    excel_path : str or pathlib.Path, optional
        Path to the workbook.  Defaults to ``data/rawdata.xlsx`` relative to the
        current working directory.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from sheet name to the assembled DataFrame for that sheet.
    """
    excel_path = pathlib.Path(excel_path)
    if not excel_path.exists():
        # try the alternate raw_data directory if the default path was used
        alt = excel_path.parent.parent / "raw_data" / excel_path.name
        if alt.exists():
            excel_path = alt
        else:
            raise FileNotFoundError(f"Excel file not found at {excel_path}")

    # read all sheets without inferring headers so the first row is row 0
    all_sheets = pd.read_excel(excel_path, sheet_name=None, header=None)
    results: dict[str, pd.DataFrame] = {}

    for name, raw in all_sheets.items():
        if name == "Summary":
            continue
        df = _parse_sheet(raw, name)
        results[name] = df

    return results


# load sheets at import time so module-level variables are available
_sheets = load_all_sheets()

# create convenient module-level references (fx_df, usd_df, etc.)
for sheet_name, df in _sheets.items():
    var_name = f"{sheet_name.lower()}_df"
    globals()[var_name] = df


if __name__ == "__main__":
    sheets = load_all_sheets()
    print("loaded sheets:", list(sheets.keys()))
    for name, df in sheets.items():
        idx = df.index
        lo = idx.min() if not idx.empty else None
        hi = idx.max() if not idx.empty else None
        print(
            f"{name!r}: shape={df.shape}, dates=({lo}, {hi})",
        )
