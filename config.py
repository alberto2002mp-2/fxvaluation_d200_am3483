# config.py
"""Project-wide configuration for paths and environment settings.

This module centralizes all path definitions to ensure consistent file resolution
across different execution environments (scripts, notebooks, tests, etc.).
"""

from pathlib import Path

# Compute the project root as the directory containing this file
PROJ_ROOT = Path(__file__).resolve().parent

# Data paths
RAW_EXCEL_PATH = PROJ_ROOT / "data" / "rawdata.xlsx"

# Notebooks directory
NOTEBOOKS_DIR = PROJ_ROOT / "notebooks"

# Source code directory
SRC_DIR = PROJ_ROOT / "src"
