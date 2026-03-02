# Package marker for src.data

# re-export convenient names from the loader module so that clients can write
# ``from src.data import fx_df`` rather than digging into the submodule.
from .load_excel_sheets import *  # noqa: F401,F403
