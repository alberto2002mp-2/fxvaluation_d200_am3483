import numpy as np
import pandas as pd
from IPython.display import display
import statsmodels.api as sm

from pathlib import Path
import sys

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

ultimate_df = build_ultimate_df()


# Drop "BBDXY" column from all dataframes in ultimate_df if present
for key, df in ultimate_df.items():
    if "BBDXY" in df.columns:
        ultimate_df[key] = df.drop(columns=["BBDXY"])


pd.set_option("display.float_format", "{:,.4f}".format)

DEFAULT_Y_COL_MAP = {
    "eur": "EURUSD",
    "gbp": "GBPUSD",
    "jpy": "USDJPY",
    "chf": "USDCHF",
    "cad": "USDCAD",
    "aud": "AUDUSD",
    "nzd": "NZDUSD",
    "nok": "USDNOK",
    "sek": "USDSEK",
}

def latest_window_multivariate_ols(df: pd.DataFrame, y_col: str, window: int = 252):
    """Fit multivariate OLS on the most recent rolling window and return (summary, r2)."""
    if y_col not in df.columns:
        raise KeyError(f"{y_col} not found in dataframe columns.")

    window_df = df.tail(window).dropna()
    if len(window_df) < 2:
        raise ValueError("Not enough non-NaN rows in the latest rolling window.")

    y = window_df[y_col]
    X = window_df.drop(columns=[y_col])
    X_const = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, X_const).fit()
    r2 = model.rsquared

    betas = model.params.drop("const")
    pvals = model.pvalues.drop("const")
    signif_pct = (1 - pvals) * 100

    ranked_idx = betas.abs().sort_values(ascending=False).index
    betas = betas.reindex(ranked_idx)
    signif_pct = signif_pct.reindex(ranked_idx)

    summary = pd.DataFrame({
        "Driver Name": betas.index,
        "Beta Coefficient": betas.values,
        "Rank": np.arange(1, len(betas) + 1),
        "Significance %": signif_pct.values,
    })
    summary["Beta Coefficient"] = summary["Beta Coefficient"].round(4)
    summary["Significance %"] = summary["Significance %"].round(2)
    return summary, r2

def run_processed_df_regression(df_name: str, y_col: str | None = None, window: int = 252):
    """Convenience wrapper for ultimate_df by name."""
    if df_name not in ultimate_df:
        raise KeyError(f"{df_name} not found in ultimate_df.")

    if y_col is None:
        y_col = DEFAULT_Y_COL_MAP.get(df_name)
        if y_col is None:
            raise KeyError(f"No default y_col found for {df_name}.")

    summary, r2 = latest_window_multivariate_ols(ultimate_df[df_name], y_col, window=window)
    display(summary)
    print(f"Current 1Y Rolling R2: {r2:.4f}")
    return summary, r2


def run_processed_df_regression_sig_only(
    df_name: str,
    y_col: str | None = None,
    window: int = 252,
    min_significance: float = 95.0,
):
    """Same as run_processed_df_regression but filters to >= min_significance."""
    summary, r2 = run_processed_df_regression(df_name, y_col=y_col, window=window)
    filtered = summary[summary["Significance %"] >= min_significance].reset_index(drop=True)
    display(filtered)
    print(f"Current 1Y Rolling R2: {r2:.4f}")
    return filtered, r2


