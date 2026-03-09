import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS


def _infer_original_key(df_name, original_df_map):
    if original_df_map is None:
        return None
    if df_name in original_df_map:
        return df_name

    candidates = []
    if df_name.endswith("_df_processed"):
        base = df_name.replace("_df_processed", "")
        candidates += [f"{base}_df2", f"{base}_df", base]
    elif df_name.endswith("_processed"):
        base = df_name.replace("_processed", "")
        candidates += [base, f"{base}_df", f"{base}_df2"]

    for key in candidates:
        if key in original_df_map:
            return key

    return None


def _get_actual_level_series(df_name, y_name, original_df_map):
    if original_df_map is None:
        return None

    key = _infer_original_key(df_name, original_df_map)
    if key is None:
        return None

    orig_df = original_df_map[key]
    if y_name in orig_df.columns:
        return orig_df[y_name]
    return orig_df.iloc[:, 0]


def _reconstruct_level(pred_series, base_value, method="log_return", start_index=None):
    if start_index is None:
        start_index = pred_series.first_valid_index()

    if start_index is None:
        return pd.Series(index=pred_series.index, dtype=float)

    pred_valid = pred_series.loc[start_index:]

    if method == "log_return":
        level = base_value * np.exp(pred_valid.cumsum())
    elif method == "diff":
        level = base_value + pred_valid.cumsum()
    else:
        raise ValueError("method must be 'log_return' or 'diff'")

    return level.reindex(pred_series.index)


def rolling_univariate_ols_predictions(df, window=252, min_obs=None):
    """
    Run univariate RollingOLS for each driver and return predicted changes.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataframe. First column is y, remaining columns are X drivers.
    window : int
        Rolling window size (business days).
    min_obs : int or None
        Minimum observations required after dropping NaNs. If None, uses window.

    Returns
    -------
    pd.DataFrame
        Columns are predicted changes for each driver, aligned to df.index.
    dict
        Params dict: {driver: params_df} with rolling alpha/beta.
    """
    if df.shape[1] < 2:
        raise ValueError("DataFrame must have at least 2 columns (y and X).")

    y = df.iloc[:, 0]
    drivers = df.columns[1:]
    if min_obs is None:
        min_obs = window

    preds = pd.DataFrame(index=df.index)
    params_map = {}

    for driver in drivers:
        data = pd.concat([y, df[driver]], axis=1).dropna()
        if len(data) < min_obs:
            preds[driver] = np.nan
            params_map[driver] = pd.DataFrame(index=df.index, columns=["const", driver], dtype=float)
            continue

        exog = sm.add_constant(data[driver], has_constant="add")
        model = RollingOLS(data[y.name], exog, window=window, min_nobs=min_obs).fit()

        params = model.params.reindex(df.index)
        params_map[driver] = params

        alpha = params["const"]
        beta = params[driver]
        preds[driver] = alpha + beta * df[driver]

    return preds, params_map


def build_univariate_fair_value_maps(
    processed_df_map,
    original_df_map=None,
    window=252,
    method_map=None,
    base_value_map=None,
):
    """
    Build predicted changes and fair value levels for each dataset and driver.

    Returns
    -------
    predicted_changes_map : dict[str, pd.DataFrame]
        Predicted daily changes per driver.
    fair_value_map : dict[str, dict[str, pd.DataFrame]]
        fair_value_map[df_name][driver] -> DataFrame with actual level and fair value.
    """
    predicted_changes_map = {}
    fair_value_map = {}

    for df_name, df in processed_df_map.items():
        y_name = df.columns[0]
        method = "log_return"
        if method_map and df_name in method_map:
            method = method_map[df_name]

        preds_df, _params = rolling_univariate_ols_predictions(df, window=window)
        predicted_changes_map[df_name] = preds_df

        actual_level = _get_actual_level_series(df_name, y_name, original_df_map)
        fair_value_map[df_name] = {}

        for driver in preds_df.columns:
            pred_series = preds_df[driver]
            start_idx = pred_series.first_valid_index()

            base_value = None
            if base_value_map and df_name in base_value_map:
                base_value = base_value_map[df_name]
            elif actual_level is not None and start_idx is not None:
                base_value = actual_level.reindex(df.index).loc[start_idx]

            if base_value is None or pd.isna(base_value):
                raise ValueError(
                    f"Base value not found for {df_name} ({driver}). "
                    "Provide original_df_map or base_value_map."
                )

            fair_level = _reconstruct_level(pred_series, base_value, method=method, start_index=start_idx)

            result = pd.DataFrame(index=df.index)
            if actual_level is not None:
                result[y_name] = actual_level.reindex(df.index)
            result["predicted_change"] = pred_series
            result["fair_value_level"] = fair_level

            fair_value_map[df_name][driver] = result

    return predicted_changes_map, fair_value_map


def get_fair_value_comparison(currency, driver, fair_value_map):
    """
    Return DataFrame with actual level and predicted fair value for a given currency and driver.
    """
    if currency not in fair_value_map:
        raise KeyError(f"{currency} not found in fair_value_map.")
    if driver not in fair_value_map[currency]:
        raise KeyError(f"{driver} not found for {currency}.")
    return fair_value_map[currency][driver]


# Example usage (run in a notebook or script where processed_df_map/original_df_map exist):
# predicted_changes_map, fair_value_map = build_univariate_fair_value_maps(
#     processed_df_map,
#     original_df_map=df2_map,
#     window=252,
# )
# df_out = get_fair_value_comparison("nok_df_processed", "BBDXY", fair_value_map)
