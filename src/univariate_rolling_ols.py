import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

# 1. Univariate Rolling OLS and Fair Value Calculation

def univariate_rolling_ols_fair_value(processed_df_map, df2_map=None, window=252, method_map=None):
    """
    For each currency pair in processed_df_map, perform univariate rolling OLS for each driver (X),
    calculate predicted changes, and recover fair value levels.
    Returns:
        predicted_changes_map: dict[currency][driver] = pd.Series of predicted changes
        fair_value_map: dict[currency][driver] = pd.Series of fair value levels
    """
    predicted_changes_map = {}
    fair_value_map = {}
    for currency, df in processed_df_map.items():
        y_name = df.columns[0]
        y = df[y_name]
        predicted_changes_map[currency] = {}
        fair_value_map[currency] = {}
        for driver in df.columns[1:]:
            X = df[[driver]]
            X_const = sm.add_constant(X)
            # Rolling OLS
            model = RollingOLS(y, X_const, window=window)
            rres = model.fit()
            alpha = rres.params['const']
            beta = rres.params[driver]
            # Predicted daily change
            y_hat = alpha + beta * X[driver]
            predicted_changes_map[currency][driver] = y_hat
            # Recover price level
            # Get original level series
            if df2_map and currency in df2_map:
                orig_level = df2_map[currency][y_name]
            else:
                orig_level = None
                # Try to get from processed_df first column if available
                if y_name in df:
                    orig_level = df[y_name]
            # Determine method
            method = 'log_return'
            if method_map and currency in method_map:
                method = method_map[currency]
            # Find first non-NaN base value
            base_value = orig_level.dropna().iloc[0] if orig_level is not None else 1.0
            # Align indices (rolling window warm-up)
            y_hat_aligned = y_hat.reindex(orig_level.index)
            y_hat_aligned = y_hat_aligned.fillna(0)
            if method == 'log_return':
                fair_value = base_value * np.exp(y_hat_aligned.cumsum())
            elif method == 'diff':
                fair_value = base_value + y_hat_aligned.cumsum()
            else:
                raise ValueError('Unknown method for fair value recovery')
            fair_value_map[currency][driver] = fair_value
    return predicted_changes_map, fair_value_map

# 2. Output function

def get_fair_value_comparison(currency, driver, processed_df_map, df2_map, predicted_changes_map, fair_value_map):
    """
    Returns a DataFrame with Actual Level and Predicted Fair Value Level for a given currency and driver.
    """
    y_name = processed_df_map[currency].columns[0]
    if df2_map and currency in df2_map:
        actual_level = df2_map[currency][y_name]
    else:
        actual_level = processed_df_map[currency][y_name]
    fair_value = fair_value_map[currency][driver]
    # Align indices
    df_out = pd.DataFrame({
        'Actual_Level': actual_level,
        f'Fair_Value_{driver}': fair_value
    })
    return df_out
