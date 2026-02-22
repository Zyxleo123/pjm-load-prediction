"""
Feature engineering for the Hong et al. (2011) GLMLF-B naïve MLR benchmark.

Implements Equation 16:
    E(Load) = β₀ + β₁·Trend + β₂·Day×Hour + β₃·Month
            + β₄·Month·T + β₅·Month·T² + β₆·Month·T³
            + β₇·Hour·T  + β₈·Hour·T²  + β₉·Hour·T³

X matrix column layout (281 columns total):
    [0]       trend
    [1..11]   M_2 .. M_12          (11 month main-effect dummies, Jan=reference)
    [12..178] DH_0_1 .. DH_6_23   (167 Day×Hour interaction dummies, Mon-0am=reference)
    [179..211] M_2_TMP1 .. M_12_TMP3  (33 month × temperature polynomial cols)
    [212..280] H_1_TMP1 .. H_23_TMP3  (69 hour × temperature polynomial cols)
"""

import numpy as np
import pandas as pd
from typing import Tuple


def celsius_to_fahrenheit(temp_c: pd.Series) -> pd.Series:
    """Convert Celsius to Fahrenheit."""
    return temp_c * 9.0 / 5.0 + 32.0


def build_features(
    df: pd.DataFrame,
    trend_offset: int = 0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the GLMLF-B feature matrix from Hong et al. (2011), Equation 16.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex (hourly, naive Eastern time), and columns:
            'load_mw' : float, electricity demand in MW
            'temp'    : float, temperature in degrees Celsius
        Rows with NaN in either column are dropped.
    trend_offset : int, optional
        Starting offset for the Trend counter.  The first row of df gets
        Trend = trend_offset + 1.  Use 0 (default) when df is the training set.
        When building test features, pass len(df_train) so that Trend continues
        from where training ended.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with 281 columns, same index as the cleaned df.
    y : pd.Series
        Target series (load_mw), same index as X.

    Raises
    ------
    ValueError
        If 'load_mw' or 'temp' columns are missing.
    """
    required = {'load_mw', 'temp'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    df = df[['load_mw', 'temp']].dropna()
    n = len(df)

    tmp_f = celsius_to_fahrenheit(df['temp'])
    tmp_f2 = tmp_f ** 2
    tmp_f3 = tmp_f ** 3

    # ------------------------------------------------------------------
    # 1. Trend
    # ------------------------------------------------------------------
    trend = pd.Series(
        np.arange(trend_offset + 1, trend_offset + 1 + n, dtype=float),
        index=df.index,
        name='trend',
    )

    # ------------------------------------------------------------------
    # 2. Month main-effect dummies (11 cols, January = reference)
    # ------------------------------------------------------------------
    month_cat = pd.Categorical(df.index.month, categories=range(1, 13), ordered=False)
    month_dummies = pd.get_dummies(month_cat, prefix='M', drop_first=True)
    month_dummies.index = df.index
    # Columns: M_2 .. M_12  (11 cols)

    # ------------------------------------------------------------------
    # 3. Day×Hour interaction dummies (167 cols, Monday-0am = reference)
    #    Fixed category list ensures consistent columns across any window.
    # ------------------------------------------------------------------
    dh_labels = [f"{d}_{h}" for d, h in zip(df.index.dayofweek, df.index.hour)]
    all_dh_cats = [f"{d}_{h}" for d in range(7) for h in range(24)]  # 168 in order
    dh_cat = pd.Categorical(dh_labels, categories=all_dh_cats, ordered=False)
    dh_dummies = pd.get_dummies(dh_cat, prefix='DH', drop_first=True)
    dh_dummies.index = df.index
    # Columns: DH_0_1 .. DH_6_23  (167 cols)

    # ------------------------------------------------------------------
    # 4. Hour dummies for temperature interactions (NOT added to X)
    #    23 cols: H_1 .. H_23 (hour 0 = reference, dropped)
    # ------------------------------------------------------------------
    hour_cat = pd.Categorical(df.index.hour, categories=range(24), ordered=False)
    hour_dummies = pd.get_dummies(hour_cat, prefix='H', drop_first=True)
    hour_dummies.index = df.index
    # Columns: H_1 .. H_23  (23 cols, for interactions only)

    # ------------------------------------------------------------------
    # 5. Month × temperature polynomial interactions (33 cols)
    # ------------------------------------------------------------------
    month_tmp_parts = {}
    for col in month_dummies.columns:  # M_2 .. M_12
        d = month_dummies[col]
        month_tmp_parts[f'{col}_TMP1'] = d * tmp_f
        month_tmp_parts[f'{col}_TMP2'] = d * tmp_f2
        month_tmp_parts[f'{col}_TMP3'] = d * tmp_f3
    month_tmp_df = pd.DataFrame(month_tmp_parts, index=df.index)

    # ------------------------------------------------------------------
    # 6. Hour × temperature polynomial interactions (69 cols)
    # ------------------------------------------------------------------
    hour_tmp_parts = {}
    for col in hour_dummies.columns:  # H_1 .. H_23
        d = hour_dummies[col]
        hour_tmp_parts[f'{col}_TMP1'] = d * tmp_f
        hour_tmp_parts[f'{col}_TMP2'] = d * tmp_f2
        hour_tmp_parts[f'{col}_TMP3'] = d * tmp_f3
    hour_tmp_df = pd.DataFrame(hour_tmp_parts, index=df.index)

    # ------------------------------------------------------------------
    # Assemble X in canonical column order
    # ------------------------------------------------------------------
    X = pd.concat(
        [trend, month_dummies, dh_dummies, month_tmp_df, hour_tmp_df],
        axis=1,
    )

    y = df['load_mw']
    return X, y
