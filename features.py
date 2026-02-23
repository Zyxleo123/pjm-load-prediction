"""
Feature engineering for load forecasting.

Implements features from two models:

  Hong et al. (2011) GLMLF-B naïve MLR benchmark (Equation 16):
      E(Load) = β₀ + β₁·Trend + β₂·Day×Hour + β₃·Month
              + β₄·Month·T + β₅·Month·T² + β₆·Month·T³
              + β₇·Hour·T  + β₈·Hour·T²  + β₉·Hour·T³

  Wang et al. (2016) recency effect extension:
      Lagged hourly temperatures  T_{t−k},  k = 1 … n_lags
      Daily moving-average temps  L_{t,d},  d = 1 … n_mavg_days
          where L_{t,d} = (1/24) Σ_{s=1}^{24d} T_{t−s}
      Each recency variable U gets the same f(U) treatment as T in the
      base temperature features (cubic, with month- and hour-dummy
      interactions):
          f(U) = U + U² + U³ + U·M + U²·M + U³·M + U·H + U²·H + U³·H
      yielding 3 + 33 + 69 = 105 columns per variable.

Default X matrix column layout (281 columns, base features only):
    [0]        trend
    [1..11]    M_2 .. M_12            (11 month main-effect dummies, Jan=reference)
    [12..178]  DH_0_1 .. DH_6_23     (167 Day×Hour interaction dummies)
    [179..211] M_2_TMP1 .. M_12_TMP3 (33 month × temperature polynomial cols)
    [212..280] H_1_TMP1 .. H_23_TMP3 (69 hour × temperature polynomial cols)

Recency columns appended when temp_lags / temp_mavg is True (105 cols each):
    Per lag k  (k = 1 … n_lags):
        lag{k}_TMP1, lag{k}_TMP2, lag{k}_TMP3             (3 raw poly)
        M_2_lag{k}_TMP1, M_2_lag{k}_TMP2, M_2_lag{k}_TMP3, …   (33 month×poly)
        H_1_lag{k}_TMP1, H_1_lag{k}_TMP2, H_1_lag{k}_TMP3, …   (69 hour×poly)
    Per moving-average day d  (d = 1 … n_mavg_days):
        mavg{d}_TMP1, mavg{d}_TMP2, mavg{d}_TMP3          (3 raw poly)
        M_2_mavg{d}_TMP1, M_2_mavg{d}_TMP2, M_2_mavg{d}_TMP3, … (33 month×poly)
        H_1_mavg{d}_TMP1, H_1_mavg{d}_TMP2, H_1_mavg{d}_TMP3, … (69 hour×poly)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Default feature configuration
# ---------------------------------------------------------------------------

DEFAULT_FEATURES: Dict[str, bool] = {
    "trend":      True,   # linear trend index
    "month":      True,   # month main-effect dummies (M_2..M_12)
    "day_hour":   True,   # day-of-week × hour interaction dummies
    "month_temp": True,   # month dummy × temperature polynomial (T, T², T³)
    "hour_temp":  True,   # hour dummy × temperature polynomial  (T, T², T³)
    "temp_lags":  False,  # lagged hourly temperatures (Wang et al. 2016)
    "temp_mavg":  False,  # daily moving-average temperatures (Wang et al. 2016)
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def celsius_to_fahrenheit(temp_c: pd.Series) -> pd.Series:
    """Convert Celsius to Fahrenheit."""
    return temp_c * 9.0 / 5.0 + 32.0


def _recency_poly_features(
    tmp_f: pd.Series,
    prefix: str,
    month_dummies: pd.DataFrame,
    hour_dummies: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """
    Build the f(U) features for one recency temperature variable (Wang et al. 2016).

    Produces 105 columns:
        3  raw cubic terms (U, U², U³)
        33 month dummy × cubic  (11 months × 3 degrees)
        69 hour dummy  × cubic  (23 hours  × 3 degrees)

    Parameters
    ----------
    tmp_f : pd.Series
        Temperature in Fahrenheit (may contain NaN for warm-up rows).
    prefix : str
        Column name prefix, e.g. ``'lag1'`` or ``'mavg2'``.
    month_dummies : pd.DataFrame
        Columns M_2..M_12 (11 cols, same index as tmp_f).
    hour_dummies : pd.DataFrame
        Columns H_1..H_23 (23 cols, same index as tmp_f).

    Returns
    -------
    dict mapping column name → pd.Series
    """
    tmp_f2 = tmp_f ** 2
    tmp_f3 = tmp_f ** 3
    parts: Dict[str, pd.Series] = {}

    # Raw cubic polynomial (consistent with base temperature features)
    parts[f"{prefix}_TMP1"] = tmp_f
    parts[f"{prefix}_TMP2"] = tmp_f2
    parts[f"{prefix}_TMP3"] = tmp_f3

    # Month dummy × polynomial interactions
    for col in month_dummies.columns:       # M_2 .. M_12
        d = month_dummies[col].astype(float)
        parts[f"{col}_{prefix}_TMP1"] = d * tmp_f
        parts[f"{col}_{prefix}_TMP2"] = d * tmp_f2
        parts[f"{col}_{prefix}_TMP3"] = d * tmp_f3

    # Hour dummy × polynomial interactions
    for col in hour_dummies.columns:        # H_1 .. H_23
        d = hour_dummies[col].astype(float)
        parts[f"{col}_{prefix}_TMP1"] = d * tmp_f
        parts[f"{col}_{prefix}_TMP2"] = d * tmp_f2
        parts[f"{col}_{prefix}_TMP3"] = d * tmp_f3

    return parts


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    trend_offset: int = 0,
    features: Optional[Dict[str, bool]] = None,
    n_lags: int = 12,
    n_mavg_days: int = 2,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix for load forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex (hourly, naive Eastern time), and columns:
            ``'load_mw'`` : float, electricity demand in MW
            ``'temp'``    : float, temperature in degrees Celsius
        Rows with NaN in ``load_mw`` or ``temp`` are dropped. Additional rows
        at the start of the series are dropped when recency features are
        enabled (warm-up rows lacking sufficient history).
    trend_offset : int, optional
        Value such that the trend counter for the i-th row of the cleaned df
        is ``trend_offset + i``.  Pass 0 (default) for training.  For
        prediction, pass the last trend value observed during training so
        that the counter continues without a gap.
    features : dict, optional
        Mapping of feature-group name → bool controlling which groups are
        included.  Keys absent from ``DEFAULT_FEATURES`` are silently
        ignored; keys not supplied fall back to ``DEFAULT_FEATURES``.
        Available keys:

        +---------------+-----------------------------------------------+
        | Key           | Description                                   |
        +===============+===============================================+
        | ``trend``     | Linear trend index                            |
        +---------------+-----------------------------------------------+
        | ``month``     | Month main-effect dummies (M_2..M_12)         |
        +---------------+-----------------------------------------------+
        | ``day_hour``  | Day-of-week × hour interaction dummies        |
        +---------------+-----------------------------------------------+
        | ``month_temp``| Month dummy × temperature poly (T, T², T³)   |
        +---------------+-----------------------------------------------+
        | ``hour_temp`` | Hour dummy  × temperature poly (T, T², T³)   |
        +---------------+-----------------------------------------------+
        | ``temp_lags`` | Lagged hourly temps T_{t-k}, k=1..n_lags     |
        |               | with cubic month/hour interactions             |
        |               | (Wang et al. 2016)                            |
        +---------------+-----------------------------------------------+
        | ``temp_mavg`` | Daily moving-average temps L_{t,d},           |
        |               | d=1..n_mavg_days, with cubic interactions     |
        |               | (Wang et al. 2016)                            |
        +---------------+-----------------------------------------------+

    n_lags : int, optional
        Number of lagged hourly temperature variables when
        ``features['temp_lags']`` is True.  Default 12.
    n_mavg_days : int, optional
        Number of daily moving-average temperature variables when
        ``features['temp_mavg']`` is True.  Default 2.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.  Index is a subset of ``df.index`` (warm-up rows
        dropped when recency features are active).
    y : pd.Series
        Target ``load_mw``, same index as X.

    Raises
    ------
    ValueError
        If ``'load_mw'`` or ``'temp'`` columns are missing.
    """
    required = {"load_mw", "temp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # Resolve feature flags: start from defaults, overlay user choices
    feat = DEFAULT_FEATURES.copy()
    if features is not None:
        feat.update({k: v for k, v in features.items() if k in DEFAULT_FEATURES})

    # Work on a clean copy; drop rows missing load or temperature
    df = df[["load_mw", "temp"]].dropna(subset=["load_mw", "temp"])
    n = len(df)

    tmp_f  = celsius_to_fahrenheit(df["temp"])
    tmp_f2 = tmp_f ** 2
    tmp_f3 = tmp_f ** 3

    # ------------------------------------------------------------------
    # 1. Trend
    # ------------------------------------------------------------------
    trend = pd.Series(
        np.arange(trend_offset + 1, trend_offset + 1 + n, dtype=float),
        index=df.index,
        name="trend",
    )

    # ------------------------------------------------------------------
    # 2. Month main-effect dummies (11 cols, January = reference)
    # ------------------------------------------------------------------
    month_cat    = pd.Categorical(df.index.month, categories=range(1, 13), ordered=False)
    month_dummies = pd.get_dummies(month_cat, prefix="M", drop_first=True)
    month_dummies.index = df.index

    # ------------------------------------------------------------------
    # 3. Day×Hour interaction dummies (167 cols, Monday-0am = reference)
    # ------------------------------------------------------------------
    dh_labels   = [f"{d}_{h}" for d, h in zip(df.index.dayofweek, df.index.hour)]
    all_dh_cats = [f"{d}_{h}" for d in range(7) for h in range(24)]
    dh_cat      = pd.Categorical(dh_labels, categories=all_dh_cats, ordered=False)
    dh_dummies  = pd.get_dummies(dh_cat, prefix="DH", drop_first=True)
    dh_dummies.index = df.index

    # ------------------------------------------------------------------
    # 4. Hour dummies for base-temperature interactions (H_1..H_23)
    # ------------------------------------------------------------------
    hour_cat    = pd.Categorical(df.index.hour, categories=range(24), ordered=False)
    hour_dummies = pd.get_dummies(hour_cat, prefix="H", drop_first=True)
    hour_dummies.index = df.index

    # ------------------------------------------------------------------
    # 5. Month × temperature polynomial interactions (33 cols)
    # ------------------------------------------------------------------
    month_tmp_parts: Dict[str, pd.Series] = {}
    for col in month_dummies.columns:           # M_2 .. M_12
        d = month_dummies[col].astype(float)
        month_tmp_parts[f"{col}_TMP1"] = d * tmp_f
        month_tmp_parts[f"{col}_TMP2"] = d * tmp_f2
        month_tmp_parts[f"{col}_TMP3"] = d * tmp_f3
    month_tmp_df = pd.DataFrame(month_tmp_parts, index=df.index)

    # ------------------------------------------------------------------
    # 6. Hour × temperature polynomial interactions (69 cols)
    # ------------------------------------------------------------------
    hour_tmp_parts: Dict[str, pd.Series] = {}
    for col in hour_dummies.columns:            # H_1 .. H_23
        d = hour_dummies[col].astype(float)
        hour_tmp_parts[f"{col}_TMP1"] = d * tmp_f
        hour_tmp_parts[f"{col}_TMP2"] = d * tmp_f2
        hour_tmp_parts[f"{col}_TMP3"] = d * tmp_f3
    hour_tmp_df = pd.DataFrame(hour_tmp_parts, index=df.index)

    # ------------------------------------------------------------------
    # 7. Lagged hourly temperatures  T_{t-k}  (Wang et al. 2016)
    #    70 columns per lag: raw (T,T²) + month×(T,T²) + hour×(T,T²)
    # ------------------------------------------------------------------
    lag_parts: Dict[str, pd.Series] = {}
    if feat["temp_lags"]:
        for k in range(1, n_lags + 1):
            tmp_lag = celsius_to_fahrenheit(df["temp"].shift(k))
            lag_parts.update(
                _recency_poly_features(tmp_lag, f"lag{k}", month_dummies, hour_dummies)
            )

    # ------------------------------------------------------------------
    # 8. Daily moving-average temperatures  L_{t,d}  (Wang et al. 2016)
    #    L_{t,d} = (1/24) Σ_{s=1}^{24d} T_{t-s}   (Equation 3)
    #    70 columns per day: raw (T,T²) + month×(T,T²) + hour×(T,T²)
    # ------------------------------------------------------------------
    mavg_parts: Dict[str, pd.Series] = {}
    if feat["temp_mavg"]:
        for d in range(1, n_mavg_days + 1):
            window   = 24 * d
            # shift(1) excludes the current hour, rolling gives the mean of
            # T_{t-1}, T_{t-2}, ..., T_{t-24d}  — matches Equation 3
            tmp_mavg = celsius_to_fahrenheit(df["temp"].shift(1).rolling(window).mean())
            mavg_parts.update(
                _recency_poly_features(tmp_mavg, f"mavg{d}", month_dummies, hour_dummies)
            )

    # ------------------------------------------------------------------
    # Assemble X in canonical order, honouring feature flags
    # ------------------------------------------------------------------
    parts_list = []
    if feat["trend"]:
        parts_list.append(trend.to_frame())
    if feat["month"]:
        parts_list.append(month_dummies)
    if feat["day_hour"]:
        parts_list.append(dh_dummies)
    if feat["month_temp"]:
        parts_list.append(month_tmp_df)
    if feat["hour_temp"]:
        parts_list.append(hour_tmp_df)
    if feat["temp_lags"] and lag_parts:
        parts_list.append(pd.DataFrame(lag_parts, index=df.index))
    if feat["temp_mavg"] and mavg_parts:
        parts_list.append(pd.DataFrame(mavg_parts, index=df.index))

    X = pd.concat(parts_list, axis=1) if parts_list else pd.DataFrame(index=df.index)

    # Drop warm-up rows where any recency feature is NaN
    valid_mask = X.notna().all(axis=1)
    X = X.loc[valid_mask]
    y = df.loc[valid_mask, "load_mw"]

    return X, y
