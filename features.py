"""
Feature engineering for load forecasting.

Implements features from two models:

  Hong et al. (2011) GLMLF-B naïve MLR benchmark (Equation 16):
      E(Load) = β₀ + β₁·Trend + β₂·Day×Hour + β₃·Month
              + β₄·Month·W + β₅·Month·W² + β₆·Month·W³
              + β₇·Hour·W  + β₈·Hour·W²  + β₉·Hour·W³
      where W is any active weather variable; applied independently to each.

  Wang et al. (2016) recency effect extension:
      Lagged hourly values  W_{t−k},  k = 1 … n_lags
      Daily moving-average  L_{t,d},  d = 1 … n_mavg_days
          where L_{t,d} = (1/24) Σ_{s=1}^{24d} W_{t−s}
      Each recency variable U gets the same f(U) treatment as W:
          f(U) = U + U² + U³ + U·M + U²·M + U³·M + U·H + U²·H + U³·H
      yielding 105 columns per weather variable per lag/mavg period.

Weather column naming (from load.py):
    Single station : 'temp', 'rh', 'dwpt', 'wspd'
    Multi-station  : '{station}_temp', '{station}_rh', etc.
    Temperature columns (plain 'temp' or ending in '_temp') are converted
    to Fahrenheit before use; all other weather columns are used as-is.

Feature tag convention (used in column names):
    Column name uppercased, e.g.:
        'temp'              → 'TEMP'
        'philadelphia_temp' → 'PHILADELPHIA_TEMP'
        'rh'                → 'RH'

DEFAULT_FEATURES keys:
    Feature-group flags:
        trend                : linear trend index
        month                : month main-effect dummies (M_2..M_12)
        day_hour             : day-of-week × hour interaction dummies
        month_weather        : month dummy × cubic polynomial for each active weather var
        hour_weather         : hour dummy  × cubic polynomial for each active weather var
        weather_lags         : lagged hourly weather values (Wang et al. 2016)
        weather_mavg         : daily moving-average weather values (Wang et al. 2016)
        weather_interactions : pairwise products of weather vars within the same station

    Per-variable filters (variable part of the column name):
        temp  : include temperature columns
        rh    : include relative-humidity columns
        dwpt  : include dew-point columns
        wspd  : include wind-speed columns
    Setting a variable filter to False removes that variable type entirely —
    no raw features and no interaction features that involve it.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Default feature configuration
# ---------------------------------------------------------------------------

DEFAULT_FEATURES: Dict[str, bool] = {
    # Feature-group flags
    "trend":                True,   # linear trend index
    "month":                True,   # month main-effect dummies (M_2..M_12)
    "day_hour":             True,   # day-of-week × hour interaction dummies
    "month_weather":        True,   # month dummy × weather polynomial for all active weather vars
    "hour_weather":         True,   # hour dummy  × weather polynomial for all active weather vars
    "weather_lags":           False,  # lagged hourly weather values (Wang et al. 2016)
    "weather_mavg":           False,  # daily moving-average weather values (Wang et al. 2016)
    "weather_interactions":   False,  # pairwise within-station weather variable products
    "recency_no_interaction": True,  # if True, recency vars get only 3 poly cols (no month/hour interactions)
    # Per-variable filters
    "temp":                 True,   # include temperature columns
    "rh":                   False,   # include relative-humidity columns
    "dwpt":                 False,   # include dew-point columns
    "wspd":                 False,   # include wind-speed columns
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def celsius_to_fahrenheit(temp_c: pd.Series) -> pd.Series:
    """Convert Celsius to Fahrenheit."""
    return temp_c * 9.0 / 5.0 + 32.0


def _is_temp_col(col: str) -> bool:
    """True for temperature columns: plain 'temp' or any column ending in '_temp'."""
    return col == "temp" or col.endswith("_temp")


def _var_of(col: str) -> str:
    """Return the variable-type part of a weather column name.

    Examples
    --------
    'philadelphia_temp' → 'temp'
    'rh'                → 'rh'
    """
    return col.rsplit("_", 1)[1] if "_" in col else col


def _weather_col_tag(col: str) -> str:
    """Return the uppercase feature-name tag for a weather column."""
    return col.upper()


def _station_of(col: str) -> str:
    """Return the station prefix of a weather column.

    Examples
    --------
    'philadelphia_temp' → 'philadelphia'
    'temp'              → ''           (unnamed / single station)
    """
    return col.rsplit("_", 1)[0] if "_" in col else ""


def _prepare_weather_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return the weather series for *col*, converting temperature to Fahrenheit."""
    s = df[col].astype(float)
    return celsius_to_fahrenheit(s) if _is_temp_col(col) else s


def _recency_poly_features(
    series: pd.Series,
    var_tag: str,
    prefix: str,
    month_dummies: pd.DataFrame,
    hour_dummies: pd.DataFrame,
    no_interaction: bool = False,
) -> Dict[str, pd.Series]:
    """
    Build the f(U) features for one recency weather variable (Wang et al. 2016).

    Produces 105 columns by default (no_interaction=False):
        3  raw cubic terms   (U, U², U³)
        33 month × cubic     (11 months × 3 degrees)
        69 hour  × cubic     (23 hours  × 3 degrees)

    When no_interaction=True, produces only 3 columns (U, U², U³) — no
    month or hour interaction terms.

    Parameters
    ----------
    series : pd.Series
        Weather values (may contain NaN for warm-up rows).
    var_tag : str
        Uppercase tag, e.g. ``'TEMP'`` or ``'PHILADELPHIA_RH'``.
    prefix : str
        Recency prefix, e.g. ``'lag1'`` or ``'mavg2'``.
    month_dummies : pd.DataFrame
        Columns M_2..M_12 (same index as *series*).
    hour_dummies : pd.DataFrame
        Columns H_1..H_23 (same index as *series*).
    no_interaction : bool, optional
        If True, skip month and hour interaction terms and return only the
        3 raw polynomial columns.  Default False.
    """
    s2 = series ** 2
    s3 = series ** 3
    parts: Dict[str, pd.Series] = {}

    # Raw cubic polynomial
    parts[f"{prefix}_{var_tag}1"] = series
    parts[f"{prefix}_{var_tag}2"] = s2
    parts[f"{prefix}_{var_tag}3"] = s3

    if not no_interaction:
        # Month dummy × polynomial
        for col in month_dummies.columns:       # M_2 .. M_12
            d = month_dummies[col].astype(float)
            parts[f"{col}_{prefix}_{var_tag}1"] = d * series
            parts[f"{col}_{prefix}_{var_tag}2"] = d * s2
            parts[f"{col}_{prefix}_{var_tag}3"] = d * s3

        # Hour dummy × polynomial
        for col in hour_dummies.columns:        # H_1 .. H_23
            d = hour_dummies[col].astype(float)
            parts[f"{col}_{prefix}_{var_tag}1"] = d * series
            parts[f"{col}_{prefix}_{var_tag}2"] = d * s2
            parts[f"{col}_{prefix}_{var_tag}3"] = d * s3

    return parts


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    trend_offset: int = 0,
    features: Optional[Dict[str, bool]] = None,
    n_lags: int = 12,
    n_mavg_days: int = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix for load forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex (hourly, naive Eastern time), and columns:
            ``'load_mw'``           : float, electricity demand in MW (required)
            ``'<station>_<var>'``   : weather variables for one or more stations.
            Temperature columns (``'temp'`` or ``'*_temp'``) are converted to
            Fahrenheit; all other weather columns are used as-is.
        Rows with NaN in any column are dropped.  Additional rows at the start
        are dropped when recency features are enabled (warm-up period).
    trend_offset : int, optional
        Value such that the trend counter for the i-th row of the cleaned df
        is ``trend_offset + i``.  Pass 0 (default) for training.  For
        prediction, pass the last trend value observed during training so
        that the counter continues without a gap.
    features : dict, optional
        Mapping of feature-group name → bool.  Keys absent from
        ``DEFAULT_FEATURES`` are silently ignored; keys not supplied fall
        back to ``DEFAULT_FEATURES``.
    n_lags : int, optional
        Number of lagged hourly weather variables when
        ``features['weather_lags']`` is True.  Default 12.
    n_mavg_days : int, optional
        Number of daily moving-average weather variables when
        ``features['weather_mavg']`` is True.  Default 2.

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
        If ``'load_mw'`` column is missing.
    """
    if "load_mw" not in df.columns:
        raise ValueError("DataFrame is missing required column: 'load_mw'")

    # Resolve feature flags: start from defaults, overlay user choices
    feat = DEFAULT_FEATURES.copy()
    if features is not None:
        feat.update({k: v for k, v in features.items() if k in DEFAULT_FEATURES})

    # All weather columns present in df, filtered by per-variable flags.
    # A column 'philadelphia_temp' is kept only if feat['temp'] is True.
    weather_cols: List[str] = [
        c for c in df.columns
        if c != "load_mw" and feat.get(_var_of(c), True)
    ]

    # Work on a clean copy; drop rows missing any column
    all_cols = ["load_mw"] + weather_cols
    df = df[all_cols].dropna(subset=all_cols)
    n = len(df)

    # Prepare weather series (Fahrenheit for temp cols, raw for others)
    weather_series: Dict[str, pd.Series] = {
        col: _prepare_weather_series(df, col) for col in weather_cols
    }

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
    month_cat     = pd.Categorical(df.index.month, categories=range(1, 13), ordered=False)
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
    # 4. Hour dummies for base-weather interactions (H_1..H_23)
    # ------------------------------------------------------------------
    hour_cat     = pd.Categorical(df.index.hour, categories=range(24), ordered=False)
    hour_dummies = pd.get_dummies(hour_cat, prefix="H", drop_first=True)
    hour_dummies.index = df.index

    # ------------------------------------------------------------------
    # 5. Month × weather polynomial interactions (33 cols per weather var)
    # ------------------------------------------------------------------
    month_weather_parts: Dict[str, pd.Series] = {}
    if feat["month_weather"]:
        for col in weather_cols:
            tag = _weather_col_tag(col)
            s   = weather_series[col]
            s2  = s ** 2
            s3  = s ** 3
            for mcol in month_dummies.columns:      # M_2 .. M_12
                d = month_dummies[mcol].astype(float)
                month_weather_parts[f"{mcol}_{tag}1"] = d * s
                month_weather_parts[f"{mcol}_{tag}2"] = d * s2
                month_weather_parts[f"{mcol}_{tag}3"] = d * s3

    # ------------------------------------------------------------------
    # 6. Hour × weather polynomial interactions (69 cols per weather var)
    # ------------------------------------------------------------------
    hour_weather_parts: Dict[str, pd.Series] = {}
    if feat["hour_weather"]:
        for col in weather_cols:
            tag = _weather_col_tag(col)
            s   = weather_series[col]
            s2  = s ** 2
            s3  = s ** 3
            for hcol in hour_dummies.columns:       # H_1 .. H_23
                d = hour_dummies[hcol].astype(float)
                hour_weather_parts[f"{hcol}_{tag}1"] = d * s
                hour_weather_parts[f"{hcol}_{tag}2"] = d * s2
                hour_weather_parts[f"{hcol}_{tag}3"] = d * s3

    # ------------------------------------------------------------------
    # 7. Lagged weather values  W_{t-k}  (Wang et al. 2016)
    #    105 columns per weather var per lag k
    # ------------------------------------------------------------------
    no_interaction = feat.get("recency_no_interaction", False)
    lag_parts: Dict[str, pd.Series] = {}
    if feat["weather_lags"]:
        for col in weather_cols:
            tag = _weather_col_tag(col)
            for k in range(1, n_lags + 1):
                shifted = df[col].shift(k)
                if _is_temp_col(col):
                    shifted = celsius_to_fahrenheit(shifted)
                lag_parts.update(
                    _recency_poly_features(shifted, tag, f"lag{k}", month_dummies, hour_dummies,
                                           no_interaction=no_interaction)
                )

    # ------------------------------------------------------------------
    # 8. Daily moving-average weather values  L_{t,d}  (Wang et al. 2016)
    #    L_{t,d} = (1/24) Σ_{s=1}^{24d} W_{t-s}   (Equation 3)
    #    105 columns per weather var per mavg day d
    # ------------------------------------------------------------------
    mavg_parts: Dict[str, pd.Series] = {}
    if feat["weather_mavg"]:
        for col in weather_cols:
            tag = _weather_col_tag(col)
            for d in range(1, n_mavg_days + 1):
                window = 24 * d
                # shift(1) excludes the current hour; rolling gives the mean of
                # W_{t-1}, W_{t-2}, ..., W_{t-24d}  — matches Equation 3
                rolled = df[col].shift(1).rolling(window).mean()
                if _is_temp_col(col):
                    rolled = celsius_to_fahrenheit(rolled)
                mavg_parts.update(
                    _recency_poly_features(rolled, tag, f"mavg{d}", month_dummies, hour_dummies,
                                           no_interaction=no_interaction)
                )

    # ------------------------------------------------------------------
    # 9. Pairwise within-station weather interactions
    #    One column per pair: TAG_A_X_TAG_B  (simple product, no polynomial)
    #    Only considers active weather_cols (disabled variables already excluded)
    # ------------------------------------------------------------------
    interaction_parts: Dict[str, pd.Series] = {}
    if feat["weather_interactions"]:
        by_station: Dict[str, List[str]] = defaultdict(list)
        for col in weather_cols:
            by_station[_station_of(col)].append(col)
        for cols_in_station in by_station.values():
            for i in range(len(cols_in_station)):
                for j in range(i + 1, len(cols_in_station)):
                    col_a = cols_in_station[i]
                    col_b = cols_in_station[j]
                    tag_a = _weather_col_tag(col_a)
                    tag_b = _weather_col_tag(col_b)
                    interaction_parts[f"{tag_a}_X_{tag_b}"] = (
                        weather_series[col_a] * weather_series[col_b]
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
    if feat["month_weather"] and month_weather_parts:
        parts_list.append(pd.DataFrame(month_weather_parts, index=df.index))
    if feat["hour_weather"] and hour_weather_parts:
        parts_list.append(pd.DataFrame(hour_weather_parts, index=df.index))
    if feat["weather_lags"] and lag_parts:
        parts_list.append(pd.DataFrame(lag_parts, index=df.index))
    if feat["weather_mavg"] and mavg_parts:
        parts_list.append(pd.DataFrame(mavg_parts, index=df.index))
    if feat["weather_interactions"] and interaction_parts:
        parts_list.append(pd.DataFrame(interaction_parts, index=df.index))

    X = pd.concat(parts_list, axis=1) if parts_list else pd.DataFrame(index=df.index)

    # Drop warm-up rows where any recency feature is NaN
    valid_mask = X.notna().all(axis=1)
    X = X.loc[valid_mask]
    y = df.loc[valid_mask, "load_mw"]

    return X, y


# ---------------------------------------------------------------------------
# Train / predict orchestration (trend continuation + recency warm-up)
# ---------------------------------------------------------------------------


def merge_feature_flags(features: Optional[Dict[str, bool]]) -> Dict[str, bool]:
    """``DEFAULT_FEATURES`` with optional user overrides (only known keys)."""
    feat = DEFAULT_FEATURES.copy()
    if features is not None:
        feat.update({k: v for k, v in features.items() if k in DEFAULT_FEATURES})
    return feat


def lookback_row_count(
    features: Optional[Dict[str, bool]],
    n_lags: int = 12,
    n_mavg_days: int = 2,
) -> int:
    """Rows of training history needed before the first row of a prediction window."""
    feat = merge_feature_flags(features)
    lags_rows = n_lags if feat["weather_lags"] else 0
    mavg_rows = (24 * n_mavg_days + 1) if feat["weather_mavg"] else 0
    return max(lags_rows, mavg_rows)


@dataclass(frozen=True)
class ForecastPrepState:
    """Carry-over from training for :func:`predict_feature_matrix`."""

    trend_end: int
    train_tail: Optional[pd.DataFrame]
    feature_names: Tuple[str, ...]


def fit_feature_matrices(
    df_train: pd.DataFrame,
    features: Optional[Dict[str, bool]] = None,
    n_lags: int = 12,
    n_mavg_days: int = 2,
) -> Tuple[pd.DataFrame, pd.Series, ForecastPrepState]:
    """
    Build training ``X, y`` and state needed to build aligned test features.

    Handles the same trend and recency warm-up contract as the NaiveMLR benchmark.
    """
    X, y = build_features(
        df_train,
        trend_offset=0,
        features=features,
        n_lags=n_lags,
        n_mavg_days=n_mavg_days,
    )
    trend_end = int(X["trend"].iloc[-1]) if "trend" in X.columns else len(X)
    lookback = lookback_row_count(features, n_lags, n_mavg_days)
    train_tail = None
    if lookback > 0:
        weather_cols = [c for c in df_train.columns if c != "load_mw"]
        train_tail = df_train[["load_mw"] + weather_cols].tail(lookback).copy()
    state = ForecastPrepState(
        trend_end=trend_end,
        train_tail=train_tail,
        feature_names=tuple(X.columns),
    )
    return X, y, state


def _feature_matrix_has_recency(feature_names: Tuple[str, ...]) -> bool:
    return any(n.startswith("lag") or n.startswith("mavg") for n in feature_names)


def predict_feature_matrix(
    df_test: pd.DataFrame,
    state: ForecastPrepState,
    features: Optional[Dict[str, bool]] = None,
    n_lags: int = 12,
    n_mavg_days: int = 2,
    trend_offset: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build the test design matrix with the same columns as training.

    Prepends ``state.train_tail`` when recency columns are present so lags/mavg
    are defined on the first test hour.
    """
    if trend_offset is None:
        trend_offset = state.trend_end

    df_pred = df_test.copy()
    if "load_mw" not in df_pred.columns:
        df_pred["load_mw"] = 0.0

    use_history = (
        state.train_tail is not None
        and _feature_matrix_has_recency(state.feature_names)
    )

    if use_history:
        tail = state.train_tail.copy()
        tail["load_mw"] = tail.get("load_mw", 0.0)
        df_combined = pd.concat([tail, df_pred])
        combined_offset = state.trend_end - len(tail)
        X_combined, _ = build_features(
            df_combined,
            trend_offset=combined_offset,
            features=features,
            n_lags=n_lags,
            n_mavg_days=n_mavg_days,
        )
        test_idx = df_test.index
        X_test = X_combined.loc[X_combined.index.isin(test_idx)]
    else:
        X_test, _ = build_features(
            df_pred,
            trend_offset=trend_offset,
            features=features,
            n_lags=n_lags,
            n_mavg_days=n_mavg_days,
        )

    if tuple(X_test.columns) != state.feature_names:
        raise ValueError(
            f"Feature mismatch: expected {len(state.feature_names)} cols, "
            f"got {len(X_test.columns)}. Make sure test data covers all "
            "calendar combinations (use a full-year or multi-year window)."
        )
    return X_test
