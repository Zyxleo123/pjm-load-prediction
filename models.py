"""
Hong et al. (2011) GLMLF-B Naïve Multiple Linear Regression benchmark,
extended with the Wang et al. (2016) recency-effect feature engineering.

References:
    Hong, T., Gui, M., Baran, M., & Willis, H. L. (2011).
    A Naïve Multiple Linear Regression Benchmark for Short Term Load Forecasting.
    IEEE Power and Energy Society General Meeting.

    Wang, Y., et al. (2016).
    Electric load forecasting with recency effect: A big data approach.
    International Journal of Forecasting.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, Optional

from features import DEFAULT_FEATURES, build_features


class NaiveMLR:
    """
    GLMLF-B: Naïve MLR benchmark for short-term load forecasting.

    Wraps sklearn LinearRegression with the feature engineering from
    ``features.build_features()``.  When recency features are enabled
    (``features['temp_lags']`` or ``features['temp_mavg']``), the model
    automatically stores the tail of the training data and prepends it when
    predicting, so that lag/moving-average features can be computed for all
    test rows without a warm-up gap.

    Parameters
    ----------
    features : dict, optional
        Feature group → bool mapping passed to ``build_features()``.
        Missing keys fall back to ``DEFAULT_FEATURES``.  Example::

            NaiveMLR(features={"temp_lags": True, "temp_mavg": True})

    n_lags : int, optional
        Number of lagged hourly temperature variables (Wang et al. 2016).
        Default 72.
    n_mavg_days : int, optional
        Number of daily moving-average temperature variables (Wang et al.
        2016).  Default 7.

    Usage
    -----
    model = NaiveMLR()                          # base 281-feature model
    model = NaiveMLR(features={"temp_lags": True}, n_lags=4)
    model = NaiveMLR(features={"temp_mavg": True}, n_mavg_days=2)
    model = NaiveMLR(features={"temp_lags": True, "temp_mavg": True},
                     n_lags=4, n_mavg_days=2)

    preds = model.fit_predict(df_train, df_test)
    """

    def __init__(
        self,
        features: Optional[Dict[str, bool]] = None,
        n_lags: int = 72,
        n_mavg_days: int = 7,
    ) -> None:
        self._features    = features
        self._n_lags      = n_lags
        self._n_mavg_days = n_mavg_days

        self._lr: Optional[LinearRegression] = None
        self.feature_names_: Optional[list]  = None
        self.n_train_rows_: int = 0

        # Last trend value seen during fit(); used as trend_offset in predict()
        # so the test-set trend continues without a gap.
        self._trend_end: int = 0

        # Tail of training data kept for recency-feature warm-up in predict()
        self._train_tail: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _needs_history(self) -> bool:
        """True when the resolved feature set includes any recency features."""
        if self.feature_names_ is None:
            return False
        return any(
            c.startswith("lag") or c.startswith("mavg")
            for c in self.feature_names_
        )

    def _lookback_rows(self) -> int:
        """Number of historical rows required before the first test row."""
        feat = DEFAULT_FEATURES.copy()
        if self._features is not None:
            feat.update({k: v for k, v in self._features.items()
                         if k in DEFAULT_FEATURES})
        lags_rows = self._n_lags if feat["temp_lags"] else 0
        mavg_rows = (24 * self._n_mavg_days + 1) if feat["temp_mavg"] else 0
        return max(lags_rows, mavg_rows)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df_train: pd.DataFrame) -> "NaiveMLR":
        """
        Fit OLS on df_train.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data with DatetimeIndex, ``'load_mw'``, and ``'temp'``.

        Returns
        -------
        self
        """
        X, y = build_features(
            df_train,
            trend_offset=0,
            features=self._features,
            n_lags=self._n_lags,
            n_mavg_days=self._n_mavg_days,
        )
        self._lr = LinearRegression(fit_intercept=True)
        self._lr.fit(X.values, y.values)
        self.feature_names_ = list(X.columns)
        self.n_train_rows_  = len(X)
        self._trend_end     = int(X["trend"].iloc[-1]) if "trend" in X.columns else len(X)

        # Store training tail for recency warm-up during predict()
        lookback = self._lookback_rows()
        if lookback > 0:
            self._train_tail = df_train[["load_mw", "temp"]].tail(lookback).copy()

        return self

    def predict(
        self,
        df_test: pd.DataFrame,
        trend_offset: Optional[int] = None,
    ) -> pd.Series:
        """
        Predict load for df_test.

        Parameters
        ----------
        df_test : pd.DataFrame
            Test/forecast data with DatetimeIndex and ``'temp'`` column.
            ``'load_mw'`` is not required.
        trend_offset : int, optional
            Override for the trend counter start.  Defaults to the last
            trend value from training (set by ``fit()``), so the test
            trend continues without a gap.

        Returns
        -------
        pd.Series
            Predicted load in MW, named ``'load_mw_pred'``.
        """
        if self._lr is None:
            raise RuntimeError("Call fit() before predict().")

        if trend_offset is None:
            trend_offset = self._trend_end

        # Inject placeholder load_mw if absent (not needed for prediction)
        df_pred = df_test.copy()
        if "load_mw" not in df_pred.columns:
            df_pred["load_mw"] = 0.0

        if self._needs_history() and self._train_tail is not None:
            # Prepend training tail so lag/rolling features can be computed
            # for all test rows.  Adjust trend_offset so that the tail rows
            # absorb the "warm-up" counts and the first test row gets
            # trend = _trend_end + 1.
            tail = self._train_tail.copy()
            tail["load_mw"] = tail.get("load_mw", 0.0)  # ensure column present
            df_combined = pd.concat([tail, df_pred])
            combined_offset = self._trend_end - len(tail)

            X_combined, _ = build_features(
                df_combined,
                trend_offset=combined_offset,
                features=self._features,
                n_lags=self._n_lags,
                n_mavg_days=self._n_mavg_days,
            )
            # Keep only the rows that belong to the original test set
            test_idx = df_test.index
            X_test   = X_combined.loc[X_combined.index.isin(test_idx)]
        else:
            X_test, _ = build_features(
                df_pred,
                trend_offset=trend_offset,
                features=self._features,
                n_lags=self._n_lags,
                n_mavg_days=self._n_mavg_days,
            )

        if list(X_test.columns) != self.feature_names_:
            raise ValueError(
                f"Feature mismatch: expected {len(self.feature_names_)} cols, "
                f"got {len(X_test.columns)}. Make sure test data covers all "
                "calendar combinations (use a full-year or multi-year window)."
            )

        preds = self._lr.predict(X_test.values)
        return pd.Series(preds, index=X_test.index, name="load_mw_pred")

    def fit_predict(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> pd.Series:
        """
        Convenience: fit on df_train, then predict on df_test.

        Trend for df_test begins at the last trend value from training + 1.
        """
        self.fit(df_train)
        return self.predict(df_test)


# ---------------------------------------------------------------------------
# Rolling forecast
# ---------------------------------------------------------------------------

def rolling_forecast(
    df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    min_train_days: int = 30,
    model: Optional[NaiveMLR] = None,
) -> pd.DataFrame:
    """
    Rolling daily retrain-and-forecast loop.

    For each calendar day d in [forecast_start, forecast_end]:
      1. Training set = all rows where timestamp < midnight of day d.
      2. Skip day d if training set has < min_train_days * 24 rows.
      3. Fit a NaiveMLR on the training set.
      4. Predict the 24 hours of day d (23 on spring-forward, 25 on fall-back).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with DatetimeIndex, ``'load_mw'``, and ``'temp'``.
    forecast_start : pd.Timestamp
        First day to forecast (inclusive).
    forecast_end : pd.Timestamp
        Last day to forecast (inclusive).
    min_train_days : int, optional
        Minimum training days required before forecasting (default 30).
    model : NaiveMLR, optional
        Model template whose ``features``, ``n_lags``, and ``n_mavg_days``
        settings are reused for every daily refitting.  A plain
        ``NaiveMLR()`` (default, 281-feature base model) is used if not
        provided.

    Returns
    -------
    pd.DataFrame
        Columns ``['load_mw_actual', 'load_mw_pred']``, DatetimeIndex.
    """
    forecast_days = pd.date_range(
        forecast_start.normalize(), forecast_end.normalize(), freq="D"
    )

    # Use model settings as a template; create a fresh instance per day
    if model is None:
        model_kwargs: Dict = {}
    else:
        model_kwargs = dict(
            features    = model._features,
            n_lags      = model._n_lags,
            n_mavg_days = model._n_mavg_days,
        )

    results = []

    for day in forecast_days:
        day_ts  = pd.Timestamp(day.date())
        df_train = df[df.index < day_ts]

        if len(df_train) < min_train_days * 24:
            continue

        day_end = day_ts + pd.Timedelta(hours=23)
        df_day  = df.loc[day_ts:day_end]
        if df_day.empty:
            continue

        if df_day["temp"].isna().any():
            warnings.warn(
                f"NaN temperature on {day_ts.date()}; those hours will be skipped."
            )
            df_day = df_day.dropna(subset=["temp"])
            if df_day.empty:
                continue

        try:
            m     = NaiveMLR(**model_kwargs)
            preds = m.fit_predict(df_train, df_day)
        except Exception as exc:
            warnings.warn(f"Forecast failed for {day_ts.date()}: {exc}")
            continue

        results.append(
            pd.DataFrame(
                {"load_mw_actual": df_day["load_mw"], "load_mw_pred": preds}
            )
        )

    if not results:
        return pd.DataFrame(columns=["load_mw_actual", "load_mw_pred"])
    return pd.concat(results)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    groupby: Optional[str] = None,
) -> Dict:
    """
    Compute point-forecast accuracy metrics.

    Parameters
    ----------
    actual : pd.Series
        Actual load values (MW), DatetimeIndex.
    predicted : pd.Series
        Predicted load values (MW), same index as actual.
    groupby : {'month', 'hour', 'dayofweek'} or None
        If given, compute metrics for each group separately and return a
        dict-of-dicts keyed by group label.

    Returns
    -------
    dict
        Keys: ``'MAE'``, ``'RMSE'``, ``'MAPE'``, ``'CVRMSE'`` (all floats;
        MAPE and CVRMSE are percentages).
    """
    actual    = pd.Series(actual,    dtype=float)
    predicted = pd.Series(predicted, dtype=float)

    def _metrics(a: np.ndarray, p: np.ndarray) -> Dict[str, float]:
        errors     = a - p
        abs_errors = np.abs(errors)
        mae    = float(np.mean(abs_errors))
        rmse   = float(np.sqrt(np.mean(errors ** 2)))
        nonzero = a != 0
        mape   = float(np.mean(abs_errors[nonzero] / np.abs(a[nonzero])) * 100)
        cvrmse = rmse / float(np.mean(a)) * 100 if np.mean(a) != 0 else float("nan")
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "CVRMSE": cvrmse}

    if groupby is None:
        return _metrics(actual.values, predicted.values)

    group_key = {
        "month":     actual.index.month,
        "hour":      actual.index.hour,
        "dayofweek": actual.index.dayofweek,
    }.get(groupby)
    if group_key is None:
        raise ValueError(
            f"groupby must be 'month', 'hour', or 'dayofweek', got {groupby!r}"
        )

    out = {}
    for label in sorted(set(group_key)):
        mask = group_key == label
        out[label] = _metrics(actual.values[mask], predicted.values[mask])
    return out


def evaluate_forecast(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute metrics from the output of ``rolling_forecast()``.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns ``'load_mw_actual'`` and ``'load_mw_pred'``.

    Returns
    -------
    dict
        ``{'MAE': ..., 'RMSE': ..., 'MAPE': ..., 'CVRMSE': ...}``
    """
    return compute_metrics(results_df["load_mw_actual"], results_df["load_mw_pred"])
