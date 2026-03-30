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

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, Optional

from features import (
    ForecastPrepState,
    fit_feature_matrices,
    predict_feature_matrix,
)


class NaiveMLR:
    """
    GLMLF-B: Naïve MLR benchmark for short-term load forecasting.

    Wraps sklearn LinearRegression with feature matrices from
    ``features.fit_feature_matrices`` / ``features.predict_feature_matrix``.
    When recency features are enabled, the model stores a training tail and
    prepends it for prediction so lag/moving-average features are defined on
    the first test hour.

    Parameters
    ----------
    features : dict, optional
        Feature group → bool; merged with ``DEFAULT_FEATURES`` in
        ``features.build_features``.  Example::

            NaiveMLR(features={"weather_lags": True, "weather_mavg": True})

    n_lags : int, optional
        Number of lagged hourly weather variables (Wang et al. 2016).
        Default 12.
    n_mavg_days : int, optional
        Number of daily moving-average weather variables (Wang et al.
        2016).  Default 2.

    Usage
    -----
    model = NaiveMLR()                             # base features only
    model = NaiveMLR(features={"weather_lags": True}, n_lags=4)
    model = NaiveMLR(features={"weather_mavg": True}, n_mavg_days=2)
    model = NaiveMLR(features={"weather_lags": True, "weather_mavg": True},
                     n_lags=4, n_mavg_days=2)

    preds = model.fit_predict(df_train, df_test)
    """

    def __init__(
        self,
        features: Optional[Dict[str, bool]] = None,
        n_lags: int = 12,
        n_mavg_days: int = 2,
    ) -> None:
        self._features    = features
        self._n_lags      = n_lags
        self._n_mavg_days = n_mavg_days

        self._lr: Optional[LinearRegression] = None
        self.feature_names_: Optional[list]  = None
        self.n_train_rows_: int = 0
        self._prep_state: Optional[ForecastPrepState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df_train: pd.DataFrame) -> "NaiveMLR":
        """
        Fit OLS on df_train.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data with DatetimeIndex, ``'load_mw'``, and one or more
            weather columns (e.g. ``'temp'``, ``'philadelphia_temp'``, ``'rh'``).

        Returns
        -------
        self
        """
        X, y, self._prep_state = fit_feature_matrices(
            df_train,
            features=self._features,
            n_lags=self._n_lags,
            n_mavg_days=self._n_mavg_days,
        )
        self._lr = LinearRegression(fit_intercept=True)
        self._lr.fit(X.values, y.values)
        self.feature_names_ = list(self._prep_state.feature_names)
        self.n_train_rows_  = len(X)
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
            Test/forecast data with DatetimeIndex and the same weather columns
            used during training.  ``'load_mw'`` is not required.
        trend_offset : int, optional
            Override for the trend counter start.  Defaults to the last
            trend value from training (set by ``fit()``), so the test
            trend continues without a gap.

        Returns
        -------
        pd.Series
            Predicted load in MW, named ``'load_mw_pred'``.
        """
        if self._lr is None or self._prep_state is None:
            raise RuntimeError("Call fit() before predict().")

        X_test = predict_feature_matrix(
            df_test,
            self._prep_state,
            features=self._features,
            n_lags=self._n_lags,
            n_mavg_days=self._n_mavg_days,
            trend_offset=trend_offset,
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
# Evaluation metrics
# ---------------------------------------------------------------------------

def _peak_valley_mapes(
    actual: pd.Series,
    predicted: pd.Series,
) -> Dict[str, float]:
    """
    Compute four daily peak/valley MAPE metrics (averaged across all days).

    peak_load_mape
        MAPE between predicted daily maximum and actual daily maximum.
        Measures how well the model predicts the magnitude of the peak,
        regardless of when it occurs.

    valley_load_mape
        Same but for the daily minimum (valley).

    peak_hour_load_mape
        For each day, find the hour where the *actual* peak occurs, then
        compute MAPE between the prediction at that hour and the actual load
        at that hour.  Measures how well the model predicts the load at the
        true peak time (even if it predicts a different hour as the peak).

    valley_hour_load_mape
        Same but at the hour of the actual daily valley.

    Both series must have a DatetimeIndex.
    """
    dates = actual.index.date
    unique_dates = np.unique(dates)

    peak_load_apes:       list = []
    valley_load_apes:     list = []
    peak_hour_apes:       list = []
    valley_hour_apes:     list = []

    for d in unique_dates:
        mask  = dates == d
        a_day = actual[mask]
        p_day = predicted[mask]

        # deduplicate
        a_day = a_day[~a_day.index.duplicated(keep='first')]
        p_day = p_day[~p_day.index.duplicated(keep='first')]

        if len(a_day) == 0:
            continue

        # 1) Peak load: predicted daily max vs actual daily max
        a_max = float(a_day.max())
        p_max = float(p_day.max())
        if a_max != 0:
            peak_load_apes.append(abs(p_max - a_max) / abs(a_max))

        # 2) Valley load: predicted daily min vs actual daily min
        a_min = float(a_day.min())
        p_min = float(p_day.min())
        if a_min != 0:
            valley_load_apes.append(abs(p_min - a_min) / abs(a_min))

        # 3) Peak-hour load: at the actual-peak timestamp, pred vs actual
        peak_ts = a_day.idxmax()
        a_at_peak = float(a_day[peak_ts])
        p_at_peak = float(p_day[peak_ts])
        if a_at_peak != 0:
            peak_hour_apes.append(abs(p_at_peak - a_at_peak) / abs(a_at_peak))

        # 4) Valley-hour load: at the actual-valley timestamp, pred vs actual
        valley_ts = a_day.idxmin()
        a_at_valley = float(a_day[valley_ts])
        p_at_valley = float(p_day[valley_ts])
        if a_at_valley != 0:
            valley_hour_apes.append(abs(p_at_valley - a_at_valley) / abs(a_at_valley))

    def _mean_pct(apes: list) -> float:
        return float(np.mean(apes) * 100) if apes else float("nan")

    return {
        "peak_load_mape":        _mean_pct(peak_load_apes),
        "valley_load_mape":      _mean_pct(valley_load_apes),
        "peak_hour_load_mape":   _mean_pct(peak_hour_apes),
        "valley_hour_load_mape": _mean_pct(valley_hour_apes),
    }


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
        When ``groupby`` is None, keys are:
        ``'MAE'``, ``'RMSE'``, ``'MAPE'``, ``'CVRMSE'`` (all floats;
        MAPE and CVRMSE are percentages), plus four daily peak/valley MAPE
        metrics (see :func:`_peak_valley_mapes`):
        ``'peak_load_mape'``, ``'valley_load_mape'``,
        ``'peak_hour_load_mape'``, ``'valley_hour_load_mape'``.

        When ``groupby`` is given, returns a dict-of-dicts keyed by group
        label with only the base metrics (no peak/valley metrics per group).
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
        result = _metrics(actual.values, predicted.values)
        result.update(_peak_valley_mapes(actual, predicted))
        return result

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
        ``{'MAE': ..., 'RMSE': ..., 'MAPE': ..., 'CVRMSE': ...,
           'peak_load_mape': ..., 'valley_load_mape': ...,
           'peak_hour_load_mape': ..., 'valley_hour_load_mape': ...}``
    """
    return compute_metrics(results_df["load_mw_actual"], results_df["load_mw_pred"])
