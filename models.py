"""
Hong et al. (2011) GLMLF-B Naïve Multiple Linear Regression benchmark.

Reference:
    Hong, T., Gui, M., Baran, M., & Willis, H. L. (2011).
    A Naïve Multiple Linear Regression Benchmark for Short Term Load Forecasting.
    IEEE Power and Energy Society General Meeting.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, Optional

from features import build_features


class NaiveMLR:
    """
    GLMLF-B: Naïve MLR benchmark for short-term load forecasting.

    Wraps sklearn LinearRegression with the feature engineering from
    features.build_features().  Maintains the global Trend counter across
    fit / predict calls so that test-set Trend values continue from where
    the training set ended.

    Usage
    -----
    model = NaiveMLR()
    model.fit(df_train)
    preds = model.predict(df_test)
    # or simply:
    preds = model.fit_predict(df_train, df_test)
    """

    def __init__(self) -> None:
        self._lr: Optional[LinearRegression] = None
        self.feature_names_: Optional[list] = None
        self.n_train_rows_: int = 0

    def fit(self, df_train: pd.DataFrame) -> "NaiveMLR":
        """
        Fit OLS on df_train.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data with DatetimeIndex, 'load_mw', and 'temp' columns.

        Returns
        -------
        self
        """
        X, y = build_features(df_train, trend_offset=0)
        self._lr = LinearRegression(fit_intercept=True)
        self._lr.fit(X.values, y.values)
        self.feature_names_ = list(X.columns)
        self.n_train_rows_ = len(X)
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
            Test/forecast data with DatetimeIndex and 'temp' column.
            'load_mw' is not required (but harmless if present).
        trend_offset : int, optional
            Value added to the local 1-based Trend counter so that Trend
            continues from where training ended.  Defaults to n_train_rows_
            (set automatically by fit).

        Returns
        -------
        pd.Series
            Predicted load in MW, named 'load_mw_pred', same index as df_test
            after NaN rows are dropped.
        """
        if self._lr is None:
            raise RuntimeError("Call fit() before predict().")

        if trend_offset is None:
            trend_offset = self.n_train_rows_

        # build_features requires 'load_mw'; inject a placeholder if absent
        df_pred = df_test.copy()
        if 'load_mw' not in df_pred.columns:
            df_pred['load_mw'] = 0.0

        X_test, _ = build_features(df_pred, trend_offset=trend_offset)

        if list(X_test.columns) != self.feature_names_:
            raise ValueError(
                f"Feature mismatch: expected {len(self.feature_names_)} cols, "
                f"got {len(X_test.columns)}. Make sure test data covers all "
                "calendar combinations (use a full-year or multi-year window)."
            )

        preds = self._lr.predict(X_test.values)
        return pd.Series(preds, index=X_test.index, name='load_mw_pred')

    def fit_predict(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> pd.Series:
        """
        Convenience: fit on df_train, then predict on df_test.

        Trend for df_test begins at len(df_train) + 1.
        """
        self.fit(df_train)
        return self.predict(df_test, trend_offset=self.n_train_rows_)


# ---------------------------------------------------------------------------
# Rolling forecast
# ---------------------------------------------------------------------------

def rolling_forecast(
    df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    min_train_days: int = 30,
) -> pd.DataFrame:
    """
    Rolling daily retrain-and-forecast loop.

    For each calendar day d in [forecast_start, forecast_end]:
      1. Training set = all rows where timestamp < midnight of day d.
      2. Skip day d if training set has < min_train_days * 24 rows.
      3. Fit NaiveMLR on training set.
      4. Predict the 24 hours of day d (23 on spring-forward, 25 on fall-back).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with DatetimeIndex, 'load_mw', and 'temp'.
    forecast_start : pd.Timestamp
        First day to forecast (inclusive).
    forecast_end : pd.Timestamp
        Last day to forecast (inclusive).
    min_train_days : int, optional
        Minimum training days required before forecasting (default 30).

    Returns
    -------
    pd.DataFrame
        Columns ['load_mw_actual', 'load_mw_pred'], DatetimeIndex.
    """
    forecast_days = pd.date_range(
        forecast_start.normalize(), forecast_end.normalize(), freq='D'
    )
    model = NaiveMLR()
    results = []

    for day in forecast_days:
        day_ts = pd.Timestamp(day.date())
        df_train = df[df.index < day_ts]

        if len(df_train) < min_train_days * 24:
            continue

        day_end = day_ts + pd.Timedelta(hours=23)
        df_day = df.loc[day_ts:day_end]
        if df_day.empty:
            continue

        if df_day['temp'].isna().any():
            warnings.warn(
                f"NaN temperature on {day_ts.date()}; those hours will be skipped."
            )
            df_day = df_day.dropna(subset=['temp'])
            if df_day.empty:
                continue

        try:
            preds = model.fit_predict(df_train, df_day)
        except Exception as exc:
            warnings.warn(f"Forecast failed for {day_ts.date()}: {exc}")
            continue

        results.append(
            pd.DataFrame(
                {'load_mw_actual': df_day['load_mw'], 'load_mw_pred': preds}
            )
        )

    if not results:
        return pd.DataFrame(columns=['load_mw_actual', 'load_mw_pred'])
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
        Keys: 'MAE', 'RMSE', 'MAPE', 'CVRMSE' (all floats; MAPE and CVRMSE
        are percentages).
    """
    actual = pd.Series(actual, dtype=float)
    predicted = pd.Series(predicted, dtype=float)

    def _metrics(a: np.ndarray, p: np.ndarray) -> Dict[str, float]:
        errors = a - p
        abs_errors = np.abs(errors)
        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        nonzero = a != 0
        mape = float(np.mean(abs_errors[nonzero] / np.abs(a[nonzero])) * 100)
        cvrmse = rmse / float(np.mean(a)) * 100 if np.mean(a) != 0 else float('nan')
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'CVRMSE': cvrmse}

    if groupby is None:
        return _metrics(actual.values, predicted.values)

    group_key = {
        'month': actual.index.month,
        'hour': actual.index.hour,
        'dayofweek': actual.index.dayofweek,
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
    Compute metrics from the output of rolling_forecast().

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns 'load_mw_actual' and 'load_mw_pred'.

    Returns
    -------
    dict
        {'MAE': ..., 'RMSE': ..., 'MAPE': ..., 'CVRMSE': ...}
    """
    return compute_metrics(results_df['load_mw_actual'], results_df['load_mw_pred'])
