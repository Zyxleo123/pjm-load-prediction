#!/usr/bin/env python3
"""
run_experiment.py — Load-forecasting experiment (OLS on engineered features) with
                    configurable feature flags and weather station selection.

Stations (from exploration.ipynb)
----------------------------------
PHL, Center_City, KOP, West_Chester, Coatesville,
Doylestown, Pottstown, Quakertown, Levittown, Peach_Bottom

Feature flags
-------------
Feature groups : trend, month, day_hour, month_weather, hour_weather,
                 weather_lags, weather_mavg, weather_interactions
Weather vars   : temp, rh, dwpt, wspd

Usage examples
--------------
# Default: Center_City only, default features (matching FEATURE_DEFAULTS below)
python run_experiment.py

# Multiple stations
python run_experiment.py --stations=PHL,KOP,Levittown

# Toggle features (true/false, case-insensitive)
python run_experiment.py --weather_lags=true --rh=false --wspd=false

# Persist artifacts under results/ (off by default; metrics still print to log)
python run_experiment.py --save-results-json
python run_experiment.py --save-predictions-csv
python run_experiment.py --save-results-json --save-predictions-csv

# 95% quantile (one-sided upper bound) + calibration metrics
python run_experiment.py --quantile=0.95

# Conformalized QR: offsets from the last training year, per season (spring cal → spring test)
python run_experiment.py --quantile=0.95 --cqr

# Pool all months in the calibration year (no seasonal stratification)
python run_experiment.py --quantile=0.95 --cqr --cqr-pooled

# Scale conformal η per season (order: Winter,Spring,Summer,Fall); e.g. only winter:
python run_experiment.py --quantile=0.95 --cqr --cqr-season-weights=1,0,0,0

# Quieter logs, or log every training epoch (quantile mode)
python run_experiment.py --log-level=WARNING
python run_experiment.py --quantile=0.95 --qr-log-interval=1
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression

from features import fit_feature_matrices, predict_feature_matrix
from load import create_dataset
from models import (
    NaiveQuantileMLR,
    apply_seasonal_cqr_adjustment,
    compute_metrics,
    compute_quantile_interval_metrics,
    compute_seasonal_cqr_offsets,
)

# ---------------------------------------------------------------------------
# Station registry (from exploration.ipynb)
# ---------------------------------------------------------------------------

PECO_COORDS = {
    "PHL":          (39.877, -75.225),
    "Center_City":  (39.950, -75.158),
    "KOP":          (40.097, -75.386),
    "West_Chester": (39.960, -75.605),
    "Coatesville":  (39.983, -75.826),
    "Doylestown":   (40.310, -75.130),
    "Pottstown":    (40.245, -75.649),
    "Quakertown":   (40.441, -75.344),
    "Levittown":    (40.155, -74.830),
    "Peach_Bottom": (39.750, -76.224),
    "Funkytown":      (38.750, -76.224),
}

FEATURE_FLAGS = [
    "trend", "month", "day_hour",
    "month_weather", "hour_weather",
    "weather_lags", "weather_mavg", "weather_interactions",
    "temp", "rh", "dwpt", "wspd",
]

FEATURE_DEFAULTS = {
    "trend":                True,
    "month":                True,
    "day_hour":             True,
    "month_weather":        True,
    "hour_weather":         True,
    "weather_lags":         False,
    "weather_mavg":         True,
    "weather_interactions": False,
    "temp":                 True,
    "rh":                   False,
    "dwpt":                 True,
    "wspd":                 False,
}

SEASONS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Fall":   [9, 10, 11],
}

TRAIN_SLICE = "2016", "2022"
TEST_SLICE = "2023"


# ---------------------------------------------------------------------------
# Argument parsing & logging
# ---------------------------------------------------------------------------

def _parse_bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def _parse_cqr_season_weights(value: str) -> Dict[str, float]:
    """Comma-separated weights in SEASONS key order: Winter,Spring,Summer,Fall."""
    parts = [p.strip() for p in value.split(",")]
    keys = list(SEASONS.keys())
    if len(parts) != len(keys):
        raise argparse.ArgumentTypeError(
            f"Expected {len(keys)} comma-separated weights "
            f"({', '.join(keys)}), got {len(parts)} values."
        )
    out: Dict[str, float] = {}
    for k, p in zip(keys, parts):
        try:
            w = float(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid weight for {k!r}: {p!r}"
            ) from exc
        if w < 0 or not math.isfinite(w):
            raise argparse.ArgumentTypeError(
                f"Weight for {k!r} must be finite and >= 0, got {w}"
            )
        out[k] = w
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OLS load forecast with selectable stations and features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stations",
        default="Center_City",
        help=(
            "Comma-separated station names (default: Center_City). "
            "Available: " + ", ".join(PECO_COORDS.keys())
        ),
    )
    for flag in FEATURE_FLAGS:
        parser.add_argument(
            f"--{flag}",
            type=_parse_bool,
            default=FEATURE_DEFAULTS[flag],
            metavar="BOOL",
        )
    parser.add_argument(
        "--name",
        default=None,
        help="Output JSON basename (no path). Defaults to YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "--save-results-json",
        action="store_true",
        help=(
            "Write results/<name>.json (metrics, config, optional quantile_metrics). "
            "Default: no JSON file."
        ),
    )
    parser.add_argument(
        "--save-predictions-csv",
        action="store_true",
        help=(
            "Write results/<name>_predictions.csv (load_mw_pred series). "
            "Default: no predictions file."
        ),
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=None,
        metavar="TAU",
        help=(
            "If set (e.g. 0.95), fit linear quantile regression at level TAU "
            "instead of OLS mean. Omit this flag for pure OLS (mean forecast). "
            "With quantile: reports ECE and sharpness for the one-sided "
            "upper interval (-inf, q_hat]."
        ),
    )
    parser.add_argument(
        "--qr-alpha",
        type=float,
        default=1e-6,
        metavar="A",
        help="L1 penalty on feature weights in torch quantile fit (only with --quantile).",
    )
    parser.add_argument(
        "--qr-lr",
        type=float,
        default=0.05,
        metavar="LR",
        help="Adam learning rate for torch quantile fit.",
    )
    parser.add_argument(
        "--qr-epochs",
        type=int,
        default=400,
        metavar="N",
        help="Training epochs (full passes over training rows) for torch quantile fit.",
    )
    parser.add_argument(
        "--qr-batch-size",
        type=int,
        default=8192,
        metavar="B",
        help="Minibatch size for torch quantile fit.",
    )
    parser.add_argument(
        "--qr-seed",
        type=int,
        default=0,
        metavar="S",
        help="RNG seed for torch quantile fit.",
    )
    parser.add_argument(
        "--qr-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for torch quantile fit (cuda if available when auto).",
    )
    parser.add_argument(
        "--qr-log-interval",
        type=int,
        default=1,
        metavar="E",
        help=(
            "Log torch quantile training every E epochs (default: ~20 progress "
            "lines per run)."
        ),
    )
    parser.add_argument(
        "--cqr",
        action="store_true",
        help=(
            "Conformalize the QR upper bound using the last training year as a "
            "calibration set (requires --quantile). By default, offsets are "
            "computed separately per season (same SEASONS as metrics); use "
            "--cqr-pooled for one global offset."
        ),
    )
    parser.add_argument(
        "--cqr-cal-year",
        type=int,
        default=None,
        metavar="Y",
        help=(
            "Calendar year for CQR calibration (default: last year of TRAIN_SLICE, "
            "currently the end year of the train range)."
        ),
    )
    parser.add_argument(
        "--cqr-pooled",
        action="store_true",
        help=(
            "Use a single conformal offset from all calibration-year rows instead "
            "of per-season offsets."
        ),
    )
    parser.add_argument(
        "--cqr-min-season-n",
        type=int,
        default=24,
        metavar="N",
        help=(
            "Minimum finite calibration scores per season before falling back to "
            "the pooled offset (only when not --cqr-pooled)."
        ),
    )
    parser.add_argument(
        "--cqr-offset-quantile",
        type=float,
        default=None,
        metavar="Q",
        help=(
            "Quantile rank for calibration residual offsets (default: same as "
            "--quantile). Use a slightly lower Q (e.g. 0.93 with --quantile=0.95) "
            "if CQR makes bounds too loose (high coverage / ECE vs target). "
            "Metrics still compare coverage to --quantile."
        ),
    )
    parser.add_argument(
        "--cqr-season-weights",
        default=None,
        type=_parse_cqr_season_weights,
        metavar="W,W,W,W",
        help=(
            "Per-season multipliers on conformal offsets η (not compatible with "
            "--cqr-pooled). Four comma-separated non-negative numbers in calendar "
            "season order matching SEASONS: Winter,Spring,Summer,Fall. "
            "0 disables CQR for that season; 1 applies full η; values in (0,1) "
            "partially shrink the bump (e.g. 0.93,0,0,0 = winter only at 93%% of η)."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console log verbosity for runtime messages.",
    )
    return parser


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_metrics(m: Dict[str, float]) -> Dict[str, float]:
    return {k: round(v, 4) for k, v in m.items()}


def _round_quantile_metrics(m: Dict[str, float]) -> Dict[str, float]:
    return {k: round(v, 6) for k, v in m.items()}


def parse_station_names(stations_arg: str) -> List[str]:
    return [s.strip() for s in stations_arg.split(",") if s.strip()]


def validate_stations(station_names: List[str], log: logging.Logger) -> None:
    invalid = [s for s in station_names if s not in PECO_COORDS]
    if invalid:
        log.error("Unknown station(s): %s", invalid)
        log.error("Valid options: %s", list(PECO_COORDS.keys()))
        sys.exit(1)
    if not station_names:
        log.error("--stations must not be empty.")
        sys.exit(1)


def feature_dict_from_args(args: argparse.Namespace) -> Dict[str, bool]:
    return {flag: getattr(args, flag) for flag in FEATURE_FLAGS}


def load_or_fetch_dataset(
    log: logging.Logger,
    station_names: List[str],
) -> pd.DataFrame:
    selected = {name: PECO_COORDS[name] for name in station_names}
    df_name = "-".join(sorted(station_names))
    cache_path = os.path.join("data", "full", f"{df_name}.csv")
    os.makedirs(os.path.join("data", "full"), exist_ok=True)

    t0 = time.perf_counter()
    if os.path.exists(cache_path):
        log.info("Loading cached dataset: %s", cache_path)
        df_loc = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        log.info("Fetching dataset (will be cached for future runs)...")
        df_loc = create_dataset(
            "hrl_load_metered_combined.csv", "PE", selected
        )
        df_loc.to_csv(cache_path)
        log.info("Cached to %s", cache_path)
    log.info(
        "Dataset load done in %.2fs (%d rows, %d columns)",
        time.perf_counter() - t0,
        len(df_loc),
        len(df_loc.columns),
    )
    return df_loc


def train_test_split(
    df_loc: pd.DataFrame,
    log: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df_loc.loc[TRAIN_SLICE[0] : TRAIN_SLICE[1]]
    df_test = df_loc.loc[TEST_SLICE]
    log.info(
        "Train: %s → %s (%s rows)",
        df_train.index[0].date(),
        df_train.index[-1].date(),
        f"{len(df_train):,}",
    )
    log.info(
        "Test:  %s → %s (%s rows)",
        df_test.index[0].date(),
        df_test.index[-1].date(),
        f"{len(df_test):,}",
    )
    return df_train, df_test


def _default_cqr_cal_year() -> int:
    """Last calendar year in the fixed train range (calibration slice)."""
    return int(TRAIN_SLICE[1])


def _resolve_cqr_cal_year(args: argparse.Namespace) -> int:
    if args.cqr_cal_year is not None:
        return int(args.cqr_cal_year)
    return _default_cqr_cal_year()


def fit_predict_quantile_cqr(
    args: argparse.Namespace,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_selection: Dict[str, bool],
    log: logging.Logger,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Fit QR on the full training window, then conformalize test predictions.

    The QR model is fit on all of ``df_train`` so lag / moving-average features
    stay continuous into the test year. Conformal scores use only rows from
    ``cqr_cal_year`` (default: last training year). Per-season offsets map
    calibration months to the same season at test time.
    """
    import torch

    if not 0.0 < args.quantile < 1.0:
        log.error("--quantile must be strictly between 0 and 1.")
        sys.exit(1)

    if args.cqr_pooled and args.cqr_season_weights is not None:
        log.error("--cqr-season-weights is not compatible with --cqr-pooled.")
        sys.exit(1)

    cal_year = _resolve_cqr_cal_year(args)
    train_years = sorted(set(df_train.index.year))
    if cal_year not in train_years:
        log.error(
            "CQR calibration year %s not in training index years %s",
            cal_year,
            train_years,
        )
        sys.exit(1)

    df_cal = df_train[df_train.index.year == cal_year]
    if len(df_cal) == 0:
        log.error("CQR: no training rows for calibration year %s", cal_year)
        sys.exit(1)

    oq = args.cqr_offset_quantile
    if oq is not None and not 0.0 < oq < 1.0:
        log.error("--cqr-offset-quantile must be strictly between 0 and 1.")
        sys.exit(1)

    qr_device = _resolve_qr_device(args, log)
    log.info(
        "CQR: fit QR τ=%s on full train (%s → %s), calibrate on year %s (%s rows, %s)",
        args.quantile,
        df_train.index[0].date(),
        df_train.index[-1].date(),
        cal_year,
        f"{len(df_cal):,}",
        "pooled" if args.cqr_pooled else "per-season",
    )
    if oq is not None and oq != args.quantile:
        log.info(
            "CQR residual rank Q=%s (QR fit still targets τ=%s)",
            oq,
            args.quantile,
        )

    model = NaiveQuantileMLR(
        quantile=args.quantile,
        features=feature_selection,
        alpha=args.qr_alpha,
        learning_rate=args.qr_lr,
        max_epochs=args.qr_epochs,
        batch_size=args.qr_batch_size,
        device=qr_device,
        seed=args.qr_seed,
        logger=log,
        log_interval=args.qr_log_interval,
    )
    model.fit(df_train)
    q_cal = model.predict(df_cal)
    offsets, detail = compute_seasonal_cqr_offsets(
        df_cal["load_mw"],
        q_cal,
        SEASONS,
        args.quantile,
        offset_quantile=oq,
        pooled=args.cqr_pooled,
        min_season_n=args.cqr_min_season_n,
    )
    sw = args.cqr_season_weights
    if not args.cqr_pooled:
        for name, eta in offsets.items():
            w = 1.0 if sw is None else float(sw[name])
            eff = eta * w
            if sw is None or w == 1.0:
                log.info("  CQR offset %-8s η = %.4f MW", name, eta)
            else:
                log.info(
                    "  CQR offset %-8s η = %.4f MW × w=%.4g → %.4f MW",
                    name,
                    eta,
                    w,
                    eff,
                )
    else:
        log.info("  CQR pooled η = %.4f MW", offsets["__pooled__"])

    t0 = time.perf_counter()
    q_test = model.predict(df_test)
    preds = apply_seasonal_cqr_adjustment(
        q_test,
        SEASONS,
        offsets,
        pooled=args.cqr_pooled,
        season_weights=sw,
    )
    log.info("CQR test prediction done in %.2fs", time.perf_counter() - t0)
    dev_used = (
        qr_device
        if qr_device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Quantile fit device: %s", dev_used)

    meta: Dict[str, Any] = {
        "cal_year": cal_year,
        "pooled": args.cqr_pooled,
        "cqr_offset_quantile": oq if oq is not None else args.quantile,
        "offsets": {k: round(v, 6) for k, v in offsets.items()},
        "detail": detail,
    }
    if sw is not None:
        meta["season_weights"] = {k: round(v, 6) for k, v in sw.items()}
        meta["effective_offsets"] = {
            k: round(float(sw[k]) * float(offsets[k]), 6) for k in offsets
        }
    return preds, meta


def _resolve_qr_device(args: argparse.Namespace, log: logging.Logger):
    import torch

    if args.qr_device == "cpu":
        return torch.device("cpu")
    if args.qr_device == "cuda":
        if not torch.cuda.is_available():
            log.error("--qr-device=cuda but CUDA is not available.")
            sys.exit(1)
        return torch.device("cuda")
    return None


def fit_predict_quantile(
    args: argparse.Namespace,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_selection: Dict[str, bool],
    log: logging.Logger,
) -> pd.Series:
    import torch

    if not 0.0 < args.quantile < 1.0:
        log.error("--quantile must be strictly between 0 and 1.")
        sys.exit(1)

    qr_device = _resolve_qr_device(args, log)
    log.info(
        "Quantile mode τ=%s (PyTorch Adam); building features and fitting...",
        args.quantile,
    )
    model = NaiveQuantileMLR(
        quantile=args.quantile,
        features=feature_selection,
        alpha=args.qr_alpha,
        learning_rate=args.qr_lr,
        max_epochs=args.qr_epochs,
        batch_size=args.qr_batch_size,
        device=qr_device,
        seed=args.qr_seed,
        logger=log,
        log_interval=args.qr_log_interval,
    )
    model.fit(df_train)
    t0 = time.perf_counter()
    preds = model.predict(df_test)
    log.info("Test prediction done in %.2fs", time.perf_counter() - t0)
    dev_used = (
        qr_device
        if qr_device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Quantile fit device: %s", dev_used)
    return preds


def fit_predict_ols(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_selection: Dict[str, bool],
    log: logging.Logger,
) -> pd.Series:
    log.info("OLS mode: building feature matrices and fitting...")
    t0 = time.perf_counter()
    X_train, y_train, prep_state = fit_feature_matrices(
        df_train, features=feature_selection
    )
    log.info(
        "Training features shape %s built in %.2fs",
        X_train.shape,
        time.perf_counter() - t0,
    )
    t0 = time.perf_counter()
    X_test = predict_feature_matrix(
        df_test, prep_state, features=feature_selection
    )
    log.info(
        "Test features shape %s built in %.2fs",
        X_test.shape,
        time.perf_counter() - t0,
    )
    est = LinearRegression(fit_intercept=True)
    log.info("Fitting LinearRegression...")
    t_fit = time.perf_counter()
    est.fit(X_train.values, y_train.values)
    log.info("Regression fit done in %.2fs", time.perf_counter() - t_fit)
    t_pred = time.perf_counter()
    pred_arr = est.predict(X_test.values)
    log.info("predict done in %.2fs", time.perf_counter() - t_pred)
    return pd.Series(pred_arr, index=X_test.index, name="load_mw_pred")


def fit_and_predict(
    args: argparse.Namespace,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_selection: Dict[str, bool],
    log: logging.Logger,
) -> pd.Series:
    if args.quantile is not None:
        return fit_predict_quantile(
            args, df_train, df_test, feature_selection, log
        )
    return fit_predict_ols(df_train, df_test, feature_selection, log)


def evaluate_predictions(
    df_test: pd.DataFrame,
    preds: pd.Series,
    quantile: Optional[float],
    log: logging.Logger,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    t0 = time.perf_counter()
    overall = compute_metrics(df_test["load_mw"], preds)
    metrics: Dict[str, Any] = {"overall": _round_metrics(overall)}

    for season, months in SEASONS.items():
        mask = df_test.index.month.isin(months)
        m = compute_metrics(df_test.loc[mask, "load_mw"], preds[mask])
        metrics[season] = _round_metrics(m)
    log.info("Point metrics computed in %.2fs", time.perf_counter() - t0)

    quantile_metrics: Optional[Dict[str, Any]] = None
    if quantile is not None:
        q_overall = compute_quantile_interval_metrics(
            df_test["load_mw"], preds, quantile
        )
        quantile_metrics = {"overall": _round_quantile_metrics(q_overall)}
        for season, months in SEASONS.items():
            mask = df_test.index.month.isin(months)
            qm = compute_quantile_interval_metrics(
                df_test.loc[mask, "load_mw"],
                preds[mask],
                quantile,
            )
            quantile_metrics[season] = _round_quantile_metrics(qm)

    return metrics, quantile_metrics


def log_results_tables(
    log: logging.Logger,
    metrics: Dict[str, Any],
    quantile_metrics: Optional[Dict[str, Any]],
) -> None:
    log.info("")
    log.info("── Results ──────────────────────────────────────────────")
    for section, vals in metrics.items():
        log.info(
            "  %-8s  MAE=%7.1f  RMSE=%7.1f  MAPE=%.3f%%  CVRMSE=%.3f%%",
            section,
            vals["MAE"],
            vals["RMSE"],
            vals["MAPE"],
            vals["CVRMSE"],
        )
    if quantile_metrics is not None:
        log.info("── Quantile interval (one-sided upper) ──────────────────")
        for section, vals in quantile_metrics.items():
            log.info(
                "  %-8s  coverage=%.4f  ECE=%.4f  sharpness=%.6f (mean (q-y)/y)",
                section,
                vals["coverage"],
                vals["ece"],
                vals["sharpness"],
            )


def build_experiment_config(
    args: argparse.Namespace,
    station_names: List[str],
    feature_selection: Dict[str, bool],
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "stations":        station_names,
        "features":        feature_selection,
        "train":           f"{TRAIN_SLICE[0]}-{TRAIN_SLICE[1]}",
        "test":            TEST_SLICE,
        "quantile":        args.quantile,
        "qr_alpha":        args.qr_alpha if args.quantile is not None else None,
        "qr_lr":           args.qr_lr if args.quantile is not None else None,
        "qr_epochs":       args.qr_epochs if args.quantile is not None else None,
        "qr_batch_size":   args.qr_batch_size if args.quantile is not None else None,
        "qr_seed":         args.qr_seed if args.quantile is not None else None,
        "qr_device":       args.qr_device if args.quantile is not None else None,
        "qr_log_interval": args.qr_log_interval if args.quantile is not None else None,
        "cqr":             bool(args.cqr),
        "cqr_cal_year":    _resolve_cqr_cal_year(args) if args.cqr else None,
        "cqr_pooled":      bool(args.cqr_pooled) if args.cqr else None,
        "cqr_min_season_n": args.cqr_min_season_n if args.cqr else None,
        "cqr_offset_quantile": (
            args.cqr_offset_quantile if args.cqr else None
        ),
        "cqr_season_weights": (
            {k: round(v, 6) for k, v in args.cqr_season_weights.items()}
            if args.cqr and args.cqr_season_weights is not None
            else None
        ),
        "log_level":            args.log_level,
        "save_results_json":    args.save_results_json,
        "save_predictions_csv": args.save_predictions_csv,
    }
    return cfg


def write_json(path: str, payload: Dict[str, Any]) -> float:
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return time.perf_counter() - t0


def write_predictions_csv(path: str, preds: pd.Series) -> None:
    preds.to_csv(path, header=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)
    log = logging.getLogger(__name__)

    station_names = parse_station_names(args.stations)
    validate_stations(station_names, log)
    feature_selection = feature_dict_from_args(args)

    if args.cqr_season_weights is not None and not args.cqr:
        log.error("--cqr-season-weights requires --cqr.")
        sys.exit(1)

    log.info("=" * 60)
    log.info("Stations: %s", ", ".join(station_names))
    log.info("Features:")
    for k, v in feature_selection.items():
        log.info("  %-25s: %s", k, v)
    log.info("=" * 60)

    df_loc = load_or_fetch_dataset(log, station_names)
    df_train, df_test = train_test_split(df_loc, log)

    if args.cqr:
        if args.quantile is None:
            log.error("--cqr requires --quantile.")
            sys.exit(1)
        if args.cqr_season_weights is not None and args.cqr_pooled:
            log.error("--cqr-season-weights is not compatible with --cqr-pooled.")
            sys.exit(1)
        preds, cqr_meta = fit_predict_quantile_cqr(
            args, df_train, df_test, feature_selection, log
        )
    else:
        cqr_meta = None
        preds = fit_and_predict(
            args, df_train, df_test, feature_selection, log
        )
    metrics, quantile_metrics = evaluate_predictions(
        df_test, preds, args.quantile, log
    )
    log_results_tables(log, metrics, quantile_metrics)

    result: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": build_experiment_config(args, station_names, feature_selection),
        "metrics": metrics,
    }
    if quantile_metrics is not None:
        result["quantile_metrics"] = quantile_metrics
    if cqr_meta is not None:
        result["cqr"] = cqr_meta

    out_name = args.name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "results"
    if args.save_results_json or args.save_predictions_csv:
        os.makedirs(out_dir, exist_ok=True)
    artifact_base = os.path.join(out_dir, out_name)

    if args.save_results_json:
        json_path = f"{artifact_base}.json"
        json_s = write_json(json_path, result)
        log.info("Wrote JSON in %.2fs — %s", json_s, json_path)
    elif args.save_predictions_csv:
        log.info(
            "Results JSON skipped (use --save-results-json to write metrics bundle)"
        )

    if args.save_predictions_csv:
        pred_path = f"{artifact_base}_predictions.csv"
        write_predictions_csv(pred_path, preds)
        log.info("Wrote predictions CSV — %s", pred_path)
    elif args.save_results_json:
        log.info(
            "Predictions CSV skipped (use --save-predictions-csv to write series)"
        )

    if not args.save_results_json and not args.save_predictions_csv:
        log.info("No files written under %s/; see log above for metrics.", out_dir)


if __name__ == "__main__":
    main()
