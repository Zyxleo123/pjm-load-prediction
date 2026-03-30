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
# All stations, default features (matching DEFAULT_FEATURES in features.py)
python run_experiment.py

# Select specific stations
python run_experiment.py --stations=PHL,KOP,Levittown

# Toggle features (true/false, case-insensitive)
python run_experiment.py --weather_lags=true --rh=false --wspd=false

# Combine stations + features + custom output path
python run_experiment.py --stations=PHL,KOP --weather_lags=true --rh=false --output=results/phl_kop.json
"""

import argparse
import json
import os
import sys
import datetime

import pandas as pd
from sklearn.linear_model import LinearRegression

from features import fit_feature_matrices, predict_feature_matrix
from load import create_dataset
from models import compute_metrics

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
    "Funkytown": (38.750, -76.224),
}

# Feature flags (matching DEFAULT_FEATURES in features.py)
FEATURE_FLAGS = [
    "trend", "month", "day_hour",
    "month_weather", "hour_weather",
    "weather_lags", "weather_mavg", "weather_interactions",
    "temp", "rh", "dwpt", "wspd",
]

# Defaults mirror DEFAULT_FEATURES in features.py
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

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OLS load forecast with selectable stations and features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stations",
        default=",".join(PECO_COORDS.keys()),
        help=(
            "Comma-separated station names. "
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
        help="Output JSON name. Defaults to YYYYMMDD_HHMMSS",
    )
    return parser

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── Validate stations ─────────────────────────────────────────────────────
    station_names = [s.strip() for s in args.stations.split(",") if s.strip()]
    invalid = [s for s in station_names if s not in PECO_COORDS]
    if invalid:
        print(f"ERROR: Unknown station(s): {invalid}")
        print(f"Valid options: {list(PECO_COORDS.keys())}")
        sys.exit(1)
    if not station_names:
        print("ERROR: --stations must not be empty.")
        sys.exit(1)

    selected_stations = {name: PECO_COORDS[name] for name in station_names}

    # ── Resolve feature flags ─────────────────────────────────────────────────
    feature_selection = {flag: getattr(args, flag) for flag in FEATURE_FLAGS}

    # ── Print configuration ───────────────────────────────────────────────────
    print("=" * 60)
    print(f"Stations  : {', '.join(station_names)}")
    print("Features  :")
    for k, v in feature_selection.items():
        print(f"  {k:25s}: {v}")
    print("=" * 60)

    # ── Load / cache dataset ──────────────────────────────────────────────────
    df_name    = "-".join(sorted(station_names))
    cache_path = os.path.join("data", "full", f"{df_name}.csv")
    os.makedirs(os.path.join("data", "full"), exist_ok=True)

    if os.path.exists(cache_path):
        print(f"\nLoading cached dataset: {cache_path}")
        df_loc = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        print("\nFetching dataset (will be cached for future runs)...")
        df_loc = create_dataset(
            "hrl_load_metered_combined.csv", "PE", selected_stations
        )
        df_loc.to_csv(cache_path)
        print(f"Cached to {cache_path}")

    df_train = df_loc.loc["2016":"2022"]
    df_test  = df_loc.loc["2023"]
    print(
        f"Train: {df_train.index[0].date()} → {df_train.index[-1].date()} "
        f"({len(df_train):,} rows)"
    )
    print(
        f"Test : {df_test.index[0].date()} → {df_test.index[-1].date()} "
        f"({len(df_test):,} rows)"
    )

    # ── Features + fit & predict ─────────────────────────────────────────────
    print("\nBuilding feature matrices and fitting OLS...")
    X_train, y_train, prep_state = fit_feature_matrices(
        df_train, features=feature_selection
    )
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X_train.values, y_train.values)
    X_test = predict_feature_matrix(
        df_test, prep_state, features=feature_selection
    )
    preds = pd.Series(
        lr.predict(X_test.values),
        index=X_test.index,
        name="load_mw_pred",
    )

    # ── Compute metrics ───────────────────────────────────────────────────────
    def _round_metrics(m: dict) -> dict:
        return {k: round(v, 4) for k, v in m.items()}

    overall = compute_metrics(df_test["load_mw"], preds)
    metrics: dict = {"overall": _round_metrics(overall)}

    for season, months in SEASONS.items():
        mask = df_test.index.month.isin(months)
        m = compute_metrics(df_test.loc[mask, "load_mw"], preds[mask])
        metrics[season] = _round_metrics(m)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── Results ──────────────────────────────────────────────")
    for section, vals in metrics.items():
        print(
            f"  {section:8s}  MAE={vals['MAE']:7.1f}  RMSE={vals['RMSE']:7.1f}  "
            f"MAPE={vals['MAPE']:.3f}%  CVRMSE={vals['CVRMSE']:.3f}%"
        )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "stations":  station_names,
            "features":  feature_selection,
            "train":     "2016-2022",
            "test":      "2023",
        },
        "metrics": metrics,
    }

    if args.name is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = ts

    out_dir = 'results'
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(out_dir, f"{args.name}.json") if out_dir else f"{args.name}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    pred_path = os.path.splitext(output_path)[0] + "_predictions.csv"
    preds.to_csv(pred_path, header=True)

    print(f"\nSaved to:      {output_path}")
    print(f"Predictions:   {pred_path}")


if __name__ == "__main__":
    main()
