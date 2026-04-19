#!/usr/bin/env python3
"""
For each results/<name>_predictions.csv, load results/<name>.json config and
re-run run_experiment.py with 5 seeds. Report mean and STE (std/sqrt(n)) for
MAPE (overall + seasonal + variants), coverage, and calibration sharpness
(quantile / CQR runs only).

If results/<name>_multiseed_s<seed>.json already exists, it is reused and the
experiment is not re-run (unless --force-rerun). Use --aggregate-only to only
read existing multiseed JSONs and write the summary (no subprocess). If some
seed files are missing, aggregates over the rest and warns.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# Keep in sync with run_experiment.SEASONS key order for --cqr-season-weights
SEASON_ORDER = ("Winter", "Spring", "Summer", "Fall")

FEATURE_FLAGS = [
    "trend",
    "month",
    "day_hour",
    "month_weather",
    "hour_weather",
    "weather_lags",
    "weather_mavg",
    "weather_interactions",
    "temp",
    "rh",
    "dwpt",
    "wspd",
]


def _bool_arg(v: bool) -> str:
    return "true" if v else "false"


def config_to_argv(cfg: Dict[str, Any], seed: int, out_name: str) -> List[str]:
    argv: List[str] = [
        "run_experiment.py",
        "--stations",
        ",".join(cfg["stations"]),
        "--log-level",
        "WARNING",
        "--save-results-json",
        "--name",
        out_name,
    ]
    feats = cfg["features"]
    for k in FEATURE_FLAGS:
        argv.append(f"--{k}={_bool_arg(bool(feats[k]))}")

    q = cfg.get("quantile")
    if q is not None:
        argv.extend(
            [
                "--quantile",
                str(q),
                "--qr-alpha",
                str(cfg.get("qr_alpha", 1e-6)),
                "--qr-lr",
                str(cfg.get("qr_lr", 0.05)),
                "--qr-epochs",
                str(cfg.get("qr_epochs", 400)),
                "--qr-batch-size",
                str(cfg.get("qr_batch_size", 8192)),
                "--qr-seed",
                str(seed),
                "--qr-device",
                str(cfg.get("qr_device") or "auto"),
                "--qr-log-interval",
                str(cfg.get("qr_log_interval", 1)),
            ]
        )

    if cfg.get("cqr"):
        argv.append("--cqr")
        if cfg.get("cqr_cal_year") is not None:
            argv.extend(["--cqr-cal-year", str(int(cfg["cqr_cal_year"]))])
        if cfg.get("cqr_pooled"):
            argv.append("--cqr-pooled")
        if cfg.get("cqr_min_season_n") is not None:
            argv.extend(["--cqr-min-season-n", str(int(cfg["cqr_min_season_n"]))])
        if cfg.get("cqr_offset_quantile") is not None:
            argv.extend(
                ["--cqr-offset-quantile", str(cfg["cqr_offset_quantile"])]
            )
        sw = cfg.get("cqr_season_weights")
        if sw is not None:
            parts = [str(float(sw[s])) for s in SEASON_ORDER]
            argv.extend(["--cqr-season-weights", ",".join(parts)])

    return argv


def run_once(script_argv: List[str]) -> None:
    r = subprocess.run(
        [sys.executable, *script_argv],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.stderr.write(r.stdout)
        raise RuntimeError(f"run_experiment failed with code {r.returncode}")


def ste(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


SECTIONS = ("overall", "Winter", "Spring", "Summer", "Fall")
SEASONS_ONLY = ("Winter", "Spring", "Summer", "Fall")
MAPE_KEYS = (
    "MAPE",
    "peak_load_mape",
    "valley_load_mape",
    "peak_hour_load_mape",
    "valley_hour_load_mape",
)


def collect_numeric_rows(
    runs: List[Dict[str, Any]], has_q: bool
) -> Dict[str, Dict[str, List[float]]]:
    """section -> metric -> values across seeds."""
    out: Dict[str, Dict[str, List[float]]] = {}
    for sec in SECTIONS:
        out[sec] = {}
        for mk in MAPE_KEYS:
            out[sec][mk] = []
        if has_q:
            out[sec]["coverage"] = []
            out[sec]["sharpness"] = []

    for payload in runs:
        mets = payload["metrics"]
        qm = payload.get("quantile_metrics")
        for sec in SECTIONS:
            for mk in MAPE_KEYS:
                out[sec][mk].append(float(mets[sec][mk]))
            if has_q and qm is not None:
                out[sec]["coverage"].append(float(qm[sec]["coverage"]))
                out[sec]["sharpness"].append(float(qm[sec]["sharpness"]))
    return out


def fmt(x: float) -> str:
    if math.isnan(x):
        return "nan"
    # Use significant digits, not a fixed 6 decimal places — small STE was
    # often shown as 0.000000 despite being nonzero.
    return format(x, ".12g")


def _table_seasonal_mape(rows: Dict[str, Dict[str, List[float]]]) -> List[str]:
    """Compact seasonal headline MAPE (%)."""
    out = [
        "",
        "Seasonal MAPE (headline %, mean ± STE across seeds)",
        f"{'Season':<10}  {'MAPE_mean':>18}  {'MAPE_STE':>18}",
        "-" * 50,
    ]
    for sec in SEASONS_ONLY:
        xs = rows[sec]["MAPE"]
        out.append(
            f"{sec:<10}  {fmt(mean(xs)):>18}  {fmt(ste(xs)):>18}"
        )
    return out


def _table_calibration(rows: Dict[str, Dict[str, List[float]]]) -> List[str]:
    """Overall + per-season coverage and calibration sharpness."""
    out = [
        "",
        "Calibration: coverage & sharpness (mean ± STE; sharpness = mean (q−y)/y on test)",
        f"{'Scope':<10}  {'coverage_m':>16}  {'coverage_STE':>18}  "
        f"{'sharpness_m':>16}  {'sharpness_STE':>18}",
        "-" * 88,
    ]
    for sec in ("overall", *SEASONS_ONLY):
        cv = rows[sec]["coverage"]
        sh = rows[sec]["sharpness"]
        out.append(
            f"{sec:<10}  {fmt(mean(cv)):>16}  {fmt(ste(cv)):>18}  "
            f"{fmt(mean(sh)):>16}  {fmt(ste(sh)):>18}"
        )
    return out


def _multiseed_json_path(base: str, seed: int) -> Path:
    return RESULTS / f"{base}_multiseed_s{seed}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Only load existing results/<name>_multiseed_s*.json files; "
            "do not run experiments. Missing seeds are skipped; mean/STE use "
            "whatever files exist (STE is nan if n<2)."
        ),
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Always run run_experiment.py for each seed (overwrite JSON).",
    )
    parser.add_argument(
        "--only",
        default=None,
        metavar="BASE",
        help=(
            "Comma-separated result basenames matching results/<BASE>_predictions.csv "
            "(e.g. winter0.25). Default: every experiment that has a predictions CSV."
        ),
    )
    args = parser.parse_args()
    if args.aggregate_only and args.force_rerun:
        parser.error("--aggregate-only and --force-rerun are mutually exclusive.")

    only_bases: set[str] | None = None
    if args.only:
        only_bases = {b.strip() for b in args.only.split(",") if b.strip()}

    seeds = [0, 1, 2, 3, 4]
    csv_files = sorted(RESULTS.glob("*_predictions.csv"))
    if not csv_files:
        print("No results/*_predictions.csv found.", file=sys.stderr)
        sys.exit(1)

    if only_bases is None:
        summary_path = RESULTS / "multiseed_summary.txt"
    else:
        tag = "_".join(sorted(only_bases))
        summary_path = RESULTS / f"multiseed_summary_only_{tag}.txt"
        print(
            f"Note: --only → writing {summary_path.name} "
            f"(full all-experiment summary is still multiseed_summary.txt).",
            flush=True,
        )

    lines: List[str] = []

    for csv_path in csv_files:
        base = csv_path.name.removesuffix("_predictions.csv")
        if only_bases is not None and base not in only_bases:
            continue
        json_path = RESULTS / f"{base}.json"
        if not json_path.is_file():
            print(f"Skip {csv_path.name}: missing {json_path.name}", file=sys.stderr)
            continue

        with open(json_path) as f:
            bundle = json.load(f)
        cfg = bundle["config"]

        has_q = cfg.get("quantile") is not None
        runs_data: List[Dict[str, Any]] = []
        seeds_used: List[int] = []

        print(f"\n=== {base} (target seeds {seeds}) ===", flush=True)
        for seed in seeds:
            out_name = f"{base}_multiseed_s{seed}"
            out_json = _multiseed_json_path(base, seed)

            if args.aggregate_only:
                if not out_json.is_file():
                    print(f"  seed {seed}  (missing {out_json.name})", flush=True)
                    continue
                print(f"  seed {seed}  (loaded {out_json.name})", flush=True)
                with open(out_json) as f:
                    runs_data.append(json.load(f))
                seeds_used.append(seed)
                continue

            argv = config_to_argv(cfg, seed, out_name)
            if args.force_rerun or not out_json.is_file():
                print(f"  seed {seed}  (running) ...", flush=True)
                run_once(argv)
            else:
                print(f"  seed {seed}  (reuse {out_json.name})", flush=True)
            with open(out_json) as f:
                runs_data.append(json.load(f))
            seeds_used.append(seed)

        if not runs_data:
            print(
                f"Skip {base}: no multiseed JSONs under results/.",
                file=sys.stderr,
            )
            continue

        if args.aggregate_only and len(seeds_used) < len(seeds):
            missing = [s for s in seeds if s not in seeds_used]
            print(
                f"  WARNING: partial aggregate n={len(seeds_used)}; "
                f"missing seeds {missing}",
                flush=True,
            )

        rows = collect_numeric_rows(runs_data, has_q)

        block_start = len(lines)
        lines.append("")
        lines.append("=" * 80)
        lines.append(
            f"Experiment: {base}  (n={len(seeds_used)} seeds: {seeds_used})"
        )
        if args.aggregate_only:
            lines.append("Source: existing results/<name>_multiseed_s*.json (--aggregate-only)")
        lines.append(f"Quantile mode: {has_q}")
        if has_q and cfg.get("quantile") is not None:
            lines.append(f"Target quantile τ: {cfg['quantile']}")
        lines.append("=" * 80)

        lines.extend(_table_seasonal_mape(rows))

        if has_q:
            lines.extend(_table_calibration(rows))
        else:
            lines.append("")
            lines.append(
                "(No coverage or calibration sharpness — OLS mean forecast; "
                "QR seed does not apply.)"
            )

        lines.append("")
        lines.append("Detail — all MAPE variants by scope (overall + seasons)")
        lines.append("-" * 72)
        for sec in SECTIONS:
            lines.append(f"\n--- {sec} ---")
            for mk in MAPE_KEYS:
                xs = rows[sec][mk]
                lines.append(
                    f"  {mk:22s}  mean={fmt(mean(xs))}  STE={fmt(ste(xs))}"
                )
            if has_q:
                lines.append("  (calibration metrics in summary table above)")

        print("\n".join(lines[block_start:]), flush=True)

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote full report to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
