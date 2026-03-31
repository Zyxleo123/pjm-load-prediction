#!/usr/bin/env python3
"""
Figures for progress talks: QR / CQR 95% quantile forecasts vs actual load.

Loads predictions from results/<run>_predictions.csv, actuals + weather from
data/full/<station>.csv (same pipeline as run_experiment).

Failure (conservative shortfall): actual load > predicted bound.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from run_experiment import (
    SEASONS,
    fit_predict_ols,
    load_or_fetch_dataset,
    train_test_split,
)

# ---------------------------------------------------------------------------
# Defaults (match saved runs under results/)
# ---------------------------------------------------------------------------

DEFAULT_RUNS = {
    "qr_lr5.0": "Plain QR τ=0.95",
    "cqr_lr5.0_og": "CQR (full winter offset)",
    "winter0.25": "CQR (winter offset × 0.25)",
}

DEFAULT_STATIONS = ["Center_City"]
DEFAULT_QR_JSON = "qr_lr5.0.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def feature_selection_from_results_json(path: Path) -> Dict[str, bool]:
    cfg = load_json(path)["config"]["features"]
    return {k: bool(v) for k, v in cfg.items()}


def load_predictions_csv(path: Path) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze(axis=1)
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.name = "load_mw_pred"
    if s.index.duplicated().any():
        s = s[~s.index.duplicated(keep="first")]
    return s


def season_for_index(dt_index: pd.DatetimeIndex) -> pd.Series:
    out = pd.Series(index=dt_index, dtype=object)
    for name, months in SEASONS.items():
        out.loc[dt_index.month.isin(months)] = name
    return out


def failure_mask(actual: pd.Series, pred: pd.Series) -> pd.Series:
    aligned = pd.DataFrame({"y": actual, "q": pred}).dropna()
    return aligned["y"] > aligned["q"]


def coverage_fraction(actual: pd.Series, pred: pd.Series) -> float:
    m = failure_mask(actual, pred)
    return float(1.0 - m.mean())


def _dedupe_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.duplicated().any():
        return df[~df.index.duplicated(keep="first")]
    return df


def load_test_frame(
    stations: List[str],
    results_dir: Path,
    qr_json_name: str,
) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    log = logging.getLogger("plot_progress_presentation")
    if not log.handlers:
        logging.basicConfig(level=logging.WARNING)
    df = _dedupe_index(load_or_fetch_dataset(log, stations))
    _, df_test = train_test_split(df, log)
    jpath = results_dir / qr_json_name
    features = feature_selection_from_results_json(jpath)
    return df_test, features


def compute_ols_mean_forecast(
    stations: List[str],
    features: Dict[str, bool],
    results_dir: Path,
) -> pd.Series:
    log = logging.getLogger("plot_progress_presentation.ols")
    if not log.handlers:
        log.addHandler(logging.NullHandler())
    df = _dedupe_index(load_or_fetch_dataset(log, stations))
    df_train, df_test = train_test_split(df, log)
    return fit_predict_ols(df_train, df_test, features, log)


def align_actual_pred(
    df_test: pd.DataFrame,
    pred: pd.Series,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "load_mw": df_test["load_mw"],
            "pred": pred.reindex(df_test.index),
        }
    )
    # temperature column for Center_City cache
    temp_col = None
    for c in df_test.columns:
        if c.lower().endswith("_temp") or c == "temp":
            temp_col = c
            break
    if temp_col is not None:
        df["temp"] = df_test[temp_col]
    return df.dropna(subset=["load_mw", "pred"])


def plot_ece_coverage_bars(
    results_dir: Path,
    run_labels: Mapping[str, str],
    ax: Optional[plt.Axes] = None,
    title: str = "ECE = empirical coverage − τ (closer to 0 is better)",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    runs = [r for r in run_labels if (results_dir / f"{r}.json").exists()]
    seasons_list = ["overall"] + list(SEASONS.keys())
    x = np.arange(len(seasons_list))
    width = min(0.22, 0.8 / max(len(runs), 1))
    for i, run in enumerate(runs):
        data = load_json(results_dir / f"{run}.json")
        qm = data.get("quantile_metrics") or {}
        eces = []
        for key in seasons_list:
            block = qm.get(key, {})
            eces.append(block.get("ece", np.nan))
        offset = (i - (len(runs) - 1) / 2) * width
        ax.bar(
            x + offset,
            eces,
            width,
            label=run_labels.get(run, run),
            alpha=0.85,
        )
    ax.axhline(0.0, color="gray", ls="--", lw=1, label="perfect calibration")
    ax.set_xticks(x)
    ax.set_xticklabels(["Year"] + list(SEASONS.keys()))
    ax.set_ylabel("ECE (coverage − τ)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_coverage_subplot(
    results_dir: Path,
    run_labels: Mapping[str, str],
    promised_q: float,
    ax: plt.Axes,
    title: str = "Empirical coverage vs nominal τ (one-sided upper bound)",
) -> None:
    runs = [r for r in run_labels if (results_dir / f"{r}.json").exists()]
    seasons_list = ["overall"] + list(SEASONS.keys())
    x = np.arange(len(seasons_list))
    width = min(0.22, 0.8 / max(len(runs), 1))
    for i, run in enumerate(runs):
        data = load_json(results_dir / f"{run}.json")
        qm = data.get("quantile_metrics") or {}
        covs = []
        for key in seasons_list:
            covs.append((qm.get(key) or {}).get("coverage", np.nan))
        offset = (i - (len(runs) - 1) / 2) * width
        ax.bar(x + offset, covs, width, label=run_labels.get(run, run), alpha=0.85)
    ax.axhline(promised_q, color="crimson", ls="--", lw=1.2, label="target coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(["Year"] + list(SEASONS.keys()))
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.set_ylim(0.88, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_coverage_figure_only(
    results_dir: Path,
    run_labels: Mapping[str, str],
    promised_q: float,
    figsize: Tuple[float, float] = (10, 4.5),
) -> plt.Figure:
    """Single-panel coverage chart (formerly the bottom panel of ece_and_coverage)."""
    fig, ax = plt.subplots(figsize=figsize)
    plot_coverage_subplot(results_dir, run_labels, promised_q, ax)
    fig.tight_layout()
    return fig


def _hourly_contiguous(idx: pd.DatetimeIndex) -> bool:
    if len(idx) < 2:
        return True
    secs = idx.to_series().diff().dt.total_seconds().iloc[1:].astype(float)
    return bool((np.abs(secs.values - 3600.0) < 120.0).all())


def find_worst_winter_week(
    df: pd.DataFrame,
    *,
    min_hours: int = 168,
) -> Tuple[pd.DataFrame, float]:
    """
    Pick a contiguous hourly window of ``min_hours`` in Winter (Dec/Jan/Feb)
    with the largest mean positive gap ``mean(max(0, actual - pred))``.
    Falls back to the full test year if winter has no qualifying window.
    """
    df = df.sort_index()
    winter = df[df.index.month.isin((12, 1, 2))]
    candidates: List[pd.DataFrame] = (
        [winter] if len(winter) >= min_hours else [df]
    )
    best_score = -1.0
    best_chunk: Optional[pd.DataFrame] = None
    for pool in candidates:
        for i in range(0, len(pool) - min_hours + 1):
            chunk = pool.iloc[i : i + min_hours]
            if not _hourly_contiguous(chunk.index):
                continue
            gap = np.maximum(
                chunk["load_mw"].to_numpy(dtype=float)
                - chunk["pred"].to_numpy(dtype=float),
                0.0,
            )
            score = float(gap.mean())
            if score > best_score:
                best_score = score
                best_chunk = chunk
    if best_chunk is None:
        raise ValueError("Could not find a contiguous hourly window for worst-week plot.")
    return best_chunk, best_score


def slice_season_week_near_nominal_coverage(
    df: pd.DataFrame,
    *,
    min_hours: int = 168,
    target: float = 0.95,
    margin: float = 0.005,
    months: Tuple[int, ...],
    season_label: str = "season",
) -> Tuple[pd.DataFrame, float, bool]:
    """
    Contiguous ``min_hours`` window restricted to ``months`` whose empirical
    coverage ``mean(actual <= pred)`` is within ``margin`` of ``target`` if
    any exists; otherwise the window with coverage closest to ``target``.

    Returns
    -------
    chunk, coverage, in_band
        ``in_band`` is True iff ``|coverage - target| <= margin``.
    """
    df = df.sort_index()
    sub = df[df.index.month.isin(months)]
    best_in_band: Optional[Tuple[pd.DataFrame, float, float]] = None
    best_dist = float("inf")
    best_fallback: Optional[Tuple[pd.DataFrame, float, float]] = None
    best_fb_dist = float("inf")

    for i in range(0, len(sub) - min_hours + 1):
        chunk = sub.iloc[i : i + min_hours]
        if not _hourly_contiguous(chunk.index):
            continue
        a = chunk["load_mw"].to_numpy(dtype=float)
        p = chunk["pred"].to_numpy(dtype=float)
        cov = float(np.mean(a <= p))
        dist = abs(cov - target)
        if dist < best_fb_dist:
            best_fb_dist = dist
            best_fallback = (chunk, cov, dist)
        if dist <= margin and dist < best_dist:
            best_dist = dist
            best_in_band = (chunk, cov, dist)

    if best_in_band is not None:
        ch, cov, _ = best_in_band
        return ch, cov, True
    if best_fallback is not None:
        ch, cov, _ = best_fallback
        return ch, cov, False
    raise ValueError(
        f"No contiguous {min_hours}h window found in {season_label} for pred vs actual plot."
    )


def _compact_week_range_label(s0: pd.Timestamp, s1: pd.Timestamp) -> str:
    if s0.year == s1.year and s0.month == s1.month:
        return f"{s0.strftime('%b')} {s0.day}–{s1.day}, {s0.year}"
    if s0.year == s1.year:
        return f"{s0.strftime('%b %d')} – {s1.strftime('%b %d, %Y')}"
    return f"{s0.strftime('%b %d, %Y')} – {s1.strftime('%b %d, %Y')}"


def plot_qr_pred_vs_actual_scatter(
    df: pd.DataFrame,
    *,
    title: str,
    note_lines: Optional[List[str]] = None,
    y_lo: float = 0.0,
    y_hi: float = 5000.0,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Scatter: x = actual load, y = predicted upper bound.

    *x*-limits follow actuals (min–max with padding, floored at 0).
    *y*-limits default to 0–5000 MW so vertical sharpness is readable.
    The ``y = x`` segment is clipped to both visible ranges.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))
    a = df["load_mw"].to_numpy(dtype=float)
    p = df["pred"].to_numpy(dtype=float)
    ax.scatter(a, p, s=16, alpha=0.55, c="steelblue", edgecolors="none")
    amin = float(np.nanmin(a))
    amax = float(np.nanmax(a))
    pad = max((amax - amin) * 0.03, 1.0)
    x_lo = max(0.0, amin - pad)
    x_hi = amax + pad
    diag_lo = max(x_lo, y_lo)
    diag_hi = min(x_hi, y_hi)
    if diag_lo < diag_hi:
        ax.plot(
            [diag_lo, diag_hi],
            [diag_lo, diag_hi],
            "k--",
            lw=1.2,
            label="y = x",
        )
    cov = float(np.mean(a <= p))
    lines = [f"Share actual ≤ pred: {cov:.1%}  (nominal 95%)"]
    if note_lines:
        lines.extend(note_lines)
    ax.text(
        0.04,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Actual load (MW)")
    ax.set_ylabel("Predicted 95% upper bound (MW)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_actual_vs_pred_timeseries(
    df: pd.DataFrame,
    *,
    title: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Hourly actual load and predicted 95% upper bound (line plot only)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        df.index,
        df["load_mw"],
        color="black",
        lw=1.35,
        label="Actual load",
    )
    ax.plot(
        df.index,
        df["pred"],
        color="steelblue",
        lw=1.35,
        label="Pred (95% upper bound)",
    )
    ax.set_ylabel("MW")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def mean_upper_sharpness(sub: pd.DataFrame) -> float:
    """Mean (q_pred - actual) / actual; matches ``compute_quantile_interval_metrics``."""
    a = sub["load_mw"].to_numpy(dtype=float)
    q = sub["pred"].to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(q) & (a != 0.0)
    if not np.any(ok):
        return float("nan")
    return float(np.mean((q[ok] - a[ok]) / a[ok]))


def plot_holiday_vs_non_sharpness(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Sharpness on holidays vs non-holidays (lower = tighter upper bound on average)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    hol = us_federal_holidays_2023().normalize()
    idx = df.index.normalize()
    is_hol = pd.Series(idx.isin(hol), index=df.index)
    groups = [
        ("US federal holiday hour", is_hol),
        ("Non-holiday hour", ~is_hol),
    ]
    vals: List[float] = []
    names: List[str] = []
    for name, m in groups:
        sub = df.loc[m]
        if len(sub) == 0:
            continue
        vals.append(mean_upper_sharpness(sub))
        names.append(name)
    ax.bar(names, vals, color=["darkslateblue", "cadetblue"])
    ax.set_ylabel("Sharpness: mean (pred − actual) / actual")
    ax.set_title("Upper-bound sharpness: holidays vs rest (2023 test)")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
    return ax


def plot_timeseries_actual_pred_fill(
    df: pd.DataFrame,
    mean_pred: Optional[pd.Series] = None,
    resample: Optional[str] = "7D",
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Red fill where actual > pred (under-estimated bound); green otherwise.
    If resample is set, aggregate by mean for readability.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    work = df.copy()
    if resample:
        work = work.resample(resample).mean()
    t = work.index
    y = work["load_mw"].values
    p = work["pred"].values
    ax.plot(t, y, color="black", lw=1.2, label="Actual load")
    ax.plot(t, p, color="steelblue", lw=1.4, label="Pred (τ quantile)")
    if mean_pred is not None:
        mp = mean_pred.reindex(df.index)
        if resample:
            mp = mp.resample(resample).mean()
        ax.plot(
            mp.index,
            mp.values,
            color="darkorange",
            lw=1.0,
            ls="--",
            label="OLS mean forecast",
        )
    ax.fill_between(
        t,
        y,
        p,
        where=(y > p),
        interpolate=True,
        color="red",
        alpha=0.25,
        label="Actual > pred (shortfall)",
    )
    ax.fill_between(
        t,
        y,
        p,
        where=(y <= p),
        interpolate=True,
        color="green",
        alpha=0.2,
        label="Actual ≤ pred",
    )
    ax.set_title(title)
    ax.set_ylabel("MW (mean over window)" if resample else "MW")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    return ax


def plot_seasonal_failure_rates(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "Share of hours with actual > pred (by season)",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    seas = season_for_index(df.index)
    rates = []
    labels = []
    for name in SEASONS:
        mask = seas == name
        sub = df.loc[mask]
        if len(sub) == 0:
            continue
        fail = (sub["load_mw"] > sub["pred"]).mean()
        rates.append(fail)
        labels.append(name)
    tgt = 1.0 - 0.95
    x = np.arange(len(labels))
    ax.bar(x, rates, color="slategray", edgecolor="black", alpha=0.85)
    ax.axhline(tgt, color="crimson", ls="--", label="exact τ=0.95 slack (5%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of hours")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return ax


def us_federal_holidays_2023() -> pd.DatetimeIndex:
    cal = USFederalHolidayCalendar()
    return cal.holidays(start="2023-01-01", end="2023-12-31")


def assign_temperature_bins(df: pd.DataFrame, n_bins: int = 8) -> pd.DataFrame:
    """Add ``temp_bin`` column (quantile bins, or equal-width fallback)."""
    if "temp" not in df.columns:
        return pd.DataFrame()
    df2 = df.dropna(subset=["temp"]).copy()
    if df2.empty:
        return df2
    try:
        df2["temp_bin"] = pd.qcut(df2["temp"], q=n_bins, duplicates="drop")
    except ValueError:
        df2["temp_bin"] = pd.cut(df2["temp"], bins=n_bins)
    return df2


def _temp_bin_ticklabels(interval_index: pd.Index) -> List[str]:
    return [f"{iv.left:.0f}–{iv.right:.0f}°" for iv in interval_index]


def plot_failure_by_temp_bins(
    df: pd.DataFrame,
    n_bins: int = 8,
    ax: Optional[plt.Axes] = None,
    title: str = "QR: fraction actual > pred by temperature",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    df2 = assign_temperature_bins(df, n_bins)
    if df2.empty or "temp_bin" not in df2.columns:
        ax.text(0.5, 0.5, "No temperature data", ha="center", va="center", transform=ax.transAxes)
        return ax

    def _fail_frac(s: pd.DataFrame) -> float:
        return float((s["load_mw"] > s["pred"]).mean())

    rate = df2.groupby("temp_bin", observed=True).apply(_fail_frac)
    centers = _temp_bin_ticklabels(rate.index)
    ax.bar(range(len(rate)), rate.values, tick_label=centers, color="teal", alpha=0.8)
    ax.set_xlabel("Temperature bin (°C)")
    ax.set_ylabel("Fraction actual > pred")
    ax.set_title(title)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_sharpness_by_temp_bins(
    df: pd.DataFrame,
    n_bins: int = 8,
    ax: Optional[plt.Axes] = None,
    title: str = "QR: sharpness by temperature",
) -> plt.Axes:
    """
    Mean (pred - actual) / actual per temperature bin (same bins as failure chart).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    df2 = assign_temperature_bins(df, n_bins)
    if df2.empty or "temp_bin" not in df2.columns:
        ax.text(0.5, 0.5, "No temperature data", ha="center", va="center", transform=ax.transAxes)
        return ax

    sharp = df2.groupby("temp_bin", observed=True).apply(mean_upper_sharpness)
    centers = _temp_bin_ticklabels(sharp.index)
    ax.bar(range(len(sharp)), sharp.values, tick_label=centers, color="darkorchid", alpha=0.8)
    ax.set_xlabel("Temperature bin (°C)")
    ax.set_ylabel("Sharpness: mean (pred - actual) / actual")
    ax.set_title(title)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.0, color="gray", ls=":", lw=1)
    return ax


def plot_run_comparison_timeseries(
    dfs: Mapping[str, pd.DataFrame],
    resample: str = "7D",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    for label, d in dfs.items():
        w = d.resample(resample).mean()
        ax.plot(w.index, w["pred"], lw=1.3, label=f"{label} pred")
    base = next(iter(dfs.values()))
    w0 = base.resample(resample).mean()
    ax.plot(w0.index, w0["load_mw"], color="black", lw=1.5, label="Actual")
    ax.set_ylabel("MW")
    ax.set_title(f"Predicted quantile bounds ({resample} mean)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return ax


def plot_cqr_runs_same_week(
    week_index: pd.DatetimeIndex,
    dfs: Mapping[str, pd.DataFrame],
    *,
    title: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Hourly actual + predicted bounds for several runs on the same 7-day index."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.5))
    first = True
    for label, d in dfs.items():
        sub = d.reindex(week_index)
        if first:
            ax.plot(
                week_index,
                sub["load_mw"].values,
                color="black",
                lw=1.8,
                label="Actual",
            )
            first = False
        ax.plot(week_index, sub["pred"].values, lw=1.3, label=label)
    ax.set_ylabel("MW")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    return ax


def build_all_figures(
    out_dir: Path,
    results_dir: Path,
    stations: List[str],
    qr_json_name: str,
    run_labels: Mapping[str, str],
    promised_q: float = 0.95,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_test, features = load_test_frame(stations, results_dir, qr_json_name)

    mean_ols = compute_ols_mean_forecast(stations, features, results_dir)

    def aligned_for(run: str) -> pd.DataFrame:
        pred = load_predictions_csv(results_dir / f"{run}_predictions.csv")
        return align_actual_pred(df_test, pred)

    qr_df = aligned_for("qr_lr5.0")

    qr_week, week_score = find_worst_winter_week(qr_df)
    week_index = pd.DatetimeIndex(qr_week.index)
    t0, t1 = week_index[0], week_index[-1]

    # 1) Coverage only (was bottom panel of ece_and_coverage)
    fig = plot_coverage_figure_only(results_dir, run_labels, promised_q)
    fig.savefig(out_dir / "ece_and_coverage.png", dpi=150)
    plt.close(fig)

    # 2) QR: worst winter week (hourly, 7 days) — max mean under-prediction gap
    fig, ax = plt.subplots(figsize=(12, 4.5))
    plot_timeseries_actual_pred_fill(
        qr_week,
        mean_pred=mean_ols.reindex(qr_week.index),
        resample=None,
        title=(
            f"QR tau=0.95 — worst winter week by mean shortfall "
            f"({t0:%Y-%m-%d %H:%M} -> {t1:%Y-%m-%d %H:%M}, "
            f"mean max(0, actual-pred)={week_score:.1f} MW)"
        ),
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "qr_worst_winter_week.png", dpi=150)
    plt.close(fig)

    # 3) QR pred vs actual — spring week with coverage ~95% (±0.5 pp if possible)
    spring_months = tuple(SEASONS["Spring"])
    scatter_wk, scatter_cov, scatter_in_band = slice_season_week_near_nominal_coverage(
        qr_df,
        target=promised_q,
        margin=0.005,
        months=spring_months,
        season_label="spring (Mar–May)",
    )
    s0, s1 = scatter_wk.index[0], scatter_wk.index[-1]
    week_lbl = _compact_week_range_label(s0, s1)
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    plot_qr_pred_vs_actual_scatter(
        scatter_wk,
        title="QR 95% bound vs actual (spring week)",
        note_lines=[week_lbl],
        y_lo=0.0,
        y_hi=5000.0,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "qr_pred_vs_actual_spring_week.png", dpi=150)
    plt.close(fig)

    fig_ts, ax_ts = plt.subplots(figsize=(12, 4))
    plot_actual_vs_pred_timeseries(
        scatter_wk,
        title=f"QR: actual vs predicted bound (same week: {week_lbl})",
        ax=ax_ts,
    )
    fig_ts.autofmt_xdate()
    fig_ts.tight_layout()
    fig_ts.savefig(out_dir / "qr_spring_week_timeseries.png", dpi=150)
    plt.close(fig_ts)

    # 4) Seasonal failure histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_seasonal_failure_rates(qr_df, ax=ax, title="QR: seasonal under-prediction rate")
    fig.tight_layout()
    fig.savefig(out_dir / "qr_seasonal_failure_rate.png", dpi=150)
    plt.close(fig)

    # 5) CQR vs winter0.25 on the same worst winter week (hourly)
    cqr_dfs: Dict[str, pd.DataFrame] = {}
    for key in ("cqr_lr5.0_og", "winter0.25"):
        p = results_dir / f"{key}_predictions.csv"
        if p.exists():
            cqr_dfs[run_labels.get(key, key)] = aligned_for(key)
    if len(cqr_dfs) >= 1:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        plot_cqr_runs_same_week(
            week_index,
            cqr_dfs,
            title=(
                f"CQR predicted bounds — same week as QR worst case "
                f"({t0:%Y-%m-%d} → {t1:%Y-%m-%d})"
            ),
            ax=ax,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "cqr_pred_comparison_7d.png", dpi=150)
        plt.close(fig)

    # 6) Holiday sharpness (QR), separate file
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_holiday_vs_non_sharpness(qr_df, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "qr_holiday_sharpness.png", dpi=150)
    plt.close(fig)

    # 7) Failure rate by temperature (QR)
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_failure_by_temp_bins(qr_df, n_bins=8, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "qr_failure_by_temperature.png", dpi=150)
    plt.close(fig)

    # 8) Sharpness by temperature (QR)
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_sharpness_by_temp_bins(qr_df, n_bins=8, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "qr_sharpness_by_temperature.png", dpi=150)
    plt.close(fig)

    print(f"Wrote figures under {out_dir.resolve()}")
    print(
        f"Worst winter week (QR, mean shortfall {week_score:.2f} MW): "
        f"{t0} → {t1}"
    )
    print(
        f"Spring pred-vs-actual week: {s0} → {s1} "
        f"(empirical coverage {scatter_cov:.2%}, in ±0.5pp band: {scatter_in_band})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/figures_progress",
        help="Directory for PNG outputs",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory with <run>.json and <run>_predictions.csv",
    )
    parser.add_argument(
        "--qr-json",
        type=str,
        default=DEFAULT_QR_JSON,
        help="JSON file (under results-dir) to read feature config for OLS mean line",
    )
    args = parser.parse_args()
    root = _repo_root()
    os.chdir(root)
    build_all_figures(
        out_dir=root / args.out_dir,
        results_dir=root / args.results_dir,
        stations=list(DEFAULT_STATIONS),
        qr_json_name=args.qr_json,
        run_labels=dict(DEFAULT_RUNS),
    )


if __name__ == "__main__":
    main()
