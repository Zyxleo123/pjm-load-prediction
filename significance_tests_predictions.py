#!/usr/bin/env python3
"""
Statistical tests on saved prediction CSVs (PECO / zone PE load, 2023 test).
Uses scipy and statsmodels as requested.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.weightstats import DescrStatsW

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
# Authoritative hourly actuals (load_mw is identical across single-station caches)
ACTUALS_PATH = ROOT / "data" / "full" / "West_Chester.csv"

WINTER_MONTHS = {12, 1, 2}


def load_predictions_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.columns[0] in ("timestamp", "Timestamp") or (
        df.columns[0] != "load_mw_pred" and "timestamp" in df.columns
    ):
        df = df.set_index(pd.to_datetime(df["timestamp"]))
    else:
        df = df.set_index(pd.to_datetime(df.iloc[:, 0]))
    s = df["load_mw_pred"].copy()
    s.index.name = "timestamp"
    s = s[~s.index.duplicated(keep="first")]
    return s.sort_index()


def load_actuals_2023() -> pd.Series:
    if not ACTUALS_PATH.exists():
        alt = ROOT / "data" / "full" / "PHL.csv"
        if not alt.exists():
            raise FileNotFoundError(
                f"Need cached dataset at {ACTUALS_PATH} or {alt} for load_mw actuals."
            )
        p = alt
    else:
        p = ACTUALS_PATH
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    y = df["load_mw"].copy()
    y.index.name = "timestamp"
    y = y[~y.index.duplicated(keep="first")].sort_index()
    return y.loc["2023"]


def test1_mape_paired_t(
    baseline_path: Path,
    improved_path: Path,
    baseline_label: str,
    improved_label: str,
) -> None:
    y = load_actuals_2023()
    p_base = load_predictions_csv(baseline_path).reindex(y.index)
    p_imp = load_predictions_csv(improved_path).reindex(y.index)
    m = y.notna() & p_base.notna() & p_imp.notna()
    y = y[m]
    p_base = p_base[m]
    p_imp = p_imp[m]

    ape_base = (p_base - y).abs() / y
    ape_imp = (p_imp - y).abs() / y

    daily_base = ape_base.groupby(ape_base.index.normalize()).mean()
    daily_imp = ape_imp.groupby(ape_imp.index.normalize()).mean()
    common_days = daily_base.index.intersection(daily_imp.index)
    daily_base = daily_base.reindex(common_days).dropna()
    daily_imp = daily_imp.reindex(common_days).dropna()
    # align lengths (should match)
    idx = daily_base.index.intersection(daily_imp.index)
    a = daily_base.loc[idx].values * 100.0  # report in %
    b = daily_imp.loc[idx].values * 100.0
    n = len(a)
    diff = a - b  # baseline - improved; positive => baseline worse

    tt = stats.ttest_rel(a, b, alternative="greater", nan_policy="omit")
    ci_low, ci_high = DescrStatsW(diff).tconfint_mean(alpha=0.05)

    print("\n" + "=" * 72)
    print("TEST 1 — Daily MAPE (paired t-test, one-sided)")
    print("=" * 72)
    print(f"Baseline (H0 reference): {baseline_label}")
    print(f"  File: {baseline_path.name}")
    print(f"Improved (expected lower error): {improved_label}")
    print(f"  File: {improved_path.name}")
    print(f"Hourly APE = |y_pred - y_true| / y_true; daily MAPE = mean hourly APE that day.")
    print(f"Paired days (2023, full overlap): n = {n}")
    print(f"Mean daily MAPE (%): baseline = {a.mean():.4f}, improved = {b.mean():.4f}")
    print(f"Mean paired difference (baseline − improved) %: {diff.mean():.4f}")
    print(f"95% CI for mean difference (baseline − improved) %: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Paired t-statistic (H1: baseline mean > improved mean): {tt.statistic:.4f}")
    print(f"p-value (one-sided): {tt.pvalue:.6g}")


def _mcnemar_table(qr_bin: np.ndarray, cqr_bin: np.ndarray) -> np.ndarray:
    """statsmodels: [0,0] both 0, [0,1] first=0 second=1, [1,0] first=1 second=0, [1,1] both 1."""
    qr_bin = qr_bin.astype(int)
    cqr_bin = cqr_bin.astype(int)
    n00 = int(np.sum((qr_bin == 0) & (cqr_bin == 0)))
    n01 = int(np.sum((qr_bin == 0) & (cqr_bin == 1)))
    n10 = int(np.sum((qr_bin == 1) & (cqr_bin == 0)))
    n11 = int(np.sum((qr_bin == 1) & (cqr_bin == 1)))
    return np.array([[n00, n01], [n10, n11]])


def test2_winter_coverage(
    qr_path: Path,
    cqr_path: Path,
    qr_label: str,
    cqr_label: str,
) -> None:
    y = load_actuals_2023()
    p_qr = load_predictions_csv(qr_path).reindex(y.index)
    p_cqr = load_predictions_csv(cqr_path).reindex(y.index)
    m = y.notna() & p_qr.notna() & p_cqr.notna()
    y = y[m]
    p_qr = p_qr[m]
    p_cqr = p_cqr[m]

    win = y.index.month.isin(WINTER_MONTHS)
    y_w = y[win]
    u_qr = p_qr[win]
    u_cqr = p_cqr[win]

    cov_qr = (y_w.values <= u_qr.values).astype(int)
    cov_cqr = (y_w.values <= u_cqr.values).astype(int)
    n_h = len(cov_qr)

    table_h = _mcnemar_table(cov_qr, cov_cqr)
    res_h = mcnemar(table_h, exact=False, correction=True)

    bt_qr = stats.binomtest(int(cov_qr.sum()), n_h, p=0.95, alternative="two-sided")
    bt_cqr = stats.binomtest(int(cov_cqr.sum()), n_h, p=0.95, alternative="two-sided")

    # Daily aggregation: same winter subset, group by calendar date
    df_h = pd.DataFrame(
        {"y": y_w.values, "cov_qr": cov_qr, "cov_cqr": cov_cqr},
        index=y_w.index,
    )
    g = df_h.groupby(df_h.index.normalize())
    daily_frac_qr = g["cov_qr"].mean()
    daily_frac_cqr = g["cov_cqr"].mean()
    day_idx = daily_frac_qr.index.intersection(daily_frac_cqr.index)
    daily_frac_qr = daily_frac_qr.reindex(day_idx)
    daily_frac_cqr = daily_frac_cqr.reindex(day_idx)

    # Binary day outcomes: (1) strict all hours covered; (2) ≥95% hours covered that day
    strict_qr = (daily_frac_qr >= 1.0 - 1e-12).astype(int).values
    strict_cqr = (daily_frac_cqr >= 1.0 - 1e-12).astype(int).values

    thr_qr = (daily_frac_qr.values >= 0.95).astype(int)
    thr_cqr = (daily_frac_cqr.values >= 0.95).astype(int)

    table_d_strict = _mcnemar_table(strict_qr, strict_cqr)
    res_d_strict = mcnemar(table_d_strict, exact=False, correction=True)
    table_d_thr = _mcnemar_table(thr_qr, thr_cqr)
    res_d_thr = mcnemar(table_d_thr, exact=False, correction=True)

    print("\n" + "=" * 72)
    print("TEST 2 — Winter coverage (months 12, 1, 2); upper-bound intervals")
    print("=" * 72)
    print(f"QR (first measurement in McNemar table): {qr_label}")
    print(f"  File: {qr_path.name}")
    print(f"CQR (second measurement): {cqr_label}")
    print(f"  File: {cqr_path.name}")
    print(f"Winter hours used: n = {n_h}")
    print(f"Observed marginal coverage: QR = {cov_qr.mean():.4f}, CQR = {cov_cqr.mean():.4f}")
    print("\n--- McNemar (hour level), continuity correction ---")
    print("Contingency [rows=QR covered?, cols=CQR covered?] → statsmodels layout:")
    print("         CQR=0    CQR=1")
    print(f"QR=0   {table_h[0,0]:6d}  {table_h[0,1]:6d}")
    print(f"QR=1   {table_h[1,0]:6d}  {table_h[1,1]:6d}")
    print(f"χ² McNemar statistic: {res_h.statistic:.4f}")
    print(f"p-value: {res_h.pvalue:.6g}")

    print("\n--- Binomial test vs target p = 0.95 (two-sided) ---")
    print(f"QR:  observed coverage = {cov_qr.mean():.6f},  p-value = {bt_qr.pvalue:.6g}")
    print(f"CQR: observed coverage = {cov_cqr.mean():.6f},  p-value = {bt_cqr.pvalue:.6g}")

    print("\n--- Daily coverage (mean fraction of hours covered per calendar day) ---")
    print(f"Days in winter window: {len(day_idx)}")
    print(
        f"Mean daily coverage fraction: QR = {daily_frac_qr.mean():.4f}, "
        f"CQR = {daily_frac_cqr.mean():.4f}"
    )

    print("\nMcNemar at day level — binary = all hours in day covered (fraction = 1)")
    print("         CQR=0    CQR=1")
    print(f"QR=0   {table_d_strict[0,0]:6d}  {table_d_strict[0,1]:6d}")
    print(f"QR=1   {table_d_strict[1,0]:6d}  {table_d_strict[1,1]:6d}")
    print(f"χ² statistic: {res_d_strict.statistic:.4f}, p-value: {res_d_strict.pvalue:.6g}")

    print("\nMcNemar at day level — binary = daily coverage fraction ≥ 0.95")
    print("         CQR=0    CQR=1")
    print(f"QR=0   {table_d_thr[0,0]:6d}  {table_d_thr[0,1]:6d}")
    print(f"QR=1   {table_d_thr[1,0]:6d}  {table_d_thr[1,1]:6d}")
    print(f"χ² statistic: {res_d_thr.statistic:.4f}, p-value: {res_d_thr.pvalue:.6g}")


def main() -> None:
    os.chdir(ROOT)
    test1_mape_paired_t(
        RESULTS / "single-station_predictions.csv",
        RESULTS / "West_Chester_predictions.csv",
        baseline_label="single-station (PHL)",
        improved_label="West_Chester",
    )
    test2_winter_coverage(
        RESULTS / "qr_lr5.0_predictions.csv",
        RESULTS / "winter0.25_predictions.csv",
        qr_label="qr_lr5.0 (QR upper bound)",
        cqr_label="winter0.25 (CQR upper bound)",
    )


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
