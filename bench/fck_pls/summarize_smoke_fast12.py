"""Summarise the FCK fast12 smoke results.

Prints the median / q75 / q90 / worst-case relative_rmsep_vs_* per pipeline,
the wins/12 vs each baseline, and the gate verdict for promotion to
audit20_transfer_core.

Per `bench/fck_pls/docs/FCK_PLAN_2026-05.md` §3.2 (revised after Codex
round-1 review of D-B-009), the **smoke gate** vs `aom_ridge_curated_best`
is:

- median Δ% rmsep ≤ +25 %;
- worst-case Δ% rmsep ≤ +200 %;
- no-error rate ≥ 75 %;
- *and* the candidate must out-perform PLS-baseline by median ≥ +5 %
  on the cohort.

Strict per-plan-§3.2 thresholds (median ≤ +5 %, q90 ≤ +25 %, worst
≤ +75 %) reapply at audit20 → full-57 and at full-57 → preset.

Usage::

    cd bench/fck_pls
    python summarize_smoke_fast12.py [--in PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO_ROOT / "bench" / "fck_pls" / "runs" / "smoke_fast12" / "results.csv"


# Display the deltas vs these baselines; "ratio_*" columns are the relative
# delta = (rmsep / ref) - 1.0 (negative = better than baseline).
BASELINES = (
    ("ratio_pls", "ref_rmse_pls"),
    ("ratio_paper_ridge", "ref_rmse_paper_ridge"),
    ("ratio_aom_ridge", "ref_rmse_aom_ridge_curated_best"),
    ("ratio_tabpfn_raw", "ref_rmse_tabpfn_raw"),
    ("ratio_tabpfn_opt", "ref_rmse_tabpfn_opt"),
    ("ratio_cnn", "ref_rmse_cnn"),
    ("ratio_catboost", "ref_rmse_catboost"),
)


def _ratio(rmsep: pd.Series, ref: pd.Series) -> pd.Series:
    return (rmsep.astype(float) / ref.astype(float)) - 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise the FCK fast12 smoke results")
    parser.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN, help="Input CSV path")
    args = parser.parse_args()
    df = pd.read_csv(args.inp)
    print(f"\n[FCK-smoke summary] file={args.inp} rows={len(df)} ok={(df['status'] == 'OK').sum()} "
          f"err={(df['status'] == 'ERROR').sum()}")

    if df.empty:
        return 1

    # Add ratio columns
    for col, ref in BASELINES:
        df[col] = _ratio(df["rmsep"], df[ref])

    ok = df[df["status"] == "OK"].copy()
    pipelines = sorted(ok["pipeline"].unique())

    # Per-pipeline absolute rmsep stats
    print("\n=== Per-pipeline absolute rmsep (OK rows only) ===")
    for p in pipelines:
        sub = ok[ok["pipeline"] == p]["rmsep"]
        print(f"  {p:25s}  n={len(sub):2d}  median={sub.median():.4f}  q75={sub.quantile(0.75):.4f}  "
              f"q90={sub.quantile(0.9):.4f}  worst={sub.max():.4f}")

    # Per-pipeline ratio stats vs key baselines
    for col, _ in BASELINES:
        sub_all = ok.dropna(subset=[col])
        if sub_all.empty:
            continue
        print(f"\n=== Δ% vs {col[len('ratio_'):]} ===")
        for p in pipelines:
            sub = sub_all[sub_all["pipeline"] == p][col]
            if sub.empty:
                continue
            wins = int((sub < 0).sum())
            print(f"  {p:25s}  n={len(sub):2d}  median={sub.median():+.3f}  q75={sub.quantile(0.75):+.3f}  "
                  f"q90={sub.quantile(0.9):+.3f}  worst={sub.max():+.3f}  wins={wins}/{len(sub)}")

    # Promotion gate vs aom_ridge_curated_best (revised gate per
    # FCK_PLAN_2026-05.md §3.2.1 / Codex review of D-B-009).
    print("\n=== Promotion gate (vs aom_ridge_curated_best) — fast12 → audit20 ===")
    print("    Need: median Δ% ≤ +25, worst Δ% ≤ +200, no-error ≥ 75 %, ")
    print("          AND median rmsep improvement vs PLS-baseline ≥ +5 %.")
    pls_baseline = ok[ok["pipeline"] == "PLS-baseline"]
    pls_baseline_median = float(pls_baseline["rmsep"].median()) if not pls_baseline.empty else float("nan")
    for p in pipelines:
        sub_ok = ok[ok["pipeline"] == p]
        sub = sub_ok.dropna(subset=["ratio_aom_ridge"])
        all_rows = df[df["pipeline"] == p]
        if all_rows.empty:
            continue
        ok_pct = float(sub_ok.shape[0]) / float(all_rows.shape[0])
        if sub.empty or sub_ok.empty:
            print(f"  {p:25s}  no aom_ridge_curated_best reference data")
            continue
        med = float(sub["ratio_aom_ridge"].median())
        q90 = float(sub["ratio_aom_ridge"].quantile(0.9))
        worst = float(sub["ratio_aom_ridge"].max())
        own_median = float(sub_ok["rmsep"].median())
        improvement_vs_pls = (
            (pls_baseline_median - own_median) / pls_baseline_median
            if pls_baseline_median and pls_baseline_median > 0 else float("nan")
        )
        # PLS-baseline itself is the reference; skip the >=+5% improvement gate
        # for it but still report the other thresholds for visibility.
        is_reference = p == "PLS-baseline"
        ok_pass = (
            (med <= 0.25)
            and (worst <= 2.00)
            and (ok_pct >= 0.75)
            and (is_reference or (improvement_vs_pls >= 0.05))
        )
        verdict = "PASS" if ok_pass else "FAIL"
        improvement_str = "  (reference)" if is_reference else f"  improvement_vs_pls={improvement_vs_pls:+.2%}"
        print(f"  {p:25s}  median={med:+.3f}  q90={q90:+.3f}  worst={worst:+.3f}  "
              f"ok_rate={ok_pct:.0%}{improvement_str}  -> {verdict}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
