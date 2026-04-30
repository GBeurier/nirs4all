"""Statistical analysis for multi-kernel benchmark results.

Adds:

- Wilcoxon signed-rank test on per-dataset **log** RMSEP ratios
  (one variant vs PLS, etc.).
- Sign / binomial test for wins / losses / ties.
- Holm correction for planned multiple comparisons.
- Bootstrap confidence intervals on the median ratio.
- Branch-lift analysis WITHIN solver family.
- Failure / non-convergence summary.

Usage:

```bash
.venv/bin/python bench/AOM_v0/Multi-kernel/benchmarks/stats_multikernel.py \
  bench/AOM_v0/Multi-kernel/benchmark_runs/extended12/results.csv \
  --out-dir bench/AOM_v0/Multi-kernel/benchmark_runs/extended12
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    return df


def _wilcoxon_one_sided(diffs: np.ndarray) -> float:
    """Wilcoxon signed-rank p-value testing 'median(diff) < 0'."""
    from scipy.stats import wilcoxon
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 5:
        return float("nan")
    res = wilcoxon(diffs, alternative="less")
    return float(res.pvalue)


def _sign_test(diffs: np.ndarray) -> tuple[int, int, int, float]:
    """Sign test: count wins (diff < 0), losses (diff > 0), ties.

    Returns ``(wins, losses, ties, p_one_sided_against_no_difference)``.
    """
    from scipy.stats import binomtest
    diffs = diffs[~np.isnan(diffs)]
    wins = int((diffs < 0).sum())
    losses = int((diffs > 0).sum())
    ties = int((diffs == 0).sum())
    n = wins + losses
    if n == 0:
        return wins, losses, ties, float("nan")
    res = binomtest(wins, n=n, p=0.5, alternative="greater")
    return wins, losses, ties, float(res.pvalue)


def _bootstrap_median_ratio_ci(
    rmsep: np.ndarray, ref: np.ndarray,
    n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI on the median(rmsep / ref).

    Returns (median, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    ratios = rmsep / ref
    ratios = ratios[~np.isnan(ratios) & np.isfinite(ratios)]
    if len(ratios) < 5:
        return float(np.median(ratios)) if len(ratios) else float("nan"), \
               float("nan"), float("nan")
    medians = []
    for _ in range(n_boot):
        sample = rng.choice(ratios, size=len(ratios), replace=True)
        medians.append(float(np.median(sample)))
    medians = np.array(medians)
    return float(np.median(ratios)), float(np.quantile(medians, alpha / 2)), \
           float(np.quantile(medians, 1 - alpha / 2))


def _holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni correction for a vector of p-values."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    cum_max = 0.0
    for rank_i, idx in enumerate(order):
        adj = pvalues[idx] * (n - rank_i)
        cum_max = max(cum_max, adj)
        adjusted[idx] = min(cum_max, 1.0)
    return adjusted


def per_variant_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per variant, with median rel-RMSEP, IQR, wins/losses/ties,
    failures, convergence rate, fit time."""
    rows = []
    grp = df.groupby("variant")
    for variant, sub in grp:
        ok = sub[sub.status == "ok"]
        n_total = len(sub)
        n_ok = len(ok)
        n_fail = n_total - n_ok
        n_converged = int(ok["converged"].fillna(True).astype(bool).sum()) \
            if "converged" in ok.columns else n_ok
        rel = ok["rel_rmsep_vs_pls"].dropna().values
        med, lo, hi = _bootstrap_median_ratio_ci(
            ok["rmsep"].values,
            ok["ref_rmse_pls"].values,
        )
        wins, losses, ties, _ = _sign_test(np.log(rel) if len(rel) else np.array([]))
        fit_time_med = float(ok["fit_time_s"].median()) if n_ok else float("nan")
        rows.append({
            "variant": variant,
            "n_total": n_total, "n_ok": n_ok, "n_fail": n_fail,
            "n_converged": n_converged,
            "median_rel_pls": med,
            "ci_low_rel_pls": lo, "ci_high_rel_pls": hi,
            "wins_vs_pls": wins, "losses_vs_pls": losses, "ties_vs_pls": ties,
            "median_rel_ridge": float(ok["rel_rmsep_vs_ridge"].median()),
            "median_rel_tabpfn_opt": float(ok["rel_rmsep_vs_tabpfn_opt"].median()),
            "median_fit_time_s": fit_time_med,
        })
    return pd.DataFrame(rows).sort_values("median_rel_pls")


def pairwise_wilcoxon(df: pd.DataFrame, base_variant: str) -> pd.DataFrame:
    """Compare every variant against ``base_variant`` via Wilcoxon
    signed-rank on log(rmsep_variant / rmsep_base) per dataset.

    Reports pvalue (one-sided "variant better than base"), Holm-adjusted
    pvalue, median log-ratio, n_paired.
    """
    base = df[(df.variant == base_variant) & (df.status == "ok")][
        ["dataset_group", "dataset", "rmsep"]
    ].rename(columns={"rmsep": "rmsep_base"})
    rows = []
    other_variants = sorted(set(df.variant) - {base_variant})
    pvalues = []
    for variant in other_variants:
        sub = df[(df.variant == variant) & (df.status == "ok")][
            ["dataset_group", "dataset", "rmsep"]
        ].rename(columns={"rmsep": "rmsep_variant"})
        merged = pd.merge(base, sub, on=["dataset_group", "dataset"])
        if len(merged) < 5:
            rows.append({
                "variant": variant, "n_paired": len(merged),
                "median_log_ratio": float("nan"), "pvalue": float("nan"),
            })
            pvalues.append(float("nan"))
            continue
        log_ratio = np.log(merged["rmsep_variant"].values / merged["rmsep_base"].values)
        p = _wilcoxon_one_sided(log_ratio)
        rows.append({
            "variant": variant,
            "n_paired": len(merged),
            "median_log_ratio": float(np.median(log_ratio)),
            "pvalue": p,
        })
        pvalues.append(p)
    pvalues = np.array(pvalues, dtype=float)
    finite = ~np.isnan(pvalues)
    if finite.any():
        adj = np.full(len(pvalues), float("nan"))
        adj[finite] = _holm(pvalues[finite])
        for i, r in enumerate(rows):
            r["holm_pvalue"] = float(adj[i]) if finite[i] else float("nan")
    return pd.DataFrame(rows).sort_values("median_log_ratio")


def branch_lift_within_family(df: pd.DataFrame) -> pd.DataFrame:
    """Compare variants within the same family (mkR-softmax_cv,
    MKM-reml, etc.) for branch-lift estimation.

    For each (family, base-no-branch), report median log(rmsep_branch /
    rmsep_no_branch) across datasets and Wilcoxon p-value.
    """
    rows = []
    for family_base in ("mkR-softmax_cv", "MKM-reml"):
        base = df[(df.variant == family_base) & (df.status == "ok")][
            ["dataset_group", "dataset", "rmsep"]
        ].rename(columns={"rmsep": "rmsep_base"})
        for branch in ("snv", "msc", "asls"):
            variant = f"{family_base}-{branch}"
            sub = df[(df.variant == variant) & (df.status == "ok")][
                ["dataset_group", "dataset", "rmsep"]
            ].rename(columns={"rmsep": "rmsep_variant"})
            merged = pd.merge(base, sub, on=["dataset_group", "dataset"])
            if len(merged) < 5:
                rows.append({
                    "family_base": family_base, "branch": branch,
                    "n_paired": len(merged),
                    "median_log_ratio": float("nan"),
                    "pvalue": float("nan"),
                })
                continue
            log_ratio = np.log(merged["rmsep_variant"].values / merged["rmsep_base"].values)
            p = _wilcoxon_one_sided(log_ratio)
            rows.append({
                "family_base": family_base,
                "branch": branch,
                "n_paired": len(merged),
                "median_log_ratio": float(np.median(log_ratio)),
                "pvalue": p,
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--pairwise-base", type=str, default="mkR-softmax_cv",
        help="Base variant for pairwise Wilcoxon tests.",
    )
    args = parser.parse_args(argv)
    df = _load(args.results_csv)
    if df.empty:
        print("[stats] empty CSV; nothing to do")
        return 0
    out_dir = args.out_dir or args.results_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pv = per_variant_table(df)
    print("== per_variant_stats ==")
    print(pv.to_string(index=False))
    pv.to_csv(out_dir / "per_variant_stats.csv", index=False)
    print(f"  -> {out_dir / 'per_variant_stats.csv'}")

    pw = pairwise_wilcoxon(df, args.pairwise_base)
    print()
    print(f"== pairwise vs {args.pairwise_base} ==")
    print(pw.to_string(index=False))
    pw.to_csv(out_dir / "pairwise_wilcoxon.csv", index=False)
    print(f"  -> {out_dir / 'pairwise_wilcoxon.csv'}")

    bl = branch_lift_within_family(df)
    print()
    print("== branch_lift_within_family ==")
    print(bl.to_string(index=False))
    bl.to_csv(out_dir / "branch_lift.csv", index=False)
    print(f"  -> {out_dir / 'branch_lift.csv'}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
