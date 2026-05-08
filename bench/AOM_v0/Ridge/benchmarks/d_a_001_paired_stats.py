"""D-A-001 paired statistics computation for Codex round-6 GATE.

Inputs:
    bench/AOM_v0/Ridge/benchmark_runs/da001_partial_fast12_seeds012/results.csv

Outputs:
    bench/AOM_v0/Ridge/docs/D_A_001_fast12_paired_stats.csv
    bench/AOM_v0/Ridge/docs/D_A_001_FAST12_PAIRED_STATS.md

Per Codex round-6 verdict (SYNC.md 2026-05-07 04:10 CEST):
  - Selectors: AOMRidge-Blender-headline-spxy3, AOMRidge-AutoSelect-headline-spxy3
  - Baselines: Ridge-tuned-cv5, ASLS-AOM-compact-cv5-numpy,
               AOMRidge-global-compact-none, AOMRidge-Local-compact-knn50
  - Wilcoxon: paired one-sided (target < baseline) on log RMSEP deltas
  - Primary unit: dataset-level seed mean (N=12) ; sensitivity: row-level (N=36)
  - Holm correction across the 4 baselines x 2 selectors = 8 comparisons
  - Effect size: Cliff's delta + median delta% + q90 ratio
  - Threshold: p_Holm < 0.05, median delta <= -3% (preferably -5%),
               |Cliff's delta| >= 0.147, q90 target/baseline <= 1.10

Usage: python d_a_001_paired_stats.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Paths -----------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent / "benchmark_runs" / "da001_partial_fast12_seeds012"
RESULTS_CSV = RUN_DIR / "results.csv"
DOCS_DIR = HERE.parent / "docs"
OUT_CSV = DOCS_DIR / "D_A_001_fast12_paired_stats.csv"
OUT_MD = DOCS_DIR / "D_A_001_FAST12_PAIRED_STATS.md"

SELECTORS = [
    "AOMRidge-Blender-headline-spxy3",
    "AOMRidge-AutoSelect-headline-spxy3",
]
BASELINES = [
    "Ridge-tuned-cv5",
    "ASLS-AOM-compact-cv5-numpy",
    "AOMRidge-global-compact-none",
    "AOMRidge-Local-compact-knn50",
]


def cliffs_delta(target: np.ndarray, baseline: np.ndarray) -> float:
    """Paired-sign Cliff's delta in [-1, +1].

    Convention: lower RMSEP = better. We compute
    ``(#(target<baseline) - #(target>baseline)) / N`` so a positive value
    means the target wins more often than the baseline; the favourable
    direction in `_verdict()` is therefore ``cliff >= +0.147``.
    """
    n = target.size
    if n == 0:
        return float("nan")
    diff = target - baseline
    return float((np.sum(diff < 0) - np.sum(diff > 0)) / n)


def holm_pvalues(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment.

    Returns p-values adjusted so a single < alpha threshold can be applied.
    """
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, i in enumerate(order):
        adjusted_p = (n - rank) * pvals[i]
        running_max = max(running_max, adjusted_p)
        adjusted[i] = min(1.0, running_max)
    return adjusted


def compute_pair_stats(df: pd.DataFrame, target: str, baseline: str) -> dict:
    """Paired comparison of `target` vs `baseline` on the OK-rows subset.

    Drops dataset/seed pairs where either side is not OK.
    Returns dict with row-level and dataset-level statistics.
    """
    cols = ["dataset", "seed", "rmsep"]
    t = (
        df[df["canonical_name"] == target][cols]
        .rename(columns={"rmsep": "rmsep_target"})
    )
    b = (
        df[df["canonical_name"] == baseline][cols]
        .rename(columns={"rmsep": "rmsep_baseline"})
    )
    paired = t.merge(b, on=["dataset", "seed"], how="inner")
    paired = paired.dropna(subset=["rmsep_target", "rmsep_baseline"])
    paired = paired[(paired["rmsep_target"] > 0) & (paired["rmsep_baseline"] > 0)]
    if paired.empty:
        return {
            "target": target,
            "baseline": baseline,
            "n_rows": 0,
            "n_datasets": 0,
        }

    paired["log_delta"] = np.log(paired["rmsep_target"]) - np.log(
        paired["rmsep_baseline"]
    )
    paired["ratio"] = paired["rmsep_target"] / paired["rmsep_baseline"]
    paired["delta_pct"] = (paired["ratio"] - 1.0) * 100.0

    # Per-dataset means (primary unit per Codex round-6).
    by_ds = (
        paired.groupby("dataset")
        .agg(
            rmsep_target=("rmsep_target", "mean"),
            rmsep_baseline=("rmsep_baseline", "mean"),
        )
        .reset_index()
    )
    by_ds["log_delta"] = np.log(by_ds["rmsep_target"]) - np.log(
        by_ds["rmsep_baseline"]
    )
    by_ds["ratio"] = by_ds["rmsep_target"] / by_ds["rmsep_baseline"]
    by_ds["delta_pct"] = (by_ds["ratio"] - 1.0) * 100.0

    # Wilcoxon paired one-sided (alternative='less' means target<baseline win).
    def _wilcoxon_less(deltas: np.ndarray) -> float:
        deltas = deltas[deltas != 0.0]
        if deltas.size == 0:
            return float("nan")
        try:
            return float(wilcoxon(deltas, alternative="less").pvalue)
        except ValueError:
            return float("nan")

    p_row = _wilcoxon_less(paired["log_delta"].to_numpy())
    p_ds = _wilcoxon_less(by_ds["log_delta"].to_numpy())

    # Worst regression at the dataset level (largest positive ratio = baseline wins).
    worst_row = by_ds.loc[by_ds["ratio"].idxmax()]

    return {
        "target": target,
        "baseline": baseline,
        "n_rows": int(len(paired)),
        "n_datasets": int(len(by_ds)),
        "wins_rows": int((paired["log_delta"] < 0).sum()),
        "wins_datasets": int((by_ds["log_delta"] < 0).sum()),
        "median_ratio_rows": float(paired["ratio"].median()),
        "median_ratio_ds": float(by_ds["ratio"].median()),
        "median_delta_pct_ds": float(by_ds["delta_pct"].median()),
        "q75_ratio_ds": float(by_ds["ratio"].quantile(0.75)),
        "q90_ratio_ds": float(by_ds["ratio"].quantile(0.90)),
        "max_ratio_ds": float(by_ds["ratio"].max()),
        "worst_dataset": str(worst_row["dataset"]),
        "worst_ratio": float(worst_row["ratio"]),
        "p_wilcoxon_rows": p_row,
        "p_wilcoxon_ds": p_ds,
        "cliffs_delta_rows": cliffs_delta(
            paired["rmsep_target"].to_numpy(), paired["rmsep_baseline"].to_numpy()
        ),
        "cliffs_delta_ds": cliffs_delta(
            by_ds["rmsep_target"].to_numpy(), by_ds["rmsep_baseline"].to_numpy()
        ),
        "_paired_rows": paired,
        "_per_dataset": by_ds,
    }


def main() -> None:
    if not RESULTS_CSV.exists():
        raise SystemExit(f"results.csv not found at {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)
    df_ok = df[df["status"] == "ok"].copy()

    rows: list[dict] = []
    pvals_ds: list[float] = []
    pvals_rows: list[float] = []
    keys: list[tuple[str, str]] = []
    full_results: dict[tuple[str, str], dict] = {}

    for selector in SELECTORS:
        for baseline in BASELINES:
            stats = compute_pair_stats(df_ok, selector, baseline)
            full_results[(selector, baseline)] = stats
            keys.append((selector, baseline))
            pvals_ds.append(stats.get("p_wilcoxon_ds", float("nan")) or float("nan"))
            pvals_rows.append(stats.get("p_wilcoxon_rows", float("nan")) or float("nan"))

    # Holm adjust ignoring NaNs.
    def _adjust(pvals: list[float]) -> list[float]:
        finite_idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
        finite_p = [pvals[i] for i in finite_idx]
        adjusted = holm_pvalues(finite_p) if finite_p else []
        out = [float("nan")] * len(pvals)
        for i, p in zip(finite_idx, adjusted, strict=False):
            out[i] = p
        return out

    holm_ds = _adjust(pvals_ds)
    holm_rows = _adjust(pvals_rows)

    for k, p_ds_holm, p_rows_holm in zip(keys, holm_ds, holm_rows, strict=False):
        full_results[k]["p_wilcoxon_ds_holm"] = p_ds_holm
        full_results[k]["p_wilcoxon_rows_holm"] = p_rows_holm
        s = full_results[k]
        rows.append(
            {
                "selector": s["target"],
                "baseline": s["baseline"],
                "n_rows": s.get("n_rows", 0),
                "n_datasets": s.get("n_datasets", 0),
                "wins_rows_36": s.get("wins_rows"),
                "wins_datasets_12": s.get("wins_datasets"),
                "median_ratio_ds": s.get("median_ratio_ds"),
                "median_delta_pct_ds": s.get("median_delta_pct_ds"),
                "q75_ratio_ds": s.get("q75_ratio_ds"),
                "q90_ratio_ds": s.get("q90_ratio_ds"),
                "worst_ratio": s.get("worst_ratio"),
                "worst_dataset": s.get("worst_dataset"),
                "cliffs_delta_ds": s.get("cliffs_delta_ds"),
                "p_wilcoxon_ds": s.get("p_wilcoxon_ds"),
                "p_wilcoxon_ds_holm": s.get("p_wilcoxon_ds_holm"),
                "p_wilcoxon_rows": s.get("p_wilcoxon_rows"),
                "p_wilcoxon_rows_holm": s.get("p_wilcoxon_rows_holm"),
            }
        )

    summary_df = pd.DataFrame(rows)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_CSV, index=False)

    # Decision-grade summary in markdown.
    md_lines: list[str] = []
    md_lines.append("# D-A-001 fast12 paired statistics (Codex round-6 GATE)")
    md_lines.append("")
    md_lines.append(
        "Source: `bench/AOM_v0/Ridge/benchmark_runs/da001_partial_fast12_seeds012/results.csv`"
    )
    md_lines.append("")
    md_lines.append(
        "Cohort: fast12_transfer_core x seeds 0/1/2 (12 datasets x 3 seeds = 36 rows per candidate)."
    )
    md_lines.append("")
    md_lines.append(
        "Wilcoxon = paired one-sided (target < baseline) on log RMSEP deltas. "
        "Primary unit = per-dataset seed-mean (N=12). Row-level (N=36) reported "
        "as sensitivity. Holm correction across the 4 x 2 = 8 comparisons."
    )
    md_lines.append("")
    md_lines.append("Win threshold conventions (from Codex round-6, SYNC 04:10 CEST):")
    md_lines.append("- median_delta_pct_ds <= -3 % (preferably <= -5 % for headline language)")
    md_lines.append("- |cliffs_delta_ds| >= 0.147 in the favourable direction")
    md_lines.append("- q90_ratio_ds <= 1.10 (no-harm sanity check)")
    md_lines.append("- p_wilcoxon_ds_holm < 0.05")
    md_lines.append("")

    md_lines.append("## Summary table")
    md_lines.append("")
    header = (
        "| Selector | Baseline | N_ds | Wins/12 | Wins/36 | "
        "Median Δ% | q75 ratio | q90 ratio | Worst ratio (dataset) | "
        "Cliff's δ | p (ds, Holm) | Verdict |"
    )
    sep = "|" + "|".join(["---"] * 12) + "|"
    md_lines.append(header)
    md_lines.append(sep)

    for r in rows:
        verdict = _verdict(r)
        worst = (
            f"{r['worst_ratio']:.3f} ({r['worst_dataset']})"
            if r["worst_ratio"] is not None and not math.isnan(r["worst_ratio"])
            else "NA"
        )
        md_lines.append(
            "| "
            + " | ".join(
                [
                    r["selector"],
                    r["baseline"],
                    str(r["n_datasets"]),
                    str(r["wins_datasets_12"]),
                    str(r["wins_rows_36"]),
                    f"{r['median_delta_pct_ds']:+.2f}",
                    f"{r['q75_ratio_ds']:.3f}",
                    f"{r['q90_ratio_ds']:.3f}",
                    worst,
                    f"{r['cliffs_delta_ds']:+.3f}",
                    f"{r['p_wilcoxon_ds_holm']:.4f}",
                    verdict,
                ]
            )
            + " |"
        )

    md_lines.append("")
    md_lines.append("## Per-comparison detail")
    md_lines.append("")

    for selector in SELECTORS:
        md_lines.append(f"### {selector}")
        md_lines.append("")
        for baseline in BASELINES:
            s = full_results[(selector, baseline)]
            md_lines.append(f"#### vs {baseline}")
            md_lines.append("")
            md_lines.append(
                f"- Rows kept: {s.get('n_rows', 0)} (out of 36 per side); "
                f"datasets kept: {s.get('n_datasets', 0)} (out of 12)"
            )
            md_lines.append(
                f"- Wins (per-row, N=36): {s.get('wins_rows')}/{s.get('n_rows', 0)} ; "
                f"Wins (per-dataset, N=12): {s.get('wins_datasets')}/{s.get('n_datasets', 0)}"
            )
            md_lines.append(
                f"- Median ratio (ds): {s.get('median_ratio_ds'):.3f}  "
                f"(median Δ% = {s.get('median_delta_pct_ds'):+.2f} %)"
            )
            md_lines.append(
                f"- q75 / q90 / worst ratio (ds): {s.get('q75_ratio_ds'):.3f} / "
                f"{s.get('q90_ratio_ds'):.3f} / {s.get('max_ratio_ds'):.3f}"
            )
            md_lines.append(
                f"- Worst-regression dataset: **{s.get('worst_dataset')}** "
                f"(ratio = {s.get('worst_ratio'):.3f})"
            )
            md_lines.append(
                f"- Cliff's δ (ds, paired): {s.get('cliffs_delta_ds'):+.3f}"
            )
            md_lines.append(
                f"- Wilcoxon (ds, one-sided less): p = {s.get('p_wilcoxon_ds'):.4f} "
                f"-> Holm-adjusted = {s.get('p_wilcoxon_ds_holm'):.4f}"
            )
            md_lines.append(
                f"- Wilcoxon (rows, one-sided less): p = {s.get('p_wilcoxon_rows'):.4f} "
                f"-> Holm-adjusted = {s.get('p_wilcoxon_rows_holm'):.4f}"
            )
            md_lines.append("")

    md_lines.append("## Pre-registered descriptive Friedman-Nemenyi (5 AOMRidge variants)")
    md_lines.append("")
    md_lines.append(
        "Variants pre-registered before looking at ranks: "
        "`AOMRidge-global-compact-none`, `AOMRidge-Local-compact-knn50`, "
        "`AOMRidge-MultiBranchMKL-compact-shrink03`, "
        "`AOMRidge-Blender-headline-spxy3`, `AOMRidge-AutoSelect-headline-spxy3`."
    )
    fn_variants = [
        "AOMRidge-global-compact-none",
        "AOMRidge-Local-compact-knn50",
        "AOMRidge-MultiBranchMKL-compact-shrink03",
        "AOMRidge-Blender-headline-spxy3",
        "AOMRidge-AutoSelect-headline-spxy3",
    ]
    pivot = (
        df_ok[df_ok["canonical_name"].isin(fn_variants)][
            ["dataset", "seed", "canonical_name", "rmsep"]
        ]
        .groupby(["dataset", "seed", "canonical_name"], as_index=False)["rmsep"]
        .mean()
    )
    wide = pivot.pivot_table(
        index=["dataset", "seed"], columns="canonical_name", values="rmsep"
    )
    wide = wide.dropna()
    md_lines.append("")
    md_lines.append(f"Rows with all 5 variants OK: {len(wide)} (out of 36 possible).")
    if len(wide) >= 6:
        from scipy.stats import friedmanchisquare

        ranks = wide[fn_variants].rank(axis=1)
        mean_ranks = ranks.mean(axis=0)
        chi2, pf = friedmanchisquare(*[wide[v].to_numpy() for v in fn_variants])
        md_lines.append("")
        md_lines.append(
            f"Friedman: chi^2 = {chi2:.3f}, p = {pf:.4f} (descriptive only; "
            "omnibus is reserved for the production/full-57 escalation)."
        )
        md_lines.append("")
        md_lines.append("Mean rank per variant (1 = best):")
        for v in fn_variants:
            md_lines.append(f"- `{v}`: {mean_ranks[v]:.3f}")

    md_lines.append("")
    md_lines.append("## Selector diagnostics")
    md_lines.append("")
    md_lines.append(
        "Selector diagnostics (AutoSelect chosen-candidate counts, Blender "
        "weight mean/std, OOF fold RMSE variance) are not surfaced in this "
        "results.csv schema; the existing harness logs them only to per-run "
        "JSON sidecars under `benchmark_runs/.../<canonical_name>/`. A "
        "follow-up pass to aggregate those into a markdown table is staged "
        "but out of scope for this initial GATE deliverable."
    )

    OUT_MD.write_text("\n".join(md_lines))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


def _verdict(row: dict) -> str:
    """Apply the Codex round-6 multi-criteria gate.

    With our `cliffs_delta` convention (paired (target - baseline) with
    lower RMSE = better), a positive value means target wins more often
    than baseline; the favourable direction is ``cliff >= +0.147``.
    """
    p = row.get("p_wilcoxon_ds_holm")
    median = row.get("median_delta_pct_ds")
    cliff = row.get("cliffs_delta_ds")
    q90 = row.get("q90_ratio_ds")
    if p is None or math.isnan(p):
        return "NA"
    pass_p = p < 0.05
    pass_median = median is not None and median <= -3.0
    pass_strong_median = median is not None and median <= -5.0
    pass_cliff = cliff is not None and cliff >= 0.147
    pass_q90 = q90 is not None and q90 <= 1.10
    if pass_p and pass_strong_median and pass_cliff and pass_q90:
        return "WIN_strong"
    if pass_p and pass_median and pass_cliff and pass_q90:
        return "WIN_practical"
    if pass_p and pass_cliff and pass_q90:
        return "TREND"
    return "NO_WIN"


if __name__ == "__main__":
    main()
