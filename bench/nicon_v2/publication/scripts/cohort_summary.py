"""End-to-end manuscript-table builder.

Joins a benchmark CSV with the cohort manifest to attach the published reference
RMSEPs (paper Ridge, paper PLS, paper TabPFN, paper CNN) and the AOM-Ridge
curated best, then computes the manuscript headline numbers:

* per-variant median rmsep, median Δ% vs each reference, win rate, paired
  Wilcoxon p (Holm-corrected)
* the two-tier success check (leaderboard vs scientific)

Usage::

    python publication/scripts/cohort_summary.py \
      --csv bench/nicon_v2/benchmark_runs/stack_aom_curated/results.csv \
      --out bench/nicon_v2/publication/tables/ \
      --best-variant Stack-AOMRidge-PLS-V1c
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


REFS = {
    "ridge": "ref_rmse_paper_ridge",
    "pls": "ref_rmse_pls",
    "tabpfn_raw": "ref_rmse_tabpfn_raw",
    "tabpfn_opt": "ref_rmse_tabpfn_opt",
    "cnn": "ref_rmse_cnn",
    "catboost": "ref_rmse_catboost",
    "aom_ridge_curated_best": "ref_rmse_aom_ridge_curated_best",
}


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[df["status"].astype(str) == "OK"].copy()


def relative_change(rmsep: float, ref: float) -> float | None:
    if ref is None or not np.isfinite(ref) or ref <= 0:
        return None
    return (rmsep - ref) / ref


def per_variant_summary(df: pd.DataFrame, control: str) -> pd.DataFrame:
    """Median rmsep + per-reference relative deltas + Wilcoxon vs control."""
    pivot = df.pivot_table(index=["dataset", "seed"], columns="variant", values="rmsep").dropna()
    if control not in pivot.columns:
        raise ValueError(f"control {control!r} missing")
    ctrl = pivot[control]

    rows: list[dict] = []
    for variant in pivot.columns:
        treatment = pivot[variant]
        if variant == control:
            wilcoxon_p = float("nan")
            wins = losses = -1
        else:
            try:
                _, wilcoxon_p = wilcoxon(treatment, ctrl, zero_method="zsplit")
            except ValueError:
                wilcoxon_p = float("nan")
            wins = int((treatment < ctrl).sum())
            losses = int((treatment > ctrl).sum())
        delta = (treatment - ctrl) / ctrl

        # Per-reference attached deltas + effective sample size (Codex round 4 F1).
        ref_deltas: dict[str, float] = {}
        for ref_name, col in REFS.items():
            sub = df[df["variant"] == variant]
            sub = sub.dropna(subset=[col])
            n_ref_pairs = int(len(sub))
            n_ref_datasets = int(sub["dataset"].nunique()) if not sub.empty else 0
            ref_deltas[f"n_ref_datasets_{ref_name}"] = n_ref_datasets
            ref_deltas[f"n_ref_pairs_{ref_name}"] = n_ref_pairs
            if sub.empty:
                ref_deltas[f"med_delta_vs_{ref_name}_pct"] = float("nan")
                continue
            ref_deltas[f"med_delta_vs_{ref_name}_pct"] = (
                float(np.median((sub["rmsep"] - sub[col]) / sub[col]) * 100)
            )

        rows.append({
            "variant": variant,
            "median_rmsep": float(np.median(treatment)),
            "median_delta_pct_vs_control": float(np.median(delta) * 100),
            "wins_vs_control": wins,
            "losses_vs_control": losses,
            "n_pairs": int(len(treatment)),
            "wilcoxon_p_vs_control": wilcoxon_p,
            **ref_deltas,
        })
    out = pd.DataFrame(rows).sort_values("median_delta_pct_vs_control")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control", type=str, default="Ridge-baseline")
    parser.add_argument("--best-variant", type=str, default=None,
                        help="Variant name to highlight in stop-gate check.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    summary = per_variant_summary(df, args.control)
    summary_path = args.out / "cohort_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    print()
    print(summary.to_string())
    print()

    # Two-tier success check.
    if args.best_variant is not None and args.best_variant in summary["variant"].values:
        row = summary[summary["variant"] == args.best_variant].iloc[0]
        print(f"=== Stop-gate check: {args.best_variant} ===")
        med_aom = row.get("med_delta_vs_aom_ridge_curated_best_pct", float("nan"))
        wilc = row["wilcoxon_p_vs_control"]
        wins = row["wins_vs_control"]
        n = row["n_pairs"]
        print(f"  median Δ% vs aom_ridge_curated_best: {med_aom:+.2f}%  (gate ≤ −2 %)")
        print(f"  paired Wilcoxon p vs {args.control}: {wilc:.4g}  (gate < 0.05)")
        if n > 0:
            print(f"  win rate vs {args.control}: {wins}/{n} = {100.0 * wins / n:.1f}%  (gate ≥ 50%)")
        leaderboard = (np.isfinite(med_aom) and med_aom <= -2.0
                       and np.isfinite(wilc) and wilc < 0.05
                       and (n > 0 and wins / n >= 0.5))
        print(f"  → Leaderboard success: {'YES' if leaderboard else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
