"""Aggregate the r21_curated_oof_multiseed results.

Reads ``bench/nicon_v2/benchmark_runs/r21_curated_oof_multiseed/results.csv``
and produces:

- a per-dataset summary CSV with median / std / IQR of ``s*`` over seeds and
  the per-seed Δ% rmsep vs Ridge / AOM-Ridge / paper PLS / TabPFN;
- a cohort-level summary table printed to stdout;
- the four production stop gates from `PLAN_REPRISE_2026-05.md` §7;
- the shrinkage Option-A reopen flag per the Codex round-2 condition on
  D-B-002c-revised: any per-dataset IQR(``s*``) > 0.3 sets
  ``option_a_reopen_needed = True``.

Usage::

    cd bench/nicon_v2
    python benchmarks/aggregate_r21.py [--in PATH] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IN = (
    REPO_ROOT
    / "bench"
    / "nicon_v2"
    / "benchmark_runs"
    / "r21_curated_oof_multiseed"
    / "results.csv"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "bench" / "nicon_v2" / "benchmark_runs" / "r21_curated_oof_multiseed"
)


def _ratio(rmsep: pd.Series, ref: pd.Series) -> pd.Series:
    return rmsep.astype(float) / ref.astype(float) - 1.0


def _quantile_rmse_diff(group: pd.DataFrame, ref_col: str) -> pd.Series:
    deltas = _ratio(group["rmsep"], group[ref_col])
    deltas = deltas.dropna()
    if deltas.empty:
        return pd.Series({"median": np.nan, "q75": np.nan, "q90": np.nan, "worst": np.nan})
    return pd.Series(
        {
            "median": float(deltas.median()),
            "q75": float(deltas.quantile(0.75)),
            "q90": float(deltas.quantile(0.9)),
            "worst": float(deltas.max()),
        }
    )


def _wins_per_seed(df: pd.DataFrame, ref_col: str) -> tuple[int, int]:
    """Per-(dataset, seed) wins where the variant beat the baseline."""
    deltas = _ratio(df["rmsep"], df[ref_col])
    valid = ~deltas.isna()
    wins = int((deltas[valid] < 0).sum())
    total = int(valid.sum())
    return wins, total


def _extract_s_star(row: pd.Series) -> float | None:
    """Pull s_star from explicit column or the hyperparams_json fallback."""
    val = row.get("shrinkage_s_star")
    if val is not None and not pd.isna(val):
        return float(val)
    raw = row.get("hyperparams_json")
    if isinstance(raw, str) and raw:
        try:
            hp = json.loads(raw)
            if "shrinkage_s_star" in hp:
                return float(hp["shrinkage_s_star"])
        except json.JSONDecodeError:
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate r21 multiseed results")
    parser.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    df = pd.read_csv(args.inp)
    df = df[df["status"] == "OK"].copy()

    target = "V2L-Residual-AOMPLS-shrinkage"
    sub = df[df["variant"] == target].copy()
    if sub.empty:
        print(f"[r21] no rows found for variant {target!r}; aborting")
        return 1
    print(f"[r21] {target}: {len(sub)} rows across "
          f"{sub['dataset'].nunique()} datasets × {sub['seed'].nunique()} seeds")

    sub["s_star"] = sub.apply(_extract_s_star, axis=1)

    # --- Per-dataset s* stability ---
    per_ds = sub.groupby("dataset")["s_star"].agg(
        s_mean="mean", s_std="std", s_min="min", s_max="max",
        s_q25=lambda x: x.quantile(0.25), s_q75=lambda x: x.quantile(0.75)
    ).reset_index()
    per_ds["s_iqr"] = per_ds["s_q75"] - per_ds["s_q25"]
    iqr_violations = per_ds[per_ds["s_iqr"] > 0.3]
    option_a_reopen = bool(len(iqr_violations) > 0)

    # --- Per-dataset Δ% vs each baseline ---
    deltas: dict[str, pd.DataFrame] = {}
    for ref_label, ref_col in (
        ("vs_pls", "ref_rmse_pls"),
        ("vs_paper_ridge", "ref_rmse_paper_ridge"),
        ("vs_tabpfn_raw", "ref_rmse_tabpfn_raw"),
        ("vs_tabpfn_opt", "ref_rmse_tabpfn_opt"),
        ("vs_cnn", "ref_rmse_cnn"),
        ("vs_catboost", "ref_rmse_catboost"),
        ("vs_aom_ridge", "ref_rmse_aom_ridge_curated_best"),
    ):
        rows: list[dict] = []
        for ds, ds_sub in sub.groupby("dataset"):
            for seed, seed_sub in ds_sub.groupby("seed"):
                if seed_sub[ref_col].isna().any():
                    continue
                d = float(_ratio(seed_sub["rmsep"], seed_sub[ref_col]).iloc[0])
                rows.append({"dataset": ds, "seed": seed, ref_label: d})
        deltas[ref_label] = pd.DataFrame(rows)

    # --- Stop gates from plan §7 ---
    print()
    print("=== Cohort summary ===")
    for ref_label, ref_col in (
        ("paper PLS", "ref_rmse_pls"),
        ("paper Ridge", "ref_rmse_paper_ridge"),
        ("paper TabPFN-raw", "ref_rmse_tabpfn_raw"),
        ("paper TabPFN-opt", "ref_rmse_tabpfn_opt"),
        ("paper CNN", "ref_rmse_cnn"),
        ("paper CatBoost", "ref_rmse_catboost"),
        ("aom_ridge_curated_best", "ref_rmse_aom_ridge_curated_best"),
    ):
        stats = _quantile_rmse_diff(sub, ref_col)
        wins, total = _wins_per_seed(sub, ref_col)
        if total == 0:
            print(f"  vs {ref_label:25s} : no reference data")
            continue
        print(f"  vs {ref_label:25s} : median={stats['median']:+.3f}  "
              f"q75={stats['q75']:+.3f}  q90={stats['q90']:+.3f}  "
              f"worst={stats['worst']:+.3f}  wins={wins}/{total}")

    # Stop gates: production / science / do-no-harm
    median_aom_ridge = float(_ratio(sub["rmsep"], sub["ref_rmse_aom_ridge_curated_best"]).median())
    wins_aom, total_aom = _wins_per_seed(sub, "ref_rmse_aom_ridge_curated_best")
    win_rate_aom = wins_aom / total_aom if total_aom else float("nan")
    median_paper_nicon = float(_ratio(sub["rmsep"], sub["ref_rmse_cnn"]).median())
    wins_cnn, total_cnn = _wins_per_seed(sub, "ref_rmse_cnn")
    win_rate_cnn = wins_cnn / total_cnn if total_cnn else float("nan")
    catastrophic_rate = float(sub["catastrophic"].astype(bool).sum()) / len(sub)

    print()
    print("=== Plan §7 stop gates ===")
    prod_pass = (median_aom_ridge <= -0.02) and (win_rate_aom >= 0.5)
    print("  Production gate (≥ −2 % median vs aom_ridge_curated_best AND ≥ 50 % wins):")
    print(f"    median Δ% = {median_aom_ridge:+.3f}, win rate = {win_rate_aom:.1%}  -> "
          f"{'PASS' if prod_pass else 'FAIL'}")
    sci_pass = (median_paper_nicon <= -0.05) and (win_rate_cnn >= 0.75)
    print("  Science gate (≥ −5 % median vs paper CNN AND ≥ 75 % wins):")
    print(f"    median Δ% = {median_paper_nicon:+.3f}, win rate = {win_rate_cnn:.1%}  -> "
          f"{'PASS' if sci_pass else 'FAIL'}")
    dnh_pass = catastrophic_rate <= 0.05
    print("  Do-no-harm gate (≤ 5 % catastrophic per (dataset, seed)):")
    print(f"    catastrophic rate = {catastrophic_rate:.1%}  -> "
          f"{'PASS' if dnh_pass else 'FAIL'}")

    # --- Shrinkage stability (Codex round-2 condition) ---
    print()
    print("=== Shrinkage s* stability (Codex round-2 condition) ===")
    print(f"  Datasets: {len(per_ds)}")
    print(f"  s* IQR > 0.3 on {len(iqr_violations)} datasets")
    if len(iqr_violations) > 0:
        print("  Datasets requiring Option-A reopen:")
        print(iqr_violations[["dataset", "s_min", "s_max", "s_iqr"]].to_string(index=False))
    print("  s* per-seed histogram (cohort-wide):")
    hist = sub["s_star"].value_counts().sort_index()
    for s, count in hist.items():
        print(f"    s={s:.2f}: {count}")
    print(f"\n  option_a_reopen_needed = {option_a_reopen}")

    # --- Persist outputs ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_ds_path = args.out_dir / "per_dataset_s_star.csv"
    per_ds.to_csv(per_ds_path, index=False)
    print(f"\n[r21] wrote per-dataset s* table to {per_ds_path}")

    summary = {
        "n_rows": len(sub),
        "n_datasets": int(sub["dataset"].nunique()),
        "n_seeds": int(sub["seed"].nunique()),
        "median_delta_vs_aom_ridge": median_aom_ridge,
        "win_rate_vs_aom_ridge": win_rate_aom,
        "median_delta_vs_paper_cnn": median_paper_nicon,
        "win_rate_vs_paper_cnn": win_rate_cnn,
        "catastrophic_rate": catastrophic_rate,
        "production_gate_pass": prod_pass,
        "science_gate_pass": sci_pass,
        "do_no_harm_gate_pass": dnh_pass,
        "n_datasets_iqr_violation": int(len(iqr_violations)),
        "option_a_reopen_needed": option_a_reopen,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[r21] wrote cohort summary to {summary_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
