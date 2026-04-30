"""End-to-end comparison table joining nicon_v2 results with all reference baselines.

Aggregates rmsep per dataset for:
  * Paper baselines: CNN (paper Nicon), CatBoost, PLS, Ridge, TabPFN-Raw, TabPFN-opt
    — read from `bench/AOM_v0/publication/tables/master_pivot.csv`.
  * AOM-PLS best — minimum rmsep across all variants in
    `bench/AOM_v0/publication/tables/summary_per_dataset.csv`.
  * AOM-Ridge best — minimum rmsep across all variants in
    `bench/AOM_v0/Ridge/benchmark_runs/<dir>/results.csv`.
  * Our nicon_v2 baselines and stacks: from `bench/nicon_v2/benchmark_runs/<run>/results.csv`.

Output columns (long form, one row per (dataset, model_class)):
  database_name, dataset, model_class, rmsep, source

Also produces a wide pivot for the manuscript table.

Usage::

    python publication/scripts/full_comparison.py \
      --our-csv bench/nicon_v2/benchmark_runs/stack_curated/results.csv \
      --our-variant Stack-Ridge-PLS-V1c \
      --out bench/nicon_v2/publication/tables/full_comparison/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NIRS4ALL_PKG_ROOT = Path("/home/delete/nirs4all/nirs4all")
PAPER_PIVOT_CSV     = NIRS4ALL_PKG_ROOT / "bench" / "AOM_v0" / "publication" / "tables" / "master_pivot.csv"
AOMPLS_PER_DS_CSV   = NIRS4ALL_PKG_ROOT / "bench" / "AOM_v0" / "publication" / "tables" / "summary_per_dataset.csv"
AOM_RIDGE_DIRS      = [
    NIRS4ALL_PKG_ROOT / "bench" / "AOM_v0" / "Ridge" / "benchmark_runs" / "curated_v2",
    NIRS4ALL_PKG_ROOT / "bench" / "AOM_v0" / "Ridge" / "benchmark_runs" / "curated",
]


def load_paper_baselines() -> pd.DataFrame:
    df = pd.read_csv(PAPER_PIVOT_CSV)
    long = df.melt(id_vars=["database_name", "dataset"],
                   value_vars=["CNN", "Catboost", "PLS", "Ridge", "TabPFN-Raw", "TabPFN-opt"],
                   var_name="model_class", value_name="rmsep")
    long["source"] = "paper (TabPFN bench master_pivot)"
    long["model_class"] = "paper " + long["model_class"]
    return long.dropna(subset=["rmsep"])


def load_aompls_best() -> pd.DataFrame:
    df = pd.read_csv(AOMPLS_PER_DS_CSV)
    variant_cols = [c for c in df.columns if c not in ("database_name", "dataset")]
    pls_baseline = df[["database_name", "dataset", "PLS-standard-numpy"]].rename(
        columns={"PLS-standard-numpy": "rmsep"}).dropna(subset=["rmsep"])
    pls_baseline["model_class"] = "AOM-PLS PLS-standard"
    pls_baseline["source"] = "AOM_v0 publication"

    aom_only = [c for c in variant_cols if c not in ("PLS-standard-numpy",)]
    df["AOM-best-rmsep"] = df[aom_only].min(axis=1, skipna=True)
    df["AOM-best-variant"] = df[aom_only].idxmin(axis=1, skipna=True)
    aom_best = df[["database_name", "dataset", "AOM-best-rmsep"]].rename(
        columns={"AOM-best-rmsep": "rmsep"}).dropna(subset=["rmsep"])
    aom_best["model_class"] = "AOM-PLS-best"
    aom_best["source"] = "AOM_v0 publication"
    return pd.concat([pls_baseline, aom_best], ignore_index=True)


def load_aom_ridge_best() -> pd.DataFrame:
    rows: list[dict] = []
    for d in AOM_RIDGE_DIRS:
        csv = d / "results.csv"
        if not csv.is_file():
            continue
        sub = pd.read_csv(csv)
        sub = sub[sub["status"].astype(str).str.lower() == "ok"]
        if sub.empty:
            continue
        # Best per dataset across all AOM-Ridge variants.
        for ds, g in sub.groupby("dataset"):
            best = g.loc[g["rmsep"].idxmin()]
            rows.append({
                "database_name": best.get("dataset_group", ""),
                "dataset": ds,
                "rmsep": float(best["rmsep"]),
                "model_class": "AOM-Ridge-best",
                "source": f"AOM_v0/Ridge/benchmark_runs/{d.name}",
            })
        break  # take the first available curated dir
    return pd.DataFrame(rows)


def load_our_results(csv: Path, variant: str) -> pd.DataFrame:
    if not csv.is_file():
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df = df[df["status"].astype(str) == "OK"]
    base = df[df["variant"] == "Ridge-baseline"][["dataset_group", "dataset", "rmsep"]].copy()
    base["model_class"] = "nicon_v2 internal Ridge"
    base["source"] = f"nicon_v2/{csv.parent.name}"
    base = base.rename(columns={"dataset_group": "database_name"})
    base = base.groupby(["database_name", "dataset", "model_class", "source"], as_index=False)["rmsep"].median()

    pls = df[df["variant"] == "PLS-baseline"][["dataset_group", "dataset", "rmsep"]].copy()
    pls["model_class"] = "nicon_v2 internal PLS"
    pls["source"] = f"nicon_v2/{csv.parent.name}"
    pls = pls.rename(columns={"dataset_group": "database_name"})
    pls = pls.groupby(["database_name", "dataset", "model_class", "source"], as_index=False)["rmsep"].median()

    cnn = df[df["variant"] == "NiconV1c-concat-bjerrum"][["dataset_group", "dataset", "rmsep"]].copy()
    cnn["model_class"] = "nicon_v2 V1c-concat-bjerrum (CNN-only)"
    cnn["source"] = f"nicon_v2/{csv.parent.name}"
    cnn = cnn.rename(columns={"dataset_group": "database_name"})
    cnn = cnn.groupby(["database_name", "dataset", "model_class", "source"], as_index=False)["rmsep"].median()

    target = df[df["variant"] == variant][["dataset_group", "dataset", "rmsep"]].copy()
    target["model_class"] = f"nicon_v2 {variant}"
    target["source"] = f"nicon_v2/{csv.parent.name}"
    target = target.rename(columns={"dataset_group": "database_name"})
    target = target.groupby(["database_name", "dataset", "model_class", "source"], as_index=False)["rmsep"].median()
    return pd.concat([base, pls, cnn, target], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--our-csv", type=Path, required=True)
    parser.add_argument("--our-variant", type=str, default="Stack-Ridge-PLS-V1c")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cnn-variant", type=str, default=None,
                        help="Optional second nicon_v2 variant column (e.g. a new pure-CNN model).")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    paper = load_paper_baselines()
    aompls = load_aompls_best()
    aomridge = load_aom_ridge_best()
    ours = load_our_results(args.our_csv, args.our_variant)
    if args.cnn_variant:
        cnn_extra = load_our_results(args.our_csv, args.cnn_variant)
        cnn_extra = cnn_extra[cnn_extra["model_class"] == f"nicon_v2 {args.cnn_variant}"]
        ours = pd.concat([ours, cnn_extra], ignore_index=True)

    long = pd.concat([paper, aompls, aomridge, ours], ignore_index=True)
    long["rmsep"] = long["rmsep"].astype(float)
    long.to_csv(args.out / "long_per_dataset.csv", index=False)
    print(f"wrote {args.out / 'long_per_dataset.csv'} ({len(long)} rows)")

    # Wide pivot.
    wide = long.pivot_table(index=["database_name", "dataset"], columns="model_class",
                              values="rmsep", aggfunc="first").reset_index()
    wide.to_csv(args.out / "wide_per_dataset.csv", index=False)
    print(f"wrote {args.out / 'wide_per_dataset.csv'} ({len(wide)} rows × {len(wide.columns)} cols)")
    print()
    print("=== columns in wide table ===")
    for c in wide.columns:
        non_na = wide[c].notna().sum()
        print(f"  {c:<55} {non_na} datasets")

    # Win-rate table — per (model A, model B), how often A < B.
    cols = [c for c in wide.columns if c not in ("database_name", "dataset")]
    print()
    print("=== Win-rates: model wins (lower rmsep) over each opponent ===")
    print(f"  {'model A':<45} {'opponent (B)':<35} {'A<B':>5} / {'#valid':>6}  {'median(A/B)':>12}")
    target_col = f"nicon_v2 {args.our_variant}"
    for opp in cols:
        if opp == target_col:
            continue
        valid = wide[[target_col, opp]].dropna()
        if valid.empty:
            continue
        wins = int((valid[target_col] < valid[opp]).sum())
        n = len(valid)
        med_ratio = float((valid[target_col] / valid[opp]).median())
        print(f"  {target_col:<45} {opp:<35} {wins:>5d} / {n:>6d}  {med_ratio:>12.4f}")

    # Save win-rate table
    win_rows = []
    for opp in cols:
        if opp == target_col:
            continue
        valid = wide[[target_col, opp]].dropna()
        if valid.empty:
            continue
        wins = int((valid[target_col] < valid[opp]).sum())
        n = len(valid)
        med_ratio = float((valid[target_col] / valid[opp]).median())
        win_rows.append({
            "our_variant": args.our_variant,
            "opponent": opp,
            "wins": wins,
            "n_valid": n,
            "win_rate": wins / n if n else float("nan"),
            "median_rmsep_ratio": med_ratio,
            "median_pct_delta": (med_ratio - 1.0) * 100,
        })
    win_df = pd.DataFrame(win_rows).sort_values("median_pct_delta")
    win_df.to_csv(args.out / "winrate_summary.csv", index=False)
    print(f"\nwrote {args.out / 'winrate_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
