"""Score table for the 11-dataset representative cohort.

Joins our V2A/V2B/Stack-V1c results with paper baselines (CNN/CatBoost/PLS/
Ridge/TabPFN-Raw/TabPFN-opt) + AOM-PLS-best + AOM-Ridge-best, side by side
per dataset, with bold-mark of the winner. Designed for the manuscript's
representative-cohort table and for quick interactive use during iteration.

Usage::

    python publication/scripts/representative_table.py \
      --our-csv bench/nicon_v2/benchmark_runs/v2a_rep/results.csv \
      --variants V2B-extended-trainable V2A-compact-frozen Stack-Ridge-PLS-V1c \
      --out bench/nicon_v2/publication/tables/representative/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NIRS4ALL_PKG_ROOT = Path("/home/delete/nirs4all/nirs4all")
PAPER_PIVOT_CSV = NIRS4ALL_PKG_ROOT / "bench" / "AOM_v0" / "publication" / "tables" / "master_pivot.csv"
AOMPLS_PER_DS_CSV = NIRS4ALL_PKG_ROOT / "bench" / "AOM_v0" / "publication" / "tables" / "summary_per_dataset.csv"
AOM_RIDGE_DIRS = [
    NIRS4ALL_PKG_ROOT / "bench" / "AOM_v0" / "Ridge" / "benchmark_runs" / "curated_v2",
    NIRS4ALL_PKG_ROOT / "bench" / "AOM_v0" / "Ridge" / "benchmark_runs" / "curated",
]


def load_paper_baselines(datasets: list[str]) -> pd.DataFrame:
    df = pd.read_csv(PAPER_PIVOT_CSV)
    df = df[df["dataset"].isin(datasets)].copy()
    return df.rename(columns={
        "CNN": "paper Nicon",
        "Catboost": "paper CatBoost",
        "PLS": "paper PLS",
        "Ridge": "paper Ridge",
        "TabPFN-Raw": "paper TabPFN-raw",
        "TabPFN-opt": "paper TabPFN-opt",
    })


def load_aompls_best(datasets: list[str]) -> pd.DataFrame:
    df = pd.read_csv(AOMPLS_PER_DS_CSV)
    df = df[df["dataset"].isin(datasets)].copy()
    variant_cols = [c for c in df.columns if c not in ("database_name", "dataset")]
    df["AOM-PLS-best"] = df[variant_cols].min(axis=1, skipna=True)
    return df[["dataset", "AOM-PLS-best"]]


def load_aom_ridge_best(datasets: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for d in AOM_RIDGE_DIRS:
        csv = d / "results.csv"
        if not csv.is_file():
            continue
        sub = pd.read_csv(csv)
        sub = sub[sub["status"].astype(str).str.lower() == "ok"]
        for ds, g in sub.groupby("dataset"):
            if ds in datasets:
                rows.append({"dataset": ds, "AOM-Ridge-best": float(g["rmsep"].min())})
    return pd.DataFrame(rows).drop_duplicates(subset=["dataset"], keep="first")


def load_our_variants(csv: Path, variants: list[str]) -> pd.DataFrame:
    if not csv.is_file():
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df = df[df["status"].astype(str) == "OK"]
    out = pd.DataFrame({"dataset": df["dataset"].unique()}).set_index("dataset")
    for v in variants:
        sub = df[df["variant"] == v].groupby("dataset")["rmsep"].median()
        out[v] = sub
    return out.reset_index()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--our-csv", type=Path, required=True)
    parser.add_argument("--variants", type=str, nargs="+", default=["V2B-extended-trainable", "V2A-compact-frozen"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Load our results to know which datasets we have.
    if not args.our_csv.is_file():
        print(f"missing csv: {args.our_csv}")
        return 1
    df = pd.read_csv(args.our_csv)
    df = df[df["status"].astype(str) == "OK"]
    datasets = sorted(df["dataset"].unique())
    print(f"datasets in our run: {len(datasets)}")

    paper = load_paper_baselines(datasets)
    aompls = load_aompls_best(datasets)
    aomridge = load_aom_ridge_best(datasets)
    ours = load_our_variants(args.our_csv, args.variants)

    table = paper.copy()
    if not aompls.empty:
        table = table.merge(aompls, on="dataset", how="left")
    if not aomridge.empty:
        table = table.merge(aomridge, on="dataset", how="left")
    table = table.merge(ours, on="dataset", how="left")

    # Reorder columns: dataset + paper refs + AOM refs + our variants.
    paper_cols = ["paper Ridge", "paper PLS", "paper Nicon", "paper CatBoost", "paper TabPFN-raw", "paper TabPFN-opt"]
    aom_cols = ["AOM-PLS-best", "AOM-Ridge-best"]
    final_cols = ["dataset"] + paper_cols + aom_cols + args.variants
    final_cols = [c for c in final_cols if c in table.columns]
    table = table[final_cols]
    table.to_csv(args.out / "representative_scores.csv", index=False)
    print(f"wrote {args.out / 'representative_scores.csv'}")

    # Win-rate analysis: for each opponent, how many datasets does each of our variants win?
    opp_cols = paper_cols + aom_cols
    opp_cols = [c for c in opp_cols if c in table.columns]
    summary_rows = []
    for our in args.variants:
        for opp in opp_cols:
            valid = table[[our, opp]].dropna()
            if valid.empty:
                continue
            wins = int((valid[our] < valid[opp]).sum())
            n = len(valid)
            ratio = float((valid[our] / valid[opp]).median())
            summary_rows.append({
                "our_variant": our,
                "opponent": opp,
                "wins": wins,
                "n_valid": n,
                "win_rate": wins / n,
                "median_ratio": ratio,
                "median_pct": (ratio - 1.0) * 100,
            })
    summary = pd.DataFrame(summary_rows).sort_values(["our_variant", "median_pct"])
    summary.to_csv(args.out / "representative_winrate.csv", index=False)

    # Print headline.
    print()
    print("=== Per-dataset rmsep (lower is better) ===")
    pretty = table.copy()
    pretty[final_cols[1:]] = pretty[final_cols[1:]].round(4)
    print(pretty.to_string(index=False))
    print()
    print("=== Win-rate per (our variant, opponent) ===")
    for our in args.variants:
        sub = summary[summary["our_variant"] == our]
        print(f"\n>>> {our}")
        print(f"  {'opponent':<25} {'wins/n':>8} {'win_rate':>10} {'median_pct':>12}")
        for _, r in sub.iterrows():
            print(f"  {r['opponent']:<25} {f'{int(r['wins'])}/{int(r['n_valid'])}':>8} "
                  f"{r['win_rate']*100:>9.1f}% {r['median_pct']:>+11.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
