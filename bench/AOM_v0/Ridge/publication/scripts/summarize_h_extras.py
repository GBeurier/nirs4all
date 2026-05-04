"""Quick summariser of the diverse_iter3_hextras and diverse_iter3_headline
benchmark runs.

Reads the two results CSVs (h_extras + headline) plus the iter2 baseline
and prints a table per variant (median delta, mean capped delta,
win-rate vs paper Ridge HPO, worst-case delta) and a per-dataset best
table so we can see which extension wins where.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_HEXTRAS = Path(
    "bench/AOM_v0/Ridge/benchmark_runs/diverse_iter3_hextras/results.csv"
)
DEFAULT_HEADLINE = Path(
    "bench/AOM_v0/Ridge/benchmark_runs/diverse_iter3_headline/results.csv"
)
DEFAULT_ITER2 = Path(
    "bench/AOM_v0/Ridge/benchmark_runs/diverse_iter2/results.csv"
)


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["rmsep", "ref_rmse_ridge"])
    df["delta_pct"] = 100.0 * (df["rmsep"] - df["ref_rmse_ridge"]) / df["ref_rmse_ridge"]
    df["win"] = (df["rmsep"] < df["ref_rmse_ridge"]).astype(int)
    return df


def _summary(df: pd.DataFrame, cap: float = 200.0) -> pd.DataFrame:
    df = df.copy()
    df["delta_pct_capped"] = df["delta_pct"].clip(-cap, cap)
    grp = df.groupby("variant").agg(
        median=("delta_pct", "median"),
        mean_capped=("delta_pct_capped", "mean"),
        worst=("delta_pct", "max"),
        wins=("win", "sum"),
        n=("win", "count"),
    )
    grp["win_rate_pct"] = 100.0 * grp["wins"] / grp["n"]
    grp = grp.sort_values("median")
    return grp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hextras", type=Path, default=DEFAULT_HEXTRAS)
    parser.add_argument("--headline", type=Path, default=DEFAULT_HEADLINE)
    parser.add_argument("--iter2", type=Path, default=DEFAULT_ITER2)
    args = parser.parse_args(argv)

    parts = []
    for label, path in [("iter2", args.iter2), ("hextras", args.hextras), ("headline", args.headline)]:
        df = _load(path)
        if df is None:
            print(f"  [{label}] not found: {path}")
            continue
        df["bench"] = label
        parts.append(df)
        print(f"  [{label}] {len(df)} rows / {df['variant'].nunique()} variants / {df['dataset'].nunique()} datasets")

    if not parts:
        print("No data loaded.")
        return 1

    big = pd.concat(parts, ignore_index=True)
    big_dedup = big.sort_values("bench").drop_duplicates(["dataset", "variant"], keep="last")
    print()
    print("=== Per-variant median delta (sorted) ===")
    print(_summary(big_dedup).round(2).to_string())

    print()
    print("=== Per-dataset winner (lowest RMSEP) ===")
    pivot = big_dedup.pivot_table(index="dataset", columns="variant", values="rmsep", aggfunc="min")
    winners = pivot.idxmin(axis=1)
    rmsep_win = pivot.min(axis=1)
    paper_rmse = big_dedup.groupby("dataset")["ref_rmse_ridge"].first()
    delta = 100.0 * (rmsep_win - paper_rmse) / paper_rmse
    table = pd.DataFrame({"winner": winners, "rmsep": rmsep_win, "paper_ridge": paper_rmse, "delta_pct": delta})
    print(table.round(4).to_string())

    print()
    print("=== Oracle envelope vs paper Ridge HPO ===")
    n = len(table)
    wins = (table["delta_pct"] < 0).sum()
    print(f"  Win rate: {wins}/{n} = {100.0*wins/n:.1f}%")
    print(f"  Median delta: {table['delta_pct'].median():+.2f}%")
    print(f"  Mean delta (capped at 200): {table['delta_pct'].clip(-200, 200).mean():+.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
