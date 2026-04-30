"""Render LaTeX tables for the AOM-Ridge paper.

Outputs (all written under ``../tables`` as ``.tex``):

- ``table_per_dataset_results.tex``
    Full per-dataset RMSEP for the best AOM-Ridge variant alongside the
    six TabPFN paper baselines (Ridge, PLS, CNN, Catboost, TabPFN-Raw,
    TabPFN-opt). Includes ``n_train``, ``n_test``, ``p`` (read from the
    cohort metadata) and a ``win`` indicator vs paper Ridge HPO.
- ``table_summary.tex``
    Cohort-level summary statistics: median delta vs paper Ridge,
    capped mean delta, and win-rate of AOM-Ridge against each paper
    baseline.
- ``table_per_method_summary.tex``
    Per AOM-Ridge variant: number of wins (vs paper Ridge), mean rank
    across datasets (lower is better), median delta vs paper Ridge.

Usage::

    python bench/AOM_v0/Ridge/publication/scripts/make_aomridge_tables.py

CLI::

    --results        path to the AOM-Ridge results CSV (default:
                     curated_v2 if present, else curated).
    --master         path to the TabPFN master pivot CSV.
    --cohort         path to the cohort metadata CSV (n_train, n_test, p).
    --out            output directory (default: ../tables).

This script never modifies the production library or the AOM-Ridge
package; it is a read-only consumer of the curated benchmark CSV.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PUB_ROOT = HERE.parent
RIDGE_ROOT = PUB_ROOT.parent  # bench/AOM_v0/Ridge
AOM_ROOT = RIDGE_ROOT.parent  # bench/AOM_v0

DEFAULT_OUT = PUB_ROOT / "tables"
DEFAULT_RESULTS_V2 = RIDGE_ROOT / "benchmark_runs" / "curated_v2" / "results.csv"
DEFAULT_RESULTS_V1 = RIDGE_ROOT / "benchmark_runs" / "curated" / "results.csv"
DEFAULT_MASTER = AOM_ROOT / "publication" / "tables" / "master_pivot.csv"
DEFAULT_COHORT = AOM_ROOT / "benchmarks" / "cohort_regression.csv"

# Datasets to drop from analysis (degenerate references).
EXCLUDED_DATASETS = {"QUARTZ"}

PAPER_BASELINES = ["Ridge", "PLS", "CNN", "Catboost", "TabPFN-Raw", "TabPFN-opt"]
DELTA_CAP_PCT = 200.0  # cap relative deltas at +/-200% before averaging


# ---------------------------------------------------------------------------
# IO + helpers
# ---------------------------------------------------------------------------

def _resolve_results_path(arg: Path | None) -> Path:
    if arg is not None:
        return Path(arg)
    if DEFAULT_RESULTS_V2.exists():
        return DEFAULT_RESULTS_V2
    return DEFAULT_RESULTS_V1


def _load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["rmsep"] = pd.to_numeric(df["rmsep"], errors="coerce")
    df = df.dropna(subset=["rmsep"])
    df = df[~df["dataset_group"].isin(EXCLUDED_DATASETS)].copy()
    return df


def _aomridge_best_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Best AOM-Ridge variant per dataset (argmin relative_rmsep_vs_paper_ridge)."""
    is_aomridge = df["variant"].str.startswith("AOMRidge-")
    aom = df[is_aomridge].copy()
    if aom.empty:
        return aom
    rel = pd.to_numeric(aom.get("relative_rmsep_vs_paper_ridge"), errors="coerce")
    aom = aom.assign(_rel=rel.fillna(np.inf))
    aom = aom.sort_values(
        ["dataset_group", "dataset", "_rel", "rmsep", "fit_time_s", "variant"]
    )
    best = aom.groupby(["dataset_group", "dataset"], as_index=False).first()
    return best.drop(columns=["_rel"])


def _escape_latex(s: object) -> str:
    s = str(s)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def _fmt(x: float, fmt: str = "{:.4f}") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "---"
    return fmt.format(x)


def _capped_mean(deltas_pct: pd.Series, cap: float = DELTA_CAP_PCT) -> float:
    arr = pd.to_numeric(deltas_pct, errors="coerce").dropna().values
    if arr.size == 0:
        return float("nan")
    arr = np.clip(arr, -cap, cap)
    return float(arr.mean())


# ---------------------------------------------------------------------------
# Table 1 - per-dataset results
# ---------------------------------------------------------------------------

def render_table_per_dataset_results(
    df: pd.DataFrame, master_path: Path, cohort_path: Path, out: Path,
) -> None:
    best = _aomridge_best_per_dataset(df)
    if best.empty:
        out.write_text(
            "% No AOM-Ridge results available.\n"
            "\\begin{tabular}{l}\\toprule (no data) \\\\\\bottomrule\\end{tabular}\n"
        )
        return
    if not master_path.exists():
        out.write_text(
            f"% master_pivot.csv not found at {master_path}\n"
            "\\begin{tabular}{l}\\toprule (master missing) \\\\\\bottomrule\\end{tabular}\n"
        )
        return
    master = pd.read_csv(master_path)
    master = master[~master["database_name"].isin(EXCLUDED_DATASETS)]
    cohort = None
    if cohort_path.exists():
        cohort = pd.read_csv(cohort_path)
        cohort = cohort[~cohort["database_name"].isin(EXCLUDED_DATASETS)]

    # Build the joined table with best AOM-Ridge + paper baselines.
    aom_part = best.rename(columns={
        "dataset_group": "database_name",
        "rmsep": "AOM-Ridge",
        "variant": "AOM-Ridge variant",
    })[["database_name", "dataset", "AOM-Ridge", "AOM-Ridge variant"]]

    merged = aom_part.merge(
        master[["database_name", "dataset"] + [c for c in PAPER_BASELINES if c in master.columns]],
        on=["database_name", "dataset"], how="left",
    )
    if cohort is not None:
        merged = merged.merge(
            cohort[["database_name", "dataset", "n_train", "n_test", "p"]],
            on=["database_name", "dataset"], how="left",
        )
    else:
        for col in ("n_train", "n_test", "p"):
            merged[col] = np.nan

    merged["delta_pct"] = (
        100.0 * (merged["AOM-Ridge"] - merged["Ridge"]) / merged["Ridge"]
    )
    merged["win_vs_ridge"] = (merged["AOM-Ridge"] < merged["Ridge"]).astype(int)
    merged = merged.sort_values(["database_name", "dataset"])

    # Build LaTeX table. Columns:
    #  database, dataset, n_tr, n_te, p, AOM-Ridge, Ridge, PLS, CNN, Catboost, TabPFN-Raw, TabPFN-opt, win
    cols_baselines = [c for c in PAPER_BASELINES if c in merged.columns]
    n_cols = 5 + 1 + len(cols_baselines) + 1
    align = "l" * 2 + "r" * 3 + "r" + "r" * len(cols_baselines) + "c"
    assert len(align) == n_cols

    lines: list[str] = []
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\toprule")
    header = [
        "database", "dataset", "$n_\\text{tr}$", "$n_\\text{te}$", "$p$",
        "AOM-Ridge",
    ] + cols_baselines + ["win"]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for _, row in merged.iterrows():
        cells: list[str] = [
            _escape_latex(row["database_name"]),
            _escape_latex(row["dataset"]),
        ]
        for c in ("n_train", "n_test", "p"):
            v = row.get(c)
            if pd.isna(v):
                cells.append("---")
            else:
                cells.append(f"{int(v)}")
        cells.append(_fmt(row.get("AOM-Ridge")))
        for c in cols_baselines:
            cells.append(_fmt(row.get(c)))
        cells.append("$\\checkmark$" if row["win_vs_ridge"] == 1 else "")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\n% AOM-Ridge column is the best variant per dataset (argmin RMSEP)."
                 " Win = AOM-Ridge RMSEP strictly below paper Ridge HPO RMSEP.\n")
    out.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 2 - cohort summary
# ---------------------------------------------------------------------------

def render_table_summary(df: pd.DataFrame, master_path: Path, out: Path) -> None:
    best = _aomridge_best_per_dataset(df)
    if best.empty or not master_path.exists():
        out.write_text(
            "% Missing inputs for summary table.\n"
            "\\begin{tabular}{l}\\toprule (no data) \\\\\\bottomrule\\end{tabular}\n"
        )
        return
    master = pd.read_csv(master_path)
    master = master[~master["database_name"].isin(EXCLUDED_DATASETS)]

    aom_part = best.rename(columns={
        "dataset_group": "database_name", "rmsep": "AOM-Ridge",
    })[["database_name", "dataset", "AOM-Ridge"]]
    merged = aom_part.merge(master, on=["database_name", "dataset"], how="left")

    rows: list[list[str]] = []
    n_total = len(merged)
    for baseline in PAPER_BASELINES:
        if baseline not in merged.columns:
            continue
        sub = merged[["AOM-Ridge", baseline]].dropna()
        if sub.empty:
            rows.append([baseline, "---", "---", "---", "0/0"])
            continue
        delta_pct = 100.0 * (sub["AOM-Ridge"] - sub[baseline]) / sub[baseline]
        median_delta = float(np.median(delta_pct))
        capped_mean = _capped_mean(delta_pct)
        wins = int((sub["AOM-Ridge"] < sub[baseline]).sum())
        n = int(len(sub))
        win_rate = 100.0 * wins / n
        rows.append([
            baseline,
            f"{median_delta:+.2f}",
            f"{capped_mean:+.2f}",
            f"{win_rate:.1f}",
            f"{wins}/{n}",
        ])

    lines: list[str] = []
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Baseline & median $\\Delta$ (\\%) & mean $\\Delta$ (capped, \\%) "
        "& win-rate (\\%) & wins / N \\\\"
    )
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        f"\n% Cohort: N={n_total} datasets after excluding "
        f"{sorted(EXCLUDED_DATASETS)}. "
        f"$\\Delta$ = (AOM-Ridge $-$ baseline) / baseline. "
        f"Mean delta is capped at +/-{int(DELTA_CAP_PCT)}\\%.\n"
    )
    out.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 3 - per AOM-Ridge variant summary
# ---------------------------------------------------------------------------

def render_table_per_method_summary(
    df: pd.DataFrame, master_path: Path, out: Path,
) -> None:
    is_aomridge = df["variant"].str.startswith("AOMRidge-")
    aom = df[is_aomridge].copy()
    if aom.empty or not master_path.exists():
        out.write_text(
            "% Missing inputs for per-method summary.\n"
            "\\begin{tabular}{l}\\toprule (no data) \\\\\\bottomrule\\end{tabular}\n"
        )
        return
    master = pd.read_csv(master_path)
    master = master[~master["database_name"].isin(EXCLUDED_DATASETS)]

    # Build wide table dataset x AOM-Ridge variant of RMSEP.
    aom_wide = aom.pivot_table(
        index=["dataset_group", "dataset"],
        columns="variant",
        values="rmsep",
        aggfunc="mean",
    )
    # Ranks per dataset (lower RMSEP = lower rank).
    ranks = aom_wide.rank(axis=1, method="average")
    mean_rank = ranks.mean(axis=0)
    n_datasets = aom_wide.shape[0]

    # Win counts vs paper Ridge.
    aom_long = aom_wide.reset_index().melt(
        id_vars=["dataset_group", "dataset"], var_name="variant", value_name="rmsep",
    ).dropna(subset=["rmsep"])
    aom_long = aom_long.merge(
        master[["database_name", "dataset", "Ridge"]],
        left_on=["dataset_group", "dataset"],
        right_on=["database_name", "dataset"],
        how="left",
    )
    aom_long["delta_pct"] = (
        100.0 * (aom_long["rmsep"] - aom_long["Ridge"]) / aom_long["Ridge"]
    )
    aom_long["win"] = (aom_long["rmsep"] < aom_long["Ridge"]).astype(int)
    grouped = aom_long.groupby("variant").agg(
        wins=("win", "sum"),
        n=("rmsep", "count"),
        median_delta_pct=("delta_pct", "median"),
    )
    grouped["mean_rank"] = grouped.index.map(mean_rank.to_dict())
    grouped = grouped.sort_values("mean_rank")

    lines: list[str] = []
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append(
        "AOM-Ridge variant & wins / N & win-rate (\\%) "
        "& mean rank & median $\\Delta$ vs Ridge (\\%) \\\\"
    )
    lines.append("\\midrule")
    for variant, row in grouped.iterrows():
        wins = int(row["wins"])
        n = int(row["n"])
        win_rate = 100.0 * wins / n if n else float("nan")
        mr = row["mean_rank"]
        md = row["median_delta_pct"]
        lines.append(
            f"{_escape_latex(variant)} & "
            f"{wins}/{n} & "
            f"{_fmt(win_rate, '{:.1f}')} & "
            f"{_fmt(mr, '{:.2f}')} & "
            f"{_fmt(md, '{:+.2f}')} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        f"\n% Mean rank computed across {n_datasets} datasets "
        f"(complete-case ranking per dataset). Median delta is per-variant.\n"
    )
    out.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render AOM-Ridge LaTeX tables.")
    parser.add_argument("--results", type=Path, default=None,
                        help="Path to AOM-Ridge results CSV.")
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    results_path = _resolve_results_path(args.results)
    master_path = Path(args.master)
    cohort_path = Path(args.cohort)
    out: Path = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[make_aomridge_tables] results: {results_path}")
    print(f"[make_aomridge_tables] master:  {master_path}")
    print(f"[make_aomridge_tables] cohort:  {cohort_path}")
    print(f"[make_aomridge_tables] out:     {out}")

    if not results_path.exists():
        print(f"ERROR: results CSV not found: {results_path}", file=sys.stderr)
        return 2
    df = _load_results(results_path)
    print(f"[make_aomridge_tables] loaded {len(df)} rows over "
          f"{df['dataset'].nunique()} datasets and {df['variant'].nunique()} variants")

    artefacts: list[Path] = []
    artefacts.append(out / "table_per_dataset_results.tex")
    render_table_per_dataset_results(df, master_path, cohort_path, artefacts[-1])
    print("  - table_per_dataset_results.tex")

    artefacts.append(out / "table_summary.tex")
    render_table_summary(df, master_path, artefacts[-1])
    print("  - table_summary.tex")

    artefacts.append(out / "table_per_method_summary.tex")
    render_table_per_method_summary(df, master_path, artefacts[-1])
    print("  - table_per_method_summary.tex")

    print("\nGenerated artefacts:")
    for p in artefacts:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
