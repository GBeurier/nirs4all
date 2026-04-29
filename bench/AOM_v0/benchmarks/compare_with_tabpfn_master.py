"""Compare AOM_v0 results with the TabPFN paper master_results.csv.

For every dataset in our benchmark, joins the AOM_v0 RMSEP against the
TabPFN-Raw, TabPFN-opt, Catboost, PLS, and Ridge reference RMSEPs from
the published TabPFN paper. Produces:

- `tabpfn_comparison_per_dataset.csv` with per-dataset deltas.
- `tabpfn_comparison_per_variant.csv` with per-variant aggregate stats.
- `tabpfn_dominance_table.tex` with median win/tie/loss vs each TabPFN baseline.

Usage:

    PYTHONPATH=bench/AOM_v0 .venv/bin/python \
      bench/AOM_v0/benchmarks/compare_with_tabpfn_master.py \
      --results bench/AOM_v0/benchmark_runs/full/results.csv \
      --master bench/tabpfn_paper/master_results.csv \
      --out bench/AOM_v0/publication/tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_MODELS = ["TabPFN-Raw", "TabPFN-opt", "Catboost", "PLS", "Ridge", "CNN"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results)
    df = df[df["status"] == "ok"].copy()
    df["RMSEP"] = pd.to_numeric(df["RMSEP"], errors="coerce")
    df = df.dropna(subset=["RMSEP"])
    master = pd.read_csv(args.master)
    master["RMSEP"] = pd.to_numeric(master["RMSEP"], errors="coerce")
    master_pivot = master[master["model"].isin(REFERENCE_MODELS)].pivot_table(
        index=["database_name", "dataset"],
        columns="model",
        values="RMSEP",
        aggfunc="mean",
    ).reset_index()
    # Join AOM_v0 results with reference scores per (database_name, dataset).
    joined = df.merge(master_pivot, on=["database_name", "dataset"], how="left", suffixes=("", "_ref"))
    # Per-dataset delta vs each reference
    rows = []
    for variant in joined["aom_variant"].unique():
        sub = joined[joined["aom_variant"] == variant].copy()
        for ref in REFERENCE_MODELS:
            if ref in sub.columns:
                sub[f"delta_vs_{ref}"] = sub["RMSEP"] - sub[ref]
                sub[f"rel_vs_{ref}"] = sub["RMSEP"] / sub[ref]
        rows.append(sub)
    full = pd.concat(rows, ignore_index=True)
    full.to_csv(out / "tabpfn_comparison_per_dataset.csv", index=False)
    # Per-variant summary
    variant_rows = []
    for variant in df["aom_variant"].unique():
        sub = full[full["aom_variant"] == variant]
        d = {"aom_variant": variant, "n_datasets": int(sub["dataset"].nunique())}
        for ref in REFERENCE_MODELS:
            col = f"rel_vs_{ref}"
            if col in sub.columns and sub[col].notna().any():
                rel = sub[col].dropna()
                d[f"{ref}_n_pairs"] = int(len(rel))
                d[f"{ref}_median_ratio"] = float(rel.median())
                d[f"{ref}_mean_ratio"] = float(rel.mean())
                d[f"{ref}_wins"] = int((rel < 1.0).sum())
                d[f"{ref}_losses"] = int((rel > 1.0).sum())
        variant_rows.append(d)
    var_df = pd.DataFrame(variant_rows).sort_values("aom_variant")
    var_df.to_csv(out / "tabpfn_comparison_per_variant.csv", index=False)
    # LaTeX dominance table — for each variant, median ratio vs each TabPFN
    latex = ["\\begin{tabular}{l" + "rr" * len(REFERENCE_MODELS) + "}\n\\toprule"]
    header = "Variant" + "".join([f" & \\multicolumn{{2}}{{c}}{{{ref}}}" for ref in REFERENCE_MODELS])
    latex.append(header + " \\\\")
    sub_header = "" + "".join([" & median & wins" for _ in REFERENCE_MODELS])
    latex.append(sub_header + " \\\\")
    latex.append("\\midrule")
    for _, row in var_df.iterrows():
        cells = [row["aom_variant"]]
        for ref in REFERENCE_MODELS:
            med = row.get(f"{ref}_median_ratio", float("nan"))
            wins = row.get(f"{ref}_wins", "-")
            n = row.get(f"{ref}_n_pairs", 0)
            cells.append(f"{med:.3f}" if not pd.isna(med) else "--")
            cells.append(f"{int(wins)}/{int(n)}" if not pd.isna(wins) else "--")
        latex.append(" & ".join(str(c) for c in cells) + " \\\\")
    latex.append("\\bottomrule\n\\end{tabular}")
    (out / "tabpfn_dominance_table.tex").write_text("\n".join(latex))
    print(f"Wrote {out}/tabpfn_comparison_per_dataset.csv")
    print(f"Wrote {out}/tabpfn_comparison_per_variant.csv")
    print(f"Wrote {out}/tabpfn_dominance_table.tex")
    print(f"\nVariants summary:")
    print(var_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
