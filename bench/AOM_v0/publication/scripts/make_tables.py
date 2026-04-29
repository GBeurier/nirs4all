"""Render LaTeX tables for the AOM_v0 paper.

Outputs (all written into ``../tables/``):

- ``table_variants.tex``               : variant matrix
                                          (selection x engine x backend
                                          x task), built from the
                                          constants in
                                          ``benchmarks/run_aompls_benchmark.py``.
- ``table_regression_main.tex``        : already produced by
                                          ``summarize_results.py``;
                                          this script regenerates it
                                          when ``summary_per_variant.csv``
                                          is present.
- ``table_classification_main.tex``    : ditto for classification.
- ``table_ablation.tex``               : ablation skeleton (placeholder
                                          rows that are populated when
                                          full ablation runs are
                                          available).
- ``table_operator_bank.tex``          : compact / default / extended
                                          bank composition.
- ``summary_per_variant.csv.tex``      : LaTeX-friendly inline copy of
                                          the per-variant smoke summary.

The script reads the variant constants directly from the runner so the
manuscript can never disagree with the benchmark script.

Usage::

    python bench/AOM_v0/publication/scripts/make_tables.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

HERE = Path(__file__).resolve().parent
PUB_ROOT = HERE.parent
ROOT = PUB_ROOT.parent  # bench/AOM_v0
DEFAULT_OUT = PUB_ROOT / "tables"
DEFAULT_BENCH = ROOT / "benchmarks" / "run_aompls_benchmark.py"

sys.path.insert(0, str(ROOT))

from aompls.banks import (  # noqa: E402
    compact_bank,
    default_bank,
    extended_bank,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_module_from_path(path: Path, name: str = "_run_aompls_benchmark"):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _escape_latex(s: str) -> str:
    return (
        s.replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\~")
        .replace("^", "\\^")
    )


# ---------------------------------------------------------------------------
# Variant matrix
# ---------------------------------------------------------------------------


def _variant_rows() -> List[Dict[str, str]]:
    bench_module = _load_module_from_path(DEFAULT_BENCH)
    rows: List[Dict[str, str]] = []
    for v in bench_module.REGRESSION_VARIANTS:
        rows.append({
            "label": v["label"],
            "task": "regression",
            "selection": v["selection"],
            "engine": v["engine"],
            "operator_bank": v["operator_bank"],
            "backend": v["backend"],
            "experimental": "yes" if v.get("experimental") else "",
        })
    for v in bench_module.CLASSIFICATION_VARIANTS:
        rows.append({
            "label": v["label"],
            "task": "classification",
            "selection": v["selection"],
            "engine": v["engine"],
            "operator_bank": v["operator_bank"],
            "backend": v["backend"],
            "experimental": "",
        })
    return rows


def render_table_variants(out: Path) -> None:
    rows = _variant_rows()
    lines = []
    lines.append("\\begin{tabular}{llllll}")
    lines.append("\\toprule")
    lines.append("Label & Task & Selection & Engine & Bank & Backend \\\\")
    lines.append("\\midrule")
    for r in rows:
        cells = [
            _escape_latex(r["label"]),
            _escape_latex(r["task"]),
            _escape_latex(r["selection"]),
            _escape_latex(r["engine"]),
            _escape_latex(r["operator_bank"]),
            _escape_latex(r["backend"]),
        ]
        suffix = " (exp.)" if r["experimental"] else ""
        cells[0] = cells[0] + suffix
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Operator bank table
# ---------------------------------------------------------------------------


def render_table_operator_bank(out: Path) -> None:
    banks = {
        "compact": compact_bank(),
        "default": default_bank(),
        "extended": extended_bank(),
    }
    names = {k: [op.name for op in v] for k, v in banks.items()}
    max_len = max(len(v) for v in names.values())
    for k in names:
        names[k] = names[k] + [""] * (max_len - len(names[k]))
    lines = []
    lines.append("\\begin{tabular}{rlll}")
    lines.append("\\toprule")
    lines.append("\\# & compact ($|B|=" + str(len(banks['compact'])) + "$)"
                  " & default ($|B|=" + str(len(banks['default'])) + "$)"
                  " & extended ($|B|=" + str(len(banks['extended'])) + "$) \\\\")
    lines.append("\\midrule")
    for i in range(max_len):
        cells = [
            str(i + 1),
            _escape_latex(names["compact"][i] or ""),
            _escape_latex(names["default"][i] or ""),
            _escape_latex(names["extended"][i] or ""),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Regression main / classification main from summarize_results CSVs
# ---------------------------------------------------------------------------


def render_summary_table(csv_path: Path, out: Path,
                          metric_name: str, ascending: bool) -> None:
    if not csv_path.exists():
        out.write_text(
            "% summary CSV not found: " + str(csv_path) + "\n"
            "\\begin{tabular}{lrr}\n\\toprule\n"
            f"AOM variant & mean {metric_name} & median {metric_name} \\\\\n"
            "\\midrule\n"
            "(no rows yet) & --- & --- \\\\\n"
            "\\bottomrule\n\\end{tabular}\n"
        )
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        out.write_text(
            "\\begin{tabular}{lrr}\n\\toprule\n"
            f"AOM variant & mean {metric_name} & median {metric_name} \\\\\n"
            "\\midrule\n"
            "(no rows yet) & --- & --- \\\\\n"
            "\\bottomrule\n\\end{tabular}\n"
        )
        return
    if "rmsep_mean" in df.columns:
        df = df.sort_values("rmsep_mean", ascending=ascending)
        mean_col, med_col = "rmsep_mean", "rmsep_median"
    else:
        df = df.sort_values("balanced_acc_mean", ascending=not ascending)
        mean_col, med_col = "balanced_acc_mean", "balanced_acc_median"
    lines = []
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\toprule")
    lines.append(f"AOM variant & mean {metric_name} & median {metric_name} \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        lines.append(
            f"{_escape_latex(str(row['aom_variant']))} & "
            f"{row[mean_col]:.4f} & {row[med_col]:.4f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Ablation skeleton
# ---------------------------------------------------------------------------

ABLATION_ROWS: List[List[str]] = [
    ["Bank size", "compact (9)", "default (13)", "extended (18)"],
    ["Criterion", "covariance", "CV", "approx PRESS"],
    ["Orthogonalisation", "transformed", "original", "auto"],
    ["Engine", "NIPALS adjoint", "SIMPLS covariance", "SIMPLS materialized"],
]


def render_table_ablation(out: Path) -> None:
    lines = []
    lines.append("\\begin{tabular}{llll}")
    lines.append("\\toprule")
    lines.append("Factor & Variant 1 & Variant 2 & Variant 3 \\\\")
    lines.append("\\midrule")
    for row in ABLATION_ROWS:
        cells = [_escape_latex(c) for c in row]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\n% Numerical cells filled in by the ablation runner once the\n"
                  "% full benchmark is available; the present rendering provides\n"
                  "% the factor x level skeleton referenced in Section 10.\n")
    out.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Inline copy of summary_per_variant.csv into LaTeX
# ---------------------------------------------------------------------------


def render_inline_summary(csv_path: Path, out: Path) -> None:
    if not csv_path.exists():
        out.write_text(
            "\\begin{tabular}{lrr}\n\\toprule\n"
            "AOM variant & mean RMSEP & median RMSEP \\\\\n\\midrule\n"
            "(smoke summary not generated) & --- & --- \\\\\n"
            "\\bottomrule\n\\end{tabular}\n"
        )
        return
    df = pd.read_csv(csv_path)
    if df.empty or "rmsep_mean" not in df.columns:
        out.write_text(
            "\\begin{tabular}{lrr}\n\\toprule\n"
            "AOM variant & mean RMSEP & median RMSEP \\\\\n\\midrule\n"
            "(smoke summary empty) & --- & --- \\\\\n"
            "\\bottomrule\n\\end{tabular}\n"
        )
        return
    df = df.sort_values("rmsep_mean")
    lines = []
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\toprule")
    lines.append("AOM variant & mean RMSEP (smoke) & median RMSEP (smoke) \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        lines.append(
            f"{_escape_latex(str(row['aom_variant']))} & "
            f"{row['rmsep_mean']:.4f} & {row['rmsep_median']:.4f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render AOM_v0 paper tables.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-reg", type=Path,
                         default=DEFAULT_OUT / "summary_per_variant.csv")
    parser.add_argument("--summary-cla", type=Path,
                         default=DEFAULT_OUT / "summary_classification_per_variant.csv")
    args = parser.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"[make_tables] output dir: {out}")

    render_table_variants(out / "table_variants.tex")
    print("  - table_variants.tex")

    render_table_operator_bank(out / "table_operator_bank.tex")
    print("  - table_operator_bank.tex")

    render_summary_table(args.summary_reg, out / "table_regression_main.tex",
                          metric_name="RMSEP", ascending=True)
    print("  - table_regression_main.tex")

    render_summary_table(args.summary_cla, out / "table_classification_main.tex",
                          metric_name="bal.acc.", ascending=True)
    print("  - table_classification_main.tex")

    render_table_ablation(out / "table_ablation.tex")
    print("  - table_ablation.tex")

    render_inline_summary(args.summary_reg, out / "summary_per_variant.csv.tex")
    print("  - summary_per_variant.csv.tex")

    return 0


if __name__ == "__main__":
    sys.exit(main())
