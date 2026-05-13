"""Selector, operator and failure diagnostics extractor for the AOM paper (P6).

This module ingests benchmark CSV outputs and produces inspectable per-row
diagnostics plus aggregated tables for the paper supplement:

  - operator frequency for the compact AOM-PLS bank;
  - selected component-count distribution;
  - compact bank vs default bank ratio (per-dataset paired);
  - strict-linear AOM (``AOM-compact-cv5-numpy``) vs ASLS-branch variant
    (``ASLS-AOM-compact-cv5-numpy``);
  - AutoSelect chosen-candidate counts and Blender weight mean/std across
    seeds/datasets;
  - seed stability for AutoSelect picks and Blender weight variance;
  - failure-mode table (QUARTZ, Brix, Tleaf, FinalScore, LMA, LUCAS_SOC_*,
    y-based / outlier splits) with absolute RMSEP and a ``ratio_meaningful``
    flag for denominators close to zero.

The CLI writes:

  - ``paper_aom/review/selector_diagnostics.csv``
  - ``paper_aom/review/operator_frequency.csv``
  - ``paper_aom/review/compact_bank_justification.md``
  - ``paper_aom/tables/table_selector_diagnostics.tex``

Usage::

    python paper_aom/review/selector_diagnostics.py \
        --aompls bench/AOM_v0/benchmark_runs/full/results.csv \
        --aomridge bench/AOM_v0/Ridge/benchmark_runs/paper_aom_aomridge_seeds012/results.csv \
        --out paper_aom/review/
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Compact AOM-PLS bank in canonical order (matches scores JSON key order in
# the AOM_v0 reference implementation; verified against
# ``bench/AOM_v0/benchmark_runs/full/results.csv``).
COMPACT_BANK_ORDER: tuple[str, ...] = (
    "identity",
    "sg_smooth_w11_p2",
    "sg_smooth_w21_p3",
    "sg_d1_w11_p2",
    "sg_d1_w21_p3",
    "sg_d2_w11_p2",
    "detrend_d1",
    "detrend_d2",
    "fd_d1",
)

# Compact-bank AOM variants used in the paper.
COMPACT_VARIANTS: tuple[str, ...] = (
    "AOM-compact-cv5-numpy",
    "ASLS-AOM-compact-cv5-numpy",
)

DEFAULT_VARIANT: str = "nirs4all-AOM-PLS-default"
STRICT_VARIANT: str = "AOM-compact-cv5-numpy"
ASLS_BRANCH_VARIANT: str = "ASLS-AOM-compact-cv5-numpy"

# Failure-mode dataset substrings (matched against ``dataset`` column).
FAILURE_MODE_SUBSTRINGS: tuple[str, ...] = (
    "Quartz",
    "QUARTZ",
    "Brix",
    "Tleaf",
    "FinalScore",
    "LMA",
    "LUCAS_SOC_all",
    "LUCAS_SOC_Cropland",
    "Ybased",
    "YbasedSplit",
    "wOutlier",
    "woOutlier",
)

# Threshold below which a denominator RMSEP is considered too small for the
# relative ratio to be informative.
RATIO_DENOM_EPS: float = 1e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_load_json(value: object) -> object | None:
    """Parse a JSON string; return ``None`` for missing / unparsable input."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed: object = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed


def _resolve_op_name(index: int, bank: Iterable[str]) -> str:
    """Map an integer operator index to a name; fall back to ``op_{idx}``."""
    bank_list = list(bank)
    if 0 <= index < len(bank_list):
        return bank_list[index]
    return f"op_{index}"


# ---------------------------------------------------------------------------
# AOM-PLS diagnostics (selected operator sequences)
# ---------------------------------------------------------------------------


def load_aompls_diagnostics(results_csv_path: str | Path) -> pd.DataFrame:
    """Parse ``selected_operator_sequence_json`` into per-row diagnostics.

    Args:
        results_csv_path: Path to an AOM-PLS benchmark ``results.csv``.

    Returns:
        DataFrame with one row per fit and columns ``dataset``, ``variant``,
        ``seed``, ``n_components_selected``, ``selected_op_<k>`` for each
        component, ``unique_ops_count``, ``most_frequent_op``, ``rmsep``,
        ``operator_bank``, ``parsed`` (False rows that could not be parsed are
        kept with ``parsed=False`` for transparency).
    """
    path = Path(results_csv_path)
    df = pd.read_csv(path, low_memory=False)

    required = {
        "dataset",
        "result_label",
        "seed",
        "selected_operator_sequence_json",
        "selected_operator_scores_json",
        "operator_bank",
        "RMSEP",
        "n_components_selected",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        variant = row["result_label"]
        seq = _safe_load_json(row["selected_operator_sequence_json"])
        scores = _safe_load_json(row["selected_operator_scores_json"])
        bank_names: tuple[str, ...]
        if isinstance(scores, dict) and scores:
            bank_names = tuple(scores.keys())
        elif row.get("operator_bank") == "compact":
            bank_names = COMPACT_BANK_ORDER
        else:
            bank_names = ()

        record: dict[str, object] = {
            "dataset": row["dataset"],
            "variant": variant,
            "seed": row["seed"],
            "operator_bank": row.get("operator_bank"),
            "rmsep": row.get("RMSEP"),
            "n_components_selected": row.get("n_components_selected"),
            "parsed": False,
            "skip_reason": None,
        }

        if not isinstance(seq, list) or not seq:
            record["skip_reason"] = "missing_or_empty_sequence"
            records.append(record)
            continue

        names: list[str] = []
        for raw in seq:
            if isinstance(raw, int):
                names.append(_resolve_op_name(raw, bank_names))
            elif isinstance(raw, str):
                names.append(raw)
            else:
                names.append(f"op_{raw}")

        for k, name in enumerate(names, start=1):
            record[f"selected_op_{k}"] = name

        counter = Counter(names)
        record["unique_ops_count"] = len(counter)
        record["most_frequent_op"] = counter.most_common(1)[0][0]
        record["parsed"] = True
        records.append(record)

    return pd.DataFrame(records)


def summarize_operator_frequency(
    diag: pd.DataFrame,
    variant_filter: list[str],
) -> pd.DataFrame:
    """Return per-operator frequency across selected variants.

    Args:
        diag: Output of :func:`load_aompls_diagnostics`.
        variant_filter: List of variant labels to include.

    Returns:
        DataFrame with ``operator``, ``n_times_selected``,
        ``fraction_of_components``, ``n_datasets_using_it``.
    """
    sub = diag[diag["variant"].isin(variant_filter) & diag["parsed"]].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "operator",
                "n_times_selected",
                "fraction_of_components",
                "n_datasets_using_it",
            ],
        )

    op_cols = [c for c in sub.columns if c.startswith("selected_op_")]
    melted = sub.melt(
        id_vars=["dataset", "variant", "seed"],
        value_vars=op_cols,
        value_name="operator",
    ).dropna(subset=["operator"])

    counts = melted.groupby("operator").size().rename("n_times_selected")
    datasets_per_op = (
        melted.groupby("operator")["dataset"].nunique().rename("n_datasets_using_it")
    )
    total_components = int(counts.sum())
    fraction = (counts / total_components).rename("fraction_of_components")
    out = pd.concat([counts, fraction, datasets_per_op], axis=1).reset_index()
    return out.sort_values("n_times_selected", ascending=False).reset_index(drop=True)


def summarize_n_components(
    diag: pd.DataFrame,
    variant_filter: list[str],
) -> pd.DataFrame:
    """Distribution of selected ``n_components`` per variant.

    The component count is taken from the ``n_components_selected`` column
    directly and does **not** require a parsed operator sequence, so it works
    for variants such as ``nirs4all-AOM-PLS-default`` that record only the
    component count.

    Returns:
        DataFrame with ``variant``, ``n_fits``, ``mean``, ``std``, ``min``,
        ``p25``, ``median``, ``p75``, ``max``.
    """
    sub = diag[diag["variant"].isin(variant_filter)].copy()
    sub["n_components_selected"] = pd.to_numeric(
        sub["n_components_selected"], errors="coerce",
    )
    sub = sub.dropna(subset=["n_components_selected"])
    grouped = sub.groupby("variant")["n_components_selected"].agg(
        n_fits="count",
        mean="mean",
        std="std",
        min="min",
        p25=lambda s: float(s.quantile(0.25)),
        median="median",
        p75=lambda s: float(s.quantile(0.75)),
        max="max",
    )
    return grouped.reset_index()


# ---------------------------------------------------------------------------
# AOM-Ridge diagnostics (AutoSelect + Blender)
# ---------------------------------------------------------------------------


def load_aomridge_selector_diagnostics(results_csv_path: str | Path) -> pd.DataFrame:
    """Parse AOM-Ridge AutoSelect / Blender selector diagnostics.

    Supports two CSV schemas:
      1. The audit schema (32 cols) — only ``canonical_name`` + ``selection``
         are available; selector diagnostics are *not* persisted. Rows are
         emitted with ``parsed=False`` and ``skip_reason='no_diagnostics_column'``.
      2. The diverse / paper schema (35 cols) — ``ridgepls_diagnostics``
         contains a JSON blob with ``selected_variant_label``,
         ``candidate_labels``, ``cv_scores`` and (for Blender) ``weights``.

    Returns:
        DataFrame with one row per fit and columns ``dataset``, ``variant``,
        ``seed``, ``selection``, ``selected_variant_label``,
        ``candidate_labels`` (list), ``cv_scores`` (list), ``weights`` (list
        or None), ``rmsep``, ``parsed``, ``skip_reason``.
    """
    path = Path(results_csv_path)
    df = pd.read_csv(path, low_memory=False)

    # Map column name variants. The two schemas differ on variant/seed names.
    variant_col = (
        "variant"
        if "variant" in df.columns
        else ("canonical_name" if "canonical_name" in df.columns else None)
    )
    seed_col = (
        "random_state"
        if "random_state" in df.columns
        else ("seed" if "seed" in df.columns else None)
    )
    rmsep_col = (
        "rmsep" if "rmsep" in df.columns else ("RMSEP" if "RMSEP" in df.columns else None)
    )
    if variant_col is None or seed_col is None:
        raise ValueError(f"Cannot locate variant/seed columns in {path}")

    has_diag_col = "ridgepls_diagnostics" in df.columns

    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        variant = row[variant_col]
        record: dict[str, object] = {
            "dataset": row.get("dataset"),
            "variant": variant,
            "seed": row[seed_col],
            "selection": row.get("selection"),
            "rmsep": row.get(rmsep_col) if rmsep_col else None,
            "selected_variant_label": None,
            "candidate_labels": None,
            "cv_scores": None,
            "weights": None,
            "parsed": False,
            "skip_reason": None,
        }

        if not has_diag_col:
            record["skip_reason"] = "no_diagnostics_column"
            records.append(record)
            continue

        diag = _safe_load_json(row["ridgepls_diagnostics"])
        if not isinstance(diag, dict):
            record["skip_reason"] = "missing_or_unparsable_diagnostics"
            records.append(record)
            continue

        record["selected_variant_label"] = diag.get("selected_variant_label")
        record["candidate_labels"] = diag.get("candidate_labels")
        record["cv_scores"] = diag.get("cv_scores")
        weights = diag.get("weights")
        record["weights"] = weights if isinstance(weights, list) else None
        record["parsed"] = bool(record["selected_variant_label"])
        if not record["parsed"]:
            record["skip_reason"] = "diagnostics_lacked_selected_variant_label"
        records.append(record)

    return pd.DataFrame(records)


def summarize_autoselect(diag: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset, per-seed AutoSelect picks plus aggregated counts.

    Filters to rows that look like AutoSelect picks (variant contains
    ``AutoSelect``) and that have a parsed ``selected_variant_label``.

    Returns:
        DataFrame with ``dataset``, ``seed``, ``autoselect_variant``,
        ``chosen_candidate``, plus an aggregated count column
        ``n_datasets_seeds`` per (autoselect_variant, chosen_candidate).
    """
    sub = diag[
        diag["parsed"]
        & diag["variant"].astype(str).str.contains("AutoSelect", na=False)
    ].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "autoselect_variant",
                "chosen_candidate",
                "n_datasets_seeds",
                "per_row",
            ],
        )

    per_row = sub.rename(
        columns={
            "variant": "autoselect_variant",
            "selected_variant_label": "chosen_candidate",
        }
    )[["dataset", "seed", "autoselect_variant", "chosen_candidate"]]

    counts = (
        per_row.groupby(["autoselect_variant", "chosen_candidate"])
        .size()
        .rename("n_datasets_seeds")
        .reset_index()
        .sort_values(
            ["autoselect_variant", "n_datasets_seeds"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )
    return counts.merge(
        per_row.groupby(["autoselect_variant", "chosen_candidate"])
        .apply(
            lambda g: g[["dataset", "seed"]].to_dict("records"),
            include_groups=False,
        )
        .rename("per_row")
        .reset_index(),
        on=["autoselect_variant", "chosen_candidate"],
        how="left",
    )


def summarize_blender_weights(diag: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Blender weights per candidate across datasets/seeds.

    Returns:
        DataFrame with ``blender_variant``, ``candidate``, ``mean_weight``,
        ``std_weight``, ``n_fits``, ``n_datasets_with_nonzero_weight``.
    """
    sub = diag[
        diag["parsed"]
        & diag["variant"].astype(str).str.contains("Blender", na=False)
        & diag["weights"].notna()
    ].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "blender_variant",
                "candidate",
                "mean_weight",
                "std_weight",
                "n_fits",
                "n_datasets_with_nonzero_weight",
            ],
        )

    rows: list[dict[str, object]] = []
    for _, r in sub.iterrows():
        labels = r["candidate_labels"]
        weights = r["weights"]
        if not isinstance(labels, list) or not isinstance(weights, list):
            continue
        if len(labels) != len(weights):
            continue
        for lab, w in zip(labels, weights, strict=True):
            rows.append(
                {
                    "blender_variant": r["variant"],
                    "dataset": r["dataset"],
                    "seed": r["seed"],
                    "candidate": lab,
                    "weight": float(w),
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        return pd.DataFrame(
            columns=[
                "blender_variant",
                "candidate",
                "mean_weight",
                "std_weight",
                "n_fits",
                "n_datasets_with_nonzero_weight",
            ],
        )

    nonzero = long[long["weight"] > 0]
    n_nonzero = (
        nonzero.groupby(["blender_variant", "candidate"])["dataset"]
        .nunique()
        .rename("n_datasets_with_nonzero_weight")
    )
    agg = long.groupby(["blender_variant", "candidate"]).agg(
        mean_weight=("weight", "mean"),
        std_weight=("weight", "std"),
        n_fits=("weight", "count"),
    )
    agg = agg.join(n_nonzero, how="left").fillna({"n_datasets_with_nonzero_weight": 0})
    agg["n_datasets_with_nonzero_weight"] = agg[
        "n_datasets_with_nonzero_weight"
    ].astype(int)
    return agg.reset_index().sort_values(
        ["blender_variant", "mean_weight"], ascending=[True, False],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Compact vs default, strict vs branch
# ---------------------------------------------------------------------------


def _paired_ratio(
    df: pd.DataFrame, numerator_variant: str, denominator_variant: str,
) -> pd.DataFrame:
    """Helper: per-dataset paired RMSEP ratio numerator/denominator."""
    metric_col = "RMSEP" if "RMSEP" in df.columns else "rmsep"
    if metric_col not in df.columns:
        raise ValueError("No RMSEP/rmsep column in input DataFrame")
    if "result_label" in df.columns:
        variant_col = "result_label"
    elif "variant" in df.columns:
        variant_col = "variant"
    else:
        raise ValueError("No variant column in input DataFrame")

    num = (
        df[df[variant_col] == numerator_variant]
        .groupby("dataset")[metric_col]
        .mean()
        .rename("numerator_rmsep")
    )
    den = (
        df[df[variant_col] == denominator_variant]
        .groupby("dataset")[metric_col]
        .mean()
        .rename("denominator_rmsep")
    )
    out = pd.concat([num, den], axis=1).dropna()
    out["ratio"] = out["numerator_rmsep"] / out["denominator_rmsep"]
    out["ratio_meaningful"] = out["denominator_rmsep"].abs() > RATIO_DENOM_EPS
    out["numerator_variant"] = numerator_variant
    out["denominator_variant"] = denominator_variant
    return out.reset_index()


def compare_compact_vs_default(df: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset paired ratio of ``AOM-compact-cv5-numpy`` vs default bank.

    Lower ratio means the compact bank is *better* than the default bank.
    """
    return _paired_ratio(df, STRICT_VARIANT, DEFAULT_VARIANT)


def compare_strict_vs_branch(df: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset paired ratio of strict-linear AOM vs ASLS+AOM branch.

    Numerator = ``AOM-compact-cv5-numpy`` (strict linear AOM).
    Denominator = ``ASLS-AOM-compact-cv5-numpy`` (ASLS branch + AOM).
    """
    return _paired_ratio(df, STRICT_VARIANT, ASLS_BRANCH_VARIANT)


# ---------------------------------------------------------------------------
# Seed stability and failure-mode tables
# ---------------------------------------------------------------------------


def seed_stability_selector(
    diag: pd.DataFrame, candidates: list[str],
) -> pd.DataFrame:
    """Count datasets where the selector picks differ across seeds.

    For AutoSelect variants: count datasets where ``chosen_candidate`` varies
    across seeds.

    For Blender variants: compute per-dataset std of each candidate's weight
    across seeds and report the maximum std observed per dataset.

    Args:
        diag: Output of :func:`load_aomridge_selector_diagnostics`.
        candidates: Allow-list of candidate labels to consider (kept for
            symmetry with the spec; non-empty list filters the Blender
            weight stability output).

    Returns:
        DataFrame with ``selector_variant``, ``mode``
        (``autoselect``/``blender``), ``dataset``, ``n_seeds``,
        ``n_unique_choices`` (autoselect) or ``max_weight_std`` (blender),
        ``is_unstable``.
    """
    sub = diag[diag["parsed"]].copy()
    out: list[dict[str, object]] = []

    autosel = sub[sub["variant"].astype(str).str.contains("AutoSelect", na=False)]
    for (variant, dataset), grp in autosel.groupby(["variant", "dataset"]):
        choices = grp["selected_variant_label"].dropna().unique().tolist()
        out.append(
            {
                "selector_variant": variant,
                "mode": "autoselect",
                "dataset": dataset,
                "n_seeds": int(grp["seed"].nunique()),
                "n_unique_choices": len(choices),
                "max_weight_std": None,
                "is_unstable": len(choices) > 1,
                "details": ",".join(map(str, sorted(choices))),
            }
        )

    blender = sub[
        sub["variant"].astype(str).str.contains("Blender", na=False)
        & sub["weights"].notna()
    ]
    for (variant, dataset), grp in blender.groupby(["variant", "dataset"]):
        rows: list[dict[str, float]] = []
        for _, r in grp.iterrows():
            labels = r["candidate_labels"]
            weights = r["weights"]
            if not isinstance(labels, list) or not isinstance(weights, list):
                continue
            for lab, w in zip(labels, weights, strict=True):
                if candidates and lab not in candidates:
                    continue
                rows.append({"candidate": lab, "weight": float(w), "seed": r["seed"]})
        long = pd.DataFrame(rows)
        if long.empty:
            continue
        std_per_candidate = long.groupby("candidate")["weight"].std(ddof=0)
        max_std = float(std_per_candidate.max())
        out.append(
            {
                "selector_variant": variant,
                "mode": "blender",
                "dataset": dataset,
                "n_seeds": int(grp["seed"].nunique()),
                "n_unique_choices": None,
                "max_weight_std": max_std,
                "is_unstable": max_std > 0.1,
                "details": "",
            }
        )

    return pd.DataFrame(out)


def failure_mode_table(
    df: pd.DataFrame, focus_datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Failure-mode RMSEP table for QUARTZ-like datasets.

    Returns absolute RMSEP per (dataset, variant) pair plus a ``ratio_meaningful``
    flag: ``False`` when any candidate variant has RMSEP within ``RATIO_DENOM_EPS``
    of zero, signalling that ratio-based comparisons should not be reported.

    Args:
        df: Raw benchmark results DataFrame.
        focus_datasets: Optional explicit list of dataset names. If omitted,
            datasets whose name contains any of
            :data:`FAILURE_MODE_SUBSTRINGS` are selected.
    """
    metric_col = "RMSEP" if "RMSEP" in df.columns else "rmsep"
    variant_col = "result_label" if "result_label" in df.columns else "variant"
    if metric_col not in df.columns or variant_col not in df.columns:
        raise ValueError("Need RMSEP/rmsep and result_label/variant columns")

    if focus_datasets is None:
        pattern = "|".join(FAILURE_MODE_SUBSTRINGS)
        mask = df["dataset"].astype(str).str.contains(pattern, case=False, na=False)
    else:
        mask = df["dataset"].isin(focus_datasets)

    sub = df.loc[mask, ["dataset", variant_col, metric_col]].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=["dataset", "variant", "rmsep", "ratio_meaningful"],
        )

    pivot = (
        sub.rename(columns={variant_col: "variant", metric_col: "rmsep"})
        .groupby(["dataset", "variant"])["rmsep"]
        .mean()
        .reset_index()
    )
    # ratio_meaningful: True if RMSEP > eps; ratios with smaller denominators
    # would be numerically unstable.
    pivot["ratio_meaningful"] = pivot["rmsep"].abs() > RATIO_DENOM_EPS
    return pivot.sort_values(["dataset", "variant"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Markdown / LaTeX writers
# ---------------------------------------------------------------------------


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}\\%"


def write_compact_bank_justification_md(
    op_freq: pd.DataFrame,
    n_comp: pd.DataFrame,
    compact_vs_default: pd.DataFrame,
    strict_vs_branch: pd.DataFrame,
    out_path: Path,
) -> None:
    """Auto-generate the compact-bank justification markdown."""
    lines: list[str] = []
    lines.append("# Compact AOM-PLS bank: empirical justification\n")
    lines.append(
        "Auto-generated by `paper_aom/review/selector_diagnostics.py`. "
        "Do not edit by hand; re-run the script after new benchmark seeds.\n",
    )

    lines.append("## Compact bank contents\n")
    lines.append("Compact bank order (matches `bench/AOM_lib` and the AOM_v0 reference):\n")
    for i, name in enumerate(COMPACT_BANK_ORDER, start=1):
        lines.append(f"{i}. `{name}`")
    lines.append("")

    lines.append("## Operator frequency (compact-bank variants)\n")
    if op_freq.empty:
        lines.append("_No parsed selections available._\n")
    else:
        total_components = int(op_freq["n_times_selected"].sum())
        top = op_freq.head(3)
        names = ", ".join(top["operator"].tolist())
        fraction = float(top["fraction_of_components"].sum())
        lines.append(
            f"Compact 9-op bank, top 3 operators **{names}** account for "
            f"**{_format_pct(fraction)}** of selections "
            f"({int(top['n_times_selected'].sum())}/{total_components}).\n",
        )
        lines.append("| Operator | Selections | Fraction | Datasets using it |")
        lines.append("|---|---|---|---|")
        for _, r in op_freq.iterrows():
            lines.append(
                f"| `{r['operator']}` | {int(r['n_times_selected'])} | "
                f"{_format_pct(float(r['fraction_of_components']))} | "
                f"{int(r['n_datasets_using_it'])} |",
            )
        lines.append("")

    lines.append("## Selected component count\n")
    if n_comp.empty:
        lines.append("_No parsed component counts available._\n")
    else:
        lines.append(
            "| Variant | n_fits | mean | std | min | p25 | median | p75 | max |",
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in n_comp.iterrows():
            lines.append(
                f"| `{r['variant']}` | {int(r['n_fits'])} | "
                f"{r['mean']:.2f} | {r['std']:.2f} | {int(r['min'])} | "
                f"{r['p25']:.1f} | {r['median']:.1f} | {r['p75']:.1f} | {int(r['max'])} |",
            )
        lines.append("")

    lines.append("## Compact bank vs default bank\n")
    if compact_vs_default.empty:
        lines.append("_No paired rows._\n")
    else:
        ratios = compact_vs_default["ratio"].dropna()
        n_wins = int((ratios < 1.0).sum())
        n_total = int(len(ratios))
        lines.append(
            f"`{STRICT_VARIANT}` beats `{DEFAULT_VARIANT}` on "
            f"**{n_wins}/{n_total}** datasets (geometric mean ratio "
            f"**{math.exp(ratios.apply(math.log).mean()):.4f}**, "
            f"lower is better).\n",
        )

    lines.append("## Strict-linear AOM vs ASLS branch\n")
    if strict_vs_branch.empty:
        lines.append("_No paired rows._\n")
    else:
        ratios = strict_vs_branch["ratio"].dropna()
        n_wins = int((ratios < 1.0).sum())
        n_total = int(len(ratios))
        lines.append(
            f"`{STRICT_VARIANT}` beats `{ASLS_BRANCH_VARIANT}` on "
            f"**{n_wins}/{n_total}** datasets (geometric mean ratio "
            f"**{math.exp(ratios.apply(math.log).mean()):.4f}**).\n",
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selector_diagnostics_tex(
    autoselect_summary: pd.DataFrame,
    blender_summary: pd.DataFrame,
    seed_stability: pd.DataFrame,
    out_path: Path,
) -> None:
    """Write the LaTeX selector-diagnostics table for the paper."""
    lines: list[str] = []
    lines.append("% Auto-generated by paper_aom/review/selector_diagnostics.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{AOM-Ridge selector diagnostics: AutoSelect chosen-candidate "
        "counts, Blender weight means, and cross-seed stability.}",
    )
    lines.append("\\label{tab:selector-diagnostics}")
    lines.append("\\begin{tabular}{llrr}")
    lines.append("\\toprule")
    lines.append("Selector & Candidate & Count / Mean weight & Std \\\\")
    lines.append("\\midrule")

    if not autoselect_summary.empty:
        lines.append("\\multicolumn{4}{l}{\\textit{AutoSelect chosen-candidate counts}} \\\\")
        for _, r in autoselect_summary.iterrows():
            lines.append(
                f"{r['autoselect_variant']} & {r['chosen_candidate']} & "
                f"{int(r['n_datasets_seeds'])} & -- \\\\",
            )
        lines.append("\\midrule")

    if not blender_summary.empty:
        lines.append("\\multicolumn{4}{l}{\\textit{Blender weight mean (std) per candidate}} \\\\")
        for _, r in blender_summary.iterrows():
            std = r["std_weight"]
            std_str = f"{std:.3f}" if pd.notna(std) else "--"
            lines.append(
                f"{r['blender_variant']} & {r['candidate']} & "
                f"{r['mean_weight']:.3f} & {std_str} \\\\",
            )
        lines.append("\\midrule")

    if not seed_stability.empty:
        unstable = seed_stability[seed_stability["is_unstable"]]
        header = (
            "\\multicolumn{4}{l}{\\textit{Cross-seed instability ("
            f"{len(unstable)}/{len(seed_stability)} dataset rows)"
            "}} \\\\"
        )
        lines.append(header)
        for _, r in unstable.iterrows():
            metric = (
                f"{int(r['n_unique_choices'])} choices"
                if r["mode"] == "autoselect"
                else f"max std = {r['max_weight_std']:.3f}"
            )
            lines.append(
                f"{r['selector_variant']} & {r['dataset']} & {metric} & -- \\\\",
            )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract AOM selector / operator / failure diagnostics.",
    )
    parser.add_argument(
        "--aompls",
        type=Path,
        default=Path("bench/AOM_v0/benchmark_runs/full/results.csv"),
        help="AOM-PLS benchmark results.csv with selected_operator_sequence_json.",
    )
    parser.add_argument(
        "--aomridge",
        type=Path,
        default=Path(
            "bench/AOM_v0/Ridge/benchmark_runs/paper_aom_aomridge_seeds012/results.csv",
        ),
        help="AOM-Ridge benchmark results.csv with ridgepls_diagnostics.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paper_aom/review/"),
        help="Output directory for review CSVs and md.",
    )
    parser.add_argument(
        "--tables",
        type=Path,
        default=Path("paper_aom/tables/"),
        help="Output directory for LaTeX tables.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)

    print(f"[selector_diagnostics] AOM-PLS source: {args.aompls}")
    aompls_diag = load_aompls_diagnostics(args.aompls)
    print(
        f"  parsed {int(aompls_diag['parsed'].sum())}/{len(aompls_diag)} rows "
        f"(skipped reasons: "
        f"{aompls_diag.loc[~aompls_diag['parsed'], 'skip_reason'].value_counts().to_dict()})",
    )

    aompls_diag.to_csv(args.out / "selector_diagnostics.csv", index=False)

    op_freq = summarize_operator_frequency(aompls_diag, list(COMPACT_VARIANTS))
    op_freq.to_csv(args.out / "operator_frequency.csv", index=False)
    n_comp = summarize_n_components(
        aompls_diag, [STRICT_VARIANT, ASLS_BRANCH_VARIANT, DEFAULT_VARIANT],
    )

    # Need the raw frame for paired comparisons / failure mode.
    raw_aompls = pd.read_csv(args.aompls, low_memory=False)
    compact_vs_default = compare_compact_vs_default(raw_aompls)
    strict_vs_branch = compare_strict_vs_branch(raw_aompls)
    failure_modes = failure_mode_table(raw_aompls, focus_datasets=None)
    failure_modes.to_csv(args.out / "failure_mode_table.csv", index=False)

    write_compact_bank_justification_md(
        op_freq=op_freq,
        n_comp=n_comp,
        compact_vs_default=compact_vs_default,
        strict_vs_branch=strict_vs_branch,
        out_path=args.out / "compact_bank_justification.md",
    )

    print(f"[selector_diagnostics] AOM-Ridge source: {args.aomridge}")
    ridge_diag = load_aomridge_selector_diagnostics(args.aomridge)
    print(
        f"  parsed {int(ridge_diag['parsed'].sum())}/{len(ridge_diag)} rows "
        f"(skipped reasons: "
        f"{ridge_diag.loc[~ridge_diag['parsed'], 'skip_reason'].value_counts().to_dict()})",
    )
    autoselect_summary = summarize_autoselect(ridge_diag)
    blender_summary = summarize_blender_weights(ridge_diag)

    # Candidates allow-list for seed stability: union of parsed candidate labels.
    candidate_labels: set[str] = set()
    for labs in ridge_diag.loc[ridge_diag["parsed"], "candidate_labels"]:
        if isinstance(labs, list):
            candidate_labels.update(labs)
    seed_stab = seed_stability_selector(ridge_diag, sorted(candidate_labels))
    seed_stab.to_csv(args.out / "selector_seed_stability.csv", index=False)

    write_selector_diagnostics_tex(
        autoselect_summary=autoselect_summary.drop(columns=["per_row"], errors="ignore"),
        blender_summary=blender_summary,
        seed_stability=seed_stab,
        out_path=args.tables / "table_selector_diagnostics.tex",
    )

    # Print headlines.
    if not op_freq.empty:
        top = op_freq.head(5)
        print("\n[selector_diagnostics] Top operators (compact-bank variants):")
        for _, r in top.iterrows():
            print(
                f"  {r['operator']:20s}  n={int(r['n_times_selected']):4d}  "
                f"fraction={float(r['fraction_of_components']) * 100:5.1f}%  "
                f"datasets={int(r['n_datasets_using_it'])}",
            )
    if not compact_vs_default.empty:
        ratios = compact_vs_default["ratio"].dropna()
        wins = int((ratios < 1.0).sum())
        print(
            f"[selector_diagnostics] compact-vs-default: "
            f"{wins}/{len(ratios)} datasets favour compact (geom. mean "
            f"{math.exp(ratios.apply(math.log).mean()):.4f}).",
        )
    if not autoselect_summary.empty:
        print("[selector_diagnostics] AutoSelect picks:")
        for _, r in autoselect_summary.iterrows():
            print(
                f"  {r['autoselect_variant']:40s} -> {r['chosen_candidate']:45s} "
                f"({int(r['n_datasets_seeds'])} fits)",
            )

    print(f"\nWritten:\n  {args.out / 'selector_diagnostics.csv'}")
    print(f"  {args.out / 'operator_frequency.csv'}")
    print(f"  {args.out / 'failure_mode_table.csv'}")
    print(f"  {args.out / 'selector_seed_stability.csv'}")
    print(f"  {args.out / 'compact_bank_justification.md'}")
    print(f"  {args.tables / 'table_selector_diagnostics.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
