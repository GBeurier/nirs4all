#!/usr/bin/env python3
"""Evaluate representative benchmark subsets from benchmark_master_results.csv.

This is a transfer-first replacement for the older class-zscore-only subset
selection.  The selected subsets below were chosen from a constrained random
search over the 57-dataset regression core, using one dataset per dataset_group
and weighting these scopes:

- all high-coverage candidates
- no-TabPFN candidates
- linear spectral candidates
- AOM-Ridge only
- AOM-PLS only
- multi-kernel ridge only
- TabPFN only
- nonlinear/challenger candidates

Run:
    python3 bench/Subset_analysis/rethink_subset_selection.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MASTER = ROOT / "bench/benchmark_master_results.csv"
CORE_MATRIX = OUT / "main_class_score_matrix.csv"

MIN_GLOBAL_COVERAGE = 0.90

LINEAR_CLASSES = {"PLS", "Ridge", "AOM-PLS", "AOM-Ridge", "POP-PLS", "FCK-PLS"}
NONLINEAR_CLASSES = {
    "TabPFN",
    "CatBoost",
    "NICON/CNN",
    "Hybrid CNN+linear",
    "Hybrid CNN+AOM",
    "Multi-kernel ridge",
    "Meta-selector/MoE",
    "Other",
}

SCOPES = {
    "all_candidates": {},
    "no_tabpfn": {"exclude": {"TabPFN"}},
    "linear_core": {"include": {"PLS", "Ridge", "AOM-PLS", "AOM-Ridge", "POP-PLS"}},
    "aom_ridge_only": {"include": {"AOM-Ridge"}},
    "aom_pls_only": {"include": {"AOM-PLS"}},
    "multi_kernel_only": {"include": {"Multi-kernel ridge"}},
    "tabpfn_only": {"include": {"TabPFN"}},
    "nonlinear_core": {
        "include": {
            "TabPFN",
            "CatBoost",
            "NICON/CNN",
            "Hybrid CNN+linear",
            "Hybrid CNN+AOM",
            "Multi-kernel ridge",
        }
    },
}

SUBSETS = {
    "fast12_transfer_core": {
        "purpose": "Primary fast iteration gate. One dataset per group, AOM-Ridge-compatible, balanced between linear wins, nonlinear wins, and ties.",
        "datasets": [
            "DIESEL_bp50_246_hlb-a",
            "Corn_Oil_80_ZhengChenPelegYbaseSplit",
            "MP_spxyG",
            "TIC_spxy70",
            "WUEinst_spxyG70_30_byCultivar_MicroNIR_NeoSpectra",
            "brix_groupSampleID_stratDateVar_balRows",
            "All_manure_K2O_SPXY_strat_Manure_type",
            "Biscuit_Sucrose_40_RandomSplit",
            "Ccar_spxyG_block2deg",
            "LUCAS_pH_Organic_1763_LiuRandomOrganic",
            "N_woOutlier",
            "Beer_OriginalExtract_60_KS",
        ],
    },
    "audit20_transfer_core": {
        "purpose": "Second-pass audit subset. Still AOM-Ridge-compatible, broader dataset-group and behavior coverage before running the full 57.",
        "datasets": [
            "All_manure_K2O_SPXY_strat_Manure_type",
            "Rd25_GTtestSite",
            "MP_spxyG",
            "An_spxyG70_30_byCultivar_MicroNIR_NeoSpectra",
            "Biscuit_Sucrose_40_RandomSplit",
            "Ccar_spxyG_block2deg",
            "DIESEL_bp50_246_b-a",
            "Milk_Urea_1224_KS",
            "ALPINE_P_291_KS",
            "Rice_Amylose_313_YbasedSplit",
            "Quartz_spxy70",
            "LUCAS_pH_Organic_1763_LiuRandomOrganic",
            "Fv_Fm_grp70_30",
            "Escitalopramt_310_Zhao",
            "WOOD_N_402_Olale",
            "ph_groupSampleID_stratDateVar_balRows",
            "C_woOutlier",
            "TIC_spxy70",
            "Beer_OriginalExtract_60_YbaseSplit",
            "Firmness_spxy70",
        ],
    },
}

AOM_RIDGE_COVERAGE_HOLES = [
    "Brix_spxy70",
    "LUCAS_SOC_Cropland_8731_NocitaKS",
    "Malaria_Oocist_333_Maia",
    "Malaria_Sporozoite_229_Maia",
]


def clean_str(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def load_core() -> list[str]:
    cols = pd.read_csv(CORE_MATRIX, nrows=0).columns.tolist()
    if len(cols) < 2:
        raise RuntimeError(f"Cannot infer core datasets from {CORE_MATRIX}")
    return cols[1:]


def load_master(core: list[str]) -> pd.DataFrame:
    df = pd.read_csv(MASTER, low_memory=False)
    for col in [
        "status",
        "task",
        "evaluation_split",
        "record_type",
        "model_class",
        "source_family",
        "model_name",
        "variant",
        "dataset",
        "dataset_group",
        "strategy_family",
    ]:
        df[col] = clean_str(df[col])

    keep = (
        df["record_type"].isin(["observed", "reference_paper"])
        & df["status"].str.lower().eq("ok")
        & df["evaluation_split"].str.lower().eq("test")
        & df["task"].str.lower().eq("regression")
        & df["dataset"].isin(core)
    )
    df = df.loc[keep].copy()
    df["score_ratio_vs_dataset_pls"] = pd.to_numeric(
        df["score_ratio_vs_dataset_pls"], errors="coerce"
    )
    df = df[df["score_ratio_vs_dataset_pls"].notna()].copy()
    df["candidate_id"] = (
        df["source_family"]
        + " | "
        + df["model_class"]
        + " | "
        + df["model_name"]
        + " | "
        + df["variant"]
    )
    return df


def build_candidate_matrix(df: pd.DataFrame, core: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = df.pivot_table(
        index="candidate_id",
        columns="dataset",
        values="score_ratio_vs_dataset_pls",
        aggfunc="median",
    ).reindex(columns=core)
    metadata = (
        df.sort_values(["candidate_id", "dataset"])
        .drop_duplicates("candidate_id")
        .set_index("candidate_id")[
            ["source_family", "model_class", "model_name", "variant", "strategy_family"]
        ]
    )
    return matrix, metadata.reindex(matrix.index)


def apply_scope(matrix: pd.DataFrame, metadata: pd.DataFrame, scope: str) -> pd.DataFrame:
    rule = SCOPES[scope]
    keep = pd.Series(True, index=matrix.index)
    classes = metadata["model_class"].reindex(matrix.index)
    if "include" in rule:
        keep &= classes.isin(rule["include"])
    if "exclude" in rule:
        keep &= ~classes.isin(rule["exclude"])
    return matrix.loc[keep]


def spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame["x"].rank().corr(frame["y"].rank()))


def evaluate_subset(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    subset_name: str,
    subset: list[str],
    scope: str,
) -> dict[str, object]:
    scoped = apply_scope(matrix, metadata, scope)
    global_coverage = scoped.notna().mean(axis=1)
    high_coverage = scoped.loc[global_coverage >= MIN_GLOBAL_COVERAGE]
    complete = high_coverage.loc[high_coverage[subset].notna().all(axis=1)]
    if len(complete) < 3:
        return {
            "subset_name": subset_name,
            "scope": scope,
            "subset_size": len(subset),
            "eligible_candidates": int(len(complete)),
            "status": "too_few_candidates",
        }

    full_metric = complete.median(axis=1, skipna=True)
    subset_metric = complete[subset].median(axis=1, skipna=True)
    full_rank = full_metric.rank(method="min")

    subset_winner = str(subset_metric.idxmin())
    full_winner = str(full_metric.idxmin())
    top_k = min(5, len(full_metric))
    top_overlap = len(
        set(subset_metric.nsmallest(top_k).index) & set(full_metric.nsmallest(top_k).index)
    ) / top_k

    return {
        "subset_name": subset_name,
        "scope": scope,
        "subset_size": len(subset),
        "eligible_candidates": int(len(complete)),
        "status": "ok",
        "spearman_subset_vs_global": spearman(subset_metric, full_metric),
        "top5_overlap": top_overlap,
        "subset_winner_candidate": subset_winner,
        "subset_winner_class": metadata.loc[subset_winner, "model_class"],
        "subset_winner_global_median_ratio": float(full_metric.loc[subset_winner]),
        "global_oracle_candidate": full_winner,
        "global_oracle_class": metadata.loc[full_winner, "model_class"],
        "global_oracle_median_ratio": float(full_metric.loc[full_winner]),
        "winner_global_rank": int(full_rank.loc[subset_winner]),
        "global_regret_abs": float(full_metric.loc[subset_winner] - full_metric.loc[full_winner]),
    }


def dataset_profile(df: pd.DataFrame, core: list[str]) -> pd.DataFrame:
    per_candidate = (
        df.groupby(["dataset", "model_class", "candidate_id"])["score_ratio_vs_dataset_pls"]
        .median()
        .reset_index()
    )
    class_best = (
        per_candidate.groupby(["dataset", "model_class"])["score_ratio_vs_dataset_pls"]
        .min()
        .unstack("model_class")
        .reindex(core)
    )
    rows = []
    groups = df.drop_duplicates("dataset").set_index("dataset")["dataset_group"]
    for dataset in core:
        row = class_best.loc[dataset]
        linear_best = row[[c for c in class_best.columns if c in LINEAR_CLASSES]].min(
            skipna=True
        )
        nonlinear_best = row[[c for c in class_best.columns if c in NONLINEAR_CLASSES]].min(
            skipna=True
        )
        ratio = nonlinear_best / linear_best if linear_best and np.isfinite(linear_best) else np.nan
        if ratio < 0.85:
            behavior = "nonlinear_strong"
        elif ratio < 0.98:
            behavior = "nonlinear_mild"
        elif ratio <= 1.02:
            behavior = "tie"
        else:
            behavior = "linear"
        rows.append(
            {
                "dataset": dataset,
                "dataset_group": groups.get(dataset, ""),
                "best_model_class": row.idxmin(skipna=True),
                "linear_best_ratio": linear_best,
                "nonlinear_best_ratio": nonlinear_best,
                "nonlinear_to_linear_ratio": ratio,
                "behavior": behavior,
            }
        )
    return pd.DataFrame(rows)


def summarize_subset(profile: pd.DataFrame, subset_name: str, subset: list[str]) -> dict[str, object]:
    sub = profile[profile["dataset"].isin(subset)].copy()
    behavior_counts = Counter(sub["behavior"])
    group_counts = Counter(sub["dataset_group"])
    return {
        "subset_name": subset_name,
        "subset_size": len(subset),
        "n_dataset_groups": int(sub["dataset_group"].nunique()),
        "max_per_group": max(group_counts.values()) if group_counts else 0,
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "best_class_counts": dict(sorted(Counter(sub["best_model_class"]).items())),
    }


def write_outputs(
    core: list[str],
    profile: pd.DataFrame,
    transfer: pd.DataFrame,
    subset_summaries: list[dict[str, object]],
) -> None:
    payload = {
        "protocol_version": "transfer_first_2026_05_05",
        "source": "bench/benchmark_master_results.csv",
        "core_dataset_count": len(core),
        "candidate_filter": {
            "rows": "observed/reference_paper, regression, test, status=ok",
            "score": "score_ratio_vs_dataset_pls",
            "candidate": "source_family | model_class | model_name | variant",
            "candidate_aggregation": "median duplicate rows per candidate/dataset",
            "minimum_global_coverage": MIN_GLOBAL_COVERAGE,
            "subset_requirement": "candidate must have complete scores on the subset",
        },
        "selection_notes": [
            "The old class-balanced subset matched model-class fingerprints but failed no-TabPFN transfer.",
            "The new subsets were selected to preserve concrete candidate rankings across linear and nonlinear/challenger scopes.",
            "The primary subsets intentionally avoid current AOM-Ridge coverage holes so linear-family audits remain valid.",
        ],
        "aom_ridge_coverage_holes_not_in_primary_subsets": AOM_RIDGE_COVERAGE_HOLES,
        "subsets": SUBSETS,
        "subset_summaries": subset_summaries,
    }
    (OUT / "rethought_subsets.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    transfer.to_csv(OUT / "rethought_subset_transfer_summary.csv", index=False)
    profile.to_csv(OUT / "rethought_dataset_behavior.csv", index=False)

    lines = [
        "# Rethought Representative Subsets",
        "",
        "Generated from `bench/benchmark_master_results.csv`.",
        "",
        "## Method",
        "",
        "- Universe: the 57 regression datasets already used by `main_class_score_matrix.csv`.",
        "- Rows: `record_type in {observed, reference_paper}`, regression, test split, `status=ok`.",
        "- Score: `score_ratio_vs_dataset_pls`; lower is better.",
        "- Candidate: `source_family | model_class | model_name | variant`, aggregated by median per dataset.",
        f"- Eligibility: at least {MIN_GLOBAL_COVERAGE:.0%} global coverage and complete subset coverage.",
        "- Objective: preserve concrete candidate transfer across all-candidate, no-TabPFN, linear-core, AOM-Ridge, AOM-PLS, multi-kernel, TabPFN, and nonlinear scopes.",
        "",
        "## Recommended Subsets",
        "",
    ]
    summary_by_name = {row["subset_name"]: row for row in subset_summaries}
    for subset_name, spec in SUBSETS.items():
        summary = summary_by_name[subset_name]
        lines.extend(
            [
                f"### {subset_name}",
                "",
                spec["purpose"],
                "",
                f"- Size: {summary['subset_size']}",
                f"- Dataset groups: {summary['n_dataset_groups']}",
                f"- Behavior counts: `{summary['behavior_counts']}`",
                "",
            ]
        )
        for dataset in spec["datasets"]:
            row = profile[profile["dataset"] == dataset].iloc[0]
            lines.append(
                f"- `{dataset}` ({row.dataset_group}; {row.behavior}; best={row.best_model_class})"
            )
        lines.append("")

    lines.extend(
        [
            "## Transfer Check",
            "",
            "| Subset | Scope | Eligible | Spearman | Winner Rank | Regret | Winner Class |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in transfer.itertuples(index=False):
        if row.status != "ok":
            lines.append(
                f"| `{row.subset_name}` | `{row.scope}` | {row.eligible_candidates} |  |  |  | {row.status} |"
            )
            continue
        lines.append(
            f"| `{row.subset_name}` | `{row.scope}` | {row.eligible_candidates} | {row.spearman_subset_vs_global:.3f} | {row.winner_global_rank} | {row.global_regret_abs:.4f} | `{row.subset_winner_class}` |"
        )

    lines.extend(
        [
            "",
            "## Coverage Caution",
            "",
            "The following core datasets are useful stress cases, but current high-coverage AOM-Ridge rows are missing there, so they are not part of the default subsets:",
            "",
        ]
    )
    for dataset in AOM_RIDGE_COVERAGE_HOLES:
        lines.append(f"- `{dataset}`")
    lines.append("")

    (OUT / "RETHOUGHT_SUBSETS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    core = load_core()
    df = load_master(core)
    matrix, metadata = build_candidate_matrix(df, core)
    profile = dataset_profile(df, core)

    rows = []
    for subset_name, spec in SUBSETS.items():
        missing = sorted(set(spec["datasets"]) - set(core))
        if missing:
            raise RuntimeError(f"{subset_name} contains datasets outside the core: {missing}")
        for scope in SCOPES:
            rows.append(evaluate_subset(matrix, metadata, subset_name, spec["datasets"], scope))
    transfer = pd.DataFrame(rows)
    subset_summaries = [
        summarize_subset(profile, subset_name, spec["datasets"])
        for subset_name, spec in SUBSETS.items()
    ]
    write_outputs(core, profile, transfer, subset_summaries)
    print(f"Wrote rethought subset outputs to {OUT}")


if __name__ == "__main__":
    main()
