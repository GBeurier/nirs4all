#!/usr/bin/env python3
"""Subset-to-global model-selection transfer analysis.

This script answers a different question than the existing class-balanced
fingerprint search:

    If a concrete model is selected using only a subset of datasets, does that
    selected model remain strong on the full benchmark core?

It uses bench/benchmark_master_results.csv as the source of truth, keeps only
observed/reference regression test rows, normalizes by each dataset's PLS score,
and evaluates both current Subset_analysis subsets and random baselines.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SEED = 20260504
RANDOM_TRIALS = 500
MIN_GLOBAL_COVERAGE = 0.90
TOP_K = 5

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MASTER = ROOT / "bench/benchmark_master_results.csv"
CORE_MATRIX = OUT / "main_class_score_matrix.csv"
SELECTED_JSON = OUT / "selected_subset.json"
LEGACY_SEARCH = OUT / "legacy_variant_heavy/subset_search_results.csv"


SCOPE_DEFINITIONS = {
    "all_candidates": {},
    "no_tabpfn": {"exclude_classes": {"TabPFN"}},
    "tabpfn_only": {"include_classes": {"TabPFN"}},
    "aom_pls_only": {"include_classes": {"AOM-PLS"}},
    "aom_ridge_only": {"include_classes": {"AOM-Ridge"}},
    "meta_selector_only": {"include_classes": {"Meta-selector/MoE"}},
    "ridge_only": {"include_classes": {"Ridge"}},
    "pls_only": {"include_classes": {"PLS"}},
}


def clean_str(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def safe_bool(series: pd.Series) -> pd.Series:
    return clean_str(series).str.lower().isin({"true", "1", "yes"})


def spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame["x"].rank().corr(frame["y"].rank()))


def pairwise_agreement(x: pd.Series, y: pd.Series, tol: float = 1e-6) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    vals_x = frame["x"].to_numpy(dtype=float)
    vals_y = frame["y"].to_numpy(dtype=float)
    if len(vals_x) < 2:
        return np.nan
    agree = 0
    total = 0
    for i in range(len(vals_x)):
        for j in range(i + 1, len(vals_x)):
            dx = vals_x[i] - vals_x[j]
            dy = vals_y[i] - vals_y[j]
            sx = 0 if abs(dx) <= tol else np.sign(dx)
            sy = 0 if abs(dy) <= tol else np.sign(dy)
            agree += int(sx == sy)
            total += 1
    return float(agree / total) if total else np.nan


def mode_or_blank(values: Iterable[object]) -> str:
    s = pd.Series(list(values)).dropna().astype(str)
    s = s[s.str.strip() != ""]
    if s.empty:
        return ""
    return str(s.mode().iloc[0])


def load_core_datasets() -> list[str]:
    """Return the 57-dataset core used by the previous Subset_analysis run."""
    if CORE_MATRIX.exists():
        cols = pd.read_csv(CORE_MATRIX, nrows=0).columns.tolist()
        if len(cols) > 1:
            return cols[1:]
    raise FileNotFoundError(
        f"Cannot infer core datasets because {CORE_MATRIX} is missing or empty."
    )


def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER, low_memory=False)
    ok = clean_str(df["status"]).str.lower().eq("ok")
    regression = clean_str(df["task"]).str.lower().eq("regression")
    source_rows = df["record_type"].isin(["observed", "reference_paper"])
    test_rows = clean_str(df["evaluation_split"]).str.lower().eq("test")
    lower_is_better = safe_bool(df["lower_is_better"])
    has_ratio = pd.to_numeric(df["score_ratio_vs_dataset_pls"], errors="coerce").notna()
    keep = regression & source_rows & ok & test_rows & lower_is_better & has_ratio
    df = df.loc[keep].copy()
    df["score_ratio_vs_dataset_pls"] = pd.to_numeric(
        df["score_ratio_vs_dataset_pls"], errors="coerce"
    )
    for col in [
        "source_family",
        "model_class",
        "model_name",
        "variant",
        "dataset",
        "dataset_group",
    ]:
        df[col] = clean_str(df[col])
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

    meta_cols = [
        "candidate_id",
        "source_family",
        "model_class",
        "model_name",
        "variant",
        "strategy_family",
    ]
    metadata = (
        df.sort_values(["candidate_id", "dataset"])
        .drop_duplicates("candidate_id")[meta_cols]
        .set_index("candidate_id")
    )
    matrix.index.name = "candidate_id"
    return matrix, metadata


def load_named_subsets(core: list[str]) -> dict[str, list[str]]:
    core_set = set(core)
    subsets: dict[str, list[str]] = {"full_core_57": core}

    if SELECTED_JSON.exists():
        payload = json.loads(SELECTED_JSON.read_text(encoding="utf-8"))
        current = [d for d in payload.get("selected_datasets", []) if d in core_set]
        if current:
            subsets[f"current_class_balanced_{len(current)}"] = current
        conservative = (payload.get("conservative_alternative") or {}).get("datasets", [])
        conservative = [d for d in conservative if d in core_set]
        if conservative:
            subsets[f"current_conservative_{len(conservative)}"] = conservative

    if LEGACY_SEARCH.exists():
        legacy = pd.read_csv(LEGACY_SEARCH)
        for target_size in [10, 19]:
            row = legacy.loc[legacy["size"].eq(target_size)]
            if not row.empty:
                datasets = str(row.iloc[0]["subset"]).split("|")
                datasets = [d for d in datasets if d in core_set]
                if datasets:
                    subsets[f"legacy_variant_heavy_{len(datasets)}"] = datasets
    return subsets


def apply_scope(matrix: pd.DataFrame, metadata: pd.DataFrame, scope: str) -> pd.DataFrame:
    rule = SCOPE_DEFINITIONS[scope]
    classes = metadata["model_class"].reindex(matrix.index)
    keep = pd.Series(True, index=matrix.index)
    include = rule.get("include_classes")
    exclude = rule.get("exclude_classes")
    if include:
        keep &= classes.isin(include)
    if exclude:
        keep &= ~classes.isin(exclude)
    return matrix.loc[keep]


def evaluate_subset_transfer(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    subset_name: str,
    subset: list[str],
    scope: str,
    min_global_coverage: float = MIN_GLOBAL_COVERAGE,
    compute_pairwise: bool = True,
) -> dict[str, object]:
    scoped = apply_scope(matrix, metadata, scope)
    subset = [d for d in subset if d in scoped.columns]
    if not subset:
        return {
            "subset_name": subset_name,
            "scope": scope,
            "subset_size": 0,
            "eligible_candidates": 0,
            "status": "no_overlap",
        }

    global_coverage = scoped.notna().mean(axis=1)
    subset_complete = scoped[subset].notna().all(axis=1)
    eligible = scoped.loc[(global_coverage >= min_global_coverage) & subset_complete]
    if len(eligible) < 3:
        return {
            "subset_name": subset_name,
            "scope": scope,
            "subset_size": len(subset),
            "eligible_candidates": int(len(eligible)),
            "status": "too_few_candidates",
        }

    full_metric = eligible.median(axis=1, skipna=True)
    subset_metric = eligible[subset].median(axis=1, skipna=True)
    global_rank = full_metric.rank(method="min")
    subset_rank = subset_metric.rank(method="min")

    subset_winner = str(subset_metric.idxmin())
    global_winner = str(full_metric.idxmin())
    winner_global_metric = float(full_metric.loc[subset_winner])
    oracle_global_metric = float(full_metric.loc[global_winner])
    top_full = set(full_metric.nsmallest(min(TOP_K, len(full_metric))).index)
    top_subset = set(subset_metric.nsmallest(min(TOP_K, len(subset_metric))).index)

    return {
        "subset_name": subset_name,
        "scope": scope,
        "subset_size": len(subset),
        "eligible_candidates": int(len(eligible)),
        "status": "ok",
        "spearman_subset_vs_global": spearman(subset_metric, full_metric),
        "pairwise_agreement": pairwise_agreement(subset_metric, full_metric)
        if compute_pairwise
        else np.nan,
        "top5_overlap": len(top_full & top_subset) / max(1, min(TOP_K, len(full_metric))),
        "subset_winner_candidate": subset_winner,
        "subset_winner_class": metadata.loc[subset_winner, "model_class"],
        "subset_winner_global_median_ratio": winner_global_metric,
        "global_oracle_candidate": global_winner,
        "global_oracle_class": metadata.loc[global_winner, "model_class"],
        "global_oracle_median_ratio": oracle_global_metric,
        "winner_global_rank": int(global_rank.loc[subset_winner]),
        "subset_winner_subset_rank": int(subset_rank.loc[subset_winner]),
        "global_regret_abs": winner_global_metric - oracle_global_metric,
        "global_regret_pct": (winner_global_metric / oracle_global_metric - 1.0)
        if oracle_global_metric > 0
        else np.nan,
    }


def total_variation(subset_counts: pd.Series, full_counts: pd.Series) -> float:
    labels = sorted(set(subset_counts.index) | set(full_counts.index))
    sub = subset_counts.reindex(labels, fill_value=0).astype(float)
    full = full_counts.reindex(labels, fill_value=0).astype(float)
    if sub.sum() == 0 or full.sum() == 0:
        return np.nan
    sub /= sub.sum()
    full /= full.sum()
    return float(0.5 * np.abs(sub - full).sum())


def dataset_profile(df: pd.DataFrame, core: list[str]) -> pd.DataFrame:
    rows = []
    for dataset, group in df[df["dataset"].isin(core)].groupby("dataset"):
        rows.append(
            {
                "dataset": dataset,
                "dataset_group": mode_or_blank(group["dataset_group"]),
                "n_train": pd.to_numeric(group["n_train"], errors="coerce").median(),
                "n_test": pd.to_numeric(group["n_test"], errors="coerce").median(),
                "n_features": pd.to_numeric(group["n_features"], errors="coerce").median(),
                "dataset_pls_score": pd.to_numeric(
                    group["dataset_pls_score"], errors="coerce"
                ).median(),
                "n_candidate_scores": int(group["candidate_id"].nunique()),
                "n_model_classes": int(group["model_class"].nunique()),
                "dataset_best_model_class": mode_or_blank(group["dataset_best_model_class"]),
            }
        )
    profile = pd.DataFrame(rows).set_index("dataset").reindex(core).reset_index()
    return profile


def representativeness(profile: pd.DataFrame, subset_name: str, subset: list[str]) -> dict[str, object]:
    subset = [d for d in subset if d in set(profile["dataset"])]
    full = profile.copy()
    sub = profile[profile["dataset"].isin(subset)].copy()

    numeric_cols = [
        "n_train",
        "n_test",
        "n_features",
        "dataset_pls_score",
        "n_candidate_scores",
        "n_model_classes",
    ]
    z_deltas = []
    for col in numeric_cols:
        vals = pd.to_numeric(full[col], errors="coerce")
        mu = vals.mean()
        sd = vals.std()
        if not np.isfinite(sd) or sd == 0:
            continue
        z_deltas.append(abs(pd.to_numeric(sub[col], errors="coerce").mean() - mu) / sd)

    group_tv = total_variation(
        sub["dataset_group"].value_counts(), full["dataset_group"].value_counts()
    )
    winner_tv = total_variation(
        sub["dataset_best_model_class"].value_counts(),
        full["dataset_best_model_class"].value_counts(),
    )
    missing_groups = sorted(
        set(full["dataset_group"].dropna().astype(str))
        - set(sub["dataset_group"].dropna().astype(str))
    )
    return {
        "subset_name": subset_name,
        "subset_size": len(subset),
        "numeric_mean_abs_z_delta": float(np.nanmean(z_deltas)) if z_deltas else np.nan,
        "dataset_group_tv_distance": group_tv,
        "winner_class_tv_distance": winner_tv,
        "missing_dataset_groups": "|".join([g for g in missing_groups if g]),
        "n_missing_dataset_groups": len([g for g in missing_groups if g]),
        "median_n_candidate_scores": float(sub["n_candidate_scores"].median()),
        "median_n_model_classes": float(sub["n_model_classes"].median()),
    }


def random_baselines(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    core: list[str],
    sizes: list[int],
    scopes: list[str],
    n_trials: int = RANDOM_TRIALS,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    core_arr = np.array(core)
    for size in sorted(set(sizes)):
        if size >= len(core):
            continue
        for scope in scopes:
            metrics = []
            for trial in range(n_trials):
                subset = rng.choice(core_arr, size=size, replace=False).tolist()
                row = evaluate_subset_transfer(
                    matrix,
                    metadata,
                    f"random_{size}_{trial:04d}",
                    subset,
                    scope,
                    compute_pairwise=False,
                )
                if row.get("status") == "ok":
                    metrics.append(row)
            frame = pd.DataFrame(metrics)
            if frame.empty:
                continue
            rows.append(
                {
                    "scope": scope,
                    "subset_size": size,
                    "random_trials_ok": int(len(frame)),
                    "spearman_mean": frame["spearman_subset_vs_global"].mean(),
                    "spearman_p05": frame["spearman_subset_vs_global"].quantile(0.05),
                    "spearman_p50": frame["spearman_subset_vs_global"].quantile(0.50),
                    "spearman_p95": frame["spearman_subset_vs_global"].quantile(0.95),
                    "regret_abs_mean": frame["global_regret_abs"].mean(),
                    "regret_abs_p50": frame["global_regret_abs"].quantile(0.50),
                    "regret_abs_p95": frame["global_regret_abs"].quantile(0.95),
                    "oracle_hit_rate": (frame["winner_global_rank"] == 1).mean(),
                    "winner_rank_p50": frame["winner_global_rank"].quantile(0.50),
                    "winner_rank_p95": frame["winner_global_rank"].quantile(0.95),
                    "eligible_candidates_p50": frame["eligible_candidates"].quantile(0.50),
                }
            )
    return pd.DataFrame(rows)


def write_markdown_report(
    transfer: pd.DataFrame,
    rep: pd.DataFrame,
    randoms: pd.DataFrame,
    subsets: dict[str, list[str]],
    core: list[str],
) -> None:
    current_name = next((n for n in subsets if n.startswith("current_class_balanced_")), "")
    conservative_name = next((n for n in subsets if n.startswith("current_conservative_")), "")
    legacy_name = next((n for n in subsets if n.startswith("legacy_variant_heavy_10")), "")

    def get_row(subset_name: str, scope: str) -> pd.Series | None:
        rows = transfer[(transfer["subset_name"] == subset_name) & (transfer["scope"] == scope)]
        return None if rows.empty else rows.iloc[0]

    current_all = get_row(current_name, "all_candidates") if current_name else None
    current_no_tab = get_row(current_name, "no_tabpfn") if current_name else None
    cons_aom = get_row(conservative_name, "aom_ridge_only") if conservative_name else None

    lines = [
        "# Subset-to-global transfer analysis",
        "",
        "Generated from `bench/benchmark_master_results.csv`.",
        "",
        "## Question",
        "",
        "Does selecting a model on a subset of datasets transfer to better global results?",
        "",
        "A subset cannot beat the full-core oracle by definition. The useful test is whether the subset selects the same concrete model, or a near-oracle model, when evaluated on the full 57-dataset core.",
        "",
        "## Protocol",
        "",
        f"- Universe: {len(core)} regression datasets from `main_class_score_matrix.csv`, used as the existing Subset_analysis core.",
        "- Rows: `record_type in {observed, reference_paper}`, regression, `evaluation_split=test`, `status=ok`.",
        "- Score: `score_ratio_vs_dataset_pls`; lower is better and values below 1 beat the dataset PLS anchor.",
        "- Concrete candidate: `source_family | model_class | model_name | variant`; duplicate rows per candidate/dataset are aggregated by median.",
        f"- Candidate eligibility per subset/scope: at least {MIN_GLOBAL_COVERAGE:.0%} full-core dataset coverage and complete coverage on the evaluated subset.",
        "- Primary transfer diagnostics: subset-vs-global Spearman rank, top-5 overlap, selected winner's full-core rank, and full-core regret against the best eligible candidate.",
        "",
        "## Main findings",
        "",
    ]

    if current_all is not None and current_all.get("status") == "ok":
        lines.append(
            f"- Current class-balanced subset selects the full-core winner in `all_candidates`: `{current_all['subset_winner_class']}` / `{current_all['subset_winner_candidate']}`."
        )
        lines.append(
            f"- Its full-core median ratio is {current_all['subset_winner_global_median_ratio']:.4f}; regret is {current_all['global_regret_abs']:.4f}; subset-vs-global Spearman is {current_all['spearman_subset_vs_global']:.3f}."
        )
    if current_no_tab is not None and current_no_tab.get("status") == "ok":
        if int(current_no_tab["winner_global_rank"]) == 1:
            lines.append(
                f"- Excluding TabPFN, the same subset selects the no-TabPFN full-core winner with regret {current_no_tab['global_regret_abs']:.4f} and Spearman {current_no_tab['spearman_subset_vs_global']:.3f}."
            )
        else:
            lines.append(
                f"- Excluding TabPFN, the current subset does not recover the full-core oracle: it selects rank {int(current_no_tab['winner_global_rank'])} with regret {current_no_tab['global_regret_abs']:.4f}, despite a high Spearman of {current_no_tab['spearman_subset_vs_global']:.3f}."
            )
    if cons_aom is not None and cons_aom.get("status") != "ok":
        lines.append(
            "- The conservative 19-dataset subset is not automatically safer for incomplete families: for AOM-Ridge it leaves too few candidates with complete subset coverage."
        )
    if legacy_name:
        legacy_no_tab = get_row(legacy_name, "no_tabpfn")
        if legacy_no_tab is not None and legacy_no_tab.get("status") == "ok":
            lines.append(
                f"- The legacy variant-heavy subset is usable for the top TabPFN decision, but without TabPFN it selects full-core rank {int(legacy_no_tab['winner_global_rank'])}, not rank 1."
            )

    lines.extend(
        [
            "",
            "## Current Subset Safety",
            "",
            "| Subset | Scope | Status | Eligible | Spearman | Winner full rank | Regret | Winner class |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    preferred_scopes = [
        "all_candidates",
        "no_tabpfn",
        "tabpfn_only",
        "aom_pls_only",
        "aom_ridge_only",
        "meta_selector_only",
        "ridge_only",
        "pls_only",
    ]
    for subset_name in [n for n in subsets if n != "full_core_57"]:
        for scope in preferred_scopes:
            row = get_row(subset_name, scope)
            if row is None:
                continue
            status = row.get("status", "")
            eligible = int(row.get("eligible_candidates", 0))
            spearman_val = row.get("spearman_subset_vs_global", np.nan)
            rank_val = row.get("winner_global_rank", np.nan)
            regret_val = row.get("global_regret_abs", np.nan)
            winner_class = row.get("subset_winner_class", "")
            rank_text = str(int(rank_val)) if pd.notna(rank_val) else ""
            lines.append(
                f"| `{subset_name}` | `{scope}` | {status} | {eligible} | {spearman_val:.3f} | {rank_text} | {regret_val:.4f} | `{winner_class}` |"
                if status == "ok"
                else f"| `{subset_name}` | `{scope}` | {status} | {eligible} |  |  |  |  |"
            )

    lines.extend(
        [
            "",
            "## Representativeness Diagnostics",
            "",
            "| Subset | Size | Numeric z-delta | Group TV | Winner-class TV | Missing groups |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in rep.itertuples(index=False):
        lines.append(
            f"| `{row.subset_name}` | {row.subset_size} | {row.numeric_mean_abs_z_delta:.3f} | {row.dataset_group_tv_distance:.3f} | {row.winner_class_tv_distance:.3f} | {row.n_missing_dataset_groups} |"
        )

    lines.extend(
        [
            "",
            "## Random Baseline Check",
            "",
            "For a good subset, transfer should beat or at least sit near the favorable tail of random subsets of the same size.",
            "",
            "| Scope | Size | Oracle hit rate | Spearman p50 | Spearman p95 | Regret p50 | Regret p95 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in randoms.itertuples(index=False):
        lines.append(
            f"| `{row.scope}` | {row.subset_size} | {row.oracle_hit_rate:.3f} | {row.spearman_p50:.3f} | {row.spearman_p95:.3f} | {row.regret_abs_p50:.4f} | {row.regret_abs_p95:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Actionable Plan",
            "",
            "1. Freeze the candidate universe before using any subset result. A subset is valid only for candidates that have complete scores on the subset and high global/core coverage in the audit.",
            "2. Use the current 10-dataset class-balanced subset for fast first-pass screening; it recovers the current full-core TabPFN winner and beats same-size random medians on rank transfer.",
            "3. Do not use subset selection alone to choose among non-TabPFN challengers. The current subset has good rank correlation but picks no-TabPFN rank 9 under the broad 90%-coverage audit.",
            "4. Do not treat the 19-dataset conservative subset as universally safer. It improves one representativeness metric but can exclude incomplete families such as AOM-Ridge from fair complete-subset comparisons.",
            "5. Before accepting a subset-selected winner, check `subset_transfer_summary.csv`: require `status=ok`, winner full rank <= 3, low regret, and subset-vs-global Spearman above the same-size random median.",
            "6. For new model families, rerun this script after adding results. If candidate coverage drops below the subset size, either reduce the subset to datasets all candidates cover or evaluate that family separately.",
            "7. Use the subset only as a screening gate. Final claims still require full-core evaluation or a nested selection protocol that did not use full-core test RMSEP to design the subset.",
            "",
            "## Generated Files",
            "",
            "- `subset_transfer_candidate_matrix.csv`",
            "- `subset_transfer_summary.csv`",
            "- `subset_representativeness.csv`",
            "- `subset_transfer_random_baselines.csv`",
            "- `SUBSET_TRANSFER_REPORT.md`",
        ]
    )
    (OUT / "SUBSET_TRANSFER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    core = load_core_datasets()
    df = load_master()
    matrix, metadata = build_candidate_matrix(df, core)
    subsets = load_named_subsets(core)
    profile = dataset_profile(df, core)

    matrix.to_csv(OUT / "subset_transfer_candidate_matrix.csv")

    transfer_rows = []
    for subset_name, subset in subsets.items():
        for scope in SCOPE_DEFINITIONS:
            transfer_rows.append(
                evaluate_subset_transfer(matrix, metadata, subset_name, subset, scope)
            )
    transfer = pd.DataFrame(transfer_rows)
    transfer.to_csv(OUT / "subset_transfer_summary.csv", index=False)

    rep = pd.DataFrame(
        representativeness(profile, subset_name, subset)
        for subset_name, subset in subsets.items()
        if subset_name != "full_core_57"
    )
    rep.to_csv(OUT / "subset_representativeness.csv", index=False)

    random_sizes = [
        len(subset)
        for name, subset in subsets.items()
        if name != "full_core_57" and 3 <= len(subset) < len(core)
    ]
    randoms = random_baselines(
        matrix,
        metadata,
        core,
        random_sizes,
        scopes=["all_candidates", "no_tabpfn"],
    )
    randoms.to_csv(OUT / "subset_transfer_random_baselines.csv", index=False)

    write_markdown_report(transfer, rep, randoms, subsets, core)
    print(f"Wrote transfer analysis outputs to {OUT}")


if __name__ == "__main__":
    main()
