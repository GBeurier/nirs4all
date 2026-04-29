"""Extended benchmark: run on at least 20 regression datasets + classification.

Uses `criterion="holdout"` (single 20% internal validation split, seed=42)
which mirrors the production nirs4all AOM-PLS selection mechanism. This is
the apples-to-apples comparison the user asked for.

Variants run:

- AOM_v0 numpy (PLS-standard, AOM/POP nipals/simpls covariance, AOM-compact)
- nirs4all production AOMPLSRegressor and POPPLSRegressor (call-through)

Output: `bench/AOM_v0/benchmark_runs/extended/results.csv` (resumable).

The script picks the first 20 cohort-OK regression splits ordered by
n_train (smallest first) for tractable wall-clock. To run on all 61 splits,
pass `--limit 0`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.build_cohorts import build_classification_cohort, build_regression_cohort  # noqa: E402
from benchmarks.run_aompls_benchmark import (  # noqa: E402
    REGRESSION_VARIANTS,
    CLASSIFICATION_VARIANTS,
    _existing_keys,
    run_dataset,
)


# Variants to run by default in the extended benchmark.
EXT_REG_VARIANTS = [
    {"label": "PLS-standard-numpy", "kind": "regression", "selection": "none", "engine": "pls_standard", "operator_bank": "identity", "backend": "numpy"},
    {"label": "AOM-default-simpls-covariance-numpy", "kind": "regression", "selection": "global", "engine": "simpls_covariance", "operator_bank": "default", "backend": "numpy"},
    {"label": "AOM-compact-simpls-covariance-numpy", "kind": "regression", "selection": "global", "engine": "simpls_covariance", "operator_bank": "compact", "backend": "numpy"},
    {"label": "AOM-default-nipals-adjoint-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "default", "backend": "numpy"},
    {"label": "POP-simpls-covariance-numpy", "kind": "regression", "selection": "per_component", "engine": "simpls_covariance", "operator_bank": "compact", "backend": "numpy"},
    {"label": "POP-nipals-adjoint-numpy", "kind": "regression", "selection": "per_component", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy"},
    {"label": "ActiveSuperblock-simpls-numpy", "kind": "regression", "selection": "active_superblock", "engine": "simpls_covariance", "operator_bank": "default", "backend": "numpy"},
    {"label": "Superblock-raw-simpls-numpy", "kind": "regression", "selection": "superblock", "engine": "simpls_covariance", "operator_bank": "compact", "backend": "numpy"},
    {"label": "AOM-explorer-simpls-numpy", "kind": "regression", "selection": "explorer_global", "engine": "simpls_covariance", "operator_bank": "explorer", "backend": "numpy"},
    # Production baselines
    {"label": "nirs4all-AOM-PLS-default", "kind": "regression", "selection": "external", "engine": "nirs4all_aom", "operator_bank": "production_default", "backend": "numpy"},
    {"label": "nirs4all-POP-PLS-default", "kind": "regression", "selection": "external", "engine": "nirs4all_pop", "operator_bank": "production_compact", "backend": "numpy"},
    # Better-criterion variants on the compact bank only (PRESS/CV on the
    # 100-op default bank is impractically slow — 14s/dataset on Beer).
    {"label": "AOM-compact-press-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "criterion_override": "approx_press"},
    {"label": "AOM-compact-cv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "criterion_override": "cv"},
    # Non-linear / supervised preprocessing piped into AOM-PLS (TabPFN paper baseline)
    {"label": "SNV-AOM-default-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "default", "backend": "numpy", "preproc": "snv"},
    {"label": "MSC-AOM-default-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "default", "backend": "numpy", "preproc": "msc"},
    {"label": "OSC-AOM-default-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "default", "backend": "numpy", "preproc": "osc"},
    {"label": "SNV-AOM-compact-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "snv"},
    {"label": "MSC-AOM-compact-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "msc"},
    {"label": "OSC-AOM-compact-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "osc"},
    {"label": "SNV-OSC-AOM-default-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "default", "backend": "numpy", "preproc": "snv_osc"},
    {"label": "MSC-OSC-AOM-default-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "default", "backend": "numpy", "preproc": "msc_osc"},
]


def _build_full_pipeline_grid():
    """All combinations of {norm} x {baseline} x {osc} x {bank}.

    Mirrors the TabPFN paper's preprocessing search at
    `bench/tabpfn_paper/run_reg_pls.py:120-135`:

        norm     in {none, snv, msc, emsc1, emsc2}
        baseline in {none, asls}                       (Detrend lives in AOM bank)
        osc      in {none, osc1, osc2, osc3}
        bank     in {compact, default}

    Total 5 x 2 x 4 x 2 = 80. The two bare cases
    (none-none-none x {compact, default}) duplicate
    `AOM-compact-simpls-covariance-numpy` and
    `AOM-default-nipals-adjoint-numpy` from EXT_REG_VARIANTS, so they
    are skipped here; the remaining 78 entries are emitted. Variant
    labels follow the format `pipeline-<norm>-<baseline>-<osc>-<bank>`
    using `x` as the empty-step token.
    """
    norms = ["x", "snv", "msc", "emsc1", "emsc2"]
    baselines = ["x", "asls"]
    oscs = ["x", "osc1", "osc2", "osc3"]
    banks = ["compact", "default"]
    variants = []
    for bank in banks:
        for norm in norms:
            for baseline in baselines:
                for osc in oscs:
                    if norm == "x" and baseline == "x" and osc == "x":
                        continue  # bare bank already in EXT_REG_VARIANTS
                    label = f"pipeline-{norm}-{baseline}-{osc}-{bank}"
                    preproc_name = (
                        ("snv" if norm == "snv" else
                         "msc" if norm == "msc" else
                         "emsc1" if norm == "emsc1" else
                         "emsc2" if norm == "emsc2" else
                         "none")
                        + "+" +
                        ("asls" if baseline == "asls" else "none")
                        + "+" +
                        ("osc1" if osc == "osc1" else
                         "osc2" if osc == "osc2" else
                         "osc3" if osc == "osc3" else
                         "none")
                    )
                    variants.append({
                        "label": label,
                        "kind": "regression",
                        "selection": "global",
                        "engine": "nipals_adjoint",
                        "operator_bank": bank,
                        "backend": "numpy",
                        "preproc": preproc_name,
                    })
    return variants


# Append the exhaustive pipeline grid to EXT_REG_VARIANTS so that the
# benchmark emits both the curated variants above and the full grid.
EXT_REG_VARIANTS = EXT_REG_VARIANTS + _build_full_pipeline_grid()


# ---------------------------------------------------------------------------
# P1-P5 improvement variants (driven by docs/AOMPLS_ALGO_IMPROVEMENT_REPORT.md)
# ---------------------------------------------------------------------------
P_VARIANTS = [
    # P1: selector stability (cv5, repeated CV, one-SE rule)
    {"label": "AOM-compact-cv5-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "criterion_override": "cv", "cv_override": 5},
    {"label": "AOM-compact-repcv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "criterion_override": "cv", "cv_override": 3, "repeats_override": 3},
    {"label": "AOM-compact-oneSEcv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "criterion_override": "cv", "cv_override": 3, "one_se_override": True},
    {"label": "AOM-compact-cv5-oneSE-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "criterion_override": "cv", "cv_override": 5, "one_se_override": True},
    {"label": "AOM-compact-repcv3-oneSE-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "criterion_override": "cv", "cv_override": 3, "repeats_override": 3, "one_se_override": True},
    # Apply one-SE on the deployed-equivalent default bank too.
    {"label": "AOM-default-oneSEcv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "default", "backend": "numpy", "criterion_override": "cv", "cv_override": 3, "one_se_override": True},
    # P2: adaptive bank (family-pruned and response-dedup over the default
    # 100-op bank). Uses the same holdout criterion as production for a clean
    # apples-to-apples ablation against `nirs4all-AOM-PLS-default`.
    {"label": "AOM-family-pruned-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "family_pruned", "backend": "numpy"},
    {"label": "AOM-response-dedup-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "response_dedup", "backend": "numpy"},
    {"label": "AOM-family-pruned-cv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "family_pruned", "backend": "numpy", "criterion_override": "cv", "cv_override": 3},
    {"label": "AOM-response-dedup-cv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "response_dedup", "backend": "numpy", "criterion_override": "cv", "cv_override": 3},
    # P3: LocalSNV (windowed SNV) piped before AOM-compact.
    {"label": "LocalSNV-w31-AOM-compact", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "lsnv31"},
    {"label": "LocalSNV-w51-AOM-compact", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "lsnv51"},
    {"label": "LocalSNV-w101-AOM-compact", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "lsnv101"},
    # P4: regularised POP (lower Kmax, optional one-SE).
    {"label": "POP-K3-numpy", "kind": "regression", "selection": "per_component", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "max_components_override": 3},
    {"label": "POP-K5-numpy", "kind": "regression", "selection": "per_component", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "max_components_override": 5},
    {"label": "POP-K8-numpy", "kind": "regression", "selection": "per_component", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "max_components_override": 8},
    {"label": "POP-K8-cv3-numpy", "kind": "regression", "selection": "per_component", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "max_components_override": 8, "criterion_override": "cv", "cv_override": 3},
    {"label": "POP-K8-cv3-oneSE-numpy", "kind": "regression", "selection": "per_component", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "max_components_override": 8, "criterion_override": "cv", "cv_override": 3, "one_se_override": True},
    # P5: multi-view full (ActiveSuperblock with deep3 on all 57 datasets).
    {"label": "ActiveSuperblock-deep3-numpy", "kind": "regression", "selection": "active_superblock", "engine": "simpls_covariance", "operator_bank": "deep3", "backend": "numpy"},
    # SNV+ASLS+AOM-compact-cv3 (compose the two best ideas: ASLS preproc + CV-3 selector).
    {"label": "SNV-AOM-compact-cv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "snv", "criterion_override": "cv", "cv_override": 3},
    {"label": "ASLS-AOM-compact-cv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "none+asls+none", "criterion_override": "cv", "cv_override": 3},
    {"label": "SNV-AOM-compact-oneSE-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "snv", "criterion_override": "cv", "cv_override": 3, "one_se_override": True},
]
EXT_REG_VARIANTS = EXT_REG_VARIANTS + P_VARIANTS


# ---------------------------------------------------------------------------
# Stabilisation variants: combine the two winners discovered in the P run.
# The benchmark found two strong new variants on the 57-dataset cohort:
#   * ASLS-AOM-compact-cv3 (median 0.978, 38/57 wins)
#   * AOM-compact-repcv3   (median 0.984, 39/57 wins)
# The combinations below test whether stacking these mechanisms (ASLS pre-
# processing + repeated CV criterion + one-SE shrinkage) compounds further.
# ---------------------------------------------------------------------------
STABILIZE_VARIANTS = [
    # ASLS preproc + repeated CV
    {"label": "ASLS-AOM-compact-repcv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "none+asls+none", "criterion_override": "cv", "cv_override": 3, "repeats_override": 3},
    # ASLS preproc + repeated CV + one-SE
    {"label": "ASLS-AOM-compact-repcv3-oneSE-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "none+asls+none", "criterion_override": "cv", "cv_override": 3, "repeats_override": 3, "one_se_override": True},
    # SNV preproc + repeated CV (parallel to ASLS+repcv3)
    {"label": "SNV-AOM-compact-repcv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "snv", "criterion_override": "cv", "cv_override": 3, "repeats_override": 3},
    # SNV+ASLS combined preproc + repeated CV
    {"label": "SNV-ASLS-AOM-compact-repcv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "snv+asls+none", "criterion_override": "cv", "cv_override": 3, "repeats_override": 3},
    # ASLS + cv5 (alternative to repcv3)
    {"label": "ASLS-AOM-compact-cv5-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "none+asls+none", "criterion_override": "cv", "cv_override": 5},
    # ASLS + family-pruned bank (instead of compact) with cv3
    {"label": "ASLS-AOM-family-pruned-cv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "family_pruned", "backend": "numpy", "preproc": "none+asls+none", "criterion_override": "cv", "cv_override": 3},
    # ASLS + response-dedup bank + cv3
    {"label": "ASLS-AOM-response-dedup-cv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "response_dedup", "backend": "numpy", "preproc": "none+asls+none", "criterion_override": "cv", "cv_override": 3},
]
EXT_REG_VARIANTS = EXT_REG_VARIANTS + STABILIZE_VARIANTS


# ---------------------------------------------------------------------------
# HPO variants: per-dataset Optuna search over (norm, asls_lambda, asls_p,
# cv, max_components, one_se_rule) on the three top banks. Reported fit
# time is the SUM of HPO search time and final-fit time, so the reported
# cost is directly comparable to Ridge-HPO and TabPFN-opt timings.
# ---------------------------------------------------------------------------
HPO_VARIANTS = [
    {"label": "HPO-AOM-compact-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "hpo": True, "hpo_trials": 25},
    {"label": "HPO-AOM-family-pruned-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "family_pruned", "backend": "numpy", "hpo": True, "hpo_trials": 25},
    {"label": "HPO-AOM-response-dedup-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "response_dedup", "backend": "numpy", "hpo": True, "hpo_trials": 25},
]
EXT_REG_VARIANTS = EXT_REG_VARIANTS + HPO_VARIANTS

EXT_CLF_VARIANTS = [
    {"label": "PLS-DA-standard", "kind": "classification", "selection": "none", "engine": "pls_standard", "operator_bank": "identity", "backend": "numpy"},
    {"label": "AOM-PLS-DA-default-simpls-covariance", "kind": "classification", "selection": "global", "engine": "simpls_covariance", "operator_bank": "default", "backend": "numpy"},
    {"label": "POP-PLS-DA-simpls-covariance", "kind": "classification", "selection": "per_component", "engine": "simpls_covariance", "operator_bank": "compact", "backend": "numpy"},
]


def _select_cohort(cohort_path: str, limit: int = 20, max_n: int = 1500) -> pd.DataFrame:
    df = pd.read_csv(cohort_path)
    df_ok = df[df["status"] == "ok"].copy()
    if df_ok.empty:
        return df_ok
    df_ok = df_ok[df_ok["n_train"].fillna(0) <= max_n].sort_values("n_train")
    if limit > 0:
        df_ok = df_ok.head(limit)
    return df_ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="bench/AOM_v0/benchmark_runs/extended")
    parser.add_argument("--limit", type=int, default=20, help="Max datasets per task (0 = all)")
    parser.add_argument("--max-n-train", type=int, default=1500, help="Skip datasets larger than this")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--criterion", default="holdout")
    parser.add_argument("--max-components", type=int, default=15)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--include-classification", action="store_true")
    parser.add_argument("--variants", default="", help="Comma-separated variant labels; empty = all extended")
    args = parser.parse_args(argv)
    cohort_reg_path = "bench/AOM_v0/benchmarks/cohort_regression.csv"
    cohort_clf_path = "bench/AOM_v0/benchmarks/cohort_classification.csv"
    if not Path(cohort_reg_path).exists():
        build_regression_cohort(out_path=cohort_reg_path)
    if not Path(cohort_clf_path).exists():
        build_classification_cohort(out_path=cohort_clf_path)
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results_reg = workspace / "results.csv"
    results_clf = workspace / "results_classification.csv"
    smoke_reg = _select_cohort(cohort_reg_path, limit=args.limit, max_n=args.max_n_train)
    print(f"[extended] regression datasets: {len(smoke_reg)}")
    variants = EXT_REG_VARIANTS
    if args.variants:
        wanted = set([v.strip() for v in args.variants.split(",")])
        variants = [v for v in variants if v["label"] in wanted]
    existing = _existing_keys(results_reg)
    total = 0
    for _, row in smoke_reg.iterrows():
        n = run_dataset(
            cohort_row=row,
            variants=variants,
            results_path=results_reg,
            seeds=[args.seed],
            criterion=args.criterion,
            max_components=args.max_components,
            cv=args.cv,
            classification=False,
            existing_keys=existing,
        )
        total += n
        print(f"[ext reg] {row['database_name']}/{row['dataset']} (n_train={row['n_train']}) +{n} rows")
    print(f"[ext reg] wrote {total} new rows -> {results_reg}")
    if args.include_classification:
        smoke_clf = _select_cohort(cohort_clf_path, limit=10, max_n=args.max_n_train)
        print(f"[extended] classification datasets: {len(smoke_clf)}")
        existing_clf = _existing_keys(results_clf)
        total_c = 0
        for _, row in smoke_clf.iterrows():
            n = run_dataset(
                cohort_row=row,
                variants=EXT_CLF_VARIANTS,
                results_path=results_clf,
                seeds=[args.seed],
                criterion=args.criterion,
                max_components=args.max_components,
                cv=args.cv,
                classification=True,
                existing_keys=existing_clf,
            )
            total_c += n
            print(f"[ext clf] {row['database_name']}/{row['dataset']} +{n} rows")
        print(f"[ext clf] wrote {total_c} new rows -> {results_clf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
