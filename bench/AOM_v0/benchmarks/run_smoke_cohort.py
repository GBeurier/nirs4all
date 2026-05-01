"""Fast iteration benchmark on the 11-dataset smoke cohort.

The smoke cohort is a curated subset of the 57-dataset TabPFN/NIRS
cohort that the user verified is representative of the full diversity
(small/medium/large n, narrow/wide p, chemistry vs biological targets).
Use this for fast iteration on new variants before launching the full
57-dataset benchmark.

Datasets:
- All_manure_P2O5_SPXY_strat_Manure_type
- Fv_Fm_grp70_30
- ta_groupSampleID_stratDateVar_balRows
- WOOD_N_402_Olale
- An_spxyG70_30_byCultivar_MicroNIR
- All_manure_CaO_SPXY_strat_Manure_type
- DIESEL_bp50_246_b-a
- Malaria_Sporozoite_229_Maia
- Chla+b_spxyG_block2deg
- DIESEL_bp50_246_hla-b
- Beef_Marbling_RandomSplit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.run_aompls_benchmark import _existing_keys, run_dataset  # noqa: E402


SMOKE_DATASETS = [
    "All_manure_MgO_SPXY_strat_Manure_type",
    "An_spxyG70_30_byCultivar_NeoSpectra",
    "TIC_spxy70",
    "Chla+b_spxyG_species",
    "ALPINE_P_291_KS",
    "Beer_OriginalExtract_60_YbaseSplit",
    "All_manure_Total_N_SPXY_strat_Manure_type",
    "Chla+b_spxyG_block2deg",
    "N_woOutlier",
    "grapevine_chloride_556_KS",
]


def _spxy_factory(n_splits):
    def _factory(seed):
        from nirs4all.operators.splitters import SPXYFold
        return SPXYFold(n_splits=int(n_splits), random_state=int(seed))
    return _factory


SMOKE_VARIANTS = [
    # Reference: PLS-standard
    {"label": "PLS-standard-numpy", "kind": "regression", "selection": "none", "engine": "pls_standard", "operator_bank": "identity", "backend": "numpy"},
    # Champion (random KFold + ASLS)
    {"label": "ASLS-AOM-compact-cv5-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "none+asls+none", "criterion_override": "cv", "cv_override": 5},
    {"label": "ASLS-AOM-family-pruned-cv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "family_pruned", "backend": "numpy", "preproc": "none+asls+none", "criterion_override": "cv", "cv_override": 3},
    {"label": "ASLS-AOM-response-dedup-cv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "response_dedup", "backend": "numpy", "preproc": "none+asls+none", "criterion_override": "cv", "cv_override": 3},
    # SPXY variants (lost from full CSV — re-running on smoke for verification)
    {"label": "SPXY-AOM-compact-cv5-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy", "preproc": "none+asls+none", "criterion_override": "cv", "cv_override": 5, "cv_splitter_factory": _spxy_factory(5)},
    {"label": "SPXY-AOM-family-pruned-cv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "family_pruned", "backend": "numpy", "preproc": "none+asls+none", "criterion_override": "cv", "cv_override": 3, "cv_splitter_factory": _spxy_factory(3)},
    {"label": "SPXY-AOM-response-dedup-cv3-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "response_dedup", "backend": "numpy", "preproc": "none+asls+none", "criterion_override": "cv", "cv_override": 3, "cv_splitter_factory": _spxy_factory(3)},
    # Production reference
    {"label": "nirs4all-AOM-PLS-default", "kind": "regression", "selection": "external", "engine": "nirs4all_aom", "operator_bank": "production_default", "backend": "numpy"},
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="bench/AOM_v0/benchmark_runs/smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-components", type=int, default=15)
    args = parser.parse_args(argv)

    cohort_path = "bench/AOM_v0/benchmarks/cohort_regression.csv"
    cohort = pd.read_csv(cohort_path)
    smoke = cohort[cohort["dataset"].isin(SMOKE_DATASETS) & (cohort["status"] == "ok")].copy()
    print(f"[smoke] running {len(smoke)} datasets x {len(SMOKE_VARIANTS)} variants")

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results = workspace / "results.csv"
    existing = _existing_keys(results)
    total = 0
    for _, row in smoke.iterrows():
        n = run_dataset(
            cohort_row=row,
            variants=SMOKE_VARIANTS,
            results_path=results,
            seeds=[args.seed],
            criterion="holdout",
            max_components=args.max_components,
            cv=3,
            classification=False,
            existing_keys=existing,
        )
        total += n
        print(f"[smoke] {row['database_name']}/{row['dataset']} +{n} rows")
    print(f"[smoke] total: {total} new rows -> {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
