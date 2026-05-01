"""Smoke-10 escalation runner for the multi-view winners on the
10-dataset smoke cohort.

Variants (defined to keep CSV variant-column stable across phases):

References:
- PLS-standard-numpy, AOM-PLS-compact-numpy, MBPLS-blocks3-vanilla

Phase 2 winners:
- lazy-V1-POP-blocks3-holdout
- lazy-V2-AOM-combined-compact-holdout
- block-sparse-V1-blocks3-holdout (slow on n>2000 datasets)

Phase 3 winners:
- moe-view-soft-pls
- moe-preproc-soft-pls-compact

Use `--skip-slow` to drop the block-sparse variant on datasets with
n_train > 1500 (Chla+b series, N_woOutlier).

Outputs `bench/AOM_v0/multiview/results/smoke10.csv`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(os.path.abspath(__file__)).resolve()
_MULTIVIEW_ROOT = _HERE.parent.parent
_AOM_ROOT = _MULTIVIEW_ROOT.parent
if str(_AOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_AOM_ROOT))
if str(_MULTIVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_MULTIVIEW_ROOT))

from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402

from benchmarks.run_smoke4 import (  # noqa: E402
    _load_csv_array,
    _load_csv_target,
    _existing_keys,
    _append,
    COLUMNS,
    _run_pls_standard,
    _run_aom_pls,
    _run_mbpls_blocks3,
)
from benchmarks.run_smoke4_phase2_lazy import (  # noqa: E402
    _runner_lazy_v1_pop_blocks_holdout,
    _runner_lazy_v2_aom_combined_holdout,
)
from benchmarks.run_smoke4_phase2_blocksparse import (  # noqa: E402
    _runner_v1_blocks_holdout as _runner_blocksparse_v1,
)
from benchmarks.run_smoke4_phase3_moe import (  # noqa: E402
    _runner_view_soft as _runner_moe_view_soft,
    _runner_preproc_soft as _runner_moe_preproc_soft,
)


SMOKE10_DATASETS = [
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

SLOW_THRESHOLD_N = 1500


VARIANTS = [
    ("PLS-standard-numpy", _run_pls_standard, False),
    ("AOM-PLS-compact-numpy", _run_aom_pls, False),
    ("MBPLS-blocks3-vanilla", _run_mbpls_blocks3, False),
    ("lazy-V1-POP-blocks3-holdout", _runner_lazy_v1_pop_blocks_holdout, False),
    ("lazy-V2-AOM-combined-compact-holdout", _runner_lazy_v2_aom_combined_holdout, False),
    ("block-sparse-V1-blocks3-holdout", _runner_blocksparse_v1, True),  # slow on n>1500
    ("moe-view-soft-pls", _runner_moe_view_soft, False),
    ("moe-preproc-soft-pls-compact", _runner_moe_preproc_soft, False),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", default="bench/AOM_v0/multiview/results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-components", type=int, default=15)
    parser.add_argument("--skip-slow", action="store_true")
    parser.add_argument(
        "--cohort",
        default="bench/AOM_v0/benchmarks/cohort_regression.csv",
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results_path = workspace / "smoke10.csv"
    existing = _existing_keys(results_path)

    cohort = pd.read_csv(args.cohort)
    smoke = cohort[
        cohort["dataset"].isin(SMOKE10_DATASETS) & (cohort["status"] == "ok")
    ].copy()
    print(
        f"[smoke10] {len(smoke)} datasets x {len(VARIANTS)} variants "
        f"on seed {args.seed} -> {results_path}"
    )

    n_appended = 0
    for _, cohort_row in smoke.iterrows():
        Xtr = _load_csv_array(cohort_row["train_path"])
        Xte = _load_csv_array(cohort_row["test_path"])
        ytr = _load_csv_target(cohort_row["ytrain_path"]).astype(float)
        yte = _load_csv_target(cohort_row["ytest_path"]).astype(float)
        n_train = Xtr.shape[0]
        for label, runner, slow in VARIANTS:
            if slow and (args.skip_slow or n_train > SLOW_THRESHOLD_N):
                # Auto-skip slow variants on big datasets.
                continue
            key = (str(cohort_row["dataset"]), label, int(args.seed))
            if key in existing:
                print(f"[smoke10] skip {key}")
                continue
            base = {
                "database_name": cohort_row.get("database_name", ""),
                "dataset": cohort_row["dataset"],
                "variant": label,
                "seed": int(args.seed),
                "n_train": int(Xtr.shape[0]),
                "n_test": int(Xte.shape[0]),
                "n_features": int(Xtr.shape[1]),
                "status": "ok",
            }
            try:
                metrics = runner(Xtr, ytr, Xte, yte, args.seed, args.max_components)
                row = {**base, **metrics}
            except Exception as exc:
                row = {
                    **base,
                    "status": "error",
                    "status_details": str(exc)[:200],
                }
            _append(results_path, row, COLUMNS)
            existing.add(key)
            n_appended += 1
            print(
                f"[smoke10] {cohort_row['dataset']:<48s} {label:<42s} "
                f"rmsep={row.get('rmsep', 'NA')} r2={row.get('r2', 'NA')} "
                f"k={row.get('n_components', 'NA')} t={row.get('fit_time_s', 'NA')}"
            )

    print(f"[smoke10] appended {n_appended} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
