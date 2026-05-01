"""Full 57-dataset benchmark with the smoke-10 winners.

Variants chosen from smoke-10 by median rel-RMSEP (most consistent
multi-view winners, plus references and one niche specialist):

References:
- PLS-standard-numpy
- AOM-PLS-compact-numpy

Multi-view variants:
- moe-view-soft-pls (smoke-10: 7/10 wins, median 0.892)
- moe-preproc-soft-pls-compact (smoke-10: 9/10 wins, median 0.917)
- lazy-V2-AOM-combined-compact-holdout (smoke-10: 6/10 wins, median 0.957)
- lazy-V1-POP-blocks3-holdout (niche specialist for block2deg-style)

Block-sparse-V1 is excluded — too slow on n>1500 datasets without a faster
incremental engine path.

Outputs `bench/AOM_v0/multiview/results/full57.csv`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(os.path.abspath(__file__)).resolve()
_MULTIVIEW_ROOT = _HERE.parent.parent
_AOM_ROOT = _MULTIVIEW_ROOT.parent
if str(_AOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_AOM_ROOT))
if str(_MULTIVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_MULTIVIEW_ROOT))

from benchmarks.run_smoke4 import (  # noqa: E402
    _load_csv_array,
    _load_csv_target,
    _existing_keys,
    _append,
    COLUMNS,
    _run_pls_standard,
    _run_aom_pls,
)
from benchmarks.run_smoke4_phase2_lazy import (  # noqa: E402
    _runner_lazy_v1_pop_blocks_holdout,
    _runner_lazy_v2_aom_combined_holdout,
)
from benchmarks.run_smoke4_phase3_moe import (  # noqa: E402
    _runner_view_soft as _runner_moe_view_soft,
    _runner_preproc_soft as _runner_moe_preproc_soft,
)


VARIANTS = [
    ("PLS-standard-numpy", _run_pls_standard),
    ("AOM-PLS-compact-numpy", _run_aom_pls),
    ("lazy-V1-POP-blocks3-holdout", _runner_lazy_v1_pop_blocks_holdout),
    ("lazy-V2-AOM-combined-compact-holdout", _runner_lazy_v2_aom_combined_holdout),
    ("moe-view-soft-pls", _runner_moe_view_soft),
    ("moe-preproc-soft-pls-compact", _runner_moe_preproc_soft),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", default="bench/AOM_v0/multiview/results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-components", type=int, default=15)
    parser.add_argument(
        "--cohort",
        default="bench/AOM_v0/benchmarks/cohort_regression.csv",
    )
    parser.add_argument(
        "--dataset", default="", help="Optional single dataset name filter (debug)"
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results_path = workspace / "full57.csv"
    existing = _existing_keys(results_path)

    cohort = pd.read_csv(args.cohort)
    valid = cohort[cohort["status"] == "ok"].copy()
    if args.dataset:
        valid = valid[valid["dataset"] == args.dataset].copy()
    print(
        f"[full57] {len(valid)} datasets x {len(VARIANTS)} variants "
        f"on seed {args.seed} -> {results_path}"
    )

    n_appended = 0
    for _, cohort_row in valid.iterrows():
        try:
            Xtr = _load_csv_array(cohort_row["train_path"])
            Xte = _load_csv_array(cohort_row["test_path"])
            ytr = _load_csv_target(cohort_row["ytrain_path"]).astype(float)
            yte = _load_csv_target(cohort_row["ytest_path"]).astype(float)
        except Exception as exc:
            print(f"[full57] ERROR loading {cohort_row['dataset']}: {exc}")
            continue
        for label, runner in VARIANTS:
            key = (str(cohort_row["dataset"]), label, int(args.seed))
            if key in existing:
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
                f"[full57] {cohort_row['dataset']:<48s} {label:<42s} "
                f"rmsep={row.get('rmsep', 'NA')} "
                f"k={row.get('n_components', 'NA')} t={row.get('fit_time_s', 'NA')}"
            )

    print(f"[full57] appended {n_appended} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
