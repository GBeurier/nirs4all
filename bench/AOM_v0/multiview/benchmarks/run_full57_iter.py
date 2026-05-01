"""Phase 7 full-57 iteration with the smoke-10 winners.

Variants:
- moe-view-soft-K5 (smoke-10: 5/10 vs TabPFN, beats TabPFN on Beer)
- ridge-stack-multiview (smoke-10 median champion 0.873)
- nnls-stack-multiview (NNLS variant, slightly different bias)

Outputs append to full57.csv.
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
    _load_csv_array, _load_csv_target, _existing_keys, _append, COLUMNS,
)
from benchmarks.run_smoke10_iterate2 import (  # noqa: E402
    _runner_moe_view_K5, _runner_ridge_stack, _runner_nnls_stack,
)


VARIANTS = [
    ("moe-view-soft-K5", _runner_moe_view_K5),
    ("ridge-stack-multiview", _runner_ridge_stack),
    ("nnls-stack-multiview", _runner_nnls_stack),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", default="bench/AOM_v0/multiview/results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-components", type=int, default=15)
    parser.add_argument(
        "--cohort", default="bench/AOM_v0/benchmarks/cohort_regression.csv",
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results_path = workspace / "full57.csv"
    existing = _existing_keys(results_path)

    cohort = pd.read_csv(args.cohort)
    valid = cohort[cohort["status"] == "ok"].copy()
    print(
        f"[full57-iter] {len(valid)} datasets x {len(VARIANTS)} variants "
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
            print(f"[full57-iter] ERROR loading {cohort_row['dataset']}: {exc}")
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
                row = {**base, "status": "error", "status_details": str(exc)[:200]}
            _append(results_path, row, COLUMNS)
            existing.add(key)
            n_appended += 1
            print(
                f"[full57-iter] {cohort_row['dataset']:<48s} {label:<26s} "
                f"rmsep={row.get('rmsep', 'NA')} t={row.get('fit_time_s', 'NA')}"
            )

    print(f"[full57-iter] appended {n_appended} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
