"""Phase 7 full-57: just moe-view-soft-K5 (fast, no stacking).

K=5 was the smoke-10 winner vs TabPFN-opt (5/10 wins). On Beer it hits
0.147 RMSEP, beating TabPFN-opt (0.152) for the first time.

Run separately so the slow stack variants don't block this fast one.

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
from benchmarks.run_smoke10_iterate2 import _runner_moe_view_K5  # noqa: E402


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
        f"[full57-K5] {len(valid)} datasets x 1 variant on seed {args.seed}"
    )

    n_appended = 0
    for _, cohort_row in valid.iterrows():
        try:
            Xtr = _load_csv_array(cohort_row["train_path"])
            Xte = _load_csv_array(cohort_row["test_path"])
            ytr = _load_csv_target(cohort_row["ytrain_path"]).astype(float)
            yte = _load_csv_target(cohort_row["ytest_path"]).astype(float)
        except Exception as exc:
            print(f"[full57-K5] ERROR loading {cohort_row['dataset']}: {exc}")
            continue
        label = "moe-view-soft-K5"
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
            metrics = _runner_moe_view_K5(Xtr, ytr, Xte, yte, args.seed, args.max_components)
            row = {**base, **metrics}
        except Exception as exc:
            row = {**base, "status": "error", "status_details": str(exc)[:200]}
        _append(results_path, row, COLUMNS)
        existing.add(key)
        n_appended += 1
        print(
            f"[full57-K5] {cohort_row['dataset']:<48s} "
            f"rmsep={row.get('rmsep', 'NA')} t={row.get('fit_time_s', 'NA')}"
        )

    print(f"[full57-K5] appended {n_appended} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
