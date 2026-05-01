"""TabPFN-standalone on the full-57 cohort.

Single TabPFN fit per dataset (no stacking, no OOF). Bigger datasets are
subsampled to n_max=5000 (TabPFN's pretraining limit). Output appended to
full57.csv as variant `tabpfn-standalone`.

After this, we can compute test-time ensembles of TabPFN + multi-view by
averaging predictions per dataset (without retraining).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

_HERE = Path(os.path.abspath(__file__)).resolve()
_MULTIVIEW_ROOT = _HERE.parent.parent
_AOM_ROOT = _MULTIVIEW_ROOT.parent
if str(_AOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_AOM_ROOT))
if str(_MULTIVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_MULTIVIEW_ROOT))

from multiview.hetero_stack import make_tabpfn  # noqa: E402

from run_smoke4 import (  # noqa: E402
    _load_csv_array, _load_csv_target, _existing_keys, _append, COLUMNS,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", default="bench/AOM_v0/multiview/results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cohort", default="bench/AOM_v0/benchmarks/cohort_regression.csv",
    )
    parser.add_argument("--n-max", type=int, default=5000)
    parser.add_argument("--max-time-s", type=int, default=600,
                        help="skip dataset if est fit time > limit (default 10 min)")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results_path = workspace / "full57.csv"
    existing = _existing_keys(results_path)

    cohort = pd.read_csv(args.cohort)
    valid = cohort[cohort["status"] == "ok"].copy()
    print(f"[tabpfn-full57] {len(valid)} datasets, n_max={args.n_max}")

    n_appended = 0
    for _, cohort_row in valid.iterrows():
        try:
            Xtr = _load_csv_array(cohort_row["train_path"])
            Xte = _load_csv_array(cohort_row["test_path"])
            ytr = _load_csv_target(cohort_row["ytrain_path"]).astype(float)
            yte = _load_csv_target(cohort_row["ytest_path"]).astype(float)
        except Exception as exc:
            print(f"[tabpfn-full57] ERROR loading {cohort_row['dataset']}: {exc}")
            continue
        label = "tabpfn-standalone"
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
            est = make_tabpfn(seed=args.seed, n_max=args.n_max)
            t0 = time.perf_counter()
            est.fit(Xtr, ytr)
            pred = est.predict(Xte).ravel()
            fit_time = time.perf_counter() - t0
            row = {
                **base,
                "rmsep": float(np.sqrt(mean_squared_error(yte, pred))),
                "r2": float(r2_score(yte, pred)),
                "n_components": -1,
                "fit_time_s": float(fit_time),
            }
        except Exception as exc:
            row = {**base, "status": "error", "status_details": str(exc)[:200]}
        _append(results_path, row, COLUMNS)
        existing.add(key)
        n_appended += 1
        print(
            f"[tabpfn-full57] {cohort_row['dataset']:<48s} "
            f"rmsep={row.get('rmsep', 'NA')} t={row.get('fit_time_s', 'NA')}"
        )

    print(f"[tabpfn-full57] appended {n_appended} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
