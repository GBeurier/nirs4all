"""Phase 8 smoke-4: heterogeneous Ridge-stack + standalone TabPFN/AOM-Ridge.

Variants:
- tabpfn-standalone — base TabPFNRegressor (no stacking)
- aom-ridge-standalone — base AOMRidgeRegressor (parallel session)
- hetero-ridge-stack — Ridge-stack of {aom_pls, moe_preproc_soft,
  moe_view_soft, lazy_v2_aom, aom_ridge, tabpfn}
- hetero-nnls-stack — same bases, NNLS meta

Outputs append to smoke4_baseline.csv.
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
# Insert AOM_ROOT first, then MULTIVIEW_ROOT — multiview ends up at index 0
# so its `benchmarks` package wins over `bench/AOM_v0/benchmarks` (both
# have __init__.py).
if str(_AOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_AOM_ROOT))
if str(_MULTIVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_MULTIVIEW_ROOT))

from multiview.hetero_stack import (  # noqa: E402
    collect_hetero_bases, make_aom_ridge, make_tabpfn,
)
from multiview.stacking import StackingHybrid  # noqa: E402

from run_smoke4 import (  # noqa: E402
    SMOKE4_DATASETS, _load_csv_array, _load_csv_target,
    _existing_keys, _append, COLUMNS,
)


def _metrics(t0, yte, pred, n_components):
    pred = np.asarray(pred).ravel()
    return {
        "rmsep": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "n_components": int(n_components),
        "fit_time_s": float(time.perf_counter() - t0),
    }


def _runner_tabpfn(Xtr, ytr, Xte, yte, seed, max_components):
    est = make_tabpfn(seed=seed, n_max=5000)
    if est is None:
        return {"rmsep": float("nan"), "r2": float("nan"), "n_components": -1, "fit_time_s": 0.0,
                "selected_operators": "tabpfn unavailable"}
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)
    return _metrics(t0, yte, pred, -1)


def _runner_aom_ridge(Xtr, ytr, Xte, yte, seed, max_components):
    est = make_aom_ridge(seed=seed)
    if est is None:
        return {"rmsep": float("nan"), "r2": float("nan"), "n_components": -1, "fit_time_s": 0.0,
                "selected_operators": "aom_ridge unavailable"}
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)
    return _metrics(t0, yte, pred, -1)


def _runner_hetero_stack(Xtr, ytr, Xte, yte, seed, max_components, *, nonneg: bool = False):
    p = Xtr.shape[1]
    bases = collect_hetero_bases(
        seed=seed, max_components=max_components, p=p,
        include_aom_ridge=True, include_tabpfn=True, include_nicon=False,
    )
    est = StackingHybrid(
        base_estimators=bases, n_oof_folds=3, meta_alpha=1.0,
        random_state=seed, nonneg=nonneg,
    )
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)
    out = _metrics(t0, yte, pred, -1)
    out["selected_operators"] = ",".join(
        f"{n}:{w:+.3f}" for (n, _), w in zip(est.base_estimators, est.meta_weights_)
    )
    return out


def _runner_hetero_ridge(Xtr, ytr, Xte, yte, seed, max_components):
    return _runner_hetero_stack(Xtr, ytr, Xte, yte, seed, max_components, nonneg=False)


def _runner_hetero_nnls(Xtr, ytr, Xte, yte, seed, max_components):
    return _runner_hetero_stack(Xtr, ytr, Xte, yte, seed, max_components, nonneg=True)


VARIANTS = [
    ("tabpfn-standalone", _runner_tabpfn),
    ("aom-ridge-standalone", _runner_aom_ridge),
    ("hetero-ridge-stack", _runner_hetero_ridge),
    ("hetero-nnls-stack", _runner_hetero_nnls),
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
    results_path = workspace / "smoke4_baseline.csv"
    existing = _existing_keys(results_path)

    cohort = pd.read_csv(args.cohort)
    smoke = cohort[
        cohort["dataset"].isin(SMOKE4_DATASETS) & (cohort["status"] == "ok")
    ].copy()
    print(
        f"[smoke4-hetero] {len(smoke)} datasets x {len(VARIANTS)} variants "
        f"on seed {args.seed} -> {results_path}"
    )

    n_appended = 0
    for _, cohort_row in smoke.iterrows():
        Xtr = _load_csv_array(cohort_row["train_path"])
        Xte = _load_csv_array(cohort_row["test_path"])
        ytr = _load_csv_target(cohort_row["ytrain_path"]).astype(float)
        yte = _load_csv_target(cohort_row["ytest_path"]).astype(float)
        for label, runner in VARIANTS:
            key = (str(cohort_row["dataset"]), label, int(args.seed))
            if key in existing:
                print(f"[smoke4-hetero] skip {key}")
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
                f"[smoke4-hetero] {cohort_row['dataset']:<48s} {label:<24s} "
                f"rmsep={row.get('rmsep', 'NA')} r2={row.get('r2', 'NA')} "
                f"t={row.get('fit_time_s', 'NA')}"
            )

    print(f"[smoke4-hetero] appended {n_appended} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
