"""Phase 9 smoke-4: enhanced moe-view variants vs baseline K=3.

Variants:
- moe-view-soft-K3-baseline (constant NNLS gate, current top by TabPFN wins)
- moe-view-stacked-K3 (Ridge meta on [X_pca | OOF_preds])
- moe-view-stacked-K5
- moe-view-multiK-3-5-7 (average across K)
- moe-view-persample-K3 (per-sample argmax classifier gate)

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
if str(_AOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_AOM_ROOT))
if str(_MULTIVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_MULTIVIEW_ROOT))

from multiview.moe import AOMMoERegressor  # noqa: E402
from multiview.moe_advanced import (  # noqa: E402
    AOMMoEMultiK, AOMMoEPerSampleRouting, AOMMoEStacked,
)

from run_smoke4 import (  # noqa: E402
    SMOKE4_DATASETS, _load_csv_array, _load_csv_target,
    _existing_keys, _append, COLUMNS,
)


def _metrics(t0, yte, pred, k):
    pred = np.asarray(pred).ravel()
    return {
        "rmsep": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "n_components": int(k),
        "fit_time_s": float(time.perf_counter() - t0),
    }


def _runner_baseline_K3(Xtr, ytr, Xte, yte, seed, max_components):
    t0 = time.perf_counter()
    est = AOMMoERegressor(
        expert_layout="per_view", routing="soft", K=3,
        per_expert_components=min(10, max_components), random_state=seed,
    )
    est.fit(Xtr, ytr)
    return _metrics(t0, yte, est.predict(Xte), 10)


def _runner_stacked_K3(Xtr, ytr, Xte, yte, seed, max_components):
    t0 = time.perf_counter()
    est = AOMMoEStacked(
        expert_layout="per_view", K=3,
        per_expert_components=min(10, max_components),
        x_pca_components=5, meta_alpha=1.0, random_state=seed,
    )
    est.fit(Xtr, ytr)
    return _metrics(t0, yte, est.predict(Xte), 10)


def _runner_stacked_K5(Xtr, ytr, Xte, yte, seed, max_components):
    t0 = time.perf_counter()
    est = AOMMoEStacked(
        expert_layout="per_view", K=5,
        per_expert_components=min(10, max_components),
        x_pca_components=5, meta_alpha=1.0, random_state=seed,
    )
    est.fit(Xtr, ytr)
    return _metrics(t0, yte, est.predict(Xte), 10)


def _runner_multiK(Xtr, ytr, Xte, yte, seed, max_components):
    t0 = time.perf_counter()
    est = AOMMoEMultiK(
        K_list=(3, 5, 7),
        per_expert_components=min(10, max_components), random_state=seed,
    )
    est.fit(Xtr, ytr)
    return _metrics(t0, yte, est.predict(Xte), 10)


def _runner_persample_K3(Xtr, ytr, Xte, yte, seed, max_components):
    t0 = time.perf_counter()
    est = AOMMoEPerSampleRouting(
        expert_layout="per_view", K=3,
        per_expert_components=min(10, max_components),
        gate_classifier="logreg", random_state=seed,
    )
    est.fit(Xtr, ytr)
    return _metrics(t0, yte, est.predict(Xte), 10)


VARIANTS = [
    ("moe-view-soft-K3-baseline", _runner_baseline_K3),
    ("moe-view-stacked-K3", _runner_stacked_K3),
    ("moe-view-stacked-K5", _runner_stacked_K5),
    ("moe-view-multiK-3-5-7", _runner_multiK),
    ("moe-view-persample-K3", _runner_persample_K3),
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
        f"[phase9-smoke4] {len(smoke)} datasets x {len(VARIANTS)} variants "
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
                print(f"[phase9-smoke4] skip {key}")
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
                f"[phase9-smoke4] {cohort_row['dataset']:<48s} {label:<32s} "
                f"rmsep={row.get('rmsep', 'NA')} r2={row.get('r2', 'NA')} "
                f"t={row.get('fit_time_s', 'NA')}"
            )

    print(f"[phase9-smoke4] appended {n_appended} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
