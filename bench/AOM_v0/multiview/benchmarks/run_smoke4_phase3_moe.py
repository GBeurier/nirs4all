"""Phase 3 MoE variants on smoke-4.

Hard / soft MoE with PLS experts (per-view K=3 / per-preproc compact).
Outputs appended to `bench/AOM_v0/multiview/results/smoke4_baseline.csv`.
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

from multiview.moe import AOMMoERegressor  # noqa: E402

from benchmarks.run_smoke4 import (  # noqa: E402
    SMOKE4_DATASETS,
    _load_csv_array,
    _load_csv_target,
    _existing_keys,
    _append,
    COLUMNS,
)


def _run_moe(
    *, Xtr, ytr, Xte, yte, seed,
    expert_layout, routing, K=3, bank_name="compact",
    per_expert_components=10,
):
    t0 = time.perf_counter()
    est = AOMMoERegressor(
        expert_layout=expert_layout,
        routing=routing,
        K=K,
        bank_name=bank_name,
        per_expert_components=per_expert_components,
        random_state=seed,
    )
    est.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    pred = est.predict(Xte).ravel()
    gate = est.get_gate_weights()
    return {
        "rmsep": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "n_components": int(per_expert_components),
        "fit_time_s": float(fit_time),
        "selected_operators": ",".join(
            f"e{i}={w:.3f}" for i, w in enumerate(gate) if w > 1e-3
        ),
    }


def _runner_view_hard(Xtr, ytr, Xte, yte, seed, max_components):
    return _run_moe(
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, seed=seed,
        expert_layout="per_view", routing="hard", K=3,
        per_expert_components=min(10, max_components),
    )


def _runner_view_soft(Xtr, ytr, Xte, yte, seed, max_components):
    return _run_moe(
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, seed=seed,
        expert_layout="per_view", routing="soft", K=3,
        per_expert_components=min(10, max_components),
    )


def _runner_preproc_hard(Xtr, ytr, Xte, yte, seed, max_components):
    return _run_moe(
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, seed=seed,
        expert_layout="per_preproc", routing="hard", bank_name="compact",
        per_expert_components=min(10, max_components),
    )


def _runner_preproc_soft(Xtr, ytr, Xte, yte, seed, max_components):
    return _run_moe(
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, seed=seed,
        expert_layout="per_preproc", routing="soft", bank_name="compact",
        per_expert_components=min(10, max_components),
    )


VARIANTS = [
    ("moe-view-hard-pls", _runner_view_hard),
    ("moe-view-soft-pls", _runner_view_soft),
    ("moe-preproc-hard-pls-compact", _runner_preproc_hard),
    ("moe-preproc-soft-pls-compact", _runner_preproc_soft),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", default="bench/AOM_v0/multiview/results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-components", type=int, default=10)
    parser.add_argument(
        "--cohort",
        default="bench/AOM_v0/benchmarks/cohort_regression.csv",
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
        f"[smoke4-phase3-moe] {len(smoke)} datasets x {len(VARIANTS)} variants "
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
                print(f"[smoke4-phase3-moe] skip {key}")
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
                f"[smoke4-phase3-moe] {cohort_row['dataset']:<48s} {label:<32s} "
                f"rmsep={row.get('rmsep', 'NA')} r2={row.get('r2', 'NA')} "
                f"k={row.get('n_components', 'NA')} t={row.get('fit_time_s', 'NA')}"
            )

    print(f"[smoke4-phase3-moe] appended {n_appended} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
