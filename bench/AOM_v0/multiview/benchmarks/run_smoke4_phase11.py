"""Phase 11 smoke-4: Codex-reviewed super-learner variants.

Codex review priorities (in order):
- trimmed-mean-4 — drop top/bottom, average middle. Zero-cost robust.
- nnls-stack-atoms — NNLS simplex on 4 atom bases (no recipes).
- nnls-stack-calibrated — NNLS + per-base shrinkage calibration.
- adaptive-superlearner — n_train threshold: <100 recipe-select, >=100 NNLS.

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

from run_smoke4 import (  # noqa: E402
    COLUMNS,
    SMOKE4_DATASETS,
    _append,
    _existing_keys,
    _load_csv_array,
    _load_csv_target,
)

from multiview.atoms import (  # noqa: E402
    AOMMoEMultiK,
    AOMMoERegressor,
    AOMPLSRegressor,
    LazyV2AOM,
)
from multiview.moe_advanced import MultiVariantMeanEnsemble  # noqa: E402
from multiview.super_learner import (  # noqa: E402
    AdaptiveSuperLearner,
    NNLSSimplexStacker,
    TrimmedMeanEnsemble,
)


def _metrics(t0, yte, pred):
    pred = np.asarray(pred).ravel()
    return {
        "rmsep": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "n_components": 10,
        "fit_time_s": float(time.perf_counter() - t0),
    }


def _atom_bases(seed, max_components):
    """The 4 atom bases per Codex review."""
    return [
        ("multiK-3-5-7", AOMMoEMultiK(
            K_list=(3, 5, 7), per_expert_components=10, random_state=seed,
        )),
        ("moe-preproc-soft", AOMMoERegressor(
            expert_layout="per_preproc", routing="soft",
            bank_name="compact", per_expert_components=min(10, max_components),
            random_state=seed,
        )),
        ("lazy-V2-AOM", LazyV2AOM(max_components=max_components, random_state=seed)),
        ("aom-pls-compact", AOMPLSRegressor(
            n_components="auto", max_components=max_components,
            engine="simpls_covariance", selection="global",
            criterion="holdout", operator_bank="compact", random_state=seed,
        )),
    ]


def _runner_trimmed_mean_4(Xtr, ytr, Xte, yte, seed, max_components):
    bases = _atom_bases(seed, max_components)
    t0 = time.perf_counter()
    est = TrimmedMeanEnsemble(bases=bases, n_drop=1)
    est.fit(Xtr, ytr)
    return _metrics(t0, yte, est.predict(Xte))


def _runner_nnls_stack_atoms(Xtr, ytr, Xte, yte, seed, max_components):
    bases = _atom_bases(seed, max_components)
    t0 = time.perf_counter()
    est = NNLSSimplexStacker(
        bases=bases, n_oof_folds=3, calibrate=False,
        min_margin=0.005, random_state=seed,
    )
    est.fit(Xtr, ytr)
    return _metrics(t0, yte, est.predict(Xte))


def _runner_nnls_stack_calibrated(Xtr, ytr, Xte, yte, seed, max_components):
    bases = _atom_bases(seed, max_components)
    t0 = time.perf_counter()
    est = NNLSSimplexStacker(
        bases=bases, n_oof_folds=3, calibrate=True,
        min_margin=0.005, random_state=seed,
    )
    est.fit(Xtr, ytr)
    return _metrics(t0, yte, est.predict(Xte))


def _runner_adaptive(Xtr, ytr, Xte, yte, seed, max_components):
    atoms = _atom_bases(seed, max_components)
    recipes = [
        ("multiK-3-5-7", AOMMoEMultiK(
            K_list=(3, 5, 7), per_expert_components=10, random_state=seed,
        )),
        ("multiK-wide", AOMMoEMultiK(
            K_list=(2, 3, 4, 5, 7, 10), per_expert_components="auto", random_state=seed,
        )),
        ("mean-ensemble-4", MultiVariantMeanEnsemble(bases=atoms)),
        ("aom-pls-compact", AOMPLSRegressor(
            n_components="auto", max_components=max_components,
            engine="simpls_covariance", selection="global",
            criterion="holdout", operator_bank="compact", random_state=seed,
        )),
    ]
    # Light atoms for huge n: drop lazy-V2-AOM (slowest atom) to keep cost
    # bounded on Chla+b-species/LUCAS-scale datasets.
    light_atoms = [a for a in atoms if a[0] != "lazy-V2-AOM"]
    t0 = time.perf_counter()
    est = AdaptiveSuperLearner(
        atoms=atoms, recipes=recipes, light_atoms=light_atoms,
        small_threshold=100, big_threshold=200, huge_threshold=3000,
        n_oof_folds=3, calibrate=False, random_state=seed,
    )
    est.fit(Xtr, ytr)
    return _metrics(t0, yte, est.predict(Xte))


VARIANTS = [
    ("trimmed-mean-4", _runner_trimmed_mean_4),
    ("nnls-stack-atoms", _runner_nnls_stack_atoms),
    ("nnls-stack-calibrated", _runner_nnls_stack_calibrated),
    ("adaptive-super-learner", _runner_adaptive),
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
    print(f"[phase11-smoke4] {len(smoke)} datasets x {len(VARIANTS)} variants")

    n_appended = 0
    for _, cohort_row in smoke.iterrows():
        Xtr = _load_csv_array(cohort_row["train_path"])
        Xte = _load_csv_array(cohort_row["test_path"])
        ytr = _load_csv_target(cohort_row["ytrain_path"]).astype(float)
        yte = _load_csv_target(cohort_row["ytest_path"]).astype(float)
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
                f"[phase11-smoke4] {cohort_row['dataset']:<48s} {label:<26s} "
                f"rmsep={row.get('rmsep', 'NA')} t={row.get('fit_time_s', 'NA')}"
            )

    print(f"[phase11-smoke4] appended {n_appended} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
