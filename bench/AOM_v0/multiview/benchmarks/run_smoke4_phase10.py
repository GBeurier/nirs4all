"""Phase 10 smoke-4: wider multiK + multi-view mean ensemble.

Variants:
- moe-view-multiK-2-3-4-5-7-10 (wider K sweep, 6 K values)
- moe-view-multiK-3-5-7-auto (Phase 9 winner with adaptive components)
- mean-ensemble-3 (avg of multiK-3-5-7, moe-preproc-soft, lazy-V2-AOM)
- mean-ensemble-4 (avg of 4 top variants including AOM-PLS)

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

from aompls.estimators import AOMPLSRegressor  # noqa: E402

from multiview.moe import AOMMoERegressor  # noqa: E402
from multiview.moe_advanced import AOMMoEMultiK, MultiVariantMeanEnsemble  # noqa: E402
from multiview.views import ViewBuilder  # noqa: E402

from run_smoke4 import (  # noqa: E402
    SMOKE4_DATASETS, _load_csv_array, _load_csv_target,
    _existing_keys, _append, COLUMNS,
)


def _metrics(t0, yte, pred):
    pred = np.asarray(pred).ravel()
    return {
        "rmsep": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "n_components": 10,
        "fit_time_s": float(time.perf_counter() - t0),
    }


def _runner_multiK_wide(Xtr, ytr, Xte, yte, seed, max_components):
    t0 = time.perf_counter()
    est = AOMMoEMultiK(
        K_list=(2, 3, 4, 5, 7, 10),
        per_expert_components="auto", random_state=seed,
    )
    est.fit(Xtr, ytr)
    return _metrics(t0, yte, est.predict(Xte))


def _runner_multiK_357_auto(Xtr, ytr, Xte, yte, seed, max_components):
    t0 = time.perf_counter()
    est = AOMMoEMultiK(
        K_list=(3, 5, 7),
        per_expert_components="auto", random_state=seed,
    )
    est.fit(Xtr, ytr)
    return _metrics(t0, yte, est.predict(Xte))


def _build_lazy_v2_aom(seed, max_components, p):
    bank = ViewBuilder.combined(
        bank_name="compact", K=3, strategy="equal_width", include_global=True,
    ).build(p=p)
    return AOMPLSRegressor(
        n_components="auto", max_components=max_components,
        engine="simpls_covariance", selection="global",
        criterion="holdout", operator_bank=bank, random_state=seed,
    )


def _runner_mean_ensemble_3(Xtr, ytr, Xte, yte, seed, max_components):
    p = Xtr.shape[1]
    bases = [
        ("multiK-3-5-7-auto", AOMMoEMultiK(
            K_list=(3, 5, 7), per_expert_components="auto", random_state=seed,
        )),
        ("moe-preproc-soft", AOMMoERegressor(
            expert_layout="per_preproc", routing="soft",
            bank_name="compact", per_expert_components=min(10, max_components),
            random_state=seed,
        )),
        ("lazy-V2-AOM", _build_lazy_v2_aom(seed, max_components, p)),
    ]
    t0 = time.perf_counter()
    est = MultiVariantMeanEnsemble(bases=bases)
    est.fit(Xtr, ytr)
    return _metrics(t0, yte, est.predict(Xte))


def _runner_mean_ensemble_4(Xtr, ytr, Xte, yte, seed, max_components):
    p = Xtr.shape[1]
    bases = [
        ("multiK-3-5-7-auto", AOMMoEMultiK(
            K_list=(3, 5, 7), per_expert_components="auto", random_state=seed,
        )),
        ("moe-preproc-soft", AOMMoERegressor(
            expert_layout="per_preproc", routing="soft",
            bank_name="compact", per_expert_components=min(10, max_components),
            random_state=seed,
        )),
        ("lazy-V2-AOM", _build_lazy_v2_aom(seed, max_components, p)),
        ("aom-pls-compact", AOMPLSRegressor(
            n_components="auto", max_components=max_components,
            engine="simpls_covariance", selection="global",
            criterion="holdout", operator_bank="compact", random_state=seed,
        )),
    ]
    t0 = time.perf_counter()
    est = MultiVariantMeanEnsemble(bases=bases)
    est.fit(Xtr, ytr)
    return _metrics(t0, yte, est.predict(Xte))


VARIANTS = [
    ("moe-view-multiK-wide-2-10", _runner_multiK_wide),
    ("moe-view-multiK-3-5-7-auto", _runner_multiK_357_auto),
    ("mean-ensemble-3", _runner_mean_ensemble_3),
    ("mean-ensemble-4", _runner_mean_ensemble_4),
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
    print(f"[phase10-smoke4] {len(smoke)} datasets x {len(VARIANTS)} variants")

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
                f"[phase10-smoke4] {cohort_row['dataset']:<48s} {label:<32s} "
                f"rmsep={row.get('rmsep', 'NA')} t={row.get('fit_time_s', 'NA')}"
            )

    print(f"[phase10-smoke4] appended {n_appended} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
