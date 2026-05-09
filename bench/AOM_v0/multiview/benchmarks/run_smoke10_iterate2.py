"""Phase 7 round 2: iterate moe-view-soft (the TabPFN-opt champion).

Test:
- moe-view-soft K=4 (4 wavelength blocks instead of 3)
- moe-view-soft K=5
- moe-view-soft K=3 with per_expert_components=15 (vs default 10)
- moe-view-soft K=3 with per_expert_components=20
- ridge-stack-multiview (StackingHybrid with Ridge meta on real OOF)
- nnls-stack-multiview (StackingHybrid with NNLS meta on real OOF)

Outputs append to smoke10.csv.
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

from aompls.estimators import AOMPLSRegressor, POPPLSRegressor  # noqa: E402
from benchmarks.run_smoke4 import (  # noqa: E402
    COLUMNS,
    _append,
    _existing_keys,
    _load_csv_array,
    _load_csv_target,
)
from benchmarks.run_smoke10 import SMOKE10_DATASETS  # noqa: E402

from multiview.atoms import LazyV2AOM  # noqa: E402
from multiview.moe import AOMMoERegressor  # noqa: E402
from multiview.stacking import StackingHybrid  # noqa: E402
from multiview.views import ViewBuilder  # noqa: E402


def _make_moe_view(K: int, components: int, seed: int) -> AOMMoERegressor:
    return AOMMoERegressor(
        expert_layout="per_view", routing="soft",
        K=K, per_expert_components=components,
        n_oof_folds=3, random_state=seed,
    )


def _make_moe_preproc(bank_name: str, components: int, seed: int) -> AOMMoERegressor:
    return AOMMoERegressor(
        expert_layout="per_preproc", routing="soft",
        bank_name=bank_name, per_expert_components=components,
        n_oof_folds=3, random_state=seed,
    )


def _build_lazy_v1_pop(seed, max_components, p):
    bank = ViewBuilder.blocks_only(K=3, strategy="equal_width").build(p=p)
    return POPPLSRegressor(
        n_components="auto", max_components=max_components,
        engine="simpls_covariance", selection="per_component",
        criterion="holdout", operator_bank=bank, random_state=seed,
    )


def _runner_moe_view_K4(Xtr, ytr, Xte, yte, seed, max_components):
    return _run_simple(_make_moe_view(4, min(10, max_components), seed),
                       Xtr, ytr, Xte, yte, max_components)


def _runner_moe_view_K5(Xtr, ytr, Xte, yte, seed, max_components):
    return _run_simple(_make_moe_view(5, min(10, max_components), seed),
                       Xtr, ytr, Xte, yte, max_components)


def _runner_moe_view_K3_c15(Xtr, ytr, Xte, yte, seed, max_components):
    return _run_simple(_make_moe_view(3, 15, seed), Xtr, ytr, Xte, yte, max_components)


def _runner_moe_view_K3_c20(Xtr, ytr, Xte, yte, seed, max_components):
    return _run_simple(_make_moe_view(3, 20, seed), Xtr, ytr, Xte, yte, max_components)


def _runner_ridge_stack(Xtr, ytr, Xte, yte, seed, max_components):
    bases = [
        ("aom_pls", AOMPLSRegressor(
            n_components="auto", max_components=max_components,
            engine="simpls_covariance", selection="global",
            criterion="holdout", operator_bank="compact", random_state=seed,
        )),
        ("moe_preproc", _make_moe_preproc("compact", min(10, max_components), seed)),
        ("moe_view", _make_moe_view(3, min(10, max_components), seed)),
        ("lazy_v2_aom", LazyV2AOM(max_components=max_components, random_state=seed)),
    ]
    est = StackingHybrid(
        base_estimators=bases, n_oof_folds=3, meta_alpha=1.0,
        random_state=seed, nonneg=False,
    )
    return _run_with_diagnostics(est, Xtr, ytr, Xte, yte, max_components,
                                  diag=lambda e: ",".join(
                                      f"{n}:{w:+.3f}" for (n, _), w in zip(e.base_estimators, e.meta_weights_)
                                  ))


def _runner_nnls_stack(Xtr, ytr, Xte, yte, seed, max_components):
    bases = [
        ("aom_pls", AOMPLSRegressor(
            n_components="auto", max_components=max_components,
            engine="simpls_covariance", selection="global",
            criterion="holdout", operator_bank="compact", random_state=seed,
        )),
        ("moe_preproc", _make_moe_preproc("compact", min(10, max_components), seed)),
        ("moe_view", _make_moe_view(3, min(10, max_components), seed)),
        ("lazy_v2_aom", LazyV2AOM(max_components=max_components, random_state=seed)),
    ]
    est = StackingHybrid(
        base_estimators=bases, n_oof_folds=3, meta_alpha=1.0,
        random_state=seed, nonneg=True,
    )
    return _run_with_diagnostics(est, Xtr, ytr, Xte, yte, max_components,
                                  diag=lambda e: ",".join(
                                      f"{n}:{w:.3f}" for (n, _), w in zip(e.base_estimators, e.meta_weights_)
                                  ))


def _run_simple(est, Xtr, ytr, Xte, yte, max_components):
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    pred = np.asarray(est.predict(Xte)).ravel()
    return _metrics(t0, yte, pred, max_components)


def _run_with_diagnostics(est, Xtr, ytr, Xte, yte, max_components, diag):
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    pred = np.asarray(est.predict(Xte)).ravel()
    out = _metrics(t0, yte, pred, max_components)
    out["selected_operators"] = diag(est)
    return out


def _metrics(t0, yte, pred, n_components):
    return {
        "rmsep": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "n_components": int(n_components),
        "fit_time_s": float(time.perf_counter() - t0),
    }


VARIANTS = [
    ("moe-view-soft-K4", _runner_moe_view_K4),
    ("moe-view-soft-K5", _runner_moe_view_K5),
    ("moe-view-soft-K3-c15", _runner_moe_view_K3_c15),
    ("moe-view-soft-K3-c20", _runner_moe_view_K3_c20),
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
    results_path = workspace / "smoke10.csv"
    existing = _existing_keys(results_path)

    cohort = pd.read_csv(args.cohort)
    smoke = cohort[
        cohort["dataset"].isin(SMOKE10_DATASETS) & (cohort["status"] == "ok")
    ].copy()
    print(
        f"[iter2] {len(smoke)} datasets x {len(VARIANTS)} variants "
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
                f"[iter2] {cohort_row['dataset']:<48s} {label:<28s} "
                f"rmsep={row.get('rmsep', 'NA')} r2={row.get('r2', 'NA')} "
                f"k={row.get('n_components', 'NA')} t={row.get('fit_time_s', 'NA')}"
            )

    print(f"[iter2] appended {n_appended} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
