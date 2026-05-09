"""Phase 7 iteration on smoke-10: push past TabPFN-opt.

New variants:
- ASLS+moe-preproc-soft-compact (AOM_v0 champion's ASLS preproc + our MoE)
- ASLS+moe-view-soft-K3
- moe-preproc-soft-family_pruned (15-op bank instead of 8-op compact)
- moe-preproc-soft-response_dedup (47-op bank, big diversity)
- bestof-multiview (per-dataset best variant selection via inner holdout)
- bestof-multiview-asls (same but with ASLS outer)

Outputs append to `smoke10.csv`.
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

from multiview.asls_wrapper import ASLSPreprocWrapper  # noqa: E402
from multiview.atoms import LazyV2AOM  # noqa: E402
from multiview.estimators_mbpls import BlockSparseAOMMBPLSRegressor  # noqa: E402
from multiview.moe import AOMMoERegressor  # noqa: E402
from multiview.stacking_select import BestOfStackedRegressor  # noqa: E402
from multiview.views import ViewBuilder  # noqa: E402


def _make_moe_preproc(bank_name: str, seed: int, n_components: int) -> AOMMoERegressor:
    return AOMMoERegressor(
        expert_layout="per_preproc", routing="soft",
        bank_name=bank_name, per_expert_components=min(10, n_components),
        n_oof_folds=3, random_state=seed,
    )


def _make_moe_view(K: int, seed: int, n_components: int) -> AOMMoERegressor:
    return AOMMoERegressor(
        expert_layout="per_view", routing="soft",
        K=K, per_expert_components=min(10, n_components),
        n_oof_folds=3, random_state=seed,
    )


def _runner_asls_moe_preproc_compact(Xtr, ytr, Xte, yte, seed, max_components):
    t0 = time.perf_counter()
    est = ASLSPreprocWrapper(estimator=_make_moe_preproc("compact", seed, max_components))
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)
    return _metrics(t0, yte, pred, max_components)


def _runner_asls_moe_view_K3(Xtr, ytr, Xte, yte, seed, max_components):
    t0 = time.perf_counter()
    est = ASLSPreprocWrapper(estimator=_make_moe_view(3, seed, max_components))
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)
    return _metrics(t0, yte, pred, max_components)


def _runner_moe_preproc_family_pruned(Xtr, ytr, Xte, yte, seed, max_components):
    t0 = time.perf_counter()
    est = _make_moe_preproc("family_pruned", seed, max_components)
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)
    return _metrics(t0, yte, pred, max_components)


def _runner_moe_preproc_response_dedup(Xtr, ytr, Xte, yte, seed, max_components):
    t0 = time.perf_counter()
    est = _make_moe_preproc("response_dedup", seed, max_components)
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)
    return _metrics(t0, yte, pred, max_components)


def _build_lazy_v1_pop(seed, max_components, p):
    bank = ViewBuilder.blocks_only(K=3, strategy="equal_width").build(p=p)
    return POPPLSRegressor(
        n_components="auto", max_components=max_components,
        engine="simpls_covariance", selection="per_component",
        criterion="holdout", operator_bank=bank, random_state=seed,
    )


def _make_bestof_estimator(seed, max_components, p, with_asls=False):
    bases = [
        ("aom_pls", AOMPLSRegressor(
            n_components="auto", max_components=max_components,
            engine="simpls_covariance", selection="global",
            criterion="holdout", operator_bank="compact", random_state=seed,
        )),
        ("moe_preproc_soft", _make_moe_preproc("compact", seed, max_components)),
        ("moe_view_soft", _make_moe_view(3, seed, max_components)),
        ("lazy_v1_pop", _build_lazy_v1_pop(seed, max_components, p)),
        ("lazy_v2_aom", LazyV2AOM(max_components=max_components, random_state=seed)),
    ]
    if with_asls:
        bases = [(f"asls_{n}", ASLSPreprocWrapper(estimator=e)) for n, e in bases]
    return BestOfStackedRegressor(
        base_estimators=bases,
        holdout_fraction=0.2, random_state=seed, refit_winner=True,
    )


def _runner_bestof_multiview(Xtr, ytr, Xte, yte, seed, max_components):
    p = Xtr.shape[1]
    t0 = time.perf_counter()
    est = _make_bestof_estimator(seed, max_components, p, with_asls=False)
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)
    out = _metrics(t0, yte, pred, max_components)
    out["selected_operators"] = f"winner={est.winner_name_}"
    return out


def _runner_bestof_multiview_asls(Xtr, ytr, Xte, yte, seed, max_components):
    p = Xtr.shape[1]
    t0 = time.perf_counter()
    est = _make_bestof_estimator(seed, max_components, p, with_asls=True)
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)
    out = _metrics(t0, yte, pred, max_components)
    out["selected_operators"] = f"winner={est.winner_name_}"
    return out


def _metrics(t0, yte, pred, n_components):
    pred = np.asarray(pred).ravel()
    return {
        "rmsep": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "n_components": int(n_components),
        "fit_time_s": float(time.perf_counter() - t0),
    }


VARIANTS = [
    ("asls-moe-preproc-soft-compact", _runner_asls_moe_preproc_compact),
    ("asls-moe-view-soft-K3", _runner_asls_moe_view_K3),
    ("moe-preproc-soft-family-pruned", _runner_moe_preproc_family_pruned),
    ("moe-preproc-soft-response-dedup", _runner_moe_preproc_response_dedup),
    ("bestof-multiview", _runner_bestof_multiview),
    ("bestof-multiview-asls", _runner_bestof_multiview_asls),
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
        f"[smoke10-iter] {len(smoke)} datasets x {len(VARIANTS)} variants "
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
                print(f"[smoke10-iter] skip {key}")
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
                f"[smoke10-iter] {cohort_row['dataset']:<48s} {label:<36s} "
                f"rmsep={row.get('rmsep', 'NA')} r2={row.get('r2', 'NA')} "
                f"k={row.get('n_components', 'NA')} t={row.get('fit_time_s', 'NA')}"
            )

    print(f"[smoke10-iter] appended {n_appended} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
