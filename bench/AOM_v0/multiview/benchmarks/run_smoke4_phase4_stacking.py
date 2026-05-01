"""Phase 4 stacking hybrid on smoke-4.

Train a Ridge-meta stacking ensemble combining the Phase 1-3 winners:
- AOM-PLS-compact (best on All_manure)
- block-sparse-V1 (best on Beer)
- MoE preproc-soft (best on grapevine)
- Lazy V1 POP (best on Chla+b)

Output appended to `bench/AOM_v0/multiview/results/smoke4_baseline.csv`.

Variants:
- stack-ridge — Ridge-meta over the 4 base estimators
- stack-nnls — NNLS-meta (nonneg, simplex) over the same 4
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

from aompls.estimators import AOMPLSRegressor, POPPLSRegressor  # noqa: E402
from multiview.estimators_mbpls import BlockSparseAOMMBPLSRegressor  # noqa: E402
from multiview.moe import AOMMoERegressor  # noqa: E402
from multiview.stacking import StackingHybrid  # noqa: E402
from multiview.views import ViewBuilder  # noqa: E402

from benchmarks.run_smoke4 import (  # noqa: E402
    SMOKE4_DATASETS,
    _load_csv_array,
    _load_csv_target,
    _existing_keys,
    _append,
    COLUMNS,
)


def _make_base_estimators(seed: int, max_components: int):
    return [
        ("aom_pls", AOMPLSRegressor(
            n_components="auto", max_components=max_components,
            engine="simpls_covariance", selection="global",
            criterion="holdout", operator_bank="compact",
            random_state=seed,
        )),
        ("block_sparse_v1", BlockSparseAOMMBPLSRegressor(
            n_components="auto", max_components=max_components,
            K=3, strategy="equal_width", preproc_bank_name=None,
            criterion="holdout", random_state=seed,
        )),
        ("moe_preproc_soft", AOMMoERegressor(
            expert_layout="per_preproc", routing="soft",
            bank_name="compact",
            per_expert_components=min(10, max_components),
            random_state=seed,
        )),
        ("lazy_v1_pop", POPPLSRegressor(
            n_components="auto", max_components=max_components,
            engine="simpls_covariance", selection="per_component",
            criterion="holdout",
            operator_bank=ViewBuilder.blocks_only(K=3, strategy="equal_width").build(p=0),
            # The operator_bank is built per-fit in the stacking helper since
            # `p` is unknown until X arrives. Override below.
            random_state=seed,
        )),
    ]


class _LazyV1POPWrapper:
    """Wrapper that builds the per-p ViewBuilder bank at fit time."""
    def __init__(self, max_components, seed):
        self.max_components = max_components
        self.seed = seed
        self._estimator_type = "regressor"

    def fit(self, X, y):
        bank = ViewBuilder.blocks_only(K=3, strategy="equal_width").build(p=X.shape[1])
        self._est = POPPLSRegressor(
            n_components="auto", max_components=self.max_components,
            engine="simpls_covariance", selection="per_component",
            criterion="holdout", operator_bank=bank,
            random_state=self.seed,
        )
        self._est.fit(X, y)
        return self

    def predict(self, X):
        return self._est.predict(X)

    def get_params(self, deep=True):
        return {"max_components": self.max_components, "seed": self.seed}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def _make_stacking(seed: int, max_components: int, nonneg: bool = False):
    return StackingHybrid(
        base_estimators=[
            ("aom_pls", AOMPLSRegressor(
                n_components="auto", max_components=max_components,
                engine="simpls_covariance", selection="global",
                criterion="holdout", operator_bank="compact",
                random_state=seed,
            )),
            ("block_sparse_v1", BlockSparseAOMMBPLSRegressor(
                n_components="auto", max_components=max_components,
                K=3, strategy="equal_width", preproc_bank_name=None,
                criterion="holdout", random_state=seed,
            )),
            ("moe_preproc_soft", AOMMoERegressor(
                expert_layout="per_preproc", routing="soft",
                bank_name="compact",
                per_expert_components=min(10, max_components),
                random_state=seed,
            )),
            ("lazy_v1_pop", _LazyV1POPWrapper(max_components, seed)),
        ],
        n_oof_folds=3, meta_alpha=1.0, random_state=seed, nonneg=nonneg,
    )


def _runner_stack_ridge(Xtr, ytr, Xte, yte, seed, max_components):
    t0 = time.perf_counter()
    est = _make_stacking(seed, max_components, nonneg=False)
    est.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    pred = est.predict(Xte).ravel()
    return {
        "rmsep": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "n_components": -1,
        "fit_time_s": float(fit_time),
        "selected_operators": ",".join(
            f"{name}:{w:+.3f}" for (name, _), w in zip(est.base_estimators, est.meta_weights_)
        ),
    }


def _runner_stack_nnls(Xtr, ytr, Xte, yte, seed, max_components):
    t0 = time.perf_counter()
    est = _make_stacking(seed, max_components, nonneg=True)
    est.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    pred = est.predict(Xte).ravel()
    return {
        "rmsep": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "n_components": -1,
        "fit_time_s": float(fit_time),
        "selected_operators": ",".join(
            f"{name}:{w:.3f}" for (name, _), w in zip(est.base_estimators, est.meta_weights_)
        ),
    }


VARIANTS = [
    ("stack-ridge-4base", _runner_stack_ridge),
    ("stack-nnls-4base", _runner_stack_nnls),
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
    results_path = workspace / "smoke4_baseline.csv"
    existing = _existing_keys(results_path)

    cohort = pd.read_csv(args.cohort)
    smoke = cohort[
        cohort["dataset"].isin(SMOKE4_DATASETS) & (cohort["status"] == "ok")
    ].copy()
    print(
        f"[smoke4-phase4-stack] {len(smoke)} datasets x {len(VARIANTS)} variants "
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
                print(f"[smoke4-phase4-stack] skip {key}")
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
                f"[smoke4-phase4-stack] {cohort_row['dataset']:<48s} {label:<24s} "
                f"rmsep={row.get('rmsep', 'NA')} r2={row.get('r2', 'NA')} "
                f"t={row.get('fit_time_s', 'NA')}"
            )

    print(f"[smoke4-phase4-stack] appended {n_appended} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
