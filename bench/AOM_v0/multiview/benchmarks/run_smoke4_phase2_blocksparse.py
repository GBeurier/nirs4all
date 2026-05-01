"""Phase 2 block-sparse AOM-MBPLS variants on smoke-4.

Block-sparse V1/V2 use the new `BlockSparseAOMMBPLSRegressor` that
implements true per-block deflation: only the winning block deflates by
`t_a . p_{k*,a}^T`. Other blocks retain their full residual, so subsequent
LVs can re-pick a different block.

Variants:

- `block-sparse-V1-blocks3-holdout` — block-only bank (3 ops), holdout.
- `block-sparse-V2-combined-compact-holdout` — preproc x block bank (24 ops), holdout.
- `block-sparse-V1-blocks3-cvspxy` — block-only, SPXY-CV criterion.

Outputs are appended to `bench/AOM_v0/multiview/results/smoke4_baseline.csv`.
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

from multiview.estimators_mbpls import BlockSparseAOMMBPLSRegressor  # noqa: E402

from benchmarks.run_smoke4 import (  # noqa: E402
    SMOKE4_DATASETS,
    _load_csv_array,
    _load_csv_target,
    _existing_keys,
    _append,
    COLUMNS,
)


def _spxy_factory(seed: int, n_splits: int = 3):
    from nirs4all.operators.splitters import SPXYFold
    return SPXYFold(n_splits=int(n_splits), random_state=int(seed))


def _run_block_sparse(
    *, Xtr, ytr, Xte, yte, seed, max_components,
    K=3, preproc_bank_name=None, criterion="holdout", cv=3, cv_splitter=None,
):
    t0 = time.perf_counter()
    est = BlockSparseAOMMBPLSRegressor(
        n_components="auto", max_components=max_components,
        K=K, strategy="equal_width",
        preproc_bank_name=preproc_bank_name,
        criterion=criterion, cv=cv, cv_splitter=cv_splitter,
        random_state=seed,
    )
    est.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    pred = est.predict(Xte).ravel()
    winners = est.get_block_winners()
    return {
        "rmsep": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "n_components": int(est.n_components_),
        "fit_time_s": float(fit_time),
        "selected_operators": ",".join(
            [f"k{k}_op{op}" for (k, op) in winners]
        ),
    }


def _runner_v1_blocks_holdout(Xtr, ytr, Xte, yte, seed, max_components):
    return _run_block_sparse(
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, seed=seed,
        max_components=max_components,
        K=3, preproc_bank_name=None, criterion="holdout",
    )


def _runner_v2_combined_holdout(Xtr, ytr, Xte, yte, seed, max_components):
    return _run_block_sparse(
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, seed=seed,
        max_components=max_components,
        K=3, preproc_bank_name="compact", criterion="holdout",
    )


def _runner_v1_blocks_cvspxy(Xtr, ytr, Xte, yte, seed, max_components):
    return _run_block_sparse(
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, seed=seed,
        max_components=max_components,
        K=3, preproc_bank_name=None, criterion="cv",
        cv=3, cv_splitter=_spxy_factory(seed, 3),
    )


VARIANTS = [
    ("block-sparse-V1-blocks3-holdout", _runner_v1_blocks_holdout),
]
# V1-cvspxy (n>2000 × auto-prefix × refit-per-candidate) and
# V2-combined (24 ops × refit) stall on Chla+b (n=2925). They are
# documented in DESIGN_MBPLS §6 as Phase-3+ profile-then-escalate work.
SLOW_VARIANTS = [
    ("block-sparse-V1-blocks3-cvspxy", _runner_v1_blocks_cvspxy),
    ("block-sparse-V2-combined-compact-holdout", _runner_v2_combined_holdout),
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
        f"[smoke4-blocksparse] {len(smoke)} datasets x {len(VARIANTS)} variants "
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
                print(f"[smoke4-blocksparse] skip {key}")
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
                f"[smoke4-blocksparse] {cohort_row['dataset']:<48s} {label:<48s} "
                f"rmsep={row.get('rmsep', 'NA')} r2={row.get('r2', 'NA')} "
                f"k={row.get('n_components', 'NA')} t={row.get('fit_time_s', 'NA')}"
            )

    print(f"[smoke4-blocksparse] appended {n_appended} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
