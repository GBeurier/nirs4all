"""Phase 2 lazy AOM-MBPLS variants on the smoke-4 cohort.

Lazy V1 / V2 reuse the existing AOM-PLS POP / AOM selection on a multi-view
bank assembled by `ViewBuilder`. No new selection algorithm is introduced —
this file is the Phase-2 starting point and tells us whether per-LV
block-aware view selection alone is enough to beat AOM-PLS-compact.

Variants:

- `lazy-V1-POP-blocks3-holdout` — blocks-only bank (4 ops), POP per-component.
- `lazy-V1-POP-blocks3-cvspxy` — same bank, SPXY-CV criterion.
- `lazy-V2-POP-combined-compact-holdout` — preproc x block bank (36 ops), POP.
- `lazy-V2-AOM-combined-compact-holdout` — same bank, AOM (single-op global).
- `lazy-V2-POP-combined-compact-cvspxy` — same bank, SPXY-CV criterion.

Outputs are appended to `bench/AOM_v0/multiview/results/smoke4_baseline.csv`
so Phase 1 baselines remain in the same comparison file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

_HERE = Path(os.path.abspath(__file__)).resolve()
_MULTIVIEW_ROOT = _HERE.parent.parent
_AOM_ROOT = _MULTIVIEW_ROOT.parent
# Order matters: `bench/AOM_v0/benchmarks/` and `bench/AOM_v0/multiview/benchmarks/`
# both define a `benchmarks` package. We want the multi-view one to win when
# `from benchmarks.run_smoke4 import ...` resolves.
if str(_AOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_AOM_ROOT))
if str(_MULTIVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_MULTIVIEW_ROOT))

from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402

from aompls.estimators import AOMPLSRegressor, POPPLSRegressor  # noqa: E402
from multiview.views import ViewBuilder  # noqa: E402

# Reuse the Phase 1 IO + cohort definitions.
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


def _run_aompls_with_bank(
    *, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
    seed: int, max_components: int,
    selection: str, criterion: str, bank, cv_splitter=None, cv: int = 3,
) -> dict:
    cls = POPPLSRegressor if selection == "per_component" else AOMPLSRegressor
    t0 = time.perf_counter()
    est = cls(
        n_components="auto",
        max_components=max_components,
        engine="simpls_covariance",
        selection="per_component" if cls is POPPLSRegressor else "global",
        criterion=criterion,
        operator_bank=bank,
        random_state=seed,
        cv=cv,
        cv_splitter=cv_splitter,
    )
    est.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    pred = est.predict(Xte).ravel()
    selected = est.get_selected_operators()
    return {
        "rmsep": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "n_components": int(est.n_components_),
        "fit_time_s": float(fit_time),
        "selected_operators": ",".join(sorted(set(selected))),
    }


def _runner_lazy_v1_pop_blocks_holdout(Xtr, ytr, Xte, yte, seed, max_components):
    bank = ViewBuilder.blocks_only(K=3, strategy="equal_width").build(p=Xtr.shape[1])
    return _run_aompls_with_bank(
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, seed=seed, max_components=max_components,
        selection="per_component", criterion="holdout", bank=bank,
    )


def _runner_lazy_v1_pop_blocks_cvspxy(Xtr, ytr, Xte, yte, seed, max_components):
    bank = ViewBuilder.blocks_only(K=3, strategy="equal_width").build(p=Xtr.shape[1])
    return _run_aompls_with_bank(
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, seed=seed, max_components=max_components,
        selection="per_component", criterion="cv", bank=bank,
        cv=3, cv_splitter=_spxy_factory(seed, 3),
    )


def _runner_lazy_v2_pop_combined_holdout(Xtr, ytr, Xte, yte, seed, max_components):
    bank = ViewBuilder.combined(
        bank_name="compact", K=3, strategy="equal_width", include_global=True
    ).build(p=Xtr.shape[1])
    return _run_aompls_with_bank(
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, seed=seed, max_components=max_components,
        selection="per_component", criterion="holdout", bank=bank,
    )


def _runner_lazy_v2_aom_combined_holdout(Xtr, ytr, Xte, yte, seed, max_components):
    bank = ViewBuilder.combined(
        bank_name="compact", K=3, strategy="equal_width", include_global=True
    ).build(p=Xtr.shape[1])
    return _run_aompls_with_bank(
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, seed=seed, max_components=max_components,
        selection="global", criterion="holdout", bank=bank,
    )


def _runner_lazy_v2_pop_combined_cvspxy(Xtr, ytr, Xte, yte, seed, max_components):
    bank = ViewBuilder.combined(
        bank_name="compact", K=3, strategy="equal_width", include_global=True
    ).build(p=Xtr.shape[1])
    return _run_aompls_with_bank(
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, seed=seed, max_components=max_components,
        selection="per_component", criterion="cv", bank=bank,
        cv=3, cv_splitter=_spxy_factory(seed, 3),
    )


VARIANTS = [
    ("lazy-V1-POP-blocks3-holdout", _runner_lazy_v1_pop_blocks_holdout),
    ("lazy-V2-POP-combined-compact-holdout", _runner_lazy_v2_pop_combined_holdout),
    ("lazy-V2-AOM-combined-compact-holdout", _runner_lazy_v2_aom_combined_holdout),
]
# CV-SPXY variants disabled by default — V2-POP-combined-cvspxy takes ~45 s/fit and
# blows up to ~30 min/dataset. Re-enable explicitly via SPXY_VARIANTS once a
# holdout winner is identified.
SPXY_VARIANTS = [
    ("lazy-V1-POP-blocks3-cvspxy", _runner_lazy_v1_pop_blocks_cvspxy),
    ("lazy-V2-POP-combined-compact-cvspxy", _runner_lazy_v2_pop_combined_cvspxy),
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
        f"[smoke4-phase2-lazy] {len(smoke)} datasets x {len(VARIANTS)} variants "
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
                print(f"[smoke4-phase2-lazy] skip already-done {key}")
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
                f"[smoke4-phase2-lazy] {cohort_row['dataset']:<48s} {label:<46s} "
                f"rmsep={row.get('rmsep', 'NA')} r2={row.get('r2', 'NA')} "
                f"k={row.get('n_components', 'NA')} t={row.get('fit_time_s', 'NA')}"
            )

    print(f"[smoke4-phase2-lazy] appended {n_appended} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
