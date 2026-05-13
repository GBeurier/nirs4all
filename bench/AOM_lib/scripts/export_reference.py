"""Export AOM-PLS reference fixtures from the Python AOM_v0 implementation.

This script runs the canonical ``AOMPLSRegressor(operator_bank="compact",
engine="simpls_covariance", selection="global", cv=5, max_components=15,
random_state=0)`` on a small set of NIRS datasets, plus the same model with
SPXY folds, and dumps everything the C++ parity tests need:

* X / y / fold indices (KFold and SPXY)
* selected operator (name + index in the compact bank)
* selected n_components
* final coefficient vector B and intercept (PLS1)
* per-prefix CV RMSE curve for the selected operator
* per-operator best-prefix CV RMSE (sanity surface)
* predictions on the training set (after the AOM fit refit on full data)

Outputs land in ``bench/AOM_lib/cpp/tests/reference/<DATASET>.json``.

Run from the repo root::

    .venv/bin/python bench/AOM_lib/scripts/export_reference.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench" / "AOM_v0"))

from aompls.banks import compact_bank, fit_bank  # noqa: E402
from aompls.estimators import AOMPLSRegressor  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from nirs4all.operators.splitters import SPXYFold  # noqa: E402


# Three small datasets from the regression cohort (n_train values 40, 64, 247).
DATASETS = [
    ("BEER", "Beer_OriginalExtract_60_KS"),
    ("CORN", "Corn_Oil_80_ZhengChenPelegYbaseSplit"),  # 64 train / 16 test — small and well-behaved
    ("ALPINE", "ALPINE_P_291_KS"),
]


def _resolve_cohort_path(database: str, dataset: str) -> Tuple[Path, Path]:
    """Look up X/Y train paths from cohort_regression.csv."""
    cohort = pd.read_csv(ROOT / "bench" / "AOM_v0" / "benchmarks" / "cohort_regression.csv")
    cohort = cohort[(cohort["database_name"] == database) & (cohort["dataset"] == dataset)]
    if cohort.empty:
        raise RuntimeError(f"dataset not found in cohort: {database}/{dataset}")
    row = cohort.iloc[0]
    return ROOT / row["train_path"], ROOT / row["ytrain_path"]


def _load_xy(database: str, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    Xpath, Ypath = _resolve_cohort_path(database, dataset)
    X = pd.read_csv(Xpath, sep=";").to_numpy(dtype=float)
    Y = pd.read_csv(Ypath, sep=";").iloc[:, 0].to_numpy(dtype=float)
    if Y.ndim != 1:
        Y = Y.ravel()
    return X, Y


def _kfold_indices(n: int, n_splits: int, seed: int) -> List[List[int]]:
    """Return the *test* indices for each fold from sklearn's KFold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: List[List[int]] = []
    dummy = np.zeros((n, 1))
    for _, test_idx in kf.split(dummy):
        folds.append(test_idx.astype(int).tolist())
    return folds


def _spxy_indices(X: np.ndarray, y: np.ndarray, n_splits: int) -> List[List[int]]:
    """Return the test indices for each fold from SPXYFold."""
    splitter = SPXYFold(n_splits=n_splits)
    folds: List[List[int]] = []
    for _, test_idx in splitter.split(X, y):
        folds.append(np.asarray(test_idx, dtype=int).tolist())
    return folds


def _compute_operator_rmse_curves(
    X: np.ndarray, y: np.ndarray, fold_test_indices: Sequence[Sequence[int]], max_components: int
) -> List[List[float]]:
    """For each operator in the compact bank and each prefix k, the mean CV RMSE.

    Uses the same auto-prefix scoring logic the AOM selector applies internally:
    one fit per fold with K=max_components, evaluate every prefix via coef_prefix.
    """
    # Match the AOMPLSRegressor's internal scoring exactly: it uses
    # simpls_covariance with orthogonalization='transformed' (auto-resolved from
    # selection='global'), which delegates to simpls_materialized_fixed.
    from aompls.simpls import simpls_materialized_fixed

    bank = compact_bank(p=X.shape[1])
    fit_bank(bank, X, y)
    n = X.shape[0]
    all_idx = set(range(n))
    n_ops = len(bank)
    n_folds = len(fold_test_indices)
    rmse_curves = [[math.inf] * max_components for _ in range(n_ops)]
    fold_collect: List[List[List[float]]] = [
        [[math.inf] * max_components for _ in range(n_folds)] for _ in range(n_ops)
    ]
    for f_idx, test_idx in enumerate(fold_test_indices):
        test_set = set(int(i) for i in test_idx)
        train_idx = np.array(sorted(all_idx - test_set), dtype=int)
        test_arr = np.array(sorted(test_set), dtype=int)
        Xtr = X[train_idx]
        ytr = y[train_idx]
        Xva = X[test_arr]
        yva = y[test_arr]
        x_mean = Xtr.mean(axis=0)
        y_mean = float(ytr.mean())
        Xc = Xtr - x_mean
        yc = ytr - y_mean
        for b, op in enumerate(bank):
            op.fit(X)
            res = simpls_materialized_fixed(Xc, yc.reshape(-1, 1), op, max_components)
            K = int(res.n_components)
            for k in range(1, K + 1):
                coef_k = res.coef_prefix(k).ravel()
                pred = (Xva - x_mean) @ coef_k + y_mean
                err = pred - yva
                fold_collect[b][f_idx][k - 1] = float(np.sqrt(np.mean(err * err)))
    # Match np.mean semantics: any inf at prefix k → mean is inf.
    for b in range(n_ops):
        for k in range(max_components):
            vals = [fold_collect[b][f][k] for f in range(n_folds)]
            if any(not math.isfinite(v) for v in vals):
                rmse_curves[b][k] = math.inf
            else:
                rmse_curves[b][k] = float(np.mean(vals))
    return rmse_curves


def _fit_and_export(
    X: np.ndarray,
    y: np.ndarray,
    fold_test_indices: Sequence[Sequence[int]],
    cv_label: str,
    max_components: int = 15,
    one_se_rule: bool = False,
) -> Dict[str, Any]:
    """Run AOMPLSRegressor against a fixed fold list and capture all parity quantities."""
    from sklearn.model_selection import BaseCrossValidator

    class FixedFolds(BaseCrossValidator):
        def __init__(self, folds: Sequence[Sequence[int]]):
            self.folds = [np.asarray(f, dtype=int) for f in folds]
            self.n_splits = len(self.folds)

        def split(self, X, y=None, groups=None):
            n = X.shape[0]
            all_idx = np.arange(n)
            for test in self.folds:
                train = np.setdiff1d(all_idx, test, assume_unique=False)
                yield train, test

        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits

    splitter = FixedFolds(fold_test_indices)
    # n_components="auto" enables auto-prefix scoring (argmin over both operator
    # and prefix k=1..max_components). This is the canonical AOM-PLS behaviour
    # described in the publication; passing an integer here disables prefix
    # selection (single-K scoring) which we explicitly do NOT want.
    est = AOMPLSRegressor(
        n_components="auto",
        max_components=max_components,
        engine="simpls_covariance",
        selection="global",
        criterion="cv",
        operator_bank="compact",
        orthogonalization="auto",
        center=True,
        scale=False,
        cv=splitter.n_splits,
        random_state=0,
        backend="numpy",
        repeats=1,
        one_se_rule=one_se_rule,
        cv_splitter=splitter,
    )
    t0 = time.perf_counter()
    est.fit(X, y)
    fit_s = time.perf_counter() - t0

    coef = np.asarray(est.coef_, dtype=float).reshape(-1)
    intercept = float(np.asarray(est.intercept_).reshape(-1)[0])
    selected_idx = int(est.selected_operator_indices_[0]) if est.selected_operator_indices_ else 0
    selected_name = est.selected_operators_[0] if est.selected_operators_ else "identity"
    bank_names = list(est.operator_scores_.keys())
    n_components = int(est.n_components_)

    # Per-prefix CV RMSE for the selected operator (already computed inside select_global,
    # exposed via diagnostics["score_curve"]).
    diag = est.get_diagnostics()
    extras = diag.get("extras") or {}
    selected_curve = extras.get("score_curve") or [math.inf] * max_components

    # Full n_ops x K score surface (re-computed independently — slow but produces the
    # cross-check the C++ parity test consumes).
    rmse_curves = _compute_operator_rmse_curves(X, y, fold_test_indices, max_components)

    preds = est.predict(X).reshape(-1)

    return {
        "cv_label": cv_label,
        "max_components": max_components,
        "one_se_rule": bool(one_se_rule),
        "n": int(X.shape[0]),
        "p": int(X.shape[1]),
        "X": X.tolist(),
        "y": y.tolist(),
        "fold_test_indices": [list(map(int, f)) for f in fold_test_indices],
        "bank_names": bank_names,
        "selected_operator_index": selected_idx,
        "selected_operator_name": selected_name,
        "n_components_selected": n_components,
        "coef": coef.tolist(),
        "intercept": intercept,
        "x_mean": np.asarray(est.x_mean_, dtype=float).tolist(),
        "y_mean": float(np.asarray(est.y_mean_).reshape(-1)[0]),
        "rmse_curve_selected": [float(x) for x in selected_curve],
        "rmse_curves": rmse_curves,
        "predictions_train": preds.tolist(),
        "fit_time_s": fit_s,
    }


def _dump(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  wrote {out_path.relative_to(ROOT)} ({size_mb:.2f} MiB)")


def main() -> int:
    out_dir = ROOT / "bench" / "AOM_lib" / "cpp" / "tests" / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    for database, dataset in DATASETS:
        print(f"[{database}] {dataset}")
        X, y = _load_xy(database, dataset)
        print(f"  X={X.shape} y={y.shape} (mean={y.mean():.4g}, std={y.std():.4g})")

        kfold = _kfold_indices(X.shape[0], n_splits=5, seed=0)
        spxy = _spxy_indices(X, y, n_splits=5)

        payload: Dict[str, Any] = {"database": database, "dataset": dataset}
        payload["kfold5"] = _fit_and_export(X, y, kfold, cv_label="kfold5", max_components=15)
        payload["kfold5_oneSE"] = _fit_and_export(X, y, kfold, cv_label="kfold5_oneSE", max_components=15, one_se_rule=True)
        payload["spxy5"] = _fit_and_export(X, y, spxy, cv_label="spxy5", max_components=15)

        _dump(out_dir / f"{database}.json", payload)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
