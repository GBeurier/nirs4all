"""Outer-K-fold grid HPO for AOMPLSCompact (PLS1)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import KFold

from .sklearn import AOMPLSCompact


@dataclass
class TuneResult:
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Tuple[Dict[str, Any], float]]
    refit_model: AOMPLSCompact


def tune(
    X: np.ndarray,
    y: np.ndarray,
    max_components_grid: Sequence[int] = (5, 10, 15, 20, 25),
    preproc_grid: Sequence[str] = ("none", "snv", "osc", "asls", "asls+osc"),
    n_folds: int = 5,
    cv_mode: str = "kfold",
    outer_folds: int = 5,
    one_se_rule: bool = False,
    random_state: int = 0,
) -> TuneResult:
    """Grid search over (max_components, preproc) with outer K-fold validation.

    For each (max_components, preproc) combination, fits AOMPLSCompact on the
    outer-train fold (with the inner CV strategy specified by ``cv_mode``) and
    evaluates RMSE on the outer-test fold. Returns the best config and a model
    refit on the full data with that config.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
    outer = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)

    grid: List[Dict[str, Any]] = [
        {"max_components": int(K), "preproc": str(p)}
        for K, p in product(max_components_grid, preproc_grid)
    ]
    all_results: List[Tuple[Dict[str, Any], float]] = []
    for cfg in grid:
        rmses = []
        for tr_idx, va_idx in outer.split(X):
            est = AOMPLSCompact(
                max_components=cfg["max_components"],
                n_folds=n_folds,
                cv_mode=cv_mode,
                one_se_rule=one_se_rule,
                random_state=random_state,
                preproc=cfg["preproc"],
            )
            est.fit(X[tr_idx], y[tr_idx])
            pred = est.predict(X[va_idx])
            rmses.append(float(np.sqrt(np.mean((pred - y[va_idx]) ** 2))))
        all_results.append((cfg, float(np.mean(rmses))))
    all_results.sort(key=lambda kv: kv[1])
    best_params, best_score = all_results[0]
    refit = AOMPLSCompact(
        max_components=best_params["max_components"],
        n_folds=n_folds,
        cv_mode=cv_mode,
        one_se_rule=one_se_rule,
        random_state=random_state,
        preproc=best_params["preproc"],
    ).fit(X, y)
    return TuneResult(best_params=best_params, best_score=best_score, all_results=all_results, refit_model=refit)
