"""Per-dataset best-variant selection via inner holdout.

Cheaper than full stacking: split training into train/val, train each base
estimator on train, score each on val, pick the one with lowest MSE, refit
on full training, predict on test. No OOF cross-validation, no Ridge meta.

This is "model selection by inner-holdout validation" — captures most of
what stacking can do without paying the OOF cost.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_squared_error


class BestOfStackedRegressor(BaseEstimator, RegressorMixin):
    """Train each base estimator on inner train fold, pick the one with
    lowest validation RMSE, refit it on full training, use for prediction.

    Parameters
    ----------
    base_estimators : list of (name, estimator)
        Base sklearn estimators. Cloned per fit.
    holdout_fraction : float
        Validation fraction (default 0.2).
    random_state : int
        Seed for permutation.
    refit_winner : bool
        If True (default), refit the winning estimator on full training data.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        base_estimators: Optional[List[Tuple[str, BaseEstimator]]] = None,
        holdout_fraction: float = 0.2,
        random_state: int = 0,
        refit_winner: bool = True,
    ) -> None:
        self.base_estimators = base_estimators
        self.holdout_fraction = holdout_fraction
        self.random_state = random_state
        self.refit_winner = refit_winner

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BestOfStackedRegressor":
        if not self.base_estimators:
            raise ValueError("base_estimators must be non-empty")
        start = time.perf_counter()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n)
        n_val = max(3, int(round(n * self.holdout_fraction)))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        X_tr, X_va = X[tr_idx], X[val_idx]
        y_tr, y_va = y[tr_idx], y[val_idx]

        scores = {}
        for name, est in self.base_estimators:
            est_e = clone(est)
            try:
                est_e.fit(X_tr, y_tr)
                pred = np.asarray(est_e.predict(X_va)).ravel()
                rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
            except Exception as exc:
                rmse = float("inf")
            scores[name] = rmse

        best_name = min(scores, key=scores.get)
        self.candidate_scores_ = dict(scores)
        self.winner_name_ = best_name

        winner_est = next(est for n, est in self.base_estimators if n == best_name)
        winner = clone(winner_est)
        if self.refit_winner:
            winner.fit(X, y)
        else:
            winner.fit(X_tr, y_tr)
        self.winner_ = winner
        self.fit_time_s_ = float(time.perf_counter() - start)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "winner_"):
            raise RuntimeError("Estimator not fitted")
        X = np.asarray(X, dtype=float)
        return np.asarray(self.winner_.predict(X)).ravel()
