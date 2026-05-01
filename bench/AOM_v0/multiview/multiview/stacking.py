"""Phase 4 stacking hybrid: Ridge-meta over multi-view winners.

Train K base models (block-sparse-V1, MoE preproc-soft, AOM-PLS-compact),
generate out-of-fold predictions via cross-validation, then train a Ridge
meta-model on the OOF predictions to combine them. Predict by getting
each base model's prediction on test data and feeding through Ridge.

Conceptually similar to AOMMoERegressor but the experts are heterogeneous
NIRS estimators (one block-sparse, one MoE, one AOM-PLS) rather than
homogeneous PLS-on-views.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


class StackingHybrid(BaseEstimator, RegressorMixin):
    """Stacking ensemble of pre-configured base estimators with Ridge meta.

    Parameters
    ----------
    base_estimators : list of (name, estimator) tuples
        Base sklearn estimators. Will be cloned per fold.
    n_oof_folds : int
        Folds for OOF prediction generation.
    meta_alpha : float
        Ridge regularisation for the meta-model.
    random_state : int
        Seed for fold splits.
    nonneg : bool
        If True, fit a non-negative Ridge (use NNLS instead). Default False.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        base_estimators: Optional[List[Tuple[str, BaseEstimator]]] = None,
        n_oof_folds: int = 3,
        meta_alpha: float = 1.0,
        random_state: int = 0,
        nonneg: bool = False,
    ) -> None:
        self.base_estimators = base_estimators
        self.n_oof_folds = n_oof_folds
        self.meta_alpha = meta_alpha
        self.random_state = random_state
        self.nonneg = nonneg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingHybrid":
        if not self.base_estimators:
            raise ValueError("base_estimators must be a non-empty list")
        start = time.perf_counter()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        E = len(self.base_estimators)

        # OOF predictions per base estimator.
        kf = KFold(n_splits=self.n_oof_folds, shuffle=True, random_state=self.random_state)
        oof = np.zeros((n, E), dtype=float)
        for train_idx, val_idx in kf.split(X):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr = y[train_idx]
            for e, (_name, est) in enumerate(self.base_estimators):
                est_e = clone(est)
                try:
                    est_e.fit(X_tr, y_tr)
                    oof[val_idx, e] = np.asarray(est_e.predict(X_va)).ravel()
                except Exception:
                    oof[val_idx, e] = float(y_tr.mean())

        # Train meta-Ridge (or NNLS).
        if self.nonneg:
            from scipy.optimize import nnls
            weights, _ = nnls(oof, y)
            self.meta_intercept_ = 0.0
            self.meta_weights_ = weights
        else:
            ridge = Ridge(alpha=self.meta_alpha, fit_intercept=True)
            ridge.fit(oof, y)
            self.meta_intercept_ = float(ridge.intercept_)
            self.meta_weights_ = ridge.coef_.copy()

        # Re-fit base estimators on full data for prediction.
        full_estimators: List[BaseEstimator] = []
        for _name, est in self.base_estimators:
            est_full = clone(est)
            est_full.fit(X, y)
            full_estimators.append(est_full)
        self.full_estimators_ = full_estimators
        self.fit_time_s_ = float(time.perf_counter() - start)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "full_estimators_"):
            raise RuntimeError("Estimator not fitted")
        X = np.asarray(X, dtype=float)
        E = len(self.full_estimators_)
        preds = np.zeros((X.shape[0], E))
        for e, est in enumerate(self.full_estimators_):
            preds[:, e] = np.asarray(est.predict(X)).ravel()
        return preds @ self.meta_weights_ + self.meta_intercept_
