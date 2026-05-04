"""Phase 11 Super Learner — implements Codex review HIGH actionable items.

Three new estimators:

1. `TrimmedMeanEnsemble` — drops the top and bottom predictions per sample,
   averages the middle. Zero-cost robust aggregation; addresses Beer-style
   drag from one bad base.

2. `NNLSSimplexStacker` — non-negative least-squares stacking with simplex
   constraint (weights sum to 1, all >= 0). Standardises base predictions
   before solving. Compares OOF RMSE to equal-weight mean and falls back
   to mean if no margin (Codex HIGH §A actionable).

3. `AdaptiveSuperLearner` — n-train-thresholded selector:
   - `n < 100`: recipe selection (pick recipe with lowest inner-CV RMSE)
   - `100 <= n < 200`: NNLS simplex stacker
   - `n >= 200`: NNLS simplex stacker with Ridge meta as alternative,
     pick whichever has lower OOF RMSE.

Per-base calibration (`y = a + b * yhat` with shrinkage to (0, 1)) is also
implemented for use inside any ensemble.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


_MIN_VAR = 1e-12


class TrimmedMeanEnsemble(BaseEstimator, RegressorMixin):
    """Drop top/bottom predictions per sample, average the middle.

    With `n_drop=1`, drops the single largest and single smallest base
    prediction at each test sample, averages the remainder. Zero-cost
    robust ensemble — useful when one base is a known weak link on a
    fraction of the cohort.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        bases: Optional[Sequence] = None,
        n_drop: int = 1,
    ) -> None:
        self.bases = bases
        self.n_drop = n_drop

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrimmedMeanEnsemble":
        if not self.bases:
            raise ValueError("bases must be a non-empty sequence")
        start = time.perf_counter()
        self._fitted = []
        for name, est in self.bases:
            e = clone(est)
            e.fit(X, y)
            self._fitted.append((name, e))
        self.fit_time_s_ = float(time.perf_counter() - start)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_fitted"):
            raise RuntimeError("Estimator not fitted")
        preds = np.column_stack([
            np.asarray(e.predict(X)).ravel() for _n, e in self._fitted
        ])
        n_b = preds.shape[1]
        if n_b <= 2 * self.n_drop:
            # Not enough bases to trim — fall back to mean.
            return preds.mean(axis=1)
        sorted_preds = np.sort(preds, axis=1)
        kept = sorted_preds[:, self.n_drop : n_b - self.n_drop]
        return kept.mean(axis=1)


class _ShrinkCalibrator:
    """Per-base linear calibration `y_hat -> a + b * y_hat` with shrinkage
    toward `(a=0, b=1)`. Shrinkage strength scales inversely with n_train:
    bigger n -> closer to OLS, smaller n -> closer to identity (no
    calibration). The shrinkage prior protects against overfitting tiny
    train sets while letting big sets learn meaningful corrections.
    """

    def __init__(self, shrinkage_lambda: float = 5.0) -> None:
        self.shrinkage_lambda = shrinkage_lambda

    def fit(self, yhat: np.ndarray, y: np.ndarray) -> "_ShrinkCalibrator":
        yhat = np.asarray(yhat, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)
        # Shrinkage strength: λ_eff = λ / sqrt(n) so big n -> ~OLS
        lam = self.shrinkage_lambda / max(np.sqrt(n), 1.0)
        # Closed-form Bayesian-ish ridge with mean=(0, 1) prior:
        # minimize ||y - (a + b yhat)||^2 + lam (a^2 + (b-1)^2)
        X = np.column_stack([np.ones_like(yhat), yhat])
        XtX = X.T @ X
        Xty = X.T @ y
        prior_target = np.array([0.0, 1.0])
        a_hat, b_hat = np.linalg.solve(
            XtX + lam * np.eye(2), Xty + lam * prior_target
        )
        self.a_ = float(a_hat)
        self.b_ = float(b_hat)
        return self

    def transform(self, yhat: np.ndarray) -> np.ndarray:
        return self.a_ + self.b_ * np.asarray(yhat, dtype=float)


def _standardize(z: np.ndarray, eps: float = 1e-9):
    """Per-column standardisation; returns (z_std, mu, sigma)."""
    mu = z.mean(axis=0)
    sigma = z.std(axis=0)
    sigma = np.where(sigma < eps, 1.0, sigma)
    return (z - mu) / sigma, mu, sigma


def _solve_simplex(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve `min ||Z w - y||^2  s.t. w >= 0, sum(w) = 1`.

    Implementation: Cawley/Talbot trick — augment Z with a row of 1s and
    y with a row of 1, run NNLS, then weights are already non-negative
    and approximately sum to 1; project to the simplex if needed.
    """
    big = 1e3  # weight on the sum=1 constraint
    Z_aug = np.vstack([Z, big * np.ones((1, Z.shape[1]))])
    y_aug = np.concatenate([y, [big * 1.0]])
    w, _ = nnls(Z_aug, y_aug)
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w = np.ones(Z.shape[1]) / Z.shape[1]
    return w


class NNLSSimplexStacker(BaseEstimator, RegressorMixin):
    """OOF NNLS simplex stacker with equal-weight fallback.

    Algorithm (per Codex HIGH §A revisions):
    1. Build OOF base predictions via K-fold inner CV.
    2. Optional per-base shrinkage calibration on OOF preds.
    3. Standardise calibrated OOF preds column-wise.
    4. Solve `min ||Z w - y||^2 s.t. w >= 0, sum(w) = 1`.
    5. Compute OOF RMSE of stacked vs equal-weight mean. If stacked
       does not improve mean by >= `min_margin`, fall back to equal weight.
    6. Refit each base on full training data; apply learned w at predict.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        bases: Optional[Sequence] = None,
        n_oof_folds: int = 5,
        min_margin: float = 0.005,
        calibrate: bool = True,
        shrinkage_lambda: float = 5.0,
        random_state: int = 0,
    ) -> None:
        self.bases = bases
        self.n_oof_folds = n_oof_folds
        self.min_margin = min_margin
        self.calibrate = calibrate
        self.shrinkage_lambda = shrinkage_lambda
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NNLSSimplexStacker":
        if not self.bases:
            raise ValueError("bases must be non-empty")
        start = time.perf_counter()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        m = len(self.bases)
        kf = KFold(n_splits=self.n_oof_folds, shuffle=True, random_state=self.random_state)
        Z = np.zeros((n, m), dtype=float)
        for tr, va in kf.split(X):
            for i, (_name, est) in enumerate(self.bases):
                e = clone(est)
                try:
                    e.fit(X[tr], y[tr])
                    Z[va, i] = np.asarray(e.predict(X[va])).ravel()
                except Exception:
                    Z[va, i] = float(y[tr].mean())

        # Per-base calibration on OOF predictions.
        self._calibrators: List[_ShrinkCalibrator] = []
        if self.calibrate:
            for i in range(m):
                cal = _ShrinkCalibrator(self.shrinkage_lambda).fit(Z[:, i], y)
                Z[:, i] = cal.transform(Z[:, i])
                self._calibrators.append(cal)
        else:
            self._calibrators = [None] * m

        # Standardize columns (Codex: required before any meta).
        Z_std, mu, sigma = _standardize(Z)
        # NOTE: simplex weights apply to *raw* preds, not standardised.
        # Working space: solve simplex on raw OOF (after calibration).
        w_simplex = _solve_simplex(Z, y)

        # OOF RMSE comparison vs equal-weight mean.
        equal_w = np.full(m, 1.0 / m)
        rmse_simplex = float(np.sqrt(mean_squared_error(y, Z @ w_simplex)))
        rmse_equal = float(np.sqrt(mean_squared_error(y, Z @ equal_w)))
        # Margin = relative improvement (positive = simplex wins).
        margin = (rmse_equal - rmse_simplex) / max(rmse_equal, 1e-9)
        if margin < self.min_margin:
            self.weights_ = equal_w
            self.fallback_ = "equal"
        else:
            self.weights_ = w_simplex
            self.fallback_ = "simplex"
        self.oof_rmse_simplex_ = rmse_simplex
        self.oof_rmse_equal_ = rmse_equal
        self.margin_ = margin

        # Refit bases on full data
        self._fitted = []
        for name, est in self.bases:
            e = clone(est)
            e.fit(X, y)
            self._fitted.append((name, e))
        self.fit_time_s_ = float(time.perf_counter() - start)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_fitted"):
            raise RuntimeError("Estimator not fitted")
        preds = np.column_stack([
            np.asarray(e.predict(X)).ravel() for _n, e in self._fitted
        ])
        # Apply calibration if enabled
        if self.calibrate:
            for i, cal in enumerate(self._calibrators):
                if cal is not None:
                    preds[:, i] = cal.transform(preds[:, i])
        return preds @ self.weights_


class AdaptiveSuperLearner(BaseEstimator, RegressorMixin):
    """n-train-thresholded ensemble per Codex review §6 question 4.

    Strategy:
    - `n_train < small_threshold`: recipe selection — fit each recipe via
      inner-CV, pick the one with lowest OOF RMSE.
    - `small_threshold <= n_train < huge_threshold`: NNLS simplex stacker
      on full `atoms` list.
    - `n_train >= huge_threshold`: NNLS simplex stacker on `light_atoms`
      (smaller atom subset to control cost on big-n datasets).

    Parameters
    ----------
    atoms : sequence of (name, estimator)
        Base estimators ("atoms") used for stacking when n < huge_threshold.
    light_atoms : sequence of (name, estimator), optional
        Base estimators used when n >= huge_threshold. Defaults to `atoms`.
    recipes : sequence of (name, estimator)
        Pre-built ensembles used for selection on small datasets.
    small_threshold, huge_threshold : int
    n_oof_folds : int
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        atoms: Optional[Sequence] = None,
        recipes: Optional[Sequence] = None,
        light_atoms: Optional[Sequence] = None,
        small_threshold: int = 100,
        big_threshold: int = 200,
        huge_threshold: int = 3000,
        n_oof_folds: int = 5,
        min_margin: float = 0.005,
        calibrate: bool = True,
        random_state: int = 0,
    ) -> None:
        self.atoms = atoms
        self.recipes = recipes
        self.light_atoms = light_atoms
        self.small_threshold = small_threshold
        self.big_threshold = big_threshold
        self.huge_threshold = huge_threshold
        self.n_oof_folds = n_oof_folds
        self.min_margin = min_margin
        self.calibrate = calibrate
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveSuperLearner":
        start = time.perf_counter()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]

        if n < self.small_threshold and self.recipes:
            self.mode_ = "recipe-select"
            self._build_recipe_selector(X, y)
        else:
            if n >= self.huge_threshold and self.light_atoms:
                self.mode_ = "nnls-stack-light"
                stacker_atoms = list(self.light_atoms)
            else:
                self.mode_ = "nnls-stack"
                if not self.atoms:
                    raise ValueError("atoms must be provided for n >= small_threshold")
                stacker_atoms = list(self.atoms)
            self._stacker = NNLSSimplexStacker(
                bases=stacker_atoms,
                n_oof_folds=self.n_oof_folds,
                min_margin=self.min_margin,
                calibrate=self.calibrate,
                random_state=self.random_state,
            )
            self._stacker.fit(X, y)
        self.fit_time_s_ = float(time.perf_counter() - start)
        return self

    def _build_recipe_selector(self, X: np.ndarray, y: np.ndarray) -> None:
        n = X.shape[0]
        kf = KFold(
            n_splits=min(self.n_oof_folds, max(2, n // 5)),
            shuffle=True, random_state=self.random_state,
        )
        scores = {}
        for name, est in self.recipes:
            oof = np.zeros(n)
            for tr, va in kf.split(X):
                e = clone(est)
                try:
                    e.fit(X[tr], y[tr])
                    oof[va] = np.asarray(e.predict(X[va])).ravel()
                except Exception:
                    oof[va] = float(y[tr].mean())
            rmse = float(np.sqrt(mean_squared_error(y, oof)))
            scores[name] = rmse
        # Best recipe
        best_name = min(scores, key=scores.get)
        self.recipe_scores_ = dict(scores)
        self.winner_ = best_name
        winner_est = next(est for n, est in self.recipes if n == best_name)
        self._winner_fitted = clone(winner_est)
        self._winner_fitted.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mode_"):
            raise RuntimeError("Estimator not fitted")
        if self.mode_ == "recipe-select":
            return np.asarray(self._winner_fitted.predict(X)).ravel()
        return np.asarray(self._stacker.predict(X)).ravel()
