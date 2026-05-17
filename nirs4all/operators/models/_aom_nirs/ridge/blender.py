"""Convex non-negative blender of AOM-Ridge variant out-of-fold predictions.

For each candidate variant the blender:

1. Runs an *outer* K-fold cross-validation. In every fold a fresh estimator
   is built from the candidate spec, fitted on the outer-train slice, and
   predictions are recorded on the outer-validation slice. The concatenated
   per-fold predictions form an out-of-fold (OOF) prediction column.
2. Stacks the OOF columns into ``Z`` of shape ``(n_samples, n_candidates)``
   (or ``(n_samples, n_targets, n_candidates)`` for multi-output ``y``).
3. Solves the regularised convex QP

   .. math::
      \\min_{w} \\tfrac{1}{2} \\| y - Z w \\|^2
              + \\tfrac{\\lambda}{2} \\| w - \\tfrac{1}{K} \\mathbf{1} \\|^2
      \\;\\text{s.t.}\\; w \\ge 0,\\; \\sum_k w_k = 1.

   Using ``scipy.optimize.minimize(method="SLSQP")``. The regularisation
   biases the solution toward the uniform mixture ``w = 1/K`` so that small
   datasets fall back to plain averaging.
4. Refits every candidate on the full training set. ``predict`` builds
   ``Z_test`` of test-side predictions and returns ``Z_test @ weights_``.

Anti-leakage invariant: an outer fold's validation rows never participate in
candidate fitting nor in OOF prediction for that fold's variant, because
each fold rebuilds the estimator with ``X[outer_train_idx]`` only and the
branch preprocessor (when any) is fitted on outer-train rows only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .auto_selector import (
    _apply_branch,
    _default_headline_candidates,
    _dispatch_candidate,
    _ravel_match,
    _resolve_outer_cv,
)

VariantSpec = dict[str, Any]


# ----------------------------------------------------------------------
# OOF prediction helpers
# ----------------------------------------------------------------------


def _oof_predictions_for_candidate(
    spec: VariantSpec,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> np.ndarray:
    """Return OOF predictions for one candidate, aligned with ``y``.

    Output shape mirrors ``y``: ``(n,)`` for 1D, ``(n, q)`` for 2D.
    """
    if y.ndim == 1:
        oof = np.full(y.shape[0], np.nan, dtype=float)
    else:
        oof = np.full(y.shape, np.nan, dtype=float)
    for tr_idx, va_idx in folds:
        X_tr_raw, X_va_raw = X[tr_idx], X[va_idx]
        y_tr = y[tr_idx]
        est, branch = _dispatch_candidate(spec, seed=seed, inner_cv=3)
        X_tr, X_va = _apply_branch(branch, X_tr_raw, y_tr, X_va_raw)
        est.fit(X_tr, y_tr)
        y_pred = est.predict(X_va)
        y_pred = _ravel_match(y_pred, y[va_idx])
        oof[va_idx] = y_pred
    if np.isnan(oof).any():
        # Splitter did not cover every row — fill missing rows with the
        # column mean of observed predictions so the QP stays well-posed.
        # In practice ``KFold``/``SPXYFold`` cover all rows exactly once.
        if y.ndim == 1:
            mean_val = float(np.nanmean(oof)) if np.isfinite(oof).any() else 0.0
            oof = np.where(np.isnan(oof), mean_val, oof)
        else:
            col_means = np.nanmean(oof, axis=0)
            col_means = np.where(np.isfinite(col_means), col_means, 0.0)
            for j in range(oof.shape[1]):
                col = oof[:, j]
                col[np.isnan(col)] = col_means[j]
                oof[:, j] = col
    return oof


# ----------------------------------------------------------------------
# Convex weight solver
# ----------------------------------------------------------------------


def _solve_simplex_qp(
    Z: np.ndarray,
    y: np.ndarray,
    regularizer: float,
) -> np.ndarray:
    """Solve the regularised non-negative simplex QP for blending weights.

    Minimises ``0.5 * ||y - Z w||^2 + 0.5 * lambda * ||w - 1/K||^2`` subject
    to ``w >= 0`` and ``sum(w) = 1`` using SLSQP. ``Z`` has shape
    ``(n_samples * n_targets, n_candidates)`` when stacking multi-output
    predictions; ``y`` is the matching flat vector.
    """
    k = Z.shape[1]
    if k <= 0:
        raise ValueError("Z must have at least one candidate column")
    uniform = np.full(k, 1.0 / k, dtype=float)
    lam = float(max(regularizer, 0.0))

    # Closed-form gradient: g(w) = Z.T @ (Z w - y) + lam * (w - 1/K)
    ZtZ = Z.T @ Z
    Zty = Z.T @ y

    def objective(w: np.ndarray) -> float:
        residual = Z @ w - y
        data_term = 0.5 * float(residual @ residual)
        reg_term = 0.5 * lam * float(np.sum((w - uniform) ** 2))
        return data_term + reg_term

    def gradient(w: np.ndarray) -> np.ndarray:
        return ZtZ @ w - Zty + lam * (w - uniform)

    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0),
                    "jac": lambda w: np.ones_like(w)},)
    bounds = [(0.0, 1.0)] * k

    result = minimize(
        objective,
        x0=uniform,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10, "disp": False},
    )
    w = np.asarray(result.x, dtype=float)
    # Numerical clean-up: clip then renormalise to land exactly on the simplex.
    w = np.clip(w, 0.0, None)
    s = float(np.sum(w))
    if s <= 0.0:
        w = uniform.copy()
    else:
        w = w / s
    return w


# ----------------------------------------------------------------------
# Estimator
# ----------------------------------------------------------------------


class AOMRidgeBlender(BaseEstimator, RegressorMixin):
    """Convex non-negative blender of AOM-Ridge variant OOF predictions.

    For each candidate variant: compute outer-CV out-of-fold predictions,
    stack them into ``Z`` of shape ``(n_samples, n_candidates)``, then solve

    .. math::
       \\min_w \\| y - Z w \\|^2
            + \\lambda \\| w - 1/K \\|^2
       \\;\\text{s.t.}\\; w \\ge 0,\\; \\sum_k w_k = 1.

    via SLSQP. Each candidate is then refit on the full training set; new
    predictions are blended as ``Z_test @ weights_``.

    Parameters
    ----------
    candidates : sequence of variant specs (dicts) or callables, or ``None``
        Same convention as :class:`AOMRidgeAutoSelector`. When ``None``
        (default), uses the 8 HEADLINE variants minus any ``auto_select``
        and ``blender`` entries (recursion guard).
    outer_cv : int or splitter
        K-fold outer CV. Integer is interpreted by ``outer_cv_kind``.
    outer_cv_kind : {"spxy", "kfold", "spxy_repeated"}
        Splitter style when ``outer_cv`` is integer.
    outer_cv_repeats : int
        Repeats for ``outer_cv_kind="spxy_repeated"`` (ignored otherwise).
    regularizer : float
        ``lambda`` for the bias-toward-uniform penalty in the QP. Larger
        values shrink weights toward ``1/K`` (uniform averaging).
    scoring : {"mse", "rmse"}
        Reported in :meth:`get_diagnostics` for completeness; the QP
        objective is always squared error so the blender weights are
        invariant under this choice.
    random_state : int
        Seed forwarded to splitters and inner CVs.
    n_jobs : int
        joblib parallelism over candidates' outer-CV passes. ``1`` runs
        serially, ``-1`` uses all cores.

    Attributes
    ----------
    weights_ : ndarray of shape (n_candidates,)
        Convex blending weights on the simplex.
    oof_predictions_ : ndarray of shape (n_samples, n_candidates) for 1D y,
        or ``(n_samples, n_targets, n_candidates)`` for 2D y.
        OOF predictions used to solve the QP.
    selected_variant_label_ : str
        Label of the candidate carrying the largest weight (for diagnostics).
    refit_estimators_ : list
        One refit estimator per candidate, fitted on the full training set.
    refit_branch_preprocs_ : list
        Per-candidate branch preprocessor fitted on the full training set
        (or ``None`` when the candidate has no branch).
    candidates_ : list of dict
        Resolved candidate specs.
    cv_scores_ : list of float
        OOF RMSE per candidate (diagnostic).
    """

    def __init__(
        self,
        candidates: Sequence[VariantSpec | Callable[[], BaseEstimator]] | None = None,
        outer_cv: int | object = 3,
        outer_cv_kind: str = "spxy",
        outer_cv_repeats: int = 1,
        regularizer: float = 0.01,
        scoring: str = "mse",
        random_state: int = 0,
        n_jobs: int = 1,
    ) -> None:
        self.candidates = candidates
        self.outer_cv = outer_cv
        self.outer_cv_kind = outer_cv_kind
        self.outer_cv_repeats = outer_cv_repeats
        self.regularizer = regularizer
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # Spec normalisation
    # ------------------------------------------------------------------

    def _normalise_candidates(self) -> list[VariantSpec]:
        if self.candidates is None:
            base = _default_headline_candidates()
        else:
            base = []
            for i, c in enumerate(self.candidates):
                if callable(c):
                    base.append({"label": f"candidate_{i}", "factory": c})
                elif isinstance(c, dict):
                    spec = dict(c)
                    spec.setdefault("label", f"candidate_{i}")
                    base.append(spec)
                else:
                    raise TypeError(
                        f"candidate {i} must be a dict spec or a callable factory; "
                        f"got {type(c).__name__}"
                    )
        # Recursion guard: drop any auto_select / blender / residual_tabpfn entries.
        out = [
            spec for spec in base
            if spec.get("selection") not in ("auto_select", "blender", "residual_tabpfn")
        ]
        if not out:
            raise ValueError(
                "candidates must be non-empty after dropping aggregator entries"
            )
        return out

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> AOMRidgeBlender:
        if self.scoring not in ("mse", "rmse"):
            raise ValueError("scoring must be 'mse' or 'rmse'")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        y_arr = np.asarray(y, dtype=float)
        if y_arr.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if y_arr.ndim not in (1, 2):
            raise ValueError("y must be 1D or 2D")

        candidates = self._normalise_candidates()
        cv_obj = _resolve_outer_cv(
            self.outer_cv,
            kind=self.outer_cv_kind,
            repeats=self.outer_cv_repeats,
            random_state=self.random_state,
        )
        folds = list(cv_obj.split(X, y_arr))
        if not folds:
            raise ValueError("outer CV produced no folds")

        seed = int(self.random_state)

        # OOF predictions per candidate (parallel over candidates).
        if int(self.n_jobs) == 1:
            oof_list = [
                _oof_predictions_for_candidate(spec, X, y_arr, folds, seed)
                for spec in candidates
            ]
        else:
            oof_list = Parallel(n_jobs=int(self.n_jobs), backend="loky")(
                delayed(_oof_predictions_for_candidate)(spec, X, y_arr, folds, seed)
                for spec in candidates
            )

        # Stack: (n,) -> (n, K); (n, q) -> (n, q, K).
        if y_arr.ndim == 1:
            Z = np.column_stack([np.asarray(o, dtype=float) for o in oof_list])
            Z_flat = Z
            y_flat = y_arr
        else:
            Z = np.stack([np.asarray(o, dtype=float) for o in oof_list], axis=-1)
            Z_flat = Z.reshape(-1, Z.shape[-1])
            y_flat = y_arr.reshape(-1)

        weights = _solve_simplex_qp(Z_flat, y_flat, regularizer=float(self.regularizer))

        # Per-candidate OOF RMSE (diagnostic).
        cv_scores: list[float] = []
        for k_idx in range(Z_flat.shape[1]):
            diff = Z_flat[:, k_idx] - y_flat
            cv_scores.append(float(np.sqrt(np.mean(diff * diff))))

        # Refit every candidate on the full training set.
        refit_estimators: list[BaseEstimator] = []
        refit_branches: list[Any] = []
        from .branches import fit_transform_branch, make_branch_preproc
        for spec in candidates:
            est, branch = _dispatch_candidate(spec, seed=seed, inner_cv=3)
            branch_preproc = None
            X_refit = X
            if branch:
                branch_preproc = make_branch_preproc(branch)
                if branch_preproc is not None:
                    X_refit = fit_transform_branch(
                        branch_preproc, np.asarray(X, dtype=float),
                        np.asarray(y_arr, dtype=float),
                    )
            est.fit(X_refit, y_arr)
            refit_estimators.append(est)
            refit_branches.append(branch_preproc)

        best_idx = int(np.argmax(weights))
        self.candidates_ = candidates
        self.weights_ = weights
        self.oof_predictions_ = Z
        self.cv_scores_ = cv_scores
        self.selected_variant_index_ = best_idx
        self.selected_variant_label_ = str(
            candidates[best_idx].get("label", f"candidate_{best_idx}")
        )
        self.refit_estimators_ = refit_estimators
        self.refit_branch_preprocs_ = refit_branches
        self.n_features_in_ = int(X.shape[1])
        self.n_targets_ = 1 if y_arr.ndim == 1 else int(y_arr.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "refit_estimators_")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")

        preds = []
        for est, branch_preproc in zip(
            self.refit_estimators_, self.refit_branch_preprocs_, strict=True,
        ):
            X_in = X
            if branch_preproc is not None:
                X_in = np.asarray(branch_preproc.transform(X), dtype=float)
            preds.append(est.predict(X_in))

        if self.n_targets_ == 1:
            normed = [np.asarray(p, dtype=float).ravel() for p in preds]
            stack = np.column_stack(normed)
            return stack @ self.weights_
        # Multi-output: align each candidate's output to (n, q).
        n = X.shape[0]
        q = int(self.n_targets_)
        normed = []
        for p in preds:
            arr = np.asarray(p, dtype=float)
            if arr.ndim == 1 and q == 1:
                arr = arr.reshape(-1, 1)
            normed.append(arr.reshape(n, q))
        # (K, n, q) -> blend along K axis with weights_.
        stack = np.stack(normed, axis=-1)
        return stack @ self.weights_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        check_is_fitted(self, "refit_estimators_")
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        y_arr = np.asarray(y, dtype=float)
        y_pred = _ravel_match(y_pred, y_arr)
        return float(r2_score(y_arr, y_pred, multioutput="uniform_average"))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        """Return a JSON-serialisable summary of the blending solution."""
        check_is_fitted(self, "refit_estimators_")
        weights = [float(w) for w in self.weights_]
        labels = [
            str(c.get("label", f"candidate_{i}"))
            for i, c in enumerate(self.candidates_)
        ]
        ranking = sorted(
            zip(labels, weights, strict=True), key=lambda t: t[1], reverse=True,
        )
        return {
            "model": "AOMRidgeBlender",
            "weights": weights,
            "candidate_labels": labels,
            "selected_variant_label": self.selected_variant_label_,
            "selected_variant_index": int(self.selected_variant_index_),
            "cv_scores": [float(s) for s in self.cv_scores_],
            "outer_cv_kind": self.outer_cv_kind,
            "outer_cv_repeats": int(self.outer_cv_repeats),
            "regularizer": float(self.regularizer),
            "scoring": self.scoring,
            "weight_ranking": [
                {"label": lab, "weight": w} for lab, w in ranking
            ],
        }


__all__ = ["AOMRidgeBlender"]
