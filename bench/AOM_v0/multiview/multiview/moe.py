"""AOM-MoE: mixture-of-experts wrapper around PLS.

Implements hard / soft MoE with PLS experts (per-view or per-preproc).
The gate is trained via NNLS on out-of-fold predictions; this is the v1
"constant gate" specified in `DESIGN_MOE.md` §3.2.

The AOM-per-LV routing variant is mathematically equivalent to Phase 2
block-sparse AOM-MBPLS — see DESIGN_MOE §3.3. It is not re-implemented here.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from aompls.banks import bank_by_name
from aompls.operators import IdentityOperator, LinearSpectralOperator
from .views import _equal_width_blocks


class AOMMoERegressor(BaseEstimator, RegressorMixin):
    """Mixture-of-experts regressor with PLS experts.

    Parameters
    ----------
    expert_layout : str
        "per_view" — one PLS per equal-width wavelength block (K experts).
        "per_preproc" — one PLS per operator in the preproc bank
        (8 ops + identity ≈ 9 experts when bank='compact').
    routing : str
        "hard" — gate picks single expert via argmax of OOF score.
        "soft" — NNLS-weighted mixture.
    K : int
        Number of equal-width blocks (per_view only).
    bank_name : str
        Preproc bank name (per_preproc only). Default 'compact'.
    per_expert_components : int
        Fixed n_components for every PLS expert. Default 10.
    n_oof_folds : int
        K-fold count for generating OOF predictions to train the gate.
    random_state : int
        Seed for fold splits.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        expert_layout: str = "per_view",
        routing: str = "soft",
        K: int = 3,
        bank_name: str = "compact",
        per_expert_components: int = 10,
        n_oof_folds: int = 3,
        random_state: int = 0,
    ) -> None:
        self.expert_layout = expert_layout
        self.routing = routing
        self.K = K
        self.bank_name = bank_name
        self.per_expert_components = per_expert_components
        self.n_oof_folds = n_oof_folds
        self.random_state = random_state

    # ------------------------------------------------------------------ Experts

    def _build_experts(self, p: int) -> List[Tuple[str, callable]]:
        """Return a list of `(name, view_fn)` where `view_fn(X)` produces the
        per-expert design matrix.
        """
        if self.expert_layout == "per_view":
            blocks = _equal_width_blocks(p, self.K)
            return [
                (f"view_{k}_[{s},{e})", _make_block_view(s, e))
                for k, (s, e) in enumerate(blocks)
            ]
        if self.expert_layout == "per_preproc":
            bank = bank_by_name(self.bank_name, p=p)
            return [
                (op.name, _make_op_view(op))
                for op in bank
            ]
        raise ValueError(f"unknown expert_layout: {self.expert_layout!r}")

    # ------------------------------------------------------------------ Fit

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AOMMoERegressor":
        start = time.perf_counter()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        experts = self._build_experts(p)
        E = len(experts)

        # Out-of-fold predictions per expert via K-fold.
        kf = KFold(n_splits=self.n_oof_folds, shuffle=True, random_state=self.random_state)
        oof_pred = np.zeros((n, E), dtype=float)
        for train_idx, val_idx in kf.split(X):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr = y[train_idx]
            for e, (_name, view_fn) in enumerate(experts):
                X_tr_e = view_fn(X_tr)
                X_va_e = view_fn(X_va)
                k = min(self.per_expert_components, max(1, X_tr.shape[0] - 1), X_tr_e.shape[1])
                try:
                    pls = PLSRegression(n_components=k)
                    pls.fit(X_tr_e, y_tr)
                    oof_pred[val_idx, e] = pls.predict(X_va_e).ravel()
                except Exception:
                    oof_pred[val_idx, e] = float(y_tr.mean())

        # Train gate via NNLS minimising || oof_pred · w - y ||.
        # NNLS yields nonneg weights (interpretable as expert weights).
        weights, _ = nnls(oof_pred, y)
        if weights.sum() == 0:
            weights = np.ones(E) / E
        else:
            weights = weights / weights.sum()  # normalise to sum to 1

        if self.routing == "hard":
            argmax = int(np.argmax(weights))
            weights = np.zeros(E)
            weights[argmax] = 1.0

        # Re-fit each expert on the FULL training data for prediction.
        full_experts = []
        for _name, view_fn in experts:
            X_full = view_fn(X)
            k = min(self.per_expert_components, max(1, n - 1), X_full.shape[1])
            pls = PLSRegression(n_components=k)
            pls.fit(X_full, y)
            full_experts.append(pls)

        self.experts_ = experts
        self.full_experts_ = full_experts
        self.gate_weights_ = weights
        self.fit_time_s_ = float(time.perf_counter() - start)
        return self

    # ------------------------------------------------------------------ Predict

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "full_experts_"):
            raise RuntimeError("Estimator not fitted")
        X = np.asarray(X, dtype=float)
        E = len(self.experts_)
        preds = np.zeros((X.shape[0], E))
        for e, ((_name, view_fn), pls) in enumerate(zip(self.experts_, self.full_experts_)):
            X_e = view_fn(X)
            preds[:, e] = pls.predict(X_e).ravel()
        return preds @ self.gate_weights_

    def get_gate_weights(self) -> np.ndarray:
        return self.gate_weights_.copy()


# ---------------------------------------------------------------------------
# View constructors (closures so estimators are pickle-friendly outside tests)
# ---------------------------------------------------------------------------


def _make_block_view(start: int, end: int):
    def _view(X: np.ndarray) -> np.ndarray:
        return X[:, start:end]
    return _view


def _make_op_view(op: LinearSpectralOperator):
    def _view(X: np.ndarray) -> np.ndarray:
        # Operators may need to be fit on the first call; this keeps the
        # transform idempotent for shape-only operators (BlockMask, identity).
        try:
            op.fit(X)
        except Exception:
            pass
        return op.transform(X)
    return _view
