"""Single-chain PLS-Ridge.

Given a *single* ``(base, chain)`` candidate, fit a PLS-Ridge regressor
on the transformed spectra ``B(X) A^T``: extract ``H`` PLS components
on the centred data and shrink the latent regression coefficients with

    c_h = (t_h^T y) / (t_h^T t_h + lambda_h),
    lambda_h = lambda_0 * h ** gamma   if component_shrinkage_gamma > 0,

so the final prediction is ``yhat = T @ c + y_mean``. The class is
sklearn-style with ``fit / predict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

from nirs4all.operators.models._aom_nirs.fast.bases import BaseTransform
from nirs4all.operators.models._aom_nirs.fast.operator_chain import OperatorChain

from sklearn.model_selection import KFold

from ._common import center_y, extract_pls_scores, ridge_on_scores


class SingleChainPLSRidge(BaseEstimator, RegressorMixin):
    """PLS-Ridge on a single preprocessing chain over a single base.

    Attributes (after :meth:`fit`):
        x_mean_: training-fold mean of ``B(X)``
        y_mean_: training-fold mean of ``y``
        T_: latent scores ``(n, H)``
        W_: PLS weights ``(p, H)``
        P_: PLS loadings ``(p, H)``
        coef_latent_: Ridge coefficients on latent scores ``(H,)``
        coef_: regression coefficients in original space ``(p,)``
        lambda_: chosen ``lambda_0``
        n_components_: actual number of components extracted

    Args:
        base: Fold-aware nonlinear base transform.
        chain: Operator chain to apply after the base.
        n_components: Maximum number of latent components ``H``.
        lambdas: Candidate ``lambda_0`` grid for Ridge.
        component_shrinkage_gamma: If not None, late components are
            shrunk more aggressively: ``lambda_h = lambda_0 * h ** gamma``.
        center_y: Always center ``y`` (set to False only for tests).
    """

    def __init__(
        self,
        base: BaseTransform,
        chain: OperatorChain,
        n_components: int = 15,
        lambdas: Optional[Sequence[float]] = None,
        component_shrinkage_gamma: Optional[float] = None,
        center_y: bool = True,
        cv_n_components: bool = False,
        cv_folds: int = 5,
        cv_random_state: int = 0,
    ) -> None:
        self.base = base
        self.chain = chain
        self.n_components = int(n_components)
        if lambdas is None:
            lambdas = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
        self.lambdas = tuple(float(l) for l in lambdas)
        self.component_shrinkage_gamma = component_shrinkage_gamma
        self.center_y = bool(center_y)
        self.cv_n_components = bool(cv_n_components)
        self.cv_folds = int(cv_folds)
        self.cv_random_state = int(cv_random_state)

    def _transform(self, X: np.ndarray, fit: bool, y: Optional[np.ndarray] = None) -> np.ndarray:
        if fit:
            # Supervised bases (OSCBase, SNVOSCBase) require ``y`` at fit
            # time. The base.fit_transform contract accepts ``y=None`` for
            # unsupervised bases (raw, SNV, MSC), so always pass it through.
            Xb = self.base.fit_transform(X, y=y)
        else:
            Xb = self.base.transform(X)
        Xb = np.asarray(Xb, dtype=float)
        return self.chain.transform(Xb)

    def _cv_select_n_components(
        self, Xc: np.ndarray, yc: np.ndarray
    ) -> int:
        """K-fold CV inside training fold to pick ``n_components_*`` in {1..max}.

        Each fold:
          1. Centre Xc, yc on the train sub-fold.
          2. Extract up to ``self.n_components`` PLS components.
          3. For ``k = 1..n_extracted``, compute the validation RMSE on the
             held-out fold using the first ``k`` latent dimensions.
          4. Record the per-k validation RMSE.

        Pick the ``k`` that minimises the mean validation RMSE across folds.
        Implements the same protocol as ``aompls.selection`` ``criterion="cv"``.
        """
        n = Xc.shape[0]
        if n < 2 * self.cv_folds:  # not enough samples — fall back to max
            return self.n_components
        # Use sklearn's KFold with shuffle=True so the fold splits match
        # the protocol used by ``aompls.selection`` (see selection.py line
        # 266) for apples-to-apples comparison with ``ASLS-AOM-compact-cv5``.
        splitter = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.cv_random_state)
        folds = [(tr, va) for tr, va in splitter.split(Xc)]
        per_k_errors: list = []
        max_k = max(1, min(self.n_components, n - n // self.cv_folds - 1))
        for tr, va in folds:
            X_tr = Xc[tr]
            y_tr = yc[tr]
            X_va = Xc[va]
            y_va = yc[va]
            mu_x = X_tr.mean(axis=0)
            mu_y = float(y_tr.mean())
            X_tr_c = X_tr - mu_x
            y_tr_c = y_tr - mu_y
            X_va_c = X_va - mu_x
            T_tr, W, P = extract_pls_scores(X_tr_c, y_tr_c, n_components=max_k)
            H = T_tr.shape[1]
            if H == 0:
                continue
            # Build prediction for k = 1..H using prefix of W, P
            PtW = P.T @ W  # (H, H)
            try:
                inv_PtW = np.linalg.inv(PtW)
            except np.linalg.LinAlgError:
                inv_PtW = np.linalg.pinv(PtW)
            # latent regression coefficients via Ridge
            c_full, _, _ = ridge_on_scores(
                T_tr, y_tr_c, self.lambdas,
                component_shrinkage_gamma=self.component_shrinkage_gamma,
            )
            errors_k = np.full(self.n_components, np.inf)
            for k in range(1, H + 1):
                W_k = W[:, :k]
                P_k = P[:, :k]
                c_k = c_full[:k]
                # beta_k = W_k @ inv(P_k^T W_k) @ c_k
                try:
                    inv_k = np.linalg.inv(P_k.T @ W_k)
                except np.linalg.LinAlgError:
                    inv_k = np.linalg.pinv(P_k.T @ W_k)
                beta_k = W_k @ inv_k @ c_k
                yhat = X_va_c @ beta_k + mu_y
                errors_k[k - 1] = float(np.mean((y_va - yhat) ** 2))
            per_k_errors.append(errors_k)
        if not per_k_errors:
            return self.n_components
        mean_err = np.mean(np.stack(per_k_errors), axis=0)
        best_k = int(np.argmin(mean_err)) + 1
        return best_k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SingleChainPLSRidge":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        Xt = self._transform(X, fit=True, y=y)
        self.x_mean_ = Xt.mean(axis=0)
        Xc = Xt - self.x_mean_
        if self.center_y:
            yc, y_mean = center_y(y)
        else:
            yc, y_mean = y.copy(), 0.0
        if self.cv_n_components and Xc.shape[0] >= 2 * self.cv_folds:
            chosen_k = self._cv_select_n_components(Xc, yc)
        else:
            chosen_k = self.n_components
        T, W, P = extract_pls_scores(Xc, yc, n_components=chosen_k)
        c, lam, loss = ridge_on_scores(
            T,
            yc,
            self.lambdas,
            component_shrinkage_gamma=self.component_shrinkage_gamma,
        )
        # Regression coefficient in original (transformed) space:
        # beta = W (P^T W)^{-1} c
        H = T.shape[1]
        if H == 0:
            self.coef_ = np.zeros(Xc.shape[1])
        else:
            PtW = P.T @ W  # (H, H)
            try:
                inv_PtW = np.linalg.inv(PtW)
            except np.linalg.LinAlgError:
                inv_PtW = np.linalg.pinv(PtW)
            self.coef_ = W @ inv_PtW @ c
        self.intercept_ = float(y_mean - self.x_mean_ @ self.coef_)
        self.y_mean_ = y_mean
        self.T_ = T
        self.W_ = W
        self.P_ = P
        self.coef_latent_ = c
        self.lambda_ = lam
        self.n_components_ = int(T.shape[1])
        self.train_loss_ = loss
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("SingleChainPLSRidge.predict called before fit")
        X = np.asarray(X, dtype=float)
        Xt = self._transform(X, fit=False)
        return Xt @ self.coef_ + self.intercept_
