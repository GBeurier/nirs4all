"""Soft AOM-chain PLS-Ridge.

A relaxation of :class:`HardAOMChainPLSRidge` where each latent
component is a *non-negative sparse mixture* of candidate vectors,
solved by coordinate-descent NNLS on a column-stacked design.

At step ``h``:

  1. Build candidate proposals ``u_s = X_s X_s^T residual_y`` for every
     ``(base, chain)`` candidate, with exact transformed matrices.
  2. Orthogonalise each ``u_s`` against the existing latent scores
     ``T_{<h}`` (Gram-Schmidt with the same projection coefficients
     reused on test data).
  3. Solve

         min_{a >= 0} || residual_y - U a ||^2 + rho ||a||_1

     via coordinate descent.
  4. Form the latent score ``t_h = U a / ||U a||`` (training side),
     ``T_test[:, h] = U_test a / ||U_train a||`` (test side).

After ``H`` components we fit a Ridge on the latent scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from FastAOM.bases import BaseTransform
from FastAOM.lowrank import LowRankBase
from FastAOM.operator_chain import OperatorChain

from ._common import ridge_on_scores


@dataclass
class SoftAOMComponent:
    candidate_indices: List[int]
    weights: List[float]
    base_indices: List[int]
    chains: List[OperatorChain]
    base_names: List[str]
    proj_coef: np.ndarray  # projection of mixed train direction onto T_prev
    norm: float


def _nonneg_lasso_cd(
    U: np.ndarray,
    y: np.ndarray,
    rho: float,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> np.ndarray:
    """Solve ``min_{a >= 0} 0.5 ||y - U a||^2 + rho ||a||_1`` via coordinate descent.

    Note the **0.5** factor on the squared-loss term: the soft-threshold is
    ``max(0, (U_j^T y_res - rho) / ||U_j||^2)`` accordingly. If callers prefer
    the unscaled-loss convention ``||y - U a||^2 + rho ||a||_1``, halve their
    ``rho`` before passing it in.
    """
    n, m = U.shape
    if m == 0:
        return np.zeros(0)
    a = np.zeros(m)
    col_sqnorm = np.sum(U * U, axis=0)
    residual = y.copy()
    for _ in range(max_iter):
        a_prev = a.copy()
        for j in range(m):
            if col_sqnorm[j] < 1e-12:
                continue
            residual = residual + U[:, j] * a[j]
            rho_j = float(U[:, j] @ residual)
            new_aj = max(0.0, (rho_j - rho) / col_sqnorm[j])
            a[j] = new_aj
            residual = residual - U[:, j] * a[j]
        if np.max(np.abs(a - a_prev)) < tol:
            break
    return a


class SoftAOMChainPLSRidge(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        bases: Sequence[BaseTransform],
        lowrank_bases: Sequence[LowRankBase],
        candidates: Sequence[Tuple[int, OperatorChain]],
        n_components: int = 15,
        rho: float = 0.05,
        lambdas: Optional[Sequence[float]] = None,
        component_shrinkage_gamma: Optional[float] = None,
        max_mixture_size: Optional[int] = 5,
        cd_iters: int = 200,
        eps: float = 1e-12,
    ) -> None:
        if len(bases) != len(lowrank_bases):
            raise ValueError("bases and lowrank_bases must have the same length")
        self.bases = list(bases)
        self.lowrank_bases = list(lowrank_bases)
        self.candidates = list(candidates)
        self.n_components = int(n_components)
        self.rho = float(rho)
        if lambdas is None:
            lambdas = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
        self.lambdas = tuple(float(l) for l in lambdas)
        self.component_shrinkage_gamma = component_shrinkage_gamma
        self.max_mixture_size = max_mixture_size
        self.cd_iters = int(cd_iters)
        self.eps = float(eps)

    def _exact_kernel_apply(
        self,
        base_idx: int,
        chain: OperatorChain,
        v: np.ndarray,
        cache: Dict[Tuple[int, str], np.ndarray],
    ) -> np.ndarray:
        key = (base_idx, chain.signature)
        Xt = cache.get(key)
        if Xt is None:
            Xt = chain.transform(self.lowrank_bases[base_idx].X_centred)
            cache[key] = Xt
        return Xt @ (Xt.T @ v)

    def fit(self, X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None) -> "SoftAOMChainPLSRidge":
        y_centred = self.lowrank_bases[0].y_centred
        n = y_centred.shape[0]
        if not self.candidates:
            raise ValueError("No candidates provided")
        Xt_train_cache: Dict[Tuple[int, str], np.ndarray] = {}

        T_cols: List[np.ndarray] = []
        components: List[SoftAOMComponent] = []
        residual_y = y_centred.copy()
        Tmat_current: Optional[np.ndarray] = None

        for h in range(self.n_components):
            cols: List[np.ndarray] = []
            for base_idx, chain in self.candidates:
                u_raw = self._exact_kernel_apply(base_idx, chain, residual_y, Xt_train_cache)
                cols.append(u_raw)
            U_raw = np.column_stack(cols)
            # Orthogonalize each column against T_train
            if Tmat_current is not None and Tmat_current.shape[1] > 0:
                Gram = Tmat_current.T @ Tmat_current
                try:
                    coef_mat = np.linalg.solve(Gram, Tmat_current.T @ U_raw)
                except np.linalg.LinAlgError:
                    coef_mat = np.linalg.lstsq(Gram, Tmat_current.T @ U_raw, rcond=None)[0]
                U_orth = U_raw - Tmat_current @ coef_mat
            else:
                U_orth = U_raw

            a = _nonneg_lasso_cd(U_orth, residual_y, rho=self.rho, max_iter=self.cd_iters)
            if self.max_mixture_size is not None and a.size > self.max_mixture_size:
                contribution = np.abs(a) * np.linalg.norm(U_orth, axis=0)
                top_idx = np.argsort(-contribution)[: self.max_mixture_size]
                mask = np.zeros_like(a, dtype=bool)
                mask[top_idx] = True
                a = np.where(mask, a, 0.0)
            nz = np.where(a > self.eps)[0]
            if nz.size == 0:
                inner = U_orth.T @ y_centred
                norms = np.linalg.norm(U_orth, axis=0)
                norms = np.where(norms < self.eps, 1.0, norms)
                score = (inner ** 2) / (norms ** 2)
                best = int(np.argmax(score))
                a = np.zeros_like(a)
                a[best] = 1.0
                nz = np.array([best])
            t_unnorm = U_orth @ a
            t_norm = float(np.linalg.norm(t_unnorm))
            if t_norm < self.eps:
                break
            t_h = t_unnorm / t_norm
            T_cols.append(t_h)
            # Compute the mixed projection coefficient on T_train (for predict reuse):
            # The mixed raw direction is U_raw @ a; project onto T_train.
            mixed_raw = U_raw @ a
            if Tmat_current is not None and Tmat_current.shape[1] > 0:
                Gram = Tmat_current.T @ Tmat_current
                try:
                    proj_coef = np.linalg.solve(Gram, Tmat_current.T @ mixed_raw)
                except np.linalg.LinAlgError:
                    proj_coef = np.linalg.lstsq(Gram, Tmat_current.T @ mixed_raw, rcond=None)[0]
            else:
                proj_coef = np.zeros(0)
            cand_idx_list = [int(i) for i in nz]
            components.append(
                SoftAOMComponent(
                    candidate_indices=cand_idx_list,
                    weights=[float(a[i]) for i in cand_idx_list],
                    base_indices=[int(self.candidates[i][0]) for i in cand_idx_list],
                    chains=[self.candidates[i][1] for i in cand_idx_list],
                    base_names=[self.bases[self.candidates[i][0]].signature for i in cand_idx_list],
                    proj_coef=proj_coef,
                    norm=t_norm,
                )
            )
            Tmat_current = np.column_stack(T_cols)
            Gram = Tmat_current.T @ Tmat_current
            try:
                coef_proj = np.linalg.solve(Gram, Tmat_current.T @ y_centred)
            except np.linalg.LinAlgError:
                coef_proj = np.linalg.lstsq(Gram, Tmat_current.T @ y_centred, rcond=None)[0]
            residual_y = y_centred - Tmat_current @ coef_proj

        if not T_cols:
            raise RuntimeError("SoftAOMChainPLSRidge extracted no latent components")

        Tmat = np.column_stack(T_cols)
        c, lam, loss = ridge_on_scores(
            Tmat,
            y_centred,
            self.lambdas,
            component_shrinkage_gamma=self.component_shrinkage_gamma,
        )
        self.T_ = Tmat
        self.coef_latent_ = c
        self.lambda_ = lam
        self.train_loss_ = loss
        self.n_components_ = Tmat.shape[1]
        self.components_ = components
        self.y_mean_ = float(np.asarray(y_train).mean()) if y_train is not None else 0.0
        self._Xt_train_cache_ = Xt_train_cache
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "components_") or not self.components_:
            raise RuntimeError("SoftAOMChainPLSRidge.predict called before fit")
        X = np.asarray(X, dtype=float)
        n_test = X.shape[0]
        H = len(self.components_)
        T_test = np.zeros((n_test, H))
        y_centred = self.lowrank_bases[0].y_centred
        residual_y_train = y_centred.copy()
        T_train_cols: List[np.ndarray] = []
        base_test_cache: Dict[int, np.ndarray] = {}

        for h, comp in enumerate(self.components_):
            mixed_train_raw = np.zeros_like(y_centred)
            mixed_test_raw = np.zeros(n_test)
            for w, cand_idx, base_idx, chain in zip(
                comp.weights, comp.candidate_indices, comp.base_indices, comp.chains
            ):
                lr = self.lowrank_bases[base_idx]
                key = (base_idx, chain.signature)
                Xt_train = self._Xt_train_cache_.get(key)
                if Xt_train is None:
                    Xt_train = chain.transform(lr.X_centred)
                    self._Xt_train_cache_[key] = Xt_train
                if base_idx not in base_test_cache:
                    Xb_test = self.bases[base_idx].transform(X)
                    base_test_cache[base_idx] = Xb_test - lr.mean
                Xc_test = base_test_cache[base_idx]
                Xt_test = chain.transform(Xc_test)
                w_s = Xt_train.T @ residual_y_train
                mixed_train_raw = mixed_train_raw + w * (Xt_train @ w_s)
                mixed_test_raw = mixed_test_raw + w * (Xt_test @ w_s)
            if T_train_cols:
                Tmat_train = np.column_stack(T_train_cols)
                mixed_train = mixed_train_raw - Tmat_train @ comp.proj_coef
                mixed_test = mixed_test_raw - T_test[:, :h] @ comp.proj_coef
            else:
                mixed_train = mixed_train_raw
                mixed_test = mixed_test_raw
            norm = comp.norm
            if norm < self.eps:
                t_train = np.zeros_like(mixed_train)
                T_test[:, h] = 0.0
            else:
                t_train = mixed_train / norm
                T_test[:, h] = mixed_test / norm
            T_train_cols.append(t_train)
            Tmat = np.column_stack(T_train_cols)
            Gram = Tmat.T @ Tmat
            try:
                coef_y = np.linalg.solve(Gram, Tmat.T @ y_centred)
            except np.linalg.LinAlgError:
                coef_y = np.linalg.lstsq(Gram, Tmat.T @ y_centred, rcond=None)[0]
            residual_y_train = y_centred - Tmat @ coef_y

        yhat_centred = T_test @ self.coef_latent_
        return yhat_centred + self.y_mean_
