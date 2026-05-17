"""Hard AOM-chain PLS-Ridge.

At each latent step ``h`` we pick the single ``(base, chain)``
candidate whose contribution to the residual ``y_res`` has the highest
normalised squared correlation

    score(s, h) = (u_s^T y_centred)^2 / ||u_s||^2

after orthogonalising ``u_s = X_s X_s^T y_res`` against the
already-selected scores ``T_{<h}``. The latent score is then ``t_h =
u_{s_h} / ||u_{s_h}||``, and the residual ``y_res`` is updated by
least-squares projection onto ``T``. After ``H`` components, we fit a
Ridge on the latent scores (optionally with component-wise shrinkage).

Train and test predictions use the *exact* transformed matrices
``X_s = X_centred A_s^T`` (not the SVD approximation); the SVD-based
low-rank machinery is reserved for the screening stage above.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from FastAOM.bases import BaseTransform
from FastAOM.lowrank import LowRankBase
from FastAOM.operator_chain import OperatorChain

from ._common import center_y, ridge_on_scores


_DEFAULT_XT_CACHE_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB


@dataclass
class HardAOMComponent:
    base_index: int
    base_name: str
    chain: OperatorChain
    score: float
    norm: float


def _chain_transform_train(lr: LowRankBase, chain: OperatorChain) -> np.ndarray:
    """Return ``chain.transform(lr.X_centred)``."""
    return chain.transform(lr.X_centred)


def _chain_transform_test(
    bases: Sequence[BaseTransform],
    lowrank_bases: Sequence[LowRankBase],
    X: np.ndarray,
    base_idx: int,
    chain: OperatorChain,
    cache: Dict[int, np.ndarray],
) -> np.ndarray:
    """Return ``chain.transform(base.transform(X) - lr.mean)``, cached per base."""
    if base_idx not in cache:
        Xb_test = bases[base_idx].transform(X)
        cache[base_idx] = Xb_test - lowrank_bases[base_idx].mean
    return chain.transform(cache[base_idx])


class HardAOMChainPLSRidge(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        bases: Sequence[BaseTransform],
        lowrank_bases: Sequence[LowRankBase],
        candidates: Sequence[Tuple[int, OperatorChain]],
        n_components: int = 15,
        lambdas: Optional[Sequence[float]] = None,
        component_shrinkage_gamma: Optional[float] = None,
        early_stop_patience: Optional[int] = 3,
        eps: float = 1e-12,
    ) -> None:
        if len(bases) != len(lowrank_bases):
            raise ValueError("bases and lowrank_bases must have the same length")
        self.bases = list(bases)
        self.lowrank_bases = list(lowrank_bases)
        self.candidates = list(candidates)
        self.n_components = int(n_components)
        if lambdas is None:
            lambdas = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
        self.lambdas = tuple(float(l) for l in lambdas)
        self.component_shrinkage_gamma = component_shrinkage_gamma
        self.early_stop_patience = early_stop_patience
        self.eps = float(eps)

    def fit(self, X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None) -> "HardAOMChainPLSRidge":
        y_centred = self.lowrank_bases[0].y_centred
        n = y_centred.shape[0]
        if not self.candidates:
            raise ValueError("No candidates provided")

        # LRU-cap the per-candidate transformed-training cache: each entry is
        # ``n × p × 8`` bytes (float64). For large-n datasets (e.g. n=40k,
        # p=200) one entry is ~64 MiB, so an unbounded cache over hundreds of
        # candidates blows past system RAM. We size the cache budget to 4 GiB
        # by default; eviction is least-recently-used.
        per_entry_bytes = max(1, n * max(1, self.lowrank_bases[0].X_centred.shape[1]) * 8)
        max_entries = max(8, _DEFAULT_XT_CACHE_BYTES // per_entry_bytes)
        Xt_cache: "OrderedDict[Tuple[int, str], np.ndarray]" = OrderedDict()

        def kernel_apply_exact(base_idx: int, chain: OperatorChain, v: np.ndarray) -> np.ndarray:
            key = (base_idx, chain.signature)
            Xt = Xt_cache.get(key)
            if Xt is None:
                Xt = _chain_transform_train(self.lowrank_bases[base_idx], chain)
                Xt_cache[key] = Xt
                if len(Xt_cache) > max_entries:
                    Xt_cache.popitem(last=False)  # LRU eviction
            else:
                # Move to end (mark as recently used).
                Xt_cache.move_to_end(key)
            return Xt @ (Xt.T @ v)

        T_cols: List[np.ndarray] = []
        components: List[HardAOMComponent] = []
        residual_y = y_centred.copy()
        proj_coefs: List[np.ndarray] = []  # per-component projection coefficients
        best_loss = float("inf")
        patience_counter = 0
        Tmat_current: Optional[np.ndarray] = None

        for h in range(self.n_components):
            best_score = -np.inf
            best: Optional[Tuple[int, OperatorChain, np.ndarray, np.ndarray, int, float]] = None
            # (cand_idx, chain, u_raw, u_orth, base_idx, norm)
            for cand_idx, (base_idx, chain) in enumerate(self.candidates):
                u_raw = kernel_apply_exact(base_idx, chain, residual_y)
                if Tmat_current is not None and Tmat_current.shape[1] > 0:
                    Gram = Tmat_current.T @ Tmat_current
                    try:
                        coef_proj = np.linalg.solve(Gram, Tmat_current.T @ u_raw)
                    except np.linalg.LinAlgError:
                        coef_proj = np.linalg.lstsq(Gram, Tmat_current.T @ u_raw, rcond=None)[0]
                    u_orth = u_raw - Tmat_current @ coef_proj
                else:
                    u_orth = u_raw
                    coef_proj = np.zeros(0)
                norm = float(np.linalg.norm(u_orth))
                if norm < self.eps:
                    continue
                inner = float(np.dot(u_orth, y_centred))
                score = inner * inner / max(norm * norm, self.eps)
                if score > best_score:
                    best_score = score
                    best = (cand_idx, chain, u_raw, u_orth, base_idx, norm)
            if best is None:
                break
            cand_idx, chain, u_raw, u_orth, base_idx, norm = best
            # Recompute and *store* the projection coefficient against the latest
            # ``T_train`` so prediction can reproduce orthogonalisation exactly.
            if Tmat_current is not None and Tmat_current.shape[1] > 0:
                Gram = Tmat_current.T @ Tmat_current
                try:
                    coef_proj = np.linalg.solve(Gram, Tmat_current.T @ u_raw)
                except np.linalg.LinAlgError:
                    coef_proj = np.linalg.lstsq(Gram, Tmat_current.T @ u_raw, rcond=None)[0]
            else:
                coef_proj = np.zeros(0)
            t_h = u_orth / norm
            T_cols.append(t_h)
            proj_coefs.append(coef_proj)
            components.append(
                HardAOMComponent(
                    base_index=base_idx,
                    base_name=self.bases[base_idx].signature,
                    chain=chain,
                    score=float(best_score),
                    norm=float(norm),
                )
            )
            Tmat_current = np.column_stack(T_cols)
            # Update residual via least-squares projection
            Gram = Tmat_current.T @ Tmat_current
            try:
                coef_y = np.linalg.solve(Gram, Tmat_current.T @ y_centred)
            except np.linalg.LinAlgError:
                coef_y = np.linalg.lstsq(Gram, Tmat_current.T @ y_centred, rcond=None)[0]
            residual_y = y_centred - Tmat_current @ coef_y
            loss = float(np.mean(residual_y ** 2))
            if self.early_stop_patience is not None:
                if loss < best_loss - 1e-12:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop_patience:
                        break

        if not T_cols:
            raise RuntimeError("No latent components could be extracted")
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
        self.proj_coefs_ = proj_coefs
        # Cache training-side norms / scores so predict reproduces them deterministically.
        self.train_norms_ = [comp.norm for comp in components]
        self.y_mean_ = float(np.asarray(y_train).mean()) if y_train is not None else 0.0
        self._Xt_train_cache_ = Xt_cache
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "components_") or not self.components_:
            raise RuntimeError("HardAOMChainPLSRidge.predict called before fit")
        X = np.asarray(X, dtype=float)
        n_test = X.shape[0]
        H = len(self.components_)
        T_test = np.zeros((n_test, H))
        y_centred = self.lowrank_bases[0].y_centred
        residual_y_train = y_centred.copy()
        T_train_cols: List[np.ndarray] = []

        cache: Dict[int, np.ndarray] = {}
        for h, comp in enumerate(self.components_):
            lr = self.lowrank_bases[comp.base_index]
            key = (comp.base_index, comp.chain.signature)
            Xt_train = self._Xt_train_cache_.get(key)
            if Xt_train is None:
                Xt_train = _chain_transform_train(lr, comp.chain)
                self._Xt_train_cache_[key] = Xt_train
            Xt_test = _chain_transform_test(self.bases, self.lowrank_bases, X, comp.base_index, comp.chain, cache)
            # u_train_raw = X_train_s @ X_train_s.T @ residual_y_train
            w_s = Xt_train.T @ residual_y_train
            u_train_raw = Xt_train @ w_s
            u_test_raw = Xt_test @ w_s
            proj_coef = self.proj_coefs_[h]
            if T_train_cols:
                Tmat_train = np.column_stack(T_train_cols)
                u_train = u_train_raw - Tmat_train @ proj_coef
                u_test = u_test_raw - T_test[:, :h] @ proj_coef
            else:
                u_train = u_train_raw
                u_test = u_test_raw
            norm = self.train_norms_[h]
            if norm < self.eps:
                t_train = np.zeros_like(u_train)
                T_test[:, h] = 0.0
            else:
                t_train = u_train / norm
                T_test[:, h] = u_test / norm
            T_train_cols.append(t_train)
            # Update residual exactly as in fit
            Tmat = np.column_stack(T_train_cols)
            Gram = Tmat.T @ Tmat
            try:
                coef_y = np.linalg.solve(Gram, Tmat.T @ y_centred)
            except np.linalg.LinAlgError:
                coef_y = np.linalg.lstsq(Gram, Tmat.T @ y_centred, rcond=None)[0]
            residual_y_train = y_centred - Tmat @ coef_y

        yhat_centred = T_test @ self.coef_latent_
        return yhat_centred + self.y_mean_
