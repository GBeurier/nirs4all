"""Sparse Multi-Kernel Ridge.

A multi-kernel Ridge with sparse non-negative kernel weights:

    K_theta = sum_s theta_s K_s,  theta_s >= 0
    alpha   = (K_theta + lambda I)^{-1} y_centred
    yhat    = K_theta alpha + y_mean

Selection of the chains is greedy:

  1. Start with no chain.
  2. At each step pick the chain whose kernel best aligns with the
     current residual ``y_res``:
        score(s) = (y_res^T K_s y_res) / sqrt(trace(K_s K_s) + eps)
  3. Add it to the active set and refit ``theta`` (non-negative
     constrained Ridge on residual) and ``lambda`` (GCV on the
     aggregated kernel).
  4. Stop when ``max_chains`` is reached or the gain falls below
     ``tol``.

The non-negative refit is performed by projected gradient descent.

All kernels are computed in *exact* form
``K_s = X_centred A^T A X_centred^T`` to keep training and prediction
consistent (the SVD low-rank machinery is reserved for the upstream
screening stage that picks the candidate pool).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from nirs4all.operators.models._aom_nirs.fast.bases import BaseTransform
from nirs4all.operators.models._aom_nirs.fast.lowrank import LowRankBase
from nirs4all.operators.models._aom_nirs.fast.operator_chain import OperatorChain

from ._common import ridge_gcv_lambda


@dataclass
class SparseKernelComponent:
    base_index: int
    base_name: str
    chain: OperatorChain
    theta: float
    score: float


def _projected_gradient_nnls_kernels(
    grams: np.ndarray,
    Ky: np.ndarray,
    init: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    """``min_{theta >= 0} 0.5 theta^T G theta - Ky^T theta`` via projected gradient."""
    if grams.size == 0:
        return np.array([])
    eigvals = np.linalg.eigvalsh(grams)
    L = float(max(np.max(eigvals), 1e-6))
    theta = np.clip(np.asarray(init, dtype=float), 0.0, None)
    for _ in range(max_iter):
        grad = grams @ theta - Ky
        theta_new = np.clip(theta - grad / L, 0.0, None)
        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new
    return theta


class SparseMultiKernelRidge(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        bases: Sequence[BaseTransform],
        lowrank_bases: Sequence[LowRankBase],
        candidates: Sequence[Tuple[int, OperatorChain]],
        max_chains: int = 8,
        lambdas: Optional[Sequence[float]] = None,
        tol: float = 1e-6,
        nnls_iters: int = 200,
    ) -> None:
        if len(bases) != len(lowrank_bases):
            raise ValueError("bases and lowrank_bases must have the same length")
        self.bases = list(bases)
        self.lowrank_bases = list(lowrank_bases)
        self.candidates = list(candidates)
        self.max_chains = int(max_chains)
        if lambdas is None:
            lambdas = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
        self.lambdas = tuple(float(l) for l in lambdas)
        self.tol = float(tol)
        self.nnls_iters = int(nnls_iters)

    def _exact_kernel_apply(
        self,
        base_idx: int,
        chain: OperatorChain,
        v: np.ndarray,
        train_cache: Dict[Tuple[int, str], np.ndarray],
    ) -> np.ndarray:
        key = (base_idx, chain.signature)
        Xt = train_cache.get(key)
        if Xt is None:
            Xt = chain.transform(self.lowrank_bases[base_idx].X_centred)
            train_cache[key] = Xt
        return Xt @ (Xt.T @ v)

    def fit(self, X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None) -> "SparseMultiKernelRidge":
        y_centred = self.lowrank_bases[0].y_centred
        n = y_centred.shape[0]
        if not self.candidates:
            raise ValueError("No candidates provided")
        Xt_train_cache: Dict[Tuple[int, str], np.ndarray] = {}
        # Precompute K_s y_centred for every candidate (exact)
        Ky_per_cand = [
            self._exact_kernel_apply(b, c, y_centred, Xt_train_cache) for b, c in self.candidates
        ]
        y_norm_sq = float(np.dot(y_centred, y_centred)) + 1e-12

        selected: List[int] = []
        theta_active = np.zeros(0)
        # Chains whose NNLS-refit set theta=0 are permanently blacklisted to
        # avoid an infinite greedy loop: without this, a chain that gets
        # selected by the residual-alignment score but ends up with theta=0
        # in NNLS would be re-selected next iteration (residual unchanged),
        # loop forever.
        blacklisted: set = set()

        while len(selected) < self.max_chains:
            if selected:
                K_active_y = np.zeros(n)
                for active_i, sel in enumerate(selected):
                    K_active_y = K_active_y + theta_active[active_i] * Ky_per_cand[sel]
                residual = y_centred - K_active_y
            else:
                residual = y_centred.copy()
            best_score = -np.inf
            best_idx = -1
            for cand_idx in range(len(self.candidates)):
                if cand_idx in selected or cand_idx in blacklisted:
                    continue
                base_idx, chain = self.candidates[cand_idx]
                Kr = self._exact_kernel_apply(base_idx, chain, residual, Xt_train_cache)
                Kr_norm = float(np.dot(Kr, Kr))
                if Kr_norm < 1e-12:
                    continue
                score = float(np.dot(residual, Kr)) / np.sqrt(Kr_norm + 1e-12)
                if score > best_score:
                    best_score = score
                    best_idx = cand_idx
            if best_idx < 0 or best_score < self.tol:
                break
            selected.append(best_idx)
            Ky_active = [Ky_per_cand[i] for i in selected]
            stacked = np.column_stack(Ky_active)
            grams = stacked.T @ stacked
            Ky_inner = stacked.T @ y_centred
            init = np.zeros(len(selected))
            if theta_active.shape[0] == len(selected) - 1:
                init[: len(selected) - 1] = theta_active
            theta_active = _projected_gradient_nnls_kernels(grams, Ky_inner, init, max_iter=self.nnls_iters)
            keep_mask = theta_active > 1e-12
            if not np.all(keep_mask):
                # Blacklist any chain that got dropped, including the newest one
                # if it was the one set to zero (otherwise we'd reselect it
                # next iteration and loop forever).
                kept = []
                kept_theta = []
                for s, k, t in zip(selected, keep_mask, theta_active):
                    if k:
                        kept.append(s)
                        kept_theta.append(t)
                    else:
                        blacklisted.add(s)
                selected = kept
                theta_active = np.asarray(kept_theta, dtype=float) if kept_theta else np.zeros(0)

        if not selected:
            raise RuntimeError("SparseMultiKernelRidge selected no chains")

        # Memory guard: the kernel matrix K_theta is dense ``n × n``. For
        # very large ``n`` (typical NIRS cohorts cap around 40k samples) this
        # would allocate ~12 GB at float64. We refuse to build it and raise so
        # the benchmark runner records a clear ``status=skipped`` row rather
        # than crashing the entire process via OOM.
        if n > 8000:
            raise RuntimeError(
                f"SparseMultiKernelRidge: n={n} exceeds dense-kernel guard (8000); "
                "skip this variant on very large datasets"
            )

        # Build K_theta exactly
        K_theta = np.zeros((n, n))
        for theta_s, idx in zip(theta_active, selected):
            base_idx, chain = self.candidates[idx]
            Xt = Xt_train_cache[(base_idx, chain.signature)]
            K_s = Xt @ Xt.T
            K_theta = K_theta + theta_s * K_s
        K_theta = 0.5 * (K_theta + K_theta.T)
        lam, gcv = ridge_gcv_lambda(K_theta, y_centred, self.lambdas)
        alpha = np.linalg.solve(K_theta + lam * np.eye(n), y_centred)

        self.selected_candidates_ = [self.candidates[i] for i in selected]
        self.theta_ = theta_active.copy()
        self.lambda_ = float(lam)
        self.alpha_ = alpha
        self.K_theta_train_ = K_theta
        self._Xt_train_cache_ = Xt_train_cache
        self.y_mean_ = float(np.asarray(y_train).mean()) if y_train is not None else 0.0
        self.components_ = [
            SparseKernelComponent(
                base_index=self.candidates[i][0],
                base_name=self.bases[self.candidates[i][0]].signature,
                chain=self.candidates[i][1],
                theta=float(t),
                score=0.0,
            )
            for i, t in zip(selected, theta_active)
        ]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "selected_candidates_"):
            raise RuntimeError("SparseMultiKernelRidge.predict called before fit")
        X = np.asarray(X, dtype=float)
        n_test = X.shape[0]
        n_train = self.alpha_.shape[0]
        K_cross = np.zeros((n_test, n_train))
        base_test_cache: Dict[int, np.ndarray] = {}
        for theta_s, (base_idx, chain) in zip(self.theta_, self.selected_candidates_):
            if base_idx not in base_test_cache:
                Xb_test = self.bases[base_idx].transform(X)
                base_test_cache[base_idx] = Xb_test - self.lowrank_bases[base_idx].mean
            Xc_test = base_test_cache[base_idx]
            Xt_test = chain.transform(Xc_test)
            Xt_train = self._Xt_train_cache_[(base_idx, chain.signature)]
            K_cross += theta_s * (Xt_test @ Xt_train.T)
        return K_cross @ self.alpha_ + self.y_mean_
