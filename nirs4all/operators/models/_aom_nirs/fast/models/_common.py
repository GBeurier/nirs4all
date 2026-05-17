"""Shared utilities for the FastAOM model family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from nirs4all.operators.models._aom_nirs.fast.bases import BaseTransform
from nirs4all.operators.models._aom_nirs.fast.operator_chain import OperatorChain


@dataclass(frozen=True)
class CandidateView:
    """Wrap a candidate so models share a uniform interface.

    A view bundles a *fitted* base transform (already applied to the
    training fold) together with the linear chain. It exposes the
    minimal API that the four models need: transform, kernel-vector
    product, kernel matrix, and chain norm.
    """

    base: BaseTransform
    chain: OperatorChain
    base_index: int
    score: float = 0.0

    @property
    def signature(self) -> str:
        return f"{self.base.signature}|{self.chain.signature}"


def center_y(y: np.ndarray) -> Tuple[np.ndarray, float]:
    yc = np.asarray(y, dtype=float).ravel()
    mean = float(yc.mean())
    return yc - mean, mean


def ridge_gcv_lambda(
    K: np.ndarray,
    y: np.ndarray,
    lambdas: Sequence[float],
) -> Tuple[float, float]:
    """Pick the ``lambda`` minimising the leave-one-out GCV on a kernel matrix.

    Uses the closed-form Hutter-style identity
    ``yhat = K (K + lam I)^{-1} y`` and the standard leave-one-out
    diagonal trick. Returns ``(lambda*, gcv_loss)``.
    """
    n = K.shape[0]
    eigvals, eigvecs = np.linalg.eigh(K)
    # eigvals can be negative due to numerical drift on a PSD kernel; clip.
    eigvals = np.clip(eigvals, 0.0, None)
    z = eigvecs.T @ y
    best = (None, np.inf)
    for lam in lambdas:
        denom = eigvals + lam
        # Hat matrix in eigenbasis: diag(eigvals / denom)
        hat_diag = (eigvals / denom)
        yhat = eigvecs @ (hat_diag * z)
        residual = y - yhat
        trace_term = (1.0 - hat_diag.sum() / n)
        if trace_term <= 1e-12:
            continue
        gcv = float(np.mean(residual ** 2) / (trace_term ** 2))
        if gcv < best[1]:
            best = (float(lam), gcv)
    if best[0] is None:
        return float(lambdas[0]), float("inf")
    return best


def extract_pls_scores(
    X_centred: np.ndarray,
    y_centred: np.ndarray,
    n_components: int,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run NIPALS (PLS1) on ``(X_centred, y_centred)`` and return scores/weights/loadings.

    Returns:
        T: latent scores ``(n, H)``
        W: weights ``(p, H)``
        P: loadings ``(p, H)``

    The implementation is a minimal in-house PLS1 NIPALS used by
    :class:`SingleChainPLSRidge`. It avoids importing the heavier
    ``aompls.nipals`` machinery so the model has a clean dependency
    surface for the tests.
    """
    n, p = X_centred.shape
    H = min(n_components, p, max(1, n - 1))
    Xr = X_centred.copy()
    yr = y_centred.copy()
    T = np.zeros((n, H))
    W = np.zeros((p, H))
    P = np.zeros((p, H))
    for h in range(H):
        w = Xr.T @ yr
        norm = float(np.linalg.norm(w))
        if norm < tol:
            T = T[:, :h]
            W = W[:, :h]
            P = P[:, :h]
            break
        w /= norm
        t = Xr @ w
        t_norm_sq = float(np.dot(t, t))
        if t_norm_sq < tol:
            T = T[:, :h]
            W = W[:, :h]
            P = P[:, :h]
            break
        p_h = Xr.T @ t / t_norm_sq
        c_h = float(np.dot(yr, t) / t_norm_sq)
        Xr = Xr - np.outer(t, p_h)
        yr = yr - c_h * t
        T[:, h] = t
        W[:, h] = w
        P[:, h] = p_h
    return T, W, P


def ridge_on_scores(
    T: np.ndarray,
    y_centred: np.ndarray,
    lambdas: Sequence[float],
    component_shrinkage_gamma: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """Fit Ridge on latent scores: ``c = (T^T T + Lambda)^{-1} T^T y``.

    Args:
        T: Latent scores ``(n, H)``.
        y_centred: Centred response (``y - y_mean``).
        lambdas: Candidate ``lambda_0`` values; the best is picked via
            internal LOO-style minimisation of the residual.
        component_shrinkage_gamma: If not None, use
            ``lambda_h = lambda_0 * (h + 1) ** gamma`` (1-indexed) so the
            late components are more aggressively shrunk.

    Returns:
        (c, best_lambda, best_loss)
    """
    n, H = T.shape
    if H == 0:
        return np.zeros(0), float(lambdas[0]) if lambdas else 0.0, float("inf")
    TtT = T.T @ T
    Tty = T.T @ y_centred
    best = (None, None, np.inf)
    for lam in lambdas:
        if component_shrinkage_gamma is None:
            reg = lam * np.eye(H)
        else:
            diag = lam * np.power(np.arange(1, H + 1, dtype=float), component_shrinkage_gamma)
            reg = np.diag(diag)
        try:
            c = np.linalg.solve(TtT + reg, Tty)
        except np.linalg.LinAlgError:
            continue
        yhat = T @ c
        loss = float(np.mean((y_centred - yhat) ** 2))
        if loss < best[2]:
            best = (c, float(lam), loss)
    if best[0] is None:
        c = Tty / max(1e-12, TtT[0, 0])
        return c, float(lambdas[0]) if lambdas else 0.0, float("inf")
    return best


def cross_val_kfold_indices(n: int, n_splits: int, rng: np.random.Generator) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Shuffled K-fold index generator (no sklearn dependency)."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    idx = rng.permutation(n)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    folds = []
    start = 0
    for fs in fold_sizes:
        val = idx[start : start + fs]
        train = np.concatenate([idx[:start], idx[start + fs :]])
        folds.append((train, val))
        start += fs
    return folds
