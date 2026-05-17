"""Truncated-SVD bases for fast candidate evaluation.

Each nonlinear base ``B(X) ∈ R^{n x p}`` is decomposed once as
``B(X) ≈ U diag(S) V^T`` with rank ``r`` (typically 50 - 300). The
decomposition then lets every linear chain ``A_s`` evaluate

  * the Frobenius norm ``||B(X) A_s^T||_F^2 ≈ ||A_s V diag(S)||_F^2``
    used by the screening denominator;
  * the kernel ``K_s = B(X) A_s^T A_s B(X)^T ≈ U C_s U^T`` where
    ``C_s = F_s^T F_s`` and ``F_s = A_s V diag(S) ∈ R^{p x r}``;
  * the kernel-vector product ``K_s v ≈ U (C_s (U^T v))`` in
    ``O(n r + r^2)`` per chain instead of ``O(n^2)``.

The decomposition is computed from the *centred* training data (``B(X) -
mean``) because PLS and Ridge are translation-invariant once both ``X``
and ``y`` are centred — that keeps the screening and low-rank kernels
aligned with the downstream solvers.

The :class:`LowRankBase` is intentionally a frozen container so that the
SVD is computed once per fold and reused across all chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .bases import BaseTransform
from .operator_chain import OperatorChain


@dataclass
class LowRankBase:
    """Truncated SVD of a nonlinear base applied to a training fold.

    Attributes:
        base: The (fitted) :class:`BaseTransform` instance.
        X_centred: The centred ``B(X) - mean(B(X))`` training matrix.
        mean: Per-feature mean vector (``shape = (p,)``).
        U: Left singular vectors, ``shape = (n, r)``.
        S: Singular values, ``shape = (r,)``.
        Vt: Right singular vectors transposed, ``shape = (r, p)``.
        rank: Effective rank ``r`` after truncation.
        y_centred: Centred response (matches ``X_centred``).
        g0: Cross-covariance vector ``B(X)^T y`` (centred), pre-computed
            for fast screening.
    """

    base: BaseTransform
    X_centred: np.ndarray
    mean: np.ndarray
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    rank: int
    y_centred: np.ndarray
    g0: np.ndarray

    @property
    def V(self) -> np.ndarray:
        return self.Vt.T

    def F(self, chain: OperatorChain) -> np.ndarray:
        """Compute ``F_s = A_s V diag(S)`` for a chain.

        ``A_s`` is the chain matrix; using the parent convention
        ``transform(X) = X @ A.T``, the apply on a column is
        ``apply_cov(V_col) = A @ V_col``. We apply ``apply_cov`` to
        ``V * S[None, :]`` to get the desired ``A V diag(S)``.
        """
        VS = self.V * self.S[None, :]
        return chain.apply_cov(VS)  # shape (p, r)

    def chain_norm_F_sq(self, chain: OperatorChain) -> float:
        """``||B(X) A_s^T||_F^2 ≈ ||F_s||_F^2``."""
        F_s = self.F(chain)
        return float(np.sum(F_s * F_s))

    def kernel_matrix_lowrank(self, chain: OperatorChain) -> np.ndarray:
        """``K_s ≈ U F_s^T F_s U^T``. Computes the dense ``n x n`` kernel."""
        F_s = self.F(chain)  # (p, r)
        C_s = F_s.T @ F_s    # (r, r)
        return self.U @ C_s @ self.U.T

    def kernel_apply(self, chain: OperatorChain, v: np.ndarray) -> np.ndarray:
        """``K_s @ v ≈ U @ C_s @ (U^T @ v)`` for ``v`` of shape ``(n,)`` or ``(n, k)``."""
        F_s = self.F(chain)
        C_s = F_s.T @ F_s
        z = self.U.T @ v
        return self.U @ (C_s @ z)


def fit_lowrank_bases(
    bases: Sequence[BaseTransform],
    X: np.ndarray,
    y: np.ndarray,
    rank: int = 200,
    random_state: int = 0,
) -> List[LowRankBase]:
    """Fit each base on ``X`` (training fold) and compute its truncated SVD.

    Args:
        bases: Sequence of fold-aware base transforms.
        X: Training spectra.
        y: Training response (1D or 2D).
        rank: Truncation rank ``r``. Reduced to ``min(rank, n, p)``.
        random_state: Currently unused (deterministic SVD); reserved for
            future randomised SVD variants.

    Returns:
        List of :class:`LowRankBase`, one per input base, fitted on ``X``.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr.ravel()
    if y_arr.ndim != 1:
        raise ValueError("y must be 1D (PLS1 / Ridge) for FastAOM low-rank evaluation")
    out: List[LowRankBase] = []
    for base in bases:
        Xb = base.fit_transform(X, y=y_arr)
        Xb = np.asarray(Xb, dtype=float)
        mean = Xb.mean(axis=0)
        Xc = Xb - mean
        y_mean = float(y_arr.mean())
        yc = y_arr - y_mean
        n, p = Xc.shape
        eff_rank = int(min(rank, n, p))
        # Full SVD for small problems; thin SVD for larger ones via numpy's lapack.
        U_full, S_full, Vt_full = np.linalg.svd(Xc, full_matrices=False)
        U = np.ascontiguousarray(U_full[:, :eff_rank])
        S = np.ascontiguousarray(S_full[:eff_rank])
        Vt = np.ascontiguousarray(Vt_full[:eff_rank, :])
        g0 = Xc.T @ yc
        out.append(
            LowRankBase(
                base=base,
                X_centred=Xc,
                mean=mean,
                U=U,
                S=S,
                Vt=Vt,
                rank=eff_rank,
                y_centred=yc,
                g0=g0,
            )
        )
    return out
