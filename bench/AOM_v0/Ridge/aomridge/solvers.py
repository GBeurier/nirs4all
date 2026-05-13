"""Dual / kernel Ridge solvers for AOM-Ridge.

Given a symmetric PSD train kernel ``K`` and a target ``Y``, the dual Ridge
problem is

```text
C = (K + alpha I)^-1 Y
```

with ``alpha > 0``. Two solver paths are available:

- ``cholesky`` — factor ``K + alpha I`` once with Cholesky and back-substitute
  one alpha at a time. Used by default.
- ``eigh`` — eigendecompose ``K = V diag(lam) V^T`` once and reuse it across
  the entire alpha grid; this is much faster when several alphas are tested.

Both paths produce the same result (modulo floating point). Tiny negative
eigenvalues from numerical error are clipped to zero as a documented safeguard.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# ----------------------------------------------------------------------
# Alpha grid
# ----------------------------------------------------------------------


def make_alpha_grid(
    K: np.ndarray,
    n_grid: int = 50,
    low: float = -6.0,
    high: float = 6.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build a trace-relative alpha grid.

    The grid is ``base * 10**linspace(low, high, n_grid)`` with
    ``base = max(trace(K) / n, eps)``.
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square 2D matrix")
    if n_grid < 1:
        raise ValueError("n_grid must be >= 1")
    n = K.shape[0]
    base = max(float(np.trace(K)) / max(n, 1), eps)
    return base * np.logspace(low, high, n_grid)


def alpha_at_boundary(
    rmse_per_alpha: np.ndarray, edge_tolerance: int = 2
) -> bool:
    """Return True if the CV optimum sits at (or within `edge_tolerance` of) a grid edge.

    This is used to detect under-/over-regularised optima where the grid
    must be expanded.
    """
    if rmse_per_alpha.ndim != 1 or rmse_per_alpha.size == 0:
        raise ValueError("rmse_per_alpha must be a non-empty 1D array")
    idx = int(np.argmin(rmse_per_alpha))
    n = rmse_per_alpha.size
    return idx <= edge_tolerance or idx >= n - 1 - edge_tolerance


# ----------------------------------------------------------------------
# Symmetrize / jitter helper
# ----------------------------------------------------------------------


def _symmetrize(K: np.ndarray) -> np.ndarray:
    return 0.5 * (K + K.T)


def _cholesky_solve_with_jitter(
    Ka: np.ndarray, Y: np.ndarray, base_jitter: float
) -> np.ndarray:
    """Cholesky-solve ``(Ka + jitter I) C = Y`` with retries on failure."""
    jitter = 0.0
    last_error: Exception | None = None
    Ka_sym = _symmetrize(Ka)
    for _ in range(6):
        if jitter > 0:
            mat = Ka_sym + jitter * np.eye(Ka_sym.shape[0])
        else:
            mat = Ka_sym
        try:
            c, low = cho_factor(mat, lower=True, check_finite=False)
            return cho_solve((c, low), Y, check_finite=False)
        except np.linalg.LinAlgError as exc:
            last_error = exc
            jitter = max(base_jitter, 10.0 * jitter)
    # Eigen fallback — safe for any symmetric input
    if last_error is None:
        last_error = np.linalg.LinAlgError("Cholesky failure with no recorded error")
    return _eigh_solve(Ka_sym, Y, alpha=0.0)


def _eigh_solve(K: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve ``(K + alpha I) C = Y`` via the symmetric eigendecomposition.

    Negative eigenvalues are clipped to zero. AOM-Ridge always feeds in
    ``K = Phi Phi^T`` (PSD by construction, then symmetrized), so this only
    suppresses tiny floating-point negatives. If a non-PSD ``K`` is passed
    in, the eigh path silently solves on the PSD part of ``K``; callers that
    need exact ``(K + alpha I)^-1 Y`` for indefinite ``K`` must use
    ``method="cholesky"`` (which will fall back to a jittered solve) or
    handle indefiniteness themselves.
    """
    K_sym = _symmetrize(K)
    lam, V = np.linalg.eigh(K_sym)
    lam = np.clip(lam, 0.0, None)
    rhs = V.T @ Y
    inv = 1.0 / (lam + alpha)
    C = V @ (rhs * inv[:, None])
    return C


# ----------------------------------------------------------------------
# Public solver
# ----------------------------------------------------------------------


def solve_dual_ridge(
    K: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    method: str = "auto",
    base_jitter: float = 1e-10,
) -> np.ndarray:
    """Return ``C = (K + alpha I)^-1 Y``.

    ``Y`` may be 1D or 2D; the returned ``C`` matches its shape. ``alpha`` must
    be strictly positive.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square 2D matrix")
    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim == 1:
        Y2 = Y_arr.reshape(-1, 1)
    else:
        Y2 = Y_arr
    method = method.lower()
    if method not in ("auto", "cholesky", "eigh"):
        raise ValueError("method must be 'auto', 'cholesky', or 'eigh'")
    Ka = K + alpha * np.eye(K.shape[0])
    if method == "eigh":
        C = _eigh_solve(K, Y2, alpha=alpha)
    else:
        C = _cholesky_solve_with_jitter(Ka, Y2, base_jitter=base_jitter)
    if Y_arr.ndim == 1:
        return C.ravel()
    return C


def solve_dual_ridge_path_eigh(
    K: np.ndarray,
    Y: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """Solve the dual Ridge for every alpha in ``alphas`` via one eigendecomposition.

    Returns an array with shape ``(len(alphas), n, q)`` if ``Y`` is 2D, or
    ``(len(alphas), n)`` if ``Y`` is 1D.
    """
    if alphas.ndim != 1:
        raise ValueError("alphas must be 1D")
    if np.any(alphas <= 0.0):
        raise ValueError("all alphas must be positive")
    K_sym = _symmetrize(K)
    lam, V = np.linalg.eigh(K_sym)
    lam = np.clip(lam, 0.0, None)
    Y_arr = np.asarray(Y, dtype=float)
    Y2 = Y_arr.reshape(-1, 1) if Y_arr.ndim == 1 else Y_arr
    rhs = V.T @ Y2
    out = np.empty((alphas.size, *Y2.shape), dtype=float)
    for i, a in enumerate(alphas):
        out[i] = V @ (rhs * (1.0 / (lam + a))[:, None])
    if Y_arr.ndim == 1:
        return out.reshape(alphas.size, Y2.shape[0])
    return out


def predict_dual(K_cross: np.ndarray, dual: np.ndarray) -> np.ndarray:
    """Predict from ``K_cross`` (shape ``n_pred x n_train``) and dual coefs."""
    return K_cross @ dual
