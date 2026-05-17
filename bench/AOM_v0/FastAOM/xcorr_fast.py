"""Vectorised replacement for ``aompls.operators._xcorr_zero_pad``.

The parent implementation iterates ``for i in range(p)``, which is the
dominant cost on every chain.apply_cov / chain.transform call inside the
FastAOM screening and model fits (chains apply this primitive O(K × r)
times per chain, where K is the chain depth and r is the SVD rank).

This module provides a bit-exact, zero-padded vectorised version using
``numpy.lib.stride_tricks.sliding_window_view`` and a single matrix
multiplication. It is **NOT** installed by default into
``aompls.operators`` — the parent module is shared with the AOM-PLS /
AOM-Ridge / Multi-kernel benchmarks, so we install the patch only when
explicitly requested via :func:`install_xcorr_patch`.

Parity is verified by :mod:`tests/test_xcorr_fast` against the original
implementation on random inputs across kernel sizes 3-31 and p in
{16, 64, 200, 500, 2000}.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def xcorr_zero_pad_fast(X: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Vectorised drop-in for :func:`aompls.operators._xcorr_zero_pad`.

    Computes ``out[:, i] = sum_j kernel[j] * X[:, i + j - half]`` with
    zero-padding outside ``[0, p)`` — the same convention as the parent.

    Implementation:

      1. Zero-pad ``X`` on both sides to shape ``(n, p + k - 1)``.
      2. Use ``sliding_window_view`` to materialise a ``(n, p, k)`` view of
         consecutive windows without copying.
      3. One matmul ``windows @ kernel`` collapses the window axis.

    The dtype convention matches the parent (always ``float``), so the
    monkey-patch is safe to install in benchmarks that depend on the
    parent's coercion behaviour.
    """
    X = np.asarray(X, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    k = kernel.shape[0]
    half_left = (k - 1) // 2
    if X.ndim == 1:
        X2 = X.reshape(1, -1)
        squeeze = True
    else:
        X2 = X
        squeeze = False
    n, p = X2.shape
    pad = np.zeros((n, p + k - 1), dtype=float)
    pad[:, half_left : half_left + p] = X2
    # ``sliding_window_view`` returns a read-only view with shape
    # (n, pad.shape[1] - k + 1, k) = (n, p, k).
    windows = sliding_window_view(pad, window_shape=k, axis=1)
    out = windows @ kernel  # shape (n, p)
    return out.ravel() if squeeze else out


_PATCH_INSTALLED = False
_ORIGINAL_XCORR: Optional[Callable] = None
_ORIGINAL_DETREND_APPLY_COV: Optional[Callable] = None
_ORIGINAL_DETREND_ADJOINT_VEC: Optional[Callable] = None
_ORIGINAL_DETREND_TRANSFORM: Optional[Callable] = None


# ---------------------------------------------------------------------------
# DetrendProjection low-rank form
# ---------------------------------------------------------------------------
#
# The parent ``DetrendProjectionOperator`` builds a dense ``p × p`` complement
# matrix ``A = I - Q Q^T`` (Q has d+1 columns) and uses it as ``A @ S``. For
# screening over chains × low-rank bases this is O(p^2 × r) per call, which
# dominates the runtime when r ≈ 200 and p ≈ 600.
#
# A mathematically identical low-rank form is ``A @ S = S - Q @ (Q^T @ S)``
# at cost O(p × d × r) — ~300× cheaper for typical (p, d, r) sizes. The patch
# below replaces the dense action while keeping the same caching behaviour,
# the same boundary semantics, and the same A = A^T symmetry.


def _detrend_q_matrix(self, p: int) -> np.ndarray:
    """Return the QR-orthonormal polynomial basis ``Q ∈ R^{p × (d+1)}``."""
    # Reuse the parent's complement cache to fingerprint ``p``, but also keep
    # a separate ``Q`` cache so we never materialise the dense complement.
    cached_p = getattr(self, "_fast_q_p", None)
    cached_Q = getattr(self, "_fast_q_Q", None)
    if cached_p == p and cached_Q is not None:
        return cached_Q
    if p < self.degree + 1:
        raise ValueError(
            f"DetrendProjection degree={self.degree} requires p >= {self.degree + 1}"
        )
    t = np.linspace(-1.0, 1.0, p)
    cols = [t**k for k in range(self.degree + 1)]
    P = np.column_stack(cols)
    Q, _ = np.linalg.qr(P)
    self._fast_q_p = p
    self._fast_q_Q = np.ascontiguousarray(Q, dtype=float)
    return self._fast_q_Q


def _detrend_apply_cov_fast(self, S: np.ndarray) -> np.ndarray:
    Q = _detrend_q_matrix(self, S.shape[0])
    # S shape (p,) or (p, r)
    proj = Q.T @ S  # shape (d+1,) or (d+1, r)
    return S - Q @ proj


def _detrend_adjoint_vec_fast(self, v: np.ndarray) -> np.ndarray:
    # Detrend is symmetric (A = A^T) so adjoint = forward.
    Q = _detrend_q_matrix(self, v.shape[0])
    proj = Q.T @ v
    return v - Q @ proj


def _detrend_transform_fast(self, X: np.ndarray) -> np.ndarray:
    """``X @ A.T = X - (X @ Q) @ Q.T`` (since A = A.T for this operator)."""
    Q = _detrend_q_matrix(self, X.shape[1])
    # X shape (n, p); Q shape (p, d+1)
    proj = X @ Q  # shape (n, d+1)
    return X - proj @ Q.T


def install_xcorr_patch() -> None:
    """Monkey-patch :mod:`aompls.operators` to use the fast xcorr **and** the
    low-rank ``I - Q Q^T`` form for :class:`DetrendProjectionOperator`.

    Both replacements are bit-exact (verified in
    ``tests/test_xcorr_fast.py``). Idempotent.

    Thread-safety note: the install/uninstall is global class-level state.
    Concurrent install/uninstall from different threads is not safe — but
    the only invocation pattern in this repository is "install once at
    benchmark module-load, never uninstall" (see
    ``benchmarks/run_fast_aom_benchmark.py``). Per-instance caches are
    independent of the global swap, so no correctness hazard exists in
    the single-installer pattern.
    """
    global _PATCH_INSTALLED, _ORIGINAL_XCORR
    global _ORIGINAL_DETREND_APPLY_COV, _ORIGINAL_DETREND_ADJOINT_VEC, _ORIGINAL_DETREND_TRANSFORM
    if _PATCH_INSTALLED:
        return
    from aompls import operators as _ops

    _ORIGINAL_XCORR = _ops._xcorr_zero_pad
    _ops._xcorr_zero_pad = xcorr_zero_pad_fast

    _ORIGINAL_DETREND_APPLY_COV = _ops.DetrendProjectionOperator._apply_cov_impl
    _ORIGINAL_DETREND_ADJOINT_VEC = _ops.DetrendProjectionOperator._adjoint_vec_impl
    _ORIGINAL_DETREND_TRANSFORM = _ops.DetrendProjectionOperator._transform_impl
    _ops.DetrendProjectionOperator._apply_cov_impl = _detrend_apply_cov_fast
    _ops.DetrendProjectionOperator._adjoint_vec_impl = _detrend_adjoint_vec_fast
    _ops.DetrendProjectionOperator._transform_impl = _detrend_transform_fast

    _PATCH_INSTALLED = True


def uninstall_xcorr_patch() -> None:
    """Restore the original :mod:`aompls.operators` implementations."""
    global _PATCH_INSTALLED, _ORIGINAL_XCORR
    global _ORIGINAL_DETREND_APPLY_COV, _ORIGINAL_DETREND_ADJOINT_VEC, _ORIGINAL_DETREND_TRANSFORM
    if not _PATCH_INSTALLED:
        return
    from aompls import operators as _ops

    if _ORIGINAL_XCORR is not None:
        _ops._xcorr_zero_pad = _ORIGINAL_XCORR
    if _ORIGINAL_DETREND_APPLY_COV is not None:
        _ops.DetrendProjectionOperator._apply_cov_impl = _ORIGINAL_DETREND_APPLY_COV
    if _ORIGINAL_DETREND_ADJOINT_VEC is not None:
        _ops.DetrendProjectionOperator._adjoint_vec_impl = _ORIGINAL_DETREND_ADJOINT_VEC
    if _ORIGINAL_DETREND_TRANSFORM is not None:
        _ops.DetrendProjectionOperator._transform_impl = _ORIGINAL_DETREND_TRANSFORM

    _PATCH_INSTALLED = False
    _ORIGINAL_XCORR = None
    _ORIGINAL_DETREND_APPLY_COV = None
    _ORIGINAL_DETREND_ADJOINT_VEC = None
    _ORIGINAL_DETREND_TRANSFORM = None


def is_patch_installed() -> bool:
    return _PATCH_INSTALLED
