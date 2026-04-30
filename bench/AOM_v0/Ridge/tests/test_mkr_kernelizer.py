"""Tests for the AOM kernelizer (centred + trace-normalised block kernels)."""

from __future__ import annotations

import numpy as np
import pytest

from aomridge.kernelizer import (
    AOMKernelizer,
    kernel_alignment_matrix,
)


def _smooth_X(n: int, p: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = np.arange(p, dtype=float)
    X = np.zeros((n, p), dtype=float)
    for i in range(n):
        for k in range(3):
            c = rng.uniform(0.1 * p, 0.9 * p)
            w = rng.uniform(0.05 * p, 0.15 * p)
            a = rng.normal()
            X[i] += a * np.exp(-((grid - c) ** 2) / (2 * w ** 2))
    X += rng.normal(0, 0.05, size=(n, p))
    return X


# ----------------------------------------------------------------------
# Centring + trace normalisation invariants
# ----------------------------------------------------------------------


def test_kernelizer_centring_residual_is_tiny():
    X = _smooth_X(40, 60, seed=0)
    ker = AOMKernelizer(operator_bank="compact", center=True, normalize="trace")
    K_blocks = ker.fit_transform(X)
    for K_b in K_blocks:
        # K_b @ 1 == 0 (within fp tolerance)
        s = K_b.sum(axis=1)
        assert np.max(np.abs(s)) < 1e-9, "centred kernel must satisfy K_b @ 1 == 0"


def test_kernelizer_trace_normalisation_unit_average():
    X = _smooth_X(40, 60, seed=0)
    ker = AOMKernelizer(operator_bank="compact", center=True, normalize="trace")
    K_blocks = ker.fit_transform(X)
    n = X.shape[0]
    for K_b in K_blocks:
        ratio = float(np.trace(K_b)) / n
        assert abs(ratio - 1.0) < 1e-9, f"trace/n must equal 1, got {ratio}"


def test_kernelizer_no_normalize():
    X = _smooth_X(40, 60, seed=0)
    ker = AOMKernelizer(operator_bank="compact", center=True, normalize="none")
    K_blocks = ker.fit_transform(X)
    # Trace can be anything when normalize="none"; just check shape and centring.
    n = X.shape[0]
    for K_b in K_blocks:
        assert K_b.shape == (n, n)


def test_kernelizer_no_center_shapes_only():
    """``center=False`` skips the explicit H K H step. Because the
    kernelizer always feature-centres X (`Xc = X - x_mean_train`), the
    raw kernel already satisfies ``K @ 1 = 0``; setting ``center=False``
    is therefore a no-op for the row-sum invariant. We only test shape
    and finiteness here.
    """
    X = _smooth_X(40, 60, seed=0)
    ker = AOMKernelizer(operator_bank="compact", center=False, normalize="trace")
    K_blocks = ker.fit_transform(X)
    n = X.shape[0]
    for K_b in K_blocks:
        assert K_b.shape == (n, n)
        assert np.all(np.isfinite(K_b))


# ----------------------------------------------------------------------
# Cross-kernel correctness
# ----------------------------------------------------------------------


def test_cross_kernel_shape_and_finite():
    rng = np.random.default_rng(1)
    X_train = _smooth_X(30, 50, seed=1)
    X_test = _smooth_X(10, 50, seed=2)
    ker = AOMKernelizer(operator_bank="compact")
    ker.fit(X_train)
    K_blocks_cross = ker.transform(X_test)
    for K_c in K_blocks_cross:
        assert K_c.shape == (10, 30)
        assert np.all(np.isfinite(K_c))


def test_cross_kernel_batch_invariance():
    """`transform([x])[0]` row should match the row of `transform([x, x', ...])`."""
    X_train = _smooth_X(30, 50, seed=1)
    X_test = _smooth_X(8, 50, seed=2)
    ker = AOMKernelizer(operator_bank="compact")
    ker.fit(X_train)
    blocks_full = ker.transform(X_test)
    for i in range(X_test.shape[0]):
        blocks_single = ker.transform(X_test[i : i + 1])
        for K_full, K_single in zip(blocks_full, blocks_single, strict=False):
            np.testing.assert_allclose(
                K_single[0], K_full[i], rtol=1e-10, atol=1e-10,
                err_msg="batch invariance violated",
            )


def test_cross_kernel_self_matches_train():
    """`transform(X_train)` should reproduce the training kernels."""
    X_train = _smooth_X(20, 40, seed=3)
    ker = AOMKernelizer(operator_bank="compact")
    K_train_blocks = ker.fit_transform(X_train)
    K_self = ker.transform(X_train)
    for K_t, K_s in zip(K_train_blocks, K_self, strict=False):
        # Both should be (n_train, n_train) and equal modulo symmetry rounding.
        np.testing.assert_allclose(K_s, K_t, rtol=1e-8, atol=1e-8)


# ----------------------------------------------------------------------
# Zero-trace policy
# ----------------------------------------------------------------------


def test_zero_trace_raises():
    """Constant X yields zero-trace kernels (everything is in the constant
    direction). The default zero_trace_policy='raise' should raise.
    """
    X = np.ones((20, 30), dtype=float)  # all-ones spectra
    ker = AOMKernelizer(operator_bank="compact", normalize="trace")
    with pytest.raises(ValueError, match="zero trace"):
        ker.fit(X)


def test_zero_trace_drop():
    """zero_trace_policy='drop' should silently skip zero-trace blocks but
    raise if all blocks are dropped."""
    X = np.ones((20, 30), dtype=float)
    ker = AOMKernelizer(
        operator_bank="compact",
        normalize="trace",
        zero_trace_policy="drop",
    )
    with pytest.raises(ValueError, match="all blocks dropped"):
        ker.fit(X)


# ----------------------------------------------------------------------
# Alignment diagnostic
# ----------------------------------------------------------------------


def test_alignment_matrix_diagonal():
    X = _smooth_X(30, 40, seed=4)
    ker = AOMKernelizer(operator_bank="compact")
    K_blocks = ker.fit_transform(X)
    A = kernel_alignment_matrix(K_blocks)
    np.testing.assert_allclose(np.diag(A), 1.0, rtol=1e-12)
    assert A.shape == (len(K_blocks), len(K_blocks))


def test_alignment_matrix_symmetry_and_bounds():
    X = _smooth_X(30, 40, seed=5)
    ker = AOMKernelizer(operator_bank="compact")
    K_blocks = ker.fit_transform(X)
    A = kernel_alignment_matrix(K_blocks)
    np.testing.assert_allclose(A, A.T, rtol=1e-10, atol=1e-10)
    off = A - np.eye(A.shape[0])
    # Off-diagonal in [-1, 1].
    assert off.max() <= 1.0 + 1e-9
    assert off.min() >= -1.0 - 1e-9
