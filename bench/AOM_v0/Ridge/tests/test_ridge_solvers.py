"""Phase 2 tests for dual Ridge solvers."""

from __future__ import annotations

import numpy as np
import pytest
from aomridge.solvers import (
    make_alpha_grid,
    predict_dual,
    solve_dual_ridge,
    solve_dual_ridge_path_eigh,
)


def _make_psd_kernel(n=20, p=15, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    K = X @ X.T
    return K, X


def test_alpha_grid_basic_properties():
    K, _ = _make_psd_kernel()
    alphas = make_alpha_grid(K, n_grid=50, low=-6.0, high=6.0)
    assert alphas.shape == (50,)
    assert np.all(alphas > 0)
    # Logarithmic spacing: log10 difference is monotone
    log_d = np.diff(np.log10(alphas))
    assert np.allclose(log_d, log_d[0], atol=1e-12)
    # base = trace(K) / n
    base = np.trace(K) / K.shape[0]
    assert np.isclose(alphas[0], base * 1e-6)
    assert np.isclose(alphas[-1], base * 1e6)


def test_cholesky_matches_direct_solve_univariate():
    K, _ = _make_psd_kernel()
    n = K.shape[0]
    rng = np.random.default_rng(1)
    y = rng.normal(size=n)
    alpha = 1e-1
    C = solve_dual_ridge(K, y, alpha=alpha, method="cholesky")
    C_ref = np.linalg.solve(K + alpha * np.eye(n), y)
    np.testing.assert_allclose(C, C_ref, atol=1e-10, rtol=1e-10)
    assert C.shape == (n,)


def test_cholesky_matches_direct_solve_multivariate():
    K, _ = _make_psd_kernel()
    n = K.shape[0]
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(n, 3))
    alpha = 5e-2
    C = solve_dual_ridge(K, Y, alpha=alpha, method="cholesky")
    C_ref = np.linalg.solve(K + alpha * np.eye(n), Y)
    np.testing.assert_allclose(C, C_ref, atol=1e-10, rtol=1e-10)
    assert C.shape == (n, 3)


def test_eigh_matches_cholesky():
    K, _ = _make_psd_kernel(n=25, p=10, seed=3)
    rng = np.random.default_rng(4)
    Y = rng.normal(size=(K.shape[0], 2))
    alpha = 0.7
    C_chol = solve_dual_ridge(K, Y, alpha=alpha, method="cholesky")
    C_eigh = solve_dual_ridge(K, Y, alpha=alpha, method="eigh")
    np.testing.assert_allclose(C_chol, C_eigh, atol=1e-9, rtol=1e-9)


def test_path_solve_matches_individual_solves():
    K, _ = _make_psd_kernel()
    n = K.shape[0]
    rng = np.random.default_rng(5)
    Y = rng.normal(size=(n, 2))
    alphas = make_alpha_grid(K, n_grid=8)
    path = solve_dual_ridge_path_eigh(K, Y, alphas)
    assert path.shape == (alphas.size, n, 2)
    for i, a in enumerate(alphas):
        ref = solve_dual_ridge(K, Y, alpha=a, method="eigh")
        np.testing.assert_allclose(path[i], ref, atol=1e-9, rtol=1e-9)


def test_alpha_must_be_positive():
    K, _ = _make_psd_kernel()
    with pytest.raises(ValueError):
        solve_dual_ridge(K, np.zeros(K.shape[0]), alpha=0.0)
    with pytest.raises(ValueError):
        solve_dual_ridge(K, np.zeros(K.shape[0]), alpha=-1.0)


def test_jitter_recovers_singular_kernel():
    # Construct a rank-deficient K
    rng = np.random.default_rng(6)
    Z = rng.normal(size=(15, 4))
    K = Z @ Z.T  # rank 4 in R^15
    n = K.shape[0]
    Y = rng.normal(size=(n, 1))
    alpha = 1e-6
    C = solve_dual_ridge(K, Y, alpha=alpha, method="cholesky")
    # Compare to a numpy-stable solve with the same alpha
    C_ref = np.linalg.solve(K + alpha * np.eye(n), Y)
    np.testing.assert_allclose(C, C_ref, atol=1e-6, rtol=1e-6)


def test_predict_dual_shape():
    rng = np.random.default_rng(7)
    K_cross = rng.normal(size=(5, 12))
    dual = rng.normal(size=(12, 3))
    out = predict_dual(K_cross, dual)
    assert out.shape == (5, 3)
    np.testing.assert_allclose(out, K_cross @ dual)


def test_symmetrization_handles_asymmetric_input():
    # Kernels passed in with floating-point asymmetry must still solve
    K, _ = _make_psd_kernel()
    K_perturbed = K + 1e-12 * np.random.default_rng(0).normal(size=K.shape)
    n = K.shape[0]
    y = np.random.default_rng(8).normal(size=n)
    C = solve_dual_ridge(K_perturbed, y, alpha=1.0, method="cholesky")
    K_sym = 0.5 * (K_perturbed + K_perturbed.T)
    C_ref = np.linalg.solve(K_sym + np.eye(n), y)
    np.testing.assert_allclose(C, C_ref, atol=1e-9, rtol=1e-9)
