"""Tests for the REML / ML negative log-likelihood and analytic gradient."""

from __future__ import annotations

import numpy as np
import pytest

from mkm.kernelizer import AOMKernelizer
from mkm.likelihood import (
    compute_neg_log_ml,
    compute_neg_log_reml,
    compute_neg_log_reml_grad,
    fit_fixed_effects,
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


def _setup(n: int = 30, p: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = _smooth_X(n, p, seed)
    y = rng.normal(0, 1.0, size=n)
    ker = AOMKernelizer(operator_bank="compact")
    K_blocks = ker.fit_transform(X)
    X_f = np.ones((n, 1), dtype=float)
    return K_blocks, y, X_f


# ----------------------------------------------------------------------
# Brute-force REML reference (uses np.linalg.solve / slogdet)
# ----------------------------------------------------------------------


def _brute_force_neg_log_reml(theta, K_blocks, y, X_f, jitter=1e-8):
    n = K_blocks[0].shape[0]
    sigma2 = np.exp(theta[:-1])
    sigma2_e = float(np.exp(theta[-1]))
    V = np.zeros((n, n), dtype=float)
    for K_b, s2 in zip(K_blocks, sigma2, strict=False):
        V += float(s2) * K_b
    V += sigma2_e * np.eye(n) + jitter * np.eye(n)
    V = 0.5 * (V + V.T)
    sign_V, logdet_V = np.linalg.slogdet(V)
    Vinv_y = np.linalg.solve(V, y)
    Vinv_Xf = np.linalg.solve(V, X_f)
    M = X_f.T @ Vinv_Xf
    sign_M, logdet_M = np.linalg.slogdet(M)
    beta_hat = np.linalg.solve(M, X_f.T @ Vinv_y)
    resid = y - X_f @ beta_hat
    quad = float(resid @ np.linalg.solve(V, resid))
    p_f = X_f.shape[1]
    return 0.5 * (logdet_V + logdet_M + quad + (n - p_f) * np.log(2 * np.pi))


def test_neg_log_reml_matches_bruteforce():
    K_blocks, y, X_f = _setup()
    rng = np.random.default_rng(99)
    B = len(K_blocks)
    for trial in range(5):
        theta = rng.normal(0.0, 1.0, size=B + 1)
        ours = compute_neg_log_reml(theta, K_blocks, y, X_f).neg_log_lik
        brute = _brute_force_neg_log_reml(theta, K_blocks, y, X_f)
        np.testing.assert_allclose(ours, brute, rtol=1e-8, atol=1e-8,
                                   err_msg=f"trial {trial}: ours {ours:.6f}, "
                                           f"brute {brute:.6f}")


def test_neg_log_ml_matches_bruteforce():
    K_blocks, y, X_f = _setup()
    rng = np.random.default_rng(7)
    B = len(K_blocks)

    def brute_force_ml(theta):
        n = K_blocks[0].shape[0]
        sigma2 = np.exp(theta[:-1])
        sigma2_e = float(np.exp(theta[-1]))
        V = np.zeros((n, n), dtype=float)
        for K_b, s2 in zip(K_blocks, sigma2, strict=False):
            V += float(s2) * K_b
        V += sigma2_e * np.eye(n) + 1e-8 * np.eye(n)
        V = 0.5 * (V + V.T)
        sign_V, logdet_V = np.linalg.slogdet(V)
        Vinv_y = np.linalg.solve(V, y)
        Vinv_Xf = np.linalg.solve(V, X_f)
        M = X_f.T @ Vinv_Xf
        beta_hat = np.linalg.solve(M, X_f.T @ Vinv_y)
        resid = y - X_f @ beta_hat
        quad = float(resid @ np.linalg.solve(V, resid))
        return 0.5 * (logdet_V + quad + n * np.log(2 * np.pi))

    for trial in range(3):
        theta = rng.normal(0.0, 1.0, size=B + 1)
        ours = compute_neg_log_ml(theta, K_blocks, y, X_f).neg_log_lik
        brute = brute_force_ml(theta)
        np.testing.assert_allclose(ours, brute, rtol=1e-8, atol=1e-8)


# ----------------------------------------------------------------------
# Gradient: analytic vs finite difference
# ----------------------------------------------------------------------


def test_gradient_matches_finite_difference():
    K_blocks, y, X_f = _setup(n=24, p=30, seed=42)
    rng = np.random.default_rng(42)
    B = len(K_blocks)
    theta0 = rng.normal(0.0, 0.5, size=B + 1)
    res0 = compute_neg_log_reml(theta0, K_blocks, y, X_f)
    g_analytic = compute_neg_log_reml_grad(theta0, K_blocks, res0)

    eps = 1e-5
    g_fd = np.zeros_like(theta0)
    for j in range(theta0.size):
        tp = theta0.copy(); tp[j] += eps
        tm = theta0.copy(); tm[j] -= eps
        fp = compute_neg_log_reml(tp, K_blocks, y, X_f).neg_log_lik
        fm = compute_neg_log_reml(tm, K_blocks, y, X_f).neg_log_lik
        g_fd[j] = (fp - fm) / (2 * eps)

    # Combined absolute + relative tolerance.
    np.testing.assert_allclose(g_analytic, g_fd, rtol=1e-4, atol=1e-4,
                               err_msg=f"grad mismatch:\n  analytic={g_analytic}\n  fd={g_fd}")


# ----------------------------------------------------------------------
# Fixed-effect rank handling
# ----------------------------------------------------------------------


def test_fit_fixed_effects_full_rank():
    rng = np.random.default_rng(3)
    X_f = rng.normal(size=(20, 3))
    X_used, p_f = fit_fixed_effects(X_f)
    assert p_f == 3
    assert X_used.shape == (20, 3)


def test_fit_fixed_effects_rank_deficient():
    X_f = np.column_stack([np.ones(20), np.ones(20)])  # collinear
    X_used, p_f = fit_fixed_effects(X_f)
    assert p_f == 1
    assert X_used.shape == (20, 1)


def test_fit_fixed_effects_intercept_only():
    X_f = np.ones((20, 1))
    X_used, p_f = fit_fixed_effects(X_f)
    assert p_f == 1
    assert X_used.shape == (20, 1)
