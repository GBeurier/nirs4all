"""Tests for mkR per-block weight strategies."""

from __future__ import annotations

import numpy as np
import pytest

from aomridge.kernelizer import AOMKernelizer
from aomridge.weights import (
    kta_simplex_weights,
    manual_weights,
    softmax_cv_weights,
    uniform_weights,
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
# Uniform
# ----------------------------------------------------------------------


def test_uniform_weights_simplex():
    eta = uniform_weights(5)
    np.testing.assert_allclose(eta, np.full(5, 0.2), rtol=1e-12)
    assert eta.sum() == pytest.approx(1.0)


def test_uniform_weights_invalid():
    with pytest.raises(ValueError):
        uniform_weights(0)


# ----------------------------------------------------------------------
# Manual
# ----------------------------------------------------------------------


def test_manual_weights_projects_to_simplex():
    eta = manual_weights([1.0, 2.0, 1.0], 3)
    np.testing.assert_allclose(eta, [0.25, 0.5, 0.25], rtol=1e-12)


def test_manual_weights_clips_negatives():
    eta = manual_weights([1.0, -1.0, 1.0], 3)
    np.testing.assert_allclose(eta, [0.5, 0.0, 0.5], rtol=1e-12)


def test_manual_weights_rejects_all_zero():
    with pytest.raises(ValueError):
        manual_weights([0.0, 0.0, 0.0], 3)


def test_manual_weights_wrong_shape():
    with pytest.raises(ValueError):
        manual_weights([1.0, 2.0], 3)


# ----------------------------------------------------------------------
# KTA simplex
# ----------------------------------------------------------------------


def test_kta_returns_simplex():
    X = _smooth_X(40, 60, seed=10)
    rng = np.random.default_rng(10)
    y = rng.normal(0, 1.0, size=40)
    ker = AOMKernelizer(operator_bank="compact")
    K_blocks = ker.fit_transform(X)
    eta = kta_simplex_weights(K_blocks, y)
    assert eta.shape == (len(K_blocks),)
    assert np.all(eta >= 0.0)
    assert eta.sum() == pytest.approx(1.0, abs=1e-10)


def test_kta_top_k_zeros_outside():
    X = _smooth_X(40, 60, seed=11)
    rng = np.random.default_rng(11)
    y = rng.normal(0, 1.0, size=40)
    ker = AOMKernelizer(operator_bank="compact")
    K_blocks = ker.fit_transform(X)
    eta = kta_simplex_weights(K_blocks, y, top_k=3)
    nonzero = int(np.count_nonzero(eta))
    assert nonzero <= 3, f"top_k=3 should leave at most 3 nonzero weights"
    assert eta.sum() == pytest.approx(1.0, abs=1e-10)


def test_kta_constant_y_uniform_fallback():
    X = _smooth_X(30, 40, seed=12)
    y = np.full(30, 5.0)
    ker = AOMKernelizer(operator_bank="compact")
    K_blocks = ker.fit_transform(X)
    eta = kta_simplex_weights(K_blocks, y)
    np.testing.assert_allclose(eta, np.full(len(K_blocks), 1.0 / len(K_blocks)), rtol=1e-10)


# ----------------------------------------------------------------------
# softmax_cv
# ----------------------------------------------------------------------


def test_softmax_cv_returns_simplex():
    X = _smooth_X(60, 50, seed=20)
    rng = np.random.default_rng(20)
    y = rng.normal(0, 1.0, size=60)
    ker = AOMKernelizer(operator_bank="compact")
    K_blocks = ker.fit_transform(X)
    alphas = np.logspace(-3, 3, 10)
    res = softmax_cv_weights(
        K_blocks, y - y.mean(),
        alphas=alphas,
        cv_n_splits=3,
        n_restarts=2,
        max_iter=20,
        random_state=0,
    )
    assert res.eta.shape == (len(K_blocks),)
    assert np.all(res.eta >= 0.0)
    assert res.eta.sum() == pytest.approx(1.0, abs=1e-10)
    assert res.alpha > 0.0
    assert np.isfinite(res.inner_cv_rmse)


def test_softmax_cv_uniform_under_strong_kl_reg():
    """With very large lambda_eta, softmax_cv should collapse close to uniform."""
    X = _smooth_X(60, 50, seed=21)
    rng = np.random.default_rng(21)
    y = rng.normal(0, 1.0, size=60)
    ker = AOMKernelizer(operator_bank="compact")
    K_blocks = ker.fit_transform(X)
    alphas = np.logspace(-3, 3, 10)
    res = softmax_cv_weights(
        K_blocks, y - y.mean(),
        alphas=alphas,
        cv_n_splits=3,
        n_restarts=2,
        max_iter=20,
        lambda_eta=100.0,  # very strong reg toward uniform
        random_state=0,
    )
    B = len(K_blocks)
    # Should be close to uniform.
    np.testing.assert_allclose(res.eta, np.full(B, 1.0 / B), atol=0.1)
