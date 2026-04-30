"""Tests for AOMMultiKernelMixedModel."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from mkm.estimator import AOMMultiKernelMixedModel
from synthetic import make_R1, make_R2


def _smooth_X_y(n: int, p: int, seed: int = 0):
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
    beta = rng.normal(0, 0.5, size=p)
    y = X @ beta + rng.normal(0, 0.5, size=n)
    return X, y


# ----------------------------------------------------------------------
# Sklearn API
# ----------------------------------------------------------------------


def test_clone_preserves_params():
    est = AOMMultiKernelMixedModel(method="ml", n_random_restarts=2)
    cloned = clone(est)
    assert cloned.method == "ml"
    assert cloned.n_random_restarts == 2
    assert not hasattr(cloned, "alpha_dual_")


def test_get_set_params():
    est = AOMMultiKernelMixedModel()
    p = est.get_params()
    assert "method" in p
    est.set_params(method="ml")
    assert est.method == "ml"


def test_fit_predict_basic():
    X, y = _smooth_X_y(40, 50, seed=0)
    est = AOMMultiKernelMixedModel(
        method="reml",
        n_random_restarts=2,
        max_iter=50,
        random_state=0,
    )
    est.fit(X, y)
    assert hasattr(est, "alpha_dual_")
    assert est.alpha_dual_.shape == (40,)
    assert est.sigma2_blocks_.shape == (est.B_,)
    assert est.sigma2_residual_ > 0.0
    # Predict shape OK
    y_pred = est.predict(X)
    assert y_pred.shape == (40,)
    assert np.all(np.isfinite(y_pred))
    # Score finite
    assert np.isfinite(est.score(X, y))
    # Variance contributions sum to ~1
    rc = list(est.relative_contributions_.values())
    assert abs(sum(rc) - 1.0) < 1e-9


# ----------------------------------------------------------------------
# Synthetic R1: variance recovery
# ----------------------------------------------------------------------


def test_R1_variance_recovery():
    """On R1 (1 active block, high SNR), the active block should receive
    dominant variance share."""
    ds = make_R1(n=120, p=200, snr=8.0, seed=7)
    est = AOMMultiKernelMixedModel(
        method="reml",
        n_random_restarts=4,
        max_iter=100,
        random_state=0,
    )
    est.fit(ds.X, ds.y)
    rel = est.relative_contributions_
    block_names = est.block_names_
    active_idx = int(np.argmax(ds.true_eta))
    active_name = block_names[active_idx]
    active_share = float(rel[active_name])
    # Should be by far the largest random-effect contribution.
    other_shares = [
        v for k, v in rel.items()
        if k != "_residual" and k != active_name
    ]
    assert active_share >= max(other_shares) - 1e-9, (
        f"active block share {active_share:.3f} should be >= max other "
        f"{max(other_shares):.3f}"
    )
    # Active share should explain notable fraction of total variance.
    assert active_share > 0.1, f"active share {active_share:.3f} too small"


# ----------------------------------------------------------------------
# Multi-output rejection
# ----------------------------------------------------------------------


def test_multi_output_rejected():
    X, y = _smooth_X_y(20, 30, seed=8)
    Y = np.column_stack([y, y])
    est = AOMMultiKernelMixedModel(n_random_restarts=1, max_iter=10)
    with pytest.raises(ValueError, match="multi-output"):
        est.fit(X, Y)


def test_unknown_method_rejected():
    X, y = _smooth_X_y(20, 30, seed=9)
    est = AOMMultiKernelMixedModel(method="bogus", n_random_restarts=1,
                                     max_iter=10)
    with pytest.raises(ValueError, match="method"):
        est.fit(X, y)
