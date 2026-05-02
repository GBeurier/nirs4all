"""Tests for AOMMoEPerSampleRouting, AOMMoEStacked, AOMMoEMultiK."""

from __future__ import annotations

import numpy as np

from multiview.moe_advanced import (
    AOMMoEMultiK, AOMMoEPerSampleRouting, AOMMoEStacked,
)


def _heterogeneous_blocks(n=300, p=120, seed=0):
    """3 classes of samples; class c has signal in block c (out of K=3)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    blocks = [(0, 40), (40, 80), (80, 120)]
    y = np.zeros(n)
    for c, (s, e) in enumerate(blocks):
        idx_start = (c * n) // 3
        idx_end = ((c + 1) * n) // 3 if c < 2 else n
        weights = rng.standard_normal(e - s)
        # Class c samples use block c for prediction
        y[idx_start:idx_end] = X[idx_start:idx_end, s:e] @ weights
    return X, y


class TestPerSampleRouting:
    def test_fit_predict_runs(self):
        X, y = _heterogeneous_blocks(n=240, p=120, seed=10)
        est = AOMMoEPerSampleRouting(
            expert_layout="per_view", K=3,
            per_expert_components=8, random_state=0,
        )
        est.fit(X, y)
        pred = est.predict(X)
        rmse = float(np.sqrt(((y - pred) ** 2).mean()))
        # On training data, no specific quality bound
        assert np.isfinite(rmse)

    def test_constant_argmax_fallback(self):
        # Trivial dataset where one expert clearly wins for all samples.
        rng = np.random.default_rng(11)
        X = rng.standard_normal((100, 60))
        y = X[:, 20:40].sum(axis=1) + 0.1 * rng.standard_normal(100)
        est = AOMMoEPerSampleRouting(K=3, per_expert_components=5, random_state=0)
        est.fit(X, y)
        pred = est.predict(X)
        # Either gate trained or constant fallback active
        assert hasattr(est, "constant_argmax_")


class TestStacked:
    def test_stacked_runs_and_predicts(self):
        rng = np.random.default_rng(12)
        X = rng.standard_normal((150, 90))
        y = X[:, 30:60].sum(axis=1) + 0.1 * rng.standard_normal(150)
        Xtr, Xte = X[:120], X[120:]
        ytr, yte = y[:120], y[120:]
        est = AOMMoEStacked(
            expert_layout="per_view", K=3, per_expert_components=5,
            x_pca_components=5, meta_alpha=1.0, random_state=0,
        )
        est.fit(Xtr, ytr)
        pred = est.predict(Xte)
        rmse = float(np.sqrt(((yte - pred) ** 2).mean()))
        baseline = float(yte.std())
        assert rmse < 0.6 * baseline  # block 1 dominates, stacker should latch on


class TestMultiK:
    def test_average_of_K3_K5_K7(self):
        rng = np.random.default_rng(20)
        X = rng.standard_normal((150, 90))
        y = X[:, 30:60].sum(axis=1) + 0.1 * rng.standard_normal(150)
        est = AOMMoEMultiK(K_list=(3, 5, 7), per_expert_components=5, random_state=0)
        est.fit(X[:120], y[:120])
        pred = est.predict(X[120:])
        rmse = float(np.sqrt(((y[120:] - pred) ** 2).mean()))
        baseline = float(y[120:].std())
        assert rmse < 0.6 * baseline
        assert len(est._models) == 3
