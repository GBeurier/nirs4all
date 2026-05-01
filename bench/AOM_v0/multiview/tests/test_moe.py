"""Tests for AOMMoERegressor (Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest

from multiview.moe import AOMMoERegressor


def _block_signal(n=200, p=120, signal=(40, 80), seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    s, e = signal
    weights = rng.standard_normal(e - s)
    y = X[:, s:e] @ weights + 0.05 * rng.standard_normal(n)
    return X, y


class TestAOMMoEPerView:
    def test_per_view_picks_signal_block(self):
        X, y = _block_signal(n=200, p=120, signal=(40, 80), seed=10)
        Xtr, Xte = X[:150], X[150:]
        ytr, yte = y[:150], y[150:]
        est = AOMMoERegressor(
            expert_layout="per_view", routing="hard",
            K=3, per_expert_components=8, random_state=0,
        )
        est.fit(Xtr, ytr)
        gate = est.get_gate_weights()
        # Hard routing → 1-hot
        assert np.isclose(gate.sum(), 1.0)
        assert np.count_nonzero(gate) == 1
        # Block 1 (signal block) wins.
        assert int(np.argmax(gate)) == 1

    def test_soft_routing_predicts_well(self):
        X, y = _block_signal(n=200, p=120, signal=(40, 80), seed=11)
        Xtr, Xte = X[:150], X[150:]
        ytr, yte = y[:150], y[150:]
        est = AOMMoERegressor(
            expert_layout="per_view", routing="soft",
            K=3, per_expert_components=8, random_state=0,
        )
        est.fit(Xtr, ytr)
        pred = est.predict(Xte)
        rmse = float(np.sqrt(((yte - pred) ** 2).mean()))
        baseline = float(yte.std())
        assert rmse < 0.6 * baseline


class TestAOMMoEPerPreproc:
    def test_per_preproc_predicts(self):
        X, y = _block_signal(n=200, p=120, signal=(40, 80), seed=12)
        Xtr, Xte = X[:150], X[150:]
        ytr, yte = y[:150], y[150:]
        est = AOMMoERegressor(
            expert_layout="per_preproc", routing="soft",
            bank_name="compact", per_expert_components=8,
            random_state=0,
        )
        est.fit(Xtr, ytr)
        pred = est.predict(Xte)
        rmse = float(np.sqrt(((yte - pred) ** 2).mean()))
        baseline = float(yte.std())
        # The compact bank (SG smoothers, derivatives) should at least beat null.
        assert rmse < baseline


class TestGateWeights:
    def test_soft_gate_sums_to_one(self):
        X, y = _block_signal(n=160, p=120, signal=(40, 80), seed=13)
        est = AOMMoERegressor(
            expert_layout="per_view", routing="soft",
            K=3, per_expert_components=4, random_state=0,
        )
        est.fit(X, y)
        gate = est.get_gate_weights()
        np.testing.assert_allclose(gate.sum(), 1.0)
        assert (gate >= 0).all()
