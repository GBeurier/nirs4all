"""Tests for StackingHybrid (Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from multiview.stacking import StackingHybrid


def _block_signal(n=200, p=120, signal=(40, 80), seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    s, e = signal
    weights = rng.standard_normal(e - s)
    y = X[:, s:e] @ weights + 0.05 * rng.standard_normal(n)
    return X, y


class TestStackingHybrid:
    def test_two_pls_experts_predicts_well(self):
        X, y = _block_signal(n=200, p=120, signal=(40, 80), seed=20)
        Xtr, Xte = X[:150], X[150:]
        ytr, yte = y[:150], y[150:]
        est = StackingHybrid(
            base_estimators=[
                ("pls5", PLSRegression(n_components=5)),
                ("pls10", PLSRegression(n_components=10)),
            ],
            n_oof_folds=3, meta_alpha=1.0, random_state=0,
        )
        est.fit(Xtr, ytr)
        pred = est.predict(Xte)
        rmse = float(np.sqrt(((yte - pred) ** 2).mean()))
        baseline = float(yte.std())
        assert rmse < 0.6 * baseline

    def test_nonneg_weights(self):
        X, y = _block_signal(n=160, p=120, signal=(40, 80), seed=21)
        est = StackingHybrid(
            base_estimators=[
                ("pls3", PLSRegression(n_components=3)),
                ("pls7", PLSRegression(n_components=7)),
            ],
            n_oof_folds=3, random_state=0, nonneg=True,
        )
        est.fit(X, y)
        # NNLS gives nonneg weights
        assert (est.meta_weights_ >= 0).all()
