"""Tests for AOMMoEClassifier and BlockSparseAOMMBPLSClassifier (Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest

from multiview.classifiers import (
    AOMMoEClassifier,
    BlockSparseAOMMBPLSClassifier,
)


def _three_class_block_signal(n=300, p=120, seed=0):
    """Three classes; class 0 has signal in block 0, class 1 in block 1, class 2 in block 2."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    blocks = [(0, 40), (40, 80), (80, 120)]
    y = np.zeros(n, dtype=int)
    third = n // 3
    for c, (s, e) in enumerate(blocks):
        idx_start = c * third
        idx_end = (c + 1) * third if c < 2 else n
        # Boost class c spectra in their respective block
        X[idx_start:idx_end, s:e] += rng.standard_normal((idx_end - idx_start, e - s)) * 2.0 + c
        y[idx_start:idx_end] = c
    perm = rng.permutation(n)
    return X[perm], y[perm]


class TestAOMMoEClassifier:
    def test_per_view_predicts_three_classes(self):
        X, y = _three_class_block_signal(n=300, p=120, seed=30)
        Xtr, Xte = X[:240], X[240:]
        ytr, yte = y[:240], y[240:]
        clf = AOMMoEClassifier(
            expert_layout="per_view", routing="soft", K=3,
            per_expert_components=8, random_state=0,
        )
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = float((pred == yte).mean())
        # Random would be 1/3; we should clearly beat that.
        assert acc > 0.5

    def test_predict_proba_shape(self):
        X, y = _three_class_block_signal(n=180, p=120, seed=31)
        clf = AOMMoEClassifier(
            expert_layout="per_view", routing="soft", K=3,
            per_expert_components=4, random_state=0,
        )
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (180, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


class TestBlockSparseAOMMBPLSClassifier:
    def test_v1_predicts(self):
        X, y = _three_class_block_signal(n=300, p=120, seed=32)
        Xtr, Xte = X[:240], X[240:]
        ytr, yte = y[:240], y[240:]
        clf = BlockSparseAOMMBPLSClassifier(
            K=3, preproc_bank_name=None, max_components=8,
            criterion="holdout", random_state=0,
        )
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = float((pred == yte).mean())
        assert acc > 0.5
