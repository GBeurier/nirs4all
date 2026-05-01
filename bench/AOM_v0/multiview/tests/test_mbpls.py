"""Tests for block-sparse AOM-MBPLS (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from aompls.scorers import CriterionConfig
from multiview.estimators_mbpls import BlockSparseAOMMBPLSRegressor
from multiview.selection_mbpls import (
    MBPLSResult,
    _fit_block_sparse_fixed,
    derive_block_metadata,
    fit_block_sparse_aom,
)
from multiview.views import BlockMaskOperator, ViewBuilder


# ---------------------------------------------------------------------------
# Result coefficient assembly
# ---------------------------------------------------------------------------


class TestMBPLSResult:
    def test_empty_result_returns_zero_coef(self):
        res = MBPLSResult(
            Z=np.zeros((10, 0)), P=np.zeros((10, 0)),
            Q=np.zeros((1, 0)), T=np.zeros((20, 0)),
            op_indices=[], block_winners=[], n_components=0,
        )
        coef = res.coef()
        assert coef.shape == (10, 1)
        assert np.all(coef == 0.0)

    def test_coef_prefix_truncates(self):
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((10, 5))
        P = rng.standard_normal((10, 5))
        Q = rng.standard_normal((1, 5))
        T = rng.standard_normal((20, 5))
        res = MBPLSResult(
            Z=Z, P=P, Q=Q, T=T,
            op_indices=[0, 1, 2, 3, 4],
            block_winners=[(0, 0)] * 5,
            n_components=5,
        )
        coef_full = res.coef()
        coef_3 = res.coef_prefix(3)
        # Same input dim, same output dim
        assert coef_full.shape == coef_3.shape == (10, 1)
        # Different from full
        assert not np.allclose(coef_full, coef_3)
        # Truncating beyond n_components saturates
        coef_10 = res.coef_prefix(10)
        np.testing.assert_allclose(coef_10, coef_full)


# ---------------------------------------------------------------------------
# Block-sparse algorithm: math invariants
# ---------------------------------------------------------------------------


def _make_block_signal_data(n=100, p=120, signal_block=(40, 80), seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    # Signal lives only in the middle block.
    s, e = signal_block
    weights = rng.standard_normal(e - s)
    y = X[:, s:e] @ weights + 0.05 * rng.standard_normal(n)
    return X, y


class TestBlockSparseFixed:
    def test_block_sparse_fixed_pls_when_K1(self):
        """K=1 with identity operator should produce a non-trivial coefficient."""
        X, y = _make_block_signal_data(n=80, p=60, signal_block=(20, 50), seed=1)
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        # Full-cover BlockMaskOperator is rejected at construction; use IdentityOperator
        # as the K=1 stand-in. It exercises the same `_fit_block_sparse_fixed` path.
        from aompls.operators import IdentityOperator
        op = IdentityOperator(p=60)
        op.fit(Xc)
        block_masks = {0: np.ones(60)}
        op_to_block = [0]
        result = _fit_block_sparse_fixed(
            Xc=Xc, yc=yc, operators=[op],
            op_to_block=op_to_block, block_masks=block_masks,
            indices=[0, 0, 0],
        )
        assert result.n_components == 3
        # coef should be a non-trivial regressor
        assert np.linalg.norm(result.coef()) > 0

    def test_block_sparse_loading_is_block_supported(self):
        """P column should be zero outside the winning block."""
        X, y = _make_block_signal_data(n=80, p=120, signal_block=(40, 80), seed=2)
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        op = BlockMaskOperator(start=40, end=80, p=120)
        op.fit(Xc)
        block_masks = {0: np.zeros(120)}
        block_masks[0][40:80] = 1.0
        result = _fit_block_sparse_fixed(
            Xc=Xc, yc=yc, operators=[op],
            op_to_block=[0], block_masks=block_masks,
            indices=[0, 0, 0],
        )
        # Loading is zero outside [40, 80) for every committed LV
        for col in range(result.n_components):
            np.testing.assert_allclose(result.P[:40, col], 0.0)
            np.testing.assert_allclose(result.P[80:, col], 0.0)

    def test_block_sparse_predicts_signal(self):
        """With signal in a known block, the block-sparse fit should predict well."""
        X, y = _make_block_signal_data(n=200, p=120, signal_block=(40, 80), seed=3)
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        # K=3 equal-width blocks, only block 1 covers signal
        ops = []
        op_to_block = []
        block_masks = {}
        for k, (s, e) in enumerate([(0, 40), (40, 80), (80, 120)]):
            mask = np.zeros(120)
            mask[s:e] = 1.0
            block_masks[k] = mask
            op = BlockMaskOperator(start=s, end=e, p=120)
            op.fit(Xc)
            ops.append(op)
            op_to_block.append(k)
        # Fix the sequence: pick block 1 every LV
        result = _fit_block_sparse_fixed(
            Xc=Xc, yc=yc, operators=ops,
            op_to_block=op_to_block, block_masks=block_masks,
            indices=[1, 1, 1, 1, 1],
        )
        coef = result.coef()
        pred_train = Xc @ coef + y.mean()
        rmse_train = float(np.sqrt(np.mean((y - pred_train.ravel()) ** 2)))
        # Should fit much better than predicting the mean (~1.0 std).
        assert rmse_train < 0.5


# ---------------------------------------------------------------------------
# Block-sparse greedy selection (full algorithm)
# ---------------------------------------------------------------------------


class TestBlockSparseAOM:
    def test_picks_signal_block(self):
        """When signal is in block 1, greedy should pick block 1 first."""
        X, y = _make_block_signal_data(n=200, p=120, signal_block=(40, 80), seed=4)
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        ops = []
        op_to_block = []
        block_masks = {}
        for k, (s, e) in enumerate([(0, 40), (40, 80), (80, 120)]):
            mask = np.zeros(120)
            mask[s:e] = 1.0
            block_masks[k] = mask
            op = BlockMaskOperator(start=s, end=e, p=120)
            op.fit(Xc)
            ops.append(op)
            op_to_block.append(k)
        criterion = CriterionConfig(kind="holdout", random_state=0)
        result = fit_block_sparse_aom(
            Xc=Xc, yc=yc, operators=ops,
            op_to_block=op_to_block, block_masks=block_masks,
            n_components_max=5, criterion=criterion,
        )
        # First winner is block 1
        first_block, _ = result.block_winners[0]
        assert first_block == 1


# ---------------------------------------------------------------------------
# derive_block_metadata
# ---------------------------------------------------------------------------


class TestDeriveBlockMetadata:
    def test_blocks_only_filters_identity(self):
        bank = ViewBuilder.blocks_only(K=3, strategy="equal_width").build(p=200)
        # bank includes 1 identity + 3 BlockMaskOperators
        assert len(bank) == 4
        ops, op_to_block, masks = derive_block_metadata(
            bank=bank, blocks=[(0, 67), (67, 133), (133, 200)], p=200,
        )
        # Identity dropped
        assert len(ops) == 3
        assert all(isinstance(op, BlockMaskOperator) for op in ops)
        assert op_to_block == [0, 1, 2]
        assert all(masks[k].sum() > 0 for k in (0, 1, 2))

    def test_combined_filters_identity_and_bare_preproc(self):
        bank = ViewBuilder.combined(
            bank_name="compact", K=3, strategy="equal_width", include_global=True,
        ).build(p=200)
        # 36 entries: identity + 3 masks + 8 bare preproc + 24 composed
        ops, op_to_block, masks = derive_block_metadata(
            bank=bank, blocks=[(0, 67), (67, 133), (133, 200)], p=200,
        )
        # Kept: 3 masks + 24 composed = 27 entries
        assert len(ops) == 27
        # All entries must have a block
        assert len(op_to_block) == 27
        assert set(op_to_block) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Estimator integration
# ---------------------------------------------------------------------------


class TestBlockSparseAOMMBPLSRegressor:
    def test_fit_predict_v1_holdout(self):
        X, y = _make_block_signal_data(n=200, p=120, signal_block=(40, 80), seed=5)
        Xtr, Xte = X[:150], X[150:]
        ytr, yte = y[:150], y[150:]
        est = BlockSparseAOMMBPLSRegressor(
            n_components="auto", max_components=8,
            K=3, strategy="equal_width", preproc_bank_name=None,
            criterion="holdout", random_state=0,
        )
        est.fit(Xtr, ytr)
        pred = est.predict(Xte)
        rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
        # Should beat the null model (std of yte) significantly.
        baseline = float(yte.std())
        assert rmse < 0.6 * baseline

    def test_fit_predict_v2_holdout(self):
        X, y = _make_block_signal_data(n=200, p=120, signal_block=(40, 80), seed=6)
        Xtr, Xte = X[:150], X[150:]
        ytr, yte = y[:150], y[150:]
        est = BlockSparseAOMMBPLSRegressor(
            n_components="auto", max_components=8,
            K=3, strategy="equal_width", preproc_bank_name="compact",
            criterion="holdout", random_state=0,
        )
        est.fit(Xtr, ytr)
        pred = est.predict(Xte)
        rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
        baseline = float(yte.std())
        assert rmse < 0.6 * baseline

    def test_block_winners_and_selected_operators(self):
        X, y = _make_block_signal_data(n=200, p=120, signal_block=(40, 80), seed=7)
        est = BlockSparseAOMMBPLSRegressor(
            n_components="auto", max_components=4,
            K=3, strategy="equal_width", preproc_bank_name=None,
            criterion="holdout", random_state=0,
        )
        est.fit(X, y)
        winners = est.get_block_winners()
        assert all(0 <= k <= 2 for k, _ in winners)
        names = est.get_selected_operators()
        assert all(name.startswith("mask_") for name in names)
