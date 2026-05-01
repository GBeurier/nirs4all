"""Tests for `BlockMaskOperator` and `ViewBuilder` (Phase 1).

Covers:
- BlockMaskOperator strict-linearity, idempotency, symmetry, edge cases.
- Composition order: `M . A_preproc` (mask after preprocessing).
- ViewBuilder bank structure and validation.
- cv_splitter threading fix in `_criterion_score_at_indices`.
"""

from __future__ import annotations

import numpy as np
import pytest

from aompls.operators import (
    ComposedOperator,
    DetrendProjectionOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    LinearSpectralOperator,
    SavitzkyGolayOperator,
)
from multiview.views import BlockMaskOperator, ViewBuilder


# ---------------------------------------------------------------------------
# BlockMaskOperator
# ---------------------------------------------------------------------------


class TestBlockMaskOperatorConstruction:
    def test_valid_block(self):
        op = BlockMaskOperator(start=10, end=50, p=200)
        assert op.start == 10
        assert op.end == 50
        assert op.p == 200
        assert op.is_strict_linear

    def test_default_name(self):
        op = BlockMaskOperator(start=10, end=50, p=200)
        assert op.name == "mask_10_50"

    def test_custom_name(self):
        op = BlockMaskOperator(start=10, end=50, p=200, name="block_low")
        assert op.name == "block_low"

    @pytest.mark.parametrize("start,end,p", [
        (0, 0, 100),    # empty
        (10, 10, 100),  # empty
        (50, 30, 100),  # reversed
    ])
    def test_empty_or_reversed_rejected(self, start, end, p):
        with pytest.raises(ValueError, match="non-empty"):
            BlockMaskOperator(start=start, end=end, p=p)

    def test_full_cover_rejected(self):
        with pytest.raises(ValueError, match="degenerate"):
            BlockMaskOperator(start=0, end=100, p=100)

    @pytest.mark.parametrize("start,end,p", [
        (-1, 50, 100),
        (10, 101, 100),
        (50, 200, 100),
    ])
    def test_out_of_bounds_rejected(self, start, end, p):
        with pytest.raises(ValueError, match="out of bounds"):
            BlockMaskOperator(start=start, end=end, p=p)


class TestBlockMaskOperatorMath:
    @pytest.fixture
    def op(self):
        return BlockMaskOperator(start=10, end=50, p=100)

    def test_matrix_is_diagonal(self, op):
        M = op.matrix(100)
        assert M.shape == (100, 100)
        # Diagonal: ones on [10, 50), zeros elsewhere
        diag = np.diag(M)
        assert diag[:10].sum() == 0
        assert diag[10:50].sum() == 40
        assert diag[50:].sum() == 0
        # Off-diagonal: zeros
        off = M - np.diag(diag)
        assert np.allclose(off, 0.0)

    def test_symmetry(self, op):
        M = op.matrix(100)
        np.testing.assert_allclose(M, M.T)

    def test_idempotency(self, op):
        M = op.matrix(100)
        np.testing.assert_allclose(M @ M, M)

    def test_transform_zeroes_outside_block(self, op):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        Xb = op.transform(X)
        assert Xb.shape == X.shape
        np.testing.assert_allclose(Xb[:, :10], 0.0)
        np.testing.assert_allclose(Xb[:, 50:], 0.0)
        np.testing.assert_allclose(Xb[:, 10:50], X[:, 10:50])

    def test_transform_matches_explicit_matrix(self, op):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        M = op.matrix(100)
        # X . A^T is the convention for `transform`.
        np.testing.assert_allclose(op.transform(X), X @ M.T)

    def test_apply_cov_matches_explicit_matrix(self, op):
        rng = np.random.default_rng(0)
        S = rng.standard_normal((100, 4))
        M = op.matrix(100)
        np.testing.assert_allclose(op.apply_cov(S), M @ S)

    def test_adjoint_vec_matches_explicit_matrix(self, op):
        rng = np.random.default_rng(0)
        v = rng.standard_normal(100)
        M = op.matrix(100)
        np.testing.assert_allclose(op.adjoint_vec(v), M.T @ v)

    def test_cross_covariance_identity(self, op):
        """The strict-linearity identity `(X A^T)^T Y = A X^T Y` must hold."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 100))
        Y = rng.standard_normal((30, 1))
        Xb = op.transform(X)
        lhs = Xb.T @ Y
        rhs = op.apply_cov(X.T @ Y)
        np.testing.assert_allclose(lhs, rhs)


# ---------------------------------------------------------------------------
# Composition: BlockMask AFTER preprocessing
# ---------------------------------------------------------------------------


class TestCompositionOrder:
    @pytest.fixture
    def p(self):
        return 100

    @pytest.fixture
    def mask(self, p):
        return BlockMaskOperator(start=20, end=70, p=p)

    @pytest.fixture
    def sg_d1(self, p):
        return SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=p)

    def test_mask_after_preproc_via_composed_operator(self, mask, sg_d1, p):
        """`ComposedOperator([sg, mask])` corresponds to matrix `M . A_sg`.

        Per `bench/AOM_v0/aompls/operators.py:513-516`, ComposedOperator
        applies operators leftmost-first. So `transform(X) = mask(sg(X))`,
        which has effective matrix `M . A_sg`.
        """
        composed = ComposedOperator([sg_d1, mask])
        composed.fit(np.zeros((1, p)))
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, p))
        # Compute expected: SG first, then mask.
        X_sg = sg_d1.transform(X)
        X_expected = mask.transform(X_sg)
        np.testing.assert_allclose(composed.transform(X), X_expected)

    def test_mask_d_neq_d_mask(self, mask, sg_d1, p):
        """Mask and derivative do NOT commute: `M . D ≠ D . M`."""
        compose_M_D = ComposedOperator([sg_d1, mask])  # mask after SG
        compose_D_M = ComposedOperator([mask, sg_d1])  # SG after mask
        compose_M_D.fit(np.zeros((1, p)))
        compose_D_M.fit(np.zeros((1, p)))
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, p))
        out_M_D = compose_M_D.transform(X)
        out_D_M = compose_D_M.transform(X)
        # They must differ near block edges (boundary semantics, DESIGN_VIEWS §3.3).
        assert not np.allclose(out_M_D, out_D_M)

    def test_mask_after_preproc_zero_outside_block(self, mask, sg_d1, p):
        """Outside the block, the composed operator output is exactly zero."""
        composed = ComposedOperator([sg_d1, mask])
        composed.fit(np.zeros((1, p)))
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, p))
        out = composed.transform(X)
        np.testing.assert_allclose(out[:, :20], 0.0)
        np.testing.assert_allclose(out[:, 70:], 0.0)

    def test_composed_remains_strict_linear(self, mask, sg_d1):
        composed = ComposedOperator([sg_d1, mask])
        assert composed.is_strict_linear


# ---------------------------------------------------------------------------
# ViewBuilder
# ---------------------------------------------------------------------------


class TestViewBuilderPreprocOnly:
    def test_compact_size_and_identity(self):
        builder = ViewBuilder.preproc_only(bank_name="compact")
        bank = builder.build(p=200)
        # Compact bank from `bank_by_name` returns 9 ops including identity.
        assert len(bank) == 9
        # Identity is first or present.
        assert any(isinstance(op, IdentityOperator) for op in bank)

    def test_strict_linear_enforcement_rejects_non_strict(self):
        # Construct a fake non-strict operator and inject via the underlying
        # bank_by_name - we instead patch the bank with a non-strict op directly.
        class _Hostile(LinearSpectralOperator):
            is_strict_linear = False

            def __init__(self, p=None):
                super().__init__(name="hostile", p=p)

            def _matrix_impl(self, p):
                return np.eye(p)

        # Drive the strict-linear check directly on the helper.
        with pytest.raises(ValueError, match="strict-linear"):
            ViewBuilder._enforce_strict_linear([_Hostile(p=10)])


class TestViewBuilderBlocksOnly:
    def test_default_K3_size(self):
        builder = ViewBuilder.blocks_only(K=3, strategy="equal_width")
        bank = builder.build(p=200)
        # 1 identity + 3 block masks
        assert len(bank) == 4
        assert isinstance(bank[0], IdentityOperator)
        masks = [op for op in bank if isinstance(op, BlockMaskOperator)]
        assert len(masks) == 3
        starts = [m.start for m in masks]
        ends = [m.end for m in masks]
        # Contiguous, covers [0, p)
        assert starts == [0, 67, 133]
        assert ends == [67, 133, 200]

    def test_K_too_small_rejected(self):
        builder = ViewBuilder.blocks_only(K=1, strategy="equal_width")
        with pytest.raises(ValueError, match="K must be"):
            builder.build(p=100)

    def test_K_exceeds_p_rejected(self):
        builder = ViewBuilder.blocks_only(K=200, strategy="equal_width")
        with pytest.raises(ValueError, match="exceeds p"):
            builder.build(p=100)

    def test_K_equals_p_rejected(self):
        builder = ViewBuilder.blocks_only(K=10, strategy="equal_width")
        with pytest.raises(ValueError, match="degenerate"):
            builder.build(p=10)

    def test_p_too_small_rejected(self):
        builder = ViewBuilder.blocks_only(K=3, strategy="equal_width")
        with pytest.raises(ValueError, match="too narrow"):
            builder.build(p=5)  # 5 < 2*3

    def test_unknown_strategy_rejected(self):
        builder = ViewBuilder.blocks_only(K=3, strategy="banana")
        with pytest.raises(ValueError, match="unknown strategy"):
            builder.build(p=200)

    def test_phase2_strategies_stub(self):
        builder = ViewBuilder.blocks_only(K=3, strategy="quantile_width")
        with pytest.raises(NotImplementedError, match="quantile_width"):
            builder.build(p=200)
        builder = ViewBuilder.blocks_only(K=3, strategy="chemistry_NIR")
        with pytest.raises(NotImplementedError, match="chemistry_NIR"):
            builder.build(p=200)


class TestViewBuilderCombined:
    def test_compact_K3_with_global(self):
        builder = ViewBuilder.combined(
            bank_name="compact", K=3, strategy="equal_width", include_global=True,
        )
        bank = builder.build(p=200)
        # Layout: 1 identity + 3 masks + 8 non-identity preproc + (3 * 8) composed = 36
        assert len(bank) == 36
        # First entry: identity
        assert isinstance(bank[0], IdentityOperator)
        # Composed entries are last
        composed = [op for op in bank if isinstance(op, ComposedOperator)]
        assert len(composed) == 24

    def test_compact_K3_without_global(self):
        builder = ViewBuilder.combined(
            bank_name="compact", K=3, strategy="equal_width", include_global=False,
        )
        bank = builder.build(p=200)
        # 1 + 3 + (3 * 8) = 28
        assert len(bank) == 28
        # No bare global preproc operators (only identity, masks, composed)
        bare_preproc = [
            op for op in bank
            if not isinstance(op, (IdentityOperator, BlockMaskOperator, ComposedOperator))
        ]
        assert bare_preproc == []

    def test_composed_entry_is_mask_after_preproc(self):
        builder = ViewBuilder.combined(
            bank_name="compact", K=3, strategy="equal_width", include_global=False,
        )
        bank = builder.build(p=200)
        composed = [op for op in bank if isinstance(op, ComposedOperator)]
        assert composed
        first = composed[0]
        # ComposedOperator stores `self.operators` in apply order. With our
        # `[preproc, mask]` constructor, operators[0] is preproc, operators[1]
        # is the mask.
        assert isinstance(first.operators[0], (SavitzkyGolayOperator,
                                                 DetrendProjectionOperator,
                                                 FiniteDifferenceOperator))
        assert isinstance(first.operators[1], BlockMaskOperator)


# ---------------------------------------------------------------------------
# cv_splitter threading regression test
# ---------------------------------------------------------------------------


class TestCVSplitterThreaded:
    """Phase 1 foundational fix: `cv_splitter` must reach
    `_criterion_score_at_indices` so POP+SPXY-CV works.
    """

    def test_cv_score_regression_uses_provided_splitter(self):
        from aompls.scorers import cv_score_regression
        from sklearn.model_selection import KFold

        rng = np.random.default_rng(0)
        Xc = rng.standard_normal((40, 8))
        yc = rng.standard_normal(40)

        def fit_predict(X_tr, y_tr, X_va):
            return np.full(X_va.shape[0], y_tr.mean())

        # A custom splitter must drive folds; pass a deterministic 4-fold splitter.
        custom = KFold(n_splits=4, shuffle=True, random_state=12345)
        score_custom = cv_score_regression(
            Xc, yc, fit_predict, n_splits=2, random_state=0, cv_splitter=custom,
        )

        # Without the splitter, the n_splits=2 default branch would be used.
        score_default = cv_score_regression(
            Xc, yc, fit_predict, n_splits=2, random_state=0, cv_splitter=None,
        )

        # Both must be finite RMSEs but with 2 vs 4 folds the variance differs.
        # We only assert that the custom splitter is *used*: count how many
        # times fit_predict is called by counting unique val-set sizes.
        sizes = []

        def fit_predict_track(X_tr, y_tr, X_va):
            sizes.append(X_va.shape[0])
            return np.full(X_va.shape[0], y_tr.mean())

        cv_score_regression(
            Xc, yc, fit_predict_track, n_splits=2, random_state=0,
            cv_splitter=KFold(n_splits=5, shuffle=True, random_state=99),
        )
        # 5 folds were used, not 2. Sum of fold sizes equals n_samples.
        assert len(sizes) == 5
        assert sum(sizes) == 40
        # Reference values are not asserted; we only care that the splitter is honoured.
        assert np.isfinite(score_custom)
        assert np.isfinite(score_default)

    def test_criterion_score_at_indices_threads_splitter(self):
        """When `_criterion_score_at_indices` runs CV, the criterion's
        cv_splitter must reach `cv_score_regression`.
        """
        from aompls.banks import bank_by_name
        from aompls.scorers import CriterionConfig
        from aompls.selection import _criterion_score_at_indices
        from sklearn.model_selection import KFold

        rng = np.random.default_rng(0)
        n, p = 60, 50
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()

        bank = bank_by_name("compact", p=p)

        # Baseline: random KFold.
        crit_default = CriterionConfig(kind="cv", cv=3, random_state=0)
        score_default, _ = _criterion_score_at_indices(
            Xc, yc, bank, "simpls_covariance", [0, 0, 0],
            crit_default, "transformed",
        )

        # With a deterministic splitter — same fold count, different splits.
        custom = KFold(n_splits=3, shuffle=True, random_state=42)
        crit_custom = CriterionConfig(
            kind="cv", cv=3, random_state=0, cv_splitter=custom,
        )
        score_custom, _ = _criterion_score_at_indices(
            Xc, yc, bank, "simpls_covariance", [0, 0, 0],
            crit_custom, "transformed",
        )

        # Both must be finite numbers; if the splitter is ignored, the two
        # would silently equal one another (because `random_state=0` would
        # be reused). With the fix wired through, the partition differs and
        # score_custom != score_default.
        assert np.isfinite(score_default)
        assert np.isfinite(score_custom)
        assert not np.isclose(score_custom, score_default)
