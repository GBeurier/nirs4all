"""Tests for FCKOperator and the FCK-augmented banks.

The FCK operator wraps a single fractional-derivative kernel as a
strict linear spectral operator. The bank presets test that the operator
is registered correctly and that the AOM-PLS covariance fast-paths
remain consistent with the explicit-matrix path.
"""

from __future__ import annotations

import numpy as np
import pytest
from aompls.banks import (
    bank_by_name,
    compact_with_fck_bank,
    fck_compact_bank,
)
from aompls.operators import FCKOperator, _fck_kernel


class TestFCKKernel:
    def test_l1_normalised(self):
        for alpha in (0.5, 1.0, 1.5, 2.0):
            for scale in (1.0, 2.0):
                k = _fck_kernel(alpha, scale, 31, sigma=3.0)
                assert np.isclose(np.sum(np.abs(k)), 1.0, atol=1e-6)
                assert k.shape == (31,)

    def test_zero_mean_for_alpha_above_zero(self):
        for alpha in (0.5, 1.0, 1.5, 2.0):
            k = _fck_kernel(alpha, scale=1.0, kernel_size=31, sigma=3.0)
            assert abs(np.mean(k)) < 1e-6

    def test_alpha_zero_is_gaussian(self):
        k = _fck_kernel(0.0, scale=1.0, kernel_size=15, sigma=3.0)
        # Gaussian smoother -> non-negative
        assert np.all(k >= 0)

    def test_invalid_kernel_size(self):
        with pytest.raises(ValueError, match="kernel_size"):
            _fck_kernel(1.0, 1.0, 14, sigma=3.0)

    def test_invalid_scale(self):
        with pytest.raises(ValueError, match="scale"):
            _fck_kernel(1.0, 0.0, 15, sigma=3.0)


class TestFCKOperator:
    def test_strict_linearity(self):
        """A strict linear operator must satisfy A(aX + bY) = a A(X) + b A(Y)."""
        rng = np.random.RandomState(0)
        op = FCKOperator(alpha=1.5, scale=2.0, kernel_size=31).fit(rng.randn(4, 200))
        X = rng.randn(8, 200)
        Y = rng.randn(8, 200)
        a, b = 0.7, -1.3
        left = op.transform(a * X + b * Y)
        right = a * op.transform(X) + b * op.transform(Y)
        np.testing.assert_allclose(left, right, atol=1e-10)

    def test_apply_cov_matches_matrix(self):
        rng = np.random.RandomState(0)
        op = FCKOperator(alpha=1.0, scale=1.0, kernel_size=15).fit(rng.randn(4, 100))
        S = rng.randn(100, 3)
        cheap = op.apply_cov(S)
        explicit = op.matrix(100) @ S
        np.testing.assert_allclose(cheap, explicit, atol=1e-10)

    def test_adjoint_matches_matrix_transpose(self):
        rng = np.random.RandomState(0)
        op = FCKOperator(alpha=2.0, scale=1.0, kernel_size=31).fit(rng.randn(4, 80))
        v = rng.randn(80)
        cheap = op.adjoint_vec(v)
        explicit = op.matrix(80).T @ v
        np.testing.assert_allclose(cheap, explicit, atol=1e-10)

    def test_transform_matches_matrix_path(self):
        """Cross-check the cheap convolution against the explicit X @ A^T path."""
        rng = np.random.RandomState(0)
        op = FCKOperator(alpha=1.5, scale=2.0, kernel_size=21).fit(rng.randn(4, 64))
        X = rng.randn(10, 64)
        cheap = op.transform(X)
        explicit = X @ op.matrix(64).T
        np.testing.assert_allclose(cheap, explicit, atol=1e-10)

    def test_unique_name_per_hyperparams(self):
        names = {
            FCKOperator(alpha=a, scale=s, kernel_size=31).name
            for a in (0.5, 1.0, 1.5, 2.0)
            for s in (1.0, 2.0)
        }
        assert len(names) == 8


class TestFCKBanks:
    def test_fck_compact_bank_size(self):
        bank = fck_compact_bank(p=200)
        assert len(bank) == 8                    # 4 alphas × 2 scales
        assert all(isinstance(op, FCKOperator) for op in bank)
        # All names unique
        assert len({op.name for op in bank}) == 8

    def test_compact_with_fck_size(self):
        bank = compact_with_fck_bank(p=200)
        # compact has 9 operators; +8 FCK = 17
        assert len(bank) == 17

    def test_bank_by_name_alias_compact_with_fck(self):
        bank = bank_by_name("compact_with_fck", p=200)
        assert len(bank) == 17

    def test_bank_by_name_alias_fck_compact(self):
        bank = bank_by_name("fck_compact", p=200)
        assert len(bank) == 8
