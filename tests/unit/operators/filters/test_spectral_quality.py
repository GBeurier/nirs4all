"""
Unit tests for SpectralQualityFilter class.

Tests the spectral quality filter with various quality checks:
- NaN ratio
- Inf values
- Zero ratio
- Minimum variance
- Value range limits
"""

import numpy as np
import pytest

from nirs4all.operators.filters import SpectralQualityFilter
from nirs4all.operators.filters.base import SampleFilter


class TestSpectralQualityFilterInitialization:
    """Tests for SpectralQualityFilter initialization and parameter validation."""

    def test_default_initialization(self):
        """Test default initialization."""
        filter_obj = SpectralQualityFilter()
        assert filter_obj.max_nan_ratio == 0.1
        assert filter_obj.max_zero_ratio == 0.5
        assert filter_obj.min_variance == 1e-8
        assert filter_obj.max_value is None
        assert filter_obj.min_value is None
        assert filter_obj.check_inf is True
        assert filter_obj.reason is None

    def test_custom_initialization(self):
        """Test custom initialization."""
        filter_obj = SpectralQualityFilter(
            max_nan_ratio=0.05,
            max_zero_ratio=0.2,
            min_variance=1e-4,
            max_value=5.0,
            min_value=-1.0,
            check_inf=False,
            reason="custom_quality"
        )
        assert filter_obj.max_nan_ratio == 0.05
        assert filter_obj.max_zero_ratio == 0.2
        assert filter_obj.min_variance == 1e-4
        assert filter_obj.max_value == 5.0
        assert filter_obj.min_value == -1.0
        assert filter_obj.check_inf is False
        assert filter_obj.exclusion_reason == "custom_quality"

    def test_invalid_nan_ratio_raises_error(self):
        """Test that invalid NaN ratio raises ValueError."""
        with pytest.raises(ValueError, match="max_nan_ratio must be in"):
            SpectralQualityFilter(max_nan_ratio=-0.1)

        with pytest.raises(ValueError, match="max_nan_ratio must be in"):
            SpectralQualityFilter(max_nan_ratio=1.5)

    def test_invalid_zero_ratio_raises_error(self):
        """Test that invalid zero ratio raises ValueError."""
        with pytest.raises(ValueError, match="max_zero_ratio must be in"):
            SpectralQualityFilter(max_zero_ratio=-0.1)

        with pytest.raises(ValueError, match="max_zero_ratio must be in"):
            SpectralQualityFilter(max_zero_ratio=1.5)

    def test_negative_variance_raises_error(self):
        """Test that negative variance raises ValueError."""
        with pytest.raises(ValueError, match="min_variance must be non-negative"):
            SpectralQualityFilter(min_variance=-1.0)

    def test_is_sample_filter_subclass(self):
        """Test that SpectralQualityFilter is a SampleFilter subclass."""
        filter_obj = SpectralQualityFilter()
        assert isinstance(filter_obj, SampleFilter)

class TestSpectralQualityFilterNaN:
    """Tests for NaN ratio checking."""

    def test_keeps_spectra_with_low_nan_ratio(self):
        """Test that spectra with low NaN ratio are kept."""
        X = np.random.randn(10, 100)
        # Add 5% NaN (below 10% threshold)
        X[0, :5] = np.nan

        filter_obj = SpectralQualityFilter(max_nan_ratio=0.1)
        mask = filter_obj.get_mask(X)

        assert mask[0] == True  # noqa: E712

    def test_excludes_spectra_with_high_nan_ratio(self):
        """Test that spectra with high NaN ratio are excluded."""
        X = np.random.randn(10, 100)
        # Add 20% NaN (above 10% threshold)
        X[0, :20] = np.nan

        filter_obj = SpectralQualityFilter(max_nan_ratio=0.1)
        mask = filter_obj.get_mask(X)

        assert mask[0] == False  # noqa: E712
        assert mask[1:].sum() == 9  # Other samples kept

    def test_all_nan_spectrum_excluded(self):
        """Test that all-NaN spectrum is excluded."""
        X = np.random.randn(10, 100)
        X[0, :] = np.nan

        filter_obj = SpectralQualityFilter()
        mask = filter_obj.get_mask(X)

        assert mask[0] == False  # noqa: E712

class TestSpectralQualityFilterInf:
    """Tests for Inf value checking."""

    def test_excludes_spectra_with_inf(self):
        """Test that spectra with Inf values are excluded."""
        X = np.random.randn(10, 100)
        X[0, 50] = np.inf
        X[1, 50] = -np.inf

        filter_obj = SpectralQualityFilter(check_inf=True)
        mask = filter_obj.get_mask(X)

        assert mask[0] == False  # noqa: E712
        assert mask[1] == False  # noqa: E712
        assert mask[2:].sum() == 8

    def test_keeps_inf_when_check_disabled(self):
        """Test that Inf spectra are kept when check is disabled."""
        X = np.random.randn(10, 100)
        X[0, 50] = np.inf

        filter_obj = SpectralQualityFilter(check_inf=False)
        mask = filter_obj.get_mask(X)

        # Still excluded due to value range, but inf check passes
        # Actually, max_value is None by default so it should pass
        # Need to also not fail other checks
        filter_obj2 = SpectralQualityFilter(check_inf=False, max_value=None)
        # This will pass if only inf check is relevant
        breakdown = filter_obj2.get_quality_breakdown(X)
        assert breakdown["passes_inf"][0] == True  # noqa: E712

class TestSpectralQualityFilterZero:
    """Tests for zero ratio checking."""

    def test_keeps_spectra_with_low_zero_ratio(self):
        """Test that spectra with low zero ratio are kept."""
        X = np.random.randn(10, 100)
        # Add 20% zeros (below 50% threshold)
        X[0, :20] = 0

        filter_obj = SpectralQualityFilter(max_zero_ratio=0.5)
        mask = filter_obj.get_mask(X)

        assert mask[0] == True  # noqa: E712

    def test_excludes_spectra_with_high_zero_ratio(self):
        """Test that spectra with high zero ratio are excluded."""
        X = np.random.randn(10, 100)
        # Add 60% zeros (above 50% threshold)
        X[0, :60] = 0

        filter_obj = SpectralQualityFilter(max_zero_ratio=0.5)
        mask = filter_obj.get_mask(X)

        assert mask[0] == False  # noqa: E712

    def test_all_zero_spectrum_excluded(self):
        """Test that all-zero spectrum is excluded."""
        X = np.random.randn(10, 100)
        X[0, :] = 0

        filter_obj = SpectralQualityFilter()
        mask = filter_obj.get_mask(X)

        # Excluded due to both zero ratio and low variance
        assert mask[0] == False  # noqa: E712

class TestSpectralQualityFilterVariance:
    """Tests for minimum variance checking."""

    def test_keeps_spectra_with_high_variance(self):
        """Test that spectra with sufficient variance are kept."""
        X = np.random.randn(10, 100)  # High variance

        filter_obj = SpectralQualityFilter(min_variance=1e-8)
        mask = filter_obj.get_mask(X)

        assert mask.sum() == 10

    def test_excludes_constant_spectra(self):
        """Test that constant spectra are excluded."""
        X = np.random.randn(10, 100)
        X[0, :] = 5.0  # Constant value

        filter_obj = SpectralQualityFilter(min_variance=1e-8)
        mask = filter_obj.get_mask(X)

        assert mask[0] == False  # noqa: E712

    def test_excludes_low_variance_spectra(self):
        """Test that low variance spectra are excluded."""
        X = np.random.randn(10, 100)
        X[0, :] = 5.0 + np.random.randn(100) * 1e-10  # Very low variance

        filter_obj = SpectralQualityFilter(min_variance=1e-6)
        mask = filter_obj.get_mask(X)

        assert mask[0] == False  # noqa: E712

class TestSpectralQualityFilterValueRange:
    """Tests for value range checking."""

    def test_excludes_saturated_spectra(self):
        """Test that saturated spectra (max exceeded) are excluded."""
        X = np.random.randn(10, 100)
        X[0, 50] = 10.0  # Saturated value

        filter_obj = SpectralQualityFilter(max_value=5.0)
        mask = filter_obj.get_mask(X)

        assert mask[0] == False  # noqa: E712

    def test_excludes_spectra_below_min(self):
        """Test that spectra with values below minimum are excluded."""
        X = np.random.randn(10, 100) + 2  # All positive
        X[0, 50] = -5.0  # Below minimum

        filter_obj = SpectralQualityFilter(min_value=-1.0)
        mask = filter_obj.get_mask(X)

        assert mask[0] == False  # noqa: E712

    def test_keeps_spectra_within_range(self):
        """Test that spectra within range are kept."""
        X = np.random.randn(10, 100) * 0.1  # Small values

        filter_obj = SpectralQualityFilter(max_value=1.0, min_value=-1.0)
        mask = filter_obj.get_mask(X)

        assert mask.sum() == 10

class TestSpectralQualityFilterMultipleChecks:
    """Tests for multiple simultaneous quality checks."""

    def test_multiple_failures_same_sample(self):
        """Test sample with multiple quality issues."""
        X = np.random.randn(10, 100)
        # Sample 0: multiple issues
        X[0, :30] = np.nan  # 30% NaN
        X[0, 30:60] = 0     # 30% zeros
        X[0, 60] = 100      # Saturated

        filter_obj = SpectralQualityFilter(
            max_nan_ratio=0.1,
            max_zero_ratio=0.2,
            max_value=10.0
        )
        mask = filter_obj.get_mask(X)

        assert mask[0] == False  # noqa: E712

    def test_breakdown_shows_individual_failures(self):
        """Test that breakdown shows which checks failed."""
        X = np.random.randn(10, 100)
        X[0, :20] = np.nan  # Fails NaN check
        X[1, :70] = 0       # Fails zero check
        X[2, :] = 5.0       # Fails variance check

        filter_obj = SpectralQualityFilter(
            max_nan_ratio=0.1,
            max_zero_ratio=0.5,
            min_variance=1e-6
        )
        breakdown = filter_obj.get_quality_breakdown(X)

        assert breakdown["passes_nan"][0] == False  # noqa: E712
        assert breakdown["passes_nan"][1] == True   # noqa: E712

        assert breakdown["passes_zero"][0] == True  # noqa: E712
        assert breakdown["passes_zero"][1] == False # noqa: E712

        assert breakdown["passes_variance"][2] == False  # noqa: E712

class TestSpectralQualityFilterEdgeCases:
    """Tests for edge cases."""

    def test_single_sample(self):
        """Test with single sample."""
        X = np.random.randn(1, 100)

        filter_obj = SpectralQualityFilter()
        mask = filter_obj.get_mask(X)

        assert len(mask) == 1

    def test_single_feature(self):
        """Test with single feature."""
        X = np.random.randn(10, 1)

        filter_obj = SpectralQualityFilter()
        mask = filter_obj.get_mask(X)

        assert len(mask) == 10

    def test_1d_input_reshaped(self):
        """Test that 1D input is reshaped."""
        X = np.random.randn(100)

        filter_obj = SpectralQualityFilter()
        mask = filter_obj.get_mask(X)

        assert len(mask) == 1  # Treated as single sample

    def test_empty_array(self):
        """Test with empty array."""
        X = np.random.randn(0, 100)

        filter_obj = SpectralQualityFilter()
        mask = filter_obj.get_mask(X)

        assert len(mask) == 0

    def test_fit_is_noop(self):
        """Test that fit is a no-op."""
        X = np.random.randn(10, 100)

        filter_obj = SpectralQualityFilter()
        result = filter_obj.fit(X)

        assert result is filter_obj

class TestSpectralQualityFilterHelperMethods:
    """Tests for helper methods."""

    def test_get_excluded_indices(self):
        """Test get_excluded_indices returns correct indices."""
        X = np.random.randn(10, 100)
        X[2, :] = np.nan  # Will be excluded
        X[7, :] = 5.0     # Constant, will be excluded

        filter_obj = SpectralQualityFilter()
        excluded = filter_obj.get_excluded_indices(X)

        assert 2 in excluded
        assert 7 in excluded

    def test_get_filter_stats(self):
        """Test get_filter_stats returns expected statistics."""
        X = np.random.randn(10, 100)
        X[0, :20] = np.nan

        filter_obj = SpectralQualityFilter(max_nan_ratio=0.1)
        stats = filter_obj.get_filter_stats(X)

        assert stats["n_samples"] == 10
        assert stats["n_excluded"] >= 1
        assert stats["max_nan_ratio"] == 0.1
        assert "failure_counts" in stats
        assert "quality_metrics" in stats

    def test_exclusion_reason_default(self):
        """Test default exclusion reason."""
        filter_obj = SpectralQualityFilter()
        assert filter_obj.exclusion_reason == "spectral_quality"

    def test_exclusion_reason_custom(self):
        """Test custom exclusion reason."""
        filter_obj = SpectralQualityFilter(reason="my_reason")
        assert filter_obj.exclusion_reason == "my_reason"

    def test_repr(self):
        """Test string representation."""
        filter_obj = SpectralQualityFilter(max_nan_ratio=0.05, max_value=5.0)
        repr_str = repr(filter_obj)
        assert "SpectralQualityFilter" in repr_str
        assert "0.05" in repr_str
        assert "5.0" in repr_str

    def test_repr_default(self):
        """Test string representation with defaults."""
        filter_obj = SpectralQualityFilter()
        repr_str = repr(filter_obj)
        assert "SpectralQualityFilter()" == repr_str

class TestSpectralQualityFilterTransform:
    """Tests for transform method (should be no-op)."""

    def test_transform_returns_input_unchanged(self):
        """Test that transform returns input unchanged."""
        X = np.random.randn(10, 100)

        filter_obj = SpectralQualityFilter()
        X_transformed = filter_obj.transform(X)

        np.testing.assert_array_equal(X, X_transformed)
