"""
Unit tests for HighLeverageFilter class.

Tests the high leverage filter with various methods:
- Hat matrix diagonal (direct)
- PCA-based leverage
"""

import numpy as np
import pytest

from nirs4all.operators.filters import HighLeverageFilter
from nirs4all.operators.filters.base import SampleFilter


class TestHighLeverageFilterInitialization:
    """Tests for HighLeverageFilter initialization and parameter validation."""

    def test_default_initialization(self):
        """Test default initialization."""
        filter_obj = HighLeverageFilter()
        assert filter_obj.method == "hat"
        assert filter_obj.threshold_multiplier == 2.0
        assert filter_obj.absolute_threshold is None
        assert filter_obj.n_components is None
        assert filter_obj.center is True
        assert filter_obj.reason is None

    def test_custom_initialization(self):
        """Test custom initialization."""
        filter_obj = HighLeverageFilter(
            method="pca",
            threshold_multiplier=3.0,
            n_components=5,
            center=False,
            reason="custom_leverage"
        )
        assert filter_obj.method == "pca"
        assert filter_obj.threshold_multiplier == 3.0
        assert filter_obj.n_components == 5
        assert filter_obj.center is False
        assert filter_obj.exclusion_reason == "custom_leverage"

    def test_absolute_threshold(self):
        """Test initialization with absolute threshold."""
        filter_obj = HighLeverageFilter(absolute_threshold=0.3)
        assert filter_obj.absolute_threshold == 0.3

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            HighLeverageFilter(method="invalid")

    def test_invalid_multiplier_raises_error(self):
        """Test that invalid threshold_multiplier raises ValueError."""
        with pytest.raises(ValueError, match="threshold_multiplier must be positive"):
            HighLeverageFilter(threshold_multiplier=0)

        with pytest.raises(ValueError, match="threshold_multiplier must be positive"):
            HighLeverageFilter(threshold_multiplier=-1)

    def test_invalid_absolute_threshold_raises_error(self):
        """Test that invalid absolute_threshold raises ValueError."""
        with pytest.raises(ValueError, match="absolute_threshold must be in"):
            HighLeverageFilter(absolute_threshold=0)

        with pytest.raises(ValueError, match="absolute_threshold must be in"):
            HighLeverageFilter(absolute_threshold=1.0)

        with pytest.raises(ValueError, match="absolute_threshold must be in"):
            HighLeverageFilter(absolute_threshold=1.5)

    def test_is_sample_filter_subclass(self):
        """Test that HighLeverageFilter is a SampleFilter subclass."""
        filter_obj = HighLeverageFilter()
        assert isinstance(filter_obj, SampleFilter)


class TestHighLeverageFilterHatMethod:
    """Tests for hat matrix method."""

    def test_hat_fit_computes_precision(self):
        """Test that hat method fit computes precision matrix."""
        np.random.seed(42)
        X = np.random.randn(50, 5)

        filter_obj = HighLeverageFilter(method="hat")
        filter_obj.fit(X)

        assert filter_obj.precision_ is not None
        assert filter_obj.mean_ is not None
        assert filter_obj.threshold_ is not None

    def test_hat_keeps_normal_samples(self):
        """Test that hat method keeps normal samples."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        filter_obj = HighLeverageFilter(method="hat", threshold_multiplier=2.0)
        filter_obj.fit(X)
        mask = filter_obj.get_mask(X)

        # Most samples should be kept
        assert mask.sum() >= 80

    def test_hat_excludes_high_leverage_points(self):
        """Test that hat method excludes high leverage points."""
        np.random.seed(42)
        X = np.random.randn(99, 5)
        # Add a high leverage point (far from center in X space)
        high_leverage = np.array([[20, 20, 20, 20, 20]])
        X_with_hl = np.vstack([X, high_leverage])

        filter_obj = HighLeverageFilter(method="hat", threshold_multiplier=2.0)
        filter_obj.fit(X_with_hl)
        mask = filter_obj.get_mask(X_with_hl)

        # High leverage point should be excluded
        assert mask[-1] == False  # noqa: E712

    def test_hat_with_absolute_threshold(self):
        """Test hat method with absolute threshold."""
        np.random.seed(42)
        X = np.random.randn(50, 5)

        filter_obj = HighLeverageFilter(method="hat", absolute_threshold=0.5)
        filter_obj.fit(X)

        assert filter_obj.threshold_ == 0.5


class TestHighLeverageFilterPCAMethod:
    """Tests for PCA-based method."""

    def test_pca_fit_creates_pca(self):
        """Test that PCA method creates PCA model."""
        np.random.seed(42)
        X = np.random.randn(50, 100)  # High-dimensional

        filter_obj = HighLeverageFilter(method="pca", n_components=10)
        filter_obj.fit(X)

        assert filter_obj.pca_ is not None
        assert filter_obj.pca_.n_components_ == 10

    def test_pca_keeps_normal_samples(self):
        """Test that PCA method keeps normal samples."""
        np.random.seed(42)
        X = np.random.randn(100, 50)

        filter_obj = HighLeverageFilter(method="pca", n_components=10)
        filter_obj.fit(X)
        mask = filter_obj.get_mask(X)

        # Most samples should be kept
        assert mask.sum() >= 80

    def test_pca_excludes_high_leverage_points(self):
        """Test that PCA method excludes high leverage points."""
        np.random.seed(42)
        X = np.random.randn(99, 50)
        # Add high leverage point
        high_leverage = np.ones((1, 50)) * 10
        X_with_hl = np.vstack([X, high_leverage])

        filter_obj = HighLeverageFilter(method="pca", n_components=10)
        filter_obj.fit(X_with_hl)
        mask = filter_obj.get_mask(X_with_hl)

        # High leverage point should be excluded
        assert mask[-1] == False  # noqa: E712

    def test_pca_auto_n_components(self):
        """Test automatic n_components selection."""
        np.random.seed(42)
        X = np.random.randn(50, 100)

        filter_obj = HighLeverageFilter(method="pca")  # n_components=None
        filter_obj.fit(X)

        # Should use min(n_samples-1, n_features, 50)
        assert filter_obj.pca_.n_components_ <= 49


class TestHighLeverageFilterHighDimensional:
    """Tests for high-dimensional data (n_features >= n_samples)."""

    def test_high_dim_uses_pca_automatically(self):
        """Test that high-dimensional data uses PCA for hat method."""
        np.random.seed(42)
        X = np.random.randn(30, 100)  # More features than samples

        filter_obj = HighLeverageFilter(method="hat")
        filter_obj.fit(X)

        # Should use PCA internally
        assert filter_obj.pca_ is not None

    def test_high_dim_works_correctly(self):
        """Test that high-dimensional filtering works."""
        np.random.seed(42)
        X = np.random.randn(30, 100)

        filter_obj = HighLeverageFilter(method="hat")
        filter_obj.fit(X)
        mask = filter_obj.get_mask(X)

        assert len(mask) == 30
        assert mask.sum() > 0


class TestHighLeverageFilterCentering:
    """Tests for data centering."""

    def test_centering_enabled(self):
        """Test that centering is applied when enabled."""
        np.random.seed(42)
        X = np.random.randn(50, 5) + 100  # Offset data

        filter_obj = HighLeverageFilter(center=True)
        filter_obj.fit(X)

        # Mean should be close to data mean
        np.testing.assert_array_almost_equal(
            filter_obj.mean_,
            np.mean(X, axis=0),
            decimal=10
        )

    def test_centering_disabled(self):
        """Test that centering is not applied when disabled."""
        np.random.seed(42)
        X = np.random.randn(50, 5)

        filter_obj = HighLeverageFilter(center=False)
        filter_obj.fit(X)

        # Mean should be zeros
        np.testing.assert_array_equal(filter_obj.mean_, np.zeros(5))


class TestHighLeverageFilterEdgeCases:
    """Tests for edge cases."""

    def test_single_feature(self):
        """Test with single feature."""
        np.random.seed(42)
        X = np.random.randn(50, 1)

        filter_obj = HighLeverageFilter()
        filter_obj.fit(X)
        mask = filter_obj.get_mask(X)

        assert len(mask) == 50

    def test_1d_input_reshaped(self):
        """Test that 1D input is reshaped."""
        np.random.seed(42)
        X = np.random.randn(50)

        filter_obj = HighLeverageFilter()
        filter_obj.fit(X)
        mask = filter_obj.get_mask(X)

        assert len(mask) == 50

    def test_get_mask_before_fit_raises_error(self):
        """Test that get_mask before fit raises ValueError."""
        X = np.random.randn(10, 5)
        filter_obj = HighLeverageFilter()

        with pytest.raises(ValueError, match="has not been fitted"):
            filter_obj.get_mask(X)

    def test_insufficient_samples_raises_error(self):
        """Test that insufficient samples raises ValueError."""
        X = np.random.randn(1, 5)  # Only 1 sample
        filter_obj = HighLeverageFilter()

        with pytest.raises(ValueError, match="at least 2 samples"):
            filter_obj.fit(X)

    def test_two_samples(self):
        """Test with minimal (2) samples."""
        np.random.seed(42)
        X = np.random.randn(2, 5)

        filter_obj = HighLeverageFilter()
        filter_obj.fit(X)
        mask = filter_obj.get_mask(X)

        assert len(mask) == 2


class TestHighLeverageFilterGetLeverages:
    """Tests for get_leverages method."""

    def test_get_leverages_returns_array(self):
        """Test that get_leverages returns leverage values."""
        np.random.seed(42)
        X = np.random.randn(50, 5)

        filter_obj = HighLeverageFilter()
        filter_obj.fit(X)
        leverages = filter_obj.get_leverages(X)

        assert len(leverages) == 50
        assert np.all(leverages >= 0)

    def test_get_leverages_before_fit_raises_error(self):
        """Test that get_leverages before fit raises error."""
        X = np.random.randn(10, 5)
        filter_obj = HighLeverageFilter()

        with pytest.raises(ValueError, match="has not been fitted"):
            filter_obj.get_leverages(X)

    def test_high_leverage_point_has_higher_value(self):
        """Test that high leverage points have higher leverage values."""
        np.random.seed(42)
        X = np.random.randn(99, 5)
        high_leverage = np.array([[15, 15, 15, 15, 15]])
        X_with_hl = np.vstack([X, high_leverage])

        filter_obj = HighLeverageFilter()
        filter_obj.fit(X_with_hl)
        leverages = filter_obj.get_leverages(X_with_hl)

        # High leverage point should have highest leverage
        assert leverages[-1] == np.max(leverages)


class TestHighLeverageFilterHelperMethods:
    """Tests for helper methods."""

    def test_get_excluded_indices(self):
        """Test get_excluded_indices returns correct indices."""
        np.random.seed(42)
        X = np.random.randn(99, 5)
        high_leverage = np.array([[20, 20, 20, 20, 20]])
        X_with_hl = np.vstack([X, high_leverage])

        filter_obj = HighLeverageFilter(threshold_multiplier=2.0)
        filter_obj.fit(X_with_hl)
        excluded = filter_obj.get_excluded_indices(X_with_hl)

        assert 99 in excluded

    def test_get_filter_stats(self):
        """Test get_filter_stats returns expected statistics."""
        np.random.seed(42)
        X = np.random.randn(50, 5)

        filter_obj = HighLeverageFilter(threshold_multiplier=2.0)
        filter_obj.fit(X)
        stats = filter_obj.get_filter_stats(X)

        assert stats["n_samples"] == 50
        assert stats["method"] == "hat"
        assert stats["threshold_multiplier"] == 2.0
        assert "leverage_stats" in stats
        assert "threshold" in stats

    def test_exclusion_reason_default(self):
        """Test default exclusion reason."""
        filter_obj = HighLeverageFilter()
        assert filter_obj.exclusion_reason == "high_leverage"

    def test_exclusion_reason_custom(self):
        """Test custom exclusion reason."""
        filter_obj = HighLeverageFilter(reason="my_reason")
        assert filter_obj.exclusion_reason == "my_reason"

    def test_repr(self):
        """Test string representation."""
        filter_obj = HighLeverageFilter(method="hat", threshold_multiplier=3.0)
        repr_str = repr(filter_obj)
        assert "HighLeverageFilter" in repr_str
        assert "hat" in repr_str
        assert "3.0" in repr_str

    def test_repr_with_absolute_threshold(self):
        """Test repr with absolute threshold."""
        filter_obj = HighLeverageFilter(absolute_threshold=0.5)
        repr_str = repr(filter_obj)
        assert "absolute_threshold=0.5" in repr_str


class TestHighLeverageFilterTransform:
    """Tests for transform method (should be no-op)."""

    def test_transform_returns_input_unchanged(self):
        """Test that transform returns input unchanged."""
        np.random.seed(42)
        X = np.random.randn(10, 5)

        filter_obj = HighLeverageFilter()
        filter_obj.fit(X)
        X_transformed = filter_obj.transform(X)

        np.testing.assert_array_equal(X, X_transformed)
