"""
Unit tests for YOutlierFilter class.

Tests the Y-based outlier detection filter with various methods:
- IQR (Interquartile Range)
- Z-score
- Percentile
- MAD (Median Absolute Deviation)
"""

import numpy as np
import pytest

from nirs4all.operators.filters import YOutlierFilter
from nirs4all.operators.filters.base import SampleFilter


class TestYOutlierFilterInitialization:
    """Tests for YOutlierFilter initialization and parameter validation."""

    def test_default_initialization(self):
        """Test default initialization with IQR method."""
        filter_obj = YOutlierFilter()
        assert filter_obj.method == "iqr"
        assert filter_obj.threshold == 1.5
        assert filter_obj.lower_percentile == 1.0
        assert filter_obj.upper_percentile == 99.0
        assert filter_obj.reason is None

    def test_custom_initialization(self):
        """Test custom initialization."""
        filter_obj = YOutlierFilter(
            method="zscore",
            threshold=3.0,
            reason="custom_outlier"
        )
        assert filter_obj.method == "zscore"
        assert filter_obj.threshold == 3.0
        assert filter_obj.exclusion_reason == "custom_outlier"

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            YOutlierFilter(method="invalid_method")

    def test_negative_threshold_raises_error(self):
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be positive"):
            YOutlierFilter(threshold=-1.0)

    def test_zero_threshold_raises_error(self):
        """Test that zero threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be positive"):
            YOutlierFilter(threshold=0.0)

    def test_invalid_percentiles_raises_error(self):
        """Test that invalid percentiles raise ValueError."""
        with pytest.raises(ValueError, match="Percentiles must satisfy"):
            YOutlierFilter(method="percentile", lower_percentile=90.0, upper_percentile=10.0)

        with pytest.raises(ValueError, match="Percentiles must satisfy"):
            YOutlierFilter(method="percentile", lower_percentile=-1.0)

        with pytest.raises(ValueError, match="Percentiles must satisfy"):
            YOutlierFilter(method="percentile", upper_percentile=101.0)

    def test_is_sample_filter_subclass(self):
        """Test that YOutlierFilter is a SampleFilter subclass."""
        filter_obj = YOutlierFilter()
        assert isinstance(filter_obj, SampleFilter)

class TestYOutlierFilterIQR:
    """Tests for IQR (Interquartile Range) method."""

    def test_fit_computes_iqr_bounds(self):
        """Test that fit computes IQR-based bounds correctly."""
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        X = np.random.rand(10, 5)

        filter_obj = YOutlierFilter(method="iqr", threshold=1.5)
        filter_obj.fit(X, y)

        # Q1=3.25, Q3=7.75, IQR=4.5
        # lower = 3.25 - 1.5*4.5 = -3.5
        # upper = 7.75 + 1.5*4.5 = 14.5
        assert filter_obj.lower_bound_ is not None
        assert filter_obj.upper_bound_ is not None
        assert filter_obj.lower_bound_ < 1  # Should not exclude min value
        assert filter_obj.upper_bound_ > 10  # Should not exclude max value

    def test_get_mask_keeps_normal_values(self):
        """Test that normal values are kept (mask=True)."""
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        X = np.random.rand(10, 5)

        filter_obj = YOutlierFilter(method="iqr", threshold=1.5)
        filter_obj.fit(X, y)
        mask = filter_obj.get_mask(X, y)

        # All values should be kept (no extreme outliers)
        assert mask.sum() == 10

    def test_get_mask_excludes_outliers(self):
        """Test that outliers are excluded (mask=False)."""
        # Normal data with one extreme outlier
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])  # 100 is outlier
        X = np.random.rand(10, 5)

        filter_obj = YOutlierFilter(method="iqr", threshold=1.5)
        filter_obj.fit(X, y)
        mask = filter_obj.get_mask(X, y)

        # The outlier (100) should be excluded
        assert mask[-1] == False  # noqa: E712
        assert mask[:-1].sum() == 9  # Other samples kept

    def test_iqr_with_low_outlier(self):
        """Test IQR detection of low outliers."""
        y = np.array([-100, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # -100 is outlier
        X = np.random.rand(10, 5)

        filter_obj = YOutlierFilter(method="iqr", threshold=1.5)
        filter_obj.fit(X, y)
        mask = filter_obj.get_mask(X, y)

        assert mask[0] == False  # -100 should be excluded  # noqa: E712

class TestYOutlierFilterZscore:
    """Tests for Z-score method."""

    def test_zscore_fit_computes_mean_std(self):
        """Test that zscore fit computes mean and std."""
        y = np.array([1, 2, 3, 4, 5])
        X = np.random.rand(5, 5)

        filter_obj = YOutlierFilter(method="zscore", threshold=2.0)
        filter_obj.fit(X, y)

        assert filter_obj.center_ == pytest.approx(3.0)  # mean
        assert filter_obj.scale_ > 0  # std

    def test_zscore_excludes_outliers(self):
        """Test that z-score method excludes statistical outliers."""
        # Normal data with outlier
        np.random.seed(42)
        y = np.concatenate([np.random.normal(0, 1, 99), [10]])  # 10 is ~10 std away
        X = np.random.rand(100, 5)

        filter_obj = YOutlierFilter(method="zscore", threshold=3.0)
        filter_obj.fit(X, y)
        mask = filter_obj.get_mask(X, y)

        # The extreme value should be excluded
        assert mask[-1] == False  # noqa: E712

    def test_zscore_with_zero_std(self):
        """Test z-score with constant y values (std=0)."""
        y = np.array([5, 5, 5, 5, 5])  # Constant
        X = np.random.rand(5, 5)

        filter_obj = YOutlierFilter(method="zscore", threshold=3.0)
        filter_obj.fit(X, y)
        mask = filter_obj.get_mask(X, y)

        # All samples should be kept (no variation)
        assert mask.sum() == 5

class TestYOutlierFilterPercentile:
    """Tests for percentile method."""

    def test_percentile_fit_computes_bounds(self):
        """Test that percentile fit computes correct bounds."""
        y = np.arange(1, 101)  # 1 to 100
        X = np.random.rand(100, 5)

        filter_obj = YOutlierFilter(
            method="percentile",
            lower_percentile=5.0,
            upper_percentile=95.0
        )
        filter_obj.fit(X, y)

        assert filter_obj.lower_bound_ == pytest.approx(5.95, rel=0.1)
        assert filter_obj.upper_bound_ == pytest.approx(95.05, rel=0.1)

    def test_percentile_excludes_tails(self):
        """Test that percentile method excludes values in tails."""
        y = np.arange(1, 101)  # 1 to 100
        X = np.random.rand(100, 5)

        filter_obj = YOutlierFilter(
            method="percentile",
            lower_percentile=10.0,
            upper_percentile=90.0
        )
        filter_obj.fit(X, y)
        mask = filter_obj.get_mask(X, y)

        # Roughly 80% should be kept
        assert 75 <= mask.sum() <= 85

class TestYOutlierFilterMAD:
    """Tests for MAD (Median Absolute Deviation) method."""

    def test_mad_fit_computes_median_based_bounds(self):
        """Test that MAD fit computes median-based bounds."""
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        X = np.random.rand(9, 5)

        filter_obj = YOutlierFilter(method="mad", threshold=3.5)
        filter_obj.fit(X, y)

        assert filter_obj.center_ == 5.0  # median
        assert filter_obj.scale_ > 0

    def test_mad_robust_to_outliers(self):
        """Test that MAD is robust to outliers in fitting."""
        # Data with outlier - MAD should be more robust than z-score
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 100])
        X = np.random.rand(9, 5)

        filter_obj = YOutlierFilter(method="mad", threshold=3.5)
        filter_obj.fit(X, y)
        mask = filter_obj.get_mask(X, y)

        # The outlier should be excluded
        assert mask[-1] == False  # noqa: E712

class TestYOutlierFilterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_fit_without_y_raises_error(self):
        """Test that fit without y raises ValueError."""
        X = np.random.rand(10, 5)
        filter_obj = YOutlierFilter()

        with pytest.raises(ValueError, match="requires y values"):
            filter_obj.fit(X, None)

    def test_get_mask_without_y_raises_error(self):
        """Test that get_mask without y raises ValueError."""
        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        filter_obj = YOutlierFilter()
        filter_obj.fit(X, y)

        with pytest.raises(ValueError, match="requires y values"):
            filter_obj.get_mask(X, None)

    def test_get_mask_before_fit_raises_error(self):
        """Test that get_mask before fit raises ValueError."""
        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        filter_obj = YOutlierFilter()

        with pytest.raises(ValueError, match="has not been fitted"):
            filter_obj.get_mask(X, y)

    def test_handles_nan_in_y(self):
        """Test that NaN values in y are handled (excluded)."""
        y = np.array([1, 2, 3, np.nan, 5])
        X = np.random.rand(5, 5)

        filter_obj = YOutlierFilter(method="iqr", threshold=1.5)
        filter_obj.fit(X, y)
        mask = filter_obj.get_mask(X, y)

        # NaN should be excluded
        assert mask[3] == False  # noqa: E712

    def test_all_nan_y_raises_error(self):
        """Test that all-NaN y raises ValueError."""
        y = np.array([np.nan, np.nan, np.nan])
        X = np.random.rand(3, 5)

        filter_obj = YOutlierFilter()
        with pytest.raises(ValueError, match="no valid"):
            filter_obj.fit(X, y)

    def test_multidimensional_y_is_flattened(self):
        """Test that 2D y arrays are flattened."""
        y = np.array([[1], [2], [3], [4], [100]])  # 2D with outlier
        X = np.random.rand(5, 5)

        filter_obj = YOutlierFilter(method="iqr", threshold=1.5)
        filter_obj.fit(X, y)
        mask = filter_obj.get_mask(X, y)

        assert len(mask) == 5
        assert mask[-1] == False  # Outlier excluded  # noqa: E712

class TestYOutlierFilterHelperMethods:
    """Tests for helper methods."""

    def test_get_excluded_indices(self):
        """Test get_excluded_indices returns correct indices."""
        y = np.array([1, 2, 3, 4, 100])  # 100 is outlier at index 4
        X = np.random.rand(5, 5)

        filter_obj = YOutlierFilter(method="iqr", threshold=1.5)
        filter_obj.fit(X, y)
        excluded = filter_obj.get_excluded_indices(X, y)

        assert 4 in excluded

    def test_get_kept_indices(self):
        """Test get_kept_indices returns correct indices."""
        y = np.array([1, 2, 3, 4, 100])  # 100 is outlier at index 4
        X = np.random.rand(5, 5)

        filter_obj = YOutlierFilter(method="iqr", threshold=1.5)
        filter_obj.fit(X, y)
        kept = filter_obj.get_kept_indices(X, y)

        assert 4 not in kept
        assert len(kept) == 4

    def test_get_filter_stats(self):
        """Test get_filter_stats returns expected statistics."""
        y = np.array([1, 2, 3, 4, 100])
        X = np.random.rand(5, 5)

        filter_obj = YOutlierFilter(method="iqr", threshold=1.5)
        filter_obj.fit(X, y)
        stats = filter_obj.get_filter_stats(X, y)

        assert stats["n_samples"] == 5
        assert stats["n_excluded"] >= 1  # At least outlier
        assert stats["method"] == "iqr"
        assert stats["threshold"] == 1.5
        assert stats["lower_bound"] is not None
        assert stats["upper_bound"] is not None
        assert "y_range" in stats

    def test_exclusion_reason_default(self):
        """Test default exclusion reason."""
        filter_obj = YOutlierFilter(method="zscore")
        assert filter_obj.exclusion_reason == "y_outlier_zscore"

    def test_exclusion_reason_custom(self):
        """Test custom exclusion reason."""
        filter_obj = YOutlierFilter(reason="my_custom_reason")
        assert filter_obj.exclusion_reason == "my_custom_reason"

    def test_repr(self):
        """Test string representation."""
        filter_obj = YOutlierFilter(method="iqr", threshold=2.0)
        repr_str = repr(filter_obj)
        assert "YOutlierFilter" in repr_str
        assert "iqr" in repr_str
        assert "2.0" in repr_str

class TestYOutlierFilterTransform:
    """Tests for transform method (should be no-op)."""

    def test_transform_returns_input_unchanged(self):
        """Test that transform returns input unchanged."""
        X = np.random.rand(10, 5)
        y = np.random.rand(10)

        filter_obj = YOutlierFilter()
        filter_obj.fit(X, y)
        X_transformed = filter_obj.transform(X)

        np.testing.assert_array_equal(X, X_transformed)

    def test_fit_transform_returns_input_unchanged(self):
        """Test that fit_transform returns input unchanged."""
        X = np.random.rand(10, 5)
        y = np.random.rand(10)

        filter_obj = YOutlierFilter()
        X_transformed = filter_obj.fit_transform(X, y)

        np.testing.assert_array_equal(X, X_transformed)
