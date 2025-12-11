"""
Unit tests for XOutlierFilter class.

Tests the X-based outlier detection filter with various methods:
- Mahalanobis distance
- Robust Mahalanobis (MinCovDet)
- PCA residual (Q-statistic)
- PCA leverage (Hotelling's T²)
- Isolation Forest
- Local Outlier Factor
"""

import numpy as np
import pytest

from nirs4all.operators.filters import XOutlierFilter
from nirs4all.operators.filters.base import SampleFilter


class TestXOutlierFilterInitialization:
    """Tests for XOutlierFilter initialization and parameter validation."""

    def test_default_initialization(self):
        """Test default initialization with Mahalanobis method."""
        filter_obj = XOutlierFilter()
        assert filter_obj.method == "mahalanobis"
        assert filter_obj.threshold is None
        assert filter_obj.n_components is None
        assert filter_obj.contamination == 0.1
        assert filter_obj.reason is None

    def test_custom_initialization(self):
        """Test custom initialization."""
        filter_obj = XOutlierFilter(
            method="pca_residual",
            threshold=5.0,
            n_components=5,
            reason="custom_outlier"
        )
        assert filter_obj.method == "pca_residual"
        assert filter_obj.threshold == 5.0
        assert filter_obj.n_components == 5
        assert filter_obj.exclusion_reason == "custom_outlier"

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            XOutlierFilter(method="invalid_method")

    def test_invalid_contamination_raises_error(self):
        """Test that invalid contamination raises ValueError."""
        with pytest.raises(ValueError, match="contamination must be in"):
            XOutlierFilter(contamination=0.0)

        with pytest.raises(ValueError, match="contamination must be in"):
            XOutlierFilter(contamination=0.6)

        with pytest.raises(ValueError, match="contamination must be in"):
            XOutlierFilter(contamination=-0.1)

    def test_is_sample_filter_subclass(self):
        """Test that XOutlierFilter is a SampleFilter subclass."""
        filter_obj = XOutlierFilter()
        assert isinstance(filter_obj, SampleFilter)


class TestXOutlierFilterMahalanobis:
    """Tests for Mahalanobis distance methods."""

    def test_mahalanobis_fit_computes_covariance(self):
        """Test that Mahalanobis fit computes covariance."""
        np.random.seed(42)
        X = np.random.randn(50, 5)

        filter_obj = XOutlierFilter(method="mahalanobis", threshold=3.0)
        filter_obj.fit(X)

        assert filter_obj.center_ is not None
        assert filter_obj.precision_ is not None
        assert filter_obj.threshold_ == 3.0

    def test_mahalanobis_keeps_normal_samples(self):
        """Test that Mahalanobis keeps normal samples."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        filter_obj = XOutlierFilter(method="mahalanobis", threshold=3.0)
        filter_obj.fit(X)
        mask = filter_obj.get_mask(X)

        # Most samples should be kept (allowing some statistical variation)
        assert mask.sum() >= 80

    def test_mahalanobis_excludes_outliers(self):
        """Test that Mahalanobis excludes outliers."""
        np.random.seed(42)
        X = np.random.randn(99, 5)
        outlier = np.array([[20, 20, 20, 20, 20]])  # Far from center
        X_with_outlier = np.vstack([X, outlier])

        filter_obj = XOutlierFilter(method="mahalanobis", threshold=3.0)
        filter_obj.fit(X_with_outlier)
        mask = filter_obj.get_mask(X_with_outlier)

        # The outlier should be excluded
        assert mask[-1] == False  # noqa: E712

    def test_robust_mahalanobis_fit(self):
        """Test robust Mahalanobis fitting with MinCovDet."""
        np.random.seed(42)
        X = np.random.randn(50, 5)

        filter_obj = XOutlierFilter(method="robust_mahalanobis", threshold=3.0)
        filter_obj.fit(X)

        assert filter_obj.center_ is not None
        assert filter_obj.precision_ is not None

    def test_high_dimensional_uses_pca(self):
        """Test that high-dimensional data uses PCA for Mahalanobis."""
        np.random.seed(42)
        X = np.random.randn(30, 100)  # More features than samples

        filter_obj = XOutlierFilter(method="mahalanobis", threshold=3.0)
        filter_obj.fit(X)

        # Should use PCA internally
        assert filter_obj.pca_ is not None


class TestXOutlierFilterPCA:
    """Tests for PCA-based methods."""

    def test_pca_residual_fit(self):
        """Test PCA residual (Q-statistic) fitting."""
        np.random.seed(42)
        X = np.random.randn(50, 10)

        filter_obj = XOutlierFilter(method="pca_residual", n_components=5)
        filter_obj.fit(X)

        assert filter_obj.pca_ is not None
        assert filter_obj.pca_.n_components_ == 5
        assert filter_obj.threshold_ is not None

    def test_pca_residual_excludes_unusual_spectra(self):
        """Test PCA residual excludes spectra with unusual patterns."""
        np.random.seed(42)
        # Normal spectra with smooth patterns
        X = np.cumsum(np.random.randn(99, 50), axis=1)
        # Outlier with random noise pattern
        outlier = np.random.randn(1, 50) * 10
        X_with_outlier = np.vstack([X, outlier])

        filter_obj = XOutlierFilter(method="pca_residual", n_components=5)
        filter_obj.fit(X_with_outlier)
        mask = filter_obj.get_mask(X_with_outlier)

        # Outlier likely excluded (high Q-statistic)
        # Note: May not always be excluded depending on random state
        assert mask.sum() < len(X_with_outlier)

    def test_pca_leverage_fit(self):
        """Test PCA leverage (Hotelling's T²) fitting."""
        np.random.seed(42)
        X = np.random.randn(50, 10)

        filter_obj = XOutlierFilter(method="pca_leverage", n_components=5)
        filter_obj.fit(X)

        assert filter_obj.pca_ is not None
        assert filter_obj.threshold_ is not None

    def test_pca_leverage_with_custom_threshold(self):
        """Test PCA leverage with custom threshold."""
        np.random.seed(42)
        X = np.random.randn(50, 10)

        filter_obj = XOutlierFilter(method="pca_leverage", threshold=10.0)
        filter_obj.fit(X)
        mask = filter_obj.get_mask(X)

        assert filter_obj.threshold_ == 10.0
        assert mask.sum() > 0


class TestXOutlierFilterSklearn:
    """Tests for sklearn-based methods (Isolation Forest, LOF)."""

    def test_isolation_forest_fit(self):
        """Test Isolation Forest fitting."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        filter_obj = XOutlierFilter(
            method="isolation_forest",
            contamination=0.1,
            random_state=42
        )
        filter_obj.fit(X)

        assert filter_obj.detector_ is not None

    def test_isolation_forest_excludes_outliers(self):
        """Test Isolation Forest excludes outliers."""
        np.random.seed(42)
        X = np.random.randn(99, 5)
        outlier = np.array([[10, 10, 10, 10, 10]])
        X_with_outlier = np.vstack([X, outlier])

        filter_obj = XOutlierFilter(
            method="isolation_forest",
            contamination=0.05,
            random_state=42
        )
        filter_obj.fit(X_with_outlier)
        mask = filter_obj.get_mask(X_with_outlier)

        # Outlier should be excluded
        assert mask[-1] == False  # noqa: E712

    def test_lof_fit(self):
        """Test Local Outlier Factor fitting."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        filter_obj = XOutlierFilter(method="lof", contamination=0.1)
        filter_obj.fit(X)

        assert filter_obj.detector_ is not None

    def test_lof_excludes_outliers(self):
        """Test LOF excludes outliers."""
        np.random.seed(42)
        X = np.random.randn(99, 5)
        outlier = np.array([[10, 10, 10, 10, 10]])
        X_with_outlier = np.vstack([X, outlier])

        filter_obj = XOutlierFilter(method="lof", contamination=0.05)
        filter_obj.fit(X_with_outlier)
        mask = filter_obj.get_mask(X_with_outlier)

        # Outlier should be excluded
        assert mask[-1] == False  # noqa: E712


class TestXOutlierFilterEdgeCases:
    """Tests for edge cases."""

    def test_fit_with_single_feature(self):
        """Test fitting with single feature."""
        np.random.seed(42)
        X = np.random.randn(50, 1)

        filter_obj = XOutlierFilter(method="mahalanobis")
        filter_obj.fit(X)
        mask = filter_obj.get_mask(X)

        assert len(mask) == 50

    def test_fit_with_1d_input(self):
        """Test fitting with 1D input (auto-reshaped)."""
        np.random.seed(42)
        X = np.random.randn(50)

        filter_obj = XOutlierFilter(method="mahalanobis")
        filter_obj.fit(X)
        mask = filter_obj.get_mask(X)

        assert len(mask) == 50

    def test_get_mask_before_fit_raises_error(self):
        """Test that get_mask before fit raises ValueError."""
        X = np.random.randn(10, 5)
        filter_obj = XOutlierFilter()

        with pytest.raises(ValueError, match="has not been fitted"):
            filter_obj.get_mask(X)

    def test_insufficient_samples_raises_error(self):
        """Test that insufficient samples raises ValueError."""
        X = np.random.randn(1, 5)  # Only 1 sample
        filter_obj = XOutlierFilter()

        with pytest.raises(ValueError, match="at least 2 samples"):
            filter_obj.fit(X)


class TestXOutlierFilterHelperMethods:
    """Tests for helper methods."""

    def test_get_excluded_indices(self):
        """Test get_excluded_indices returns correct indices."""
        np.random.seed(42)
        X = np.random.randn(99, 5)
        outlier = np.array([[15, 15, 15, 15, 15]])
        X_with_outlier = np.vstack([X, outlier])

        filter_obj = XOutlierFilter(method="mahalanobis", threshold=3.0)
        filter_obj.fit(X_with_outlier)
        excluded = filter_obj.get_excluded_indices(X_with_outlier)

        assert 99 in excluded  # Outlier index

    def test_get_filter_stats(self):
        """Test get_filter_stats returns expected statistics."""
        np.random.seed(42)
        X = np.random.randn(50, 5)

        filter_obj = XOutlierFilter(method="mahalanobis", threshold=3.0)
        filter_obj.fit(X)
        stats = filter_obj.get_filter_stats(X)

        assert stats["n_samples"] == 50
        assert stats["method"] == "mahalanobis"
        assert stats["threshold"] == 3.0
        assert "distance_stats" in stats

    def test_exclusion_reason_default(self):
        """Test default exclusion reason."""
        filter_obj = XOutlierFilter(method="pca_residual")
        assert filter_obj.exclusion_reason == "x_outlier_pca_residual"

    def test_exclusion_reason_custom(self):
        """Test custom exclusion reason."""
        filter_obj = XOutlierFilter(reason="my_reason")
        assert filter_obj.exclusion_reason == "my_reason"

    def test_repr(self):
        """Test string representation."""
        filter_obj = XOutlierFilter(method="mahalanobis", threshold=3.0)
        repr_str = repr(filter_obj)
        assert "XOutlierFilter" in repr_str
        assert "mahalanobis" in repr_str
        assert "3.0" in repr_str


class TestXOutlierFilterTransform:
    """Tests for transform method (should be no-op)."""

    def test_transform_returns_input_unchanged(self):
        """Test that transform returns input unchanged."""
        np.random.seed(42)
        X = np.random.randn(10, 5)

        filter_obj = XOutlierFilter()
        filter_obj.fit(X)
        X_transformed = filter_obj.transform(X)

        np.testing.assert_array_equal(X, X_transformed)
