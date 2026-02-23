"""Tests for nirs4all.analysis.projections."""

import numpy as np
import pytest

from nirs4all.analysis.projections import compute_pca_projection


class TestComputePcaProjection:
    """Tests for compute_pca_projection."""

    def test_basic_shape(self):
        """PCA coordinates have shape (n_samples, n_components)."""
        X = np.random.RandomState(42).randn(50, 20)
        result = compute_pca_projection(X, max_components=5)

        coords = np.array(result["coordinates"])
        assert coords.shape == (50, 5)
        assert len(result["explained_variance_ratio"]) == 5
        assert len(result["explained_variance"]) == 5
        assert result["n_components"] == 5

    def test_caps_at_n_samples(self):
        """Components capped at n_samples when n_samples < max_components."""
        X = np.random.RandomState(42).randn(5, 20)
        result = compute_pca_projection(X, max_components=10)

        assert result["n_components"] == 5
        coords = np.array(result["coordinates"])
        assert coords.shape == (5, 5)

    def test_caps_at_n_features(self):
        """Components capped at n_features when n_features < max_components."""
        X = np.random.RandomState(42).randn(50, 3)
        result = compute_pca_projection(X, max_components=10)

        assert result["n_components"] == 3
        coords = np.array(result["coordinates"])
        assert coords.shape == (50, 3)

    def test_variance_ratio_sums_to_one(self):
        """Explained variance ratios sum to ~1.0 when all components are kept."""
        X = np.random.RandomState(42).randn(30, 5)
        result = compute_pca_projection(X, max_components=5)

        total = sum(result["explained_variance_ratio"])
        assert abs(total - 1.0) < 1e-6

    def test_threshold_at_least_3(self):
        """n_components_threshold is at least 3 (clamped)."""
        X = np.random.RandomState(42).randn(50, 20)
        result = compute_pca_projection(X)

        assert result["n_components_threshold"] >= 3

    def test_threshold_at_most_n_components(self):
        """n_components_threshold never exceeds actual n_components."""
        X = np.random.RandomState(42).randn(50, 20)
        result = compute_pca_projection(X)

        assert result["n_components_threshold"] <= result["n_components"]

    def test_raises_on_single_sample(self):
        """ValueError raised when X has fewer than 2 samples."""
        X = np.random.RandomState(42).randn(1, 10)
        with pytest.raises(ValueError, match="at least 2 samples"):
            compute_pca_projection(X)

    def test_raises_on_zero_features(self):
        """ValueError raised when X has 0 features."""
        X = np.empty((10, 0))
        with pytest.raises(ValueError, match="at least 1 feature"):
            compute_pca_projection(X)

    def test_two_samples(self):
        """Works with exactly 2 samples (edge case)."""
        X = np.random.RandomState(42).randn(2, 50)
        result = compute_pca_projection(X, max_components=10)

        assert result["n_components"] == 2
        coords = np.array(result["coordinates"])
        assert coords.shape == (2, 2)
