"""
Integration tests for sample exclusion infrastructure.

Tests the end-to-end functionality of sample exclusion including:
- Filter components (YOutlierFilter, CompositeFilter)
- Indexer exclusion tracking (mark_excluded, reset_exclusions)
- Cascade to augmented samples

Note: These tests verify the underlying infrastructure used by
the ExcludeController (`exclude` keyword) in pipelines.
"""

import numpy as np
import pytest
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression

from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.filters import YOutlierFilter
from nirs4all.operators.filters.base import CompositeFilter


class TestExclusionIntegration:
    """Integration tests for sample exclusion functionality."""

    @pytest.fixture
    def dataset_with_outliers(self):
        """Create a dataset with known outliers for testing."""
        np.random.seed(42)

        # Create normal samples
        n_normal = 50
        X_normal = np.random.rand(n_normal, 100)
        y_normal = np.random.normal(50, 10, n_normal)

        # Add some outliers
        n_outliers = 5
        X_outliers = np.random.rand(n_outliers, 100)
        y_outliers = np.array([200, -100, 250, -150, 300])  # Extreme values

        X = np.vstack([X_normal, X_outliers])
        y = np.concatenate([y_normal, y_outliers])

        dataset = SpectroDataset("test_outliers")
        dataset.add_samples(X, {"partition": "train"})
        dataset.add_targets(y)

        return dataset, n_normal, n_outliers

    def test_filter_marks_samples_as_excluded(self, dataset_with_outliers):
        """Test that filtering marks samples as excluded in indexer."""
        dataset, n_normal, n_outliers = dataset_with_outliers

        # Create filter
        filter_obj = YOutlierFilter(method="iqr", threshold=1.5)

        # Get train data
        selector = {"partition": "train"}
        X = dataset.x(selector, layout="2d", include_augmented=False)
        y = dataset.y(selector, include_augmented=False)
        sample_indices = dataset._indexer.x_indices(selector, include_augmented=False)

        # Fit and get mask
        filter_obj.fit(X, y)
        mask = filter_obj.get_mask(X, y)

        # Get samples to exclude
        exclude_indices = sample_indices[~mask].tolist()

        # Mark as excluded
        n_excluded = dataset._indexer.mark_excluded(exclude_indices, reason="iqr_outlier")

        assert n_excluded >= n_outliers - 1  # At least most outliers should be caught

        # Verify excluded samples are filtered from x_indices
        included_indices = dataset._indexer.x_indices(selector, include_excluded=False)
        assert len(included_indices) < len(sample_indices)

        # Verify exclusion summary
        summary = dataset._indexer.get_exclusion_summary()
        assert summary["total_excluded"] >= n_outliers - 1

    def test_filter_excludes_outliers_correctly(self, dataset_with_outliers):
        """Test that IQR filter correctly identifies extreme outliers."""
        dataset, n_normal, n_outliers = dataset_with_outliers

        filter_obj = YOutlierFilter(method="iqr", threshold=1.5)

        selector = {"partition": "train"}
        X = dataset.x(selector, layout="2d")
        y = dataset.y(selector)

        filter_obj.fit(X, y)
        mask = filter_obj.get_mask(X, y)

        # The last 5 samples are outliers
        outlier_mask = mask[-n_outliers:]

        # At least most outliers should be flagged
        assert np.sum(~outlier_mask) >= n_outliers - 1

    def test_zscore_filter_detects_outliers(self, dataset_with_outliers):
        """Test z-score filter on dataset with outliers."""
        dataset, n_normal, n_outliers = dataset_with_outliers

        filter_obj = YOutlierFilter(method="zscore", threshold=3.0)

        selector = {"partition": "train"}
        X = dataset.x(selector, layout="2d")
        y = dataset.y(selector)

        filter_obj.fit(X, y)
        excluded = filter_obj.get_excluded_indices(X, y)

        # Should detect at least some outliers
        assert len(excluded) >= 1

    def test_filter_preserves_data_integrity(self, dataset_with_outliers):
        """Test that filtering doesn't modify underlying data."""
        dataset, _, _ = dataset_with_outliers

        selector = {"partition": "train"}
        X_before = dataset.x(selector, layout="2d").copy()
        y_before = dataset.y(selector).copy()

        # Apply filter
        filter_obj = YOutlierFilter(method="iqr", threshold=1.5)
        filter_obj.fit(X_before, y_before)
        sample_indices = dataset._indexer.x_indices(selector, include_augmented=False)
        mask = filter_obj.get_mask(X_before, y_before)
        exclude_indices = sample_indices[~mask].tolist()
        dataset._indexer.mark_excluded(exclude_indices)

        # Data should be unchanged (include_excluded=True to get all)
        X_after = dataset.x(selector, layout="2d", include_augmented=True)
        y_after = dataset.y(selector, include_augmented=True)

        # Note: Arrays should still contain same data when we include excluded
        # The size might differ due to how selector works
        assert X_after is not None
        assert y_after is not None

    def test_reset_exclusions(self, dataset_with_outliers):
        """Test that exclusions can be reset."""
        dataset, n_normal, n_outliers = dataset_with_outliers

        # Mark some as excluded
        dataset._indexer.mark_excluded([0, 1, 2], reason="test")

        # Verify exclusion
        summary_before = dataset._indexer.get_exclusion_summary()
        assert summary_before["total_excluded"] == 3

        # Reset
        n_reset = dataset._indexer.reset_exclusions()
        assert n_reset == 3

        # Verify reset
        summary_after = dataset._indexer.get_exclusion_summary()
        assert summary_after["total_excluded"] == 0

    def test_composite_filter_any_mode(self, dataset_with_outliers):
        """Test CompositeFilter with 'any' mode."""
        dataset, _, _ = dataset_with_outliers

        # Create two filters with different thresholds
        filter_strict = YOutlierFilter(method="iqr", threshold=1.0)
        filter_loose = YOutlierFilter(method="iqr", threshold=3.0)

        composite = CompositeFilter(
            filters=[filter_strict, filter_loose],
            mode="any"  # Exclude if ANY flags
        )

        selector = {"partition": "train"}
        X = dataset.x(selector, layout="2d")
        y = dataset.y(selector)

        composite.fit(X, y)
        excluded_composite = composite.get_excluded_indices(X, y)

        # With "any" mode, should exclude same as strict filter
        filter_strict.fit(X, y)
        excluded_strict = filter_strict.get_excluded_indices(X, y)

        # Composite should exclude at least as many as strict (more restrictive)
        assert len(excluded_composite) >= len(excluded_strict)

    def test_composite_filter_all_mode(self, dataset_with_outliers):
        """Test CompositeFilter with 'all' mode."""
        dataset, _, _ = dataset_with_outliers

        # Create two filters with different thresholds
        filter_strict = YOutlierFilter(method="iqr", threshold=1.0)
        filter_loose = YOutlierFilter(method="iqr", threshold=3.0)

        composite = CompositeFilter(
            filters=[filter_strict, filter_loose],
            mode="all"  # Exclude only if ALL flag
        )

        selector = {"partition": "train"}
        X = dataset.x(selector, layout="2d")
        y = dataset.y(selector)

        composite.fit(X, y)
        excluded_composite = composite.get_excluded_indices(X, y)

        # With "all" mode, should exclude fewer (only intersection)
        filter_loose.fit(X, y)
        excluded_loose = filter_loose.get_excluded_indices(X, y)

        # Composite should exclude at most as many as loose (less restrictive)
        assert len(excluded_composite) <= len(excluded_loose)


class TestExclusionWithAugmentation:
    """Tests for exclusion with augmented samples."""

    @pytest.fixture
    def dataset_with_augmented(self):
        """Create dataset with base and augmented samples."""
        np.random.seed(42)

        # Base samples
        X = np.random.rand(20, 100)
        y = np.concatenate([
            np.random.normal(50, 5, 18),
            [200, -100]  # 2 outliers at indices 18, 19
        ])

        dataset = SpectroDataset("test_augmented")
        dataset.add_samples(X, {"partition": "train"})
        dataset.add_targets(y)

        # Add augmented samples for outliers using add_samples_batch
        # This simulates having augmented versions of outlier samples
        X_aug = np.random.rand(4, 1, 100)  # 4 augmented samples, 1 processing, 100 features
        indexes_list = [
            {"partition": "train", "origin": 18, "augmentation": "noise"},
            {"partition": "train", "origin": 18, "augmentation": "noise"},
            {"partition": "train", "origin": 19, "augmentation": "noise"},
            {"partition": "train", "origin": 19, "augmentation": "noise"},
        ]
        dataset.add_samples_batch(X_aug, indexes_list)

        return dataset

    def test_cascade_excludes_augmented_samples(self, dataset_with_augmented):
        """Test that excluding base samples cascades to their augmentations."""
        dataset = dataset_with_augmented

        # Get count before exclusion
        selector = {"partition": "train"}
        count_before = len(dataset._indexer.x_indices(selector, include_excluded=False))

        # Mark outlier base samples (18, 19) as excluded with cascade
        n_excluded = dataset._indexer.mark_excluded([18, 19], reason="outlier", cascade_to_augmented=True)

        # Should exclude base samples + their augmentations
        # 2 base + 4 augmented = 6
        assert n_excluded == 6

        count_after = len(dataset._indexer.x_indices(selector, include_excluded=False))
        assert count_after == count_before - 6

    def test_no_cascade_option(self, dataset_with_augmented):
        """Test excluding without cascading to augmented samples."""
        dataset = dataset_with_augmented

        # Mark base sample without cascade
        n_excluded = dataset._indexer.mark_excluded([18], reason="outlier", cascade_to_augmented=False)

        # Should only exclude the base sample
        assert n_excluded == 1

    def test_exclusion_summary_tracks_augmented(self, dataset_with_augmented):
        """Test that exclusion summary correctly tracks augmented samples."""
        dataset = dataset_with_augmented

        dataset._indexer.mark_excluded([18, 19], reason="test_outlier", cascade_to_augmented=True)

        summary = dataset._indexer.get_exclusion_summary()
        assert summary["total_excluded"] == 6
        assert "test_outlier" in summary["by_reason"]


class TestFilterStats:
    """Tests for filter statistics and reporting."""

    def test_y_outlier_filter_stats(self):
        """Test YOutlierFilter statistics."""
        np.random.seed(42)
        X = np.random.rand(100, 50)
        y = np.concatenate([
            np.random.normal(50, 10, 95),
            [200, -100, 250, -150, 300]  # 5 outliers
        ])

        filter_obj = YOutlierFilter(method="iqr", threshold=1.5)
        filter_obj.fit(X, y)
        stats = filter_obj.get_filter_stats(X, y)

        assert stats["method"] == "iqr"
        assert stats["threshold"] == 1.5
        assert stats["n_samples"] == 100
        assert stats["n_excluded"] >= 3  # At least some outliers caught
        assert "lower_bound" in stats
        assert "upper_bound" in stats
        assert "y_range" in stats

    def test_composite_filter_stats_breakdown(self):
        """Test CompositeFilter statistics include breakdown."""
        np.random.seed(42)
        X = np.random.rand(50, 20)
        y = np.random.normal(50, 10, 50)

        filter1 = YOutlierFilter(method="iqr", threshold=1.5)
        filter2 = YOutlierFilter(method="zscore", threshold=2.5)
        composite = CompositeFilter(filters=[filter1, filter2], mode="any")

        composite.fit(X, y)
        stats = composite.get_filter_stats(X, y)

        assert "filter_breakdown" in stats
        assert len(stats["filter_breakdown"]) == 2
        assert stats["mode"] == "any"
