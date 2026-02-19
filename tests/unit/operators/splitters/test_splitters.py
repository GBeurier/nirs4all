"""Tests for data splitters."""

import numpy as np
import pytest

from nirs4all.operators.splitters import (
    KennardStoneSplitter,
    SPXYFold,
    SPXYGFold,
    SPXYSplitter,
)


class TestSPXYGFold:
    """Test suite for SPXYGFold splitter."""

    @pytest.fixture
    def sample_regression_data(self):
        """Generate sample regression data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        return X, y

    @pytest.fixture
    def sample_classification_data(self):
        """Generate sample classification data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)  # 3 classes
        return X, y

    @pytest.fixture
    def sample_grouped_data(self):
        """Generate sample data with groups."""
        np.random.seed(42)
        n_groups = 20
        samples_per_group = 5
        n_samples = n_groups * samples_per_group
        n_features = 50

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        groups = np.repeat(np.arange(n_groups), samples_per_group)
        return X, y, groups

    # --- Basic K-Fold Tests ---

    def test_kfold_basic(self, sample_regression_data):
        """Test basic K-fold functionality."""
        X, y = sample_regression_data
        n_splits = 5

        splitter = SPXYGFold(n_splits=n_splits)
        folds = list(splitter.split(X, y))

        assert len(folds) == n_splits

        # Check all samples are used
        all_test_indices = np.concatenate([test for _, test in folds])
        assert len(np.unique(all_test_indices)) == len(X)

        # Check no overlap between train and test
        for train, test in folds:
            assert len(set(train) & set(test)) == 0

    def test_kfold_sizes_balanced(self, sample_regression_data):
        """Test that fold sizes are approximately balanced."""
        X, y = sample_regression_data
        n_splits = 5

        splitter = SPXYGFold(n_splits=n_splits)
        folds = list(splitter.split(X, y))

        test_sizes = [len(test) for _, test in folds]
        expected_size = len(X) // n_splits

        # Allow +/- 1 sample difference
        for size in test_sizes:
            assert abs(size - expected_size) <= 1

    def test_get_n_splits(self, sample_regression_data):
        """Test get_n_splits method."""
        X, y = sample_regression_data

        splitter = SPXYGFold(n_splits=5)
        assert splitter.get_n_splits(X, y) == 5

        splitter = SPXYGFold(n_splits=10)
        assert splitter.get_n_splits() == 10

    # --- Single Split Mode (Backward Compatibility) ---

    def test_single_split_mode(self, sample_regression_data):
        """Test single split mode (n_splits=1) for backward compatibility."""
        X, y = sample_regression_data
        test_size = 0.25

        splitter = SPXYGFold(n_splits=1, test_size=test_size)
        folds = list(splitter.split(X, y))

        assert len(folds) == 1
        train, test = folds[0]

        # Check proportions
        expected_test_size = int(len(X) * test_size)
        assert abs(len(test) - expected_test_size) <= 1

        # Check no overlap
        assert len(set(train) & set(test)) == 0

    def test_single_split_default_test_size(self, sample_regression_data):
        """Test that single split defaults to test_size=0.25."""
        X, y = sample_regression_data

        splitter = SPXYGFold(n_splits=1)  # No test_size specified
        folds = list(splitter.split(X, y))

        train, test = folds[0]
        expected_test_size = int(len(X) * 0.25)
        assert abs(len(test) - expected_test_size) <= 1

    # --- Classification Mode ---

    def test_classification_hamming(self, sample_classification_data):
        """Test classification mode with Hamming distance."""
        X, y = sample_classification_data
        n_splits = 5

        splitter = SPXYGFold(n_splits=n_splits, y_metric="hamming")
        folds = list(splitter.split(X, y))

        assert len(folds) == n_splits

        # All samples should be covered
        all_test_indices = np.concatenate([test for _, test in folds])
        assert len(np.unique(all_test_indices)) == len(X)

    def test_classification_preserves_class_distribution(self, sample_classification_data):
        """Test that classification mode produces reasonable class distribution per fold."""
        X, y = sample_classification_data
        n_splits = 5

        splitter = SPXYGFold(n_splits=n_splits, y_metric="hamming")
        folds = list(splitter.split(X, y))

        # Each fold should have samples from multiple classes (not all same class)
        for train, test in folds:
            unique_classes_train = np.unique(y[train])
            unique_classes_test = np.unique(y[test])
            # At least 2 classes in training set
            assert len(unique_classes_train) >= 2

    # --- X-only Mode (Kennard-Stone) ---

    def test_xonly_mode(self, sample_regression_data):
        """Test X-only mode (y_metric=None) - pure Kennard-Stone."""
        X, y = sample_regression_data
        n_splits = 5

        splitter = SPXYGFold(n_splits=n_splits, y_metric=None)
        # Should work without y
        folds = list(splitter.split(X))

        assert len(folds) == n_splits

    def test_y_required_when_y_metric_set(self, sample_regression_data):
        """Test that y is required when y_metric is not None."""
        X, _ = sample_regression_data

        splitter = SPXYGFold(n_splits=5, y_metric="euclidean")

        with pytest.raises(ValueError, match="y is required"):
            list(splitter.split(X))  # No y provided

    # --- Group-Aware Splitting ---

    def test_group_aware_basic(self, sample_grouped_data):
        """Test group-aware splitting keeps groups together."""
        X, y, groups = sample_grouped_data
        n_splits = 4

        splitter = SPXYGFold(n_splits=n_splits)
        folds = list(splitter.split(X, y, groups=groups))

        assert len(folds) == n_splits

        # Check that groups are not split across folds
        for train, test in folds:
            train_groups = set(groups[train])
            test_groups = set(groups[test])
            # No group should appear in both train and test
            assert len(train_groups & test_groups) == 0

    def test_group_aware_all_groups_used(self, sample_grouped_data):
        """Test that all groups appear exactly once in test sets."""
        X, y, groups = sample_grouped_data
        n_splits = 4

        splitter = SPXYGFold(n_splits=n_splits)
        folds = list(splitter.split(X, y, groups=groups))

        all_test_groups = []
        for _, test in folds:
            all_test_groups.extend(groups[test])

        unique_test_groups = np.unique(all_test_groups)
        assert len(unique_test_groups) == len(np.unique(groups))

    def test_group_aggregation_mean(self, sample_grouped_data):
        """Test that mean aggregation is used for groups."""
        X, y, groups = sample_grouped_data

        splitter = SPXYGFold(n_splits=4, aggregation="mean")
        folds = list(splitter.split(X, y, groups=groups))

        # Just verify it runs without error
        assert len(folds) == 4

    def test_group_aggregation_median(self, sample_grouped_data):
        """Test that median aggregation is used for groups."""
        X, y, groups = sample_grouped_data

        splitter = SPXYGFold(n_splits=4, aggregation="median")
        folds = list(splitter.split(X, y, groups=groups))

        # Just verify it runs without error
        assert len(folds) == 4

    # --- PCA Components ---

    def test_pca_components(self, sample_regression_data):
        """Test PCA dimensionality reduction."""
        X, y = sample_regression_data

        splitter = SPXYGFold(n_splits=5, pca_components=10)
        folds = list(splitter.split(X, y))

        assert len(folds) == 5

    # --- Edge Cases ---

    def test_too_many_splits(self, sample_regression_data):
        """Test error when n_splits > n_samples."""
        X, y = sample_regression_data

        splitter = SPXYGFold(n_splits=200)  # More than 100 samples

        with pytest.raises(ValueError, match="Cannot have n_splits"):
            list(splitter.split(X, y))

    def test_invalid_n_splits(self):
        """Test error for invalid n_splits."""
        with pytest.raises(ValueError, match="n_splits must be at least 1"):
            SPXYGFold(n_splits=0)

    def test_invalid_aggregation(self):
        """Test error for invalid aggregation method."""
        with pytest.raises(ValueError, match="aggregation must be"):
            SPXYGFold(aggregation="invalid")

    def test_small_dataset(self):
        """Test with very small dataset."""
        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        splitter = SPXYGFold(n_splits=2)
        folds = list(splitter.split(X, y))

        assert len(folds) == 2
        for train, test in folds:
            assert len(train) + len(test) == 10

    # --- Reproducibility ---

    def test_reproducible_with_random_state(self, sample_regression_data):
        """Test that splits are reproducible with same random_state."""
        X, y = sample_regression_data

        splitter1 = SPXYGFold(n_splits=5, random_state=42)
        splitter2 = SPXYGFold(n_splits=5, random_state=42)

        folds1 = list(splitter1.split(X, y))
        folds2 = list(splitter2.split(X, y))

        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

    def test_single_split_sizes(self, sample_regression_data):
        """Test that single-split mode produces correct proportions."""
        X, y = sample_regression_data
        test_size = 0.3

        splitter = SPXYGFold(n_splits=1, test_size=test_size)
        train, test = next(splitter.split(X, y))

        # Check proportions
        expected_test_size = int(len(X) * test_size)
        assert abs(len(test) - expected_test_size) <= 1
        assert len(train) + len(test) == len(X)

class TestSPXYFold:
    """Test suite for SPXYFold splitter (no group handling)."""

    @pytest.fixture
    def sample_regression_data(self):
        """Generate sample regression data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        return X, y

    @pytest.fixture
    def sample_classification_data(self):
        """Generate sample classification data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)
        return X, y

    # --- Basic K-Fold Tests ---

    def test_kfold_basic(self, sample_regression_data):
        """Test basic K-fold functionality."""
        X, y = sample_regression_data
        n_splits = 5

        splitter = SPXYFold(n_splits=n_splits)
        folds = list(splitter.split(X, y))

        assert len(folds) == n_splits

        all_test_indices = np.concatenate([test for _, test in folds])
        assert len(np.unique(all_test_indices)) == len(X)

        for train, test in folds:
            assert len(set(train) & set(test)) == 0

    def test_kfold_sizes_balanced(self, sample_regression_data):
        """Test that fold sizes are approximately balanced."""
        X, y = sample_regression_data
        n_splits = 5

        splitter = SPXYFold(n_splits=n_splits)
        folds = list(splitter.split(X, y))

        test_sizes = [len(test) for _, test in folds]
        expected_size = len(X) // n_splits

        for size in test_sizes:
            assert abs(size - expected_size) <= 1

    def test_all_samples_in_train_and_test(self, sample_regression_data):
        """Test that train + test covers all samples in each fold."""
        X, y = sample_regression_data

        splitter = SPXYFold(n_splits=5)
        folds = list(splitter.split(X, y))

        for train, test in folds:
            assert len(train) + len(test) == len(X)

    def test_get_n_splits(self):
        """Test get_n_splits method."""
        splitter = SPXYFold(n_splits=5)
        assert splitter.get_n_splits() == 5

        splitter = SPXYFold(n_splits=10)
        assert splitter.get_n_splits() == 10

    # --- Classification Mode ---

    def test_classification_hamming(self, sample_classification_data):
        """Test classification mode with Hamming distance."""
        X, y = sample_classification_data
        n_splits = 5

        splitter = SPXYFold(n_splits=n_splits, y_metric="hamming")
        folds = list(splitter.split(X, y))

        assert len(folds) == n_splits

        all_test_indices = np.concatenate([test for _, test in folds])
        assert len(np.unique(all_test_indices)) == len(X)

    def test_classification_preserves_class_distribution(self, sample_classification_data):
        """Test that classification mode produces reasonable class distribution per fold."""
        X, y = sample_classification_data
        n_splits = 5

        splitter = SPXYFold(n_splits=n_splits, y_metric="hamming")
        folds = list(splitter.split(X, y))

        for train, test in folds:
            unique_classes_train = np.unique(y[train])
            assert len(unique_classes_train) >= 2

    # --- X-only Mode (Kennard-Stone) ---

    def test_xonly_mode(self, sample_regression_data):
        """Test X-only mode (y_metric=None) - pure Kennard-Stone."""
        X, _ = sample_regression_data
        n_splits = 5

        splitter = SPXYFold(n_splits=n_splits, y_metric=None)
        folds = list(splitter.split(X))

        assert len(folds) == n_splits

    def test_y_required_when_y_metric_set(self, sample_regression_data):
        """Test that y is required when y_metric is not None."""
        X, _ = sample_regression_data

        splitter = SPXYFold(n_splits=5, y_metric="euclidean")

        with pytest.raises(ValueError, match="y is required"):
            list(splitter.split(X))

    # --- Groups Parameter Ignored ---

    def test_groups_ignored(self, sample_regression_data):
        """Test that groups parameter is accepted but ignored."""
        X, y = sample_regression_data
        groups = np.repeat(np.arange(20), 5)

        splitter = SPXYFold(n_splits=5)
        folds_with_groups = list(splitter.split(X, y, groups=groups))
        folds_without_groups = list(splitter.split(X, y))

        for (train1, test1), (train2, test2) in zip(folds_with_groups, folds_without_groups):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

    # --- PCA Components ---

    def test_pca_components(self, sample_regression_data):
        """Test PCA dimensionality reduction."""
        X, y = sample_regression_data

        splitter = SPXYFold(n_splits=5, pca_components=10)
        folds = list(splitter.split(X, y))

        assert len(folds) == 5

    # --- Edge Cases ---

    def test_too_many_splits(self, sample_regression_data):
        """Test error when n_splits > n_samples."""
        X, y = sample_regression_data

        splitter = SPXYFold(n_splits=200)

        with pytest.raises(ValueError, match="Cannot have n_splits"):
            list(splitter.split(X, y))

    def test_invalid_n_splits_zero(self):
        """Test error for n_splits=0."""
        with pytest.raises(ValueError, match="n_splits must be at least 2"):
            SPXYFold(n_splits=0)

    def test_invalid_n_splits_one(self):
        """Test error for n_splits=1 (use SPXYSplitter for single split)."""
        with pytest.raises(ValueError, match="n_splits must be at least 2"):
            SPXYFold(n_splits=1)

    def test_small_dataset(self):
        """Test with very small dataset."""
        np.random.seed(42)
        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        splitter = SPXYFold(n_splits=2)
        folds = list(splitter.split(X, y))

        assert len(folds) == 2
        for train, test in folds:
            assert len(train) + len(test) == 10

    # --- Reproducibility ---

    def test_reproducible_with_random_state(self, sample_regression_data):
        """Test that splits are reproducible with same random_state."""
        X, y = sample_regression_data

        splitter1 = SPXYFold(n_splits=5, random_state=42)
        splitter2 = SPXYFold(n_splits=5, random_state=42)

        folds1 = list(splitter1.split(X, y))
        folds2 = list(splitter2.split(X, y))

        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

    def test_deterministic_without_random_state(self, sample_regression_data):
        """Test that splits are deterministic (algorithm is deterministic)."""
        X, y = sample_regression_data

        splitter1 = SPXYFold(n_splits=5)
        splitter2 = SPXYFold(n_splits=5)

        folds1 = list(splitter1.split(X, y))
        folds2 = list(splitter2.split(X, y))

        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)
