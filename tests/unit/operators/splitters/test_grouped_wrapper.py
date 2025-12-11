"""Tests for GroupedSplitterWrapper."""

import numpy as np
import pytest
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

from nirs4all.operators.splitters import GroupedSplitterWrapper


class TestGroupedSplitterWrapper:
    """Test suite for GroupedSplitterWrapper."""

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

    @pytest.fixture
    def sample_classification_grouped_data(self):
        """Generate sample classification data with groups."""
        np.random.seed(42)
        n_groups = 20
        samples_per_group = 5
        n_samples = n_groups * samples_per_group
        n_features = 50

        X = np.random.randn(n_samples, n_features)
        # Assign each group to one class (groups 0-9 -> class 0, 10-19 -> class 1)
        y = np.repeat(np.arange(n_groups) // 10, samples_per_group)
        groups = np.repeat(np.arange(n_groups), samples_per_group)
        return X, y, groups

    @pytest.fixture
    def unequal_grouped_data(self):
        """Generate data with unequal group sizes."""
        np.random.seed(42)
        n_features = 50

        # Groups with different sizes: 2, 5, 3, 7, 1, 4
        group_sizes = [2, 5, 3, 7, 1, 4]
        n_samples = sum(group_sizes)

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        groups = np.concatenate([np.full(size, i) for i, size in enumerate(group_sizes)])
        return X, y, groups

    # --- KFold Tests ---

    def test_kfold_respects_groups(self, sample_grouped_data):
        """Test that KFold respects groups through wrapper - no group split."""
        X, y, groups = sample_grouped_data

        wrapper = GroupedSplitterWrapper(KFold(n_splits=5))

        for train_idx, test_idx in wrapper.split(X, y, groups=groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])

            # No group should appear in both train and test
            assert len(train_groups & test_groups) == 0, (
                "Groups should not overlap between train and test"
            )

    def test_kfold_all_samples_used(self, sample_grouped_data):
        """Test that all samples are used exactly once across test folds."""
        X, y, groups = sample_grouped_data
        n_splits = 5

        wrapper = GroupedSplitterWrapper(KFold(n_splits=n_splits))
        folds = list(wrapper.split(X, y, groups=groups))

        assert len(folds) == n_splits

        # All samples should appear exactly once in test sets
        all_test_indices = np.concatenate([test for _, test in folds])
        unique_test_indices = np.unique(all_test_indices)
        assert len(unique_test_indices) == len(X)

    def test_kfold_no_train_test_overlap(self, sample_grouped_data):
        """Test no sample appears in both train and test for any fold."""
        X, y, groups = sample_grouped_data

        wrapper = GroupedSplitterWrapper(KFold(n_splits=5))

        for train_idx, test_idx in wrapper.split(X, y, groups=groups):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, "Train and test indices should not overlap"

    # --- ShuffleSplit Tests ---

    def test_shuffle_split_respects_groups(self, sample_grouped_data):
        """Test that ShuffleSplit respects groups through wrapper."""
        X, y, groups = sample_grouped_data

        wrapper = GroupedSplitterWrapper(
            ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        )

        for train_idx, test_idx in wrapper.split(X, y, groups=groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])

            # No group should appear in both train and test
            assert len(train_groups & test_groups) == 0

    def test_shuffle_split_test_size(self, sample_grouped_data):
        """Test that ShuffleSplit produces approximately correct test size."""
        X, y, groups = sample_grouped_data
        n_groups = len(np.unique(groups))
        test_size = 0.2

        wrapper = GroupedSplitterWrapper(
            ShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        )

        for train_idx, test_idx in wrapper.split(X, y, groups=groups):
            test_groups = len(set(groups[test_idx]))
            expected_test_groups = int(n_groups * test_size)

            # Allow some tolerance due to rounding
            assert abs(test_groups - expected_test_groups) <= 1

    # --- StratifiedKFold Tests ---

    def test_stratified_kfold_respects_groups(self, sample_classification_grouped_data):
        """Test StratifiedKFold respects groups through wrapper."""
        X, y, groups = sample_classification_grouped_data

        wrapper = GroupedSplitterWrapper(
            StratifiedKFold(n_splits=3, shuffle=False),
            y_aggregation="mode"
        )

        for train_idx, test_idx in wrapper.split(X, y, groups=groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])

            # No group should appear in both train and test
            assert len(train_groups & test_groups) == 0

    def test_stratified_kfold_class_distribution(self, sample_classification_grouped_data):
        """Test StratifiedKFold maintains class distribution in folds."""
        X, y, groups = sample_classification_grouped_data

        wrapper = GroupedSplitterWrapper(
            StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
            y_aggregation="mode"
        )

        for train_idx, test_idx in wrapper.split(X, y, groups=groups):
            # Each fold should have samples from both classes
            test_classes = set(y[test_idx])
            assert len(test_classes) == 2, "Test fold should contain both classes"

    # --- Aggregation Method Tests ---

    def test_mean_aggregation(self, sample_grouped_data):
        """Test mean aggregation for X values."""
        X, y, groups = sample_grouped_data

        wrapper = GroupedSplitterWrapper(KFold(n_splits=5), aggregation="mean")

        # Just verify it runs without error and produces valid splits
        folds = list(wrapper.split(X, y, groups=groups))
        assert len(folds) == 5

    def test_median_aggregation(self, sample_grouped_data):
        """Test median aggregation for X values."""
        X, y, groups = sample_grouped_data

        wrapper = GroupedSplitterWrapper(KFold(n_splits=5), aggregation="median")

        folds = list(wrapper.split(X, y, groups=groups))
        assert len(folds) == 5

    def test_first_aggregation(self, sample_grouped_data):
        """Test first aggregation for X values."""
        X, y, groups = sample_grouped_data

        wrapper = GroupedSplitterWrapper(KFold(n_splits=5), aggregation="first")

        folds = list(wrapper.split(X, y, groups=groups))
        assert len(folds) == 5

    def test_y_aggregation_mode(self, sample_classification_grouped_data):
        """Test mode aggregation for y values (classification)."""
        X, y, groups = sample_classification_grouped_data

        wrapper = GroupedSplitterWrapper(
            StratifiedKFold(n_splits=2),
            y_aggregation="mode"
        )

        folds = list(wrapper.split(X, y, groups=groups))
        assert len(folds) == 2

    def test_y_aggregation_mean(self, sample_grouped_data):
        """Test mean aggregation for y values (regression)."""
        X, y, groups = sample_grouped_data

        wrapper = GroupedSplitterWrapper(KFold(n_splits=5), y_aggregation="mean")

        folds = list(wrapper.split(X, y, groups=groups))
        assert len(folds) == 5

    def test_y_aggregation_first(self, sample_grouped_data):
        """Test first aggregation for y values."""
        X, y, groups = sample_grouped_data

        wrapper = GroupedSplitterWrapper(KFold(n_splits=5), y_aggregation="first")

        folds = list(wrapper.split(X, y, groups=groups))
        assert len(folds) == 5

    # --- Unequal Group Sizes ---

    def test_unequal_group_sizes(self, unequal_grouped_data):
        """Test wrapper handles unequal group sizes correctly."""
        X, y, groups = unequal_grouped_data
        n_groups = len(np.unique(groups))

        wrapper = GroupedSplitterWrapper(KFold(n_splits=min(3, n_groups)))

        for train_idx, test_idx in wrapper.split(X, y, groups=groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])

            # No group overlap
            assert len(train_groups & test_groups) == 0

            # All train indices belong to train groups
            for idx in train_idx:
                assert groups[idx] in train_groups

            # All test indices belong to test groups
            for idx in test_idx:
                assert groups[idx] in test_groups

    # --- Backward Compatibility (No Groups) ---

    def test_no_groups_delegates_to_splitter(self, sample_grouped_data):
        """Test wrapper is transparent when no groups provided."""
        X, y, _ = sample_grouped_data

        inner_splitter = KFold(n_splits=5, shuffle=False)
        wrapper = GroupedSplitterWrapper(inner_splitter)

        # Get splits from wrapper (without groups)
        wrapper_folds = list(wrapper.split(X, y))

        # Get splits from original splitter
        original_folds = list(inner_splitter.split(X, y))

        # Should be identical
        assert len(wrapper_folds) == len(original_folds)
        for (w_train, w_test), (o_train, o_test) in zip(wrapper_folds, original_folds):
            np.testing.assert_array_equal(w_train, o_train)
            np.testing.assert_array_equal(w_test, o_test)

    # --- get_n_splits Tests ---

    def test_get_n_splits(self, sample_grouped_data):
        """Test get_n_splits returns inner splitter's n_splits."""
        X, y, groups = sample_grouped_data

        wrapper = GroupedSplitterWrapper(KFold(n_splits=5))
        assert wrapper.get_n_splits(X, y, groups) == 5

        wrapper = GroupedSplitterWrapper(ShuffleSplit(n_splits=10))
        assert wrapper.get_n_splits() == 10

    # --- Validation Tests ---

    def test_invalid_aggregation_raises(self):
        """Test that invalid aggregation parameter raises ValueError."""
        with pytest.raises(ValueError, match="aggregation must be one of"):
            GroupedSplitterWrapper(KFold(n_splits=5), aggregation="invalid")

    def test_invalid_y_aggregation_raises(self):
        """Test that invalid y_aggregation parameter raises ValueError."""
        with pytest.raises(ValueError, match="y_aggregation must be one of"):
            GroupedSplitterWrapper(KFold(n_splits=5), y_aggregation="invalid")

    # --- Repr Test ---

    def test_repr(self):
        """Test string representation of wrapper."""
        wrapper = GroupedSplitterWrapper(
            KFold(n_splits=5),
            aggregation="median",
            y_aggregation="mode"
        )

        repr_str = repr(wrapper)
        assert "GroupedSplitterWrapper" in repr_str
        assert "KFold" in repr_str
        assert "median" in repr_str
        assert "mode" in repr_str

    # --- Y None Test ---

    def test_split_without_y(self, sample_grouped_data):
        """Test splitting works when y is None."""
        X, _, groups = sample_grouped_data

        wrapper = GroupedSplitterWrapper(KFold(n_splits=5))

        folds = list(wrapper.split(X, y=None, groups=groups))
        assert len(folds) == 5

        for train_idx, test_idx in folds:
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert len(train_groups & test_groups) == 0

    # --- Inferred Y Aggregation Test ---

    def test_infer_y_aggregation_stratified(self):
        """Test that y_aggregation is inferred as 'mode' for Stratified splitters."""
        wrapper = GroupedSplitterWrapper(StratifiedKFold(n_splits=3))
        assert wrapper._infer_y_aggregation() == "mode"

        wrapper = GroupedSplitterWrapper(StratifiedShuffleSplit(n_splits=3))
        assert wrapper._infer_y_aggregation() == "mode"

    def test_infer_y_aggregation_non_stratified(self):
        """Test that y_aggregation is inferred as 'mean' for non-Stratified splitters."""
        wrapper = GroupedSplitterWrapper(KFold(n_splits=3))
        assert wrapper._infer_y_aggregation() == "mean"

        wrapper = GroupedSplitterWrapper(ShuffleSplit(n_splits=3))
        assert wrapper._infer_y_aggregation() == "mean"


class TestYBinning:
    """Tests for y-binning functionality used with force_group='y'."""

    @pytest.fixture
    def controller(self):
        """Create CrossValidatorController instance for testing _bin_y_for_groups."""
        from nirs4all.controllers.splitters.split import CrossValidatorController
        return CrossValidatorController()

    def test_bin_y_basic(self, controller):
        """Test basic y-binning into quantile groups."""
        np.random.seed(42)
        y = np.random.randn(100)

        groups = controller._bin_y_for_groups(y, n_bins=5)

        assert len(groups) == len(y)
        assert len(np.unique(groups)) <= 5
        # All samples should have a bin assignment
        assert not np.any(np.isnan(groups))

    def test_bin_y_balanced_distribution(self, controller):
        """Test that y-binning creates approximately balanced bins."""
        np.random.seed(42)
        y = np.random.randn(100)

        groups = controller._bin_y_for_groups(y, n_bins=5)

        # Count samples per bin
        unique_bins, counts = np.unique(groups, return_counts=True)

        # Each bin should have approximately 20 samples (100/5)
        # Allow some tolerance due to quantile boundaries
        for count in counts:
            assert 10 <= count <= 30, f"Bin count {count} is too unbalanced"

    def test_bin_y_preserves_ordering(self, controller):
        """Test that lower y values go to lower bin numbers."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        groups = controller._bin_y_for_groups(y, n_bins=5)

        # Lower y values should have lower or equal bin indices
        for i in range(len(y) - 1):
            if y[i] < y[i + 1]:
                assert groups[i] <= groups[i + 1], \
                    f"y[{i}]={y[i]} (bin {groups[i]}) should have bin <= y[{i+1}]={y[i+1]} (bin {groups[i+1]})"

    def test_bin_y_few_unique_values(self, controller):
        """Test y-binning when fewer unique y values than bins."""
        y = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])  # Only 3 unique values

        groups = controller._bin_y_for_groups(y, n_bins=10)

        # Should fall back to unique value binning
        assert len(np.unique(groups)) == 3
        # Same y values should be in same bin
        assert groups[0] == groups[1]  # Both y=1.0
        assert groups[2] == groups[3]  # Both y=2.0
        assert groups[4] == groups[5]  # Both y=3.0

    def test_bin_y_identical_values(self, controller):
        """Test y-binning when all y values are identical."""
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        groups = controller._bin_y_for_groups(y, n_bins=5)

        # All should be in the same bin
        assert len(np.unique(groups)) == 1

    def test_bin_y_different_nbins(self, controller):
        """Test y-binning with different number of bins."""
        np.random.seed(42)
        y = np.random.randn(100)

        for n_bins in [2, 3, 5, 10]:
            groups = controller._bin_y_for_groups(y, n_bins=n_bins)
            n_unique = len(np.unique(groups))
            assert n_unique <= n_bins, f"Got {n_unique} bins but requested {n_bins}"
            assert n_unique >= 2, "Should have at least 2 bins"

    def test_bin_y_skewed_distribution(self, controller):
        """Test y-binning with skewed y distribution."""
        np.random.seed(42)
        # Create heavily skewed distribution
        y = np.abs(np.random.randn(100)) ** 3

        groups = controller._bin_y_for_groups(y, n_bins=5)

        # Should still create bins (quantile-based should handle skew)
        assert len(np.unique(groups)) >= 2

    def test_bin_y_with_kfold_wrapper(self):
        """Test that y-binned groups work with GroupedSplitterWrapper."""
        from nirs4all.controllers.splitters.split import CrossValidatorController

        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100)

        controller = CrossValidatorController()
        groups = controller._bin_y_for_groups(y, n_bins=10)

        # Use wrapper with binned groups
        wrapper = GroupedSplitterWrapper(KFold(n_splits=5))

        for train_idx, test_idx in wrapper.split(X, y, groups=groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])

            # No group (bin) should appear in both train and test
            assert len(train_groups & test_groups) == 0, \
                "Y-bins should not overlap between train and test"

    def test_bin_y_with_stratified_wrapper(self):
        """Test y-binning with KFold to verify balanced y distribution in folds.

        Note: StratifiedKFold with force_group='y' is primarily useful when
        you have many more samples than bins, so that after aggregation there
        are still multiple groups per bin for stratification. This test uses
        KFold and verifies that y-value distribution is balanced across folds.
        """
        from nirs4all.controllers.splitters.split import CrossValidatorController

        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randn(200)

        controller = CrossValidatorController()
        n_bins = 20  # More bins = smaller groups, allows proper CV
        groups = controller._bin_y_for_groups(y, n_bins=n_bins)

        # Use KFold with binned groups for balanced y distribution
        wrapper = GroupedSplitterWrapper(
            KFold(n_splits=5, shuffle=True, random_state=42)
        )

        fold_y_means = []
        for train_idx, test_idx in wrapper.split(X, y, groups=groups):
            # Check train/test separation
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert len(train_groups & test_groups) == 0

            # Record mean y value for each fold
            fold_y_means.append(y[test_idx].mean())

        # With y-binning, the y means across folds should be balanced
        # (groups with similar y values stay together but are distributed)
        y_mean_range = max(fold_y_means) - min(fold_y_means)
        # The range should be smaller than overall y std
        assert y_mean_range < 2 * np.std(y), \
            f"Fold y means range ({y_mean_range:.3f}) should be less than 2*std ({2*np.std(y):.3f})"


class TestPerformanceBenchmark:
    """Performance benchmarks comparing force_group wrapper vs native GroupKFold.

    These tests verify that the GroupedSplitterWrapper provides similar
    performance to native group splitters while offering more flexibility.
    """

    @pytest.fixture
    def large_grouped_data(self):
        """Generate larger dataset for performance testing."""
        np.random.seed(42)
        n_groups = 100
        samples_per_group = 10
        n_samples = n_groups * samples_per_group
        n_features = 100

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        groups = np.repeat(np.arange(n_groups), samples_per_group)
        return X, y, groups

    def test_wrapper_vs_native_groupkfold_results(self, large_grouped_data):
        """Test that wrapper produces same fold structure as native GroupKFold.

        Both approaches should produce the same number of folds with the same
        group separation (no groups split across train/test).
        """
        from sklearn.model_selection import GroupKFold as NativeGroupKFold

        X, y, groups = large_grouped_data
        n_splits = 5

        # Native GroupKFold
        native = NativeGroupKFold(n_splits=n_splits)
        native_folds = list(native.split(X, y, groups=groups))

        # Wrapper with KFold
        wrapper = GroupedSplitterWrapper(KFold(n_splits=n_splits))
        wrapper_folds = list(wrapper.split(X, y, groups=groups))

        # Same number of folds
        assert len(native_folds) == len(wrapper_folds)

        # Both should respect group boundaries
        for train_idx, test_idx in native_folds:
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert len(train_groups & test_groups) == 0, "Native: groups overlap"

        for train_idx, test_idx in wrapper_folds:
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert len(train_groups & test_groups) == 0, "Wrapper: groups overlap"

    def test_wrapper_performance_timing(self, large_grouped_data):
        """Benchmark wrapper performance vs native GroupKFold.

        This test verifies that the wrapper overhead is acceptable.
        The wrapper may be slightly slower due to aggregation step.
        """
        import time
        from sklearn.model_selection import GroupKFold as NativeGroupKFold

        X, y, groups = large_grouped_data
        n_splits = 5
        n_iterations = 10

        # Time native GroupKFold
        native = NativeGroupKFold(n_splits=n_splits)
        start_native = time.perf_counter()
        for _ in range(n_iterations):
            list(native.split(X, y, groups=groups))
        time_native = time.perf_counter() - start_native

        # Time wrapper with KFold
        wrapper = GroupedSplitterWrapper(KFold(n_splits=n_splits))
        start_wrapper = time.perf_counter()
        for _ in range(n_iterations):
            list(wrapper.split(X, y, groups=groups))
        time_wrapper = time.perf_counter() - start_wrapper

        # Calculate overhead ratio
        overhead_ratio = time_wrapper / time_native if time_native > 0 else float('inf')

        # Log performance for visibility
        print(f"\n  Performance benchmark ({n_iterations} iterations):")
        print(f"    Native GroupKFold: {time_native * 1000:.2f} ms")
        print(f"    Wrapped KFold:     {time_wrapper * 1000:.2f} ms")
        print(f"    Overhead ratio:    {overhead_ratio:.2f}x")

        # Wrapper should not be more than 10x slower
        # (aggregation adds overhead but should still be fast)
        assert overhead_ratio < 10, \
            f"Wrapper is too slow ({overhead_ratio:.2f}x slower than native)"

    def test_wrapper_memory_efficiency(self, large_grouped_data):
        """Test that wrapper doesn't create excessive memory copies."""
        import sys

        X, y, groups = large_grouped_data
        n_splits = 5

        # Get baseline memory for data
        data_size = sys.getsizeof(X) + sys.getsizeof(y) + sys.getsizeof(groups)

        # Create wrapper and run split
        wrapper = GroupedSplitterWrapper(KFold(n_splits=n_splits))

        # The wrapper should work without crashing (memory test)
        folds = list(wrapper.split(X, y, groups=groups))

        # Verify folds are valid numpy arrays (not excessive copies)
        for train_idx, test_idx in folds:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            # Index arrays should be much smaller than data
            assert sys.getsizeof(train_idx) < data_size
            assert sys.getsizeof(test_idx) < data_size

    def test_wrapper_with_many_small_groups(self):
        """Test wrapper performance with many small groups (edge case)."""
        np.random.seed(42)
        n_groups = 500
        samples_per_group = 2  # Small groups
        n_samples = n_groups * samples_per_group
        n_features = 50

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        groups = np.repeat(np.arange(n_groups), samples_per_group)

        wrapper = GroupedSplitterWrapper(KFold(n_splits=5))
        folds = list(wrapper.split(X, y, groups=groups))

        assert len(folds) == 5

        # Verify all groups are respected
        for train_idx, test_idx in folds:
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert len(train_groups & test_groups) == 0

    def test_wrapper_with_few_large_groups(self):
        """Test wrapper with few large groups (edge case)."""
        np.random.seed(42)
        n_groups = 6
        samples_per_group = 100  # Large groups
        n_samples = n_groups * samples_per_group
        n_features = 50

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        groups = np.repeat(np.arange(n_groups), samples_per_group)

        wrapper = GroupedSplitterWrapper(KFold(n_splits=3))
        folds = list(wrapper.split(X, y, groups=groups))

        assert len(folds) == 3

        # Verify each fold uses about half the samples
        for train_idx, test_idx in folds:
            assert len(train_idx) > n_samples * 0.5
            assert len(test_idx) > 0

            # Groups are respected
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert len(train_groups & test_groups) == 0
