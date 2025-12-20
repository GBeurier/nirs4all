"""Tests for Predictions.aggregate() method with method and exclude_outliers parameters."""

import pytest
import numpy as np
from nirs4all.data.predictions import Predictions


class TestPredictionsAggregateMethod:
    """Test suite for aggregate method parameter."""

    def test_aggregate_default_method_mean_for_float_data(self):
        """Test that default aggregation method is mean for regression (float data)."""
        # Use clearly float data to avoid auto-detection as classification
        y_pred = np.array([1.1, 2.3, 3.7, 4.9])
        group_ids = np.array(['A', 'A', 'B', 'B'])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids)

        # Mean of [1.1, 2.3] = 1.7, Mean of [3.7, 4.9] = 4.3
        assert result['y_pred'][0] == pytest.approx(1.7)
        assert result['y_pred'][1] == pytest.approx(4.3)

    def test_aggregate_method_mean_explicit(self):
        """Test explicit mean method with float data."""
        y_pred = np.array([1.1, 2.3, 3.7, 4.9])
        group_ids = np.array(['A', 'A', 'B', 'B'])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids, method='mean')

        assert result['y_pred'][0] == pytest.approx(1.7)
        assert result['y_pred'][1] == pytest.approx(4.3)

    def test_aggregate_method_median(self):
        """Test median aggregation method."""
        y_pred = np.array([1.0, 2.0, 10.0,  # Group A: median = 2.0
                          3.0, 4.0, 5.0])    # Group B: median = 4.0
        group_ids = np.array(['A', 'A', 'A', 'B', 'B', 'B'])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids, method='median')

        assert result['y_pred'][0] == pytest.approx(2.0)  # Group A median
        assert result['y_pred'][1] == pytest.approx(4.0)  # Group B median

    def test_aggregate_method_median_even_samples(self):
        """Test median with even number of samples (average of middle two)."""
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])  # 4 samples, 2 groups
        group_ids = np.array(['A', 'A', 'B', 'B'])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids, method='median')

        # Median of [1, 2] = 1.5, Median of [3, 4] = 3.5
        assert result['y_pred'][0] == pytest.approx(1.5)
        assert result['y_pred'][1] == pytest.approx(3.5)

    def test_aggregate_method_vote_for_classification(self):
        """Test majority voting for classification predictions."""
        y_pred = np.array([0, 0, 1,  # Group A: 0 wins (2 vs 1)
                          1, 1, 0])   # Group B: 1 wins (2 vs 1)
        group_ids = np.array(['A', 'A', 'A', 'B', 'B', 'B'])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids, method='vote')

        assert result['y_pred'][0] == 0  # Group A vote
        assert result['y_pred'][1] == 1  # Group B vote

    def test_aggregate_method_vote_tie_breaking(self):
        """Test vote method tie-breaking (scipy.mode returns first mode)."""
        y_pred = np.array([0, 1,  # Group A: tie, first mode returned
                          1, 0])  # Group B: tie
        group_ids = np.array(['A', 'A', 'B', 'B'])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids, method='vote')

        # Both groups have ties, scipy.mode returns first mode
        assert result['y_pred'][0] in [0, 1]
        assert result['y_pred'][1] in [0, 1]

    def test_aggregate_method_affects_y_true(self):
        """Test that method also affects y_true aggregation."""
        y_pred = np.array([1.5, 2.5, 10.5, 3.5, 4.5, 5.5])  # Clearly float data
        y_true = np.array([1.1, 2.1, 9.0, 3.1, 4.1, 5.1])
        group_ids = np.array(['A', 'A', 'A', 'B', 'B', 'B'])

        result_mean = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids,
                                            y_true=y_true, method='mean')
        result_median = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids,
                                              y_true=y_true, method='median')

        # Mean: (1.1 + 2.1 + 9.0) / 3 = 4.0666...
        # Median: 2.1
        assert result_mean['y_true'][0] == pytest.approx(4.0666, rel=1e-2)
        assert result_median['y_true'][0] == pytest.approx(2.1)


class TestPredictionsAggregateExcludeOutliers:
    """Test suite for exclude_outliers parameter with MAD-based detection."""

    def test_exclude_outliers_false_by_default(self):
        """Test that outliers are not excluded by default."""
        y_pred = np.array([1.0, 1.5, 1.3, 100.0])  # 100 is outlier
        group_ids = np.array(['A', 'A', 'A', 'A'])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids)

        # Mean includes outlier: (1 + 1.5 + 1.3 + 100) / 4 = 25.95
        assert result['y_pred'][0] == pytest.approx(25.95)
        assert 'outliers_excluded' not in result

    def test_exclude_outliers_true_removes_outlier(self):
        """Test that exclude_outliers=True removes extreme outliers."""
        y_pred = np.array([1.0, 1.5, 1.3, 100.0])  # 100 is extreme outlier
        group_ids = np.array(['A', 'A', 'A', 'A'])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids, exclude_outliers=True)

        # Mean should exclude 100: (1 + 1.5 + 1.3) / 3 ≈ 1.27
        assert result['y_pred'][0] == pytest.approx(1.2667, rel=1e-2)
        assert 'outliers_excluded' in result
        assert result['outliers_excluded'][0] == 1  # 1 outlier excluded

    def test_exclude_outliers_no_false_positives_normal_data(self):
        """Test that normal data doesn't have false positive outliers."""
        # Data with normal variation
        y_pred = np.array([2.0, 2.5, 2.3, 2.4])
        group_ids = np.array(['A', 'A', 'A', 'A'])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids, exclude_outliers=True)

        # No outliers should be detected
        assert result['outliers_excluded'][0] == 0
        assert result['y_pred'][0] == pytest.approx(2.3)

    def test_exclude_outliers_with_multiple_groups(self):
        """Test outlier exclusion across multiple groups."""
        y_pred = np.array([1.0, 1.5, 1.3, 100.0,  # Group A: 100 is outlier
                          2.0, 2.5, 2.3, 2.4])     # Group B: no outliers
        group_ids = np.array(['A', 'A', 'A', 'A',
                              'B', 'B', 'B', 'B'])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids, exclude_outliers=True)

        # Group A should have outlier excluded
        assert result['outliers_excluded'][0] == 1
        assert result['y_pred'][0] == pytest.approx(1.2667, rel=1e-2)

        # Group B should have no outliers
        assert result['outliers_excluded'][1] == 0
        assert result['y_pred'][1] == pytest.approx(2.3)

    def test_exclude_outliers_needs_minimum_samples(self):
        """Test that outlier detection requires at least 3 samples."""
        # Use clearly float data
        y_pred = np.array([1.5, 100.5])  # Only 2 samples
        group_ids = np.array(['A', 'A'])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids, exclude_outliers=True)

        # With < 3 samples, no outlier detection applied
        assert result['outliers_excluded'][0] == 0
        assert result['y_pred'][0] == pytest.approx(51.0)  # Mean of [1.5, 100.5]

    def test_exclude_outliers_with_median_method(self):
        """Test that exclude_outliers works with median method."""
        y_pred = np.array([1.0, 1.5, 1.3, 100.0, 1.2])
        group_ids = np.array(['A', 'A', 'A', 'A', 'A'])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids,
                                       method='median', exclude_outliers=True)

        # Outlier excluded, median of [1.0, 1.5, 1.3, 1.2] = 1.25
        assert result['outliers_excluded'][0] == 1
        assert result['y_pred'][0] == pytest.approx(1.25)

    def test_exclude_outliers_y_true_also_filtered(self):
        """Test that y_true is also filtered when excluding outliers."""
        y_pred = np.array([1.0, 1.5, 1.3, 100.0])
        y_true = np.array([1.1, 1.4, 1.2, 99.0])  # 99 corresponds to outlier pred
        group_ids = np.array(['A', 'A', 'A', 'A'])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids,
                                       y_true=y_true, exclude_outliers=True)

        # y_true should exclude the sample corresponding to y_pred outlier
        # Mean of [1.1, 1.4, 1.2] = 1.233...
        assert result['y_true'][0] == pytest.approx(1.2333, rel=1e-2)


class TestPredictionsAggregateEdgeCases:
    """Test edge cases for Predictions.aggregate()."""

    def test_single_sample_per_group(self):
        """Test aggregation with single sample per group."""
        y_pred = np.array([1.0, 2.0, 3.0])
        group_ids = np.array(['A', 'B', 'C'])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids)

        assert len(result['y_pred']) == 3
        assert result['y_pred'][0] == 1.0
        assert result['y_pred'][1] == 2.0
        assert result['y_pred'][2] == 3.0

    def test_all_identical_values_in_group(self):
        """Test aggregation when all values in a group are identical."""
        y_pred = np.array([5.0, 5.0, 5.0])
        group_ids = np.array(['A', 'A', 'A'])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids, method='median')

        assert result['y_pred'][0] == 5.0

    def test_empty_arrays(self):
        """Test aggregation with empty arrays."""
        y_pred = np.array([])
        group_ids = np.array([])

        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids)

        assert len(result['y_pred']) == 0
        assert len(result['group_ids']) == 0
