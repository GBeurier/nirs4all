"""Boundary-condition tests for nirs4all.core.task_detection.detect_task_type."""

import numpy as np
import pytest

from nirs4all.core.task_detection import detect_task_type
from nirs4all.core.task_type import TaskType


class TestIntegerTargets:
    """Integer-valued arrays: behaviour driven by n_unique."""

    def test_binary_integer(self):
        y = np.array([0, 1, 0, 1, 1, 0], dtype=float)
        assert detect_task_type(y) == TaskType.BINARY_CLASSIFICATION

    def test_multiclass_integer(self):
        y = np.array([0, 1, 2, 0, 1, 2, 3], dtype=float)
        assert detect_task_type(y) == TaskType.MULTICLASS_CLASSIFICATION

    def test_integer_regression_many_unique(self):
        """101 unique integer values exceed max_classes (100) → REGRESSION."""
        y = np.arange(101, dtype=float)
        assert detect_task_type(y) == TaskType.REGRESSION

    def test_integer_exactly_100_classes(self):
        """100 unique integer values (== max_classes) → MULTICLASS_CLASSIFICATION."""
        y = np.arange(100, dtype=float)
        assert detect_task_type(y) == TaskType.MULTICLASS_CLASSIFICATION

class TestContinuousTargets:
    """Continuous float arrays that do NOT satisfy the integer check."""

    def test_typical_regression_large_range(self):
        rng = np.random.RandomState(0)
        y = rng.uniform(0.0, 100.0, 500)
        assert detect_task_type(y) == TaskType.REGRESSION

    def test_continuous_uniform_01_many_unique(self):
        """Regression targets uniformly spread in [0,1] - many unique values.

        The [0,1] branch is entered, but n_unique >> len(y)*threshold
        so the threshold condition is False → falls through to REGRESSION.
        """
        rng = np.random.RandomState(42)
        y = rng.uniform(0.0, 1.0, 200)
        # 200 uniform samples → effectively all unique; threshold=0.05 → 10 unique needed
        assert detect_task_type(y) == TaskType.REGRESSION

    def test_continuous_01_all_distinct(self):
        """500 distinct floats in [0,1] should always be REGRESSION."""
        rng = np.random.RandomState(7)
        y = rng.uniform(0.0, 1.0, 500)
        assert detect_task_type(y) == TaskType.REGRESSION

class TestBoundaryCondition01WithFewUnique:
    """The specific edge-case from CC-03: values in [0,1] with few unique values.

    This is the problematic region where a continuous regression target with
    few distinct values (e.g., 0.1, 0.2, 0.3, 0.4) may be misclassified.
    Tests document the *current* heuristic behaviour so regressions are caught.
    """

    def test_three_unique_values_in_01_small_sample(self):
        """3 unique floats in [0,1] across 10 samples.

        n_unique=3, len=10, threshold=0.05 → 0.05*10=0.5 → 3 > 0.5 → REGRESSION.
        """
        y = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.5])
        result = detect_task_type(y)
        # 3 unique values, 10 samples → n_unique (3) > 10*0.05 (0.5) → REGRESSION
        assert result == TaskType.REGRESSION

    def test_two_unique_exact_01(self):
        """Only {0.0, 1.0} → BINARY_CLASSIFICATION (special-cased before threshold check)."""
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        assert detect_task_type(y) == TaskType.BINARY_CLASSIFICATION

    def test_few_unique_triggers_classification(self):
        """Verify the threshold condition: n_unique <= len(y)*threshold.

        With threshold=0.05 and 200 samples, a target needs ≤10 unique values
        to be classified.  Use 3 non-{0,1} floats in [0,1] across 200 samples.
        n_unique=3, len=200 → 3 <= 200*0.05=10 → MULTICLASS_CLASSIFICATION.
        This documents the known edge-case where regression targets with few
        distinct values may be misclassified.
        """
        y = np.tile([0.2, 0.5, 0.8], 67)[:200]  # 200 samples, 3 unique floats
        assert len(np.unique(y)) == 3
        # Current heuristic classifies this as multiclass
        result = detect_task_type(y, threshold=0.05)
        assert result == TaskType.MULTICLASS_CLASSIFICATION

    def test_custom_threshold_changes_decision(self):
        """Lower threshold → fewer unique values accepted → borderline → REGRESSION."""
        y = np.tile([0.2, 0.5, 0.8], 67)[:200]  # 200 samples, 3 unique floats
        # With threshold=0.01, 200*0.01=2 → n_unique (3) > 2 → REGRESSION
        result = detect_task_type(y, threshold=0.01)
        assert result == TaskType.REGRESSION

class TestEdgeCases:
    """Additional edge cases and input robustness."""

    def test_all_nan_raises(self):
        y = np.array([np.nan, np.nan, np.nan])
        with pytest.raises(ValueError, match="NaN"):
            detect_task_type(y)

    def test_nan_values_ignored(self):
        """NaN values are stripped; non-NaN values drive detection."""
        y = np.array([0.0, 1.0, np.nan, 0.0, 1.0])
        assert detect_task_type(y) == TaskType.BINARY_CLASSIFICATION

    def test_single_class_integer(self):
        """Only one unique integer value: n_unique=1 is not 2 and not in (2,100] → REGRESSION."""
        y = np.array([5.0, 5.0, 5.0])
        # n_unique=1 → not binary (!=2), not multi (not >2), falls to else → REGRESSION
        result = detect_task_type(y)
        assert result == TaskType.REGRESSION

    def test_2d_input_flattened(self):
        """2D arrays are flattened before analysis."""
        y = np.array([[0.0], [1.0], [0.0], [1.0]])
        assert detect_task_type(y) == TaskType.BINARY_CLASSIFICATION

    def test_negative_values_not_in_01_branch(self):
        """Negative values skip the [0,1] check and default to REGRESSION."""
        y = np.array([-1.0, 0.5, 1.5, -0.5, 0.0])
        assert detect_task_type(y) == TaskType.REGRESSION

    def test_values_above_1_not_in_01_branch(self):
        """Any value > 1 exits the [0,1] branch → REGRESSION."""
        y = np.array([0.1, 0.5, 0.9, 1.1])
        assert detect_task_type(y) == TaskType.REGRESSION
