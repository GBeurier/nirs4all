"""Integration tests for refactored Targets class."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from nirs4all.data.targets import Targets


class TestTargetsRefactoredIntegration:
    """Integration tests for the refactored Targets class."""

    def test_full_workflow_classification(self):
        """Test complete workflow with classification data."""
        # Create targets with string labels
        labels = np.array(['cat', 'dog', 'bird', 'cat', 'dog'])
        targets = Targets()
        targets.add_targets(labels)

        # Check initial state
        assert targets.num_samples == 5
        assert targets.num_classes == 3
        assert "raw" in targets.processing_ids
        assert "numeric" in targets.processing_ids

        # Get numeric data
        numeric = targets.get_targets("numeric")
        assert numeric.dtype == np.float32
        assert len(np.unique(numeric)) == 3

    def test_full_workflow_regression(self):
        """Test complete workflow with regression data."""
        values = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        targets = Targets()
        targets.add_targets(values)

        # Check initial state
        assert targets.num_samples == 5
        assert "raw" in targets.processing_ids
        assert "numeric" in targets.processing_ids

        # Add scaled processing
        numeric_data = targets.get_targets("numeric")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        targets.add_processed_targets("scaled", scaled_data, ancestor="numeric", transformer=scaler)

        # Check scaled data
        assert "scaled" in targets.processing_ids
        scaled = targets.get_targets("scaled")
        assert np.allclose(scaled.mean(), 0, atol=1e-6)

    def test_backward_compatibility_basic_operations(self):
        """Test that basic operations work as before refactoring."""
        # String labels
        labels = np.array(['A', 'B', 'C', 'A', 'B'])
        targets = Targets()
        targets.add_targets(labels)

        # These should work exactly as before
        assert targets.num_samples == 5
        assert targets.num_classes == 3

        # Get numeric data
        numeric = targets.get_targets("numeric")
        assert numeric.dtype == np.float32
        assert len(numeric) == 5

    def test_multiple_transformations(self):
        """Test chaining multiple transformations."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        targets = Targets()
        targets.add_targets(data)

        # First transformation
        numeric = targets.get_targets("numeric")
        scaler1 = StandardScaler()
        scaled = scaler1.fit_transform(numeric)
        targets.add_processed_targets("scaled", scaled, ancestor="numeric", transformer=scaler1)

        # Second transformation
        scaler2 = StandardScaler()
        normalized = scaler2.fit_transform(scaled)
        targets.add_processed_targets("normalized", normalized, ancestor="scaled", transformer=scaler2)

        # Check all states exist
        assert "numeric" in targets.processing_ids
        assert "scaled" in targets.processing_ids
        assert "normalized" in targets.processing_ids

    def test_predictions_transformation(self):
        """Test that prediction transformation works."""
        labels = np.array(['x', 'y', 'z'])
        targets = Targets()
        targets.add_targets(labels)

        # Transform
        numeric = targets.get_targets("numeric")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric)
        targets.add_processed_targets("scaled", scaled, ancestor="numeric", transformer=scaler)

        # Simulate predictions in scaled space
        predictions = targets.get_targets("scaled").copy()

        # Transform back to numeric
        numeric_preds = targets.transform_predictions(predictions, "scaled", "numeric")

        # Should be close to encoded labels
        assert numeric_preds.dtype == np.float32
        assert len(np.unique(numeric_preds)) <= 3

    def test_add_targets_append(self):
        """Test appending more targets."""
        targets = Targets()
        targets.add_targets(np.array([1, 2, 3]))

        assert targets.num_samples == 3

        # Append more data
        targets.add_targets(np.array([4, 5]))

        assert targets.num_samples == 5

    def test_error_handling_invalid_state(self):
        """Test error handling for invalid states."""
        data = np.array([1, 2, 3])
        targets = Targets()
        targets.add_targets(data)

        with pytest.raises(ValueError):
            targets.get_targets("nonexistent_state")

    def test_copy_behavior(self):
        """Test that data copies work correctly."""
        data = np.array([1, 2, 3, 4, 5])
        targets = Targets()
        targets.add_targets(data)

        # Get data
        retrieved = targets.get_targets("raw")

        # Modify retrieved data
        retrieved[0] = 999

        # Original should be unchanged
        original = targets.get_targets("raw")
        assert original[0] != 999

    def test_edge_case_single_sample(self):
        """Test with single sample."""
        data = np.array([5.0])
        targets = Targets()
        targets.add_targets(data)

        numeric = targets.get_targets("numeric")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric)
        targets.add_processed_targets("scaled", scaled, ancestor="numeric", transformer=scaler)

        # Should handle gracefully
        assert targets.num_samples == 1

    def test_edge_case_empty_array(self):
        """Test with empty array."""
        targets = Targets()

        assert targets.num_samples == 0
        assert targets.num_processings == 0

    def test_multidimensional_targets(self):
        """Test with multi-dimensional target data."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        targets = Targets()
        targets.add_targets(data)

        numeric = targets.get_targets("numeric")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric)
        targets.add_processed_targets("scaled", scaled, ancestor="numeric", transformer=scaler)

        # Check shape preservation
        assert targets.get_targets("scaled").shape == (3, 2)
        assert targets.get_targets("raw").shape == (3, 2)

    def test_num_classes_caching(self):
        """Test that num_classes property works."""
        labels = np.array(['a', 'b', 'c'] * 100)
        targets = Targets()
        targets.add_targets(labels)

        # Access num_classes
        nc = targets.num_classes

        assert nc == 3

    def test_string_representation(self):
        """Test that string representation works."""
        data = np.array([1, 2, 3])
        targets = Targets()
        targets.add_targets(data)

        # Should have a string representation
        str_repr = str(targets)
        assert "Targets" in str_repr

        repr_str = repr(targets)
        assert "Targets" in repr_str

    def test_task_type_update_on_discretization(self):
        """Test that task type is updated when processed targets change the task nature."""
        from nirs4all.operators.transforms.targets import RangeDiscretizer
        from nirs4all.utils.task_type import TaskType

        # Start with regression data
        values = np.array([10.5, 15.2, 22.8, 28.3, 35.1, 42.9, 48.7])
        targets = Targets()
        targets.add_targets(values)

        # Verify it's detected as regression
        assert targets._task_type == TaskType.REGRESSION

        # Apply discretization to convert to classification
        discretizer = RangeDiscretizer([15, 25, 35, 45])
        numeric_data = targets.get_targets("numeric")
        discretized = discretizer.fit_transform(numeric_data.reshape(-1, 1)).ravel()

        # Add discretized targets
        targets.add_processed_targets("discretized", discretized, ancestor="numeric", transformer=discretizer)

        # Task type should now be classification
        assert targets._task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        assert targets.num_classes > 1
