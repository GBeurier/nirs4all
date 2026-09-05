"""Explicit regression preserves measurements, even when they resemble labels."""

import numpy as np
import pytest

from nirs4all.core.task_type import TaskType
from nirs4all.data.targets import Targets


@pytest.mark.parametrize("before", [True, False])
@pytest.mark.parametrize("values", [np.array([101., 102., 103.]), np.array([[101, 201], [102, 202], [103, 203]]), np.array(["101", "102", "103"]), np.array(["0", "1", "2"])])
def test_forced_regression_preserves_raw_numeric_values_before_or_after_ingestion(before, values):
    targets = Targets()
    if before:
        targets.set_task_type(TaskType.REGRESSION)
    targets.add_targets(values)
    if not before:
        targets.set_task_type(TaskType.REGRESSION)
    expected = values.astype(np.float32).reshape(3, -1)
    np.testing.assert_array_equal(targets.get_targets(), expected)
    np.testing.assert_array_equal(targets.transform_predictions(expected, "numeric", "raw"), expected)
    assert targets.task_type == TaskType.REGRESSION
    assert targets.task_type_forced
    assert targets.get_task_type_for_processing("numeric") == TaskType.REGRESSION
    targets.add_targets(values)
    np.testing.assert_array_equal(targets.get_targets(), np.vstack([expected, expected]))


def test_auto_classification_still_encodes_nonzero_labels():
    targets = Targets()
    targets.add_targets([101, 102, 101])
    np.testing.assert_array_equal(targets.get_targets().ravel(), [0, 1, 0])
    assert targets.task_type.is_classification


def test_regression_cannot_silently_reinterpret_already_processed_label_codes():
    from sklearn.preprocessing import StandardScaler

    targets = Targets()
    targets.add_targets([101, 102, 103])
    scaler = StandardScaler()
    targets.add_processed_targets("scaled", scaler.fit_transform(targets.get_targets()), transformer=scaler)
    before = targets.get_targets()
    with pytest.raises(ValueError, match="before target preprocessing"):
        targets.set_task_type(TaskType.REGRESSION)
    np.testing.assert_array_equal(targets.get_targets(), before)
    assert not targets.task_type_forced


def test_invalid_explicit_regression_input_leaves_target_storage_empty():
    targets = Targets()
    targets.set_task_type(TaskType.REGRESSION)
    with pytest.raises(ValueError, match="numeric values"):
        targets.add_targets(["cat", "dog"])
    assert targets.num_samples == 0
    assert targets.processing_ids == []
