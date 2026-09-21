"""Target inference must precede storage rounding and use only training rows."""

import numpy as np
import pytest

from nirs4all.core.task_type import TaskType
from nirs4all.data.targets import Targets


@pytest.mark.parametrize("offset", [1e-8, -1e-8])
def test_continuous_targets_remain_regression_after_float32_rounding(offset: float) -> None:
    raw = np.asarray([1 + offset, 2 + offset, 1 + 2 * offset, 2 + 2 * offset])
    targets = Targets()
    targets.add_targets(raw)

    assert targets.task_type == TaskType.REGRESSION
    assert targets.get_task_type_for_processing("numeric") == TaskType.REGRESSION
    assert targets.get_task_type_for_processing("raw") == TaskType.REGRESSION
    np.testing.assert_array_equal(targets.get_targets("raw").ravel(), raw)
    np.testing.assert_array_equal(targets.get_targets().ravel(), raw.astype(np.float32))


def test_training_precision_preserves_interleaved_regression_with_new_test_values() -> None:
    raw = np.asarray([1.5, 1.00000001, 2.00000001, 7.25, 1.00000002, 2.00000002])
    targets = Targets()
    targets.add_targets(raw, fit_indices=[1, 2, 4, 5])

    assert targets.task_type == TaskType.REGRESSION
    np.testing.assert_array_equal(targets.get_targets("raw").ravel(), raw)
    np.testing.assert_array_equal(targets.get_targets().ravel(), raw.astype(np.float32))


@pytest.mark.parametrize("forced", [False, True])
def test_large_integer_class_labels_remain_exact_before_numeric_storage(forced: bool) -> None:
    raw = np.asarray([2**63 - 2, 2**63 - 1, 2**63 - 2, 2**63 - 1], dtype=np.int64)
    targets = Targets()
    if forced:
        targets.set_task_type(TaskType.BINARY_CLASSIFICATION)
    targets.add_targets(raw, fit_indices=[0, 1])

    assert targets.task_type == TaskType.BINARY_CLASSIFICATION
    numeric = targets.get_targets()
    np.testing.assert_array_equal(numeric.ravel(), [0, 1, 0, 1])
    np.testing.assert_array_equal(targets.transform_predictions(numeric, "numeric", "raw").ravel(), raw)


def test_explicit_fractional_class_labels_are_encoded_without_rounding() -> None:
    raw = np.asarray([1.00000001, 1.00000002, 1.00000001, 1.00000002])
    targets = Targets()
    targets.set_task_type(TaskType.BINARY_CLASSIFICATION)
    targets.add_targets(raw, fit_indices=[0, 1])

    numeric = targets.get_targets()
    assert targets.task_type == TaskType.BINARY_CLASSIFICATION
    np.testing.assert_array_equal(numeric.ravel(), [0, 1, 0, 1])
    np.testing.assert_array_equal(targets.transform_predictions(numeric, "numeric", "raw").ravel(), raw)
