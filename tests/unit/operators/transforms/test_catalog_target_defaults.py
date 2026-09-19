import numpy as np
import pytest
from sklearn.base import clone

from nirs4all.operators.transforms.targets import IntegerKBinsDiscretizer, RangeDiscretizer


@pytest.mark.parametrize("bins", [1, 5, [2, 4, 6, 8], []])
def test_range_discretizer_round_trip_has_finite_representatives(bins):
    values = np.linspace(0, 10, 41).reshape(-1, 1)
    operator = clone(RangeDiscretizer(bins)).fit(values)
    labels = operator.transform(values)
    assert np.isfinite(operator.inverse_transform(labels)).all()
    if bins == 5:
        assert len(np.unique(labels)) == 5
        np.testing.assert_allclose(operator.inverse_transform(np.arange(5)), [[1], [3], [5], [7], [9]])


def test_constant_targets_remain_finite():
    operator = RangeDiscretizer().fit(np.full((10, 1), 3.5))
    np.testing.assert_array_equal(operator.inverse_transform([[0]]), [[3.5]])


def test_invalid_predicted_class_is_explicit_and_old_archives_still_work():
    operator = RangeDiscretizer([2, 4]).fit([[0], [6]])
    del operator._centers_
    np.testing.assert_array_equal(operator.inverse_transform([[0], [1], [2]]), [[1], [3], [5]])
    with pytest.raises(ValueError, match="use a classifier"):
        operator.inverse_transform([[3.5]])


def test_integer_bins_honors_set_params_before_fitting():
    operator = IntegerKBinsDiscretizer().set_params(n_bins=2)
    transformed = operator.fit_transform(np.arange(20).reshape(-1, 1))
    assert len(np.unique(transformed)) == 2
