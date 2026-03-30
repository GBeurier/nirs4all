"""Tests for task-type-aware chart behavior in BaseChart."""

from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure

from nirs4all.visualization.charts.base import BaseChart


class ConcreteChart(BaseChart):
    """Minimal concrete chart for testing base class methods."""

    def render(self, *args, **kwargs):
        return MagicMock(spec=Figure)

    def validate_inputs(self, *args, **kwargs):
        pass


def _make_predictions_mock(task_types: list[str]):
    """Create a mock Predictions object with given task types."""
    mock = MagicMock()
    mock.num_predictions = len(task_types)
    mock.get_unique_values.return_value = task_types
    return mock


class TestGetDefaultMetric:
    def test_regression_only(self):
        preds = _make_predictions_mock(["regression"])
        chart = ConcreteChart(preds)
        assert chart._get_default_metric() == "rmse"

    def test_classification_only(self):
        preds = _make_predictions_mock(["binary_classification"])
        chart = ConcreteChart(preds)
        assert chart._get_default_metric() == "balanced_accuracy"

    def test_mixed_defaults_to_rmse(self):
        preds = _make_predictions_mock(["regression", "binary_classification"])
        chart = ConcreteChart(preds)
        assert chart._get_default_metric() == "rmse"

    def test_with_classification_filter(self):
        preds = _make_predictions_mock(["regression", "binary_classification"])
        chart = ConcreteChart(preds)
        assert chart._get_default_metric(task_type_filter="classification") == "balanced_accuracy"

    def test_with_regression_filter(self):
        preds = _make_predictions_mock(["regression", "binary_classification"])
        chart = ConcreteChart(preds)
        assert chart._get_default_metric(task_type_filter="regression") == "rmse"

    def test_with_clf_alias(self):
        preds = _make_predictions_mock([])
        chart = ConcreteChart(preds)
        assert chart._get_default_metric(task_type_filter="clf") == "balanced_accuracy"

    def test_with_reg_alias(self):
        preds = _make_predictions_mock([])
        chart = ConcreteChart(preds)
        assert chart._get_default_metric(task_type_filter="reg") == "rmse"

    def test_multiclass_only(self):
        preds = _make_predictions_mock(["multiclass_classification"])
        chart = ConcreteChart(preds)
        assert chart._get_default_metric() == "balanced_accuracy"

    def test_empty_predictions(self):
        preds = _make_predictions_mock([])
        preds.num_predictions = 0
        chart = ConcreteChart(preds)
        assert chart._get_default_metric() == "rmse"


class TestGetUniqueTaskTypes:
    def test_returns_types(self):
        preds = _make_predictions_mock(["regression", "binary_classification"])
        chart = ConcreteChart(preds)
        result = chart._get_unique_task_types()
        assert set(result) == {"regression", "binary_classification"}

    def test_filters_none(self):
        mock = MagicMock()
        mock.get_unique_values.return_value = ["regression", None, "binary_classification"]
        chart = ConcreteChart(mock)
        result = chart._get_unique_task_types()
        assert None not in result
        assert len(result) == 2

    def test_handles_exception(self):
        mock = MagicMock()
        mock.get_unique_values.side_effect = Exception("no data")
        chart = ConcreteChart(mock)
        assert chart._get_unique_task_types() == []
