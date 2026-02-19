"""Tests for metrics defaults and new metrics."""

import numpy as np
import pytest

from nirs4all.core import metrics as evaluator


class TestMetricsDefaults:
    """Test suite for metrics defaults."""

    def test_get_default_metrics_regression(self):
        """Test default metrics list for regression."""
        defaults = evaluator.get_default_metrics("regression")
        expected = ['r2', 'rmse', 'mse', 'sep', 'mae', 'rpd', 'bias', 'consistency', 'nrmse', 'nmse', 'nmae', 'pearson_r', 'spearman_r']
        assert set(defaults) == set(expected)

    def test_get_default_metrics_classification(self):
        """Test default metrics list for classification."""
        defaults = evaluator.get_default_metrics("multiclass_classification")
        expected = ['accuracy', 'balanced_accuracy', 'precision', 'balanced_precision', 'recall', 'balanced_recall', 'f1', 'specificity']
        assert set(defaults) == set(expected)

    def test_get_default_metrics_binary(self):
        """Test default metrics list for binary classification."""
        defaults = evaluator.get_default_metrics("binary_classification")
        expected = ['accuracy', 'balanced_accuracy', 'precision', 'balanced_precision', 'recall', 'balanced_recall', 'f1', 'specificity', 'roc_auc', 'jaccard']
        assert set(defaults) == set(expected)

    def test_new_metrics_calculation(self):
        """Test calculation of new metrics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        # RMSE approx 0.1
        # Range = 4.0
        # Mean = 3.0
        # Std = 1.414

        # NRMSE = RMSE / Range = 0.1 / 4.0 = 0.025
        nrmse = evaluator.eval(y_true, y_pred, "nrmse")
        assert 0.02 < nrmse < 0.03

        # Consistency = 1 - (RMSE / Std)
        consistency = evaluator.eval(y_true, y_pred, "consistency")
        assert 0.9 < consistency < 1.0

        # Bias = mean(y_pred - y_true)
        bias = evaluator.eval(y_true, y_pred, "bias")
        assert abs(bias) < 0.1

    def test_eval_list(self):
        """Test eval_list returns correct number of metrics."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 2.1])
        metrics = ["rmse", "mae", "r2"]

        scores = evaluator.eval_list(y_true, y_pred, metrics)
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
