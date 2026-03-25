"""Tests for metrics defaults and new metrics."""

import numpy as np
import pytest

from nirs4all.core import metrics as evaluator
from nirs4all.core.metrics import HIGHER_IS_BETTER_METRICS, infer_ascending, is_higher_better


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


class TestMetricDirection:
    """Tests for the centralized metric direction functions."""

    @pytest.mark.parametrize("metric", [
        "rmse", "mse", "mae", "mape", "log_loss", "nrmse", "nmse", "nmae",
        "bias", "sep", "hamming_loss",
    ])
    def test_lower_is_better_metrics(self, metric):
        assert is_higher_better(metric) is False
        assert infer_ascending(metric) is True

    @pytest.mark.parametrize("metric", [
        "r2", "accuracy", "balanced_accuracy", "f1", "precision", "recall",
        "auc", "roc_auc", "kappa", "cohen_kappa", "rpd", "rpiq",
        "specificity", "matthews_corrcoef", "mcc", "jaccard",
        "f1_micro", "f1_macro", "precision_micro", "recall_macro",
        "explained_variance", "pearson_r", "spearman_r", "consistency",
    ])
    def test_higher_is_better_metrics(self, metric):
        assert is_higher_better(metric) is True
        assert infer_ascending(metric) is False

    def test_case_insensitive(self):
        assert is_higher_better("R2") is True
        assert is_higher_better("RMSE") is False
        assert infer_ascending("Balanced_Accuracy") is False

    def test_unknown_metric_defaults_to_lower_is_better(self):
        assert is_higher_better("unknown_metric") is False
        assert infer_ascending("unknown_metric") is True

    def test_metric_metadata_consistency(self):
        """Ensure METRIC_METADATA in pipeline/run.py is consistent with centralized direction."""
        from nirs4all.pipeline.run import METRIC_METADATA

        for metric_name, meta in METRIC_METADATA.items():
            if metric_name == "default":
                continue
            assert is_higher_better(metric_name) == meta["higher_is_better"], (
                f"Mismatch for '{metric_name}': is_higher_better={is_higher_better(metric_name)} "
                f"but METRIC_METADATA says higher_is_better={meta['higher_is_better']}"
            )
