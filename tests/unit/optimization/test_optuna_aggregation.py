"""Unit tests for OptunaManager._aggregate_scores (BUG-2 regression test)."""

import numpy as np
import pytest

from nirs4all.optimization.optuna import OptunaManager


class TestAggregateScores:
    """Tests for _aggregate_scores method."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_mean_returns_average_not_sum(self, manager):
        """BUG-2 regression: 'mean' must return np.mean, not np.sum."""
        scores = [0.5, 0.6, 0.7]
        result = manager._aggregate_scores(scores, "mean")
        assert result == pytest.approx(0.6)
        assert result != pytest.approx(1.8)  # Was the old bug

    def test_mean_single_score(self, manager):
        result = manager._aggregate_scores([0.42], "mean")
        assert result == pytest.approx(0.42)

    def test_best_returns_minimum(self, manager):
        scores = [0.5, 0.6, 0.7]
        result = manager._aggregate_scores(scores, "best")
        assert result == pytest.approx(0.5)

    def test_best_with_inf(self, manager):
        scores = [float('inf'), 0.5, 0.6]
        result = manager._aggregate_scores(scores, "best")
        assert result == pytest.approx(0.5)

    def test_robust_best_excludes_inf(self, manager):
        scores = [float('inf'), 0.5, 0.6]
        result = manager._aggregate_scores(scores, "robust_best")
        assert result == pytest.approx(0.5)

    def test_robust_best_all_inf(self, manager):
        scores = [float('inf'), float('inf')]
        result = manager._aggregate_scores(scores, "robust_best")
        assert result == float('inf')

    def test_unknown_eval_mode_raises(self, manager):
        """Unknown eval_mode must raise ValueError, not silently fallback.

        Note: 'avg' is normalized to 'mean' upstream in finetune(), but
        _aggregate_scores itself must reject it — it only accepts canonical values.
        """
        with pytest.raises(ValueError, match="Unknown eval_mode 'avg'"):
            manager._aggregate_scores([0.5, 0.6], "avg")

    def test_unknown_eval_mode_sum_raises(self, manager):
        with pytest.raises(ValueError, match="Unknown eval_mode"):
            manager._aggregate_scores([0.5, 0.6], "sum")


@pytest.mark.parametrize("mode", ["best", "mean", "robust_best"])
def test_maximize_fold_evidence(mode):
    manager = OptunaManager()
    expected = 0.8 if mode != "mean" else 0.6
    assert manager._aggregate_scores([0.4, 0.8], mode, "maximize") == pytest.approx(expected)


@pytest.mark.parametrize("failed", [float("inf"), float("-inf"), float("nan")])
def test_failed_controller_scores_cannot_win_maximization(failed):
    manager = OptunaManager()
    assert manager._aggregate_scores([failed, 0.8], "best", "maximize") == pytest.approx(0.8)
    assert manager._aggregate_scores([failed], "robust_best", "maximize") == float("-inf")
    assert manager._aggregate_scores([failed, 0.8], "mean", "maximize") == float("-inf")


def test_classification_default_minimizes_negative_accuracy():
    from types import SimpleNamespace

    from nirs4all.optimization.n4m_engine import N4MFinetuneManager

    dataset = SimpleNamespace(task_type="binary_classification")
    for manager in (OptunaManager(), N4MFinetuneManager()):
        assert manager._resolve_metric_direction({}, dataset)["direction"] == "minimize"
        assert manager._resolve_metric_direction({"metric": "balanced_accuracy"}, dataset)["direction"] == "maximize"


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_maximize_study_selects_valid_trial(failure):
    from types import SimpleNamespace

    class Controller:
        def _get_model_instance(self, dataset, config, force_params):
            return force_params["quality"]

        def _prepare_data(self, X, y, context):
            return X, y

        def _train_model(self, model, *args, **kwargs):
            if model == "failed" and failure == "exception":
                raise ValueError("deliberate failed fit")
            return model

        def _evaluate_model(self, model, *args, **kwargs):
            return float("inf") if model == "failed" else 0.8

    manager = OptunaManager()
    dataset = SimpleNamespace(task_type="binary_classification")
    params = {"metric": "accuracy", "direction": "maximize", "sampler": "grid", "seed": 3,
              "model_params": {"quality": ["failed", "valid"]}}
    X, y = np.ones((4, 2)), np.array([0, 1, 0, 1])
    result = manager._run_single_optimization(dataset, {}, X, y, X, y, params, 2, None, Controller(), verbose=0)
    assert result.best_params["quality"] == "valid"
    assert result.best_value == pytest.approx(0.8)
