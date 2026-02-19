"""Semantic tests for finetuning end-to-end behavior (Phase 7 - ISSUE-19).

These tests verify that the optimization pipeline integrates correctly:
- FinetuneResult flows from OptunaManager through BaseModelController
- Custom metric actually changes the objective function
- Optimization summary is stored in predictions
"""

import numpy as np
import pytest

from nirs4all.optimization.optuna import (
    METRIC_DIRECTION,
    FinetuneResult,
    OptunaManager,
)


class MockDataset:
    """Minimal mock for SpectroDataset."""

    def __init__(self, task_type="regression"):
        self.task_type = task_type
        self.name = "mock_dataset"

class MockContext:
    """Minimal mock for ExecutionContext."""

    class State:
        step_number = 1

    state = State()

class MockController:
    """Minimal mock controller that trains a simple linear model."""

    def _get_model_instance(self, dataset, model_config, force_params=None):
        from sklearn.linear_model import Ridge
        params = force_params or {}
        return Ridge(**params)

    def _prepare_data(self, X, y, context):
        return X, y

    def _train_model(self, model, X_train, y_train, X_val, y_val, **kwargs):
        y_train_flat = y_train.ravel() if y_train.ndim > 1 else y_train
        model.fit(X_train, y_train_flat)
        return model

    def _evaluate_model(self, model, X_val, y_val, metric=None, direction="minimize"):
        y_val_1d = y_val.ravel() if y_val.ndim > 1 else y_val
        y_pred = model.predict(X_val)
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        if metric is not None:
            from nirs4all.core import metrics as evaluator_mod
            return evaluator_mod.eval(y_val_1d, y_pred, metric)

        from sklearn.metrics import mean_squared_error
        return mean_squared_error(y_val_1d, y_pred)

@pytest.fixture
def manager():
    return OptunaManager()

@pytest.fixture
def controller():
    return MockController()

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(50) * 0.1
    return X, y

@pytest.fixture
def folds(sample_data):
    X, y = sample_data
    n = len(X)
    idx = np.arange(n)
    mid = n // 2
    return [(idx[:mid], idx[mid:])]

class TestFinetuneReturnsFinetuneResult:
    """Verify that finetune() returns FinetuneResult, not raw dicts."""

    def test_single_approach_returns_finetune_result(self, manager, controller, sample_data, folds):
        X, y = sample_data
        dataset = MockDataset()
        context = MockContext()

        finetune_params = {
            "n_trials": 2,
            "sampler": "random",
            "approach": "single",
            "seed": 42,
            "model_params": {
                "alpha": ('float_log', 1e-4, 1e2),
            },
        }

        result = manager.finetune(
            dataset=dataset, model_config={}, X_train=X, y_train=y,
            X_test=X[:5], y_test=y[:5], folds=None,
            finetune_params=finetune_params, context=context, controller=controller,
        )

        assert isinstance(result, FinetuneResult)
        assert result.n_trials == 2
        assert isinstance(result.best_params, dict)
        assert "alpha" in result.best_params
        assert result.best_value != float("inf")
        assert len(result.trials) == 2

    def test_grouped_approach_returns_finetune_result(self, manager, controller, sample_data, folds):
        X, y = sample_data
        dataset = MockDataset()
        context = MockContext()

        finetune_params = {
            "n_trials": 2,
            "sampler": "random",
            "approach": "grouped",
            "seed": 42,
            "model_params": {
                "alpha": ('float_log', 1e-4, 1e2),
            },
        }

        result = manager.finetune(
            dataset=dataset, model_config={}, X_train=X, y_train=y,
            X_test=X[:5], y_test=y[:5], folds=folds,
            finetune_params=finetune_params, context=context, controller=controller,
        )

        assert isinstance(result, FinetuneResult)
        assert result.n_trials == 2

    def test_individual_approach_returns_list_of_finetune_results(self, manager, controller, sample_data, folds):
        X, y = sample_data
        dataset = MockDataset()
        context = MockContext()

        finetune_params = {
            "n_trials": 2,
            "sampler": "random",
            "approach": "individual",
            "seed": 42,
            "model_params": {
                "alpha": ('float_log', 1e-4, 1e2),
            },
        }

        result = manager.finetune(
            dataset=dataset, model_config={}, X_train=X, y_train=y,
            X_test=X[:5], y_test=y[:5], folds=folds,
            finetune_params=finetune_params, context=context, controller=controller,
        )

        assert isinstance(result, list)
        assert len(result) == len(folds)
        for r in result:
            assert isinstance(r, FinetuneResult)
            assert r.n_trials == 2

    def test_multiphase_returns_finetune_result(self, manager, controller, sample_data, folds):
        X, y = sample_data
        dataset = MockDataset()
        context = MockContext()

        finetune_params = {
            "seed": 42,
            "phases": [
                {"n_trials": 2, "sampler": "random"},
                {"n_trials": 2, "sampler": "tpe"},
            ],
            "model_params": {
                "alpha": ('float_log', 1e-4, 1e2),
            },
        }

        result = manager.finetune(
            dataset=dataset, model_config={}, X_train=X, y_train=y,
            X_test=X[:5], y_test=y[:5], folds=folds,
            finetune_params=finetune_params, context=context, controller=controller,
        )

        assert isinstance(result, FinetuneResult)
        assert result.n_trials == 4  # 2 + 2 from both phases

class TestCustomMetricChangesObjective:
    """Verify that setting metric= changes the objective function."""

    def test_r2_metric_maximizes(self, manager, controller, sample_data, folds):
        X, y = sample_data
        dataset = MockDataset()
        context = MockContext()

        finetune_params = {
            "n_trials": 3,
            "sampler": "random",
            "approach": "single",
            "seed": 42,
            "metric": "r2",
            "model_params": {
                "alpha": ('float_log', 1e-4, 1e2),
            },
        }

        result = manager.finetune(
            dataset=dataset, model_config={}, X_train=X, y_train=y,
            X_test=X[:5], y_test=y[:5], folds=None,
            finetune_params=finetune_params, context=context, controller=controller,
        )

        assert isinstance(result, FinetuneResult)
        assert result.direction == "maximize"
        assert result.metric == "r2"
        # R2 should be positive for this well-correlated data
        assert result.best_value > 0.0

    def test_rmse_metric_minimizes(self, manager, controller, sample_data, folds):
        X, y = sample_data
        dataset = MockDataset()
        context = MockContext()

        finetune_params = {
            "n_trials": 3,
            "sampler": "random",
            "approach": "single",
            "seed": 42,
            "metric": "rmse",
            "model_params": {
                "alpha": ('float_log', 1e-4, 1e2),
            },
        }

        result = manager.finetune(
            dataset=dataset, model_config={}, X_train=X, y_train=y,
            X_test=X[:5], y_test=y[:5], folds=None,
            finetune_params=finetune_params, context=context, controller=controller,
        )

        assert isinstance(result, FinetuneResult)
        assert result.direction == "minimize"
        assert result.metric == "rmse"
        # RMSE should be a small positive value
        assert 0 < result.best_value < 10.0

class TestToSummaryDictIntegration:
    """Verify that to_summary_dict produces correct output from real runs."""

    def test_summary_dict_from_real_optimization(self, manager, controller, sample_data):
        X, y = sample_data
        dataset = MockDataset()
        context = MockContext()

        finetune_params = {
            "n_trials": 3,
            "sampler": "tpe",
            "approach": "single",
            "seed": 42,
            "metric": "rmse",
            "model_params": {
                "alpha": ('float_log', 1e-4, 1e2),
            },
        }

        result = manager.finetune(
            dataset=dataset, model_config={}, X_train=X, y_train=y,
            X_test=X[:5], y_test=y[:5], folds=None,
            finetune_params=finetune_params, context=context, controller=controller,
        )

        summary = result.to_summary_dict()
        assert summary["n_trials"] == 3
        assert summary["n_pruned"] == 0
        assert summary["n_failed"] == 0
        assert summary["metric"] == "rmse"
        assert summary["direction"] == "minimize"
        assert isinstance(summary["best_value"], float)
        assert summary["best_value"] > 0
