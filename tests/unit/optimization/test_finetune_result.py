"""Unit tests for FinetuneResult and TrialSummary dataclasses (Phase 6)."""

import pytest

from nirs4all.optimization.optuna import (
    FinetuneResult,
    TrialSummary,
    METRIC_DIRECTION,
    OptunaManager,
)


class TestTrialSummary:
    """Tests for TrialSummary dataclass."""

    def test_basic_construction(self):
        ts = TrialSummary(
            number=0,
            params={"n_components": 5},
            value=0.123,
            duration_seconds=1.5,
            state="COMPLETE",
        )
        assert ts.number == 0
        assert ts.params == {"n_components": 5}
        assert ts.value == 0.123
        assert ts.duration_seconds == 1.5
        assert ts.state == "COMPLETE"

    def test_pruned_trial(self):
        ts = TrialSummary(
            number=1,
            params={"alpha": 0.01},
            value=None,
            duration_seconds=0.3,
            state="PRUNED",
        )
        assert ts.value is None
        assert ts.state == "PRUNED"

    def test_failed_trial(self):
        ts = TrialSummary(
            number=2,
            params={},
            value=None,
            duration_seconds=0.0,
            state="FAIL",
        )
        assert ts.state == "FAIL"


class TestFinetuneResult:
    """Tests for FinetuneResult dataclass."""

    def test_basic_construction(self):
        result = FinetuneResult(
            best_params={"n_components": 5},
            best_value=0.05,
            n_trials=10,
        )
        assert result.best_params == {"n_components": 5}
        assert result.best_value == 0.05
        assert result.n_trials == 10
        assert result.n_pruned == 0
        assert result.n_failed == 0
        assert result.trials == []
        assert result.study_name is None
        assert result.metric is None
        assert result.direction == "minimize"

    def test_full_construction(self):
        trials = [
            TrialSummary(0, {"alpha": 0.1}, 0.5, 1.0, "COMPLETE"),
            TrialSummary(1, {"alpha": 0.01}, 0.3, 1.2, "COMPLETE"),
            TrialSummary(2, {"alpha": 1.0}, None, 0.5, "PRUNED"),
        ]
        result = FinetuneResult(
            best_params={"alpha": 0.01},
            best_value=0.3,
            n_trials=3,
            n_pruned=1,
            n_failed=0,
            trials=trials,
            study_name="test_study",
            metric="rmse",
            direction="minimize",
        )
        assert result.n_pruned == 1
        assert len(result.trials) == 3
        assert result.study_name == "test_study"
        assert result.metric == "rmse"

    def test_to_summary_dict(self):
        result = FinetuneResult(
            best_params={"n_components": 5},
            best_value=0.05,
            n_trials=10,
            n_pruned=2,
            n_failed=1,
            study_name="my_study",
            metric="rmse",
            direction="minimize",
        )
        summary = result.to_summary_dict()
        assert summary == {
            "n_trials": 10,
            "n_pruned": 2,
            "n_failed": 1,
            "best_value": 0.05,
            "study_name": "my_study",
            "metric": "rmse",
            "direction": "minimize",
        }

    def test_to_summary_dict_no_metric(self):
        result = FinetuneResult(
            best_params={},
            best_value=float("inf"),
            n_trials=0,
        )
        summary = result.to_summary_dict()
        assert summary["metric"] is None
        assert summary["n_trials"] == 0

    def test_empty_result_for_unavailable_optuna(self):
        """When Optuna is unavailable, an empty FinetuneResult is returned."""
        result = FinetuneResult(best_params={}, best_value=float("inf"), n_trials=0)
        assert result.best_params == {}
        assert result.n_trials == 0

    def test_maximize_direction(self):
        result = FinetuneResult(
            best_params={"C": 1.0},
            best_value=0.95,
            n_trials=20,
            metric="accuracy",
            direction="maximize",
        )
        assert result.direction == "maximize"
        assert result.metric == "accuracy"


class TestMetricDirection:
    """Tests for METRIC_DIRECTION mapping."""

    def test_regression_metrics_minimize(self):
        assert METRIC_DIRECTION["mse"] == "minimize"
        assert METRIC_DIRECTION["rmse"] == "minimize"
        assert METRIC_DIRECTION["mae"] == "minimize"

    def test_regression_r2_maximize(self):
        assert METRIC_DIRECTION["r2"] == "maximize"

    def test_classification_metrics_maximize(self):
        assert METRIC_DIRECTION["accuracy"] == "maximize"
        assert METRIC_DIRECTION["balanced_accuracy"] == "maximize"
        assert METRIC_DIRECTION["f1"] == "maximize"


class TestResolveMetricDirection:
    """Tests for _resolve_metric_direction method."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def _mock_dataset(self, task_type="regression"):
        """Create a minimal mock dataset."""

        class MockDataset:
            pass

        ds = MockDataset()
        ds.task_type = task_type
        return ds

    def test_explicit_metric_infers_direction(self, manager):
        params = manager._resolve_metric_direction(
            {"metric": "rmse"}, self._mock_dataset()
        )
        assert params["metric"] == "rmse"
        assert params["direction"] == "minimize"

    def test_explicit_metric_r2_maximize(self, manager):
        params = manager._resolve_metric_direction(
            {"metric": "r2"}, self._mock_dataset()
        )
        assert params["direction"] == "maximize"

    def test_explicit_direction_overrides_inference(self, manager):
        params = manager._resolve_metric_direction(
            {"metric": "rmse", "direction": "maximize"}, self._mock_dataset()
        )
        assert params["direction"] == "maximize"

    def test_no_metric_regression_defaults_minimize(self, manager):
        params = manager._resolve_metric_direction({}, self._mock_dataset("regression"))
        assert params["direction"] == "minimize"
        assert params.get("metric") is None

    def test_no_metric_classification_defaults_maximize(self, manager):
        params = manager._resolve_metric_direction(
            {}, self._mock_dataset("classification")
        )
        assert params["direction"] == "maximize"

    def test_unknown_metric_raises(self, manager):
        with pytest.raises(ValueError, match="Unknown metric 'bogus'"):
            manager._resolve_metric_direction(
                {"metric": "bogus"}, self._mock_dataset()
            )
