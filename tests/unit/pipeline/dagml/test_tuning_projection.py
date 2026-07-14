"""Unit locks for native tuning result projection."""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.api.result import RunResult
from nirs4all.pipeline.dagml.tuning_adapters import ObjectiveTuningRunResult
from nirs4all.pipeline.dagml.tuning_contracts import TrialResult, TuningResult, parse_tuning_spec
from nirs4all.pipeline.dagml.tuning_projection import project_objective_tuning_to_run_result


def _result() -> TuningResult:
    tuning = parse_tuning_spec(
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "metric": "rmse",
            "direction": "minimize",
            "n_trials": 2,
            "sampler": "grid",
        }
    )
    return TuningResult(
        tuning=tuning,
        best_params={"alpha": 0.2},
        best_value=0.2,
        trials=(
            TrialResult(number=0, params={"alpha": 0.9}, value=0.9, state="COMPLETE", diagnostics={}),
            TrialResult(number=1, params={"alpha": 0.2}, value=0.2, state="COMPLETE", diagnostics={}),
        ),
        optimizer="optuna",
    )


class _PredictingEstimator:
    def predict(self, x: object) -> np.ndarray:
        assert np.asarray(x).shape == (2, 1)
        return np.asarray([1.25, 2.75])


def test_project_objective_tuning_to_run_result_carries_tuning_without_predictions() -> None:
    tuning_result = _result()
    projected = project_objective_tuning_to_run_result(
        ObjectiveTuningRunResult(tuning_result=tuning_result, refit_estimator=object(), tuning_id="tune-main"),
        per_dataset={"dataset": {"status": "tuned"}},
    )

    assert isinstance(projected, RunResult)
    assert projected.num_predictions == 0
    assert projected.per_dataset == {"dataset": {"status": "tuned"}}
    assert projected.tuning_id == "tune-main"
    assert projected.tuning_result is tuning_result
    assert projected.tuning_best_params == {"alpha": 0.2}
    assert projected.tuning_best_value == pytest.approx(0.2)
    assert projected.validate(raise_on_failure=False)["valid"] is True


def test_project_objective_tuning_to_run_result_projects_explicit_winner_prediction() -> None:
    tuning_result = _result()
    projected = project_objective_tuning_to_run_result(
        ObjectiveTuningRunResult(tuning_result=tuning_result, refit_estimator=_PredictingEstimator(), tuning_id="tune-main"),
        winner_x=np.asarray([[1.0], [2.0]]),
        winner_y_true=np.asarray([1.5, 2.5]),
        winner_score=0.1,
        winner_metric="rmse",
        winner_sample_ids=("sample-a", "sample-b"),
        winner_dataset_name="heldout",
        winner_model_name="WinnerEstimator",
        winner_metadata={"split": "external_test"},
    )

    assert projected.num_predictions == 1
    assert projected.best["dataset_name"] == "heldout"
    assert projected.best["model_name"] == "WinnerEstimator"
    assert projected.best["fold_id"] == "final"
    assert projected.best["refit_context"] == "tuning_winner"
    assert projected.best_score == pytest.approx(0.1)
    assert projected.tuning_best_params == {"alpha": 0.2}
    assert projected.validate(raise_on_failure=False)["valid"] is True

    [entry] = projected.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [1.25, 2.75])
    np.testing.assert_allclose(entry["y_true"], [1.5, 2.5])
    assert entry["best_params"] == {"alpha": 0.2}
    assert entry["scores"] == {"test": {"rmse": 0.1}}
    assert entry["metadata"] == {
        "split": "external_test",
        "physical_sample_id": ["sample-a", "sample-b"],
    }


def test_project_objective_tuning_to_run_result_requires_winner_score_and_metric() -> None:
    with pytest.raises(ValueError, match="winner_score and winner_metric"):
        project_objective_tuning_to_run_result(
            ObjectiveTuningRunResult(tuning_result=_result(), refit_estimator=_PredictingEstimator()),
            winner_x=np.asarray([[1.0], [2.0]]),
            winner_score=0.1,
        )


def test_project_objective_tuning_to_run_result_requires_refit_estimator_for_winner_projection() -> None:
    with pytest.raises(ValueError, match="refit_estimator"):
        project_objective_tuning_to_run_result(
            ObjectiveTuningRunResult(tuning_result=_result(), refit_estimator=None),
            winner_x=np.asarray([[1.0], [2.0]]),
            winner_score=0.1,
            winner_metric="rmse",
        )


def test_project_objective_tuning_to_run_result_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="ObjectiveTuningRunResult"):
        project_objective_tuning_to_run_result(object())  # type: ignore[arg-type]
