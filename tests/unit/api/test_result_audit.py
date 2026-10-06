"""Regression coverage for CV-selected refit accessors and export (API-01)."""

from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.data.predictions import Predictions


@pytest.mark.parametrize("per_dataset", [True, False])
def test_refit_accessors_and_export_share_cv_selected_model(tmp_path, per_dataset):
    """The first row and the held-out-test winner are both worse on CV."""
    cv = Predictions()
    finals = Predictions()
    for name, val_score, test_score in [("first", 0.5, 0.01), ("selected", 0.1, 0.8)]:
        cv.add_prediction(dataset_name="data", model_name=name, fold_id="avg", partition="val", metric="rmse", val_score=val_score)
        finals.add_prediction(dataset_name="data", model_name=name, fold_id="final", partition="test", metric="rmse", val_score=val_score, test_score=test_score)
    if not per_dataset:
        cv.extend_from_list(finals.iter_entries())
    runner = Mock()
    runner.export.return_value = tmp_path / "model.n4a"
    result = RunResult(cv, {"data": {"run_predictions": finals}} if per_dataset else {}, _runner=runner, _owns_runner=False)
    assert result.best["model_name"] == "selected"
    assert result.final["model_name"] == "selected"
    assert result.final_score == 0.8
    assert result.best_score == 0.8
    result.export(tmp_path / "model.n4a")
    assert runner.export.call_args.kwargs["source"]["model_name"] == "selected"
    models = result.models
    assert set(models) == {"first", "selected"}
    for name in models:
        assert models[name].final_entry["model_name"] == name
        assert models[name].cv_entry["model_name"] == name
    assert models["first"].cv_score == 0.5
    assert models["selected"].cv_score == 0.1


def test_per_dataset_and_global_final_candidates_are_ranked_together():
    global_predictions = Predictions()
    local_predictions = Predictions()
    global_predictions.add_prediction(dataset_name="A", model_name="Ridge", fold_id="final", partition="test", metric="rmse", val_score=0.5)
    local_predictions.add_prediction(dataset_name="B", model_name="Ridge", fold_id="final", partition="test", metric="rmse", val_score=0.1)
    result = RunResult(global_predictions, {"B": {"run_predictions": local_predictions}})
    assert result.final["dataset_name"] == "B"


def test_legacy_multimodel_export_replays_selected_refit(tmp_path):
    """Use a real legacy run and replay to catch exporting a constant model."""
    rng = np.random.default_rng(37)
    X = rng.normal(size=(48, 5))
    y = X[:, 0] * 4 + X[:, 1] * 2
    result = nirs4all.run(
        [StandardScaler(), KFold(3), {"model": {"_or_": [DummyRegressor(), Ridge(alpha=0.01)]}}],
        {"train_x": X[:36], "train_y": y[:36], "test_x": X[36:], "test_y": y[36:]},
        engine="legacy", workspace_path=tmp_path / "workspace", save_artifacts=True, save_charts=False, verbose=0,
    )
    assert result.final["model_name"] == result.best["model_name"]
    assert "Ridge" in result.final["model_name"]
    assert len(result.models) == 2
    for name, model in result.models.items():
        assert model.cv_entry["model_name"] == name
    bundle = result.export(tmp_path / "selected.n4a")
    replay = nirs4all.predict(bundle, X[36:], engine="legacy", verbose=0)
    np.testing.assert_allclose(replay.y_pred.ravel(), result.final["y_pred"].ravel(), atol=1e-8)
    assert np.std(replay.y_pred) > 1.0
    result.close()


def test_models_pairs_refit_with_its_selected_cv_configuration():
    """The same model name can identify multiple hyperparameter variants."""
    cv = Predictions()
    for config, score in [("selected_config", 0.4), ("other_config", 0.1)]:
        cv.add_prediction(dataset_name="data", config_name=config, model_name="Ridge", fold_id="avg", partition="val", metric="rmse", val_score=score)
    finals = Predictions()
    finals.add_prediction(dataset_name="data", config_name="selected_config_refit", model_name="Ridge", fold_id="final", partition="test", metric="rmse", val_score=0.4, test_score=0.3)
    result = RunResult(cv, {"data": {"run_predictions": finals}})
    model = result.models["Ridge"]
    assert model.cv_entry["config_name"] == "selected_config"
    assert model.cv_score == 0.4
