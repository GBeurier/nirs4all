"""Actual estimator controls inside native CV/REFIT scopes, without a legacy run."""

import copy
import json
import sys

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.data import SpectroDataset
from nirs4all.pipeline.dagml import training_controls as controls
from nirs4all.pipeline.dagml.run_paths import _run_concrete


def test_controls_merge_only_for_refit_without_mutating_graph():
    metadata = {"nirs4all_train_params": {"alpha": 3.0, "tol": 0.01}, "nirs4all_refit_params": {"alpha": 7.0}}
    original = copy.deepcopy(metadata)
    assert controls.effective_training_controls(metadata, "FIT_CV") == {"alpha": 3.0, "tol": 0.01}
    assert controls.effective_training_controls(metadata, "REFIT") == {"alpha": 7.0, "tol": 0.01}
    assert metadata == original


def test_nested_component_controls_survive_json_without_repr():
    pipeline = make_pipeline(StandardScaler(), Ridge())
    metadata = {"nirs4all_train_params": controls.encode_training_controls({"ridge": Ridge(9)}, name="train_params")}
    evidence = controls.apply_model_training_controls(pipeline, json.loads(json.dumps(metadata)), "FIT_CV")
    assert pipeline.steps[-1][1].alpha == 9
    json.dumps(evidence, allow_nan=False)


@pytest.mark.parametrize("value", [[], "ignored", {1: 2}, {"alpha": float("nan")}, {"alpha": object()}])
def test_training_metadata_cannot_lossily_serialize(value):
    with pytest.raises((TypeError, ValueError)):
        controls.encode_training_controls(value, name="train_params")


def test_unknown_controls_are_not_historically_ignored():
    with pytest.raises(ValueError, match="would have been ignored"):
        controls.apply_model_training_controls(Ridge(), {"nirs4all_train_params": {"nonexistent_parameter": 19}}, "FIT_CV")


def test_refit_warm_start_is_not_faked_with_fresh_estimator():
    with pytest.raises(NotImplementedError, match="CV-weight transfer"):
        controls.apply_model_training_controls(Ridge(), {"nirs4all_refit_params": {"warm_start": True}}, "REFIT")


def test_verbose_is_observable_controller_output_not_estimator_parameter(monkeypatch):
    messages = []
    monkeypatch.setattr(controls.logger, "info", messages.append)
    model = Ridge()
    evidence = controls.apply_model_training_controls(model, {"nirs4all_train_params": {"verbose": 2}}, "FIT_CV")
    controls.report_model_training_controls(evidence, model, 24)
    assert len(messages) == 1 and "24 training rows" in messages[0] and "FIT_CV" in messages[0]
    assert "verbose" not in model.get_params()


@pytest.mark.parametrize("refit_alpha", [3.0, 7.0])
def test_real_native_cv_and_refit_use_historical_parameter_precedence(tmp_path, monkeypatch, refit_alpha):
    rng = np.random.default_rng(912)
    X = rng.normal(size=(36, 8)).astype(np.float32)
    y = X[:, 0] * 2 + X[:, 1] * 0.3 + rng.normal(scale=0.07, size=36)
    dataset = SpectroDataset("training-controls")
    dataset.add_samples(X, {"partition": "train"})
    dataset.add_targets(y)
    X = dataset.x({"partition": "train"}, layout="2d")
    y = np.asarray(dataset.y({"partition": "train"}), dtype=float).ravel()
    pipeline = [StandardScaler(), KFold(3, shuffle=True, random_state=31), {
        "model": Ridge(0.1), "train_params": {"alpha": 3.0, "verbose": 0}, "refit_params": {"alpha": refit_alpha},
    }]
    fits = []
    original = Ridge.fit

    def record(self, X, y, **kwargs):
        fits.append((len(X), self.alpha))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(Ridge, "fit", record)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy execution"))
    result = _run_concrete(pipeline, dataset, "", "unused-in-process", sys.executable, tmp_path)
    assert fits == [(24, 3.0)] * 3 + [(36, refit_alpha)]
    monkeypatch.setattr(Ridge, "fit", original)
    oof = np.empty(len(y))
    for train, val in KFold(3, shuffle=True, random_state=31).split(X):
        fitted = make_pipeline(StandardScaler(), Ridge(3.0)).fit(X[train], y[train])
        oof[val] = fitted.predict(X[val])
    assert result.cv_best_score == pytest.approx(np.sqrt(np.mean((y - oof) ** 2)), abs=1e-10)
    artifact = result._dagml_refit_artifacts[0]["estimator"]
    expected = make_pipeline(StandardScaler(), Ridge(refit_alpha)).fit(X, y)
    np.testing.assert_array_equal(artifact.predict(X), expected.predict(X))
    assert artifact._nirs4all_training_controls["phase"] == "REFIT"
    assert artifact._nirs4all_training_controls["model_params"] == {"alpha": refit_alpha}
    result.close()
