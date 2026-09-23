"""Model training controls in native scopes and a focused legacy oracle."""

import copy
import json

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.data import SpectroDataset
from nirs4all.pipeline.dagml import training_controls as controls


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


def test_pipeline_fold_policy_is_owned_only_by_aom_models():
    from n4m.model_selection.aom_search import AOMPLSRegressor

    metadata = {"nirs4all_train_params": {"use_pipeline_folds_for_aom": "required"}}
    evidence = controls.apply_model_training_controls(AOMPLSRegressor(), metadata, "FIT_CV")
    assert evidence["pipeline_fold_policy_for_aom"] == "required"
    assert evidence["model_params"] == {}
    with pytest.raises(ValueError, match="requires an AOM estimator"):
        controls.apply_model_training_controls(Ridge(), metadata, "FIT_CV")


def test_materialized_pipeline_folds_are_restricted_to_the_current_aom_scope():
    from n4m.model_selection.aom_search import AOMPLSRegressor

    model = AOMPLSRegressor(cv=4)
    fold_set = {
        "sample_ids": [f"s{i}" for i in range(8)],
        "folds": [
            {
                "fold_id": f"fold{fold}",
                "train_sample_ids": [f"s{i}" for i in range(8) if i % 4 != fold],
                "validation_sample_ids": [f"s{i}" for i in range(8) if i % 4 == fold],
            }
            for fold in range(4)
        ],
    }
    metadata = {
        "nirs4all_train_params": {"use_pipeline_folds_for_aom": "required"},
        "nirs4all_pipeline_fold_set": fold_set,
    }
    assert controls.apply_pipeline_folds_to_model(model, metadata, "FIT_CV", ["s0", "s1", "s2", "s4", "s5", "s6"])
    assert model.cv == 3
    np.testing.assert_array_equal(model.fold_ids, np.array([0, 1, 2, 0, 1, 2], dtype=np.int32))


def test_refit_warm_start_is_not_faked_with_fresh_estimator():
    with pytest.raises(NotImplementedError, match="CV-weight transfer"):
        controls.apply_model_training_controls(Ridge(), {"nirs4all_refit_params": {"warm_start": True}}, "REFIT")


def test_legacy_public_refit_warm_start_has_no_cv_artifact_to_transfer(tmp_path, monkeypatch):
    """Legacy's public refit currently cold-fits even when warm_start is requested.

    This is an oracle for the existing behavior, not a warm-start contract: the
    legacy training context has no artifact provider during CV or refit.
    """
    import nirs4all

    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 4))
    y = X[:, 0] ** 2 + 0.3 * X[:, 1]
    fits = []
    fitted_models = []
    original = GradientBoostingRegressor.fit

    def record(self, features, targets, **kwargs):
        fits.append((len(features), self.n_estimators, hasattr(self, "estimators_")))
        fitted = original(self, features, targets, **kwargs)
        fitted_models.append(fitted)
        return fitted

    monkeypatch.setattr(GradientBoostingRegressor, "fit", record)
    pipeline = [
        KFold(3, shuffle=True, random_state=4),
        {
            "model": GradientBoostingRegressor(n_estimators=3, random_state=3),
            "refit_params": {"warm_start": True, "warm_start_fold": "fold_1", "n_estimators": 6},
        },
    ]
    result = nirs4all.run(
        pipeline, (X, y), engine="legacy", workspace_path=tmp_path,
        save_artifacts=True, save_charts=False, verbose=0,
    )
    assert fits == [(20, 3, False)] * 3 + [(30, 6, False)]
    assert np.isfinite(result.cv_best_score)
    cold = GradientBoostingRegressor(n_estimators=6, random_state=3, warm_start=True)
    original(cold, X, y)
    np.testing.assert_allclose(fitted_models[-1].predict(X), cold.predict(X), rtol=0, atol=1e-6)


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
    import nirs4all

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
    result = nirs4all.run(pipeline, dataset, workspace_path=tmp_path, save_charts=False, save_artifacts=False, verbose=0)
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
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("export/replay fit"))
    archive = result.export(tmp_path / "controlled.n4a")
    np.testing.assert_array_equal(nirs4all.predict(archive, X[:7]).y_pred, expected.predict(X[:7]))
    result.close()


def test_invalid_training_control_is_diagnosed_before_any_hpo_or_preprocessing_fit(tmp_path, monkeypatch):
    import nirs4all

    rng = np.random.default_rng(91)
    X = rng.normal(size=(24, 4))
    y = X[:, 0] + 0.2
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("invalid control reached model fit"))
    monkeypatch.setattr(StandardScaler, "fit", lambda *args, **kwargs: pytest.fail("invalid control reached preprocessing fit"))
    with pytest.raises(Exception, match="would have been ignored"):
        nirs4all.run([StandardScaler(), KFold(3), {"model": Ridge(), "train_params": {"nonexistent_parameter": 9},
                       "finetune_params": {"n_trials": 2, "sampler": "grid", "approach": "single", "model_params": {"alpha": [0.1, 1]}}}],
                      (X, y), workspace_path=tmp_path, save_artifacts=False, save_charts=False, verbose=0)


def test_hpo_evaluates_actual_training_overrides_and_records_refit_precedence(tmp_path, monkeypatch):
    import nirs4all

    rng = np.random.default_rng(984)
    X = rng.normal(size=(30, 5)).astype(np.float32)
    y = (X @ np.arange(1.0, 6.0) + rng.normal(scale=0.1, size=30)).astype(np.float32)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy scheduler"))
    result = nirs4all.run([StandardScaler(), KFold(3, shuffle=True, random_state=23), {
        "model": Ridge(), "train_params": {"tol": 0.003, "verbose": 0}, "refit_params": {"alpha": 5.0},
        "finetune_params": {"approach": "single", "sampler": "grid", "n_trials": 2, "seed": 42,
                            "model_params": {"alpha": [0.01, 1.0]}},
    }], (X, y), workspace_path=tmp_path, save_charts=False, save_artifacts=False, verbose=0)
    fitted = result._dagml_refit_artifacts[0]["estimator"]
    evidence = fitted._nirs4all_host_hpo
    assert evidence["training_controls"] == {"alpha": 5.0, "tol": 0.003, "verbose": 0}
    assert evidence["effective_selected_model_params"]["alpha"] == fitted.steps[-1][1].alpha == 5.0
    train, val = next(ShuffleSplit(1, test_size=0.2, random_state=42).split(X))
    for trial in evidence["trials"]:
        assert trial["effective_model_params"] == {"alpha": 5.0, "tol": 0.003}
        expected = make_pipeline(StandardScaler(), Ridge(**trial["effective_model_params"])).fit(X[train], y[train].astype(float))
        score = np.sqrt(np.mean((expected.predict(X[val]).astype(float) - y[val].astype(float)) ** 2))
        assert trial["score"] == pytest.approx(score, abs=1e-10)
    assert len(fitted._nirs4all_host_hpo_history) == 4
    for search in fitted._nirs4all_host_hpo_history[:-1]:
        assert search["scope"]["phase"] == "FIT_CV"
        assert search["training_controls"] == {"tol": 0.003, "verbose": 0}
        assert all(trial["effective_model_params"]["alpha"] == trial["params"]["alpha"] for trial in search["trials"])
    result.close()


def test_nested_stacking_applies_each_branch_and_meta_control_in_native_scopes(tmp_path, monkeypatch):
    import nirs4all

    rng = np.random.default_rng(923)
    X = rng.normal(size=(36, 5))
    y = X @ np.arange(1.0, 6.0) + 0.3
    fits = []
    original = Ridge.fit

    def record(self, X, y, **kwargs):
        fits.append((len(X), self.alpha))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(Ridge, "fit", record)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy scheduler"))
    result = nirs4all.run([StandardScaler(), KFold(3, shuffle=True, random_state=23), {"branch": [
        [{"model": Ridge(11), "train_params": {"alpha": 1}, "refit_params": {"alpha": 4}}],
        [{"model": Ridge(22), "train_params": {"alpha": 2}, "refit_params": {"alpha": 5}}],
    ]}, {"merge": "predictions"}, {"model": Ridge(33), "train_params": {"alpha": 3}, "refit_params": {"alpha": 7}}],
        (X, y), workspace_path=tmp_path, save_artifacts=False, save_charts=False, verbose=0)
    assert np.isfinite(result.cv_best_score)
    assert fits.count((24, 3)) == 3  # Three outer meta fits, inner OOF only.
    assert sorted(alpha for rows, alpha in fits if rows == 36) == [4, 5, 7]
    assert {alpha for _, alpha in fits} == {1, 2, 3, 4, 5, 7}
    assert result._dagml_score_set and result._dagml_refit_artifacts
    result.close()


def test_full_training_without_splitter_applies_refit_controls_once(tmp_path, monkeypatch):
    import nirs4all

    rng = np.random.default_rng(924)
    X = rng.normal(size=(24, 5)).astype(np.float32)
    y = (X[:, 0] + 0.2).astype(np.float32)
    fits = []
    original = Ridge.fit

    def record(self, X, y, **kwargs):
        fits.append((len(X), self.alpha))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(Ridge, "fit", record)
    with pytest.warns(UserWarning, match="No splitter"):
        result = nirs4all.run([StandardScaler(), {"model": Ridge(), "train_params": {"alpha": 3}, "refit_params": {"alpha": 7}}],
                             (X, y), workspace_path=tmp_path, save_artifacts=False, save_charts=False, verbose=0)
    assert fits == [(24, 7)]
    assert np.isnan(result.cv_best_score)
    assert result._dagml_refit_artifacts[0]["estimator"]._nirs4all_training_controls["model_params"] == {"alpha": 7}
    result.close()
