"""Real per-source nested OOF stacking, not the historical spectral concat."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.data import SpectroDataset


def _data(poison=False):
    rng = np.random.default_rng(883)
    first = rng.normal(size=(48, 2))
    second = rng.normal(size=(48, 3)) * 3 + 8
    y = first[:, 0] * 2 + second[:, 0] * 0.7 + rng.normal(scale=0.1, size=48)
    if poison:
        y[:12] += 10000
    dataset = SpectroDataset("two-source-stacking")
    dataset.add_samples([first[:36], second[:36]], {"partition": "train"})
    dataset.add_targets(y[:36])
    dataset.add_samples([first[36:], second[36:]], {"partition": "test"})
    dataset.add_targets(y[36:])
    return dataset


def _run(dataset, tmp_path, *, hpo=False, meta_hpo=False):
    import nirs4all

    model = {"model": Ridge(0.3), "train_params": {"tol": 0.003}, "refit_params": {"alpha": 0.7}}
    if hpo:
        model["finetune_params"] = {"approach": hpo if isinstance(hpo, str) else "single", "n_trials": 2, "sampler": "grid", "seed": 42,
                                    "model_params": {"alpha": [0.1, 1.0]}}
    meta = {"model": Ridge(0.1), "refit_params": {"alpha": 0.2}}
    if meta_hpo:
        meta["finetune_params"] = {"approach": "single", "n_trials": 2, "sampler": "grid", "model_params": {"alpha": [0.1, 1.0]}}
    return nirs4all.run([KFold(3), {"branch": {"by_source": True, "steps": [StandardScaler(), model]}},
                        {"merge": "predictions"}, meta],
                       dataset, workspace_path=tmp_path, save_artifacts=False, save_charts=False, verbose=0)


def test_source_models_fit_only_their_columns_and_replay_full_layout(tmp_path, monkeypatch):
    import nirs4all
    from nirs4all.pipeline.dagml import run_paths

    dataset = _data()
    fitted_shapes = []
    native_calls = []
    original_fit = StandardScaler.fit
    original_run = run_paths.run_cv_refit_bundle

    def observe_run(*args, **kwargs):
        native_calls.append({"dsl": kwargs["dsl"], "graph": kwargs["graph"]})
        return original_run(*args, **kwargs)

    def record(self, X, *args, **kwargs):
        fitted_shapes.append(X.shape)
        return original_fit(self, X, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record)
    monkeypatch.setattr(run_paths, "run_cv_refit_bundle", observe_run)
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._run_model_on_precomputed_matrix", lambda *args, **kwargs: pytest.fail("old Python CV loop"))
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy execution"))
    result = _run(dataset, tmp_path)
    assert len(result.runs) == 3
    assert len(native_calls) == 1  # One native campaign, not a Python CV loop per source.
    assert native_calls[0]["dsl"]["inner_cv"]["kind"] == "kfold"
    assert result._dagml_score_set is not None
    assert sorted(shape for shape in fitted_shapes if shape[0] == 36) == [(36, 2), (36, 3)]
    assert {shape[1] for shape in fitted_shapes} == {2, 3}
    assert any(shape[0] < 24 for shape in fitted_shapes)
    assert {row["branch_name"] for row in result.predictions.filter_predictions()} == {"source_0", "source_1", None}
    monkeypatch.setattr(StandardScaler, "fit", original_fit)
    X = dataset.x({"partition": "train"}, layout="2d")
    y = np.asarray(dataset.y({"partition": "train"}), dtype=float).ravel()
    X_new = dataset.x({"partition": "test"}, layout="2d")
    base_predictions = []
    for index, child in enumerate(result.runs[:2]):
        columns = slice(0, 2) if index == 0 else slice(2, 5)
        # sklearn ColumnTransformer's integer-list selection yields F-order
        # blocks. Match storage order as well as values for float32 BLAS parity.
        expected = make_pipeline(StandardScaler(), Ridge(0.7, tol=0.003)).fit(np.asfortranarray(X[:, columns]), y)
        actual = child._dagml_refit_artifacts[0]["estimator"]
        np.testing.assert_array_equal(actual.predict(X_new), expected.predict(np.asfortranarray(X_new[:, columns])))
        base_predictions.append(actual.predict(X_new))
        layout = child.per_dataset[dataset.name]["source_stacking"]["layout"]
        assert [source["column_count"] for source in layout["sources"]] == [2, 3]
        source_nodes = [node for node in native_calls[0]["graph"]["nodes"] if node["kind"] == "model" and node["id"].startswith(f"branch:{index}.")]
        assert len(source_nodes) == 1
        binding = source_nodes[0]["metadata"]["nirs4all_source_stacking"]
        assert binding["layout_fingerprint"] == layout["fingerprint"]
        assert binding["source"] == layout["sources"][index]
    meta = result.runs[-1]
    fitted_meta = next(artifact["estimator"] for artifact in meta._dagml_refit_artifacts if artifact["controller_id"] == "controller:nirs4all.meta_model")
    assert fitted_meta.n_features_in_ == 2  # Prediction columns, not 5/8 spectral columns.
    assert fitted_meta.alpha == 0.2
    expected_prediction = fitted_meta.predict(np.column_stack(base_predictions))
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("export or replay fitted"))
    archive = meta.export(tmp_path / "source-stack.n4a")
    np.testing.assert_array_equal(nirs4all.predict(archive, X_new).y_pred, expected_prediction)
    from nirs4all.pipeline.dagml.general_archive import load_general_archive

    recorded_layout = load_general_archive(archive)["manifest"]["source_stacking"]["layout"]
    assert recorded_layout == meta.per_dataset[dataset.name]["source_stacking"]["layout"]
    result.close()


@pytest.mark.parametrize("hpo", [False, "single", "grouped"])
def test_outer_validation_targets_cannot_affect_source_stacking_predictions(tmp_path, monkeypatch, hpo):
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy execution"))
    before = _run(_data(), tmp_path / "before", hpo=hpo)
    after = _run(_data(poison=True), tmp_path / "after", hpo=hpo)

    def outer_prediction(result):
        rows = result.runs[-1].predictions.filter_predictions(partition="val", load_arrays=True)
        return next(row["y_pred"] for row in rows if set(row["sample_indices"]) == set(range(12)))

    np.testing.assert_array_equal(outer_prediction(before), outer_prediction(after))
    if hpo:
        for child in before.runs[:2]:
            fitted = child._dagml_refit_artifacts[0]["estimator"]
            assert fitted._nirs4all_host_hpo["scope"]["phase"] == "REFIT"
            assert fitted._nirs4all_host_hpo["effective_selected_model_params"]["alpha"] == 0.7
    before.close()
    after.close()


def test_meta_hpo_is_not_silently_ignored_or_fitted_on_precomputed_oof(tmp_path, monkeypatch):
    monkeypatch.setattr(StandardScaler, "fit", lambda *args, **kwargs: pytest.fail("meta HPO was rejected after preprocessing fit"))
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("meta HPO was rejected after model fit"))
    with pytest.raises(Exception, match="meta-model HPO requires a native whole-stack nested search"):
        _run(_data(), tmp_path, meta_hpo=True)


def test_classification_uses_real_source_predictions_and_native_selection(tmp_path, monkeypatch):
    import nirs4all
    from nirs4all.pipeline.dagml import run_paths

    rng = np.random.default_rng(893)
    y = np.tile([0, 1], 24)
    first = rng.normal(size=(48, 2)) + y[:, None] * 2
    second = rng.normal(size=(48, 3)) + y[:, None] * 1.5
    dataset = SpectroDataset("source-classification")
    dataset.add_samples([first[:36], second[:36]], {"partition": "train"})
    dataset.add_targets(y[:36])
    dataset.add_samples([first[36:], second[36:]], {"partition": "test"})
    dataset.add_targets(y[36:])
    native_dsls = []
    original_run = run_paths.run_cv_refit_bundle

    def observe_run(*args, **kwargs):
        native_dsls.append(kwargs["dsl"])
        return original_run(*args, **kwargs)

    monkeypatch.setattr(run_paths, "run_cv_refit_bundle", observe_run)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy scheduler"))
    result = nirs4all.run([KFold(3), {"branch": {"by_source": True, "steps": [StandardScaler(), {
        "model": LogisticRegression(), "train_params": {"C": 0.3}, "refit_params": {"C": 0.7},
    }]}}, {"merge": "predictions"}, {"model": LogisticRegression(C=2)}],
        dataset, workspace_path=tmp_path, save_artifacts=False, save_charts=False, verbose=0)
    meta = result.runs[-1]
    inner_cv = native_dsls[0]["inner_cv"]
    assert inner_cv["kind"] == "stratified_kfold" and len(inner_cv["strata"]) == 36
    assert set(inner_cv["strata"].values()) == {float(0).hex(), float(1).hex()}
    assert meta.cv_best["metric"] == "balanced_accuracy"
    assert 0 <= meta.cv_best_score <= 1
    X = dataset.x({"partition": "test"}, layout="2d")
    base_predictions = [child._dagml_refit_artifacts[0]["estimator"].predict(X) for child in result.runs[:2]]
    estimator = next(artifact["estimator"] for artifact in meta._dagml_refit_artifacts if artifact["controller_id"] == "controller:nirs4all.meta_model")
    expected = estimator.predict(np.column_stack(base_predictions))
    monkeypatch.setattr(LogisticRegression, "fit", lambda *args, **kwargs: pytest.fail("replay fit"))
    np.testing.assert_array_equal(nirs4all.predict(meta.export(tmp_path / "classification.n4a"), X).y_pred, expected)
    result.close()
