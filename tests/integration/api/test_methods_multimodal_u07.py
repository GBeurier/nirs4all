"""Real raw-source native encoders, public campaigns and target-free portable replay."""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

import nirs4all
from nirs4all.api.portable_archive import read_portable_predictor_archive_v2
from nirs4all.pipeline.dagml.methods_multimodal import PROFILE, bind_methods_dsl, recipe_from_estimator, source_schemas_from_cohort

EXAMPLE = Path(__file__).resolve().parents[3] / "examples/user/02_data_handling/U07_multimodal.py"
spec = importlib.util.spec_from_file_location("canonical_u07_native_test", EXAMPLE)
assert spec is not None and spec.loader is not None
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def _model(**params: Any) -> Any:
    return example.make_pipeline(backend="methods")[-1]["model"].set_params(**params)


def _blocks(cohort: Any, rows: Any) -> dict[str, Any]:
    return {name: source.values[rows] for name, source in cohort.sources.items()}


def _replace(cohort: Any, name: str, values: Any, **changes: Any) -> Any:
    old = cohort.sources[name]
    sources = dict(cohort.sources)
    options = {"representation_id": old.representation_id, "axes": old.axes, "feature_names": old.feature_names,
               "axis_units": old.axis_units, "axis_coordinates": old.axis_coordinates, "presence_mask": old.presence_mask}
    options.update(changes)
    sources[name] = TensorSource(values, cohort.sample_ids, **options)
    return MultimodalDataset(sources, sample_ids=cohort.sample_ids, y=cohort.y, groups=cohort.groups,
                             partitions=cohort.partitions, target_names=cohort.target_names, name=cohort.name)


def _oracle(model: Any, cohort: Any, fit: Any, predict: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Independent sklearn encoder/Ridge oracle, used only in this test."""
    fitted, encoded_fit, encoded_predict = {}, [], []
    for name, transformer in model.transformers.items():
        estimator = clone(transformer).fit(cohort.sources[name].values[fit], cohort.y[fit])
        fitted[name] = estimator
        weight = (model.source_weights or {}).get(name, 1.0)
        encoded_fit.append(np.asarray(estimator.transform(cohort.sources[name].values[fit]), dtype=float) * weight)
        encoded_predict.append(np.asarray(estimator.transform(cohort.sources[name].values[predict]), dtype=float) * weight)
    ridge = clone(model.model).fit(np.concatenate(encoded_fit, axis=1), cohort.y[fit])
    return ridge.predict(np.concatenate(encoded_predict, axis=1)), fitted


@pytest.mark.parametrize("seed", [17, 23])
@pytest.mark.parametrize("components,weight,alpha", [(2, 0.5, 0.1), (4, 1.0, 1.0)])
def test_real_native_encoder_and_ridge_oracle(seed: int, components: int, weight: float, alpha: float) -> None:
    from n4m import MultimodalPipeline

    cohort = example.make_cohort(seed)
    # The direct public binding requires storage matching IO's object schema.
    # SDK tests below retain raw Unicode arrays to exercise its host boundary.
    cohort = _replace(cohort, "metadata", cohort.sources["metadata"].values.astype(object))
    train = np.flatnonzero(np.asarray(cohort.partitions) == "train")
    test = np.flatnonzero(np.asarray(cohort.partitions) == "test")
    model = _model(transformers__image__n_components=components, source_weights__image=weight, model__alpha=alpha)
    model.transformers["image"].random_state = seed
    model.transformers["series"].random_state = seed
    schemas = source_schemas_from_cohort(cohort)
    oracle, fitted = _oracle(model, cohort, train, test)
    native = MultimodalPipeline(recipe_from_estimator(model), schemas)
    try:
        native.fit(_blocks(cohort, train), cohort.y[train])
        np.testing.assert_allclose(np.asarray(native.predict(_blocks(cohort, test))).reshape(-1), oracle, rtol=2e-7, atol=2e-7)
        encoded = np.asarray(native.transform(_blocks(cohort, test)))
        offset = 0
        for name in cohort.sources:
            expected = np.asarray(fitted[name].transform(cohort.sources[name].values[test]), dtype=float)
            expected *= (model.source_weights or {}).get(name, 1.0)
            actual = encoded[:, offset:offset + expected.shape[1]]
            if name in {"image", "series"}:
                # Component signs are mathematically ambiguous. This compares
                # the entire score subspace without changing production values.
                np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, rtol=2e-7, atol=2e-7)
            else:
                np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)
            offset += expected.shape[1]
        assert encoded.shape[1] == offset
        state = native.export_state()
        assert isinstance(state, bytes) and state[:4] == b"N4MF"
        restored = MultimodalPipeline.from_state(state, recipe=recipe_from_estimator(model), source_schemas=schemas)
        try:
            np.testing.assert_array_equal(restored.predict(_blocks(cohort, test)), native.predict(_blocks(cohort, test)))
        finally:
            restored.close()
    finally:
        native.close()


def test_categories_are_learned_on_fit_rows_and_unknown_is_zero() -> None:
    from n4m import MultimodalPipeline

    cohort = example.make_cohort()
    values = cohort.sources["metadata"].values.astype(object)
    train = np.flatnonzero(np.asarray(cohort.partitions) == "train")
    test = np.flatnonzero(np.asarray(cohort.partitions) == "test")
    values[train, 1] = np.where(np.arange(len(train)) % 2, "café", "α")
    values[test, 1] = "未知"
    values[:, 0] = 7.0  # Native ddof=0 scaler's zero-variance behavior.
    cohort = _replace(cohort, "metadata", values)
    model = _model()
    schemas = source_schemas_from_cohort(cohort)
    expected, fitted = _oracle(model, cohort, train, test)
    native = MultimodalPipeline(recipe_from_estimator(model), schemas)
    try:
        native.fit(_blocks(cohort, train), cohort.y[train])
        np.testing.assert_allclose(np.asarray(native.predict(_blocks(cohort, test))).reshape(-1), expected, rtol=2e-7, atol=2e-7)
        encoded = np.asarray(native.transform(_blocks(cohort, test)))
        metadata = fitted["metadata"].transform(values[test])
        np.testing.assert_array_equal(encoded[:, -metadata.shape[1]:], np.zeros_like(metadata))
        assert list(fitted["metadata"].named_transformers_["category"].categories_[0]) == ["café", "α"]
    finally:
        native.close()


def _observe_direct_native_lifecycle(monkeypatch: Any) -> tuple[list[Any], set[int], list[tuple[str, int]]]:
    """Observe real native allocation/FIT/close without replacing numerical work."""
    from n4m import MultimodalPipeline

    created: list[Any] = []
    live: set[int] = set()
    events: list[tuple[str, int]] = []
    create, fit, close = MultimodalPipeline.__init__, MultimodalPipeline.fit, MultimodalPipeline.close

    def observed_create(self: Any, *args: Any, **kwargs: Any) -> None:
        create(self, *args, **kwargs)
        created.append(self)
        live.add(id(self))

    def observed_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = fit(self, *args, **kwargs)
        events.append(("fit-complete", id(self)))
        return result

    def observed_close(self: Any) -> None:
        close(self)
        live.discard(id(self))
        events.append(("close", id(self)))

    monkeypatch.setattr(MultimodalPipeline, "__init__", observed_create)
    monkeypatch.setattr(MultimodalPipeline, "fit", observed_fit)
    monkeypatch.setattr(MultimodalPipeline, "close", observed_close)
    return created, live, events


@pytest.mark.parametrize("failure", ["nonfinite_raw", "metadata_numeric", "shape", "native_pca_rows", "after_native_fit"])
def test_direct_native_failed_refit_preserves_previous_predictor(monkeypatch: Any, failure: str) -> None:
    from n4m import MultimodalPipeline

    created, live, _ = _observe_direct_native_lifecycle(monkeypatch)
    cohort, model = example.make_cohort(), _model(transformers__image__n_components=4)
    train = np.flatnonzero(np.asarray(cohort.partitions) == "train")
    test = np.flatnonzero(np.asarray(cohort.partitions) == "test")
    schemas = source_schemas_from_cohort(cohort)
    expected = _oracle(model, cohort, train, test)[0]
    try:
        model.fit(list(_blocks(cohort, train).values()), cohort.y[train], source_schemas=schemas)
        previous = model.native_pipeline_
        previous_handle = previous._handle.value
        state = previous.export_state()
        prediction = model.predict(list(_blocks(cohort, test).values()), source_schemas=schemas)
        np.testing.assert_allclose(prediction, expected, rtol=2e-7, atol=2e-7)
        identities = {name: getattr(model, name) for name in ("source_schemas_", "source_names_", "input_shapes_", "target_ndim_", "n_outputs_")}
        invalid = [np.array(block, copy=True) for block in _blocks(cohort, train).values()]
        targets = cohort.y[train]
        if failure == "nonfinite_raw":
            invalid[0][0, 0] = np.nan
        elif failure == "metadata_numeric":
            invalid[3][0, 0] = "not-a-number"
        elif failure == "shape":
            invalid[1] = invalid[1][:, :7]
        elif failure == "native_pca_rows":
            invalid = [block[:2] for block in invalid]
            targets = targets[:2]  # Actual native PCA count=4 exceeds FIT rows.
        else:
            real_fit = MultimodalPipeline.fit

            def fail_after_real_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
                real_fit(self, *args, **kwargs)
                assert self.export_state()[:4] == b"N4MF"
                raise RuntimeError("injected after real candidate native FIT")

            monkeypatch.setattr(MultimodalPipeline, "fit", fail_after_real_fit)
        with pytest.raises(Exception):
            model.fit(invalid, targets, source_schemas=schemas)
        assert len(created) == 2 and live == {id(previous)}
        assert model.native_pipeline_ is previous and previous._handle.value == previous_handle
        assert all(getattr(model, name) is identity for name, identity in identities.items())
        assert previous.export_state() == state
        np.testing.assert_array_equal(model.predict(list(_blocks(cohort, test).values()), source_schemas=schemas), prediction)
        with pytest.raises(RuntimeError, match="closed"):
            created[1].export_state()
    finally:
        model.close()
        model.close()
    assert not live


def test_direct_native_successful_refit_replaces_shape_state_and_closes_old_handle(monkeypatch: Any) -> None:
    created, live, events = _observe_direct_native_lifecycle(monkeypatch)
    first, model = example.make_cohort(), _model()
    first_train = np.flatnonzero(np.asarray(first.partitions) == "train")
    first_schemas = source_schemas_from_cohort(first)
    second = example.make_cohort(23)
    second = _replace(second, "nir", second.sources["nir"].values[:, :12], axis_coordinates={"wavelength": np.linspace(900, 1700, 12)})
    second = _replace(second, "image", second.sources["image"].values[:, :5, :6])
    second = _replace(second, "series", second.sources["series"].values[:, :10],
                      axis_coordinates={"time": np.linspace(0, 2 * np.pi, 10), "variable": ["sensor_a", "sensor_b"]})
    second = _replace(second, "metadata", second.sources["metadata"].values, feature_names=["temperature", "device"])
    train = np.flatnonzero(np.asarray(second.partitions) == "train")
    test = np.flatnonzero(np.asarray(second.partitions) == "test")
    schemas = source_schemas_from_cohort(second)
    expected = _oracle(model, second, train, test)[0]
    try:
        model.fit(list(_blocks(first, first_train).values()), first.y[first_train], source_schemas=first_schemas)
        previous = model.native_pipeline_
        original_state = previous.export_state()
        model.fit(list(_blocks(second, train).values()), second.y[train, None], source_schemas=schemas)
        candidate = model.native_pipeline_
        assert len(created) == 2 and candidate is not previous and live == {id(candidate)}
        assert events.index(("fit-complete", id(candidate))) < events.index(("close", id(previous)))
        assert model.source_schemas_ == schemas and model.source_schemas_ is not schemas
        assert model.input_shapes_ == {name: tuple(source.values.shape[1:]) for name, source in second.sources.items()}
        assert model.target_ndim_ == 2 and model.n_outputs_ == 1
        assert candidate.export_state() != original_state
        prediction = model.predict(list(_blocks(second, test).values()), source_schemas=schemas)
        assert prediction.shape == (len(test), 1)
        np.testing.assert_allclose(prediction[:, 0], expected, rtol=2e-7, atol=2e-7)
        with pytest.raises(RuntimeError, match="closed"):
            previous.export_state()
        with pytest.raises(ValueError, match="input shape"):
            model.predict(list(_blocks(first, first_train).values()), source_schemas=first_schemas)
    finally:
        model.close()
        model.close()
    assert not live


def _run(pipeline: Any, cohort: Any, workspace: Path, **options: Any) -> Any:
    return nirs4all.run(pipeline, cohort, engine="dag-ml", verbose=0, save_charts=False, random_state=17,
                        workspace_path=workspace, **options)


@pytest.mark.parametrize("explicit_id", [None, "model:declared-u07"])
def test_native_declaration_preserves_compiler_resolved_model_identity(explicit_id: str | None) -> None:
    import dag_ml

    from nirs4all.data.multimodal import MultimodalSpectroDataset
    from nirs4all.pipeline.dagml.cli_runner import assemble_cv_refit_dsl
    from nirs4all.pipeline.dagml.envelope import build_envelope
    from nirs4all.pipeline.dagml.folds import _build_folds, _split_group_grain
    from nirs4all.pipeline.dagml.identity import mint_identity
    from nirs4all.pipeline.dagml_bridge import controller_manifests

    cohort = example.make_cohort()
    pipeline = example.make_pipeline(backend="methods")
    dataset = MultimodalSpectroDataset(cohort)
    pool = dataset.index_column("sample", {"partition": "train"})
    identity = mint_identity(dataset)
    folds = _build_folds(pipeline[0], dataset, pool, set())
    groups = _split_group_grain(pipeline[0], dataset, pool)
    envelope = build_envelope(dataset, identity, sample_ints=pool, group_by_sample=groups)
    dsl = assemble_cv_refit_dsl(pipeline[1:], identity, envelope, folds, dsl_id="identity-u07", n_splits=3)
    if explicit_id is not None:
        dsl["pipeline"][0]["id"] = explicit_id
        probe = deepcopy(dsl)
        probe.pop("data_bindings")
        compiled = dag_ml.compile_pipeline_dsl_artifact_with_controllers(probe, controller_manifests())
        resolved = next(node["id"] for node in compiled.graph.to_dict()["nodes"] if node["kind"] == "model")
        assert resolved == explicit_id
        dsl["data_bindings"][0]["node_id"] = resolved
    expected_id = dsl["data_bindings"][0]["node_id"]
    original = deepcopy(dsl)
    declaration = bind_methods_dsl(dsl, pipeline[-1]["model"], cohort)
    compiled = dag_ml.compile_pipeline_dsl_artifact_with_controllers(declaration["dsl"], [declaration["manifest"]])
    assert [node["id"] for node in compiled.graph.to_dict()["nodes"]] == [expected_id]
    bound_binding = declaration["dsl"]["data_bindings"][0]
    assert bound_binding["node_id"] == original["data_bindings"][0]["node_id"]
    assert bound_binding["source_ids"] == original["data_bindings"][0]["source_ids"]
    assert bound_binding["view_policy"]["include_augmented_train"] is False
    assert bound_binding["view_policy"]["include_refit_test_view"] is True
    assert dsl == original
    if explicit_id is not None:
        mismatched = deepcopy(dsl)
        mismatched["pipeline"][0]["id"] = "model:another-u07"
        with pytest.raises(ValueError, match="native data binding"):
            bind_methods_dsl(mismatched, pipeline[-1]["model"], cohort)


def test_public_grid_selects_independent_oracle_and_portable_archive(tmp_path: Path) -> None:
    cohort = example.make_cohort()
    pipeline = example.make_pipeline(backend="methods")
    train = np.flatnonzero(np.asarray(cohort.partitions) == "train")
    test = np.flatnonzero(np.asarray(cohort.partitions) == "test")
    folds = list(GroupKFold(3).split(train[:, None], groups=np.asarray(cohort.groups)[train]))
    choices = []
    for alpha, weight, components in itertools.product([0.1, 1.0], [0.5, 1.0], [2, 4]):
        model = _model(model__alpha=alpha, source_weights__image=weight, transformers__image__n_components=components)
        oof = np.empty(len(train))
        for fit, validation in folds:
            oof[validation] = _oracle(model, cohort, train[fit], train[validation])[0]
        choices.append((float(np.sqrt(np.mean((oof - cohort.y[train]) ** 2))), model, oof))
    expected_score, expected_model, _ = min(choices, key=lambda choice: choice[0])
    result = _run(pipeline, cohort, tmp_path / "workspace")
    try:
        assert result.native_profile == PROFILE
        np.testing.assert_allclose(result.cv_best_score, expected_score, rtol=2e-7, atol=2e-7)
        fit_events = [event for event in result.methods_multimodal_audit if event["operation"] == "fit"]
        # Native SELECT evaluates all candidates then reruns selected CV for
        # retained OOF closure, followed by exactly one full REFIT.
        assert len(fit_events) == 8 * 3 + 3 + 1
        for event in fit_events:
            assert not set(event["sample_ids"]).intersection(cohort.sample_ids[index] for index in test)
            if event["fold"] is not None:
                number = int(event["fold"][4:])
                assert set(event["sample_ids"]) == {cohort.sample_ids[train[index]] for index in folds[number][0]}
        test_rows = result.predictions.filter_predictions(partition="test", fold_id="final")
        assert len(test_rows) == 1
        test_indices = test_rows[0]["sample_indices"]
        assert len(test_indices) == len(test) and set(test_indices) == set(test)
        expected_test = _oracle(expected_model, cohort, train, test_indices)[0]
        np.testing.assert_allclose(np.asarray(test_rows[0]["y_pred"]).reshape(-1), expected_test, rtol=2e-7, atol=2e-7)
        np.testing.assert_array_equal(np.asarray(test_rows[0]["y_true"]).reshape(-1), cohort.y[test_indices])
        expected_test_rmse = float(np.sqrt(np.mean((expected_test - cohort.y[test_indices]) ** 2)))
        np.testing.assert_allclose(result.best_rmse, expected_test_rmse, rtol=2e-7, atol=2e-7)
        native_outcome = result._methods_multimodal_outcome.to_dict()
        test_reports = [report for report in native_outcome["score_set"]["reports"]
                        if report["partition"] == "test" and report.get("fold_id") is None
                        and report.get("variant_id") == native_outcome["selected_variant_id"]]
        assert len(test_reports) == 1
        np.testing.assert_allclose(test_reports[0]["metrics"]["rmse"], expected_test_rmse, rtol=2e-7, atol=2e-7)
        archive = result.export(tmp_path / "complete.n4a")
        package = read_portable_predictor_archive_v2(archive).to_dict()
        records = package["execution_bundle"]["refit_artifacts"]
        assert len(records) == 1 and records[0]["artifact"]["kind"] == "methods_multimodal_pipeline"
        assert records[0]["artifact"]["backend"] == "raw"
        assert "host_sidecar_payloads" not in package["execution_bundle"] or not package["execution_bundle"]["host_sidecar_payloads"]
        # Reading retains Core/DAG validation and gives no controller fit permission.
        prediction_rows = test[::-1]
        replay = nirs4all.predict(archive, cohort.take([cohort.sample_ids[index] for index in prediction_rows]), verbose=0)
        expected = _oracle(expected_model, cohort, train, prediction_rows)[0]
        np.testing.assert_allclose(np.asarray(replay.y_pred).reshape(-1), expected, rtol=2e-7, atol=2e-7)
        assert replay.metadata["training_performed"] is False
        assert replay.metadata["sample_ids"] == [cohort.sample_ids[index] for index in prediction_rows]
        predict_events = [event for event in replay.metadata["methods_multimodal_audit"] if event["operation"] == "PREDICT"]
        assert len(predict_events) == 1
        assert len(predict_events[0]["sample_ids"]) == len(replay.metadata["sample_ids"])
        assert set(predict_events[0]["sample_ids"]) == set(replay.metadata["sample_ids"])
    finally:
        result.close()


@pytest.mark.parametrize("dotted_paths", [False, True])
def test_same_native_hpo_proposals_and_scores_across_backends(tmp_path: Path, dotted_paths: bool) -> None:
    cohort = example.make_cohort()
    results = []
    try:
        for backend in ("sklearn", "methods"):
            pipeline = example.make_pipeline(backend=backend)
            space = pipeline[-1].pop("_grid_")
            if dotted_paths:
                space = {key.replace("__", "."): value for key, value in space.items()}
            results.append(_run(pipeline, cohort, tmp_path / backend, tuning={
                "engine": "n4m", "sampler": "random", "seed": 17, "n_trials": 4,
                "space": space, "storage": (tmp_path / f"search-{backend}").as_uri(), "study_name": "canonical",
            }))
        left, right = [result.tuning_result for result in results]
        assert [trial.params for trial in left.trials] == [trial.params for trial in right.trials]
        assert all(trial.state == "COMPLETE" for trial in right.trials)
        np.testing.assert_allclose([trial.value for trial in left.trials], [trial.value for trial in right.trials], rtol=2e-7, atol=2e-7)
        assert left.best_params == right.best_params
        native_request = results[1].methods_multimodal_search_request["request"]
        target = native_request["target_node"]
        assert native_request["parameter_bindings"] == {
            "model.alpha": {"node_id": target, "param_path": "model__alpha"},
            "source_weights.image": {"node_id": target, "param_path": "source_weights__image"},
            "transformers.image.n_components": {"node_id": target, "param_path": "transformers__image__n_components"},
        }
    finally:
        for result in results:
            result.close()


@pytest.mark.parametrize("generated_grid", [False, True])
def test_installed_fresh_replay_without_workspace_fit_or_hpo(tmp_path: Path, generated_grid: bool) -> None:
    python = os.environ.get("NIRS4ALL_U07_INSTALLED_PYTHON")
    assert python and Path(python).is_file(), "mandatory U07 installed replay requires NIRS4ALL_U07_INSTALLED_PYTHON"
    cohort = example.make_cohort()
    pipeline = example.make_pipeline(backend="methods")
    if not generated_grid:
        pipeline[-1].pop("_grid_")
    workspace = tmp_path / "workspace"
    result = _run(pipeline, cohort, workspace)
    archive = result.export(tmp_path / "complete.n4a")
    result.close()
    predict = example.make_cohort(29, prediction=True)
    expected = nirs4all.predict(archive, predict, verbose=0).y_pred
    declaration = tmp_path / "prediction.json"
    declaration.write_text(json.dumps(predict.to_dict(), ensure_ascii=False, allow_nan=False))
    shutil.rmtree(workspace)
    script = tmp_path / "replay.py"
    script.write_text('''import json, sys
import numpy as np
import nirs4all, n4m, dag_ml
from nirs4all_io import MultimodalDataset
def refuse(*args, **kwargs):
    raise AssertionError("FIT/HPO forbidden during installed archive replay")
n4m.MultimodalPipeline.fit = refuse
dag_ml.execute_training = refuse
dag_ml.run_host_hpo_search_in_process = refuse
data = MultimodalDataset.from_dict(json.loads(open(sys.argv[2]).read()))
result = nirs4all.predict(sys.argv[1], data, verbose=0)
assert result.metadata["training_performed"] is False
print(json.dumps({"values": np.asarray(result.y_pred).reshape(-1).tolist(), "module": nirs4all.__file__}))
''')
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    process = subprocess.run([python, "-I", str(script), str(archive), str(declaration)], cwd=tmp_path,
                             env=env, capture_output=True, text=True, timeout=120, check=True)
    evidence = json.loads(process.stdout.strip().splitlines()[-1])
    assert ".codex-targets" in evidence["module"] or "site-packages" in evidence["module"]
    np.testing.assert_array_equal(np.asarray(evidence["values"]), np.asarray(expected).reshape(-1))


def test_real_native_handles_close_on_fit_failure(monkeypatch: Any, tmp_path: Path) -> None:
    from n4m import MultimodalPipeline

    live: set[int] = set()
    create, fit, close = MultimodalPipeline.__init__, MultimodalPipeline.fit, MultimodalPipeline.close

    def observed_create(self: Any, *args: Any, **kwargs: Any) -> None:
        create(self, *args, **kwargs)
        live.add(id(self))

    def failed_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        fit(self, *args, **kwargs)
        raise RuntimeError("injected after real native encoder fit")

    def observed_close(self: Any) -> None:
        close(self)
        live.discard(id(self))

    monkeypatch.setattr(MultimodalPipeline, "__init__", observed_create)
    monkeypatch.setattr(MultimodalPipeline, "fit", failed_fit)
    monkeypatch.setattr(MultimodalPipeline, "close", observed_close)
    pipeline = example.make_pipeline(backend="methods")
    pipeline[-1].pop("_grid_")
    with pytest.raises(Exception, match="injected|callback|native"):
        _run(pipeline, example.make_cohort(), tmp_path / "workspace")
    assert not live


@pytest.mark.parametrize("change", ["dtype", "coordinates", "feature_names", "shape"])
def test_schema_refusals_happen_before_native_import(monkeypatch: Any, tmp_path: Path, change: str) -> None:
    from n4m import MultimodalPipeline

    cohort = example.make_cohort()
    pipeline = example.make_pipeline(backend="methods")
    pipeline[-1].pop("_grid_")
    result = _run(pipeline, cohort, tmp_path / "workspace")
    archive = result.export(tmp_path / "complete.n4a")
    result.close()
    current = example.make_cohort(29, prediction=True)
    if change == "dtype":
        current = _replace(current, "nir", current.sources["nir"].values.astype(np.float32))
    elif change == "coordinates":
        current = _replace(current, "nir", current.sources["nir"].values, axis_coordinates={"wavelength": np.linspace(901, 1701, 24)})
    elif change == "feature_names":
        current = _replace(current, "metadata", current.sources["metadata"].values, feature_names=["different", "category"])
    else:
        current = _replace(current, "image", current.sources["image"].values[:, :7], axis_coordinates={"channel": ["R", "G", "B"]})

    def refused_import(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("incompatible raw schema reached native hydration")

    monkeypatch.setattr(MultimodalPipeline, "from_state", refused_import)
    with pytest.raises(ValueError, match="schema|shape"):
        nirs4all.predict(archive, current, verbose=0)


def prepare_qualification(output: Path, seed: int = 17) -> dict[str, Any]:
    """Capture one actual public U07 grid and native eight-proposal search.

    Cross-language qualification drivers consume this capture rather than
    generating another data set or scheduling their own folds. This helper
    performs real training when invoked; it is never a collection-time fit.
    """
    output.mkdir(parents=True, exist_ok=True)
    cohort = example.make_cohort(seed)
    prediction = example.make_cohort(29, prediction=True)
    pipeline = example.make_pipeline(backend="methods")
    for name in ("image", "series"):
        pipeline[-1]["model"].transformers[name].random_state = seed
    result = _run(pipeline, cohort, output / "grid-workspace")
    try:
        import dag_ml

        archive = result.export(output / "complete.n4a")
        package = read_portable_predictor_archive_v2(archive).to_dict()
        plan = package["effective_plan"]
        graph = plan["graph_plan"]["graph"]
        operators = {node["id"]: node["operator"] for node in graph["nodes"]}
        node_params = {node_id: node["params"] for node_id, node in plan["node_plans"].items()}
        source_ids = next(iter(plan["node_plans"].values()))["data_bindings"][0]["source_ids"]
        schemas = source_schemas_from_cohort(cohort)
        prediction_schemas = source_schemas_from_cohort(prediction)
        replay_inputs: dict[str, Any] = {}
        replay = dag_ml.replay_loaded_predictor_package

        def observe_replay(package: Any, request: Any, envelopes: Any, caches: Any, *args: Any, **kwargs: Any) -> Any:
            replay_inputs.update(prediction_request=request.to_dict(), prediction_envelopes=deepcopy(envelopes),
                                 prediction_inputs=deepcopy(caches))
            return replay(package, request, envelopes, caches, *args, **kwargs)

        # Observe and delegate to actual native replay. This does not replace
        # numerical execution or independently manufacture input fingerprints.
        with patch.object(dag_ml, "replay_loaded_predictor_package", side_effect=observe_replay):
            expected_prediction = np.asarray(nirs4all.predict(archive, prediction, verbose=0).y_pred).reshape(-1).tolist()

        def wire_sources(data: Any, descriptors: Any) -> dict[str, Any]:
            wire = {}
            for name, source in data.sources.items():
                item = {"sample_ids": list(data.sample_ids), "descriptor": descriptors[name], "shape": list(source.values.shape)}
                if name == "metadata":
                    item["rows"] = source.values.tolist()
                else:
                    item["data"] = source.values.reshape(-1, order="C").tolist()
                wire[name] = item
            return wire

        capture = {
            "training_request": result.methods_multimodal_training_request,
            "training_inputs": result.methods_multimodal_training_inputs,
            "training_outcome": result._methods_multimodal_outcome.to_dict(),
            "package": package, "operators": operators, "node_params": node_params, "source_ids": source_ids,
            "sources": wire_sources(cohort, schemas),
            "targets": {"sample_ids": list(cohort.sample_ids), "values": np.asarray(cohort.y).reshape(-1, 1).tolist()},
            "target_names": list(cohort.target_names),
            "prediction_sources": wire_sources(prediction, prediction_schemas), "prediction_dataset": prediction.to_dict(),
            "expected_prediction": expected_prediction, **replay_inputs,
            "archive": str(archive.resolve()), "seed": seed,
        }
    finally:
        result.close()
    search_pipeline = example.make_pipeline(backend="methods")
    for name in ("image", "series"):
        search_pipeline[-1]["model"].transformers[name].random_state = seed
    space = search_pipeline[-1].pop("_grid_")
    search = _run(search_pipeline, cohort, output / "search-workspace", tuning={
        "engine": "n4m", "sampler": "random", "seed": 17, "n_trials": 8, "space": space,
        "storage": (output / "search-ledger").resolve().as_uri(), "study_name": "canonical-u07",
    })
    try:
        capture["search_request"] = search.methods_multimodal_search_request
        capture["search_result"] = search.methods_multimodal_tuning_evidence
    finally:
        search.close()
    destination = output / "canonical-u07.json"
    destination.write_text(json.dumps(capture, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return capture
