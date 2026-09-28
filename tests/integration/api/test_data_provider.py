"""Real native provider preparation, grouped training and independent replay."""

from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import DataProvider, MultimodalDataset, TensorSource
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.operators.models.multimodal import MultimodalClassifier, MultimodalRegressor, TensorPCA
from nirs4all.pipeline.dagml.cancellation import DagRunCancelled
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped
from nirs4all.pipeline.dagml.native_results import read_native_results
from tests.integration.api.test_multimodal_dagml import _cohort, _model
from tests.integration.api.test_multimodal_targets import _classification_cohort, _classifier
from tests.integration.api.test_multimodal_tuning import _checkpoint, _tuning


def _generate(*, seed: int, params: dict[str, Any], context: dict[str, Any]) -> MultimodalDataset:
    cohort = _cohort(unequal_groups=True)
    return MultimodalDataset(
        cohort.sources, sample_ids=cohort.sample_ids,
        y=cohort.y + np.random.default_rng(seed).normal(scale=params.get("noise", 0.01), size=len(cohort.sample_ids)),
        groups=cohort.groups, partitions=cohort.partitions, name=cohort.name,
    )


def _provider(**kwargs: Any) -> DataProvider:
    return DataProvider(_generate, provider_id="qualification.synthetic", seed=19, **kwargs)


def _run(provider: DataProvider, directory: Path, **kwargs: Any) -> Any:
    return nirs4all.run(
        [GroupKFold(3), _model()], provider, workspace_path=directory,
        random_state=19, verbose=0, save_charts=False, **kwargs,
    )


@pytest.fixture(autouse=True)
def no_legacy_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("Legacy scheduler executed"))


def test_native_source_executes_once_before_grouped_fits_and_survives_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def generate(**kwargs: Any) -> MultimodalDataset:
        calls.append(kwargs["seed"])
        return _generate(**kwargs)

    provider = DataProvider(generate, provider_id="qualification.synthetic", seed=19)
    fits: list[frozenset[int]] = []
    original_fit = TensorPCA.fit

    def observe(self: TensorPCA, X: Any, y: Any = None) -> Any:
        assert len(calls) == 1
        values = np.asarray(X)
        fits.append(frozenset(int(value) for value in values.reshape(len(values), -1)[:, 0]))
        return original_fit(self, X, y)

    monkeypatch.setattr(TensorPCA, "fit", observe)
    result = _run(provider, tmp_path / "workspace")
    try:
        assert len(calls) == 1
        cohort = provider.cohort
        groups = np.asarray(cohort.groups)[:12]
        expected = [frozenset(train) for train, _ in GroupKFold(3).split(np.zeros((12, 1)), groups=groups)] + [frozenset(range(12))]
        assert sorted(fits, key=lambda ids: (len(ids), sorted(ids))) == sorted(expected * 2, key=lambda ids: (len(ids), sorted(ids)))
        evidence = next(iter(result.per_dataset.values()))["data_provider_evidence"]
        assert evidence["execution"]["task_seed"] == calls[0]
        assert evidence["execution"]["metadata"]["content_fingerprint"] == provider.fingerprint
        wrapped = MultimodalSpectroDataset(cohort)
        assert pickle.loads(pickle.dumps(wrapped))._data_provider_evidence == evidence
        archive = result.export(tmp_path / "provider.n4a")
        with zipfile.ZipFile(archive) as bundle:
            manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["multimodal_host"]["data_provider"] == evidence
        monkeypatch.setattr(DataProvider, "materialize", lambda *a, **k: pytest.fail("Replay regenerated training data"))
        monkeypatch.setattr(TensorPCA, "fit", lambda *a, **k: pytest.fail("Replay fitted an encoder"))
        prediction = nirs4all.predict(archive, _cohort(prediction=True))
        assert len(prediction.values) == 5
        assert len(calls) == 1
    finally:
        result.close()


def test_partial_provider_adds_raw_modalities_to_fixed_base(tmp_path: Path) -> None:
    complete = _cohort()
    base = MultimodalDataset(
        {"nir": complete.sources["nir"]}, sample_ids=complete.sample_ids,
        y=complete.y, groups=complete.groups, partitions=complete.partitions, name=complete.name,
    )

    def generate(**kwargs: Any) -> dict[str, Any]:
        return {"sample_ids": complete.sample_ids, "sources": {key: value for key, value in complete.sources.items() if key != "nir"}}

    provider = DataProvider(generate, provider_id="qualification.partial", base=base)
    result = _run(provider, tmp_path)
    try:
        assert np.isfinite(result.best_rmse)
        assert set(provider.cohort.sources) == {"nir", "image", "series", "metadata"}
        np.testing.assert_array_equal(provider.cohort.sources["nir"].values, base.sources["nir"].values)
        np.testing.assert_array_equal(provider.cohort.y, base.y)
    finally:
        result.close()


def test_view_provider_refuses_before_eager_materialization_or_fit(tmp_path: Path) -> None:
    base = _cohort()

    def forbidden(**kwargs: Any) -> Any:
        pytest.fail("An unconnected fold-view provider executed")

    provider = DataProvider(
        forbidden,
        provider_id="qualification.view.pending",
        base=base,
        generate_view=forbidden,
    )
    with pytest.raises(NotImplementedError, match="fold-view and training-content attestation"):
        _run(provider, tmp_path)
    with pytest.raises(NotImplementedError, match="fold-view and training-content attestation"):
        nirs4all.run(
            [KFold(3, shuffle=True), {"model": Ridge()}], provider,
            engine="dag-ml", refit=True, save_artifacts=False,
            save_charts=False, verbose=0,
        )


def test_generated_views_fit_each_native_fold_and_persist_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = _cohort()
    base = MultimodalDataset(
        {"nir": original.sources["nir"]}, sample_ids=original.sample_ids,
        y=original.y, groups=original.groups, partitions=original.partitions,
        name=original.name,
    )
    generated_fits: list[np.ndarray] = []
    generated_predicts: list[np.ndarray] = []
    view_scopes: list[tuple[str, list[str], np.ndarray]] = []
    original_fit = Ridge.fit
    original_predict = Ridge.predict

    def generate(**_: Any) -> dict[str, Any]:
        return {"sample_ids": list(base.sample_ids), "sources": {"nir": base.sources["nir"]}}

    def generate_view(*, sample_ids: list[str], context: dict[str, Any], **_: Any) -> dict[str, Any]:
        partition = context["_dag_ml_view"]["partition"]
        offset = {"fold_train": 10.0, "fold_validation": 20.0, "full_train": 30.0, "predict": 40.0}[partition]
        source = base.take(sample_ids).sources["nir"]
        values = np.asarray(source.values) + offset
        view_scopes.append((partition, list(sample_ids), values.copy()))
        return {"sample_ids": sample_ids, "sources": {"nir": TensorSource(
            values, sample_ids, representation_id=source.representation_id,
            axis_units=source.axis_units, axis_coordinates=source.axis_coordinates,
        )}}

    def observe_fit(self: Ridge, X: Any, y: Any, **kwargs: Any) -> Any:
        generated_fits.append(np.asarray(X).copy())
        return original_fit(self, X, y, **kwargs)

    def observe_predict(self: Ridge, X: Any, **kwargs: Any) -> Any:
        generated_predicts.append(np.asarray(X).copy())
        return original_predict(self, X, **kwargs)

    monkeypatch.setattr(Ridge, "fit", observe_fit)
    monkeypatch.setattr(Ridge, "predict", observe_predict)
    provider = DataProvider(
        generate, generate_view=generate_view, provider_id="qualification.view.concrete",
        base=base, replace_sources=["nir"],
    )
    result = nirs4all.run(
        [KFold(3), {"model": Ridge(alpha=0.2)}], provider,
        engine="dag-ml", refit=True, save_artifacts=False, save_charts=False,
        results_path=tmp_path / "native", random_state=19, verbose=0,
    )
    try:
        expected_fits = [values for partition, _ids, values in view_scopes if partition in {"fold_train", "full_train"}]
        assert len(expected_fits) == len(generated_fits) == 4
        assert all(sum(np.array_equal(actual, expected) for actual in generated_fits) == 1 for expected in expected_fits)
        expected_validation = [values for partition, _ids, values in view_scopes if partition == "fold_validation"]
        assert len(expected_validation) == 3
        assert all(any(np.array_equal(actual, expected) for actual in generated_predicts)
                   for expected in expected_validation)
        assert {partition for partition, _ids, _values in view_scopes} >= {"fold_train", "fold_validation", "full_train"}
        assert np.isfinite(result.best_rmse)
        manifest = result._dagml_generated_view_manifest
        assert manifest["schema_version"] == 1
        assert len(manifest["views"]) == len(view_scopes)
        native = read_native_results(result._dagml_results_dir)
        assert native["generated_view_manifest"] == manifest
        assert native["artifacts"] == []
        assert native["manifest"]["capabilities"]["has_model_artifacts"] is False
        assert result._dagml_refit_artifacts == []
        assert result.to_rt_result().manifest["capabilities"]["has_model_artifacts"] is False
        for compatibility in (None, "legacy-refit"):
            with pytest.raises(Exception, match="generated data views have no replay contract"):
                result.export(tmp_path / "generated.n4a", compatibility=compatibility)
            with pytest.raises(Exception, match="generated data views have no replay contract"):
                result.export_model(tmp_path / "generated.joblib", compatibility=compatibility)
        assert not (tmp_path / "generated.n4a").exists()
        assert not (tmp_path / "generated.joblib").exists()
    finally:
        result.close()


@pytest.mark.parametrize("splitter", [KFold(3), GroupKFold(3)])
def test_generated_nir_views_fit_with_fixed_image_series_and_metadata(
    splitter: KFold | GroupKFold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fold can regenerate NIR without changing the other three PLAN sources."""
    base = _cohort(unequal_groups=type(splitter) is GroupKFold)
    view_scopes: list[tuple[str, str | None, list[str], np.ndarray]] = []
    fitted_blocks: list[tuple[tuple[str, ...], list[np.ndarray]]] = []
    predicted_blocks: list[tuple[tuple[str, ...], list[np.ndarray]]] = []
    original_fit = MultimodalRegressor.fit
    original_predict = MultimodalRegressor.predict

    def generate(**_: Any) -> dict[str, Any]:
        return {"sample_ids": list(base.sample_ids), "sources": {"nir": base.sources["nir"]}}

    def generate_view(*, sample_ids: list[str], context: dict[str, Any], **_: Any) -> dict[str, Any]:
        partition = context["_dag_ml_view"]["partition"]
        source = base.take(sample_ids).sources["nir"]
        values = np.asarray(source.values) + {
            "fold_train": 10.0, "fold_validation": 20.0, "full_train": 30.0, "predict": 40.0,
        }[partition]
        view_scopes.append((partition, context["_dag_ml_view"]["fold_id"], list(sample_ids), values.copy()))
        return {"sample_ids": sample_ids, "sources": {"nir": TensorSource(
            values, sample_ids, representation_id=source.representation_id,
            axis_units=source.axis_units, axis_coordinates=source.axis_coordinates,
        )}}

    def observe_fit(self: MultimodalRegressor, X: list[Any], y: Any, **kwargs: Any) -> Any:
        fitted_blocks.append((tuple(self.transformers), [np.asarray(block).copy() for block in X]))
        return original_fit(self, X, y, **kwargs)

    def observe_predict(self: MultimodalRegressor, X: list[Any], **kwargs: Any) -> Any:
        predicted_blocks.append((tuple(self.transformers), [np.asarray(block).copy() for block in X]))
        return original_predict(self, X, **kwargs)

    monkeypatch.setattr(MultimodalRegressor, "fit", observe_fit)
    monkeypatch.setattr(MultimodalRegressor, "predict", observe_predict)
    provider = DataProvider(
        generate, generate_view=generate_view, provider_id="qualification.view.multimodal",
        base=base, replace_sources=["nir"],
    )
    result = nirs4all.run(
        [splitter, {"model": _model()}], provider,
        engine="dag-ml", refit=True, save_artifacts=False, save_charts=False,
        results_path=tmp_path / "native", random_state=19, verbose=0,
    )
    try:
        assert np.isfinite(result.best_rmse)
        assert len(fitted_blocks) == 4
        assert {partition for partition, _fold, _ids, _values in view_scopes} >= {"fold_train", "fold_validation", "full_train"}
        if type(splitter) is GroupKFold:
            groups_by_id = dict(zip(base.sample_ids, base.groups, strict=True))
            by_fold = {
                fold: {
                    partition: {groups_by_id[sample_id] for sample_id in ids}
                    for partition, view_fold, ids, _values in view_scopes if view_fold == fold
                }
                for fold in {fold for _partition, fold, _ids, _values in view_scopes if fold is not None}
            }
            assert len(by_fold) == 3
            for partitions in by_fold.values():
                assert partitions["fold_train"].isdisjoint(partitions["fold_validation"])
            expected_groups = {groups_by_id[sample_id] for sample_id, partition in zip(base.sample_ids, base.partitions, strict=True) if partition == "train"}
            assert set.union(*(partitions["fold_validation"] for partitions in by_fold.values())) == expected_groups
        for partition, _fold, ids, nir in view_scopes:
            expected = base.take(ids)
            matches = [(names, blocks) for names, blocks in (fitted_blocks if partition in {"fold_train", "full_train"} else predicted_blocks)
                       if np.array_equal(blocks[names.index("nir")], nir)]
            assert matches, f"No model call consumed the generated {partition} view"
            for names, blocks in matches:
                assert set(names) == set(base.sources)
                for name, block in zip(names, blocks, strict=True):
                    np.testing.assert_array_equal(block, nir if name == "nir" else expected.sources[name].values)
        assert result._dagml_refit_artifacts == []
    finally:
        result.close()


@pytest.mark.parametrize("splitter", [StratifiedKFold(3), StratifiedGroupKFold(3)])
def test_generated_classification_views_keep_stratified_folds(
    splitter: StratifiedKFold | StratifiedGroupKFold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stratification uses PLAN targets while each model call consumes its own NIR view."""
    original = _classification_cohort()
    base = original if type(splitter) is StratifiedGroupKFold else MultimodalDataset(
        original.sources, sample_ids=original.sample_ids, y=original.y,
        partitions=original.partitions, name=original.name,
    )
    views: list[tuple[str, str | None, tuple[str, ...], np.ndarray]] = []
    fitted: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    original_fit = MultimodalClassifier.fit
    original_predict = MultimodalClassifier.predict

    def generate(**_: Any) -> dict[str, Any]:
        return {"sample_ids": list(base.sample_ids), "sources": {"nir": base.sources["nir"]}}

    def generate_view(*, sample_ids: list[str], context: dict[str, Any], **_: Any) -> dict[str, Any]:
        scope = context["_dag_ml_view"]
        source = base.take(sample_ids).sources["nir"]
        values = np.asarray(source.values) + {
            "fold_train": 10.0, "fold_validation": 20.0, "full_train": 30.0, "predict": 40.0,
        }[scope["partition"]]
        views.append((scope["partition"], scope["fold_id"], tuple(sample_ids), values))
        return {"sample_ids": sample_ids, "sources": {"nir": TensorSource(
            values, sample_ids, representation_id=source.representation_id,
            axis_units=source.axis_units, axis_coordinates=source.axis_coordinates,
        )}}

    def observe_fit(self: MultimodalClassifier, X: list[Any], y: Any, **kwargs: Any) -> Any:
        fitted.append(np.asarray(X[list(self.transformers).index("nir")]).copy())
        return original_fit(self, X, y, **kwargs)

    def observe_predict(self: MultimodalClassifier, X: list[Any], **kwargs: Any) -> Any:
        predicted.append(np.asarray(X[list(self.transformers).index("nir")]).copy())
        return original_predict(self, X, **kwargs)

    monkeypatch.setattr(MultimodalClassifier, "fit", observe_fit)
    monkeypatch.setattr(MultimodalClassifier, "predict", observe_predict)
    provider = DataProvider(
        generate, generate_view=generate_view, provider_id="qualification.view.classification",
        base=base, replace_sources=["nir"],
    )
    result = nirs4all.run(
        [splitter, {"model": _classifier()}], provider,
        engine="dag-ml", refit=True, save_artifacts=False, save_charts=False,
        results_path=tmp_path / "native", random_state=19, verbose=0,
    )
    try:
        assert result.best["task_type"] == "classification"
        assert len(fitted) == 4
        labels_by_id = dict(zip(base.sample_ids, base.y, strict=True))
        validation_by_fold = {
            fold: ids for partition, fold, ids, _values in views if partition == "fold_validation"
        }
        assert len(validation_by_fold) == 3
        assert all({labels_by_id[sid] for sid in ids} == set(base.y[:12]) for ids in validation_by_fold.values())
        assert set.union(*(set(ids) for ids in validation_by_fold.values())) == set(base.sample_ids[:12])
        if type(splitter) is StratifiedGroupKFold:
            groups_by_id = dict(zip(base.sample_ids, base.groups, strict=True))
            for fold, ids in validation_by_fold.items():
                train_groups = {groups_by_id[sid] for partition, view_fold, train_ids, _values in views
                                if view_fold == fold and partition == "fold_train" for sid in train_ids}
                assert train_groups.isdisjoint({groups_by_id[sid] for sid in ids})
        for partition, _fold, ids, values in views:
            calls = fitted if partition in {"fold_train", "full_train"} else predicted
            assert any(np.array_equal(call, values) for call in calls), f"No model call consumed the {partition} NIR view for {ids}"
        assert result._dagml_refit_artifacts == []
    finally:
        result.close()


def test_generated_view_manifest_repeats_for_same_seed_and_changes_for_new_seed(tmp_path: Path) -> None:
    """A fresh run can verify the same generated-view content without a saved callback."""
    base = _cohort()

    def provider() -> DataProvider:
        def generate(**_: Any) -> dict[str, Any]:
            return {"sample_ids": list(base.sample_ids), "sources": {"nir": base.sources["nir"]}}

        def generate_view(*, sample_ids: list[str], seed: int, **_: Any) -> dict[str, Any]:
            source = base.take(sample_ids).sources["nir"]
            values = np.asarray(source.values) + float(seed % 7)
            return {"sample_ids": sample_ids, "sources": {"nir": TensorSource(
                values, sample_ids, representation_id=source.representation_id,
                axis_units=source.axis_units, axis_coordinates=source.axis_coordinates,
            )}}

        return DataProvider(
            generate, generate_view=generate_view, provider_id="qualification.view.repeatable",
            base=base, replace_sources=["nir"],
        )

    def run(index: int, seed: int) -> tuple[dict[str, Any], float]:
        result = nirs4all.run(
            [KFold(3), {"model": _model()}], provider(), engine="dag-ml",
            refit=True, save_artifacts=False, save_charts=False,
            results_path=tmp_path / f"run-{index}", random_state=seed, verbose=0,
        )
        try:
            return result._dagml_generated_view_manifest, result.best_rmse
        finally:
            result.close()

    first, first_score = run(0, 19)
    repeated, repeated_score = run(1, 19)
    changed, _ = run(2, 20)
    assert first == repeated
    assert first_score == repeated_score
    assert first["fingerprint"] != changed["fingerprint"]


def test_generated_views_fit_sklearn_pipeline_preprocessor_on_train_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A concrete sklearn Pipeline may learn its scaler from generated training X."""
    cohort = _cohort()
    base = MultimodalDataset(
        {"nir": cohort.sources["nir"]}, sample_ids=cohort.sample_ids,
        y=cohort.y, groups=cohort.groups, partitions=cohort.partitions, name=cohort.name,
    )
    views: list[tuple[str, np.ndarray]] = []
    scaler_fits: list[np.ndarray] = []
    original_fit = StandardScaler.fit

    def generate(**_: Any) -> dict[str, Any]:
        return {"sample_ids": list(base.sample_ids), "sources": {"nir": base.sources["nir"]}}

    def generate_view(*, sample_ids: list[str], context: dict[str, Any], **_: Any) -> dict[str, Any]:
        partition = context["_dag_ml_view"]["partition"]
        source = base.take(sample_ids).sources["nir"]
        values = np.asarray(source.values) + {
            "fold_train": 10.0, "fold_validation": 20.0, "full_train": 30.0, "predict": 40.0,
        }[partition]
        views.append((partition, values.copy()))
        return {"sample_ids": sample_ids, "sources": {"nir": TensorSource(
            values, sample_ids, representation_id=source.representation_id,
            axis_units=source.axis_units, axis_coordinates=source.axis_coordinates,
        )}}

    def observe_fit(self: StandardScaler, X: Any, y: Any = None, **kwargs: Any) -> Any:
        scaler_fits.append(np.asarray(X).copy())
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", observe_fit)
    provider = DataProvider(
        generate, generate_view=generate_view, provider_id="qualification.view.sklearn-pipeline",
        base=base, replace_sources=["nir"],
    )
    result = nirs4all.run(
        [KFold(3), {"model": Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])}],
        provider, engine="dag-ml", refit=True, save_artifacts=False, save_charts=False,
        results_path=tmp_path / "native", random_state=19, verbose=0,
    )
    try:
        expected_fits = [values for partition, values in views if partition in {"fold_train", "full_train"}]
        assert len(expected_fits) == len(scaler_fits) == 4
        assert all(sum(np.array_equal(actual, expected) for actual in scaler_fits) == 1 for expected in expected_fits)
        assert np.isfinite(result.best_rmse)
    finally:
        result.close()


@pytest.mark.parametrize("random_state", [-1, 2**32])
def test_generated_view_rejects_invalid_run_seed_before_plan(random_state: int, tmp_path: Path) -> None:
    def forbidden(**_: Any) -> Any:
        pytest.fail("Invalid run seed executed the provider")

    base = _cohort()
    provider = DataProvider(
        forbidden, generate_view=forbidden, provider_id="qualification.view.invalid-seed",
        base=base, replace_sources=["nir"],
    )
    with pytest.raises(ValueError, match="non-negative 32-bit integer random_state"):
        nirs4all.run(
            [KFold(3), {"model": _model()}], provider,
            engine="dag-ml", refit=True, save_artifacts=False, save_charts=False,
            results_path=tmp_path / "native", random_state=random_state, verbose=0,
        )


def test_generated_view_rejects_subprocess_before_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(**_: Any) -> Any:
        pytest.fail("Subprocess mode executed the provider")

    base = _cohort()
    provider = DataProvider(
        forbidden, generate_view=forbidden, provider_id="qualification.view.subprocess-refused",
        base=base, replace_sources=["nir"],
    )
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "off")
    with pytest.raises(NotImplementedError, match="subprocess execution is not qualified"):
        nirs4all.run(
            [KFold(3), {"model": _model()}], provider,
            engine="dag-ml", refit=True, save_artifacts=False, save_charts=False,
            results_path=tmp_path / "native", random_state=19, verbose=0,
        )


@pytest.mark.parametrize("engine", ["native", "legacy", "dual"])
def test_unsupported_engine_refuses_before_provider_execution(engine: str, tmp_path: Path) -> None:
    def forbidden(**kwargs: Any) -> Any:
        pytest.fail("Unsupported engine executed the provider")

    with pytest.raises(ValueError, match="DataProvider requires"):
        _run(DataProvider(forbidden, provider_id="qualification.refused"), tmp_path, engine=engine)


def test_cancelled_run_refuses_before_provider_execution(tmp_path: Path) -> None:
    def forbidden(**kwargs: Any) -> Any:
        pytest.fail("Cancelled run executed the provider")

    with pytest.raises(DagRunCancelled, match="cancelled"):
        _run(DataProvider(forbidden, provider_id="qualification.cancelled"), tmp_path, should_stop=lambda: True)


def test_cancellation_during_generation_retains_public_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stop = False

    def generate(**kwargs: Any) -> MultimodalDataset:
        nonlocal stop
        stop = True
        return _generate(**kwargs)

    monkeypatch.setattr(TensorPCA, "fit", lambda *a, **k: pytest.fail("Cancelled provider reached training"))
    with pytest.raises(DagRunCancelled, match="cancelled"):
        _run(DataProvider(generate, provider_id="qualification.cancelled"), tmp_path, should_stop=lambda: stop)


def test_provider_hpo_resume_matches_continuous_and_binds_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "study"
    tuning = {**_tuning(directory), "n_trials": 2}
    with pytest.raises(MultimodalTuningStopped):
        _run(_provider(), tmp_path / "stopped", tuning={**tuning, "progress_callback": lambda event: len(event["checkpoint"]["trials"]) < 1})
    prefix = _checkpoint(directory)["native_checkpoint"]["trials"]
    resumed = _run(_provider(), tmp_path / "resumed", tuning={**tuning, "resume": True})
    continuous = _run(_provider(), tmp_path / "continuous", tuning={**_tuning(tmp_path / "other-study"), "n_trials": 2})
    try:
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [trial.to_dict() for trial in continuous.tuning_result.trials]
        assert _checkpoint(directory)["native_checkpoint"]["trials"][:1] == prefix
        assert resumed.tuning_best_params == continuous.tuning_best_params
        resumed_archive, continuous_archive = resumed.export(tmp_path / "resumed.n4a"), continuous.export(tmp_path / "continuous.n4a")
        np.testing.assert_allclose(nirs4all.predict(resumed_archive, _cohort(prediction=True)).values,
                                   nirs4all.predict(continuous_archive, _cohort(prediction=True)).values, rtol=0, atol=0)
        before = (directory / "multimodal.n4mopt.json").read_bytes()
        monkeypatch.setattr(TensorPCA, "fit", lambda *a, **k: pytest.fail("Changed provider recipe reached fit"))
        with pytest.raises(Exception, match="checkpoint .* binding mismatch"):
            _run(_provider(provider_version="2"), tmp_path / "changed", tuning={**tuning, "resume": True})
        assert (directory / "multimodal.n4mopt.json").read_bytes() == before
    finally:
        resumed.close()
        continuous.close()


@pytest.mark.parametrize("tune", [False, True])
def test_provider_lineage_is_retained_by_late_fusion_export(tune: bool, tmp_path: Path) -> None:
    from tests.integration.api.test_multimodal_late_fusion import _pipeline

    provider = _provider()
    options: dict[str, Any] = {
        "tuning": {
            **_tuning(tmp_path / "study"),
            "n_trials": 2,
            "space": {"meta.alpha": [0.1, 1.0]},
        }
    } if tune else {}
    result: Any = nirs4all.run(
        _pipeline(),
        provider,
        workspace_path=tmp_path / "workspace",
        random_state=19,
        verbose=0,
        save_charts=False,
        **options,
    )
    try:
        views = [result, *getattr(result, "runs", ())]
        evidence = getattr(provider.cohort, "_data_provider_evidence")
        assert all(
            metadata["data_provider_evidence"] == evidence
            for view in views
            for metadata in view.per_dataset.values()
        )
        ensemble = result if tune else next(
            view
            for view in result.runs
            if any(item.get("producer_node") == "merge:stack" for item in view.per_dataset.values())
        )
        archive = ensemble.export(tmp_path / "late.n4a")
        with zipfile.ZipFile(archive) as bundle:
            manifest = json.loads(bundle.read("manifest.json"))
        contract = manifest["multimodal_host"]
        assert contract["selected_model"]["fusion"] == "late_oof"
        assert contract["data_provider"] == evidence
        assert len(nirs4all.predict(archive, _cohort(prediction=True)).values) == 5
    finally:
        result.close()


def test_should_stop_ends_search_at_saved_trial_boundary(tmp_path: Path) -> None:
    stop = False
    directory = tmp_path / "study"

    def progress(event: dict[str, Any]) -> bool:
        nonlocal stop
        stop = len(event["checkpoint"]["trials"]) >= 1
        return True

    tuning = {**_tuning(directory), "n_trials": 2}
    with pytest.raises(DagRunCancelled, match="checkpoint saved"):
        _run(_provider(), tmp_path / "stopped", should_stop=lambda: stop,
             tuning={**tuning, "progress_callback": progress})
    saved = _checkpoint(directory)["native_checkpoint"]["trials"]
    assert len(saved) == 1 and saved[0]["state"] == "complete"
    resumed = _run(_provider(), tmp_path / "resumed", tuning={**tuning, "resume": True})
    try:
        assert len(resumed.tuning_result.trials) == 2
        assert _checkpoint(directory)["native_checkpoint"]["trials"][:1] == saved
    finally:
        resumed.close()
