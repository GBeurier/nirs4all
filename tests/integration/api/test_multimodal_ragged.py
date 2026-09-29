"""Variable-length raw series through native CV, masks, tuning and replay."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import DataProvider, MultimodalDataset, RaggedSeriesBatch, RaggedSeriesSource
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.operators.models.multimodal import MultimodalRegressor
from nirs4all.operators.models.sklearn.mbpls import MBPLS
from nirs4all.operators.transforms import SequenceSummary
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped
from tests.integration.api.test_multimodal_dagml import _cohort, _model
from tests.integration.api.test_multimodal_tuning import _checkpoint, _stop_after, _tuning


def _ragged_cohort(*, prediction: bool = False, missing: bool = False, partial_targets: bool = False) -> MultimodalDataset:
    cohort = _cohort(prediction=prediction, unequal_groups=not prediction)
    n = len(cohort.sample_ids)
    lengths = np.arange(n) % 5 + (9 if prediction else 2)
    offsets = np.r_[0, lengths.cumsum()]
    values = np.concatenate([np.tile(cohort.sources["series"].values[index, 1], (length, 1)) for index, length in enumerate(lengths)])
    for index, offset in enumerate(offsets[:-1]):
        values[offset, 0] = index
    times = np.concatenate([np.arange(length, dtype=float) / 4 for length in lengths])
    presence = np.ones(n, dtype=bool)
    if missing:
        presence[np.arange(n) % 4 == 1] = False
    series = RaggedSeriesSource(values, offsets, cohort.sample_ids, time_coordinates=times,
                                channel_names=["sensor_a", "sensor_b"], time_unit="s", presence_mask=presence)
    targets, mask = cohort.y, None
    if partial_targets and not prediction:
        targets = np.column_stack([cohort.y, -0.5 * cohort.y])
        mask = np.ones(targets.shape, dtype=bool)
        mask[1::4, 0], mask[2::4, 1] = False, False
    return MultimodalDataset({**cohort.sources, "series": series}, sample_ids=cohort.sample_ids,
                             y=targets, groups=cohort.groups, partitions=cohort.partitions,
                             task_type="regression", target_mask=mask, name=cohort.name)


def _ragged_model(*, fusion: str = "early", missing: bool = False, partial_targets: bool = False) -> Any:
    model = _model()
    model.set_params(transformers__series=Pipeline([("summary", SequenceSummary()), ("scale", StandardScaler())]))
    if fusion == "intermediate":
        model.set_params(fusion=fusion, model=MBPLS(n_components=2))
    return model.set_params(missing_source_policy="zero_with_indicator" if missing else "error",
                            target_policy="per_target" if partial_targets else "complete")


def _run(cohort: MultimodalDataset, directory: Path, **kwargs: Any) -> Any:
    model = kwargs.pop("model", _ragged_model())
    return nirs4all.run([GroupKFold(3), model], cohort, random_state=19, verbose=0,
                        save_charts=False, workspace_path=directory, **kwargs)


def _replace_series(cohort: MultimodalDataset, **changes: Any) -> MultimodalDataset:
    source = cohort.sources["series"]
    block = RaggedSeriesSource(**{
        "values": source.values.values, "offsets": source.offsets, "sample_ids": source.sample_ids,
        "time_coordinates": source.time_coordinates, "channel_names": source.channel_names,
        "time_unit": source.time_unit, "presence_mask": source.presence_mask, **changes,
    })
    return MultimodalDataset({**cohort.sources, "series": block}, sample_ids=cohort.sample_ids,
                             y=cohort.y, target_mask=cohort.target_mask, groups=cohort.groups,
                             partitions=cohort.partitions, name=cohort.name, task_type=cohort.task_type)


def _shorten_series(cohort: MultimodalDataset, row: int) -> MultimodalDataset:
    source = cohort.sources["series"]
    point = source.offsets[row + 1] - 1
    offsets = source.offsets.copy()
    offsets[row + 1:] -= 1
    return _replace_series(cohort, values=np.delete(source.values.values, point, axis=0), offsets=offsets,
                           time_coordinates=np.delete(source.time_coordinates, point))


@pytest.fixture(autouse=True)
def native_scheduler_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("Legacy scheduler executed"))


@pytest.mark.parametrize("fusion", ["early", "intermediate"])
def test_ragged_native_folds_and_archive_accept_new_lengths(fusion: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cohort = _ragged_cohort()
    seen: list[frozenset[int]] = []
    original_fit = SequenceSummary.fit

    def observe(self: SequenceSummary, X: Any, y: Any = None) -> Any:
        assert isinstance(X, RaggedSeriesBatch)
        seen.append(frozenset(int(X[index][0, 0]) for index in range(len(X))))
        return original_fit(self, X, y)

    monkeypatch.setattr(SequenceSummary, "fit", observe)
    result = _run(cohort, tmp_path / "workspace", model=_ragged_model(fusion=fusion))
    try:
        groups = np.asarray(cohort.groups)[:12]
        expected = [frozenset(train) for train, _ in GroupKFold(3).split(np.zeros((12, 1)), groups=groups)] + [frozenset(range(12))]
        assert sorted(seen, key=lambda ids: (len(ids), sorted(ids))) == sorted(expected, key=lambda ids: (len(ids), sorted(ids)))
        assert np.isfinite(result.best_rmse)
        archive = result.export(tmp_path / "ragged.n4a")
        with zipfile.ZipFile(archive) as bundle:
            schema = json.loads(bundle.read("manifest.json"))["multimodal_host"]["input_schema"]["series"]
        assert schema["shape"] == [None, None, 2]
        assert schema["native_representation"]["ragged"] is True
        monkeypatch.setattr(SequenceSummary, "fit", lambda *a, **k: pytest.fail("Replay fitted a series encoder"))
        prediction = nirs4all.predict(archive, _ragged_cohort(prediction=True))
        assert np.asarray(prediction.values).shape == (5,)
        assert np.isfinite(prediction.values).all()
    finally:
        result.close()


def test_generated_ragged_views_reach_each_grouped_model_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fold-specific lengths and times must reach the actual four-source model."""
    base = _ragged_cohort()
    views: list[tuple[str, str | None, tuple[str, ...], RaggedSeriesBatch]] = []
    fits: list[RaggedSeriesBatch] = []
    predictions: list[RaggedSeriesBatch] = []
    original_fit = MultimodalRegressor.fit
    original_predict = MultimodalRegressor.predict

    def generate(**_: Any) -> dict[str, Any]:
        return {"sample_ids": list(base.sample_ids), "sources": {"series": base.sources["series"]}}

    def generate_view(*, sample_ids: list[str], context: dict[str, Any], **_: Any) -> dict[str, Any]:
        scope = context["_dag_ml_view"]
        series = _shorten_series(base.take(sample_ids), 0).sources["series"]
        views.append((scope["partition"], scope["fold_id"], tuple(sample_ids), series.values))
        return {"sample_ids": sample_ids, "sources": {"series": series}}

    def observe_fit(self: MultimodalRegressor, X: list[Any], y: Any, **kwargs: Any) -> Any:
        assert set(self.transformers) == set(base.sources)
        series = X[list(self.transformers).index("series")]
        assert isinstance(series, RaggedSeriesBatch)
        fits.append(series)
        return original_fit(self, X, y, **kwargs)

    def observe_predict(self: MultimodalRegressor, X: list[Any], **kwargs: Any) -> Any:
        series = X[list(self.transformers).index("series")]
        assert isinstance(series, RaggedSeriesBatch)
        predictions.append(series)
        return original_predict(self, X, **kwargs)

    monkeypatch.setattr(MultimodalRegressor, "fit", observe_fit)
    monkeypatch.setattr(MultimodalRegressor, "predict", observe_predict)
    provider = DataProvider(
        generate, generate_view=generate_view, provider_id="qualification.view.ragged",
        base=base, replace_sources=["series"],
    )
    result = nirs4all.run(
        [GroupKFold(3), {"model": _ragged_model()}], provider,
        engine="dag-ml", refit=True, save_artifacts=False, save_charts=False,
        results_path=tmp_path / "native", random_state=19, verbose=0,
    )
    try:
        assert np.isfinite(result.best_rmse)
        assert len(fits) == 4
        assert {partition for partition, _fold, _ids, _series in views} >= {"fold_train", "fold_validation", "full_train"}
        groups_by_id = dict(zip(base.sample_ids, base.groups, strict=True))
        for fold in {fold for _partition, fold, _ids, _series in views if fold is not None}:
            train = {groups_by_id[sid] for partition, view_fold, ids, _series in views
                     if view_fold == fold and partition == "fold_train" for sid in ids}
            validation = {groups_by_id[sid] for partition, view_fold, ids, _series in views
                          if view_fold == fold and partition == "fold_validation" for sid in ids}
            assert train and validation and train.isdisjoint(validation)
        for partition, _fold, ids, expected in views:
            calls = fits if partition in {"fold_train", "full_train"} else predictions
            assert any(
                np.array_equal(actual.offsets, expected.offsets)
                and np.array_equal(actual.values, expected.values)
                and np.array_equal(actual.time_coordinates, expected.time_coordinates)
                for actual in calls
            ), f"No model call consumed the {partition} ragged view for {ids}"
        assert len(result._dagml_refit_artifacts) == 1
        assert result._dagml_generated_prediction_contract["mode"] == "explicit_cohort_predict_only"
    finally:
        result.close()


def test_ragged_masks_keep_only_observed_source_target_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cohort = _ragged_cohort(missing=True, partial_targets=True)
    source = cohort.sources["series"]
    assert isinstance(source, RaggedSeriesSource)
    seen: list[frozenset[int]] = []
    original_fit = SequenceSummary.fit

    def observe(self: SequenceSummary, X: RaggedSeriesBatch, y: Any = None) -> Any:
        seen.append(frozenset(int(X[index][0, 0]) for index in range(len(X))))
        return original_fit(self, X, y)

    monkeypatch.setattr(SequenceSummary, "fit", observe)
    result = _run(cohort, tmp_path, model=_ragged_model(missing=True, partial_targets=True))
    try:
        expected = []
        fits = [train for train, _ in GroupKFold(3).split(np.zeros((12, 1)), groups=np.asarray(cohort.groups)[:12])] + [np.arange(12)]
        for rows in fits:
            for target in range(2):
                expected.append(frozenset(index for index in rows if source.presence_mask[index] and cohort.target_mask[index, target]))
        assert sorted(seen, key=lambda ids: (len(ids), sorted(ids))) == sorted(expected, key=lambda ids: (len(ids), sorted(ids)))
        prediction = nirs4all.predict(result.export(tmp_path / "partial.n4a"), _ragged_cohort(prediction=True, missing=True))
        assert np.asarray(prediction.values).shape == (5, 2)
    finally:
        result.close()


def test_ragged_hash_binds_times_and_lengths_but_excludes_absent_buffers() -> None:
    cohort = _ragged_cohort(missing=True)
    source = cohort.sources["series"]
    assert isinstance(source, RaggedSeriesSource)
    original = MultimodalSpectroDataset(cohort).content_hash()

    def changed(*, row: int, times: bool = False) -> str:
        values, coordinates = source.values.values.copy(), source.time_coordinates.copy()
        start, end = source.offsets[row:row + 2]
        if times:
            coordinates[start:end] += 1
        else:
            values[start:end] += 100
        block = RaggedSeriesSource(values, source.offsets, source.sample_ids, time_coordinates=coordinates,
                                   channel_names=source.channel_names, time_unit="s", presence_mask=source.presence_mask)
        updated = MultimodalDataset({**cohort.sources, "series": block}, sample_ids=cohort.sample_ids,
                                    y=cohort.y, groups=cohort.groups, partitions=cohort.partitions, name=cohort.name, task_type="regression")
        return MultimodalSpectroDataset(updated).content_hash()

    assert changed(row=0) != original
    assert changed(row=0, times=True) != original
    assert MultimodalSpectroDataset(_shorten_series(cohort, 0)).content_hash() != original
    assert changed(row=1) == original
    assert changed(row=1, times=True) == original
    assert MultimodalSpectroDataset(_shorten_series(cohort, 1)).content_hash() == original


def test_ragged_tuning_resumes_identically_and_replays_selected_encoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cohort = _ragged_cohort()
    study = tmp_path / "study"

    def config(directory: Path, **options: Any) -> dict[str, Any]:
        return {**_tuning(directory), "space": {"model__alpha": [0.1, 1.0],
                "transformers__series__summary__include_length": [True, False]}, "n_trials": 3, **options}

    with pytest.raises(MultimodalTuningStopped):
        _run(cohort, tmp_path / "stopped", tuning=config(study, progress_callback=_stop_after(1, [])))
    prefix = _checkpoint(study)["native_checkpoint"]["trials"]
    resumed = _run(cohort, tmp_path / "resumed", tuning=config(study, resume=True))
    continuous = _run(cohort, tmp_path / "continuous", tuning=config(tmp_path / "continuous-study"))
    try:
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [trial.to_dict() for trial in continuous.tuning_result.trials]
        assert _checkpoint(study)["native_checkpoint"]["trials"][:1] == prefix
        assert resumed.tuning_best_params == continuous.tuning_best_params
        first, second = resumed.export(tmp_path / "resumed.n4a"), continuous.export(tmp_path / "continuous.n4a")
        monkeypatch.setattr(SequenceSummary, "fit", lambda *a, **k: pytest.fail("HPO replay fitted an encoder"))
        new = _ragged_cohort(prediction=True)
        np.testing.assert_array_equal(nirs4all.predict(first, new).values, nirs4all.predict(second, new).values)
    finally:
        resumed.close()
        continuous.close()


@pytest.mark.parametrize("mutation", ["times", "lengths"])
def test_ragged_resume_refuses_changed_observed_series_before_fit(mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cohort = _ragged_cohort()
    study = tmp_path / "study"
    config = {**_tuning(study), "space": {"model__alpha": [0.1, 1.0]}}
    with pytest.raises(MultimodalTuningStopped):
        _run(cohort, tmp_path / "stopped", tuning={**config, "progress_callback": _stop_after(1, [])})
    checkpoint = study / "multimodal.n4mopt.json"
    before = checkpoint.read_bytes()
    if mutation == "lengths":
        changed = _shorten_series(cohort, 0)
    else:
        times = cohort.sources["series"].time_coordinates.copy()
        times[:cohort.sources["series"].offsets[1]] += 1
        changed = _replace_series(cohort, time_coordinates=times)
    monkeypatch.setattr(SequenceSummary, "fit", lambda *a, **k: pytest.fail("Changed ragged input reached fitting"))
    with pytest.raises(Exception, match="(?i)(checkpoint|fingerprint|mismatch|contract)"):
        _run(changed, tmp_path / "invalid", tuning={**config, "resume": True})
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("mutation", ["channels", "units", "coordinates"])
def test_ragged_replay_rejects_incompatible_schema_before_encoding(mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run(_ragged_cohort(), tmp_path / "workspace")
    try:
        archive = result.export(tmp_path / "model.n4a")
    finally:
        result.close()
    changes = {"channels": {"channel_names": ["sensor_b", "sensor_a"]},
               "units": {"time_unit": "ms"}, "coordinates": {"time_coordinates": None}}[mutation]
    new = _replace_series(_ragged_cohort(prediction=True), **changes)
    monkeypatch.setattr(SequenceSummary, "transform", lambda *a, **k: pytest.fail("Invalid schema reached encoding"))
    with pytest.raises(ValueError, match="(?i)(schema|contract|mismatch)"):
        nirs4all.predict(archive, new)


def test_ragged_prediction_can_omit_every_series_with_explicit_missing_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run(_ragged_cohort(missing=True), tmp_path / "workspace", model=_ragged_model(missing=True))
    try:
        archive = result.export(tmp_path / "model.n4a")
    finally:
        result.close()
    new = _ragged_cohort(prediction=True)
    new = _replace_series(new, values=np.empty((0, 2)), offsets=np.zeros(len(new) + 1, dtype=int),
                          time_coordinates=np.empty(0), presence_mask=np.zeros(len(new), dtype=bool))
    monkeypatch.setattr(SequenceSummary, "transform", lambda *a, **k: pytest.fail("Missing series reached encoding"))
    prediction = nirs4all.predict(archive, new)
    assert np.isfinite(prediction.values).all()
    assert np.asarray(prediction.values).shape == (5,)


def test_ragged_late_fusion_preserves_inner_groups_and_replays_new_lengths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.integration.api.test_multimodal_late_fusion import _pipeline

    cohort = _ragged_cohort()
    pipeline = _pipeline()
    pipeline[1]["branch"]["steps"]["series"][0] = SequenceSummary()
    seen: list[frozenset[int]] = []
    original_fit = SequenceSummary.fit

    def observe(self: SequenceSummary, X: RaggedSeriesBatch, y: Any = None) -> Any:
        assert isinstance(X, RaggedSeriesBatch)
        seen.append(frozenset(int(X[index][0, 0]) for index in range(len(X))))
        return original_fit(self, X, y)

    monkeypatch.setattr(SequenceSummary, "fit", observe)
    result = nirs4all.run(pipeline, cohort, random_state=19, workspace_path=tmp_path / "workspace", verbose=0, save_charts=False)
    try:
        assert any(len(rows) < 8 for rows in seen)
        assert all(rows <= set(range(12)) for rows in seen)
        grouped = [{index for index in range(12) if cohort.groups[index] == group} for group in set(cohort.groups[:12])]
        assert all(not rows.intersection(group) or group <= rows for rows in seen for group in grouped)
        meta = next(run for run in result.runs if any(item["controller_id"] == "controller:nirs4all.meta_model" for item in run._dagml_refit_artifacts))
        archive = meta.export(tmp_path / "late.n4a")
        assert np.isfinite(meta.cv_best_score)
    finally:
        result.close()
    monkeypatch.setattr(SequenceSummary, "fit", lambda *a, **k: pytest.fail("Late replay fitted a series encoder"))
    prediction = nirs4all.predict(archive, _ragged_cohort(prediction=True))
    assert np.asarray(prediction.values).shape == (5,)
    assert np.isfinite(prediction.values).all()


def test_raw_multimodal_prediction_refuses_legacy_runner_before_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    predict_module = importlib.import_module("nirs4all.api.predict")
    monkeypatch.setattr(predict_module, "PipelineRunner", lambda *a, **k: pytest.fail("Raw modalities reached the legacy runner"))
    with pytest.raises(TypeError, match="captured DAG-ML artifact"):
        nirs4all.predict({"pipeline_id": "legacy"}, _ragged_cohort(prediction=True), engine="legacy")
