"""Native nested OOF and archive replay with explicit source availability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset, RaggedSeriesBatch
from sklearn.model_selection import GroupKFold

import nirs4all
from nirs4all.operators.transforms import SequenceSummary
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped
from tests.integration.api.test_multimodal_late_fusion import _pipeline
from tests.integration.api.test_multimodal_ragged import _ragged_cohort, _replace_series
from tests.integration.api.test_multimodal_tuning import _checkpoint, _stop_after, _tuning


def _late_pipeline(*, policy: str = "zero_with_indicator") -> list[Any]:
    pipeline = _pipeline()
    pipeline[1]["branch"]["missing_source_policy"] = policy
    pipeline[1]["branch"]["steps"]["series"][0] = SequenceSummary()
    pipeline[1]["branch"]["steps"]["image"][0].set_params(n_components=1)
    return pipeline


def _run(cohort: MultimodalDataset, directory: Path, **kwargs: Any) -> Any:
    pipeline = kwargs.pop("pipeline", _late_pipeline())
    return nirs4all.run(
        pipeline, cohort, engine="dag-ml", refit=True, save_artifacts=True,
        random_state=19, verbose=0, save_charts=False, workspace_path=directory, **kwargs,
    )


def _meta(result: Any) -> Any:
    return next(run for run in result.runs if any(item["controller_id"] == "controller:nirs4all.meta_model" for item in run._dagml_refit_artifacts))


@pytest.fixture(autouse=True)
def forbid_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("Legacy scheduler executed"))


def test_late_missing_series_filters_every_native_fit_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[frozenset[int]] = []
    original_fit = SequenceSummary.fit

    def observe(self: SequenceSummary, X: RaggedSeriesBatch, y: Any = None) -> Any:
        seen.append(frozenset(int(X[index][0, 0]) for index in range(len(X))))
        return original_fit(self, X, y)

    monkeypatch.setattr(SequenceSummary, "fit", observe)
    complete = _run(_ragged_cohort(), tmp_path / "complete")
    complete.close()
    expected_scopes = seen.copy()
    seen.clear()
    cohort = _ragged_cohort(missing=True)
    present = {index for index, value in enumerate(cohort.sources["series"].presence_mask) if value}
    result = _run(cohort, tmp_path / "partial")
    try:
        expected = [scope & present for scope in expected_scopes]
        assert sorted(seen, key=lambda ids: (len(ids), sorted(ids))) == sorted(expected, key=lambda ids: (len(ids), sorted(ids)))
        assert all(seen) and any(len(scope) < 8 for scope in seen)
        assert np.isfinite(_meta(result).cv_best_score)
        fitted = next(item["estimator"] for item in _meta(result)._dagml_refit_artifacts if item["controller_id"] == "controller:nirs4all.meta_model")
        assert fitted.n_features_in_ == 8
        assert fitted.multimodal_source_names == tuple(cohort.sources)
    finally:
        result.close()


def test_late_missing_source_hidden_values_never_affect_oof(tmp_path: Path) -> None:
    cohort = _ragged_cohort(missing=True)
    source = cohort.sources["series"]
    values, times = source.values.values.copy(), source.time_coordinates.copy()
    for row in np.flatnonzero(~source.presence_mask):
        start, end = source.offsets[row:row + 2]
        values[start:end] += 1e7
        times[start:end] += 1e6
    poisoned = _replace_series(cohort, values=values, time_coordinates=times)
    before = _run(cohort, tmp_path / "before")
    after = _run(poisoned, tmp_path / "poisoned")
    try:
        for left, right in zip(before.runs, after.runs, strict=True):
            for fold in range(3):
                expected = left.predictions.filter_predictions(fold_id=str(fold), partition="val", load_arrays=True)[0]
                actual = right.predictions.filter_predictions(fold_id=str(fold), partition="val", load_arrays=True)[0]
                assert expected["sample_indices"] == actual["sample_indices"]
                np.testing.assert_array_equal(expected["y_pred"], actual["y_pred"])
    finally:
        before.close()
        after.close()


@pytest.mark.parametrize("policy", ["error", "zero_with_indicator"])
def test_late_missing_source_requires_observations_in_every_fit(policy: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cohort = _ragged_cohort()
    cohort = _replace_series(cohort, presence_mask=np.zeros(len(cohort), dtype=bool))
    monkeypatch.setattr(SequenceSummary, "fit", lambda *a, **k: pytest.fail("Unavailable series reached fitting"))
    message = "complete modalities" if policy == "error" else "no observed rows"
    with pytest.raises(Exception, match=message):
        _run(cohort, tmp_path, pipeline=_late_pipeline(policy=policy))


def test_late_missing_source_refuses_an_empty_inner_training_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cohort = _ragged_cohort()
    groups = np.asarray(cohort.groups)[:12]
    outer = list(GroupKFold(3).split(np.zeros((12, 1)), groups=groups))
    train = outer[0][0]
    inner = next(GroupKFold(2).split(np.zeros((len(train), 1)), groups=groups[train]))[0]
    presence = np.ones(len(cohort), dtype=bool)
    presence[train[inner]] = False
    assert all(presence[rows].any() for rows, _ in outer)
    original_fit = SequenceSummary.fit

    def observed_only(self: SequenceSummary, X: RaggedSeriesBatch, y: Any = None) -> Any:
        assert len(X) > 0
        assert all(presence[int(X[index][0, 0])] for index in range(len(X)))
        return original_fit(self, X, y)

    monkeypatch.setattr(SequenceSummary, "fit", observed_only)
    with pytest.raises(Exception, match="no observed rows in the native training view"):
        _run(_replace_series(cohort, presence_mask=presence), tmp_path)


def test_missing_source_meta_oof_never_uses_outer_validation_targets(tmp_path: Path) -> None:
    cohort = _ragged_cohort(missing=True)
    validation = next(GroupKFold(3).split(np.zeros((12, 1)), groups=np.asarray(cohort.groups)[:12]))[1]
    targets = cohort.y.copy()
    targets[validation] += 1e6
    poisoned = MultimodalDataset(cohort.sources, sample_ids=cohort.sample_ids, y=targets,
                                 groups=cohort.groups, partitions=cohort.partitions, name=cohort.name, task_type="regression")
    before, after = _run(cohort, tmp_path / "before"), _run(poisoned, tmp_path / "after")
    try:
        for left, right in zip(before.runs, after.runs, strict=True):
            expected = left.predictions.filter_predictions(fold_id="0", partition="val", load_arrays=True)[0]
            actual = right.predictions.filter_predictions(fold_id="0", partition="val", load_arrays=True)[0]
            np.testing.assert_array_equal(expected["y_pred"], actual["y_pred"])
    finally:
        before.close()
        after.close()


def _late_tuning(study: Path, **options: Any) -> dict[str, Any]:
    return {**_tuning(study), "n_trials": 3,
            "space": {"branches.series.0.include_length": [True, False], "meta.alpha": [0.1, 1.0]}, **options}


def test_missing_source_whole_stack_hpo_resumes_exactly(tmp_path: Path) -> None:
    cohort = _ragged_cohort(missing=True)
    study = tmp_path / "study"
    with pytest.raises(MultimodalTuningStopped):
        _run(cohort, tmp_path / "stopped", tuning=_late_tuning(study, progress_callback=_stop_after(1, [])))
    prefix = _checkpoint(study)["native_checkpoint"]["trials"]
    resumed = _run(cohort, tmp_path / "resumed", tuning=_late_tuning(study, resume=True))
    continuous = _run(cohort, tmp_path / "continuous", tuning=_late_tuning(tmp_path / "continuous-study"))
    try:
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [trial.to_dict() for trial in continuous.tuning_result.trials]
        assert resumed.tuning_best_params == continuous.tuning_best_params
        assert _checkpoint(study)["native_checkpoint"]["trials"][:1] == prefix
        assert all(trial.diagnostics["test_used"] is False for trial in resumed.tuning_result.trials)
    finally:
        resumed.close()
        continuous.close()


def test_missing_source_hpo_binds_presence_and_ignores_hidden_buffers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cohort = _ragged_cohort(missing=True)
    study = tmp_path / "study"
    with pytest.raises(MultimodalTuningStopped):
        _run(cohort, tmp_path / "stopped", tuning=_late_tuning(study, progress_callback=_stop_after(1, [])))
    checkpoint = study / "multimodal.n4mopt.json"
    before = checkpoint.read_bytes()
    with monkeypatch.context() as scoped:
        scoped.setattr(SequenceSummary, "fit", lambda *a, **k: pytest.fail("Changed availability reached fitting"))
        presence = cohort.sources["series"].presence_mask.copy()
        presence[1] = True
        with pytest.raises(Exception, match="(?i)(checkpoint|fingerprint|mismatch|contract)"):
            _run(_replace_series(cohort, presence_mask=presence), tmp_path / "changed", tuning=_late_tuning(study, resume=True))
    assert checkpoint.read_bytes() == before
    source = cohort.sources["series"]
    values = source.values.values.copy()
    values[source.offsets[1]:source.offsets[2]] += 1e8
    hidden = _replace_series(cohort, values=values)
    resumed = _run(hidden, tmp_path / "hidden", tuning=_late_tuning(study, resume=True))
    try:
        assert len(resumed.tuning_result.trials) == 3
    finally:
        resumed.close()


def test_missing_source_branch_cannot_consume_other_raw_modalities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nirs4all.operators.models.multimodal import MultimodalRegressor
    from tests.integration.api.test_multimodal_ragged import _ragged_model

    pipeline = _late_pipeline()
    pipeline[1]["branch"]["steps"]["series"] = [_ragged_model(missing=True)]
    monkeypatch.setattr(MultimodalRegressor, "fit", lambda *a, **k: pytest.fail("A source-bound base saw every raw modality"))
    with pytest.raises(Exception, match="single-source base models"):
        _run(_ragged_cohort(missing=True), tmp_path, pipeline=pipeline)


@pytest.mark.parametrize("outputs", [1, 2])
def test_dense_missing_sources_and_target_transform_replay(outputs: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from tests.integration.api.test_multimodal_dagml import _cohort

    def cohort(*, prediction: bool = False) -> MultimodalDataset:
        data = _cohort(prediction=prediction, unequal_groups=not prediction)
        sources = dict(data.sources)
        presence = np.zeros(len(data), dtype=bool) if prediction else np.arange(len(data)) % 4 != 1
        sources["series"] = replace(sources["series"], presence_mask=presence)
        # A second missing modality has a different availability pattern.
        sources["image"] = replace(sources["image"], presence_mask=np.arange(len(data)) % 4 != 2)
        y = None if prediction else data.y if outputs == 1 else np.column_stack([data.y, 100 + data.y * 3])
        return MultimodalDataset(sources, sample_ids=data.sample_ids, y=y, groups=data.groups,
                                 partitions=data.partitions, task_type="regression", name=data.name)

    pipeline = _pipeline()
    pipeline[1]["branch"]["missing_source_policy"] = "zero_with_indicator"
    for source in ("series", "image"):
        pipeline[1]["branch"]["steps"][source][0].set_params(n_components=1)
    pipeline[1]["branch"]["steps"]["series"][-1] = TransformedTargetRegressor(regressor=Ridge(), transformer=StandardScaler())
    result = _run(cohort(), tmp_path / "training", pipeline=pipeline)
    try:
        meta = _meta(result)
        fitted = next(item["estimator"] for item in meta._dagml_refit_artifacts if item["controller_id"] == "controller:nirs4all.meta_model")
        assert fitted.n_features_in_ == 4 * (outputs + 1)
        archive = meta.export(tmp_path / "dense.n4a")
    finally:
        result.close()
    monkeypatch.setattr(TransformedTargetRegressor, "predict", lambda *a, **k: pytest.fail("Unavailable source reached target-transform prediction"))
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("Replay trained a model"))
    prediction = nirs4all.predict(archive, cohort(prediction=True))
    assert np.asarray(prediction.values).shape == ((5,) if outputs == 1 else (5, 2))
    assert np.isfinite(prediction.values).all()
