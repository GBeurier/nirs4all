"""Raw source presence through native folds, durable search and fitted archives."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.operators.models.multimodal import MultimodalClassifier, MultimodalRegressor, TensorPCA
from nirs4all.operators.models.sklearn.mbpls import MBPLS
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped
from tests.integration.api.test_multimodal_dagml import _cohort as complete_cohort
from tests.integration.api.test_multimodal_dagml import _model as complete_model
from tests.integration.api.test_multimodal_late_fusion import _pipeline as late_pipeline
from tests.integration.api.test_multimodal_partial_targets import _assert_masked_metrics, _predictions


def _cohort(*, hidden: float = np.nan, partial_targets: bool = False, classification: bool = False, prediction: bool = False) -> MultimodalDataset:
    original = complete_cohort(prediction=prediction, unequal_groups=True)
    rows = np.arange(len(original))
    masks = {"nir": rows % 5 != 0, "image": rows % 4 != 1, "series": rows % 5 != 2, "metadata": rows % 5 != 3}
    sources = {}
    for name, source in original.sources.items():
        values = np.array(source.values, copy=True)
        values[~masks[name]] = [hidden, f"hidden-only-{hidden}"] if name == "metadata" else hidden
        sources[name] = replace(source, values=values, presence_mask=masks[name])
    targets = original.y
    target_mask = None
    target_names = ["condition"] if classification else ["response"]
    if classification and targets is not None:
        targets = np.asarray(["healthy", "mild", "severe"])[rows % 3]
    if partial_targets:
        target_names = ["sugar", "protein"]
        if targets is not None:
            targets = np.column_stack([targets, -1.7 * targets + np.cos(rows)])
            target_mask = np.column_stack([rows % 2 == 0, rows % 2 == 1])
            targets[~target_mask] = hidden
    return MultimodalDataset(
        sources, sample_ids=original.sample_ids, y=targets, target_names=target_names,
        target_mask=target_mask, task_type="classification" if classification else "regression",
        groups=original.groups, partitions=original.partitions, name="missing-raw-modalities",
    )


def _updated(cohort: MultimodalDataset, sources: dict[str, Any], **changes: Any) -> MultimodalDataset:
    options = {
        "sample_ids": cohort.sample_ids, "y": cohort.y, "target_names": cohort.target_names,
        "target_mask": cohort.target_mask, "task_type": cohort.task_type,
        "groups": cohort.groups, "partitions": cohort.partitions, "name": cohort.name,
    }
    options.update(changes)
    return MultimodalDataset(sources, **options)


def _model(*, fusion: str = "early", partial_targets: bool = False, classification: bool = False) -> Any:
    model = complete_model().set_params(
        missing_source_policy="zero_with_indicator", target_policy="per_target" if partial_targets else "complete",
        transformers__image__n_components=1, transformers__series__n_components=1,
    )
    if classification:
        return MultimodalClassifier(
            model.transformers, LogisticRegression(max_iter=300), source_weights=model.source_weights,
            missing_source_policy="zero_with_indicator",
        )
    if fusion == "intermediate":
        model.set_params(fusion=fusion, model=MBPLS(n_components=1, standardize=False))
    return model


def _run(cohort: MultimodalDataset, workspace: Path, *, model: Any = None, tuning: dict[str, Any] | None = None, pipeline: Any = None) -> Any:
    return nirs4all.run(
        [GroupKFold(3), {"model": _model() if model is None else model}] if pipeline is None else pipeline,
        cohort, tuning=tuning, engine="dag-ml", workspace_path=workspace,
        refit=True, save_artifacts=True, save_charts=False, verbose=0, random_state=19,
    )


@pytest.mark.parametrize("fusion", ["early", "intermediate"])
@pytest.mark.parametrize("partial_targets", [False, True])
def test_encoders_see_only_present_source_rows_in_each_grouped_target_fold(
    fusion: str, partial_targets: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    cohort = _cohort(partial_targets=partial_targets)
    fits: dict[str, list[frozenset[int]]] = {name: [] for name in cohort.sources}
    tensor_fit, scaler_fit, table_fit = TensorPCA.fit, StandardScaler.fit, ColumnTransformer.fit_transform
    nir_rows = {tuple(row): index for index, row in enumerate(cohort.sources["nir"].values) if cohort.sources["nir"].presence_mask[index]}

    def observe(name: str, rows: list[int], y: Any) -> None:
        fits[name].append(frozenset(rows))
        assert np.all(cohort.sources[name].presence_mask[rows])
        expected = cohort.y[rows, 0 if all(row % 2 == 0 for row in rows) else 1] if partial_targets else cohort.y[rows]
        np.testing.assert_allclose(np.asarray(y).ravel(), expected, rtol=1e-6, atol=1e-7)

    def observe_tensor(self: TensorPCA, X: Any, y: Any = None) -> TensorPCA:
        values = np.asarray(X)
        assert values.ndim in (3, 4)
        observe("image" if values.ndim == 4 else "series", [int(value) for value in values.reshape(len(values), -1)[:, 0]], y)
        return tensor_fit(self, X, y)

    def observe_scaler(self: StandardScaler, X: Any, y: Any = None, sample_weight: Any = None) -> StandardScaler:
        values = np.asarray(X)
        if values.shape[1] == 6:
            observe("nir", [nir_rows[tuple(row)] for row in values], y)
        return scaler_fit(self, X, y, sample_weight=sample_weight)

    def observe_table(self: ColumnTransformer, X: Any, y: Any = None, **params: Any) -> Any:
        observe("metadata", [round(float(value) * len(cohort)) for value in np.asarray(X)[:, 0]], y)
        return table_fit(self, X, y, **params)

    monkeypatch.setattr(TensorPCA, "fit", observe_tensor)
    monkeypatch.setattr(StandardScaler, "fit", observe_scaler)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", observe_table)
    result = _run(cohort, tmp_path / "workspace", model=_model(fusion=fusion, partial_targets=partial_targets))
    try:
        folds = list(GroupKFold(3).split(np.zeros((12, 1)), groups=cohort.groups[:12]))
        for train, validation in folds:
            assert set(cohort.groups[train]).isdisjoint(cohort.groups[validation])
        pools = [train for train, _ in folds] + [np.arange(12)]
        for name, source in cohort.sources.items():
            expected_rows = []
            for pool in pools:
                observed = cohort.target_mask[pool] if partial_targets else np.ones((len(pool), 1), dtype=bool)
                for target in range(observed.shape[1]):
                    expected_rows.append(frozenset(pool[observed[:, target] & source.presence_mask[pool]].tolist()))
            assert Counter(fits[name]) == Counter(expected_rows), name
            assert all(rows.isdisjoint(range(12, 16)) for rows in fits[name])
        if partial_targets:
            _assert_masked_metrics(result, cohort)
        assert np.isfinite(result.cv_best_score)
    finally:
        result.close()


@pytest.mark.parametrize("fusion", ["early", "intermediate"])
def test_hidden_numeric_and_categorical_source_values_never_affect_native_predictions_or_scores(fusion: str, tmp_path: Path) -> None:
    baseline = _run(_cohort(), tmp_path / "baseline", model=_model(fusion=fusion))
    poisoned = _run(_cohort(hidden=-1e100), tmp_path / "poisoned", model=_model(fusion=fusion))
    try:
        assert baseline._dagml_score_set == poisoned._dagml_score_set
        expected, actual = _predictions(baseline), _predictions(poisoned)
        assert expected.keys() == actual.keys()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
    finally:
        baseline.close()
        poisoned.close()


@pytest.mark.parametrize("classification", [False, True])
def test_archive_reorders_sources_and_predicts_with_empty_image_source_without_fit(
    classification: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    cohort = _cohort(classification=classification)
    result = _run(cohort, tmp_path / "workspace", model=_model(classification=classification))
    prediction = _cohort(prediction=True, classification=classification)
    sources = dict(reversed(list(prediction.sources.items())))
    sources["image"] = replace(sources["image"], values=np.empty((0, 2, 2, 3)), sample_ids=[], presence_mask=np.empty(0, dtype=bool))
    prediction = _updated(prediction, sources, source_alignment="left")
    assert not prediction.sources["image"].presence_mask.any()
    try:
        artifact = result._dagml_refit_artifacts[0]
        estimator = artifact["estimator"]
        blocks = [prediction.sources[name].values for name in estimator.source_names]
        masks = prediction.source_presence()
        expected = estimator.predict(blocks, source_masks=masks)
        if classification:
            probabilities = estimator.predict_proba(blocks, source_masks=masks)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
            np.testing.assert_array_equal(estimator.classes_[probabilities.argmax(axis=1)], expected)
            expected = artifact["y_transform"].decode(expected.reshape(-1, 1)).ravel()
        archive = result.export(tmp_path / "missing-sources.n4a")
    finally:
        result.close()
    shutil.rmtree(tmp_path / "workspace")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("partial-source archive attempted fit")

    for operator in (MultimodalClassifier, MultimodalRegressor, TensorPCA, Ridge, LogisticRegression, StandardScaler, ColumnTransformer):
        monkeypatch.setattr(operator, "fit", forbidden)
    replay = nirs4all.predict(archive, MultimodalDataset.from_dict(prediction.to_dict()))
    np.testing.assert_array_equal(replay.y_pred, expected)
    assert replay.metadata["training_performed"] is False
    assert replay.metadata["target_names"] == list(cohort.target_names)


@pytest.mark.parametrize("unsupported", ["default", "late", "upstream", "empty_training_source"])
def test_missing_source_unsupported_paths_fail_explicitly(unsupported: str, tmp_path: Path) -> None:
    cohort = _cohort()
    pipeline = None
    model = _model()
    if unsupported == "default":
        model.set_params(missing_source_policy="error")
    elif unsupported == "late":
        pipeline = late_pipeline()
    elif unsupported == "upstream":
        pipeline = [GroupKFold(3), StandardScaler(), {"model": model}]
    else:
        sources = dict(cohort.sources)
        sources["image"] = replace(sources["image"], presence_mask=np.zeros(len(cohort), dtype=bool))
        cohort = _updated(cohort, sources)
    with pytest.raises(Exception, match="(?i)(partial modalities|missing_source_policy|complete modalities|present training row)"):
        _run(cohort, tmp_path / "workspace", model=model, pipeline=pipeline)


def _tuning(directory: Path, *, resume: bool = False) -> dict[str, Any]:
    return {
        "engine": "n4m", "sampler": "random", "seed": 19, "n_trials": 3,
        "space": {"model__alpha": [0.1, 1.0], "source_weights__image": [0.5, 1.0]},
        "storage": directory.as_uri(), "study_name": "missing", "resume": resume,
    }


@pytest.mark.parametrize("change_test_presence", [False, True])
def test_search_resume_ignores_hidden_source_values_and_test_presence(change_test_presence: bool, tmp_path: Path) -> None:
    directory = tmp_path / "resumed-study"
    with pytest.raises(MultimodalTuningStopped):
        _run(_cohort(), tmp_path / "interrupted", tuning={
            **_tuning(directory), "progress_callback": lambda event: len(event["checkpoint"]["trials"]) < 1,
        })
    checkpoint = directory / "missing.n4mopt.json"
    prefix = json.loads(checkpoint.read_text())["native_checkpoint"]["trials"]
    changed = _cohort(hidden=-1e100)
    if change_test_presence:
        sources = dict(changed.sources)
        mask = np.array(sources["image"].presence_mask, copy=True)
        mask[12] = False
        sources["image"] = replace(sources["image"], presence_mask=mask)
        changed = _updated(changed, sources)
    resumed = _run(changed, tmp_path / "resumed", tuning=_tuning(directory, resume=True))
    continuous_directory = tmp_path / "continuous-study"
    continuous = _run(_cohort(), tmp_path / "continuous", tuning=_tuning(continuous_directory))
    try:
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [trial.to_dict() for trial in continuous.tuning_result.trials]
        saved = json.loads(checkpoint.read_text())["native_checkpoint"]["trials"]
        assert len(saved) == 3 and saved[:1] == prefix
        assert saved == json.loads((continuous_directory / "missing.n4mopt.json").read_text())["native_checkpoint"]["trials"]
        assert resumed.tuning_best_params == continuous.tuning_best_params
    finally:
        resumed.close()
        continuous.close()


def test_resume_refuses_changed_training_presence_before_fit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    directory = tmp_path / "study"
    with pytest.raises(MultimodalTuningStopped):
        _run(_cohort(), tmp_path / "interrupted", tuning={
            **_tuning(directory), "progress_callback": lambda event: len(event["checkpoint"]["trials"]) < 1,
        })
    checkpoint = directory / "missing.n4mopt.json"
    before = checkpoint.read_bytes()
    cohort = _cohort()
    sources = dict(cohort.sources)
    mask = np.array(sources["image"].presence_mask, copy=True)
    mask[0] = False
    sources["image"] = replace(sources["image"], presence_mask=mask)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("changed training presence reached fit")

    monkeypatch.setattr(MultimodalRegressor, "fit", forbidden)
    with pytest.raises(Exception, match="(?i)(checkpoint|fingerprint|mismatch|contract)"):
        _run(_updated(cohort, sources), tmp_path / "invalid", tuning=_tuning(directory, resume=True))
    assert checkpoint.read_bytes() == before
