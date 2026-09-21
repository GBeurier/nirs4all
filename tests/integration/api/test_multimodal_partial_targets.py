"""Partial labels remain explicit through real native CV, search and replay."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.operators.models.multimodal import MultimodalClassifier, MultimodalRegressor, TensorPCA
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped
from tests.integration.api.test_multimodal_dagml import _cohort, _model
from tests.integration.api.test_multimodal_late_fusion import _pipeline as _late_pipeline


def _partial_cohort(*, hidden: float = 1e12) -> MultimodalDataset:
    original = _cohort(unequal_groups=True)
    rows = np.arange(len(original))
    targets = np.column_stack([original.y, -1.7 * original.y + np.cos(rows)])
    mask = np.column_stack([rows % 2 == 0, rows % 2 == 1])
    targets[~mask] = hidden
    return MultimodalDataset(
        original.sources, sample_ids=original.sample_ids, y=targets,
        target_names=["sugar", "protein"], target_mask=mask, task_type="regression",
        groups=original.groups, partitions=original.partitions, name="partial-targets",
    )


def _updated(cohort: MultimodalDataset, **changes: Any) -> MultimodalDataset:
    values = {
        "sample_ids": cohort.sample_ids, "y": cohort.y, "target_names": cohort.target_names,
        "target_mask": cohort.target_mask, "task_type": cohort.task_type,
        "groups": cohort.groups, "partitions": cohort.partitions, "name": cohort.name,
    }
    values.update(changes)
    return MultimodalDataset(cohort.sources, **values)


def _run(cohort: MultimodalDataset, workspace: Path, *, model: Any = None, tuning: dict[str, Any] | None = None, late: bool = False) -> Any:
    estimator = _model().set_params(target_policy="per_target") if model is None else model
    return nirs4all.run(
        _late_pipeline() if late else [GroupKFold(3), {"model": estimator}], cohort, tuning=tuning,
        engine="dag-ml", workspace_path=workspace, refit=True, save_artifacts=True,
        save_charts=False, verbose=0, random_state=19,
    )


def _assert_masked_metrics(result: Any, cohort: MultimodalDataset) -> None:
    reports = {report["prediction_id"]: report for report in result._dagml_score_set["reports"] if report.get("fold_id") != "avg"}
    positions = {sample: index for index, sample in enumerate(cohort.sample_ids)}
    checked = 0
    validation_ids: list[str] = []
    for node in result._dagml_node_results:
        for block in node.get("predictions", []):
            report = reports.get(block["prediction_id"])
            if report is None:
                continue
            ids = block["sample_ids"]
            rows = [positions[sample] for sample in ids]
            target = next(target for target in node["regression_targets"] if [unit["id"] for unit in target["unit_ids"]] == ids)
            assert block["target_names"] == target["target_names"] == list(cohort.target_names)
            mask = np.asarray(target["validity_masks"], dtype=bool)
            np.testing.assert_array_equal(mask, cohort.target_mask[rows])
            observed, predicted = np.asarray(target["values"]), np.asarray(block["values"])
            assert observed.shape == predicted.shape == mask.shape == (len(rows), 2)
            np.testing.assert_allclose(observed[mask], cohort.y[rows][mask])
            assert report["row_count"] == len(rows)
            per_target_rmse = []
            for index, name in enumerate(cohort.target_names):
                valid = mask[:, index]
                assert valid.any()
                # Compare scoring against the actual native input values. The
                # host target manager may store the caller's float64 as float32.
                truth, pred = observed[:, index][valid], predicted[:, index][valid]
                expected = {
                    "rmse": root_mean_squared_error(truth, pred),
                    "mse": mean_squared_error(truth, pred),
                    "mae": mean_absolute_error(truth, pred),
                }
                for metric, value in expected.items():
                    assert report["metrics"][f"{metric}:{name}"] == pytest.approx(value, rel=1e-10, abs=1e-12)
                per_target_rmse.append(expected["rmse"])
            assert report["metrics"]["rmse"] == pytest.approx(np.mean(per_target_rmse), rel=1e-10, abs=1e-12)
            if report["partition"] == "validation":
                validation_ids.extend(ids)
            checked += 1
    assert checked == 5  # Three native OOF folds, full-train refit, held-out test.
    assert Counter(validation_ids) == Counter(cohort.sample_ids[:12])


def _predictions(result: Any) -> dict[str, np.ndarray]:
    return {
        block["prediction_id"]: np.asarray(block["values"])
        for node in result._dagml_node_results for block in node.get("predictions", [])
    }


def _tuning(directory: Path, *, resume: bool = False) -> dict[str, Any]:
    return {
        "engine": "n4m", "sampler": "random", "seed": 19, "n_trials": 2,
        "metric": "rmse", "direction": "minimize", "storage": directory.as_uri(),
        "study_name": "partial", "resume": resume,
        "space": {"model__alpha": [0.1, 1.0], "source_weights__image": [0.5, 1.0]},
    }


def _checkpoint(directory: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((directory / "partial.n4mopt.json").read_text(encoding="utf-8"))
    return payload


def test_native_partial_targets_fit_each_complete_source_chain_on_observed_fold_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cohort = _partial_cohort()
    fits: dict[str, list[frozenset[int]]] = {name: [] for name in cohort.sources}
    tensor_fit, scaler_fit, table_fit = TensorPCA.fit, StandardScaler.fit, ColumnTransformer.fit_transform
    nir_rows = {tuple(row): index for index, row in enumerate(cohort.sources["nir"].values)}

    def observe_tensor(self: TensorPCA, X: Any, y: Any = None) -> TensorPCA:
        values = np.asarray(X)
        name = "image" if values.ndim == 4 else "series"
        fits[name].append(frozenset(int(value) for value in values.reshape(len(values), -1)[:, 0]))
        return tensor_fit(self, X, y)

    def observe_scaler(self: StandardScaler, X: Any, y: Any = None, sample_weight: Any = None) -> StandardScaler:
        values = np.asarray(X)
        if values.shape[1] == 6:
            fits["nir"].append(frozenset(nir_rows[tuple(row)] for row in values))
        return scaler_fit(self, X, y, sample_weight=sample_weight)

    def observe_table(self: ColumnTransformer, X: Any, y: Any = None, **params: Any) -> Any:
        fits["metadata"].append(frozenset(round(float(value) * len(cohort)) for value in np.asarray(X)[:, 0]))
        return table_fit(self, X, y, **params)

    monkeypatch.setattr(TensorPCA, "fit", observe_tensor)
    monkeypatch.setattr(StandardScaler, "fit", observe_scaler)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", observe_table)
    result = _run(cohort, tmp_path / "training")
    try:
        expected = []
        for train, validation in GroupKFold(3).split(np.zeros((12, 1)), groups=cohort.groups[:12]):
            assert set(cohort.groups[train]).isdisjoint(cohort.groups[validation])
            for target in range(2):
                expected.append(frozenset(train[cohort.target_mask[train, target]].tolist()))
        expected.extend(frozenset(np.flatnonzero(cohort.target_mask[:12, target]).tolist()) for target in range(2))
        for name, observed in fits.items():
            assert Counter(observed) == Counter(expected), name
            assert all(rows.isdisjoint(range(12, 16)) for rows in observed)
        fitted = result._dagml_refit_artifacts[0]["estimator"]._model
        np.testing.assert_array_equal(fitted.target_counts_, [6, 6])
        assert len(fitted.target_models_) == 2 and not hasattr(fitted, "transformers_")
        first, second = fitted.target_models_
        assert first.model_ is not second.model_
        for source in cohort.sources:
            assert first.transformers_[source] is not second.transformers_[source]
        for target, child in enumerate(fitted.target_models_):
            rows = np.flatnonzero(cohort.target_mask[:12, target])
            assert child.transformers_["image"].pca_.n_samples_ == len(rows)
            assert child.transformers_["series"].pca_.n_samples_ == len(rows)
            np.testing.assert_allclose(child.transformers_["nir"].mean_, cohort.sources["nir"].values[rows].mean(axis=0), atol=1e-12)
        _assert_masked_metrics(result, cohort)
    finally:
        result.close()


def test_hidden_finite_extremes_and_nan_never_change_native_predictions_or_scores(tmp_path: Path) -> None:
    baseline = _run(_partial_cohort(hidden=-1e100), tmp_path / "finite")
    hidden_nan = _run(_partial_cohort(hidden=np.nan), tmp_path / "nan")
    try:
        expected, actual = _predictions(baseline), _predictions(hidden_nan)
        assert expected.keys() == actual.keys()
        for prediction_id in expected:
            np.testing.assert_array_equal(actual[prediction_id], expected[prediction_id])
        assert baseline._dagml_score_set == hidden_nan._dagml_score_set
        assert baseline.cv_best_score == hidden_nan.cv_best_score
        assert baseline.best_rmse == hidden_nan.best_rmse
    finally:
        baseline.close()
        hidden_nan.close()


def test_partial_target_archive_replays_new_ids_without_fitting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "training"
    result = _run(_partial_cohort(hidden=np.nan), workspace)
    prediction = _cohort(prediction=True)
    prediction = _updated(prediction, target_names=["sugar", "protein"], task_type="regression")
    captured = result._dagml_refit_artifacts[0]["estimator"]
    expected = captured.predict([prediction.sources[name].values for name in captured.source_names])
    archive = result.export(tmp_path / "partial-targets.n4a")
    result.close()
    shutil.rmtree(workspace)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("partial-target archive attempted to fit an encoder or estimator")

    for estimator in (MultimodalRegressor, TensorPCA, Ridge, StandardScaler):
        monkeypatch.setattr(estimator, "fit", forbidden)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", forbidden)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", forbidden)
    replay = nirs4all.predict(archive, prediction)
    assert replay.y_pred.shape == (len(prediction), 2)
    np.testing.assert_allclose(replay.y_pred, expected, atol=1e-10, rtol=1e-10)
    assert replay.metadata["target_names"] == ["sugar", "protein"]
    assert replay.metadata["sample_ids"] == list(prediction.sample_ids)
    assert replay.metadata["training_performed"] is False
    assert replay.metadata["artifact_integrity_verified"] is True
    assert replay.metadata["scores"] is None


@pytest.mark.parametrize("unsupported", ["complete", "classifier", "empty_target", "implicit_task", "late"])
def test_unsupported_partial_target_requests_are_rejected_before_any_encoder_fit(unsupported: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cohort = _partial_cohort()
    model: Any = _model().set_params(target_policy="per_target")
    if unsupported == "complete":
        model.set_params(target_policy="complete")
    elif unsupported == "classifier":
        cohort = _updated(cohort, y=np.arange(len(cohort)) % 2, target_mask=cohort.target_mask[:, 0], target_names=["class"], task_type="classification")
        model = MultimodalClassifier(_model().transformers, LogisticRegression())
    elif unsupported == "empty_target":
        mask = np.array(cohort.target_mask, copy=True)
        mask[:12, 0] = False
        cohort = _updated(cohort, target_mask=mask)
    elif unsupported == "implicit_task":
        cohort = _updated(cohort, task_type=None)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("unsupported partial targets reached a source encoder fit")

    for estimator in (TensorPCA, StandardScaler, Ridge, LogisticRegression):
        monkeypatch.setattr(estimator, "fit", forbidden)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", forbidden)
    with pytest.raises(Exception, match="(?i)(partial|target|observed|mask)"):
        _run(cohort, tmp_path / "refused", model=model, late=unsupported == "late")


def test_native_scoring_refuses_a_validation_target_with_no_observed_labels(tmp_path: Path) -> None:
    cohort = _partial_cohort()
    _, validation = next(GroupKFold(3).split(np.zeros((12, 1)), groups=cohort.groups[:12]))
    mask = np.array(cohort.target_mask, copy=True)
    mask[validation, 0] = False
    # Each training fold still has enough observed rows for its source PCAs.
    for train, _ in GroupKFold(3).split(np.zeros((12, 1)), groups=cohort.groups[:12]):
        assert (mask[train].sum(axis=0) >= 2).all()
    with pytest.raises(Exception, match="(?i)(observed|valid|empty).*(target|label)|(target|label).*(observed|valid|empty)"):
        _run(_updated(cohort, target_mask=mask), tmp_path / "empty-score")


def test_partial_target_search_resume_ignores_test_labels_masks_and_hidden_training_values(tmp_path: Path) -> None:
    cohort = _partial_cohort()
    directory = tmp_path / "resumed-study"
    with pytest.raises(MultimodalTuningStopped):
        _run(cohort, tmp_path / "interrupted", tuning={
            **_tuning(directory), "progress_callback": lambda event: len(event["checkpoint"]["trials"]) < 1,
        })
    prefix = _checkpoint(directory)["native_checkpoint"]["trials"]
    assert len(prefix) == 1 and prefix[0]["state"] == "complete"
    targets, masks = np.array(cohort.y, copy=True), np.array(cohort.target_mask, copy=True)
    targets[:12][~masks[:12]] = np.nan
    targets[12:] += 1e8
    masks[12:] = ~masks[12:]
    resumed = _run(_updated(cohort, y=targets, target_mask=masks), tmp_path / "resumed", tuning=_tuning(directory, resume=True))
    continuous_directory = tmp_path / "continuous-study"
    continuous = _run(cohort, tmp_path / "continuous", tuning=_tuning(continuous_directory))
    try:
        assert resumed.tuning_result is not None and continuous.tuning_result is not None
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [trial.to_dict() for trial in continuous.tuning_result.trials]
        assert all(trial.diagnostics["test_used"] is False for trial in resumed.tuning_result.trials)
        assert resumed.tuning_best_value == continuous.tuning_best_value
        assert resumed.tuning_best_params == continuous.tuning_best_params
        saved = _checkpoint(directory)["native_checkpoint"]["trials"]
        assert saved[:1] == prefix
        assert saved == _checkpoint(continuous_directory)["native_checkpoint"]["trials"]
        assert resumed.best_rmse != pytest.approx(continuous.best_rmse)
        for result in (resumed, continuous):
            np.testing.assert_array_equal(result._dagml_refit_artifacts[0]["estimator"]._model.target_counts_, [6, 6])
        prediction = _cohort(prediction=True)
        first = resumed._dagml_refit_artifacts[0]["estimator"]
        second = continuous._dagml_refit_artifacts[0]["estimator"]
        np.testing.assert_array_equal(
            first.predict([prediction.sources[name].values for name in first.source_names]),
            second.predict([prediction.sources[name].values for name in second.source_names]),
        )
    finally:
        resumed.close()
        continuous.close()


@pytest.mark.parametrize("mutation", ["mask", "task", "names"])
def test_resuming_changed_partial_target_contract_is_refused_before_fit(mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cohort = _partial_cohort()
    directory = tmp_path / "study"
    with pytest.raises(MultimodalTuningStopped):
        _run(cohort, tmp_path / "interrupted", tuning={
            **_tuning(directory), "progress_callback": lambda event: len(event["checkpoint"]["trials"]) < 1,
        })
    checkpoint = directory / "partial.n4mopt.json"
    before = checkpoint.read_bytes()
    if mutation == "mask":
        mask = np.array(cohort.target_mask, copy=True)
        mask[0, 1] = True
        changed = _updated(cohort, target_mask=mask)
    elif mutation == "task":
        changed = _updated(cohort, task_type="classification")
    else:
        changed = _updated(cohort, target_names=["protein", "sugar"])

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("changed target contract reached estimator.fit during resume")

    monkeypatch.setattr(MultimodalRegressor, "fit", forbidden)
    with pytest.raises(Exception, match="(?i)(checkpoint|fingerprint|mismatch|contract|task_type|regression)"):
        _run(changed, tmp_path / "invalid-resume", tuning=_tuning(directory, resume=True))
    assert checkpoint.read_bytes() == before
