"""Native classification and multi-output regression from named raw modalities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset
from sklearn.base import is_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold

import nirs4all
from nirs4all.operators.models.multimodal import MultimodalClassifier, MultimodalRegressor, TensorPCA
from nirs4all.operators.models.sklearn.mbpls import MBPLS
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped
from tests.integration.api.test_multimodal_dagml import _cohort, _model


def _classification_cohort(labels: str = "string") -> MultimodalDataset:
    original = _cohort(unequal_groups=True)
    classes = {
        "string": np.array(["healthy", "mild", "severe"]),
        "numeric": np.array([2, 7, 42]),
        "int64": np.array([2, 2**32 + 2, 2**63 - 1], dtype=np.int64),
    }[labels]
    return MultimodalDataset(
        original.sources, sample_ids=original.sample_ids, y=classes[np.arange(len(original)) % 3],
        groups=original.groups, partitions=original.partitions, name=original.name,
    )


def _classifier() -> MultimodalClassifier:
    return MultimodalClassifier(_model().transformers, LogisticRegression(C=0.3, max_iter=300))


def test_unknown_test_class_refuses_before_classifier_fit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original = _classification_cohort()
    labels = np.asarray(original.y, dtype=object).copy()
    labels[-1] = "new-diagnosis"
    cohort = MultimodalDataset(original.sources, sample_ids=original.sample_ids, y=labels,
                               groups=original.groups, partitions=original.partitions)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("unknown held-out class reached classifier fitting")

    monkeypatch.setattr(MultimodalClassifier, "fit", forbidden)
    with pytest.raises(ValueError, match="labels absent from training"):
        _run(_classifier(), cohort, tmp_path / "unknown-class")


def _run(model: Any, cohort: MultimodalDataset, workspace: Path, *, tuning: dict[str, Any] | None = None) -> Any:
    return nirs4all.run(
        [GroupKFold(3), {"model": model}], cohort, tuning=tuning, engine="dag-ml",
        refit=True, save_artifacts=True, save_charts=False, verbose=0, random_state=19, workspace_path=workspace,
    )


def _assert_native_metrics(result: Any, classification: bool = False) -> None:
    reports = {report["prediction_id"]: report for report in result._dagml_score_set["reports"] if report.get("fold_id") != "avg"}
    checked = 0
    for node in result._dagml_node_results:
        for block in node.get("predictions", []):
            report = reports.get(block["prediction_id"])
            if report is None:
                continue
            ids = block["sample_ids"]
            target = next(target for target in node["regression_targets"] if [unit["id"] for unit in target["unit_ids"]] == ids)
            observed = np.asarray(target["values"])
            predicted = np.asarray(block["values"])
            assert observed.shape == predicted.shape
            assert report["row_count"] == len(ids)
            for index, name in enumerate(block["target_names"]):
                y, p = observed[:, index], predicted[:, index]
                metrics = {"accuracy": accuracy_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, p)} if classification else {
                    "rmse": root_mean_squared_error(y, p), "mse": mean_squared_error(y, p), "mae": mean_absolute_error(y, p),
                }
                for metric, value in metrics.items():
                    assert report["metrics"][f"{metric}:{name}"] == pytest.approx(value)
            if not classification:
                assert report["metrics"]["rmse"] == pytest.approx(root_mean_squared_error(observed, predicted))
            checked += 1
    assert checked == 8  # Three OOF folds, three CV test views, refit and final test.


@pytest.mark.parametrize("labels", ["string", "numeric", "int64"])
def test_classification_native_scores_and_archive_preserve_labels(labels: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _run(_classifier(), _classification_cohort(labels), tmp_path / "workspace")
    try:
        assert result.best["task_type"] == "classification"
        _assert_native_metrics(result, classification=True)
        artifact = result._dagml_refit_artifacts[0]
        estimator = artifact["estimator"]
        assert is_classifier(estimator)
        prediction = _cohort(prediction=True)
        blocks = [prediction.sources[name].values for name in estimator.source_names]
        probabilities = estimator.predict_proba(blocks)
        assert probabilities.shape == (len(prediction), 3)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
        np.testing.assert_array_equal(estimator.classes_[probabilities.argmax(axis=1)], estimator.predict(blocks))
        encoded = estimator.predict(blocks)
        expected = artifact["y_transform"].decode(encoded.reshape(-1, 1)).ravel()
        archive = result.export(tmp_path / "classifier.n4a")

        def forbidden(*args: Any, **kwargs: Any) -> None:
            pytest.fail("classification archive attempted training")

        monkeypatch.setattr(MultimodalClassifier, "fit", forbidden)
        monkeypatch.setattr(LogisticRegression, "fit", forbidden)
        monkeypatch.setattr(TensorPCA, "fit", forbidden)
        replay = nirs4all.predict(archive, prediction)
        np.testing.assert_array_equal(replay.y_pred, expected)
        assert set(replay.y_pred) <= set(_classification_cohort(labels).y)
        assert replay.metadata["training_performed"] is False
        assert replay.metadata["scores"] is None
    finally:
        result.close()


@pytest.mark.parametrize("fusion", ["early", "intermediate"])
def test_multioutput_native_scores_and_archive_keep_target_columns(fusion: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original = _cohort(unequal_groups=True)
    targets = np.column_stack([original.y, original.y * 2 + np.arange(len(original)) * 0.1])
    cohort = MultimodalDataset(
        original.sources, sample_ids=original.sample_ids, y=targets,
        groups=original.groups, partitions=original.partitions, name=original.name,
        target_names=["concentration", "moisture"], task_type="regression",
    )
    model = _model()
    if fusion == "intermediate":
        model.set_params(model=MBPLS(n_components=2, standardize=False), fusion=fusion)
    result = _run(model, cohort, tmp_path / "workspace")
    try:
        assert result.best["task_type"] == "regression"
        _assert_native_metrics(result)
        prediction = _cohort(prediction=True)
        estimator = result._dagml_refit_artifacts[0]["estimator"]
        expected = estimator.predict([prediction.sources[name].values for name in estimator.source_names])
        assert expected.shape == (len(prediction), 2)
        archive = result.export(tmp_path / "two-targets.n4a")

        def forbidden(*args: Any, **kwargs: Any) -> None:
            pytest.fail("multi-output archive attempted training")

        monkeypatch.setattr(MultimodalRegressor, "fit", forbidden)
        monkeypatch.setattr(TensorPCA, "fit", forbidden)
        replay = nirs4all.predict(archive, prediction)
        assert replay.y_pred.shape == expected.shape
        assert replay.metadata["target_names"] == ["concentration", "moisture"]
        np.testing.assert_allclose(replay.y_pred, expected, rtol=1e-9, atol=1e-9)
    finally:
        result.close()


def test_classification_tuning_resumes_and_maximizes_native_score(tmp_path: Path) -> None:
    cohort = _classification_cohort()
    tuning = {
        "engine": "n4m", "sampler": "random", "seed": 19, "n_trials": 4,
        "storage": (tmp_path / "resumed-study").as_uri(), "study_name": "classification",
        "space": {"model__C": [0.1, 1.0], "source_weights__image": [0.5, 1.0], "transformers__image__n_components": [1, 2]},
    }
    with pytest.raises(MultimodalTuningStopped):
        _run(_classifier(), cohort, tmp_path / "interrupted", tuning={
            **tuning, "progress_callback": lambda event: len(event["checkpoint"]["trials"]) < 2,
        })
    resumed = _run(_classifier(), cohort, tmp_path / "resumed", tuning={**tuning, "resume": True})
    continuous = _run(_classifier(), cohort, tmp_path / "continuous", tuning={**tuning, "storage": (tmp_path / "continuous-study").as_uri()})
    try:
        assert resumed.tuning_result.tuning.metric == "balanced_accuracy"
        assert resumed.tuning_result.tuning.direction == "maximize"
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [trial.to_dict() for trial in continuous.tuning_result.trials]
        assert resumed.tuning_best_value == max(trial.value for trial in resumed.tuning_result.trials)
        assert resumed.tuning_best_params == continuous.tuning_best_params
        first = resumed.export(tmp_path / "resumed.n4a")
        second = continuous.export(tmp_path / "continuous.n4a")
        np.testing.assert_array_equal(nirs4all.predict(first, _cohort(prediction=True)).y_pred, nirs4all.predict(second, _cohort(prediction=True)).y_pred)
    finally:
        resumed.close()
        continuous.close()


def test_late_fusion_multioutput_preserves_named_targets_and_replays_without_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sklearn.linear_model import Ridge

    from tests.integration.api.test_multimodal_late_fusion import _cohort as late_cohort
    from tests.integration.api.test_multimodal_late_fusion import _pipeline as late_pipeline

    original = late_cohort()
    targets = np.column_stack([original.y, original.y * 2 + np.arange(len(original)) * 0.1])
    cohort = MultimodalDataset(
        original.sources, sample_ids=original.sample_ids, y=targets,
        target_names=["concentration", "moisture"], task_type="regression",
        groups=original.groups, partitions=original.partitions, name=original.name,
    )
    result: Any = nirs4all.run(
        late_pipeline(), cohort, engine="dag-ml", refit=True, save_artifacts=True,
        save_charts=False, verbose=0, random_state=17, workspace_path=tmp_path / "workspace",
    )
    try:
        meta = result.runs[-1]
        model = next(item["estimator"] for item in meta._dagml_refit_artifacts if item["controller_id"] == "controller:nirs4all.meta_model")
        assert model.n_features_in_ == 8  # Four source predictors, two columns each.
        prediction = late_cohort(prediction=True)
        base_outputs = [
            child._dagml_refit_artifacts[0]["estimator"].predict(prediction.sources[name].values)
            for name, child in zip(cohort.sources, result.runs[:4], strict=True)
        ]
        expected = model.predict(np.column_stack(base_outputs))
        assert expected.shape == (len(prediction), 2)
        reports = [report for report in meta._dagml_score_set["reports"] if report["producer_node"] == "merge:stack"]
        assert reports and all(report["target_names"] == ["concentration", "moisture"] for report in reports)
        assert all("rmse:concentration" in report["metrics"] and "rmse:moisture" in report["metrics"] for report in reports)
        archive = meta.export(tmp_path / "late-two-targets.n4a")

        def forbidden(*args: Any, **kwargs: Any) -> None:
            pytest.fail("late multi-output replay attempted fit")

        monkeypatch.setattr(Ridge, "fit", forbidden)
        monkeypatch.setattr(TensorPCA, "fit", forbidden)
        replay = nirs4all.predict(archive, prediction)
        assert replay.metadata["target_names"] == ["concentration", "moisture"]
        assert replay.y_pred.shape == expected.shape
        np.testing.assert_allclose(replay.y_pred, expected, rtol=1e-9, atol=1e-9)
    finally:
        result.close()


@pytest.mark.parametrize("labels", ["string", "numeric"])
def test_late_fusion_classification_decodes_only_the_final_output(labels: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.integration.api.test_multimodal_late_fusion import _cohort as late_cohort
    from tests.integration.api.test_multimodal_late_fusion import _pipeline as late_pipeline

    original = late_cohort()
    classes = np.array(["healthy", "mild", "severe"]) if labels == "string" else np.array([-8, 4, 37])
    cohort = MultimodalDataset(
        original.sources, sample_ids=original.sample_ids, y=classes[np.arange(len(original)) % 3],
        task_type="classification", target_names=["diagnosis"],
        groups=original.groups, partitions=original.partitions, name=original.name,
    )
    pipeline = late_pipeline()
    for branch in pipeline[1]["branch"]["steps"].values():
        branch[-1] = LogisticRegression(C=0.2, max_iter=500)
    pipeline[-1] = LogisticRegression(C=0.5, max_iter=500)
    result: Any = nirs4all.run(
        pipeline, cohort, engine="dag-ml", refit=True, save_artifacts=True,
        save_charts=False, verbose=0, random_state=17, workspace_path=tmp_path / "workspace",
    )
    try:
        meta = result.runs[-1]
        assert meta.best["task_type"] == "classification"
        prediction = late_cohort(prediction=True)
        base_outputs = [
            child._dagml_refit_artifacts[0]["estimator"].predict(prediction.sources[name].values)
            for name, child in zip(cohort.sources, result.runs[:4], strict=True)
        ]
        meta_artifact = next(item for item in meta._dagml_refit_artifacts if item["controller_id"] == "controller:nirs4all.meta_model")
        encoded = meta_artifact["estimator"].predict(np.column_stack(base_outputs))
        expected = meta_artifact["y_transform"].decode(encoded.reshape(-1, 1)).ravel()
        archive = meta.export(tmp_path / "late-classifier.n4a")

        def forbidden(*args: Any, **kwargs: Any) -> None:
            pytest.fail("late classification replay attempted fit")

        monkeypatch.setattr(LogisticRegression, "fit", forbidden)
        monkeypatch.setattr(TensorPCA, "fit", forbidden)
        replay = nirs4all.predict(archive, prediction)
        np.testing.assert_array_equal(replay.y_pred, expected)
        assert set(replay.y_pred) <= set(classes)
        assert replay.metadata["target_names"] == ["diagnosis"]
        assert replay.metadata["training_performed"] is False
        native_values = replay.metadata["node_results"][0]["predictions"][0]["values"]
        assert all(isinstance(row[0], int | float) for row in native_values)
    finally:
        result.close()
