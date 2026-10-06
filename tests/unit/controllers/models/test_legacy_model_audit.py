"""Focused legacy regression witnesses for CTM-02/03/04/05/06/17."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.controllers.models.meta_model import MetaModelController
from nirs4all.controllers.models.stacking.exceptions import MaxStackingLevelExceededError
from nirs4all.controllers.models.stacking.reconstructor import TrainingSetReconstructor
from nirs4all.controllers.splitters.fold_file_loader import FoldFileLoaderController
from nirs4all.data import SpectroDataset
from nirs4all.operators.filters.metadata import MetadataFilter
from nirs4all.operators.models.meta import CoverageStrategy, MetaModel, StackingConfig
from nirs4all.operators.models.meta import TestAggregation as Aggregation
from nirs4all.pipeline.config.context import ExecutionContext


def _dataset():
    X = np.random.default_rng(17).normal(size=(42, 6))
    y = 3 * X[:, 0] - X[:, 2] + 0.1 * X[:, 4]
    dataset = SpectroDataset("legacy-model-audit")
    dataset.add_samples(X[:36], {"partition": "train"})
    dataset.add_samples(X[36:], {"partition": "test"})
    dataset.add_targets(y)
    dataset.add_metadata(pl.DataFrame({"batch": np.where(np.arange(42) == 0, "bad", "good"), "sample_id": np.arange(42)}))
    return dataset, X, y


@pytest.mark.parametrize("with_meta", [False, True])
def test_excluded_sample_predictions_keep_actual_ids_and_oof_targets(with_meta, tmp_path):
    dataset, _, y = _dataset()
    pipeline = [{"exclude": MetadataFilter("batch", values_to_exclude=["bad"])}, KFold(3), Ridge()]
    if with_meta:
        pipeline.append(MetaModel(Ridge(alpha=0.1)))
    result = nirs4all.run(pipeline, dataset, engine="legacy", workspace_path=tmp_path,
                          save_charts=False, save_artifacts=False, verbose=0, refit=False)
    try:
        rows = result.predictions.filter_predictions(load_arrays=True)
        assert rows
        for row in rows:
            ids = np.asarray(row["sample_indices"], dtype=int)
            np.testing.assert_allclose(np.asarray(row["y_true"]).ravel(), y[ids], rtol=1e-6)
            assert 0 not in ids
            assert row["metadata"]["sample_id"] == ids.tolist()
        val_rows = [row for row in rows if row["partition"] == "val" and row["fold_id"] not in {"avg", "w_avg"}]
        assert set(np.concatenate([row["sample_indices"] for row in val_rows])) == set(range(1, 36))
    finally:
        result.close()


@pytest.mark.parametrize("coverage", [CoverageStrategy.STRICT, CoverageStrategy.IMPUTE_ZERO])
def test_default_meta_refit_produces_nonconstant_final_predictions(coverage, tmp_path):
    dataset, X, y = _dataset()
    pipeline = [KFold(3), Ridge(), MetaModel(Ridge(alpha=0.1), stacking_config=StackingConfig(coverage_strategy=coverage))]
    result = nirs4all.run(pipeline, dataset, engine="legacy", workspace_path=tmp_path,
                          save_charts=False, save_artifacts=False, verbose=0)
    try:
        rows = result.predictions.filter_predictions(partition="test", load_arrays=True)
        finals = [row for row in rows if row.get("fold_id") == "final" and "MetaModel" in row["model_name"]]
        assert finals, "default refit must retain a final meta-learner"
        assert np.std(finals[0]["y_pred"]) > 0.1
        base = Ridge().fit(X[:36], y[:36])
        meta = Ridge(alpha=0.1).fit(base.predict(X[:36])[:, None], y[:36])
        np.testing.assert_allclose(np.asarray(finals[0]["y_pred"]).ravel(), meta.predict(base.predict(X[36:])[:, None]), rtol=1e-6)
    finally:
        result.close()


def test_forced_meta_parameters_create_independent_unfitted_models():
    original = Ridge(alpha=1.0).fit([[0.0], [1.0], [2.0]], [0.0, 2.0, 4.0])
    config = {"model_instance": MetaModel(original)}
    controller = MetaModelController()
    first = controller._get_model_instance(None, config, {"model__alpha": 2.0})
    second = controller._get_model_instance(None, config, {"model__alpha": 3.0})
    assert first is not second and first is not original and second is not original
    assert original.alpha == 1.0
    assert (first.alpha, second.alpha) == (2.0, 3.0)
    assert not hasattr(first, "coef_") and not hasattr(second, "coef_")
    first.fit([[0.0], [1.0]], [0.0, 1.0])
    assert not hasattr(second, "coef_")


class _Store:
    def __init__(self, rows):
        self.rows = rows

    def filter_predictions(self, **filters):
        return [row for row in self.rows if all(key == "load_arrays" or row.get(key) == value for key, value in filters.items())]


@pytest.mark.parametrize("metric,scores,best,weighted", [
    ("rmse", [1.0, 9.0], 0.0, 0.1),
    ("r2", [-3.0, -1.0], 1.0, 1.0),
    ("balanced_accuracy", [0.2, 0.8], 1.0, 0.8),
    ("rmse", [None, 9.0], 1.0, 1.0),
    ("rmse", [np.nan, np.inf], 0.5, 0.5),
])
@pytest.mark.parametrize("aggregation", [Aggregation.BEST_FOLD, Aggregation.WEIGHTED_MEAN])
def test_test_fold_aggregation_uses_stored_metric(metric, scores, best, weighted, aggregation):
    rows = [{"model_name": "base", "partition": "test", "step_idx": 1, "fold_id": fold,
             "sample_indices": np.array([20, 30]), "y_pred": np.full(2, fold, dtype=float), "val_score": score, "metric": metric}
            for fold, score in enumerate(scores)]
    store = _Store(rows)
    context = SimpleNamespace(selector=SimpleNamespace(branch_id=None), state=SimpleNamespace(step_number=2))
    reconstructor = TrainingSetReconstructor(store, ["base"], StackingConfig(test_aggregation=aggregation))
    expected = best if aggregation == Aggregation.BEST_FOLD else weighted
    actual = reconstructor._collect_test_predictions("base", None, 2, {20: 0, 30: 1}, 2)
    np.testing.assert_allclose(actual, expected)
    actual = MetaModelController()._aggregate_test_predictions_for_model("base", store, context, 2, use_proba=False, aggregation=aggregation)
    np.testing.assert_allclose(actual, expected)
    actual = reconstructor._aggregate_test_fold_features([row["y_pred"] for row in rows], scores, aggregation, 1, 2, metric=metric)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("mode", ["predict", "explain"])
def test_fold_file_replay_does_not_read_or_validate_training_file(mode, tmp_path):
    dataset = SpectroDataset("new-cohort")
    dataset.add_samples(np.ones((4, 6)), {"partition": "test"})
    provider = Mock()
    provider.get_fold_artifacts.return_value = [(fold, object()) for fold in range(3)]
    runtime = SimpleNamespace(artifact_provider=provider, target_model={"step_idx": 2})
    context = ExecutionContext()
    step = SimpleNamespace(operator=tmp_path / "unavailable-original-folds.json")
    result_context, _ = FoldFileLoaderController().execute(step, dataset, context, runtime, mode=mode)
    assert result_context is context
    assert len(dataset.folds) == 3
    assert all(len(train) == 4 and validation == [] for train, validation in dataset.folds)


def test_fold_file_public_prediction_replays_saved_cv_model(tmp_path):
    dataset, X, _ = _dataset()
    fold_path = tmp_path / "folds.json"
    fold_path.write_text(json.dumps([{"train": train.tolist(), "val": val.tolist()} for train, val in KFold(3).split(X[:36])]))
    result = nirs4all.run([{"split": str(fold_path)}, Ridge()], dataset, engine="legacy", workspace_path=tmp_path / "workspace",
                          save_charts=False, verbose=0, refit=False)
    try:
        fold_path.unlink()
        prediction = nirs4all.predict(result.best, X[36:], engine="legacy", verbose=0, workspace_path=tmp_path / "workspace")
        assert len(prediction.y_pred) == 6
        assert np.isfinite(prediction.y_pred).all()
    finally:
        result.close()


@pytest.mark.parametrize("error,exception", [
    ("Detected level 2 exceeds maximum 1", MaxStackingLevelExceededError),
    ("Unsupported meta-source configuration", ValueError),
])
def test_invalid_multilevel_result_always_raises(error, exception):
    result = SimpleNamespace(errors=[error], detected_level=2, circular_dependencies=[])
    with pytest.raises(exception):
        MetaModelController()._raise_multi_level_error(result, StackingConfig(max_level=1), ["base"])


@pytest.mark.parametrize("with_folds", [False, True])
def test_augmented_predictions_keep_child_ids_and_origin_metadata(with_folds, tmp_path):
    dataset, X, y = _dataset()
    dataset.augment_samples(X[:36] + 0.01, processings=["raw"], augmentation_id="shift", selector={"partition": "train"})
    pipeline = ([KFold(3)] if with_folds else []) + [Ridge()]
    result = nirs4all.run(pipeline, dataset, engine="legacy", workspace_path=tmp_path,
                          save_charts=False, save_artifacts=False, verbose=0, refit=False)
    try:
        rows = result.predictions.filter_predictions(load_arrays=True)
        assert any(np.max(row["sample_indices"]) >= 42 for row in rows if row["partition"] == "train")
        for row in rows:
            origins = dataset._indexer.get_origins_for_samples(list(row["sample_indices"]))
            np.testing.assert_allclose(np.asarray(row["y_true"]).ravel(), y[origins], rtol=1e-6)
            assert row["metadata"]["sample_id"] == list(origins)
            if row["partition"] == "val" and with_folds:
                assert np.max(row["sample_indices"]) < 36
    finally:
        result.close()
