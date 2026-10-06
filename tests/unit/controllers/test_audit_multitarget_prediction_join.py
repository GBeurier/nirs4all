"""Refuse target-axis corruption without blocking class probability features."""

from copy import deepcopy

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.controllers.data.merge import MergeController
from nirs4all.controllers.models.meta_model import MetaModelController
from nirs4all.controllers.models.stacking.classification import (
    ClassificationFeatureExtractor,
    ClassificationInfo,
    StackingTaskType,
)
from nirs4all.controllers.models.stacking.reconstructor import TrainingSetReconstructor
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.operators.models import MetaModel
from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
from nirs4all.pipeline.steps.parser import StepParser


def _dataset(n_targets):
    rng = np.random.default_rng(791)
    dataset = SpectroDataset("multi_target_join")
    dataset.add_samples(rng.normal(size=(30, 10)), {"partition": "train"})
    dataset.add_samples(rng.normal(size=(10, 10)), {"partition": "test"})
    dataset.add_targets(rng.normal(size=(40, n_targets)))
    dataset.set_task_type("regression")
    return dataset


@pytest.mark.parametrize("step", [
    {"merge": "predictions"},
    {"merge": {"features": True, "predictions": True}},
    {"merge_predictions": "all"},
    MetaModel(Ridge()),
])
@pytest.mark.parametrize("n_targets", [2, 5])
def test_join_refuses_before_feature_and_context_mutation(step, n_targets):
    dataset = _dataset(n_targets)
    context = ExecutionContext()
    context.custom["in_branch_mode"] = True
    runtime = RuntimeContext()
    original_features = dataset.x(None).copy()
    original_targets = dataset.y(None).copy()
    original_custom = deepcopy(context.custom)
    parsed = StepParser().parse(step)
    controller = MetaModelController() if isinstance(step, MetaModel) else MergeController()
    with pytest.raises(NotImplementedError, match="only one target per sample"):
        controller.execute(parsed, dataset, context, runtime, prediction_store=Predictions())
    np.testing.assert_array_equal(dataset.x(None), original_features)
    np.testing.assert_array_equal(dataset.y(None), original_targets)
    assert context.custom == original_custom
    assert runtime.current_context is None


def test_direct_reconstruction_refuses_multitarget_without_reading_predictions(monkeypatch):
    store = Predictions()
    monkeypatch.setattr(store, "filter_predictions", lambda **kwargs: pytest.fail("prediction access before target validation"))
    with pytest.raises(NotImplementedError, match="only one target per sample"):
        TrainingSetReconstructor(store, ["base"]).reconstruct(_dataset(5), ExecutionContext())


@pytest.mark.parametrize("n_targets", [2, 5])
def test_real_legacy_cv_branch_merge_refuses_multitarget(tmp_path, n_targets):
    dataset = _dataset(n_targets)
    pipeline = [KFold(3), {"branch": [[Ridge(alpha=1)], [Ridge(alpha=2)]]}, {"merge": "predictions"}]
    with pytest.raises(RuntimeError, match="Multi-target predictions cannot be flattened or truncated"):
        nirs4all.run(pipeline, dataset, engine="legacy", refit=False, save_charts=False,
                     workspace_path=tmp_path, verbose=0)


@pytest.mark.parametrize("n_targets", [2, 5])
def test_prediction_extractor_refuses_multioutput_even_without_dataset(n_targets):
    extractor = ClassificationFeatureExtractor(ClassificationInfo(StackingTaskType.REGRESSION))
    with pytest.raises(NotImplementedError, match="multi-target predictions cannot be flattened"):
        extractor.extract_features({"y_pred": np.arange(4 * n_targets).reshape(4, n_targets)}, 4)


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_class_probability_columns_keep_class_axis(n_classes):
    probabilities = np.arange(1, 4 * n_classes + 1, dtype=float).reshape(4, n_classes)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    task = StackingTaskType.BINARY_CLASSIFICATION if n_classes == 2 else StackingTaskType.MULTICLASS_CLASSIFICATION
    extractor = ClassificationFeatureExtractor(ClassificationInfo(task, n_classes=n_classes), use_proba=True)
    result = extractor.extract_features({"y_pred": probabilities.argmax(axis=1), "y_proba": probabilities}, 4)
    np.testing.assert_array_equal(result, probabilities[:, 1] if n_classes == 2 else probabilities)
