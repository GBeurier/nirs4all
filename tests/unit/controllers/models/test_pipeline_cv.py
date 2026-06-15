from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

from nirs4all.controllers.models.base_model import BaseModelController
from nirs4all.controllers.models.pipeline_cv import (
    PrecomputedFoldSplitter,
    apply_pipeline_folds_to_aom_estimator,
    make_pipeline_fold_splitter,
)
from nirs4all.controllers.models.sklearn_model import SklearnModelController
from nirs4all.pipeline.config.context import ExecutionPhase


class AOMRecordingRegressor(BaseEstimator):
    def __init__(self, cv=5, cv_splitter=None, external_folds=None, selection="cv", repeats=1):
        self.cv = cv
        self.cv_splitter = cv_splitter
        self.external_folds = external_folds
        self.selection = selection
        self.repeats = repeats

    def fit(self, X, y):
        self.fit_cv_ = self.cv
        self.fit_cv_splitter_ = self.cv_splitter
        self.fit_external_folds_ = self.external_folds
        self.fit_selection_ = self.selection
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])


class RecordingController(BaseModelController):
    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return False

    def _get_model_instance(self, dataset: Any, model_config: dict[str, Any], force_params: dict[str, Any] | None = None) -> Any:
        return model_config["model"]

    def _train_model(self, model: Any, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None, **kwargs) -> Any:
        self.received_train_kwargs = kwargs
        return model

    def _predict_model(self, model: Any, X: Any) -> np.ndarray:
        return np.zeros((X.shape[0], 1))

    def _prepare_data(self, X: Any, y: Any, context: Any) -> tuple[Any, Any]:
        return np.asarray(X), None if y is None else np.asarray(y)

    def _clone_model(self, model: Any) -> Any:
        return model

    def _evaluate_model(self, model: Any, X_val: Any, y_val: Any, metric: str | None = None, direction: str = "minimize") -> float:
        return 0.0


def _pipeline_folds():
    return [
        (np.array([0, 1, 2, 3]), np.array([4, 5])),
        (np.array([0, 1, 4, 5]), np.array([2, 3])),
        (np.array([2, 3, 4, 5]), np.array([0, 1])),
    ]


class DummyContext:
    custom: dict[str, Any] = {}
    selector: object = object()

    def with_partition(self, partition: str) -> "DummyContext":
        return self


class DummyFoldDataset:
    task_type = SimpleNamespace(is_classification=False)

    def __init__(self) -> None:
        self._folds = [([10, 11, 12, 13, 14, 15], [])]
        self._nirs4all_aom_pipeline_folds = [
            ([10, 11, 12, 13], [14, 15]),
            ([10, 11, 14, 15], [12, 13]),
            ([12, 13, 14, 15], [10, 11]),
        ]
        self._indexer = SimpleNamespace(
            x_indices=lambda selector, include_augmented=True, include_excluded=False: np.array([10, 11, 12, 13, 14, 15])
        )

    @property
    def folds(self):
        return self._folds

    def set_folds(self, folds_iterable) -> None:
        self._folds = list(folds_iterable)


def test_make_pipeline_fold_splitter_remaps_to_fold_training_slice():
    splitter = make_pipeline_fold_splitter(
        _pipeline_folds(),
        n_samples=4,
        train_indices=np.array([0, 1, 2, 3]),
    )

    assert splitter is not None
    folds = list(splitter.split(np.zeros((4, 2))))
    assert len(folds) == 2
    np.testing.assert_array_equal(folds[0][1], [2, 3])
    np.testing.assert_array_equal(folds[0][0], [0, 1])
    np.testing.assert_array_equal(folds[1][1], [0, 1])
    np.testing.assert_array_equal(folds[1][0], [2, 3])


def test_refit_aom_pipeline_folds_are_remapped_without_replacing_full_train_fold():
    controller = RecordingController()
    dataset = DummyFoldDataset()
    original_full_train_fold = list(dataset.folds)

    remapped = controller._remap_aom_pipeline_folds_to_positions(dataset, DummyContext(), "train")

    assert dataset.folds == original_full_train_fold
    assert remapped == [
        ([0, 1, 2, 3], [4, 5]),
        ([0, 1, 4, 5], [2, 3]),
        ([2, 3, 4, 5], [0, 1]),
    ]


def test_precomputed_fold_splitter_validates_row_count():
    splitter = PrecomputedFoldSplitter.from_folds([([0, 1], [2]), ([1, 2], [0])], n_samples=3)

    with pytest.raises(ValueError, match="expected 3 rows"):
        list(splitter.split(np.zeros((4, 2))))


def test_apply_pipeline_folds_to_aom_sets_supported_params():
    splitter = PrecomputedFoldSplitter.from_folds([([0, 1], [2]), ([1, 2], [0])], n_samples=3)
    model = AOMRecordingRegressor()

    changed = apply_pipeline_folds_to_aom_estimator(model, splitter)

    assert changed is True
    assert model.cv is splitter.get_n_splits()
    assert model.cv_splitter is splitter
    assert model.external_folds == [[2], [0]]
    assert model.selection == "external"


def test_apply_pipeline_folds_leaves_non_aom_estimator_untouched():
    splitter = PrecomputedFoldSplitter.from_folds([([0, 1], [2]), ([1, 2], [0])], n_samples=3)
    model = Ridge()

    changed = apply_pipeline_folds_to_aom_estimator(model, splitter)

    assert changed is False
    assert model.get_params()["alpha"] == 1.0


def test_required_pipeline_folds_raise_for_aom_when_missing():
    with pytest.raises(ValueError, match="Pipeline folds are required"):
        apply_pipeline_folds_to_aom_estimator(
            AOMRecordingRegressor(),
            None,
            policy="required",
            unavailable_reason="no folds",
        )


def test_sklearn_controller_injects_pipeline_cv_before_fit():
    splitter = PrecomputedFoldSplitter.from_folds([([0, 1], [2]), ([1, 2], [0])], n_samples=3)
    controller = SklearnModelController()
    model = AOMRecordingRegressor()

    trained = controller._train_model(
        model,
        np.zeros((3, 2)),
        np.arange(3, dtype=float),
        _pipeline_fold_splitter_for_aom=splitter,
        _pipeline_fold_policy_for_aom="auto",
        _pipeline_fold_unavailable_reason_for_aom=None,
        task_type=None,
    )

    assert trained.fit_cv_splitter_ is splitter
    assert trained.fit_external_folds_ == [[2], [0]]
    assert trained.fit_selection_ == "external"


def test_base_controller_does_not_leak_pipeline_cv_kwargs_to_non_aom_model():
    controller = RecordingController()
    controller.verbose = 0

    controller._build_and_train_model(
        dataset=SimpleNamespace(task_type=SimpleNamespace(is_classification=False)),
        model_config={"model": object(), "train_params": {"use_pipeline_folds_for_aom": "required"}},
        context=SimpleNamespace(custom={}),
        runtime_context=SimpleNamespace(phase=ExecutionPhase.CV, artifact_provider=None, step_number=0),
        identifiers=SimpleNamespace(name="non_aom"),
        X_train=np.zeros((4, 2)),
        y_train=np.arange(4, dtype=float),
        X_val=np.zeros((0, 2)),
        y_val=np.zeros((0,), dtype=float),
        X_test=np.zeros((0, 2)),
        best_params=None,
        train_indices=np.arange(4),
        pipeline_folds=_pipeline_folds(),
    )

    assert "task_type" in controller.received_train_kwargs
    assert not any(key.startswith("_pipeline_fold") for key in controller.received_train_kwargs)
