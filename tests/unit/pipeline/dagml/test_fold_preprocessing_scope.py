"""Learned preprocessing must never observe its model fold's validation data."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.dagml import run_paths


class RecordingTransform(TransformerMixin, BaseEstimator):
    fits: ClassVar[list[tuple[tuple[float, ...], tuple[float, ...]]]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> RecordingTransform:
        self.fits.append((tuple(X[:, 0]), tuple(np.asarray(y).ravel())))
        self.mean_ = X.mean(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X - self.mean_


class FoldData:
    name = "fold-scope"

    def __init__(self) -> None:
        self.X = np.column_stack((np.arange(7.0), np.arange(7.0) ** 2))
        self.targets = (np.arange(7.0) + 10).reshape(-1, 1)

    def x_rows(self, samples: list[int], layout: str = "2d") -> np.ndarray:
        return self.X[samples]

    def y(self, selector: dict[str, Any], include_augmented: bool = False) -> np.ndarray:
        return self.targets[sorted(selector["sample"])]

    def index_column(self, column: str, selector: dict[str, Any]) -> list[int]:
        if "sample" in selector:
            return sorted(selector["sample"])
        if not selector:
            return list(range(7))
        return list(range(6)) if selector["partition"] == "train" else [6]

    def metadata_column(self, column: str, selector: dict[str, Any]) -> np.ndarray:
        return np.asarray(["a", "b", "a", "b", "a", "b", "a"])


@pytest.fixture
def fold_scope(monkeypatch: pytest.MonkeyPatch) -> tuple[FoldData, list[tuple[list[int], list[int]]]]:
    data = FoldData()
    folds = [(train.tolist(), val.tolist()) for train, val in KFold(3).split(data.X[:6])]
    RecordingTransform.fits.clear()
    monkeypatch.setattr(run_paths, "_build_folds", lambda *_args: folds)
    return data, folds


def test_concat_preprocessing_fits_fold_train_only_and_refit_once(fold_scope: Any) -> None:
    data, folds = fold_scope
    pipeline = [KFold(3), RecordingTransform(), {"concat_transform": [StandardScaler()]}, {"model": Ridge()}]
    result = run_paths._run_concat_transform_prematerialized(pipeline, data, "rmse", "regression")
    assert RecordingTransform.fits[:3] == [
        (tuple(data.X[train, 0]), tuple(data.targets[train].ravel())) for train, _ in folds
    ]
    assert RecordingTransform.fits[3:] == [(tuple(data.X[:6, 0]), tuple(data.targets[:6].ravel()))]
    result.close()


def test_validation_changes_do_not_change_first_fold_preprocessing_fit(fold_scope: Any) -> None:
    data, folds = fold_scope
    pipeline = [KFold(3), RecordingTransform(), {"concat_transform": [StandardScaler()]}, {"model": Ridge()}]
    first = run_paths._run_concat_transform_prematerialized(pipeline, data, "rmse", "regression")
    initial_fit = RecordingTransform.fits[0]
    first.close()
    data.X[folds[0][1], :] += 1000
    data.targets[folds[0][1]] += 2000
    RecordingTransform.fits.clear()
    second = run_paths._run_concat_transform_prematerialized(pipeline, data, "rmse", "regression")
    assert RecordingTransform.fits[0] == initial_fit
    second.close()


def test_duplication_branch_preprocessing_fits_each_fold_only(fold_scope: Any) -> None:
    data, folds = fold_scope
    result = run_paths._run_duplication_merge_all_branch_result(
        [RecordingTransform(), {"model": Ridge()}], 0, "branch", KFold(3), data, "rmse", "regression", "scope",
    )
    assert RecordingTransform.fits == [
        (tuple(data.X[train, 0]), tuple(data.targets[train].ravel())) for train, _ in folds
    ]
    result.close()


def test_separation_preprocessing_is_fitted_again_for_each_fold(fold_scope: Any) -> None:
    data, folds = fold_scope
    branch = {"branch": {"by_metadata": "group", "steps": [RecordingTransform()]}}
    pipeline = [KFold(3), branch, {"merge": "concat"}, {"model": Ridge()}]
    result = run_paths._run_separation_preproc_concat(
        pipeline, branch, [RecordingTransform()], [{"model": Ridge()}], data, "rmse", "regression",
    )
    expected = [
        (tuple(data.X[train, 0]), tuple(data.targets[train].ravel()))
        for train, _ in folds for _branch in range(2)
    ]
    assert RecordingTransform.fits[:6] == expected
    assert RecordingTransform.fits[6:] == [(tuple(data.X[:6, 0]), tuple(data.targets[:6].ravel()))] * 2
    result.close()


def test_named_branch_keeps_transform_unfitted_until_fold_training(fold_scope: Any) -> None:
    data, folds = fold_scope
    model, X, X_test, preprocessing = run_paths._prepare_named_branch_feature_matrix(
        [RecordingTransform(), {"model": Ridge()}], data, list(range(6)), [6],
    )
    assert RecordingTransform.fits == []
    predictions = Predictions()
    run_paths._project_cv_model_rows(
        predictions, X_train_all=X, X_test=X_test, train_pool=list(range(6)), test_pool=[6],
        folds=folds, model_template=model["model"], preprocessing=preprocessing,
        model_name="Ridge", model_classname="Ridge", step_idx=1, branch_id=0,
        branch_name="branch", spectro=data, metric="rmse", task_type="regression", config_name="scope",
    )
    assert RecordingTransform.fits == [
        (tuple(data.X[train, 0]), tuple(data.targets[train].ravel())) for train, _ in folds
    ]
