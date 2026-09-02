"""Regression tests for named-dict stacking result projection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from nirs4all.pipeline.dagml.run_paths import _named_dict_stacking_legacy_projection


class _Spectro:
    """Minimal dataset surface consumed by the named stacking projection."""

    name = "stacking-regression"

    def __init__(self) -> None:
        self._x = np.arange(12, dtype=float).reshape(-1, 1)
        self._y = 3.0 * self._x.ravel() + 2.0

    def index_column(self, _column: str, query: dict[str, str]) -> list[int]:
        return list(range(9)) if query["partition"] == "train" else list(range(9, 12))

    def x(self, query: dict[str, list[int]], *, layout: str) -> np.ndarray:
        assert layout == "2d"
        return self._x[query["sample"]]

    def y(self, query: dict[str, list[int]]) -> np.ndarray:
        return self._y[query["sample"]]


def test_meta_validation_rows_keep_cross_fitted_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never replace a meta fold's OOF score with an optimistic in-sample score."""

    def fixed_cross_val_predict(_estimator: Any, _x: np.ndarray, y: np.ndarray, *, cv: Any) -> np.ndarray:
        assert cv is not None
        return np.asarray(y, dtype=float) + 10.0

    monkeypatch.setattr("sklearn.model_selection.cross_val_predict", fixed_cross_val_predict)
    folds = [
        ([3, 4, 5, 6, 7, 8], [0, 1, 2]),
        ([0, 1, 2, 6, 7, 8], [3, 4, 5]),
        ([0, 1, 2, 3, 4, 5], [6, 7, 8]),
    ]
    result = _named_dict_stacking_legacy_projection(
        pipeline=[],
        branches=[[{"model": Ridge(alpha=0.5)}], [{"model": Ridge(alpha=1.5)}]],
        meta_learner=Ridge(alpha=1.0),
        spectro=_Spectro(),
        folds=folds,
        metric="rmse",
        task_type="regression",
        config_name="projection-regression",
        scores=None,
    )

    meta_val_rows = [
        row
        for row in result.predictions.filter_predictions(partition="val", load_arrays=False)
        if row.get("branch_id") is None and str(row.get("fold_id")) in {"0", "1", "2"}
    ]
    assert len(meta_val_rows) == 3
    assert [row["val_score"] for row in meta_val_rows] == pytest.approx([10.0, 10.0, 10.0])
