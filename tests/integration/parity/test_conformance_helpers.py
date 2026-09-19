"""Focused tests for dual-engine conformance helpers."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from ._conformance_helpers import _top_distinct_model_names, assert_native_score_evidence


class _FoldRowResult:
    """Minimal ``RunResult.top`` surface with repeated CV rows per model."""

    rows = [
        {"model_name": "GBR", "score": 1.0},
        {"model_name": "PLS", "score": 2.0},
        {"model_name": "RF", "score": 3.0},
        {"model_name": "PLS", "score": 4.0},
        {"model_name": "PLS", "score": 5.0},
        {"model_name": "Ridge", "score": 6.0},
    ]

    def top(self, n: int, **kwargs: Any) -> list[dict[str, Any]]:
        if kwargs.get("group_by") != "model_name":
            return self.rows[:n]
        seen: set[str] = set()
        grouped: list[dict[str, Any]] = []
        for row in self.rows:
            name = str(row["model_name"])
            if name not in seen:
                seen.add(name)
                grouped.append(row)
        return grouped


def test_top_distinct_model_names_does_not_spend_slots_on_cv_fold_rows() -> None:
    """A model after repeated fold rows remains in the distinct-model top-N."""
    result = _FoldRowResult()

    assert {row["model_name"] for row in result.top(5)} == {"GBR", "PLS", "RF"}
    assert _top_distinct_model_names(result, 5) == {"GBR", "PLS", "RF", "Ridge"}


def _score_result(rows: list[dict[str, Any]]) -> Any:
    return SimpleNamespace(predictions=SimpleNamespace(filter_predictions=lambda **kwargs: rows))


def _measured_row() -> dict[str, Any]:
    return {
        "partition": "val", "fold_id": "0", "metric": "rmse",
        "train_score": None, "test_score": None, "val_score": 1.0,
        "scores": {"val": {"rmse": 1.0}}, "result_metadata": {},
        "y_true": [1.0, 2.0], "y_pred": [2.0, 3.0], "sample_indices": [4, 7],
    }


def test_general_dag_scores_require_own_prediction_evidence() -> None:
    row = _measured_row()
    assert_native_score_evidence(_score_result([row]))
    row["val_score"] = 0.5
    with pytest.raises(AssertionError):
        assert_native_score_evidence(_score_result([row]))


def test_general_dag_scores_reject_missing_arrays() -> None:
    row = _measured_row()
    row.update(y_true=[], y_pred=[], sample_indices=[])
    with pytest.raises(AssertionError, match="target evidence"):
        assert_native_score_evidence(_score_result([row]))


def test_compact_projection_cannot_drop_one_rows_provenance() -> None:
    row = _measured_row()
    row["result_metadata"] = {"dagml_projection": {"score_provenance": {
        "val": {"purpose": "measurement", "partition": "validation"},
    }}}
    assert_native_score_evidence(_score_result([row]))
    broken = deepcopy(row)
    broken["result_metadata"] = {}
    with pytest.raises(KeyError, match="dagml_projection"):
        assert_native_score_evidence(_score_result([row, broken]))


def test_score_only_projection_requires_matching_native_measurement() -> None:
    row = _measured_row()
    row.update(y_true=[], y_pred=[], sample_indices=[])
    row["result_metadata"] = {"dagml_projection": {"score_provenance": {"val": {
        "purpose": "measurement", "partition": "validation", "fold_id": "fold0", "variant_id": "variant:base",
    }}}}
    result = _score_result([row])
    result._dagml_score_set = {"reports": [{
        "partition": "validation", "fold_id": "fold0", "variant_id": "variant:base", "level": "sample",
        "metrics": {"rmse": 1.0}, "row_count": 2,
    }]}
    assert_native_score_evidence(result)
    result._dagml_score_set["reports"][0]["metrics"]["rmse"] = 0.5
    with pytest.raises(AssertionError):
        assert_native_score_evidence(result)
