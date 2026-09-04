"""Focused tests for dual-engine conformance helpers."""

from __future__ import annotations

from typing import Any

from ._conformance_helpers import _top_distinct_model_names


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
