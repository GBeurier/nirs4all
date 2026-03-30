"""
Task Type Enumeration - Shared across the library

This module provides the TaskType enum that is shared across data, controllers, and utilities.
Kept as a minimal standalone module to avoid circular import issues.
"""

from __future__ import annotations

from enum import StrEnum


class TaskType(StrEnum):
    """Enumeration of machine learning task types."""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"

    @property
    def is_classification(self) -> bool:
        """Check if this is a classification task."""
        return self in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

    @property
    def is_regression(self) -> bool:
        """Check if this is a regression task."""
        return self == TaskType.REGRESSION


# Aliases for fuzzy task-type matching in user-facing APIs.
_TASK_TYPE_ALIASES: dict[str, str] = {
    "regression": "regression",
    "reg": "regression",
    "binary": "binary_classification",
    "binary_classification": "binary_classification",
    "multiclass": "multiclass_classification",
    "multiclass_classification": "multiclass_classification",
    "classification": "_any_classification",
    "clf": "_any_classification",
}


def matches_task_type(record_task_type: str | None, filter_value: str) -> bool:
    """Check whether a prediction record's task_type matches a user-supplied filter.

    Supports fuzzy aliases:
      - ``"regression"`` / ``"reg"`` -> matches ``"regression"``
      - ``"binary"`` -> matches ``"binary_classification"``
      - ``"multiclass"`` -> matches ``"multiclass_classification"``
      - ``"classification"`` / ``"clf"`` -> matches both binary and multiclass

    Args:
        record_task_type: The ``task_type`` value stored in a prediction record.
        filter_value: The user-supplied filter string.

    Returns:
        ``True`` if the record matches the filter.
    """
    if record_task_type is None:
        return False
    resolved = _TASK_TYPE_ALIASES.get(filter_value.lower())
    if resolved is None:
        return record_task_type.lower() == filter_value.lower()
    if resolved == "_any_classification":
        return "classification" in record_task_type.lower()
    return record_task_type.lower() == resolved


def resolve_task_type_sql(filter_value: str) -> tuple[str, list[str]]:
    """Resolve a user-supplied task_type filter into a SQL condition fragment.

    Returns:
        ``(sql_fragment, params)`` — e.g. ``("task_type = ?", ["regression"])``
        or ``("task_type LIKE ?", ["%classification%"])``.
    """
    resolved = _TASK_TYPE_ALIASES.get(filter_value.lower())
    if resolved is None:
        return "task_type = ?", [filter_value]
    if resolved == "_any_classification":
        return "task_type LIKE ?", ["%classification%"]
    return "task_type = ?", [resolved]
