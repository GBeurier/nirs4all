"""
Task Type Enumeration - Shared across the library

This module provides the TaskType enum that is shared across data, controllers, and utilities.
Kept as a minimal standalone module to avoid circular import issues.
"""

from enum import Enum


class TaskType(str, Enum):
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
