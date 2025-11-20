"""
Task Type Detection - Standalone utility to avoid circular imports

This module provides task type detection functionality that can be used
by both data and controller modules without creating circular import issues.
"""

import numpy as np
from nirs4all.core.task_type import TaskType


def detect_task_type(y: np.ndarray, threshold: float = 0.05) -> TaskType:
    """
    Detect task type based on target values.

    Args:
        y: Target values array
        threshold: Threshold for determining if values are continuous (regression)
                  vs discrete (classification). For integer values, if n_unique <= max_classes
                  or n_unique <= len(y) * threshold, it's considered classification.

    Returns:
        TaskType: Detected task type
    """
    # Flatten y to handle various shapes
    y_flat = np.asarray(y).ravel()

    # Remove NaN values if any
    y_clean = y_flat[~np.isnan(y_flat)]

    if len(y_clean) == 0:
        raise ValueError("Target array contains only NaN values")

    # Check if all values are integers (potential classification)
    if np.all(np.equal(np.mod(y_clean, 1), 0)):
        unique_values = np.unique(y_clean)
        n_unique = len(unique_values)

        # Maximum reasonable number of classes for classification
        max_classes = 100

        # Binary classification: exactly 2 unique values
        if n_unique == 2:
            return TaskType.BINARY_CLASSIFICATION

        # Multi-class classification: more than 2 but reasonable number of classes
        elif n_unique > 2 and n_unique <= max_classes:
            return TaskType.MULTICLASS_CLASSIFICATION

        # Too many unique integer values - likely regression with integer targets
        else:
            return TaskType.REGRESSION

    # Check if values are in [0, 1] range (potential binary classification probabilities)
    if np.all(y_clean >= 0) and np.all(y_clean <= 1):
        unique_values = np.unique(y_clean)
        n_unique = len(unique_values)

        # If mostly 0s and 1s, treat as binary classification
        if n_unique == 2 and set(unique_values) == {0.0, 1.0}:
            return TaskType.BINARY_CLASSIFICATION

        # If few unique values in [0,1], might be classification probabilities
        elif n_unique <= len(y_clean) * threshold:
            if n_unique == 2:
                return TaskType.BINARY_CLASSIFICATION
            else:
                return TaskType.MULTICLASS_CLASSIFICATION

    # Default to regression for continuous values
    return TaskType.REGRESSION
