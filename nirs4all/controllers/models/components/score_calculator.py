"""
Score Calculator - Calculate evaluation scores consistently

This component centralizes score calculation logic using ModelUtils and Evaluator.
Extracted from launch_training() lines 449-461 and various controller methods.
"""

from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np

from nirs4all.core import metrics as evaluator
from nirs4all.core.task_type import TaskType

from ..utilities import ModelControllerUtils as ModelUtils


@dataclass
class PartitionScores:
    """Scores for a single partition."""

    train: float
    val: float
    test: float
    metric: str
    higher_is_better: bool
    detailed_scores: dict[str, float] | None = None

class ScoreCalculator:
    """Calculates evaluation scores for models.

    Uses ModelUtils to select appropriate metrics based on task type,
    and Evaluator to compute scores.

    Example:
        >>> calculator = ScoreCalculator()
        >>> scores = calculator.calculate(
        ...     y_true={'train': y_train, 'val': y_val, 'test': y_test},
        ...     y_pred={'train': y_train_pred, 'val': y_val_pred, 'test': y_test_pred},
        ...     task_type='regression'
        ... )
        >>> scores.test
        0.88
    """

    def calculate(
        self,
        y_true: dict[str, np.ndarray],
        y_pred: dict[str, np.ndarray],
        task_type: str | TaskType
    ) -> PartitionScores:
        """Calculate scores for all partitions.

        Args:
            y_true: Dictionary of true values per partition
            y_pred: Dictionary of predictions per partition
            task_type: Task type string (e.g., 'regression', 'classification')

        Returns:
            PartitionScores with scores for train, val, test
        """
        # Get best metric for task type
        task_type_enum = task_type if isinstance(task_type, TaskType) else TaskType(task_type)
        metric, higher_is_better = ModelUtils.get_best_score_metric(task_type_enum)

        # Calculate scores for each partition
        scores: dict[str, float] = {}
        for partition in ['train', 'val', 'test']:
            if partition in y_true and partition in y_pred:
                if y_true[partition].shape[0] > 0 and y_pred[partition].shape[0] > 0:
                    scores[partition] = cast(float, evaluator.eval(
                        y_true[partition],
                        y_pred[partition],
                        metric
                    ))
                else:
                    scores[partition] = 0.0
            else:
                scores[partition] = 0.0

        return PartitionScores(
            train=scores.get('train', 0.0),
            val=scores.get('val', 0.0),
            test=scores.get('test', 0.0),
            metric=metric,
            higher_is_better=higher_is_better
        )

    def calculate_single(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str | TaskType,
        metric: str | None = None
    ) -> float:
        """Calculate score for a single partition.

        Args:
            y_true: True values
            y_pred: Predictions
            task_type: Task type string
            metric: Optional metric name (if None, uses best metric for task)

        Returns:
            Score value
        """
        if y_true.shape[0] == 0 or y_pred.shape[0] == 0:
            return 0.0

        if metric is None:
            task_type_enum = task_type if isinstance(task_type, TaskType) else TaskType(task_type)
            metric, _ = ModelUtils.get_best_score_metric(task_type_enum)

        return cast(float, evaluator.eval(y_true, y_pred, metric))

    def format_scores(self, scores: PartitionScores) -> str:
        """Format scores as a readable string.

        Args:
            scores: PartitionScores instance

        Returns:
            Formatted string like "Train: 0.95 | Val: 0.90 | Test: 0.88 (R2)"
        """
        from nirs4all.core.logging.formatters import get_symbols
        direction = get_symbols().direction(scores.higher_is_better)
        return (
            f"Train: {scores.train:.4f} | "
            f"Val: {scores.val:.4f} | "
            f"Test: {scores.test:.4f} "
            f"({scores.metric} {direction})"
        )
