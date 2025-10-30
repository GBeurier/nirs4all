"""
Score Calculator - Calculate evaluation scores consistently

This component centralizes score calculation logic using ModelUtils and Evaluator.
Extracted from launch_training() lines 449-461 and various controller methods.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
import numpy as np

from ..utilities import ModelControllerUtils as ModelUtils
from nirs4all.core import metrics as evaluator


@dataclass
class PartitionScores:
    """Scores for a single partition."""

    train: float
    val: float
    test: float
    metric: str
    higher_is_better: bool
    detailed_scores: Optional[Dict[str, float]] = None


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
        y_true: Dict[str, np.ndarray],
        y_pred: Dict[str, np.ndarray],
        task_type: str
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
        metric, higher_is_better = ModelUtils.get_best_score_metric(task_type)

        # Calculate scores for each partition
        scores = {}
        for partition in ['train', 'val', 'test']:
            if partition in y_true and partition in y_pred:
                if y_true[partition].shape[0] > 0 and y_pred[partition].shape[0] > 0:
                    scores[partition] = evaluator.eval(
                        y_true[partition],
                        y_pred[partition],
                        metric
                    )
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
        task_type: str,
        metric: Optional[str] = None
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
            metric, _ = ModelUtils.get_best_score_metric(task_type)

        return evaluator.eval(y_true, y_pred, metric)

    def format_scores(self, scores: PartitionScores) -> str:
        """Format scores as a readable string.

        Args:
            scores: PartitionScores instance

        Returns:
            Formatted string like "Train: 0.95 | Val: 0.90 | Test: 0.88 (R2)"
        """
        direction = "↑" if scores.higher_is_better else "↓"
        return (
            f"Train: {scores.train:.4f} | "
            f"Val: {scores.val:.4f} | "
            f"Test: {scores.test:.4f} "
            f"({scores.metric} {direction})"
        )
