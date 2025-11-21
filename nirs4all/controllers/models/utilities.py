"""
Model Controller Utilities - Task detection, loss/metric configuration, and scoring

This module provides controller-specific utilities that were relocated from utils/model_utils.py
to be co-located with model controllers.

Provides:
- Task type detection and enumeration
- Default loss function and metric selection based on task type
- Loss function validation
- Score formatting utilities
"""

from typing import Dict, List, Tuple
import numpy as np

from nirs4all.core.task_type import TaskType
from nirs4all.core.task_detection import detect_task_type as _detect_task_type


class ModelControllerUtils:
    """Utilities for model controller configuration and evaluation."""

    # Default loss functions by task type
    DEFAULT_LOSSES = {
        TaskType.REGRESSION: "mse",
        TaskType.BINARY_CLASSIFICATION: "binary_crossentropy",
        TaskType.MULTICLASS_CLASSIFICATION: "sparse_categorical_crossentropy"
    }

    # Default metrics by task type
    DEFAULT_METRICS = {
        TaskType.REGRESSION: ["mae", "mse"],
        TaskType.BINARY_CLASSIFICATION: ["balanced_accuracy", "accuracy", "auc"],
        TaskType.MULTICLASS_CLASSIFICATION: ["balanced_accuracy", "accuracy", "categorical_accuracy"]
    }

    # Sklearn scoring metrics by task type
    SKLEARN_SCORING = {
        TaskType.REGRESSION: "neg_mean_squared_error",
        TaskType.BINARY_CLASSIFICATION: "balanced_accuracy",
        TaskType.MULTICLASS_CLASSIFICATION: "balanced_accuracy"
    }

    @staticmethod
    def detect_task_type(y: np.ndarray, threshold: float = 0.05) -> TaskType:
        """
        Detect task type based on target values.

        Delegates to standalone task_detection module to avoid circular imports.

        Args:
            y: Target values array
            threshold: Threshold for determining if values are continuous (regression)
                      vs discrete (classification).

        Returns:
            TaskType: Detected task type
        """
        return _detect_task_type(y, threshold)

    @staticmethod
    def get_default_loss(task_type: TaskType, framework: str = "sklearn") -> str:
        """
        Get default loss function for task type and framework.

        Args:
            task_type: Detected task type
            framework: ML framework ("sklearn", "tensorflow", "pytorch")

        Returns:
            str: Default loss function name
        """
        base_loss = ModelControllerUtils.DEFAULT_LOSSES[task_type]

        # Framework-specific adjustments
        if framework == "sklearn":
            # Sklearn uses different naming conventions
            if base_loss == "mse":
                return "squared_error"
            elif base_loss == "binary_crossentropy":
                return "log_loss"
            elif base_loss == "categorical_crossentropy":
                return "log_loss"

        return base_loss

    @staticmethod
    def get_default_metrics(task_type: TaskType, framework: str = "sklearn") -> List[str]:
        """
        Get default metrics for task type and framework.

        Args:
            task_type: Detected task type
            framework: ML framework ("sklearn", "tensorflow", "pytorch")

        Returns:
            List[str]: List of default metric names
        """
        base_metrics = ModelControllerUtils.DEFAULT_METRICS[task_type].copy()

        # Framework-specific adjustments
        if framework == "sklearn":
            # Sklearn has different metric names
            sklearn_mapping = {
                "mae": "mean_absolute_error",
                "mse": "mean_squared_error",
                "auc": "roc_auc",
                "categorical_accuracy": "accuracy"
            }
            base_metrics = [sklearn_mapping.get(m, m) for m in base_metrics]
        elif framework == "tensorflow":
            # Remove balanced metrics as they are not standard Keras metric strings
            # Keras will use loss for optimization and we calculate balanced metrics
            # using sklearn in _log_training_results
            for metric in ["balanced_accuracy", "balanced_precision", "balanced_recall"]:
                if metric in base_metrics:
                    base_metrics.remove(metric)

        return base_metrics

    @staticmethod
    def get_scoring_metric(task_type: TaskType, framework: str = "sklearn") -> str:
        """
        Get default scoring metric for hyperparameter optimization.

        Args:
            task_type: Detected task type
            framework: ML framework

        Returns:
            str: Scoring metric name
        """
        return ModelControllerUtils.SKLEARN_SCORING[task_type]

    @staticmethod
    def validate_loss_compatibility(loss: str, task_type: TaskType, framework: str = "sklearn") -> bool:
        """
        Validate if loss function is compatible with task type.

        Args:
            loss: Loss function name
            task_type: Task type
            framework: ML framework

        Returns:
            bool: True if compatible, False otherwise
        """
        # Regression losses
        regression_losses = {
            "mse", "mean_squared_error", "squared_error",
            "mae", "mean_absolute_error",
            "huber", "huber_loss",
            "quantile", "quantile_loss"
        }

        # Classification losses
        classification_losses = {
            "binary_crossentropy", "log_loss", "logistic",
            "categorical_crossentropy", "sparse_categorical_crossentropy",
            "hinge", "squared_hinge"
        }

        if task_type == TaskType.REGRESSION:
            return loss.lower() in regression_losses
        else:  # Binary or multi-class classification
            return loss.lower() in classification_losses

    @staticmethod
    def get_best_score_metric(task_type: TaskType) -> Tuple[str, bool]:
        """
        Get the primary metric for determining "best" score.

        Args:
            task_type: Task type

        Returns:
            Tuple[str, bool]: (metric_name, higher_is_better)
        """
        if task_type == TaskType.REGRESSION:
            return "mse", False  # Lower MSE is better
        else:  # Classification
            return "balanced_accuracy", True  # Higher balanced accuracy is better

    @staticmethod
    def format_scores(scores: Dict[str, float], precision: int = 4) -> str:
        """
        Format scores dictionary for pretty printing.

        Args:
            scores: Dictionary of scores
            precision: Number of decimal places

        Returns:
            str: Formatted scores string
        """
        if not scores:
            return "No scores available"

        formatted_items = []
        for metric, score in scores.items():
            formatted_items.append(f"{metric}: {score:.{precision}f}")

        return ", ".join(formatted_items)


# Backward compatibility - keep ModelUtils as alias
ModelUtils = ModelControllerUtils
