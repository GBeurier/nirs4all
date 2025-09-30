"""
Model utility functions for task type detection, loss/metric configuration, and scoring.

This module provides utilities for:
- Automatic detection of regression, binary classification, or multi-class classification
- Default loss function and metric selection based on task type
- Score calculation and validation
"""

from typing import Dict, List, Any, Tuple, Optional, Union
from enum import Enum
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, classification_report
)
from sklearn.exceptions import UndefinedMetricWarning
import warnings


class TaskType(Enum):
    """Enumeration of machine learning task types."""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


class ModelUtils:
    """Utilities for model configuration and evaluation."""

    # Default loss functions by task type
    DEFAULT_LOSSES = {
        TaskType.REGRESSION: "mse",
        TaskType.BINARY_CLASSIFICATION: "binary_crossentropy",
        TaskType.MULTICLASS_CLASSIFICATION: "categorical_crossentropy"
    }

    # Default metrics by task type
    DEFAULT_METRICS = {
        TaskType.REGRESSION: ["mae", "mse"],
        TaskType.BINARY_CLASSIFICATION: ["accuracy", "auc"],
        TaskType.MULTICLASS_CLASSIFICATION: ["accuracy", "categorical_accuracy"]
    }

    # Sklearn scoring metrics by task type
    SKLEARN_SCORING = {
        TaskType.REGRESSION: "neg_mean_squared_error",
        TaskType.BINARY_CLASSIFICATION: "roc_auc",
        TaskType.MULTICLASS_CLASSIFICATION: "accuracy"
    }

    @staticmethod
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
        base_loss = ModelUtils.DEFAULT_LOSSES[task_type]

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
        base_metrics = ModelUtils.DEFAULT_METRICS[task_type].copy()

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
        return ModelUtils.SKLEARN_SCORING[task_type]

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
    def calculate_scores(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: TaskType,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate scores for predictions based on task type.

        Args:
            y_true: True values
            y_pred: Predicted values
            task_type: Task type
            metrics: List of metrics to calculate (None for defaults)

        Returns:
            Dict[str, float]: Dictionary of metric names and scores
        """
        if metrics is None:
            metrics = ModelUtils.get_default_metrics(task_type, "sklearn")

        scores = {}
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        try:
            if task_type == TaskType.REGRESSION:
                # Regression metrics
                if "mean_squared_error" in metrics or "mse" in metrics:
                    scores["mse"] = mean_squared_error(y_true, y_pred)
                if "mean_absolute_error" in metrics or "mae" in metrics:
                    scores["mae"] = mean_absolute_error(y_true, y_pred)
                if "r2_score" in metrics or "r2" in metrics:
                    scores["r2"] = r2_score(y_true, y_pred)
                if "rmse" in metrics:
                    scores["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))

            else:  # Classification
                # Ensure y_true and y_pred are suitable for classification
                try:
                    # For binary classification with probabilities, threshold at 0.5
                    if task_type == TaskType.BINARY_CLASSIFICATION and np.all((y_pred >= 0) & (y_pred <= 1)):
                        y_pred_class = (y_pred > 0.5).astype(int)
                    else:
                        # For classification, convert to integers if they are continuous
                        y_pred_class = np.round(y_pred).astype(int)

                    # Ensure y_true is also integer for classification
                    y_true_class = np.round(y_true).astype(int)

                    # Check if the data is actually suitable for classification
                    unique_true = np.unique(y_true_class)
                    unique_pred = np.unique(y_pred_class)

                    # If there are too many unique values, it might be a regression problem
                    if len(unique_true) > 100 or len(unique_pred) > 100:
                        raise ValueError("Too many unique classes - might be regression data")

                    scores["accuracy"] = accuracy_score(y_true_class, y_pred_class)

                    if "f1_score" in metrics or "f1" in metrics:
                        average = "binary" if task_type == TaskType.BINARY_CLASSIFICATION else "weighted"
                        scores["f1"] = f1_score(y_true_class, y_pred_class, average=average)

                    # Suppress sklearn warnings for precision and recall
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                        if "precision" in metrics:
                            average = "binary" if task_type == TaskType.BINARY_CLASSIFICATION else "weighted"
                            scores["precision"] = precision_score(y_true_class, y_pred_class, average=average, zero_division=0)

                        if "recall" in metrics:
                            average = "binary" if task_type == TaskType.BINARY_CLASSIFICATION else "weighted"
                            scores["recall"] = recall_score(y_true_class, y_pred_class, average=average, zero_division=0)

                    # AUC for binary classification with probabilities
                    if "auc" in metrics or "roc_auc" in metrics:
                        if task_type == TaskType.BINARY_CLASSIFICATION and len(np.unique(y_true_class)) == 2:
                            try:
                                scores["auc"] = roc_auc_score(y_true_class, y_pred)
                            except ValueError:
                                # If y_pred are class predictions, skip AUC
                                pass

                except (ValueError, TypeError) as class_error:
                    # If classification metrics fail, try to redetect task type
                    print(f"⚠️ Classification metrics failed ({class_error}), retrying with auto-detection")

                    # Re-detect task type more conservatively
                    actual_task_type = ModelUtils.detect_task_type(y_true, threshold=0.01)  # More strict threshold
                    if actual_task_type == TaskType.REGRESSION:
                        # Recalculate as regression
                        if "mse" in metrics or "mean_squared_error" in metrics:
                            scores["mse"] = mean_squared_error(y_true, y_pred)
                        if "mae" in metrics or "mean_absolute_error" in metrics:
                            scores["mae"] = mean_absolute_error(y_true, y_pred)
                        if "r2" in metrics or "r2_score" in metrics:
                            scores["r2"] = r2_score(y_true, y_pred)
                        if "rmse" in metrics:
                            scores["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
                    else:
                        # Still classification but data is problematic, skip metrics
                        print("⚠️ Unable to calculate classification metrics for problematic data")
                        scores["accuracy"] = 0.0
                        scores["f1"] = 0.0
                        scores["precision"] = 0.0
                        scores["recall"] = 0.0

                if "log_loss" in metrics:
                    try:
                        scores["log_loss"] = log_loss(y_true, y_pred)
                    except ValueError:
                        # If y_pred are class predictions, skip log_loss
                        pass

        except Exception as e:
            print(f"⚠️ Error calculating scores: {e}")

        return scores

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
            return "accuracy", True  # Higher accuracy is better

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
