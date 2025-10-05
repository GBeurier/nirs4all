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

    # Deprecated methods from controllers (for backward compatibility)
    @staticmethod
    def deprec_calculate_scores(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str = "auto"
    ) -> Dict[str, float]:
        """
        DEPRECATED: Calculate scores for the predictions (simplified version).
        Use calculate_scores() instead for better functionality.
        """
        try:
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        except ImportError:
            return {}

        # Ensure arrays are numpy and flatten
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # Auto-detect task type if needed
        if task_type == "auto":
            task_type = ModelUtils.deprec_detect_task_type(y_true)

        scores = {}

        try:
            if task_type == "regression":
                # Regression metrics
                scores['mse'] = mean_squared_error(y_true, y_pred)
                scores['rmse'] = np.sqrt(scores['mse'])
                scores['mae'] = mean_absolute_error(y_true, y_pred)
                scores['r2'] = r2_score(y_true, y_pred)

            elif task_type == "classification":
                # Classification metrics
                scores['accuracy'] = accuracy_score(y_true, y_pred)
                scores['f1'] = f1_score(y_true, y_pred, average='weighted')
                scores['precision'] = precision_score(y_true, y_pred, average='weighted')
                scores['recall'] = recall_score(y_true, y_pred, average='weighted')

        except Exception as e:
            print(f"⚠️ Error calculating scores: {e}")

        return scores

    @staticmethod
    def deprec_detect_task_type(y: np.ndarray) -> str:
        """
        DEPRECATED: Detect if this is a regression or classification task (simplified).
        Use detect_task_type() instead for better functionality.
        """
        y = np.asarray(y).flatten()

        # Check if all values are integers and within a reasonable range for classification
        if np.all(y == y.astype(int)) and len(np.unique(y)) < 50:
            return "classification"
        else:
            return "regression"

    @staticmethod
    def deprec_get_best_metric_for_task(task_type: str) -> tuple[str, bool]:
        """
        DEPRECATED: Get the best metric for a given task type (simplified).
        Use get_best_score_metric() instead.
        """
        if task_type == "regression":
            return "rmse", False  # Lower RMSE is better
        elif task_type == "classification":
            return "accuracy", True  # Higher accuracy is better
        else:
            return "rmse", False  # Default

    @staticmethod
    def deprec_format_scores(scores: Dict[str, float]) -> str:
        """
        DEPRECATED: Format scores dictionary into a readable string (simplified).
        Use format_scores() instead.
        """
        if not scores:
            return "no scores"

        formatted = []
        for metric, value in scores.items():
            if isinstance(value, (int, float)):
                formatted.append(f"{metric}={value:.4f}")
            else:
                formatted.append(f"{metric}={value}")

        return ", ".join(formatted)

    # Weighted averaging functionality (moved from controllers)
    @staticmethod
    def compute_weighted_average(
        arrays: List[np.ndarray],
        scores: List[float],
        metric: Optional[str] = None,
        higher_is_better: Optional[bool] = None
    ) -> np.ndarray:
        """
        Compute weighted average of arrays based on their scores.

        Args:
            arrays: List of numpy arrays to average (must have same shape)
            scores: List of scores corresponding to each array
            metric: Name of the metric (used to determine if higher is better)
                   Supported: 'mse', 'rmse', 'mae', 'r2', 'accuracy', 'f1', 'precision', 'recall'
            higher_is_better: Boolean indicating if higher scores are better
                             If None, will be inferred from metric name

        Returns:
            Weighted average array

        Raises:
            ValueError: If arrays have different shapes or invalid parameters
        """
        if not arrays:
            raise ValueError("arrays list cannot be empty")

        if len(arrays) != len(scores):
            raise ValueError(f"Number of arrays ({len(arrays)}) must match number of scores ({len(scores)})")

        # Convert to numpy arrays and validate shapes
        arrays = [np.asarray(arr) for arr in arrays]
        base_shape = arrays[0].shape

        for i, arr in enumerate(arrays):
            if arr.shape != base_shape:
                raise ValueError(f"Array {i} has shape {arr.shape}, expected {base_shape}")

        scores_array = np.asarray(scores, dtype=float)

        # Determine if higher scores are better
        if higher_is_better is None:
            if metric is None:
                raise ValueError("Either 'metric' or 'higher_is_better' must be specified")
            higher_is_better = ModelUtils._is_higher_better(metric)

        # Convert scores to weights
        weights = ModelUtils._scores_to_weights(scores_array, higher_is_better)

        # Compute weighted average
        weighted_sum = np.zeros_like(arrays[0], dtype=float)
        for arr, weight in zip(arrays, weights):
            weighted_sum += weight * arr

        return weighted_sum

    @staticmethod
    def _is_higher_better(metric: str) -> bool:
        """
        Determine if higher values are better for a given metric.
        """
        # Metrics where higher is better
        higher_better_metrics = {
            'r2', 'accuracy', 'f1', 'precision', 'recall',
            'auc', 'roc_auc', 'score'
        }

        # Metrics where lower is better
        lower_better_metrics = {
            'mse', 'rmse', 'mae', 'loss', 'error',
            'mean_squared_error', 'mean_absolute_error', 'root_mean_squared_error'
        }

        metric_lower = metric.lower()

        if metric_lower in higher_better_metrics:
            return True
        elif metric_lower in lower_better_metrics:
            return False
        else:
            # Default assumption: if it contains 'error', 'loss', or 'mse', lower is better
            if any(term in metric_lower for term in ['error', 'loss', 'mse', 'mae']):
                return False
            else:
                # Default to higher is better for unknown metrics
                return True

    @staticmethod
    def _scores_to_weights(scores: np.ndarray, higher_is_better: bool) -> np.ndarray:
        """
        Convert scores to normalized weights for weighted averaging.
        """
        scores = scores.astype(float)

        # Handle edge case: all scores are the same
        if np.allclose(scores, scores[0]):
            return np.ones_like(scores) / len(scores)

        if higher_is_better:
            # For higher-is-better metrics, use scores directly
            # Ensure non-negative by shifting if needed
            if np.min(scores) < 0:
                shifted_scores = scores - np.min(scores)
            else:
                shifted_scores = scores.copy()

            # Handle case where all shifted scores are zero
            if np.allclose(shifted_scores, 0):
                return np.ones_like(scores) / len(scores)

            weights = shifted_scores
        else:
            # For lower-is-better metrics, invert the scores
            min_score = np.min(scores)

            if min_score <= 0:
                # Shift scores to be positive
                shifted_scores = scores - min_score + 1e-8
            else:
                shifted_scores = scores.copy()

            # Invert: better (lower) scores get higher weights
            weights = 1.0 / shifted_scores

        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)

        return weights

    @staticmethod
    def compute_ensemble_prediction(
        predictions_data: List[Dict[str, Any]],
        score_metric: str = "test_score",
        prediction_key: str = "y_pred",
        metric_for_direction: Optional[str] = None,
        higher_is_better: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Compute ensemble prediction from a list of prediction dictionaries.
        """
        if not predictions_data:
            raise ValueError("predictions_data cannot be empty")

        # Extract arrays and scores
        arrays = []
        scores = []
        metadata = {
            'model_names': [],
            'individual_scores': [],
            'weights': [],
            'n_models': len(predictions_data)
        }

        for pred_dict in predictions_data:
            # Get prediction array
            if prediction_key not in pred_dict:
                raise ValueError(f"Prediction key '{prediction_key}' not found in prediction data")

            pred_array = pred_dict[prediction_key]
            if isinstance(pred_array, list):
                pred_array = np.array(pred_array)
            elif not isinstance(pred_array, np.ndarray):
                pred_array = np.asarray(pred_array)

            arrays.append(pred_array)

            # Get score
            if score_metric not in pred_dict:
                raise ValueError(f"Score metric '{score_metric}' not found in prediction data")

            score = pred_dict[score_metric]
            if score is None:
                raise ValueError(f"Score metric '{score_metric}' is None for one of the predictions")

            scores.append(float(score))

            # Collect metadata
            metadata['model_names'].append(pred_dict.get('model_name', 'unknown'))
            metadata['individual_scores'].append(score)

        # Determine scoring direction
        if higher_is_better is None:
            if metric_for_direction is None:
                # Try to infer from score_metric name
                metric_for_direction = score_metric
            higher_is_better = ModelUtils._is_higher_better(metric_for_direction)

        # Compute weighted average
        ensemble_pred = ModelUtils.compute_weighted_average(
            arrays=arrays,
            scores=scores,
            higher_is_better=higher_is_better
        )

        # Calculate weights for metadata
        weights = ModelUtils._scores_to_weights(np.array(scores), higher_is_better)
        metadata['weights'] = weights.tolist()
        metadata['weight_sum'] = float(np.sum(weights))  # Should be 1.0
        metadata['score_direction'] = 'higher_better' if higher_is_better else 'lower_better'

        # Create result dictionary
        result = {
            'y_pred': ensemble_pred,
            'ensemble_method': 'weighted_average',
            'score_metric': score_metric,
            'n_models': len(predictions_data),
            'metadata': metadata
        }

        # Copy other common fields from first prediction
        first_pred = predictions_data[0]
        for key in ['dataset_name', 'partition', 'task_type', 'y_true', 'n_samples', 'n_features']:
            if key in first_pred:
                result[key] = first_pred[key]

        return result
