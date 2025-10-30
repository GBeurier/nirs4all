"""
Ensemble Prediction Utilities - Weighted averaging for ensemble predictions

This module provides utilities for combining predictions from multiple models
using weighted averaging based on their scores. Relocated from utils/model_utils.py
to be with data/prediction modules.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class EnsembleUtils:
    """Utilities for ensemble prediction with weighted averaging."""

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
            higher_is_better = EnsembleUtils._is_higher_better(metric)

        # Convert scores to weights
        weights = EnsembleUtils._scores_to_weights(scores_array, higher_is_better)

        # Compute weighted average
        weighted_sum = np.zeros_like(arrays[0], dtype=float)
        for arr, weight in zip(arrays, weights):
            weighted_sum += weight * arr

        return weighted_sum

    @staticmethod
    def _is_higher_better(metric: str) -> bool:
        """
        Determine if higher values are better for a given metric.

        Args:
            metric: Metric name

        Returns:
            True if higher is better, False if lower is better
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

        Args:
            scores: Array of scores
            higher_is_better: Whether higher scores are better

        Returns:
            Array of normalized weights (sum to 1.0)
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

        Args:
            predictions_data: List of prediction dictionaries
            score_metric: Key to extract score from each prediction
            prediction_key: Key to extract predictions array from each prediction
            metric_for_direction: Metric name to infer direction (if higher_is_better is None)
            higher_is_better: Whether higher scores are better (None to infer)

        Returns:
            Dictionary with ensemble prediction and metadata

        Raises:
            ValueError: If predictions_data is empty or missing required keys
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
            higher_is_better = EnsembleUtils._is_higher_better(metric_for_direction)

        # Compute weighted average
        ensemble_pred = EnsembleUtils.compute_weighted_average(
            arrays=arrays,
            scores=scores,
            higher_is_better=higher_is_better
        )

        # Calculate weights for metadata
        weights = EnsembleUtils._scores_to_weights(np.array(scores), higher_is_better)
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
