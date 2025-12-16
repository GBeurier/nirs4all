"""
Ensemble Prediction Utilities - Weighted averaging for ensemble predictions

This module provides utilities for combining predictions from multiple models
using weighted averaging based on their scores. Relocated from utils/model_utils.py
to be with data/prediction modules.

Supports both regression (numeric averaging) and classification (soft/hard voting).
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class EnsembleUtils:
    """Utilities for ensemble prediction with weighted averaging and voting."""

    # =========================================================================
    # Classification Ensemble Methods (Soft/Hard Voting)
    # =========================================================================

    @staticmethod
    def compute_soft_voting_average(
        probability_arrays: List[np.ndarray],
        weights: Optional[np.ndarray] = None,
        use_confidence_weighting: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute soft voting average of class probabilities.

        Averages probability distributions from multiple models (weighted or simple),
        then takes argmax to get final class predictions.

        Args:
            probability_arrays: List of probability arrays, each shape (n_samples, n_classes).
                               Arrays can have different numbers of classes; they will be
                               padded/aligned to the maximum number of classes found.
            weights: Optional weights for each model (fold weights based on validation scores).
                    If None, uses uniform weights.
            use_confidence_weighting: If True, additionally weight each fold's contribution
                    per-sample by its prediction confidence (max probability).
                    This gives more influence to confident predictions.

        Returns:
            Tuple of:
                - class_predictions: Class labels as (n_samples, 1) array
                - averaged_probabilities: Averaged probabilities (n_samples, n_classes)

        Raises:
            ValueError: If probability_arrays is empty or sample counts don't match.
        """
        if not probability_arrays:
            raise ValueError("probability_arrays cannot be empty")

        # Validate sample counts and find max classes
        n_samples = probability_arrays[0].shape[0]
        max_classes = probability_arrays[0].shape[1] if probability_arrays[0].ndim > 1 else 1

        for i, arr in enumerate(probability_arrays):
            if arr.shape[0] != n_samples:
                raise ValueError(f"Array {i} has {arr.shape[0]} samples, expected {n_samples}")
            n_classes = arr.shape[1] if arr.ndim > 1 else 1
            max_classes = max(max_classes, n_classes)

        # Align arrays to max_classes by padding with zeros
        aligned_arrays = []
        for arr in probability_arrays:
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            n_classes = arr.shape[1]
            if n_classes < max_classes:
                # Pad with zeros for missing classes
                padding = np.zeros((n_samples, max_classes - n_classes), dtype=arr.dtype)
                arr = np.hstack([arr, padding])
            aligned_arrays.append(arr)

        n_models = len(aligned_arrays)

        # Default to uniform fold weights
        if weights is None:
            fold_weights = np.ones(n_models) / n_models
        else:
            fold_weights = np.asarray(weights, dtype=float)
            # Normalize weights to sum to 1
            fold_weights = fold_weights / np.sum(fold_weights)

        if use_confidence_weighting:
            # Confidence-weighted averaging: each sample gets per-fold weights
            # based on prediction confidence (max probability)
            averaged_probs = np.zeros((n_samples, max_classes), dtype=float)

            for sample_idx in range(n_samples):
                # Compute confidence for each fold at this sample
                confidences = np.array([
                    np.max(probs[sample_idx]) for probs in aligned_arrays
                ])

                # Combine fold weights with confidence weights
                combined_weights = fold_weights * confidences
                combined_weights = combined_weights / np.sum(combined_weights)  # Normalize

                # Weighted average for this sample
                for fold_idx, probs in enumerate(aligned_arrays):
                    averaged_probs[sample_idx] += combined_weights[fold_idx] * probs[sample_idx]
        else:
            # Standard weighted average of probabilities
            averaged_probs = np.zeros((n_samples, max_classes), dtype=float)
            for probs, w in zip(aligned_arrays, fold_weights):
                averaged_probs += w * probs

        # Get class predictions via argmax
        class_predictions = np.argmax(averaged_probs, axis=1).reshape(-1, 1).astype(float)

        return class_predictions, averaged_probs

    @staticmethod
    def compute_hard_voting(
        class_predictions: List[np.ndarray],
        weights: Optional[np.ndarray] = None,
        n_classes: Optional[int] = None
    ) -> np.ndarray:
        """Compute hard voting (majority vote) from class predictions.

        Each model votes for a class, and the class with most votes wins.
        Supports weighted voting where each model's vote is weighted.

        Args:
            class_predictions: List of class prediction arrays, each shape (n_samples,) or (n_samples, 1).
            weights: Optional weights for each model's vote.
                    If None, uses uniform weights (standard majority vote).
            n_classes: Number of classes. If None, inferred from predictions.

        Returns:
            Final class predictions as (n_samples, 1) array.

        Raises:
            ValueError: If class_predictions is empty.
        """
        if not class_predictions:
            raise ValueError("class_predictions cannot be empty")

        n_models = len(class_predictions)
        n_samples = class_predictions[0].shape[0]

        # Flatten all predictions to 1D
        predictions = [np.asarray(p).flatten().astype(int) for p in class_predictions]

        # Infer n_classes if not provided
        if n_classes is None:
            n_classes = max(p.max() for p in predictions) + 1

        # Default to uniform weights
        if weights is None:
            weights = np.ones(n_models)
        else:
            weights = np.asarray(weights, dtype=float)

        # Count weighted votes for each class per sample
        vote_counts = np.zeros((n_samples, n_classes), dtype=float)
        for pred, w in zip(predictions, weights):
            for sample_idx in range(n_samples):
                class_idx = pred[sample_idx]
                vote_counts[sample_idx, class_idx] += w

        # Get winning class (most votes)
        final_predictions = np.argmax(vote_counts, axis=1).reshape(-1, 1).astype(float)

        return final_predictions

    # =========================================================================
    # Regression Ensemble Methods (Weighted Averaging)
    # =========================================================================

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
