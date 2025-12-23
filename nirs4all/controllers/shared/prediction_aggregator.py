"""
PredictionAggregator - Utility class for aggregating predictions from multiple models.

This module provides aggregation strategies for combining predictions within
a branch or across models for stacking and merge operations.

Phase 2 Implementation (Stacking Restoration):
    Extracted from MergeController to provide shared prediction aggregation logic
    for both MergeController and MetaModelController.

Aggregation Strategies:
    - SEPARATE: Stack predictions as separate features (n_models features)
    - MEAN: Simple average of predictions (1 feature)
    - WEIGHTED_MEAN: Weighted average by validation score (1 feature)
    - PROBA_MEAN: Average class probabilities for classification (n_classes features)

Example:
    >>> from nirs4all.controllers.shared import PredictionAggregator
    >>> from nirs4all.operators.data.merge import AggregationStrategy
    >>> import numpy as np
    >>>
    >>> predictions = {
    ...     "PLS": np.array([1.0, 2.0, 3.0]),
    ...     "RF": np.array([1.1, 1.9, 3.1]),
    ... }
    >>> aggregated = PredictionAggregator.aggregate(
    ...     predictions=predictions,
    ...     strategy=AggregationStrategy.MEAN,
    ... )
    >>> print(aggregated.shape)  # (3, 1)
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from nirs4all.core.logging import get_logger
from nirs4all.operators.data.merge import AggregationStrategy

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class PredictionAggregator:
    """Utility class for aggregating predictions from multiple models.

    Handles aggregation strategies (separate, mean, weighted_mean, proba_mean)
    for combining predictions within a branch or across models.

    This class is shared between MergeController and MetaModelController
    to avoid code duplication.

    All methods are static as no instance state is needed.
    """

    # Reference to LOWER_IS_BETTER_METRICS for weighted_mean
    LOWER_IS_BETTER_METRICS = {"rmse", "mse", "mae", "mape", "log_loss", "nrmse", "nmse", "nmae"}

    @staticmethod
    def aggregate(
        predictions: Dict[str, np.ndarray],
        strategy: AggregationStrategy,
        model_scores: Optional[Dict[str, float]] = None,
        proba: bool = False,
        metric: Optional[str] = None,
    ) -> np.ndarray:
        """Aggregate predictions from multiple models.

        Args:
            predictions: Dictionary mapping model names to prediction arrays.
                Each array has shape (n_samples,) for regression or
                (n_samples, n_classes) for classification probabilities.
            strategy: Aggregation strategy to use.
            model_scores: Optional dictionary of model scores for weighted averaging.
            proba: Whether predictions are class probabilities.
            metric: Metric name (for determining weight direction).

        Returns:
            Aggregated predictions with shape:
                - SEPARATE: (n_samples, n_models)
                - MEAN/WEIGHTED_MEAN: (n_samples, 1)
                - PROBA_MEAN: (n_samples, n_classes)

        Raises:
            ValueError: If predictions dict is empty.
        """
        if not predictions:
            raise ValueError("Cannot aggregate empty predictions dictionary")

        model_names = list(predictions.keys())

        if strategy == AggregationStrategy.SEPARATE:
            return PredictionAggregator._aggregate_separate(predictions, model_names)

        elif strategy == AggregationStrategy.MEAN:
            return PredictionAggregator._aggregate_mean(predictions, model_names, proba)

        elif strategy == AggregationStrategy.WEIGHTED_MEAN:
            return PredictionAggregator._aggregate_weighted_mean(
                predictions, model_names, model_scores, metric, proba
            )

        elif strategy == AggregationStrategy.PROBA_MEAN:
            return PredictionAggregator._aggregate_proba_mean(predictions, model_names)

        # Default to separate
        return PredictionAggregator._aggregate_separate(predictions, model_names)

    @staticmethod
    def _aggregate_separate(
        predictions: Dict[str, np.ndarray],
        model_names: List[str],
    ) -> np.ndarray:
        """Stack predictions as separate features.

        Args:
            predictions: Dictionary mapping model names to prediction arrays.
            model_names: Ordered list of model names.

        Returns:
            2D array with shape (n_samples, n_models).
        """
        arrays = []
        for name in model_names:
            arr = predictions[name]
            # Ensure 2D
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim > 2:
                # Flatten extra dimensions
                arr = arr.reshape(arr.shape[0], -1)
            arrays.append(arr)

        return np.hstack(arrays)

    @staticmethod
    def _aggregate_mean(
        predictions: Dict[str, np.ndarray],
        model_names: List[str],
        proba: bool = False,
    ) -> np.ndarray:
        """Simple average of predictions.

        Args:
            predictions: Dictionary mapping model names to prediction arrays.
            model_names: Ordered list of model names.
            proba: Whether predictions are class probabilities.

        Returns:
            2D array with shape (n_samples, 1) for regression,
            or (n_samples, n_classes) for proba.
        """
        arrays = [predictions[name] for name in model_names]

        if proba:
            # Average probabilities, maintaining class dimension
            # Ensure all arrays have same shape
            max_classes = max(arr.shape[1] if arr.ndim > 1 else 1 for arr in arrays)
            aligned = []
            for arr in arrays:
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                if arr.shape[1] < max_classes:
                    # Pad with zeros
                    padding = np.zeros((arr.shape[0], max_classes - arr.shape[1]))
                    arr = np.hstack([arr, padding])
                aligned.append(arr)

            stacked = np.stack(aligned, axis=0)  # (n_models, n_samples, n_classes)
            return np.mean(stacked, axis=0)  # (n_samples, n_classes)
        else:
            # Stack and average regression predictions
            stacked = np.stack([arr.flatten() for arr in arrays], axis=0)  # (n_models, n_samples)
            mean_pred = np.mean(stacked, axis=0)  # (n_samples,)
            return mean_pred.reshape(-1, 1)  # (n_samples, 1)

    @staticmethod
    def _aggregate_weighted_mean(
        predictions: Dict[str, np.ndarray],
        model_names: List[str],
        model_scores: Optional[Dict[str, float]],
        metric: Optional[str],
        proba: bool = False,
    ) -> np.ndarray:
        """Weighted average of predictions based on validation scores.

        Args:
            predictions: Dictionary mapping model names to prediction arrays.
            model_names: Ordered list of model names.
            model_scores: Dictionary of model scores for weighting.
            metric: Metric name (for determining weight direction).
            proba: Whether predictions are class probabilities.

        Returns:
            2D array with shape (n_samples, 1) for regression,
            or (n_samples, n_classes) for proba.
        """
        if not model_scores:
            logger.warning(
                "No model scores provided for weighted_mean aggregation. "
                "Falling back to simple mean."
            )
            return PredictionAggregator._aggregate_mean(predictions, model_names, proba)

        # Compute weights from scores
        weights = []
        valid_models = []

        # Determine if higher or lower scores are better
        lower_is_better = (metric or "rmse").lower() in PredictionAggregator.LOWER_IS_BETTER_METRICS

        for name in model_names:
            score = model_scores.get(name)
            if score is not None and np.isfinite(score):
                # For metrics where lower is better, invert the score
                if lower_is_better:
                    # Use 1/score for weighting (better = higher weight)
                    weight = 1.0 / (score + 1e-10) if score >= 0 else abs(score)
                else:
                    # Use score directly (higher = better)
                    weight = max(score, 0.0)
                weights.append(weight)
                valid_models.append(name)

        if not weights:
            logger.warning("No valid weights computed. Falling back to simple mean.")
            return PredictionAggregator._aggregate_mean(predictions, model_names, proba)

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        logger.debug(
            f"Weighted mean weights: {list(zip(valid_models, [f'{w:.3f}' for w in weights]))}"
        )

        if proba:
            # Weighted average of probabilities
            max_classes = max(
                predictions[name].shape[1] if predictions[name].ndim > 1 else 1
                for name in valid_models
            )
            n_samples = predictions[valid_models[0]].shape[0]
            weighted_proba = np.zeros((n_samples, max_classes))

            for name, weight in zip(valid_models, weights):
                arr = predictions[name]
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                if arr.shape[1] < max_classes:
                    padding = np.zeros((arr.shape[0], max_classes - arr.shape[1]))
                    arr = np.hstack([arr, padding])
                weighted_proba += weight * arr

            return weighted_proba
        else:
            # Weighted average of regression predictions
            n_samples = predictions[valid_models[0]].shape[0]
            weighted_sum = np.zeros(n_samples)

            for name, weight in zip(valid_models, weights):
                weighted_sum += weight * predictions[name].flatten()

            return weighted_sum.reshape(-1, 1)

    @staticmethod
    def _aggregate_proba_mean(
        predictions: Dict[str, np.ndarray],
        model_names: List[str],
    ) -> np.ndarray:
        """Average class probabilities from classifiers.

        Args:
            predictions: Dictionary mapping model names to probability arrays.
            model_names: Ordered list of model names.

        Returns:
            2D array with shape (n_samples, n_classes).
        """
        arrays = [predictions[name] for name in model_names]

        # Ensure all have same class dimension
        max_classes = max(arr.shape[1] if arr.ndim > 1 else 1 for arr in arrays)
        aligned = []

        for arr in arrays:
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[1] < max_classes:
                # Pad with zeros for missing classes
                padding = np.zeros((arr.shape[0], max_classes - arr.shape[1]))
                arr = np.hstack([arr, padding])
            aligned.append(arr)

        stacked = np.stack(aligned, axis=0)  # (n_models, n_samples, n_classes)
        return np.mean(stacked, axis=0)  # (n_samples, n_classes)

    @staticmethod
    def aggregate_folds(
        fold_predictions: List[np.ndarray],
        fold_scores: Optional[List[float]] = None,
        strategy: str = "mean",
        metric: Optional[str] = None,
    ) -> np.ndarray:
        """Aggregate predictions across CV folds.

        Useful for combining test predictions from different folds.

        Args:
            fold_predictions: List of prediction arrays, one per fold.
            fold_scores: Optional list of validation scores per fold.
            strategy: Aggregation strategy ("mean", "weighted_mean", "best").
            metric: Metric name for weighted aggregation.

        Returns:
            Aggregated predictions.
        """
        if not fold_predictions:
            raise ValueError("Cannot aggregate empty fold predictions")

        if len(fold_predictions) == 1:
            return fold_predictions[0]

        if strategy == "mean":
            return np.mean(fold_predictions, axis=0)

        elif strategy == "weighted_mean":
            if fold_scores is None:
                logger.warning("No fold scores for weighted aggregation. Using mean.")
                return np.mean(fold_predictions, axis=0)

            # Compute weights
            lower_is_better = (metric or "rmse").lower() in PredictionAggregator.LOWER_IS_BETTER_METRICS

            weights = []
            for score in fold_scores:
                if lower_is_better:
                    weight = 1.0 / (score + 1e-10) if score >= 0 else abs(score)
                else:
                    weight = max(score, 0.0)
                weights.append(weight)

            # Normalize
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                weights = [1.0 / len(weights)] * len(weights)

            return np.average(fold_predictions, axis=0, weights=weights)

        elif strategy == "best":
            if fold_scores is None:
                return fold_predictions[0]

            lower_is_better = (metric or "rmse").lower() in PredictionAggregator.LOWER_IS_BETTER_METRICS

            if lower_is_better:
                best_idx = np.argmin(fold_scores)
            else:
                best_idx = np.argmax(fold_scores)

            return fold_predictions[best_idx]

        else:
            logger.warning(f"Unknown fold aggregation strategy: {strategy}. Using mean.")
            return np.mean(fold_predictions, axis=0)
