"""
ModelSelector - Utility class for selecting models based on validation metrics.

This module provides model ranking and selection strategies for stacking
and branch merging operations. It handles model selection based on
validation metrics with support for various strategies.

Phase 2 Implementation (Stacking Restoration):
    Extracted from MergeController to provide shared model selection logic
    for both MergeController and MetaModelController.

Selection Strategies:
    - ALL: Use all available models
    - BEST: Use single best model by metric
    - TOP_K: Use top K models by metric
    - EXPLICIT: Use explicitly named models
    - REGEX: Use models matching pattern (future)
    - THRESHOLD: Use models above/below metric threshold (future)

Example:
    >>> from nirs4all.controllers.shared import ModelSelector
    >>> from nirs4all.operators.data.merge import BranchPredictionConfig
    >>>
    >>> selector = ModelSelector(prediction_store, context)
    >>> config = BranchPredictionConfig(branch=0, select="best", metric="rmse")
    >>> selected = selector.select_models(["PLS", "RF", "XGB"], config, branch_id=0)
    >>> print(selected)  # ["PLS"] (assuming PLS has best RMSE)
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from nirs4all.core.logging import get_logger
from nirs4all.operators.data.merge import (
    BranchPredictionConfig,
    SelectionStrategy,
)

if TYPE_CHECKING:
    from nirs4all.data.predictions import Predictions
    from nirs4all.pipeline.config.context import ExecutionContext

logger = get_logger(__name__)


class ModelSelector:
    """Utility class for selecting models based on validation metrics.

    Handles model ranking and selection strategies (all, best, top_k, explicit)
    for per-branch prediction collection and stacking operations.

    This class is shared between MergeController and MetaModelController
    to avoid code duplication.

    Attributes:
        prediction_store: Prediction storage instance.
        context: Execution context.
        LOWER_IS_BETTER_METRICS: Set of metrics where lower values are better.
    """

    # Metrics where lower values are better (for ascending sort)
    LOWER_IS_BETTER_METRICS = {"rmse", "mse", "mae", "mape", "log_loss", "nrmse", "nmse", "nmae"}

    def __init__(
        self,
        prediction_store: "Predictions",
        context: "ExecutionContext",
    ):
        """Initialize the model selector.

        Args:
            prediction_store: Prediction storage instance.
            context: Execution context.
        """
        self.prediction_store = prediction_store
        self.context = context
        self._score_cache: Dict[str, Dict[str, float]] = {}

    def select_models(
        self,
        available_models: List[str],
        config: BranchPredictionConfig,
        branch_id: int,
    ) -> List[str]:
        """Select models from available models based on config.

        Args:
            available_models: List of available model names in the branch.
            config: Per-branch prediction configuration.
            branch_id: Branch identifier.

        Returns:
            List of selected model names.

        Raises:
            ValueError: If explicit model selection references unknown models.
        """
        strategy = config.get_selection_strategy()

        if strategy == SelectionStrategy.ALL:
            return available_models

        elif strategy == SelectionStrategy.BEST:
            return self._select_best(
                available_models,
                config.metric,
                branch_id,
            )

        elif strategy == SelectionStrategy.TOP_K:
            assert isinstance(config.select, dict)
            k = config.select.get("top_k", 1)
            return self._select_top_k(
                available_models,
                k,
                config.metric,
                branch_id,
            )

        elif strategy == SelectionStrategy.EXPLICIT:
            assert isinstance(config.select, list)
            return self._select_explicit(
                available_models,
                config.select,
                branch_id,
            )

        # Fallback to all
        return available_models

    def _select_best(
        self,
        available_models: List[str],
        metric: Optional[str],
        branch_id: int,
    ) -> List[str]:
        """Select the single best model by validation metric.

        Args:
            available_models: List of available model names.
            metric: Metric to rank by (default: rmse).
            branch_id: Branch identifier.

        Returns:
            List with single best model name, or empty if no valid scores.
        """
        ranked = self._rank_models_by_metric(
            available_models, metric or "rmse", branch_id
        )
        return [ranked[0]] if ranked else []

    def _select_top_k(
        self,
        available_models: List[str],
        k: int,
        metric: Optional[str],
        branch_id: int,
    ) -> List[str]:
        """Select top K models by validation metric.

        Args:
            available_models: List of available model names.
            k: Number of models to select.
            metric: Metric to rank by (default: rmse).
            branch_id: Branch identifier.

        Returns:
            List of top K model names.
        """
        ranked = self._rank_models_by_metric(
            available_models, metric or "rmse", branch_id
        )
        return ranked[:min(k, len(ranked))]

    def _select_explicit(
        self,
        available_models: List[str],
        model_names: List[str],
        branch_id: int,
    ) -> List[str]:
        """Select explicitly named models.

        Args:
            available_models: List of available model names.
            model_names: Explicit list of model names to select.
            branch_id: Branch identifier.

        Returns:
            List of selected model names (intersection with available).

        Raises:
            ValueError: If any named model is not available.
        """
        available_set = set(available_models)
        selected = []

        for name in model_names:
            if name in available_set:
                selected.append(name)
            else:
                logger.warning(
                    f"Explicit model '{name}' not found in branch {branch_id}. "
                    f"Available models: {available_models}. Skipping."
                )

        return selected

    def _rank_models_by_metric(
        self,
        available_models: List[str],
        metric: str,
        branch_id: int,
    ) -> List[str]:
        """Rank models by validation metric score.

        Args:
            available_models: List of available model names.
            metric: Metric name to rank by.
            branch_id: Branch identifier.

        Returns:
            List of model names sorted by metric (best first).
        """
        model_scores: List[Tuple[str, float]] = []

        for model_name in available_models:
            score = self._get_model_validation_score(model_name, metric, branch_id)
            if score is not None and np.isfinite(score):
                model_scores.append((model_name, score))

        if not model_scores:
            logger.warning(
                f"No valid validation scores found for metric '{metric}' "
                f"in branch {branch_id}. Returning all models."
            )
            return available_models

        # Determine sort order based on metric
        ascending = metric.lower() in self.LOWER_IS_BETTER_METRICS

        # Sort by score
        model_scores.sort(key=lambda x: x[1], reverse=not ascending)

        logger.debug(
            f"Model ranking for branch {branch_id} by {metric}: "
            f"{[(m, f'{s:.4f}') for m, s in model_scores[:5]]}..."
        )

        return [m for m, _ in model_scores]

    def _get_model_validation_score(
        self,
        model_name: str,
        metric: str,
        branch_id: int,
    ) -> Optional[float]:
        """Get validation score for a model.

        Uses caching to avoid repeated prediction store queries.

        Args:
            model_name: Model name.
            metric: Metric name.
            branch_id: Branch identifier.

        Returns:
            Validation score or None if not found.
        """
        cache_key = f"{model_name}:{metric}:{branch_id}"

        if cache_key in self._score_cache:
            return self._score_cache.get(cache_key, {}).get(metric)

        # Query prediction store for validation predictions
        current_step = getattr(self.context.state, 'step_number', float('inf'))

        filter_kwargs = {
            'model_name': model_name,
            'branch_id': branch_id,
            'partition': 'val',
            'load_arrays': False,
        }

        predictions = self.prediction_store.filter_predictions(**filter_kwargs)

        # Filter by step
        predictions = [
            p for p in predictions
            if p.get('step_idx', 0) < current_step
        ]

        if not predictions:
            # Try without branch_id for pre-branch models
            filter_kwargs_no_branch = {
                'model_name': model_name,
                'partition': 'val',
                'load_arrays': False,
            }
            predictions = self.prediction_store.filter_predictions(**filter_kwargs_no_branch)
            predictions = [
                p for p in predictions
                if p.get('step_idx', 0) < current_step and p.get('branch_id') is None
            ]

        if not predictions:
            return None

        # Get score from first matching prediction
        # Priority: scores dict > val_score field
        for pred in predictions:
            # Try scores JSON dict first
            import json
            scores_json = pred.get("scores")
            if scores_json:
                try:
                    scores_dict = json.loads(scores_json) if isinstance(scores_json, str) else scores_json
                    if "val" in scores_dict and metric in scores_dict["val"]:
                        score = scores_dict["val"][metric]
                        self._score_cache[cache_key] = {metric: score}
                        return score
                except (json.JSONDecodeError, TypeError):
                    pass

            # Fallback to val_score if metric matches
            if metric == pred.get("metric"):
                score = pred.get("val_score")
                if score is not None:
                    self._score_cache[cache_key] = {metric: score}
                    return score

        return None

    def get_model_scores(
        self,
        model_names: List[str],
        metric: str,
        branch_id: int,
    ) -> Dict[str, float]:
        """Get validation scores for multiple models.

        Used for weighted aggregation.

        Args:
            model_names: List of model names.
            metric: Metric name.
            branch_id: Branch identifier.

        Returns:
            Dictionary mapping model name to score.
        """
        scores = {}
        for name in model_names:
            score = self._get_model_validation_score(name, metric, branch_id)
            if score is not None:
                scores[name] = score
        return scores

    def select_models_global(
        self,
        available_models: List[str],
        selection: Any,
        metric: Optional[str] = None,
    ) -> List[str]:
        """Select models globally (without branch context).

        This is used by MetaModelController for pipelines without branches.

        Args:
            available_models: List of available model names.
            selection: Selection configuration:
                - "all": Use all models
                - "best": Use best model
                - {"top_k": N}: Use top N models
                - ["model1", "model2"]: Explicit list
            metric: Optional metric for ranking.

        Returns:
            List of selected model names.
        """
        if selection == "all" or selection is None:
            return available_models

        if selection == "best":
            ranked = self._rank_models_by_metric_global(
                available_models, metric or "rmse"
            )
            return [ranked[0]] if ranked else []

        if isinstance(selection, dict):
            if "top_k" in selection:
                k = selection["top_k"]
                ranked = self._rank_models_by_metric_global(
                    available_models, metric or "rmse"
                )
                return ranked[:min(k, len(ranked))]

        if isinstance(selection, list):
            # Explicit list
            available_set = set(available_models)
            selected = [m for m in selection if m in available_set]
            for m in selection:
                if m not in available_set:
                    logger.warning(
                        f"Explicit model '{m}' not found. "
                        f"Available models: {available_models}. Skipping."
                    )
            return selected

        return available_models

    def _rank_models_by_metric_global(
        self,
        available_models: List[str],
        metric: str,
    ) -> List[str]:
        """Rank models globally (across all branches) by validation metric.

        Args:
            available_models: List of available model names.
            metric: Metric name to rank by.

        Returns:
            List of model names sorted by metric (best first).
        """
        model_scores: List[Tuple[str, float]] = []
        current_step = getattr(self.context.state, 'step_number', float('inf'))

        for model_name in available_models:
            # Get all predictions for this model across all branches
            filter_kwargs = {
                'model_name': model_name,
                'partition': 'val',
                'load_arrays': False,
            }
            predictions = self.prediction_store.filter_predictions(**filter_kwargs)
            predictions = [
                p for p in predictions
                if p.get('step_idx', 0) < current_step
            ]

            # Find best score across all folds/branches
            best_score = None
            for pred in predictions:
                # Try scores JSON dict first
                import json
                scores_json = pred.get("scores")
                if scores_json:
                    try:
                        scores_dict = json.loads(scores_json) if isinstance(scores_json, str) else scores_json
                        if "val" in scores_dict and metric in scores_dict["val"]:
                            score = scores_dict["val"][metric]
                            if best_score is None:
                                best_score = score
                            elif metric.lower() in self.LOWER_IS_BETTER_METRICS:
                                best_score = min(best_score, score)
                            else:
                                best_score = max(best_score, score)
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Fallback to val_score
                if best_score is None and metric == pred.get("metric"):
                    score = pred.get("val_score")
                    if score is not None:
                        best_score = score

            if best_score is not None and np.isfinite(best_score):
                model_scores.append((model_name, best_score))

        if not model_scores:
            logger.warning(
                f"No valid validation scores found for metric '{metric}'. "
                f"Returning all models."
            )
            return available_models

        # Determine sort order based on metric
        ascending = metric.lower() in self.LOWER_IS_BETTER_METRICS

        # Sort by score
        model_scores.sort(key=lambda x: x[1], reverse=not ascending)

        logger.debug(
            f"Global model ranking by {metric}: "
            f"{[(m, f'{s:.4f}') for m, s in model_scores[:5]]}..."
        )

        return [m for m, _ in model_scores]
