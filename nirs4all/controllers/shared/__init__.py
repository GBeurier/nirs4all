"""
Shared utilities for controllers.

This module provides shared utilities used by multiple controllers,
particularly for model selection and prediction aggregation which are
needed by both MergeController and MetaModelController.

Phase 2 Components (Stacking Restoration):
    ModelSelector: Utility class for selecting models based on validation metrics.
    PredictionAggregator: Utility class for aggregating predictions from multiple models.

These utilities were extracted from MergeController to avoid code duplication
and provide a single source of truth for model selection and aggregation logic.

Example:
    >>> from nirs4all.controllers.shared import ModelSelector, PredictionAggregator
    >>> from nirs4all.operators.data.merge import AggregationStrategy
    >>>
    >>> selector = ModelSelector(prediction_store, context)
    >>> ranked_models = selector.select_models(available_models, config, branch_id=0)
    >>>
    >>> aggregated = PredictionAggregator.aggregate(
    ...     predictions={"PLS": pls_preds, "RF": rf_preds},
    ...     strategy=AggregationStrategy.MEAN,
    ... )
"""

from .model_selector import ModelSelector
from .prediction_aggregator import PredictionAggregator

__all__ = [
    'ModelSelector',
    'PredictionAggregator',
]
