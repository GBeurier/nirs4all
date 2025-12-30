"""
PredictionAnalyzer - Orchestrator for prediction analysis and visualization.

This module provides a unified interface for creating various prediction visualizations.
Delegates to specialized chart classes for rendering.

Leverages the refactored Predictions API (predictions.top(), PredictionResult, etc.)
for efficient data access and avoids redundant calculations.

Includes a caching layer (PredictionCache) to avoid recomputing expensive aggregations
when multiple charts use the same parameters.
"""
from matplotlib.figure import Figure
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import os
import re
import glob
import numpy as np
import pandas as pd
import polars as pl

from nirs4all.data.predictions import Predictions
from nirs4all.visualization.prediction_cache import PredictionCache
from nirs4all.visualization.charts import (
    ChartConfig,
    ScoreHistogramChart,
    CandlestickChart,
    ConfusionMatrixChart,
    TopKComparisonChart,
    HeatmapChart
)


def _get_default_figures_dir() -> str:
    """Get the default figures output directory.

    Checks NIRS4ALL_WORKSPACE environment variable first, then falls back
    to ./workspace in the current working directory.

    Returns:
        Default figures directory path as string.
    """
    env_workspace = os.environ.get("NIRS4ALL_WORKSPACE")
    if env_workspace:
        return str(Path(env_workspace) / "figures")
    return "workspace/figures"
from nirs4all.core import metrics as evaluator
from nirs4all.core.metrics import abbreviate_metric
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


class PredictionAnalyzer:
    """Orchestrator for prediction analysis and visualization.

    Provides a unified interface for creating various prediction visualizations.
    Delegates to specialized chart classes for rendering.

    Includes a caching layer (PredictionCache) to avoid recomputing expensive
    aggregations when multiple charts use the same parameters. The cache is
    keyed by (aggregate, rank_metric, rank_partition, display_partition, group_by,
    filters) and stores the results of predictions.top() calls.

    Leverages the refactored Predictions API (predictions.top(), PredictionResult, etc.)
    for efficient data access and avoids redundant calculations.

    Attributes:
        predictions: Predictions object containing prediction data.
        dataset_name_override: Optional dataset name override for display.
        config: ChartConfig for customization across all charts.
        output_dir: Directory to save generated charts.
        cache: PredictionCache for caching aggregated results.
        default_aggregate: Default aggregation column for all visualization methods.

    Example:
        >>> from nirs4all.data.predictions import Predictions
        >>> predictions = Predictions.load('predictions.json')
        >>> analyzer = PredictionAnalyzer(predictions)
        >>>
        >>> # Plot top 5 models - first call computes aggregation
        >>> fig = analyzer.plot_top_k(k=5, aggregate='ID')
        >>>
        >>> # Plot heatmap - uses cached aggregation (fast!)
        >>> fig = analyzer.plot_heatmap('model_name', 'preprocessings', aggregate='ID')
        >>>
        >>> # Check cache stats
        >>> print(analyzer.get_cache_stats())
        >>>
        >>> # With default aggregation from dataset config
        >>> runner = PipelineRunner()
        >>> predictions, _ = runner.run(pipeline, DatasetConfigs(path, aggregate='sample_id'))
        >>> analyzer = PredictionAnalyzer(predictions, default_aggregate=runner.last_aggregate)
        >>> # All plots now use sample_id aggregation by default
        >>> fig = analyzer.plot_top_k(k=5)  # Aggregated automatically
    """

    def __init__(
        self,
        predictions_obj: Predictions,
        dataset_name_override: Optional[str] = None,
        config: Optional[ChartConfig] = None,
        output_dir: Optional[str] = None,
        cache_size: int = 50,
        default_aggregate: Optional[str] = None,
        default_aggregate_method: Optional[str] = None,
        default_aggregate_exclude_outliers: bool = False
    ):
        """Initialize analyzer with predictions object.

        Args:
            predictions_obj: The predictions object containing prediction data.
            dataset_name_override: Optional dataset name override for display.
            config: Optional ChartConfig for customization across all charts.
            output_dir: Directory to save generated charts. If None, uses
                NIRS4ALL_WORKSPACE/figures or defaults to "workspace/figures".
            cache_size: Maximum number of cached query results. Defaults to 50.
            default_aggregate: Default aggregation column for all visualization methods.
                If set, all plots will use this aggregation unless overridden.
                Can be 'y' (aggregate by target values) or a metadata column name.
                Typically obtained from `runner.last_aggregate` after a pipeline run.
            default_aggregate_method: Default aggregation method for all visualization methods.
                - None (default): Use 'mean' for regression, 'vote' for classification
                - 'mean': Average predictions within each group
                - 'median': Median prediction within each group
                - 'vote': Majority voting (for classification)
            default_aggregate_exclude_outliers: If True, exclude outliers using T² statistic
                before aggregation (default: False).

        Example:
            >>> # With default aggregation from dataset config
            >>> runner = PipelineRunner()
            >>> predictions, _ = runner.run(pipeline, DatasetConfigs(path, aggregate='sample_id'))
            >>> analyzer = PredictionAnalyzer(predictions, default_aggregate=runner.last_aggregate)
            >>> # All plots now use sample_id aggregation by default
            >>> fig = analyzer.plot_top_k(k=5)  # Aggregated
            >>> fig = analyzer.plot_top_k(k=5, aggregate=None)  # Explicit override to no aggregation
        """
        self.predictions = predictions_obj
        self.dataset_name_override = dataset_name_override
        self.config = config or ChartConfig()
        self.output_dir = output_dir if output_dir is not None else _get_default_figures_dir()
        self._cache = PredictionCache(max_entries=cache_size)
        self.default_aggregate = default_aggregate
        self.default_aggregate_method = default_aggregate_method
        self.default_aggregate_exclude_outliers = default_aggregate_exclude_outliers

    def clear_cache(self) -> None:
        """Clear all caches.

        Call this if the underlying predictions data has been modified
        to ensure fresh results are computed. Clears both:
        - Analyzer's query result cache
        - Ranker's aggregation and score caches
        """
        self._cache.clear()
        self.predictions.clear_caches()

    def _resolve_aggregate(self, aggregate: Optional[str]) -> Optional[str]:
        """Resolve effective aggregate value, considering default.

        Args:
            aggregate: Explicit aggregate value passed to method.
                - None: Use default_aggregate (if set).
                - Explicit value: Use that value (overrides default).

        Returns:
            Effective aggregate column name or None.

        Note:
            To explicitly disable aggregation when a default is set,
            callers should use an empty string '' which is truthy enough
            to override but evaluates to no aggregation.
        """
        # If aggregate is explicitly provided (not None), use it
        if aggregate is not None:
            return aggregate if aggregate else None  # '' means no aggregation
        # Otherwise, fall back to default
        return self.default_aggregate

    def _resolve_aggregate_method(self, method: Optional[str]) -> Optional[str]:
        """Resolve effective aggregate method, considering default.

        Args:
            method: Explicit method value passed to method.
                - None: Use default_aggregate_method (if set).
                - Explicit value: Use that value (overrides default).

        Returns:
            Effective aggregate method or None.
        """
        if method is not None:
            return method
        return self.default_aggregate_method

    def _resolve_aggregate_exclude_outliers(self, exclude: Optional[bool]) -> bool:
        """Resolve effective aggregate exclude_outliers, considering default.

        Args:
            exclude: Explicit exclude_outliers value passed to method.
                - None: Use default_aggregate_exclude_outliers.
                - Explicit value: Use that value (overrides default).

        Returns:
            Effective exclude_outliers setting.
        """
        if exclude is not None:
            return exclude
        return self.default_aggregate_exclude_outliers

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with stats for both analyzer and ranker caches:
            - analyzer_cache: Query result cache stats
            - ranker_cache: Aggregation and score cache stats
        """
        return {
            'analyzer_cache': self._cache.get_stats(),
            'ranker_cache': self.predictions.get_cache_stats()
        }

    def get_cached_predictions(
        self,
        n: int,
        rank_metric: str,
        rank_partition: str = 'val',
        display_partition: str = 'test',
        display_metrics: Optional[List[str]] = None,
        aggregate: Optional[str] = None,
        aggregate_method: Optional[str] = None,
        aggregate_exclude_outliers: Optional[bool] = None,
        group_by: Optional[Union[str, List[str]]] = None,
        aggregate_partitions: bool = True,
        **filters
    ):
        """Get predictions with caching support.

        This method wraps predictions.top() with a caching layer.
        Charts should call this method instead of directly calling
        predictions.top() to benefit from caching.

        The cache key includes: aggregate, rank_metric, rank_partition,
        display_partition, group_by, and all filters.

        Args:
            n: Number of top predictions to return.
            rank_metric: Metric for ranking.
            rank_partition: Partition for ranking (default: 'val').
            display_partition: Partition for display (default: 'test').
            display_metrics: List of metrics to compute for display.
            aggregate: Aggregation column (e.g., 'ID') or None.
            aggregate_method: Aggregation method ('mean', 'median', 'vote').
                If None, uses default_aggregate_method from constructor.
            aggregate_exclude_outliers: If True, exclude outliers using T² before aggregation.
                If None, uses default_aggregate_exclude_outliers from constructor.
            group_by: Grouping column(s) for deduplication.
            aggregate_partitions: If True, include all partition data.
            **filters: Additional filter criteria.

        Returns:
            PredictionResultsList from cache or fresh computation.

        Example:
            >>> # First call computes and caches
            >>> preds = analyzer.get_cached_predictions(
            ...     n=5, rank_metric='rmse', aggregate='ID'
            ... )
            >>> # Second call with same params is instant
            >>> preds = analyzer.get_cached_predictions(
            ...     n=5, rank_metric='rmse', aggregate='ID'
            ... )
        """
        # Resolve effective aggregate settings
        effective_method = aggregate_method if aggregate_method is not None else self.default_aggregate_method
        effective_exclude = aggregate_exclude_outliers if aggregate_exclude_outliers is not None else self.default_aggregate_exclude_outliers

        # Create cache key (n is intentionally excluded as we cache large result sets)
        # We request more than needed and slice, so the same cache entry serves
        # multiple calls with different n values
        cache_n = max(n, 1000)  # Cache a larger result set

        cache_key = self._cache.make_key(
            aggregate=aggregate,
            aggregate_method=effective_method,
            aggregate_exclude_outliers=effective_exclude,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            display_partition=display_partition,
            group_by=group_by,
            **filters
        )

        def compute():
            # Determine ascending order based on metric type
            ascending = not self._is_higher_better(rank_metric)

            # Prepare display_metrics if not provided
            effective_display_metrics = display_metrics
            if effective_display_metrics is None:
                effective_display_metrics = [rank_metric]
            elif rank_metric not in effective_display_metrics:
                effective_display_metrics = [rank_metric] + list(effective_display_metrics)

            return self.predictions.top(
                n=cache_n,
                rank_metric=rank_metric,
                rank_partition=rank_partition,
                display_partition=display_partition,
                display_metrics=effective_display_metrics,
                ascending=ascending,
                aggregate_partitions=aggregate_partitions,
                aggregate=aggregate,
                aggregate_method=effective_method,
                aggregate_exclude_outliers=effective_exclude,
                group_by=group_by,
                **filters
            )

        # Get from cache or compute
        result = self._cache.get_or_compute(cache_key, compute)

        # Return only the requested number of results
        return result[:n] if len(result) > n else result

    def _get_default_metric(self) -> str:
        """Get default metric based on task type from predictions.

        Returns:
            'balanced_accuracy' for classification, 'rmse' for regression.
        """
        if self.predictions.num_predictions > 0:
            try:
                task_types = self.predictions.get_unique_values('task_type')
                if any(t and 'classification' in str(t).lower() for t in task_types):
                    return 'balanced_accuracy'
            except Exception:
                pass
        return 'rmse'

    @staticmethod
    def _is_higher_better(metric: str) -> bool:
        """Check if metric is higher-is-better.

        Args:
            metric: Metric name to check.

        Returns:
            True if higher values are better, False otherwise.
        """
        metric_lower = metric.lower()
        higher_is_better = [
            'accuracy', 'balanced_accuracy',
            'precision', 'balanced_precision', 'precision_micro', 'precision_macro',
            'recall', 'balanced_recall', 'recall_micro', 'recall_macro',
            'f1', 'f1_micro', 'f1_macro',
            'specificity', 'roc_auc', 'auc',
            'matthews_corrcoef', 'cohen_kappa', 'jaccard',
            'r2', 'r2_score'
        ]
        return metric_lower in higher_is_better

    def _save_figure(self, fig: Figure, chart_type: str, dataset_name: str = None):
        """Save figure to disk with versioning.

        Args:
            fig: Matplotlib Figure to save.
            chart_type: Type of chart (e.g., 'top_k', 'heatmap').
            dataset_name: Name of the dataset associated with the chart.
        """
        if not self.output_dir:
            return

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Determine dataset name
        if dataset_name:
            ds_name = dataset_name
        elif self.dataset_name_override:
            ds_name = self.dataset_name_override
        else:
            # Try to infer from predictions if single dataset
            datasets = self.predictions.get_datasets()
            if len(datasets) == 1:
                ds_name = datasets[0]
            else:
                ds_name = "combined"

        # Sanitize names
        ds_name = re.sub(r'[^\w\-]', '_', str(ds_name))
        chart_type = re.sub(r'[^\w\-]', '_', str(chart_type))

        # Base filename pattern
        base_name = f"{ds_name}_{chart_type}"

        # Find next counter
        # Escape glob special characters in base_name just in case, though sanitization handles most
        pattern = os.path.join(self.output_dir, f"{base_name}_*.png")
        existing_files = glob.glob(pattern)

        max_counter = 0
        for f in existing_files:
            # Extract filename from path
            fname = os.path.basename(f)
            # Match pattern to extract number
            match = re.match(rf"{re.escape(base_name)}_(\d+)\.png$", fname)
            if match:
                max_counter = max(max_counter, int(match.group(1)))

        next_counter = max_counter + 1
        filename = f"{base_name}_{next_counter}.png"
        filepath = os.path.join(self.output_dir, filename)

        try:
            fig.savefig(filepath, bbox_inches='tight')
            logger.info(f"Saved chart to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save chart to {filepath}: {e}")

    def plot_top_k(
        self,
        k: int = 5,
        rank_metric: Optional[str] = None,
        rank_partition: str = 'val',
        display_metric: str = '',
        display_partition: str = 'all',
        show_scores: bool = True,
        aggregate: Optional[str] = None,
        config: Optional[ChartConfig] = None,
        **kwargs
    ) -> Union[Figure, List[Figure]]:
        """Plot top K model comparison (scatter + residuals).

        Models are ranked by rank_metric on rank_partition, then predictions
        from display_partition(s) are shown.

        When multiple datasets are present and no dataset_name is specified,
        creates one figure per dataset.

        Args:
            k: Number of top models to show (default: 5).
            rank_metric: Metric for ranking models (default: auto-detect from task type).
            rank_partition: Partition used for ranking (default: 'val').
            display_metric: Metric to display in titles (default: same as rank_metric).
            display_partition: Partition(s) to display ('all' or specific partition).
            show_scores: If True, show scores in chart titles (default: True).
            aggregate: If provided, aggregate predictions by this metadata column or 'y'.
                      When 'y', groups by y_true values.
                      When a column name (e.g., 'ID'), groups by that metadata column.
                      Aggregated predictions have recalculated metrics.
            config: Optional ChartConfig to override analyzer's default config for this chart.
            **kwargs: Additional parameters (dataset_name, figsize, filters).

        Returns:
            matplotlib Figure object or list of Figure objects (one per dataset).

        Example:
            >>> fig = analyzer.plot_top_k(k=3, rank_metric='r2')
            >>> fig = analyzer.plot_top_k(k=3, aggregate='ID')  # Aggregated by ID
        """
        effective_config = config if config is not None else self.config
        effective_aggregate = self._resolve_aggregate(aggregate)
        chart = TopKComparisonChart(
            self.predictions,
            self.dataset_name_override,
            effective_config,
            analyzer=self  # Pass analyzer for cached data access
        )

        # Check if dataset_name is specified in kwargs
        if 'dataset_name' not in kwargs:
            # Get all datasets
            datasets = self.predictions.get_datasets()

            # If multiple datasets, create one figure per dataset
            if len(datasets) > 1:
                figures = []
                for dataset in datasets:
                    fig = chart.render(
                        k=k,
                        rank_metric=rank_metric,
                        rank_partition=rank_partition,
                        display_metric=display_metric,
                        display_partition=display_partition,
                        show_scores=show_scores,
                        aggregate=effective_aggregate,
                        dataset_name=dataset,
                        **kwargs
                    )
                    self._save_figure(fig, "top_k", dataset)
                    figures.append(fig)
                return figures

        # Single dataset or dataset_name specified
        fig = chart.render(
            k=k,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            display_metric=display_metric,
            display_partition=display_partition,
            show_scores=show_scores,
            aggregate=effective_aggregate,
            **kwargs
        )
        self._save_figure(fig, "top_k", kwargs.get('dataset_name'))
        return fig

    def plot_confusion_matrix(
        self,
        k: int = 5,
        rank_metric: Optional[str] = None,
        rank_partition: str = 'val',
        display_metric: Union[str, List[str]] = '',
        display_partition: str = 'test',
        show_scores: bool = True,
        aggregate: Optional[str] = None,
        config: Optional[ChartConfig] = None,
        **kwargs
    ) -> Union[Figure, List[Figure]]:
        """Plot confusion matrices for top K classification models.

        When multiple datasets are present and no dataset_name is specified,
        creates one figure per dataset.

        Args:
            k: Number of top models to show (default: 5).
            rank_metric: Metric for ranking (default: auto-detect from task type).
            rank_partition: Partition used for ranking models (default: 'val').
            display_metric: Metric(s) to display in titles. Can be a single string
                          (e.g., 'accuracy') or a list of strings for multiple metrics
                          (e.g., ['balanced_accuracy', 'accuracy']). Metric names are
                          shown in abbreviated form (default: same as rank_metric).
            display_partition: Partition to display confusion matrix from (default: 'test').
            show_scores: If True, show scores in chart titles (default: True).
            aggregate: If provided, aggregate predictions by this metadata column or 'y'.
            config: Optional ChartConfig to override analyzer's default config for this chart.
            **kwargs: Additional parameters (dataset_name, figsize, filters).

        Returns:
            matplotlib Figure object or list of Figure objects (one per dataset).

        Example:
            >>> fig = analyzer.plot_confusion_matrix(k=3, rank_metric='f1')
            >>> fig = analyzer.plot_confusion_matrix(k=3, aggregate='ID')
            >>> # Multiple metrics displayed with abbreviated names
            >>> fig = analyzer.plot_confusion_matrix(
            ...     k=3,
            ...     display_metric=['balanced_accuracy', 'accuracy']
            ... )
        """
        effective_config = config if config is not None else self.config
        effective_aggregate = self._resolve_aggregate(aggregate)
        chart = ConfusionMatrixChart(
            self.predictions,
            self.dataset_name_override,
            effective_config,
            analyzer=self  # Pass analyzer for cached data access
        )

        # Check if dataset_name is specified in kwargs
        if 'dataset_name' not in kwargs:
            # Get all datasets
            datasets = self.predictions.get_datasets()

            # If multiple datasets, create one figure per dataset
            if len(datasets) > 1:
                figures = []
                for dataset in datasets:
                    fig = chart.render(
                        k=k,
                        rank_metric=rank_metric,
                        rank_partition=rank_partition,
                        display_metric=display_metric,
                        display_partition=display_partition,
                        show_scores=show_scores,
                        aggregate=effective_aggregate,
                        dataset_name=dataset,
                        **kwargs
                    )
                    self._save_figure(fig, "confusion_matrix", dataset)
                    figures.append(fig)
                return figures

        # Single dataset or dataset_name specified
        fig = chart.render(
            k=k,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            display_metric=display_metric,
            display_partition=display_partition,
            show_scores=show_scores,
            aggregate=effective_aggregate,
            **kwargs
        )
        self._save_figure(fig, "confusion_matrix", kwargs.get('dataset_name'))
        return fig

    def plot_histogram(
        self,
        display_metric: Optional[str] = None,
        display_partition: str = 'test',
        aggregate: Optional[str] = None,
        config: Optional[ChartConfig] = None,
        **kwargs
    ) -> Union[Figure, List[Figure]]:
        """Plot score distribution histogram.

        When multiple datasets are present and no dataset_name is specified,
        creates one figure per dataset.

        Args:
            display_metric: Metric to plot (default: auto-detect from task type).
            display_partition: Partition to display scores from (default: 'test').
            aggregate: If provided, aggregate predictions by this metadata column or 'y'.
                      When 'y', groups by y_true values.
                      When a column name (e.g., 'ID'), groups by that metadata column.
                      Aggregated predictions have recalculated metrics.
            config: Optional ChartConfig to override analyzer's default config for this chart.
            **kwargs: Additional parameters (dataset_name, bins, figsize, filters).

        Returns:
            matplotlib Figure object or list of Figure objects (one per dataset).

        Example:
            >>> fig = analyzer.plot_histogram(display_metric='r2', display_partition='val')
            >>> fig = analyzer.plot_histogram(display_metric='rmse', aggregate='ID')
        """
        effective_config = config if config is not None else self.config
        effective_aggregate = self._resolve_aggregate(aggregate)
        chart = ScoreHistogramChart(
            self.predictions,
            self.dataset_name_override,
            effective_config,
            analyzer=self  # Pass analyzer for cached data access
        )

        # Check if dataset_name is specified in kwargs
        if 'dataset_name' not in kwargs:
            # Get all datasets
            datasets = self.predictions.get_datasets()

            # If multiple datasets, create one figure per dataset
            if len(datasets) > 1:
                figures = []
                for dataset in datasets:
                    fig = chart.render(
                        display_metric=display_metric,
                        display_partition=display_partition,
                        aggregate=effective_aggregate,
                        dataset_name=dataset,
                        **kwargs
                    )
                    self._save_figure(fig, "histogram", dataset)
                    figures.append(fig)
                return figures

        # Single dataset or dataset_name specified
        fig = chart.render(
            display_metric=display_metric,
            display_partition=display_partition,
            aggregate=effective_aggregate,
            **kwargs
        )
        self._save_figure(fig, "histogram", kwargs.get('dataset_name'))
        return fig

    def plot_heatmap(
        self,
        x_var: str,
        y_var: str,
        rank_metric: Optional[str] = None,
        rank_partition: str = 'val',
        display_metric: str = '',
        display_partition: str = 'test',
        normalize: bool = False,
        rank_agg: str = 'best',
        display_agg: str = 'best',
        show_counts: bool = True,
        local_scale: bool = False,
        column_scale: bool = False,
        aggregate: Optional[str] = None,
        top_k: Optional[int] = None,
        sort_by_value: bool = False,
        sort_by: Optional[str] = None,
        config: Optional[ChartConfig] = None,
        **kwargs
    ) -> Figure:
        """Plot performance heatmap across two variables.

        For each (x_var, y_var) cell:
        1. Rank predictions by rank_metric on rank_partition using rank_agg
        2. Display display_metric from display_partition using display_agg
        3. Normalize per dataset if requested
        4. Show counts if requested

        Args:
            x_var: Variable for x-axis (e.g., 'model_name', 'preprocessings').
            y_var: Variable for y-axis (e.g., 'dataset_name', 'partition').
            rank_metric: Metric used to rank/select models (default: auto-detect from task type).
            rank_partition: Partition used for ranking models (default: 'val').
            display_metric: Metric to display in heatmap (default: same as rank_metric).
            display_partition: Partition to display scores from (default: 'test').
            normalize: If True, show normalized scores in cells. Colors always use normalized (default: False).
            rank_agg: Aggregation for ranking ('best', 'worst', 'mean', 'median') (default: 'best').
            display_agg: Aggregation for display scores ('best', 'worst', 'mean', 'median') (default: 'mean').
            show_counts: Show prediction counts in cells (default: True).
            local_scale: If True, colorbar shows actual metric values; if False, shows 0-1 normalized (default: False).
            column_scale: If True, normalize colors per column (best in column = 1.0).
                         Automatically sets local_scale=False when enabled (default: False).
            aggregate: If provided, aggregate predictions by this metadata column (e.g., 'ID').
            top_k: If provided, show only top K models. Selection uses Borda count:
                   first keeps top-1 per column, then ranks by Borda count.
            sort_by_value: If True, sort Y-axis by ranking score (best first) instead
                          of alphabetically. Uses rank_metric on rank_partition.
                          Deprecated: use sort_by='value' instead.
            sort_by: Sorting method for Y-axis (rows). Options:
                - None: Alphabetical sorting (default).
                - 'value': Sort by ranking score on rank_partition column.
                - 'mean': Sort by mean score across all columns.
                - 'median': Sort by median score across all columns.
                - 'borda': Sort by Borda count (sum of ranks across columns).
                - 'condorcet': Sort by pairwise wins (Copeland method).
                - 'consensus': Sort by consensus (geometric mean of normalized ranks).
            config: Optional ChartConfig to override analyzer's default config for this chart.
            **kwargs: Additional filters (dataset_name, model_name, etc.).

        Returns:
            matplotlib Figure object.

        Example:
            >>> # Rank on best val RMSE, display mean test RMSE
            >>> fig = analyzer.plot_heatmap('model_name', 'dataset_name')
            >>>
            >>> # Rank on mean val R2, display best test F1
            >>> fig = analyzer.plot_heatmap(
            ...     'model_name', 'dataset_name',
            ...     rank_metric='r2',
            ...     rank_agg='mean',
            ...     display_metric='f1',
            ...     display_agg='best'
            ... )
            >>>
            >>> # Use column normalization for comparing across partitions
            >>> fig = analyzer.plot_heatmap(
            ...     'partition', 'model_name',
            ...     column_scale=True
            ... )
        """
        # Handle backward compatibility with old 'aggregation' parameter
        if 'aggregation' in kwargs:
            aggregation = kwargs.pop('aggregation')
            if rank_agg == 'best':  # Only override if not explicitly set
                rank_agg = aggregation
            if display_agg == 'mean':  # Only override if not explicitly set
                display_agg = aggregation

        effective_config = config if config is not None else self.config
        effective_aggregate = self._resolve_aggregate(aggregate)
        chart = HeatmapChart(
            self.predictions,
            self.dataset_name_override,
            effective_config,
            analyzer=self  # Pass analyzer for cached data access
        )
        fig = chart.render(
            x_var=x_var,
            y_var=y_var,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            display_metric=display_metric,
            display_partition=display_partition,
            normalize=normalize,
            rank_agg=rank_agg,
            display_agg=display_agg,
            show_counts=show_counts,
            local_scale=local_scale,
            column_scale=column_scale,
            aggregate=effective_aggregate,
            top_k=top_k,
            sort_by_value=sort_by_value,
            sort_by=sort_by,
            **kwargs
        )
        self._save_figure(fig, "heatmap", kwargs.get('dataset_name'))
        return fig

    def plot_candlestick(
        self,
        variable: str,
        display_metric: Optional[str] = None,
        display_partition: str = 'test',
        aggregate: Optional[str] = None,
        config: Optional[ChartConfig] = None,
        **kwargs
    ) -> Figure:
        """Plot candlestick chart for score distribution by variable.

        Args:
            variable: Variable to group by (e.g., 'model_name', 'preprocessings').
            display_metric: Metric to analyze (default: auto-detect from task type).
            display_partition: Partition to display scores from (default: 'test').
            aggregate: If provided, aggregate predictions by this metadata column or 'y'.
                      When 'y', groups by y_true values.
                      When a column name (e.g., 'ID'), groups by that metadata column.
                      Aggregated predictions have recalculated metrics.
            config: Optional ChartConfig to override analyzer's default config for this chart.
            **kwargs: Additional parameters (dataset_name, figsize, filters).

        Returns:
            matplotlib Figure object.

        Example:
            >>> fig = analyzer.plot_candlestick('model_name', display_metric='rmse')
            >>> fig = analyzer.plot_candlestick('model_name', display_metric='rmse', aggregate='ID')
        """
        effective_config = config if config is not None else self.config
        effective_aggregate = self._resolve_aggregate(aggregate)
        chart = CandlestickChart(
            self.predictions,
            self.dataset_name_override,
            effective_config,
            analyzer=self  # Pass analyzer for cached data access
        )
        fig = chart.render(
            variable=variable,
            display_metric=display_metric,
            display_partition=display_partition,
            aggregate=effective_aggregate,
            **kwargs
        )
        self._save_figure(fig, "candlestick", kwargs.get('dataset_name'))
        return fig

    # # Backward compatibility aliases
    # def plot_top_k_comparison(self, *args, **kwargs):
    #     """Alias for plot_top_k() (backward compatibility)."""
    #     return self.plot_top_k(*args, **kwargs)

    # def plot_top_k_confusionMatrix(self, *args, **kwargs):
    #     """Alias for plot_confusion_matrix() (backward compatibility).

    #     Note: Old 'partition' kwarg is mapped to both 'rank_partition' and 'display_partition'
    #     for backward compatibility with the old single-partition behavior.
    #     """
    #     # Map old 'partition' param if present and new params not specified
    #     if 'partition' in kwargs:
    #         old_partition = kwargs.pop('partition')
    #         if 'rank_partition' not in kwargs:
    #             kwargs['rank_partition'] = old_partition
    #         if 'display_partition' not in kwargs:
    #             kwargs['display_partition'] = old_partition
    #     return self.plot_confusion_matrix(*args, **kwargs)

    # def plot_score_histogram(self, *args, **kwargs):
    #     """Alias for plot_histogram() (backward compatibility)."""
    #     return self.plot_histogram(*args, **kwargs)

    # def plot_heatmap_v2(self, *args, **kwargs) -> Figure:
    #     """Alias for plot_heatmap() (backward compatibility)."""
    #     return self.plot_heatmap(*args, **kwargs)

    # def plot_variable_heatmap(self, x_var: str, y_var: str, filters: dict = None,
    #                           partition: str = 'val', metric: str = 'rmse',
    #                           score_partition: str = 'test', score_metric: str = '',
    #                           **kwargs) -> Figure:
    #     """Alias for plot_heatmap() (backward compatibility).

    #     Maps old parameters to new API:
    #     - filters['partition'] -> rank_partition
    #     - partition -> rank_partition
    #     - metric -> rank_metric
    #     - score_partition -> display_partition
    #     - score_metric -> display_metric
    #     """
    #     # Extract filters if provided
    #     extra_filters = filters.copy() if filters else {}

    #     # Map old parameters to new ones
    #     rank_partition = extra_filters.pop('partition', partition)
    #     rank_metric = metric
    #     display_partition = score_partition
    #     display_metric = score_metric if score_metric else metric

    #     # Merge remaining filters
    #     kwargs.update(extra_filters)

    #     return self.plot_heatmap(
    #         x_var=x_var,
    #         y_var=y_var,
    #         rank_metric=rank_metric,
    #         rank_partition=rank_partition,
    #         display_metric=display_metric,
    #         display_partition=display_partition,
    #         **kwargs
    #     )

    # def plot_variable_candlestick(self, filters: dict, variable: str,
    #                                metric: str = 'rmse', **kwargs) -> Figure:
    #     """Alias for plot_candlestick() (backward compatibility).

    #     Maps old parameters to new API:
    #     - filters -> extracted and passed as kwargs
    #     """
    #     # Extract filters
    #     extra_filters = filters.copy() if filters else {}
    #     partition = extra_filters.pop('partition', None)

    #     # Merge filters
    #     if partition:
    #         kwargs['partition'] = partition
    #     kwargs.update(extra_filters)

    #     return self.plot_candlestick(variable=variable, metric=metric, **kwargs)

    # =========================================================================
    # BRANCH ANALYSIS METHODS (Phase 4)
    # =========================================================================

    def get_branches(self) -> List[str]:
        """Get list of unique branch names in predictions.

        Returns:
            List of branch names (empty list if no branches)

        Examples:
            >>> branches = analyzer.get_branches()
            >>> print(branches)  # ['snv_pca', 'msc_detrend', 'derivative']
        """
        try:
            return self.predictions.get_unique_values('branch_name')
        except (ValueError, KeyError):
            return []

    def get_branch_ids(self) -> List[int]:
        """Get list of unique branch IDs in predictions.

        Returns:
            List of branch IDs (empty list if no branches)

        Examples:
            >>> branch_ids = analyzer.get_branch_ids()
            >>> print(branch_ids)  # [0, 1, 2]
        """
        try:
            ids = self.predictions.get_unique_values('branch_id')
            # Filter out None values and convert to int
            return sorted([int(x) for x in ids if x is not None])
        except (ValueError, KeyError):
            return []

    def branch_summary(
        self,
        metrics: Optional[List[str]] = None,
        display_partition: str = 'test',
        aggregate: Optional[str] = None,
        as_dataframe: bool = True,
        **filters
    ) -> Union[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Generate summary statistics comparing branch performance.

        Computes mean, std, min, max for each metric across branches.

        Args:
            metrics: List of metrics to compute (default: ['rmse', 'r2'] or
                    ['balanced_accuracy', 'f1'] for classification).
            display_partition: Partition to compute metrics from (default: 'test').
            aggregate: If provided, aggregate predictions by this metadata column
                      (e.g., 'ID') before computing statistics.
            as_dataframe: If True, return pandas DataFrame. If False, return dict.
            **filters: Additional filter criteria.

        Returns:
            DataFrame or dict with branch summary statistics:
                - branch_name: Branch identifier
                - branch_id: Numeric branch ID
                - count: Number of predictions
                - {metric}_mean: Mean value
                - {metric}_std: Standard deviation
                - {metric}_min: Minimum value
                - {metric}_max: Maximum value

        Examples:
            >>> summary = analyzer.branch_summary(metrics=['rmse', 'r2'])
            >>> print(summary.to_markdown())

            >>> summary = analyzer.branch_summary(
            ...     metrics=['balanced_accuracy', 'f1'],
            ...     aggregate='ID'
            ... )
        """
        # Auto-detect metrics if not provided
        if metrics is None:
            metrics = ['rmse', 'r2'] if self._is_higher_better('r2') else ['rmse', 'r2']
            if self._get_default_metric() in ['balanced_accuracy', 'accuracy', 'f1']:
                metrics = ['balanced_accuracy', 'f1']

        effective_aggregate = self._resolve_aggregate(aggregate)

        # Get all predictions
        all_preds = self.get_cached_predictions(
            n=100000,  # Large number to get all
            rank_metric=metrics[0],
            rank_partition='val',
            display_partition=display_partition,
            display_metrics=metrics,
            aggregate=effective_aggregate,
            aggregate_partitions=True,
            **filters
        )

        if not all_preds:
            if as_dataframe:
                return pd.DataFrame()
            return {}

        # Group predictions by branch
        branch_data: Dict[str, List[Dict[str, Any]]] = {}

        for pred in all_preds:
            branch_name = pred.get('branch_name', 'no_branch')
            branch_id = pred.get('branch_id')

            if branch_name not in branch_data:
                branch_data[branch_name] = []
            branch_data[branch_name].append(pred)

        # Compute statistics for each branch
        summary_rows = []

        for branch_name, preds in sorted(branch_data.items()):
            # Get branch_id from first prediction
            branch_id = preds[0].get('branch_id') if preds else None

            row = {
                'branch_name': branch_name,
                'branch_id': branch_id,
                'count': len(preds),
            }

            # Extract scores for each metric
            for metric in metrics:
                scores = []
                for pred in preds:
                    partitions = pred.get('partitions', {})
                    partition_data = partitions.get(display_partition, {})

                    # Try to get score directly or compute it
                    score = partition_data.get(metric)
                    if score is None:
                        y_true = partition_data.get('y_true')
                        y_pred = partition_data.get('y_pred')
                        if y_true is not None and y_pred is not None:
                            try:
                                score = evaluator.eval(y_true, y_pred, metric)
                            except Exception:
                                pass

                    if score is not None:
                        scores.append(float(score))

                # Compute statistics
                if scores:
                    row[f'{metric}_mean'] = np.mean(scores)
                    row[f'{metric}_std'] = np.std(scores)
                    row[f'{metric}_min'] = np.min(scores)
                    row[f'{metric}_max'] = np.max(scores)
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
                    row[f'{metric}_min'] = np.nan
                    row[f'{metric}_max'] = np.nan

            summary_rows.append(row)

        if as_dataframe:
            return pd.DataFrame(summary_rows)

        return {row['branch_name']: row for row in summary_rows}

    def plot_branch_comparison(
        self,
        rank_metric: Optional[str] = None,
        display_metric: Optional[str] = None,
        display_partition: str = 'test',
        aggregate: Optional[str] = None,
        show_ci: bool = True,
        ci_level: float = 0.95,
        figsize: Optional[tuple] = None,
        config: Optional[ChartConfig] = None,
        **filters
    ) -> Figure:
        """Plot bar chart comparing branch performance with confidence intervals.

        Creates a grouped bar chart showing mean metric values for each branch
        with optional confidence intervals.

        Args:
            rank_metric: Metric for ranking models (default: auto-detect).
            display_metric: Metric to display (default: same as rank_metric).
            display_partition: Partition to display results from (default: 'test').
            aggregate: If provided, aggregate predictions by this metadata column.
            show_ci: If True, show confidence intervals (default: True).
            ci_level: Confidence level for intervals (default: 0.95).
            figsize: Figure size tuple (default: auto-computed).
            config: Optional ChartConfig to override defaults.
            **filters: Additional filter criteria.

        Returns:
            matplotlib Figure with branch comparison bar chart.

        Examples:
            >>> fig = analyzer.plot_branch_comparison(display_metric='rmse')
            >>> fig = analyzer.plot_branch_comparison(
            ...     display_metric='r2',
            ...     aggregate='ID',
            ...     show_ci=True
            ... )
        """
        import matplotlib.pyplot as plt
        from scipy import stats

        effective_config = config if config is not None else self.config

        # Auto-detect metric
        if rank_metric is None:
            rank_metric = self._get_default_metric()
        if display_metric is None:
            display_metric = rank_metric

        effective_aggregate = self._resolve_aggregate(aggregate)

        # Get branch summary
        summary_df = self.branch_summary(
            metrics=[display_metric],
            display_partition=display_partition,
            aggregate=effective_aggregate,
            as_dataframe=True,
            **filters
        )

        if summary_df.empty:
            if figsize is None:
                figsize = effective_config.get_figsize('small')
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No branch data available',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig

        # Prepare data
        branch_names = summary_df['branch_name'].tolist()
        means = summary_df[f'{display_metric}_mean'].tolist()
        stds = summary_df[f'{display_metric}_std'].tolist()
        counts = summary_df['count'].tolist()

        # Compute confidence intervals
        if show_ci:
            errors = []
            for std, n in zip(stds, counts):
                if n > 1 and not np.isnan(std):
                    # Use t-distribution for small samples
                    t_crit = stats.t.ppf((1 + ci_level) / 2, n - 1)
                    errors.append(t_crit * std / np.sqrt(n))
                else:
                    errors.append(0)
        else:
            errors = None

        # Create figure
        if figsize is None:
            width = max(6, len(branch_names) * 1.2)
            figsize = (width, 5)

        fig, ax = plt.subplots(figsize=figsize)

        # Create bar chart
        x = np.arange(len(branch_names))
        higher_better = self._is_higher_better(display_metric)

        # Color bars based on ranking
        if higher_better:
            best_idx = np.nanargmax(means)
        else:
            best_idx = np.nanargmin(means)

        colors = ['#2ecc71' if i == best_idx else '#3498db'
                  for i in range(len(branch_names))]

        bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor='black')

        # Add error bars
        if errors:
            ax.errorbar(
                x, means, yerr=errors, fmt='none', color='black',
                capsize=5, capthick=2, linewidth=2
            )

        # Customize plot
        ax.set_xticks(x)
        ax.set_xticklabels(
            branch_names, rotation=45, ha='right',
            fontsize=effective_config.tick_fontsize
        )
        ax.set_ylabel(
            f'{abbreviate_metric(display_metric)} [{display_partition}]',
            fontsize=effective_config.label_fontsize
        )
        ax.set_xlabel('Branch', fontsize=effective_config.label_fontsize)

        # Build title
        title = f'Branch Comparison: {abbreviate_metric(display_metric)} [{display_partition}]'
        if aggregate:
            title += f' (aggregated by {aggregate})'
        ax.set_title(title, fontsize=effective_config.title_fontsize)

        # Add value labels on bars
        for bar, mean, count in zip(bars, means, counts):
            if not np.isnan(mean):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{mean:.3f}\n(n={count})',
                    ha='center', va='bottom', fontsize=8
                )

        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        self._save_figure(fig, 'branch_comparison')
        return fig

    def plot_branch_boxplot(
        self,
        rank_metric: Optional[str] = None,
        display_metric: Optional[str] = None,
        display_partition: str = 'test',
        aggregate: Optional[str] = None,
        figsize: Optional[tuple] = None,
        config: Optional[ChartConfig] = None,
        **filters
    ) -> Figure:
        """Plot boxplot comparing score distributions across branches.

        Creates a boxplot showing the distribution of metric values for each branch.

        Args:
            rank_metric: Metric for ranking models (default: auto-detect).
            display_metric: Metric to display (default: same as rank_metric).
            display_partition: Partition to display results from (default: 'test').
            aggregate: If provided, aggregate predictions by this metadata column.
            figsize: Figure size tuple (default: auto-computed).
            config: Optional ChartConfig to override defaults.
            **filters: Additional filter criteria.

        Returns:
            matplotlib Figure with branch comparison boxplot.

        Examples:
            >>> fig = analyzer.plot_branch_boxplot(display_metric='rmse')
            >>> fig = analyzer.plot_branch_boxplot(
            ...     display_metric='r2',
            ...     aggregate='ID'
            ... )
        """
        import matplotlib.pyplot as plt

        effective_config = config if config is not None else self.config

        # Auto-detect metric
        if rank_metric is None:
            rank_metric = self._get_default_metric()
        if display_metric is None:
            display_metric = rank_metric

        effective_aggregate = self._resolve_aggregate(aggregate)

        # Get all predictions
        all_preds = self.get_cached_predictions(
            n=100000,
            rank_metric=rank_metric,
            rank_partition='val',
            display_partition=display_partition,
            display_metrics=[display_metric],
            aggregate=effective_aggregate,
            aggregate_partitions=True,
            **filters
        )

        if not all_preds:
            if figsize is None:
                figsize = effective_config.get_figsize('small')
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No branch data available',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig

        # Group scores by branch
        branch_scores: Dict[str, List[float]] = {}

        for pred in all_preds:
            branch_name = pred.get('branch_name', 'no_branch')
            partitions = pred.get('partitions', {})
            partition_data = partitions.get(display_partition, {})

            # Get score
            score = partition_data.get(display_metric)
            if score is None:
                y_true = partition_data.get('y_true')
                y_pred = partition_data.get('y_pred')
                if y_true is not None and y_pred is not None:
                    try:
                        score = evaluator.eval(y_true, y_pred, display_metric)
                    except Exception:
                        pass

            if score is not None:
                if branch_name not in branch_scores:
                    branch_scores[branch_name] = []
                branch_scores[branch_name].append(float(score))

        if not branch_scores:
            if figsize is None:
                figsize = effective_config.get_figsize('small')
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No score data available',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig

        # Sort branches by median score
        higher_better = self._is_higher_better(display_metric)
        sorted_branches = sorted(
            branch_scores.keys(),
            key=lambda b: np.median(branch_scores[b]),
            reverse=higher_better
        )

        # Create figure
        if figsize is None:
            width = max(6, len(sorted_branches) * 1.0)
            figsize = (width, 5)

        fig, ax = plt.subplots(figsize=figsize)

        # Create boxplot
        data = [branch_scores[b] for b in sorted_branches]
        bp = ax.boxplot(data, labels=sorted_branches, patch_artist=True)

        # Color boxes
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_branches)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Highlight best branch
        best_idx = 0  # First is best due to sorting
        bp['boxes'][best_idx].set_facecolor('#2ecc71')

        # Customize plot
        ax.set_xticklabels(
            sorted_branches, rotation=45, ha='right',
            fontsize=effective_config.tick_fontsize
        )
        ax.set_ylabel(
            f'{abbreviate_metric(display_metric)} [{display_partition}]',
            fontsize=effective_config.label_fontsize
        )
        ax.set_xlabel('Branch', fontsize=effective_config.label_fontsize)

        # Build title
        title = f'Branch Distribution: {abbreviate_metric(display_metric)} [{display_partition}]'
        if aggregate:
            title += f' (aggregated by {aggregate})'
        ax.set_title(title, fontsize=effective_config.title_fontsize)

        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        self._save_figure(fig, 'branch_boxplot')
        return fig

    def plot_branch_heatmap(
        self,
        y_var: str = 'fold_id',
        rank_metric: Optional[str] = None,
        display_metric: Optional[str] = None,
        display_partition: str = 'test',
        aggregate: Optional[str] = None,
        config: Optional[ChartConfig] = None,
        **kwargs
    ) -> Figure:
        """Plot heatmap of branch performance across folds or other variable.

        Creates a heatmap with branches on x-axis and another variable (e.g., fold_id)
        on y-axis.

        Args:
            y_var: Variable for y-axis (default: 'fold_id').
            rank_metric: Metric for ranking (default: auto-detect).
            display_metric: Metric to display (default: same as rank_metric).
            display_partition: Partition to display (default: 'test').
            aggregate: If provided, aggregate predictions by this metadata column.
            config: Optional ChartConfig to override defaults.
            **kwargs: Additional parameters passed to plot_heatmap.

        Returns:
            matplotlib Figure with branch heatmap.

        Examples:
            >>> fig = analyzer.plot_branch_heatmap(display_metric='rmse')
            >>> fig = analyzer.plot_branch_heatmap(
            ...     y_var='model_name',
            ...     display_metric='r2'
            ... )
        """
        if rank_metric is None:
            rank_metric = self._get_default_metric()
        if display_metric is None:
            display_metric = rank_metric

        return self.plot_heatmap(
            x_var='branch_name',
            y_var=y_var,
            rank_metric=rank_metric,
            display_metric=display_metric,
            display_partition=display_partition,
            aggregate=aggregate,
            config=config,
            **kwargs
        )

    def plot_branch_diagram(
        self,
        show_metrics: bool = True,
        metric: Optional[str] = None,
        partition: str = 'test',
        figsize: Optional[tuple] = None,
        title: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Figure:
        """Plot DAG diagram showing the branching structure of the pipeline.

        Creates a visual diagram showing shared steps, branch nodes, and
        post-branch models in a hierarchical layout.

        Args:
            show_metrics: If True, show metrics in branch nodes (default: True).
            metric: Metric to display (default: auto-detect).
            partition: Partition for metrics (default: 'test').
            figsize: Figure size tuple (default: auto-computed).
            title: Optional title for the diagram.
            config: Additional configuration dict for BranchDiagram.

        Returns:
            matplotlib Figure with branch DAG diagram.

        Examples:
            >>> fig = analyzer.plot_branch_diagram(metric='rmse')
            >>> fig = analyzer.plot_branch_diagram(
            ...     show_metrics=True,
            ...     metric='r2',
            ...     partition='val'
            ... )
        """
        from nirs4all.visualization.branch_diagram import BranchDiagram

        if metric is None:
            metric = self._get_default_metric()

        cfg = config or {}
        diagram = BranchDiagram(self.predictions, config=cfg)
        fig = diagram.render(
            show_metrics=show_metrics,
            metric=metric,
            partition=partition,
            figsize=figsize,
            title=title,
        )

        self._save_figure(fig, 'branch_diagram')
        return fig

    def plot_nested_branches(
        self,
        level1_var: str = 'branch_path_level1',
        level2_var: str = 'branch_path_level2',
        metric: Optional[str] = None,
        partition: str = 'test',
        plot_type: str = 'grouped_bar',
        figsize: Optional[tuple] = None,
        config: Optional[ChartConfig] = None,
        **filters
    ) -> Figure:
        """Plot nested branch comparison for hierarchical experiments.

        Creates grouped bar charts or faceted plots for nested branch structures.

        Args:
            level1_var: Variable for first level grouping (outer group).
            level2_var: Variable for second level grouping (inner group/x-axis).
            metric: Metric to display (default: auto-detect).
            partition: Partition for metrics (default: 'test').
            plot_type: Type of plot ('grouped_bar', 'facet').
            figsize: Figure size tuple.
            config: Optional ChartConfig to override defaults.
            **filters: Additional filter criteria.

        Returns:
            matplotlib Figure with nested branch visualization.

        Examples:
            >>> # Compare outlier strategies × preprocessing
            >>> fig = analyzer.plot_nested_branches(
            ...     level1_var='outlier_strategy',
            ...     level2_var='preprocessing',
            ...     metric='rmse'
            ... )
        """
        import matplotlib.pyplot as plt

        effective_config = config if config is not None else self.config

        if metric is None:
            metric = self._get_default_metric()

        # Get all predictions
        all_preds = self.get_cached_predictions(
            n=100000,
            rank_metric=metric,
            rank_partition='val',
            display_partition=partition,
            display_metrics=[metric],
            aggregate_partitions=True,
            **filters
        )

        if not all_preds:
            if figsize is None:
                figsize = effective_config.get_figsize('medium')
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No nested branch data available',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig

        # Group data by level1 and level2
        # For branch_path, we extract levels from the tuple
        nested_data: Dict[str, Dict[str, List[float]]] = {}

        for pred in all_preds:
            # Try to get level variables from prediction
            level1 = None
            level2 = None

            # Support branch_path tuple if available
            branch_path = pred.get('branch_path')
            if branch_path and isinstance(branch_path, (list, tuple)) and len(branch_path) >= 2:
                level1 = str(branch_path[0])
                level2 = str(branch_path[1])
            else:
                # Fallback to specified variables
                level1 = pred.get(level1_var)
                level2 = pred.get(level2_var)

            if level1 is None or level2 is None:
                # Try to parse from branch_name if structured
                branch_name = pred.get('branch_name', '')
                if '_' in str(branch_name):
                    parts = str(branch_name).split('_', 1)
                    if len(parts) == 2:
                        level1, level2 = parts

            if level1 is None or level2 is None:
                continue

            level1 = str(level1)
            level2 = str(level2)

            # Get score
            partitions = pred.get('partitions', {})
            partition_data = partitions.get(partition, {})
            score = partition_data.get(metric)

            if score is None:
                y_true = partition_data.get('y_true')
                y_pred = partition_data.get('y_pred')
                if y_true is not None and y_pred is not None:
                    try:
                        score = evaluator.eval(y_true, y_pred, metric)
                    except Exception:
                        continue

            if score is not None:
                if level1 not in nested_data:
                    nested_data[level1] = {}
                if level2 not in nested_data[level1]:
                    nested_data[level1][level2] = []
                nested_data[level1][level2].append(float(score))

        if not nested_data:
            if figsize is None:
                figsize = effective_config.get_figsize('medium')
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No nested structure detected in data',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig

        # Create grouped bar chart
        level1_labels = sorted(nested_data.keys())
        level2_labels = sorted(set(
            l2 for l1_data in nested_data.values() for l2 in l1_data.keys()
        ))

        if figsize is None:
            width = max(8, len(level1_labels) * len(level2_labels) * 0.5)
            figsize = (width, 6)

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(level1_labels))
        width_bar = 0.8 / len(level2_labels)
        colors = plt.cm.Set2(np.linspace(0, 1, len(level2_labels)))

        for i, level2 in enumerate(level2_labels):
            means = []
            stds = []
            for level1 in level1_labels:
                scores = nested_data.get(level1, {}).get(level2, [])
                if scores:
                    means.append(np.mean(scores))
                    stds.append(np.std(scores))
                else:
                    means.append(0)
                    stds.append(0)

            offset = (i - len(level2_labels) / 2 + 0.5) * width_bar
            ax.bar(x + offset, means, width_bar, label=level2,
                   color=colors[i], yerr=stds, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(
            level1_labels, rotation=45, ha='right',
            fontsize=effective_config.tick_fontsize
        )
        ax.set_ylabel(
            f'{abbreviate_metric(metric)} [{partition}]',
            fontsize=effective_config.label_fontsize
        )
        ax.set_xlabel(
            level1_var.replace('_', ' ').title(),
            fontsize=effective_config.label_fontsize
        )
        ax.legend(title=level2_var.replace('_', ' ').title())
        ax.set_title(
            f'Nested Branch Comparison: {abbreviate_metric(metric)}',
            fontsize=effective_config.title_fontsize
        )
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, 'nested_branches')
        return fig

    def generate_report(
        self,
        output_path: str,
        branch_comparison: bool = True,
        include_diagrams: bool = True,
        include_tables: bool = True,
        metrics: Optional[List[str]] = None,
        partition: str = 'test',
        title: Optional[str] = None
    ) -> str:
        """Generate HTML report with branch analysis.

        Creates a comprehensive HTML report with branch comparisons,
        visualizations, and statistical tables.

        Args:
            output_path: Path for the output HTML file.
            branch_comparison: If True, include branch comparison section.
            include_diagrams: If True, include branch diagram visualization.
            include_tables: If True, include summary statistics tables.
            metrics: List of metrics to include (default: ['rmse', 'r2']).
            partition: Partition for metrics (default: 'test').
            title: Report title (default: 'Branch Comparison Report').

        Returns:
            Path to the generated HTML file.

        Examples:
            >>> path = analyzer.generate_report(
            ...     'reports/branch_comparison.html',
            ...     branch_comparison=True,
            ...     metrics=['rmse', 'r2', 'mae']
            ... )
        """
        import os
        import base64
        from io import BytesIO
        import matplotlib.pyplot as plt

        if metrics is None:
            metrics = ['rmse', 'r2'] if self._is_higher_better('r2') else ['rmse', 'r2']
            if self._get_default_metric() in ['balanced_accuracy', 'accuracy', 'f1']:
                metrics = ['balanced_accuracy', 'f1']

        if title is None:
            title = 'Branch Comparison Report'

        # Get branch summary
        summary = self.branch_summary(metrics=metrics, display_partition=partition)

        # Start building HTML
        html_parts = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '<meta charset="utf-8">',
            f'<title>{title}</title>',
            '<style>',
            self._get_report_styles(),
            '</style>',
            '</head>',
            '<body>',
            f'<h1>{title}</h1>',
            f'<p class="timestamp">Generated: {self._get_timestamp()}</p>',
        ]

        # Branch overview
        branches = self.get_branches()
        html_parts.append('<h2>Overview</h2>')
        html_parts.append(f'<p>Total branches: <strong>{len(branches)}</strong></p>')
        html_parts.append(f'<p>Total predictions: <strong>{self.predictions.num_predictions}</strong></p>')
        html_parts.append(f'<p>Metrics analyzed: {", ".join(metrics)}</p>')

        # Summary table
        if include_tables and branch_comparison:
            html_parts.append('<h2>Branch Performance Summary</h2>')
            html_parts.append(self._summary_to_html_table(summary, metrics))

        # Visualizations
        if branch_comparison:
            html_parts.append('<h2>Visualizations</h2>')

            # Branch comparison bar chart
            try:
                fig = self.plot_branch_comparison(
                    display_metric=metrics[0],
                    display_partition=partition
                )
                html_parts.append('<h3>Branch Comparison</h3>')
                html_parts.append(self._figure_to_html(fig))
                plt.close(fig)
            except Exception as e:
                html_parts.append(f'<p class="error">Error creating comparison chart: {e}</p>')

            # Branch boxplot
            try:
                fig = self.plot_branch_boxplot(
                    display_metric=metrics[0],
                    display_partition=partition
                )
                html_parts.append('<h3>Score Distribution</h3>')
                html_parts.append(self._figure_to_html(fig))
                plt.close(fig)
            except Exception as e:
                html_parts.append(f'<p class="error">Error creating boxplot: {e}</p>')

            # Branch heatmap
            try:
                fig = self.plot_branch_heatmap(
                    y_var='fold_id',
                    display_metric=metrics[0],
                    display_partition=partition
                )
                html_parts.append('<h3>Branch × Fold Heatmap</h3>')
                html_parts.append(self._figure_to_html(fig))
                plt.close(fig)
            except Exception as e:
                html_parts.append(f'<p class="error">Error creating heatmap: {e}</p>')

        # Branch diagram
        if include_diagrams:
            try:
                fig = self.plot_branch_diagram(
                    show_metrics=True,
                    metric=metrics[0],
                    partition=partition
                )
                html_parts.append('<h3>Pipeline Branching Structure</h3>')
                html_parts.append(self._figure_to_html(fig))
                plt.close(fig)
            except Exception as e:
                html_parts.append(f'<p class="error">Error creating diagram: {e}</p>')

        # Close HTML
        html_parts.extend([
            '</body>',
            '</html>',
        ])

        # Write file
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))

        logger.info(f"Report generated: {output_path}")
        return output_path

    def _get_report_styles(self) -> str:
        """Get CSS styles for HTML report."""
        return """
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            h3 { color: #7f8c8d; }
            .timestamp { color: #95a5a6; font-size: 0.9em; }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            th, td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }
            th { background: #3498db; color: white; }
            tr:nth-child(even) { background: #f9f9f9; }
            tr:hover { background: #f1f1f1; }
            .best { background: #d5f5e3 !important; font-weight: bold; }
            img { max-width: 100%; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .error { color: #e74c3c; font-style: italic; }
        """

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _summary_to_html_table(self, summary, metrics: List[str]) -> str:
        """Convert branch summary to HTML table.

        Args:
            summary: Either a DataFrame or BranchSummary object.
            metrics: List of metrics to display.

        Returns:
            HTML table string.
        """
        import pandas as pd

        # Handle DataFrame
        if isinstance(summary, pd.DataFrame):
            if summary.empty:
                return '<p>No branch data available</p>'
            # Convert DataFrame rows to dicts
            data = summary.to_dict('records')
        else:
            # BranchSummary or similar object with .data attribute
            if not hasattr(summary, 'data') or not summary.data:
                return '<p>No branch data available</p>'
            data = summary.data

        # Build table
        rows = ['<table>', '<tr>']

        # Header
        headers = ['Branch', 'ID', 'Count']
        for metric in metrics:
            headers.append(f'{metric.upper()} (mean ± std)')
        rows.append(''.join(f'<th>{h}</th>' for h in headers))
        rows.append('</tr>')

        # Find best values for highlighting
        best_values = {}
        for metric in metrics:
            values = [row.get(f'{metric}_mean', float('nan')) for row in data]
            valid_values = [v for v in values if v is not None and not np.isnan(v)]
            if valid_values:
                higher_better = self._is_higher_better(metric)
                if higher_better:
                    best_values[metric] = max(valid_values)
                else:
                    best_values[metric] = min(valid_values)

        # Data rows
        for row in data:
            rows.append('<tr>')
            rows.append(f'<td>{row.get("branch_name", "")}</td>')
            rows.append(f'<td>{row.get("branch_id", "")}</td>')
            rows.append(f'<td>{row.get("count", "")}</td>')

            for metric in metrics:
                mean = row.get(f'{metric}_mean')
                std = row.get(f'{metric}_std')
                in_best = metric in best_values
                is_best = mean is not None and in_best and abs(mean - best_values[metric]) < 1e-9
                cell_class = ' class="best"' if is_best else ''

                if mean is not None and not np.isnan(mean):
                    if std is not None and not np.isnan(std):
                        rows.append(f'<td{cell_class}>{mean:.4f} ± {std:.4f}</td>')
                    else:
                        rows.append(f'<td{cell_class}>{mean:.4f}</td>')
                else:
                    rows.append('<td>N/A</td>')

            rows.append('</tr>')

        rows.append('</table>')
        return '\n'.join(rows)

    def _figure_to_html(self, fig: Figure) -> str:
        """Convert matplotlib figure to embedded HTML image."""
        import base64
        from io import BytesIO

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        return f'<img src="data:image/png;base64,{img_base64}" alt="Chart">'
