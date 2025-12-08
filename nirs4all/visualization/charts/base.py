"""
BaseChart - Abstract base class for all prediction visualization charts.
"""
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Any, TYPE_CHECKING
import re
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from nirs4all.visualization.charts.config import ChartConfig
from nirs4all.core.metrics import abbreviate_metric

if TYPE_CHECKING:
    from nirs4all.visualization.predictions import PredictionAnalyzer


class BaseChart(ABC):
    """Abstract base class for all prediction visualization charts.

    Provides common interface and shared functionality for chart implementations.
    Each chart type should inherit from this class and implement required methods.

    Charts can be initialized in two modes:
    1. With predictions only (legacy): Direct access to Predictions object
    2. With analyzer (recommended): Access through PredictionAnalyzer with caching

    When an analyzer is provided, charts use analyzer.get_cached_predictions()
    to benefit from caching of expensive aggregation operations.

    Designed to be operator-ready for future integration with the controller/operator
    pattern (see SpectraChartController for reference pattern).

    Attributes:
        predictions: Predictions object containing prediction data.
        analyzer: Optional PredictionAnalyzer for cached data access.
        dataset_name_override: Optional dataset name override for display.
        config: ChartConfig instance for customization.
    """

    def __init__(
        self,
        predictions,
        dataset_name_override: Optional[str] = None,
        config: Optional[Union[ChartConfig, Dict[str, Any]]] = None,
        analyzer: Optional['PredictionAnalyzer'] = None
    ):
        """Initialize chart with predictions object.

        Args:
            predictions: Predictions object instance.
            dataset_name_override: Optional dataset name override.
            config: Optional ChartConfig or dict for customization.
                   If a dict is provided, it will be used to create a ChartConfig.
            analyzer: Optional PredictionAnalyzer for cached data access.
                     When provided, charts use analyzer.get_cached_predictions()
                     instead of direct predictions.top() calls.
        """
        self.predictions = predictions
        self.analyzer = analyzer
        self.dataset_name_override = dataset_name_override
        # Handle dict config - convert to ChartConfig
        if isinstance(config, dict):
            self.config = ChartConfig(**config)
        else:
            self.config = config or ChartConfig()

    @abstractmethod
    def render(self, **kwargs) -> Figure:
        """Render the chart and return matplotlib Figure.

        This method must be implemented by all chart subclasses.

        Args:
            **kwargs: Chart-specific rendering parameters.

        Returns:
            matplotlib Figure object.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> None:
        """Validate input parameters for the chart.

        This method should be called before rendering to ensure
        all required parameters are present and valid.

        Args:
            **kwargs: Chart-specific parameters to validate.

        Raises:
            ValueError: If validation fails.
        """
        pass

    def _create_empty_figure(self, figsize, message: str) -> Figure:
        """Create empty figure with message.

        Args:
            figsize: Figure size tuple.
            message: Message to display.

        Returns:
            matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14)
        ax.set_title('No Data Available')
        ax.axis('off')
        return fig

    @staticmethod
    def _natural_sort_key(text: str):
        """Generate natural sorting key for strings with numbers.

        E.g., 'PLSRegression_10_cp' will sort after 'PLSRegression_2_cp'.

        Args:
            text: Input string to generate key for.

        Returns:
            List of alternating strings and integers for sorting.
        """
        def convert(part):
            return int(part) if part.isdigit() else part.lower()
        return [convert(c) for c in re.split(r'(\d+)', str(text))]

    def _get_default_metric(self) -> str:
        """Get default metric based on task type from predictions.

        Uses task_type already stored in predictions - does NOT recompute.

        Returns:
            'balanced_accuracy' for classification, 'rmse' for regression.
        """
        if self.predictions.num_predictions > 0:
            try:
                # Try to get unique task types using public API
                task_types = self.predictions.get_unique_values('task_type')
                # Check if ANY task type is classification
                if any(t and 'classification' in str(t).lower() for t in task_types):
                    return 'balanced_accuracy'
            except Exception:
                pass
        return 'rmse'

    def _get_score(self, partition_data: Dict[str, Any], metric: str) -> Optional[float]:
        """Get score from partition data, computing it if necessary.

        Args:
            partition_data: Dictionary containing partition data (metrics, y_true, y_pred).
            metric: Metric name to retrieve.

        Returns:
            Score value or None if not found/computable.
        """
        if not partition_data:
            return None

        # 1. Try direct access
        if metric in partition_data:
            return float(partition_data[metric])

        # 2. Compute from y_true/y_pred
        y_true = partition_data.get('y_true')
        y_pred = partition_data.get('y_pred')

        if y_true is not None and y_pred is not None:
            try:
                from nirs4all.core import metrics as evaluator
                return float(evaluator.eval(y_true, y_pred, metric))
            except Exception:
                pass

        return None

    def _get_ranked_predictions(
        self,
        n: int,
        rank_metric: str,
        rank_partition: str = 'val',
        display_partition: str = 'test',
        display_metrics: Optional[List[str]] = None,
        aggregate: Optional[str] = None,
        group_by: Optional[Union[str, List[str]]] = None,
        aggregate_partitions: bool = True,
        **filters
    ):
        """
        Get ranked predictions using a common interface for all charts.

        This method provides a unified way to retrieve ranked predictions
        with consistent behavior across all chart types. When an analyzer
        is available, it uses the analyzer's caching layer to avoid
        redundant computations.

        The ranking flow follows this order:
        1. FILTER: Apply user-provided filters (dataset_name, model_name, etc.)
        2. SELECT PARTITION: Filter to rank_partition for ranking scores
        3. AGGREGATE (optional): If aggregate is set, aggregate by metadata column
        4. SORT: Sort globally by rank_score
        5. GROUP (optional): If group_by is set, keep first (best) per group
        6. LIMIT: Take top n results
        7. FETCH DISPLAY: Get data from display_partition for results

        Args:
            n: Number of top predictions to return.
            rank_metric: Metric to rank by (e.g., 'rmse', 'balanced_accuracy').
            rank_partition: Partition to rank on (default: 'val').
            display_partition: Partition to display results from (default: 'test').
            display_metrics: List of metrics to compute for display.
            aggregate: If provided, aggregate predictions by this metadata column.
                      Example: 'ID' to average multiple scans per sample.
            group_by: Group predictions and keep only the best per group.
                     Can be a single column name (str) or list of columns.
                     Examples: 'model_name', ['model_name', 'preprocessings']
            aggregate_partitions: If True, include all partition data (default: True).
            **filters: Additional filter criteria (dataset_name, config_name, etc.)

        Returns:
            PredictionResultsList containing ranked predictions.

        Notes:
            - When analyzer is provided, uses cached data access for performance
            - Charts that need one result per model should use group_by=['model_name']
            - Charts that show distributions should NOT use group_by (or set it to None)
            - The global sort order is always preserved - group_by takes first per group

        Examples:
            >>> # TopK/ConfusionMatrix: one best per model
            >>> preds = self._get_ranked_predictions(
            ...     n=5, rank_metric='rmse', group_by=['model_name']
            ... )
            >>>
            >>> # Histogram/Candlestick: all predictions for distribution
            >>> preds = self._get_ranked_predictions(
            ...     n=10000, rank_metric='rmse', group_by=None
            ... )
            >>>
            >>> # Heatmap: one best per (model_name, config_name) cell
            >>> preds = self._get_ranked_predictions(
            ...     n=10000, rank_metric='rmse',
            ...     group_by=['model_name', 'preprocessings']
            ... )
        """
        # Use analyzer's cached method if available
        if self.analyzer is not None:
            return self.analyzer.get_cached_predictions(
                n=n,
                rank_metric=rank_metric,
                rank_partition=rank_partition,
                display_partition=display_partition,
                display_metrics=display_metrics,
                aggregate=aggregate,
                group_by=group_by,
                aggregate_partitions=aggregate_partitions,
                **filters
            )

        # Fallback to direct predictions.top() (legacy path)
        # Determine ascending order based on metric type
        ascending = not self._is_higher_better(rank_metric)

        # Prepare display_metrics if not provided
        if display_metrics is None:
            display_metrics = [rank_metric]
        elif rank_metric not in display_metrics:
            display_metrics = [rank_metric] + list(display_metrics)

        return self.predictions.top(
            n=n,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            display_partition=display_partition,
            display_metrics=display_metrics,
            ascending=ascending,
            aggregate_partitions=aggregate_partitions,
            aggregate=aggregate,
            group_by=group_by,
            **filters
        )

    @staticmethod
    def _is_higher_better(metric: str) -> bool:
        """Check if metric is higher-is-better.

        Args:
            metric: Metric name to check.

        Returns:
            True if higher values are better, False otherwise.
        """
        metric_lower = metric.lower()
        # Classification metrics (higher is better)
        higher_is_better = [
            'accuracy', 'balanced_accuracy',
            'precision', 'balanced_precision', 'precision_micro', 'precision_macro',
            'recall', 'balanced_recall', 'recall_micro', 'recall_macro',
            'f1', 'f1_micro', 'f1_macro',
            'specificity', 'roc_auc', 'auc',
            'matthews_corrcoef', 'cohen_kappa', 'jaccard',
            # Regression metrics (higher is better)
            'r2', 'r2_score'
        ]
        return metric_lower in higher_is_better

    def _format_score_display(
        self,
        pred: Dict[str, Any],
        show_scores: Union[bool, str, List[str], Dict],
        rank_metric: str,
        rank_partition: str,
        display_metric: str = None,
        display_partition: str = None,
        show_rank_score: bool = False
    ) -> str:
        """Format scores for chart title based on show_scores parameter.

        Compact format: metric1: score [part] | metric2: score [part] | rank: score [part]
        Example: acc: 0.95 [test] | f1: 0.87 [test] | rank(bal_acc): 0.92 [val]

        Args:
            pred: Prediction dictionary with 'partitions' data.
            show_scores: Control parameter for score display.
            rank_metric: Metric used for ranking.
            rank_partition: Partition used for ranking.
            display_metric: Primary display metric.
            display_partition: Primary display partition.
            show_rank_score: If True, also display the ranking score when it differs
                            from display metrics/partition.

        Returns:
            Formatted string for chart title.
        """
        # Handle show_scores parameter
        if show_scores is False:
            return ""  # No scores

        # Determine which metrics and partitions to show
        if show_scores is True:
            # Default: display_metric on display_partition
            metrics = [display_metric or rank_metric]
            partitions = [display_partition or 'test']
        elif show_scores == 'rank_only':
            # Only ranking metric and partition
            metrics = [rank_metric]
            partitions = [rank_partition]
        elif show_scores == 'all':
            # Display_metric on all partitions
            metrics = [display_metric or rank_metric]
            partitions = ['train', 'val', 'test']
        elif isinstance(show_scores, list):
            # Multiple metrics on display_partition
            metrics = show_scores
            partitions = [display_partition or 'test']
        elif isinstance(show_scores, dict):
            # Full control
            metrics = show_scores.get('metrics', [display_metric or rank_metric])
            partitions = show_scores.get('partitions', [display_partition or 'test'])
        else:
            metrics = [display_metric or rank_metric]
            partitions = [display_partition or 'test']

        # Extract scores from prediction
        partitions_data = pred.get('partitions', {})

        # Collect all score parts for compact display
        score_parts = []

        # Add display metrics scores
        for metric in metrics:
            for partition in partitions:
                partition_data = partitions_data.get(partition, {})
                score = self._get_score(partition_data, metric)

                if score is not None:
                    abbrev = abbreviate_metric(metric)
                    score_parts.append(f"{abbrev}: {score:.3f} [{partition}]")

        # Add ranking score if requested and different from display
        if show_rank_score:
            rank_already_shown = rank_metric in metrics and rank_partition in partitions
            if not rank_already_shown:
                rank_partition_data = partitions_data.get(rank_partition, {})
                rank_score = self._get_score(rank_partition_data, rank_metric)
                if rank_score is not None:
                    rank_abbrev = abbreviate_metric(rank_metric)
                    score_parts.append(f"rank({rank_abbrev}): {rank_score:.3f} [{rank_partition}]")

        # Join with separator for compact display
        return " | ".join(score_parts)
