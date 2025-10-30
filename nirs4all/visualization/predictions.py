"""
PredictionAnalyzer - Orchestrator for prediction analysis and visualization.

This module provides a unified interface for creating various prediction visualizations.
Delegates to specialized chart classes for rendering.

Leverages the refactored Predictions API (predictions.top(), PredictionResult, etc.)
for efficient data access and avoids redundant calculations.
"""
from matplotlib.figure import Figure
from typing import Optional, Union, List
from nirs4all.data.predictions import Predictions
from nirs4all.visualization.charts import (
    ChartConfig,
    ScoreHistogramChart,
    CandlestickChart,
    ConfusionMatrixChart,
    TopKComparisonChart,
    HeatmapChart
)


class PredictionAnalyzer:
    """Orchestrator for prediction analysis and visualization.

    Provides a unified interface for creating various prediction visualizations.
    Delegates to specialized chart classes for rendering.

    Leverages the refactored Predictions API (predictions.top(), PredictionResult, etc.)
    for efficient data access and avoids redundant calculations.

    Attributes:
        predictions: Predictions object containing prediction data.
        dataset_name_override: Optional dataset name override for display.
        config: ChartConfig for customization across all charts.

    Example:
        >>> from nirs4all.data.predictions import Predictions
        >>> predictions = Predictions.load('predictions.json')
        >>> analyzer = PredictionAnalyzer(predictions)
        >>>
        >>> # Plot top 5 models
        >>> fig = analyzer.plot_top_k(k=5)
        >>>
        >>> # Plot heatmap
        >>> fig = analyzer.plot_heatmap('model_name', 'preprocessings')
    """

    def __init__(
        self,
        predictions_obj: Predictions,
        dataset_name_override: Optional[str] = None,
        config: Optional[ChartConfig] = None
    ):
        """Initialize analyzer with predictions object.

        Args:
            predictions_obj: The predictions object containing prediction data.
            dataset_name_override: Optional dataset name override for display.
            config: Optional ChartConfig for customization across all charts.
        """
        self.predictions = predictions_obj
        self.dataset_name_override = dataset_name_override
        self.config = config or ChartConfig()

    def plot_top_k(
        self,
        k: int = 5,
        rank_metric: str = 'rmse',
        rank_partition: str = 'val',
        display_partition: str = 'all',
        **kwargs
    ) -> Union[Figure, List[Figure]]:
        """Plot top K model comparison (scatter + residuals).

        Models are ranked by rank_metric on rank_partition, then predictions
        from display_partition(s) are shown.

        When multiple datasets are present and no dataset_name is specified,
        creates one figure per dataset.

        Args:
            k: Number of top models to show (default: 5).
            rank_metric: Metric for ranking models (default: 'rmse').
            rank_partition: Partition used for ranking (default: 'val').
            display_partition: Partition(s) to display ('all' or specific partition).
            **kwargs: Additional parameters (dataset_name, figsize, filters).

        Returns:
            matplotlib Figure object or list of Figure objects (one per dataset).

        Example:
            >>> fig = analyzer.plot_top_k(k=3, rank_metric='r2')
        """
        chart = TopKComparisonChart(
            self.predictions,
            self.dataset_name_override,
            self.config
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
                        display_partition=display_partition,
                        dataset_name=dataset,
                        **kwargs
                    )
                    figures.append(fig)
                return figures

        # Single dataset or dataset_name specified
        return chart.render(
            k=k,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            display_partition=display_partition,
            **kwargs
        )

    def plot_confusion_matrix(
        self,
        k: int = 5,
        metric: str = 'accuracy',
        rank_partition: str = 'val',
        display_partition: str = 'test',
        **kwargs
    ) -> Figure:
        """Plot confusion matrices for top K classification models.

        Args:
            k: Number of top models to show (default: 5).
            metric: Metric for ranking (default: 'accuracy').
            rank_partition: Partition used for ranking models (default: 'val').
            display_partition: Partition to display confusion matrix from (default: 'test').
            **kwargs: Additional parameters (dataset_name, figsize, filters).

        Returns:
            matplotlib Figure object.

        Example:
            >>> fig = analyzer.plot_confusion_matrix(k=3, metric='f1')
        """
        chart = ConfusionMatrixChart(
            self.predictions,
            self.dataset_name_override,
            self.config
        )
        return chart.render(
            k=k,
            metric=metric,
            rank_partition=rank_partition,
            display_partition=display_partition,
            **kwargs
        )

    def plot_histogram(
        self,
        metric: str = 'rmse',
        partition: Optional[str] = None,
        **kwargs
    ) -> Union[Figure, List[Figure]]:
        """Plot score distribution histogram.

        When multiple datasets are present and no dataset_name is specified,
        creates one figure per dataset.

        Args:
            metric: Metric to plot (default: 'rmse').
            partition: Partition to display scores from (default: 'test').
            **kwargs: Additional parameters (dataset_name, bins, figsize, filters).

        Returns:
            matplotlib Figure object or list of Figure objects (one per dataset).

        Example:
            >>> fig = analyzer.plot_histogram(metric='r2', partition='val')
        """
        chart = ScoreHistogramChart(
            self.predictions,
            self.dataset_name_override,
            self.config
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
                        metric=metric,
                        partition=partition,
                        dataset_name=dataset,
                        **kwargs
                    )
                    figures.append(fig)
                return figures

        # Single dataset or dataset_name specified
        return chart.render(metric=metric, partition=partition, **kwargs)

    def plot_heatmap(
        self,
        x_var: str,
        y_var: str,
        rank_metric: str = 'rmse',
        rank_partition: str = 'val',
        display_metric: str = '',
        display_partition: str = 'test',
        normalize: bool = False,
        rank_agg: str = 'best',
        display_agg: str = 'best',
        show_counts: bool = True,
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
            rank_metric: Metric used to rank/select models (default: 'rmse').
            rank_partition: Partition used for ranking models (default: 'val').
            display_metric: Metric to display in heatmap (default: same as rank_metric).
            display_partition: Partition to display scores from (default: 'test').
            normalize: If True, show normalized scores in cells. Colors always use normalized (default: False).
            rank_agg: Aggregation for ranking ('best', 'worst', 'mean', 'median') (default: 'best').
            display_agg: Aggregation for display scores ('best', 'worst', 'mean', 'median') (default: 'mean').
            show_counts: Show prediction counts in cells (default: True).
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
        """
        # Handle backward compatibility with old 'aggregation' parameter
        if 'aggregation' in kwargs:
            aggregation = kwargs.pop('aggregation')
            if rank_agg == 'best':  # Only override if not explicitly set
                rank_agg = aggregation
            if display_agg == 'mean':  # Only override if not explicitly set
                display_agg = aggregation

        chart = HeatmapChart(
            self.predictions,
            self.dataset_name_override,
            self.config
        )
        return chart.render(
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
            **kwargs
        )

    def plot_candlestick(
        self,
        variable: str,
        metric: str = 'rmse',
        **kwargs
    ) -> Figure:
        """Plot candlestick chart for score distribution by variable.

        Args:
            variable: Variable to group by (e.g., 'model_name', 'preprocessings').
            metric: Metric to analyze (default: 'rmse').
            **kwargs: Additional parameters (dataset_name, partition, figsize, filters).

        Returns:
            matplotlib Figure object.

        Example:
            >>> fig = analyzer.plot_candlestick('model_name', metric='rmse')
        """
        chart = CandlestickChart(
            self.predictions,
            self.dataset_name_override,
            self.config
        )
        return chart.render(variable=variable, metric=metric, **kwargs)

    # Backward compatibility aliases
    def plot_top_k_comparison(self, *args, **kwargs):
        """Alias for plot_top_k() (backward compatibility)."""
        return self.plot_top_k(*args, **kwargs)

    def plot_top_k_confusionMatrix(self, *args, **kwargs):
        """Alias for plot_confusion_matrix() (backward compatibility).

        Note: Old 'partition' kwarg is mapped to both 'rank_partition' and 'display_partition'
        for backward compatibility with the old single-partition behavior.
        """
        # Map old 'partition' param if present and new params not specified
        if 'partition' in kwargs:
            old_partition = kwargs.pop('partition')
            if 'rank_partition' not in kwargs:
                kwargs['rank_partition'] = old_partition
            if 'display_partition' not in kwargs:
                kwargs['display_partition'] = old_partition
        return self.plot_confusion_matrix(*args, **kwargs)

    def plot_score_histogram(self, *args, **kwargs):
        """Alias for plot_histogram() (backward compatibility)."""
        return self.plot_histogram(*args, **kwargs)

    def plot_heatmap_v2(self, *args, **kwargs) -> Figure:
        """Alias for plot_heatmap() (backward compatibility)."""
        return self.plot_heatmap(*args, **kwargs)

    def plot_variable_heatmap(self, x_var: str, y_var: str, filters: dict = None,
                              partition: str = 'val', metric: str = 'rmse',
                              score_partition: str = 'test', score_metric: str = '',
                              **kwargs) -> Figure:
        """Alias for plot_heatmap() (backward compatibility).

        Maps old parameters to new API:
        - filters['partition'] -> rank_partition
        - partition -> rank_partition
        - metric -> rank_metric
        - score_partition -> display_partition
        - score_metric -> display_metric
        """
        # Extract filters if provided
        extra_filters = filters.copy() if filters else {}

        # Map old parameters to new ones
        rank_partition = extra_filters.pop('partition', partition)
        rank_metric = metric
        display_partition = score_partition
        display_metric = score_metric if score_metric else metric

        # Merge remaining filters
        kwargs.update(extra_filters)

        return self.plot_heatmap(
            x_var=x_var,
            y_var=y_var,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            display_metric=display_metric,
            display_partition=display_partition,
            **kwargs
        )

    def plot_variable_candlestick(self, filters: dict, variable: str,
                                   metric: str = 'rmse', **kwargs) -> Figure:
        """Alias for plot_candlestick() (backward compatibility).

        Maps old parameters to new API:
        - filters -> extracted and passed as kwargs
        """
        # Extract filters
        extra_filters = filters.copy() if filters else {}
        partition = extra_filters.pop('partition', None)

        # Merge filters
        if partition:
            kwargs['partition'] = partition
        kwargs.update(extra_filters)

        return self.plot_candlestick(variable=variable, metric=metric, **kwargs)

