"""
PredictionAnalyzer - Orchestrator for prediction analysis and visualization.

This module provides a unified interface for creating various prediction visualizations.
Delegates to specialized chart classes for rendering.

Leverages the refactored Predictions API (predictions.top(), PredictionResult, etc.)
for efficient data access and avoids redundant calculations.
"""
from matplotlib.figure import Figure
from typing import Optional, Union, List
import os
import re
import glob
from pathlib import Path
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
        output_dir: Directory to save generated charts.

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
        config: Optional[ChartConfig] = None,
        output_dir: Optional[str] = "workspace/figures"
    ):
        """Initialize analyzer with predictions object.

        Args:
            predictions_obj: The predictions object containing prediction data.
            dataset_name_override: Optional dataset name override for display.
            config: Optional ChartConfig for customization across all charts.
            output_dir: Directory to save generated charts. Defaults to "workspace/figures".
        """
        self.predictions = predictions_obj
        self.dataset_name_override = dataset_name_override
        self.config = config or ChartConfig()
        self.output_dir = output_dir

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
            print(f"Saved chart to {filepath}")
        except Exception as e:
            print(f"Failed to save chart to {filepath}: {e}")

    def plot_top_k(
        self,
        k: int = 5,
        rank_metric: Optional[str] = None,
        rank_partition: str = 'val',
        display_metric: str = '',
        display_partition: str = 'all',
        show_scores: bool = True,
        aggregate: Optional[str] = None,
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
            **kwargs: Additional parameters (dataset_name, figsize, filters).

        Returns:
            matplotlib Figure object or list of Figure objects (one per dataset).

        Example:
            >>> fig = analyzer.plot_top_k(k=3, rank_metric='r2')
            >>> fig = analyzer.plot_top_k(k=3, aggregate='ID')  # Aggregated by ID
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
                        display_metric=display_metric,
                        display_partition=display_partition,
                        show_scores=show_scores,
                        aggregate=aggregate,
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
            aggregate=aggregate,
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
        chart = ConfusionMatrixChart(
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
                        display_metric=display_metric,
                        display_partition=display_partition,
                        show_scores=show_scores,
                        aggregate=aggregate,
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
            aggregate=aggregate,
            **kwargs
        )
        self._save_figure(fig, "confusion_matrix", kwargs.get('dataset_name'))
        return fig

    def plot_histogram(
        self,
        display_metric: Optional[str] = None,
        display_partition: str = 'test',
        aggregate: Optional[str] = None,
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
            **kwargs: Additional parameters (dataset_name, bins, figsize, filters).

        Returns:
            matplotlib Figure object or list of Figure objects (one per dataset).

        Example:
            >>> fig = analyzer.plot_histogram(display_metric='r2', display_partition='val')
            >>> fig = analyzer.plot_histogram(display_metric='rmse', aggregate='ID')
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
                        display_metric=display_metric,
                        display_partition=display_partition,
                        aggregate=aggregate,
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
            aggregate=aggregate,
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

        chart = HeatmapChart(
            self.predictions,
            self.dataset_name_override,
            self.config
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
            aggregate=aggregate,
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
            **kwargs: Additional parameters (dataset_name, figsize, filters).

        Returns:
            matplotlib Figure object.

        Example:
            >>> fig = analyzer.plot_candlestick('model_name', display_metric='rmse')
            >>> fig = analyzer.plot_candlestick('model_name', display_metric='rmse', aggregate='ID')
        """
        chart = CandlestickChart(
            self.predictions,
            self.dataset_name_override,
            self.config
        )
        fig = chart.render(
            variable=variable,
            display_metric=display_metric,
            display_partition=display_partition,
            aggregate=aggregate,
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

