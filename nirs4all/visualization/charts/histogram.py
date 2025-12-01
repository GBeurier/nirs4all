"""
ScoreHistogramChart - Histogram of score distributions.
"""
import numpy as np
import polars as pl
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional
from nirs4all.visualization.charts.base import BaseChart
from nirs4all.visualization.chart_utils.annotator import ChartAnnotator


class ScoreHistogramChart(BaseChart):
    """Histogram of score distributions.

    Displays distribution of a metric across predictions with
    statistical annotations.
    """

    def __init__(self, predictions, dataset_name_override: Optional[str] = None,
                 config=None):
        """Initialize histogram chart.

        Args:
            predictions: Predictions object instance.
            dataset_name_override: Optional dataset name override.
            config: Optional ChartConfig for customization.
        """
        super().__init__(predictions, dataset_name_override, config)
        self.annotator = ChartAnnotator(config)

    def validate_inputs(self, display_metric: Optional[str], **kwargs) -> None:
        """Validate histogram inputs.

        Args:
            display_metric: Metric name to plot.
            **kwargs: Additional parameters (ignored).

        Raises:
            ValueError: If display_metric is invalid.
        """
        if display_metric and not isinstance(display_metric, str):
            raise ValueError("display_metric must be a string")

    def render(self, display_metric: Optional[str] = None, display_partition: str = 'test',
               dataset_name: Optional[str] = None, bins: int = 20,
               figsize: Optional[tuple] = None, aggregate: Optional[str] = None,
               **filters) -> Figure:
        """Render score distribution histogram (Optimized with Polars).

        Args:
            display_metric: Metric to plot (default: auto-detect from task type).
            display_partition: Partition to display scores from (default: 'test').
            bins: Number of histogram bins (default: 20).
            figsize: Figure size tuple (default: from config).
            aggregate: If provided, aggregate predictions by this metadata column or 'y'.
                      When 'y', groups by y_true values.
                      When a column name (e.g., 'ID'), groups by that metadata column.
                      Aggregated predictions have recalculated metrics.
            dataset_name: Optional dataset filter.
            **filters: Additional filters (model_name, config_name, etc.).

        Returns:
            matplotlib Figure object.
        """
        t0 = time.time()

        # Auto-detect metric if not provided
        if display_metric is None:
            display_metric = self._get_default_metric()

        self.validate_inputs(display_metric)

        if figsize is None:
            figsize = self.config.get_figsize('small')

        # Build filters
        all_filters = filters.copy()
        if dataset_name:
            all_filters['dataset_name'] = dataset_name

        # If aggregation is requested, use the slower but accurate path
        if aggregate is not None:
            return self._render_with_aggregation(
                display_metric=display_metric,
                display_partition=display_partition,
                bins=bins,
                figsize=figsize,
                aggregate=aggregate,
                **all_filters
            )

        # --- POLARS OPTIMIZATION START ---
        df = self.predictions.to_dataframe()

        # Add partition filter (already have other filters in all_filters)
        all_filters['partition'] = display_partition

        # Apply filters
        for k, v in all_filters.items():
            if k in df.columns:
                df = df.filter(pl.col(k) == v)

        if df.height == 0:
            return self._create_empty_figure(
                figsize,
                f'No predictions found for metric={display_metric}, partition={display_partition}'
            )

        # Extract score (Vectorized)
        col_score = f"{display_partition}_score"
        regex = f'"{display_partition}"\\s*:\\s*\\{{[^}}]*"{display_metric}"\\s*:\\s*([\\d\\.]+)'

        df = df.with_columns(
            pl.when(pl.col("metric") == display_metric)
            .then(pl.col(col_score))
            .otherwise(
                pl.col("scores").str.extract(regex, 1).cast(pl.Float64, strict=False)
            )
            .alias("score")
        )

        # Filter null scores
        df = df.filter(pl.col("score").is_not_null())

        if df.height == 0:
            return self._create_empty_figure(
                figsize,
                f'No valid scores found for metric={display_metric}, partition={display_partition}'
            )

        scores = df["score"].to_list()

        t1 = time.time()
        print(f"Histogram data wrangling time: {t1 - t0:.4f} seconds")

        # --- POLARS OPTIMIZATION END ---

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot histogram
        ax.hist(scores, bins=bins, alpha=self.config.alpha,
                edgecolor='black', color='#35B779')
        ax.set_xlabel(f'{display_metric} score', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Frequency', fontsize=self.config.label_fontsize)

        # Title
        title = f'Score Histogram - {display_metric} [{display_partition}]\n{len(scores)} predictions'
        if dataset_name:
            title = f'{title}\nDataset: {dataset_name}'
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.grid(True, alpha=0.3)

        # Add mean and median lines
        mean_val = float(np.mean(scores))
        median_val = float(np.median(scores))

        ax.axvline(mean_val, color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='g', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.4f}')

        # Add statistics box
        self.annotator.add_statistics_box(ax, scores, position='upper right')

        ax.legend()
        plt.tight_layout()

        t2 = time.time()
        print(f"Matplotlib render time: {t2 - t1:.4f} seconds")

        return fig

    def _render_with_aggregation(
        self,
        display_metric: str,
        display_partition: str,
        bins: int,
        figsize: tuple,
        aggregate: str,
        **filters
    ) -> Figure:
        """Render histogram with aggregation support.

        This is slower than the default render because it needs to load arrays
        and recalculate metrics after aggregation.
        """
        from nirs4all.core import metrics as evaluator
        t0 = time.time()

        # Get all predictions with aggregation applied
        # Use a large n to get all predictions
        try:
            all_preds = self.predictions.top(
                n=10000,  # Large number to get all
                rank_metric=display_metric,
                rank_partition=display_partition,
                display_metrics=[display_metric],
                display_partition=display_partition,
                aggregate_partitions=True,
                aggregate=aggregate,
                **filters
            )
        except Exception as e:
            return self._create_empty_figure(
                figsize,
                f'Error getting aggregated predictions: {e}'
            )

        if not all_preds:
            return self._create_empty_figure(
                figsize,
                f'No predictions found for metric={display_metric}, partition={display_partition}'
            )

        # Extract scores from aggregated predictions
        scores = []
        for pred in all_preds:
            partitions = pred.get('partitions', {})
            partition_data = partitions.get(display_partition, {})

            # Try to get pre-calculated score
            score = partition_data.get(display_metric)

            # If not available, calculate from y_true/y_pred
            if score is None:
                y_true = partition_data.get('y_true')
                y_pred = partition_data.get('y_pred')
                if y_true is not None and y_pred is not None:
                    try:
                        score = evaluator.eval(y_true, y_pred, display_metric)
                    except Exception:
                        pass

            if score is not None:
                scores.append(score)

        if not scores:
            return self._create_empty_figure(
                figsize,
                f'No valid scores found for metric={display_metric}, partition={display_partition}'
            )

        t1 = time.time()
        print(f"Histogram data wrangling time (with aggregation): {t1 - t0:.4f} seconds")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot histogram
        ax.hist(scores, bins=bins, alpha=self.config.alpha,
                edgecolor='black', color='#35B779')
        ax.set_xlabel(f'{display_metric} score', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Frequency', fontsize=self.config.label_fontsize)

        # Title with aggregation note
        title = f'Score Histogram - {display_metric} [{display_partition}]\n{len(scores)} predictions [aggregated by {aggregate}]'
        dataset_name = filters.get('dataset_name')
        if dataset_name:
            title = f'{title}\nDataset: {dataset_name}'
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.grid(True, alpha=0.3)

        # Add mean and median lines
        mean_val = float(np.mean(scores))
        median_val = float(np.median(scores))

        ax.axvline(mean_val, color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='g', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.4f}')

        # Add statistics box
        self.annotator.add_statistics_box(ax, scores, position='upper right')

        ax.legend()
        plt.tight_layout()

        t2 = time.time()
        print(f"Matplotlib render time: {t2 - t1:.4f} seconds")

        return fig
