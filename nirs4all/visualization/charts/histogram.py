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
               figsize: Optional[tuple] = None, **filters) -> Figure:
        """Render score distribution histogram (Optimized with Polars).

        Args:
            display_metric: Metric to plot (default: auto-detect from task type).
            display_partition: Partition to display scores from (default: 'test').
            dataset_name: Optional dataset filter.
            bins: Number of histogram bins (default: 20).
            figsize: Figure size tuple (default: from config).
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

        # --- POLARS OPTIMIZATION START ---
        df = self.predictions.to_dataframe()

        # Build filters
        all_filters = filters.copy()
        if dataset_name:
            all_filters['dataset_name'] = dataset_name
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
            ).alias("score")
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
