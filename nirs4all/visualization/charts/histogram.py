"""
ScoreHistogramChart - Histogram of score distributions.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional
from nirs4all.visualization.charts.base import BaseChart
from nirs4all.visualization.chart_utils.predictions_adapter import PredictionsAdapter
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
        self.adapter = PredictionsAdapter(predictions)
        self.annotator = ChartAnnotator(config)

    def validate_inputs(self, metric: str, **kwargs) -> None:
        """Validate histogram inputs.

        Args:
            metric: Metric name to plot.
            **kwargs: Additional parameters (ignored).

        Raises:
            ValueError: If metric is invalid.
        """
        if not metric or not isinstance(metric, str):
            raise ValueError("metric must be a non-empty string")

    def render(self, metric: str = 'rmse', dataset_name: Optional[str] = None,
               partition: Optional[str] = None, bins: int = 20,
               figsize: Optional[tuple] = None, **filters) -> Figure:
        """Render score distribution histogram.

        Args:
            metric: Metric to plot (default: 'rmse').
            dataset_name: Optional dataset filter.
            partition: Partition to display scores from (default: 'test').
            bins: Number of histogram bins (default: 20).
            figsize: Figure size tuple (default: from config).
            **filters: Additional filters (model_name, config_name, etc.).

        Returns:
            matplotlib Figure object.
        """
        self.validate_inputs(metric)

        if figsize is None:
            figsize = self.config.get_figsize('small')

        # Build filters
        if dataset_name:
            filters['dataset_name'] = dataset_name
        if partition:
            filters['partition'] = partition
        else:
            partition = 'test'
            filters['partition'] = partition

        # Get all predictions for the specified partition
        predictions_list = self.adapter.get_top_models(
            n=self.predictions.num_predictions,
            rank_metric=metric,
            rank_partition=partition,
            **filters
        )

        if not predictions_list:
            return self._create_empty_figure(
                figsize,
                f'No predictions found for metric={metric}, partition={partition}'
            )

        # Extract scores
        scores = self.adapter.extract_metric_values(predictions_list, metric, partition)

        if not scores:
            return self._create_empty_figure(
                figsize,
                f'No valid scores found for metric={metric}, partition={partition}'
            )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot histogram
        ax.hist(scores, bins=bins, alpha=self.config.alpha,
                edgecolor='black', color='#35B779')
        ax.set_xlabel(f'{metric.upper()} Score', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Frequency', fontsize=self.config.label_fontsize)

        # Title
        partition_label = partition if partition else 'test'
        title = f'Distribution of {metric.upper()} Scores\n({len(scores)} predictions, partition: {partition_label})'
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

        return fig
