"""
TopKComparisonChart - Scatter plots comparing predicted vs observed values for top K models.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional
from nirs4all.visualization.charts.base import BaseChart
from nirs4all.visualization.chart_utils.predictions_adapter import PredictionsAdapter


class TopKComparisonChart(BaseChart):
    """Scatter plots comparing predicted vs observed values for top K models.

    Displays predicted vs true scatter plots alongside residual plots
    for the best performing models according to a ranking metric.
    """

    def __init__(self, predictions, dataset_name_override: Optional[str] = None,
                 config=None):
        """Initialize top K comparison chart.

        Args:
            predictions: Predictions object instance.
            dataset_name_override: Optional dataset name override.
            config: Optional ChartConfig for customization.
        """
        super().__init__(predictions, dataset_name_override, config)
        self.adapter = PredictionsAdapter(predictions)

    def validate_inputs(self, k: int, rank_metric: str, **kwargs) -> None:
        """Validate top K comparison inputs.

        Args:
            k: Number of top models.
            rank_metric: Metric name for ranking.
            **kwargs: Additional parameters (ignored).

        Raises:
            ValueError: If inputs are invalid.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not rank_metric or not isinstance(rank_metric, str):
            raise ValueError("rank_metric must be a non-empty string")

    def render(self, k: int = 5, rank_metric: str = 'rmse',
               rank_partition: str = 'val', display_partition: str = 'all',
               dataset_name: Optional[str] = None,
               figsize: Optional[tuple] = None, **filters) -> Figure:
        """Plot top K models with predicted vs true and residuals.

        Uses the top() method to rank models by a metric on rank_partition,
        then displays predictions from display_partition(s).

        Args:
            k: Number of top models to show (default: 5).
            rank_metric: Metric for ranking models (default: 'rmse').
            rank_partition: Partition used for ranking (default: 'val').
            display_partition: Partition(s) to display ('all' for train/val/test, or 'test', 'val', 'train').
            dataset_name: Optional dataset filter.
            figsize: Figure size tuple (default: from config).
            **filters: Additional filters.

        Returns:
            matplotlib Figure object.
        """
        self.validate_inputs(k, rank_metric)

        if figsize is None:
            figsize = self.config.get_figsize('large')

        # Build filters
        if dataset_name:
            filters['dataset_name'] = dataset_name

        # Determine which partitions to display
        show_all_partitions = display_partition in ['all', 'ALL', 'All', '_all_', '']

        if show_all_partitions:
            partitions_to_display = ['train', 'val', 'test']
        else:
            partitions_to_display = [display_partition]

        # Get top models using predictions.top() with aggregate_partitions
        ascending = not self.adapter.is_higher_better(rank_metric)

        top_predictions = self.predictions.top(
            n=k,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            display_partition='test',  # Ignored when aggregate_partitions=True
            ascending=ascending,
            aggregate_partitions=True,
            **filters
        )

        if not top_predictions:
            return self._create_empty_figure(
                figsize,
                f'No predictions found for top {k} models with metric={rank_metric}'
            )

        # Create figure
        n_plots = len(top_predictions)
        cols = 2
        rows = n_plots

        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # Handle different subplot configurations
        if n_plots == 1:
            axes = axes.reshape(1, -1)
        elif rows == 1:
            axes = axes.reshape(1, -1)

        # Create figure title
        fig_title = f'Top {k} Models - Best {rank_metric.upper()} ({rank_partition})'
        if dataset_name:
            fig_title = f'{fig_title}\nDataset: {dataset_name}'
        fig.suptitle(fig_title, fontsize=self.config.title_fontsize, fontweight='bold')
        fig.subplots_adjust(top=0.95)

        # Plot each model
        for i, pred in enumerate(top_predictions):
            ax_scatter = axes[i, 0]
            ax_residuals = axes[i, 1]

            model_name = pred.get('model_name', 'Unknown')

            # Get rank score
            rank_score_field = f'{rank_partition}_score'
            rank_score = pred.get(rank_score_field)

            # Collect data from partitions
            all_y_true = []
            all_y_pred = []
            all_colors = []

            partitions_data = pred.get('partitions', {})

            for partition in partitions_to_display:
                partition_data = partitions_data.get(partition, {})
                y_true = partition_data.get('y_true')
                y_pred = partition_data.get('y_pred')

                if y_true is not None and y_pred is not None and len(y_true) > 0:
                    all_y_true.extend(y_true)
                    all_y_pred.extend(y_pred)
                    color = self.config.partition_colors.get(partition, '#333333')
                    all_colors.extend([color] * len(y_true))

            if not all_y_true:
                ax_scatter.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax_scatter.set_title(f'Model {i+1}: {model_name}')
                ax_scatter.axis('off')
                ax_residuals.axis('off')
                continue

            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            residuals = all_y_pred - all_y_true

            # Scatter plot: Predicted vs True
            for partition in partitions_to_display:
                partition_data = partitions_data.get(partition, {})
                y_true = partition_data.get('y_true')
                y_pred = partition_data.get('y_pred')

                if y_true is not None and y_pred is not None and len(y_true) > 0:
                    color = self.config.partition_colors.get(partition, '#333333')
                    ax_scatter.scatter(y_true, y_pred, alpha=self.config.alpha,
                                     s=30, color=color, label=partition)

            # Add diagonal line
            min_val = min(all_y_true.min(), all_y_pred.min())
            max_val = max(all_y_true.max(), all_y_pred.max())
            ax_scatter.plot([min_val, max_val], [min_val, max_val],
                          'k--', lw=1.5, alpha=0.7, label='Perfect prediction')

            ax_scatter.set_xlabel('True Values', fontsize=self.config.label_fontsize)
            ax_scatter.set_ylabel('Predicted Values', fontsize=self.config.label_fontsize)

            # Title with model info
            if isinstance(rank_score, (int, float)):
                rank_score_str = f'{rank_score:.4f}'
            else:
                rank_score_str = str(rank_score)
            title = f'{model_name}\n{rank_metric.upper()}={rank_score_str} ({rank_partition})'
            ax_scatter.set_title(title, fontsize=self.config.label_fontsize)
            ax_scatter.legend(fontsize=8)
            ax_scatter.grid(True, alpha=0.3)

            # Residual plot
            for partition in partitions_to_display:
                partition_data = partitions_data.get(partition, {})
                y_true = partition_data.get('y_true')
                y_pred = partition_data.get('y_pred')

                if y_true is not None and y_pred is not None and len(y_true) > 0:
                    y_true = np.array(y_true)
                    y_pred = np.array(y_pred)
                    residuals_p = y_pred - y_true
                    color = self.config.partition_colors.get(partition, '#333333')
                    ax_residuals.scatter(y_true, residuals_p, alpha=self.config.alpha,
                                       s=30, color=color, label=partition)

            ax_residuals.axhline(y=0, color='k', linestyle='--', lw=1.5, alpha=0.7)
            ax_residuals.set_xlabel('True Values', fontsize=self.config.label_fontsize)
            ax_residuals.set_ylabel('Residuals', fontsize=self.config.label_fontsize)

            # Build residual title with partition info
            if show_all_partitions:
                residual_title = 'Residuals (train/val/test)'
            else:
                residual_title = f'Residuals ({display_partition})'
            ax_residuals.set_title(residual_title, fontsize=self.config.label_fontsize)
            ax_residuals.legend(fontsize=8)
            ax_residuals.grid(True, alpha=0.3)

            # Add partition scores as text annotation
            scores_text = []
            for partition in ['train', 'val', 'test']:
                score_field = f'{partition}_score'
                score = pred.get(score_field)
                if score is not None:
                    scores_text.append(f'{partition}: {score:.4f}')

            if scores_text:
                scores_str = '\n'.join(scores_text)
                ax_residuals.text(
                    0.98, 0.02, scores_str,
                    transform=ax_residuals.transAxes,
                    fontsize=8, verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

        plt.tight_layout()
        return fig
