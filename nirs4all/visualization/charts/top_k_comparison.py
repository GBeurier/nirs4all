"""
TopKComparisonChart - Scatter plots comparing predicted vs observed values for top K models.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, TYPE_CHECKING
from nirs4all.visualization.charts.base import BaseChart

if TYPE_CHECKING:
    from nirs4all.visualization.predictions import PredictionAnalyzer


class TopKComparisonChart(BaseChart):
    """Scatter plots comparing predicted vs observed values for top K models.

    Displays predicted vs true scatter plots alongside residual plots
    for the best performing models according to a ranking metric.
    """

    def __init__(
        self,
        predictions,
        dataset_name_override: Optional[str] = None,
        config=None,
        analyzer: Optional['PredictionAnalyzer'] = None
    ):
        """Initialize top K comparison chart.

        Args:
            predictions: Predictions object instance.
            dataset_name_override: Optional dataset name override.
            config: Optional ChartConfig for customization.
            analyzer: Optional PredictionAnalyzer for cached data access.
        """
        super().__init__(predictions, dataset_name_override, config, analyzer=analyzer)

    def validate_inputs(self, k: int, rank_metric: Optional[str], **kwargs) -> None:
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
        if rank_metric and not isinstance(rank_metric, str):
            raise ValueError("rank_metric must be a string")

    def render(self, k: int = 5, rank_metric: Optional[str] = None,
               rank_partition: str = 'val', display_metric: str = '',
               display_partition: str = 'all', show_scores: bool = True,
               dataset_name: Optional[str] = None,
               figsize: Optional[tuple] = None,
               aggregate: Optional[str] = None,
               **filters) -> Figure:
        """Plot top K models with predicted vs true and residuals.

        Uses the top() method to rank models by a metric on rank_partition,
        then displays predictions from display_partition(s).

        Args:
            k: Number of top models to show (default: 5).
            rank_metric: Metric for ranking models (default: auto-detect from task type).
            rank_partition: Partition used for ranking (default: 'val').
            display_metric: Metric to display in titles (default: same as rank_metric).
            display_partition: Partition(s) to display ('all' for train/val/test, or 'test', 'val', 'train').
            show_scores: If True, show scores in chart titles (default: True).
            dataset_name: Optional dataset filter.
            figsize: Figure size tuple (default: from config).
            aggregate: If provided, aggregate predictions by this metadata column or 'y'.
            **filters: Additional filters.

        Returns:
            matplotlib Figure object.
        """
        # Auto-detect metric if not provided
        if rank_metric is None:
            rank_metric = self._get_default_metric()
        if not display_metric:
            display_metric = rank_metric

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

        # Get top models using common helper with group_by for deduplication
        top_predictions = self._get_ranked_predictions(
            n=k,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            display_partition='test',  # Ignored when aggregate_partitions=True
            display_metrics=[display_metric] if display_metric else None,
            aggregate_partitions=True,
            aggregate=aggregate,
            group_by=['model_name'],  # Keep only best per model_name
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

        # Check if aggregation was actually applied (check first model's partitions)
        aggregation_applied = False
        if aggregate and top_predictions:
            first_pred = top_predictions[0]
            partitions_data = first_pred.get('partitions', {})
            for partition in partitions_to_display:
                partition_data = partitions_data.get(partition, {})
                if partition_data.get('aggregated', False):
                    aggregation_applied = True
                    break

        # Create figure title
        fig_title = f'Top {k} Models Comparison - Ranked by best {rank_metric} [{rank_partition}]'
        if aggregate:
            if aggregation_applied:
                fig_title += f' [aggregated by {aggregate}]'
            else:
                fig_title += f' [aggregation by {aggregate} not applied ⚠️]'
        if dataset_name:
            fig_title = f'{fig_title}\nDataset: {dataset_name}'
        fig.suptitle(fig_title, fontsize=self.config.title_fontsize, fontweight='bold')
        fig.subplots_adjust(top=0.95)

        # Plot each model
        for i, pred in enumerate(top_predictions):
            ax_scatter = axes[i, 0]
            ax_residuals = axes[i, 1]

            model_name = pred.get('model_name', 'Unknown')

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
                    # Flatten arrays to ensure homogeneous shape
                    y_true_flat = np.asarray(y_true).flatten()
                    y_pred_flat = np.asarray(y_pred).flatten()
                    all_y_true.extend(y_true_flat)
                    all_y_pred.extend(y_pred_flat)
                    color = self.config.partition_colors.get(partition, '#333333')
                    all_colors.extend([color] * len(y_true_flat))

            if not all_y_true:
                ax_scatter.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax_scatter.set_title(f'Model {i+1}: {model_name}')
                ax_scatter.axis('off')
                ax_residuals.axis('off')
                continue

            all_y_true = np.array(all_y_true, dtype=np.float64)
            all_y_pred = np.array(all_y_pred, dtype=np.float64)

            # Scatter plot: Predicted vs True
            for partition in partitions_to_display:
                partition_data = partitions_data.get(partition, {})
                y_true = partition_data.get('y_true')
                y_pred = partition_data.get('y_pred')

                if y_true is not None and y_pred is not None and len(y_true) > 0:
                    # Flatten arrays to ensure homogeneous shape
                    y_true_flat = np.asarray(y_true, dtype=np.float64).flatten()
                    y_pred_flat = np.asarray(y_pred, dtype=np.float64).flatten()
                    color = self.config.partition_colors.get(partition, '#333333')
                    ax_scatter.scatter(y_true_flat, y_pred_flat, alpha=self.config.alpha * 0.7,
                                     s=15, color=color, label=partition)

            # Add diagonal line
            min_val = min(all_y_true.min(), all_y_pred.min())
            max_val = max(all_y_true.max(), all_y_pred.max())
            ax_scatter.plot([min_val, max_val], [min_val, max_val],
                          'k--', lw=1.5, alpha=0.7, label='Perfect prediction')

            ax_scatter.set_xlabel('True Values', fontsize=self.config.label_fontsize)
            ax_scatter.set_ylabel('Predicted Values', fontsize=self.config.label_fontsize)

            # Title with model info and scores for all partitions
            # By default, show scores for all partitions (train, val, test)
            if show_scores:
                # Use 'all' mode to show scores for all partitions
                # If show_scores is True, we default to showing all partitions for TopK
                score_mode = 'all' if show_scores is True else show_scores

                title_scores = self._format_score_display(
                    pred, score_mode, rank_metric, rank_partition,
                    display_metric, display_partition
                )

                # Add ranking info if different from display
                if rank_metric != display_metric:
                    rank_score = pred.get('rank_score')
                    if rank_score is not None:
                        title_scores += f"\n(Rank: {rank_metric}={rank_score:.4f} [{rank_partition}])"

                title = f'{model_name}\n{title_scores}'
            else:
                title = model_name

            ax_scatter.set_title(title, fontsize=self.config.label_fontsize)
            ax_scatter.legend(fontsize=self.config.legend_fontsize)
            ax_scatter.grid(True, alpha=0.3)

            # Residual plot
            for partition in partitions_to_display:
                partition_data = partitions_data.get(partition, {})
                y_true = partition_data.get('y_true')
                y_pred = partition_data.get('y_pred')

                if y_true is not None and y_pred is not None and len(y_true) > 0:
                    # Flatten arrays to ensure homogeneous shape
                    y_true_flat = np.asarray(y_true, dtype=np.float64).flatten()
                    y_pred_flat = np.asarray(y_pred, dtype=np.float64).flatten()
                    residuals_p = y_pred_flat - y_true_flat
                    color = self.config.partition_colors.get(partition, '#333333')
                    ax_residuals.scatter(y_true_flat, residuals_p, alpha=self.config.alpha * 0.7,
                                       s=15, color=color, label=partition)

            ax_residuals.axhline(y=0, color='k', linestyle='--', lw=1.5, alpha=0.7)
            ax_residuals.set_xlabel('True Values', fontsize=self.config.label_fontsize)
            ax_residuals.set_ylabel('Residuals', fontsize=self.config.label_fontsize)

            # Build residual title with scores for all partitions (like obs chart)
            if show_scores:
                title_lines = []

                # Add partition label
                if show_all_partitions:
                    title_lines.append('Residuals [train/val/test]')
                else:
                    title_lines.append(f'Residuals [{display_partition}]')

                # Use 'all' mode to show scores for all partitions
                score_mode = 'all' if show_scores is True else show_scores

                scores_str = self._format_score_display(
                    pred, score_mode, rank_metric, rank_partition,
                    display_metric, display_partition
                )

                if scores_str:
                    title_lines.append(scores_str)

                residual_title = '\n'.join(title_lines)
            else:
                # Build residual title with partition info
                if show_all_partitions:
                    residual_title = 'Residuals [train/val/test]'
                else:
                    residual_title = f'Residuals [{display_partition}]'

            ax_residuals.set_title(residual_title, fontsize=self.config.label_fontsize)
            ax_residuals.legend(fontsize=self.config.legend_fontsize)
            ax_residuals.grid(True, alpha=0.3)

            # Commented out: Add partition scores as text annotation (small box)
            # scores_text = []
            # for partition in ['train', 'val', 'test']:
            #     score_field = f'{partition}_score'
            #     score = pred.get(score_field)
            #     if score is not None:
            #         scores_text.append(f'{partition}: {score:.4f}')
            #
            # if scores_text:
            #     scores_str = '\\n'.join(scores_text)
            #     ax_residuals.text(
            #         0.98, 0.02, scores_str,
            #         transform=ax_residuals.transAxes,
            #         fontsize=8, verticalalignment='bottom',
            #         horizontalalignment='right',
            #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            #     )

        plt.tight_layout()
        return fig
