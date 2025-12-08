"""
ConfusionMatrixChart - Confusion matrix visualizations for classification models.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Union, List, TYPE_CHECKING
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from nirs4all.visualization.charts.base import BaseChart
from nirs4all.core.metrics import abbreviate_metric

if TYPE_CHECKING:
    from nirs4all.visualization.predictions import PredictionAnalyzer


class ConfusionMatrixChart(BaseChart):
    """Confusion matrix visualizations for classification models.

    Displays confusion matrices for top K classification models,
    with proper handling of multi-class predictions.
    """

    def __init__(
        self,
        predictions,
        dataset_name_override: Optional[str] = None,
        config=None,
        analyzer: Optional['PredictionAnalyzer'] = None
    ):
        """Initialize confusion matrix chart.

        Args:
            predictions: Predictions object instance.
            dataset_name_override: Optional dataset name override.
            config: Optional ChartConfig for customization.
            analyzer: Optional PredictionAnalyzer for cached data access.
        """
        super().__init__(predictions, dataset_name_override, config, analyzer=analyzer)

    def validate_inputs(self, k: int, rank_metric: Optional[str], **kwargs) -> None:
        """Validate confusion matrix inputs.

        Args:
            k: Number of top models.
            rank_metric: Metric name.
            **kwargs: Additional parameters (ignored).

        Raises:
            ValueError: If inputs are invalid.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if rank_metric and not isinstance(rank_metric, str):
            raise ValueError("rank_metric must be a string")

    def render(self, k: int = 5, rank_metric: Optional[str] = None,
               rank_partition: str = 'val', display_metric: Union[str, List[str]] = '',
               display_partition: str = 'test', show_scores: bool = True,
               dataset_name: Optional[str] = None,
               figsize: Optional[tuple] = None,
               aggregate: Optional[str] = None,
               **filters) -> Union[Figure, List[Figure]]:
        """Plot confusion matrices for top K classification models per dataset.

        Models are ranked by the metric on rank_partition, then confusion matrices
        are displayed using predictions from display_partition. Returns one figure
        per dataset to avoid mixing predictions from different datasets.

        Args:
            k: Number of top models to show per dataset (default: 5).
            rank_metric: Metric for ranking (default: auto-detect from task type).
            rank_partition: Partition used for ranking models (default: 'val').
            display_metric: Metric(s) to display in titles. Can be a string for single
                          metric or a list of strings for multiple metrics (e.g.,
                          ['balanced_accuracy', 'accuracy']). Default: same as rank_metric.
            display_partition: Partition to display confusion matrix from (default: 'test').
            show_scores: If True, show scores in chart titles (default: True).
            dataset_name: Optional dataset filter. If provided, only shows that dataset.
            figsize: Figure size tuple (default: from config).
            aggregate: If provided, aggregate predictions by this metadata column or 'y'.
            **filters: Additional filters (e.g., config_name="config1").

        Returns:
            Single Figure if one dataset, List[Figure] if multiple datasets.
        """
        # Auto-detect metric if not provided
        if rank_metric is None:
            rank_metric = self._get_default_metric()

        # Normalize display_metric to list
        # For confusion matrix, always include accuracy and f1 alongside the display_metric
        if not display_metric:
            display_metrics = [rank_metric]
        elif isinstance(display_metric, str):
            display_metrics = [display_metric]
        else:
            display_metrics = list(display_metric)

        # Always ensure accuracy and f1 are included for classification confusion matrix
        classification_essentials = ['accuracy', 'f1']
        for essential_metric in classification_essentials:
            if essential_metric not in display_metrics:
                display_metrics.append(essential_metric)

        self.validate_inputs(k, rank_metric)

        if figsize is None:
            figsize = self.config.get_figsize('large')

        # Build filters
        if dataset_name:
            filters['dataset_name'] = dataset_name

        # Get list of datasets to process
        if dataset_name:
            datasets = [dataset_name]
        else:
            # Get all unique datasets from predictions
            datasets = self.predictions.get_datasets()
            if not datasets:
                return self._create_empty_figure(
                    figsize,
                    'No datasets found in predictions'
                )

        # Create one figure per dataset
        figures = []
        for ds in datasets:
            # Get top models for this specific dataset using common helper
            ds_filters = {**filters, 'dataset_name': ds}
            top_preds = self._get_ranked_predictions(
                n=k,
                rank_metric=rank_metric,
                rank_partition=rank_partition,
                display_metrics=display_metrics,
                display_partition=display_partition,
                aggregate_partitions=True,
                aggregate=aggregate,
                group_by=['model_name'],  # Keep only best per model_name
                **ds_filters
            )

            if not top_preds:
                # Create empty figure for this dataset
                fig = self._create_empty_figure(
                    figsize,
                    f'No predictions found for dataset: {ds}'
                )
                figures.append(fig)
                continue

            # Calculate grid dimensions for this dataset
            n_plots = len(top_preds)
            cols = int(np.ceil(np.sqrt(n_plots)))
            rows = int(np.ceil(n_plots / cols))

            fig, axes = plt.subplots(rows, cols, figsize=figsize)

            # Handle different subplot configurations
            if n_plots == 1:
                axes = np.array([axes])
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            # Plot each model for this dataset
            for i, pred in enumerate(top_preds):
                ax = axes[i]

                # Get predictions for display_partition
                partition_data = pred.get('partitions', {}).get(display_partition, {})
                y_true = partition_data.get('y_true')
                y_pred = partition_data.get('y_pred')

                # Check for missing or empty data
                y_true_empty = y_true is None or (hasattr(y_true, '__len__') and len(y_true) == 0)
                y_pred_empty = y_pred is None or (hasattr(y_pred, '__len__') and len(y_pred) == 0)
                if y_true_empty or y_pred_empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    model_name = pred.get('model_name', 'Unknown')
                    ax.set_title(f'{model_name}: No data')
                    ax.axis('off')
                    continue

                # Auto-convert probabilities to labels for binary classification
                if len(np.unique(y_true)) == 2 and np.issubdtype(y_pred.dtype, np.floating):
                     unique_vals = np.unique(y_pred)
                     if np.min(y_pred) >= 0 and np.max(y_pred) <= 1 and \
                       not (len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0.0, 1.0]))):
                        y_pred = (y_pred > 0.5).astype(int)

                # Compute confusion matrix
                cm = sk_confusion_matrix(y_true, y_pred)

                # Display confusion matrix
                im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=self.config.tick_fontsize)

                # Set ticks
                n_classes = cm.shape[0]
                tick_marks = np.arange(n_classes)
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(tick_marks, fontsize=self.config.tick_fontsize)
                ax.set_yticklabels(tick_marks, fontsize=self.config.tick_fontsize)

                # Add text annotations
                thresh = cm.max() / 2.
                for row in range(cm.shape[0]):
                    for col in range(cm.shape[1]):
                        ax.text(col, row, format(cm[row, col], 'd'),
                               ha="center", va="center",
                               color="white" if cm[row, col] > thresh else "black",
                               fontsize=self.config.annotation_fontsize)

                # Labels
                ax.set_ylabel('True label', fontsize=self.config.label_fontsize)
                ax.set_xlabel('Predicted label', fontsize=self.config.label_fontsize)

                # Title with model info and scores
                model_name = pred.get('model_name', 'Unknown')

                # Check if aggregation was requested but not applied
                was_aggregated = partition_data.get('aggregated', False)
                agg_indicator = ""
                if aggregate and not was_aggregated:
                    agg_indicator = " ⚠️"  # Warning indicator for non-aggregated

                if show_scores:
                    # Pass display_metrics list to show multiple metrics
                    title_scores = self._format_score_display(
                        pred, display_metrics, rank_metric, rank_partition,
                        display_metrics[0], display_partition,
                        show_rank_score=(rank_partition != display_partition or rank_metric not in display_metrics)
                    )
                    # Split scores into 2 lines if too long
                    score_parts = title_scores.split(' | ')
                    if len(score_parts) > 2:
                        mid = (len(score_parts) + 1) // 2
                        line1 = ' | '.join(score_parts[:mid])
                        line2 = ' | '.join(score_parts[mid:])
                        title_scores = f'{line1}\n{line2}'
                    title = f'{model_name}{agg_indicator}\n{title_scores}'
                else:
                    title = f'{model_name}{agg_indicator}'
                ax.set_title(title, fontsize=self.config.label_fontsize, pad=4)

            # Hide empty subplots
            for i in range(n_plots, len(axes)):
                axes[i].axis('off')

            # Create overall title for this dataset
            rank_metric_abbrev = abbreviate_metric(rank_metric)
            overall_title = f'Top {k} Models - {rank_metric_abbrev} [{rank_partition}→{display_partition}]'
            if aggregate:
                overall_title += f' (agg: {aggregate})'
            fig.suptitle(overall_title, fontsize=self.config.title_fontsize, fontweight='bold')
            plt.tight_layout(rect=(0, 0, 1, 0.95))  # Leave space for suptitle

            figures.append(fig)

        # Return single figure or list of figures
        if len(figures) == 1:
            return figures[0]
        return figures
