"""
ConfusionMatrixChart - Confusion matrix visualizations for classification models.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Union, List
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from nirs4all.visualization.charts.base import BaseChart
from nirs4all.visualization.chart_utils.predictions_adapter import PredictionsAdapter


class ConfusionMatrixChart(BaseChart):
    """Confusion matrix visualizations for classification models.

    Displays confusion matrices for top K classification models,
    with proper handling of multi-class predictions.
    """

    def __init__(self, predictions, dataset_name_override: Optional[str] = None,
                 config=None):
        """Initialize confusion matrix chart.

        Args:
            predictions: Predictions object instance.
            dataset_name_override: Optional dataset name override.
            config: Optional ChartConfig for customization.
        """
        super().__init__(predictions, dataset_name_override, config)
        self.adapter = PredictionsAdapter(predictions)

    def validate_inputs(self, k: int, metric: str, **kwargs) -> None:
        """Validate confusion matrix inputs.

        Args:
            k: Number of top models.
            metric: Metric name.
            **kwargs: Additional parameters (ignored).

        Raises:
            ValueError: If inputs are invalid.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not metric or not isinstance(metric, str):
            raise ValueError("metric must be a non-empty string")

    def render(self, k: int = 5, metric: str = 'accuracy',
               rank_partition: str = 'val', display_partition: str = 'test',
               dataset_name: Optional[str] = None,
               figsize: Optional[tuple] = None, **filters) -> Union[Figure, List[Figure]]:
        """Plot confusion matrices for top K classification models per dataset.

        Models are ranked by the metric on rank_partition, then confusion matrices
        are displayed using predictions from display_partition. Returns one figure
        per dataset to avoid mixing predictions from different datasets.

        Args:
            k: Number of top models to show per dataset (default: 5).
            metric: Metric for ranking (default: 'accuracy').
            rank_partition: Partition used for ranking models (default: 'val').
            display_partition: Partition to display confusion matrix from (default: 'test').
            dataset_name: Optional dataset filter. If provided, only shows that dataset.
            figsize: Figure size tuple (default: from config).
            **filters: Additional filters (e.g., config_name="config1").

        Returns:
            Single Figure if one dataset, List[Figure] if multiple datasets.
        """
        self.validate_inputs(k, metric)

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
            # Get top models for this specific dataset
            ds_filters = {**filters, 'dataset_name': ds}
            top_preds = self.predictions.top(
                n=k,
                rank_metric=metric,
                rank_partition=rank_partition,
                display_metrics=[metric],
                display_partition=display_partition,
                aggregate_partitions=True,
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

                if y_true is None or y_pred is None or len(y_true) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(f'Model {i+1}: No data')
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
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

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
                               fontsize=self.config.tick_fontsize)

                # Labels
                ax.set_ylabel('True label', fontsize=self.config.label_fontsize)
                ax.set_xlabel('Predicted label', fontsize=self.config.label_fontsize)

                # Title with model info and score
                model_name = pred.get('model_name', 'Unknown')
                score_field = f'{display_partition}_score'
                score = pred.get(score_field, 'N/A')
                if isinstance(score, (int, float)):
                    score_str = f'{score:.4f}'
                else:
                    score_str = str(score)

                title = f'{model_name}\n{metric.upper()}={score_str} ({display_partition})'
                ax.set_title(title, fontsize=self.config.label_fontsize)

            # Hide empty subplots
            for i in range(n_plots, len(axes)):
                axes[i].axis('off')

            # Create overall title for this dataset
            overall_title = f'Dataset: {ds} - Top {k} Models\nConfusion Matrices (ranked by {metric.upper()} on {rank_partition})'
            fig.suptitle(overall_title, fontsize=self.config.title_fontsize, fontweight='bold')
            plt.tight_layout()

            figures.append(fig)

        # Return single figure or list of figures
        if len(figures) == 1:
            return figures[0]
        return figures
