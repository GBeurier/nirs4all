"""
BaseChart - Abstract base class for all prediction visualization charts.
"""
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Any
import re
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from nirs4all.visualization.charts.config import ChartConfig


class BaseChart(ABC):
    """Abstract base class for all prediction visualization charts.

    Provides common interface and shared functionality for chart implementations.
    Each chart type should inherit from this class and implement required methods.

    Designed to be operator-ready for future integration with the controller/operator
    pattern (see SpectraChartController for reference pattern).

    Attributes:
        predictions: Predictions object containing prediction data.
        dataset_name_override: Optional dataset name override for display.
        config: ChartConfig instance for customization.
    """

    def __init__(self, predictions, dataset_name_override: Optional[str] = None,
                 config: Optional[ChartConfig] = None):
        """Initialize chart with predictions object.

        Args:
            predictions: Predictions object instance.
            dataset_name_override: Optional dataset name override.
            config: Optional ChartConfig for customization.
        """
        self.predictions = predictions
        self.dataset_name_override = dataset_name_override
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
        # Get task_type from first prediction record
        if self.predictions.num_predictions > 0:
            try:
                task_type = self.predictions._storage.df['task_type'].iloc[0]
                if task_type and 'classification' in str(task_type).lower():
                    return 'balanced_accuracy'
            except (AttributeError, KeyError, IndexError):
                pass
        return 'rmse'

    def _format_score_display(
        self,
        pred: Dict[str, Any],
        show_scores: Union[bool, str, List[str], Dict],
        rank_metric: str,
        rank_partition: str,
        display_metric: str = None,
        display_partition: str = None
    ) -> str:
        """Format scores for chart title based on show_scores parameter.

        New format: metric: score1 [partition1]  score2 [partition2]
        Example: accuracy: 0.95 [val]  0.67 [test]

        Args:
            pred: Prediction dictionary with 'partitions' data.
            show_scores: Control parameter for score display.
            rank_metric: Metric used for ranking.
            rank_partition: Partition used for ranking.
            display_metric: Primary display metric.
            display_partition: Primary display partition.

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

        # Group scores by metric
        metric_lines = []
        for metric in metrics:
            partition_scores = []
            for partition in partitions:
                partition_data = partitions_data.get(partition, {})
                if not partition_data:
                    continue

                # Try to get pre-computed metric
                score = partition_data.get(metric)

                # If not found, compute from y_true and y_pred
                if score is None:
                    y_true = partition_data.get('y_true')
                    y_pred = partition_data.get('y_pred')
                    if y_true is not None and y_pred is not None:
                        try:
                            from nirs4all.core import metrics as evaluator
                            score = evaluator.eval(y_true, y_pred, metric)
                        except Exception:
                            continue

                if score is not None:
                    partition_scores.append(f"{score:.4f} [{partition}]")

            if partition_scores:
                # Format: metric: score1 [partition1]  score2 [partition2]
                metric_lines.append(f"{metric}: {' '.join(partition_scores)}")

        return '\n'.join(metric_lines)
