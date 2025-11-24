"""
CandlestickChart - Candlestick/box plot for score distributions by variable.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Dict, Any
from collections import defaultdict
from nirs4all.visualization.charts.base import BaseChart
from nirs4all.visualization.chart_utils.predictions_adapter import PredictionsAdapter


class CandlestickChart(BaseChart):
    """Candlestick/box plot for score distributions by variable.

    Shows score distribution statistics (min, Q25, mean, Q75, max)
    for each value of a grouping variable.
    """

    def __init__(self, predictions, dataset_name_override: Optional[str] = None,
                 config=None):
        """Initialize candlestick chart.

        Args:
            predictions: Predictions object instance.
            dataset_name_override: Optional dataset name override.
            config: Optional ChartConfig for customization.
        """
        super().__init__(predictions, dataset_name_override, config)
        self.adapter = PredictionsAdapter(predictions)

    def validate_inputs(self, variable: str, display_metric: Optional[str], **kwargs) -> None:
        """Validate candlestick inputs.

        Args:
            variable: Variable name to group by.
            display_metric: Metric name to analyze.
            **kwargs: Additional parameters (ignored).

        Raises:
            ValueError: If variable or display_metric is invalid.
        """
        if not variable or not isinstance(variable, str):
            raise ValueError("variable must be a non-empty string")
        if display_metric and not isinstance(display_metric, str):
            raise ValueError("display_metric must be a string")

    def render(self, variable: str, display_metric: Optional[str] = None,
               display_partition: str = 'test', dataset_name: Optional[str] = None,
               figsize: Optional[tuple] = None, **filters) -> Figure:
        """Render candlestick chart showing metric distribution by variable.

        Args:
            variable: Variable to group by (e.g., 'model_name', 'preprocessings').
            display_metric: Metric to analyze (default: auto-detect from task type).
            display_partition: Partition to display scores from (default: 'test').
            dataset_name: Optional dataset filter.
            figsize: Figure size tuple (default: from config).
            **filters: Additional filters (config_name, etc.).

        Returns:
            matplotlib Figure object.
        """
        # Auto-detect metric if not provided
        if display_metric is None:
            display_metric = self._get_default_metric()

        self.validate_inputs(variable, display_metric)

        if figsize is None:
            figsize = self.config.get_figsize('medium')

        # Build filters
        if dataset_name:
            filters['dataset_name'] = dataset_name
        filters['partition'] = display_partition

        # Get all predictions
        predictions_list = self.adapter.get_top_models(
            n=self.predictions.num_predictions,
            rank_metric=display_metric,
            rank_partition=display_partition,
            **filters
        )

        if not predictions_list:
            return self._create_empty_figure(
                figsize,
                f'No predictions found for variable={variable}, metric={display_metric}'
            )

        # Group scores by variable
        variable_scores = defaultdict(list)

        for pred in predictions_list:
            var_value = pred.get(variable)
            if var_value is None:
                continue

            # Extract score
            score_field = f'{display_partition}_score'
            score = pred.get(score_field)
            if score is not None:
                variable_scores[var_value].append(float(score))

        if not variable_scores:
            return self._create_empty_figure(
                figsize,
                f'No valid scores found for variable={variable}'
            )

        # Sort variable values naturally
        var_values = sorted(variable_scores.keys(), key=self._natural_sort_key)

        # Compute statistics for each variable value
        stats_data = []
        for var_val in var_values:
            scores = variable_scores[var_val]
            stats = {
                'min': float(np.min(scores)),
                'q25': float(np.percentile(scores, 25)),
                'mean': float(np.mean(scores)),
                'median': float(np.median(scores)),
                'q75': float(np.percentile(scores, 75)),
                'max': float(np.max(scores)),
                'n': len(scores)
            }
            stats_data.append(stats)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        x_positions = range(len(var_values))

        # Plot candlesticks
        for i, stats in enumerate(stats_data):
            # Vertical line from min to max
            ax.plot([i, i], [stats['min'], stats['max']], 'k-', linewidth=1)

            # Box from Q25 to Q75
            box_height = stats['q75'] - stats['q25']
            box = plt.Rectangle((i - 0.2, stats['q25']), 0.4, box_height,
                                facecolor='lightblue', edgecolor='black', linewidth=1.5)
            ax.add_patch(box)

            # Mean line
            ax.plot([i - 0.2, i + 0.2], [stats['mean'], stats['mean']],
                   'r-', linewidth=2, label='Mean' if i == 0 else '')

            # Median line
            ax.plot([i - 0.2, i + 0.2], [stats['median'], stats['median']],
                   'g--', linewidth=2, label='Median' if i == 0 else '')

        # Set labels and title
        ax.set_xticks(x_positions)
        var_labels = [str(v)[:25] + '...' if len(str(v)) > 25 else str(v)
                     for v in var_values]
        ax.set_xticklabels(var_labels, rotation=45, ha='right',
                          fontsize=self.config.tick_fontsize)
        ax.set_xlabel(variable.replace('_', ' ').title(),
                     fontsize=self.config.label_fontsize)
        ax.set_ylabel(f'{display_metric} score',
                     fontsize=self.config.label_fontsize)

        title = f'Candlestick - {display_metric} by {variable.replace("_", " ").title()} [{display_partition}]'
        ax.set_title(title, fontsize=self.config.title_fontsize)

        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()

        plt.tight_layout()

        return fig
