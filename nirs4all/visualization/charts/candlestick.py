"""
CandlestickChart - Candlestick/box plot for score distributions by variable.
"""
import numpy as np
import polars as pl
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Dict, Any, TYPE_CHECKING
from nirs4all.visualization.charts.base import BaseChart
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.visualization.predictions import PredictionAnalyzer


class CandlestickChart(BaseChart):
    """Candlestick/box plot for score distributions by variable.

    Shows score distribution statistics (min, Q25, mean, Q75, max)
    for each value of a grouping variable.
    """

    def __init__(
        self,
        predictions,
        dataset_name_override: Optional[str] = None,
        config=None,
        analyzer: Optional['PredictionAnalyzer'] = None
    ):
        """Initialize candlestick chart.

        Args:
            predictions: Predictions object instance.
            dataset_name_override: Optional dataset name override.
            config: Optional ChartConfig for customization.
            analyzer: Optional PredictionAnalyzer for cached data access.
        """
        super().__init__(predictions, dataset_name_override, config, analyzer=analyzer)

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
               figsize: Optional[tuple] = None, aggregate: Optional[str] = None,
               clip_outliers: bool = True, iqr_factor: float = 1.5,
               **filters) -> Figure:
        """Render candlestick chart showing metric distribution by variable (Optimized with Polars).

        Args:
            variable: Variable to group by (e.g., 'model_name', 'preprocessings').
            display_metric: Metric to analyze (default: auto-detect from task type).
            display_partition: Partition to display scores from (default: 'test').
            dataset_name: Optional dataset filter.
            figsize: Figure size tuple (default: from config).
            aggregate: If provided, aggregate predictions by this metadata column or 'y'.
                      When 'y', groups by y_true values.
                      When a column name (e.g., 'ID'), groups by that metadata column.
                      Aggregated predictions have recalculated metrics.
            clip_outliers: If True, constrain the y-axis to show the main distribution
                          and let extreme outliers go off-frame (default: True).
            iqr_factor: Factor to multiply IQR for determining outlier bounds.
                       Higher values show more of the tails (default: 1.5).
            **filters: Additional filters (config_name, etc.).

        Returns:
            matplotlib Figure object.
        """
        t0 = time.time()

        # Auto-detect metric if not provided
        if display_metric is None:
            display_metric = self._get_default_metric()

        self.validate_inputs(variable, display_metric)

        if figsize is None:
            figsize = self.config.get_figsize('medium')

        # Build filters
        all_filters = filters.copy()
        if dataset_name:
            all_filters['dataset_name'] = dataset_name

        # If aggregation is requested, use the slower but accurate path
        if aggregate is not None:
            return self._render_with_aggregation(
                variable=variable,
                display_metric=display_metric,
                display_partition=display_partition,
                figsize=figsize,
                aggregate=aggregate,
                clip_outliers=clip_outliers,
                iqr_factor=iqr_factor,
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
                f'No predictions found for variable={variable}, metric={display_metric}'
            )

        # Extract score (Vectorized)
        # Priority 1: Direct column if metric matches
        # Priority 2: Regex from scores JSON
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

        # Filter null scores and ensure variable exists
        df = df.filter(
            pl.col("score").is_not_null()
            & pl.col(variable).is_not_null()
        )

        if df.height == 0:
            return self._create_empty_figure(
                figsize,
                f'No valid scores found for variable={variable}'
            )

        # Group and Aggregate
        stats_df = df.group_by(variable).agg([
            pl.col("score").min().alias("min"),
            pl.col("score").quantile(0.25).alias("q25"),
            pl.col("score").mean().alias("mean"),
            pl.col("score").median().alias("median"),
            pl.col("score").quantile(0.75).alias("q75"),
            pl.col("score").max().alias("max"),
            pl.len().alias("n")
        ])

        # Convert to list of dicts for sorting and plotting
        stats_data = stats_df.to_dicts()

        # Sort by variable value naturally
        stats_data.sort(key=lambda x: self._natural_sort_key(x[variable]))

        var_values = [d[variable] for d in stats_data]

        t1 = time.time()
        logger.debug(f"Candlestick data wrangling time: {t1 - t0:.4f} seconds")

        # --- POLARS OPTIMIZATION END ---

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        x_positions = range(len(var_values))

        # Plot candlesticks
        for i, stats in enumerate(stats_data):
            # Vertical line from min to max
            ax.plot([i, i], [stats['min'], stats['max']], 'k-', linewidth=1)

            # Box from Q25 to Q75
            box_height = stats['q75'] - stats['q25']
            # Handle case where q75 == q25 (zero height box)
            if box_height == 0:
                box_height = 0.00001 # Minimal height to be visible

            box = plt.Rectangle((i - 0.2, stats['q25']), 0.4, box_height,
                                facecolor='lightblue', edgecolor='black', linewidth=1.5)
            ax.add_patch(box)

            # Mean line
            ax.plot([i - 0.2, i + 0.2], [stats['mean'], stats['mean']],
                   'r-', linewidth=2, label='Mean' if i == 0 else '')

            # Median line
            ax.plot([i - 0.2, i + 0.2], [stats['median'], stats['median']],
                   'g--', linewidth=2, label='Median' if i == 0 else '')

        # Apply y-axis clipping for outliers
        if clip_outliers and stats_data:
            y_min, y_max = self._compute_clipped_ylim(stats_data, iqr_factor)
            ax.set_ylim(y_min, y_max)

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
        ax.legend(fontsize=self.config.legend_fontsize)

        plt.tight_layout()

        t2 = time.time()
        logger.debug(f"Matplotlib render time: {t2 - t1:.4f} seconds")

        return fig

    def _render_with_aggregation(
        self,
        variable: str,
        display_metric: str,
        display_partition: str,
        figsize: tuple,
        aggregate: str,
        clip_outliers: bool = True,
        iqr_factor: float = 1.5,
        **filters
    ) -> Figure:
        """Render candlestick with aggregation support.

        This is slower than the default render because it needs to load arrays
        and recalculate metrics after aggregation.
        """
        from nirs4all.core import metrics as evaluator
        t0 = time.time()

        # Get all predictions with aggregation applied using common helper
        # NOTE: Do NOT use group_by here - candlestick shows distribution
        # of scores WITHIN each variable group, so we need all predictions
        try:
            all_preds = self._get_ranked_predictions(
                n=10000,  # Large number to get all
                rank_metric=display_metric,
                rank_partition=display_partition,
                display_metrics=[display_metric],
                display_partition=display_partition,
                aggregate_partitions=True,
                aggregate=aggregate,
                group_by=None,  # Keep all predictions for distribution
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
                f'No predictions found for variable={variable}, metric={display_metric}'
            )

        # Extract variable values and scores from aggregated predictions
        data_by_variable = {}
        for pred in all_preds:
            var_value = pred.get(variable)
            if var_value is None:
                continue

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
                if var_value not in data_by_variable:
                    data_by_variable[var_value] = []
                data_by_variable[var_value].append(score)

        if not data_by_variable:
            return self._create_empty_figure(
                figsize,
                f'No valid scores found for variable={variable}'
            )

        # Compute statistics for each variable value
        stats_data = []
        for var_value, scores in data_by_variable.items():
            scores_arr = np.array(scores)
            stats_data.append({
                variable: var_value,
                'min': float(np.min(scores_arr)),
                'q25': float(np.quantile(scores_arr, 0.25)),
                'mean': float(np.mean(scores_arr)),
                'median': float(np.median(scores_arr)),
                'q75': float(np.quantile(scores_arr, 0.75)),
                'max': float(np.max(scores_arr)),
                'n': len(scores_arr)
            })

        # Sort by variable value naturally
        stats_data.sort(key=lambda x: self._natural_sort_key(x[variable]))
        var_values = [d[variable] for d in stats_data]

        t1 = time.time()
        logger.debug(f"Candlestick data wrangling time (with aggregation): {t1 - t0:.4f} seconds")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        x_positions = range(len(var_values))

        # Plot candlesticks
        for i, stats in enumerate(stats_data):
            # Vertical line from min to max
            ax.plot([i, i], [stats['min'], stats['max']], 'k-', linewidth=1)

            # Box from Q25 to Q75
            box_height = stats['q75'] - stats['q25']
            if box_height == 0:
                box_height = 0.00001

            box = plt.Rectangle((i - 0.2, stats['q25']), 0.4, box_height,
                                facecolor='lightblue', edgecolor='black', linewidth=1.5)
            ax.add_patch(box)

            # Mean line
            ax.plot([i - 0.2, i + 0.2], [stats['mean'], stats['mean']],
                   'r-', linewidth=2, label='Mean' if i == 0 else '')

            # Median line
            ax.plot([i - 0.2, i + 0.2], [stats['median'], stats['median']],
                   'g--', linewidth=2, label='Median' if i == 0 else '')

        # Apply y-axis clipping for outliers
        if clip_outliers and stats_data:
            y_min, y_max = self._compute_clipped_ylim(stats_data, iqr_factor)
            ax.set_ylim(y_min, y_max)

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

        title = f'Candlestick - {display_metric} by {variable.replace("_", " ").title()} [{display_partition}] [aggregated by {aggregate}]'
        ax.set_title(title, fontsize=self.config.title_fontsize)

        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=self.config.legend_fontsize)

        plt.tight_layout()

        t2 = time.time()
        logger.debug(f"Matplotlib render time: {t2 - t1:.4f} seconds")

        return fig

    def _compute_clipped_ylim(
        self,
        stats_data: list,
        iqr_factor: float = 1.5
    ) -> tuple:
        """Compute y-axis limits that clip extreme outliers.

        Uses the IQR method across all groups to determine sensible bounds,
        allowing extreme min/max values to go off-frame.

        Args:
            stats_data: List of dicts with 'min', 'q25', 'q75', 'max' keys.
            iqr_factor: Multiplier for IQR to determine outlier bounds.

        Returns:
            Tuple of (y_min, y_max) for axis limits.
        """
        # Collect all Q25 and Q75 values across groups
        all_q25 = [s['q25'] for s in stats_data]
        all_q75 = [s['q75'] for s in stats_data]
        all_min = [s['min'] for s in stats_data]
        all_max = [s['max'] for s in stats_data]

        # Global IQR based on the range of quartiles
        global_q25 = min(all_q25)
        global_q75 = max(all_q75)
        global_iqr = global_q75 - global_q25

        # Handle case where IQR is 0 (all values are the same)
        if global_iqr == 0:
            global_iqr = abs(global_q75) * 0.1 if global_q75 != 0 else 0.1

        # Compute bounds using IQR
        lower_bound = global_q25 - iqr_factor * global_iqr
        upper_bound = global_q75 + iqr_factor * global_iqr

        # Ensure we show at least all the boxes (Q25 to Q75)
        # but clip extreme whiskers
        y_min = max(min(all_min), lower_bound)
        y_max = min(max(all_max), upper_bound)

        # Add small padding (5%) for visual comfort
        padding = (y_max - y_min) * 0.05
        y_min -= padding
        y_max += padding

        return y_min, y_max
