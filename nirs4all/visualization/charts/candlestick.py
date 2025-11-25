"""
CandlestickChart - Candlestick/box plot for score distributions by variable.
"""
import numpy as np
import polars as pl
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Dict, Any
from nirs4all.visualization.charts.base import BaseChart


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
        """Render candlestick chart showing metric distribution by variable (Optimized with Polars).

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
        t0 = time.time()

        # Auto-detect metric if not provided
        if display_metric is None:
            display_metric = self._get_default_metric()

        self.validate_inputs(variable, display_metric)

        if figsize is None:
            figsize = self.config.get_figsize('medium')

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
            ).alias("score")
        )

        # Filter null scores and ensure variable exists
        df = df.filter(
            pl.col("score").is_not_null() &
            pl.col(variable).is_not_null()
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
        print(f"Candlestick data wrangling time: {t1 - t0:.4f} seconds")

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

        t2 = time.time()
        print(f"Matplotlib render time: {t2 - t1:.4f} seconds")

        return fig
