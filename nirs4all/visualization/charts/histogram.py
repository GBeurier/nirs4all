"""
ScoreHistogramChart - Histogram of score distributions.
"""
import numpy as np
import polars as pl
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Literal, Optional, TYPE_CHECKING
from nirs4all.visualization.charts.base import BaseChart
from nirs4all.visualization.chart_utils.annotator import ChartAnnotator
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.visualization.predictions import PredictionAnalyzer


class ScoreHistogramChart(BaseChart):
    """Histogram of score distributions.

    Displays distribution of a metric across predictions with
    statistical annotations.
    """

    def __init__(
        self,
        predictions,
        dataset_name_override: Optional[str] = None,
        config=None,
        analyzer: Optional['PredictionAnalyzer'] = None
    ):
        """Initialize histogram chart.

        Args:
            predictions: Predictions object instance.
            dataset_name_override: Optional dataset name override.
            config: Optional ChartConfig for customization.
            analyzer: Optional PredictionAnalyzer for cached data access.
        """
        super().__init__(predictions, dataset_name_override, config, analyzer=analyzer)
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
               figsize: Optional[tuple] = None, aggregate: Optional[str] = None,
               clip_outliers: bool = True, iqr_factor: float = 1.5,
               layout: Literal['standard', 'stacked', 'staggered'] = 'standard',
               **filters) -> Figure:
        """Render score distribution histogram (Optimized with Polars).

        Args:
            display_metric: Metric to plot (default: auto-detect from task type).
            display_partition: Partition to display scores from (default: 'test').
            bins: Number of histogram bins (default: 20).
            figsize: Figure size tuple (default: from config).
            aggregate: If provided, aggregate predictions by this metadata column or 'y'.
                      When 'y', groups by y_true values.
                      When a column name (e.g., 'ID'), groups by that metadata column.
                      Aggregated predictions have recalculated metrics.
            clip_outliers: If True, constrain the x-axis to show the main distribution
                          and let extreme outliers go off-frame (default: True).
            iqr_factor: Factor to multiply IQR for determining outlier bounds.
                       Higher values show more of the tails (default: 1.5).
            layout: Histogram layout style:
                   - 'standard': overlapping histograms (default)
                   - 'stacked': bars stacked on top of each other
                   - 'staggered': bars placed side by side
            dataset_name: Optional dataset filter.
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

        # Build filters
        all_filters = filters.copy()
        if dataset_name:
            all_filters['dataset_name'] = dataset_name

        # If aggregation is requested, use the slower but accurate path
        if aggregate is not None:
            return self._render_with_aggregation(
                display_metric=display_metric,
                display_partition=display_partition,
                bins=bins,
                figsize=figsize,
                aggregate=aggregate,
                clip_outliers=clip_outliers,
                iqr_factor=iqr_factor,
                layout=layout,
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
            )
            .alias("score")
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
        logger.debug(f"Histogram data wrangling time: {t1 - t0:.4f} seconds")

        # --- POLARS OPTIMIZATION END ---

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Compute clipped range if requested
        x_min, x_max = None, None
        n_clipped = 0
        if clip_outliers:
            x_min, x_max = self._compute_clipped_xlim(scores, iqr_factor)
            # Filter scores for histogram binning within clipped range (only if valid limits)
            if x_min is not None and x_max is not None:
                clipped_scores = [s for s in scores if x_min <= s <= x_max]
                n_clipped = len(scores) - len(clipped_scores)
            else:
                clipped_scores = scores
        else:
            clipped_scores = scores

        # Plot histogram based on layout
        self._plot_histogram(ax, clipped_scores, bins, layout)

        # Apply x-axis limits if clipping
        if clip_outliers and x_min is not None and x_max is not None:
            ax.set_xlim(x_min, x_max)

        ax.set_xlabel(f'{display_metric} score', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Frequency', fontsize=self.config.label_fontsize)

        # Title
        layout_note = f' [{layout}]' if layout != 'standard' else ''
        clipped_note = f' ({n_clipped} outliers clipped)' if n_clipped > 0 else ''
        title = f'Score Histogram - {display_metric} [{display_partition}]{layout_note}\n{len(scores)} predictions{clipped_note}'
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

        ax.legend(fontsize=self.config.legend_fontsize)
        plt.tight_layout()

        t2 = time.time()
        logger.debug(f"Matplotlib render time: {t2 - t1:.4f} seconds")

        return fig

    def _render_with_aggregation(
        self,
        display_metric: str,
        display_partition: str,
        bins: int,
        figsize: tuple,
        aggregate: str,
        clip_outliers: bool = True,
        iqr_factor: float = 1.5,
        layout: Literal['standard', 'stacked', 'staggered'] = 'standard',
        **filters
    ) -> Figure:
        """Render histogram with aggregation support.

        This is slower than the default render because it needs to load arrays
        and recalculate metrics after aggregation.
        """
        from nirs4all.core import metrics as evaluator
        t0 = time.time()

        # Get all predictions with aggregation applied using common helper
        # NOTE: Do NOT use group_by here - histogram needs ALL scores
        # to show the full distribution, not just one per model
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
                f'No predictions found for metric={display_metric}, partition={display_partition}'
            )

        # Extract scores from aggregated predictions
        scores = []
        for pred in all_preds:
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
                scores.append(score)

        if not scores:
            return self._create_empty_figure(
                figsize,
                f'No valid scores found for metric={display_metric}, partition={display_partition}'
            )

        t1 = time.time()
        logger.debug(f"Histogram data wrangling time (with aggregation): {t1 - t0:.4f} seconds")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Compute clipped range if requested
        x_min, x_max = None, None
        n_clipped = 0
        if clip_outliers:
            x_min, x_max = self._compute_clipped_xlim(scores, iqr_factor)
            # Filter scores for histogram binning within clipped range (only if valid limits)
            if x_min is not None and x_max is not None:
                clipped_scores = [s for s in scores if x_min <= s <= x_max]
                n_clipped = len(scores) - len(clipped_scores)
            else:
                clipped_scores = scores
        else:
            clipped_scores = scores

        # Plot histogram based on layout
        self._plot_histogram(ax, clipped_scores, bins, layout)

        # Apply x-axis limits if clipping
        if clip_outliers and x_min is not None and x_max is not None:
            ax.set_xlim(x_min, x_max)

        ax.set_xlabel(f'{display_metric} score', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Frequency', fontsize=self.config.label_fontsize)

        # Title with aggregation note
        layout_note = f' [{layout}]' if layout != 'standard' else ''
        clipped_note = f' ({n_clipped} outliers clipped)' if n_clipped > 0 else ''
        title = f'Score Histogram - {display_metric} [{display_partition}]{layout_note}\n{len(scores)} predictions [aggregated by {aggregate}]{clipped_note}'
        dataset_name = filters.get('dataset_name')
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

        ax.legend(fontsize=self.config.legend_fontsize)
        plt.tight_layout()

        t2 = time.time()
        logger.debug(f"Matplotlib render time: {t2 - t1:.4f} seconds")

        return fig

    def _plot_histogram(
        self,
        ax,
        scores: list,
        bins: int,
        layout: Literal['standard', 'stacked', 'staggered'] = 'standard'
    ) -> None:
        """Plot histogram with specified layout style.

        Args:
            ax: Matplotlib axes object.
            scores: List of score values to plot.
            bins: Number of histogram bins.
            layout: Layout style ('standard', 'stacked', 'staggered').
        """
        if layout == 'standard':
            # Standard overlapping histogram
            ax.hist(scores, bins=bins, alpha=self.config.alpha,
                    edgecolor='black', color='#35B779')

        elif layout == 'stacked':
            # Stacked histogram - for single distribution, same as standard
            # but with histtype='barstacked' for consistency with multi-series
            ax.hist(scores, bins=bins, alpha=self.config.alpha,
                    edgecolor='black', color='#35B779', histtype='barstacked')

        elif layout == 'staggered':
            # Staggered - use 'bar' type with rwidth for spacing effect
            ax.hist(scores, bins=bins, alpha=self.config.alpha,
                    edgecolor='black', color='#35B779', histtype='bar', rwidth=0.8)

        else:
            raise ValueError(f"Unknown layout: {layout}. Use 'standard', 'stacked', or 'staggered'.")

    def _compute_clipped_xlim(
        self,
        scores: list,
        iqr_factor: float = 1.5
    ) -> tuple:
        """Compute x-axis limits that clip extreme outliers.

        Uses the IQR method to determine sensible bounds,
        allowing extreme values to go off-frame.

        Args:
            scores: List of score values.
            iqr_factor: Multiplier for IQR to determine outlier bounds.

        Returns:
            Tuple of (x_min, x_max) for axis limits, or (None, None) if
            limits cannot be computed (e.g., empty or all-NaN data).
        """
        scores_arr = np.array(scores)

        # Filter out NaN and Inf values
        valid_scores = scores_arr[np.isfinite(scores_arr)]

        # Handle empty or all-invalid data
        if len(valid_scores) == 0:
            return None, None

        q25 = float(np.quantile(valid_scores, 0.25))
        q75 = float(np.quantile(valid_scores, 0.75))
        iqr = q75 - q25

        # Handle case where IQR is 0 (all values are the same)
        if iqr == 0:
            iqr = abs(q75) * 0.1 if q75 != 0 else 0.1

        # Compute bounds using IQR
        lower_bound = q25 - iqr_factor * iqr
        upper_bound = q75 + iqr_factor * iqr

        # Ensure we don't go beyond actual data range
        data_min = float(np.min(valid_scores))
        data_max = float(np.max(valid_scores))

        x_min = max(data_min, lower_bound)
        x_max = min(data_max, upper_bound)

        # Add small padding (5%) for visual comfort
        padding = (x_max - x_min) * 0.05
        x_min -= padding
        x_max += padding

        # Final safety check for NaN/Inf
        if not np.isfinite(x_min) or not np.isfinite(x_max):
            return None, None

        return x_min, x_max
