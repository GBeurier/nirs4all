"""
HeatmapChart - Heatmap visualization of performance across two variables.

CORE LOGIC:
1. Get all predictions
2. Rank predictions by rank_metric on rank_partition using rank_agg
3. Group by (x_var, y_var)
4. For each cell, get display_metric from display_partition using display_agg
5. Normalize per dataset if requested
6. Render with color based on normalized scores
"""
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import time
from matplotlib.figure import Figure
from typing import Optional, List, Dict, Any
from nirs4all.visualization.charts.base import BaseChart
from nirs4all.visualization.chart_utils.normalizer import ScoreNormalizer
from nirs4all.visualization.chart_utils.annotator import ChartAnnotator
from nirs4all.visualization.chart_utils.matrix_builder import MatrixBuilder
from nirs4all.core import metrics as evaluator


class HeatmapChart(BaseChart):
    """Heatmap visualization of performance across two variables.

    Supports flexible ranking and display configurations with multiple
    aggregation strategies.
    """

    def __init__(self, predictions, dataset_name_override: Optional[str] = None, config=None):
        super().__init__(predictions, dataset_name_override, config)
        self.normalizer = ScoreNormalizer()
        self.annotator = ChartAnnotator(config)
        self.matrix_builder = MatrixBuilder()

    def validate_inputs(self, x_var: str, y_var: str, rank_metric: str, **kwargs) -> None:
        """Validate inputs."""
        if not x_var or not isinstance(x_var, str):
            raise ValueError("x_var must be a non-empty string")
        if not y_var or not isinstance(y_var, str):
            raise ValueError("y_var must be a non-empty string")
        if not rank_metric or not isinstance(rank_metric, str):
            raise ValueError("rank_metric must be a non-empty string")

    def render(
        self,
        x_var: str,
        y_var: str,
        rank_metric: Optional[str] = None,
        rank_partition: str = 'val',
        display_metric: str = '',
        display_partition: str = 'test',
        figsize: Optional[tuple] = None,
        normalize: bool = False,
        rank_agg: str = 'best',
        display_agg: str = 'mean',
        show_counts: bool = True,
        local_scale: bool = False,
        **filters
    ) -> Figure:
        """Render performance heatmap (Optimized with Polars).

        Uses vectorized operations for 20x+ speedup.
        """
        t0 = time.time()

        # Auto-detect metric if not provided
        if rank_metric is None:
            if display_metric:
                rank_metric = display_metric
            else:
                rank_metric = self._get_default_metric()

        self.validate_inputs(x_var, y_var, rank_metric)

        if figsize is None:
            figsize = self.config.get_figsize('medium')

        if not display_metric:
            display_metric = rank_metric

        # Determine if partition or dataset_name is used as a grouping variable
        is_partition_grouped = (x_var == 'partition' or y_var == 'partition')
        is_dataset_grouped = (x_var == 'dataset_name' or y_var == 'dataset_name')

        # Remove grouping variables from filters
        all_filters = dict(filters)
        if is_partition_grouped:
            all_filters.pop('partition', None)
        if is_dataset_grouped:
            all_filters.pop('dataset_name', None)

        # Remove internal parameters
        for k in ['aggregation', 'rank_agg', 'display_agg', 'show_counts', 'figsize']:
            all_filters.pop(k, None)

        # --- POLARS OPTIMIZATION START ---
        df = self.predictions.to_dataframe()

        # 1. Apply Filters
        for k, v in all_filters.items():
            if k in df.columns:
                df = df.filter(pl.col(k) == v)

        if df.height == 0:
            raise ValueError(f"No predictions found with filters: {all_filters}")

        # 2. Define Score Extraction Logic (Vectorized)
        def get_score_expr(metric_name, partition_name):
            # Priority 1: Direct column if metric matches
            # Priority 2: Regex from scores JSON (fast approximation)
            # Priority 3: Null

            # Direct column (e.g. 'val_score')
            col_score = f"{partition_name}_score"

            # Regex for JSON: "partition": { ... "metric": value ... }
            # Simplified regex: look for metric key inside partition block?
            # JSON structure: {"val": {"rmse": 0.1, ...}, ...}
            # Regex: "val"\s*:\s*\{[^}]*"rmse"\s*:\s*([\d\.]+)
            # Note: This is fragile but fast.
            regex = f'"{partition_name}"\\s*:\\s*\\{{[^}}]*"{metric_name}"\\s*:\\s*([\\d\\.]+)'

            return (
                pl.when(pl.col("metric") == metric_name)
                .then(pl.col(col_score))
                .otherwise(
                    pl.col("scores").str.extract(regex, 1).cast(pl.Float64, strict=False)
                )
            )

        # 3. Prepare Rank and Display Data
        # We need to join rank partition data with display partition data
        # Identity columns for joining
        join_cols = ['dataset_name', 'model_name', 'config_name', 'fold_id', 'step_idx', 'op_counter']
        # Ensure columns exist
        join_cols = [c for c in join_cols if c in df.columns]

        # Rank Data
        rank_select_cols = list(set(join_cols + ["rank_score", x_var, y_var]))
        df_rank = (
            df.filter(pl.col("partition") == rank_partition)
            .with_columns(get_score_expr(rank_metric, rank_partition).alias("rank_score"))
            .select(rank_select_cols) # Include grouping vars if present
        )

        # Display Data
        if is_partition_grouped:
            # If grouped by partition, we need all partitions, not just display_partition
            df_disp = (
                df
                .with_columns(
                    pl.when(pl.col("metric") == display_metric)
                    .then(pl.col(pl.col("partition") + "_score")) # Dynamic col name? No, hard to do.
                    # Fallback: just use the score col corresponding to the row's partition
                    .otherwise(None) # TODO: Improve for partition grouping
                    .alias("display_score")
                )
            )
            # Actually, if x_var is partition, we just take the score of that row
            # Assuming row['metric'] == display_metric for simplicity in V2
             # If we need cross-metric, it's harder.
            df_disp = df.with_columns(
                 get_score_expr(display_metric, "test").alias("display_score_test"), # Hacky
                 get_score_expr(display_metric, "val").alias("display_score_val"),
                 get_score_expr(display_metric, "train").alias("display_score_train"),
            ).with_columns(
                pl.coalesce([
                    pl.when(pl.col("partition")=="test").then("display_score_test"),
                    pl.when(pl.col("partition")=="val").then("display_score_val"),
                    pl.when(pl.col("partition")=="train").then("display_score_train")
                ]).alias("display_score")
            )
        else:
            disp_select_cols = list(set(join_cols + ["display_score", x_var, y_var]))
            df_disp = (
                df.filter(pl.col("partition") == display_partition)
                .with_columns(get_score_expr(display_metric, display_partition).alias("display_score"))
                .select(disp_select_cols)
            )

        # 4. Join
        # If rank and display partitions are same, we don't need join, just filter
        if rank_partition == display_partition and not is_partition_grouped:
            combined = df_rank.with_columns(pl.col("rank_score").alias("display_score"))
        elif is_partition_grouped:
             # If partition grouped, we join rank info (for selection) onto all rows
             # This allows selecting the "best fold" based on val, but showing all partitions for that fold
             combined = df_disp.join(df_rank.select(join_cols + ["rank_score"]), on=join_cols, how="inner")
        else:
            combined = df_disp.join(df_rank.select(join_cols + ["rank_score"]), on=join_cols, how="inner")

        # 5. Group and Aggregate
        # We need to group by (x_var, y_var)
        # And aggregate: select best model based on rank_score, then take its display_score

        # Filter out nulls
        combined = combined.filter(pl.col("rank_score").is_not_null() & pl.col("display_score").is_not_null())

        if combined.height == 0:
             raise ValueError(f"No valid scores found for {x_var} vs {y_var}")

        # Sort for ranking
        rank_higher_better = self._is_higher_better(rank_metric)
        combined = combined.sort("rank_score", descending=rank_higher_better)

        # Aggregation
        # We want one value per (x, y) group
        # Strategy: Group by x,y -> take first (best) or aggregate

        if rank_agg == 'best':
            # Since we sorted by rank_score, 'first' is the best
            agg_df = combined.group_by([x_var, y_var]).agg([
                pl.col("display_score").first().alias("agg_score"),
                pl.len().alias("count")
            ])
        elif rank_agg == 'worst':
            agg_df = combined.group_by([x_var, y_var]).agg([
                pl.col("display_score").last().alias("agg_score"),
                pl.len().alias("count")
            ])
        elif rank_agg == 'mean':
            # For mean, we might want mean of display scores of ALL models, or top K?
            # Standard behavior: mean of display scores of ALL matching models
            agg_df = combined.group_by([x_var, y_var]).agg([
                pl.col("display_score").mean().alias("agg_score"),
                pl.len().alias("count")
            ])
        else: # median
             agg_df = combined.group_by([x_var, y_var]).agg([
                pl.col("display_score").median().alias("agg_score"),
                pl.len().alias("count")
            ])

        # 6. Build Matrix (Pivot)
        # Polars pivot is great
        # We need a matrix of scores and a matrix of counts

        # Collect unique labels
        x_labels = sorted([str(x) for x in agg_df[x_var].unique().to_list()], key=self._natural_sort_key)
        y_labels = sorted([str(y) for y in agg_df[y_var].unique().to_list()], key=self._natural_sort_key)

        # Create mapping for indices
        x_map = {x: i for i, x in enumerate(x_labels)}
        y_map = {y: i for i, y in enumerate(y_labels)}

        matrix = np.full((len(y_labels), len(x_labels)), np.nan)
        count_matrix = np.zeros((len(y_labels), len(x_labels)), dtype=int)

        # Fill matrix
        # Iterate over aggregated rows (much fewer than raw predictions)
        for row in agg_df.to_dicts():
            x_val = str(row[x_var])
            y_val = str(row[y_var])
            score = row["agg_score"]
            count = row["count"]

            if x_val in x_map and y_val in y_map:
                matrix[y_map[y_val], x_map[x_val]] = score
                count_matrix[y_map[y_val], x_map[x_val]] = count

        t1 = time.time()
        print(f"Data wrangling time: {t1 - t0:.4f} seconds")

        # --- POLARS OPTIMIZATION END ---

        # Normalize for colors
        display_higher_better = self._is_higher_better(display_metric)
        normalize_per_row = is_dataset_grouped and (y_var == 'dataset_name')
        normalized_matrix = self.normalizer.normalize(
            matrix, display_higher_better, per_row=normalize_per_row
        )

        # Render
        fig = self._render_heatmap(
            matrix, normalized_matrix, count_matrix,
            x_labels, y_labels, x_var, y_var,
            rank_metric, rank_partition, rank_agg,
            display_metric, display_partition, display_agg,
            figsize, normalize, show_counts, local_scale, display_higher_better
        )

        t2 = time.time()
        print(f"Matplotlib render time: {t2 - t1:.4f} seconds")

        return fig

    @staticmethod
    def _is_higher_better(metric: str) -> bool:
        """Check if metric is higher-is-better."""
        metric_lower = metric.lower()
        # Classification metrics (higher is better)
        higher_is_better = [
            'accuracy', 'balanced_accuracy',
            'precision', 'balanced_precision', 'precision_micro', 'precision_macro',
            'recall', 'balanced_recall', 'recall_micro', 'recall_macro',
            'f1', 'f1_micro', 'f1_macro',
            'specificity', 'roc_auc', 'auc',
            'matthews_corrcoef', 'cohen_kappa', 'jaccard',
            # Regression metrics (higher is better)
            'r2', 'r2_score'
        ]
        return metric_lower in higher_is_better

    def _render_heatmap(
        self,
        matrix: np.ndarray,
        normalized_matrix: np.ndarray,
        count_matrix: np.ndarray,
        x_labels: List[str],
        y_labels: List[str],
        x_var: str,
        y_var: str,
        rank_metric: str,
        rank_partition: str,
        rank_agg: str,
        display_metric: str,
        display_partition: str,
        display_agg: str,
        figsize: tuple,
        normalize: bool,
        show_counts: bool,
        local_scale: bool,
        display_higher_better: bool
    ) -> Figure:
        """Render the heatmap figure."""
        fig, ax = plt.subplots(figsize=figsize)

        # Use normalized matrix for colors (always)
        masked_matrix = np.ma.masked_invalid(normalized_matrix)

        # Determine scaling mode
        # Force local_scale=True for regression metrics (unbounded) unless explicitly set
        is_bounded_0_1 = display_metric.lower() in [
            'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1',
            'specificity', 'auc', 'roc_auc', 'jaccard'
        ] or any(m in display_metric.lower() for m in ['accuracy', 'precision', 'recall', 'f1'])

        use_local_scale = local_scale or not is_bounded_0_1

        masked_raw = np.ma.masked_invalid(matrix)

        if use_local_scale:
            vmin = np.nanmin(matrix)
            vmax = np.nanmax(matrix)
        else:
            vmin = 0
            vmax = 1

        # Select colormap based on direction
        cmap_name = self.config.heatmap_colormap
        if not display_higher_better:
            cmap_name += '_r'

        im = ax.imshow(
            masked_raw,
            cmap=cmap_name,
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )
        cbar_label = f'{display_metric.upper()}\n(green=best, red=worst)'

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label, fontsize=self.config.label_fontsize)

        # Axis labels
        x_labels_display = [str(lbl)[:25] + '...' if len(str(lbl)) > 25 else str(lbl) for lbl in x_labels]
        y_labels_display = [str(lbl)[:25] + '...' if len(str(lbl)) > 25 else str(lbl) for lbl in y_labels]

        ax.set_xticks(range(len(x_labels)))
        ax.set_yticks(range(len(y_labels)))
        ax.set_xticklabels(x_labels_display, rotation=45, ha='right', fontsize=self.config.tick_fontsize)
        ax.set_yticklabels(y_labels_display, fontsize=self.config.tick_fontsize)
        ax.set_xlabel(x_var.replace('_', ' ').title(), fontsize=self.config.label_fontsize)
        ax.set_ylabel(y_var.replace('_', ' ').title(), fontsize=self.config.label_fontsize)

        # Title: Show display aggregation and metric, and add ranking info if different
        title_parts = [f'{display_agg.title()} {display_metric} [{display_partition}]']

        # Add ranking score info if different from display (mimic confusion matrix behavior)
        # Show both the display and ranking configurations in the title
        if rank_partition != display_partition or rank_metric != display_metric or rank_agg != display_agg:
            title_parts.append(f'(rank on {rank_agg} {rank_metric} [{rank_partition}])')

        title = ' '.join(title_parts)
        ax.set_title(title, fontsize=self.config.title_fontsize, pad=10)

        # Cell annotations
        # Use normalized matrix if normalize=True, otherwise use raw matrix
        display_matrix = normalized_matrix if normalize else matrix
        self.annotator.add_heatmap_annotations(
            ax, display_matrix, normalized_matrix, count_matrix,
            x_labels, y_labels, show_counts
        )

        plt.tight_layout()
        return fig
