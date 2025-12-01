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

    def _select_top_k_by_borda(
        self,
        matrix: np.ndarray,
        y_labels: List[str],
        top_k: int,
        higher_better: bool
    ) -> tuple:
        """Select top K models using Borda count ranking.

        Selection strategy:
        1. Always keep the top-1 model for each column (dataset/partition)
        2. Rank remaining models by Borda count (sum of ranks per column)
        3. Select top_k models total

        Args:
            matrix: Score matrix of shape (n_models, n_columns).
            y_labels: List of model names (rows).
            top_k: Number of top models to keep.
            higher_better: If True, higher scores are better.

        Returns:
            Tuple of (filtered_matrix, filtered_y_labels, selected_indices).
        """
        n_models, n_cols = matrix.shape

        if top_k >= n_models:
            return matrix, y_labels, list(range(n_models))

        # Compute per-column ranks (1 = best, n = worst)
        # Handle NaN by assigning worst rank
        ranks = np.zeros_like(matrix)
        for j in range(n_cols):
            col = matrix[:, j].copy()
            valid_mask = ~np.isnan(col)
            n_valid = valid_mask.sum()

            if n_valid == 0:
                ranks[:, j] = n_models  # All worst
                continue

            # Argsort: ascending by default
            if higher_better:
                # Higher is better -> descending order -> negate for argsort
                order = np.argsort(-col)
            else:
                # Lower is better -> ascending order
                order = np.argsort(col)

            # Assign ranks
            for rank_pos, idx in enumerate(order):
                if valid_mask[idx]:
                    ranks[idx, j] = rank_pos + 1  # 1-indexed rank
                else:
                    ranks[idx, j] = n_models  # NaN gets worst rank

        # Step 1: Find top-1 model for each column (must be kept)
        top1_indices = set()
        for j in range(n_cols):
            col = matrix[:, j]
            valid_mask = ~np.isnan(col)
            if valid_mask.any():
                if higher_better:
                    best_idx = np.nanargmax(col)
                else:
                    best_idx = np.nanargmin(col)
                top1_indices.add(best_idx)

        # Step 2: Compute Borda count for all models (lower is better - sum of ranks)
        borda_scores = np.nansum(ranks, axis=1)

        # Step 3: Select remaining models by Borda count
        # First, rank all models by Borda score
        borda_order = np.argsort(borda_scores)  # Lower borda score = better

        selected_indices = set(top1_indices)
        for idx in borda_order:
            if len(selected_indices) >= top_k:
                break
            selected_indices.add(idx)

        # Sort selected indices to maintain original order
        selected_indices = sorted(selected_indices)

        # Filter matrix and labels
        filtered_matrix = matrix[selected_indices, :]
        filtered_labels = [y_labels[i] for i in selected_indices]

        return filtered_matrix, filtered_labels, selected_indices

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
        column_scale: bool = False,
        aggregate: Optional[str] = None,
        top_k: Optional[int] = None,
        **filters
    ) -> Figure:
        """Render performance heatmap (Optimized with Polars).

        Uses vectorized operations for 20x+ speedup.
        When aggregate is provided, uses the slower but accurate aggregation path.

        Args:
            x_var: Variable for X-axis (columns).
            y_var: Variable for Y-axis (rows).
            rank_metric: Metric used for ranking models.
            rank_partition: Partition used for ranking ('val', 'test', 'train').
            display_metric: Metric displayed in cells.
            display_partition: Partition for display metric.
            figsize: Figure size (auto-computed if None).
            normalize: Whether to normalize displayed values.
            rank_agg: Ranking aggregation ('best', 'worst', 'mean', 'median').
            display_agg: Display aggregation strategy.
            show_counts: Whether to show sample counts.
            local_scale: If True, use local scale for colors.
            column_scale: If True, normalize colors per column (best in column = 1.0).
                         Automatically sets local_scale=False when enabled.
            aggregate: Aggregation column for sample-level aggregation.
            top_k: If provided, show only top K models. Selection uses Borda count:
                   first keeps top-1 per column, then ranks by Borda count.
            **filters: Additional filters for predictions.

        Returns:
            Matplotlib Figure with the heatmap.
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

        # If aggregation is requested, use the slower but accurate path
        if aggregate is not None:
            return self._render_with_aggregation(
                x_var=x_var,
                y_var=y_var,
                rank_metric=rank_metric,
                rank_partition=rank_partition,
                display_metric=display_metric,
                display_partition=display_partition,
                figsize=figsize,
                normalize=normalize,
                rank_agg=rank_agg,
                display_agg=display_agg,
                show_counts=show_counts,
                local_scale=local_scale,
                column_scale=column_scale,
                aggregate=aggregate,
                top_k=top_k,
                **filters
            )

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
        for k in ['aggregation', 'rank_agg', 'display_agg', 'show_counts', 'figsize', 'aggregate']:
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
            .select(rank_select_cols)  # Include grouping vars if present
        )

        # Display Data
        if is_partition_grouped:
            # If grouped by partition, we need all partitions, not just display_partition
            # Compute score for each partition, then select based on row's partition value
            df_disp = df.with_columns(
                get_score_expr(display_metric, "test").alias("display_score_test"),
                get_score_expr(display_metric, "val").alias("display_score_val"),
                get_score_expr(display_metric, "train").alias("display_score_train"),
            ).with_columns(
                pl.when(pl.col("partition") == "test").then(pl.col("display_score_test"))
                .when(pl.col("partition") == "val").then(pl.col("display_score_val"))
                .when(pl.col("partition") == "train").then(pl.col("display_score_train"))
                .otherwise(None)
                .alias("display_score")
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
        else:  # median
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

        # Determine if higher is better for display metric
        display_higher_better = self._is_higher_better(display_metric)

        # Apply top_k filtering if requested
        if top_k is not None and top_k > 0:
            matrix, y_labels, selected_indices = self._select_top_k_by_borda(
                matrix, y_labels, top_k, display_higher_better
            )
            count_matrix = count_matrix[selected_indices, :]

        # Auto-compute figsize based on number of labels (always recompute for dynamic sizing)
        figsize = self._compute_figsize(len(x_labels), len(y_labels))

        # Normalize for colors
        normalize_per_row = is_dataset_grouped and (y_var == 'dataset_name')

        # column_scale overrides local_scale and per_row normalization
        if column_scale:
            local_scale = False
            normalized_matrix = self.normalizer.normalize(
                matrix, display_higher_better, per_column=True
            )
        else:
            normalized_matrix = self.normalizer.normalize(
                matrix, display_higher_better, per_row=normalize_per_row
            )

        # Render
        fig = self._render_heatmap(
            matrix, normalized_matrix, count_matrix,
            x_labels, y_labels, x_var, y_var,
            rank_metric, rank_partition, rank_agg,
            display_metric, display_partition, display_agg,
            figsize, normalize, show_counts, local_scale, display_higher_better,
            column_scale, top_k
        )

        t2 = time.time()
        print(f"Matplotlib render time: {t2 - t1:.4f} seconds")

        return fig

    def _compute_figsize(self, n_x: int, n_y: int) -> tuple:
        """Compute optimal figure size based on number of labels.

        Ensures labels don't overlap by scaling height with number of Y labels
        and width with number of X labels.

        Args:
            n_x: Number of X-axis labels.
            n_y: Number of Y-axis labels.

        Returns:
            Tuple of (width, height) in inches.
        """
        # Base sizes
        base_width, base_height = self.config.get_figsize('medium')

        # Minimum cell size (in inches) to avoid overlap
        min_cell_height = 0.35  # ~0.35 inches per Y label
        min_cell_width = 0.6   # ~0.6 inches per X label (rotated labels need more width)

        # Compute required size
        required_height = max(base_height, n_y * min_cell_height + 2)  # +2 for margins/title
        required_width = max(base_width, n_x * min_cell_width + 3)     # +3 for y-labels/colorbar

        # Cap at reasonable maximum
        max_height = 40
        max_width = 30

        return (min(required_width, max_width), min(required_height, max_height))

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
        display_higher_better: bool,
        column_scale: bool = False,
        top_k: Optional[int] = None
    ) -> Figure:
        """Render the heatmap figure."""
        fig, ax = plt.subplots(figsize=figsize)

        # Determine scaling mode
        # Force local_scale=True for regression metrics (unbounded) unless explicitly set
        is_bounded_0_1 = display_metric.lower() in [
            'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1',
            'specificity', 'auc', 'roc_auc', 'jaccard'
        ] or any(m in display_metric.lower() for m in ['accuracy', 'precision', 'recall', 'f1'])

        use_local_scale = local_scale or not is_bounded_0_1

        # When column_scale is enabled, use normalized matrix for coloring
        # with 0-1 scale (each column independently normalized)
        if column_scale:
            display_data = np.ma.masked_invalid(normalized_matrix)
            vmin = 0
            vmax = 1
            cbar_label = f'{display_metric.upper()}\n(per-column: green=best)'
        elif use_local_scale:
            display_data = np.ma.masked_invalid(matrix)
            vmin = np.nanmin(matrix)
            vmax = np.nanmax(matrix)
            cbar_label = f'{display_metric.upper()}\n(green=best, red=worst)'
        else:
            display_data = np.ma.masked_invalid(matrix)
            vmin = 0
            vmax = 1
            cbar_label = f'{display_metric.upper()}\n(green=best, red=worst)'

        # Select colormap based on direction
        cmap_name = self.config.heatmap_colormap
        if not display_higher_better:
            cmap_name += '_r'

        im = ax.imshow(
            display_data,
            cmap=cmap_name,
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label, fontsize=self.config.label_fontsize)

        # Adapt font size based on number of labels to avoid overlap
        n_y = len(y_labels)
        tick_fontsize = self.config.tick_fontsize
        if n_y > 30:
            tick_fontsize = max(5, tick_fontsize - 3)
        elif n_y > 20:
            tick_fontsize = max(6, tick_fontsize - 2)
        elif n_y > 15:
            tick_fontsize = max(7, tick_fontsize - 1)

        # Axis labels
        x_labels_display = [str(lbl)[:25] + '...' if len(str(lbl)) > 25 else str(lbl) for lbl in x_labels]
        y_labels_display = [str(lbl)[:25] + '...' if len(str(lbl)) > 25 else str(lbl) for lbl in y_labels]

        ax.set_xticks(range(len(x_labels)))
        ax.set_yticks(range(len(y_labels)))
        ax.set_xticklabels(x_labels_display, rotation=45, ha='right', fontsize=tick_fontsize)
        ax.set_yticklabels(y_labels_display, fontsize=tick_fontsize)
        ax.set_xlabel(x_var.replace('_', ' ').title(), fontsize=self.config.label_fontsize)
        ax.set_ylabel(y_var.replace('_', ' ').title(), fontsize=self.config.label_fontsize)

        # Title: Show display aggregation and metric, and add ranking info if different
        title_parts = []

        # Add top_k indicator first if applicable
        if top_k is not None and top_k > 0:
            title_parts.append(f'Top {top_k}:')

        title_parts.append(f'{display_agg.title()} {display_metric} [{display_partition}]')

        # Add column_scale indicator
        if column_scale:
            title_parts.append('[column normalized]')

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

    def _render_with_aggregation(
        self,
        x_var: str,
        y_var: str,
        rank_metric: str,
        rank_partition: str,
        display_metric: str,
        display_partition: str,
        figsize: tuple,
        normalize: bool,
        rank_agg: str,
        display_agg: str,
        show_counts: bool,
        local_scale: bool,
        column_scale: bool,
        aggregate: str,
        top_k: Optional[int] = None,
        **filters
    ) -> Figure:
        """Render heatmap with aggregation support.

        This is slower than the default render because it needs to load arrays
        and recalculate metrics after aggregation.
        """
        t0 = time.time()

        # Determine if partition or dataset_name is used as a grouping variable
        is_partition_grouped = (x_var == 'partition' or y_var == 'partition')
        is_dataset_grouped = (x_var == 'dataset_name' or y_var == 'dataset_name')

        # Remove internal parameters from filters
        all_filters = dict(filters)
        for k in ['aggregation', 'rank_agg', 'display_agg', 'show_counts', 'figsize', 'aggregate', 'column_scale']:
            all_filters.pop(k, None)

        df = self.predictions.to_dataframe()

        # Apply filters
        for k, v in all_filters.items():
            if k in df.columns:
                df = df.filter(pl.col(k) == v)

        if df.height == 0:
            raise ValueError(f"No predictions found with filters: {all_filters}")

        # Get unique combinations of x_var and y_var
        # When partition is a grouping variable, don't filter to display_partition
        if is_partition_grouped:
            source_df = df
        else:
            source_df = df.filter(pl.col("partition") == display_partition)

        if source_df.height == 0:
            raise ValueError(f"No predictions found for partition: {display_partition}")

        # Get unique x and y values
        x_labels = sorted([str(x) for x in source_df[x_var].unique().to_list()], key=self._natural_sort_key)
        y_labels = sorted([str(y) for y in source_df[y_var].unique().to_list()], key=self._natural_sort_key)

        # Create mappings
        x_map = {x: i for i, x in enumerate(x_labels)}
        y_map = {y: i for i, y in enumerate(y_labels)}

        matrix = np.full((len(y_labels), len(x_labels)), np.nan)
        count_matrix = np.zeros((len(y_labels), len(x_labels)), dtype=int)

        # Get unique (x_var, y_var) combinations
        unique_combinations = source_df.select([x_var, y_var]).unique().to_dicts()

        # Process each unique cell - get predictions with aggregation using top()
        for combo in unique_combinations:
            x_val = str(combo[x_var])
            y_val = str(combo[y_var])

            if x_val not in x_map or y_val not in y_map:
                continue

            x_idx = x_map[x_val]
            y_idx = y_map[y_val]

            # Skip if already processed
            if not np.isnan(matrix[y_idx, x_idx]):
                continue

            # Get aggregated predictions for this cell
            cell_filters = {**all_filters, x_var: combo[x_var], y_var: combo[y_var]}

            # When partition is a grouping variable, use the combo's partition value
            if is_partition_grouped:
                cell_display_partition = combo.get('partition', display_partition)
                # Remove partition from cell_filters to not conflict with display_partition
                cell_filters.pop('partition', None)
            else:
                cell_display_partition = display_partition

            try:
                # Use top(1) with aggregation to get the best model for this cell
                top_preds = self.predictions.top(
                    n=1,
                    rank_metric=rank_metric,
                    rank_partition=rank_partition,
                    display_metrics=[display_metric],
                    display_partition=cell_display_partition,
                    aggregate_partitions=True,
                    aggregate=aggregate,
                    **cell_filters
                )

                if top_preds:
                    pred = top_preds[0]
                    # Get the score from the display partition
                    partitions = pred.get('partitions', {})
                    display_data = partitions.get(cell_display_partition, {})
                    score = display_data.get(display_metric)

                    if score is None:
                        # Try y_true/y_pred calculation
                        y_true = display_data.get('y_true')
                        y_pred = display_data.get('y_pred')
                        if y_true is not None and y_pred is not None:
                            try:
                                score = evaluator.eval(y_true, y_pred, display_metric)
                            except Exception:
                                pass

                    if score is not None:
                        matrix[y_idx, x_idx] = score
                        # Count aggregated samples
                        y_pred = display_data.get('y_pred')
                        if y_pred is not None:
                            count_matrix[y_idx, x_idx] = len(y_pred)
                        else:
                            count_matrix[y_idx, x_idx] = 1
            except Exception:
                pass

        t1 = time.time()
        print(f"Data wrangling time (with aggregation): {t1 - t0:.4f} seconds")

        # Determine if higher is better for display metric
        display_higher_better = self._is_higher_better(display_metric)

        # Apply top_k filtering if requested
        if top_k is not None and top_k > 0:
            matrix, y_labels, selected_indices = self._select_top_k_by_borda(
                matrix, y_labels, top_k, display_higher_better
            )
            count_matrix = count_matrix[selected_indices, :]

        # Auto-compute figsize based on number of labels (always recompute for dynamic sizing)
        figsize = self._compute_figsize(len(x_labels), len(y_labels))

        # Normalize for colors
        normalize_per_row = is_dataset_grouped and (y_var == 'dataset_name')

        # column_scale overrides local_scale and per_row normalization
        if column_scale:
            local_scale = False
            normalized_matrix = self.normalizer.normalize(
                matrix, display_higher_better, per_column=True
            )
        else:
            normalized_matrix = self.normalizer.normalize(
                matrix, display_higher_better, per_row=normalize_per_row
            )

        # Render with aggregation note in title
        fig = self._render_heatmap_aggregated(
            matrix, normalized_matrix, count_matrix,
            x_labels, y_labels, x_var, y_var,
            rank_metric, rank_partition, rank_agg,
            display_metric, display_partition, display_agg,
            figsize, normalize, show_counts, local_scale, display_higher_better,
            aggregate, column_scale, top_k
        )

        t2 = time.time()
        print(f"Matplotlib render time: {t2 - t1:.4f} seconds")

        return fig

    def _render_heatmap_aggregated(
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
        display_higher_better: bool,
        aggregate: str,
        column_scale: bool = False,
        top_k: Optional[int] = None
    ) -> Figure:
        """Render the heatmap figure with aggregation note."""
        fig, ax = plt.subplots(figsize=figsize)

        # Use normalized matrix for colors (always)
        masked_matrix = np.ma.masked_invalid(normalized_matrix)

        # Determine scaling mode
        is_bounded_0_1 = display_metric.lower() in [
            'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1',
            'specificity', 'auc', 'roc_auc', 'jaccard'
        ] or any(m in display_metric.lower() for m in ['accuracy', 'precision', 'recall', 'f1'])

        use_local_scale = local_scale or not is_bounded_0_1

        # When column_scale is enabled, use normalized matrix for coloring
        # with 0-1 scale (each column independently normalized)
        if column_scale:
            display_data = np.ma.masked_invalid(normalized_matrix)
            vmin = 0
            vmax = 1
            cbar_label = f'{display_metric.upper()}\n(per-column: green=best)'
        elif use_local_scale:
            display_data = np.ma.masked_invalid(matrix)
            vmin = np.nanmin(matrix)
            vmax = np.nanmax(matrix)
            cbar_label = f'{display_metric.upper()}\n(green=best, red=worst)'
        else:
            display_data = np.ma.masked_invalid(matrix)
            vmin = 0
            vmax = 1
            cbar_label = f'{display_metric.upper()}\n(green=best, red=worst)'

        # Select colormap based on direction
        cmap_name = self.config.heatmap_colormap
        if not display_higher_better:
            cmap_name += '_r'

        im = ax.imshow(
            display_data,
            cmap=cmap_name,
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label, fontsize=self.config.label_fontsize)

        # Adapt font size based on number of labels to avoid overlap
        n_y = len(y_labels)
        tick_fontsize = self.config.tick_fontsize
        if n_y > 30:
            tick_fontsize = max(5, tick_fontsize - 3)
        elif n_y > 20:
            tick_fontsize = max(6, tick_fontsize - 2)
        elif n_y > 15:
            tick_fontsize = max(7, tick_fontsize - 1)

        # Axis labels
        x_labels_display = [str(lbl)[:25] + '...' if len(str(lbl)) > 25 else str(lbl) for lbl in x_labels]
        y_labels_display = [str(lbl)[:25] + '...' if len(str(lbl)) > 25 else str(lbl) for lbl in y_labels]

        ax.set_xticks(range(len(x_labels)))
        ax.set_yticks(range(len(y_labels)))
        ax.set_xticklabels(x_labels_display, rotation=45, ha='right', fontsize=tick_fontsize)
        ax.set_yticklabels(y_labels_display, fontsize=tick_fontsize)
        ax.set_xlabel(x_var.replace('_', ' ').title(), fontsize=self.config.label_fontsize)
        ax.set_ylabel(y_var.replace('_', ' ').title(), fontsize=self.config.label_fontsize)

        # Title with aggregation info
        title_parts = []

        # Add top_k indicator first if applicable
        if top_k is not None and top_k > 0:
            title_parts.append(f'Top {top_k}:')

        title_parts.append(f'{display_agg.title()} {display_metric} [{display_partition}]')
        title_parts.append(f'[aggregated by {aggregate}]')

        # Add column_scale indicator
        if column_scale:
            title_parts.append('[column normalized]')

        if rank_partition != display_partition or rank_metric != display_metric or rank_agg != display_agg:
            title_parts.append(f'(rank on {rank_agg} {rank_metric} [{rank_partition}])')

        title = ' '.join(title_parts)
        ax.set_title(title, fontsize=self.config.title_fontsize, pad=10)

        # Cell annotations
        display_matrix = normalized_matrix if normalize else matrix
        self.annotator.add_heatmap_annotations(
            ax, display_matrix, normalized_matrix, count_matrix,
            x_labels, y_labels, show_counts
        )

        plt.tight_layout()
        return fig
