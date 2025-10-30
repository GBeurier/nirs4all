"""
HeatmapChart - Heatmap visualization of performance across two variables.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional
from nirs4all.visualization.charts.base import BaseChart
from nirs4all.visualization.chart_utils.predictions_adapter import PredictionsAdapter
from nirs4all.visualization.chart_utils.matrix_builder import MatrixBuilder
from nirs4all.visualization.chart_utils.normalizer import ScoreNormalizer
from nirs4all.visualization.chart_utils.annotator import ChartAnnotator
from nirs4all.core import metrics as evaluator


class HeatmapChart(BaseChart):
    """Heatmap visualization of performance across two variables.

    Unified heatmap implementation supporting flexible ranking and
    display configurations with multiple aggregation strategies.

    Optimized Implementation:
    - Uses PredictionsAdapter for efficient data access
    - Supports lazy figure creation for large heatmaps
    - Leverages predictions.top() API instead of manual calculations
    """

    def __init__(self, predictions, dataset_name_override: Optional[str] = None,
                 config=None):
        """Initialize heatmap chart.

        Args:
            predictions: Predictions object instance.
            dataset_name_override: Optional dataset name override.
            config: Optional ChartConfig for customization.
        """
        super().__init__(predictions, dataset_name_override, config)
        self.adapter = PredictionsAdapter(predictions)
        self.matrix_builder = MatrixBuilder()
        self.normalizer = ScoreNormalizer()
        self.annotator = ChartAnnotator(config)

    def validate_inputs(self, x_var: str, y_var: str, rank_metric: str, **kwargs) -> None:
        """Validate heatmap inputs.

        Args:
            x_var: X-axis variable name.
            y_var: Y-axis variable name.
            rank_metric: Ranking metric name.
            **kwargs: Additional parameters (ignored).

        Raises:
            ValueError: If inputs are invalid.
        """
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
        rank_metric: str = 'rmse',
        rank_partition: str = 'val',
        display_metric: str = 'rmse',
        display_partition: str = 'test',
        figsize: Optional[tuple] = None,
        normalize: bool = True,
        aggregation: str = 'best',
        show_counts: bool = True,
        lazy_render: bool = False,
        **filters
    ) -> Figure:
        """Render performance heatmap.

        Args:
            x_var: Variable for x-axis (e.g., 'model_name', 'preprocessings').
            y_var: Variable for y-axis.
            rank_metric: Metric used to rank/select best models (default: 'rmse').
            rank_partition: Partition used for ranking models (default: 'val').
            display_metric: Metric to display in heatmap (default: same as rank_metric).
            display_partition: Partition to display scores from (default: 'test').
            figsize: Figure size tuple (default: from config).
            normalize: Whether to normalize scores to [0,1] (default: True).
            aggregation: How to aggregate scores per cell: 'best', 'mean', 'median' (default: 'best').
            show_counts: Whether to show sample counts in cells (default: True).
            lazy_render: If True, create placeholder for deferred rendering (default: False).
            **filters: Additional filters (dataset_name, config_name, etc.).

        Returns:
            matplotlib Figure object.
        """
        self.validate_inputs(x_var, y_var, rank_metric)

        if lazy_render:
            return self._create_lazy_figure()

        if figsize is None:
            figsize = self.config.get_figsize('medium')

        # Default display_metric to rank_metric if not specified
        if not display_metric:
            display_metric = rank_metric

        # Determine if metrics are "higher is better"
        rank_higher_better = self.adapter.is_higher_better(rank_metric)
        display_higher_better = self.adapter.is_higher_better(display_metric)

        # Detect if partition or dataset_name is used as a grouping variable
        is_partition_grouped = (x_var == 'partition' or y_var == 'partition')
        is_dataset_grouped = (x_var == 'dataset_name' or y_var == 'dataset_name')

        # When partition or dataset is grouped, we need all data without filtering
        if is_partition_grouped or is_dataset_grouped:
            # Get ALL predictions without partition or dataset filtering
            all_filters = {k: v for k, v in filters.items() if k not in ['partition', 'dataset_name']}

            # Use filter_predictions to get raw data without partition-based ranking
            all_predictions = []
            df = self.predictions._storage.to_dataframe()

            # Apply only non-grouping variable filters
            # Also filter by metric to match display_metric
            if all_filters or display_metric:
                import polars as pl
                filter_conditions = [pl.col(k) == v for k, v in all_filters.items()]

                # Filter to only predictions with the requested metric
                if display_metric:
                    filter_conditions.append(pl.col('metric') == display_metric)

                filtered_df = df.filter(filter_conditions) if filter_conditions else df
            else:
                filtered_df = df

            # If no results with exact metric match, use stored metric
            if filtered_df.height == 0 and display_metric:
                # Fall back to all metrics and we'll use stored scores
                filter_conditions = [pl.col(k) == v for k, v in all_filters.items()]
                filtered_df = df.filter(filter_conditions) if filter_conditions else df
                # Get the actual metric from first row for title
                if filtered_df.height > 0:
                    actual_metric = filtered_df[0, 'metric']
                    if actual_metric != display_metric:
                        display_metric = actual_metric  # Update for accurate title

            # Convert to list of dicts and rank within each grouping
            all_predictions_raw = [row for row in filtered_df.to_dicts()]

            # Now we need to rank these by rank_metric on rank_partition
            # Group by model identity (excluding the grouping variable)
            from collections import defaultdict
            grouped = defaultdict(list)

            for pred in all_predictions_raw:
                # Create a key excluding the grouping variable
                if is_partition_grouped:
                    key = (pred.get('model_name'), pred.get('dataset_name'),
                          pred.get('config_name'), pred.get('preprocessings'), pred.get('fold_id'))
                elif is_dataset_grouped:
                    key = (pred.get('model_name'), pred.get('config_name'),
                          pred.get('preprocessings'), pred.get('fold_id'), pred.get('partition'))
                else:
                    key = (pred.get('model_name'), pred.get('dataset_name'),
                          pred.get('config_name'), pred.get('preprocessings'), pred.get('fold_id'))

                grouped[key].append(pred)

            # For each group, select the best based on rank_metric and rank_partition
            all_predictions = []
            for key, preds in grouped.items():
                # Find prediction from rank_partition
                rank_pred = None
                for p in preds:
                    if p.get('partition') == rank_partition:
                        rank_pred = p
                        break

                if rank_pred:
                    # Get rank score
                    rank_score_field = f'{rank_partition}_score'
                    rank_score = rank_pred.get(rank_score_field)

                    # Add all predictions from this group with rank score for sorting
                    for p in preds:
                        p_copy = dict(p)
                        p_copy['_rank_score'] = rank_score
                        all_predictions.append(p_copy)
                else:
                    # No rank partition found, include all
                    all_predictions.extend(preds)
        else:
            # Standard ranking approach when not grouping by partition/dataset
            rank_filters = {k: v for k, v in filters.items() if k not in ['partition', 'dataset_name']}

            # Only filter by partition if partition is not used as a grouping variable
            if not is_partition_grouped:
                rank_filters['partition'] = rank_partition

            rank_predictions = self.predictions.top(
                n=self.predictions.num_predictions,
                rank_metric=rank_metric,
                ascending=(not rank_higher_better),
                rank_partition=rank_partition,
                group_by_fold=True,  # Always True to get all folds for proper aggregation
                **rank_filters
            )

            if not rank_predictions:
                return self._create_empty_figure(
                    figsize,
                    f'No predictions found for rank_metric={rank_metric}, rank_partition={rank_partition}'
                )

            # If rank and display are the same, use the same data
            if rank_partition == display_partition and rank_metric == display_metric and not is_partition_grouped:
                all_predictions = rank_predictions
            else:
                # Get display predictions from display partition
                # Always group by fold to get all folds for proper matching
                display_filters = {k: v for k, v in filters.items() if k not in ['partition', 'dataset_name']}

                # Only filter by partition if partition is not used as a grouping variable
                if not is_partition_grouped:
                    display_filters['partition'] = display_partition

                display_predictions = self.predictions.top(
                    n=self.predictions.num_predictions,
                    rank_metric=display_metric,
                    ascending=(not display_higher_better),
                    rank_partition=display_partition,
                    group_by_fold=True,  # Always True to match with rank predictions
                    **display_filters
                )

                # Merge predictions by model identity
                all_predictions = self._merge_predictions(
                    rank_predictions, display_predictions,
                    rank_partition, display_partition
                )

        # Build score dictionary
        # When partition is grouped, we need to extract scores dynamically based on the partition
        if is_partition_grouped:
            # Special handling when partition is a grouping variable
            # We'll need to extract scores based on the partition value in each prediction
            # And use rank scores for proper aggregation
            score_dict = self.matrix_builder.build_score_dict_with_dynamic_partition(
                all_predictions, x_var, y_var, display_metric,
                use_rank_scores=True
            )
        elif is_dataset_grouped:
            # When dataset is grouped, use display scores with rank scores
            display_score_field = f'{display_partition}_score'
            rank_score_field = '_rank_score'

            score_dict = self.matrix_builder.build_score_dict(
                all_predictions, x_var, y_var,
                display_score_field, rank_score_field
            )
        else:
            display_score_field = f'{display_partition}_score'
            rank_score_field = '_rank_score' if rank_partition != display_partition else display_score_field

            score_dict = self.matrix_builder.build_score_dict(
                all_predictions, x_var, y_var,
                display_score_field, rank_score_field
            )

        if not score_dict:
            return self._create_empty_figure(
                figsize,
                f'No valid scores found for x_var={x_var}, y_var={y_var}'
            )

        # Build matrices
        y_labels, x_labels, matrix, count_matrix = self.matrix_builder.build_matrices(
            score_dict, aggregation, display_higher_better
        )

        # Detect if dataset_name is used as y_var for per-dataset normalization
        normalize_per_row = is_dataset_grouped and (y_var == 'dataset_name')

        # Normalize if requested
        normalized_matrix = self.normalizer.normalize(
            matrix, display_higher_better, per_row=normalize_per_row
        ) if normalize else matrix

        # Render the heatmap
        return self._render_heatmap(
            matrix, normalized_matrix, count_matrix, x_labels, y_labels,
            x_var, y_var, rank_metric, rank_partition, display_metric,
            display_partition, figsize, normalize, aggregation, show_counts
        )

    def _enrich_predictions_with_metric(
        self,
        predictions_list: List[Dict[str, Any]],
        metric: str,
        partitions: List[str] = ['train', 'val', 'test']
    ) -> List[Dict[str, Any]]:
        """Enrich predictions with computed metric scores if needed.

        Args:
            predictions_list: List of prediction dictionaries.
            metric: Metric to compute/extract.
            partitions: List of partitions to compute for.

        Returns:
            List of predictions with metric scores added to partition_score fields.
        """
        enriched = []
        for pred in predictions_list:
            pred_copy = dict(pred)
            stored_metric = pred.get('metric', '')

            # If stored metric matches requested metric, scores are already correct
            if stored_metric == metric:
                enriched.append(pred_copy)
                continue

            # Otherwise, compute metric from y_true/y_pred arrays for each partition
            for partition in partitions:
                score_field = f'{partition}_score'

                # Try to get arrays and compute metric
                try:
                    y_true_id = pred.get('y_true_id')
                    y_pred_id = pred.get('y_pred_id')

                    if y_true_id and y_pred_id:
                        y_true = self.predictions._storage.get_array(y_true_id)
                        y_pred = self.predictions._storage.get_array(y_pred_id)

                        if y_true is not None and y_pred is not None:
                            # Compute the requested metric
                            score = ModelUtils.compute_metric(y_true, y_pred, metric)
                            pred_copy[score_field] = score
                except Exception:
                    # If we can't compute, keep original or set None
                    pass

            enriched.append(pred_copy)

        return enriched

    def _merge_predictions(self, rank_predictions, display_predictions,
                          rank_partition, display_partition):
        """Merge ranking and display predictions by model identity.

        Args:
            rank_predictions: Predictions from rank partition.
            display_predictions: Predictions from display partition.
            rank_partition: Rank partition name.
            display_partition: Display partition name.

        Returns:
            List of merged predictions with both rank and display scores.
        """
        # Build lookup for display predictions
        display_lookup = {}
        for pred in display_predictions:
            key = self._get_model_key(pred)
            display_lookup[key] = pred

        # Merge predictions
        merged = []
        for rank_pred in rank_predictions:
            key = self._get_model_key(rank_pred)
            merged_pred = dict(rank_pred)

            # Add rank score
            rank_score_field = f'{rank_partition}_score'
            merged_pred['_rank_score'] = rank_pred.get(rank_score_field)

            # Add display score if available
            if key in display_lookup:
                display_pred = display_lookup[key]
                display_score_field = f'{display_partition}_score'
                merged_pred[display_score_field] = display_pred.get(display_score_field)

            merged.append(merged_pred)

        return merged

    def _get_model_key(self, pred):
        """Get unique key for model identity including fold.

        Args:
            pred: Prediction dictionary.

        Returns:
            Tuple of identifying fields including fold_id.
        """
        return (
            pred.get('model_name'),
            pred.get('dataset_name'),
            pred.get('config_name'),
            pred.get('preprocessings'),
            pred.get('fold_id')  # Include fold_id for proper matching
        )

    def _render_heatmap(self, matrix, normalized_matrix, count_matrix, x_labels, y_labels,
                       x_var, y_var, rank_metric, rank_partition, display_metric,
                       display_partition, figsize, normalize, aggregation, show_counts):
        """Render the final heatmap plot.

        Args:
            matrix: Original score matrix.
            normalized_matrix: Normalized matrix.
            count_matrix: Sample count matrix.
            x_labels: X-axis labels.
            y_labels: Y-axis labels.
            x_var: X variable name.
            y_var: Y variable name.
            rank_metric: Ranking metric.
            rank_partition: Ranking partition.
            display_metric: Display metric.
            display_partition: Display partition.
            figsize: Figure size.
            normalize: Whether scores are normalized.
            aggregation: Aggregation method.
            show_counts: Whether to show counts.

        Returns:
            matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Use RdYlGn colormap (Red-Yellow-Green)
        masked_matrix = np.ma.masked_invalid(normalized_matrix)
        im = ax.imshow(masked_matrix, cmap=self.config.heatmap_colormap, aspect='auto', vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar_label = f'Normalized {display_metric.upper()}\n(1=best, 0=worst)' if normalize else f'{display_metric.upper()} Score'
        cbar.set_label(cbar_label, fontsize=self.config.label_fontsize)

        # Set axis labels and ticks - Truncate long labels
        x_labels_display = [lbl[:25] + '...' if len(str(lbl)) > 25 else str(lbl) for lbl in x_labels]
        y_labels_display = [lbl[:25] + '...' if len(str(lbl)) > 25 else str(lbl) for lbl in y_labels]

        ax.set_xticks(range(len(x_labels)))
        ax.set_yticks(range(len(y_labels)))
        ax.set_xticklabels(x_labels_display, rotation=45, ha='right', fontsize=self.config.tick_fontsize)
        ax.set_yticklabels(y_labels_display, fontsize=self.config.tick_fontsize)
        ax.set_xlabel(x_var.replace('_', ' ').title(), fontsize=self.config.label_fontsize)
        ax.set_ylabel(y_var.replace('_', ' ').title(), fontsize=self.config.label_fontsize)

        # Create title
        title_parts = [f'{aggregation.title()} {display_metric.upper()}']
        if rank_partition != display_partition or rank_metric != display_metric:
            title_parts.append(f'(ranked by {rank_metric.upper()} on {rank_partition})')
        title_parts.append(f'[{display_partition}]')
        ax.set_title(' '.join(title_parts), fontsize=self.config.title_fontsize, pad=10)

        # Add text annotations to cells
        self.annotator.add_heatmap_annotations(
            ax, matrix, normalized_matrix, count_matrix,
            x_labels, y_labels, show_counts
        )

        plt.tight_layout()
        return fig

    def _create_lazy_figure(self) -> Figure:
        """Create placeholder for lazy rendering.

        Returns:
            matplotlib Figure with placeholder message.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Lazy rendering enabled\nFigure will be created on display',
                ha='center', va='center', fontsize=14)
        ax.set_title('Heatmap (Deferred)')
        ax.axis('off')
        return fig
