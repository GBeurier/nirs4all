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
from matplotlib.figure import Figure
from typing import Optional, List, Dict, Any
from collections import defaultdict
from nirs4all.visualization.charts.base import BaseChart
from nirs4all.visualization.chart_utils.normalizer import ScoreNormalizer
from nirs4all.visualization.chart_utils.annotator import ChartAnnotator
from nirs4all.visualization.chart_utils.matrix_builder import MatrixBuilder
from nirs4all.visualization.chart_utils.aggregator import DataAggregator
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
        self.aggregator = DataAggregator()

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
        """Render performance heatmap.

        Logic:
        1. Rank predictions by rank_metric on rank_partition using rank_agg
        2. Display display_metric from display_partition using display_agg
        3. Aggregate multiple values per cell
        4. Normalize per dataset if requested
        5. Color cells based on normalized scores

        Args:
            x_var: Variable for x-axis (e.g., 'model_name', 'preprocessings').
            y_var: Variable for y-axis (e.g., 'dataset_name', 'partition').
            rank_metric: Metric used to rank/select models (default: auto-detect from task type).
            rank_partition: Partition used for ranking models (default: 'val').
            display_metric: Metric to display in heatmap (default: same as rank_metric).
            display_partition: Partition to display scores from (default: 'test').
            figsize: Figure size tuple (default: from config).
            normalize: If True, show normalized scores in cells (default: False).
            rank_agg: Aggregation for ranking ('best', 'worst', 'mean', 'median') (default: 'best').
            display_agg: Aggregation for display ('best', 'worst', 'mean', 'median') (default: 'mean').
            show_counts: Show prediction counts in cells (default: True).
            local_scale: If True, colorbar shows actual metric values; if False, shows 0-1 normalized (default: False).
            **filters: Additional filters (dataset_name, model_name, etc.).

        Returns:
            matplotlib Figure object.
        """
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

        # Remove grouping variables from filters if they are used for grouping
        all_filters = dict(filters)
        if is_partition_grouped:
            all_filters.pop('partition', None)
        if is_dataset_grouped:
            all_filters.pop('dataset_name', None)

        # Remove internal parameters that aren't filters
        all_filters.pop('aggregation', None)  # Backward compatibility
        all_filters.pop('rank_agg', None)
        all_filters.pop('display_agg', None)
        all_filters.pop('show_counts', None)
        all_filters.pop('figsize', None)

        # Get all predictions with metrics computed
        # Use a high n to get all predictions
        all_preds = self.predictions.top(
            n=self.predictions.num_predictions,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            ascending=not self._is_higher_better(rank_metric),
            group_by_fold=True,
            load_arrays=True,
            aggregate_partitions=True,  # Get all partitions data
            **all_filters
        )

        if not all_preds:
            raise ValueError(f"No predictions found with filters: {all_filters}")

        # Build score dictionary
        score_dict = self._build_score_dict(
            all_preds, x_var, y_var,
            rank_metric, rank_partition, rank_agg,
            display_metric, display_partition, display_agg,
            is_partition_grouped
        )

        if not score_dict:
            raise ValueError(f"No scores found for {x_var} vs {y_var}")

        # Build matrices
        display_higher_better = self._is_higher_better(display_metric)
        y_labels, x_labels, matrix, count_matrix = self.matrix_builder.build_matrices(
            score_dict, 'identity', display_higher_better  # 'identity' because we already aggregated
        )

        # Normalize for colors (always per dataset if dataset is grouped as y_var)
        normalize_per_row = is_dataset_grouped and (y_var == 'dataset_name')
        normalized_matrix = self.normalizer.normalize(
            matrix, display_higher_better, per_row=normalize_per_row
        )

        # Render
        return self._render_heatmap(
            matrix, normalized_matrix, count_matrix,
            x_labels, y_labels, x_var, y_var,
            rank_metric, rank_partition, rank_agg,
            display_metric, display_partition, display_agg,
            figsize, normalize, show_counts, local_scale, display_higher_better
        )

    def _build_score_dict(
        self,
        predictions: List[Dict[str, Any]],
        x_var: str,
        y_var: str,
        rank_metric: str,
        rank_partition: str,
        rank_agg: str,
        display_metric: str,
        display_partition: str,
        display_agg: str,
        is_partition_grouped: bool
    ) -> Dict:
        """Build score dictionary grouped by (x_var, y_var).

        For each cell (x_val, y_val):
        1. Get all predictions matching this cell
        2. Extract rank_scores from rank_partition
        3. Extract display_scores from display_partition
        4. Aggregate both using rank_agg and display_agg
        5. Store aggregated display_score in the cell

        Args:
            predictions: List of prediction results from top().
            x_var: X-axis variable.
            y_var: Y-axis variable.
            rank_metric: Metric for ranking.
            rank_partition: Partition for ranking.
            rank_agg: Aggregation method for ranking.
            display_metric: Metric for display.
            display_partition: Partition for display.
            display_agg: Aggregation method for display.
            is_partition_grouped: Whether partition is used as grouping variable.

        Returns:
            Dict structure: {y_val: {x_val: (aggregated_display_score, count)}}
        """
        # Group predictions by (x_var, y_var)
        groups = defaultdict(lambda: defaultdict(list))

        for pred in predictions:
            x_val = pred.get(x_var)
            y_val = pred.get(y_var)

            if x_val is None or y_val is None:
                continue

            # Store the full prediction for later processing
            groups[y_val][x_val].append(pred)

        # Now aggregate each group
        score_dict = defaultdict(lambda: defaultdict(list))
        rank_higher_better = self._is_higher_better(rank_metric)
        display_higher_better = self._is_higher_better(display_metric)

        for y_val, x_dict in groups.items():
            for x_val, preds in x_dict.items():
                # Extract rank and display scores from each prediction
                rank_scores = []
                display_scores = []

                for pred in preds:
                    # Get rank score
                    rank_score = self._extract_score(
                        pred, rank_metric, rank_partition, is_partition_grouped
                    )
                    # Get display score
                    display_score = self._extract_score(
                        pred, display_metric, display_partition, is_partition_grouped
                    )

                    if rank_score is not None and display_score is not None:
                        rank_scores.append(rank_score)
                        display_scores.append(display_score)

                if not rank_scores or not display_scores:
                    continue

                # Aggregate rank scores to determine which predictions to keep
                if rank_agg == 'best':
                    if rank_higher_better:
                        best_idx = np.argmax(rank_scores)
                    else:
                        best_idx = np.argmin(rank_scores)
                    # Use the display score from the best ranked prediction
                    agg_display_score = display_scores[best_idx]
                elif rank_agg == 'worst':
                    if rank_higher_better:
                        worst_idx = np.argmin(rank_scores)
                    else:
                        worst_idx = np.argmax(rank_scores)
                    agg_display_score = display_scores[worst_idx]
                else:
                    # For mean/median, aggregate display scores directly
                    agg_display_score = self.aggregator.aggregate(
                        display_scores, display_agg, display_higher_better
                    )

                # Store aggregated display score and count
                score_dict[y_val][x_val] = (float(agg_display_score), len(preds))

        return score_dict

    def _extract_score(
        self,
        pred: Dict[str, Any],
        metric: str,
        partition: str,
        is_partition_grouped: bool
    ) -> Optional[float]:
        """Extract score from prediction for given metric and partition.

        Args:
            pred: Prediction dictionary.
            metric: Metric name.
            partition: Partition name.
            is_partition_grouped: Whether partition is a grouping variable.

        Returns:
            Score value or None if not found.
        """
        # If partition is grouped, extract from the appropriate partition
        if is_partition_grouped:
            # The partition value is in x_var or y_var
            partition = pred.get('partition', partition)

        # Try to get from partitions nested dict (from aggregate_partitions=True)
        partitions = pred.get('partitions', {})
        if partitions and partition in partitions:
            partition_data = partitions[partition]

            # Use centralized score extraction
            score = self._get_score(partition_data, metric)
            if score is not None:
                return score

        # Fallback: try direct score fields (for single partition case)
        score_field = f'{partition}_score'
        score = pred.get(score_field)

        # Check if stored metric matches
        stored_metric = pred.get('metric', '')
        if stored_metric == metric and score is not None:
            return float(score)

        # Try to convert between compatible metrics
        if score is not None:
            converted = self._convert_metric(score, stored_metric, metric)
            if converted is not None:
                return float(converted)

        # Last resort: use centralized score extraction (checks key or computes from arrays)
        return self._get_score(pred, metric)

    def _convert_metric(self, score: float, from_metric: str, to_metric: str) -> Optional[float]:
        """Convert between compatible metrics."""
        if from_metric == to_metric:
            return score

        # MSE <-> RMSE conversion
        if from_metric == 'mse' and to_metric == 'rmse':
            return np.sqrt(score)
        elif from_metric == 'rmse' and to_metric == 'mse':
            return score ** 2

        return None

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
