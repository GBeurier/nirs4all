"""
MatrixBuilder - Build matrices for heatmap visualizations.
"""
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from nirs4all.visualization.chart_utils.aggregator import DataAggregator


class MatrixBuilder:
    """Build matrices for heatmap visualizations.

    Handles grouping scores by variables and creating 2D matrices
    with support for different aggregation strategies.

    Optimized to work with PredictionResultsList from predictions.top().
    """

    @staticmethod
    def _natural_sort_key(text: str):
        """Generate natural sorting key for strings with numbers."""
        def convert(part):
            return int(part) if part.isdigit() else part.lower()
        return [convert(c) for c in re.split(r'(\d+)', str(text))]

    @staticmethod
    def build_score_dict(
        predictions_list,
        x_var: str,
        y_var: str,
        display_score_field: str,
        rank_field: Optional[str] = None
    ) -> Dict:
        """Group scores by x and y variables from PredictionResultsList.

        Args:
            predictions_list: List of prediction results.
            x_var: Variable name for x-axis grouping.
            y_var: Variable name for y-axis grouping.
            display_score_field: Field name for display scores.
            rank_field: Optional field name for ranking scores.

        Returns:
            Dict structure: {y_val: {x_val: [(display_score, rank_score), ...]}}
            or {y_val: {x_val: [score1, score2, ...]}} if no rank_field.
        """
        score_dict = defaultdict(lambda: defaultdict(list))

        for pred in predictions_list:
            x_val = pred.get(x_var)
            y_val = pred.get(y_var)

            if x_val is None or y_val is None:
                continue

            display_score = pred.get(display_score_field)
            if display_score is None:
                continue

            if rank_field and rank_field != display_score_field:
                rank_score = pred.get(rank_field)
                if rank_score is not None:
                    score_dict[y_val][x_val].append((float(display_score), float(rank_score)))
                else:
                    score_dict[y_val][x_val].append(float(display_score))
            else:
                score_dict[y_val][x_val].append(float(display_score))

        return score_dict

    @staticmethod
    def build_score_dict_with_dynamic_partition(
        predictions_list,
        x_var: str,
        y_var: str,
        metric: str,
        use_rank_scores: bool = False
    ) -> Dict:
        """Group scores by x and y variables when partition is one of the grouping variables.

        This method handles the special case where 'partition' is used as x_var or y_var.
        It extracts the score from the appropriate partition field based on the partition value.

        Args:
            predictions_list: List of prediction results.
            x_var: Variable name for x-axis grouping.
            y_var: Variable name for y-axis grouping.
            metric: Metric name to extract scores for.
            use_rank_scores: If True, include rank scores for proper aggregation.

        Returns:
            Dict structure: {y_val: {x_val: [score1, score2, ...]}} or
                           {y_val: {x_val: [(display_score, rank_score), ...]}}
        """
        score_dict = defaultdict(lambda: defaultdict(list))

        for pred in predictions_list:
            x_val = pred.get(x_var)
            y_val = pred.get(y_var)

            if x_val is None or y_val is None:
                continue

            # Determine which value is the partition
            partition_val = None
            if x_var == 'partition':
                partition_val = x_val
            elif y_var == 'partition':
                partition_val = y_val

            if partition_val is None:
                continue

            # Extract score from the appropriate partition field
            score_field = f'{partition_val}_score'
            score = pred.get(score_field)

            if score is not None:
                if use_rank_scores:
                    # Include rank score for proper aggregation
                    rank_score = pred.get('_rank_score')
                    if rank_score is not None:
                        score_dict[y_val][x_val].append((float(score), float(rank_score)))
                    else:
                        score_dict[y_val][x_val].append(float(score))
                else:
                    score_dict[y_val][x_val].append(float(score))

        return score_dict

    @staticmethod
    def build_matrices(
        score_dict: Dict,
        aggregation: str,
        higher_better: bool,
        natural_sort: bool = True
    ) -> Tuple[List, List, np.ndarray, np.ndarray]:
        """Build matrices from score dictionary.

        Args:
            score_dict: Dict of scores grouped by x and y variables.
                Can be {y: {x: [scores]}} or {y: {x: (score, count)}}.
            aggregation: Aggregation method ('best', 'mean', 'median', 'identity').
                Use 'identity' if scores are already aggregated tuples.
            higher_better: Whether higher values are better.
            natural_sort: Whether to use natural sorting for labels.

        Returns:
            Tuple of (y_labels, x_labels, score_matrix, count_matrix).
        """
        sort_key = MatrixBuilder._natural_sort_key if natural_sort else None

        y_labels = sorted(score_dict.keys(), key=sort_key)
        x_labels = sorted(
            set(x for y_data in score_dict.values() for x in y_data.keys()),
            key=sort_key
        )

        matrix = np.full((len(y_labels), len(x_labels)), np.nan)
        count_matrix = np.zeros((len(y_labels), len(x_labels)), dtype=int)

        for i, y_val in enumerate(y_labels):
            for j, x_val in enumerate(x_labels):
                cell_data = score_dict[y_val].get(x_val, [])

                if not cell_data:
                    continue

                # Check if cell_data is already aggregated (tuple of score, count)
                if isinstance(cell_data, tuple) and len(cell_data) == 2:
                    matrix[i, j] = cell_data[0]
                    count_matrix[i, j] = cell_data[1]
                # Otherwise, it's a list of scores to aggregate
                elif isinstance(cell_data, list) and cell_data:
                    matrix[i, j] = DataAggregator.aggregate(cell_data, aggregation, higher_better)
                    count_matrix[i, j] = len(cell_data)

        return y_labels, x_labels, matrix, count_matrix
