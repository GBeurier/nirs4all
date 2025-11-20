"""
PredictionsAdapter - Adapter for Predictions API with optimized data access.

Wraps the refactored Predictions API to provide convenient methods for charts.
"""
from typing import List, Optional
from nirs4all.data.predictions import PredictionResultsList


class PredictionsAdapter:
    """Adapter for Predictions API with optimized data access.

    Wraps the refactored Predictions API to provide convenient methods for charts.
    Leverages predictions.top(), lazy loading, and structured results.

    Key Optimizations:
    - Uses predictions.top() for efficient ranking
    - Supports lazy loading (load_arrays=False) for metadata-only queries
    - Works with PredictionResult/PredictionResultsList classes
    - Avoids redundant metric calculations

    Attributes:
        predictions: Predictions object instance.
    """

    def __init__(self, predictions):
        """Initialize adapter with predictions object.

        Args:
            predictions: Predictions object instance.
        """
        self.predictions = predictions

    def get_top_models(
        self,
        n: int,
        rank_metric: str,
        rank_partition: str = 'val',
        ascending: Optional[bool] = None,
        load_arrays: bool = True,
        **filters
    ) -> PredictionResultsList:
        """Get top N models using predictions.top() API.

        Args:
            n: Number of top models to retrieve.
            rank_metric: Metric to rank by.
            rank_partition: Partition to rank on (default: 'val').
            ascending: Sort order (None = auto-detect from metric).
            load_arrays: Whether to load prediction arrays (default: True).
            **filters: Additional filters (dataset_name, model_name, etc.).

        Returns:
            PredictionResultsList of top N models.
        """
        if ascending is None:
            ascending = not self.is_higher_better(rank_metric)

        return self.predictions.top(
            n=n,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            ascending=ascending,
            load_arrays=load_arrays,
            **filters
        )

    def get_all_predictions_metadata(
        self,
        rank_metric: str = 'rmse',
        rank_partition: str = 'test',
        **filters
    ) -> PredictionResultsList:
        """Get all predictions matching filters (metadata only, fast).

        Args:
            rank_metric: Metric for sorting (default: 'rmse').
            rank_partition: Partition for sorting (default: 'test').
            **filters: Filters to apply (dataset_name, model_name, etc.).

        Returns:
            PredictionResultsList with all matching predictions (no arrays loaded).
        """
        return self.predictions.top(
            n=self.predictions.num_predictions,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            load_arrays=False,
            **filters
        )

    def extract_metric_values(
        self,
        predictions_list: PredictionResultsList,
        metric: str,
        partition: str = 'test'
    ) -> List[float]:
        """Extract metric values from prediction results.

        Args:
            predictions_list: List of prediction results.
            metric: Metric name to extract.
            partition: Partition to extract from (default: 'test').

        Returns:
            List of metric values.
        """
        values = []
        for pred in predictions_list:
            try:
                # Try to get the score from the partition-specific field
                score_field = f'{partition}_score'
                if score_field in pred:
                    values.append(float(pred[score_field]))
                elif 'metrics' in pred and metric in pred['metrics']:
                    values.append(float(pred['metrics'][metric]))
                elif metric in pred:
                    values.append(float(pred[metric]))
            except (KeyError, TypeError, ValueError):
                continue
        return values

    @staticmethod
    def is_higher_better(metric: str) -> bool:
        """Check if metric is higher-is-better.

        Args:
            metric: Metric name.

        Returns:
            True if higher is better, False otherwise.
        """
        return metric.lower() in ['r2', 'accuracy', 'f1', 'precision', 'recall', 'auc']
