"""
Ranking and top-k selection for predictions.

This module provides the PredictionRanker class for ranking predictions
by metrics and selecting top-performing models.

Performance optimizations (v0.4.2):
    - AggregationCache: Caches aggregated y_true/y_pred/y_proba per (prediction_id, aggregate_key)
    - ScoreCache: Caches computed metrics per (prediction_id, aggregate_key, metric, partition)
    - Avoids redundant array loading and metric recalculation
"""

import json
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import polars as pl

from nirs4all.core import metrics as evaluator
from nirs4all.data.ensemble_utils import EnsembleUtils

from .storage import PredictionStorage
from .serializer import PredictionSerializer
from .indexer import PredictionIndexer
from .result import PredictionResult, PredictionResultsList


class AggregationCache:
    """
    LRU cache for aggregated prediction arrays.

    Caches aggregated (y_true, y_pred, y_proba) per (prediction_id, aggregate_key)
    to avoid redundant aggregation computations.

    This is the main source of slowness - aggregating arrays is O(n) per prediction
    and involves numpy operations. By caching, we only pay this cost once.
    """

    def __init__(self, max_entries: int = 5000):
        """Initialize cache with max entries."""
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._access_order: List[Tuple[str, str]] = []
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0

    def get(self, pred_id: str, aggregate_key: str) -> Optional[Dict[str, Any]]:
        """Get cached aggregation result."""
        key = (pred_id, aggregate_key)
        if key in self._cache:
            self._hits += 1
            # Move to end (LRU)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, pred_id: str, aggregate_key: str, result: Dict[str, Any]) -> None:
        """Store aggregation result."""
        key = (pred_id, aggregate_key)
        # Evict if needed
        while len(self._cache) >= self._max_entries and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

        self._cache[key] = result
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total > 0 else 0,
            'size': len(self._cache),
            'max_size': self._max_entries
        }


class ScoreCache:
    """
    Cache for computed metric scores.

    Caches scores per (prediction_id, aggregate_key, metric, partition).
    Once a metric is computed for a prediction, it's stored here and never recomputed.
    """

    def __init__(self, max_entries: int = 50000):
        """Initialize cache with max entries."""
        self._cache: Dict[Tuple[str, str, str, str], float] = {}
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0

    def get(self, pred_id: str, aggregate_key: str, metric: str, partition: str) -> Optional[float]:
        """Get cached score."""
        key = (pred_id, aggregate_key or '', metric, partition)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, pred_id: str, aggregate_key: str, metric: str, partition: str, score: float) -> None:
        """Store score."""
        if len(self._cache) >= self._max_entries:
            # Simple eviction: clear half the cache
            keys = list(self._cache.keys())[:len(self._cache) // 2]
            for k in keys:
                del self._cache[k]

        key = (pred_id, aggregate_key or '', metric, partition)
        self._cache[key] = score

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total > 0 else 0,
            'size': len(self._cache),
            'max_size': self._max_entries
        }


class PredictionRanker:
    """
    Handles ranking predictions by metrics.

    Features:
        - Top-k selection by arbitrary metrics
        - Partition-aware ranking (train/val/test)
        - Fold-grouped or cross-fold ranking
        - Ascending/descending sort orders

    Examples:
        >>> storage = PredictionStorage()
        >>> serializer = PredictionSerializer()
        >>> indexer = PredictionIndexer(storage)
        >>> ranker = PredictionRanker(storage, serializer, indexer)
        >>> top_5 = ranker.top(n=5, rank_metric="rmse", rank_partition="val")
        >>> best = ranker.get_best(metric="r2", ascending=False)

    Attributes:
        _storage: PredictionStorage instance
        _serializer: PredictionSerializer instance
        _indexer: PredictionIndexer instance
    """

    def __init__(
        self,
        storage: PredictionStorage,
        serializer: PredictionSerializer,
        indexer: PredictionIndexer
    ):
        """
        Initialize ranker with dependencies.

        Args:
            storage: PredictionStorage instance
            serializer: PredictionSerializer instance
            indexer: PredictionIndexer instance
        """
        self._storage = storage
        self._serializer = serializer
        self._indexer = indexer

        # Performance caches - avoid redundant computation
        self._aggregation_cache = AggregationCache(max_entries=5000)
        self._score_cache = ScoreCache(max_entries=50000)

    def clear_caches(self) -> None:
        """Clear all internal caches. Call when predictions data changes."""
        self._aggregation_cache.clear()
        self._score_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for debugging performance."""
        return {
            'aggregation_cache': self._aggregation_cache.stats(),
            'score_cache': self._score_cache.stats()
        }

    def _parse_vec_json(self, s: str) -> np.ndarray:
        """Parse JSON string to numpy array."""
        return np.asarray(json.loads(s), dtype=float)

    def _get_array(self, row: Dict[str, Any], field_name: str) -> Optional[np.ndarray]:
        """
        Get array from row, handling both legacy and registry formats.

        Args:
            row: Row dictionary
            field_name: Name of array field (e.g., 'y_true', 'y_pred')

        Returns:
            Numpy array or None if not found
        """
        # Try array registry format first (new format)
        array_id_field = f"{field_name}_id"
        if array_id_field in row and row[array_id_field] is not None:
            try:
                return self._storage._array_registry.get_array(row[array_id_field])
            except (KeyError, AttributeError):
                pass

        # Fall back to legacy format (JSON string)
        if field_name in row and row[field_name] is not None:
            try:
                return self._parse_vec_json(row[field_name])
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    def _get_fallback_metadata_for_avg_fold(
        self,
        row: Dict[str, Any],
        aggregate: str,
        partition: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata from a corresponding regular fold when avg/w_avg fold has empty metadata.

        For avg and w_avg folds that were created without metadata, this method looks up
        metadata from a corresponding regular fold (e.g., fold '0') that should have the
        same sample structure.

        Args:
            row: Current prediction row (avg/w_avg fold)
            aggregate: Aggregation column to look for (e.g., 'ID')
            partition: Partition name (train/val/test)

        Returns:
            Metadata dictionary with the aggregate column, or None if not found.
        """
        try:
            # Get identifying info from current row
            model_name = row.get("model_name", "")
            config_name = row.get("config_name", "")
            dataset_name = row.get("dataset_name", "")

            # Query for a regular fold (not avg/w_avg) with the same model/config/dataset
            df = self._storage._df
            regular_fold = df.filter(
                (pl.col("model_name") == model_name) &
                (pl.col("config_name") == config_name) &
                (pl.col("dataset_name") == dataset_name) &
                (pl.col("partition") == partition) &
                (~pl.col("fold_id").is_in(["avg", "w_avg"]))
            ).head(1)

            if regular_fold.height > 0:
                fold_row = regular_fold.to_dicts()[0]
                metadata_json = fold_row.get("metadata", "{}")
                metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json

                if aggregate in metadata:
                    return metadata
        except Exception:
            pass

        return None

    def _apply_aggregation(
        self,
        y_true: Optional[np.ndarray],
        y_pred: Optional[np.ndarray],
        y_proba: Optional[np.ndarray],
        metadata: Dict[str, Any],
        aggregate: str,
        model_name: str = "",
        pred_id: str = "",
        row: Optional[Dict[str, Any]] = None,
        partition: str = "test",
        method: Optional[str] = None,
        exclude_outliers: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Apply aggregation to predictions by a group column.

        Uses internal cache to avoid redundant aggregation computations.

        Args:
            y_true: True values array
            y_pred: Predicted values array
            y_proba: Optional class probabilities array
            metadata: Metadata dictionary containing group column
            aggregate: Group column name or 'y' to group by y_true
            model_name: Model name for warning messages
            pred_id: Prediction ID for caching (optional but recommended)
            row: Optional full row dict for fallback metadata lookup (for avg/w_avg folds)
            partition: Partition name for fallback lookup (default: 'test')
            method: Aggregation method ('mean', 'median', 'vote'). Default is 'mean'.
            exclude_outliers: If True, exclude outliers using T² statistic before aggregation.

        Returns:
            Tuple of (aggregated_y_true, aggregated_y_pred, aggregated_y_proba, was_aggregated)
            The was_aggregated flag is True only if aggregation was actually applied.
        """
        from nirs4all.data.predictions import Predictions

        if y_pred is None:
            return y_true, y_pred, y_proba, False

        # Check cache first if we have a prediction ID
        if pred_id:
            cached = self._aggregation_cache.get(pred_id, aggregate)
            if cached is not None:
                return (
                    cached.get('y_true'),
                    cached.get('y_pred'),
                    cached.get('y_proba'),
                    cached.get('was_aggregated', True)
                )

        # Determine group IDs
        if aggregate == 'y':
            if y_true is None:
                return y_true, y_pred, y_proba, False
            group_ids = y_true
        else:
            # Get group IDs from metadata
            effective_metadata = metadata

            # For avg/w_avg folds with empty metadata, try to get from a regular fold
            if aggregate not in metadata and row is not None:
                fold_id = row.get("fold_id", "")
                if fold_id in ["avg", "w_avg"]:
                    fallback_metadata = self._get_fallback_metadata_for_avg_fold(
                        row, aggregate, partition
                    )
                    if fallback_metadata is not None:
                        effective_metadata = fallback_metadata

            if aggregate not in effective_metadata:
                if model_name:
                    warnings.warn(
                        f"Aggregation column '{aggregate}' not found in metadata for model '{model_name}'. "
                        f"Available columns: {list(metadata.keys())}. Skipping aggregation for this model.",
                        UserWarning
                    )
                # Cache the failure to avoid repeated warnings
                if pred_id:
                    self._aggregation_cache.put(pred_id, aggregate, {
                        'y_true': y_true,
                        'y_pred': y_pred,
                        'y_proba': y_proba,
                        'was_aggregated': False
                    })
                return y_true, y_pred, y_proba, False
            group_ids = np.asarray(effective_metadata[aggregate])

        if len(group_ids) != len(y_pred):
            if model_name:
                warnings.warn(
                    f"Aggregation column '{aggregate}' length ({len(group_ids)}) doesn't match "
                    f"predictions length ({len(y_pred)}) for model '{model_name}'. Skipping aggregation.",
                    UserWarning
                )
            # Cache the failure
            if pred_id:
                self._aggregation_cache.put(pred_id, aggregate, {
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    'was_aggregated': False
                })
            return y_true, y_pred, y_proba, False

        # Apply aggregation
        result = Predictions.aggregate(
            y_pred=y_pred,
            group_ids=group_ids,
            y_proba=y_proba,
            y_true=y_true,
            method=method,
            exclude_outliers=exclude_outliers
        )

        agg_result = (
            result.get('y_true'),
            result.get('y_pred'),
            result.get('y_proba'),
            True  # Aggregation was successfully applied
        )

        # Cache the result
        if pred_id:
            self._aggregation_cache.put(pred_id, aggregate, {
                'y_true': agg_result[0],
                'y_pred': agg_result[1],
                'y_proba': agg_result[2],
                'was_aggregated': True
            })

        return agg_result

    def _get_score_cached(
        self,
        row: Dict[str, Any],
        metric: str,
        partition: str,
        aggregate: Optional[str] = None,
        load_arrays: bool = True,
        aggregate_method: Optional[str] = None,
        aggregate_exclude_outliers: bool = False
    ) -> Optional[float]:
        """
        Get score for a prediction, using cache when possible.

        This is the main performance optimization. It:
        1. Checks the score cache first
        2. If not cached, computes and caches the score
        3. Handles both aggregated and non-aggregated cases

        Args:
            row: Row dictionary from prediction storage
            metric: Metric name to compute
            partition: Partition name
            aggregate: Aggregation column name (optional)
            load_arrays: Whether to load arrays if needed
            aggregate_method: Aggregation method ('mean', 'median', 'vote'). Default is 'mean'.
            aggregate_exclude_outliers: If True, exclude outliers using T² statistic before aggregation.

        Returns:
            Score value or None
        """
        pred_id = row.get("id", "")
        aggregate_key = aggregate or ""

        # Check score cache first
        cached_score = self._score_cache.get(pred_id, aggregate_key, metric, partition)
        if cached_score is not None:
            return cached_score

        score = None

        if aggregate:
            # Aggregated path - must compute from arrays
            try:
                y_true = self._get_array(row, "y_true")
                y_pred = self._get_array(row, "y_pred")
                y_proba = self._get_array(row, "y_proba")

                if y_true is not None and y_pred is not None:
                    metadata = json.loads(row.get("metadata", "{}"))
                    model_name = row.get("model_name", "")

                    agg_y_true, agg_y_pred, _, was_aggregated = self._apply_aggregation(
                        y_true, y_pred, y_proba, metadata, aggregate, model_name, pred_id,
                        row=row, partition=partition, method=aggregate_method,
                        exclude_outliers=aggregate_exclude_outliers
                    )

                    if agg_y_true is not None and agg_y_pred is not None:
                        score = evaluator.eval(agg_y_true, agg_y_pred, metric)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        else:
            # Non-aggregated path - try pre-computed scores first
            scores_json = row.get("scores")
            if scores_json:
                try:
                    scores_dict = json.loads(scores_json)
                    if partition in scores_dict and metric in scores_dict[partition]:
                        score = scores_dict[partition][metric]
                except (json.JSONDecodeError, TypeError):
                    pass

            # Fallback to legacy field
            if score is None and metric == row.get("metric"):
                score = row.get(f"{partition}_score")

            # Last resort: compute from arrays
            if score is None and load_arrays:
                try:
                    y_true = self._get_array(row, "y_true")
                    y_pred = self._get_array(row, "y_pred")
                    if y_true is not None and y_pred is not None:
                        score = evaluator.eval(y_true, y_pred, metric)
                except (ValueError, TypeError):
                    pass

        # Cache the computed score
        if score is not None:
            self._score_cache.put(pred_id, aggregate_key, metric, partition, score)

        return score

    def _get_precomputed_score(
        self,
        row: Dict[str, Any],
        metric: str,
        partition: str
    ) -> Optional[float]:
        """
        Get pre-computed score from row, using scores JSON or legacy fields.

        Priority:
        1. scores JSON: {"val": {"rmse": 0.5}}
        2. Legacy field: val_score (if metric matches row's metric)

        Args:
            row: Row dictionary from prediction storage
            metric: Metric name to retrieve (e.g., 'rmse', 'r2')
            partition: Partition name (e.g., 'val', 'test')

        Returns:
            Pre-computed score or None if not found
        """
        # Try scores JSON first (new format)
        scores_json = row.get("scores")
        if scores_json:
            try:
                scores_dict = json.loads(scores_json)
                if partition in scores_dict and metric in scores_dict[partition]:
                    return scores_dict[partition][metric]
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback to legacy field
        if metric == row.get("metric"):
            return row.get(f"{partition}_score")

        return None

    def _make_group_key(
        self,
        row: Dict[str, Any],
        group_by: List[str]
    ) -> Tuple:
        """
        Create hashable group key from row values.

        Handles:
        - Case-insensitive comparison for string columns (model_name, model_classname, etc.)
        - None/missing values (treated as a single group)
        - Numeric columns (kept as-is for proper grouping)
        - List values (converted to tuple for hashability)

        Args:
            row: Row dictionary
            group_by: List of column names to group by

        Returns:
            Tuple of values for the group key
        """
        # Columns that should be compared case-insensitively
        case_insensitive_cols = {
            'model_name', 'model_classname', 'preprocessings',
            'dataset_name', 'config_name'
        }

        key_parts = []
        for col in group_by:
            val = row.get(col)

            # Handle None/missing values
            if val is None:
                key_parts.append(None)
                continue

            # Case-insensitive for string columns
            if isinstance(val, str) and col in case_insensitive_cols:
                val = val.lower()
            # Convert lists to tuples for hashability
            elif isinstance(val, list):
                val = tuple(val)
            # Numeric columns (int, float) kept as-is
            # This ensures proper grouping by fold_id, step_idx, etc.

            key_parts.append(val)
        return tuple(key_parts)

    def top(
        self,
        n: int,
        rank_metric: str = "",
        rank_partition: str = "val",
        display_metrics: Optional[List[str]] = None,
        display_partition: str = "test",
        aggregate_partitions: bool = False,
        ascending: Optional[bool] = None,
        group_by_fold: bool = False,
        load_arrays: bool = True,
        aggregate: Optional[str] = None,
        aggregate_method: Optional[str] = None,
        aggregate_exclude_outliers: bool = False,
        group_by: Optional[Union[str, List[str]]] = None,
        best_per_model: bool = False,
        **filters
    ) -> PredictionResultsList:
        """
        Get top n models ranked by a metric on a specific partition.

        Ranks models by performance on `rank_partition`, then returns their data
        from `display_partition`. Useful for validation-based selection with test set evaluation.

        Args:
            n: Number of top models to return
            rank_metric: Metric to rank by (if empty, uses record's metric or val_score)
            rank_partition: Partition to rank on (default: "val")
            display_metrics: Metrics to compute for display (default: task_type defaults)
            display_partition: Partition to display results from (default: "test")
            aggregate_partitions: If True, add train/val/test nested dicts in results
            ascending: Sort order. If True, sorts ascending (lower is better).
                      If False, sorts descending (higher is better).
                      If None, infers from metric (RMSE->True, Accuracy->False).
            group_by_fold: If True, include fold_id in model identity (rank per fold)
            aggregate: If provided, aggregate predictions by this metadata column or 'y'.
                      When 'y', groups by y_true values.
                      When a column name (e.g., 'ID'), groups by that metadata column.
                      Aggregated predictions have recalculated metrics.
            aggregate_method: Aggregation method for combining predictions.
                      'mean' (default), 'median', or 'vote' (for classification).
            aggregate_exclude_outliers: If True, exclude outliers using T² statistic
                      before aggregation (default: False).
            group_by: Group predictions and keep only the best per group.
                     Can be a single column name (str) or list of columns.
                     Examples: 'model_name', ['model_name', 'preprocessings']
                     The global sort order is preserved - first occurrence per group is kept.
            best_per_model: DEPRECATED - Use group_by=['model_name'] instead.
                           If True, keep only the best prediction per model_name.
            **filters: Additional filter criteria (dataset_name, config_name, etc.)

        Returns:
            PredictionResultsList containing top n models, sorted by rank_metric.
            Each result includes data from display_partition (or all partitions if aggregate=True).

        Raises:
            ValueError: If rank_partition or display_partition is invalid

        Examples:
            >>> # Get top 5 models by validation RMSE, show test results
            >>> top_5 = ranker.top(
            ...     n=5,
            ...     rank_metric="rmse",
            ...     rank_partition="val",
            ...     display_partition="test",
            ...     dataset_name="wheat"
            ... )
            >>>
            >>> # Get top 3 models with aggregated partitions
            >>> top_3_agg = ranker.top(
            ...     n=3,
            ...     rank_metric="r2",
            ...     rank_partition="val",
            ...     aggregate_partitions=True,
            ...     ascending=False  # Higher R² is better
            ... )
            >>>
            >>> # Get top 5 with predictions aggregated by sample ID
            >>> # Useful when multiple scans per sample (e.g., 4 scans averaged)
            >>> top_5_by_id = ranker.top(
            ...     n=5,
            ...     rank_metric="rmse",
            ...     aggregate='ID'  # Aggregate by metadata 'ID' column
            ... )
            >>>
            >>> # Get best prediction per model (for TopK charts)
            >>> one_per_model = ranker.top(
            ...     n=10,
            ...     rank_metric="rmse",
            ...     group_by=["model_name"]  # One result per unique model_name
            ... )
            >>>
            >>> # Get best prediction per model class (e.g., PLSRegression, SVR)
            >>> one_per_class = ranker.top(
            ...     n=10,
            ...     rank_metric="rmse",
            ...     group_by=["model_classname"]  # One result per model class
            ... )
            >>>
            >>> # Multi-column grouping for heatmaps
            >>> # Get best per (model_name, fold_id) combination
            >>> per_model_fold = ranker.top(
            ...     n=100,
            ...     rank_metric="rmse",
            ...     group_by=["model_name", "fold_id"]
            ... )

        Notes:
            - If rank_metric matches the stored metric in the data, uses precomputed scores
            - Otherwise, recomputes metric from y_true/y_pred arrays
            - ascending=True means lower scores rank higher (good for RMSE, MAE)
            - ascending=False means higher scores rank higher (good for R², accuracy)
            - group_by preserves global ranking order: sorts first, then takes first per group
            - aggregate is applied BEFORE ranking to ensure correct score calculation
        """
        # Handle display_partition as list (backward compat)
        if isinstance(display_partition, list):
            aggregate_partitions = True
            display_partition = "test"  # Reset to default for non-aggregate sections

        # Handle display_metrics as single string (backward compat)
        if isinstance(display_metrics, str):
            display_metrics = [display_metrics]

        # Remove non-filter parameters that might have been passed
        # (for backward compatibility and to prevent polars errors)
        _ = filters.pop("partition", None)
        _ = filters.pop("load_arrays", None)

        # Handle display_metric (singular) for backward compatibility
        if "display_metric" in filters:
            passed_display_metric = filters.pop("display_metric")
            if isinstance(passed_display_metric, list):
                display_metrics = passed_display_metric
            elif isinstance(passed_display_metric, str):
                display_metrics = [passed_display_metric]

        # Handle display_partition as list (backward compat)
        if "display_partition" in filters:
            passed_display_partition = filters.pop("display_partition")
            if isinstance(passed_display_partition, list):
                # If list passed, use aggregate mode
                aggregate_partitions = True
                # Keep display_partition as default "test" for non-aggregate sections
            else:
                display_partition = passed_display_partition

        # Handle display_metrics variations
        if "display_metrics" in filters:
            passed_display_metrics = filters.pop("display_metrics")
            if isinstance(passed_display_metrics, list):
                display_metrics = passed_display_metrics

        _ = filters.pop("rank_metric", None)  # Already a parameter
        _ = filters.pop("rank_partition", None)  # Already a parameter
        _ = filters.pop("aggregate_partitions", None)  # Already a parameter
        _ = filters.pop("ascending", None)  # Already a parameter
        _ = filters.pop("group_by_fold", None)  # Already a parameter
        _ = filters.pop("higher_is_better", None)  # Not a data column
        _ = filters.pop("aggregate", None)  # Already a parameter

        df = self._storage.to_dataframe()
        base = df.filter([pl.col(k) == v for k, v in filters.items()]) if filters else df

        if base.height == 0:
            return PredictionResultsList([])

        # Check if rank_partition and display_partition exist in data
        available_partitions = set(base["partition"].unique().to_list())
        if rank_partition and rank_partition not in available_partitions:
            warnings.warn(
                f"rank_partition '{rank_partition}' not found in data. "
                f"Available partitions: {sorted(available_partitions)}.",
                UserWarning
            )
        if display_partition and display_partition not in available_partitions:
            warnings.warn(
                f"display_partition '{display_partition}' not found in data. "
                f"Available partitions: {sorted(available_partitions)}.",
                UserWarning
            )

        # Default rank_metric from data if not provided
        if rank_metric == "":
            rank_metric = base[0, "metric"]

        # Adjust ascending based on metric direction if not specified
        if ascending is None:
            if EnsembleUtils._is_higher_better(rank_metric):
                ascending = False  # Higher is better -> Sort Descending
            else:
                ascending = True   # Lower is better -> Sort Ascending

        # Model identity key
        KEY = ["dataset_name", "config_name", "step_idx", "model_name"]
        if group_by_fold:
            KEY.append("fold_id")

        # 1) RANKING: Filter to rank_partition and compute scores
        rank_data = base.filter(pl.col("partition") == rank_partition)
        if rank_data.height == 0:
            return PredictionResultsList([])

        # Compute rank scores using cached method
        # CRITICAL: When aggregate is provided, we MUST aggregate predictions first
        # and then recalculate metrics on the aggregated data for ranking.
        # This ensures consistent ranking across all visualizations.
        rank_scores = []
        for row in rank_data.to_dicts():
            scores_json = row.get("scores")

            # Use cached scoring method - handles both aggregated and non-aggregated
            score = self._get_score_cached(
                row, rank_metric, rank_partition, aggregate, load_arrays,
                aggregate_method=aggregate_method,
                aggregate_exclude_outliers=aggregate_exclude_outliers
            )

            # Include additional columns that might be used for group_by filtering
            # These columns are commonly used in visualizations for grouping
            extra_cols = ["model_classname", "preprocessings", "pipeline_uid", "task_type"]
            rank_scores.append({
                **{k: row[k] for k in KEY},
                **{k: row.get(k) for k in extra_cols if k in row},
                "rank_score": score,
                "id": row["id"],
                "fold_id": row["fold_id"],
                "scores": scores_json  # Pass scores along
            })

        # Get tiebreaker scores (test partition) for models with equal rank scores
        # This ensures consistent ranking when val scores are equal
        tiebreaker_partition = "test" if rank_partition == "val" else "val"
        tiebreaker_data = base.filter(pl.col("partition") == tiebreaker_partition)
        tiebreaker_scores = {}
        for row in tiebreaker_data.to_dicts():
            key_tuple = tuple(row[k] for k in KEY) + (row.get("fold_id"),)

            # Use cached scoring method for tiebreaker
            tiebreaker_score = self._get_score_cached(
                row, rank_metric, tiebreaker_partition, aggregate, load_arrays,
                aggregate_method=aggregate_method,
                aggregate_exclude_outliers=aggregate_exclude_outliers
            )

            tiebreaker_scores[key_tuple] = tiebreaker_score

        # Add tiebreaker scores to rank_scores
        for rs in rank_scores:
            key_tuple = tuple(rs[k] for k in KEY) + (rs.get("fold_id"),)
            rs["tiebreaker_score"] = tiebreaker_scores.get(key_tuple)

        # Sort with tiebreaker: primary by rank_score, secondary by tiebreaker_score
        # Filter out None and NaN scores - these indicate missing or invalid data
        def _is_valid_score(score):
            if score is None:
                return False
            try:
                return not np.isnan(score)
            except (TypeError, ValueError):
                return True  # Non-numeric scores are kept (shouldn't happen)

        rank_scores = [r for r in rank_scores if _is_valid_score(r["rank_score"])]

        def sort_key(x):
            rank = x["rank_score"]
            tiebreaker = x.get("tiebreaker_score")
            # Handle None tiebreaker - use inf (worst) for ascending, -inf for descending
            if tiebreaker is None:
                tiebreaker = float('inf') if ascending else float('-inf')
            return (rank, tiebreaker)

        rank_scores.sort(key=sort_key, reverse=not ascending)

        # Handle group_by filtering
        # IMPORTANT: This happens AFTER global sorting, so we preserve rank order
        # and just take the first (best) occurrence per group.
        effective_group_by: Optional[List[str]] = None

        # Handle deprecated best_per_model parameter
        if best_per_model:
            warnings.warn(
                "best_per_model is deprecated and will be removed in a future version. "
                "Use group_by=['model_name'] instead.",
                DeprecationWarning,
                stacklevel=2
            )
            if group_by is None:
                effective_group_by = ['model_name']

        # Handle group_by parameter (can be string or list)
        if group_by is not None:
            if isinstance(group_by, str):
                effective_group_by = [group_by]
            else:
                effective_group_by = list(group_by)

        # Apply group-by filtering if requested
        if effective_group_by:
            seen_groups: set = set()
            filtered_scores = []
            for rs in rank_scores:
                group_key = self._make_group_key(rs, effective_group_by)
                if group_key not in seen_groups:
                    seen_groups.add(group_key)
                    filtered_scores.append(rs)
            rank_scores = filtered_scores

        top_keys = rank_scores[:n]

        if not top_keys:
            return PredictionResultsList([])

        # 2) DISPLAY: Get display partition data for top models
        results = []
        for top_key in top_keys:
            # Filter to this specific model
            model_filter = {k: top_key[k] for k in KEY}

            result = PredictionResult({
                **model_filter,
                "rank_metric": rank_metric,
                "rank_score": top_key["rank_score"],
                "rank_id": top_key["id"],
                "fold_id": top_key.get("fold_id")
            })

            if aggregate_partitions:
                # Add nested structure for all partitions
                for partition in ["train", "val", "test"]:
                    partition_data = base.filter(
                        pl.col("partition") == partition
                    ).filter([pl.col(k) == v for k, v in model_filter.items()])

                    # Filter by fold_id to get the correct fold's data
                    if top_key.get("fold_id") is not None:
                        partition_data = partition_data.filter(pl.col("fold_id") == top_key["fold_id"])

                    if partition_data.height > 0:
                        row = partition_data.to_dicts()[0]

                        y_true = None
                        y_pred = None
                        y_proba = None
                        if load_arrays:
                            y_true = self._get_array(row, "y_true")
                            y_pred = self._get_array(row, "y_pred")
                            y_proba = self._get_array(row, "y_proba")

                        # Apply aggregation if requested
                        was_aggregated = False
                        if aggregate and y_pred is not None:
                            metadata = json.loads(row.get("metadata", "{}"))
                            model_name = row.get("model_name", "")
                            pred_id = row.get("id", "")
                            y_true, y_pred, y_proba, was_aggregated = self._apply_aggregation(
                                y_true, y_pred, y_proba, metadata, aggregate, model_name, pred_id,
                                row=row, partition=partition, method=aggregate_method,
                                exclude_outliers=aggregate_exclude_outliers
                            )

                        partition_dict = {
                            "y_true": y_true,  # Keep as numpy array
                            "y_pred": y_pred,  # Keep as numpy array
                            "y_proba": y_proba,  # Keep as numpy array
                            "train_score": row.get("train_score"),
                            "val_score": row.get("val_score"),
                            "test_score": row.get("test_score"),
                            "fold_id": row.get("fold_id"),
                            "aggregated": was_aggregated
                        }

                        # Add metadata from test partition
                        if partition == "test":
                            # Get arrays using _get_array method
                            sample_indices = None
                            weights = None
                            if load_arrays:
                                sample_indices = self._get_array(row, "sample_indices")
                                weights = self._get_array(row, "weights")

                            result.update({
                                "partition": "test",
                                "dataset_name": row.get("dataset_name"),
                                "dataset_path": row.get("dataset_path"),
                                "config_path": row.get("config_path"),
                                "pipeline_uid": row.get("pipeline_uid"),
                                "model_classname": row.get("model_classname"),
                                "model_path": row.get("model_path"),
                                "model_artifact_id": row.get("model_artifact_id"),
                                "trace_id": row.get("trace_id"),
                                "fold_id": row.get("fold_id"),
                                "op_counter": row.get("op_counter"),
                                "sample_indices": sample_indices if sample_indices is not None else np.array([]),
                                "weights": weights if weights is not None else np.array([]),
                                "metadata": json.loads(row.get("metadata", "{}")),
                                "metric": row.get("metric"),
                                "task_type": row.get("task_type", "regression"),
                                "n_samples": len(y_pred) if y_pred is not None else row.get("n_samples"),
                                "n_features": row.get("n_features"),
                                "preprocessings": row.get("preprocessings"),
                                "best_params": json.loads(row.get("best_params", "{}")),
                                "train_score": row.get("train_score"),
                                "val_score": row.get("val_score"),
                                "test_score": row.get("test_score"),
                                "aggregated": was_aggregated,
                                "branch_id": row.get("branch_id"),
                                "branch_name": row.get("branch_name"),
                            })
                            result["id"] = result["rank_id"]

                        # Recalculate metrics if aggregated
                        if aggregate and y_true is not None and y_pred is not None:
                            try:
                                agg_score = evaluator.eval(y_true, y_pred, row.get("metric", rank_metric))
                                # Store recalculated score in partition_dict
                                partition_dict[f"{partition}_score_aggregated"] = agg_score
                            except Exception:
                                pass

                        # Add display metrics
                        if display_metrics:
                            # Parse scores for this partition (only used if not aggregated)
                            partition_scores = {}
                            if not aggregate and row.get("scores"):
                                try:
                                    all_scores = json.loads(row.get("scores"))
                                    partition_scores = all_scores.get(partition, {})
                                except Exception:
                                    pass

                            for metric in display_metrics:
                                # If aggregated, always recalculate metrics
                                if aggregate and y_true is not None and y_pred is not None:
                                    try:
                                        score = evaluator.eval(y_true, y_pred, metric)
                                        partition_dict[metric] = score
                                    except Exception:
                                        partition_dict[metric] = None
                                # Try pre-computed scores first
                                elif metric in partition_scores:
                                    partition_dict[metric] = partition_scores[metric]
                                else:
                                    # Fallback to legacy
                                    stored_score_key = f"{partition}_score" if partition != "val" else "val_score"
                                    if metric == row.get("metric"):
                                        partition_dict[metric] = row.get(stored_score_key)
                                    else:
                                        try:
                                            if load_arrays and y_true is not None and y_pred is not None:
                                                score = evaluator.eval(y_true, y_pred, metric)
                                                partition_dict[metric] = score
                                            else:
                                                partition_dict[metric] = None
                                        except Exception:
                                            partition_dict[metric] = None

                        # Nest partitions under 'partitions' key
                        if 'partitions' not in result:
                            result['partitions'] = {}
                        result['partitions'][partition] = partition_dict
            else:
                # Single partition display
                display_data = base.filter(
                    pl.col("partition") == display_partition
                ).filter([pl.col(k) == v for k, v in model_filter.items()])

                if top_key.get("fold_id") is not None:
                    display_data = display_data.filter(pl.col("fold_id") == top_key["fold_id"])

                if display_data.height > 0:
                    row = display_data.to_dicts()[0]

                    y_true = None
                    y_pred = None
                    y_proba = None
                    sample_indices = None
                    weights = None

                    if load_arrays:
                        y_true = self._get_array(row, "y_true")
                        y_pred = self._get_array(row, "y_pred")
                        y_proba = self._get_array(row, "y_proba")
                        sample_indices = self._get_array(row, "sample_indices")
                        weights = self._get_array(row, "weights")

                    # Get metadata for aggregation
                    metadata = json.loads(row.get("metadata", "{}"))

                    # Apply aggregation if requested
                    was_aggregated = False
                    if aggregate and y_pred is not None:
                        model_name = row.get("model_name", "")
                        pred_id = row.get("id", "")
                        y_true, y_pred, y_proba, was_aggregated = self._apply_aggregation(
                            y_true, y_pred, y_proba, metadata, aggregate, model_name, pred_id,
                            row=row, partition=display_partition, method=aggregate_method,
                            exclude_outliers=aggregate_exclude_outliers
                        )

                    result.update({
                        "partition": display_partition,
                        "dataset_name": row.get("dataset_name"),
                        "dataset_path": row.get("dataset_path"),
                        "config_path": row.get("config_path"),
                        "pipeline_uid": row.get("pipeline_uid"),
                        "model_classname": row.get("model_classname"),
                        "model_path": row.get("model_path"),
                        "model_artifact_id": row.get("model_artifact_id"),
                        "trace_id": row.get("trace_id"),
                        "fold_id": row.get("fold_id"),
                        "op_counter": row.get("op_counter"),
                        "sample_indices": sample_indices if sample_indices is not None else np.array([]),
                        "weights": weights if weights is not None else np.array([]),
                        "metadata": metadata,
                        "metric": row.get("metric"),
                        "task_type": row.get("task_type", "regression"),
                        "n_samples": len(y_pred) if y_pred is not None else row.get("n_samples"),
                        "n_features": row.get("n_features"),
                        "preprocessings": row.get("preprocessings"),
                        "best_params": json.loads(row.get("best_params", "{}")),
                        "y_true": y_true,  # Keep as numpy array
                        "y_pred": y_pred,  # Keep as numpy array
                        "y_proba": y_proba,  # Keep as numpy array
                        "train_score": row.get("train_score"),
                        "val_score": row.get("val_score"),
                        "test_score": row.get("test_score"),
                        "aggregated": was_aggregated,
                        "branch_id": row.get("branch_id"),
                        "branch_name": row.get("branch_name"),
                    })
                    result["id"] = result["rank_id"]

                    # Recalculate metrics if aggregated
                    if aggregate and y_true is not None and y_pred is not None:
                        # Store original scores for reference
                        result["original_test_score"] = row.get("test_score")
                        result["original_val_score"] = row.get("val_score")
                        result["original_train_score"] = row.get("train_score")

                        # Recalculate main metric on aggregated data
                        try:
                            agg_score = evaluator.eval(y_true, y_pred, row.get("metric", rank_metric))
                            result["test_score"] = agg_score
                        except Exception:
                            pass

                    # Add display metrics
                    if display_metrics:
                        # Parse scores for this partition (only used if not aggregated)
                        partition_scores = {}
                        if not aggregate and row.get("scores"):
                            try:
                                all_scores = json.loads(row.get("scores"))
                                partition_scores = all_scores.get(display_partition, {})
                            except Exception:
                                pass

                        for metric in display_metrics:
                            # If aggregated, always recalculate metrics
                            if aggregate and y_true is not None and y_pred is not None:
                                try:
                                    score = evaluator.eval(y_true, y_pred, metric)
                                    result[metric] = score
                                except Exception:
                                    result[metric] = None
                            # Try pre-computed scores first
                            elif metric in partition_scores:
                                result[metric] = partition_scores[metric]
                            else:
                                # Fallback to legacy
                                if metric == row.get("metric"):
                                    stored_score_key = f"{display_partition}_score" if display_partition != "val" else "val_score"
                                    result[metric] = row.get(stored_score_key)
                                else:
                                    try:
                                        if load_arrays and y_true is not None and y_pred is not None:
                                            score = evaluator.eval(y_true, y_pred, metric)
                                            result[metric] = score
                                        else:
                                            result[metric] = None
                                    except Exception:
                                        result[metric] = None

            results.append(result)

        return PredictionResultsList(results)

    def get_best(
        self,
        metric: str = "",
        ascending: Optional[bool] = None,
        aggregate_partitions: bool = False,
        aggregate: Optional[str] = None,
        aggregate_method: Optional[str] = None,
        aggregate_exclude_outliers: bool = False,
        **filters
    ) -> Optional[PredictionResult]:
        """
        Get single best prediction by metric.

        Convenience wrapper around top() to get just the best model.

        Args:
            metric: Metric to rank by
            ascending: Sort order. If True, sorts ascending (lower is better).
                      If False, sorts descending (higher is better).
                      If None, infers from metric.
            aggregate_partitions: If True, include all partition data
            aggregate: If provided, aggregate predictions by this metadata column or 'y'.
            aggregate_method: Aggregation method ('mean', 'median', 'vote').
            aggregate_exclude_outliers: If True, exclude outliers using T² before aggregation.
            **filters: Additional filter criteria

        Returns:
            Best PredictionResult or None if no matches

        Examples:
            >>> best = ranker.get_best(metric="rmse", dataset_name="wheat")
            >>> best_r2 = ranker.get_best(metric="r2", ascending=False)

        Notes:
            - Returns None if no predictions match the filter criteria
            - Use ascending=False for metrics where higher is better (R², accuracy)
            - Automatically falls back to "test" partition if "val" has no data
        """
        # Try ranking by val partition first (default behavior)
        results = self.top(
            n=1,
            rank_metric=metric,
            rank_partition="val",
            ascending=ascending,
            aggregate_partitions=aggregate_partitions,
            aggregate=aggregate,
            aggregate_method=aggregate_method,
            aggregate_exclude_outliers=aggregate_exclude_outliers,
            **filters
        )

        # Fallback to test partition if val has no data
        # This handles single-fold splits where validation becomes test
        if not results:
            results = self.top(
                n=1,
                rank_metric=metric,
                rank_partition="test",
                ascending=ascending,
                aggregate_partitions=aggregate_partitions,
                aggregate=aggregate,
                aggregate_method=aggregate_method,
                aggregate_exclude_outliers=aggregate_exclude_outliers,
                **filters
            )

        if not results:
            return None

        return results[0]
