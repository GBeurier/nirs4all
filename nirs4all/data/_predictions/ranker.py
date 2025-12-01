"""
Ranking and top-k selection for predictions.

This module provides the PredictionRanker class for ranking predictions
by metrics and selecting top-performing models.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import polars as pl

from nirs4all.core import metrics as evaluator
from nirs4all.data.ensemble_utils import EnsembleUtils

from .storage import PredictionStorage
from .serializer import PredictionSerializer
from .indexer import PredictionIndexer
from .result import PredictionResult, PredictionResultsList


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

    def _apply_aggregation(
        self,
        y_true: Optional[np.ndarray],
        y_pred: Optional[np.ndarray],
        y_proba: Optional[np.ndarray],
        metadata: Dict[str, Any],
        aggregate: str,
        model_name: str = ""
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Apply aggregation to predictions by a group column.

        Args:
            y_true: True values array
            y_pred: Predicted values array
            y_proba: Optional class probabilities array
            metadata: Metadata dictionary containing group column
            aggregate: Group column name or 'y' to group by y_true
            model_name: Model name for warning messages

        Returns:
            Tuple of (aggregated_y_true, aggregated_y_pred, aggregated_y_proba, was_aggregated)
            The was_aggregated flag is True only if aggregation was actually applied.
        """
        import warnings
        from nirs4all.data.predictions import Predictions

        if y_pred is None:
            return y_true, y_pred, y_proba, False

        # Determine group IDs
        if aggregate == 'y':
            if y_true is None:
                return y_true, y_pred, y_proba, False
            group_ids = y_true
        else:
            # Get group IDs from metadata
            if aggregate not in metadata:
                if model_name:
                    warnings.warn(
                        f"Aggregation column '{aggregate}' not found in metadata for model '{model_name}'. "
                        f"Available columns: {list(metadata.keys())}. Skipping aggregation for this model.",
                        UserWarning
                    )
                return y_true, y_pred, y_proba, False
            group_ids = np.asarray(metadata[aggregate])

        if len(group_ids) != len(y_pred):
            if model_name:
                warnings.warn(
                    f"Aggregation column '{aggregate}' length ({len(group_ids)}) doesn't match "
                    f"predictions length ({len(y_pred)}) for model '{model_name}'. Skipping aggregation.",
                    UserWarning
                )
            return y_true, y_pred, y_proba, False

        # Apply aggregation
        result = Predictions.aggregate(
            y_pred=y_pred,
            group_ids=group_ids,
            y_proba=y_proba,
            y_true=y_true
        )

        return (
            result.get('y_true'),
            result.get('y_pred'),
            result.get('y_proba'),
            True  # Aggregation was successfully applied
        )

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
            >>> top_5_by_id = ranker.top(
            ...     n=5,
            ...     rank_metric="rmse",
            ...     aggregate='ID'  # Aggregate by metadata 'ID' column
            ... )

        Notes:
            - If rank_metric matches the stored metric in the data, uses precomputed scores
            - Otherwise, recomputes metric from y_true/y_pred arrays
            - ascending=True means lower scores rank higher (good for RMSE, MAE)
            - ascending=False means higher scores rank higher (good for R², accuracy)
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

        df = self._storage.to_dataframe()
        base = df.filter([pl.col(k) == v for k, v in filters.items()]) if filters else df

        if base.height == 0:
            return PredictionResultsList([])

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

        # Compute rank scores
        rank_scores = []
        for row in rank_data.to_dicts():
            # Try to get score from pre-computed scores first
            scores_json = row.get("scores")
            score = None

            if scores_json:
                try:
                    scores_dict = json.loads(scores_json)
                    if rank_partition in scores_dict and rank_metric in scores_dict[rank_partition]:
                        score = scores_dict[rank_partition][rank_metric]
                except (json.JSONDecodeError, TypeError):
                    pass

            # Fallback to legacy methods if score not found
            if score is None:
                if rank_metric == row["metric"]:
                    # Use precomputed score for the rank_partition
                    score_field = f"{rank_partition}_score"
                    score = row.get(score_field)
                else:
                    # Compute metric from y_true/y_pred (Slow!)
                    try:
                        if load_arrays:
                            y_true = self._get_array(row, "y_true")
                            y_pred = self._get_array(row, "y_pred")
                            if y_true is not None and y_pred is not None:
                                score = evaluator.eval(y_true, y_pred, rank_metric)
                    except Exception:
                        pass

            rank_scores.append({
                **{k: row[k] for k in KEY},
                "rank_score": score,
                "id": row["id"],
                "fold_id": row["fold_id"],
                "scores": scores_json # Pass scores along
            })

        # Sort and get top n
        rank_scores = [r for r in rank_scores if r["rank_score"] is not None]
        rank_scores.sort(key=lambda x: x["rank_score"], reverse=not ascending)
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
                            y_true, y_pred, y_proba, was_aggregated = self._apply_aggregation(
                                y_true, y_pred, y_proba, metadata, aggregate, model_name
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
                                "aggregated": was_aggregated
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
                        y_true, y_pred, y_proba, was_aggregated = self._apply_aggregation(
                            y_true, y_pred, y_proba, metadata, aggregate, model_name
                        )

                    result.update({
                        "partition": display_partition,
                        "dataset_name": row.get("dataset_name"),
                        "dataset_path": row.get("dataset_path"),
                        "config_path": row.get("config_path"),
                        "pipeline_uid": row.get("pipeline_uid"),
                        "model_classname": row.get("model_classname"),
                        "model_path": row.get("model_path"),
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
                        "aggregated": was_aggregated
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
            **filters: Additional filter criteria

        Returns:
            Best PredictionResult or None if no matches

        Examples:
            >>> best = ranker.get_best(metric="rmse", dataset_name="wheat")
            >>> best_r2 = ranker.get_best(metric="r2", ascending=False)

        Notes:
            - Returns None if no predictions match the filter criteria
            - Use ascending=False for metrics where higher is better (R², accuracy)
        """
        results = self.top(
            n=1,
            rank_metric=metric,
            ascending=ascending,
            aggregate_partitions=aggregate_partitions,
            **filters
        )

        if not results:
            return None

        return results[0]
