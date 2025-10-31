"""
Partition data aggregation for predictions.

This module provides the PartitionAggregator class for combining
prediction data across train/val/test partitions.
"""

import json
from typing import Dict, Any, List
import numpy as np

from .storage import PredictionStorage
from .indexer import PredictionIndexer
from .result import PredictionResult, PredictionResultsList


class PartitionAggregator:
    """
    Aggregates prediction data across partitions.

    Features:
        - Train/val/test partition combining
        - Nested dictionary structure creation
        - Metadata preservation

    Examples:
        >>> storage = PredictionStorage()
        >>> indexer = PredictionIndexer(storage)
        >>> aggregator = PartitionAggregator(storage, indexer)
        >>> aggregated = aggregator.aggregate_partitions(
        ...     dataset_name="wheat",
        ...     model_name="PLS"
        ... )

    Attributes:
        _storage: PredictionStorage instance
        _indexer: PredictionIndexer instance
    """

    def __init__(self, storage: PredictionStorage, indexer: PredictionIndexer):
        """
        Initialize aggregator with dependencies.

        Args:
            storage: PredictionStorage instance
            indexer: PredictionIndexer instance
        """
        self._storage = storage
        self._indexer = indexer

    def _parse_vec_json(self, s: str) -> np.ndarray:
        """Parse JSON string to numpy array."""
        return np.asarray(json.loads(s), dtype=float)

    def _get_array(self, row: Dict[str, Any], field_name: str):
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

        return None

    def aggregate_partitions(
        self,
        partitions: List[str] = None,
        **filters
    ) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate data across multiple partitions into nested structure.

        Args:
            partitions: List of partitions to aggregate (default: ["train", "val", "test"])
            **filters: Filter criteria for predictions

        Returns:
            Dictionary with partition names as keys, each containing y_true/y_pred/scores

        Examples:
            >>> agg_data = aggregator.aggregate_partitions(
            ...     partitions=["train", "val", "test"],
            ...     dataset_name="wheat",
            ...     model_name="PLS"
            ... )
            >>> agg_data["test"]["y_true"]  # Access test partition data
        """
        if partitions is None:
            partitions = ["train", "val", "test"]

        aggregated = {}

        for partition in partitions:
            df_partition = self._indexer.filter(partition=partition, **filters)

            if df_partition.height > 0:
                row = df_partition.to_dicts()[0]
                y_true = self._get_array(row, "y_true")
                y_pred = self._get_array(row, "y_pred")

                aggregated[partition] = {
                    "y_true": y_true,  # Keep as numpy array
                    "y_pred": y_pred,  # Keep as numpy array
                    "train_score": row.get("train_score"),
                    "val_score": row.get("val_score"),
                    "test_score": row.get("test_score"),
                    "fold_id": row.get("fold_id"),
                    "n_samples": row.get("n_samples"),
                    "n_features": row.get("n_features"),
                    "metric": row.get("metric"),
                    "task_type": row.get("task_type", "regression")
                }

        return aggregated

    def add_partition_data(
        self,
        results: List[PredictionResult],
        partitions: List[str] = None
    ) -> List[PredictionResult]:
        """
        Add partition data to existing results list.

        Modifies results in-place to include train/val/test nested dictionaries.

        Args:
            results: List of PredictionResult objects to augment
            partitions: List of partitions to add (default: ["train", "val", "test"])

        Returns:
            Modified results list with partition data

        Examples:
            >>> results = [PredictionResult({...})]
            >>> enriched = aggregator.add_partition_data(results)
            >>> enriched[0]["train"]["y_true"]  # Access train data
        """
        if partitions is None:
            partitions = ["train", "val", "test"]

        for result in results:
            # Extract filter criteria from result
            filters = {
                "dataset_name": result.get("dataset_name"),
                "config_name": result.get("config_name"),
                "model_name": result.get("model_name"),
                "fold_id": result.get("fold_id")
            }

            # Aggregate partitions
            aggregated = self.aggregate_partitions(partitions=partitions, **filters)

            # Add to result
            for partition, data in aggregated.items():
                result[partition] = data

        return results

    def merge_partition_arrays(
        self,
        train_data: Dict,
        val_data: Dict,
        test_data: Dict
    ) -> Dict[str, Dict]:
        """
        Merge train/val/test partition data into single structure.

        Args:
            train_data: Training partition data dict
            val_data: Validation partition data dict
            test_data: Test partition data dict

        Returns:
            Dictionary with "train", "val", "test" keys containing merged data

        Examples:
            >>> merged = aggregator.merge_partition_arrays(
            ...     train_data={"y_true": [1, 2], "y_pred": [1.1, 2.2]},
            ...     val_data={"y_true": [3, 4], "y_pred": [3.3, 4.4]},
            ...     test_data={"y_true": [5, 6], "y_pred": [5.5, 6.6]}
            ... )
        """
        return {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
