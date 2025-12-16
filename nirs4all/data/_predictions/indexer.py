"""
Fast filtering and lookup operations for predictions.

This module provides the PredictionIndexer class for efficient filtering
and querying of predictions without caching overhead.
"""

from typing import Dict, Any, List, Optional
import polars as pl

from .storage import PredictionStorage


class PredictionIndexer:
    """
    Optimized filtering and indexing for predictions.

    Features:
        - Multi-column filtering (no caching - calls are rare)
        - Unique value extraction
        - Complex query building

    Examples:
        >>> storage = PredictionStorage()
        >>> indexer = PredictionIndexer(storage)
        >>> filtered = indexer.filter(dataset_name="wheat", partition="test")
        >>> datasets = indexer.get_unique_values("dataset_name")

    Attributes:
        _storage: PredictionStorage instance
    """

    def __init__(self, storage: PredictionStorage):
        """
        Initialize indexer with storage backend.

        Args:
            storage: PredictionStorage instance to index
        """
        self._storage = storage

    def filter(
        self,
        dataset_name: Optional[str] = None,
        partition: Optional[str] = None,
        config_name: Optional[str] = None,
        model_name: Optional[str] = None,
        fold_id: Optional[str] = None,
        step_idx: Optional[int] = None,
        branch_id: Optional[int] = None,
        branch_name: Optional[str] = None,
        **kwargs
    ) -> pl.DataFrame:
        """
        Filter predictions by multiple criteria.

        Args:
            dataset_name: Filter by dataset name
            partition: Filter by partition
            config_name: Filter by config name
            model_name: Filter by model name
            fold_id: Filter by fold ID
            step_idx: Filter by step index
            branch_id: Filter by branch ID (for pipeline branching)
            branch_name: Filter by branch name (for pipeline branching)
            **kwargs: Additional filter criteria

        Returns:
            Filtered Polars DataFrame

        Examples:
            >>> df = indexer.filter(dataset_name="wheat", partition="test")
            >>> df = indexer.filter(model_name="PLS", fold_id="0")
            >>> df = indexer.filter(branch_id=0)  # Get all from first branch
        """
        df = self._storage.to_dataframe()

        # Apply standard filters
        if dataset_name is not None:
            df = df.filter(pl.col("dataset_name") == dataset_name)
        if partition is not None:
            df = df.filter(pl.col("partition") == partition)
        if config_name is not None:
            df = df.filter(pl.col("config_name") == config_name)
        if model_name is not None:
            df = df.filter(pl.col("model_name") == model_name)
        if fold_id is not None:
            df = df.filter(pl.col("fold_id") == str(fold_id))
        if step_idx is not None:
            df = df.filter(pl.col("step_idx") == step_idx)
        if branch_id is not None:
            df = df.filter(pl.col("branch_id") == branch_id)
        if branch_name is not None:
            df = df.filter(pl.col("branch_name") == branch_name)

        # Apply additional filters from kwargs
        for key, value in kwargs.items():
            if key in df.columns and value is not None:
                df = df.filter(pl.col(key) == value)

        return df

    def get_unique_values(self, column: str) -> List[Any]:
        """
        Get unique values for a specific column.

        Args:
            column: Column name

        Returns:
            List of unique values

        Raises:
            ValueError: If column doesn't exist

        Examples:
            >>> datasets = indexer.get_unique_values("dataset_name")
            >>> models = indexer.get_unique_values("model_name")
        """
        df = self._storage.to_dataframe()

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in predictions")

        return df[column].unique().to_list()

    def build_filter_expression(self, **criteria) -> pl.Expr:
        """
        Build a Polars filter expression from criteria.

        Args:
            **criteria: Filter criteria

        Returns:
            Polars expression for filtering

        Examples:
            >>> expr = indexer.build_filter_expression(dataset_name="wheat")
        """
        if not criteria:
            return pl.lit(True)

        exprs = []
        for key, value in criteria.items():
            if value is not None:
                exprs.append(pl.col(key) == value)

        if len(exprs) == 0:
            return pl.lit(True)
        elif len(exprs) == 1:
            return exprs[0]
        else:
            result = exprs[0]
            for expr in exprs[1:]:
                result = result & expr
            return result

    def get_datasets(self) -> List[str]:
        """
        Get list of unique dataset names.

        Returns:
            List of dataset names

        Examples:
            >>> datasets = indexer.get_datasets()
        """
        return self.get_unique_values("dataset_name")

    def get_partitions(self) -> List[str]:
        """
        Get list of unique partitions.

        Returns:
            List of partitions

        Examples:
            >>> partitions = indexer.get_partitions()
        """
        return self.get_unique_values("partition")

    def get_configs(self) -> List[str]:
        """
        Get list of unique config names.

        Returns:
            List of config names

        Examples:
            >>> configs = indexer.get_configs()
        """
        return self.get_unique_values("config_name")

    def get_models(self) -> List[str]:
        """
        Get list of unique model names.

        Returns:
            List of model names

        Examples:
            >>> models = indexer.get_models()
        """
        return self.get_unique_values("model_name")

    def get_folds(self) -> List[str]:
        """
        Get list of unique fold IDs.

        Returns:
            List of fold IDs

        Examples:
            >>> folds = indexer.get_folds()
        """
        return self.get_unique_values("fold_id")
