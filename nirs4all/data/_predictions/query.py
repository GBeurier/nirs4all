"""
Catalog-specific query operations for predictions.

This module provides the CatalogQueryEngine for high-level catalog
browsing and analysis operations.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import polars as pl

from .storage import PredictionStorage
from .indexer import PredictionIndexer


class CatalogQueryEngine:
    """
    High-level query operations for catalog browsing.

    Features:
        - Best pipeline queries
        - Cross-dataset comparisons
        - Run listing and summary
        - Metric-based filtering

    Examples:
        >>> storage = PredictionStorage()
        >>> indexer = PredictionIndexer(storage)
        >>> query_engine = CatalogQueryEngine(storage, indexer)
        >>> best = query_engine.query_best(dataset_name="wheat", metric="test_score", n=10)
        >>> stats = query_engine.get_summary_stats(metric="test_score")

    Attributes:
        _storage: PredictionStorage instance
        _indexer: PredictionIndexer instance
    """

    def __init__(self, storage: PredictionStorage, indexer: PredictionIndexer):
        """
        Initialize query engine with dependencies.

        Args:
            storage: PredictionStorage instance
            indexer: PredictionIndexer instance
        """
        self._storage = storage
        self._indexer = indexer

    def query_best(
        self,
        dataset_name: Optional[str] = None,
        metric: str = "test_score",
        n: int = 10,
        ascending: bool = False
    ) -> pl.DataFrame:
        """
        Query for best performing pipelines by metric.

        Args:
            dataset_name: Filter by dataset name (None for all datasets)
            metric: Metric column to rank by (default: "test_score")
            n: Number of top results to return
            ascending: If True, lower scores rank higher

        Returns:
            DataFrame with top n predictions sorted by metric

        Examples:
            >>> # Get top 10 models for wheat dataset
            >>> best = query_engine.query_best(
            ...     dataset_name="wheat",
            ...     metric="test_score",
            ...     n=10,
            ...     ascending=False  # Higher test scores are better
            ... )
            >>>
            >>> # Get worst performing models (for debugging)
            >>> worst = query_engine.query_best(
            ...     dataset_name="corn",
            ...     metric="test_score",
            ...     n=5,
            ...     ascending=True
            ... )
        """
        df = self._storage.to_dataframe()

        # Filter by dataset if specified
        if dataset_name:
            df = df.filter(pl.col("dataset_name") == dataset_name)

        # Filter to non-null metric values
        df = df.filter(pl.col(metric).is_not_null())

        # Sort and limit
        df = df.sort(metric, descending=not ascending).limit(n)

        return df

    def filter_by_criteria(
        self,
        dataset_name: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        metric_thresholds: Optional[Dict[str, float]] = None
    ) -> pl.DataFrame:
        """
        Filter predictions by multiple criteria.

        Args:
            dataset_name: Filter by dataset name
            date_range: Tuple of (start_date, end_date) for filtering (not yet implemented)
            metric_thresholds: Dict of metric names to threshold values

        Returns:
            Filtered DataFrame

        Examples:
            >>> # Get predictions with R² > 0.9
            >>> good_preds = query_engine.filter_by_criteria(
            ...     dataset_name="wheat",
            ...     metric_thresholds={"test_score": 0.9}
            ... )
        """
        df = self._storage.to_dataframe()

        # Filter by dataset
        if dataset_name:
            df = df.filter(pl.col("dataset_name") == dataset_name)

        # Filter by metric thresholds
        if metric_thresholds:
            for metric_col, threshold in metric_thresholds.items():
                if metric_col in df.columns:
                    df = df.filter(pl.col(metric_col) >= threshold)

        # TODO: Implement date_range filtering when timestamp column added

        return df

    def compare_across_datasets(
        self,
        pipeline_hash: str,
        metric: str = "test_score"
    ) -> pl.DataFrame:
        """
        Compare a pipeline's performance across multiple datasets.

        Args:
            pipeline_hash: Pipeline UID to compare
            metric: Metric column to compare

        Returns:
            DataFrame with one row per dataset showing the metric value

        Examples:
            >>> comparison = query_engine.compare_across_datasets(
            ...     pipeline_hash="abc123",
            ...     metric="test_score"
            ... )
        """
        df = self._storage.to_dataframe()

        # Filter to this pipeline
        df = df.filter(pl.col("pipeline_uid") == pipeline_hash)

        # Group by dataset and get best score per dataset
        df = df.group_by("dataset_name").agg([
            pl.col(metric).max().alias(f"best_{metric}"),
            pl.col("model_name").first().alias("model_name"),
            pl.col("config_name").first().alias("config_name")
        ])

        return df

    def list_runs(self, dataset_name: Optional[str] = None) -> pl.DataFrame:
        """
        List all prediction runs with summary information.

        Args:
            dataset_name: Filter by dataset name (None for all)

        Returns:
            DataFrame with run summary information

        Examples:
            >>> runs = query_engine.list_runs(dataset_name="wheat")
            >>> all_runs = query_engine.list_runs()
        """
        df = self._storage.to_dataframe()

        if dataset_name:
            df = df.filter(pl.col("dataset_name") == dataset_name)

        # Group by run identifiers
        summary = df.group_by(["dataset_name", "config_name", "model_name"]).agg([
            pl.col("fold_id").n_unique().alias("n_folds"),
            pl.col("partition").n_unique().alias("n_partitions"),
            pl.col("test_score").mean().alias("avg_test_score"),
            pl.col("test_score").max().alias("best_test_score"),
            pl.col("id").count().alias("n_predictions")
        ])

        return summary

    def get_summary_stats(self, metric: str = "test_score") -> Dict[str, float]:
        """
        Get summary statistics for a metric across all predictions.

        Args:
            metric: Metric column to summarize

        Returns:
            Dictionary with mean, median, min, max, std statistics

        Examples:
            >>> stats = query_engine.get_summary_stats("test_score")
            >>> print(f"Average test score: {stats['mean']:.3f}")
        """
        df = self._storage.to_dataframe()

        # Filter to non-null values
        df = df.filter(pl.col(metric).is_not_null())

        if df.height == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
                "count": 0
            }

        stats = {
            "mean": df[metric].mean(),
            "median": df[metric].median(),
            "min": df[metric].min(),
            "max": df[metric].max(),
            "std": df[metric].std(),
            "count": df.height
        }

        return stats
