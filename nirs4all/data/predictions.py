"""
Predictions management using Polars.

This module contains the main Predictions facade class that delegates to
specialized components for storage, serialization, ranking, and querying.

Refactored architecture (v0.4.1):
    - Storage: PredictionStorage (DataFrame backend)
    - Serializer: PredictionSerializer (JSON/Parquet hybrid)
    - Indexer: PredictionIndexer (filtering operations)
    - Ranker: PredictionRanker (ranking and top-k)
    - Aggregator: PartitionAggregator (partition combining)
    - Query: CatalogQueryEngine (catalog operations)

Public API is preserved for backward compatibility.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import polars as pl

from nirs4all.utils.emoji import DISK, CHECK, WARNING
from nirs4all.utils import evaluator

# Import components
from .predictions_components import (
    PredictionStorage,
    PredictionSerializer,
    PredictionResult,
    PredictionResultsList,
)
from .predictions_components.indexer import PredictionIndexer
from .predictions_components.ranker import PredictionRanker
from .predictions_components.aggregator import PartitionAggregator
from .predictions_components.query import CatalogQueryEngine

# Re-export result classes for backward compatibility
__all__ = ['Predictions', 'PredictionResult', 'PredictionResultsList']


class Predictions:
    """
    Main facade for prediction management.

    Delegates to specialized components while maintaining backward-compatible public API.

    Architecture:
        - Storage: PredictionStorage (DataFrame backend)
        - Serializer: PredictionSerializer (JSON/Parquet hybrid)
        - Indexer: PredictionIndexer (filtering operations)
        - Ranker: PredictionRanker (ranking and top-k)
        - Aggregator: PartitionAggregator (partition combining)
        - Query: CatalogQueryEngine (catalog operations)

    Examples:
        >>> # Create and add predictions
        >>> pred = Predictions()
        >>> pred.add_prediction(
        ...     dataset_name="wheat",
        ...     model_name="PLS",
        ...     partition="test",
        ...     y_true=y_true,
        ...     y_pred=y_pred,
        ...     test_score=0.85
        ... )
        >>>
        >>> # Query top models
        >>> top_5 = pred.top(n=5, rank_metric="rmse", rank_partition="val")
        >>>
        >>> # Save and load
        >>> pred.save_to_file("predictions.json")
        >>> loaded = Predictions.load("predictions.json")
    """

    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize Predictions storage with component-based architecture.

        Args:
            filepath: Optional path to load predictions from (.meta.parquet file)
        """
        # Initialize components with array registry support
        self._storage = PredictionStorage()
        self._serializer = PredictionSerializer()
        self._indexer = PredictionIndexer(self._storage)
        self._ranker = PredictionRanker(self._storage, self._serializer, self._indexer)
        self._aggregator = PartitionAggregator(self._storage, self._indexer)
        self._query = CatalogQueryEngine(self._storage, self._indexer)

        # Load from file if provided
        if filepath and Path(filepath).exists():
            self.load_from_file(filepath)

    # =========================================================================
    # CORE CRUD OPERATIONS - Delegate to Storage
    # =========================================================================

    def add_prediction(
        self,
        dataset_name: str,
        dataset_path: str = "",
        config_name: str = "",
        config_path: str = "",
        pipeline_uid: Optional[str] = None,
        step_idx: int = 0,
        op_counter: int = 0,
        model_name: str = "",
        model_classname: str = "",
        model_path: str = "",
        fold_id: Optional[Union[str, int]] = None,
        sample_indices: Optional[List[int]] = None,
        weights: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        partition: str = "",
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        val_score: Optional[float] = None,
        test_score: Optional[float] = None,
        train_score: Optional[float] = None,
        metric: str = "mse",
        task_type: str = "regression",
        n_samples: int = 0,
        n_features: int = 0,
        preprocessings: str = "",
        best_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a single prediction to storage.

        Delegates to PredictionStorage component.

        Args:
            dataset_name: Dataset name
            dataset_path: Path to dataset file
            config_name: Configuration name
            config_path: Path to config file
            pipeline_uid: Unique pipeline identifier
            step_idx: Pipeline step index
            op_counter: Operation counter
            model_name: Model name
            model_classname: Model class name
            model_path: Path to saved model
            fold_id: Cross-validation fold ID
            sample_indices: Indices of samples used
            weights: Sample weights
            metadata: Additional metadata
            partition: Data partition (train/val/test)
            y_true: True labels
            y_pred: Predicted labels
            val_score: Validation score
            test_score: Test score
            train_score: Training score
            metric: Metric name
            task_type: Task type (classification/regression)
            n_samples: Number of samples
            n_features: Number of features
            preprocessings: Preprocessing steps applied
            best_params: Best hyperparameters

        Returns:
            Prediction ID
        """
        row_dict = {
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "config_name": config_name,
            "config_path": config_path,
            "pipeline_uid": pipeline_uid or "",
            "step_idx": step_idx,
            "op_counter": op_counter,
            "model_name": model_name,
            "model_classname": model_classname,
            "model_path": model_path,
            "fold_id": str(fold_id) if fold_id is not None else "",
            "sample_indices": sample_indices if sample_indices is not None else [],
            "weights": weights.tolist() if isinstance(weights, np.ndarray) else (weights if weights is not None else []),
            "metadata": metadata if metadata is not None else {},
            "partition": partition,
            "y_true": y_true if y_true is not None else np.array([]),
            "y_pred": y_pred if y_pred is not None else np.array([]),
            "val_score": val_score,
            "test_score": test_score,
            "train_score": train_score,
            "metric": metric,
            "task_type": task_type,
            "n_samples": n_samples,
            "n_features": n_features,
            "preprocessings": preprocessings,
            "best_params": best_params if best_params is not None else {},
        }

        return self._storage.add_row(row_dict)

    def add_predictions(
        self,
        dataset_name: Union[str, List[str]],
        dataset_path: Union[str, List[str]] = "",
        config_name: Union[str, List[str]] = "",
        config_path: Union[str, List[str]] = "",
        pipeline_uid: Union[Optional[str], List[Optional[str]]] = None,
        step_idx: Union[int, List[int]] = 0,
        op_counter: Union[int, List[int]] = 0,
        model_name: Union[str, List[str]] = "",
        model_classname: Union[str, List[str]] = "",
        model_path: Union[str, List[str]] = "",
        fold_id: Union[Optional[str], List[Optional[str]]] = None,
        sample_indices: Union[Optional[List[int]], List[Optional[List[int]]]] = None,
        weights: Union[Optional[List[float]], List[Optional[List[float]]]] = None,
        metadata: Union[Optional[Dict[str, Any]], List[Optional[Dict[str, Any]]]] = None,
        partition: Union[str, List[str]] = "",
        y_true: Union[Optional[np.ndarray], List[Optional[np.ndarray]]] = None,
        y_pred: Union[Optional[np.ndarray], List[Optional[np.ndarray]]] = None,
        val_score: Union[Optional[float], List[Optional[float]]] = None,
        test_score: Union[Optional[float], List[Optional[float]]] = None,
        train_score: Union[Optional[float], List[Optional[float]]] = None,
        metric: Union[str, List[str]] = "mse",
        task_type: Union[str, List[str]] = "regression",
        n_samples: Union[int, List[int]] = 0,
        n_features: Union[int, List[int]] = 0,
        preprocessings: Union[str, List[str]] = "",
        best_params: Union[Optional[Dict[str, Any]], List[Optional[Dict[str, Any]]]] = None
    ) -> None:
        """
        Add multiple predictions to storage (batch operation).

        For each parameter, if it's a single value it will be broadcast to all predictions.
        If it's a list, each index corresponds to one prediction.

        Args:
            Same as add_prediction, but can be single values or lists
        """
        # Collect all parameters
        params = {
            'dataset_name': dataset_name,
            'dataset_path': dataset_path,
            'config_name': config_name,
            'config_path': config_path,
            'pipeline_uid': pipeline_uid,
            'step_idx': step_idx,
            'op_counter': op_counter,
            'model_name': model_name,
            'model_classname': model_classname,
            'model_path': model_path,
            'fold_id': fold_id,
            'sample_indices': sample_indices,
            'weights': weights,
            'metadata': metadata,
            'partition': partition,
            'y_true': y_true,
            'y_pred': y_pred,
            'val_score': val_score,
            'test_score': test_score,
            'train_score': train_score,
            'metric': metric,
            'task_type': task_type,
            'n_samples': n_samples,
            'n_features': n_features,
            'preprocessings': preprocessings,
            'best_params': best_params,
        }

        # Find the maximum length (number of predictions)
        max_length = 1
        for param_value in params.values():
            if isinstance(param_value, list):
                max_length = max(max_length, len(param_value))

        if max_length == 1:
            # No lists, single prediction
            self.add_prediction(**params)
            return

        # Add predictions one by one (simpler and more reliable than batch)
        for i in range(max_length):
            prediction_params = {}
            for param_name, param_value in params.items():
                if isinstance(param_value, list):
                    idx = min(i, len(param_value) - 1)
                    prediction_params[param_name] = param_value[idx]
                else:
                    prediction_params[param_name] = param_value

            # Add individual prediction
            self.add_prediction(**prediction_params)

    # =========================================================================
    # FILTERING OPERATIONS - Delegate to Indexer
    # =========================================================================

    def filter_predictions(
        self,
        dataset_name: Optional[str] = None,
        partition: Optional[str] = None,
        config_name: Optional[str] = None,
        model_name: Optional[str] = None,
        fold_id: Optional[str] = None,
        step_idx: Optional[int] = None,
        load_arrays: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Filter predictions and return as list of dictionaries.

        Delegates to PredictionIndexer for filtering, then deserializes results.
        Supports lazy loading of arrays for performance optimization.

        Args:
            dataset_name: Filter by dataset name
            partition: Filter by partition
            config_name: Filter by config name
            model_name: Filter by model name
            fold_id: Filter by fold ID
            step_idx: Filter by step index
            load_arrays: If True, loads actual arrays from registry (slower).
                        If False, returns metadata only with array references (fast).
            **kwargs: Additional filter criteria

        Returns:
            List of prediction dictionaries with deserialized numpy arrays (if load_arrays=True)
            or metadata with array_id references (if load_arrays=False)

        Examples:
            >>> # Fast metadata-only query
            >>> preds = predictions.filter_predictions(dataset_name="wheat", load_arrays=False)
            >>> # Full query with arrays
            >>> preds = predictions.filter_predictions(dataset_name="wheat", load_arrays=True)
        """
        df_filtered = self._indexer.filter(
            dataset_name=dataset_name,
            partition=partition,
            config_name=config_name,
            model_name=model_name,
            fold_id=fold_id,
            step_idx=step_idx,
            **kwargs
        )

        # Deserialize results
        results = []
        for row in df_filtered.to_dicts():
            deserialized = self._serializer.deserialize_row(row)

            # Hydrate arrays if requested (always using array registry now)
            if load_arrays:
                pred_id = deserialized.get('id')
                if pred_id:
                    # Get full prediction with arrays from storage
                    full_pred = self._storage.get_by_id(pred_id, load_arrays=True)
                    if full_pred:
                        deserialized = full_pred

            results.append(deserialized)

        return results

    def get_similar(self, **filter_kwargs) -> Optional[Dict[str, Any]]:
        """
        Get the first prediction matching filter criteria.

        Args:
            **filter_kwargs: Filter criteria (same as filter_predictions)

        Returns:
            First matching prediction or None
        """
        results = self.filter_predictions(**filter_kwargs)
        return results[0] if results else None

    # =========================================================================
    # RANKING OPERATIONS - Delegate to Ranker
    # =========================================================================

    def top(
        self,
        n: int,
        rank_metric: str = "",
        rank_partition: str = "val",
        display_metrics: Optional[List[str]] = None,
        display_partition: str = "test",
        aggregate_partitions: bool = False,
        ascending: bool = True,
        group_by_fold: bool = False,
        **filters
    ) -> PredictionResultsList:
        """
        Get top n models ranked by a metric on a specific partition.

        Delegates to PredictionRanker component.

        Args:
            n: Number of top models to return
            rank_metric: Metric to rank by (if empty, uses record's metric or val_score)
            rank_partition: Partition to rank on (default: "val")
            display_metrics: Metrics to compute for display (default: task_type defaults)
            display_partition: Partition to display results from (default: "test")
            aggregate_partitions: If True, add train/val/test nested dicts in results
            ascending: If True, lower scores rank higher (for error metrics like RMSE)
            group_by_fold: If True, include fold_id in model identity (rank per fold)
            **filters: Additional filter criteria (dataset_name, config_name, etc.)

        Returns:
            PredictionResultsList containing top n models
        """
        return self._ranker.top(
            n=n,
            rank_metric=rank_metric,
            rank_partition=rank_partition,
            display_metrics=display_metrics,
            display_partition=display_partition,
            aggregate_partitions=aggregate_partitions,
            ascending=ascending,
            group_by_fold=group_by_fold,
            **filters
        )

    def get_best(
        self,
        metric: str = "",
        ascending: bool = True,
        aggregate_partitions: bool = False,
        **filters
    ) -> Optional[PredictionResult]:
        """
        Get the best prediction for a specific metric.

        Delegates to PredictionRanker component.

        Args:
            metric: Metric to optimize
            ascending: If True, lower scores are better
            aggregate_partitions: If True, add partition data
            **filters: Additional filter criteria

        Returns:
            Best prediction or None
        """
        return self._ranker.get_best(
            metric=metric,
            ascending=ascending,
            aggregate_partitions=aggregate_partitions,
            **filters
        )

    # =========================================================================
    # AGGREGATION OPERATIONS - Delegate to Aggregator
    # =========================================================================

    def _add_partition_data(
        self,
        results: List[PredictionResult],
        partitions: List[str] = None
    ) -> List[PredictionResult]:
        """
        Add partition data to results (internal helper).

        Delegates to PartitionAggregator component.

        Args:
            results: List of PredictionResult objects
            partitions: List of partitions to aggregate

        Returns:
            Results with partition data added
        """
        return self._aggregator.add_partition_data(results, partitions)

    # =========================================================================
    # CATALOG QUERY OPERATIONS - Delegate to CatalogQueryEngine
    # =========================================================================

    def query_best(
        self,
        dataset_name: Optional[str] = None,
        metric: str = "test_score",
        n: int = 10,
        ascending: bool = False
    ) -> pl.DataFrame:
        """
        Query for best performing pipelines by metric (catalog query).

        Delegates to CatalogQueryEngine component.

        Args:
            dataset_name: Filter by dataset name
            metric: Metric column to rank by
            n: Number of top results
            ascending: If True, lower scores rank higher

        Returns:
            DataFrame with top n predictions
        """
        return self._query.query_best(
            dataset_name=dataset_name,
            metric=metric,
            n=n,
            ascending=ascending
        )

    def filter_by_criteria(
        self,
        dataset_name: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        metric_thresholds: Optional[Dict[str, float]] = None
    ) -> pl.DataFrame:
        """
        Filter predictions by multiple criteria (catalog query).

        Delegates to CatalogQueryEngine component.

        Args:
            dataset_name: Filter by dataset name
            date_range: Tuple of (start_date, end_date)
            metric_thresholds: Dict of metric names to threshold values

        Returns:
            Filtered DataFrame
        """
        return self._query.filter_by_criteria(
            dataset_name=dataset_name,
            date_range=date_range,
            metric_thresholds=metric_thresholds
        )

    def compare_across_datasets(
        self,
        pipeline_hash: str,
        metric: str = "test_score"
    ) -> pl.DataFrame:
        """
        Compare a pipeline's performance across multiple datasets.

        Delegates to CatalogQueryEngine component.

        Args:
            pipeline_hash: Pipeline UID to compare
            metric: Metric column to compare

        Returns:
            DataFrame with one row per dataset
        """
        return self._query.compare_across_datasets(
            pipeline_hash=pipeline_hash,
            metric=metric
        )

    def list_runs(self, dataset_name: Optional[str] = None) -> pl.DataFrame:
        """
        List all prediction runs with summary information.

        Delegates to CatalogQueryEngine component.

        Args:
            dataset_name: Filter by dataset name (None for all)

        Returns:
            DataFrame with run summary
        """
        return self._query.list_runs(dataset_name=dataset_name)

    def get_summary_stats(self, metric: str = "test_score") -> Dict[str, float]:
        """
        Get summary statistics for a metric.

        Delegates to CatalogQueryEngine component.

        Args:
            metric: Metric column name

        Returns:
            Dictionary with min, max, mean, median, std
        """
        return self._query.get_summary_stats(metric=metric)

    # =========================================================================
    # STORAGE OPERATIONS - Delegate to Storage & Serializer
    # =========================================================================

    def save_to_file(self, filepath: str, format: str = "parquet") -> None:
        """
        Save predictions to split Parquet format with array registry.

        Args:
            filepath: Output file path (should end with .meta.parquet)
            format: Format to use (only "parquet" is supported)

        Examples:
            >>> predictions.save_to_file("predictions.meta.parquet")
        """
        if format != "parquet":
            raise ValueError(f"Only 'parquet' format is supported, got: {format}")

        filepath = Path(filepath)

        if not filepath.name.endswith(".meta.parquet"):
            raise ValueError(
                f"Expected .meta.parquet extension, got: {filepath.name}\n"
                f"Use 'predictions.meta.parquet' as the filename"
            )

        # Split Parquet with array registry
        arrays_path = filepath.with_name(
            filepath.name.replace(".meta.parquet", ".arrays.parquet")
        )
        self._storage.save_parquet(filepath, arrays_path)

    def load_from_file(self, filepath: str) -> None:
        """
        Load predictions from split Parquet format.

        Supports:
        - Split Parquet with array registry (.meta.parquet + .arrays.parquet)

        Args:
            filepath: Path to .meta.parquet file

        Examples:
            >>> predictions.load_from_file("predictions.meta.parquet")
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Must be .meta.parquet format
        if not filepath.name.endswith('.meta.parquet'):
            raise ValueError(
                f"Expected .meta.parquet file, got: {filepath}\n"
                f"Only split Parquet format is supported (use .meta.parquet + .arrays.parquet)"
            )

        arrays_path = filepath.with_name(
            filepath.name.replace('.meta.parquet', '.arrays.parquet')
        )

        if not arrays_path.exists():
            raise FileNotFoundError(
                f"Array file not found: {arrays_path}\n"
                f"Expected paired .arrays.parquet file for {filepath}"
            )

        # Load using storage
        self._storage.load_parquet(filepath, arrays_path)

        # Reinitialize dependent components
        self._indexer = PredictionIndexer(self._storage)
        self._ranker = PredictionRanker(self._storage, self._serializer, self._indexer)
        self._aggregator = PartitionAggregator(self._storage, self._indexer)
        self._query = CatalogQueryEngine(self._storage, self._indexer)

    @classmethod
    def load_from_file_cls(cls, filepath: str) -> 'Predictions':
        """
        Load predictions from JSON file as class method.

        Args:
            filepath: Input file path

        Returns:
            Predictions instance with loaded data (empty if file doesn't exist)
        """
        instance = cls()
        if Path(filepath).exists():
            instance.load_from_file(filepath)
        return instance

    @classmethod
    def load(
        cls,
        dataset_name: Optional[str] = None,
        path: str = "results",
        aggregate_partitions: bool = False,
        **filters
    ) -> 'Predictions':
        """
        Load predictions from results directory structure.

        Args:
            dataset_name: Name of dataset to load (None for all)
            path: Base path to search for predictions
            aggregate_partitions: If True, aggregate partition data
            **filters: Additional filter criteria

        Returns:
            Predictions instance with loaded data
        """
        instance = cls()
        base_path = Path(path)

        # Case 1: path is a .meta.parquet file
        if base_path.is_file() and base_path.name.endswith('.meta.parquet'):
            instance.load_from_file(str(base_path))

        # Case 2: path is a directory
        elif base_path.is_dir():
            if dataset_name:
                # Look for .meta.parquet files
                dataset_path = base_path / dataset_name / "predictions.meta.parquet"
                if dataset_path.exists():
                    temp = cls()
                    temp.load_from_file(str(dataset_path))
                    instance.merge_predictions(temp)
            else:
                # Load all datasets
                predictions_files = list(base_path.glob("*/predictions.meta.parquet"))
                for pred_file in predictions_files:
                    temp = cls()
                    temp.load_from_file(str(pred_file))
                    instance.merge_predictions(temp)

        # Apply filters if provided
        if filters:
            df = instance._storage.to_dataframe()
            for key, value in filters.items():
                if key in df.columns:
                    df = df.filter(pl.col(key) == value)
            instance._storage._df = df

        return instance

    def save_to_parquet(self, catalog_dir: Path, prediction_id: str = None) -> tuple:
        """
        Save predictions as split Parquet (metadata + arrays separate).

        Appends to existing files if they exist.

        Delegates to PredictionStorage component.

        Args:
            catalog_dir: Directory for catalog storage
            prediction_id: Optional prediction ID (generates UUID if None)

        Returns:
            Tuple of (meta_path, data_path)
        """
        pred_id = prediction_id or str(uuid4())
        df = self._storage.to_dataframe()

        if "prediction_id" not in df.columns:
            df = df.with_columns(pl.lit(pred_id).alias("prediction_id"))

        if "created_at" not in df.columns:
            df = df.with_columns(
                pl.lit(datetime.now().isoformat()).alias("created_at")
            )

        # Update storage DataFrame
        self._storage._df = df

        # Save using storage component
        catalog_dir = Path(catalog_dir)
        catalog_dir.mkdir(parents=True, exist_ok=True)

        meta_path = catalog_dir / "predictions_meta.parquet"
        data_path = catalog_dir / "predictions_data.parquet"

        # Load existing data if files exist
        if meta_path.exists() and data_path.exists():
            existing = Predictions.load_from_parquet(catalog_dir)
            # Merge with existing
            existing._storage.merge(self._storage)
            # Save the merged data
            existing._storage.save_parquet(meta_path, data_path)
        else:
            # Save new data
            self._storage.save_parquet(meta_path, data_path)

        return (meta_path, data_path)

    @classmethod
    def load_from_parquet(cls, catalog_dir: Path, prediction_ids: list = None) -> 'Predictions':
        """
        Load predictions from split Parquet storage.

        Args:
            catalog_dir: Path to catalog directory
            prediction_ids: Optional list of prediction IDs to load

        Returns:
            Predictions instance with loaded data
        """
        instance = cls()
        catalog_dir = Path(catalog_dir)

        meta_path = catalog_dir / "predictions_meta.parquet"
        data_path = catalog_dir / "predictions_data.parquet"

        if meta_path.exists() and data_path.exists():
            instance._storage.load_parquet(meta_path, data_path)

            # Filter by prediction_ids if specified
            if prediction_ids:
                df = instance._storage.to_dataframe()
                if "prediction_id" in df.columns:
                    df = df.filter(pl.col("prediction_id").is_in(prediction_ids))
                    instance._storage._df = df

        return instance

    def archive_to_catalog(
        self,
        catalog_dir: Path,
        pipeline_dir: Path,
        metrics: Dict[str, Any] = None
    ) -> str:
        """
        Archive pipeline predictions to catalog.

        Loads predictions CSV from pipeline directory, adds metadata,
        and saves to catalog.

        Delegates to PredictionStorage for CSV loading.

        Args:
            catalog_dir: Catalog directory for storage
            pipeline_dir: Pipeline directory containing predictions.csv
            metrics: Optional metadata dict to add to predictions

        Returns:
            Generated prediction ID
        """
        # Load CSV and prepare data using storage component
        pred_csv = pipeline_dir / "predictions.csv"
        pred_data = self._storage.archive_from_csv(pred_csv, metrics)

        # Add to predictions
        pred_id = self.add_prediction(**pred_data)

        # Save to catalog
        self.save_to_parquet(catalog_dir, pred_id)

        return pred_id

    def merge_predictions(self, other: 'Predictions') -> None:
        """
        Merge predictions from another Predictions instance.

        Delegates to PredictionStorage component.

        Args:
            other: Another Predictions instance to merge
        """
        self._storage.merge(other._storage)

    def clear(self) -> None:
        """
        Clear all predictions.

        Delegates to PredictionStorage component.
        """
        self._storage.clear()

    # =========================================================================
    # METADATA & UTILITY OPERATIONS - Delegate to Indexer
    # =========================================================================

    @property
    def _df(self) -> pl.DataFrame:
        """
        Backward compatibility property for direct DataFrame access.

        Delegates to storage._df for tests and legacy code.

        Returns:
            Internal Polars DataFrame
        """
        return self._storage._df

    @_df.setter
    def _df(self, value: pl.DataFrame) -> None:
        """
        Backward compatibility setter for direct DataFrame assignment.

        Args:
            value: DataFrame to set
        """
        self._storage._df = value

    @property
    def num_predictions(self) -> int:
        """Get the number of stored predictions."""
        return len(self._storage)

    def get_unique_values(self, column: str) -> List[str]:
        """
        Get unique values for a specific column.

        Delegates to PredictionIndexer component.

        Args:
            column: Column name

        Returns:
            List of unique values
        """
        return self._indexer.get_unique_values(column)

    def get_datasets(self) -> List[str]:
        """
        Get list of unique dataset names.

        Delegates to PredictionIndexer component.

        Returns:
            List of dataset names
        """
        return self._indexer.get_datasets()

    def get_partitions(self) -> List[str]:
        """
        Get list of unique partitions.

        Delegates to PredictionIndexer component.

        Returns:
            List of partitions
        """
        return self._indexer.get_partitions()

    def get_configs(self) -> List[str]:
        """
        Get list of unique config names.

        Delegates to PredictionIndexer component.

        Returns:
            List of config names
        """
        return self._indexer.get_configs()

    def get_models(self) -> List[str]:
        """
        Get list of unique model names.

        Delegates to PredictionIndexer component.

        Returns:
            List of model names
        """
        return self._indexer.get_models()

    def get_folds(self) -> List[str]:
        """
        Get list of unique fold IDs.

        Delegates to PredictionIndexer component.

        Returns:
            List of fold IDs
        """
        return self._indexer.get_folds()

    # =========================================================================
    # STATIC UTILITY METHODS
    # =========================================================================

    @staticmethod
    def save_predictions_to_csv(
        y_true: Optional[Union[np.ndarray, List[float]]] = None,
        y_pred: Optional[Union[np.ndarray, List[float]]] = None,
        filepath: str = "",
        prefix: str = "",
        suffix: str = ""
    ) -> None:
        """
        Save y_true and y_pred arrays to a CSV file.

        Args:
            y_true: True values array
            y_pred: Predicted values array
            filepath: Output CSV file path
            prefix: Optional prefix for column names
            suffix: Optional suffix for column names
        """
        if y_pred is None:
            raise ValueError("y_pred is required")

        y_pred_arr = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred
        y_pred_flat = y_pred_arr.flatten()

        data_dict = {f"{prefix}y_pred{suffix}": y_pred_flat.tolist()}

        if y_true is not None:
            y_true_arr = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
            y_true_flat = y_true_arr.flatten()

            if len(y_true_flat) != len(y_pred_flat):
                raise ValueError(
                    f"Length mismatch: y_true ({len(y_true_flat)}) != y_pred ({len(y_pred_flat)})"
                )

            data_dict[f"{prefix}y_true{suffix}"] = y_true_flat.tolist()

        df_csv = pl.DataFrame(data_dict)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df_csv.write_csv(filepath)
        print(f"{DISK}Saved predictions to {filepath}")

    @staticmethod
    def save_all_to_csv(
        predictions: 'Predictions',
        path: str = "results",
        aggregate_partitions: bool = False,
        **filters
    ) -> None:
        """
        Save all predictions to CSV files.

        Args:
            predictions: Predictions instance
            path: Base path for saving
            aggregate_partitions: If True, save one file per model with all partitions
            **filters: Additional filter criteria
        """
        if aggregate_partitions:
            all_results = predictions.top(
                n=predictions.num_predictions,
                aggregate_partitions=True,
                group_by_fold=True,
                **filters
            )
        else:
            all_results = predictions.top(
                n=predictions.num_predictions,
                aggregate_partitions=False,
                group_by_fold=True,
                **filters
            )

        for result in all_results:
            try:
                result.save_to_csv(path=path)
            except Exception as e:
                model_id = result.get('id', 'unknown')
                print(f"{WARNING}Failed to save prediction {model_id}: {e}")

        print(f"{CHECK}Saved {len(all_results)} files to {path}")

    @classmethod
    def pred_short_string(cls, entry: Dict, metrics: Optional[List[str]] = None, partition: str | List[str] = "test") -> str:
        """
        Generate short string representation of a prediction.

        Args:
            entry: Prediction dictionary
            metrics: Optional list of metrics to display

        Returns:
            Short description string
        """
        scores_str = ""
        if metrics:
            # Make a copy to avoid modifying the original list
            metrics = metrics.copy()
            if isinstance(partition, str):
                partition = [partition]

            for p in partition:
                # Handle aggregated partitions structure
                if 'partitions' in entry and p in entry['partitions']:
                    y_true = entry['partitions'][p].get('y_true')
                    y_pred = entry['partitions'][p].get('y_pred')
                else:
                    y_true = entry.get('y_true')
                    y_pred = entry.get('y_pred')

                if y_true is not None and y_pred is not None:
                    scores = evaluator.eval_list(
                        y_true,
                        y_pred,
                        metrics=metrics
                    )
                    scores_str += f" [{p}]: "
                    scores_str += ", ".join(
                        [f"[{k}:{v:.4f}]" for k, v in zip(metrics, scores)]
                    )

        desc = f"{entry['model_name']} - {entry['metric']} "
        desc += f"[test: {entry['test_score']:.4f}], [val: {entry['val_score']:.4f}]"
        if scores_str:
            desc += f", {scores_str}"
        desc += f", (fold: {entry['fold_id']}, id: {entry['op_counter']}, "
        desc += f"step: {entry['step_idx']}) - [{entry['id']}]"
        return desc

    @classmethod
    def pred_long_string(cls, entry: Dict, metrics: Optional[List[str]] = None) -> str:
        """
        Generate long string representation of a prediction.

        Args:
            entry: Prediction dictionary
            metrics: Optional list of metrics to display

        Returns:
            Long description string with config
        """
        return cls.pred_short_string(entry, metrics=metrics) + f" | [{entry['config_name']}]"

    def get_entry_partitions(self, entry: Dict) -> Dict[str, Optional[Dict]]:
        """
        Get all partition data for an entry.

        Args:
            entry: Prediction entry dictionary

        Returns:
            Dictionary with 'train', 'val', 'test' keys containing partition data
        """
        res = {}
        filter_dict = {
            'dataset_name': entry['dataset_name'],
            'config_name': entry['config_name'],
            'model_name': entry['model_name'],
            'fold_id': entry['fold_id'],
            'step_idx': entry['step_idx'],
            'op_counter': entry['op_counter']
        }

        for partition in ['train', 'val', 'test']:
            filter_dict['partition'] = partition
            predictions = self.filter_predictions(**filter_dict, load_arrays=True)
            if predictions:
                res[partition] = predictions[0]
            else:
                res[partition] = None

        return res

    # =========================================================================
    # CONVERSION METHODS
    # =========================================================================

    def to_dataframe(self) -> pl.DataFrame:
        """Get predictions as Polars DataFrame."""
        return self._storage.to_dataframe()

    def to_dicts(self, load_arrays: bool = True) -> List[Dict[str, Any]]:
        """
        Get predictions as list of dictionaries.

        Args:
            load_arrays: If True, hydrate array references with actual arrays.
                        If False, returns metadata with array IDs only (faster).

        Returns:
            List of prediction dictionaries
        """
        if load_arrays:
            # Hydrate arrays for each row
            df = self._storage.to_dataframe()
            result = []
            for row in df.to_dicts():
                hydrated = self._storage._hydrate_arrays(row)
                result.append(hydrated)
            return result
        else:
            # Return raw data with array IDs
            return self._storage.to_dataframe().to_dicts()

    def to_pandas(self):
        """Get predictions as pandas DataFrame."""
        return self._storage.to_dataframe().to_pandas()

    # =========================================================================
    # DUNDER METHODS
    # =========================================================================

    def __len__(self) -> int:
        """Return number of stored predictions."""
        return len(self._storage)

    def __repr__(self) -> str:
        """String representation."""
        if len(self._storage) == 0:
            return "Predictions(empty)"
        return f"Predictions({len(self._storage)} entries)"

    def __str__(self) -> str:
        """User-friendly string representation."""
        if len(self._storage) == 0:
            return "📈 Predictions: No predictions stored"

        datasets = self.get_datasets()
        configs = self.get_configs()
        models = self.get_models()

        return (
            f"📈 Predictions: {len(self._storage)} entries\n"
            f"   Datasets: {datasets}\n"
            f"   Configs: {configs}\n"
            f"   Models: {models}"
        )
