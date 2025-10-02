"""
Predictions management using Polars.

This module contains Predictions class for storing and managing model predictions
with metadata using Polars DataFrame as the backend for efficient data manipulation.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import polars as pl
from pathlib import Path
import json
import nirs4all.dataset.evaluator as Evaluator

class Predictions:
    """
    Storage for model predictions using Polars DataFrame backend.

    Each prediction is stored as a row with the following schema:
    - dataset_name: str
    - dataset_path: str
    - config_name: str
    - config_path: str
    - step_idx: int
    - op_counter: int
    - model_name: str
    - model_classname: str
    - model_path: str
    - fold_id: str
    - sample_indices: List[int] (stored as string)
    - weights: Optional[List[float]] (stored as string)
    - metadata: Dict[str, Any] (stored as JSON string)
    - partition: str
    - y_true: List[float] (stored as string)
    - y_pred: List[float] (stored as string)
    - val_score: Optional[float]
    - test_score: Optional[float]
    - metric: str
    - task_type: str
    - n_samples: int
    - n_features: int
    """

    def __init__(self, filepath: Optional[str] = None):
        """Initialize Predictions storage with Polars DataFrame backend."""
        self._df = pl.DataFrame(schema={
            "dataset_name": pl.Utf8,
            "dataset_path": pl.Utf8,
            "config_name": pl.Utf8,
            "config_path": pl.Utf8,
            "step_idx": pl.Int64,
            "op_counter": pl.Int64,
            "model_name": pl.Utf8,
            "model_classname": pl.Utf8,
            "model_path": pl.Utf8,
            "fold_id": pl.Utf8,
            "sample_indices": pl.Utf8,  # JSON string
            "weights": pl.Utf8,  # JSON string
            "metadata": pl.Utf8,  # JSON string
            "partition": pl.Utf8,
            "y_true": pl.Utf8,  # JSON string
            "y_pred": pl.Utf8,  # JSON string
            "val_score": pl.Float64,
            "test_score": pl.Float64,
            "metric": pl.Utf8,
            "task_type": pl.Utf8,
            "n_samples": pl.Int64,
            "n_features": pl.Int64,
        })

        if filepath and Path(filepath).exists():
            self.load_from_file(filepath)

    def add_prediction(
        self,
        dataset_name: str,
        dataset_path: str = "",
        config_name: str = "",
        config_path: str = "",
        step_idx: int = 0,
        op_counter: int = 0,
        model_name: str = "",
        model_classname: str = "",
        model_path: str = "",
        fold_id: str | int= None,
        sample_indices: Optional[List[int]] = None,
        weights: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        partition: str = "",
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        val_score: Optional[float] = None,
        test_score: Optional[float] = None,
        metric: str = "mse",
        task_type: str = "regression",
        n_samples: int = 0,
        n_features: int = 0
    ) -> None:
        """Add a new prediction to the storage."""

        # Convert numpy arrays to lists for JSON serialization
        y_true_list = y_true.tolist() if y_true is not None else []
        y_pred_list = y_pred.tolist() if y_pred is not None else []
        sample_indices_list = sample_indices if sample_indices is not None else []
        weights_list = weights if weights is not None else []
        metadata_dict = metadata if metadata is not None else {}
        fold_id = str(fold_id)

        # Create new row
        new_row = pl.DataFrame([{
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "config_name": config_name,
            "config_path": config_path,
            "step_idx": step_idx,
            "op_counter": op_counter,
            "model_name": model_name,
            "model_classname": model_classname,
            "model_path": model_path,
            "fold_id": fold_id,
            "sample_indices": json.dumps(sample_indices_list),
            "weights": json.dumps(weights_list),
            "metadata": json.dumps(metadata_dict),
            "partition": partition,
            "y_true": json.dumps(y_true_list),
            "y_pred": json.dumps(y_pred_list),
            "val_score": val_score,
            "test_score": test_score,
            "metric": metric,
            "task_type": task_type,
            "n_samples": n_samples,
            "n_features": n_features,
        }])

        # Append to main DataFrame
        self._df = pl.concat([self._df, new_row])

    def add_predictions(
        self,
        dataset_name: Union[str, List[str]],
        dataset_path: Union[str, List[str]] = "",
        config_name: Union[str, List[str]] = "",
        config_path: Union[str, List[str]] = "",
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
        metric: Union[str, List[str]] = "mse",
        task_type: Union[str, List[str]] = "regression",
        n_samples: Union[int, List[int]] = 0,
        n_features: Union[int, List[int]] = 0
    ) -> None:
        """
        Add multiple predictions to the storage.

        For each parameter:
        - If it's a single value, it will be copied to all predictions
        - If it's a list, the value at each index will be used for the corresponding prediction

        The number of predictions is determined by the longest list parameter.

        Args:
            dataset_name: Dataset name(s) - can be single string or list
            dataset_path: Dataset path(s) - can be single string or list
            config_name: Config name(s) - can be single string or list
            config_path: Config path(s) - can be single string or list
            step_idx: Step index(es) - can be single int or list
            op_counter: Operation counter(s) - can be single int or list
            model_name: Model name(s) - can be single string or list
            model_classname: Model classname(s) - can be single string or list
            model_path: Model path(s) - can be single string or list
            fold_id: Fold ID(s) - can be single string or list
            sample_indices: Sample indices - can be single list or list of lists
            weights: Weights - can be single list or list of lists
            metadata: Metadata - can be single dict or list of dicts
            partition: Partition(s) - can be single string or list
            y_true: True values - can be single array or list of arrays
            y_pred: Predicted values - can be single array or list of arrays
            val_score: Loss score(s) - can be single float or list
            test_score: Evaluation score(s) - can be single float or list
            metric: Metric(s) - can be single string or list
            task_type: Task type(s) - can be single string or list
            n_samples: Number of samples - can be single int or list
            n_features: Number of features - can be single int or list
        """
        # Collect all parameters
        params = {
            'dataset_name': dataset_name,
            'dataset_path': dataset_path,
            'config_name': config_name,
            'config_path': config_path,
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
            'metric': metric,
            'task_type': task_type,
            'n_samples': n_samples,
            'n_features': n_features
        }

        # Find the maximum length (number of predictions to create)
        max_length = 1
        list_params = {}

        for param_name, param_value in params.items():
            if isinstance(param_value, list):
                max_length = max(max_length, len(param_value))
                list_params[param_name] = param_value

        if max_length == 1:
            # No lists found, just call add_prediction once
            self.add_prediction(**params)
            return

        # Create individual predictions
        for i in range(max_length):
            prediction_params = {}

            for param_name, param_value in params.items():
                if isinstance(param_value, list):
                    # Use value at index i, or last value if list is shorter
                    idx = min(i, len(param_value) - 1)
                    prediction_params[param_name] = param_value[idx]
                else:
                    # Single value, copy to all predictions
                    prediction_params[param_name] = param_value

            # Add the individual prediction
            self.add_prediction(**prediction_params)

    def filter_predictions(
        self,
        dataset_name: Optional[str] = None,
        partition: Optional[str] = None,
        config_name: Optional[str] = None,
        model_name: Optional[str] = None,
        fold_id: Optional[str] = None,
        step_idx: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Filter predictions and return as list of dictionaries.

        Args:
            dataset_name: Filter by dataset name
            partition: Filter by partition
            config_name: Filter by config name
            model_name: Filter by model name
            fold_id: Filter by fold ID
            step_idx: Filter by step index
            **kwargs: Additional filter criteria

        Returns:
            List of prediction dictionaries
        """
        df_filtered = self._df

        # Apply filters
        if dataset_name is not None:
            df_filtered = df_filtered.filter(pl.col("dataset_name") == dataset_name)
        if partition is not None:
            df_filtered = df_filtered.filter(pl.col("partition") == partition)
        if config_name is not None:
            df_filtered = df_filtered.filter(pl.col("config_name") == config_name)
        if model_name is not None:
            df_filtered = df_filtered.filter(pl.col("model_name") == model_name)
        if fold_id is not None:
            df_filtered = df_filtered.filter(pl.col("fold_id") == fold_id)
        if step_idx is not None:
            df_filtered = df_filtered.filter(pl.col("step_idx") == step_idx)

        # Apply additional filters from kwargs
        for key, value in kwargs.items():
            if key in self._df.columns:
                df_filtered = df_filtered.filter(pl.col(key) == value)

        # Convert to list of dictionaries with JSON deserialization
        results = []
        for row in df_filtered.to_dicts():
            # Deserialize JSON fields
            row["sample_indices"] = json.loads(row["sample_indices"])
            row["weights"] = json.loads(row["weights"])
            row["metadata"] = json.loads(row["metadata"])
            row["y_true"] = np.array(json.loads(row["y_true"]))
            row["y_pred"] = np.array(json.loads(row["y_pred"]))
            results.append(row)

        return results

    def get_unique_values(self, column: str) -> List[str]:
        """Get unique values for a specific column."""
        if column not in self._df.columns:
            raise ValueError(f"Column '{column}' not found in predictions")
        return self._df[column].unique().to_list()

    def get_datasets(self) -> List[str]:
        """Get list of unique dataset names."""
        return self.get_unique_values("dataset_name")

    def get_partitions(self) -> List[str]:
        """Get list of unique partitions."""
        return self.get_unique_values("partition")

    def get_configs(self) -> List[str]:
        """Get list of unique config names."""
        return self.get_unique_values("config_name")

    def get_models(self) -> List[str]:
        """Get list of unique model names."""
        return self.get_unique_values("model_name")

    def list_keys(self) -> List[str]:
        """Get list of unique prediction keys (for compatibility with old interface)."""
        # Generate keys similar to the old format: dataset/config/model/partition/fold
        if len(self._df) == 0:
            return []

        keys = []
        for row in self._df.iter_rows(named=True):
            fold_part = f"_fold_{row['fold_id']}" if row['fold_id'] is not None else ""
            key = f"{row['dataset_name']}/{row['config_name']}/{row['model_name']}/{row['partition']}{fold_part}"
            keys.append(key)

        return list(set(keys))  # Remove duplicates

    # def get_prediction_by_key(self, key: str) -> Optional[Dict[str, Any]]:
    #     """Get prediction data by key (for compatibility with old interface)."""
    #     # Parse key format: dataset/config/model/partition[_fold_X]
    #     parts = key.split('/')
    #     if len(parts) < 4:
    #         return None

    #     dataset_name, config_name, model_name, partition_part = parts[:4]

    #     # Extract fold info if present
    #     fold_id = None
    #     partition = partition_part
    #     if '_fold_' in partition_part:
    #         partition, fold_part = partition_part.split('_fold_')
    #         try:
    #             fold_id = int(fold_part)
    #         except ValueError:
    #             pass

    #     # Query the DataFrame
    #     filter_expr = (
    #         (pl.col('dataset_name') == dataset_name) &
    #         (pl.col('config_name') == config_name) &
    #         (pl.col('model_name') == model_name) &
    #         (pl.col('partition') == partition)
    #     )

    #     if fold_id is not None:
    #         filter_expr = filter_expr & (pl.col('fold_id') == fold_id)

    #     matches = self._df.filter(filter_expr)

    #     if len(matches) == 0:
    #         return None

    #     # Return the first match as a dictionary with the expected format
    #     row = matches.row(0, named=True)
    #     return {
    #         'dataset_name': row['dataset_name'],
    #         'config_name': row['config_name'],
    #         'model_name': row['model_name'],
    #         'partition': row['partition'],
    #         'fold_id': row['fold_id'],
    #         'y_true': json.loads(row['y_true']) if row['y_true'] else [],
    #         'y_pred': json.loads(row['y_pred']) if row['y_pred'] else [],
    #         'test_score': row['test_score'],
    #         'metric': row['metric'],
    #     }

    def save_to_file(self, filepath: str) -> None:
        """Save predictions to JSON file."""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Convert DataFrame to JSON-serializable format
            data = self._df.to_dicts()

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            # print(f"💾 Saved {len(self._df)} predictions to {filepath}")

        except Exception as e:
            print(f"⚠️ Error saving predictions to {filepath}: {e}")

    def load_from_file(self, filepath: str) -> None:
        """Load predictions from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if data:
                self._df = pl.DataFrame(data)
                print(f"📥 Loaded {len(self._df)} predictions from {filepath}")

        except Exception as e:
            print(f"⚠️ Error loading predictions from {filepath}: {e}")

    @classmethod
    def load_from_file_cls(cls, filepath: str) -> 'Predictions':
        """Load predictions from JSON file as class method."""
        instance = cls()
        instance.load_from_file(filepath)
        return instance

    def save_prediction_to_csv(self, filepath: str, index: Optional[int] = None) -> None:
        """
        Save a single prediction to CSV file.

        Args:
            filepath: Output CSV file path
            index: Index of prediction to save (if None, saves all)
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            if index is not None:
                # Save single prediction
                if index >= len(self._df):
                    raise IndexError(f"Index {index} out of range")
                row = self._df[index].to_dicts()[0]

                # Deserialize arrays for CSV
                y_true = json.loads(row["y_true"])
                y_pred = json.loads(row["y_pred"])
                sample_indices = json.loads(row["sample_indices"])

                # Create CSV data
                csv_data = []
                for i, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
                    sample_idx = sample_indices[i] if i < len(sample_indices) else i
                    csv_data.append({
                        "sample_index": sample_idx,
                        "y_true": true_val,
                        "y_pred": pred_val,
                        "dataset_name": row["dataset_name"],
                        "model_name": row["model_name"],
                        "partition": row["partition"],
                        "fold_id": row["fold_id"],
                        "metric": row["metric"]
                    })

                df_csv = pl.DataFrame(csv_data)
                df_csv.write_csv(filepath)
                print(f"💾 Saved prediction {index} to {filepath}")
            else:
                # Save all predictions in expanded format
                self._df.write_csv(filepath)
                print(f"💾 Saved all {len(self._df)} predictions to {filepath}")

        except Exception as e:
            print(f"⚠️ Error saving prediction to CSV {filepath}: {e}")

    def calculate_scores(self, metrics: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Calculate scores for all predictions using external evaluator.

        Args:
            metrics: List of metrics to calculate (defaults to ['mse', 'r2', 'mae'])

        Returns:
            DataFrame with calculated scores
        """
        if metrics is None:
            metrics = ['mse', 'r2', 'mae']

        # Import the external evaluator
        try:
            from ..utils.model_utils import ModelUtils, TaskType
            model_utils = ModelUtils()
        except ImportError:
            print("⚠️ Cannot import ModelUtils for score calculation")
            return pl.DataFrame()

        scores_data = []

        for row in self._df.to_dicts():
            # Deserialize prediction arrays
            y_true = np.array(json.loads(row["y_true"]))
            y_pred = np.array(json.loads(row["y_pred"]))

            if len(y_true) > 0 and len(y_pred) > 0:
                # Calculate all scores using regression as default
                all_scores = model_utils.calculate_scores(y_true, y_pred, TaskType.REGRESSION)

                # Filter to requested metrics if they exist
                scores = {}
                for metric in metrics:
                    if metric in all_scores:
                        scores[metric] = all_scores[metric]
                    else:
                        # If metric not found, set to None
                        scores[metric] = None

                # Create score record
                score_record = {
                    "dataset_name": row["dataset_name"],
                    "config_name": row["config_name"],
                    "model_name": row["model_name"],
                    "partition": row["partition"],
                    "fold_id": row["fold_id"],
                    "step_idx": row["step_idx"],
                }

                # Add calculated metrics
                for metric, score in scores.items():
                    score_record[f"score_{metric}"] = score if isinstance(score, (int, float)) else None

                scores_data.append(score_record)

        return pl.DataFrame(scores_data)

    def top_k(self, k: int = 5, metric: str = "", ascending: bool = True, **filters) -> List[Dict[str, Any]]:
        """
        Get top K predictions ranked by metric, val_score, or test_score.
        By default filters to test partition unless otherwise specified.

        Args:
            metric: Metric name to rank by ("" for test_score, "loss" for val_score, else calculate metric on-the-fly)
            k: Number of top results to return (-1 to return all filtered predictions)
            ascending: If True, lower scores rank higher (for error metrics)
            **filters: Additional filter criteria

        Returns:
            List of top K prediction dictionaries (or all if k=-1)
        """
        # Add default partition filter if not specified
        if 'partition' not in filters:
            filters['partition'] = 'val'

        # First filter the entries
        df_filtered = self._df
        for key, value in filters.items():
            if key in df_filtered.columns:
                df_filtered = df_filtered.filter(pl.col(key) == value)

        if df_filtered.is_empty():
            return []

        # Handle different ranking scenarios
        if metric == "" or metric == "loss":
            # Use existing stored scores
            rank_col = "val_score"
            df_ranked = df_filtered.filter(pl.col(rank_col).is_not_null())
            if df_ranked.is_empty():
                return []

            df_sorted = df_ranked.sort(rank_col, descending=not ascending)

            # Return all results if k=-1, otherwise return top k
            if k == -1:
                top_k_rows = df_sorted
            else:
                top_k_rows = df_sorted.head(k)

            # Convert to list of dictionaries with JSON deserialization
            results = []
            for row in top_k_rows.to_dicts():
                # Deserialize JSON fields
                row["sample_indices"] = json.loads(row["sample_indices"])
                row["weights"] = json.loads(row["weights"]) if row["weights"] else []
                row["metadata"] = json.loads(row["metadata"]) if row["metadata"] else {}
                row["y_true"] = np.array(json.loads(row["y_true"]))
                row["y_pred"] = np.array(json.loads(row["y_pred"]))
                results.append(row)

            return results

        else:
            # Calculate metric on-the-fly for all filtered entries
            scores_data = []

            # Import the external evaluator
            try:
                from ..utils.model_utils import ModelUtils, TaskType
                model_utils = ModelUtils()
            except ImportError:
                print("⚠️ Cannot import ModelUtils for score calculation")
                return []

            for i, row in enumerate(df_filtered.to_dicts()):
                # Deserialize prediction arrays
                y_true = np.array(json.loads(row["y_true"]))
                y_pred = np.array(json.loads(row["y_pred"]))

                if len(y_true) > 0 and len(y_pred) > 0:
                    # Calculate all scores using regression as default, explicitly include the requested metric
                    all_scores = model_utils.calculate_scores(y_true, y_pred, TaskType.REGRESSION, metrics=[metric])

                    # Get the requested metric score
                    if metric in all_scores:
                        metric_score = all_scores[metric]

                        # Create score record
                        score_record = row.copy()
                        score_record["computed_score"] = metric_score
                        score_record["computed_metric"] = metric
                        score_record[metric] = metric_score

                        # Deserialize JSON fields for final output
                        score_record["sample_indices"] = json.loads(row["sample_indices"])
                        score_record["weights"] = json.loads(row["weights"]) if row["weights"] else []
                        score_record["metadata"] = json.loads(row["metadata"]) if row["metadata"] else {}
                        score_record["y_true"] = y_true
                        score_record["y_pred"] = y_pred

                        scores_data.append(score_record)

            # Sort by computed metric and return top k (or all if k=-1)
            if not scores_data:
                return []
            if ModelUtils._is_higher_better(metric):
                ascending = not ascending  # Reverse for higher is better
            scores_data.sort(key=lambda x: x["computed_score"], reverse=not ascending)

            # Return all results if k=-1, otherwise return top k
            if k == -1:
                return scores_data
            else:
                return scores_data[:k]

    def get_best(self, metric: str = "", ascending: bool = True, **filters) -> Optional[Dict[str, Any]]:
        """
        Get the best prediction for a specific metric, val_score, or test_score.
        This is an alias for top_k with k=1.

        Args:
            metric: Metric name to optimize ("" for test_score, "loss" for val_score, else metric)
            ascending: If True, lower scores are better (for error metrics)
            **filters: Additional filter criteria

        Returns:
            Best prediction dictionary or None
        """
        top_results = self.top_k(k=1, metric=metric, ascending=ascending, **filters)
        return top_results[0] if top_results else None

    def bottom_k(self, metric: str = "", k: int = 5, **filters) -> List[Dict[str, Any]]:
        """
        Get bottom K predictions (worst performing).
        This is an alias for top_k with ascending=False.
        By default filters to test partition unless otherwise specified.

        Args:
            metric: Metric name to rank by ("" for test_score, "loss" for val_score, else metric)
            k: Number of bottom results to return (-1 to return all filtered predictions)
            **filters: Additional filter criteria

        Returns:
            List of bottom K prediction dictionaries (or all if k=-1)
        """
        return self.top_k(k=k, metric=metric, ascending=False, **filters)

    def clear(self) -> None:
        """Clear all predictions."""
        self._df = self._df.clear()

    def merge_predictions(self, other: 'Predictions') -> None:
        """
        Merge predictions from another Predictions instance.

        Args:
            other: Another Predictions instance to merge from

        Note:
            - Duplicate predictions (same metadata) will be kept (no deduplication)
            - Use this method to combine results from multiple experiments
        """
        if not isinstance(other, Predictions):
            raise TypeError("Can only merge with another Predictions instance")

        if len(other._df) == 0:
            print("⚠️ No predictions to merge (source is empty)")
            return

        # Ensure schemas are compatible before concatenating
        if len(self._df) == 0:
            # If current DataFrame is empty, just copy the other
            self._df = other._df.clone()
        else:
            # Check if schemas are compatible and align them before concatenating
            self_schema = self._df.schema
            other_schema = other._df.schema

            # Check if schemas match exactly
            schemas_match = (
                len(self_schema) == len(other_schema) and
                all(col in other_schema and self_schema[col] == other_schema[col] for col in self_schema) and
                list(self._df.columns) == list(other._df.columns)
            )

            if schemas_match:
                # Schemas match, safe to concatenate directly
                self._df = pl.concat([self._df, other._df], how="vertical")
            else:
                # Schemas don't match, need to align them
                # print(f"⚠️ Schema mismatch detected, aligning schemas before merge")

                # Use the predefined schema order from __init__ to ensure consistency
                predefined_order = [
                    "dataset_name", "dataset_path", "config_name", "config_path",
                    "step_idx", "op_counter", "model_name", "model_classname", "model_path",
                    "fold_id", "sample_indices", "weights", "metadata", "partition",
                    "y_true", "y_pred", "val_score", "test_score", "metric", "task_type",
                    "n_samples", "n_features"
                ]

                # Determine the target schema by preferring non-null types, maintaining order
                unified_schema = {}
                all_columns = set(self_schema.keys()) | set(other_schema.keys())

                # Process columns in predefined order first, then any extra columns
                ordered_columns = [col for col in predefined_order if col in all_columns]
                extra_columns = [col for col in all_columns if col not in predefined_order]
                ordered_columns.extend(extra_columns)

                for col_name in ordered_columns:
                    self_type = self_schema.get(col_name, pl.Null)
                    other_type = other_schema.get(col_name, pl.Null)

                    # Prefer non-null types
                    if self_type == pl.Null and other_type != pl.Null:
                        unified_schema[col_name] = other_type
                    elif other_type == pl.Null and self_type != pl.Null:
                        unified_schema[col_name] = self_type
                    elif self_type == other_type:
                        unified_schema[col_name] = self_type
                    else:
                        # If types differ and neither is null, prefer the more specific type
                        # Float64 > Int64 > Utf8 > Null in terms of preference
                        type_priority = {pl.Null: 0, pl.Utf8: 1, pl.Int64: 2, pl.Float64: 3}
                        self_priority = type_priority.get(self_type, 1)
                        other_priority = type_priority.get(other_type, 1)

                        if other_priority > self_priority:
                            unified_schema[col_name] = other_type
                        else:
                            unified_schema[col_name] = self_type

                # Align self DataFrame to unified schema (maintaining column order)
                self_cast_expressions = []
                for col_name in ordered_columns:
                    target_type = unified_schema[col_name]
                    if col_name in self._df.columns:
                        if self._df[col_name].dtype == pl.Null:
                            self_cast_expressions.append(pl.lit(None).cast(target_type).alias(col_name))
                        else:
                            self_cast_expressions.append(pl.col(col_name).cast(target_type))
                    else:
                        self_cast_expressions.append(pl.lit(None).cast(target_type).alias(col_name))

                # Align other DataFrame to unified schema (maintaining column order)
                other_cast_expressions = []
                for col_name in ordered_columns:
                    target_type = unified_schema[col_name]
                    if col_name in other._df.columns:
                        if other._df[col_name].dtype == pl.Null:
                            other_cast_expressions.append(pl.lit(None).cast(target_type).alias(col_name))
                        else:
                            other_cast_expressions.append(pl.col(col_name).cast(target_type))
                    else:
                        other_cast_expressions.append(pl.lit(None).cast(target_type).alias(col_name))

                # Apply schema alignment
                self_aligned = self._df.select(self_cast_expressions)
                other_aligned = other._df.select(other_cast_expressions)

                # Concatenate aligned DataFrames
                self._df = pl.concat([self_aligned, other_aligned], how="vertical")


    def merge_predictions_with_dedup(self, other: 'Predictions') -> None:
        """
        Merge predictions from another Predictions instance with deduplication.

        Args:
            other: Another Predictions instance to merge from

        Note:
            - Duplicates are identified by: dataset_name, config_name, model_name,
              partition, fold_id, step_idx, op_counter
            - When duplicates are found, keeps the existing prediction (no replacement)
        """
        if not isinstance(other, Predictions):
            raise TypeError("Can only merge with another Predictions instance")

        if len(other._df) == 0:
            print("⚠️ No predictions to merge (source is empty)")
            return

        original_count = len(self._df)

        # Define key columns for duplicate detection
        key_columns = [
            "dataset_name", "config_name", "model_name",
            "partition", "fold_id", "step_idx", "op_counter"
        ]

        # Create a combined DataFrame
        combined_df = pl.concat([self._df, other._df], how="vertical")

        # Remove duplicates keeping the first occurrence (existing predictions)
        deduplicated_df = combined_df.unique(subset=key_columns, keep="first")

        self._df = deduplicated_df

        added_count = len(self._df) - original_count
        duplicates_count = len(other._df) - added_count

        # print(f"✅ Merged {added_count} new predictions, skipped {duplicates_count} duplicates. Total: {len(self._df)} predictions")

    def __len__(self) -> int:
        """Return number of stored predictions."""
        return len(self._df)

    def __repr__(self) -> str:
        if len(self._df) == 0:
            return "Predictions(empty)"
        return f"Predictions({len(self._df)} entries)"

    def __str__(self) -> str:
        if len(self._df) == 0:
            return "📈 Predictions: No predictions stored"

        datasets = self.get_datasets()
        configs = self.get_configs()
        models = self.get_models()

        return (f"📈 Predictions: {len(self._df)} entries\n"
                f"   Datasets: {datasets}\n"
                f"   Configs: {configs}\n"
                f"   Models: {models}")





    @classmethod
    def pred_short_string(cls, entry, metrics=None):
        scores_str = ""
        if metrics is not None:
            scores = Evaluator.eval_list(entry['y_true'], entry['y_pred'], metrics=metrics)
            scores_str = ", ".join([f"[{k}:{v:.4f}]" if k != 'rmse' else f"[{k}:{v:.4f}]" for k, v in zip(metrics, scores)])

        short_desc = f"{entry['model_name']} - {entry['metric']} [test: {entry['test_score']:.4f}], [val: {entry['val_score']:.4f}]"
        short_desc += f", {scores_str}"
        short_desc += f" - (fold: {entry['fold_id']}, id: {entry['op_counter']}, step: {entry['step_idx']})"
        return short_desc

    @classmethod
    def pred_long_string(cls, entry, metrics=None): ##ADAPT TO CLASSIFICATION
        return Predictions.pred_short_string(entry, metrics=metrics) + f" | pipeline: {entry['config_name']}"
