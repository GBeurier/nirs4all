"""
Predictions management using Polars.

This module contains Predictions class for storing and managing model predictions
with metadata using Polars DataFrame as the backend for efficient data manipulation.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import polars as pl
from pathlib import Path
import json
import hashlib
import nirs4all.dataset.evaluator as Evaluator
import csv
import io


class PredictionResult(dict):
    """
    A dictionary that extends standard dict with prediction-specific methods.
    Behaves like a normal dict but provides additional functionality for saving predictions.
    """
    @property
    def id(self) -> str:
        return self.get("id", "unknown")

    @property
    def fold_id(self) -> str:
        return self.get("fold_id", "unknown")

    @property
    def dataset_name(self) -> str:
        return self.get("dataset_name", "unknown")

    @property
    def model_name(self) -> str:
        return self.get("model_name", "unknown")

    @property
    def step_idx(self) -> int:
        return self.get("step_idx", 0)

    @property
    def op_counter(self) -> int:
        return self.get("op_counter", 0)

    @property
    def config_name(self) -> str:
        return self.get("config_name", "unknown")

    def save_to_csv(self, path: str = "results", force_path: Optional[str] = None) -> None:
        """
        Save prediction result to CSV file.

        Args:
            path: Base path for saving (default: "results")
            force_path: Complete path/filename override (optional)
        """
        if force_path:
            filepath = Path(force_path)
        else:
            # Generate filename from model information
            dataset_name = self.get("dataset_name", "unknown")
            model_id = self.get("id", "unknown")

            base_path = Path(path)
            filepath = base_path / dataset_name / f"{model_id}.csv"

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Determine data structure
        csv_data = []

        # Check if this is an aggregated result (has train/val/test keys)
        has_partitions = all(k in self for k in ["train", "val", "test"])

        if has_partitions:
            # Aggregated data: create columns for each partition
            for partition in ["train", "val", "test"]:
                if partition in self and self[partition] is not None:
                    partition_data = self[partition]
                    y_true = partition_data.get("y_true", [])
                    y_pred = partition_data.get("y_pred", [])

                    # Get fold_id for column naming from partition data (more reliable)
                    # For aggregated data, each partition might have its own fold_id
                    partition_fold_id = partition_data.get("fold_id", self.get("fold_id", ""))
                    if isinstance(partition_fold_id, list) and partition_fold_id:
                        partition_fold_id = partition_fold_id[0]  # Take first if it's a list

                    fold_suffix = f"_fold{partition_fold_id}" if partition_fold_id and partition in ["train", "val"] else ""

                    # Extend csv_data with this partition's data
                    max_len = max(len(y_true), len(y_pred)) if y_true or y_pred else 0

                    for i in range(max_len):
                        if i >= len(csv_data):
                            csv_data.append({})

                        if i < len(y_true):
                            csv_data[i][f"y_true_{partition}{fold_suffix}"] = y_true[i]
                        if i < len(y_pred):
                            csv_data[i][f"y_pred_{partition}{fold_suffix}"] = y_pred[i]
        else:
            # Single partition data: use y_true/y_pred from root
            y_true = self.get("y_true", [])
            y_pred = self.get("y_pred", [])

            max_len = max(len(y_true), len(y_pred)) if y_true or y_pred else 0

            for i in range(max_len):
                row = {}
                if i < len(y_true):
                    row["y_true"] = y_true[i]
                if i < len(y_pred):
                    row["y_pred"] = y_pred[i]
                csv_data.append(row)

        if csv_data:
            # Convert to DataFrame and save
            # Handle potential nested data by converting to strings
            clean_csv_data = []
            for row in csv_data:
                clean_row = {}
                for key, value in row.items():
                    if isinstance(value, (list, np.ndarray)):
                        clean_row[key] = float(value[0]) if len(value) > 0 else 0.0
                    else:
                        clean_row[key] = value
                clean_csv_data.append(clean_row)

            df_csv = pl.DataFrame(clean_csv_data)
            df_csv.write_csv(str(filepath))
            print(f"💾 Saved prediction result to {filepath}")
        else:
            print(f"⚠️ No prediction data found to save for {filepath}")

    def eval_score(self, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate scores for this prediction using specified metrics.

        Args:
            metrics: List of metrics to compute (if None, returns all available metrics)

        Returns:
            Dictionary of metric names to scores.
            For aggregated results: {"train": {...}, "val": {...}, "test": {...}}
            For single partition: {"rmse": ..., "r2": ..., ...}
        """
        scores = {}

        # Check if this is an aggregated result
        has_partitions = all(k in self for k in ["train", "val", "test"])

        if has_partitions:
            # For aggregated results, organize scores by partition in sub-dicts
            for partition in ["train", "val", "test"]:
                if partition in self and self[partition] is not None:
                    partition_data = self[partition]
                    y_true = partition_data.get("y_true", [])
                    y_pred = partition_data.get("y_pred", [])

                    if len(y_true) > 0 and len(y_pred) > 0:
                        y_true_arr = np.array(y_true)
                        y_pred_arr = np.array(y_pred)

                        if metrics is None:
                            # Get all available metrics using task_type
                            task_type = self.get("task_type", "regression")
                            partition_scores = Evaluator.eval_multi(y_true_arr, y_pred_arr, task_type)
                        else:
                            # Get specific metrics
                            partition_scores = {}
                            for metric in metrics:
                                try:
                                    partition_scores[metric] = Evaluator.eval(y_true_arr, y_pred_arr, metric)
                                except Exception:
                                    partition_scores[metric] = None

                        # Store scores in partition sub-dictionary
                        scores[partition] = partition_scores
        else:
            # Single partition: use root y_true/y_pred
            y_true = self.get("y_true", [])
            y_pred = self.get("y_pred", [])

            if len(y_true) > 0 and len(y_pred) > 0:
                y_true_arr = np.array(y_true)
                y_pred_arr = np.array(y_pred)

                if metrics is None:
                    # Get all available metrics using task_type
                    task_type = self.get("task_type", "regression")
                    scores = Evaluator.eval_multi(y_true_arr, y_pred_arr, task_type)
                else:
                    # Get specific metrics
                    for metric in metrics:
                        try:
                            scores[metric] = Evaluator.eval(y_true_arr, y_pred_arr, metric)
                        except Exception:
                            scores[metric] = None

        return scores

    def summary(self) -> str:
        """
        Generate a summary tab report for this prediction.
        Works with both aggregated and non-aggregated prediction results.

        Returns:
            Formatted string with tab report
        """
        # Import tab report manager
        try:
            from nirs4all.utils.tab_report_manager import TabReportManager
        except ImportError:
            return "⚠️ TabReportManager not available"

        # Check if this is an aggregated result (has train/val/test keys)
        has_partitions = all(k in self for k in ["train", "val", "test"])

        if has_partitions:
            # Build partition dictionary with y_true/y_pred and metadata
            best_by_partition = {}
            for partition in ["train", "val", "test"]:
                if partition in self and self[partition] is not None:
                    partition_data = self[partition].copy() if isinstance(self[partition], dict) else {}
                    # Add metadata from root level
                    partition_data['n_features'] = self.get('n_features', 0)
                    partition_data['task_type'] = self.get('task_type', 'regression')
                    best_by_partition[partition] = partition_data
        else:
            # Single partition result - treat as test partition
            partition = self.get('partition', 'test')
            best_by_partition = {
                partition: {
                    'y_true': self.get('y_true', []),
                    'y_pred': self.get('y_pred', []),
                    'n_features': self.get('n_features', 0),
                    'task_type': self.get('task_type', 'regression')
                }
            }

        # Generate tab report using TabReportManager
        formatted_string, _ = TabReportManager.generate_best_score_tab_report(best_by_partition)
        return formatted_string

    def __repr__(self) -> str:
        """String representation showing key info."""
        return f"PredictionResult(id={self.id}, model={self.model_name}, dataset={self.dataset_name}, fold={self.fold_id}, step={self.step_idx}, op={self.op_counter})"

    def __str__(self) -> str:
        """String representation showing key info."""
        return self.__repr__()


class PredictionResultsList(list):
    """
    A list wrapper for PredictionResult objects with additional functionality.
    Provides save(), get() methods and maintains compatibility with standard list operations.
    """

    def __init__(self, predictions: Optional[List[Union[Dict[str, Any], PredictionResult]]] = None):
        """Initialize with optional list of PredictionResult objects."""
        super().__init__(predictions or [])

    def save(self, path: str = "results", filename: Optional[str] = None) -> None:
        """
        Save all predictions to a single CSV file with structured headers.

        CSV Structure:
        - Line 1: dataset_name
        - Line 2: model_classname + model_id
        - Line 3: fold_id
        - Line 4: partition
        - Lines 5+: prediction data (y_true, y_pred columns)

        Args:
            path: Base directory path (default: "results")
            filename: Optional filename (if None, auto-generated from first prediction)
        """
        if not self:
            print("⚠️ No predictions to save")
            return

        # Generate filename if not provided
        if filename is None:
            first_pred = self[0]
            dataset = first_pred.get("dataset_name", "unknown")
            config = first_pred.get("config_name", "unknown")
            filename = f"{dataset}_{config}_predictions.csv"

        # Ensure path directory exists
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        filepath = path_obj / filename

        # Prepare CSV data
        output = io.StringIO()
        writer = csv.writer(output)

        # Collect all columns needed
        all_columns = []

        for pred in self:
            # Check if aggregated (has train/val/test partitions)
            has_partitions = all(k in pred for k in ["train", "val", "test"])

            if has_partitions:
                # Add columns for each partition
                for partition in ["train", "val", "test"]:
                    if partition in pred and pred[partition] is not None:
                        fold_id = pred.get("fold_id", "")
                        fold_suffix = f"_fold{fold_id}" if fold_id and partition in ["train", "val"] else ""
                        all_columns.extend([
                            f"y_true_{partition}{fold_suffix}",
                            f"y_pred_{partition}{fold_suffix}"
                        ])
            else:
                # Single partition columns
                partition = pred.get("partition", "test")
                fold_id = pred.get("fold_id", "")
                fold_suffix = f"_fold{fold_id}" if fold_id else ""
                all_columns.extend([
                    f"y_true_{partition}{fold_suffix}",
                    f"y_pred_{partition}{fold_suffix}"
                ])

        # Remove duplicates while preserving order
        seen = set()
        all_columns = [x for x in all_columns if not (x in seen or seen.add(x))]

        # Write header rows (metadata)
        for pred in self:
            # Row 1: dataset_name
            writer.writerow([pred.get("dataset_name", "unknown")])
            # Row 2: model_classname + model_id
            model_classname = pred.get("model_classname", "unknown")
            model_id = pred.get("id", "unknown")
            writer.writerow([f"{model_classname}_{model_id}"])
            # Row 3: fold_id
            writer.writerow([pred.get("fold_id", "")])
            # Row 4: partition
            writer.writerow([pred.get("partition", "test")])

        # Row 5: Column headers
        writer.writerow(all_columns)

        # Write data rows
        for pred in self:
            has_partitions = all(k in pred for k in ["train", "val", "test"])

            if has_partitions:
                # Collect data from all partitions
                row_data = {}
                for partition in ["train", "val", "test"]:
                    if partition in pred and pred[partition] is not None:
                        partition_data = pred[partition]
                        y_true = partition_data.get("y_true", [])
                        y_pred = partition_data.get("y_pred", [])

                        fold_id = pred.get("fold_id", "")
                        fold_suffix = f"_fold{fold_id}" if fold_id and partition in ["train", "val"] else ""

                        # Store in dict by column name
                        row_data[f"y_true_{partition}{fold_suffix}"] = y_true
                        row_data[f"y_pred_{partition}{fold_suffix}"] = y_pred

                # Find max length
                max_len = max((len(v) for v in row_data.values() if isinstance(v, (list, np.ndarray))), default=0)

                # Write rows
                for i in range(max_len):
                    row = []
                    for col in all_columns:
                        if col in row_data:
                            data = row_data[col]
                            row.append(data[i] if i < len(data) else "")
                        else:
                            row.append("")
                    writer.writerow(row)
            else:
                # Single partition
                y_true = pred.get("y_true", [])
                y_pred = pred.get("y_pred", [])

                partition = pred.get("partition", "test")
                fold_id = pred.get("fold_id", "")
                fold_suffix = f"_fold{fold_id}" if fold_id else ""

                col_true = f"y_true_{partition}{fold_suffix}"
                col_pred = f"y_pred_{partition}{fold_suffix}"

                max_len = max(len(y_true), len(y_pred))

                for i in range(max_len):
                    row = []
                    for col in all_columns:
                        if col == col_true:
                            row.append(y_true[i] if i < len(y_true) else "")
                        elif col == col_pred:
                            row.append(y_pred[i] if i < len(y_pred) else "")
                        else:
                            row.append("")
                    writer.writerow(row)

        # Write to file
        with open(filepath, 'w', newline='') as f:
            f.write(output.getvalue())

        output.close()
        print(f"💾 Saved {len(self)} predictions to {filepath}")

    def get(self, prediction_id: str) -> Optional[PredictionResult]:
        """
        Get a prediction by its ID.

        Args:
            prediction_id: The ID of the prediction to retrieve

        Returns:
            PredictionResult if found, None otherwise
        """
        for pred in self:
            if pred.get("id") == prediction_id:
                return pred
        return None

    def __repr__(self) -> str:
        """String representation showing count and brief info."""
        if not self:
            return "PredictionResultsList([])"
        return f"PredictionResultsList({len(self)} predictions)"


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
    - train_score: Optional[float]
    - metric: str
    - task_type: str
    - n_samples: int
    - n_features: int
    """

    def __init__(self, filepath: Optional[str] = None):
        """Initialize Predictions storage with Polars DataFrame backend."""
        self._df = pl.DataFrame(schema={
            "id": pl.Utf8,
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
            "train_score": pl.Float64,
            "metric": pl.Utf8,
            "task_type": pl.Utf8,
            "n_samples": pl.Int64,
            "n_features": pl.Int64,
            "preprocessings": pl.Utf8,
            "best_params": pl.Utf8,  # JSON string
        })

        if filepath and Path(filepath).exists():
            self.load_from_file(filepath)

    @staticmethod
    def _generate_hash(row_dict: Dict[str, Any]) -> str:
        """
        Generate a 6-character hash from row dictionary.
        Uses key identifying fields to create a diverse, reproducible hash.
        """
        # Select key fields that uniquely identify a prediction
        key_fields = [
            str(row_dict.get('dataset_name', '')),
            str(row_dict.get('config_name', '')),
            str(row_dict.get('model_name', '')),
            str(row_dict.get('fold_id', '')),
            str(row_dict.get('step_idx', 0)),
            str(row_dict.get('op_counter', 0)),
        ]

        # Create a string to hash
        hash_string = '|'.join(key_fields)

        # Generate SHA256 hash and take first 6 characters (alphanumeric)
        hash_obj = hashlib.sha256(hash_string.encode('utf-8'))
        hex_hash = hash_obj.hexdigest()

        # Convert to base36 (0-9, a-z) for better diversity in 6 chars
        # Take multiple segments of the hex and combine them
        hash_int = int(hex_hash[:16], 16)  # Use first 16 hex chars

        # Convert to base36 and take 6 characters
        base36_chars = '0123456789abcdefghijklmnopqrstuvwxyz'
        result = ''
        temp = hash_int

        for _ in range(6):
            result = base36_chars[temp % 36] + result
            temp //= 36

        return result

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
            y_true: True values array (optional, can be None for prediction-only mode)
            y_pred: Predicted values array (required)
            filepath: Output CSV file path
            prefix: Optional prefix for column names
            suffix: Optional suffix for column names
        """
        if y_pred is None:
            raise ValueError("y_pred is required")

        # Convert to numpy arrays if needed and flatten
        y_pred_arr = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred
        y_pred_flat = y_pred_arr.flatten()

        data_dict = {}
        pred_col = f"{prefix}y_pred{suffix}"
        data_dict[pred_col] = y_pred_flat.tolist()

        # Handle y_true if provided
        if y_true is not None:
            y_true_arr = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
            y_true_flat = y_true_arr.flatten()

            if len(y_true_flat) != len(y_pred_flat):
                raise ValueError(f"Length mismatch after flattening: y_true ({len(y_true_flat)}) != y_pred ({len(y_pred_flat)})")

            true_col = f"{prefix}y_true{suffix}"
            data_dict[true_col] = y_true_flat.tolist()

        # Create DataFrame and save to CSV
        import polars as pl
        df_csv = pl.DataFrame(data_dict)        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df_csv.write_csv(filepath)
        print(f"💾 Saved predictions to {filepath}")

    @staticmethod
    def save_all_to_csv(predictions: 'Predictions', path: str = "results", aggregate_partitions: bool = False, **filters) -> None:
        """
        Save all predictions to CSV files.

        Args:
            predictions: Predictions instance
            path: Base path for saving (default: "results")
            aggregate_partitions: If True, save one file per model with all partitions aggregated
                                 If False, save one file per individual prediction
            **filters: Additional filter criteria to apply before saving
        """
        if aggregate_partitions:
            # Save one file per individual model/fold with all partitions aggregated
            # Use group_by_fold=True to keep individual folds separate
            all_results = predictions.top(
                n=predictions.num_predictions,
                aggregate_partitions=True,
                group_by_fold=True,  # Include fold_id to keep individual folds
                **filters
            )

            # No need for deduplication since group_by_fold=True keeps them separate
            for result in all_results:
                try:
                    result.save_to_csv(path=path)
                except Exception as e:
                    model_id = result.get('id', 'unknown')
                    print(f"⚠️ Failed to save prediction {model_id}: {e}")

            print(f"✅ Saved {len(all_results)} aggregated model files to {path}")
        else:
            # Save one file per individual prediction (each partition/fold separately)
            all_results = predictions.top(
                n=predictions.num_predictions,
                aggregate_partitions=False,
                group_by_fold=True,  # Include fold in grouping for individual saves
                **filters
            )

            for result in all_results:
                try:
                    result.save_to_csv(path=path)
                except Exception as e:
                    model_id = result.get('id', 'unknown')
                    print(f"⚠️ Failed to save prediction {model_id}: {e}")

            print(f"✅ Saved {len(all_results)} individual prediction files to {path}")

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
        """Add a new prediction to the storage."""

        # Convert numpy arrays to lists for JSON serialization
        y_true_list = y_true.tolist() if y_true is not None else []
        y_pred_list = y_pred.tolist() if y_pred is not None else []
        sample_indices_list = sample_indices if sample_indices is not None else []
        weights_list = weights if weights is not None else []
        metadata_dict = metadata if metadata is not None else {}
        best_params_dict = best_params if best_params is not None else {}
        fold_id = str(fold_id)

        # Create row_dict with columns in schema order
        row_dict = {
            "id": "",  # Will be filled by hash generation
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
            "weights": json.dumps([w.tolist() if isinstance(w, np.ndarray) else w for w in weights_list]),
            "metadata": json.dumps(metadata_dict),
            "partition": partition,
            "y_true": json.dumps(y_true_list),
            "y_pred": json.dumps(y_pred_list),
            "val_score": val_score,
            "test_score": test_score,
            "train_score": train_score,
            "metric": metric,
            "task_type": task_type,
            "n_samples": n_samples,
            "n_features": n_features,
            "preprocessings": preprocessings,
            "best_params": json.dumps(best_params_dict),
        }

        # Generate unique ID hash for the prediction
        prediction_id = self._generate_hash(row_dict)
        row_dict["id"] = prediction_id
        new_row = pl.DataFrame([row_dict])
        self._df = pl.concat([self._df, new_row])

        return row_dict["id"]

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
        train_score: Union[Optional[float], List[Optional[float]]] = None,
        metric: Union[str, List[str]] = "mse",
        task_type: Union[str, List[str]] = "regression",
        n_samples: Union[int, List[int]] = 0,
        n_features: Union[int, List[int]] = 0,
        preprocessings: Union[str, List[str]] = "",
        best_params: Union[Optional[Dict[str, Any]], List[Optional[Dict[str, Any]]]] = None
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
            train_score: Training score(s) - can be single float or list
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
            'train_score': train_score,
            'metric': metric,
            'task_type': task_type,
            'n_samples': n_samples,
            'n_features': n_features,
            'preprocessings': preprocessings,
            'best_params': best_params,
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
            row["best_params"] = json.loads(row["best_params"]) if row["best_params"] else {}
            row["y_true"] = np.array(json.loads(row["y_true"]))
            row["y_pred"] = np.array(json.loads(row["y_pred"]))
            results.append(row)

        return results

    def get_similar(self, **filter_kwargs) -> Optional[Dict[str, Any]]:
        """
        Get the first prediction similar to the provided filter criteria.

        Args:
            **filter_kwargs: Filter criteria (same as filter_predictions)

        Returns:
            First matching prediction as dictionary, or None if no matches found
        """
        results = self.filter_predictions(**filter_kwargs)
        return results[0] if results else None

    @property
    def num_predictions(self) -> int:
        """Get the number of stored predictions."""
        return len(self._df)

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
                # print(f"📥 Loaded {len(self._df)} predictions from {filepath}")

        except Exception as e:
            pass
            # print(f"⚠️ Error loading predictions from {filepath}: {e}")

    @classmethod
    def load_from_file_cls(cls, filepath: str) -> 'Predictions':
        """Load predictions from JSON file as class method."""
        instance = cls()
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
        Load predictions from JSON files in the results directory structure.

        Args:
            dataset_name: Name of dataset to load (if None, loads all datasets in path)
            path: Base path to search for predictions (default: "results")
                  If path points to a JSON file directly, loads it without going into /dataset_name
            aggregate_partitions: If True, aggregate y_pred and y_true from all partitions
                                 with the same id (similar to top() function)
            **filters: Additional filter criteria to apply after loading

        Returns:
            Predictions instance with loaded data

        Examples:
            # Load all predictions from all datasets
            predictions = Predictions.load()

            # Load predictions for a specific dataset
            predictions = Predictions.load(dataset_name="my_dataset")

            # Load from a specific JSON file
            predictions = Predictions.load(path="results/my_dataset/predictions.json")

            # Load and filter by model name
            predictions = Predictions.load(dataset_name="my_dataset", model_name="PLS_1")

            # Load with partition aggregation
            predictions = Predictions.load(dataset_name="my_dataset", aggregate_partitions=True)
        """
        instance = cls()
        base_path = Path(path)

        # Case 1: path is a JSON file, load it directly
        if base_path.is_file() and base_path.suffix == '.json':
            instance.load_from_file(str(base_path))
            print(f"📥 Loaded {len(instance._df)} predictions from {base_path}")

        # Case 2: path is a directory
        elif base_path.is_dir():
            # If dataset_name is specified, load only that dataset
            if dataset_name:
                dataset_path = base_path / dataset_name / "predictions.json"
                if dataset_path.exists():
                    temp_instance = cls()
                    temp_instance.load_from_file(str(dataset_path))
                    instance.merge_predictions(temp_instance)
                    print(f"📥 Loaded {len(temp_instance._df)} predictions from {dataset_name}")
                else:
                    print(f"⚠️ No predictions.json found for dataset '{dataset_name}' at {dataset_path}")

            # If dataset_name is None, browse all datasets in path
            else:
                # Find all predictions.json files in subdirectories
                predictions_files = list(base_path.glob("*/predictions.json"))

                if not predictions_files:
                    print(f"⚠️ No predictions.json files found in {base_path}")
                else:
                    for pred_file in predictions_files:
                        dataset_name_from_path = pred_file.parent.name
                        temp_instance = cls()
                        temp_instance.load_from_file(str(pred_file))
                        instance.merge_predictions(temp_instance)
                        print(f"📥 Loaded {len(temp_instance._df)} predictions from dataset '{dataset_name_from_path}'")

                    print(f"✅ Total loaded: {len(instance._df)} predictions from {len(predictions_files)} datasets")

        else:
            print(f"⚠️ Path '{base_path}' does not exist or is not accessible")
            return instance

        # Apply filters if provided using existing filter on DataFrame
        if filters:
            # Build filter expressions for polars
            filter_exprs = []
            for key, value in filters.items():
                if key in instance._df.columns:
                    filter_exprs.append(pl.col(key) == value)

            if filter_exprs:
                instance._df = instance._df.filter(filter_exprs)
                print(f"🔍 Filtered to {len(instance._df)} predictions matching criteria: {filters}")

        # Apply partition aggregation if requested using existing _add_partition_data
        if aggregate_partitions and len(instance._df) > 0:
            # Get test partition predictions as base (one per model)
            test_predictions = instance.filter_predictions(partition="test")

            if test_predictions:
                # Add partition data using existing method
                aggregated = instance._add_partition_data(test_predictions, ["train", "val", "test"])
                print(f"📦 Aggregated {len(aggregated)} models with partition data")

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

    def _parse_vec_json(self, s: str) -> np.ndarray:
        """Parse JSON string to numpy array."""
        return np.asarray(json.loads(s), dtype=float)

    def _rank_score_expr(self, rank_metric: str, rank_partition: str):
        """
        Create expression to compute or retrieve rank score.
        If rank_metric matches the row's metric, use precomputed score.
        Otherwise, recompute from y_true/y_pred.
        """
        score_col = f"{rank_partition}_score"
        return (
            pl.when(pl.col("metric") == rank_metric)
              .then(pl.col(score_col))
              .otherwise(
                pl.struct(["y_true", "y_pred"]).map_elements(
                    lambda s: Evaluator.eval(
                        self._parse_vec_json(s["y_true"]),
                        self._parse_vec_json(s["y_pred"]),
                        rank_metric
                    ),
                    return_dtype=pl.Float64
                )
            )
            .alias("rank_score")
        )

    def top(
        self, n: int,
        rank_metric: str = "",
        rank_partition: str = "val",
        display_metrics: list[str] = None,
        display_partition: str = "test",
        aggregate_partitions: bool = False,
        ascending: bool = True,
        group_by_fold: bool = False,
        **filters
    ):
        """
        Get top n models ranked by a metric on a specific partition.

        Args:
            n: Number of top models to return
            rank_metric: Metric to rank by (if empty, uses record's metric or val_score)
            rank_partition: Partition to rank on (default: "val")
            display_metrics: Metrics to compute (default: task_type defaults)
            display_partition: Partition to display (default: "test")
            aggregate_partitions: If True, add train/val/test nested dicts
            ascending: If True, lower scores rank higher
            group_by_fold: If True, include fold_id in model identity
            **filters: Additional filter criteria
        """
        # Apply filters (excluding partition)
        _ = filters.pop("partition", None)
        base = self._df.filter([pl.col(k) == v for k, v in filters.items()]) if filters else self._df
        if base.height == 0:
            return PredictionResultsList([])

        # Default rank_metric from data if not provided
        if rank_metric == "":
            rank_metric = base[0, "metric"]

        # Model identity key
        KEY = ["config_name", "step_idx", "model_name"]
        if group_by_fold:
            KEY.append("fold_id")

        # 1) RANKING: Filter to rank_partition and compute scores
        rank_data = base.filter(pl.col("partition") == rank_partition)
        if rank_data.height == 0:
            return PredictionResultsList([])

        # Compute rank score: use val_score if rank_metric matches record's metric, else compute
        rank_scores = []
        for row in rank_data.to_dicts():
            if rank_metric == row["metric"]:
                # Use precomputed val_score
                score = row.get("val_score")
            else:
                # Compute metric from y_true/y_pred
                try:
                    y_true = self._parse_vec_json(row["y_true"])
                    y_pred = self._parse_vec_json(row["y_pred"])
                    score = Evaluator.eval(y_true, y_pred, rank_metric)
                except:
                    score = None

            rank_scores.append({
                **{k: row[k] for k in KEY},
                "rank_score": score,
                "id": row["id"],
                "fold_id": row["fold_id"]  # Always include fold_id for data retrieval
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
                "rank_id": top_key["id"],  # ID of the record used for ranking
                "fold_id": top_key.get("fold_id")  # Add fold_id to top level
            })

            if aggregate_partitions:
                # Add nested structure for all partitions
                for partition in ["train", "val", "test"]:
                    # Filter by model AND the specific fold_id from ranking
                    partition_data = base.filter(
                        pl.col("partition") == partition
                    ).filter([pl.col(k) == v for k, v in model_filter.items()])

                    # Also filter by fold_id to get the correct fold's data
                    if top_key.get("fold_id") is not None:
                        partition_data = partition_data.filter(pl.col("fold_id") == top_key["fold_id"])

                    if partition_data.height > 0:
                        row = partition_data.to_dicts()[0]
                        y_true = self._parse_vec_json(row["y_true"])
                        y_pred = self._parse_vec_json(row["y_pred"])

                        partition_dict = {
                            "y_true": y_true.tolist(),
                            "y_pred": y_pred.tolist(),
                            # Include all original scores (do NOT recompute)
                            "train_score": row.get("train_score"),
                            "val_score": row.get("val_score"),
                            "test_score": row.get("test_score"),
                            # Preserve fold_id for CSV column naming
                            "fold_id": row.get("fold_id")
                        }

                        # Add metadata to result from TEST partition (reference partition)
                        if partition == "test":
                            # Determine partition name: use "test" for aggregate mode (multiple partitions)
                            partition_name = "test" if aggregate_partitions else display_partition
                            result.update({
                                # DON'T overwrite the id - keep the rank_id which has the correct ID
                                "partition": partition_name,
                                "dataset_name": row.get("dataset_name"),
                                "dataset_path": row.get("dataset_path"),
                                "config_path": row.get("config_path"),
                                "model_classname": row.get("model_classname"),
                                "model_path": row.get("model_path"),
                                "fold_id": row.get("fold_id"),  # Add fold_id explicitly
                                "op_counter": row.get("op_counter"),  # Add op_counter for pred_short_string
                                "sample_indices": json.loads(row.get("sample_indices", "[]")),
                                "weights": json.loads(row.get("weights", "[]")),
                                "metadata": json.loads(row.get("metadata", "{}")),
                                "metric": row.get("metric"),
                                "task_type": row.get("task_type", "regression"),
                                "n_samples": row.get("n_samples"),
                                "n_features": row.get("n_features"),
                                "preprocessings": row.get("preprocessings"),
                                "best_params": json.loads(row.get("best_params", "{}")),
                                # Include all original scores in main result from TEST partition
                                "train_score": row.get("train_score"),
                                "val_score": row.get("val_score"),
                                "test_score": row.get("test_score")
                            })
                            # Add the correct ID from ranking after metadata update
                            result["id"] = result["rank_id"]

                        # Add display metrics using STORED scores, not recomputed
                        if display_metrics:
                            for metric in display_metrics:
                                # Use stored score if available, otherwise compute
                                stored_score_key = f"{partition}_score" if partition != "val" else "val_score"
                                if metric == row.get("metric"):
                                    # Use the precomputed score from storage
                                    partition_dict[metric] = row.get(stored_score_key)
                                else:
                                    # Compute other requested metrics
                                    try:
                                        score = Evaluator.eval(y_true, y_pred, metric)
                                        partition_dict[metric] = score
                                    except:
                                        partition_dict[metric] = None

                        result[partition] = partition_dict
            else:
                # Single partition display
                # Filter by model AND the specific fold_id from ranking
                display_data = base.filter(
                    pl.col("partition") == display_partition
                ).filter([pl.col(k) == v for k, v in model_filter.items()])

                # Also filter by fold_id to get the correct fold's data
                if top_key.get("fold_id") is not None:
                    display_data = display_data.filter(pl.col("fold_id") == top_key["fold_id"])

                if display_data.height > 0:
                    row = display_data.to_dicts()[0]
                    y_true = self._parse_vec_json(row["y_true"])
                    y_pred = self._parse_vec_json(row["y_pred"])

                    result.update({
                        # Keep the rank_id which has the correct ID from ranking
                        "partition": display_partition,  # The partition being displayed
                        "dataset_name": row.get("dataset_name"),
                        "dataset_path": row.get("dataset_path"),
                        "config_path": row.get("config_path"),
                        "model_classname": row.get("model_classname"),
                        "model_path": row.get("model_path"),
                        "fold_id": row.get("fold_id"),  # Add fold_id explicitly
                        "op_counter": row.get("op_counter"),  # Add op_counter for pred_short_string
                        "sample_indices": json.loads(row.get("sample_indices", "[]")),
                        "weights": json.loads(row.get("weights", "[]")),
                        "metadata": json.loads(row.get("metadata", "{}")),
                        "metric": row.get("metric"),
                        "task_type": row.get("task_type", "regression"),
                        "n_samples": row.get("n_samples"),
                        "n_features": row.get("n_features"),
                        "preprocessings": row.get("preprocessings"),
                        "best_params": json.loads(row.get("best_params", "{}")),
                        "y_true": y_true.tolist(),
                        "y_pred": y_pred.tolist(),
                        # Include all original scores
                        "train_score": row.get("train_score"),
                        "val_score": row.get("val_score"),
                        "test_score": row.get("test_score")
                    })
                    # Set the correct ID from ranking
                    result["id"] = result["rank_id"]

                    # Add display metrics using STORED scores when possible
                    if display_metrics:
                        for metric in display_metrics:
                            # Use stored score if it matches the record's metric
                            if metric == row.get("metric"):
                                # Use the precomputed score from storage
                                stored_score_key = f"{display_partition}_score" if display_partition != "val" else "val_score"
                                result[metric] = row.get(stored_score_key)
                            else:
                                # Compute other requested metrics
                                try:
                                    score = Evaluator.eval(y_true, y_pred, metric)
                                    result[metric] = score
                                except:
                                    result[metric] = None

            results.append(result)

        return PredictionResultsList(results)


    def top_k(self, k: int = 5, metric: str = "", ascending: bool = True, aggregate_partitions: List[str] = [], **filters) -> List[Union[Dict[str, Any], 'PredictionResult']]:
        """
        Get top K predictions ranked by metric, val_score, or test_score.
        By default filters to test partition unless otherwise specified.

        Args:
            metric: Metric name to rank by ("" for test_score, "loss" for val_score, else calculate metric on-the-fly)
            k: Number of top results to return (-1 to return all filtered predictions)
            ascending: If True, lower scores rank higher (for error metrics)
            aggregate_partitions: List of partitions to aggregate y_true and y_pred from (e.g. ['train', 'val'])
            **filters: Additional filter criteria

        Returns:
            List of top K prediction dictionaries (or all if k=-1)
        """
        # Add default partition filter if not specified
        if 'partition' not in filters:
            filters['partition'] = 'val'
        if 'partition' in filters and filters['partition'] in ['all', 'ALL', 'All', '_all_', '']:
            del filters['partition']

        # First filter the entries
        df_filtered = self._df
        for key, value in filters.items():
            if key in df_filtered.columns:
                df_filtered = df_filtered.filter(pl.col(key) == value)

        # print( f"🔍 Found {len(df_filtered)} predictions after filtering with criteria: {filters}")

        if df_filtered.is_empty():
            return PredictionResultsList([])

        # Handle different ranking scenarios
        if metric == "" or metric == "loss":
            # Use existing stored scores
            rank_col = "val_score"
            df_ranked = df_filtered.filter(pl.col(rank_col).is_not_null())

            if df_ranked.is_empty():
                return PredictionResultsList([])

            df_sorted = df_ranked.sort(rank_col, descending=not ascending)

            # Return all results if k=-1, otherwise return top k
            if k == -1:
                top_k_rows = df_sorted
            else:
                top_k_rows = df_sorted.head(k)

            # Convert to list of PredictionResult with JSON deserialization
            results = []
            for row in top_k_rows.to_dicts():
                # Deserialize JSON fields
                row["sample_indices"] = json.loads(row["sample_indices"])
                row["weights"] = json.loads(row["weights"]) if row["weights"] else []
                row["metadata"] = json.loads(row["metadata"]) if row["metadata"] else {}
                row["best_params"] = json.loads(row["best_params"]) if row["best_params"] else {}
                row["y_true"] = np.array(json.loads(row["y_true"]))
                row["y_pred"] = np.array(json.loads(row["y_pred"]))
                results.append(PredictionResult(row))

            # Add partition data if requested
            if len(aggregate_partitions) > 0:
                results = self._add_partition_data(results, aggregate_partitions)
            return PredictionResultsList(results)

        else:
            # Calculate metric on-the-fly for all filtered entries
            scores_data = []

            # Import the external evaluator
            try:
                from ..utils.model_utils import ModelUtils, TaskType
                model_utils = ModelUtils()
            except ImportError:
                print("⚠️ Cannot import ModelUtils for score calculation")
                return PredictionResultsList([])

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
                        score_record["best_params"] = json.loads(row["best_params"]) if row["best_params"] else {}
                        score_record["y_true"] = y_true
                        score_record["y_pred"] = y_pred

                        scores_data.append(score_record)

            # Sort by computed metric and return top k (or all if k=-1)
            if not scores_data:
                return PredictionResultsList([])
            if ModelUtils._is_higher_better(metric):
                ascending = not ascending  # Reverse for higher is better
            scores_data.sort(key=lambda x: x["computed_score"], reverse=not ascending)

            # Return all results if k=-1, otherwise return top k
            if k == -1:
                results = [PredictionResult(r) for r in scores_data]
            else:
                results = [PredictionResult(r) for r in scores_data[:k]]
            # Add partition data if requested
            if len(aggregate_partitions) > 0:
                results = self._add_partition_data(results, aggregate_partitions)

            return PredictionResultsList(results)  # type: ignore

    def _add_partition_data(self, results: List[Union[Dict[str, Any], 'PredictionResult']], aggregate_partitions: List[str]) -> List[Union[Dict[str, Any], 'PredictionResult']]:
        """
        Add y_true and y_pred data from all partitions to each result using simple filtering.

        Args:
            results: List of prediction results

        Returns:
            Results with added partition structure: train: {y_true: ..., y_pred: ...}, val: {...}, test: {...}
        """
        for result in results:
            # For each partition, filter once and get the data
            for partition in aggregate_partitions:
                partition_data = self.filter_predictions(
                    dataset_name=result['dataset_name'],
                    config_name=result['config_name'],
                    model_name=result['model_name'],
                    fold_id=result['fold_id'],
                    step_idx=result['step_idx'],
                    op_counter=result['op_counter'],
                    partition=partition
                )

                if partition_data:
                    # Found data for this partition
                    result[partition] = {
                        'y_true': partition_data[0]['y_true'],
                        'y_pred': partition_data[0]['y_pred']
                    }
                else:
                    # No data for this partition
                    result[partition] = {
                        'y_true': np.array([]),
                        'y_pred': np.array([])
                    }

        return results

    def get_best(self, metric: str = "", ascending: bool = True, aggregate_partitions: List[str] = [], **filters) -> Optional[Union[Dict[str, Any], 'PredictionResult']]:
        """
        Get the best prediction for a specific metric, val_score, or test_score.
        This is an alias for top_k with k=1.

        Args:
            metric: Metric name to optimize ("" for test_score, "loss" for val_score, else metric)
            ascending: If True, lower scores are better (for error metrics)
            aggregate_partitions: If True, add y_true and y_pred for all partitions (train, val, test)
            **filters: Additional filter criteria

        Returns:
            Best prediction dictionary or None
        """
        top_results = self.top_k(k=1, metric=metric, ascending=ascending, aggregate_partitions=aggregate_partitions, **filters)
        return top_results[0] if top_results else None

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
                (len(self_schema) == len(other_schema))
                and all(col in other_schema and self_schema[col] == other_schema[col] for col in self_schema)
                and (list(self._df.columns) == list(other._df.columns))
            )

            if schemas_match:
                # Schemas match, safe to concatenate directly
                self._df = pl.concat([self._df, other._df], how="vertical")
            else:
                # Schemas don't match, need to align them
                # print(f"⚠️ Schema mismatch detected, aligning schemas before merge")

                # Use the predefined schema order from __init__ to ensure consistency
                predefined_order = [
                    "id", "dataset_name", "dataset_path", "config_name", "config_path",
                    "step_idx", "op_counter", "model_name", "model_classname", "model_path",
                    "fold_id", "sample_indices", "weights", "metadata", "partition",
                    "y_true", "y_pred", "val_score", "test_score", "metric", "task_type",
                    "n_samples", "n_features", "preprocessings", "best_params"
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

        # Define key columns for duplicate detection (excluding 'id' since it's auto-generated)
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

    def get_entry_partitions(self, entry):
        res = {}
        filter = {
            'dataset_name': entry['dataset_name'],
            'config_name': entry['config_name'],
            'model_name': entry['model_name'],
            'fold_id': entry['fold_id'],
            'step_idx': entry['step_idx'],
            'op_counter': entry['op_counter']
        }

        for partition in ['train', 'val', 'test']:
            filter['partition'] = partition
            predictions = self.filter_predictions(**filter)
            if not predictions or len(predictions) == 0:
                print(f"⚠️ No predictions found for {filter}")
                res[partition] = None
            else:
                res[partition] = predictions[0]
        return res

    @classmethod
    def pred_short_string(cls, entry, metrics=None):
        scores_str = ""
        metrics.remove('rmse') if metrics is not None and 'rmse' in metrics else None
        if metrics is not None:
            scores = Evaluator.eval_list(entry['y_true'], entry['y_pred'], metrics=metrics)
            scores_str = ", ".join([f"[{k}:{v:.4f}]" if k != 'rmse' else f"[{k}:{v:.4f}]" for k, v in zip(metrics, scores)])

        short_desc = f"{entry['model_name']} - {entry['metric']} [test: {entry['test_score']:.4f}], [val: {entry['val_score']:.4f}]"
        short_desc += f", {scores_str}"
        short_desc += f", (fold: {entry['fold_id']}, id: {entry['op_counter']}, step: {entry['step_idx']}) - [{entry['id']}]"
        return short_desc

    @classmethod
    def pred_long_string(cls, entry, metrics=None):  # ADAPT TO CLASSIFICATION
        return Predictions.pred_short_string(entry, metrics=metrics) + f" | [{entry['config_name']}]"

    def to_numpy(self) -> np.ndarray:
        """
        Get all predictions as a numpy array.

        Returns:
            List of all prediction dictionaries
        """
        return self._df.to_numpy()

    def to_list(self) -> List[Dict[str, Any]]:
        """
        Get all predictions as a list of dictionaries.

        Returns:
            List of all prediction dictionaries
        """
        return self.to_numpy().tolist()

    def to_dicts(self) -> List[Dict[str, Any]]:
        """
        Get all predictions as a list of dictionaries.

        Returns:
            List of all prediction dictionaries
        """
        return self._df.to_dicts()

    def to_pandas(self) -> 'pd.DataFrame':
        """
        Get all predictions as a pandas DataFrame.

        Returns:
            Pandas DataFrame of all predictions
        """
        return self._df.to_pandas()

