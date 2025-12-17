"""
User-facing result container classes for predictions.

This module provides PredictionResult and PredictionResultsList classes
that extend standard Python dict/list with prediction-specific functionality.
"""

import csv
import io
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np
import polars as pl

from nirs4all.core.logging import get_logger
from nirs4all.core import metrics as evaluator

logger = get_logger(__name__)


class PredictionResult(dict):
    """
    Enhanced dictionary for a single prediction with convenience methods.

    Extends standard dict with property accessors and methods for saving,
    evaluating, and summarizing predictions.

    Features:
        - Property accessors (id, model_name, dataset_name, etc.)
        - save_to_csv() - save individual result
        - eval_score() - compute metrics on-the-fly
        - summary() - generate tab report

    Examples:
        >>> result = PredictionResult({
        ...     "id": "abc123",
        ...     "dataset_name": "wheat",
        ...     "model_name": "PLS",
        ...     "y_true": [1, 2, 3],
        ...     "y_pred": [1.1, 2.2, 3.3]
        ... })
        >>> result.model_name
        'PLS'
        >>> scores = result.eval_score(["rmse", "r2"])
        >>> result.save_to_csv("results")
    """

    @property
    def id(self) -> str:
        """Get prediction ID."""
        return self.get("id", "unknown")

    @property
    def fold_id(self) -> str:
        """Get fold ID."""
        return self.get("fold_id", "unknown")

    @property
    def dataset_name(self) -> str:
        """Get dataset name."""
        return self.get("dataset_name", "unknown")

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.get("model_name", "unknown")

    @property
    def step_idx(self) -> int:
        """Get pipeline step index."""
        return self.get("step_idx", 0)

    @property
    def op_counter(self) -> int:
        """Get operation counter."""
        return self.get("op_counter", 0)

    @property
    def config_name(self) -> str:
        """Get config name."""
        return self.get("config_name", "unknown")

    def save_to_csv(self, path_or_file: str = "results", filename: Optional[str] = None) -> None:
        """
        Save prediction result to CSV file.

        Args:
            path_or_file: Base path (folder) or complete file path (if ends with .csv)
            filename: Optional filename (if path_or_file is a folder)

        Examples:
            >>> result.save_to_csv("output")  # Saves to output/{dataset}/{id}.csv
            >>> result.save_to_csv("output/my_result.csv")  # Saves to output/my_result.csv
            >>> result.save_to_csv("output", "my_result.csv")  # Saves to output/my_result.csv
        """
        destinations = []
        path_obj = Path(path_or_file)

        # Check if path_or_file looks like a file (has extension .csv)
        is_file_1 = path_obj.suffix.lower() == '.csv'

        if is_file_1:
            destinations.append(path_obj)
            # If filename is also provided and looks like a file, save there too
            if filename:
                file_obj = Path(filename)
                if file_obj.suffix.lower() == '.csv':
                    destinations.append(file_obj)
        else:
            # path_or_file is a directory
            base_dir = path_obj

            if filename:
                # filename provided
                destinations.append(base_dir / filename)
            else:
                # Auto-generate filename
                dataset_name = self.get("dataset_name", "unknown")
                model_id = self.get("id", "unknown")
                destinations.append(base_dir / dataset_name / f"{model_id}.csv")

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

            # Check if arrays exist and have length (avoid ambiguous truth value with numpy arrays)
            has_y_true = y_true is not None and len(y_true) > 0
            has_y_pred = y_pred is not None and len(y_pred) > 0
            max_len = max(len(y_true) if has_y_true else 0, len(y_pred) if has_y_pred else 0)

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

            for filepath in destinations:
                # Create directory if it doesn't exist
                filepath.parent.mkdir(parents=True, exist_ok=True)
                df_csv.write_csv(str(filepath))
                logger.info(f"Saved prediction result to {filepath}")
        else:
            logger.warning("No prediction data found to save")

    def eval_score(self, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate scores for this prediction using specified metrics.

        Args:
            metrics: List of metrics to compute (if None, returns all available metrics)

        Returns:
            Dictionary of metric names to scores.
            For aggregated results: {"train": {...}, "val": {...}, "test": {...}}
            For single partition: {"rmse": ..., "r2": ..., ...}

        Examples:
            >>> scores = result.eval_score(["rmse", "r2", "mae"])
            >>> # For aggregated: scores = {"train": {"rmse": 0.5}, "val": {...}, "test": {...}}
            >>> # For single: scores = {"rmse": 0.5, "r2": 0.9}
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
                            partition_scores = evaluator.eval_multi(y_true_arr, y_pred_arr, task_type)
                        else:
                            # Get specific metrics
                            partition_scores = {}
                            for metric in metrics:
                                try:
                                    partition_scores[metric] = evaluator.eval(y_true_arr, y_pred_arr, metric)
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
                    scores = evaluator.eval_multi(y_true_arr, y_pred_arr, task_type)
                else:
                    # Get specific metrics
                    for metric in metrics:
                        try:
                            scores[metric] = evaluator.eval(y_true_arr, y_pred_arr, metric)
                        except Exception:
                            scores[metric] = None

        return scores

    def summary(self) -> str:
        """
        Generate a summary tab report for this prediction.

        Works with both aggregated and non-aggregated prediction results.

        Returns:
            Formatted string with tab report

        Examples:
            >>> report = result.summary()
            >>> print(report)
        """
        # Import tab report manager
        try:
            from nirs4all.visualization.reports import TabReportManager
        except ImportError:
            return f"{WARNING}TabReportManager not available"

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
    List container for PredictionResult objects with batch operations.

    Extends standard list with prediction-specific batch functionality.

    Features:
        - save() - batch CSV export
        - get() - retrieve by ID
        - filter() - chain filtering
        - Iterator support

    Examples:
        >>> results = PredictionResultsList([result1, result2, result3])
        >>> results.save("output/predictions.csv")
        >>> best = results.get("abc123")
        >>> len(results)
        3
    """

    def __init__(self, predictions: Optional[List[Union[Dict[str, Any], PredictionResult]]] = None):
        """
        Initialize with optional list of PredictionResult objects.

        Args:
            predictions: List of predictions (dicts or PredictionResult objects)
        """
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

        Examples:
            >>> results.save("output")
            >>> results.save("output", "my_predictions.csv")
        """
        if not self:
            logger.warning("No predictions to save")
            return

        # Generate filename if not provided
        if filename is None:
            first_pred = self[0]
            dataset_name = first_pred.get("dataset_name", "unknown")
            model_name = first_pred.get("model_name", "unknown")
            filename = f"{dataset_name}_{model_name}_predictions.csv"

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
            # Check if this is an aggregated result (has train/val/test keys)
            has_partitions = all(k in pred for k in ["train", "val", "test"])

            if has_partitions:
                for partition in ["train", "val", "test"]:
                    if partition in pred and pred[partition] is not None:
                        partition_data = pred[partition]
                        fold_id = partition_data.get("fold_id", pred.get("fold_id", ""))
                        fold_suffix = f"_fold{fold_id}" if fold_id and partition in ["train", "val"] else ""
                        all_columns.append(f"y_true_{partition}{fold_suffix}")
                        all_columns.append(f"y_pred_{partition}{fold_suffix}")
            else:
                if "y_true" not in all_columns:
                    all_columns.append("y_true")
                if "y_pred" not in all_columns:
                    all_columns.append("y_pred")

        # Remove duplicates while preserving order
        seen = set()
        all_columns = [x for x in all_columns if not (x in seen or seen.add(x))]

        # Write header rows (metadata)
        for pred in self:
            writer.writerow(['dataset_name', pred.get('dataset_name', '')])
            writer.writerow(['model_name', pred.get('model_name', '')])
            writer.writerow(['fold_id', pred.get('fold_id', '')])
            writer.writerow(['partition', pred.get('partition', '')])
            break  # Only write once

        # Row 5: Column headers
        writer.writerow(all_columns)

        # Write data rows
        for pred in self:
            has_partitions = all(k in pred for k in ["train", "val", "test"])

            if has_partitions:
                # Aggregated data
                max_len = 0
                for partition in ["train", "val", "test"]:
                    if partition in pred and pred[partition] is not None:
                        partition_data = pred[partition]
                        y_true = partition_data.get("y_true", [])
                        max_len = max(max_len, len(y_true))

                for i in range(max_len):
                    row = []
                    for col in all_columns:
                        # Parse column name to get partition and field
                        if col.startswith("y_true_") or col.startswith("y_pred_"):
                            field, partition_part = col.split("_", 1)
                            partition = partition_part.split("_fold")[0] if "_fold" in partition_part else partition_part

                            if partition in pred and pred[partition] is not None:
                                partition_data = pred[partition]
                                data_list = partition_data.get(field, [])
                                if i < len(data_list):
                                    row.append(data_list[i])
                                else:
                                    row.append("")
                            else:
                                row.append("")
                        else:
                            row.append("")
                    writer.writerow(row)
            else:
                # Single partition data
                y_true = pred.get("y_true", [])
                y_pred = pred.get("y_pred", [])
                max_len = max(len(y_true), len(y_pred))

                for i in range(max_len):
                    row = []
                    for col in all_columns:
                        if col == "y_true":
                            row.append(y_true[i] if i < len(y_true) else "")
                        elif col == "y_pred":
                            row.append(y_pred[i] if i < len(y_pred) else "")
                        else:
                            row.append("")
                    writer.writerow(row)

        # Write to file
        with open(filepath, 'w', newline='') as f:
            f.write(output.getvalue())

        output.close()
        logger.info(f"Saved {len(self)} predictions to {filepath}")

    def get(self, prediction_id: str) -> Optional[PredictionResult]:
        """
        Get a prediction by its ID.

        Args:
            prediction_id: The ID of the prediction to retrieve

        Returns:
            PredictionResult if found, None otherwise

        Examples:
            >>> result = results.get("abc123")
        """
        for pred in self:
            if pred.get("id") == prediction_id:
                return pred
        return None

    def __repr__(self) -> str:
        """String representation showing count and brief info."""
        if not self:
            return "PredictionResultsList(0 predictions)"
        return f"PredictionResultsList({len(self)} predictions)"
