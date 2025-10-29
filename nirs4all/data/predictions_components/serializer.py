"""
Serialization and deserialization for prediction data.

This module handles all serialization/deserialization operations with hybrid
format support: JSON for metadata (human-readable), Parquet for arrays (performance).
"""

import json
import hashlib
import csv
import io
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import polars as pl


class PredictionSerializer:
    """
    Handles serialization/deserialization for predictions.

    Supports:
        - JSON encoding/decoding for metadata and simple fields
        - Parquet for array data (y_true, y_pred, sample_indices)
        - CSV export with metadata headers
        - Hash generation for IDs

    Design:
        - Metadata stays in JSON for human readability and external parsing
        - Arrays use Parquet for efficiency when saved separately
        - Polars handles in-memory operations

    Examples:
        >>> serializer = PredictionSerializer()
        >>> row = {"y_true": [1, 2, 3], "y_pred": [1.1, 2.2, 3.3]}
        >>> serialized = serializer.serialize_row(row)
        >>> deserialized = serializer.deserialize_row(serialized)
    """

    @staticmethod
    def serialize_row(row: Dict[str, Any]) -> Dict[str, str]:
        """
        Serialize a prediction row by converting arrays/dicts to JSON strings.

        Args:
            row: Dictionary with prediction data (may contain numpy arrays, lists, dicts)

        Returns:
            Dictionary with all values as JSON-serialized strings where needed

        Examples:
            >>> row = {"y_true": np.array([1, 2, 3]), "metadata": {"key": "value"}}
            >>> serialized = PredictionSerializer.serialize_row(row)
            >>> serialized["y_true"]  # JSON string
            '[1, 2, 3]'
        """
        serialized = {}
        for key, value in row.items():
            if value is None:
                serialized[key] = json.dumps([])
            elif isinstance(value, np.ndarray):
                serialized[key] = json.dumps(value.tolist())
            elif isinstance(value, (list, dict)):
                # Handle nested numpy arrays in lists
                if isinstance(value, list) and len(value) > 0:
                    serialized_list = []
                    for item in value:
                        if isinstance(item, np.ndarray):
                            serialized_list.append(item.tolist())
                        else:
                            serialized_list.append(item)
                    serialized[key] = json.dumps(serialized_list)
                else:
                    serialized[key] = json.dumps(value)
            else:
                serialized[key] = value
        return serialized

    @staticmethod
    def deserialize_row(row: Dict[str, str]) -> Dict[str, Any]:
        """
        Deserialize a prediction row by parsing JSON strings back to Python objects.

        Args:
            row: Dictionary with JSON-serialized string values

        Returns:
            Dictionary with parsed Python objects (lists, numpy arrays, dicts)

        Examples:
            >>> serialized = {"y_true": '[1, 2, 3]', "metadata": '{"key": "value"}'}
            >>> deserialized = PredictionSerializer.deserialize_row(serialized)
            >>> deserialized["y_true"]
            [1, 2, 3]
        """
        deserialized = {}
        json_fields = ['y_true', 'y_pred', 'sample_indices', 'weights', 'metadata', 'best_params']

        for key, value in row.items():
            if key in json_fields and isinstance(value, str):
                try:
                    deserialized[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    deserialized[key] = value
            else:
                deserialized[key] = value
        return deserialized

    @staticmethod
    def generate_id(row: Dict[str, Any]) -> str:
        """
        Generate a unique hash ID for a prediction row.

        The hash is based on key fields that uniquely identify a prediction:
        dataset, config, model, fold, partition, and sample indices.

        Args:
            row: Prediction row dictionary

        Returns:
            SHA-256 hash string (first 16 characters)

        Examples:
            >>> row = {
            ...     "dataset_name": "wheat",
            ...     "config_name": "default",
            ...     "model_name": "PLS",
            ...     "fold_id": "0",
            ...     "partition": "test"
            ... }
            >>> id_hash = PredictionSerializer.generate_id(row)
            >>> len(id_hash)
            16
        """
        hash_fields = [
            str(row.get('dataset_name', '')),
            str(row.get('dataset_path', '')),
            str(row.get('config_name', '')),
            str(row.get('config_path', '')),
            str(row.get('model_name', '')),
            str(row.get('model_classname', '')),
            str(row.get('fold_id', '')),
            str(row.get('partition', '')),
            str(row.get('step_idx', 0)),
            str(row.get('op_counter', 0)),
            str(row.get('sample_indices', '')),
        ]
        hash_string = '|'.join(hash_fields)
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]

    @staticmethod
    def to_csv(
        predictions: List[Dict],
        filepath: Path,
        mode: str = "single"
    ) -> None:
        """
        Export predictions to CSV format.

        Supports two modes:
            - "single": One prediction per file with metadata headers
            - "batch": Multiple predictions in one file

        Args:
            predictions: List of prediction dictionaries
            filepath: Output CSV file path
            mode: Export mode ("single" or "batch")

        Examples:
            >>> predictions = [{"y_true": [1, 2], "y_pred": [1.1, 2.2]}]
            >>> PredictionSerializer.to_csv(predictions, Path("out.csv"), "single")
        """
        if not predictions:
            return

        filepath.parent.mkdir(parents=True, exist_ok=True)

        if mode == "single" and len(predictions) == 1:
            pred = predictions[0]
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write metadata headers
                writer.writerow(['dataset_name', pred.get('dataset_name', '')])
                writer.writerow(['model_name', pred.get('model_name', '')])
                writer.writerow(['fold_id', pred.get('fold_id', '')])
                writer.writerow(['partition', pred.get('partition', '')])
                writer.writerow([])  # Empty line

                # Write data
                y_true = pred.get('y_true', [])
                y_pred = pred.get('y_pred', [])
                writer.writerow(['y_true', 'y_pred'])
                for yt, yp in zip(y_true, y_pred):
                    writer.writerow([yt, yp])
        else:
            # Batch mode - write all predictions
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)

                # Collect all unique columns
                all_cols = set()
                for pred in predictions:
                    all_cols.update(pred.keys())
                all_cols = sorted(all_cols)

                writer.writerow(all_cols)
                for pred in predictions:
                    writer.writerow([pred.get(col, '') for col in all_cols])

    @staticmethod
    def from_csv(filepath: Path) -> List[Dict[str, Any]]:
        """
        Load predictions from CSV file.

        Args:
            filepath: Input CSV file path

        Returns:
            List of prediction dictionaries

        Examples:
            >>> predictions = PredictionSerializer.from_csv(Path("out.csv"))
        """
        predictions = []

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                predictions.append(dict(row))

        return predictions

    @staticmethod
    def save_arrays_parquet(
        predictions_df: pl.DataFrame,
        parquet_path: Path,
        array_columns: List[str] = None
    ) -> None:
        """
        Save array columns to Parquet format for efficient storage.

        Args:
            predictions_df: DataFrame containing predictions
            parquet_path: Output Parquet file path
            array_columns: List of columns to save (default: y_true, y_pred, sample_indices)

        Examples:
            >>> df = pl.DataFrame({"id": ["a1"], "y_true": ['[1,2,3]']})
            >>> PredictionSerializer.save_arrays_parquet(df, Path("arrays.parquet"))
        """
        if array_columns is None:
            array_columns = ["y_true", "y_pred", "sample_indices"]

        # Select only ID and array columns
        cols_to_save = ["id"] + [col for col in array_columns if col in predictions_df.columns]
        array_df = predictions_df.select(cols_to_save)

        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        array_df.write_parquet(parquet_path)

    @staticmethod
    def load_arrays_parquet(parquet_path: Path) -> pl.DataFrame:
        """
        Load array data from Parquet format.

        Args:
            parquet_path: Input Parquet file path

        Returns:
            DataFrame with array columns

        Examples:
            >>> df = PredictionSerializer.load_arrays_parquet(Path("arrays.parquet"))
        """
        return pl.read_parquet(parquet_path)

    @staticmethod
    def numpy_to_bytes(arr: np.ndarray) -> bytes:
        """
        Convert numpy array to bytes for binary storage.

        Args:
            arr: Numpy array

        Returns:
            Byte representation

        Examples:
            >>> arr = np.array([1, 2, 3])
            >>> bytes_data = PredictionSerializer.numpy_to_bytes(arr)
        """
        return arr.tobytes()

    @staticmethod
    def bytes_to_numpy(data: bytes) -> np.ndarray:
        """
        Convert bytes back to numpy array.

        Args:
            data: Byte representation

        Returns:
            Numpy array

        Examples:
            >>> bytes_data = b'...'
            >>> arr = PredictionSerializer.bytes_to_numpy(bytes_data)
        """
        return np.frombuffer(data)
