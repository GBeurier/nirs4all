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


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


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

        Note: When using ArrayRegistry, arrays are externalized before serialization,
        so this method only handles metadata fields.

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
                # Keep None as None (important for nullable columns in Polars schema)
                serialized[key] = None
            elif isinstance(value, (np.ndarray, list, dict)):
                # Convert to JSON string using NumpyEncoder
                serialized[key] = json.dumps(value, cls=NumpyEncoder)
            elif isinstance(value, (np.integer, np.floating)):
                # Handle scalar numpy types
                if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    serialized[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                    serialized[key] = float(value)
                else:
                    serialized[key] = value
            else:
                serialized[key] = value
        return serialized

    @staticmethod
    def deserialize_row(row: Dict[str, str]) -> Dict[str, Any]:
        """
        Deserialize a prediction row by parsing JSON strings back to Python objects.

        Handles both legacy format (arrays as JSON) and new format (array_id references).

        Args:
            row: Dictionary with JSON-serialized string values

        Returns:
            Dictionary with parsed Python objects (lists, numpy arrays, dicts)

        Examples:
            >>> # Legacy format
            >>> serialized = {"y_true": '[1, 2, 3]', "metadata": '{"key": "value"}'}
            >>> deserialized = PredictionSerializer.deserialize_row(serialized)
            >>> deserialized["y_true"]
            [1, 2, 3]
            >>> # New format with array references
            >>> serialized = {"y_true_id": "array_abc123", "metadata": '{"key": "value"}'}
            >>> deserialized = PredictionSerializer.deserialize_row(serialized)
            >>> deserialized["y_true_id"]
            'array_abc123'
        """
        deserialized = {}
        json_fields = ['y_true', 'y_pred', 'sample_indices', 'weights', 'metadata', 'best_params', 'scores']

        for key, value in row.items():
            if key in json_fields and isinstance(value, str):
                try:
                    deserialized[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    deserialized[key] = value
            else:
                # Pass through all other values (including array_id fields)
                deserialized[key] = value
        return deserialized

    @staticmethod
    def generate_id(row: Dict[str, Any]) -> str:
        """
        Generate a unique hash ID for a prediction row.

        The hash is based on key fields that uniquely identify a prediction:
        dataset, config, model, fold, partition, and sample indices (or sample_indices_id).

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
        # Use sample_indices_id if available (new format), otherwise sample_indices
        sample_idx_field = row.get('sample_indices_id', row.get('sample_indices', ''))

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
            str(sample_idx_field),
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
