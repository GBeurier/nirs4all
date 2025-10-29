"""
Low-level storage backend for predictions using Polars DataFrame.

This module provides the core storage operations for predictions data,
handling CRUD operations, file I/O, and schema management.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import polars as pl
import numpy as np

from .schemas import PREDICTION_SCHEMA
from .serializer import PredictionSerializer


class PredictionStorage:
    """
    Low-level storage backend using Polars DataFrame.

    Handles:
        - DataFrame schema management
        - CRUD operations (add, filter, merge, clear)
        - File I/O (JSON, Parquet)
        - Schema validation

    Examples:
        >>> storage = PredictionStorage()
        >>> row_id = storage.add_row({"dataset_name": "wheat", "model_name": "PLS"})
        >>> filtered = storage.filter(dataset_name="wheat")
        >>> storage.save_json(Path("predictions.json"))

    Attributes:
        _df: Internal Polars DataFrame storing all predictions
        _serializer: PredictionSerializer instance for data conversion
    """

    def __init__(self, schema: Optional[Dict[str, pl.DataType]] = None):
        """
        Initialize storage with optional schema.

        Args:
            schema: DataFrame schema dict (defaults to PREDICTION_SCHEMA)
        """
        self._schema = schema or PREDICTION_SCHEMA
        self._df = pl.DataFrame(schema=self._schema)
        self._serializer = PredictionSerializer()

    def add_row(self, row_dict: Dict[str, Any]) -> str:
        """
        Add a single prediction row to storage.

        Args:
            row_dict: Dictionary with prediction data

        Returns:
            Prediction ID (hash)

        Examples:
            >>> storage = PredictionStorage()
            >>> row_id = storage.add_row({
            ...     "dataset_name": "wheat",
            ...     "model_name": "PLS",
            ...     "y_true": [1, 2, 3],
            ...     "y_pred": [1.1, 2.2, 3.3]
            ... })
        """
        # Serialize row
        serialized = self._serializer.serialize_row(row_dict)

        # Generate ID if not provided
        if 'id' not in serialized or not serialized['id']:
            serialized['id'] = self._serializer.generate_id(serialized)

        # Add to DataFrame
        new_row = pl.DataFrame([serialized], schema=self._schema)
        self._df = pl.concat([self._df, new_row])

        return serialized['id']

    def add_rows(self, rows: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple prediction rows to storage (batch operation).

        Args:
            rows: List of prediction row dictionaries

        Returns:
            List of prediction IDs

        Examples:
            >>> storage = PredictionStorage()
            >>> ids = storage.add_rows([
            ...     {"dataset_name": "wheat", "model_name": "PLS"},
            ...     {"dataset_name": "corn", "model_name": "PLS"}
            ... ])
        """
        if not rows:
            return []

        serialized_rows = []
        ids = []

        for row in rows:
            serialized = self._serializer.serialize_row(row)
            if 'id' not in serialized or not serialized['id']:
                serialized['id'] = self._serializer.generate_id(serialized)
            ids.append(serialized['id'])
            serialized_rows.append(serialized)

        # Batch add to DataFrame
        new_df = pl.DataFrame(serialized_rows, schema=self._schema)
        self._df = pl.concat([self._df, new_df])

        return ids

    def filter(self, **criteria) -> pl.DataFrame:
        """
        Filter predictions by criteria.

        Args:
            **criteria: Filter criteria (e.g., dataset_name="wheat", partition="test")

        Returns:
            Filtered DataFrame

        Examples:
            >>> storage = PredictionStorage()
            >>> filtered = storage.filter(dataset_name="wheat", partition="test")
        """
        df = self._df

        for key, value in criteria.items():
            if key in df.columns and value is not None:
                df = df.filter(pl.col(key) == value)

        return df

    def get_by_id(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single prediction by its ID.

        Args:
            prediction_id: Prediction hash ID

        Returns:
            Prediction dictionary or None if not found

        Examples:
            >>> storage = PredictionStorage()
            >>> pred = storage.get_by_id("abc123def456")
        """
        filtered = self._df.filter(pl.col("id") == prediction_id)

        if len(filtered) == 0:
            return None

        row = filtered.row(0, named=True)
        return self._serializer.deserialize_row(row)

    def merge(self, other: 'PredictionStorage', deduplicate: bool = False) -> None:
        """
        Merge another storage into this one.

        Args:
            other: Another PredictionStorage instance
            deduplicate: If True, remove duplicate IDs (keep first occurrence)

        Examples:
            >>> storage1 = PredictionStorage()
            >>> storage2 = PredictionStorage()
            >>> storage1.merge(storage2, deduplicate=True)
        """
        self._df = pl.concat([self._df, other._df])

        if deduplicate:
            # Keep first occurrence of each ID
            self._df = self._df.unique(subset=["id"], keep="first")

    def clear(self) -> None:
        """
        Clear all predictions from storage.

        Examples:
            >>> storage = PredictionStorage()
            >>> storage.clear()
        """
        self._df = pl.DataFrame(schema=self._schema)

    def to_dataframe(self) -> pl.DataFrame:
        """
        Get the internal DataFrame (read-only view).

        Returns:
            Polars DataFrame with all predictions

        Examples:
            >>> storage = PredictionStorage()
            >>> df = storage.to_dataframe()
        """
        return self._df

    def save_json(self, filepath: Path) -> None:
        """
        Save predictions to JSON file.

        Args:
            filepath: Output JSON file path

        Examples:
            >>> storage = PredictionStorage()
            >>> storage.save_json(Path("predictions.json"))
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to list of dicts and save
        data = self._df.to_dicts()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_json(self, filepath: Path) -> None:
        """
        Load predictions from JSON file.

        Args:
            filepath: Input JSON file path

        Examples:
            >>> storage = PredictionStorage()
            >>> storage.load_json(Path("predictions.json"))
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        if data:
            self._df = pl.DataFrame(data, schema=self._schema)

    def save_parquet(self, meta_path: Path, data_path: Path) -> None:
        """
        Save predictions using split Parquet format (metadata + array data).

        Args:
            meta_path: Metadata Parquet file path
            data_path: Array data Parquet file path

        Examples:
            >>> storage = PredictionStorage()
            >>> storage.save_parquet(Path("meta.parquet"), Path("data.parquet"))
        """
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata (exclude large array columns)
        meta_cols = [col for col in self._df.columns
                     if col not in ['y_true', 'y_pred', 'sample_indices', 'weights']]
        meta_df = self._df.select(meta_cols)
        meta_df.write_parquet(meta_path)

        # Save array data
        array_cols = ['id', 'y_true', 'y_pred', 'sample_indices', 'weights']
        array_cols = [col for col in array_cols if col in self._df.columns]
        array_df = self._df.select(array_cols)
        array_df.write_parquet(data_path)

    def load_parquet(self, meta_path: Path, data_path: Path) -> None:
        """
        Load predictions from split Parquet format.

        Args:
            meta_path: Metadata Parquet file path
            data_path: Array data Parquet file path

        Examples:
            >>> storage = PredictionStorage()
            >>> storage.load_parquet(Path("meta.parquet"), Path("data.parquet"))
        """
        meta_df = pl.read_parquet(meta_path)
        array_df = pl.read_parquet(data_path)

        # Join on ID
        self._df = meta_df.join(array_df, on="id", how="left")

    def __len__(self) -> int:
        """Return number of predictions in storage."""
        return len(self._df)

    def __repr__(self) -> str:
        """String representation."""
        return f"PredictionStorage({len(self)} predictions)"
