"""
Low-level storage backend for predictions using Polars DataFrame.

This module provides the core storage operations for predictions data,
handling CRUD operations, file I/O, and schema management with external
array storage for performance optimization.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import polars as pl
import numpy as np

from .schemas import PREDICTION_SCHEMA
from .serializer import PredictionSerializer
from .array_registry import ArrayRegistry


class PredictionStorage:
    """
    Low-level storage backend using Polars DataFrame with external array storage.

    Uses ArrayRegistry for efficient array storage with deduplication.
    Metadata stored in DataFrame with array references.

    Handles:
        - DataFrame schema management
        - CRUD operations (add, filter, merge, clear)
        - File I/O (split Parquet with array registry)
        - Schema validation
        - Array externalization and hydration

    Examples:
        >>> storage = PredictionStorage()
        >>> row_id = storage.add_row({"dataset_name": "wheat", "model_name": "PLS"})
        >>> # Get with arrays loaded
        >>> pred = storage.get_by_id(row_id, load_arrays=True)
        >>> # Get metadata only (fast)
        >>> pred_meta = storage.get_by_id(row_id, load_arrays=False)

    Attributes:
        _df: Internal Polars DataFrame storing metadata and array references
        _array_registry: ArrayRegistry for efficient array storage
        _serializer: PredictionSerializer instance for data conversion
    """

    def __init__(self, schema: Optional[Dict[str, pl.DataType]] = None):
        """
        Initialize storage with optional schema.

        Args:
            schema: DataFrame schema dict (defaults to PREDICTION_SCHEMA)
        """
        self._schema = schema or PREDICTION_SCHEMA
        self._array_registry = ArrayRegistry()
        self._df = pl.DataFrame(schema=self._schema)
        self._serializer = PredictionSerializer()

    def add_row(self, row_dict: Dict[str, Any]) -> str:
        """
        Add a single prediction row to storage.

        Extracts arrays and stores them externally in ArrayRegistry,
        replacing with array IDs in the metadata.

        Args:
            row_dict: Dictionary with prediction data

        Returns:
            Prediction ID (hash)

        Examples:
            >>> storage = PredictionStorage()
            >>> row_id = storage.add_row({
            ...     "dataset_name": "wheat",
            ...     "model_name": "PLS",
            ...     "y_true": np.array([1.0, 2.0, 3.0]),
            ...     "y_pred": np.array([1.1, 2.2, 3.3])
            ... })
        """
        # Extract and externalize arrays
        row_dict = self._externalize_arrays(row_dict)

        # Serialize row
        serialized = self._serializer.serialize_row(row_dict)

        # Generate ID if not provided
        if 'id' not in serialized or not serialized['id']:
            serialized['id'] = self._serializer.generate_id(serialized)

        # Add to DataFrame
        new_row = pl.DataFrame([serialized], schema=self._schema)
        self._df = pl.concat([self._df, new_row])

        return serialized['id']

        return serialized['id']

    def add_rows(self, rows: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple prediction rows to storage (batch operation).

        Uses batch array registration for better performance.

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

        # Externalize arrays (batch operation)
        rows = [self._externalize_arrays(row) for row in rows]

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

    def get_by_id(self, prediction_id: str, load_arrays: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get a single prediction by its ID.

        Args:
            prediction_id: Prediction hash ID
            load_arrays: If True, hydrate array references with actual arrays.
                        If False, returns metadata only (fast).

        Returns:
            Prediction dictionary or None if not found

        Examples:
            >>> storage = PredictionStorage()
            >>> # Get with arrays (slower)
            >>> pred = storage.get_by_id("abc123def456", load_arrays=True)
            >>> # Get metadata only (faster)
            >>> meta = storage.get_by_id("abc123def456", load_arrays=False)
        """
        filtered = self._df.filter(pl.col("id") == prediction_id)

        if len(filtered) == 0:
            return None

        row = filtered.row(0, named=True)
        deserialized = self._serializer.deserialize_row(row)

        # Hydrate arrays if requested
        if load_arrays:
            deserialized = self._hydrate_arrays(deserialized)

        return deserialized

    def merge(self, other: 'PredictionStorage', deduplicate: bool = False) -> None:
        """
        Merge another storage into this one, including array registries.

        Args:
            other: Another PredictionStorage instance
            deduplicate: If True, remove duplicate IDs (keep first occurrence)

        Examples:
            >>> storage1 = PredictionStorage()
            >>> storage2 = PredictionStorage()
            >>> storage1.merge(storage2, deduplicate=True)
        """
        # Merge DataFrames
        self._df = pl.concat([self._df, other._df], how="diagonal")

        if deduplicate:
            # Keep first occurrence of each ID
            self._df = self._df.unique(subset=["id"], keep="first")

        # Merge array registries
        other_arrays = other._array_registry.get_all_arrays()
        for array_id, (array_data, array_type) in other_arrays.items():
            # Only add if not already present (avoid overwriting)
            if array_id not in self._array_registry:
                self._array_registry.add_array_with_id(array_data, array_type, array_id)

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

    def save_parquet(self, meta_path: Path, arrays_path: Path) -> None:
        """
        Save predictions using split Parquet format (metadata + array registry).

        Saves metadata and arrays separately for optimal performance.

        Args:
            meta_path: Metadata Parquet file path
            arrays_path: Array registry Parquet file path

        Examples:
            >>> storage = PredictionStorage()
            >>> storage.save_parquet(
            ...     Path("predictions.meta.parquet"),
            ...     Path("predictions.arrays.parquet")
            ... )
        """
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata DataFrame (contains array IDs)
        self._df.write_parquet(meta_path, compression="zstd")

        # Save array registry
        self._array_registry.save_to_parquet(arrays_path)

    def load_parquet(self, meta_path: Path, arrays_path: Path) -> None:
        """
        Load predictions from split Parquet format.

        Args:
            meta_path: Metadata Parquet file path
            arrays_path: Array registry Parquet file path

        Examples:
            >>> storage = PredictionStorage()
            >>> storage.load_parquet(
            ...     Path("predictions.meta.parquet"),
            ...     Path("predictions.arrays.parquet")
            ... )
        """
        # Load metadata
        self._df = pl.read_parquet(meta_path)

        # Load array registry
        self._array_registry.load_from_parquet(arrays_path)

    def _externalize_arrays(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract arrays from row and store in array registry, replacing with IDs.

        Args:
            row_dict: Row dictionary with arrays

        Returns:
            Row dictionary with array IDs instead of arrays
        """
        row_dict = row_dict.copy()

        # Array fields to externalize
        array_fields = {
            'y_true': 'y_true',
            'y_pred': 'y_pred',
            'y_proba': 'y_proba',  # Class probabilities for classification
            'sample_indices': 'indices',
            'weights': 'weights'
        }

        for field, array_type in array_fields.items():
            if field in row_dict and row_dict[field] is not None:
                value = row_dict[field]

                # Convert to numpy array if needed
                if not isinstance(value, np.ndarray):
                    try:
                        value = np.array(value, dtype=np.float64)
                    except (ValueError, TypeError):
                        # Skip if conversion fails
                        row_dict[f'{field}_id'] = None
                        if field in row_dict:
                            del row_dict[field]
                        continue

                # Store in registry and get ID
                array_id = self._array_registry.add_array(value, array_type)

                # Replace array with ID reference
                row_dict[f'{field}_id'] = array_id
                del row_dict[field]
            else:
                # Field is missing or None - set ID field to None
                row_dict[f'{field}_id'] = None
                if field in row_dict:
                    del row_dict[field]

        return row_dict

    def _hydrate_arrays(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hydrate array references by loading arrays from registry.

        Args:
            row_dict: Row dictionary with array IDs

        Returns:
            Row dictionary with actual arrays instead of IDs
        """
        row_dict = row_dict.copy()

        # Array ID fields to hydrate
        array_id_fields = ['y_true_id', 'y_pred_id', 'y_proba_id', 'sample_indices_id', 'weights_id']

        for field in array_id_fields:
            if field in row_dict and row_dict[field]:
                array_id = row_dict[field]

                # Get array from registry
                try:
                    array = self._array_registry.get_array(array_id)

                    # Put array back with original name
                    array_field = field.replace('_id', '')
                    row_dict[array_field] = array

                    # Remove ID field
                    del row_dict[field]
                except KeyError:
                    # Array not found - skip
                    pass

        return row_dict

    def get_array_registry(self) -> ArrayRegistry:
        """
        Get the array registry.

        Returns:
            ArrayRegistry instance
        """
        return self._array_registry

    def archive_from_csv(self, csv_path: Path, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load predictions from CSV file and prepare for archiving.

        Args:
            csv_path: Path to predictions CSV file
            metadata: Optional metadata dict to merge with prediction data

        Returns:
            Dictionary with prediction data ready for add_row()

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV file is empty

        Examples:
            >>> storage = PredictionStorage()
            >>> pred_data = storage.archive_from_csv(Path("predictions.csv"), {"dataset_name": "wheat"})
            >>> pred_id = storage.add_row(pred_data)
        """
        import csv

        if not csv_path.exists():
            raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")

        # Load predictions from CSV
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"Empty predictions file: {csv_path}")

        # Convert to arrays
        y_true = np.array([float(row['y_true']) for row in rows])
        y_pred = np.array([float(row['y_pred']) for row in rows])

        # Extract partition and sample_id if present
        partitions = [row.get('partition', 'unknown') for row in rows]

        # Create prediction data
        pred_data = {
            'y_true': y_true,
            'y_pred': y_pred,
            'partition': partitions[0] if len(set(partitions)) == 1 else 'mixed',
            'sample_indices': np.arange(len(y_true))
        }

        # Merge metadata if provided
        if metadata:
            # Normalize field names (model_type -> model_name)
            if 'model_type' in metadata:
                metadata = metadata.copy()
                metadata['model_name'] = metadata.pop('model_type')
            pred_data.update(metadata)

        return pred_data

    def __len__(self) -> int:
        """Return number of predictions in storage."""
        return len(self._df)

    def __repr__(self) -> str:
        """String representation."""
        return f"PredictionStorage({len(self)} predictions)"
