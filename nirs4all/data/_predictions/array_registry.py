"""
Array Registry - Efficient storage and deduplication for prediction arrays

This module provides the ArrayRegistry class which manages array storage with:
- Content-based deduplication using hash matching
- Native Parquet List[Float64] storage for performance
- Batch operations for efficiency
- Lazy loading support
"""

import hashlib
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from uuid import uuid4


# Schema for array storage with native Parquet types
ARRAY_SCHEMA = {
    "array_id": pl.Utf8,
    "array_data": pl.List(pl.Float64),  # Native Parquet list type
    "array_hash": pl.Utf8,
    "array_size": pl.Int64,
    "array_shape": pl.Utf8,  # JSON-encoded shape tuple (e.g., "[2, 3]" for 2D)
    "array_type": pl.Utf8,  # "y_true" | "y_pred" | "y_proba" | "indices" | "weights"
}


class ArrayRegistry:
    """
    Manages array storage with native Parquet types and deduplication.

    Features:
    - Content-based deduplication (arrays with same content share storage)
    - Native List[Float64] Parquet storage (no JSON serialization)
    - Fast lookup by ID
    - Batch operations for performance
    - Lazy loading support

    Example:
        >>> registry = ArrayRegistry()
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> array_id = registry.add_array(y_true, "y_true")
        >>> # Same array returns same ID (deduplication)
        >>> array_id2 = registry.add_array(y_true, "y_true")
        >>> assert array_id == array_id2
        >>> # Retrieve array
        >>> retrieved = registry.get_array(array_id)
        >>> np.testing.assert_array_equal(retrieved, y_true)
    """

    def __init__(self):
        """Initialize empty array registry."""
        self._arrays_df = pl.DataFrame(schema=ARRAY_SCHEMA)
        self._hash_cache: Dict[str, str] = {}  # hash → array_id for fast lookup
        self._id_cache: Set[str] = set()  # array_ids for fast membership check

    def add_array(
        self,
        array: np.ndarray,
        array_type: str = "data"
    ) -> str:
        """
        Add array to registry with deduplication.

        If an identical array (by content hash) already exists,
        returns the existing array_id instead of storing duplicate.

        Preserves array shape for multi-dimensional arrays (e.g., y_proba).

        Args:
            array: NumPy array to store (1D or 2D)
            array_type: Type label for array ("y_true", "y_pred", "y_proba", "indices", "weights")

        Returns:
            array_id: Unique identifier for this array

        Example:
            >>> registry = ArrayRegistry()
            >>> arr = np.array([1.0, 2.0, 3.0])
            >>> id1 = registry.add_array(arr, "y_true")
            >>> id2 = registry.add_array(arr, "y_true")  # Same content
            >>> assert id1 == id2  # Deduplication works
        """
        import json

        # Store original shape for reconstruction
        original_shape = array.shape

        # Flatten array for storage
        flat_array = array.flatten()

        # Compute content hash for deduplication
        array_hash = self._hash_array(flat_array)

        # Check if array already exists (deduplication)
        if array_hash in self._hash_cache:
            existing_id = self._hash_cache[array_hash]

            # If the new shape is multi-dimensional but stored shape is 1D,
            # update the stored shape (handles backward compatibility for y_proba)
            if len(original_shape) > 1:
                existing_row = self._arrays_df.filter(pl.col("array_id") == existing_id)
                if existing_row.height > 0:
                    stored_shape_str = existing_row[0, "array_shape"]
                    if stored_shape_str:
                        stored_shape = tuple(json.loads(stored_shape_str))
                        if len(stored_shape) == 1 and stored_shape != original_shape:
                            # Update the shape to the multi-dimensional version
                            self._arrays_df = self._arrays_df.with_columns(
                                pl.when(pl.col("array_id") == existing_id)
                                .then(pl.lit(json.dumps(original_shape)))
                                .otherwise(pl.col("array_shape"))
                                .alias("array_shape")
                            )

            return existing_id

        # New array - store it
        array_id = f"array_{uuid4().hex[:12]}"

        new_row = {
            "array_id": array_id,
            "array_data": flat_array.tolist(),  # Polars converts to List[Float64]
            "array_hash": array_hash,
            "array_size": len(flat_array),
            "array_shape": json.dumps(original_shape),  # Store shape as JSON string
            "array_type": array_type
        }

        # Add to DataFrame
        new_row_df = pl.DataFrame([new_row], schema=ARRAY_SCHEMA)
        self._arrays_df = pl.concat([self._arrays_df, new_row_df], how="vertical")

        # Update caches
        self._hash_cache[array_hash] = array_id
        self._id_cache.add(array_id)

        return array_id

    def add_arrays_batch(
        self,
        arrays: List[np.ndarray],
        array_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add multiple arrays in batch for performance.

        Uses vectorized operations and bulk DataFrame concatenation
        for better performance than repeated add_array() calls.

        Args:
            arrays: List of NumPy arrays to store
            array_types: Optional list of type labels (same length as arrays)

        Returns:
            List of array_ids in same order as input arrays

        Example:
            >>> registry = ArrayRegistry()
            >>> arrays = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
            >>> ids = registry.add_arrays_batch(arrays)
            >>> assert len(ids) == 2
        """
        import json

        if not arrays:
            return []

        if array_types is None:
            array_types = ["data"] * len(arrays)
        elif len(array_types) != len(arrays):
            raise ValueError("array_types must have same length as arrays")

        array_ids = []
        new_rows = []

        for array, array_type in zip(arrays, array_types):
            # Store original shape
            original_shape = array.shape

            # Flatten for storage
            flat_array = array.flatten()

            # Compute hash
            array_hash = self._hash_array(flat_array)

            # Check if already exists
            if array_hash in self._hash_cache:
                array_ids.append(self._hash_cache[array_hash])
                continue

            # New array
            array_id = f"array_{uuid4().hex[:12]}"
            array_ids.append(array_id)

            new_rows.append({
                "array_id": array_id,
                "array_data": flat_array.tolist(),
                "array_hash": array_hash,
                "array_size": len(flat_array),
                "array_shape": json.dumps(original_shape),
                "array_type": array_type
            })

            # Update caches
            self._hash_cache[array_hash] = array_id
            self._id_cache.add(array_id)

        # Bulk concatenation if we have new rows
        if new_rows:
            new_rows_df = pl.DataFrame(new_rows, schema=ARRAY_SCHEMA)
            self._arrays_df = pl.concat([self._arrays_df, new_rows_df], how="vertical")

        return array_ids

    def get_array(self, array_id: str) -> np.ndarray:
        """
        Retrieve array by ID.

        Restores original array shape for multi-dimensional arrays.

        Args:
            array_id: Unique identifier of array

        Returns:
            NumPy array with original shape

        Raises:
            KeyError: If array_id not found

        Example:
            >>> registry = ArrayRegistry()
            >>> arr = np.array([1.0, 2.0, 3.0])
            >>> array_id = registry.add_array(arr)
            >>> retrieved = registry.get_array(array_id)
            >>> np.testing.assert_array_equal(arr, retrieved)
        """
        import json

        if array_id not in self._id_cache:
            raise KeyError(f"Array {array_id} not found in registry")

        row = self._arrays_df.filter(pl.col("array_id") == array_id)

        if row.height == 0:
            raise KeyError(f"Array {array_id} not found in registry")

        # Extract List[Float64] and convert to NumPy
        array_data = row[0, "array_data"]
        array = np.array(array_data, dtype=np.float64)

        # Restore original shape if stored
        shape_str = row[0, "array_shape"]
        if shape_str is not None:
            original_shape = tuple(json.loads(shape_str))
            array = array.reshape(original_shape)

        return array

    def get_arrays_batch(
        self,
        array_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Retrieve multiple arrays in batch for performance.

        More efficient than repeated get_array() calls as it uses
        a single DataFrame filter operation. Restores original shapes.

        Args:
            array_ids: List of array IDs to retrieve

        Returns:
            Dictionary mapping array_id to NumPy array (with original shapes)

        Example:
            >>> registry = ArrayRegistry()
            >>> arr1 = np.array([1.0, 2.0])
            >>> arr2 = np.array([3.0, 4.0])
            >>> id1 = registry.add_array(arr1)
            >>> id2 = registry.add_array(arr2)
            >>> batch = registry.get_arrays_batch([id1, id2])
            >>> assert len(batch) == 2
        """
        import json

        if not array_ids:
            return {}

        # Filter for all requested IDs in one operation
        filtered_df = self._arrays_df.filter(
            pl.col("array_id").is_in(array_ids)
        )

        # Convert to dict with shape restoration
        result = {}
        for row in filtered_df.iter_rows(named=True):
            array_id = row["array_id"]
            array_data = row["array_data"]
            shape_str = row.get("array_shape")

            array = np.array(array_data, dtype=np.float64)

            # Restore original shape if stored
            if shape_str is not None:
                original_shape = tuple(json.loads(shape_str))
                array = array.reshape(original_shape)

            result[array_id] = array

        return result

    def has_array(self, array_id: str) -> bool:
        """
        Check if array exists in registry.

        Args:
            array_id: Array ID to check

        Returns:
            True if array exists, False otherwise
        """
        return array_id in self._id_cache

    def remove_array(self, array_id: str) -> bool:
        """
        Remove array from registry.

        Args:
            array_id: ID of array to remove

        Returns:
            True if array was removed, False if not found

        Note:
            This does not check if array is still referenced by predictions.
            Use with caution.
        """
        if array_id not in self._id_cache:
            return False

        # Remove from DataFrame
        self._arrays_df = self._arrays_df.filter(
            pl.col("array_id") != array_id
        )

        # Remove from caches
        self._id_cache.discard(array_id)

        # Remove from hash cache (need to find the hash)
        hash_to_remove = None
        for hash_val, arr_id in self._hash_cache.items():
            if arr_id == array_id:
                hash_to_remove = hash_val
                break

        if hash_to_remove:
            del self._hash_cache[hash_to_remove]

        return True

    def save_to_parquet(self, filepath: Path) -> None:
        """
        Save arrays to Parquet file with native List[Float64] storage.

        The resulting Parquet file uses native list types for efficient
        storage and fast loading, with no JSON serialization overhead.

        Args:
            filepath: Path to save Parquet file

        Example:
            >>> registry = ArrayRegistry()
            >>> registry.add_array(np.array([1.0, 2.0]))
            >>> registry.save_to_parquet(Path("arrays.parquet"))
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Write with Parquet-native List[Float64] type
        self._arrays_df.write_parquet(
            filepath,
            compression="zstd",  # Good compression for numeric data
            use_pyarrow=True
        )

    def load_from_parquet(self, filepath: Path) -> None:
        """
        Load arrays from Parquet file.

        Handles backward compatibility with older schema versions
        (e.g., files without array_shape column).

        Args:
            filepath: Path to Parquet file

        Raises:
            FileNotFoundError: If file doesn't exist

        Example:
            >>> registry = ArrayRegistry()
            >>> registry.load_from_parquet(Path("arrays.parquet"))
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Array registry file not found: {filepath}")

        self._arrays_df = pl.read_parquet(filepath, use_pyarrow=True)

        # Handle backward compatibility: add missing columns from schema
        for col_name, col_type in ARRAY_SCHEMA.items():
            if col_name not in self._arrays_df.columns:
                # Add missing column with null values
                self._arrays_df = self._arrays_df.with_columns(
                    pl.lit(None).cast(col_type).alias(col_name)
                )

        # Ensure column order matches schema
        self._arrays_df = self._arrays_df.select(list(ARRAY_SCHEMA.keys()))

        self._rebuild_caches()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about array registry.

        Returns:
            Dictionary with statistics:
            - total_arrays: Number of arrays stored
            - unique_hashes: Number of unique content hashes
            - total_elements: Total number of array elements
            - total_size_mb: Approximate memory size in MB
            - deduplication_ratio: Ratio of unique arrays to total adds
            - by_type: Count of arrays by type

        Example:
            >>> registry = ArrayRegistry()
            >>> registry.add_array(np.array([1.0, 2.0]), "y_true")
            >>> stats = registry.get_stats()
            >>> assert stats["total_arrays"] == 1
        """
        if self._arrays_df.height == 0:
            return {
                "total_arrays": 0,
                "unique_hashes": 0,
                "total_elements": 0,
                "total_size_mb": 0.0,
                "deduplication_ratio": 0.0,
                "by_type": {}
            }

        total_elements = self._arrays_df["array_size"].sum()
        total_size_mb = total_elements * 8 / 1024 / 1024  # 8 bytes per float64

        # Count by type
        by_type = (
            self._arrays_df
            .group_by("array_type")
            .agg(pl.len())
            .to_dicts()
        )
        by_type_dict = {row["array_type"]: row["len"] for row in by_type}

        return {
            "total_arrays": len(self._arrays_df),
            "unique_hashes": len(self._hash_cache),
            "total_elements": int(total_elements),
            "total_size_mb": float(total_size_mb),
            "deduplication_ratio": len(self._hash_cache) / max(len(self._id_cache), 1),
            "by_type": by_type_dict
        }

    def clear(self) -> None:
        """
        Clear all arrays from registry.

        Example:
            >>> registry = ArrayRegistry()
            >>> registry.add_array(np.array([1.0]))
            >>> registry.clear()
            >>> assert len(registry.get_stats()["by_type"]) == 0
        """
        self._arrays_df = pl.DataFrame(schema=ARRAY_SCHEMA)
        self._hash_cache.clear()
        self._id_cache.clear()

    def _hash_array(self, array: np.ndarray) -> str:
        """
        Compute content hash of array for deduplication.

        Uses SHA-256 hash of array bytes for collision resistance.

        Args:
            array: NumPy array

        Returns:
            Hexadecimal hash string
        """
        # Convert to bytes (ensure consistent byte order)
        array_bytes = np.ascontiguousarray(array).tobytes()

        # Compute SHA-256 hash
        hash_obj = hashlib.sha256(array_bytes)
        return hash_obj.hexdigest()

    def _rebuild_caches(self) -> None:
        """
        Rebuild internal caches from DataFrame.

        Called after loading from disk to reconstruct hash and ID caches.
        """
        self._hash_cache.clear()
        self._id_cache.clear()

        for row in self._arrays_df.iter_rows(named=True):
            array_id = row["array_id"]
            array_hash = row["array_hash"]

            self._hash_cache[array_hash] = array_id
            self._id_cache.add(array_id)

    def get_all_arrays(self) -> Dict[str, Tuple[np.ndarray, str]]:
        """
        Get all arrays as a dictionary.

        Returns:
            Dictionary mapping array_id → (array_data, array_type)
            Arrays are returned with their original shapes restored.

        Example:
            >>> registry = ArrayRegistry()
            >>> arr_id = registry.add_array(np.array([1.0, 2.0]), "y_true")
            >>> all_arrays = registry.get_all_arrays()
            >>> assert arr_id in all_arrays
        """
        import json

        result = {}
        for row in self._arrays_df.iter_rows(named=True):
            array_id = row["array_id"]
            array_data = np.array(row["array_data"], dtype=np.float64)
            array_type = row["array_type"]

            # Restore original shape if stored
            shape_str = row.get("array_shape")
            if shape_str is not None:
                try:
                    original_shape = tuple(json.loads(shape_str))
                    array_data = array_data.reshape(original_shape)
                except (json.JSONDecodeError, ValueError):
                    pass

            result[array_id] = (array_data, array_type)
        return result

    def add_array_with_id(
        self,
        array: np.ndarray,
        array_type: str,
        array_id: str
    ) -> str:
        """
        Add array with a specific ID (for merging registries).

        Does not check for deduplication - assumes the array_id is unique.
        Used internally when merging registries.
        Preserves original array shape for multi-dimensional arrays.

        Args:
            array: NumPy array to store (1D or 2D)
            array_type: Type label for array
            array_id: Specific ID to use

        Returns:
            array_id: The provided array_id

        Example:
            >>> registry = ArrayRegistry()
            >>> arr = np.array([1.0, 2.0, 3.0])
            >>> registry.add_array_with_id(arr, "y_true", "my_custom_id")
            'my_custom_id'
        """
        import json

        # Store original shape
        original_shape = array.shape

        # Flatten for storage
        flat_array = array.flatten()

        # Compute content hash
        array_hash = self._hash_array(flat_array)

        new_row = {
            "array_id": array_id,
            "array_data": flat_array.tolist(),
            "array_hash": array_hash,
            "array_size": len(flat_array),
            "array_shape": json.dumps(original_shape),
            "array_type": array_type
        }

        # Add to DataFrame
        new_row_df = pl.DataFrame([new_row], schema=ARRAY_SCHEMA)
        self._arrays_df = pl.concat([self._arrays_df, new_row_df], how="vertical")

        # Update caches
        self._hash_cache[array_hash] = array_id
        self._id_cache.add(array_id)

        return array_id

    def __len__(self) -> int:
        """Return number of arrays in registry."""
        return len(self._arrays_df)

    def __contains__(self, array_id: str) -> bool:
        """Check if array_id exists in registry."""
        return array_id in self._id_cache

    def __repr__(self) -> str:
        """String representation of registry."""
        stats = self.get_stats()
        return (
            f"ArrayRegistry("
            f"arrays={stats['total_arrays']}, "
            f"unique={stats['unique_hashes']}, "
            f"size={stats['total_size_mb']:.2f}MB)"
        )
