"""
Tests for ArrayRegistry - Array storage with deduplication
"""

import pytest
import numpy as np
import polars as pl
from pathlib import Path
import tempfile
import shutil

from nirs4all.data.predictions_components.array_registry import ArrayRegistry, ARRAY_SCHEMA


class TestArrayRegistryBasics:
    """Test basic ArrayRegistry operations."""

    def test_init_empty_registry(self):
        """Test initialization creates empty registry."""
        registry = ArrayRegistry()
        assert len(registry) == 0
        stats = registry.get_stats()
        assert stats["total_arrays"] == 0
        assert stats["unique_hashes"] == 0

    def test_add_single_array(self):
        """Test adding a single array."""
        registry = ArrayRegistry()
        array = np.array([1.0, 2.0, 3.0, 4.0])

        array_id = registry.add_array(array, "y_true")

        assert isinstance(array_id, str)
        assert array_id.startswith("array_")
        assert len(registry) == 1
        assert array_id in registry

    def test_retrieve_array(self):
        """Test retrieving array by ID."""
        registry = ArrayRegistry()
        original = np.array([1.5, 2.5, 3.5])

        array_id = registry.add_array(original, "y_pred")
        retrieved = registry.get_array(array_id)

        np.testing.assert_array_equal(retrieved, original)
        assert retrieved.dtype == np.float64

    def test_retrieve_nonexistent_array_raises(self):
        """Test retrieving non-existent array raises KeyError."""
        registry = ArrayRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get_array("nonexistent_id")

    def test_has_array(self):
        """Test checking array existence."""
        registry = ArrayRegistry()
        array = np.array([1.0, 2.0])

        array_id = registry.add_array(array)

        assert registry.has_array(array_id)
        assert not registry.has_array("nonexistent_id")

    def test_contains_operator(self):
        """Test __contains__ operator."""
        registry = ArrayRegistry()
        array = np.array([1.0, 2.0])

        array_id = registry.add_array(array)

        assert array_id in registry
        assert "nonexistent_id" not in registry


class TestArrayDeduplication:
    """Test deduplication functionality."""

    def test_identical_arrays_same_id(self):
        """Test identical arrays return same ID (deduplication)."""
        registry = ArrayRegistry()
        array1 = np.array([1.0, 2.0, 3.0])
        array2 = np.array([1.0, 2.0, 3.0])  # Identical content

        id1 = registry.add_array(array1, "y_true")
        id2 = registry.add_array(array2, "y_true")

        assert id1 == id2
        assert len(registry) == 1  # Only one array stored

    def test_different_arrays_different_ids(self):
        """Test different arrays get different IDs."""
        registry = ArrayRegistry()
        array1 = np.array([1.0, 2.0, 3.0])
        array2 = np.array([4.0, 5.0, 6.0])

        id1 = registry.add_array(array1)
        id2 = registry.add_array(array2)

        assert id1 != id2
        assert len(registry) == 2

    def test_deduplication_with_different_types(self):
        """Test deduplication works across different array types."""
        registry = ArrayRegistry()
        array = np.array([1.0, 2.0, 3.0])

        id1 = registry.add_array(array, "y_true")
        id2 = registry.add_array(array, "y_pred")  # Same content, different type

        assert id1 == id2  # Deduplication based on content only
        assert len(registry) == 1

    def test_deduplication_stats(self):
        """Test deduplication statistics are correct."""
        registry = ArrayRegistry()
        array1 = np.array([1.0, 2.0])
        array2 = np.array([1.0, 2.0])  # Duplicate
        array3 = np.array([3.0, 4.0])  # Unique

        registry.add_array(array1)
        registry.add_array(array2)
        registry.add_array(array3)

        stats = registry.get_stats()
        assert stats["total_arrays"] == 2  # Only 2 unique arrays
        assert stats["unique_hashes"] == 2


class TestBatchOperations:
    """Test batch add and retrieve operations."""

    def test_add_arrays_batch_empty(self):
        """Test batch add with empty list."""
        registry = ArrayRegistry()
        ids = registry.add_arrays_batch([])
        assert ids == []
        assert len(registry) == 0

    def test_add_arrays_batch_single(self):
        """Test batch add with single array."""
        registry = ArrayRegistry()
        arrays = [np.array([1.0, 2.0])]

        ids = registry.add_arrays_batch(arrays)

        assert len(ids) == 1
        assert len(registry) == 1

    def test_add_arrays_batch_multiple(self):
        """Test batch add with multiple arrays."""
        registry = ArrayRegistry()
        arrays = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0])
        ]

        ids = registry.add_arrays_batch(arrays)

        assert len(ids) == 3
        assert len(registry) == 3
        assert all(isinstance(id, str) for id in ids)

    def test_add_arrays_batch_with_types(self):
        """Test batch add with array types."""
        registry = ArrayRegistry()
        arrays = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0])
        ]
        types = ["y_true", "y_pred"]

        ids = registry.add_arrays_batch(arrays, types)

        assert len(ids) == 2
        # Verify types stored correctly
        stats = registry.get_stats()
        assert "y_true" in stats["by_type"]
        assert "y_pred" in stats["by_type"]

    def test_add_arrays_batch_with_duplicates(self):
        """Test batch add with duplicate arrays (deduplication)."""
        registry = ArrayRegistry()
        arrays = [
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),  # Duplicate
            np.array([3.0, 4.0])
        ]

        ids = registry.add_arrays_batch(arrays)

        assert len(ids) == 3  # 3 IDs returned
        assert ids[0] == ids[1]  # First two are same (deduplicated)
        assert len(registry) == 2  # Only 2 unique arrays stored

    def test_add_arrays_batch_type_length_mismatch(self):
        """Test batch add raises on type/array length mismatch."""
        registry = ArrayRegistry()
        arrays = [np.array([1.0]), np.array([2.0])]
        types = ["y_true"]  # Only one type for two arrays

        with pytest.raises(ValueError, match="same length"):
            registry.add_arrays_batch(arrays, types)

    def test_get_arrays_batch_empty(self):
        """Test batch get with empty list."""
        registry = ArrayRegistry()
        result = registry.get_arrays_batch([])
        assert result == {}

    def test_get_arrays_batch_multiple(self):
        """Test batch get with multiple arrays."""
        registry = ArrayRegistry()
        arrays = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0])
        ]

        ids = registry.add_arrays_batch(arrays)
        retrieved = registry.get_arrays_batch(ids)

        assert len(retrieved) == 3
        for i, array_id in enumerate(ids):
            np.testing.assert_array_equal(retrieved[array_id], arrays[i])

    def test_get_arrays_batch_partial(self):
        """Test batch get with some nonexistent IDs."""
        registry = ArrayRegistry()
        array = np.array([1.0, 2.0])

        array_id = registry.add_array(array)
        retrieved = registry.get_arrays_batch([array_id, "nonexistent"])

        assert len(retrieved) == 1  # Only existing array returned
        assert array_id in retrieved


class TestArrayManipulation:
    """Test array manipulation operations."""

    def test_flatten_multidimensional_array(self):
        """Test multidimensional arrays are flattened."""
        registry = ArrayRegistry()
        array_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

        array_id = registry.add_array(array_2d)
        retrieved = registry.get_array(array_id)

        assert retrieved.ndim == 1
        np.testing.assert_array_equal(retrieved, np.array([1.0, 2.0, 3.0, 4.0]))

    def test_remove_array(self):
        """Test removing array from registry."""
        registry = ArrayRegistry()
        array = np.array([1.0, 2.0])

        array_id = registry.add_array(array)
        assert array_id in registry

        removed = registry.remove_array(array_id)

        assert removed is True
        assert array_id not in registry
        assert len(registry) == 0

    def test_remove_nonexistent_array(self):
        """Test removing nonexistent array returns False."""
        registry = ArrayRegistry()

        removed = registry.remove_array("nonexistent")

        assert removed is False

    def test_clear_registry(self):
        """Test clearing all arrays from registry."""
        registry = ArrayRegistry()
        arrays = [np.array([1.0]), np.array([2.0]), np.array([3.0])]

        for arr in arrays:
            registry.add_array(arr)

        assert len(registry) == 3

        registry.clear()

        assert len(registry) == 0
        stats = registry.get_stats()
        assert stats["total_arrays"] == 0


class TestParquetIO:
    """Test Parquet save and load operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_save_empty_registry(self, temp_dir):
        """Test saving empty registry to Parquet."""
        registry = ArrayRegistry()
        filepath = temp_dir / "empty_arrays.parquet"

        registry.save_to_parquet(filepath)

        assert filepath.exists()

    def test_save_and_load_single_array(self, temp_dir):
        """Test save and load roundtrip with single array."""
        registry = ArrayRegistry()
        array = np.array([1.5, 2.5, 3.5, 4.5])

        array_id = registry.add_array(array, "y_true")

        # Save
        filepath = temp_dir / "arrays.parquet"
        registry.save_to_parquet(filepath)

        # Load into new registry
        new_registry = ArrayRegistry()
        new_registry.load_from_parquet(filepath)

        # Verify
        assert len(new_registry) == 1
        assert array_id in new_registry
        retrieved = new_registry.get_array(array_id)
        np.testing.assert_array_almost_equal(retrieved, array)

    def test_save_and_load_multiple_arrays(self, temp_dir):
        """Test save and load roundtrip with multiple arrays."""
        registry = ArrayRegistry()
        arrays = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0])
        ]
        types = ["y_true", "y_pred", "indices"]

        ids = registry.add_arrays_batch(arrays, types)

        # Save
        filepath = temp_dir / "arrays.parquet"
        registry.save_to_parquet(filepath)

        # Load
        new_registry = ArrayRegistry()
        new_registry.load_from_parquet(filepath)

        # Verify
        assert len(new_registry) == len(arrays)
        for i, array_id in enumerate(ids):
            retrieved = new_registry.get_array(array_id)
            np.testing.assert_array_almost_equal(retrieved, arrays[i])

    def test_save_and_load_with_deduplication(self, temp_dir):
        """Test save/load preserves deduplication."""
        registry = ArrayRegistry()
        array = np.array([1.0, 2.0, 3.0])

        id1 = registry.add_array(array)
        id2 = registry.add_array(array)  # Duplicate

        assert id1 == id2
        assert len(registry) == 1

        # Save and load
        filepath = temp_dir / "arrays.parquet"
        registry.save_to_parquet(filepath)

        new_registry = ArrayRegistry()
        new_registry.load_from_parquet(filepath)

        # Verify deduplication preserved
        assert len(new_registry) == 1
        assert id1 in new_registry

        # Adding same array again should still deduplicate
        id3 = new_registry.add_array(array)
        assert id3 == id1
        assert len(new_registry) == 1

    def test_load_from_nonexistent_file_raises(self, temp_dir):
        """Test loading from nonexistent file raises error."""
        registry = ArrayRegistry()
        filepath = temp_dir / "nonexistent.parquet"

        with pytest.raises(FileNotFoundError, match="not found"):
            registry.load_from_parquet(filepath)

    def test_save_creates_parent_directories(self, temp_dir):
        """Test save creates parent directories if needed."""
        registry = ArrayRegistry()
        registry.add_array(np.array([1.0, 2.0]))

        nested_path = temp_dir / "nested" / "dir" / "arrays.parquet"

        registry.save_to_parquet(nested_path)

        assert nested_path.exists()


class TestStatistics:
    """Test statistics and reporting."""

    def test_stats_empty_registry(self):
        """Test stats for empty registry."""
        registry = ArrayRegistry()
        stats = registry.get_stats()

        assert stats["total_arrays"] == 0
        assert stats["unique_hashes"] == 0
        assert stats["total_elements"] == 0
        assert stats["total_size_mb"] == 0.0
        assert stats["deduplication_ratio"] == 0.0
        assert stats["by_type"] == {}

    def test_stats_with_arrays(self):
        """Test stats with multiple arrays."""
        registry = ArrayRegistry()
        arrays = [
            np.array([1.0] * 100),  # 100 elements
            np.array([2.0] * 200),  # 200 elements
            np.array([3.0] * 300)   # 300 elements
        ]
        types = ["y_true", "y_pred", "indices"]

        registry.add_arrays_batch(arrays, types)

        stats = registry.get_stats()

        assert stats["total_arrays"] == 3
        assert stats["unique_hashes"] == 3
        assert stats["total_elements"] == 600
        assert stats["total_size_mb"] > 0
        assert stats["deduplication_ratio"] == 1.0  # No duplicates
        assert len(stats["by_type"]) == 3

    def test_stats_with_deduplication(self):
        """Test stats reflect deduplication."""
        registry = ArrayRegistry()
        array = np.array([1.0, 2.0, 3.0])

        # Add same array 5 times
        for _ in range(5):
            registry.add_array(array, "y_true")

        stats = registry.get_stats()

        assert stats["total_arrays"] == 1  # Only stored once
        assert stats["unique_hashes"] == 1
        # Note: deduplication_ratio is based on id_cache size
        # which tracks unique IDs, not add attempts

    def test_stats_by_type(self):
        """Test stats breakdown by array type."""
        registry = ArrayRegistry()

        # Add different types
        registry.add_array(np.array([1.0]), "y_true")
        registry.add_array(np.array([2.0]), "y_true")
        registry.add_array(np.array([3.0]), "y_pred")
        registry.add_array(np.array([4.0]), "indices")

        stats = registry.get_stats()

        assert stats["by_type"]["y_true"] == 2
        assert stats["by_type"]["y_pred"] == 1
        assert stats["by_type"]["indices"] == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_array(self):
        """Test handling of empty array."""
        registry = ArrayRegistry()
        array = np.array([])

        array_id = registry.add_array(array)

        assert array_id in registry
        retrieved = registry.get_array(array_id)
        assert len(retrieved) == 0

    def test_single_element_array(self):
        """Test single element array."""
        registry = ArrayRegistry()
        array = np.array([42.0])

        array_id = registry.add_array(array)
        retrieved = registry.get_array(array_id)

        np.testing.assert_array_equal(retrieved, array)

    def test_large_array(self):
        """Test handling of large array."""
        registry = ArrayRegistry()
        array = np.random.randn(10000)  # 10k elements

        array_id = registry.add_array(array)
        retrieved = registry.get_array(array_id)

        np.testing.assert_array_almost_equal(retrieved, array)

    def test_arrays_with_nan(self):
        """Test arrays containing NaN values."""
        registry = ArrayRegistry()
        array = np.array([1.0, np.nan, 3.0, np.nan])

        array_id = registry.add_array(array)
        retrieved = registry.get_array(array_id)

        # Use allclose with equal_nan for comparison
        assert np.allclose(retrieved, array, equal_nan=True)

    def test_arrays_with_inf(self):
        """Test arrays containing infinity."""
        registry = ArrayRegistry()
        array = np.array([1.0, np.inf, -np.inf, 4.0])

        array_id = registry.add_array(array)
        retrieved = registry.get_array(array_id)

        np.testing.assert_array_equal(retrieved, array)

    def test_negative_values(self):
        """Test arrays with negative values."""
        registry = ArrayRegistry()
        array = np.array([-1.0, -2.0, -3.0])

        array_id = registry.add_array(array)
        retrieved = registry.get_array(array_id)

        np.testing.assert_array_equal(retrieved, array)

    def test_very_small_values(self):
        """Test arrays with very small values."""
        registry = ArrayRegistry()
        array = np.array([1e-100, 1e-200, 1e-300])

        array_id = registry.add_array(array)
        retrieved = registry.get_array(array_id)

        np.testing.assert_array_almost_equal(retrieved, array)


class TestRepr:
    """Test string representation."""

    def test_repr_empty(self):
        """Test repr of empty registry."""
        registry = ArrayRegistry()
        repr_str = repr(registry)

        assert "ArrayRegistry" in repr_str
        assert "arrays=0" in repr_str

    def test_repr_with_data(self):
        """Test repr with data."""
        registry = ArrayRegistry()
        registry.add_array(np.array([1.0, 2.0]))

        repr_str = repr(registry)

        assert "ArrayRegistry" in repr_str
        assert "arrays=1" in repr_str
        assert "unique=1" in repr_str
        assert "MB)" in repr_str
