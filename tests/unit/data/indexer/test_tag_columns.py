"""
Unit tests for IndexStore tag column support.

Tests cover:
- Adding boolean, string, and numeric tag columns
- Setting and getting tag values
- Tag column serialization/deserialization
- Duplicate column name handling
- Removing tag columns
- Core column conflict detection
"""
import numpy as np
import polars as pl
import pytest
from nirs4all.data._indexer.index_store import IndexStore


class TestAddTagColumn:
    """Tests for add_tag_column method."""

    def test_add_boolean_tag_column(self):
        """Test adding a boolean tag column."""
        store = IndexStore()
        store.add_tag_column("is_outlier", pl.Boolean)

        assert "is_outlier" in store.columns
        assert store.has_tag_column("is_outlier")
        assert store.get_tag_dtype("is_outlier") == pl.Boolean

    def test_add_string_tag_column(self):
        """Test adding a string tag column."""
        store = IndexStore()
        store.add_tag_column("category", pl.Utf8)

        assert "category" in store.columns
        assert store.has_tag_column("category")
        assert store.get_tag_dtype("category") == pl.Utf8

    def test_add_numeric_tag_column_int(self):
        """Test adding an integer tag column."""
        store = IndexStore()
        store.add_tag_column("cluster_id", pl.Int32)

        assert "cluster_id" in store.columns
        assert store.has_tag_column("cluster_id")
        assert store.get_tag_dtype("cluster_id") == pl.Int32

    def test_add_numeric_tag_column_float(self):
        """Test adding a float tag column."""
        store = IndexStore()
        store.add_tag_column("quality_score", pl.Float64)

        assert "quality_score" in store.columns
        assert store.has_tag_column("quality_score")
        assert store.get_tag_dtype("quality_score") == pl.Float64

    def test_add_tag_column_with_string_dtype(self):
        """Test adding tag column with string dtype specification."""
        store = IndexStore()
        store.add_tag_column("bool_tag", "bool")
        store.add_tag_column("str_tag", "str")
        store.add_tag_column("int_tag", "int")
        store.add_tag_column("float_tag", "float")

        assert store.get_tag_dtype("bool_tag") == pl.Boolean
        assert store.get_tag_dtype("str_tag") == pl.Utf8
        assert store.get_tag_dtype("int_tag") == pl.Int32
        assert store.get_tag_dtype("float_tag") == pl.Float64

    def test_add_tag_column_default_dtype_is_boolean(self):
        """Test that default dtype for tag columns is Boolean."""
        store = IndexStore()
        store.add_tag_column("my_tag")

        assert store.get_tag_dtype("my_tag") == pl.Boolean


class TestSetAndGetTags:
    """Tests for set_tags and get_tags methods."""

    def test_set_and_get_tags_single_value(self):
        """Test setting a single value for multiple samples."""
        store = IndexStore()
        store.add_tag_column("is_outlier", pl.Boolean)

        # Add some rows
        store.append({
            "row": pl.Series([0, 1, 2], dtype=pl.Int32),
            "sample": pl.Series([0, 1, 2], dtype=pl.Int32),
            "origin": pl.Series([0, 1, 2], dtype=pl.Int32),
            "partition": pl.Series(["train", "train", "test"], dtype=pl.Categorical),
            "group": pl.Series([0, 0, 1], dtype=pl.Int8),
            "branch": pl.Series([0, 0, 0], dtype=pl.Int8),
            "processings": pl.Series([["raw"], ["raw"], ["raw"]], dtype=pl.List(pl.Utf8)),
            "augmentation": pl.Series([None, None, None], dtype=pl.Categorical),
            "excluded": pl.Series([False, False, False], dtype=pl.Boolean),
            "exclusion_reason": pl.Series([None, None, None], dtype=pl.Utf8),
            "is_outlier": pl.Series([None, None, None], dtype=pl.Boolean),
        })

        # Set single value for multiple samples
        store.set_tags([0, 1], "is_outlier", True)

        tags = store.get_tags("is_outlier")
        assert tags[0] is True
        assert tags[1] is True
        assert tags[2] is None

    def test_set_and_get_tags_list_values(self):
        """Test setting different values for each sample."""
        store = IndexStore()
        store.add_tag_column("cluster_id", pl.Int32)

        # Add some rows
        store.append({
            "row": pl.Series([0, 1, 2], dtype=pl.Int32),
            "sample": pl.Series([0, 1, 2], dtype=pl.Int32),
            "origin": pl.Series([0, 1, 2], dtype=pl.Int32),
            "partition": pl.Series(["train", "train", "test"], dtype=pl.Categorical),
            "group": pl.Series([0, 0, 1], dtype=pl.Int8),
            "branch": pl.Series([0, 0, 0], dtype=pl.Int8),
            "processings": pl.Series([["raw"], ["raw"], ["raw"]], dtype=pl.List(pl.Utf8)),
            "augmentation": pl.Series([None, None, None], dtype=pl.Categorical),
            "excluded": pl.Series([False, False, False], dtype=pl.Boolean),
            "exclusion_reason": pl.Series([None, None, None], dtype=pl.Utf8),
            "cluster_id": pl.Series([None, None, None], dtype=pl.Int32),
        })

        # Set different values
        store.set_tags([0, 1, 2], "cluster_id", [1, 2, 3])

        tags = store.get_tags("cluster_id")
        assert tags == [1, 2, 3]

    def test_get_tags_with_condition(self):
        """Test getting tags with a filter condition."""
        store = IndexStore()
        store.add_tag_column("is_outlier", pl.Boolean)

        # Add rows with different partitions
        store.append({
            "row": pl.Series([0, 1, 2], dtype=pl.Int32),
            "sample": pl.Series([0, 1, 2], dtype=pl.Int32),
            "origin": pl.Series([0, 1, 2], dtype=pl.Int32),
            "partition": pl.Series(["train", "train", "test"], dtype=pl.Categorical),
            "group": pl.Series([0, 0, 1], dtype=pl.Int8),
            "branch": pl.Series([0, 0, 0], dtype=pl.Int8),
            "processings": pl.Series([["raw"], ["raw"], ["raw"]], dtype=pl.List(pl.Utf8)),
            "augmentation": pl.Series([None, None, None], dtype=pl.Categorical),
            "excluded": pl.Series([False, False, False], dtype=pl.Boolean),
            "exclusion_reason": pl.Series([None, None, None], dtype=pl.Utf8),
            "is_outlier": pl.Series([True, False, True], dtype=pl.Boolean),
        })

        # Get tags for train partition only
        train_tags = store.get_tags("is_outlier", pl.col("partition") == "train")
        assert train_tags == [True, False]


class TestTagColumnSerialization:
    """Tests for tag column serialization and deserialization."""

    def test_tag_column_serialization(self):
        """Test that tag columns are serialized in to_dict()."""
        store = IndexStore()
        store.add_tag_column("is_outlier", pl.Boolean)
        store.add_tag_column("cluster_id", pl.Int32)

        state = store.to_dict()

        assert "tag_columns" in state
        assert "is_outlier" in state["tag_columns"]
        assert "cluster_id" in state["tag_columns"]
        assert state["tag_columns"]["is_outlier"] == "bool"
        assert state["tag_columns"]["cluster_id"] == "int32"

    def test_tag_column_deserialization(self):
        """Test that tag columns are restored from from_dict()."""
        # Create and populate store
        store = IndexStore()
        store.add_tag_column("is_outlier", pl.Boolean)
        store.add_tag_column("quality_score", pl.Float64)

        store.append({
            "row": pl.Series([0, 1], dtype=pl.Int32),
            "sample": pl.Series([0, 1], dtype=pl.Int32),
            "origin": pl.Series([0, 1], dtype=pl.Int32),
            "partition": pl.Series(["train", "train"], dtype=pl.Categorical),
            "group": pl.Series([0, 0], dtype=pl.Int8),
            "branch": pl.Series([0, 0], dtype=pl.Int8),
            "processings": pl.Series([["raw"], ["raw"]], dtype=pl.List(pl.Utf8)),
            "augmentation": pl.Series([None, None], dtype=pl.Categorical),
            "excluded": pl.Series([False, False], dtype=pl.Boolean),
            "exclusion_reason": pl.Series([None, None], dtype=pl.Utf8),
            "is_outlier": pl.Series([True, False], dtype=pl.Boolean),
            "quality_score": pl.Series([0.9, 0.5], dtype=pl.Float64),
        })

        # Serialize and deserialize
        state = store.to_dict()
        restored = IndexStore.from_dict(state)

        # Check tag columns are restored
        assert restored.has_tag_column("is_outlier")
        assert restored.has_tag_column("quality_score")
        assert restored.get_tag_dtype("is_outlier") == pl.Boolean
        assert restored.get_tag_dtype("quality_score") == pl.Float64

        # Check tag values are restored
        assert restored.get_tags("is_outlier") == [True, False]
        assert restored.get_tags("quality_score") == [0.9, 0.5]

    def test_serialization_roundtrip_preserves_data(self):
        """Test that serialization roundtrip preserves all data."""
        store = IndexStore()
        store.add_tag_column("tag1", pl.Boolean)
        store.add_tag_column("tag2", pl.Utf8)
        store.add_tag_column("tag3", pl.Int32)

        store.append({
            "row": pl.Series([0, 1, 2], dtype=pl.Int32),
            "sample": pl.Series([0, 1, 2], dtype=pl.Int32),
            "origin": pl.Series([0, 1, 2], dtype=pl.Int32),
            "partition": pl.Series(["train", "test", "train"], dtype=pl.Categorical),
            "group": pl.Series([1, 2, 1], dtype=pl.Int8),
            "branch": pl.Series([0, 0, 1], dtype=pl.Int8),
            "processings": pl.Series([["raw", "snv"], ["raw"], ["raw", "msc"]], dtype=pl.List(pl.Utf8)),
            "augmentation": pl.Series([None, "flip", None], dtype=pl.Categorical),
            "excluded": pl.Series([False, True, False], dtype=pl.Boolean),
            "exclusion_reason": pl.Series([None, "outlier", None], dtype=pl.Utf8),
            "tag1": pl.Series([True, False, None], dtype=pl.Boolean),
            "tag2": pl.Series(["a", "b", "c"], dtype=pl.Utf8),
            "tag3": pl.Series([1, 2, 3], dtype=pl.Int32),
        })

        # Roundtrip
        state = store.to_dict()
        restored = IndexStore.from_dict(state)

        # Verify all columns match
        assert len(restored) == 3
        assert restored.get_column("sample") == [0, 1, 2]
        assert restored.get_column("partition") == ["train", "test", "train"]
        assert restored.get_column("group") == [1, 2, 1]
        assert restored.get_tags("tag1") == [True, False, None]
        assert restored.get_tags("tag2") == ["a", "b", "c"]
        assert restored.get_tags("tag3") == [1, 2, 3]


class TestDuplicateTagColumnError:
    """Tests for duplicate tag column handling."""

    def test_duplicate_tag_column_error(self):
        """Test that adding a duplicate tag column raises ValueError."""
        store = IndexStore()
        store.add_tag_column("my_tag", pl.Boolean)

        with pytest.raises(ValueError, match="already exists"):
            store.add_tag_column("my_tag", pl.Boolean)

    def test_duplicate_tag_column_different_dtype_error(self):
        """Test that adding duplicate tag column with different dtype raises error."""
        store = IndexStore()
        store.add_tag_column("my_tag", pl.Boolean)

        with pytest.raises(ValueError, match="already exists"):
            store.add_tag_column("my_tag", pl.Int32)


class TestRemoveTagColumn:
    """Tests for remove_tag_column method."""

    def test_remove_tag_column(self):
        """Test removing a tag column."""
        store = IndexStore()
        store.add_tag_column("to_remove", pl.Boolean)
        store.add_tag_column("to_keep", pl.Int32)

        assert store.has_tag_column("to_remove")
        store.remove_tag_column("to_remove")

        assert not store.has_tag_column("to_remove")
        assert "to_remove" not in store.columns
        assert store.has_tag_column("to_keep")

    def test_remove_nonexistent_tag_column_error(self):
        """Test that removing nonexistent tag column raises ValueError."""
        store = IndexStore()

        with pytest.raises(ValueError, match="not found"):
            store.remove_tag_column("nonexistent")


class TestGetTagColumnNames:
    """Tests for get_tag_column_names method."""

    def test_get_tag_column_names_empty(self):
        """Test getting tag column names when none exist."""
        store = IndexStore()
        assert store.get_tag_column_names() == []

    def test_get_tag_column_names(self):
        """Test getting all tag column names."""
        store = IndexStore()
        store.add_tag_column("tag1", pl.Boolean)
        store.add_tag_column("tag2", pl.Int32)
        store.add_tag_column("tag3", pl.Float64)

        names = store.get_tag_column_names()
        assert set(names) == {"tag1", "tag2", "tag3"}


class TestCoreColumnConflict:
    """Tests for core column name conflict detection."""

    @pytest.mark.parametrize("core_column", [
        "row", "sample", "origin", "partition", "group",
        "branch", "processings", "augmentation", "excluded", "exclusion_reason"
    ])
    def test_core_column_conflict_error(self, core_column):
        """Test that adding a tag with core column name raises ValueError."""
        store = IndexStore()

        with pytest.raises(ValueError, match="conflicts with core column"):
            store.add_tag_column(core_column, pl.Boolean)


class TestEdgeCases:
    """Tests for edge cases in tag column operations."""

    def test_set_tags_empty_indices(self):
        """Test that setting tags with empty indices does nothing."""
        store = IndexStore()
        store.add_tag_column("my_tag", pl.Boolean)

        # Should not raise
        store.set_tags([], "my_tag", True)

    def test_set_tags_nonexistent_tag_error(self):
        """Test that setting nonexistent tag raises ValueError."""
        store = IndexStore()

        with pytest.raises(ValueError, match="not found"):
            store.set_tags([0], "nonexistent", True)

    def test_get_tags_nonexistent_tag_error(self):
        """Test that getting nonexistent tag raises ValueError."""
        store = IndexStore()

        with pytest.raises(ValueError, match="not found"):
            store.get_tags("nonexistent")

    def test_set_tags_values_length_mismatch_error(self):
        """Test that mismatched values length raises ValueError."""
        store = IndexStore()
        store.add_tag_column("my_tag", pl.Int32)

        # Add some rows first
        store.append({
            "row": pl.Series([0, 1, 2], dtype=pl.Int32),
            "sample": pl.Series([0, 1, 2], dtype=pl.Int32),
            "origin": pl.Series([0, 1, 2], dtype=pl.Int32),
            "partition": pl.Series(["train", "train", "train"], dtype=pl.Categorical),
            "group": pl.Series([0, 0, 0], dtype=pl.Int8),
            "branch": pl.Series([0, 0, 0], dtype=pl.Int8),
            "processings": pl.Series([["raw"], ["raw"], ["raw"]], dtype=pl.List(pl.Utf8)),
            "augmentation": pl.Series([None, None, None], dtype=pl.Categorical),
            "excluded": pl.Series([False, False, False], dtype=pl.Boolean),
            "exclusion_reason": pl.Series([None, None, None], dtype=pl.Utf8),
            "my_tag": pl.Series([None, None, None], dtype=pl.Int32),
        })

        with pytest.raises(ValueError, match="must match"):
            store.set_tags([0, 1, 2], "my_tag", [1, 2])  # Only 2 values for 3 indices

    def test_unknown_dtype_string_error(self):
        """Test that unknown dtype string raises ValueError."""
        store = IndexStore()

        with pytest.raises(ValueError, match="Unknown dtype"):
            store.add_tag_column("my_tag", "unknown_dtype")
