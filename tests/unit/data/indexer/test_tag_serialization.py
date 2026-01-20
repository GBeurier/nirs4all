"""
Unit tests for tag column serialization.

Tests cover:
- IndexStore to_dict/from_dict roundtrip with tags
- Tag data persistence across serialization
- Empty tag columns serialization
- Mixed data types serialization
- JSON compatibility of serialized state
"""
import json
import polars as pl
import pytest
from nirs4all.data._indexer.index_store import IndexStore


class TestTagSerializationRoundtrip:
    """Tests for tag serialization roundtrip."""

    def test_empty_store_roundtrip(self):
        """Test serialization of empty store with no tags."""
        store = IndexStore()

        state = store.to_dict()
        restored = IndexStore.from_dict(state)

        assert len(restored) == 0
        assert restored.get_tag_column_names() == []

    def test_store_with_tags_no_data_roundtrip(self):
        """Test serialization of store with tag columns but no data."""
        store = IndexStore()
        store.add_tag_column("is_outlier", pl.Boolean)
        store.add_tag_column("cluster_id", pl.Int32)
        store.add_tag_column("quality", pl.Float64)

        state = store.to_dict()
        restored = IndexStore.from_dict(state)

        assert len(restored) == 0
        assert set(restored.get_tag_column_names()) == {"is_outlier", "cluster_id", "quality"}
        assert restored.get_tag_dtype("is_outlier") == pl.Boolean
        assert restored.get_tag_dtype("cluster_id") == pl.Int32
        assert restored.get_tag_dtype("quality") == pl.Float64

    def test_full_store_roundtrip(self):
        """Test serialization of store with data and tags."""
        store = IndexStore()
        store.add_tag_column("is_outlier", pl.Boolean)
        store.add_tag_column("cluster_id", pl.Int32)

        # Add data
        store.append({
            "row": pl.Series([0, 1, 2], dtype=pl.Int32),
            "sample": pl.Series([0, 1, 2], dtype=pl.Int32),
            "origin": pl.Series([0, 1, 2], dtype=pl.Int32),
            "partition": pl.Series(["train", "train", "test"], dtype=pl.Categorical),
            "group": pl.Series([1, 1, 2], dtype=pl.Int8),
            "branch": pl.Series([0, 0, 0], dtype=pl.Int8),
            "processings": pl.Series([["raw", "snv"], ["raw"], ["raw", "msc"]], dtype=pl.List(pl.Utf8)),
            "augmentation": pl.Series([None, "flip", None], dtype=pl.Categorical),
            "excluded": pl.Series([False, True, False], dtype=pl.Boolean),
            "exclusion_reason": pl.Series([None, "test", None], dtype=pl.Utf8),
            "is_outlier": pl.Series([True, False, True], dtype=pl.Boolean),
            "cluster_id": pl.Series([1, 2, 1], dtype=pl.Int32),
        })

        # Serialize and restore
        state = store.to_dict()
        restored = IndexStore.from_dict(state)

        # Verify structure
        assert len(restored) == 3
        assert restored.has_tag_column("is_outlier")
        assert restored.has_tag_column("cluster_id")

        # Verify data
        assert restored.get_column("sample") == [0, 1, 2]
        assert restored.get_column("partition") == ["train", "train", "test"]
        assert restored.get_tags("is_outlier") == [True, False, True]
        assert restored.get_tags("cluster_id") == [1, 2, 1]

    def test_null_values_preserved(self):
        """Test that null tag values are preserved through serialization."""
        store = IndexStore()
        store.add_tag_column("tag", pl.Int32)

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
            "tag": pl.Series([1, None, 3], dtype=pl.Int32),
        })

        state = store.to_dict()
        restored = IndexStore.from_dict(state)

        tags = restored.get_tags("tag")
        assert tags[0] == 1
        assert tags[1] is None
        assert tags[2] == 3


class TestJsonCompatibility:
    """Tests for JSON compatibility of serialized state."""

    def test_state_is_json_serializable(self):
        """Test that to_dict() output is JSON-serializable."""
        store = IndexStore()
        store.add_tag_column("is_outlier", pl.Boolean)
        store.add_tag_column("score", pl.Float64)

        store.append({
            "row": pl.Series([0, 1], dtype=pl.Int32),
            "sample": pl.Series([0, 1], dtype=pl.Int32),
            "origin": pl.Series([0, 1], dtype=pl.Int32),
            "partition": pl.Series(["train", "test"], dtype=pl.Categorical),
            "group": pl.Series([0, 1], dtype=pl.Int8),
            "branch": pl.Series([0, 0], dtype=pl.Int8),
            "processings": pl.Series([["raw"], ["raw", "snv"]], dtype=pl.List(pl.Utf8)),
            "augmentation": pl.Series([None, None], dtype=pl.Categorical),
            "excluded": pl.Series([False, False], dtype=pl.Boolean),
            "exclusion_reason": pl.Series([None, None], dtype=pl.Utf8),
            "is_outlier": pl.Series([True, False], dtype=pl.Boolean),
            "score": pl.Series([0.9, 0.5], dtype=pl.Float64),
        })

        state = store.to_dict()

        # Should not raise
        json_str = json.dumps(state)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_json_roundtrip(self):
        """Test full JSON roundtrip."""
        store = IndexStore()
        store.add_tag_column("is_outlier", pl.Boolean)
        store.add_tag_column("cluster", pl.Int32)

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
            "cluster": pl.Series([1, 2], dtype=pl.Int32),
        })

        # Serialize to JSON and back
        state = store.to_dict()
        json_str = json.dumps(state)
        loaded_state = json.loads(json_str)
        restored = IndexStore.from_dict(loaded_state)

        # Verify
        assert restored.get_tags("is_outlier") == [True, False]
        assert restored.get_tags("cluster") == [1, 2]


class TestAllDataTypes:
    """Tests for serialization of all supported tag data types."""

    def test_boolean_tag_serialization(self):
        """Test boolean tag serialization."""
        store = IndexStore()
        store.add_tag_column("flag", pl.Boolean)

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
            "flag": pl.Series([True, False], dtype=pl.Boolean),
        })

        state = store.to_dict()
        assert state["tag_columns"]["flag"] == "bool"

        restored = IndexStore.from_dict(state)
        assert restored.get_tags("flag") == [True, False]

    def test_string_tag_serialization(self):
        """Test string tag serialization."""
        store = IndexStore()
        store.add_tag_column("category", pl.Utf8)

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
            "category": pl.Series(["A", "B"], dtype=pl.Utf8),
        })

        state = store.to_dict()
        assert state["tag_columns"]["category"] == "str"

        restored = IndexStore.from_dict(state)
        assert restored.get_tags("category") == ["A", "B"]

    def test_int32_tag_serialization(self):
        """Test Int32 tag serialization."""
        store = IndexStore()
        store.add_tag_column("count", pl.Int32)

        store.append({
            "row": pl.Series([0], dtype=pl.Int32),
            "sample": pl.Series([0], dtype=pl.Int32),
            "origin": pl.Series([0], dtype=pl.Int32),
            "partition": pl.Series(["train"], dtype=pl.Categorical),
            "group": pl.Series([0], dtype=pl.Int8),
            "branch": pl.Series([0], dtype=pl.Int8),
            "processings": pl.Series([["raw"]], dtype=pl.List(pl.Utf8)),
            "augmentation": pl.Series([None], dtype=pl.Categorical),
            "excluded": pl.Series([False], dtype=pl.Boolean),
            "exclusion_reason": pl.Series([None], dtype=pl.Utf8),
            "count": pl.Series([42], dtype=pl.Int32),
        })

        state = store.to_dict()
        assert state["tag_columns"]["count"] == "int32"

        restored = IndexStore.from_dict(state)
        assert restored.get_tags("count") == [42]

    def test_float64_tag_serialization(self):
        """Test Float64 tag serialization."""
        store = IndexStore()
        store.add_tag_column("score", pl.Float64)

        store.append({
            "row": pl.Series([0], dtype=pl.Int32),
            "sample": pl.Series([0], dtype=pl.Int32),
            "origin": pl.Series([0], dtype=pl.Int32),
            "partition": pl.Series(["train"], dtype=pl.Categorical),
            "group": pl.Series([0], dtype=pl.Int8),
            "branch": pl.Series([0], dtype=pl.Int8),
            "processings": pl.Series([["raw"]], dtype=pl.List(pl.Utf8)),
            "augmentation": pl.Series([None], dtype=pl.Categorical),
            "excluded": pl.Series([False], dtype=pl.Boolean),
            "exclusion_reason": pl.Series([None], dtype=pl.Utf8),
            "score": pl.Series([3.14159], dtype=pl.Float64),
        })

        state = store.to_dict()
        assert state["tag_columns"]["score"] == "float64"

        restored = IndexStore.from_dict(state)
        assert restored.get_tags("score") == [pytest.approx(3.14159)]


class TestEdgeCases:
    """Tests for edge cases in serialization."""

    def test_many_tag_columns(self):
        """Test serialization with many tag columns."""
        store = IndexStore()

        # Add many tag columns
        for i in range(10):
            store.add_tag_column(f"tag_{i}", pl.Int32)

        state = store.to_dict()
        restored = IndexStore.from_dict(state)

        assert len(restored.get_tag_column_names()) == 10
        for i in range(10):
            assert restored.has_tag_column(f"tag_{i}")

    def test_special_characters_in_tag_values(self):
        """Test serialization with special characters in string values."""
        store = IndexStore()
        store.add_tag_column("label", pl.Utf8)

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
            "label": pl.Series(["hello\nworld", "test\ttab"], dtype=pl.Utf8),
        })

        # JSON roundtrip
        state = store.to_dict()
        json_str = json.dumps(state)
        loaded_state = json.loads(json_str)
        restored = IndexStore.from_dict(loaded_state)

        assert restored.get_tags("label") == ["hello\nworld", "test\ttab"]

    def test_unicode_in_tag_values(self):
        """Test serialization with Unicode characters."""
        store = IndexStore()
        store.add_tag_column("label", pl.Utf8)

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
            "label": pl.Series(["日本語", "émoji 🎉"], dtype=pl.Utf8),
        })

        # JSON roundtrip
        state = store.to_dict()
        json_str = json.dumps(state, ensure_ascii=False)
        loaded_state = json.loads(json_str)
        restored = IndexStore.from_dict(loaded_state)

        assert restored.get_tags("label") == ["日本語", "émoji 🎉"]
