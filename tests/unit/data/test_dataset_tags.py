"""
Unit tests for SpectroDataset tag operations API.

Tests cover:
- add_tag() method
- set_tag() method
- get_tag() method
- tags property
- tag_info() method
- remove_tag() method
- has_tag() method
- Integration with dataset selectors
"""
import numpy as np
import pytest
from nirs4all.data.dataset import SpectroDataset


class TestAddTag:
    """Tests for add_tag method."""

    def test_add_boolean_tag(self):
        """Test adding a boolean tag."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})

        dataset.add_tag("is_outlier", "bool")

        assert dataset.has_tag("is_outlier")
        assert "is_outlier" in dataset.tags

    def test_add_integer_tag(self):
        """Test adding an integer tag."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})

        dataset.add_tag("cluster_id", "int")

        assert dataset.has_tag("cluster_id")

    def test_add_float_tag(self):
        """Test adding a float tag."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})

        dataset.add_tag("quality_score", "float")

        assert dataset.has_tag("quality_score")

    def test_add_string_tag(self):
        """Test adding a string tag."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})

        dataset.add_tag("category", "str")

        assert dataset.has_tag("category")

    def test_add_tag_default_dtype(self):
        """Test that default dtype is boolean."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})

        dataset.add_tag("my_tag")  # No dtype specified

        # Should be able to set boolean values
        dataset.set_tag("my_tag", [0, 1, 2], True)
        assert dataset.has_tag("my_tag")


class TestSetTag:
    """Tests for set_tag method."""

    def test_set_tag_single_value(self):
        """Test setting a single value for multiple samples."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})
        dataset.add_tag("is_outlier", "bool")

        dataset.set_tag("is_outlier", [0, 1, 2], True)

        values = dataset.get_tag("is_outlier")
        assert values[0] is True
        assert values[1] is True
        assert values[2] is True
        assert values[3] is None
        assert values[4] is None

    def test_set_tag_list_values(self):
        """Test setting different values for each sample."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})
        dataset.add_tag("cluster_id", "int")

        dataset.set_tag("cluster_id", [0, 1, 2, 3, 4], [1, 1, 2, 2, 3])

        values = dataset.get_tag("cluster_id")
        assert values.tolist() == [1, 1, 2, 2, 3]

    def test_set_tag_with_numpy_indices(self):
        """Test setting tags with numpy array indices."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})
        dataset.add_tag("is_outlier", "bool")

        indices = np.array([0, 2, 4])
        dataset.set_tag("is_outlier", indices, True)

        values = dataset.get_tag("is_outlier")
        assert values[0] is True
        assert values[1] is None
        assert values[2] is True
        assert values[3] is None
        assert values[4] is True


class TestGetTag:
    """Tests for get_tag method."""

    def test_get_tag_returns_numpy_array(self):
        """Test that get_tag returns a numpy array."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})
        dataset.add_tag("my_tag", "int")
        dataset.set_tag("my_tag", list(range(5)), list(range(5)))

        values = dataset.get_tag("my_tag")

        assert isinstance(values, np.ndarray)

    def test_get_tag_with_selector(self):
        """Test getting tags with selector filter."""
        dataset = SpectroDataset("test")
        # Add train samples
        dataset.add_samples(np.random.rand(3, 10), {"partition": "train"})
        # Add test samples
        dataset.add_samples(np.random.rand(2, 10), {"partition": "test"})

        dataset.add_tag("cluster_id", "int")
        dataset.set_tag("cluster_id", [0, 1, 2, 3, 4], [1, 2, 3, 4, 5])

        # Get only train tags
        train_tags = dataset.get_tag("cluster_id", {"partition": "train"})
        assert len(train_tags) == 3
        assert train_tags.tolist() == [1, 2, 3]

        # Get only test tags
        test_tags = dataset.get_tag("cluster_id", {"partition": "test"})
        assert len(test_tags) == 2
        assert test_tags.tolist() == [4, 5]


class TestTagsProperty:
    """Tests for tags property."""

    def test_tags_empty(self):
        """Test tags property when no tags exist."""
        dataset = SpectroDataset("test")

        assert dataset.tags == []

    def test_tags_with_multiple_tags(self):
        """Test tags property with multiple tags."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})
        dataset.add_tag("tag1", "bool")
        dataset.add_tag("tag2", "int")
        dataset.add_tag("tag3", "float")

        tags = dataset.tags
        assert set(tags) == {"tag1", "tag2", "tag3"}


class TestTagInfo:
    """Tests for tag_info method."""

    def test_tag_info_basic(self):
        """Test tag_info returns correct metadata."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})
        dataset.add_tag("is_outlier", "bool")
        dataset.set_tag("is_outlier", [0, 1], True)
        dataset.set_tag("is_outlier", [2, 3], False)
        # Sample 4 remains None

        info = dataset.tag_info()

        assert "is_outlier" in info
        assert info["is_outlier"]["non_null_count"] == 4
        assert info["is_outlier"]["total_count"] == 5
        assert set(info["is_outlier"]["unique_values"]) == {True, False}

    def test_tag_info_multiple_tags(self):
        """Test tag_info with multiple tags."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})
        dataset.add_tag("tag1", "bool")
        dataset.add_tag("tag2", "int")

        dataset.set_tag("tag1", [0, 1, 2], True)
        dataset.set_tag("tag2", list(range(5)), [1, 2, 3, 4, 5])

        info = dataset.tag_info()

        assert len(info) == 2
        assert "tag1" in info
        assert "tag2" in info
        assert info["tag1"]["non_null_count"] == 3
        assert info["tag2"]["non_null_count"] == 5


class TestRemoveTag:
    """Tests for remove_tag method."""

    def test_remove_tag(self):
        """Test removing a tag."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})
        dataset.add_tag("to_remove", "bool")
        dataset.add_tag("to_keep", "int")

        assert dataset.has_tag("to_remove")
        dataset.remove_tag("to_remove")

        assert not dataset.has_tag("to_remove")
        assert dataset.has_tag("to_keep")

    def test_remove_nonexistent_tag_error(self):
        """Test that removing nonexistent tag raises error."""
        dataset = SpectroDataset("test")

        with pytest.raises(ValueError, match="not found"):
            dataset.remove_tag("nonexistent")


class TestHasTag:
    """Tests for has_tag method."""

    def test_has_tag_true(self):
        """Test has_tag returns True when tag exists."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})
        dataset.add_tag("my_tag", "bool")

        assert dataset.has_tag("my_tag") is True

    def test_has_tag_false(self):
        """Test has_tag returns False when tag doesn't exist."""
        dataset = SpectroDataset("test")

        assert dataset.has_tag("nonexistent") is False


class TestTagIntegration:
    """Integration tests for tags with dataset operations."""

    def test_tags_survive_copy(self):
        """Test that tags are preserved when copying dataset components."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})
        dataset.add_tag("is_outlier", "bool")
        dataset.set_tag("is_outlier", [0, 1], True)

        # Verify original
        assert dataset.has_tag("is_outlier")
        values = dataset.get_tag("is_outlier")
        assert values[0] is True
        assert values[1] is True

    def test_tags_with_mixed_partitions(self):
        """Test tags work correctly with multiple partitions."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(3, 10), {"partition": "train"})
        dataset.add_samples(np.random.rand(3, 10), {"partition": "test"})
        dataset.add_samples(np.random.rand(2, 10), {"partition": "val"})

        dataset.add_tag("quality", "float")
        # Set different quality scores
        dataset.set_tag("quality", list(range(8)), [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

        # Verify filtering works
        train_quality = dataset.get_tag("quality", {"partition": "train"})
        assert len(train_quality) == 3
        assert train_quality.tolist() == [0.9, 0.8, 0.7]

        test_quality = dataset.get_tag("quality", {"partition": "test"})
        assert len(test_quality) == 3
        assert test_quality.tolist() == [0.6, 0.5, 0.4]

    def test_multiple_tags_independent(self):
        """Test that multiple tags are independent."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})

        dataset.add_tag("tag_a", "bool")
        dataset.add_tag("tag_b", "int")
        dataset.add_tag("tag_c", "float")

        # Set values independently
        dataset.set_tag("tag_a", [0, 1], True)
        dataset.set_tag("tag_b", [2, 3], [42, 43])
        dataset.set_tag("tag_c", [4], 3.14)

        # Verify independence
        a_values = dataset.get_tag("tag_a")
        b_values = dataset.get_tag("tag_b")
        c_values = dataset.get_tag("tag_c")

        assert a_values[0] is True
        assert a_values[2] is None
        assert b_values[2] == 42
        assert b_values[0] is None
        assert c_values[4] == pytest.approx(3.14)
        assert c_values[0] is None


class TestErrorHandling:
    """Tests for error handling in tag operations."""

    def test_set_tag_nonexistent_error(self):
        """Test error when setting nonexistent tag."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})

        with pytest.raises(ValueError, match="not found"):
            dataset.set_tag("nonexistent", [0], True)

    def test_get_tag_nonexistent_error(self):
        """Test error when getting nonexistent tag."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})

        with pytest.raises(ValueError, match="not found"):
            dataset.get_tag("nonexistent")

    def test_add_duplicate_tag_error(self):
        """Test error when adding duplicate tag."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})
        dataset.add_tag("my_tag", "bool")

        with pytest.raises(ValueError, match="already exists"):
            dataset.add_tag("my_tag", "int")

    def test_add_tag_core_column_conflict(self):
        """Test error when tag name conflicts with core column."""
        dataset = SpectroDataset("test")
        dataset.add_samples(np.random.rand(5, 10), {"partition": "train"})

        with pytest.raises(ValueError, match="conflicts"):
            dataset.add_tag("sample", "int")
