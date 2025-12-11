"""
Unit tests for MetadataFilter class.

Tests the metadata-based filter with various filtering modes:
- Custom condition function
- Values to exclude
- Values to keep
"""

import numpy as np
import pytest

from nirs4all.operators.filters import MetadataFilter
from nirs4all.operators.filters.base import SampleFilter


class TestMetadataFilterInitialization:
    """Tests for MetadataFilter initialization and parameter validation."""

    def test_initialization_with_condition(self):
        """Test initialization with condition function."""
        filter_obj = MetadataFilter(
            column="quality",
            condition=lambda x: x == "good"
        )
        assert filter_obj.column == "quality"
        assert filter_obj.condition is not None
        assert filter_obj.values_to_exclude is None
        assert filter_obj.values_to_keep is None

    def test_initialization_with_values_to_exclude(self):
        """Test initialization with values_to_exclude."""
        filter_obj = MetadataFilter(
            column="quality",
            values_to_exclude=["bad", "corrupted"]
        )
        assert filter_obj.column == "quality"
        assert filter_obj.values_to_exclude == ["bad", "corrupted"]
        assert filter_obj._exclude_set == {"bad", "corrupted"}

    def test_initialization_with_values_to_keep(self):
        """Test initialization with values_to_keep."""
        filter_obj = MetadataFilter(
            column="quality",
            values_to_keep=["good", "excellent"]
        )
        assert filter_obj.column == "quality"
        assert filter_obj.values_to_keep == ["good", "excellent"]
        assert filter_obj._keep_set == {"good", "excellent"}

    def test_missing_column_raises_error(self):
        """Test that missing column raises ValueError."""
        with pytest.raises(ValueError, match="column must be provided"):
            MetadataFilter(column="", values_to_exclude=["bad"])

    def test_no_criterion_raises_error(self):
        """Test that no filtering criterion raises ValueError."""
        with pytest.raises(ValueError, match="One of .* must be provided"):
            MetadataFilter(column="quality")

    def test_multiple_criteria_raises_error(self):
        """Test that multiple criteria raises ValueError."""
        with pytest.raises(ValueError, match="Only one of"):
            MetadataFilter(
                column="quality",
                values_to_exclude=["bad"],
                values_to_keep=["good"]
            )

        with pytest.raises(ValueError, match="Only one of"):
            MetadataFilter(
                column="quality",
                condition=lambda x: x == "good",
                values_to_exclude=["bad"]
            )

    def test_custom_reason(self):
        """Test custom reason."""
        filter_obj = MetadataFilter(
            column="quality",
            values_to_exclude=["bad"],
            reason="custom_reason"
        )
        assert filter_obj.exclusion_reason == "custom_reason"

    def test_default_reason(self):
        """Test default reason."""
        filter_obj = MetadataFilter(
            column="quality",
            values_to_exclude=["bad"]
        )
        assert filter_obj.exclusion_reason == "metadata_quality"

    def test_is_sample_filter_subclass(self):
        """Test that MetadataFilter is a SampleFilter subclass."""
        filter_obj = MetadataFilter(column="x", values_to_exclude=["a"])
        assert isinstance(filter_obj, SampleFilter)


class TestMetadataFilterValuesToExclude:
    """Tests for values_to_exclude mode."""

    def test_excludes_specified_values(self):
        """Test that specified values are excluded."""
        X = np.random.randn(10, 5)
        metadata = {
            "quality": np.array(["good", "bad", "good", "bad", "good",
                                "good", "bad", "good", "good", "good"])
        }

        filter_obj = MetadataFilter(column="quality", values_to_exclude=["bad"])
        mask = filter_obj.get_mask(X, metadata=metadata)

        # "bad" at indices 1, 3, 6 should be excluded
        assert mask[0] == True   # noqa: E712
        assert mask[1] == False  # noqa: E712
        assert mask[3] == False  # noqa: E712
        assert mask[6] == False  # noqa: E712
        assert mask.sum() == 7

    def test_excludes_multiple_values(self):
        """Test exclusion of multiple values."""
        X = np.random.randn(5, 5)
        metadata = {
            "status": np.array(["ok", "bad", "corrupted", "ok", "bad"])
        }

        filter_obj = MetadataFilter(
            column="status",
            values_to_exclude=["bad", "corrupted"]
        )
        mask = filter_obj.get_mask(X, metadata=metadata)

        assert mask[0] == True   # noqa: E712  ok
        assert mask[1] == False  # noqa: E712  bad
        assert mask[2] == False  # noqa: E712  corrupted
        assert mask[3] == True   # noqa: E712  ok
        assert mask[4] == False  # noqa: E712  bad

    def test_keeps_all_if_no_excluded_values_present(self):
        """Test that all are kept if no excluded values present."""
        X = np.random.randn(5, 5)
        metadata = {
            "quality": np.array(["good", "good", "good", "good", "good"])
        }

        filter_obj = MetadataFilter(column="quality", values_to_exclude=["bad"])
        mask = filter_obj.get_mask(X, metadata=metadata)

        assert mask.sum() == 5


class TestMetadataFilterValuesToKeep:
    """Tests for values_to_keep mode."""

    def test_keeps_only_specified_values(self):
        """Test that only specified values are kept."""
        X = np.random.randn(5, 5)
        metadata = {
            "type": np.array(["control", "treatment", "other", "control", "treatment"])
        }

        filter_obj = MetadataFilter(
            column="type",
            values_to_keep=["control", "treatment"]
        )
        mask = filter_obj.get_mask(X, metadata=metadata)

        assert mask[0] == True   # noqa: E712  control
        assert mask[1] == True   # noqa: E712  treatment
        assert mask[2] == False  # noqa: E712  other
        assert mask[3] == True   # noqa: E712  control
        assert mask[4] == True   # noqa: E712  treatment

    def test_excludes_all_if_no_kept_values_present(self):
        """Test that all are excluded if no kept values present."""
        X = np.random.randn(5, 5)
        metadata = {
            "type": np.array(["other", "other", "other", "other", "other"])
        }

        filter_obj = MetadataFilter(
            column="type",
            values_to_keep=["control", "treatment"]
        )
        mask = filter_obj.get_mask(X, metadata=metadata)

        assert mask.sum() == 0


class TestMetadataFilterCondition:
    """Tests for custom condition mode."""

    def test_condition_function(self):
        """Test custom condition function."""
        X = np.random.randn(5, 5)
        metadata = {
            "temperature": np.array([15, 25, 35, 20, 30])
        }

        filter_obj = MetadataFilter(
            column="temperature",
            condition=lambda x: 20 <= x <= 30
        )
        mask = filter_obj.get_mask(X, metadata=metadata)

        assert mask[0] == False  # noqa: E712  15
        assert mask[1] == True   # noqa: E712  25
        assert mask[2] == False  # noqa: E712  35
        assert mask[3] == True   # noqa: E712  20
        assert mask[4] == True   # noqa: E712  30

    def test_condition_with_string_values(self):
        """Test condition with string values."""
        X = np.random.randn(4, 5)
        metadata = {
            "name": np.array(["sample_a", "sample_b", "test_1", "sample_c"])
        }

        filter_obj = MetadataFilter(
            column="name",
            condition=lambda x: x.startswith("sample")
        )
        mask = filter_obj.get_mask(X, metadata=metadata)

        assert mask[0] == True   # noqa: E712
        assert mask[1] == True   # noqa: E712
        assert mask[2] == False  # noqa: E712
        assert mask[3] == True   # noqa: E712

    def test_condition_exception_excludes_sample(self):
        """Test that condition exception excludes the sample."""
        X = np.random.randn(3, 5)
        metadata = {
            "value": np.array([1, None, 3])  # None will cause exception
        }

        filter_obj = MetadataFilter(
            column="value",
            condition=lambda x: x > 0  # Will fail on None
        )
        mask = filter_obj.get_mask(X, metadata=metadata)

        assert mask[0] == True   # noqa: E712
        assert mask[1] == False  # noqa: E712  Exception
        assert mask[2] == True   # noqa: E712


class TestMetadataFilterMissingValues:
    """Tests for handling missing values."""

    def test_excludes_none_by_default(self):
        """Test that None values are excluded by default."""
        X = np.random.randn(5, 5)
        metadata = {
            "quality": np.array(["good", None, "good", None, "good"], dtype=object)
        }

        filter_obj = MetadataFilter(
            column="quality",
            values_to_keep=["good"],
            exclude_missing=True
        )
        mask = filter_obj.get_mask(X, metadata=metadata)

        assert mask[1] == False  # noqa: E712  None
        assert mask[3] == False  # noqa: E712  None

    def test_excludes_nan_by_default(self):
        """Test that NaN values are excluded by default."""
        X = np.random.randn(5, 5)
        metadata = {
            "value": np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        }

        filter_obj = MetadataFilter(
            column="value",
            condition=lambda x: x > 0,
            exclude_missing=True
        )
        mask = filter_obj.get_mask(X, metadata=metadata)

        assert mask[1] == False  # noqa: E712  NaN
        assert mask[3] == False  # noqa: E712  NaN

    def test_keeps_missing_when_disabled(self):
        """Test that missing values are kept when exclude_missing=False."""
        X = np.random.randn(5, 5)
        metadata = {
            "quality": np.array(["good", None, "good", None, "good"], dtype=object)
        }

        filter_obj = MetadataFilter(
            column="quality",
            values_to_keep=["good", None],  # Explicitly keep None
            exclude_missing=False
        )
        mask = filter_obj.get_mask(X, metadata=metadata)

        # None should now be kept (if in values_to_keep)
        assert mask[1] == True  # noqa: E712


class TestMetadataFilterErrorHandling:
    """Tests for error handling."""

    def test_no_metadata_raises_error(self):
        """Test that missing metadata raises ValueError."""
        X = np.random.randn(5, 5)

        filter_obj = MetadataFilter(column="quality", values_to_exclude=["bad"])

        with pytest.raises(ValueError, match="requires metadata"):
            filter_obj.get_mask(X, metadata=None)

    def test_missing_column_raises_error(self):
        """Test that missing column raises KeyError."""
        X = np.random.randn(5, 5)
        metadata = {"other_column": np.array([1, 2, 3, 4, 5])}

        filter_obj = MetadataFilter(column="quality", values_to_exclude=["bad"])

        with pytest.raises(KeyError, match="not found"):
            filter_obj.get_mask(X, metadata=metadata)

    def test_length_mismatch_raises_error(self):
        """Test that length mismatch raises ValueError."""
        X = np.random.randn(5, 5)
        metadata = {"quality": np.array(["good", "bad", "good"])}  # Wrong length

        filter_obj = MetadataFilter(column="quality", values_to_exclude=["bad"])

        with pytest.raises(ValueError, match="does not match"):
            filter_obj.get_mask(X, metadata=metadata)


class TestMetadataFilterDataFrameSupport:
    """Tests for pandas DataFrame support."""

    def test_works_with_dict_metadata(self):
        """Test that dict metadata works."""
        X = np.random.randn(5, 5)
        metadata = {"quality": np.array(["good", "bad", "good", "bad", "good"])}

        filter_obj = MetadataFilter(column="quality", values_to_exclude=["bad"])
        mask = filter_obj.get_mask(X, metadata=metadata)

        assert mask.sum() == 3

    def test_works_with_dataframe_like(self):
        """Test with DataFrame-like object (has .values attribute)."""
        X = np.random.randn(5, 5)

        # Mock DataFrame-like column
        class MockSeries:
            def __init__(self, data):
                self.values = np.array(data)

        class MockDataFrame:
            def __init__(self):
                self._data = {
                    "quality": MockSeries(["good", "bad", "good", "bad", "good"])
                }

            def __getitem__(self, key):
                return self._data[key]

            def keys(self):
                return self._data.keys()

        metadata = MockDataFrame()

        filter_obj = MetadataFilter(column="quality", values_to_exclude=["bad"])
        mask = filter_obj.get_mask(X, metadata=metadata)

        assert mask.sum() == 3


class TestMetadataFilterHelperMethods:
    """Tests for helper methods."""

    def test_fit_is_noop(self):
        """Test that fit is a no-op."""
        X = np.random.randn(10, 5)

        filter_obj = MetadataFilter(column="x", values_to_exclude=["a"])
        result = filter_obj.fit(X)

        assert result is filter_obj

    def test_get_filter_stats_with_metadata(self):
        """Test get_filter_stats with metadata."""
        X = np.random.randn(5, 5)
        metadata = {"quality": np.array(["good", "bad", "good", "bad", "good"])}

        filter_obj = MetadataFilter(column="quality", values_to_exclude=["bad"])
        stats = filter_obj.get_filter_stats(X, metadata=metadata)

        assert stats["n_samples"] == 5
        assert stats["n_excluded"] == 2
        assert stats["column"] == "quality"
        assert stats["filtering_type"] == "values_to_exclude"
        assert "value_distribution" in stats

    def test_get_filter_stats_without_metadata(self):
        """Test get_filter_stats without metadata."""
        X = np.random.randn(5, 5)

        filter_obj = MetadataFilter(column="quality", values_to_exclude=["bad"])
        stats = filter_obj.get_filter_stats(X)

        assert stats["n_samples"] == 5
        assert stats["n_excluded"] == 0  # No filtering without metadata
        assert "note" in stats

    def test_repr_with_values_to_exclude(self):
        """Test repr with values_to_exclude."""
        filter_obj = MetadataFilter(
            column="quality",
            values_to_exclude=["bad", "corrupted"]
        )
        repr_str = repr(filter_obj)

        assert "MetadataFilter" in repr_str
        assert "quality" in repr_str
        assert "values_to_exclude" in repr_str

    def test_repr_with_many_values(self):
        """Test repr with many values (truncated)."""
        filter_obj = MetadataFilter(
            column="category",
            values_to_exclude=["a", "b", "c", "d", "e"]
        )
        repr_str = repr(filter_obj)

        assert "5 values" in repr_str

    def test_repr_with_condition(self):
        """Test repr with condition function."""
        filter_obj = MetadataFilter(
            column="value",
            condition=lambda x: x > 0
        )
        repr_str = repr(filter_obj)

        assert "<function>" in repr_str


class TestMetadataFilterTransform:
    """Tests for transform method (should be no-op)."""

    def test_transform_returns_input_unchanged(self):
        """Test that transform returns input unchanged."""
        X = np.random.randn(10, 5)

        filter_obj = MetadataFilter(column="x", values_to_exclude=["a"])
        X_transformed = filter_obj.transform(X)

        np.testing.assert_array_equal(X, X_transformed)
