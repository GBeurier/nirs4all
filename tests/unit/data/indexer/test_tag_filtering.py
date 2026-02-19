"""
Unit tests for QueryBuilder tag filtering support.

Tests cover:
- Boolean condition filtering
- Numeric comparison filtering (>, <, >=, <=, ==, !=)
- Range filtering (start..end, start.., ..end)
- List membership filtering
- Lambda/callable filtering
- Null value filtering
- String exact match filtering
- Edge cases and error handling
"""
import polars as pl
import pytest

from nirs4all.data._indexer.query_builder import QueryBuilder


class TestBooleanConditionFiltering:
    """Tests for boolean tag condition filtering."""

    def test_filter_boolean_true(self):
        """Test filtering for True values."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3],
            "is_outlier": [True, False, True, None],
        })

        expr = builder.build_tag_filter("is_outlier", True)
        result = df.filter(expr)["sample"].to_list()

        assert result == [0, 2]

    def test_filter_boolean_false(self):
        """Test filtering for False values."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3],
            "is_outlier": [True, False, True, None],
        })

        expr = builder.build_tag_filter("is_outlier", False)
        result = df.filter(expr)["sample"].to_list()

        assert result == [1]

class TestNumericComparisonFiltering:
    """Tests for numeric comparison filtering."""

    def test_filter_greater_than(self):
        """Test > comparison."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "score": [0.1, 0.5, 0.7, 0.9, 0.3],
        })

        expr = builder.build_tag_filter("score", "> 0.5")
        result = df.filter(expr)["sample"].to_list()

        assert result == [2, 3]

    def test_filter_greater_than_or_equal(self):
        """Test >= comparison."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "score": [0.1, 0.5, 0.7, 0.9, 0.3],
        })

        expr = builder.build_tag_filter("score", ">= 0.5")
        result = df.filter(expr)["sample"].to_list()

        assert result == [1, 2, 3]

    def test_filter_less_than(self):
        """Test < comparison."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "score": [0.1, 0.5, 0.7, 0.9, 0.3],
        })

        expr = builder.build_tag_filter("score", "< 0.5")
        result = df.filter(expr)["sample"].to_list()

        assert result == [0, 4]

    def test_filter_less_than_or_equal(self):
        """Test <= comparison."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "score": [0.1, 0.5, 0.7, 0.9, 0.3],
        })

        expr = builder.build_tag_filter("score", "<= 0.5")
        result = df.filter(expr)["sample"].to_list()

        assert result == [0, 1, 4]

    def test_filter_equal(self):
        """Test == comparison."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "cluster_id": [1, 2, 1, 3, 2],
        })

        expr = builder.build_tag_filter("cluster_id", "== 2")
        result = df.filter(expr)["sample"].to_list()

        assert result == [1, 4]

    def test_filter_not_equal(self):
        """Test != comparison."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "cluster_id": [1, 2, 1, 3, 2],
        })

        expr = builder.build_tag_filter("cluster_id", "!= 2")
        result = df.filter(expr)["sample"].to_list()

        assert result == [0, 2, 3]

    def test_filter_with_negative_numbers(self):
        """Test comparison with negative numbers."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "delta": [-5, -2, 0, 3, 7],
        })

        expr = builder.build_tag_filter("delta", "> -3")
        result = df.filter(expr)["sample"].to_list()

        assert result == [1, 2, 3, 4]

    def test_filter_numeric_exact_match(self):
        """Test numeric exact match (not string)."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "cluster_id": [1, 2, 1, 3, 2],
        })

        # Pass integer directly, not as string
        expr = builder.build_tag_filter("cluster_id", 2)
        result = df.filter(expr)["sample"].to_list()

        assert result == [1, 4]

class TestRangeFiltering:
    """Tests for range filtering (start..end syntax)."""

    def test_filter_closed_range(self):
        """Test closed range (inclusive both ends)."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "score": [0.1, 0.5, 0.7, 0.9, 0.3],
        })

        expr = builder.build_tag_filter("score", "0.3..0.7")
        result = df.filter(expr)["sample"].to_list()

        assert result == [1, 2, 4]

    def test_filter_open_end_range(self):
        """Test open-end range (>= start)."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "score": [0.1, 0.5, 0.7, 0.9, 0.3],
        })

        expr = builder.build_tag_filter("score", "0.7..")
        result = df.filter(expr)["sample"].to_list()

        assert result == [2, 3]

    def test_filter_open_start_range(self):
        """Test open-start range (<= end)."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "score": [0.1, 0.5, 0.7, 0.9, 0.3],
        })

        expr = builder.build_tag_filter("score", "..0.3")
        result = df.filter(expr)["sample"].to_list()

        assert result == [0, 4]

    def test_filter_integer_range(self):
        """Test range with integer values."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "cluster_id": [1, 2, 3, 4, 5],
        })

        expr = builder.build_tag_filter("cluster_id", "2..4")
        result = df.filter(expr)["sample"].to_list()

        assert result == [1, 2, 3]

    def test_filter_negative_range(self):
        """Test range with negative values."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "delta": [-10, -5, 0, 5, 10],
        })

        expr = builder.build_tag_filter("delta", "-5..5")
        result = df.filter(expr)["sample"].to_list()

        assert result == [1, 2, 3]

class TestListMembershipFiltering:
    """Tests for list membership filtering."""

    def test_filter_integer_list(self):
        """Test filtering with list of integers."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "cluster_id": [1, 2, 3, 4, 5],
        })

        expr = builder.build_tag_filter("cluster_id", [2, 4])
        result = df.filter(expr)["sample"].to_list()

        assert result == [1, 3]

    def test_filter_string_list(self):
        """Test filtering with list of strings."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "category": ["A", "B", "C", "A", "B"],
        })

        expr = builder.build_tag_filter("category", ["A", "C"])
        result = df.filter(expr)["sample"].to_list()

        assert result == [0, 2, 3]

    def test_filter_empty_list_matches_nothing(self):
        """Test that empty list matches nothing."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2],
            "cluster_id": [1, 2, 3],
        })

        expr = builder.build_tag_filter("cluster_id", [])
        result = df.filter(expr)["sample"].to_list()

        assert result == []

class TestCallableFiltering:
    """Tests for lambda/callable filtering."""

    def test_filter_with_lambda(self):
        """Test filtering with lambda function."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "score": [0.1, 0.5, 0.7, 0.9, 0.3],
        })

        expr = builder.build_tag_filter("score", lambda x: x > 0.5)
        result = df.filter(expr)["sample"].to_list()

        assert result == [2, 3]

    def test_filter_with_complex_lambda(self):
        """Test filtering with complex lambda."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "score": [0.1, 0.5, 0.7, 0.9, 0.3],
        })

        # Filter for values between 0.3 and 0.7 exclusive
        expr = builder.build_tag_filter("score", lambda x: 0.3 < x < 0.7)
        result = df.filter(expr)["sample"].to_list()

        assert result == [1]

class TestNullValueFiltering:
    """Tests for null value filtering."""

    def test_filter_null_values(self):
        """Test filtering for null values."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "tag": [1, None, 3, None, 5],
        })

        expr = builder.build_tag_filter("tag", None)
        result = df.filter(expr)["sample"].to_list()

        assert result == [1, 3]

class TestStringExactMatchFiltering:
    """Tests for string exact match filtering."""

    def test_filter_string_exact_match(self):
        """Test exact string match (fallback for non-pattern strings)."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "category": ["A", "B", "A", "C", "B"],
        })

        expr = builder.build_tag_filter("category", "A")
        result = df.filter(expr)["sample"].to_list()

        assert result == [0, 2]

    def test_filter_string_with_spaces(self):
        """Test filtering string with spaces."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2],
            "label": ["class A", "class B", "class A"],
        })

        expr = builder.build_tag_filter("label", "class A")
        result = df.filter(expr)["sample"].to_list()

        assert result == [0, 2]

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_condition_type_error(self):
        """Test that unknown condition type raises ValueError."""
        builder = QueryBuilder()

        # Passing a set instead of a list should raise an error
        with pytest.raises(ValueError, match="Unknown condition format"):
            builder.build_tag_filter("tag", {1, 2, 3})

    def test_comparison_with_whitespace(self):
        """Test comparison string with extra whitespace."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2],
            "score": [0.1, 0.5, 0.9],
        })

        # Whitespace should be trimmed
        expr = builder.build_tag_filter("score", "  > 0.5  ")
        result = df.filter(expr)["sample"].to_list()

        assert result == [2]

    def test_decimal_precision_in_comparison(self):
        """Test decimal precision in comparison."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2],
            "score": [0.123, 0.456, 0.789],
        })

        expr = builder.build_tag_filter("score", ">= 0.456")
        result = df.filter(expr)["sample"].to_list()

        assert result == [1, 2]

    def test_combined_with_other_filters(self):
        """Test that tag filter can be combined with other filters."""
        builder = QueryBuilder()
        df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "partition": ["train", "train", "test", "test", "train"],
            "score": [0.1, 0.9, 0.5, 0.8, 0.3],
        })

        # Build tag filter
        tag_expr = builder.build_tag_filter("score", "> 0.5")
        # Build partition filter
        partition_expr = pl.col("partition") == "train"

        # Combine with AND
        combined = tag_expr & partition_expr
        result = df.filter(combined)["sample"].to_list()

        assert result == [1]
