"""Unit tests for branch value mapping utilities.

Tests the value condition parsing and sample grouping functionality
used by separation branches.
"""

import pytest

from nirs4all.controllers.data.branch_utils import (
    parse_value_condition,
    group_samples_by_value_mapping,
    validate_disjoint_conditions,
)


class TestParseValueCondition:
    """Tests for parse_value_condition function."""

    # =========================================================================
    # Boolean conditions
    # =========================================================================

    def test_boolean_true(self):
        """Test boolean True condition."""
        pred = parse_value_condition(True)
        assert pred(True) is True
        assert pred(False) is False
        # Note: In Python, 1 == True is True due to bool being a subclass of int

    def test_boolean_false(self):
        """Test boolean False condition."""
        pred = parse_value_condition(False)
        assert pred(False) is True
        assert pred(True) is False

    # =========================================================================
    # Numeric conditions
    # =========================================================================

    def test_integer_equality(self):
        """Test integer equality condition."""
        pred = parse_value_condition(5)
        assert pred(5) is True
        assert pred(4) is False
        assert pred(5.0) is True  # Python equality

    def test_float_equality(self):
        """Test float equality condition."""
        pred = parse_value_condition(3.14)
        assert pred(3.14) is True
        assert pred(3.0) is False

    # =========================================================================
    # List/tuple/set membership conditions
    # =========================================================================

    def test_list_membership(self):
        """Test list membership condition."""
        pred = parse_value_condition([1, 2, 3])
        assert pred(1) is True
        assert pred(2) is True
        assert pred(3) is True
        assert pred(4) is False
        assert pred(0) is False

    def test_tuple_membership(self):
        """Test tuple membership condition."""
        pred = parse_value_condition(("a", "b", "c"))
        assert pred("a") is True
        assert pred("d") is False

    def test_set_membership(self):
        """Test set membership condition."""
        pred = parse_value_condition({"x", "y", "z"})
        assert pred("x") is True
        assert pred("w") is False

    def test_empty_list(self):
        """Test empty list returns False for all values."""
        pred = parse_value_condition([])
        assert pred(1) is False
        assert pred("a") is False
        assert pred(None) is False

    # =========================================================================
    # Comparison string conditions
    # =========================================================================

    def test_greater_than(self):
        """Test greater than comparison."""
        pred = parse_value_condition("> 0.5")
        assert pred(0.6) is True
        assert pred(0.5) is False
        assert pred(0.4) is False

    def test_greater_than_equal(self):
        """Test greater than or equal comparison."""
        pred = parse_value_condition(">= 10")
        assert pred(11) is True
        assert pred(10) is True
        assert pred(9) is False

    def test_less_than(self):
        """Test less than comparison."""
        pred = parse_value_condition("< 100")
        assert pred(99) is True
        assert pred(100) is False
        assert pred(101) is False

    def test_less_than_equal(self):
        """Test less than or equal comparison."""
        pred = parse_value_condition("<= 50")
        assert pred(49) is True
        assert pred(50) is True
        assert pred(51) is False

    def test_equal_comparison(self):
        """Test equality comparison string."""
        pred = parse_value_condition("== 42")
        assert pred(42) is True
        assert pred(41) is False

    def test_not_equal_comparison(self):
        """Test not equal comparison string."""
        pred = parse_value_condition("!= 0")
        assert pred(1) is True
        assert pred(-1) is True
        assert pred(0) is False

    def test_comparison_with_spaces(self):
        """Test comparison with various spacing."""
        pred1 = parse_value_condition(">0.8")
        pred2 = parse_value_condition("> 0.8")
        pred3 = parse_value_condition(">  0.8")

        assert pred1(0.9) is True
        assert pred2(0.9) is True
        assert pred3(0.9) is True

    def test_comparison_negative_values(self):
        """Test comparison with negative values."""
        pred = parse_value_condition("> -5")
        assert pred(0) is True
        assert pred(-4) is True
        assert pred(-5) is False
        assert pred(-6) is False

    def test_comparison_float_values(self):
        """Test comparison with float values."""
        pred = parse_value_condition("<= 3.14159")
        assert pred(3.14159) is True
        assert pred(3.14) is True
        assert pred(3.15) is False

    # =========================================================================
    # Range string conditions
    # =========================================================================

    def test_closed_range(self):
        """Test closed range (both ends specified)."""
        pred = parse_value_condition("0..100")
        assert pred(0) is True
        assert pred(50) is True
        assert pred(100) is True
        assert pred(-1) is False
        assert pred(101) is False

    def test_open_end_range(self):
        """Test open-ended range (start only)."""
        pred = parse_value_condition("50..")
        assert pred(50) is True
        assert pred(100) is True
        assert pred(49) is False

    def test_open_start_range(self):
        """Test open-start range (end only)."""
        pred = parse_value_condition("..50")
        assert pred(50) is True
        assert pred(0) is True
        assert pred(-10) is True
        assert pred(51) is False

    def test_range_with_floats(self):
        """Test range with float values."""
        pred = parse_value_condition("0.0..1.0")
        assert pred(0.0) is True
        assert pred(0.5) is True
        assert pred(1.0) is True
        assert pred(-0.1) is False
        assert pred(1.1) is False

    def test_range_with_spaces(self):
        """Test range with spaces around .."""
        pred = parse_value_condition("10 .. 20")
        assert pred(15) is True
        assert pred(9) is False

    def test_range_negative_values(self):
        """Test range with negative values."""
        pred = parse_value_condition("-10..10")
        assert pred(-10) is True
        assert pred(0) is True
        assert pred(10) is True
        assert pred(-11) is False

    def test_invalid_range_both_empty(self):
        """Test that '..' alone raises error."""
        with pytest.raises(ValueError, match="at least a start or end"):
            parse_value_condition("..")

    # =========================================================================
    # String literal conditions
    # =========================================================================

    def test_string_literal_equality(self):
        """Test string literal as equality check."""
        pred = parse_value_condition("category_a")
        assert pred("category_a") is True
        assert pred("category_b") is False

    def test_numeric_string(self):
        """Test numeric string is parsed as number."""
        pred = parse_value_condition("42")
        assert pred(42) is True
        assert pred("42") is False  # Not string match

    # =========================================================================
    # Callable conditions
    # =========================================================================

    def test_callable_passthrough(self):
        """Test that callable is used directly."""
        custom_fn = lambda x: x % 2 == 0  # Even numbers
        pred = parse_value_condition(custom_fn)
        assert pred(2) is True
        assert pred(4) is True
        assert pred(1) is False
        assert pred(3) is False

    def test_callable_with_closure(self):
        """Test callable with closure."""
        threshold = 0.5

        def above_threshold(x):
            return x > threshold

        pred = parse_value_condition(above_threshold)
        assert pred(0.6) is True
        assert pred(0.4) is False

    # =========================================================================
    # Edge cases and error handling
    # =========================================================================

    def test_none_condition_raises(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="Unknown condition format"):
            parse_value_condition(None)

    def test_dict_condition_raises(self):
        """Test that dict raises ValueError (not a valid condition type)."""
        with pytest.raises(ValueError, match="Unknown condition format"):
            parse_value_condition({"key": "value"})


class TestGroupSamplesByValueMapping:
    """Tests for group_samples_by_value_mapping function."""

    def test_simple_two_group_split(self):
        """Test simple two-way split by value."""
        values = [True, False, True, True, False]
        mapping = {"outliers": True, "inliers": False}

        groups = group_samples_by_value_mapping(values, mapping)

        assert set(groups["outliers"]) == {0, 2, 3}
        assert set(groups["inliers"]) == {1, 4}

    def test_numeric_grouping_by_range(self):
        """Test grouping by numeric ranges."""
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        mapping = {
            "low": "< 0.5",
            "high": ">= 0.5"
        }

        groups = group_samples_by_value_mapping(values, mapping)

        assert groups["low"] == [0, 1]
        assert groups["high"] == [2, 3, 4]

    def test_three_way_split(self):
        """Test three-way split."""
        values = [1, 2, 3, 1, 2, 3]
        mapping = {
            "ones": 1,
            "twos": 2,
            "threes": 3,
        }

        groups = group_samples_by_value_mapping(values, mapping)

        assert groups["ones"] == [0, 3]
        assert groups["twos"] == [1, 4]
        assert groups["threes"] == [2, 5]

    def test_list_membership_grouping(self):
        """Test grouping by list membership."""
        values = ["A", "B", "C", "D", "E"]
        mapping = {
            "first_half": ["A", "B", "C"],
            "second_half": ["D", "E"],
        }

        groups = group_samples_by_value_mapping(values, mapping)

        assert groups["first_half"] == [0, 1, 2]
        assert groups["second_half"] == [3, 4]

    def test_overlapping_conditions_raises(self):
        """Test that overlapping conditions raise error."""
        values = [0.5, 0.6, 0.7]
        mapping = {
            "low": "<= 0.6",  # Includes 0.5, 0.6
            "mid": ">= 0.5",  # Includes 0.5, 0.6, 0.7 - overlaps!
        }

        with pytest.raises(ValueError, match="matches multiple branches"):
            group_samples_by_value_mapping(values, mapping)

    def test_unassigned_samples(self):
        """Test samples not matching any condition."""
        values = [1, 2, 3, 4, 5]
        mapping = {
            "low": [1, 2],
            "high": [4, 5],
            # 3 is not in any group
        }

        groups = group_samples_by_value_mapping(values, mapping)

        assert groups["low"] == [0, 1]
        assert groups["high"] == [3, 4]
        # Sample at index 2 (value=3) is not in any group

    def test_empty_values_list(self):
        """Test with empty values list."""
        groups = group_samples_by_value_mapping([], {"a": True, "b": False})
        assert groups == {"a": [], "b": []}

    def test_callable_in_mapping(self):
        """Test using callable in value mapping."""
        values = [1, 2, 3, 4, 5, 6]
        mapping = {
            "even": lambda x: x % 2 == 0,
            "odd": lambda x: x % 2 == 1,
        }

        groups = group_samples_by_value_mapping(values, mapping)

        assert groups["even"] == [1, 3, 5]  # indices of 2, 4, 6
        assert groups["odd"] == [0, 2, 4]   # indices of 1, 3, 5


class TestValidateDisjointConditions:
    """Tests for validate_disjoint_conditions function."""

    def test_disjoint_conditions_valid(self):
        """Test that disjoint conditions pass validation."""
        mapping = {
            "low": "< 0.5",
            "high": ">= 0.5",
        }
        sample_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        result = validate_disjoint_conditions(mapping, sample_values)
        assert result is True

    def test_overlapping_conditions_invalid(self):
        """Test that overlapping conditions fail validation."""
        mapping = {
            "low": "<= 0.5",
            "mid": ">= 0.3",  # Overlaps with low in 0.3-0.5 range
        }
        sample_values = [0.1, 0.4, 0.7]

        result = validate_disjoint_conditions(mapping, sample_values)
        assert result is False

    def test_without_sample_values_returns_true(self):
        """Test that without samples, validation assumes disjoint."""
        mapping = {
            "a": "< 0",
            "b": ">= 0",
        }

        result = validate_disjoint_conditions(mapping)
        assert result is True
