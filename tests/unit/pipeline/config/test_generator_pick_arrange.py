"""Tests for pick and arrange keywords in the generator module.

This module tests the Phase 1.5 implementation of explicit selection semantics:
- pick: Unordered selection (combinations)
- arrange: Ordered arrangement (permutations)
"""

import pytest
from math import comb, factorial

from nirs4all.pipeline.config.generator import (
    expand_spec, count_combinations,
    PICK_KEYWORD, ARRANGE_KEYWORD
)


class TestPickKeyword:
    """Tests for the 'pick' keyword (combinations-based selection)."""

    def test_pick_basic(self):
        """Basic pick: select exactly 2 items, order doesn't matter."""
        result = expand_spec({"_or_": ["A", "B", "C"], "pick": 2})
        # C(3,2) = 3 combinations
        assert len(result) == 3
        # Each result should be a list of 2 elements
        for r in result:
            assert len(r) == 2
        # Check the combinations (order doesn't matter within)
        combinations_set = {tuple(sorted(r)) for r in result}
        expected = {("A", "B"), ("A", "C"), ("B", "C")}
        assert combinations_set == expected

    def test_pick_single(self):
        """Pick 1 item should return each item individually."""
        result = expand_spec({"_or_": ["A", "B", "C"], "pick": 1})
        assert len(result) == 3
        # Each should be a single-element list
        assert [["A"], ["B"], ["C"]] == result or all(len(r) == 1 for r in result)

    def test_pick_all(self):
        """Pick all items should return one combination."""
        result = expand_spec({"_or_": ["A", "B", "C"], "pick": 3})
        # C(3,3) = 1
        assert len(result) == 1
        assert sorted(result[0]) == ["A", "B", "C"]

    def test_pick_range(self):
        """Pick with range: select 1 to 2 items."""
        result = expand_spec({"_or_": ["A", "B", "C"], "pick": (1, 2)})
        # C(3,1) + C(3,2) = 3 + 3 = 6
        assert len(result) == 6

    def test_pick_with_count(self):
        """Pick with count limit."""
        result = expand_spec({"_or_": ["A", "B", "C", "D"], "pick": 2, "count": 3})
        # C(4,2) = 6, but limited to 3
        assert len(result) == 3

    def test_pick_empty(self):
        """Pick 0 items should return empty list."""
        result = expand_spec({"_or_": ["A", "B", "C"], "pick": 0})
        assert len(result) == 1
        assert result[0] == []

    def test_pick_exceeds_choices(self):
        """Pick more than available choices should return nothing."""
        result = expand_spec({"_or_": ["A", "B"], "pick": 5})
        assert len(result) == 0


class TestArrangeKeyword:
    """Tests for the 'arrange' keyword (permutations-based selection)."""

    def test_arrange_basic(self):
        """Basic arrange: select exactly 2 items, order matters."""
        result = expand_spec({"_or_": ["A", "B", "C"], "arrange": 2})
        # P(3,2) = 6 permutations
        assert len(result) == 6
        # Each result should be a list of 2 elements
        for r in result:
            assert len(r) == 2
        # Check that order matters
        assert ["A", "B"] in result
        assert ["B", "A"] in result
        assert ["A", "C"] in result
        assert ["C", "A"] in result
        assert ["B", "C"] in result
        assert ["C", "B"] in result

    def test_arrange_single(self):
        """Arrange 1 item should return each item individually."""
        result = expand_spec({"_or_": ["A", "B", "C"], "arrange": 1})
        # P(3,1) = 3
        assert len(result) == 3

    def test_arrange_all(self):
        """Arrange all items should return all permutations."""
        result = expand_spec({"_or_": ["A", "B", "C"], "arrange": 3})
        # P(3,3) = 3! = 6
        assert len(result) == 6

    def test_arrange_range(self):
        """Arrange with range: arrange 1 to 2 items."""
        result = expand_spec({"_or_": ["A", "B", "C"], "arrange": (1, 2)})
        # P(3,1) + P(3,2) = 3 + 6 = 9
        assert len(result) == 9

    def test_arrange_with_count(self):
        """Arrange with count limit."""
        result = expand_spec({"_or_": ["A", "B", "C", "D"], "arrange": 2, "count": 5})
        # P(4,2) = 12, but limited to 5
        assert len(result) == 5

    def test_arrange_empty(self):
        """Arrange 0 items should return empty list."""
        result = expand_spec({"_or_": ["A", "B", "C"], "arrange": 0})
        assert len(result) == 1
        assert result[0] == []

    def test_arrange_exceeds_choices(self):
        """Arrange more than available choices should return nothing."""
        result = expand_spec({"_or_": ["A", "B"], "arrange": 5})
        assert len(result) == 0


class TestPickVsArrangeComparison:
    """Tests comparing pick and arrange behavior."""

    def test_pick_vs_arrange_count_difference(self):
        """Pick and arrange should give different counts for same spec."""
        choices = ["A", "B", "C"]
        k = 2

        pick_result = expand_spec({"_or_": choices, "pick": k})
        arrange_result = expand_spec({"_or_": choices, "arrange": k})

        # C(3,2) = 3, P(3,2) = 6
        assert len(pick_result) == 3
        assert len(arrange_result) == 6

    def test_pick_vs_arrange_order_difference(self):
        """Pick should not have duplicates with order reversed, arrange should."""
        choices = ["A", "B", "C"]
        k = 2

        pick_result = expand_spec({"_or_": choices, "pick": k})
        arrange_result = expand_spec({"_or_": choices, "arrange": k})

        # In pick results, ["A", "B"] and ["B", "A"] should not both appear
        pick_tuples = [tuple(r) for r in pick_result]
        for t in pick_tuples:
            reversed_t = tuple(reversed(t))
            if reversed_t != t:
                assert reversed_t not in pick_tuples

        # In arrange results, both should appear
        assert ["A", "B"] in arrange_result
        assert ["B", "A"] in arrange_result


class TestCountCombinationsPickArrange:
    """Tests for count_combinations with pick and arrange."""

    def test_count_pick_basic(self):
        """Count with pick keyword."""
        count = count_combinations({"_or_": ["A", "B", "C"], "pick": 2})
        assert count == 3  # C(3,2)

    def test_count_arrange_basic(self):
        """Count with arrange keyword."""
        count = count_combinations({"_or_": ["A", "B", "C"], "arrange": 2})
        assert count == 6  # P(3,2)

    def test_count_pick_range(self):
        """Count with pick range."""
        count = count_combinations({"_or_": ["A", "B", "C", "D"], "pick": (1, 3)})
        # C(4,1) + C(4,2) + C(4,3) = 4 + 6 + 4 = 14
        assert count == 14

    def test_count_arrange_range(self):
        """Count with arrange range."""
        count = count_combinations({"_or_": ["A", "B", "C"], "arrange": (1, 2)})
        # P(3,1) + P(3,2) = 3 + 6 = 9
        assert count == 9

    def test_count_with_limit(self):
        """Count with count limit."""
        count_pick = count_combinations({"_or_": ["A", "B", "C", "D"], "pick": 2, "count": 3})
        count_arrange = count_combinations({"_or_": ["A", "B", "C", "D"], "arrange": 2, "count": 5})
        assert count_pick == 3  # min(C(4,2)=6, 3)
        assert count_arrange == 5  # min(P(4,2)=12, 5)


class TestNestedOrWithPickArrange:
    """Tests for nested dict nodes with pick/arrange."""

    def test_nested_dict_with_pick(self):
        """Pick in a nested dict structure."""
        spec = {
            "model": {"_or_": ["PLS", "SVM", "RF"], "pick": 2},
            "scaler": "Standard"
        }
        result = expand_spec(spec)
        # C(3,2) = 3 combinations
        assert len(result) == 3
        for r in result:
            assert r["scaler"] == "Standard"
            assert len(r["model"]) == 2

    def test_nested_dict_with_arrange(self):
        """Arrange in a nested dict structure."""
        spec = {
            "preprocessing": {"_or_": ["SNV", "MSC", "SG"], "arrange": 2},
            "model": "PLS"
        }
        result = expand_spec(spec)
        # P(3,2) = 6 permutations
        assert len(result) == 6


class TestSecondOrderWithPickArrange:
    """Tests for second-order (nested) pick/arrange syntax.

    Note: The [outer, inner] syntax like pick: [2, 2] is ambiguous with
    serialized tuples (pick: (1, 2) becomes pick: [1, 2] in YAML).

    For second-order operations, use explicit then_pick/then_arrange syntax:
    - pick: 2, then_pick: 2 (equivalent to old [2, 2])

    See also: _handle_nested_combinations and related methods in or_strategy.py
    """

    def test_pick_then_pick_syntax(self):
        """Pick with then_pick for second-order selection."""
        # This uses explicit then_pick for second-order operations
        # Primary: C(3,2) = 3 combinations (AB, AC, BC)
        # Then: C(3,2) = 3 combinations of those pairs
        result = expand_spec({"_or_": ["A", "B", "C"], "pick": 2, "then_pick": 2})
        assert len(result) == 3
        # Each result is a pair of 2-combinations
        for r in result:
            assert len(r) == 2  # Each outer result has 2 elements
            for inner in r:
                assert len(inner) == 2  # Each inner element is a 2-combination

    def test_arrange_then_arrange_syntax(self):
        """Arrange with then_arrange for second-order selection."""
        # This uses explicit then_arrange for second-order operations
        # Inner: P(3,2) = 6 arrangements
        # Outer: P(6,2) = 30 arrangements of those
        result = expand_spec({"_or_": ["A", "B", "C"], "arrange": 2, "then_arrange": 2})
        assert len(result) == 30

    def test_pick_range_after_serialization(self):
        """Verify that [1, 2] (serialized from tuple) is treated as range, not nested."""
        # This is crucial: after YAML serialization, (1, 2) becomes [1, 2]
        # The generator should treat [int, int] as a range specification
        result = expand_spec({"_or_": ["A", "B", "C"], "pick": [1, 2]})
        # Expected: C(3,1) + C(3,2) = 3 + 3 = 6 results
        assert len(result) == 6

        # Verify we get singles and pairs, not nested arrangements
        singles = [r for r in result if len(r) == 1]
        pairs = [r for r in result if len(r) == 2]
        assert len(singles) == 3  # A, B, C
        assert len(pairs) == 3    # AB, AC, BC


class TestKeywordConstants:
    """Tests for keyword constant accessibility."""

    def test_pick_keyword_constant(self):
        """PICK_KEYWORD should be 'pick'."""
        assert PICK_KEYWORD == "pick"

    def test_arrange_keyword_constant(self):
        """ARRANGE_KEYWORD should be 'arrange'."""
        assert ARRANGE_KEYWORD == "arrange"


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_choices_pick(self):
        """Pick from empty choices."""
        result = expand_spec({"_or_": [], "pick": 1})
        assert len(result) == 0

    def test_empty_choices_arrange(self):
        """Arrange from empty choices."""
        result = expand_spec({"_or_": [], "arrange": 1})
        assert len(result) == 0

    def test_single_choice_pick(self):
        """Pick from single choice."""
        result = expand_spec({"_or_": ["A"], "pick": 1})
        assert len(result) == 1
        assert result[0] == ["A"]

    def test_single_choice_arrange(self):
        """Arrange from single choice."""
        result = expand_spec({"_or_": ["A"], "arrange": 1})
        assert len(result) == 1


# Integration tests with complex specs
class TestComplexSpecs:
    """Tests for complex specification structures."""

    def test_mixed_pick_and_values(self):
        """Mixed pick with other value expansions."""
        spec = {
            "components": {"_or_": [10, 20, 30], "pick": 2},
            "method": {"_or_": ["pca", "svd"]}
        }
        result = expand_spec(spec)
        # C(3,2) * 2 = 3 * 2 = 6
        assert len(result) == 6

    def test_list_with_pick_elements(self):
        """List containing pick elements."""
        spec = [
            {"_or_": ["A", "B", "C"], "pick": 2},
            "fixed"
        ]
        result = expand_spec(spec)
        # Each result should have 2 elements
        assert len(result) == 3
        for r in result:
            assert r[1] == "fixed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
