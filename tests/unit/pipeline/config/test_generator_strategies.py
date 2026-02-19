"""Tests for generator strategy pattern (Phase 2).

This module tests the modular architecture introduced in Phase 2:
- Strategy pattern implementation
- RangeStrategy
- OrStrategy
- Core module with strategy dispatch
"""

from math import comb, factorial

import pytest

from nirs4all.pipeline.config.generator import (
    # Strategy exports
    ExpansionStrategy,
    OrStrategy,
    RangeStrategy,
    count_combinations,
    expand_spec,
    get_strategy,
)


class TestStrategyPattern:
    """Tests for strategy pattern architecture."""

    def test_get_strategy_returns_range_for_range_node(self):
        """Range nodes should be handled by RangeStrategy."""
        node = {"_range_": [1, 5]}
        strategy = get_strategy(node)
        assert strategy is not None
        assert isinstance(strategy, RangeStrategy)

    def test_get_strategy_returns_or_for_or_node(self):
        """OR nodes should be handled by OrStrategy."""
        node = {"_or_": ["A", "B", "C"]}
        strategy = get_strategy(node)
        assert strategy is not None
        assert isinstance(strategy, OrStrategy)

    def test_get_strategy_returns_none_for_regular_dict(self):
        """Regular dicts should not have a strategy."""
        node = {"class": "MyClass", "params": {"n": 5}}
        strategy = get_strategy(node)
        assert strategy is None

    def test_get_strategy_returns_none_for_non_dict(self):
        """Non-dict values should not have a strategy."""
        assert get_strategy("string") is None
        assert get_strategy(123) is None
        assert get_strategy(["list"]) is None

    def test_range_strategy_priority_higher_than_or(self):
        """RangeStrategy should have higher priority than OrStrategy."""
        assert RangeStrategy.priority > OrStrategy.priority

    def test_strategy_instances_are_cached(self):
        """Same node type should return same strategy instance."""
        node1 = {"_range_": [1, 5]}
        node2 = {"_range_": [10, 20]}
        strategy1 = get_strategy(node1)
        strategy2 = get_strategy(node2)
        assert strategy1 is strategy2  # Same instance

class TestRangeStrategy:
    """Tests for RangeStrategy."""

    def test_range_strategy_handles(self):
        """RangeStrategy should handle pure range nodes."""
        assert RangeStrategy.handles({"_range_": [1, 5]})
        assert RangeStrategy.handles({"_range_": [1, 10], "count": 3})
        # Should not handle mixed nodes
        assert not RangeStrategy.handles({"_range_": [1, 5], "extra_key": "value"})

    def test_range_expand_basic(self):
        """Basic range expansion."""
        strategy = RangeStrategy()
        result = strategy.expand({"_range_": [1, 5]})
        assert result == [1, 2, 3, 4, 5]

    def test_range_expand_with_step(self):
        """Range expansion with step."""
        strategy = RangeStrategy()
        result = strategy.expand({"_range_": [0, 10, 2]})
        assert result == [0, 2, 4, 6, 8, 10]

    def test_range_expand_dict_syntax(self):
        """Range expansion with dict syntax."""
        strategy = RangeStrategy()
        result = strategy.expand({"_range_": {"from": 1, "to": 5, "step": 1}})
        assert result == [1, 2, 3, 4, 5]

    def test_range_expand_with_count(self):
        """Range expansion with count limit."""
        strategy = RangeStrategy()
        result = strategy.expand({"_range_": [1, 100], "count": 5}, seed=42)
        assert len(result) == 5
        # All values should be in original range
        assert all(1 <= v <= 100 for v in result)

    def test_range_count(self):
        """Range counting."""
        strategy = RangeStrategy()
        assert strategy.count({"_range_": [1, 10]}) == 10
        assert strategy.count({"_range_": [0, 10, 2]}) == 6
        assert strategy.count({"_range_": [1, 100], "count": 5}) == 5

    def test_range_validate_valid(self):
        """Validation should pass for valid range specs."""
        strategy = RangeStrategy()
        assert strategy.validate({"_range_": [1, 5]}) == []
        assert strategy.validate({"_range_": [1, 10, 2]}) == []
        assert strategy.validate({"_range_": {"from": 1, "to": 10}}) == []

    def test_range_validate_invalid(self):
        """Validation should catch invalid range specs."""
        strategy = RangeStrategy()
        errors = strategy.validate({"_range_": [1, 2, 3, 4]})  # Too many elements
        assert len(errors) > 0

class TestOrStrategy:
    """Tests for OrStrategy."""

    def test_or_strategy_handles(self):
        """OrStrategy should handle pure OR nodes."""
        assert OrStrategy.handles({"_or_": ["A", "B"]})
        assert OrStrategy.handles({"_or_": ["A", "B"], "pick": 2})
        assert OrStrategy.handles({"_or_": ["A", "B"], "arrange": 2})
        assert OrStrategy.handles({"_or_": ["A", "B"], "pick": 1})
        assert OrStrategy.handles({"_or_": ["A", "B"], "count": 1})
        # Should not handle mixed nodes
        assert not OrStrategy.handles({"_or_": ["A", "B"], "class": "MyClass"})

    def test_or_expand_basic(self):
        """Basic OR expansion."""
        strategy = OrStrategy()
        result = strategy.expand({"_or_": ["A", "B", "C"]})
        assert result == ["A", "B", "C"]

    def test_or_expand_with_pick(self):
        """OR expansion with pick (combinations)."""
        strategy = OrStrategy()
        result = strategy.expand({"_or_": ["A", "B", "C"], "pick": 2})
        assert len(result) == 3  # C(3,2)
        # Each result should be a 2-element list
        for r in result:
            assert len(r) == 2

    def test_or_expand_with_arrange(self):
        """OR expansion with arrange (permutations)."""
        strategy = OrStrategy()
        result = strategy.expand({"_or_": ["A", "B", "C"], "arrange": 2})
        assert len(result) == 6  # P(3,2)
        # Order should matter
        assert ["A", "B"] in result
        assert ["B", "A"] in result

    def test_or_expand_with_count(self):
        """OR expansion with count limit."""
        strategy = OrStrategy()
        result = strategy.expand(
            {"_or_": ["A", "B", "C", "D", "E"], "count": 2},
            seed=42
        )
        assert len(result) == 2

    def test_or_count_basic(self):
        """OR counting."""
        strategy = OrStrategy()
        assert strategy.count({"_or_": ["A", "B", "C"]}) == 3
        assert strategy.count({"_or_": ["A", "B", "C"], "pick": 2}) == 3  # C(3,2)
        assert strategy.count({"_or_": ["A", "B", "C"], "arrange": 2}) == 6  # P(3,2)

    def test_or_validate_valid(self):
        """Validation should pass for valid OR specs."""
        strategy = OrStrategy()
        assert strategy.validate({"_or_": ["A", "B"]}) == []
        assert strategy.validate({"_or_": ["A", "B"], "pick": 2}) == []
        assert strategy.validate({"_or_": ["A", "B"], "count": 1}) == []

    def test_or_validate_invalid(self):
        """Validation should catch invalid OR specs."""
        strategy = OrStrategy()
        errors = strategy.validate({"_or_": "not a list"})
        assert len(errors) > 0

class TestCoreIntegration:
    """Tests for core module integration with strategies."""

    def test_expand_uses_range_strategy(self):
        """expand_spec should use RangeStrategy for range nodes."""
        result = expand_spec({"_range_": [1, 5]})
        assert result == [1, 2, 3, 4, 5]

    def test_expand_uses_or_strategy(self):
        """expand_spec should use OrStrategy for OR nodes."""
        result = expand_spec({"_or_": ["A", "B", "C"]})
        assert result == ["A", "B", "C"]

    def test_expand_handles_nested_generators(self):
        """expand_spec should handle nested generator nodes."""
        result = expand_spec({
            "x": {"_or_": [1, 2]},
            "y": {"_range_": [10, 11]}
        })
        # 2 x 2 = 4 combinations
        assert len(result) == 4
        expected_set = {
            (1, 10), (1, 11), (2, 10), (2, 11)
        }
        result_set = {(r["x"], r["y"]) for r in result}
        assert result_set == expected_set

    def test_expand_handles_list_with_generators(self):
        """expand_spec should handle lists containing generators."""
        result = expand_spec([
            {"_or_": ["A", "B"]},
            "fixed"
        ])
        assert len(result) == 2
        assert ["A", "fixed"] in result
        assert ["B", "fixed"] in result

    def test_count_uses_strategies(self):
        """count_combinations should use strategies."""
        assert count_combinations({"_range_": [1, 100]}) == 100
        assert count_combinations({"_or_": ["A", "B", "C"]}) == 3
        assert count_combinations({"_or_": ["A", "B", "C"], "pick": 2}) == 3

    def test_count_handles_nested_generators(self):
        """count_combinations should handle nested nodes."""
        count = count_combinations({
            "x": {"_or_": [1, 2]},
            "y": {"_range_": [10, 11]}
        })
        assert count == 4  # 2 x 2

class TestSecondOrderOperations:
    """Tests for second-order (then_pick/then_arrange) operations."""

    def test_pick_then_pick(self):
        """pick with then_pick should work."""
        result = expand_spec({
            "_or_": ["A", "B", "C"],
            "pick": 1,
            "then_pick": 2
        })
        # First pick 1: 3 items (A, B, C)
        # Then pick 2 from 3: C(3,2) = 3
        assert len(result) == 3

    def test_pick_then_arrange(self):
        """pick with then_arrange should work."""
        result = expand_spec({
            "_or_": ["A", "B", "C"],
            "pick": 1,
            "then_arrange": 2
        })
        # First pick 1: 3 items (A, B, C)
        # Then arrange 2 from 3: P(3,2) = 6
        assert len(result) == 6

    def test_arrange_then_pick(self):
        """arrange with then_pick should work."""
        result = expand_spec({
            "_or_": ["A", "B"],
            "arrange": 1,
            "then_pick": 2
        })
        # First arrange 1: 2 items
        # Then pick 2 from 2: C(2,2) = 1
        assert len(result) == 1

    def test_arrange_then_arrange(self):
        """arrange with then_arrange should work."""
        result = expand_spec({
            "_or_": ["A", "B"],
            "arrange": 1,
            "then_arrange": 2
        })
        # First arrange 1: 2 items
        # Then arrange 2 from 2: P(2,2) = 2
        assert len(result) == 2

class TestFloatRanges:
    """Tests for float range support in RangeStrategy."""

    def test_float_range_basic(self):
        """Float range should work."""
        result = expand_spec({"_range_": [0.1, 0.5, 0.1]})
        assert len(result) == 5
        # Check approximate values (float precision)
        expected = [0.1, 0.2, 0.3, 0.4, 0.5]
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-9

    def test_float_range_count(self):
        """Float range count should work."""
        count = count_combinations({"_range_": [0.1, 1.0, 0.1]})
        assert count == 10

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
