"""Tests for Phase 3 generator strategies.

This module tests the new strategies introduced in Phase 3:
- LogRangeStrategy: Logarithmic sequence generation
- GridStrategy: Grid search style expansion
- ZipStrategy: Parallel iteration
- ChainStrategy: Sequential ordered expansion
- SampleStrategy: Statistical sampling
"""

import math

import pytest

from nirs4all.pipeline.config.generator import (
    ChainStrategy,
    GridStrategy,
    # Phase 3 strategies
    LogRangeStrategy,
    SampleStrategy,
    ZipStrategy,
    count_combinations,
    expand_spec,
    get_strategy,
)


class TestLogRangeStrategy:
    """Tests for LogRangeStrategy."""

    def test_log_range_strategy_handles(self):
        """LogRangeStrategy should handle pure log range nodes."""
        assert LogRangeStrategy.handles({"_log_range_": [0.001, 1, 4]})
        assert LogRangeStrategy.handles({"_log_range_": [0.001, 1, 4], "count": 2})
        # Should not handle mixed nodes
        assert not LogRangeStrategy.handles({"_log_range_": [1, 10, 4], "extra_key": "value"})

    def test_log_range_expand_basic(self):
        """Basic logarithmic range expansion."""
        strategy = LogRangeStrategy()
        result = strategy.expand({"_log_range_": [0.001, 1, 4]})
        assert len(result) == 4
        # Should be approximately [0.001, 0.01, 0.1, 1.0]
        assert abs(result[0] - 0.001) < 1e-9
        assert abs(result[1] - 0.01) < 1e-9
        assert abs(result[2] - 0.1) < 1e-9
        assert abs(result[3] - 1.0) < 1e-9

    def test_log_range_expand_integers(self):
        """Log range with integer bounds."""
        strategy = LogRangeStrategy()
        result = strategy.expand({"_log_range_": [1, 1000, 4]})
        assert len(result) == 4
        # Should be [1, 10, 100, 1000]
        assert abs(result[0] - 1) < 1e-9
        assert abs(result[1] - 10) < 1e-9
        assert abs(result[2] - 100) < 1e-9
        assert abs(result[3] - 1000) < 1e-9

    def test_log_range_expand_dict_syntax(self):
        """Log range with dict syntax."""
        strategy = LogRangeStrategy()
        result = strategy.expand({
            "_log_range_": {"from": 0.001, "to": 1, "num": 4}
        })
        assert len(result) == 4
        assert abs(result[0] - 0.001) < 1e-9

    def test_log_range_expand_with_count(self):
        """Log range with count limit."""
        strategy = LogRangeStrategy()
        result = strategy.expand(
            {"_log_range_": [0.001, 1, 10], "count": 3},
            seed=42
        )
        assert len(result) == 3
        # Values should be within range
        assert all(0.001 <= v <= 1 for v in result)

    def test_log_range_count(self):
        """Log range counting."""
        strategy = LogRangeStrategy()
        assert strategy.count({"_log_range_": [0.001, 1, 10]}) == 10
        assert strategy.count({"_log_range_": [0.001, 1, 10], "count": 5}) == 5

    def test_log_range_validate_valid(self):
        """Validation should pass for valid log range specs."""
        strategy = LogRangeStrategy()
        assert strategy.validate({"_log_range_": [0.001, 1, 4]}) == []
        assert strategy.validate({"_log_range_": {"from": 0.001, "to": 1, "num": 4}}) == []

    def test_log_range_validate_invalid(self):
        """Validation should catch invalid log range specs."""
        strategy = LogRangeStrategy()
        # Negative values not allowed
        errors = strategy.validate({"_log_range_": [-1, 1, 4]})
        assert len(errors) > 0
        # Wrong array length
        errors = strategy.validate({"_log_range_": [0.001, 1]})
        assert len(errors) > 0

class TestGridStrategy:
    """Tests for GridStrategy."""

    def test_grid_strategy_handles(self):
        """GridStrategy should handle pure grid nodes."""
        assert GridStrategy.handles({"_grid_": {"x": [1, 2]}})
        assert GridStrategy.handles({"_grid_": {"x": [1, 2]}, "count": 2})
        # Should not handle mixed nodes
        assert not GridStrategy.handles({"_grid_": {"x": [1, 2]}, "extra": "value"})

    def test_grid_expand_basic(self):
        """Basic grid expansion (Cartesian product)."""
        strategy = GridStrategy()
        result = strategy.expand({"_grid_": {"x": [1, 2], "y": ["A", "B"]}})
        assert len(result) == 4  # 2 x 2
        # Should have all combinations
        expected = [
            {"x": 1, "y": "A"}, {"x": 1, "y": "B"},
            {"x": 2, "y": "A"}, {"x": 2, "y": "B"}
        ]
        for exp in expected:
            assert exp in result

    def test_grid_expand_single_param(self):
        """Grid with single parameter."""
        strategy = GridStrategy()
        result = strategy.expand({"_grid_": {"x": [1, 2, 3]}})
        assert len(result) == 3
        assert result == [{"x": 1}, {"x": 2}, {"x": 3}]

    def test_grid_expand_three_params(self):
        """Grid with three parameters."""
        strategy = GridStrategy()
        result = strategy.expand({
            "_grid_": {"x": [1, 2], "y": ["A", "B"], "z": [True, False]}
        })
        assert len(result) == 8  # 2 x 2 x 2

    def test_grid_expand_with_count(self):
        """Grid expansion with count limit."""
        strategy = GridStrategy()
        result = strategy.expand(
            {"_grid_": {"x": [1, 2, 3], "y": ["A", "B", "C"]}, "count": 4},
            seed=42
        )
        assert len(result) == 4

    def test_grid_count(self):
        """Grid counting."""
        strategy = GridStrategy()
        assert strategy.count({"_grid_": {"x": [1, 2], "y": ["A", "B"]}}) == 4
        assert strategy.count({"_grid_": {"x": [1, 2, 3], "y": ["A", "B"]}}) == 6
        assert strategy.count({"_grid_": {"x": [1, 2], "y": ["A", "B"]}, "count": 3}) == 3

    def test_grid_empty(self):
        """Grid with empty dict."""
        strategy = GridStrategy()
        result = strategy.expand({"_grid_": {}})
        assert result == [{}]

    def test_grid_validate(self):
        """Grid validation."""
        strategy = GridStrategy()
        assert strategy.validate({"_grid_": {"x": [1, 2]}}) == []
        # Invalid: _grid_ must be dict
        errors = strategy.validate({"_grid_": [1, 2, 3]})
        assert len(errors) > 0

class TestZipStrategy:
    """Tests for ZipStrategy."""

    def test_zip_strategy_handles(self):
        """ZipStrategy should handle pure zip nodes."""
        assert ZipStrategy.handles({"_zip_": {"x": [1, 2]}})
        assert ZipStrategy.handles({"_zip_": {"x": [1, 2]}, "count": 1})
        # Should not handle mixed nodes
        assert not ZipStrategy.handles({"_zip_": {"x": [1, 2]}, "extra": "value"})

    def test_zip_expand_basic(self):
        """Basic zip expansion (parallel iteration)."""
        strategy = ZipStrategy()
        result = strategy.expand({"_zip_": {"x": [1, 2, 3], "y": ["A", "B", "C"]}})
        assert len(result) == 3
        assert result == [
            {"x": 1, "y": "A"},
            {"x": 2, "y": "B"},
            {"x": 3, "y": "C"}
        ]

    def test_zip_expand_unequal_lengths(self):
        """Zip with unequal list lengths (stops at shortest)."""
        strategy = ZipStrategy()
        result = strategy.expand({"_zip_": {"x": [1, 2, 3, 4], "y": ["A", "B"]}})
        assert len(result) == 2  # Stops at shortest
        assert result == [{"x": 1, "y": "A"}, {"x": 2, "y": "B"}]

    def test_zip_expand_three_params(self):
        """Zip with three parameters."""
        strategy = ZipStrategy()
        result = strategy.expand({
            "_zip_": {"x": [1, 2], "y": ["A", "B"], "z": [True, False]}
        })
        assert len(result) == 2
        assert result[0] == {"x": 1, "y": "A", "z": True}
        assert result[1] == {"x": 2, "y": "B", "z": False}

    def test_zip_expand_with_count(self):
        """Zip with count limit."""
        strategy = ZipStrategy()
        result = strategy.expand(
            {"_zip_": {"x": [1, 2, 3], "y": ["A", "B", "C"]}, "count": 2},
            seed=42
        )
        assert len(result) == 2

    def test_zip_count(self):
        """Zip counting."""
        strategy = ZipStrategy()
        assert strategy.count({"_zip_": {"x": [1, 2, 3], "y": ["A", "B", "C"]}}) == 3
        assert strategy.count({"_zip_": {"x": [1, 2, 3, 4], "y": ["A", "B"]}}) == 2
        assert strategy.count({"_zip_": {"x": [1, 2, 3], "y": ["A", "B", "C"]}, "count": 1}) == 1

    def test_zip_empty(self):
        """Zip with empty dict."""
        strategy = ZipStrategy()
        result = strategy.expand({"_zip_": {}})
        assert result == [{}]

class TestChainStrategy:
    """Tests for ChainStrategy."""

    def test_chain_strategy_handles(self):
        """ChainStrategy should handle pure chain nodes."""
        assert ChainStrategy.handles({"_chain_": [1, 2, 3]})
        assert ChainStrategy.handles({"_chain_": [1, 2], "count": 1})
        # Should not handle mixed nodes
        assert not ChainStrategy.handles({"_chain_": [1, 2], "extra": "value"})

    def test_chain_expand_basic(self):
        """Basic chain expansion (sequential)."""
        strategy = ChainStrategy()
        result = strategy.expand({"_chain_": [{"x": 1}, {"x": 2}, {"x": 3}]})
        assert len(result) == 3
        assert result == [{"x": 1}, {"x": 2}, {"x": 3}]

    def test_chain_expand_scalars(self):
        """Chain with scalar values."""
        strategy = ChainStrategy()
        result = strategy.expand({"_chain_": ["A", "B", "C"]})
        assert result == ["A", "B", "C"]

    def test_chain_expand_preserves_order(self):
        """Chain should preserve order."""
        strategy = ChainStrategy()
        result = strategy.expand({
            "_chain_": [
                {"stage": "init"},
                {"stage": "train"},
                {"stage": "evaluate"},
                {"stage": "deploy"}
            ]
        })
        assert len(result) == 4
        assert result[0]["stage"] == "init"
        assert result[1]["stage"] == "train"
        assert result[2]["stage"] == "evaluate"
        assert result[3]["stage"] == "deploy"

    def test_chain_expand_with_count_no_seed(self):
        """Chain with count (no seed) takes first n."""
        strategy = ChainStrategy()
        result = strategy.expand(
            {"_chain_": [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}], "count": 2}
        )
        assert len(result) == 2
        assert result == [{"x": 1}, {"x": 2}]  # First 2, not random

    def test_chain_expand_with_count_and_seed(self):
        """Chain with count and seed samples randomly."""
        strategy = ChainStrategy()
        result = strategy.expand(
            {"_chain_": [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}], "count": 2, "_seed_": 42}
        )
        assert len(result) == 2

    def test_chain_count(self):
        """Chain counting."""
        strategy = ChainStrategy()
        assert strategy.count({"_chain_": [1, 2, 3]}) == 3
        assert strategy.count({"_chain_": [1, 2, 3, 4], "count": 2}) == 2

    def test_chain_empty(self):
        """Chain with empty list."""
        strategy = ChainStrategy()
        result = strategy.expand({"_chain_": []})
        assert result == []

class TestSampleStrategy:
    """Tests for SampleStrategy."""

    def test_sample_strategy_handles(self):
        """SampleStrategy should handle pure sample nodes."""
        assert SampleStrategy.handles({"_sample_": {"distribution": "uniform"}})
        assert SampleStrategy.handles({"_sample_": {"distribution": "uniform"}, "count": 3})
        # Should not handle mixed nodes
        assert not SampleStrategy.handles({"_sample_": {"distribution": "uniform"}, "extra": "x"})

    def test_sample_uniform_basic(self):
        """Basic uniform sampling."""
        strategy = SampleStrategy()
        result = strategy.expand({
            "_sample_": {"distribution": "uniform", "from": 0, "to": 1, "num": 5}
        }, seed=42)
        assert len(result) == 5
        assert all(0 <= v <= 1 for v in result)

    def test_sample_uniform_range(self):
        """Uniform sampling with custom range."""
        strategy = SampleStrategy()
        result = strategy.expand({
            "_sample_": {"distribution": "uniform", "from": 10, "to": 20, "num": 100}
        }, seed=42)
        assert len(result) == 100
        assert all(10 <= v <= 20 for v in result)

    def test_sample_log_uniform(self):
        """Log-uniform sampling."""
        strategy = SampleStrategy()
        result = strategy.expand({
            "_sample_": {"distribution": "log_uniform", "from": 0.001, "to": 1, "num": 10}
        }, seed=42)
        assert len(result) == 10
        assert all(0.001 <= v <= 1 for v in result)

    def test_sample_normal(self):
        """Normal distribution sampling."""
        strategy = SampleStrategy()
        result = strategy.expand({
            "_sample_": {"distribution": "normal", "mean": 0, "std": 1, "num": 100}
        }, seed=42)
        assert len(result) == 100
        # Mean should be roughly 0 with large sample
        assert abs(sum(result) / len(result)) < 1  # Rough check

    def test_sample_choice(self):
        """Choice sampling."""
        strategy = SampleStrategy()
        result = strategy.expand({
            "_sample_": {"distribution": "choice", "values": ["A", "B", "C"], "num": 10}
        }, seed=42)
        assert len(result) == 10
        assert all(v in ["A", "B", "C"] for v in result)

    def test_sample_with_count(self):
        """Sample with count limit."""
        strategy = SampleStrategy()
        result = strategy.expand({
            "_sample_": {"distribution": "uniform", "from": 0, "to": 1, "num": 10},
            "count": 3
        }, seed=42)
        assert len(result) == 3

    def test_sample_deterministic_with_seed(self):
        """Sample should be deterministic with same seed."""
        strategy = SampleStrategy()
        spec = {"_sample_": {"distribution": "uniform", "from": 0, "to": 1, "num": 5}}
        result1 = strategy.expand(spec, seed=42)
        result2 = strategy.expand(spec, seed=42)
        assert result1 == result2

    def test_sample_count(self):
        """Sample counting."""
        strategy = SampleStrategy()
        assert strategy.count({"_sample_": {"num": 10}}) == 10
        assert strategy.count({"_sample_": {"num": 10}, "count": 5}) == 5

    def test_sample_validate(self):
        """Sample validation."""
        strategy = SampleStrategy()
        assert strategy.validate({"_sample_": {"distribution": "uniform"}}) == []
        # Invalid distribution
        errors = strategy.validate({"_sample_": {"distribution": "unknown"}})
        assert len(errors) > 0
        # Invalid negative std for normal
        errors = strategy.validate({"_sample_": {"distribution": "normal", "std": -1}})
        assert len(errors) > 0

class TestPhase3Integration:
    """Integration tests for Phase 3 strategies with expand_spec."""

    def test_expand_spec_uses_log_range_strategy(self):
        """expand_spec should use LogRangeStrategy for log range nodes."""
        result = expand_spec({"_log_range_": [0.001, 1, 4]})
        assert len(result) == 4
        assert abs(result[0] - 0.001) < 1e-9

    def test_expand_spec_uses_grid_strategy(self):
        """expand_spec should use GridStrategy for grid nodes."""
        result = expand_spec({"_grid_": {"x": [1, 2], "y": ["A", "B"]}})
        assert len(result) == 4

    def test_expand_spec_uses_zip_strategy(self):
        """expand_spec should use ZipStrategy for zip nodes."""
        result = expand_spec({"_zip_": {"x": [1, 2], "y": ["A", "B"]}})
        assert len(result) == 2
        assert result == [{"x": 1, "y": "A"}, {"x": 2, "y": "B"}]

    def test_expand_spec_uses_chain_strategy(self):
        """expand_spec should use ChainStrategy for chain nodes."""
        result = expand_spec({"_chain_": ["A", "B", "C"]})
        assert result == ["A", "B", "C"]

    def test_expand_spec_uses_sample_strategy(self):
        """expand_spec should use SampleStrategy for sample nodes."""
        result = expand_spec(
            {"_sample_": {"distribution": "uniform", "from": 0, "to": 1, "num": 3}},
            seed=42
        )
        assert len(result) == 3

    def test_nested_phase3_strategies(self):
        """Test nested Phase 3 generator nodes."""
        result = expand_spec({
            "learning_rate": {"_log_range_": [0.001, 0.1, 3]},
            "model": {"_or_": ["CNN", "RNN"]}
        })
        # 3 learning rates x 2 models = 6 configs
        assert len(result) == 6

    def test_count_with_phase3_strategies(self):
        """count_combinations should work with Phase 3 strategies."""
        assert count_combinations({"_log_range_": [0.001, 1, 10]}) == 10
        assert count_combinations({"_grid_": {"x": [1, 2], "y": [1, 2, 3]}}) == 6
        assert count_combinations({"_zip_": {"x": [1, 2, 3], "y": ["A", "B"]}}) == 2
        assert count_combinations({"_chain_": [1, 2, 3, 4, 5]}) == 5
        assert count_combinations({"_sample_": {"num": 20}}) == 20

class TestStrategyPriority:
    """Tests for strategy priority ordering."""

    def test_phase3_strategies_have_correct_priority(self):
        """Phase 3 strategies should have appropriate priorities."""
        # Higher priority = checked first
        assert GridStrategy.priority > LogRangeStrategy.priority
        assert LogRangeStrategy.priority > SampleStrategy.priority

    def test_get_strategy_returns_correct_phase3_strategy(self):
        """get_strategy should return correct Phase 3 strategy."""
        assert isinstance(get_strategy({"_log_range_": [1, 100, 5]}), LogRangeStrategy)
        assert isinstance(get_strategy({"_grid_": {"x": [1, 2]}}), GridStrategy)
        assert isinstance(get_strategy({"_zip_": {"x": [1, 2]}}), ZipStrategy)
        assert isinstance(get_strategy({"_chain_": [1, 2]}), ChainStrategy)
        assert isinstance(get_strategy({"_sample_": {"num": 10}}), SampleStrategy)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
