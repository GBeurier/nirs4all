"""Tests for Phase 4 generator features.

This module tests the features introduced in Phase 4:
- Iterator-based expansion (expand_spec_iter)
- Constraints (_mutex_, _requires_, _exclude_)
- Presets system (_preset_)
- Export utilities (to_dataframe, diff_configs, print_expansion_tree)
"""

import pytest
from itertools import islice

from nirs4all.pipeline.config.generator import (
    # Core API
    expand_spec,
    count_combinations,
    # Iterator API (Phase 4)
    expand_spec_iter,
    batch_iter,
    iter_with_progress,
    # Constraints (Phase 4)
    apply_mutex_constraint,
    apply_requires_constraint,
    apply_exclude_constraint,
    apply_all_constraints,
    parse_constraints,
    validate_constraints,
    # Presets (Phase 4)
    PRESET_KEYWORD,
    register_preset,
    unregister_preset,
    get_preset,
    list_presets,
    clear_presets,
    has_preset,
    is_preset_reference,
    resolve_preset,
    resolve_presets_recursive,
    # Export utilities (Phase 4)
    diff_configs,
    summarize_configs,
    get_expansion_tree,
    print_expansion_tree,
    format_config_table,
)


# =============================================================================
# Iterator Tests
# =============================================================================

class TestExpandSpecIter:
    """Tests for expand_spec_iter lazy expansion."""

    def test_basic_iter_matches_expand(self):
        """Iterator should produce same results as expand_spec."""
        spec = {"_or_": ["A", "B", "C"]}
        eager = expand_spec(spec)
        lazy = list(expand_spec_iter(spec))
        assert eager == lazy

    def test_iter_with_range(self):
        """Iterator with range should produce values lazily."""
        spec = {"_range_": [1, 100]}
        # Only get first 5 - should not generate all 100
        first_5 = list(islice(expand_spec_iter(spec), 5))
        assert first_5 == [1, 2, 3, 4, 5]

    def test_iter_with_pick(self):
        """Iterator with pick combinations."""
        spec = {"_or_": ["A", "B", "C"], "pick": 2}
        eager = expand_spec(spec)
        lazy = list(expand_spec_iter(spec))
        assert set(tuple(x) for x in eager) == set(tuple(x) for x in lazy)

    def test_iter_with_nested_dict(self):
        """Iterator with nested dict expansion."""
        spec = {"x": {"_or_": [1, 2]}, "y": {"_or_": ["A", "B"]}}
        eager = expand_spec(spec)
        lazy = list(expand_spec_iter(spec))
        assert len(eager) == len(lazy)
        # Check all configs present
        for config in eager:
            assert config in lazy

    def test_iter_with_sample_size(self):
        """Iterator with reservoir sampling."""
        spec = {"_range_": [1, 100]}
        sampled = list(expand_spec_iter(spec, seed=42, sample_size=10))
        assert len(sampled) == 10
        assert all(1 <= v <= 100 for v in sampled)

    def test_iter_deterministic_with_seed(self):
        """Sampling should be deterministic with same seed."""
        spec = {"_range_": [1, 100]}
        sample1 = list(expand_spec_iter(spec, seed=42, sample_size=10))
        sample2 = list(expand_spec_iter(spec, seed=42, sample_size=10))
        assert sample1 == sample2

    def test_iter_scalar(self):
        """Iterator with scalar input."""
        assert list(expand_spec_iter("scalar")) == ["scalar"]
        assert list(expand_spec_iter(42)) == [42]

    def test_iter_empty_list(self):
        """Iterator with empty list."""
        assert list(expand_spec_iter([])) == [[]]


class TestBatchIter:
    """Tests for batch_iter utility."""

    def test_batch_iter_exact_batches(self):
        """Batch iteration with exact batch sizes."""
        spec = {"_range_": [1, 10]}
        batches = list(batch_iter(spec, batch_size=5))
        assert len(batches) == 2
        assert batches[0] == [1, 2, 3, 4, 5]
        assert batches[1] == [6, 7, 8, 9, 10]

    def test_batch_iter_partial_last(self):
        """Batch iteration with partial last batch."""
        spec = {"_range_": [1, 7]}
        batches = list(batch_iter(spec, batch_size=3))
        assert len(batches) == 3
        assert batches[0] == [1, 2, 3]
        assert batches[1] == [4, 5, 6]
        assert batches[2] == [7]


class TestIterWithProgress:
    """Tests for iter_with_progress utility."""

    def test_iter_with_progress_yields_tuples(self):
        """Progress iteration should yield (index, config) tuples."""
        spec = {"_or_": ["A", "B", "C"]}
        results = list(iter_with_progress(spec))
        assert results == [(0, "A"), (1, "B"), (2, "C")]


# =============================================================================
# Constraint Tests
# =============================================================================

class TestMutexConstraint:
    """Tests for mutual exclusion constraints."""

    def test_mutex_basic(self):
        """Basic mutex filtering."""
        combos = [["A", "B"], ["A", "C"], ["B", "C"]]
        result = apply_mutex_constraint(combos, [["A", "B"]])
        assert result == [["A", "C"], ["B", "C"]]

    def test_mutex_multiple_groups(self):
        """Multiple mutex groups."""
        combos = [["A", "B"], ["A", "C"], ["B", "C"], ["C", "D"]]
        result = apply_mutex_constraint(combos, [["A", "B"], ["C", "D"]])
        assert result == [["A", "C"], ["B", "C"]]

    def test_mutex_no_violations(self):
        """No mutex violations returns all combos."""
        combos = [["A", "C"], ["B", "D"]]
        result = apply_mutex_constraint(combos, [["A", "B"]])
        assert result == combos

    def test_mutex_empty_groups(self):
        """Empty mutex groups returns all combos."""
        combos = [["A", "B"], ["C", "D"]]
        result = apply_mutex_constraint(combos, [])
        assert result == combos


class TestRequiresConstraint:
    """Tests for dependency requirement constraints."""

    def test_requires_basic(self):
        """Basic requires filtering."""
        combos = [["A", "B"], ["A", "C"], ["B", "C"]]
        result = apply_requires_constraint(combos, [["A", "B"]])
        # A requires B: [A,C] is invalid because A present but B not
        assert result == [["A", "B"], ["B", "C"]]

    def test_requires_no_trigger(self):
        """Combos without trigger pass."""
        combos = [["X", "Y"], ["B", "C"]]
        result = apply_requires_constraint(combos, [["A", "B"]])
        assert result == combos

    def test_requires_multiple_deps(self):
        """Trigger requires multiple items."""
        combos = [["A", "B", "C"], ["A", "B"], ["B", "C"]]
        result = apply_requires_constraint(combos, [["A", "B", "C"]])
        # A requires both B and C
        assert result == [["A", "B", "C"], ["B", "C"]]


class TestExcludeConstraint:
    """Tests for explicit exclusion constraints."""

    def test_exclude_basic(self):
        """Basic exclusion."""
        combos = [["A", "B"], ["A", "C"], ["B", "C"]]
        result = apply_exclude_constraint(combos, [["A", "B"]])
        assert result == [["A", "C"], ["B", "C"]]

    def test_exclude_multiple(self):
        """Multiple exclusions."""
        combos = [["A", "B"], ["A", "C"], ["B", "C"]]
        result = apply_exclude_constraint(combos, [["A", "B"], ["B", "C"]])
        assert result == [["A", "C"]]


class TestApplyAllConstraints:
    """Tests for combined constraint application."""

    def test_all_constraints_together(self):
        """Apply mutex, requires, and exclude together."""
        combos = [
            ["A", "B"],  # excluded by mutex
            ["A", "C"],  # excluded by requires (A needs B)
            ["B", "C"],  # OK
            ["C", "D"],  # excluded explicitly
        ]
        result = apply_all_constraints(
            combos,
            mutex_groups=[["A", "B"]],
            requires_groups=[["A", "B"]],
            exclude_combos=[["C", "D"]]
        )
        assert result == [["B", "C"]]


class TestConstraintsInExpandSpec:
    """Tests for constraint integration with expand_spec."""

    def test_expand_with_mutex(self):
        """expand_spec with _mutex_ constraint."""
        spec = {
            "_or_": ["A", "B", "C"],
            "pick": 2,
            "_mutex_": [["A", "B"]]
        }
        result = expand_spec(spec)
        # ["A", "B"] should be excluded
        assert ["A", "B"] not in result
        assert ["A", "C"] in result
        assert ["B", "C"] in result

    def test_expand_with_requires(self):
        """expand_spec with _requires_ constraint."""
        spec = {
            "_or_": ["A", "B", "C"],
            "pick": 2,
            "_requires_": [["A", "B"]]
        }
        result = expand_spec(spec)
        # ["A", "C"] should be excluded (A requires B)
        assert ["A", "C"] not in result
        assert ["A", "B"] in result  # A with B is OK
        assert ["B", "C"] in result  # No A, so OK

    def test_expand_with_exclude(self):
        """expand_spec with _exclude_ constraint."""
        spec = {
            "_or_": ["A", "B", "C"],
            "pick": 2,
            "_exclude_": [["A", "C"]]
        }
        result = expand_spec(spec)
        assert ["A", "C"] not in result
        assert ["A", "B"] in result
        assert ["B", "C"] in result


# =============================================================================
# Preset Tests
# =============================================================================

class TestPresetRegistry:
    """Tests for preset registration and retrieval."""

    def setup_method(self):
        """Clear presets before each test."""
        clear_presets()

    def teardown_method(self):
        """Clear presets after each test."""
        clear_presets()

    def test_register_and_get_preset(self):
        """Register and retrieve a preset."""
        register_preset("test", {"_or_": ["A", "B"]})
        assert has_preset("test")
        spec = get_preset("test")
        assert spec == {"_or_": ["A", "B"]}

    def test_list_presets(self):
        """List registered presets."""
        register_preset("preset1", {"x": 1})
        register_preset("preset2", {"x": 2})
        names = list_presets()
        assert "preset1" in names
        assert "preset2" in names

    def test_unregister_preset(self):
        """Remove a preset."""
        register_preset("temp", {"x": 1})
        assert has_preset("temp")
        unregister_preset("temp")
        assert not has_preset("temp")

    def test_preset_overwrite(self):
        """Overwrite existing preset."""
        register_preset("test", {"x": 1})
        with pytest.raises(ValueError):
            register_preset("test", {"x": 2})  # Should fail
        register_preset("test", {"x": 2}, overwrite=True)  # Should succeed
        assert get_preset("test") == {"x": 2}

    def test_preset_with_tags(self):
        """Preset with tags for filtering."""
        register_preset("p1", {"x": 1}, tags=["ml", "preprocessing"])
        register_preset("p2", {"x": 2}, tags=["ml", "models"])
        register_preset("p3", {"x": 3}, tags=["utils"])

        ml_presets = list_presets(tags=["ml"])
        assert "p1" in ml_presets
        assert "p2" in ml_presets
        assert "p3" not in ml_presets

    def test_preset_not_found(self):
        """Get nonexistent preset raises KeyError."""
        with pytest.raises(KeyError):
            get_preset("nonexistent")


class TestPresetResolution:
    """Tests for preset reference resolution."""

    def setup_method(self):
        clear_presets()
        register_preset("choices", {"_or_": ["A", "B", "C"]})
        register_preset("range", {"_range_": [1, 5]})

    def teardown_method(self):
        clear_presets()

    def test_is_preset_reference(self):
        """Detect preset references."""
        assert is_preset_reference({"_preset_": "test"})
        assert not is_preset_reference({"_or_": ["A", "B"]})
        assert not is_preset_reference("string")

    def test_resolve_preset(self):
        """Resolve a single preset reference."""
        ref = {"_preset_": "choices"}
        spec = resolve_preset(ref)
        assert spec == {"_or_": ["A", "B", "C"]}

    def test_resolve_presets_recursive(self):
        """Resolve nested preset references."""
        register_preset("nested", {"x": {"_preset_": "choices"}})

        spec = {"config": {"_preset_": "nested"}}
        resolved = resolve_presets_recursive(spec)

        assert resolved == {
            "config": {"x": {"_or_": ["A", "B", "C"]}}
        }

    def test_resolve_circular_reference(self):
        """Circular preset references should raise error."""
        register_preset("a", {"ref": {"_preset_": "b"}}, overwrite=True)
        register_preset("b", {"ref": {"_preset_": "a"}}, overwrite=True)

        with pytest.raises(ValueError, match="Circular"):
            resolve_presets_recursive({"_preset_": "a"})


# =============================================================================
# Export Utility Tests
# =============================================================================

class TestDiffConfigs:
    """Tests for configuration diff utility."""

    def test_diff_identical(self):
        """Identical configs have no diff."""
        config = {"a": 1, "b": 2}
        assert diff_configs(config, config) == {}

    def test_diff_simple_change(self):
        """Simple value change."""
        c1 = {"a": 1, "b": 2}
        c2 = {"a": 1, "b": 3}
        diff = diff_configs(c1, c2)
        assert diff == {"b": (2, 3)}

    def test_diff_nested(self):
        """Nested value change."""
        c1 = {"a": {"x": 1}}
        c2 = {"a": {"x": 2}}
        diff = diff_configs(c1, c2)
        assert diff == {"a.x": (1, 2)}

    def test_diff_added_key(self):
        """Key added in second config."""
        c1 = {"a": 1}
        c2 = {"a": 1, "b": 2}
        diff = diff_configs(c1, c2)
        assert diff == {"b": (None, 2)}

    def test_diff_removed_key(self):
        """Key removed in second config."""
        c1 = {"a": 1, "b": 2}
        c2 = {"a": 1}
        diff = diff_configs(c1, c2)
        assert diff == {"b": (2, None)}


class TestSummarizeConfigs:
    """Tests for config summarization."""

    def test_summarize_basic(self):
        """Basic config summary."""
        configs = [
            {"model": "PLS", "n": 5},
            {"model": "PLS", "n": 10},
            {"model": "RF", "n": 100}
        ]
        summary = summarize_configs(configs)
        assert summary["count"] == 3
        assert summary["keys"]["model"]["unique_count"] == 2
        assert summary["keys"]["n"]["unique_count"] == 3


class TestExpansionTree:
    """Tests for expansion tree visualization."""

    def test_tree_simple_or(self):
        """Tree for simple OR node."""
        spec = {"_or_": ["A", "B", "C"]}
        tree = get_expansion_tree(spec)
        assert tree.node_type == "_or_"
        assert tree.count == 3
        assert len(tree.children) == 3

    def test_tree_nested(self):
        """Tree for nested spec."""
        spec = {
            "x": {"_or_": [1, 2]},
            "y": {"_range_": [1, 3]}
        }
        tree = get_expansion_tree(spec)
        assert tree.node_type == "dict"
        assert tree.count == 6  # 2 x 3
        assert len(tree.children) == 2

    def test_print_tree_output(self):
        """Print tree produces string output."""
        spec = {"_or_": ["A", "B"]}
        output = print_expansion_tree(spec)
        assert "root" in output
        assert "_or_" in output
        assert "2 variants" in output


class TestFormatConfigTable:
    """Tests for ASCII table formatting."""

    def test_format_table_basic(self):
        """Basic table formatting."""
        configs = [
            {"model": "PLS", "n": 5},
            {"model": "RF", "n": 100}
        ]
        table = format_config_table(configs)
        assert "model" in table
        assert "PLS" in table
        assert "RF" in table

    def test_format_table_empty(self):
        """Empty configs."""
        table = format_config_table([])
        assert "no configurations" in table


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase4Integration:
    """Integration tests combining Phase 4 features."""

    def setup_method(self):
        clear_presets()

    def teardown_method(self):
        clear_presets()

    def test_preset_with_constraints(self):
        """Presets combined with constraints."""
        register_preset("transforms", {
            "_or_": ["SNV", "MSC", "Detrend", "Normalize"],
            "pick": 2,
            "_mutex_": [["SNV", "MSC"]]  # Can't use both together
        })

        spec = {"transforms": {"_preset_": "transforms"}}
        resolved = resolve_presets_recursive(spec)
        result = expand_spec(resolved["transforms"])

        # SNV and MSC should never appear together
        for combo in result:
            assert not ("SNV" in combo and "MSC" in combo)

    def test_iter_with_constraints(self):
        """Iterator works with constraints."""
        spec = {
            "_or_": ["A", "B", "C", "D"],
            "pick": 2,
            "_mutex_": [["A", "B"]]
        }

        eager = expand_spec(spec)
        lazy = list(expand_spec_iter(spec))

        # Both should have same results (constraints applied)
        assert len(eager) == len(lazy)
        for combo in eager:
            assert combo in lazy

    def test_tree_shows_constraints(self):
        """Tree visualization with constrained spec."""
        spec = {
            "_or_": ["A", "B", "C"],
            "pick": 2,
            "_mutex_": [["A", "B"]]
        }
        tree = get_expansion_tree(spec)
        # Count should reflect constraints aren't pre-applied in tree
        # (tree shows theoretical count, constraints filter at expand time)
        assert tree.count >= 2  # At least 2 valid combinations


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
