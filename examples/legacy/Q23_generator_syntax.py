"""
Q23: Generator Syntax Examples
==============================
This example demonstrates the generator syntax for creating multiple pipeline
configurations from a single specification. No models are run - this is purely
to illustrate and validate the generation mechanisms.

Generator Keywords (Phases 1-4):
--------------------------------
Phase 1-2: Core Keywords
- _or_: Choose between alternatives
- _range_: Generate numeric sequences
- size: Number of items to select from _or_ choices (legacy)
- pick: Unordered selection (combinations) - explicit intent
- arrange: Ordered arrangement (permutations) - explicit intent
- then_pick: Apply combinations to primary results
- then_arrange: Apply permutations to primary results
- count: Limit number of generated variants

Phase 3: Advanced Keywords
- _log_range_: Logarithmic numeric sequences
- _grid_: Grid search style Cartesian product
- _zip_: Parallel iteration (like Python's zip)
- _chain_: Sequential ordered choices
- _sample_: Statistical sampling (uniform, log-uniform, normal)
- _tags_: Configuration tagging for filtering
- _metadata_: Arbitrary metadata attachment

Phase 4: Production Features
- _mutex_: Mutual exclusion constraints
- _requires_: Dependency requirements
- _exclude_: Exclusion rules
- _preset_: Named configuration templates
- expand_spec_iter: Lazy iterator for large spaces

This example covers:
1. Basic _or_ expansion
2. _or_ with size (combinations) - legacy
3. pick keyword (explicit combinations)
4. arrange keyword (explicit permutations)
5. pick vs arrange comparison
6. _range_ for numeric sequences
7. Nested _or_ in dictionaries
8. Complex nested structures
9. count for limiting results
10. Second-order combinations with [outer, inner] size/pick/arrange
10.5. then_pick and then_arrange keywords
11. _log_range_ for logarithmic sequences (Phase 3)
12. _grid_ for grid search (Phase 3)
13. _zip_ for parallel iteration (Phase 3)
14. _chain_ for sequential choices (Phase 3)
15. _sample_ for statistical sampling (Phase 3)
16. Constraints: _mutex_, _requires_, _exclude_ (Phase 4)
17. Presets: _preset_ for named templates (Phase 4)
18. Iterator: expand_spec_iter for large spaces (Phase 4)
19. Export: to_dataframe, diff_configs, print_expansion_tree (Phase 4)

Usage:
    python Q23_generator_syntax.py
"""

from nirs4all.pipeline.config.generator import (
    # Core API
    expand_spec,
    count_combinations,
    # Iterator API (Phase 4)
    expand_spec_iter,
    batch_iter,
    # Detection functions
    is_generator_node,
    # Utilities
    sample_with_seed,
    # Keyword constants
    PICK_KEYWORD,
    ARRANGE_KEYWORD,
    THEN_PICK_KEYWORD,
    THEN_ARRANGE_KEYWORD,
    # Preset functions (Phase 4)
    register_preset,
    list_presets,
    clear_presets,
    resolve_presets_recursive,
    # Export utilities (Phase 4)
    to_dataframe,
    diff_configs,
    summarize_configs,
    print_expansion_tree,
)

import json


def print_section(title: str, phase: str = ""):
    """Print a section header."""
    print()
    print("=" * 70)
    if phase:
        print(f"{title} [{phase}]")
    else:
        print(title)
    print("=" * 70)
    print()


def show_expansion(name: str, spec, show_all: bool = True, max_show: int = 10, seed: int = None):
    """Show expansion results for a specification."""
    print(f"--- {name} ---")
    print(f"Spec: {json.dumps(spec, indent=2, default=str)}")
    print()

    count = count_combinations(spec)
    print(f"Count (without generating): {count}")

    results = expand_spec(spec, seed=seed)
    print(f"Actual results count: {len(results)}")

    if show_all or len(results) <= max_show:
        print("Results:")
        for i, r in enumerate(results):
            print(f"  [{i}] {r}")
    else:
        print(f"First {max_show} results:")
        for i, r in enumerate(results[:max_show]):
            print(f"  [{i}] {r}")
        print(f"  ... ({len(results) - max_show} more)")

    print()
    return results


# ============================================================
# Example 1: Basic _or_ expansion
# ============================================================
print_section("EXAMPLE 1: Basic _or_ expansion", "Phase 1")
print("The _or_ keyword creates variants from a list of choices.")
print("Each choice becomes a separate configuration.")

# Simple string choices
show_expansion(
    "String choices",
    {"_or_": ["StandardScaler", "MinMaxScaler", "RobustScaler"]}
)

# Dictionary choices (typical for pipeline steps)
show_expansion(
    "Dictionary choices",
    {"_or_": [
        {"class": "PCA", "n_components": 10},
        {"class": "TruncatedSVD", "n_components": 10},
        {"class": "FastICA", "n_components": 10},
    ]}
)

# Mixed types (scalars, dicts)
show_expansion(
    "Mixed types",
    {"_or_": [None, 5, {"window": 11}]}
)


# ============================================================
# Example 2: _or_ with size (combinations)
# ============================================================
print_section("EXAMPLE 2: _or_ with size (combinations)", "Phase 1")
print("The 'size' parameter selects combinations of N items from choices.")
print("Uses mathematical combinations C(n, k).")

# Size = 2: pick 2 from 3 choices -> C(3,2) = 3 combinations
show_expansion(
    "size=2 from 3 items (C(3,2)=3)",
    {"_or_": ["A", "B", "C"], "size": 2}
)

# Size = 2 from 4 choices -> C(4,2) = 6 combinations
show_expansion(
    "size=2 from 4 items (C(4,2)=6)",
    {"_or_": ["PCA", "SVD", "ICA", "NMF"], "size": 2}
)


# ============================================================
# Example 3: _or_ with size tuple (range of sizes)
# ============================================================
print_section("EXAMPLE 3: _or_ with size tuple (range of sizes)", "Phase 1")
print("size=(from, to) generates combinations for all sizes in range.")
print("Example: size=(1,3) generates C(n,1) + C(n,2) + C(n,3).")

# size=(1,2) from 3 items: C(3,1) + C(3,2) = 3 + 3 = 6
show_expansion(
    "size=(1,2) from 3 items",
    {"_or_": ["A", "B", "C"], "size": (1, 2)}
)


# ============================================================
# Example 3.5: pick keyword (explicit combinations)
# ============================================================
print_section("EXAMPLE 3.5: pick keyword (explicit combinations)", "Phase 2")
print("The 'pick' keyword is the explicit version of 'size'.")
print("It clearly indicates unordered selection (combinations).")
print(f"Keyword constant: PICK_KEYWORD = '{PICK_KEYWORD}'")

# pick = 2: same as size = 2
show_expansion(
    "pick=2 from 3 items (same as size=2)",
    {"_or_": ["A", "B", "C"], "pick": 2}
)

# pick with range
show_expansion(
    "pick=(1,2) from 3 items",
    {"_or_": ["A", "B", "C"], "pick": (1, 2)}
)


# ============================================================
# Example 3.6: arrange keyword (explicit permutations)
# ============================================================
print_section("EXAMPLE 3.6: arrange keyword (explicit permutations)", "Phase 2")
print("The 'arrange' keyword selects items where ORDER MATTERS.")
print("Uses mathematical permutations P(n, k).")
print(f"Keyword constant: ARRANGE_KEYWORD = '{ARRANGE_KEYWORD}'")

# arrange = 2: P(3,2) = 6 permutations
show_expansion(
    "arrange=2 from 3 items (P(3,2)=6)",
    {"_or_": ["A", "B", "C"], "arrange": 2}
)


# ============================================================
# Example 3.7: pick vs arrange comparison
# ============================================================
print_section("EXAMPLE 3.7: pick vs arrange comparison", "Phase 2")
print("Comparing pick (combinations) vs arrange (permutations).")
print("Key difference: arrange includes both [A,B] and [B,A] as separate results.")

choices = ["A", "B", "C"]

print("--- pick=2 (combinations) ---")
pick_result = expand_spec({"_or_": choices, "pick": 2})
print(f"Count: {len(pick_result)}")
print(f"Results: {pick_result}")
print("Note: [A,B] appears, but [B,A] does NOT (order doesn't matter)")
print()

print("--- arrange=2 (permutations) ---")
arrange_result = expand_spec({"_or_": choices, "arrange": 2})
print(f"Count: {len(arrange_result)}")
print(f"Results: {arrange_result}")
print("Note: BOTH [A,B] AND [B,A] appear (order matters)")
print()

print("Use case guidance:")
print("- Use 'pick' for concat_transform (feature order doesn't matter)")
print("- Use 'pick' for feature_augmentation (parallel channels)")
print("- Use 'arrange' for sequential preprocessing steps")
print("- Use 'arrange' when the order of operations affects the result")
print()


# ============================================================
# Example 4: _range_ for numeric sequences
# ============================================================
print_section("EXAMPLE 4: _range_ for numeric sequences", "Phase 1")
print("The _range_ keyword generates a sequence of numbers.")
print("Supports [start, end] or [start, end, step] syntax.")

# Basic range [from, to] - inclusive
show_expansion(
    "Range [1, 5] (inclusive)",
    {"_range_": [1, 5]}
)

# Range with step
show_expansion(
    "Range [0, 20, 5] (step=5)",
    {"_range_": [0, 20, 5]}
)

# Range in dict syntax
show_expansion(
    "Range dict syntax",
    {"_range_": {"from": 10, "to": 50, "step": 10}}
)


# ============================================================
# Example 5: Nested _or_ in dictionaries
# ============================================================
print_section("EXAMPLE 5: Nested _or_ in dictionaries", "Phase 1")
print("When _or_ is a value in a dictionary, it expands that key only.")
print("The Cartesian product is taken across all keys.")

# Multiple keys with _or_ -> Cartesian product
show_expansion(
    "Two keys with _or_ (Cartesian product)",
    {
        "n_components": {"_or_": [5, 10]},
        "random_state": {"_or_": [0, 42]}
    }
)

# Mix of _or_ and fixed values
show_expansion(
    "Mix of _or_ and fixed values",
    {
        "class": "PCA",
        "n_components": {"_or_": [5, 10, 20]},
        "whiten": True
    }
)


# ============================================================
# Example 6: Complex nested structures
# ============================================================
print_section("EXAMPLE 6: Complex nested structures", "Phase 1")
print("Generator syntax works recursively in complex structures.")

# Nested dicts
show_expansion(
    "Nested dictionary structure",
    {
        "scaler": {"class": "StandardScaler"},
        "reducer": {
            "_or_": [
                {"class": "PCA", "n_components": {"_or_": [5, 10]}},
                {"class": "SVD", "n_components": {"_or_": [5, 10]}},
            ]
        }
    }
)

# List of items with _or_
show_expansion(
    "List with _or_ elements",
    [
        {"_or_": ["A", "B"]},
        {"_or_": ["X", "Y"]}
    ]
)


# ============================================================
# Example 7: count for limiting results
# ============================================================
print_section("EXAMPLE 7: count for limiting results", "Phase 1")
print("The 'count' parameter limits the number of results returned.")
print("With seed parameter, results are deterministic.")

# count=2 from many choices with seed for reproducibility
print("--- count=2 from 5 choices (with seed for reproducibility) ---")
spec = {"_or_": ["A", "B", "C", "D", "E"], "count": 2}
print(f"Spec: {spec}")
print(f"Count: {count_combinations(spec)}")

# Run with seed for deterministic results
for i in range(3):
    results = expand_spec(spec, seed=42)
    print(f"  Run {i+1} (seed=42): {results}")
print()


# ============================================================
# Example 8: Second-order combinations with [outer, inner]
# ============================================================
print_section("EXAMPLE 8: Second-order combinations [outer, inner]", "Phase 2")
print("size=[outer, inner] creates nested combinations:")
print("- inner: permutations (order matters within sub-arrays)")
print("- outer: combinations (selecting which sub-arrays)")

# [2, 1] from 3 items
show_expansion(
    "size=[2, 1] from 3 items",
    {"_or_": ["A", "B", "C"], "size": [2, 1]}
)

# [2, 2] from 3 items
show_expansion(
    "size=[2, 2] from 3 items",
    {"_or_": ["A", "B", "C"], "size": [2, 2]},
    max_show=10
)


# ============================================================
# Example 8.5: then_pick and then_arrange keywords
# ============================================================
print_section("EXAMPLE 8.5: then_pick and then_arrange keywords", "Phase 2")
print("Second-order selection with explicit then_* keywords.")
print()
print("Keywords:")
print(f"  {THEN_PICK_KEYWORD}    -> then apply COMBINATIONS to primary results")
print(f"  {THEN_ARRANGE_KEYWORD} -> then apply PERMUTATIONS to primary results")
print()

# pick + then_arrange: first pick combinations, then arrange those
print("--- pick + then_arrange: pick first, then arrange results ---")
show_expansion(
    "pick=2, then_arrange=2",
    {"_or_": ["A", "B", "C"], "pick": 2, "then_arrange": 2}
)

# pick + then_pick: first pick combinations, then pick from those
show_expansion(
    "pick=2, then_pick=2",
    {"_or_": ["A", "B", "C"], "pick": 2, "then_pick": 2}
)


# ============================================================
# Example 9: _log_range_ for logarithmic sequences (Phase 3)
# ============================================================
print_section("EXAMPLE 9: _log_range_ for logarithmic sequences", "Phase 3")
print("The _log_range_ keyword generates logarithmically-spaced values.")
print("Useful for hyperparameter search (learning rates, regularization, etc.).")

# Basic log range
show_expansion(
    "Log range [0.001, 1, 4] (4 values from 0.001 to 1)",
    {"_log_range_": [0.001, 1, 4]}
)

# Log range for learning rates
show_expansion(
    "Log range [0.0001, 0.1, 5] (learning rate search)",
    {"_log_range_": [0.0001, 0.1, 5]}
)

# Log range with dict syntax and custom base
show_expansion(
    "Log range dict syntax with base=2",
    {"_log_range_": {"from": 1, "to": 256, "num": 9, "base": 2}}
)


# ============================================================
# Example 10: _grid_ for grid search (Phase 3)
# ============================================================
print_section("EXAMPLE 10: _grid_ for grid search", "Phase 3")
print("The _grid_ keyword generates Cartesian product of parameter spaces.")
print("Similar to sklearn's ParameterGrid.")

# Basic grid
show_expansion(
    "Grid search over 2 parameters",
    {"_grid_": {"learning_rate": [0.01, 0.1], "batch_size": [16, 32, 64]}}
)

# Grid with more parameters
show_expansion(
    "Grid search over 3 parameters",
    {"_grid_": {
        "model": ["PLS", "RF"],
        "n_components": [5, 10],
        "cv_folds": [3, 5]
    }},
    max_show=10
)


# ============================================================
# Example 11: _zip_ for parallel iteration (Phase 3)
# ============================================================
print_section("EXAMPLE 11: _zip_ for parallel iteration", "Phase 3")
print("The _zip_ keyword pairs values at the same index (like Python's zip).")
print("Unlike _grid_ which generates all combinations, _zip_ pairs by position.")

# Basic zip
show_expansion(
    "Zip pairing",
    {"_zip_": {"x": [1, 2, 3], "y": ["A", "B", "C"]}}
)

# Zip comparison with grid
print("--- Comparison: _zip_ vs _grid_ ---")
zip_spec = {"_zip_": {"x": [1, 2], "y": ["A", "B"]}}
grid_spec = {"_grid_": {"x": [1, 2], "y": ["A", "B"]}}
print(f"_zip_ result:  {expand_spec(zip_spec)}")
print(f"_grid_ result: {expand_spec(grid_spec)}")
print("_zip_ pairs: (1,A), (2,B) - position-based")
print("_grid_ products: all 4 combinations")
print()


# ============================================================
# Example 12: _chain_ for sequential choices (Phase 3)
# ============================================================
print_section("EXAMPLE 12: _chain_ for sequential choices", "Phase 3")
print("The _chain_ keyword preserves order (unlike _or_ which may be randomized).")
print("Useful for progressive experiments: baseline -> improved -> best.")

# Basic chain
show_expansion(
    "Chain of configurations in order",
    {"_chain_": [
        {"model": "baseline", "complexity": "low"},
        {"model": "improved", "complexity": "medium"},
        {"model": "best", "complexity": "high"}
    ]}
)


# ============================================================
# Example 13: _sample_ for statistical sampling (Phase 3)
# ============================================================
print_section("EXAMPLE 13: _sample_ for statistical sampling", "Phase 3")
print("The _sample_ keyword generates values from statistical distributions.")
print("Supports: uniform, log_uniform, normal/gaussian, choice")

# Uniform sampling
show_expansion(
    "Uniform sampling [0.1, 1.0]",
    {"_sample_": {"distribution": "uniform", "from": 0.1, "to": 1.0, "num": 5}},
    seed=42
)

# Log-uniform sampling (common for learning rates)
show_expansion(
    "Log-uniform sampling [0.0001, 0.1]",
    {"_sample_": {"distribution": "log_uniform", "from": 0.0001, "to": 0.1, "num": 5}},
    seed=42
)

# Normal distribution
show_expansion(
    "Normal distribution (mean=0, std=1)",
    {"_sample_": {"distribution": "normal", "mean": 0, "std": 1, "num": 5}},
    seed=42
)

# Random choice from values
show_expansion(
    "Random choice from discrete values",
    {"_sample_": {"distribution": "choice", "values": ["A", "B", "C", "D", "E"], "num": 3}},
    seed=42
)


# ============================================================
# Example 14: Constraints - _mutex_, _requires_, _exclude_ (Phase 4)
# ============================================================
print_section("EXAMPLE 14: Constraints", "Phase 4")
print("Constraint keywords filter invalid combinations:")
print("  _mutex_: Items that cannot appear together")
print("  _requires_: If A selected, B must also be selected")
print("  _exclude_: Specific combinations to exclude")

# Setup: show all combinations first
choices = ["A", "B", "C", "D"]
all_combos = expand_spec({"_or_": choices, "pick": 2})
print(f"All combinations of 2 from {choices}: {all_combos}")
print()

# Mutual exclusion
print("--- _mutex_ constraint ---")
print("A and B cannot appear together:")
mutex_combos = expand_spec({"_or_": choices, "pick": 2, "_mutex_": [["A", "B"]]})
print(f"After _mutex_: {mutex_combos}")
print()

# Requires constraint
print("--- _requires_ constraint ---")
print("If A is selected, C must also be selected:")
requires_combos = expand_spec({"_or_": choices, "pick": 2, "_requires_": [["A", "C"]]})
print(f"After _requires_: {requires_combos}")
print()

# Exclude specific combinations
print("--- _exclude_ constraint ---")
print("Exclude specific combinations [A, C] and [B, D]:")
exclude_combos = expand_spec({
    "_or_": choices,
    "pick": 2,
    "_exclude_": [["A", "C"], ["B", "D"]]
})
print(f"After _exclude_: {exclude_combos}")
print()


# ============================================================
# Example 15: Presets - Named Configuration Templates (Phase 4)
# ============================================================
print_section("EXAMPLE 15: Presets - Named Configuration Templates", "Phase 4")
print("Presets allow defining reusable configuration patterns.")
print("Reference with _preset_ keyword.")

# Clear any existing presets
clear_presets()

# Register some presets
register_preset(
    "spectral_transforms",
    {"_or_": ["SNV", "MSC", "Detrend", "SavGol"], "pick": (1, 2)},
    description="Common spectral preprocessing transforms"
)

register_preset(
    "pls_components",
    {"_range_": [2, 15]},
    description="Range of PLS n_components"
)

register_preset(
    "regularization",
    {"_log_range_": [0.0001, 1, 5]},
    description="Log-spaced regularization values"
)

print("Registered presets:", list_presets())
print()

# Use presets in config
config_with_presets = {
    "transforms": {"_preset_": "spectral_transforms"},
    "model": {
        "class": "PLSRegression",
        "n_components": {"_preset_": "pls_components"}
    }
}

print("Config with presets:")
print(json.dumps(config_with_presets, indent=2))
print()

# Resolve presets
resolved = resolve_presets_recursive(config_with_presets)
print("After resolving presets:")
print(json.dumps(resolved, indent=2, default=str))
print()

# Expand resolved config
print(f"Total combinations: {count_combinations(resolved)}")
print()


# ============================================================
# Example 16: Iterator API - Memory-Efficient Expansion (Phase 4)
# ============================================================
print_section("EXAMPLE 16: Iterator API - Memory-Efficient Expansion", "Phase 4")
print("expand_spec_iter() generates configurations lazily.")
print("Essential for large configuration spaces that won't fit in memory.")

# Example with large range
large_spec = {"_range_": [1, 1000]}
print(f"Spec: {large_spec}")
print(f"Total: {count_combinations(large_spec)} configurations")

# Use iterator to get first 10
from itertools import islice
first_10 = list(islice(expand_spec_iter(large_spec), 10))
print(f"First 10 (lazy): {first_10}")
print()

# Batch processing
print("--- Batch Processing ---")
batch_spec = {"_range_": [1, 25]}
print(f"Processing {count_combinations(batch_spec)} items in batches of 5:")
for i, batch in enumerate(batch_iter(batch_spec, batch_size=5)):
    print(f"  Batch {i}: {batch}")
print()


# ============================================================
# Example 17: Export Utilities (Phase 4)
# ============================================================
print_section("EXAMPLE 17: Export Utilities", "Phase 4")
print("Utilities for inspecting and exporting configurations.")

# Sample configs
configs = expand_spec({
    "_grid_": {
        "model": ["PLS", "RF"],
        "n_components": [5, 10, 15]
    }
})

# to_dataframe (if pandas available)
try:
    df = to_dataframe(configs)
    print("--- to_dataframe() ---")
    print(df)
    print()
except ImportError:
    print("(pandas not available for to_dataframe demo)")
    print()

# summarize_configs
print("--- summarize_configs() ---")
summary = summarize_configs(configs)
print(f"Count: {summary['count']}")
for key, info in summary['keys'].items():
    print(f"  {key}: {info['unique_count']} unique values: {info['unique_values']}")
print()

# diff_configs
print("--- diff_configs() ---")
config1 = configs[0]
config2 = configs[3]
print(f"Config 1: {config1}")
print(f"Config 2: {config2}")
diff = diff_configs(config1, config2)
print(f"Differences: {diff}")
print()

# print_expansion_tree
print("--- print_expansion_tree() ---")
tree_spec = {
    "preprocessing": {"_or_": ["StandardScaler", "MinMaxScaler"]},
    "model": {
        "_grid_": {
            "class": ["PLS", "RF"],
            "components": {"_range_": [2, 5]}
        }
    }
}
print(f"Spec: {json.dumps(tree_spec, indent=2)}")
print()
print("Expansion Tree:")
print(print_expansion_tree(tree_spec))
print()


# ============================================================
# Example 18: Pipeline-like structures
# ============================================================
print_section("EXAMPLE 18: Pipeline-like structures", "Comprehensive")
print("Demonstrating generator syntax in realistic pipeline configurations.")

# Feature augmentation with multiple transformers
pipeline_spec = {
    "preprocessing": {
        "_or_": [
            {"class": "StandardScaler"},
            {"class": "MinMaxScaler"},
        ]
    },
    "feature_extraction": {
        "_or_": [
            {"class": "PCA", "n_components": {"_or_": [10, 20, 30]}},
            {"class": "TruncatedSVD", "n_components": {"_or_": [10, 20]}},
        ]
    }
}

show_expansion("Pipeline with nested choices", pipeline_spec, max_show=10)

# Concat transform pool
concat_pool = {
    "_or_": [
        {"class": "PCA", "n_components": 10},
        {"class": "TruncatedSVD", "n_components": 10},
        {"class": "FastICA", "n_components": 10},
    ],
    "size": (1, 2)
}

show_expansion(
    "Concat pool: single or pairs of transformers",
    concat_pool
)


# ============================================================
# Example 19: Utility functions
# ============================================================
print_section("EXAMPLE 19: Utility functions", "Utilities")
print("Testing the utility functions from the generator module.")

# is_generator_node
print("--- is_generator_node() ---")
test_cases = [
    {"_or_": ["A", "B"]},
    {"_range_": [1, 10]},
    {"_log_range_": [0.01, 1, 5]},
    {"_grid_": {"x": [1, 2]}},
    {"_zip_": {"x": [1], "y": [2]}},
    {"_chain_": [1, 2, 3]},
    {"_sample_": {"distribution": "uniform", "num": 5}},
    {"class": "PCA"},
    {"n_components": 10},
]
for tc in test_cases:
    first_key = list(tc.keys())[0]
    print(f"  {first_key:15} -> {is_generator_node(tc)}")
print()

# sample_with_seed - deterministic sampling
print("--- sample_with_seed() with seed (deterministic) ---")
items = ["A", "B", "C", "D", "E", "F"]
print(f"Items: {items}")
for seed in [42, 42, 99]:
    result = sample_with_seed(items, 3, seed=seed)
    print(f"  seed={seed}: {result}")
print()


# ============================================================
# Summary
# ============================================================
print_section("SUMMARY")
print("Generator syntax allows creating multiple configurations from one spec:")
print()
print("Phase 1-2 (Core):")
print("  _or_: ['A', 'B', 'C']           -> ['A', 'B', 'C'] (3 variants)")
print("  _or_: [...], size=2             -> combinations of 2 (legacy)")
print("  _or_: [...], pick=2             -> combinations of 2 (explicit)")
print("  _or_: [...], arrange=2          -> permutations of 2 (explicit)")
print("  _range_: [1, 10]                -> [1, 2, ..., 10]")
print("  count=N                         -> limit to N samples")
print()
print("Phase 3 (Advanced):")
print("  _log_range_: [0.001, 1, 4]      -> [0.001, 0.01, 0.1, 1.0]")
print("  _grid_: {x: [1,2], y: [3,4]}    -> 4 Cartesian product combos")
print("  _zip_: {x: [1,2], y: [3,4]}     -> 2 paired combos [(1,3), (2,4)]")
print("  _chain_: [cfg1, cfg2]           -> Sequential ordered configs")
print("  _sample_: {distribution: ...}   -> Statistical sampling")
print()
print("Phase 4 (Production):")
print("  _mutex_: [['A','B']]            -> A and B cannot appear together")
print("  _requires_: [['A','B']]         -> If A selected, B must also be")
print("  _exclude_: [['A','C']]          -> Exclude specific combinations")
print("  _preset_: 'name'                -> Reference named preset")
print("  expand_spec_iter(spec)          -> Memory-efficient lazy iteration")
print()
print("Selection semantics:")
print("  pick    -> combinations C(n,k)  -> order doesn't matter")
print("  arrange -> permutations P(n,k)  -> order matters")
print("  then_pick    -> apply combinations to primary results")
print("  then_arrange -> apply permutations to primary results")
print()
print("Export utilities:")
print("  to_dataframe(configs)           -> Convert to pandas DataFrame")
print("  diff_configs(c1, c2)            -> Show differences")
print("  summarize_configs(configs)      -> Summary statistics")
print("  print_expansion_tree(spec)      -> Tree visualization")
print()
print("Use count_combinations() to get count without generating.")
print("Use sample_with_seed() for deterministic random sampling.")
print()
print("Example completed successfully!")
