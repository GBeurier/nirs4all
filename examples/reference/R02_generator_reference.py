"""
R02 - Generator Syntax Reference
================================

Complete reference for nirs4all pipeline generator syntax.

Generators create multiple pipeline variants from a single specification.
This is the core mechanism for hyperparameter sweeps, preprocessing exploration,
and automated experiment generation.

This reference covers:

* Core generators: _or_, _range_, pick, arrange, count
* Advanced generators: _log_range_, _grid_, _zip_, _chain_, _sample_
* Constraints: _mutex_, _requires_, _exclude_
* Presets: _preset_ for reusable templates
* Utilities: expand_spec, count_combinations, export tools

For pipeline syntax, see :ref:`R01_pipeline_syntax`.
For a runnable all-keywords test, see :ref:`R03_all_keywords`.

Duration: ~3 minutes (no models trained)
Difficulty: Reference material
"""

# Standard library imports
import argparse
import json

# Generator API imports
from nirs4all.pipeline.config.generator import (
    ARRANGE_KEYWORD,
    # Keywords
    PICK_KEYWORD,
    THEN_ARRANGE_KEYWORD,
    THEN_PICK_KEYWORD,
    batch_iter,
    clear_presets,
    count_combinations,
    diff_configs,
    # Core expansion functions
    expand_spec,
    # Iterator API for large spaces
    expand_spec_iter,
    # Detection
    is_generator_node,
    list_presets,
    print_expansion_tree,
    # Preset functions
    register_preset,
    resolve_presets_recursive,
    # Utilities
    sample_with_seed,
    summarize_configs,
    # Export utilities
    to_dataframe,
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='R02 Generator Syntax Reference')
parser.add_argument('--plots', action='store_true', help='(unused)')
parser.add_argument('--show', action='store_true', help='(unused)')
args = parser.parse_args()

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

def show_expansion(name: str, spec, max_show: int = 10, seed: int = None):
    """Show expansion results for a specification."""
    print(f"--- {name} ---")
    print(f"Spec: {json.dumps(spec, indent=2, default=str)}")
    print()

    count = count_combinations(spec)
    print(f"Combinations (computed without expanding): {count}")

    results = expand_spec(spec, seed=seed)
    print(f"Actual expanded count: {len(results)}")

    if len(results) <= max_show:
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

# =============================================================================
# PHASE 1: Core Keywords
# =============================================================================
print_section("PHASE 1: Core Generator Keywords")

# -----------------------------------------------------------------------------
# 1.1 _or_ - Choose between alternatives
# -----------------------------------------------------------------------------
print("1.1 _or_ - Basic alternative selection")
print("-" * 40)
print("Creates one variant per choice in the list.")
print()

show_expansion(
    "String choices",
    {"_or_": ["StandardScaler", "MinMaxScaler", "RobustScaler"]}
)

show_expansion(
    "Dictionary choices",
    {"_or_": [
        {"class": "PCA", "n_components": 10},
        {"class": "SVD", "n_components": 10},
    ]}
)

# -----------------------------------------------------------------------------
# 1.2 _range_ - Numeric sequences
# -----------------------------------------------------------------------------
print("1.2 _range_ - Generate numeric sequences")
print("-" * 40)
print("Syntax: [start, end] or [start, end, step]")
print("Range is INCLUSIVE on both ends.")
print()

show_expansion("Range [1, 5]", {"_range_": [1, 5]})
show_expansion("Range with step [0, 20, 5]", {"_range_": [0, 20, 5]})

# Dict syntax
show_expansion(
    "Range dict syntax",
    {"_range_": {"from": 10, "to": 50, "step": 10}}
)

# -----------------------------------------------------------------------------
# 1.3 count - Limit results
# -----------------------------------------------------------------------------
print("1.3 count - Limit number of results")
print("-" * 40)
print("Randomly samples N results (use seed for reproducibility).")
print()

spec_count = {"_or_": ["A", "B", "C", "D", "E"], "count": 3}
print(f"Spec: {spec_count}")
for i in range(3):
    results = expand_spec(spec_count, seed=42)
    print(f"  Run {i+1} (seed=42): {results}")
print()

# =============================================================================
# PHASE 2: Selection Keywords
# =============================================================================
print_section("PHASE 2: Selection Keywords (pick vs arrange)")

# -----------------------------------------------------------------------------
# 2.1 pick - Combinations (order doesn't matter)
# -----------------------------------------------------------------------------
print("2.1 pick - Combinations (unordered selection)")
print("-" * 40)
print("C(n,k) = n! / (k! * (n-k)!)")
print("pick=2 from [A,B,C] gives: [A,B], [A,C], [B,C]  (3 combinations)")
print()

show_expansion("pick=2 from 3", {"_or_": ["A", "B", "C"], "pick": 2})
show_expansion("pick=2 from 4", {"_or_": ["A", "B", "C", "D"], "pick": 2})

# Range of picks
show_expansion(
    "pick=(1,2) - pick 1 OR 2",
    {"_or_": ["A", "B", "C"], "pick": (1, 2)}
)

# -----------------------------------------------------------------------------
# 2.2 arrange - Permutations (order matters)
# -----------------------------------------------------------------------------
print("2.2 arrange - Permutations (ordered selection)")
print("-" * 40)
print("P(n,k) = n! / (n-k)!")
print("arrange=2 from [A,B,C] gives: [A,B], [B,A], [A,C], [C,A], [B,C], [C,B]")
print()

show_expansion("arrange=2 from 3", {"_or_": ["A", "B", "C"], "arrange": 2})

# -----------------------------------------------------------------------------
# 2.3 Comparison: pick vs arrange
# -----------------------------------------------------------------------------
print("2.3 Comparison: pick vs arrange")
print("-" * 40)
print()

choices = ["A", "B", "C"]
pick_results = expand_spec({"_or_": choices, "pick": 2})
arrange_results = expand_spec({"_or_": choices, "arrange": 2})

print(f"From {choices}:")
print(f"  pick=2 ({len(pick_results)} results):    {pick_results}")
print(f"  arrange=2 ({len(arrange_results)} results): {arrange_results}")
print()
print("Use pick when order doesn't matter (feature_augmentation, concat_transform)")
print("Use arrange when order matters (sequential preprocessing steps)")
print()

# =============================================================================
# PHASE 3: Advanced Generators
# =============================================================================
print_section("PHASE 3: Advanced Generator Keywords")

# -----------------------------------------------------------------------------
# 3.1 _log_range_ - Logarithmic sequences
# -----------------------------------------------------------------------------
print("3.1 _log_range_ - Logarithmic sequences")
print("-" * 40)
print("Useful for learning rates, regularization, etc.")
print()

show_expansion(
    "Log range [0.001, 1, 4]",
    {"_log_range_": [0.001, 1, 4]}
)

show_expansion(
    "Log range with custom base",
    {"_log_range_": {"from": 1, "to": 256, "num": 9, "base": 2}}
)

# -----------------------------------------------------------------------------
# 3.2 _grid_ - Cartesian product
# -----------------------------------------------------------------------------
print("3.2 _grid_ - Cartesian product (grid search)")
print("-" * 40)
print("All combinations of all parameters.")
print()

show_expansion(
    "Grid over 2 parameters",
    {"_grid_": {"learning_rate": [0.01, 0.1], "batch_size": [16, 32, 64]}}
)

# -----------------------------------------------------------------------------
# 3.3 _zip_ - Parallel iteration
# -----------------------------------------------------------------------------
print("3.3 _zip_ - Parallel iteration")
print("-" * 40)
print("Pairs values at same index (like Python's zip).")
print()

show_expansion(
    "Zip pairing",
    {"_zip_": {"x": [1, 2, 3], "y": ["A", "B", "C"]}}
)

print("Comparison: _zip_ vs _grid_")
zip_spec = {"_zip_": {"x": [1, 2], "y": ["A", "B"]}}
grid_spec = {"_grid_": {"x": [1, 2], "y": ["A", "B"]}}
print(f"_zip_:  {expand_spec(zip_spec)}")
print(f"_grid_: {expand_spec(grid_spec)}")
print()

# -----------------------------------------------------------------------------
# 3.4 _chain_ - Sequential ordering
# -----------------------------------------------------------------------------
print("3.4 _chain_ - Sequential choices (ordered)")
print("-" * 40)
print("Unlike _or_ which may randomize, _chain_ preserves order.")
print()

show_expansion(
    "Chain of configurations",
    {"_chain_": [
        {"model": "baseline"},
        {"model": "improved"},
        {"model": "best"}
    ]}
)

# -----------------------------------------------------------------------------
# 3.5 _sample_ - Statistical sampling
# -----------------------------------------------------------------------------
print("3.5 _sample_ - Statistical sampling")
print("-" * 40)
print("Generate values from distributions: uniform, log_uniform, normal, choice")
print()

show_expansion(
    "Uniform sampling",
    {"_sample_": {"distribution": "uniform", "from": 0.1, "to": 1.0, "num": 5}},
    seed=42
)

show_expansion(
    "Log-uniform (for learning rates)",
    {"_sample_": {"distribution": "log_uniform", "from": 0.0001, "to": 0.1, "num": 5}},
    seed=42
)

show_expansion(
    "Normal distribution",
    {"_sample_": {"distribution": "normal", "mean": 0, "std": 1, "num": 5}},
    seed=42
)

# =============================================================================
# PHASE 4: Constraints and Presets
# =============================================================================
print_section("PHASE 4: Constraints and Presets")

# -----------------------------------------------------------------------------
# 4.1 Constraints: _mutex_, _requires_, _exclude_
# -----------------------------------------------------------------------------
print("4.1 Constraints")
print("-" * 40)
print()

choices = ["A", "B", "C", "D"]
all_combos = expand_spec({"_or_": choices, "pick": 2})
print(f"All combinations of 2 from {choices}: {all_combos}")
print()

# Mutual exclusion
mutex_combos = expand_spec({"_or_": choices, "pick": 2, "_mutex_": [["A", "B"]]})
print(f"With _mutex_ [A,B] (can't appear together): {mutex_combos}")

# Requires
requires_combos = expand_spec({"_or_": choices, "pick": 2, "_requires_": [["A", "C"]]})
print(f"With _requires_ [A,C] (if A then C): {requires_combos}")

# Exclude
exclude_combos = expand_spec({"_or_": choices, "pick": 2, "_exclude_": [["A", "C"]]})
print(f"With _exclude_ [A,C]: {exclude_combos}")
print()

# -----------------------------------------------------------------------------
# 4.2 Presets: Named templates
# -----------------------------------------------------------------------------
print("4.2 Presets - Reusable templates")
print("-" * 40)
print()

clear_presets()

register_preset(
    "spectral_transforms",
    {"_or_": ["SNV", "MSC", "Detrend"], "pick": (1, 2)},
    description="Common spectral preprocessing"
)

register_preset(
    "pls_components",
    {"_range_": [2, 15]},
    description="Range of PLS components"
)

print(f"Registered presets: {list_presets()}")
print()

config = {
    "transforms": {"_preset_": "spectral_transforms"},
    "n_components": {"_preset_": "pls_components"}
}
resolved = resolve_presets_recursive(config)
print(f"Config with presets: {json.dumps(config, indent=2)}")
print(f"Resolved: {json.dumps(resolved, indent=2, default=str)}")
print(f"Total combinations: {count_combinations(resolved)}")
print()

# =============================================================================
# UTILITIES
# =============================================================================
print_section("Utility Functions")

# Iterator for large spaces
print("Iterator API for large spaces")
print("-" * 40)
large_spec = {"_range_": [1, 1000]}
print(f"Spec: {large_spec} ({count_combinations(large_spec)} items)")
print()

from itertools import islice

first_10 = list(islice(expand_spec_iter(large_spec), 10))
print(f"First 10 (lazy iteration): {first_10}")
print()

# Batch processing
print("Batch processing")
print("-" * 40)
for i, batch in enumerate(batch_iter({"_range_": [1, 15]}, batch_size=5)):
    print(f"Batch {i}: {batch}")
print()

# Export utilities
print("Export utilities")
print("-" * 40)
configs = expand_spec({"_grid_": {"model": ["PLS", "RF"], "n": [5, 10]}})
print(f"Configs: {configs}")
print()

summary = summarize_configs(configs)
print(f"Summary: {summary['count']} configs")
for key, info in summary['keys'].items():
    print(f"  {key}: {info['unique_values']}")
print()

print("diff_configs():")
print(f"  Config 0: {configs[0]}")
print(f"  Config 2: {configs[2]}")
print(f"  Diff: {diff_configs(configs[0], configs[2])}")
print()

# =============================================================================
# SUMMARY
# =============================================================================
print_section("SUMMARY")

print("""
R02 - Generator Syntax Reference
================================

CORE GENERATORS (Phase 1):
  _or_: [A, B, C]               -> 3 variants (one per choice)
  _range_: [1, 10]              -> [1, 2, 3, ..., 10]
  _range_: [1, 10, 2]           -> [1, 3, 5, 7, 9]
  count: N                      -> Limit to N random samples

SELECTION (Phase 2):
  pick: 2                       -> Combinations C(n,k) - order doesn't matter
  arrange: 2                    -> Permutations P(n,k) - order matters
  pick: (1, 3)                  -> Pick 1 OR 2 OR 3

ADVANCED (Phase 3):
  _log_range_: [0.001, 1, 4]    -> Logarithmic spacing
  _grid_: {x: [...], y: [...]}  -> Cartesian product
  _zip_: {x: [...], y: [...]}   -> Parallel pairing
  _chain_: [cfg1, cfg2]         -> Sequential (ordered)
  _sample_: {distribution: ...} -> Statistical sampling

CONSTRAINTS (Phase 4):
  _mutex_: [[A, B]]             -> A and B cannot appear together
  _requires_: [[A, B]]          -> If A selected, B must be too
  _exclude_: [[A, C]]           -> Exclude specific combination

PRESETS (Phase 4):
  register_preset("name", spec) -> Define reusable template
  _preset_: "name"              -> Reference preset

UTILITIES:
  expand_spec(spec)             -> Get all expansions
  count_combinations(spec)      -> Count without expanding
  expand_spec_iter(spec)        -> Lazy iterator
  batch_iter(spec, batch_size)  -> Batch processing

PIPELINE USAGE:
  # In feature_augmentation
  {"feature_augmentation": {"_or_": [SNV, MSC], "pick": 2, "count": 5}}

  # Model sweep
  {"_range_": [2, 20, 2], "param": "n_components", "model": PLSRegression}

  # Alternative models
  {"_or_": [{"model": PLS}, {"model": RF}]}
""")

print("Reference complete!")
