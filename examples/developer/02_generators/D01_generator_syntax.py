"""
D01 - Generator Syntax: Dynamic Pipeline Generation
====================================================

Generators enable creating multiple pipeline variants from a single
template using special keywords like ``_or_``, ``_range_``, and ``_grid_``.

This tutorial covers:

* Basic ``_or_`` for alternatives
* ``_range_`` for numeric sweeps
* ``pick`` and ``arrange`` for combinations
* Generator expansion and counting

Prerequisites
-------------
- 01_quickstart/U02_basic_regression for pipeline basics

Next Steps
----------
See D02_generator_advanced for constraints and presets.

Duration: ~5 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse
import json
from typing import Any

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import Detrend, FirstDerivative, SecondDerivative
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.pipeline.config.generator import (
    count_combinations,
    expand_spec,
    is_generator_node,
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D01 Generator Syntax Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D01 - Generator Syntax: Dynamic Pipeline Generation")
print("=" * 60)

print("""
Instead of writing multiple pipelines manually:

    pipeline_1 = [SNV(), PLSRegression(n_components=5)]
    pipeline_2 = [MSC(), PLSRegression(n_components=5)]
    ...

Use generators to express all variants compactly:

    pipeline = [
        {"_or_": ["SNV", "MSC"]},
        {"_range_": [5, 15, 5]}
    ]

This expands to 2 × 3 = 6 pipeline variants automatically!
""")

# =============================================================================
# Section 1: Basic _or_ for Alternatives
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Basic _or_ for Alternatives")
print("-" * 60)

print("""
_or_ creates one variant per option:

    {"_or_": ["A", "B", "C"]}  →  3 variants: "A", "B", "C"
""")

# Simple string choices
spec_or = {"_or_": ["StandardScaler", "MinMaxScaler", "RobustScaler"]}
print(f"Spec: {spec_or}")
print(f"Count: {count_combinations(spec_or)}")
results = expand_spec(spec_or)
print(f"Expanded: {results}")

# =============================================================================
# Section 2: _range_ for Numeric Sweeps
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: _range_ for Numeric Sweeps")
print("-" * 60)

print("""
_range_ sweeps through numeric values:

    {"_range_": [start, stop]}             # default step=1
    {"_range_": [start, stop, step]}       # custom step
""")

# Basic range
spec_range = {"_range_": [2, 10, 2]}
print(f"Spec: {spec_range}")
print(f"Count: {count_combinations(spec_range)}")
results = expand_spec(spec_range)
print(f"Expanded: {results}")

# Range in dict syntax
spec_range_dict = {"_range_": {"from": 5, "to": 20, "step": 5}}
print(f"\nDict syntax: {spec_range_dict}")
results = expand_spec(spec_range_dict)
print(f"Expanded: {results}")

# =============================================================================
# Section 3: Nested Generators (Cartesian Product)
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Nested Generators (Cartesian Product)")
print("-" * 60)

print("""
Multiple generator keys in a dict -> Cartesian product:

    {"x": {"_or_": [1, 2]}, "y": {"_or_": [A, B]}}
    → 4 combinations: (1,A), (1,B), (2,A), (2,B)
""")

spec_nested = {
    "n_components": {"_or_": [5, 10]},
    "random_state": {"_or_": [0, 42]}
}
print(f"Spec: {json.dumps(spec_nested, indent=2)}")
print(f"Count: {count_combinations(spec_nested)}")
results = expand_spec(spec_nested)
for i, r in enumerate(results):
    print(f"  [{i}]: {r}")

# =============================================================================
# Section 4: pick - Select k Items (Combinations)
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: pick - Select k Items (Combinations)")
print("-" * 60)

print("""
pick selects k items (order doesn't matter):

    {"_or_": ["A", "B", "C", "D"], "pick": 2}

Generates C(4,2) = 6 combinations: [A,B], [A,C], [A,D], [B,C], [B,D], [C,D]
""")

spec_pick = {"_or_": ["A", "B", "C", "D"], "pick": 2}
print(f"Spec: {spec_pick}")
print(f"Count: {count_combinations(spec_pick)}")
results = expand_spec(spec_pick)
for i, r in enumerate(results):
    print(f"  [{i}]: {r}")

# =============================================================================
# Section 5: arrange - Select k Items (Permutations)
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: arrange - Select k Items (Permutations)")
print("-" * 60)

print("""
arrange selects k items where order matters (permutations):

    {"_or_": ["A", "B", "C"], "arrange": 2}

Generates P(3,2) = 6 permutations: [A,B], [A,C], [B,A], [B,C], [C,A], [C,B]
""")

spec_arrange = {"_or_": ["A", "B", "C"], "arrange": 2}
print(f"Spec: {spec_arrange}")
print(f"Count: {count_combinations(spec_arrange)}")
results = expand_spec(spec_arrange)
for i, r in enumerate(results):
    print(f"  [{i}]: {r}")

# =============================================================================
# Section 6: count - Limit Expansion
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: count - Limit Expansion")
print("-" * 60)

print("""
count limits the number of generated variants:

    {"_or_": ["A", "B", "C", "D", "E"], "count": 3}

Generates only 3 randomly selected variants from 5 options.
Use seed for deterministic results.
""")

spec_count = {"_or_": ["A", "B", "C", "D", "E"], "count": 3}
print(f"Spec: {spec_count}")
print(f"Count (before limit): 5, Count (after limit): {count_combinations(spec_count)}")
results = expand_spec(spec_count, seed=42)
print(f"Expanded (seed=42): {results}")

# =============================================================================
# Section 7: _log_range_ for Logarithmic Sweeps
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: _log_range_ for Logarithmic Sweeps")
print("-" * 60)

print("""
_log_range_ sweeps logarithmically (useful for regularization):

    {"_log_range_": [0.001, 1, 4]}

Generates: [0.001, 0.01, 0.1, 1.0] (log-spaced, 4 values)
""")

spec_log = {"_log_range_": [0.001, 1, 4]}
print(f"Spec: {spec_log}")
print(f"Count: {count_combinations(spec_log)}")
results = expand_spec(spec_log)
print(f"Expanded: {results}")

# =============================================================================
# Section 8: _grid_ for Grid Search
# =============================================================================
print("\n" + "-" * 60)
print("Example 8: _grid_ for Grid Search")
print("-" * 60)

print("""
_grid_ generates Cartesian product (like sklearn ParameterGrid):

    {"_grid_": {"learning_rate": [0.01, 0.1], "batch_size": [16, 32]}}
""")

spec_grid = {
    "_grid_": {
        "model": ["PLS", "RF"],
        "n_components": [5, 10]
    }
}
print(f"Spec: {json.dumps(spec_grid, indent=2)}")
print(f"Count: {count_combinations(spec_grid)}")
results = expand_spec(spec_grid)
for i, r in enumerate(results):
    print(f"  [{i}]: {r}")

# =============================================================================
# Section 9: _zip_ for Parallel Iteration
# =============================================================================
print("\n" + "-" * 60)
print("Example 9: _zip_ for Parallel Iteration")
print("-" * 60)

print("""
_zip_ pairs values at the same index (like Python's zip):

    {"_zip_": {"x": [1, 2, 3], "y": ["A", "B", "C"]}}
    → 3 pairs: (1,A), (2,B), (3,C)

Unlike _grid_ which generates all combinations!
""")

spec_zip = {"_zip_": {"x": [1, 2, 3], "y": ["A", "B", "C"]}}
print(f"Spec: {spec_zip}")
results = expand_spec(spec_zip)
print(f"Expanded: {results}")

print("\n--- Comparison: _zip_ vs _grid_ ---")
zip_result = expand_spec({"_zip_": {"x": [1, 2], "y": ["A", "B"]}})
grid_result = expand_spec({"_grid_": {"x": [1, 2], "y": ["A", "B"]}})
print(f"_zip_ (2 pairs):  {zip_result}")
print(f"_grid_ (4 combos): {grid_result}")

# =============================================================================
# Section 10: Running Pipelines with Generator Syntax
# =============================================================================
print("\n" + "-" * 60)
print("Example 10: Running Pipelines with Generator Syntax")
print("-" * 60)

print("""
Generator syntax integrates with nirs4all.run():
The pipeline expands automatically during execution.
""")

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"_or_": [SNV(), MSC()]},  # 2 preprocessing variants
    PLSRegression(n_components=5),
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="GeneratorDemo",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nTotal predictions: {result.num_predictions}")

# =============================================================================
# Section 11: Utility Function - is_generator_node
# =============================================================================
print("\n" + "-" * 60)
print("Example 11: Utility - is_generator_node")
print("-" * 60)

print("""
is_generator_node() detects if a dict contains generator syntax:
""")

test_cases: list[dict[str, Any]] = [
    {"_or_": ["A", "B"]},
    {"_range_": [1, 10]},
    {"_log_range_": [0.01, 1, 5]},
    {"_grid_": {"x": [1, 2]}},
    {"_zip_": {"x": [1], "y": [2]}},
    {"class": "PCA"},  # Not a generator
    {"n_components": 10},  # Not a generator
]

for tc in test_cases:
    first_key = list(tc.keys())[0]
    is_gen = is_generator_node(tc)
    print(f"  {first_key:15} -> is_generator_node: {is_gen}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. _or_ creates alternatives (one variant per option)
2. _range_ sweeps numeric parameters
3. _log_range_ for logarithmic sweeps
4. pick selects k items (combinations)
5. arrange selects k items (permutations)
6. count limits total variants
7. _grid_ for Cartesian product
8. _zip_ for parallel pairing

Key functions:
- expand_spec(spec) - Expand generator to list of configs
- count_combinations(spec) - Count without generating
- is_generator_node(node) - Check if node is a generator

Key formulas:
- _or_: n variants for n options
- pick(n,k): C(n,k) = n!/(k!(n-k)!) combinations
- arrange(n,k): P(n,k) = n!/(n-k)! permutations
- Multiple generators: multiply counts

Next: D02_generator_advanced.py - Constraints, presets, and advanced patterns
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
