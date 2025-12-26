"""
D07 - Generator Advanced: Constraints, Presets, and Grid Search
================================================================

Advanced generator features for complex hyperparameter optimization
and constrained search spaces.

This tutorial covers:

* Constraints: ``_mutex_``, ``_requires_``, ``_exclude_``
* Presets for reusable configurations
* ``_grid_`` for full grid search
* ``_zip_`` for paired parameters
* ``_chain_`` for sequential generators
* ``_sample_`` for random sampling

Prerequisites
-------------
- D06_generator_syntax for basic generators

Next Steps
----------
See D08_generator_iterators for programmatic access.

Duration: ~6 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse
import json

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    MultiplicativeScatterCorrection as MSC,
    FirstDerivative,
)
from nirs4all.pipeline.config.generator import (
    expand_spec,
    count_combinations,
    register_preset,
    list_presets,
    clear_presets,
    resolve_presets_recursive,
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D07 Generator Advanced Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D07 - Generator Advanced: Constraints and Grid Search")
print("=" * 60)

print("""
Basic generators can explode combinatorially. Advanced features help:

    Constraints:  Prune invalid combinations
    Presets:      Reusable named configurations
    Grid:         Full Cartesian product
    Zip:          Paired parameter sweeps
    Sample:       Random subset of search space
""")


# =============================================================================
# Section 1: _mutex_ - Mutually Exclusive Options
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: _mutex_ - Mutually Exclusive Options")
print("-" * 60)

print("""
_mutex_ prevents certain combinations from appearing together:

    {"_or_": ["A", "B", "C", "D"], "_mutex_": [["A", "B"]], "pick": 2}

A and B cannot both appear together.
""")

spec_mutex = {
    "_or_": ["A", "B", "C", "D"],
    "_mutex_": [["A", "B"]],  # A and B cannot both appear
    "pick": 2
}

print(f"Spec: {json.dumps(spec_mutex, indent=2)}")
print(f"Without _mutex_: C(4,2) = 6 combinations")
results = expand_spec(spec_mutex)
print(f"With _mutex_: {len(results)} combinations")
for i, r in enumerate(results):
    print(f"  [{i}]: {r}")


# =============================================================================
# Section 2: _requires_ - Dependency Constraints
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: _requires_ - Dependency Constraints")
print("-" * 60)

print("""
_requires_ enforces that one option requires another:

    {"_or_": ["A", "B", "C"], "_requires_": [["A", "B"]], "pick": 2}

If A is selected, B must also be selected.
""")

spec_requires = {
    "_or_": ["A", "B", "C", "D"],
    "_requires_": [["A", "C"]],  # If A selected, C must also be selected
    "pick": 2
}

print(f"Spec: {json.dumps(spec_requires, indent=2)}")
results = expand_spec(spec_requires)
print(f"With _requires_: {len(results)} combinations")
for i, r in enumerate(results):
    print(f"  [{i}]: {r}")


# =============================================================================
# Section 3: _exclude_ - Explicit Exclusion
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: _exclude_ - Explicit Exclusion")
print("-" * 60)

print("""
_exclude_ removes specific combinations:

    {"_or_": ["A", "B", "C"], "_exclude_": [["A", "C"]], "pick": 2}

The combination ["A", "C"] will never be generated.
""")

spec_exclude = {
    "_or_": ["A", "B", "C", "D"],
    "_exclude_": [["A", "C"], ["B", "D"]],
    "pick": 2
}

print(f"Spec: {json.dumps(spec_exclude, indent=2)}")
results = expand_spec(spec_exclude)
print(f"With _exclude_: {len(results)} combinations")
for i, r in enumerate(results):
    print(f"  [{i}]: {r}")


# =============================================================================
# Section 4: _grid_ - Full Grid Search
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: _grid_ - Full Grid Search")
print("-" * 60)

print("""
_grid_ creates the Cartesian product of multiple parameters:

    {"_grid_": {
        "n_components": [3, 5, 10],
        "scale": [True, False]
    }}

Generates 3 × 2 = 6 combinations.
""")

spec_grid = {
    "_grid_": {
        "n_components": [3, 5, 7, 10],
        "scale": [True, False],
    }
}

print(f"Spec: {json.dumps(spec_grid, indent=2)}")
print(f"Count: {count_combinations(spec_grid)}")
results = expand_spec(spec_grid)
print(f"Expanded ({len(results)} items):")
for i, r in enumerate(results[:6]):
    print(f"  [{i}]: {r}")
if len(results) > 6:
    print(f"  ... and {len(results) - 6} more")


# =============================================================================
# Section 5: _zip_ - Paired Parameters
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: _zip_ - Paired Parameters")
print("-" * 60)

print("""
_zip_ pairs parameters together (like Python's zip):

    {"_zip_": {
        "alpha": [0.1, 1.0, 10.0],
        "l1_ratio": [0.2, 0.5, 0.8]
    }}

Generates 3 pairs: (0.1, 0.2), (1.0, 0.5), (10.0, 0.8)
NOT the 9 combinations from grid.
""")

spec_zip = {
    "_zip_": {
        "alpha": [0.1, 1.0, 10.0],
        "l1_ratio": [0.2, 0.5, 0.8],
    }
}

print(f"Spec: {json.dumps(spec_zip, indent=2)}")
results = expand_spec(spec_zip)
print(f"_zip_ generates: {len(results)} paired combinations")
for i, r in enumerate(results):
    print(f"  [{i}]: {r}")


# =============================================================================
# Section 6: _chain_ - Sequential Generators
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: _chain_ - Sequential Generators")
print("-" * 60)

print("""
_chain_ concatenates multiple configurations in order:

    {"_chain_": [
        {"model": "baseline", "config": "fast"},
        {"model": "improved", "config": "medium"},
        {"model": "best", "config": "slow"}
    ]}

Preserves order (unlike _or_ which may shuffle).
""")

spec_chain = {
    "_chain_": [
        {"model": "baseline", "config": "fast"},
        {"model": "improved", "config": "medium"},
        {"model": "best", "config": "slow"},
    ]
}

print(f"Spec: {json.dumps(spec_chain, indent=2)}")
results = expand_spec(spec_chain)
print(f"_chain_ generates: {len(results)} configurations in order")
for i, r in enumerate(results):
    print(f"  [{i}]: {r}")


# =============================================================================
# Section 7: _sample_ - Random Sampling
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: _sample_ - Random Sampling")
print("-" * 60)

print("""
_sample_ randomly selects from distributions:

    {"_sample_": {"distribution": "uniform", "from": 0.1, "to": 1.0, "num": 5}}
    {"_sample_": {"distribution": "log_uniform", "from": 0.001, "to": 1.0, "num": 5}}
    {"_sample_": {"distribution": "normal", "mean": 0, "std": 1, "num": 5}}
    {"_sample_": {"distribution": "choice", "values": ["A", "B", "C"], "num": 2}}
""")

# Uniform sampling
spec_uniform = {"_sample_": {"distribution": "uniform", "from": 0.1, "to": 1.0, "num": 5}}
results = expand_spec(spec_uniform, seed=42)
print(f"Uniform [0.1, 1.0]: {[round(x, 3) for x in results]}")

# Log-uniform sampling
spec_log = {"_sample_": {"distribution": "log_uniform", "from": 0.001, "to": 1.0, "num": 5}}
results = expand_spec(spec_log, seed=42)
print(f"Log-uniform [0.001, 1.0]: {[round(x, 4) for x in results]}")

# Normal distribution
spec_normal = {"_sample_": {"distribution": "normal", "mean": 0, "std": 1, "num": 5}}
results = expand_spec(spec_normal, seed=42)
print(f"Normal (mean=0, std=1): {[round(x, 3) for x in results]}")

# Random choice
spec_choice = {"_sample_": {"distribution": "choice", "values": ["A", "B", "C", "D", "E"], "num": 3}}
results = expand_spec(spec_choice, seed=42)
print(f"Choice from [A,B,C,D,E]: {results}")


# =============================================================================
# Section 8: Presets - Named Configurations
# =============================================================================
print("\n" + "-" * 60)
print("Example 8: Presets - Named Configurations")
print("-" * 60)

print("""
Presets define reusable configurations:

    register_preset("fast", {"n_components": 3})
    register_preset("accurate", {"n_components": 15})

    config = {"model_config": {"_preset_": "fast"}}
    resolved = resolve_presets_recursive(config)
""")

# Clear any existing presets
clear_presets()

# Register presets
register_preset("fast", {"n_components": 3, "max_iter": 100})
register_preset("balanced", {"n_components": 8, "max_iter": 500})
register_preset("accurate", {"n_components": 15, "max_iter": 1000})

print(f"Registered presets: {list_presets()}")

# Use presets in config
config_with_preset = {
    "model": "PLSRegression",
    "params": {"_preset_": "balanced"}
}

print(f"\nConfig with preset: {json.dumps(config_with_preset, indent=2)}")
resolved = resolve_presets_recursive(config_with_preset)
print(f"Resolved: {json.dumps(resolved, indent=2)}")


# =============================================================================
# Section 9: Running Pipeline with Generators
# =============================================================================
print("\n" + "-" * 60)
print("Example 9: Running Pipeline with Generators")
print("-" * 60)

print("""
Generator syntax works directly in pipelines:
nirs4all.run() automatically expands generators.
""")

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"_or_": [SNV(), MSC()]},  # 2 preprocessing options
    {"_grid_": {"n_components": [5, 10]}, "model": PLSRegression},  # 2 n_components
]

# This will generate 2 × 2 = 4 pipeline variants
print("Pipeline has: 2 preprocessing × 2 n_components = 4 variants")

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="AdvancedGenerator",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nTotal predictions: {result.num_predictions}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. _mutex_: Mutually exclusive options (can't appear together)
2. _requires_: Dependency constraints (if A then B)
3. _exclude_: Explicit exclusion of combinations
4. _grid_: Full Cartesian product of parameters
5. _zip_: Paired parameter sweeps (same index)
6. _chain_: Sequential configurations (preserves order)
7. _sample_: Random sampling (uniform, log, normal, choice)
8. Presets: Reusable named configurations

Key functions:
- expand_spec(spec, seed=None) - Expand generator spec
- count_combinations(spec) - Count without generating
- register_preset(name, config) - Register named config
- resolve_presets_recursive(config) - Resolve preset references

Design tips:
- Use constraints to prune invalid combinations
- Use _sample_ for large search spaces
- Use _zip_ when parameters should vary together
- Use presets for common configurations

Next: D08_generator_iterators.py - Programmatic generator access
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
