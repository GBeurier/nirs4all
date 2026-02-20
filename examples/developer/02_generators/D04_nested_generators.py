"""
D04 - Nested Generators: Complex Multi-Level Generation
========================================================

Nested generators enable sophisticated search spaces with
generators inside generators, branches, and preprocessing chains.

This tutorial covers:

* Nested _or_ for preprocessing chains
* Generators in dictionary values
* Hierarchical parameter spaces
* Complex pipeline patterns

Prerequisites
-------------
- D01_generator_syntax for basic generators
- D02_generator_advanced for constraints
- 01_branching/D01_branching_basics for branching concepts

Next Steps
----------
See 03_deep_learning/D01_pytorch_models for deep learning integration.

Duration: ~5 minutes
Difficulty: ★★★★★
"""

# Standard library imports
import argparse
import json
from typing import Any

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import Detrend, FirstDerivative, SavitzkyGolay, SecondDerivative
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.pipeline.config.generator import (
    count_combinations,
    expand_spec,
    print_expansion_tree,
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D04 Nested Generators Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D04 - Nested Generators: Complex Multi-Level Generation")
print("=" * 60)

print("""
Generators can be nested for complex search spaces:

    {"model_config": {
        "params": {
            "alpha": {"_log_range_": [0.001, 100, 5]},
            "n_components": {"_range_": [2, 10]}
        }
    }}

Each level expands independently, multiplying the total combinations.
""")

# =============================================================================
# Section 1: Nested _or_ for Preprocessing Chains
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Nested _or_ for Preprocessing Chains")
print("-" * 60)

print("""
Nested lists in _or_ create multi-step preprocessing variants:

    {"_or_": [
        ["SNV", "FirstDerivative"],  # 2-step chain
        ["MSC"],                      # 1-step chain
        ["Detrend", "SavGol", "SNV"]  # 3-step chain
    ]}
""")

spec_chains = {
    "_or_": [
        ["SNV", "FirstDerivative"],
        ["MSC"],
        ["Detrend", "SavGol", "SNV"],
    ]
}

print(f"Spec: {json.dumps(spec_chains, indent=2)}")
results = expand_spec(spec_chains)
print(f"\nExpanded ({len(results)} chains):")
for i, r in enumerate(results):
    print(f"  [{i}]: {r}")

# =============================================================================
# Section 2: Generators in Dictionary Values
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Generators in Dictionary Values")
print("-" * 60)

print("""
Each dictionary key can have its own generator:

    {
        "model": {"_or_": ["PLS", "RF"]},
        "params": {"_grid_": {...}}
    }

The result is the Cartesian product of all generators.
""")

spec_dict = {
    "model": {"_or_": ["PLS", "RF"]},
    "n_components": {"_range_": [3, 7, 2]},
}

print(f"Spec: {json.dumps(spec_dict, indent=2)}")
print(f"Count: {count_combinations(spec_dict)}")
results = expand_spec(spec_dict)
print(f"\nExpanded ({len(results)} configs):")
for i, r in enumerate(results):
    print(f"  [{i}]: {r}")

# =============================================================================
# Section 3: Deeply Nested Generators
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Deeply Nested Generators")
print("-" * 60)

print("""
Generators can be arbitrarily nested:

    {"outer": {"inner": {"_or_": [...]}}}
""")

spec_deep = {
    "preprocessing": {
        "scaler": {"_or_": ["Standard", "MinMax"]},
        "transforms": {
            "_or_": ["SNV", "MSC"]
        }
    },
    "model": {
        "type": "PLS",
        "params": {
            "n_components": {"_range_": [3, 5]}
        }
    }
}

print(f"Spec:\n{json.dumps(spec_deep, indent=2)}")
print(f"\nTotal combinations: {count_combinations(spec_deep)}")
results = expand_spec(spec_deep)
print("\nFirst 4 results:")
for i, r in enumerate(results[:4]):
    print(f"  [{i}]: {json.dumps(r)}")

# =============================================================================
# Section 4: Conditional Generators
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Conditional Generators (Grouped Options)")
print("-" * 60)

print("""
Group related parameters together using _or_ with dicts:

    {"_or_": [
        {"model": "PLS", "n_components": 5},
        {"model": "RF", "n_estimators": 100}
    ]}

This ensures consistent parameter sets.
""")

spec_conditional = {
    "_or_": [
        {"model": "PLS", "n_components": 5, "scale": True},
        {"model": "PLS", "n_components": 10, "scale": True},
        {"model": "RF", "n_estimators": 50, "max_depth": 5},
        {"model": "RF", "n_estimators": 100, "max_depth": 10},
    ]
}

print(f"Spec: {json.dumps(spec_conditional, indent=2)}")
results = expand_spec(spec_conditional)
print(f"\nExpanded ({len(results)} configs):")
for i, r in enumerate(results):
    print(f"  [{i}]: {r}")

# =============================================================================
# Section 5: Generator with Mixed Nesting
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Generator with Mixed Nesting")
print("-" * 60)

print("""
Combine different generator types at different levels:
""")

spec_mixed = {
    "stage1": {"_or_": ["A", "B"]},  # 2 options
    "stage2": {
        "_grid_": {
            "x": [1, 2],
            "y": [True, False]
        }  # 4 options
    },
    "stage3": {"_zip_": {"a": [10, 20], "b": ["X", "Y"]}}  # 2 options
}

print(f"Spec:\n{json.dumps(spec_mixed, indent=2)}")
print("\nCombination counts:")
print("  stage1: 2 options")
print("  stage2: 4 options (2×2 grid)")
print("  stage3: 2 options (zip pairs)")
print(f"  Total: 2 × 4 × 2 = {count_combinations(spec_mixed)}")

results = expand_spec(spec_mixed)
print("\nFirst 4 results:")
for i, r in enumerate(results[:4]):
    print(f"  [{i}]: {r}")

# =============================================================================
# Section 6: Visualizing Nested Structure
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Visualizing Nested Structure")
print("-" * 60)

print("""
print_expansion_tree() helps visualize complex nesting:
""")

tree_spec = {
    "preprocessing": {"_or_": ["StandardScaler", "MinMaxScaler"]},
    "feature_extraction": {
        "_or_": [
            {"class": "PCA", "n_components": {"_or_": [5, 10]}},
            {"class": "SVD", "n_components": 10}
        ]
    },
    "model": {
        "_grid_": {
            "type": ["PLS", "Ridge"],
            "alpha": {"_log_range_": [0.01, 1, 3]}
        }
    }
}

print("Expansion Tree:")
print(print_expansion_tree(tree_spec))
print(f"\nTotal combinations: {count_combinations(tree_spec)}")

# =============================================================================
# Section 7: Pipeline with Nested Generators
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Pipeline with Nested Generators")
print("-" * 60)

print("""
Nested generators work directly in pipelines:
""")

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    # Nested _or_ - each option is a preprocessing sequence
    {"_or_": [
        SNV(),  # Single step
        [SNV(), FirstDerivative()],  # Two-step chain
    ]},
    PLSRegression(n_components=5),
]

print("Pipeline with nested preprocessing options:")
print("  Option 1: SNV only")
print("  Option 2: SNV → FirstDerivative")

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="NestedGenerator",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nTotal predictions: {result.num_predictions}")

# =============================================================================
# Section 8: Debugging Nested Generators
# =============================================================================
print("\n" + "-" * 60)
print("Example 8: Debugging Nested Generators")
print("-" * 60)

print("""
Tips for debugging complex nested generators:

1. Use count_combinations() first to check expected size
2. Use print_expansion_tree() to visualize structure
3. Test each level independently
4. Use small subsets before full expansion
""")

# Debug a complex spec step by step
complex_spec: dict[str, Any] = {
    "a": {"_or_": ["x", "y"]},
    "b": {"_range_": [1, 3]},
    "c": {"_or_": ["m", "n"]}
}

# Step 1: Count
print(f"Step 1 - Count: {count_combinations(complex_spec)}")

# Step 2: Test each part
print("Step 2 - Test parts:")
print(f"  a: {expand_spec(complex_spec['a'])}")
print(f"  b: {expand_spec(complex_spec['b'])}")
print(f"  c: {expand_spec(complex_spec['c'])}")

# Step 3: Full expansion
print(f"Step 3 - Full expansion: {len(expand_spec(complex_spec))} configs")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. Nested lists in _or_ create multi-step chains
2. Dict values can each have generators (Cartesian product)
3. Arbitrary nesting depth is supported
4. Grouped options ensure parameter consistency
5. Mixed generator types at different levels
6. print_expansion_tree() for visualization
7. Debug by testing parts independently

Best practices:
- Start simple, add nesting incrementally
- Use count_combinations() to check size
- Test each level before combining
- Use print_expansion_tree() for complex specs
- Group related parameters with _or_ of dicts

Next: 03_deep_learning/D01_pytorch_models.py - Deep learning integration
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
