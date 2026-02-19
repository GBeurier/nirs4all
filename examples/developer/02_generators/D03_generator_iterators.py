"""
D03 - Generator Iterators: Programmatic Generator Access
=========================================================

Use the generator API for programmatic control over
pipeline generation, inspection, and export.

This tutorial covers:

* expand_spec_iter for lazy iteration
* batch_iter for batch processing
* Counting and inspecting variants
* Exporting to DataFrame
* Summary and diff utilities

Prerequisites
-------------
- D01_generator_syntax for basic generators
- D02_generator_advanced for constraints

Next Steps
----------
See D04_nested_generators for complex nested patterns.

Duration: ~4 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse
import json
from itertools import islice

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import (
    FirstDerivative,
)
from nirs4all.operators.transforms import (
    MultiplicativeScatterCorrection as MSC,
)
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
)
from nirs4all.pipeline.config.generator import (
    batch_iter,
    count_combinations,
    diff_configs,
    expand_spec,
    expand_spec_iter,
    is_generator_node,
    print_expansion_tree,
    summarize_configs,
    to_dataframe,
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D03 Generator Iterators Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D03 - Generator Iterators: Programmatic Generator Access")
print("=" * 60)

print("""
For large search spaces, you need programmatic control:

    expand_spec_iter():  Lazy iteration (memory efficient)
    batch_iter():        Process in batches
    count_combinations(): Count without generating
    to_dataframe():      Export to pandas
    summarize_configs(): Get statistics
""")

# =============================================================================
# Section 1: Counting Combinations
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Counting Combinations")
print("-" * 60)

print("""
count_combinations() returns the count without generating all variants.
Essential for large search spaces!
""")

spec_small = {"_or_": ["A", "B", "C"]}
spec_medium = {"_grid_": {"x": [1, 2, 3], "y": ["A", "B"], "z": [True, False]}}
spec_large = {"_range_": [1, 1000]}

print(f"Small spec: {count_combinations(spec_small)} variants")
print(f"Medium spec: {count_combinations(spec_medium)} variants")
print(f"Large spec: {count_combinations(spec_large)} variants")

# Nested spec
spec_nested = {
    "preprocessing": {"_or_": ["SNV", "MSC", "Detrend"]},
    "model_params": {
        "_grid_": {
            "n_components": {"_range_": [2, 10]},
            "scale": [True, False]
        }
    }
}
print("\nNested spec:")
print(json.dumps(spec_nested, indent=2))
print(f"Total combinations: {count_combinations(spec_nested)}")

# =============================================================================
# Section 2: Lazy Iteration with expand_spec_iter
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Lazy Iteration with expand_spec_iter")
print("-" * 60)

print("""
expand_spec_iter() generates configurations lazily.
Memory-efficient for large search spaces!
""")

large_spec = {"_range_": [1, 1000]}
print(f"Spec with {count_combinations(large_spec)} variants")

# Use islice to get first N without generating all
first_10 = list(islice(expand_spec_iter(large_spec), 10))
print(f"First 10 (lazy): {first_10}")

last_10 = list(islice(expand_spec_iter(large_spec), 990, 1000))
print(f"Last 10 (lazy): {last_10}")

# =============================================================================
# Section 3: Batch Processing with batch_iter
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Batch Processing with batch_iter")
print("-" * 60)

print("""
batch_iter() yields batches for incremental processing:

    for batch in batch_iter(spec, batch_size=10):
        process_batch(batch)
""")

batch_spec = {"_range_": [1, 25]}
print(f"Processing {count_combinations(batch_spec)} items in batches of 5:")

for i, batch in enumerate(batch_iter(batch_spec, batch_size=5)):
    print(f"  Batch {i}: {batch}")

# =============================================================================
# Section 4: Detecting Generator Nodes
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Detecting Generator Nodes")
print("-" * 60)

print("""
is_generator_node() checks if a dict contains generator syntax:
""")

test_nodes = [
    {"_or_": ["A", "B"]},
    {"_range_": [1, 10]},
    {"_grid_": {"x": [1, 2]}},
    {"_zip_": {"x": [1], "y": [2]}},
    {"_sample_": {"distribution": "uniform", "num": 5}},
    {"class": "PCA", "n_components": 10},  # Not a generator
    {"n_components": 10},  # Not a generator
]

for node in test_nodes:
    first_key = list(node.keys())[0]
    is_gen = is_generator_node(node)
    print(f"  {first_key:20} -> {is_gen}")

# =============================================================================
# Section 5: Export to DataFrame
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Export to DataFrame")
print("-" * 60)

print("""
to_dataframe() converts expanded configs to pandas DataFrame:
""")

configs = expand_spec({
    "_grid_": {
        "model": ["PLS", "RF", "Ridge"],
        "n_components": [5, 10, 15]
    }
})

try:
    df = to_dataframe(configs)
    print("DataFrame:")
    print(df.to_string())
except ImportError:
    print("(pandas not available)")

# =============================================================================
# Section 6: Summarize Configurations
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Summarize Configurations")
print("-" * 60)

print("""
summarize_configs() provides statistics about a set of configs:
""")

summary = summarize_configs(configs)
print(f"Count: {summary['count']} configurations")
print("Keys:")
for key, info in summary['keys'].items():
    print(f"  {key}: {info['unique_count']} unique values: {info['unique_values']}")

# =============================================================================
# Section 7: Compare Configurations
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Compare Configurations")
print("-" * 60)

print("""
diff_configs() shows differences between two configs:
""")

config1 = {"model": "PLS", "n_components": 5, "scale": True}
config2 = {"model": "RF", "n_components": 5, "max_depth": 10}

print(f"Config 1: {config1}")
print(f"Config 2: {config2}")

diff = diff_configs(config1, config2)
print("\nDifferences:")
print(f"  Left only:  {diff.get('left_only', {})}")
print(f"  Right only: {diff.get('right_only', {})}")
print(f"  Different:  {diff.get('different', {})}")
print(f"  Same:       {diff.get('same', {})}")

# =============================================================================
# Section 8: Expansion Tree Visualization
# =============================================================================
print("\n" + "-" * 60)
print("Example 8: Expansion Tree Visualization")
print("-" * 60)

print("""
print_expansion_tree() shows the generator structure:
""")

tree_spec = {
    "preprocessing": {"_or_": ["StandardScaler", "MinMaxScaler"]},
    "model": {
        "_grid_": {
            "class": ["PLS", "RF"],
            "n_components": {"_range_": [2, 5]}
        }
    }
}

print(f"Spec: {json.dumps(tree_spec, indent=2)}")
print("\nExpansion Tree:")
print(print_expansion_tree(tree_spec))

# =============================================================================
# Section 9: Practical Example - Progressive Search
# =============================================================================
print("\n" + "-" * 60)
print("Example 9: Progressive Search Pattern")
print("-" * 60)

print("""
Use iterators for progressive hyperparameter search:
1. Start with coarse grid
2. Narrow down based on results
3. Fine-tune best region
""")

# Coarse search
coarse_spec = {"_range_": [2, 20, 4]}  # 2, 6, 10, 14, 18
print(f"Coarse search: {expand_spec(coarse_spec)}")

# Assume n_components=10 was best, now fine-tune
fine_spec = {"_range_": [8, 12]}  # 8, 9, 10, 11, 12
print(f"Fine search around 10: {expand_spec(fine_spec)}")

# =============================================================================
# Section 10: Running with Generator API
# =============================================================================
print("\n" + "-" * 60)
print("Example 10: Running with Generator API")
print("-" * 60)

print("""
The generator API works directly with nirs4all.run():
""")

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    SNV(),
    PLSRegression(n_components=5),
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="IteratorDemo",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nPredictions: {result.num_predictions}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. count_combinations() - Count without generating
2. expand_spec_iter() - Lazy iteration (memory-efficient)
3. batch_iter() - Process in batches
4. is_generator_node() - Detect generator syntax
5. to_dataframe() - Export to pandas
6. summarize_configs() - Get statistics
7. diff_configs() - Compare configurations
8. print_expansion_tree() - Visualize structure

Best practices:
- Use count_combinations() to estimate search space size
- Use expand_spec_iter() for large spaces
- Use batch processing for parallel execution
- Export to DataFrame for analysis

Next: D04_nested_generators.py - Complex nested patterns
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
