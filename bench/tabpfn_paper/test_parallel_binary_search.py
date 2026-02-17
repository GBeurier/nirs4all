"""
Test script for parallel variant execution and gradient-based binary search.

This script demonstrates:
1. Parallel execution of pipeline variants (sweeps)
2. Gradient-based binary search sampler

Usage:
    python test_parallel_binary_search.py
"""

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate as SNV, MultiplicativeScatterCorrection as MSC

# =============================================================================
# Test 1: Parallel Variant Execution
# =============================================================================
print("=" * 70)
print("Test 1: Parallel Variant Execution (Sweeps)")
print("=" * 70)

pipeline_parallel = [
    ShuffleSplit(n_splits=2, test_size=0.3, random_state=42),
    {
        "_cartesian_": [
            {"_or_": [None, SNV, MSC]},  # 3 variants
            {"_or_": [StandardScaler, None]},  # 2 variants
        ],
        "count": 6,  # 3 * 2 = 6 total variants
    },
    {
        "model": PLSRegression(n_components=5),
        "name": "PLS",
    },
]

print("\nPipeline creates 6 variants via _cartesian_")
print("Running with n_jobs=4 (parallel execution)...")

result_parallel = nirs4all.run(
    pipeline=pipeline_parallel,
    dataset="examples/sample_data/regression",
    name="ParallelTest",
    verbose=1,
    n_jobs=4,  # Run 4 variants in parallel
)

print(f"\n✓ Completed {result_parallel.num_predictions} predictions")
print(f"✓ Best RMSE: {result_parallel.best_rmse:.4f}")

# =============================================================================
# Test 2: Gradient-Based Binary Search
# =============================================================================
print("\n" + "=" * 70)
print("Test 2: Gradient-Based Binary Search Sampler")
print("=" * 70)

pipeline_binary = [
    SNV(),
    ShuffleSplit(n_splits=2, test_size=0.3, random_state=42),
    {
        "model": PLSRegression(),
        "name": "PLS",
        "finetune_params": {
            "n_trials": 12,
            "sampler": "binary",  # Use gradient-based binary search
            "verbose": 2,
            "model_params": {
                "n_components": ('int', 1, 30),  # Search range
            },
        },
    },
]

print("\nSearching for optimal n_components in range [1, 30]")
print("Using gradient-based binary search (12 trials)...")

result_binary = nirs4all.run(
    pipeline=pipeline_binary,
    dataset="examples/sample_data/regression",
    name="BinarySearchTest",
    verbose=1,
)

print(f"\n✓ Optimization completed in {result_binary.num_predictions} trials")
print(f"✓ Best RMSE: {result_binary.best_rmse:.4f}")
best_pred = result_binary.predictions.get_best()
if best_pred:
    print(f"✓ Optimal n_components: {best_pred.get('params', {}).get('n_components', 'N/A')}")

# =============================================================================
# Test 3: Combined (Parallel + Binary Search)
# =============================================================================
print("\n" + "=" * 70)
print("Test 3: Combined Parallel Variants + Binary Search")
print("=" * 70)

pipeline_combined = [
    ShuffleSplit(n_splits=2, test_size=0.3, random_state=42),
    {
        "_or_": [None, SNV, MSC],  # 3 preprocessing variants
    },
    {
        "model": PLSRegression(),
        "name": "PLS",
        "finetune_params": {
            "n_trials": 10,
            "sampler": "binary",
            "verbose": 1,
            "model_params": {
                "n_components": ('int', 1, 25),
            },
        },
    },
]

print("\nPipeline creates 3 preprocessing variants")
print("Each variant uses binary search to find optimal n_components")
print("Running with n_jobs=3 (parallel variants)...")

result_combined = nirs4all.run(
    pipeline=pipeline_combined,
    dataset="examples/sample_data/regression",
    name="CombinedTest",
    verbose=1,
    n_jobs=3,  # Parallelize the 3 preprocessing variants
)

print(f"\n✓ Completed {result_combined.num_predictions} predictions")
print(f"✓ Best RMSE: {result_combined.best_rmse:.4f}")

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)
