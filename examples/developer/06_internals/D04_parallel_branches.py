"""
Example: Parallel Branch Execution

This example demonstrates how to use parallel branch execution to speed up
pipeline training when you have multiple independent branches.

Key features:
- Automatic detection of safe parallelization (avoids nested parallelization)
- Support for n_jobs parameter at branch level
- Support for n_jobs parameter in finetune_params (Optuna)
- Thread-safe DuckDB access for concurrent branches
- Smart detection of GPU/neural net models to avoid conflicts

Usage:
    python D04_parallel_branches.py
"""

import nirs4all
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from nirs4all.operators.transforms import StandardNormalVariate, MultiplicativeScatterCorrection, SavitzkyGolay
from nirs4all.operators.splitters.splitters import SPXYGFold

# =============================================================================
# Example 1: Basic Parallel Execution
# =============================================================================

print("=" * 80)
print("Example 1: Basic Parallel Branch Execution")
print("=" * 80)

pipeline_basic = [
    SPXYGFold(n_splits=3, random_state=42),
    {
        "branch": [
            [StandardNormalVariate(), PLSRegression(n_components=10)],
            [MultiplicativeScatterCorrection(), Ridge(alpha=1.0)],
            [StandardScaler(), RandomForestRegressor(n_estimators=50, n_jobs=1)],
        ],
        "parallel": True,  # Enable auto-detected parallelization
    },
]

# This will execute 3 branches in parallel (auto-detected: min(3, cpu_count))
result = nirs4all.run(
    pipeline=pipeline_basic,
    dataset="sample_data/regression",
    verbose=2,
)

print(f"\nBest RMSE: {result.best_rmse:.4f}")

# =============================================================================
# Example 2: Explicit n_jobs Control
# =============================================================================

print("\n" + "=" * 80)
print("Example 2: Explicit n_jobs Control")
print("=" * 80)

pipeline_explicit = [
    SPXYGFold(n_splits=3, random_state=42),
    {
        "branch": {
            "light_models": [
                [StandardNormalVariate(), PLSRegression(n_components=10)],
                [MultiplicativeScatterCorrection(), Ridge(alpha=1.0)],
            ],
            "heavy_model": [
                [StandardScaler(), RandomForestRegressor(n_estimators=100, n_jobs=-1)],
            ],
        },
        "n_jobs": 2,  # Limit to 2 parallel branches (for memory control)
    },
]

result = nirs4all.run(
    pipeline=pipeline_explicit,
    dataset="sample_data/regression",
    verbose=2,
)

print(f"\nBest RMSE: {result.best_rmse:.4f}")

# =============================================================================
# Example 3: Parallel Optuna Optimization
# =============================================================================

print("\n" + "=" * 80)
print("Example 3: Parallel Optuna Optimization (within branches)")
print("=" * 80)

pipeline_optuna = [
    SPXYGFold(n_splits=3, random_state=42),
    {
        "branch": [
            # Branch 1: PLS with parallel Optuna optimization
            [
                StandardNormalVariate(),
                {
                    "model": PLSRegression(),
                    "finetune_params": {
                        "n_trials": 20,
                        "n_jobs": 2,  # Parallel Optuna trials
                        "model_params": {
                            "n_components": ("int", 1, 20),
                        },
                    },
                },
            ],
            # Branch 2: Ridge with parallel Optuna optimization
            [
                MultiplicativeScatterCorrection(),
                {
                    "model": Ridge(),
                    "finetune_params": {
                        "n_trials": 30,
                        "n_jobs": 2,  # Parallel Optuna trials
                        "model_params": {
                            "alpha": ("float_log", 1e-5, 1e3),
                        },
                    },
                },
            ],
        ],
        "parallel": False,  # DISABLED: Avoid nested parallelization (Optuna already parallel)
    },
]

result = nirs4all.run(
    pipeline=pipeline_optuna,
    dataset="sample_data/regression",
    verbose=2,
)

print(f"\nBest RMSE: {result.best_rmse:.4f}")

# =============================================================================
# Example 4: Smart Parallelization Detection
# =============================================================================

print("\n" + "=" * 80)
print("Example 4: Smart Parallelization Detection (Auto-Disabled)")
print("=" * 80)

# This pipeline will automatically DISABLE parallel execution because:
# - RandomForestRegressor has n_jobs=-1 (internal parallelization)
# The system detects this and falls back to sequential execution

pipeline_smart = [
    SPXYGFold(n_splits=3, random_state=42),
    {
        "branch": [
            [StandardNormalVariate(), PLSRegression(n_components=10)],
            [MultiplicativeScatterCorrection(), RandomForestRegressor(n_estimators=100, n_jobs=-1)],  # Has n_jobs=-1
        ],
        "parallel": True,  # Requested, but will be auto-disabled
    },
]

result = nirs4all.run(
    pipeline=pipeline_smart,
    dataset="sample_data/regression",
    verbose=2,
)

print(f"\nBest RMSE: {result.best_rmse:.4f}")
print("\n⚠️  Notice: Parallel execution was automatically disabled due to n_jobs=-1 in RandomForest")

# =============================================================================
# Example 5: Named Branches with Parallel Execution
# =============================================================================

print("\n" + "=" * 80)
print("Example 5: Named Branches with Parallel Execution")
print("=" * 80)

pipeline_named = [
    SPXYGFold(n_splits=3, random_state=42),
    {
        "branch": {
            "preprocessing_a": [StandardNormalVariate(), PLSRegression(n_components=10)],
            "preprocessing_b": [MultiplicativeScatterCorrection(), SavitzkyGolay(), PLSRegression(n_components=10)],
            "preprocessing_c": [StandardScaler(), Ridge(alpha=1.0)],
        },
        "n_jobs": 3,  # All 3 branches in parallel
    },
]

result = nirs4all.run(
    pipeline=pipeline_named,
    dataset="sample_data/regression",
    verbose=2,
)

print(f"\nBest RMSE: {result.best_rmse:.4f}")

# =============================================================================
# Example 6: Mixed Strategy (Parallel Branches + Sequential Heavy Models)
# =============================================================================

print("\n" + "=" * 80)
print("Example 6: Mixed Strategy - Group Light and Heavy Models")
print("=" * 80)

# Strategy: Group light models for parallel execution, run heavy models sequentially
pipeline_mixed = [
    SPXYGFold(n_splits=3, random_state=42),

    # First branch group: Light models (can run in parallel)
    {
        "branch": [
            [StandardNormalVariate(), PLSRegression(n_components=10)],
            [MultiplicativeScatterCorrection(), Ridge(alpha=1.0)],
        ],
        "n_jobs": 2,  # Parallel execution
    },

    # Second branch group: Heavy model (runs alone, no parallelization)
    {
        "branch": [
            [StandardScaler(), RandomForestRegressor(n_estimators=200, n_jobs=-1)],
        ],
        "parallel": False,  # Explicitly disabled
    },
]

result = nirs4all.run(
    pipeline=pipeline_mixed,
    dataset="sample_data/regression",
    verbose=2,
)

print(f"\nBest RMSE: {result.best_rmse:.4f}")

print("\n" + "=" * 80)
print("✅ All examples completed!")
print("=" * 80)
print("\nKey Takeaways:")
print("1. Use 'parallel': True for auto-detected parallelization")
print("2. Use 'n_jobs': N for explicit worker control")
print("3. Add 'n_jobs' to finetune_params for parallel Optuna optimization")
print("4. Avoid nested parallelization (branch + Optuna + model n_jobs)")
print("5. System auto-detects unsafe parallelization (GPU, neural nets, n_jobs>1)")
print("6. Group light and heavy models for optimal performance")
