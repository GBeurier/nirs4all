"""
D05 - Binary Search Sampler for Unimodal Parameters
====================================================

Demonstrates the BinarySearchSampler for efficiently optimizing unimodal
integer parameters like PLS n_components.

Binary search reduces optimization from ~30-50 trials (TPE) to ~10-15 trials
for parameters with single-peak behavior.

Prerequisites
-------------
Understanding of hyperparameter tuning (see U02_hyperparameter_tuning).

Duration: ~2 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse
import time

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D05 Binary Search Sampler Example')
parser.add_argument('--comparison', action='store_true', help='Run comparison with TPE sampler')
args = parser.parse_args()


# =============================================================================
# Section 1: Basic Binary Search
# =============================================================================
print("\n" + "=" * 80)
print("D05 - Binary Search Sampler")
print("=" * 80)

print("""
Binary Search Sampler for Unimodal Integer Parameters
------------------------------------------------------

The BinarySearchSampler is optimized for parameters that exhibit unimodal
behavior (single peak with monotonic gradients on both sides).

Perfect for:
  • PLS/PCR n_components (most common use case)
  • KNN n_neighbors
  • Polynomial degree
  • Any integer parameter with clear peak

Strategy:
  1. Initial phase: Test low, high, midpoint (3 trials)
  2. Binary search: Narrow based on best value
  3. Refinement: Test neighbors of best value

Efficiency:
  • TPE: ~30-50 trials for 1-30 range
  • Binary: ~10-15 trials for same range
""")

pipeline_binary = [
    StandardNormalVariate(),
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
    {
        "model": PLSRegression(),
        "name": "PLS-BinarySearch",
        "finetune_params": {
            "n_trials": 12,              # Much fewer trials needed!
            "sampler": "binary",         # Binary search sampler
            "verbose": 2,                # Show trial details
            "seed": 42,
            "approach": "single",
            "model_params": {
                "n_components": ('int', 1, 30),  # Unimodal parameter
            },
        }
    },
]

print("\n" + "-" * 80)
print("Running Binary Search Optimization (12 trials)...")
print("-" * 80)

start_time = time.time()
result_binary = nirs4all.run(
    pipeline=pipeline_binary,
    dataset="sample_data/regression",
    name="BinarySearch",
    verbose=1
)
binary_time = time.time() - start_time

print(f"\n✓ Binary Search completed in {binary_time:.2f}s")
print(f"  Best RMSE: {result_binary.best_score:.4f}")
print(f"  Optimal n_components: {result_binary.best.get('best_params', {}).get('n_components', 'N/A')}")


# =============================================================================
# Section 2: Comparison with TPE (Optional)
# =============================================================================
if args.comparison:
    print("\n" + "=" * 80)
    print("Section 2: Comparison with TPE Sampler")
    print("=" * 80)

    print("""
    Running TPE with same number of trials for comparison.
    TPE typically needs 2-3x more trials for integer parameters.
    """)

    pipeline_tpe = [
        StandardNormalVariate(),
        MinMaxScaler(),
        ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
        {
            "model": PLSRegression(),
            "name": "PLS-TPE",
            "finetune_params": {
                "n_trials": 12,          # Same number of trials
                "sampler": "tpe",        # TPE for comparison
                "verbose": 2,
                "seed": 42,
                "approach": "single",
                "model_params": {
                    "n_components": ('int', 1, 30),
                },
            }
        },
    ]

    print("\n" + "-" * 80)
    print("Running TPE Optimization (12 trials)...")
    print("-" * 80)

    start_time = time.time()
    result_tpe = nirs4all.run(
        pipeline=pipeline_tpe,
        dataset="sample_data/regression",
        name="TPE-Comparison",
        verbose=1
    )
    tpe_time = time.time() - start_time

    print(f"\n✓ TPE completed in {tpe_time:.2f}s")
    print(f"  Best RMSE: {result_tpe.best_score:.4f}")
    print(f"  Optimal n_components: {result_tpe.best.get('best_params', {}).get('n_components', 'N/A')}")

    # Comparison summary
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    print(f"Binary Search:")
    print(f"  • Time: {binary_time:.2f}s")
    print(f"  • Best RMSE: {result_binary.best_score:.4f}")
    print(f"  • Optimal n_components: {result_binary.best.get('best_params', {}).get('n_components', 'N/A')}")
    print(f"\nTPE:")
    print(f"  • Time: {tpe_time:.2f}s")
    print(f"  • Best RMSE: {result_tpe.best_score:.4f}")
    print(f"  • Optimal n_components: {result_tpe.best.get('best_params', {}).get('n_components', 'N/A')}")
    print(f"\nSpeedup: {tpe_time / binary_time:.2f}x faster")
    print(f"Note: For optimal results, TPE would need ~30-50 trials, while binary needs ~12-15")


# =============================================================================
# Section 3: Multi-Phase with Binary Search
# =============================================================================
print("\n" + "=" * 80)
print("Section 3: Multi-Phase Search (Coarse → Fine)")
print("=" * 80)

print("""
Combine binary search with multi-phase optimization:
  • Phase 1: Binary search for coarse optimization
  • Phase 2: TPE for fine-tuning around best value
""")

pipeline_multiphase = [
    StandardNormalVariate(),
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
    {
        "model": PLSRegression(),
        "name": "PLS-MultiPhase",
        "finetune_params": {
            "verbose": 2,
            "seed": 42,
            "approach": "single",
            "phases": [
                {
                    "n_trials": 8,
                    "sampler": "binary",  # Phase 1: Binary search
                },
                {
                    "n_trials": 5,
                    "sampler": "tpe",     # Phase 2: TPE refinement
                },
            ],
            "model_params": {
                "n_components": ('int', 1, 30),
            },
        }
    },
]

print("\n" + "-" * 80)
print("Running Multi-Phase Optimization (8 binary + 5 TPE = 13 trials)...")
print("-" * 80)

result_multiphase = nirs4all.run(
    pipeline=pipeline_multiphase,
    dataset="sample_data/regression",
    name="MultiPhase",
    verbose=1
)

print(f"\n✓ Multi-Phase completed")
print(f"  Best RMSE: {result_multiphase.best_score:.4f}")
print(f"  Optimal n_components: {result_multiphase.best.get('best_params', {}).get('n_components', 'N/A')}")


# =============================================================================
# Section 4: When NOT to Use Binary Search
# =============================================================================
print("\n" + "=" * 80)
print("Section 4: When NOT to Use Binary Search")
print("=" * 80)

print("""
Binary search is NOT suitable for:

❌ Multi-modal parameters (multiple peaks):
   • Random Forest n_estimators (may have local optima)
   • Learning rates with multiple sweet spots

❌ Continuous float parameters:
   • Regularization alpha (use 'tpe' or 'cmaes')
   • Dropout rates

❌ Categorical parameters:
   • Activation functions (use 'grid')
   • Optimizers

❌ Non-monotonic parameters:
   • Parameters where middle values are worst

✅ Best for:
   • PLS/PCR n_components (most common)
   • KNN n_neighbors
   • Polynomial degree
   • Any integer with clear unimodal behavior
""")


# =============================================================================
# Section 5: Practical Usage Guide
# =============================================================================
print("\n" + "=" * 80)
print("Section 5: Practical Usage Guide")
print("=" * 80)

print("""
How to use Binary Search Sampler:

1. For PLS n_components (recommended):

   finetune_params = {
       "n_trials": 12,              # 10-15 trials sufficient
       "sampler": "binary",
       "model_params": {
           "n_components": ('int', 1, 30),
       }
   }

2. For multiple parameters (binary only affects integers):

   finetune_params = {
       "n_trials": 20,
       "sampler": "binary",         # Will use binary for integers
       "model_params": {
           "n_components": ('int', 1, 30),  # Binary search
           "scale": [True, False],           # Categorical (exhaustive)
       }
   }

3. Multi-phase approach (recommended for best results):

   finetune_params = {
       "seed": 42,
       "phases": [
           {"n_trials": 8, "sampler": "binary"},  # Coarse search
           {"n_trials": 5, "sampler": "tpe"},     # Fine-tuning
       ],
       "model_params": {
           "n_components": ('int', 1, 30),
       }
   }

4. Combined with preprocessing search:

   pipeline = [
       {"feature_augmentation": [SNV, MSC, Detrend], "action": "extend"},
       ShuffleSplit(n_splits=3),
       {
           "model": PLSRegression(),
           "finetune_params": {
               "n_trials": 12,
               "sampler": "binary",
               "approach": "grouped",  # Binary search per preprocessing
               "model_params": {
                   "n_components": ('int', 1, 30),
               }
           }
       }
   ]

Performance Tips:
-----------------
• For PLS n_components in range 1-30: use 10-15 trials
• For larger ranges (1-100): use 15-20 trials
• Always set seed for reproducibility
• Consider multi-phase for production models
• Use "grouped" approach with preprocessing variants
""")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("""
Binary Search Sampler - Key Points:

✓ Efficiency:
  • Reduces trials from ~30-50 (TPE) to ~10-15 (Binary)
  • 2-3x faster convergence for unimodal integers

✓ Best Use Cases:
  • PLS/PCR n_components (primary use case)
  • KNN n_neighbors
  • Polynomial degrees
  • Any unimodal integer parameter

✓ Configuration:
  {
    "n_trials": 12,
    "sampler": "binary",
    "seed": 42,
    "model_params": {
      "n_components": ('int', 1, 30),
    }
  }

✓ Advanced Pattern (Multi-Phase):
  {
    "phases": [
      {"n_trials": 8, "sampler": "binary"},   # Coarse
      {"n_trials": 5, "sampler": "tpe"},      # Fine-tune
    ],
    "model_params": {
      "n_components": ('int', 1, 30),
    }
  }

⚠ Limitations:
  • Only for integer parameters with unimodal behavior
  • Not suitable for multi-modal or continuous parameters
  • Falls back to random for non-integer parameters

Next Steps:
-----------
• Use "binary" sampler for PLS optimization in production
• Combine with "grouped" approach for preprocessing variants
• Consider multi-phase for final model tuning
""")

if __name__ == "__main__":
    pass
