"""
U01 - Hello World: Your First nirs4all Pipeline
================================================

The simplest possible nirs4all pipeline in about 20 lines of code.

This tutorial covers:

* Using ``nirs4all.run()`` to train a pipeline
* The structure of a minimal pipeline
* Reading results from the ``RunResult`` object

Prerequisites
-------------
None - this is the starting point!

Next Steps
----------
After this example, see :ref:`U02_basic_regression` for preprocessing and visualization.

Duration: ~1 minute
Difficulty: ★☆☆☆☆
"""

# Standard library imports
import argparse

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U01 Hello World Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Define a Minimal Pipeline
# =============================================================================
print("\n" + "=" * 60)
print("U01 - Hello World: Your First nirs4all Pipeline")
print("=" * 60)

# A pipeline is a simple list of processing steps
pipeline = [
    MinMaxScaler(),                              # Feature scaling
    {"y_processing": MinMaxScaler()},            # Target scaling
    ShuffleSplit(n_splits=3, test_size=0.25),    # Cross-validation
    {"model": PLSRegression(n_components=10)}    # Model
]

print("\n📋 Pipeline defined:")
print("   1. MinMaxScaler() - scale features to [0,1]")
print("   2. y_processing - scale targets")
print("   3. ShuffleSplit - 3-fold cross-validation")
print("   4. PLSRegression - PLS model with 10 components")


# =============================================================================
# Section 2: Train the Pipeline
# =============================================================================
print("\n" + "-" * 60)
print("Training the pipeline...")
print("-" * 60)

# Run the pipeline with one simple call
result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="HelloWorld",
    verbose=1,
    save_artifacts=True,
    plots_visible=args.plots
)


# =============================================================================
# Section 3: Access Results
# =============================================================================
print("\n" + "-" * 60)
print("Results")
print("-" * 60)

# The result object provides convenient accessors
print(f"\n📊 Pipeline Results:")
print(f"   Number of predictions: {result.num_predictions}")
print(f"   Best RMSE: {result.best_rmse:.4f}")
print(f"   Best R²: {result.best_r2:.4f}")

# Get the best model details
best = result.best
if best:
    print(f"\n🏆 Best Model Details:")
    print(f"   Model name: {best.get('model_name', 'unknown')}")
    print(f"   Dataset: {best.get('dataset_name', 'unknown')}")
    print(f"   Fold: {best.get('fold_id', 'unknown')}")


# =============================================================================
# Section 4: Get Top Models
# =============================================================================
print("\n" + "-" * 60)
print("Top 3 Models")
print("-" * 60)

for i, pred in enumerate(result.top(n=3), 1):
    rmse = pred.get('test_rmse', pred.get('rmse', 0))
    r2 = pred.get('test_r2', pred.get('r2', 0))
    print(f"   {i}. {pred.get('model_name', 'unknown')} - RMSE: {rmse:.4f}, R²: {r2:.4f}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. A pipeline is a list of processing steps
2. Use nirs4all.run() to train - no boilerplate needed
3. Results are accessed via result.best_rmse, result.top(n), etc.

Key API:
  result = nirs4all.run(pipeline, dataset, name, ...)
  result.best_rmse    # Best RMSE score
  result.best_r2      # Best R² score
  result.top(n=5)     # Top N predictions
  result.best         # Best prediction entry

Next: U02_basic_regression.py - Add preprocessing and visualization
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
