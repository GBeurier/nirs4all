"""
Q40 Example - New Module-Level API
==================================
Demonstrates the new simplified module-level API for nirs4all (v0.6+).
This API provides a more ergonomic interface compared to manually creating
PipelineRunner and config objects.

Features demonstrated:
    - nirs4all.run() for training pipelines
    - nirs4all.predict() for making predictions
    - RunResult convenience methods
    - Session context manager for multiple runs

Comparison with old API:
    # Old API (still works)
    runner = PipelineRunner(verbose=1)
    predictions, _ = runner.run(PipelineConfigs(pipeline), DatasetConfigs(path))

    # New API (recommended)
    result = nirs4all.run(pipeline, path, verbose=1)
"""

# Standard library imports
import argparse
import time

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports - using new module-level API
import nirs4all
from nirs4all.operators.transforms import (
    Detrend, FirstDerivative, Gaussian, SavitzkyGolay, StandardNormalVariate
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q40 New API Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()


# =============================================================================
# Example 1: Simple Training with nirs4all.run()
# =============================================================================
print("\n" + "="*70)
print("Example 1: Simple Training with nirs4all.run()")
print("="*70)

# Define pipeline as a simple list
pipeline_simple = [
    MinMaxScaler(),                          # Feature scaling
    {"y_processing": MinMaxScaler()},        # Target scaling
    ShuffleSplit(n_splits=3, test_size=0.25),  # Cross-validation
    {"model": PLSRegression(n_components=10)}  # Model
]

# Run with the new API - no need to create PipelineConfigs or DatasetConfigs
result = nirs4all.run(
    pipeline=pipeline_simple,
    dataset="sample_data/regression",
    name="SimpleTraining",
    verbose=1,
    save_artifacts=True,
    plots_visible=args.plots
)

# Access results with convenience properties
print(f"\nResults:")
print(f"  Number of predictions: {result.num_predictions}")
print(f"  Best score: {result.best_score:.4f}")
print(f"  Best RMSE: {result.best_rmse:.4f}")
print(f"  Best R²: {result.best_r2:.4f}")

# Get top 3 models
print("\nTop 3 models:")
for i, pred in enumerate(result.top(n=3), 1):
    print(f"  {i}. {pred.get('model_name', 'unknown')} - RMSE: {pred.get('test_score', 0):.4f}")


# =============================================================================
# Example 2: Multi-Model Pipeline with Generator Syntax
# =============================================================================
print("\n" + "="*70)
print("Example 2: Multi-Model Pipeline with Preprocessing Generators")
print("="*70)

# Define pipeline with preprocessing options
pipeline_advanced = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    # Feature augmentation with generator - creates multiple variants
    {"feature_augmentation": {
        "_or_": [Detrend, FirstDerivative, Gaussian, SavitzkyGolay],
        "pick": 2,
        "count": 3
    }},
    ShuffleSplit(n_splits=3, test_size=0.25),
]

# Add multiple PLS models with different n_components
for n in [5, 10, 15]:
    pipeline_advanced.append({
        "name": f"PLS-{n}",
        "model": PLSRegression(n_components=n)
    })

# Run with new API
result_advanced = nirs4all.run(
    pipeline=pipeline_advanced,
    dataset="sample_data/regression",
    name="AdvancedPipeline",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nGenerated {result_advanced.num_predictions} pipeline variants")
print(f"Best RMSE: {result_advanced.best_rmse:.4f}")

# Filter results by model
models = result_advanced.get_models()
print(f"\nModels evaluated: {models}")


# =============================================================================
# Example 3: Using Session for Multiple Runs
# =============================================================================
print("\n" + "="*70)
print("Example 3: Session for Multiple Runs (shared resources)")
print("="*70)

# Session shares workspace and configuration across runs
with nirs4all.session(verbose=1, save_artifacts=True) as s:
    # Run 1: PLS model
    result_pls = nirs4all.run(
        pipeline=[
            MinMaxScaler(),
            ShuffleSplit(n_splits=3),
            {"model": PLSRegression(n_components=10)}
        ],
        dataset="sample_data/regression",
        name="PLS_Session",
        session=s
    )

    # Run 2: Different preprocessing
    result_snv = nirs4all.run(
        pipeline=[
            MinMaxScaler(),
            StandardNormalVariate(),
            ShuffleSplit(n_splits=3),
            {"model": PLSRegression(n_components=10)}
        ],
        dataset="sample_data/regression",
        name="SNV_PLS_Session",
        session=s
    )

    print(f"\nSession results:")
    print(f"  PLS only:     RMSE = {result_pls.best_rmse:.4f}")
    print(f"  SNV + PLS:    RMSE = {result_snv.best_rmse:.4f}")

    # Determine which is better
    if result_snv.best_rmse < result_pls.best_rmse:
        print(f"  → SNV preprocessing improved RMSE by {result_pls.best_rmse - result_snv.best_rmse:.4f}")
    else:
        print(f"  → No improvement from SNV preprocessing")


# =============================================================================
# Example 4: Result Exploration and Export
# =============================================================================
print("\n" + "="*70)
print("Example 4: Result Exploration")
print("="*70)

# Use the result from Example 1
print(f"\nResult summary:")
print(result)  # Uses __str__ which calls summary()

# Get available datasets and models
print(f"\nDatasets: {result.get_datasets()}")
print(f"Models: {result.get_models()}")

# Show the best entry details
best = result.best
if best:
    print(f"\nBest model details:")
    print(f"  Name: {best.get('model_name')}")
    print(f"  Dataset: {best.get('dataset_name')}")
    print(f"  Preprocessings: {best.get('preprocessings')}")

# Note: Export requires artifacts to be saved
# result.export("exports/Q40_best_model.n4a")


# =============================================================================
# Comparison: Old API vs New API
# =============================================================================
print("\n" + "="*70)
print("API Comparison")
print("="*70)

print("""
Old API (still supported):
--------------------------
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

runner = PipelineRunner(verbose=1, save_artifacts=True)
pipeline_config = PipelineConfigs(pipeline, "MyPipeline")
dataset_config = DatasetConfigs("sample_data/regression")
predictions, per_dataset = runner.run(pipeline_config, dataset_config)
best = predictions.top(n=1)[0]


New API (recommended):
----------------------
import nirs4all

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="MyPipeline",
    verbose=1,
    save_artifacts=True
)
best = result.best  # Convenience accessor
""")

print("\nExample complete!")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
