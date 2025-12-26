"""
U06 - Multi-Datasets: Analyzing Multiple Datasets in Parallel
==============================================================

Run the same pipeline on multiple datasets and compare results.

This tutorial covers:

* Specifying multiple datasets as a list
* Per-dataset result access
* Cross-dataset comparison visualizations
* Dataset-wise performance analysis

Prerequisites
-------------
Complete :ref:`U05_flexible_inputs` first.

Next Steps
----------
See :ref:`U07_multi_source` for multi-source data (e.g., NIR + other sensors).

Duration: ~3 minutes
Difficulty: ★★☆☆☆
"""

# Standard library imports
import argparse
import matplotlib.pyplot as plt

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.data.predictions import Predictions
from nirs4all.operators.transforms import (
    Gaussian, SavitzkyGolay, StandardNormalVariate, Haar, MultiplicativeScatterCorrection
)
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U06 Multi-Datasets Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Define the Pipeline
# =============================================================================
print("\n" + "=" * 60)
print("U06 - Multi-Dataset Analysis")
print("=" * 60)

# Build pipeline with feature augmentation
pipeline = [
    # Preprocessing
    MinMaxScaler(feature_range=(0.1, 0.8)),

    # Feature augmentation with preprocessing combinations
    {"feature_augmentation": [
        MultiplicativeScatterCorrection,
        Gaussian,
        StandardNormalVariate,
        SavitzkyGolay,
        Haar,
    ]},

    # Cross-validation
    ShuffleSplit(n_splits=3, random_state=42),

    # Target scaling
    {"y_processing": MinMaxScaler()},

    # Models
    {"model": PLSRegression(n_components=5), "name": "PLS-5"},
    {"model": PLSRegression(n_components=10), "name": "PLS-10"},
    {"model": ElasticNet(alpha=0.1), "name": "ElasticNet"},
]

print("\n📋 Pipeline defined:")
print("   • MinMaxScaler + 5 preprocessing options")
print("   • 3-fold ShuffleSplit")
print("   • 3 models: PLS-5, PLS-10, ElasticNet")


# =============================================================================
# Section 2: Define Multiple Datasets
# =============================================================================
print("\n" + "-" * 60)
print("Configuring Multiple Datasets")
print("-" * 60)

# Specify multiple datasets as a list of paths
data_paths = [
    'sample_data/regression',
    'sample_data/regression_2',
    'sample_data/regression_3'
]

print("Datasets:")
for i, path in enumerate(data_paths, 1):
    print(f"   {i}. {path}")


# =============================================================================
# Section 3: Run Pipeline Across All Datasets
# =============================================================================
print("\n" + "-" * 60)
print("Training Pipeline on All Datasets")
print("-" * 60)

result = nirs4all.run(
    pipeline=pipeline,
    dataset=data_paths,
    name="MultiDataset",
    verbose=1,
    plots_visible=args.plots
)

predictions = result.predictions
predictions_per_dataset = result.predictions_per_dataset

print(f"\n📊 Training complete!")
print(f"   Total predictions: {predictions.num_predictions}")
print(f"   Datasets processed: {len(predictions_per_dataset)}")


# =============================================================================
# Section 4: Per-Dataset Results
# =============================================================================
print("\n" + "-" * 60)
print("Per-Dataset Results")
print("-" * 60)

for dataset_name, dataset_pred_dict in predictions_per_dataset.items():
    print(f"\n📁 Dataset: {dataset_name}")

    # Get the Predictions object from the dict
    dataset_predictions = dataset_pred_dict['run_predictions']

    # Get top 3 models for this dataset
    top_models = dataset_predictions.top(n=3, rank_metric='rmse')
    print("   Top 3 models:")
    for idx, model in enumerate(top_models, 1):
        rmse = model.get('test_rmse', model.get('rmse', 0))
        r2 = model.get('test_r2', model.get('r2', 0))
        print(f"   {idx}. {model['model_name']} - RMSE: {rmse:.4f}, R²: {r2:.4f}")


# =============================================================================
# Section 5: Cross-Dataset Visualizations
# =============================================================================
print("\n" + "-" * 60)
print("Creating Cross-Dataset Visualizations")
print("-" * 60)

# Create analyzer with all predictions
analyzer = PredictionAnalyzer(predictions)

# Heatmap: models vs datasets
fig1 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="dataset_name",
    rank_metric='rmse',
    display_metric='rmse',
)
print("   ✓ Heatmap: model_name vs dataset_name (RMSE)")

# Heatmap: models vs datasets (R²)
fig2 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="dataset_name",
    rank_metric='r2',
    display_metric='r2',
)
print("   ✓ Heatmap: model_name vs dataset_name (R²)")

# Candlestick by model (across all datasets)
fig3 = analyzer.plot_candlestick(
    variable="model_name",
    display_metric='rmse',
)
print("   ✓ Candlestick: model performance distribution")

# Candlestick by dataset
fig4 = analyzer.plot_candlestick(
    variable="dataset_name",
    display_metric='rmse',
)
print("   ✓ Candlestick: dataset difficulty comparison")


# =============================================================================
# Section 6: Best Model Selection
# =============================================================================
print("\n" + "-" * 60)
print("Best Model Selection")
print("-" * 60)

# Overall best model
best = predictions.get_best(ascending=True)  # ascending=True for RMSE
print(f"\n🏆 Overall Best Model:")
print(f"   Model: {best.get('model_name')}")
print(f"   Dataset: {best.get('dataset_name')}")
print(f"   RMSE: {best.get('test_rmse', best.get('rmse', 0)):.4f}")
print(f"   R²: {best.get('test_r2', best.get('r2', 0)):.4f}")

# Best model per dataset
print("\n🏆 Best Model Per Dataset:")
for dataset_name, dataset_pred_dict in predictions_per_dataset.items():
    dataset_preds = dataset_pred_dict['run_predictions']
    best_for_ds = dataset_preds.get_best(ascending=True)
    rmse = best_for_ds.get('test_rmse', best_for_ds.get('rmse', 0))
    print(f"   {dataset_name}: {best_for_ds.get('model_name')} (RMSE: {rmse:.4f})")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Multi-Dataset Analysis:

  1. Specify datasets as a list of paths:
     dataset=['path1', 'path2', 'path3']

  2. Access per-dataset results:
     result.predictions_per_dataset  # Dict of {name: predictions}

  3. Cross-dataset visualization:
     analyzer.plot_heatmap(x_var="model_name", y_var="dataset_name")

Use cases:
  • Compare model generalization across samples
  • Identify dataset-specific challenges
  • Find robust models that work well everywhere

Key insight:
  The same pipeline is applied identically to each dataset,
  making it easy to compare performance fairly.

Next: U07_multi_source.py - Multi-source data (NIR + other sensors)
""")

if args.show:
    plt.show()
