"""
U04 - Visualization: Comprehensive Result Analysis
===================================================

A tour of all visualization options in nirs4all.

This tutorial covers:

* PredictionAnalyzer methods and options
* Heatmaps, candlestick charts, histograms
* Top-k comparison plots
* Ranking and display partition configuration

Prerequisites
-------------
Complete :ref:`U02_basic_regression` and :ref:`U03_basic_classification`.

Next Steps
----------
See :ref:`U05_flexible_inputs` for different data input formats.

Duration: ~3 minutes
Difficulty: ★★☆☆☆
"""

# Standard library imports
import argparse
import matplotlib.pyplot as plt

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import (
    Gaussian, SavitzkyGolay, StandardNormalVariate, Haar
)
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U04 Visualization Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Build Pipeline with Multiple Models
# =============================================================================
print("\n" + "=" * 60)
print("U04 - Comprehensive Visualization Tutorial")
print("=" * 60)

# Build pipeline with diverse models for interesting visualizations
pipeline = [
    # Preprocessing
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},

    # Feature augmentation
    {"feature_augmentation": [
        StandardNormalVariate,
        Gaussian,
        SavitzkyGolay,
        Haar,
    ]},

    # Cross-validation
    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),

    # Multiple models for comparison
    {"model": PLSRegression(n_components=5), "name": "PLS-5"},
    {"model": PLSRegression(n_components=10), "name": "PLS-10"},
    {"model": PLSRegression(n_components=15), "name": "PLS-15"},
    {"model": Ridge(alpha=1.0), "name": "Ridge"},
    {"model": ElasticNet(alpha=0.1), "name": "ElasticNet"},
]

print("\n📋 Training pipeline with 5 models on 2 datasets...")

# Train on multiple datasets for more visualization options
result = nirs4all.run(
    pipeline=pipeline,
    dataset=['sample_data/regression', 'sample_data/regression_2'],
    name="VisualizationDemo",
    verbose=1,
    plots_visible=False  # We'll create our own plots
)

predictions = result.predictions
print(f"   Generated {result.num_predictions} predictions")


# =============================================================================
# Section 2: PredictionAnalyzer Basics
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: PredictionAnalyzer Basics")
print("-" * 60)

# Create the analyzer
analyzer = PredictionAnalyzer(predictions)

print("""
PredictionAnalyzer provides:
  • plot_top_k()       - Compare top K models side by side
  • plot_heatmap()     - 2D grid comparison
  • plot_candlestick() - Performance distribution (box plot style)
  • plot_histogram()   - Score distribution
  • plot_confusion_matrix() - For classification tasks
""")


# =============================================================================
# Section 3: Top-K Comparison Plots
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Top-K Comparison Plots")
print("-" * 60)

# Basic top-k plot
fig1 = analyzer.plot_top_k(k=3, rank_metric='rmse')
print("   ✓ plot_top_k(k=3, rank_metric='rmse')")
print("     Shows predicted vs actual for top 3 models by RMSE")

# Top-k ranked by test partition
fig2 = analyzer.plot_top_k(k=3, rank_metric='rmse', rank_partition='test')
print("   ✓ plot_top_k(k=3, rank_partition='test')")
print("     Ranks models by test RMSE (not validation)")

# Top-k ranked by R²
fig3 = analyzer.plot_top_k(k=3, rank_metric='r2', rank_partition='val')
print("   ✓ plot_top_k(k=3, rank_metric='r2')")
print("     Ranks models by R² instead of RMSE")


# =============================================================================
# Section 4: Heatmap Visualizations
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Heatmap Visualizations")
print("-" * 60)

# Heatmap: model vs preprocessing
fig4 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="preprocessings",
    rank_metric="rmse",
    display_metric="rmse",
)
print("   ✓ Heatmap: model_name vs preprocessings")

# Heatmap: model vs dataset
fig5 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="dataset_name",
    rank_metric="rmse",
    display_metric="r2",  # Display R² but rank by RMSE
)
print("   ✓ Heatmap: model_name vs dataset_name")
print("     Note: Ranking by RMSE but displaying R² values")

# Heatmap: model vs fold
fig6 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="fold_id",
    aggregation='best',
    show_counts=True,  # Show count in each cell
)
print("   ✓ Heatmap: model_name vs fold_id (with counts)")


# =============================================================================
# Section 5: Candlestick Charts
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Candlestick Charts")
print("-" * 60)

# Candlestick by model
fig7 = analyzer.plot_candlestick(
    variable="model_name",
    display_metric='rmse',
)
print("   ✓ Candlestick: performance distribution per model")

# Candlestick by dataset
fig8 = analyzer.plot_candlestick(
    variable="dataset_name",
    display_metric='r2',
)
print("   ✓ Candlestick: R² distribution per dataset")

# Candlestick by fold
fig9 = analyzer.plot_candlestick(
    variable="fold_id",
    display_metric='rmse',
)
print("   ✓ Candlestick: RMSE distribution per fold")


# =============================================================================
# Section 6: Histograms
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Histograms")
print("-" * 60)

# Histogram of RMSE
fig10 = analyzer.plot_histogram(display_metric='rmse')
print("   ✓ Histogram: RMSE distribution")

# Histogram of R²
fig11 = analyzer.plot_histogram(display_metric='r2')
print("   ✓ Histogram: R² distribution")

# Histogram of MAE
fig12 = analyzer.plot_histogram(display_metric='mae')
print("   ✓ Histogram: MAE distribution")


# =============================================================================
# Section 7: Understanding Rank vs Display
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Ranking vs Display Partitions")
print("-" * 60)

print("""
Key concept: Separate ranking from display

  rank_metric + rank_partition  → Determines which models are "best"
  display_metric + display_partition → What values to show

Example: Rank by validation RMSE, display test R²
  analyzer.plot_heatmap(
      rank_metric='rmse',
      rank_partition='val',
      display_metric='r2',
      display_partition='test'
  )

Aggregation options:
  'best'   - Show best score for each cell
  'mean'   - Show mean score
  'median' - Show median score
""")

# Demonstration
fig13 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="preprocessings",
    rank_metric='rmse',
    rank_partition='val',
    display_metric='rmse',
    display_partition='test',
    aggregation='best'
)
print("   ✓ Created heatmap: rank by val, display test")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
PredictionAnalyzer Visualization Methods:

  plot_top_k(k, rank_metric, rank_partition)
      Compare top K models with scatter plots

  plot_heatmap(x_var, y_var, rank_metric, display_metric, ...)
      2D grid: model_name, dataset_name, preprocessings, fold_id

  plot_candlestick(variable, display_metric)
      Distribution box plots per category

  plot_histogram(display_metric)
      Score distribution across all predictions

  plot_confusion_matrix(k, rank_metric)  [classification only]
      Confusion matrices for top K classifiers

Key parameters:
  rank_metric    - Metric for ranking: 'rmse', 'r2', 'mae', 'mse'
  display_metric - Metric to display (can differ from rank)
  rank_partition - Partition for ranking: 'val', 'test', 'train'
  display_partition - Partition to display
  aggregation    - 'best', 'mean', 'median'

Next: U05_flexible_inputs.py - Different data input formats
""")

if args.show:
    plt.show()
