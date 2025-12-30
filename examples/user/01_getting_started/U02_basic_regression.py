"""
U02 - Basic Regression: NIRS Preprocessing and Model Comparison
================================================================

A complete regression pipeline with NIRS-specific preprocessing and visualization.

This tutorial covers:

* NIRS-specific preprocessing (SNV, Detrend, Derivatives, Gaussian)
* Feature augmentation to explore preprocessing combinations
* Using ``PredictionAnalyzer`` for result visualization
* Comparing models with different n_components

Prerequisites
-------------
Complete :ref:`U01_hello_world` first.

Next Steps
----------
See :ref:`U03_basic_classification` for classification tasks.

Duration: ~3 minutes
Difficulty: ★★☆☆☆
"""

# Standard library imports
import argparse
import matplotlib.pyplot as plt

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import (
    Detrend, FirstDerivative, Gaussian, SavitzkyGolay,
    StandardNormalVariate, Haar, MultiplicativeScatterCorrection
)
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U02 Basic Regression Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Define Preprocessing Options
# =============================================================================
print("\n" + "=" * 60)
print("U02 - Basic Regression Pipeline")
print("=" * 60)

print("\n📋 NIRS Preprocessing Options:")
print("   • StandardNormalVariate (SNV) - scatter correction")
print("   • MultiplicativeScatterCorrection (MSC) - scatter correction")
print("   • Detrend - baseline correction")
print("   • FirstDerivative - removes baseline, enhances peaks")
print("   • SavitzkyGolay - smoothing with differentiation")
print("   • Gaussian - smoothing filter")
print("   • Haar - wavelet transform")


# =============================================================================
# Section 2: Build the Pipeline
# =============================================================================
print("\n" + "-" * 60)
print("Building Pipeline")
print("-" * 60)

# Build the pipeline with feature augmentation
pipeline = [
    # Feature scaling
    MinMaxScaler(),

    # Target scaling
    {"y_processing": MinMaxScaler()},

    # Feature augmentation: generate preprocessing combinations
    # This creates multiple variants with different preprocessing chains
    {
        "feature_augmentation": {
            "_or_": [Detrend, FirstDerivative, Gaussian, SavitzkyGolay, Haar],
            "pick": 2,      # Pick 2 at a time
            "count": 3      # Generate 3 combinations
        }
    },

    # Visualization (only shown if --plots is passed)
    "chart_2d",

    # Cross-validation
    ShuffleSplit(n_splits=3, test_size=0.25),
]

# Add PLS models with different numbers of components
for n_components in [5, 10, 15]:
    pipeline.append({
        "name": f"PLS-{n_components}",
        "model": PLSRegression(n_components=n_components)
    })

print("Pipeline includes:")
print("   • MinMaxScaler for feature and target scaling")
print("   • Feature augmentation with 3 preprocessing combinations")
print("   • 3 PLS models (5, 10, 15 components)")


# =============================================================================
# Section 3: Train the Pipeline
# =============================================================================
print("\n" + "-" * 60)
print("Training Pipeline")
print("-" * 60)

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="BasicRegression",
    verbose=1,
    save_artifacts=True,
    plots_visible=args.plots
)

print(f"\n📊 Training complete!")
print(f"   Generated {result.num_predictions} predictions")
print(f"   Best Score (MSE): {result.best_score:.4f}")


# =============================================================================
# Section 4: Display Top Models
# =============================================================================
print("\n" + "-" * 60)
print("Top 5 Models")
print("-" * 60)

# Use display_metrics to get RMSE and R² values
for i, pred in enumerate(result.top(n=5, display_metrics=['rmse', 'r2']), 1):
    model_name = pred.get('model_name', 'unknown')
    preproc = pred.get('preprocessings', 'N/A')
    rmse = pred.get('rmse', 0)
    r2 = pred.get('r2', 0)
    print(f"{i}. {model_name} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    print(f"   Preprocessing: {preproc}")


# =============================================================================
# Section 5: Visualize Results
# =============================================================================
print("\n" + "-" * 60)
print("Creating Visualizations")
print("-" * 60)

# Create the analyzer
analyzer = PredictionAnalyzer(result.predictions)

# Plot top-k comparison
fig1 = analyzer.plot_top_k(k=3, rank_metric='rmse')
print("   ✓ Created top-k comparison plot")

# Plot heatmap: models vs preprocessing
fig2 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="preprocessings",
    aggregation='best',
    rank_metric="rmse",
    rank_partition="val",
    display_metric="rmse",
    display_partition="test"
)
print("   ✓ Created heatmap: models vs preprocessing")

# Plot candlestick chart for model performance distribution
fig3 = analyzer.plot_candlestick(
    variable="model_name",
    partition="test"
)
print("   ✓ Created candlestick chart")

# Plot histogram of RMSE values
fig4 = analyzer.plot_histogram(partition="test")
print("   ✓ Created histogram")


# =============================================================================
# Validation: Ensure results are valid (no NaN metrics)
# =============================================================================
import sys
import os
# Add examples dir to find example_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from example_utils import validate_result

# Validate results - will exit with code 1 if NaN metrics found
validate_result(result, "BasicRegression")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. NIRS-specific preprocessing (SNV, MSC, derivatives, smoothing)
2. Feature augmentation to explore preprocessing combinations
3. PredictionAnalyzer for comprehensive visualizations
4. Comparing models with different hyperparameters

Key preprocessing options:
  SNV/MSC     - Scatter correction
  Detrend     - Baseline correction
  Derivatives - Enhance spectral features
  SavitzkyGolay/Gaussian - Smoothing

Key visualization methods:
  analyzer.plot_top_k()       - Compare top K models
  analyzer.plot_heatmap()     - 2D comparison
  analyzer.plot_candlestick() - Performance distribution
  analyzer.plot_histogram()   - Score distribution

Next: U03_basic_classification.py - Classification with Random Forest
""")

if args.show:
    plt.show()
