"""
U25 - SHAP Basics: Model Explainability
=======================================

Understand model predictions using SHAP (SHapley Additive exPlanations).

This tutorial covers:

* Why SHAP for model interpretation
* Running SHAP analysis with runner.explain()
* Spectral, waterfall, and beeswarm visualizations
* Binning and aggregation for spectral data

Prerequisites
-------------
Complete :ref:`U24_sklearn_integration` first.
Requires: pip install shap

Next Steps
----------
See :ref:`U26_shap_sklearn` for SHAP with sklearn wrapper.

Duration: ~5 minutes
Difficulty: ★★☆☆☆
"""

# Standard library imports
import argparse

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import Gaussian, SavitzkyGolay, StandardNormalVariate
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U25 SHAP Basics Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Why SHAP?
# =============================================================================
print("\n" + "=" * 60)
print("U25 - SHAP Basics: Model Explainability")
print("=" * 60)

print("""
SHAP explains model predictions:

  📊 WHAT SHAP TELLS YOU
     - Which wavelengths influence predictions
     - Direction of influence (positive/negative)
     - Importance across the spectrum

  🎯 NIRS-SPECIFIC VISUALIZATIONS
     - Spectral plot: SHAP values across wavelengths
     - Waterfall: Individual sample explanation
     - Beeswarm: Feature importance distribution

  🔧 KEY CONCEPTS
     - SHAP value: Contribution of each feature
     - Positive value: Increases prediction
     - Negative value: Decreases prediction
""")


# =============================================================================
# Section 2: Train a Model
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Train a Model")
print("-" * 60)

# Define pipeline
pipeline = [
    {"y_processing": MinMaxScaler},
    MinMaxScaler,
    Gaussian,
    SavitzkyGolay,
    StandardNormalVariate,
    PLSRegression(n_components=16),
]

pipeline_config = PipelineConfigs(pipeline, "U25_SHAP")
dataset_config = DatasetConfigs("sample_data/regression_2")

print("Training model...")
runner = PipelineRunner(save_artifacts=True, verbose=0)
predictions, _ = runner.run(pipeline_config, dataset_config)

# Get best model
best_prediction = predictions.top(
    n=1,
    rank_metric='rmse',
    rank_partition="test"
)[0]

print(f"\nBest model: {best_prediction['model_name']}")
print(f"RMSE: {best_prediction['rmse']:.4f}")


# =============================================================================
# Section 3: Basic SHAP Analysis
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Basic SHAP Analysis")
print("-" * 60)

print("""
Run SHAP analysis with runner.explain().
""")

# Simple SHAP configuration
shap_params = {
    'n_samples': 100,  # Number of samples to explain
    'explainer_type': 'auto',  # Auto-detect best explainer
    'visualizations': ['spectral', 'waterfall'],
}

print("Running SHAP analysis (this may take a moment)...")
shap_results, output_dir = runner.explain(
    best_prediction,
    dataset_config,
    shap_params=shap_params,
    plots_visible=args.plots
)

print(f"\n✓ SHAP analysis complete")
print(f"  Output directory: {output_dir}")


# =============================================================================
# Section 4: Advanced SHAP Configuration
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Advanced SHAP Configuration")
print("-" * 60)

print("""
Customize binning for different visualizations.
For spectral data with many wavelengths, binning groups
features into interpretable regions.
""")

# Advanced SHAP configuration with per-visualization settings
shap_params_advanced = {
    'n_samples': 200,
    'explainer_type': 'auto',  # Options: 'auto', 'tree', 'kernel', 'deep', 'linear'
    'visualizations': ['spectral', 'waterfall', 'beeswarm'],

    # Different bin sizes for each visualization
    'bin_size': {
        'spectral': 10,      # Fine-grained for spectral overview
        'waterfall': 20,     # Coarser for waterfall (fewer bars)
        'beeswarm': 20       # Medium for beeswarm
    },

    # Stride (overlap) for each visualization
    'bin_stride': {
        'spectral': 5,       # 50% overlap
        'waterfall': 10,     # 50% overlap
        'beeswarm': 20       # No overlap
    },

    # Aggregation method for each visualization
    'bin_aggregation': {
        'spectral': 'mean',   # Average importance
        'waterfall': 'mean',  # Average per region
        'beeswarm': 'mean'    # Average importance
    }
}

print("Running advanced SHAP analysis...")
shap_results_adv, output_dir_adv = runner.explain(
    best_prediction,
    dataset_config,
    shap_params=shap_params_advanced,
    plots_visible=args.plots
)

print(f"\n✓ Advanced SHAP analysis complete")


# =============================================================================
# Section 5: Understanding SHAP Visualizations
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Understanding SHAP Visualizations")
print("-" * 60)

print("""
Each visualization serves a different purpose:

  📊 SPECTRAL PLOT
     Shows SHAP values across the spectrum.
     - X-axis: Wavelength/feature index
     - Y-axis: SHAP value (importance)
     - Identifies key spectral regions

  📉 WATERFALL PLOT
     Explains a single prediction step-by-step.
     - Shows cumulative contribution
     - From base value to final prediction
     - Best for individual sample analysis

  🐝 BEESWARM PLOT
     Shows SHAP distribution for all samples.
     - Each dot is a sample
     - Color indicates feature value
     - Reveals feature importance patterns
""")


# =============================================================================
# Section 6: SHAP Configuration Reference
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: SHAP Configuration Reference")
print("-" * 60)

print("""
SHAP Parameter Reference:

  n_samples: int
      Number of samples to explain (default: 100)
      Higher = more accurate, slower

  explainer_type: str
      'auto': Auto-detect best explainer
      'tree': For tree-based models (RF, GBR)
      'kernel': Universal (slower)
      'linear': For linear models (PLS, Ridge)
      'deep': For neural networks

  visualizations: list
      List of plots to generate:
      - 'spectral': SHAP across wavelengths
      - 'waterfall': Single sample breakdown
      - 'beeswarm': Distribution summary

  bin_size: int or dict
      Number of features per bin
      Use dict for per-visualization settings

  bin_stride: int or dict
      Step between bins (overlap control)
      stride < bin_size creates overlap

  bin_aggregation: str or dict
      'mean': Average SHAP in bin
      'sum': Total SHAP in bin
      'mean_abs': Average absolute SHAP
      'sum_abs': Total absolute SHAP
""")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
SHAP Analysis Workflow:

  1. TRAIN MODEL:
     runner = PipelineRunner(save_artifacts=True)
     predictions, _ = runner.run(pipeline_config, dataset_config)
     best = predictions.top(1)[0]

  2. BASIC SHAP ANALYSIS:
     shap_params = {
         'n_samples': 100,
         'explainer_type': 'auto',
         'visualizations': ['spectral', 'waterfall']
     }
     shap_results, output_dir = runner.explain(
         best, dataset_config, shap_params=shap_params
     )

  3. ADVANCED CONFIGURATION:
     shap_params = {
         'n_samples': 200,
         'visualizations': ['spectral', 'waterfall', 'beeswarm'],
         'bin_size': {'spectral': 10, 'waterfall': 20},
         'bin_stride': {'spectral': 5, 'waterfall': 10},
         'bin_aggregation': {'spectral': 'mean', 'waterfall': 'mean'}
     }

Key Insights:
  - Spectral plot shows important wavelength regions
  - Waterfall explains individual predictions
  - Beeswarm reveals feature patterns
  - Binning makes high-dimensional data interpretable

Next: U26_shap_sklearn.py - SHAP with sklearn wrapper
""")
