"""
U27 - Feature Selection: CARS and MC-UVE
========================================

Select informative wavelengths for improved models.

This tutorial covers:

* CARS (Competitive Adaptive Reweighted Sampling)
* MC-UVE (Monte-Carlo Uninformative Variable Elimination)
* Comparing selection methods
* Integration in pipelines

Prerequisites
-------------
Complete :ref:`U26_shap_sklearn` first.

Duration: ~5 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import (
    FirstDerivative,
    StandardNormalVariate,
    CARS,
    MCUVE
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U27 Feature Selection Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Why Feature Selection?
# =============================================================================
print("\n" + "=" * 60)
print("U27 - Feature Selection: CARS and MC-UVE")
print("=" * 60)

print("""
Feature selection for NIRS data:

  📊 BENEFITS
     - Reduces dimensionality
     - Improves model interpretability
     - Reduces overfitting
     - Identifies key spectral regions

  🔧 METHODS
     - CARS: Adaptive wavelength selection
     - MC-UVE: Noise variable elimination

  🎯 USE CASES
     - High-dimensional spectral data
     - Interpretable models
     - Sensor optimization
     - Variable importance analysis
""")


# =============================================================================
# Section 2: CARS Feature Selection
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: CARS (Competitive Adaptive Reweighted Sampling)")
print("-" * 60)

print("""
CARS iteratively selects wavelengths based on:
  - PLS regression coefficient magnitudes
  - Exponential decay of variable count
  - Cross-validation RMSECV optimization
""")

# Build pipeline with CARS
cars_pipeline = [
    # Preprocessing
    {"y_processing": MinMaxScaler()},
    {"feature_augmentation": [FirstDerivative, StandardNormalVariate]},

    # CARS feature selection
    CARS(
        n_components=10,            # PLS components for internal model
        n_sampling_runs=50,         # Number of Monte-Carlo runs
        n_variables_ratio_end=0.2,  # Final ratio of variables to keep
        cv_folds=5,                 # Cross-validation folds
        random_state=42             # For reproducibility
    ),

    MinMaxScaler(),

    # Show spectra after selection
    "chart_2d",

    # Cross-validation
    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),

    # PLS regression
    PLSRegression(n_components=10),
]

# Load dataset
dataset_config = DatasetConfigs("sample_data/regression_3")
pipeline_config = PipelineConfigs(cars_pipeline, name="CARS_Selection")

# Run pipeline
print("\nRunning CARS pipeline...")
runner = PipelineRunner(
    save_artifacts=False,
    save_charts=False,
    verbose=1,
    plots_visible=args.plots
)
predictions_cars, _ = runner.run(pipeline_config, dataset_config)

# Get results
best_cars = predictions_cars.top(1, rank_metric='rmse')[0]
print(f"\nCARS Results:")
print(f"  RMSE: {best_cars['rmse']:.4f}")
print(f"  R²: {best_cars['r2']:.4f}")


# =============================================================================
# Section 3: MC-UVE Feature Selection
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: MC-UVE (Monte-Carlo Uninformative Variable Elimination)")
print("-" * 60)

print("""
MC-UVE identifies uninformative variables by:
  - Comparing real variables to random noise
  - Measuring coefficient stability under bootstrap
  - Eliminating variables with noise-like behavior
""")

# Build pipeline with MC-UVE
mcuve_pipeline = [
    # Preprocessing
    {"y_processing": MinMaxScaler()},
    {"feature_augmentation": [FirstDerivative, StandardNormalVariate]},

    # MC-UVE feature selection
    MCUVE(
        n_components=10,           # PLS components for internal model
        n_iterations=100,          # Number of bootstrap iterations
        threshold_method='auto',   # Automatic threshold selection
        random_state=42            # For reproducibility
    ),

    MinMaxScaler(),

    # Show spectra after selection
    "chart_2d",

    # Cross-validation
    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),

    # PLS regression
    PLSRegression(n_components=10),
]

pipeline_config_mcuve = PipelineConfigs(mcuve_pipeline, name="MCUVE_Selection")

# Run pipeline
print("\nRunning MC-UVE pipeline...")
predictions_mcuve, _ = runner.run(pipeline_config_mcuve, dataset_config)

# Get results
best_mcuve = predictions_mcuve.top(1, rank_metric='rmse')[0]
print(f"\nMC-UVE Results:")
print(f"  RMSE: {best_mcuve['rmse']:.4f}")
print(f"  R²: {best_mcuve['r2']:.4f}")


# =============================================================================
# Section 4: Baseline (No Selection)
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Baseline (No Feature Selection)")
print("-" * 60)

# Baseline pipeline without feature selection
baseline_pipeline = [
    {"y_processing": MinMaxScaler()},
    {"feature_augmentation": [FirstDerivative, StandardNormalVariate]},
    MinMaxScaler(),

    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
    PLSRegression(n_components=10),
]

pipeline_config_baseline = PipelineConfigs(baseline_pipeline, name="Baseline")

print("Running baseline pipeline...")
predictions_baseline, _ = runner.run(pipeline_config_baseline, dataset_config)

best_baseline = predictions_baseline.top(1, rank_metric='rmse')[0]
print(f"\nBaseline Results:")
print(f"  RMSE: {best_baseline['rmse']:.4f}")
print(f"  R²: {best_baseline['r2']:.4f}")


# =============================================================================
# Section 5: Compare Methods
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Compare Methods")
print("-" * 60)

print("""
Comparison of feature selection methods:
""")

results = [
    ("Baseline", best_baseline['rmse'], best_baseline['r2']),
    ("CARS", best_cars['rmse'], best_cars['r2']),
    ("MC-UVE", best_mcuve['rmse'], best_mcuve['r2']),
]

# Sort by RMSE
results_sorted = sorted(results, key=lambda x: x[1])

print(f"{'Method':<15} {'RMSE':>10} {'R²':>10}")
print("-" * 37)
for name, rmse, r2 in results_sorted:
    marker = "★" if name == results_sorted[0][0] else " "
    print(f"{marker}{name:<14} {rmse:>10.4f} {r2:>10.4f}")

best_method = results_sorted[0][0]
print(f"\nBest method: {best_method}")


# =============================================================================
# Section 6: CARS and MC-UVE Parameters
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Parameter Reference")
print("-" * 60)

print("""
CARS Parameters:
  n_components: int
      Number of PLS components (default: 10)

  n_sampling_runs: int
      Number of Monte-Carlo sampling runs (default: 50)
      Higher = more stable, slower

  n_variables_ratio_end: float
      Final ratio of variables to keep (default: 0.2)
      0.2 = keep 20% of original variables

  cv_folds: int
      Number of cross-validation folds (default: 5)

  random_state: int
      Random seed for reproducibility

MC-UVE Parameters:
  n_components: int
      Number of PLS components (default: 10)

  n_iterations: int
      Number of bootstrap iterations (default: 100)
      Higher = more stable, slower

  threshold_method: str
      'auto': Automatic threshold selection
      'percentile': Use percentile-based threshold
      float: Fixed threshold value

  random_state: int
      Random seed for reproducibility
""")


# =============================================================================
# Section 7: When to Use Each Method
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: When to Use Each Method")
print("-" * 60)

print("""
CARS (Competitive Adaptive Reweighted Sampling):
  ✓ Aggressive feature reduction
  ✓ Good for very high-dimensional data
  ✓ Optimizes for prediction accuracy
  ✓ Use when speed is less important

MC-UVE (Monte-Carlo Uninformative Variable Elimination):
  ✓ Conservative selection
  ✓ Good for interpretability
  ✓ Removes noise-like variables
  ✓ Use when stability is important

General Guidelines:
  - Start with baseline (no selection)
  - Try both methods and compare
  - Use CARS for aggressive reduction
  - Use MC-UVE for noise removal
  - Combine with domain knowledge
""")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Feature Selection Workflow:

  1. CARS SELECTION:
     pipeline = [
         CARS(
             n_components=10,
             n_sampling_runs=50,
             n_variables_ratio_end=0.2,
             cv_folds=5
         ),
         ShuffleSplit(n_splits=3),
         PLSRegression(n_components=10)
     ]

  2. MC-UVE SELECTION:
     pipeline = [
         MCUVE(
             n_components=10,
             n_iterations=100,
             threshold_method='auto'
         ),
         ShuffleSplit(n_splits=3),
         PLSRegression(n_components=10)
     ]

  3. COMPARE RESULTS:
     - Run baseline (no selection)
     - Run with CARS
     - Run with MC-UVE
     - Compare RMSE and R²

Method Comparison:
  CARS:
    + Aggressive reduction (keep 10-30%)
    + Optimizes for RMSECV
    - Slower computation
    - Less stable with small datasets

  MC-UVE:
    + Conservative (removes noise)
    + More stable
    + Better interpretability
    - Less aggressive reduction

Best Practice:
  1. Start with no selection
  2. Apply preprocessing first
  3. Try both methods
  4. Validate on held-out test set
  5. Consider domain knowledge

This completes the User Examples series!
Congratulations on learning nirs4all!
""")
