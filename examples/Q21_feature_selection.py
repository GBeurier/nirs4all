"""
Q21 Feature Selection - CARS and MC-UVE for Wavelength Selection
=================================================================
This example demonstrates the use of feature selection operators
(CARS and MC-UVE) to identify informative wavelengths in NIRS data.

Feature selection:
- Reduces dimensionality while preserving predictive power
- Identifies spectral regions relevant to the target property
- Improves model interpretability and reduces overfitting

Usage:
    python Q21_feature_selection.py --plots --show
"""

# Standard library imports
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q21 Feature Selection Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()


# Imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import (
    FirstDerivative,
    StandardNormalVariate,
    CARS,
    MCUVE
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer


# Example 1: CARS Feature Selection

print("=" * 70)
print("EXAMPLE 1: CARS (Competitive Adaptive Reweighted Sampling)")
print("=" * 70)
print()
print("CARS iteratively selects wavelengths based on:")
print("  - PLS regression coefficient magnitudes")
print("  - Exponential decay of variable count")
print("  - Cross-validation RMSECV optimization")
print()

# Build pipeline with CARS
cars_pipeline = [
    # Show original spectra (before selection)
    "chart_2d",

    # Standard preprocessing (on reduced features)
    {"y_processing": MinMaxScaler()},
    {"feature_augmentation": [FirstDerivative, StandardNormalVariate]},

    CARS(
        n_components=10,            # PLS components for internal model
        n_sampling_runs=50,         # Number of Monte-Carlo runs
        n_variables_ratio_end=0.2,  # Final ratio of variables to keep
        cv_folds=5,                 # Cross-validation folds
        random_state=42             # For reproducibility
    ),
    MinMaxScaler(),
    # Show spectra after CARS selection
    "chart_2d",

    # Cross-validation
    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),

    # PLS regression
    PLSRegression(n_components=10),
]

# Load regression dataset
dataset_config = DatasetConfigs("sample_data/regression_3")
pipeline_config = PipelineConfigs(cars_pipeline, name="CARS_Selection")

# Run pipeline
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
predictions_cars, _ = runner.run(pipeline_config, dataset_config)

# Analyze results
analyzer = PredictionAnalyzer(predictions_cars)
fig1 = analyzer.plot_top_k(k=1, rank_metric='rmse')

print()
print("CARS selected the most informative wavelengths for prediction.")
print("Check the 2D charts to see the feature reduction.")
print()


# Example 2: MC-UVE Feature Selection

print("=" * 70)
print("EXAMPLE 2: MC-UVE (Monte-Carlo Uninformative Variable Elimination)")
print("=" * 70)
print()
print("MC-UVE identifies uninformative variables by:")
print("  - Comparing real variables to random noise")
print("  - Measuring coefficient stability under bootstrap")
print("  - Eliminating variables with noise-like behavior")
print()

# Build pipeline with MC-UVE
mcuve_pipeline = [
    # Show original spectra
    "chart_2d",

    # Apply MC-UVE for wavelength selection

    {"y_processing": MinMaxScaler()},
    {"feature_augmentation": [FirstDerivative, StandardNormalVariate]},

     MCUVE(
        n_components=10,           # PLS components for internal model
        n_iterations=100,          # Number of bootstrap iterations
        threshold_method='auto',   # Automatic threshold selection
        random_state=42            # For reproducibility
    ),

    # Show spectra after MC-UVE selection
    "chart_2d",

    # Standard preprocessing
    MinMaxScaler(),

    # Cross-validation
    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),

    # PLS regression
    PLSRegression(n_components=10),
]

pipeline_config = PipelineConfigs(mcuve_pipeline, name="MCUVE_Selection")

# Run pipeline
predictions_mcuve, _ = runner.run(pipeline_config, dataset_config)

# Analyze results
analyzer = PredictionAnalyzer(predictions_mcuve)
fig2 = analyzer.plot_top_k(k=1, rank_metric='rmse')

print()
print("MC-UVE eliminated uninformative (noise-like) wavelengths.")
print("Check the 2D charts to see the feature reduction.")
print()


# Summary

print()
print("=" * 70)
print("SUMMARY: Feature Selection Methods")
print("=" * 70)
print()
print("CARS (Competitive Adaptive Reweighted Sampling):")
print("  - Iteratively reduces variables using exponential decay")
print("  - Uses weighted sampling based on PLS coefficients")
print("  - Selects optimal subset via RMSECV minimization")
print("  - Good for aggressive feature reduction")
print()
print("MC-UVE (Monte-Carlo Uninformative Variable Elimination):")
print("  - Compares real variables to noise variables")
print("  - Uses bootstrap stability as selection criterion")
print("  - More conservative, removes clearly uninformative features")
print("  - Good for interpretability and noise reduction")
print()
print("Both methods can be used before preprocessing to reduce")
print("the number of wavelengths while preserving predictive power.")
print("=" * 70)
