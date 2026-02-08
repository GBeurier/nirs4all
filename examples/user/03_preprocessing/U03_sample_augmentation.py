"""
U03 - Sample Augmentation: Data Augmentation for NIRS
======================================================

Augment training data with spectral transformations.

This tutorial covers:

* sample_augmentation for data augmentation
* Built-in spectral augmenters
* Balanced augmentation for class imbalance
* Visualization of augmented samples

Prerequisites
-------------
Complete :ref:`U01_preprocessing_basics` first.

Next Steps
----------
See :ref:`U04_signal_conversion` for absorbance/reflectance handling.

Duration: ~4 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, GroupKFold

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import (
    # Basic augmenters
    Rotate_Translate,
    GaussianAdditiveNoise,
    MultiplicativeNoise,

    # Spline-based
    Spline_Y_Perturbations,
    Spline_X_Simplification,

    # Baseline augmenters
    LinearBaselineDrift,
    PolynomialBaselineDrift,

    # Wavelength augmenters
    WavelengthShift,
    WavelengthStretch,
    SmoothMagnitudeWarp,

    # Advanced
    MixupAugmenter,
    ScatterSimulationMSC,
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U03 Sample Augmentation Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: What is Sample Augmentation?
# =============================================================================
print("\n" + "=" * 60)
print("U03 - Sample Augmentation")
print("=" * 60)

print("""
Sample Augmentation generates synthetic training samples by applying
realistic spectral transformations to your data.

Benefits:
  ✓ Increase effective training set size
  ✓ Improve model generalization
  ✓ Handle class imbalance
  ✓ Simulate measurement variability

Available Augmenters:

  📊 NOISE
     GaussianAdditiveNoise - Add Gaussian noise
     MultiplicativeNoise   - Multiplicative gain variations

  📈 GEOMETRY
     Rotate_Translate      - Rotate and shift spectra
     WavelengthShift       - Shift in wavelength axis
     WavelengthStretch     - Stretch/compress wavelength axis

  📉 BASELINE
     LinearBaselineDrift     - Add linear drift
     PolynomialBaselineDrift - Add polynomial drift

  🔧 SPLINE
     Spline_Y_Perturbations  - Smooth Y-axis perturbations
     Spline_X_Simplification - Reduce spectral resolution

  🎲 MIXING
     MixupAugmenter         - Linear interpolation between samples
     ScatterSimulationMSC   - Simulate scatter effects
""")


# =============================================================================
# Section 2: Basic Augmentation
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Basic Augmentation")
print("-" * 60)

print("""
Use sample_augmentation with a list of transformers.
Each original sample generates 'count' augmented samples.
""")

pipeline_basic = [
    # Show original data
    "fold_chart",

    # Augment: each sample generates 2 augmented versions
    {"sample_augmentation": {
        "transformers": [
            Rotate_Translate(p_range=2, y_factor=3),
            GaussianAdditiveNoise(sigma=0.01),
        ],
        "count": 2,
        "selection": "random",
        "random_state": 42,
    }},

    # Show augmented data
    "fold_chart",

    # Simple model
    ShuffleSplit(n_splits=2, random_state=42),
    {"model": PLSRegression(n_components=5)},
]

result_basic = nirs4all.run(
    pipeline=pipeline_basic,
    dataset="sample_data/regression",
    name="BasicAug",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nResult with augmentation: RMSE = {result_basic.best_rmse:.4f}")


# =============================================================================
# Section 3: Visualizing Augmentation Effects
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Visualizing Augmentation Effects")
print("-" * 60)

print("""
Use special chart keywords to visualize augmentation:
  augment_chart - Overlay original vs augmented
  augment_details_chart - Show each transformer separately
""")

pipeline_visual = [
    {"sample_augmentation": {
        "transformers": [
            Rotate_Translate(p_range=2, y_factor=3),
            GaussianAdditiveNoise(sigma=0.01),
            WavelengthShift(),
        ],
        "count": 2,
        "selection": "random",
        "random_state": 42,
    }},

    # Overlay: original (blue) vs augmented (orange)
    "augment_chart",

    # Details: each transformer shown separately
    "augment_details_chart",

    ShuffleSplit(n_splits=1),
    {"model": PLSRegression(n_components=5)},
]

result_visual = nirs4all.run(
    pipeline=pipeline_visual,
    dataset="sample_data/regression",
    name="VisualAug",
    verbose=0,
    plots_visible=args.plots
)

print("Charts generated (use --plots to view)")


# =============================================================================
# Section 4: Balanced Augmentation for Classification
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Balanced Augmentation for Classification")
print("-" * 60)

print("""
Use balance='y' to automatically balance classes.
The minority class is augmented to match the majority class.
""")

# Split configuration for classification
split_step = {"split": GroupKFold(n_splits=2), "group": "Sample_ID"}

pipeline_balanced = [
    "fold_chart",

    # Balanced augmentation: augment to match majority class
    {"sample_augmentation": {
        "transformers": [Rotate_Translate(p_range=2, y_factor=3)],
        "balance": "y",           # Balance by target
        "ref_percentage": 1.0,    # Match 100% of majority class
        "selection": "random",
        "random_state": 42,
    }},

    "fold_chart",
    split_step,
    {"model": RandomForestClassifier(n_estimators=5, random_state=42)},
]

result_balanced = nirs4all.run(
    pipeline=pipeline_balanced,
    dataset="sample_data/classification",
    name="BalancedAug",
    verbose=1,
    plots_visible=args.plots
)


# =============================================================================
# Section 5: Balanced Augmentation with Limits
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Balanced Augmentation with Limits")
print("-" * 60)

print("""
Control augmentation intensity with:
  target_size - Fixed number of samples per class
  max_factor  - Maximum augmentation ratio (e.g., 2x)
  ref_percentage - Percentage of majority class to target
""")

# Option 1: Fixed target size
pipeline_fixed = [
    {"sample_augmentation": {
        "transformers": [Rotate_Translate],
        "balance": "y",
        "target_size": 30,  # Each class gets exactly 30 samples
        "random_state": 42,
    }},
    split_step,
    {"model": RandomForestClassifier(n_estimators=50, random_state=42)},
]

result_fixed = nirs4all.run(
    pipeline=pipeline_fixed,
    dataset="sample_data/classification",
    name="FixedSize",
    verbose=0
)
accuracy_fixed = result_fixed.best_accuracy if hasattr(result_fixed, 'best_accuracy') and result_fixed.best_accuracy is not None else (1 - result_fixed.best_rmse if not np.isnan(result_fixed.best_rmse) else float('nan'))
print(f"   target_size=30 → Result: Accuracy = {100*accuracy_fixed:.1f}%" if not np.isnan(accuracy_fixed) else "   target_size=30 → Result: (see detailed metrics)")

# Option 2: Max factor
pipeline_maxfactor = [
    {"sample_augmentation": {
        "transformers": [Rotate_Translate],
        "balance": "y",
        "max_factor": 2.0,  # Max 2x augmentation
        "random_state": 42,
    }},
    split_step,
    {"model": RandomForestClassifier(n_estimators=50, random_state=42)},
]

result_maxfactor = nirs4all.run(
    pipeline=pipeline_maxfactor,
    dataset="sample_data/classification",
    name="MaxFactor",
    verbose=0
)
accuracy_maxfactor = result_maxfactor.best_accuracy if hasattr(result_maxfactor, 'best_accuracy') and result_maxfactor.best_accuracy is not None else (1 - result_maxfactor.best_rmse if not np.isnan(result_maxfactor.best_rmse) else float('nan'))
print(f"   max_factor=2.0 → Result: Accuracy = {100*accuracy_maxfactor:.1f}%" if not np.isnan(accuracy_maxfactor) else "   max_factor=2.0 → Result: (see detailed metrics)")


# =============================================================================
# Section 6: Regression Balancing with Binning
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Regression Balancing with Binning")
print("-" * 60)

print("""
For regression, use bins to create pseudo-classes for balancing.
  bins - Number of bins to create
  binning_strategy - 'equal_width' or 'quantile'
""")

pipeline_regression_balanced = [
    "fold_chart",

    {"sample_augmentation": {
        "transformers": [
            Rotate_Translate(p_range=2, y_factor=3),
            GaussianAdditiveNoise(sigma=0.01),
        ],
        "balance": "y",
        "bins": 5,                    # Create 5 bins from Y values
        "binning_strategy": "quantile",  # Equal population bins
        "ref_percentage": 0.8,        # Target 80% of largest bin
        "random_state": 42,
    }},

    "fold_chart",
    ShuffleSplit(n_splits=2, random_state=42),
    {"model": PLSRegression(n_components=10)},
]

result_reg_balanced = nirs4all.run(
    pipeline=pipeline_regression_balanced,
    dataset="sample_data/regression",
    name="RegBalanced",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nBalanced regression augmentation: RMSE = {result_reg_balanced.best_rmse:.4f}")


# =============================================================================
# Section 7: Comprehensive Augmentation Example
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Comprehensive Augmentation Pipeline")
print("-" * 60)

pipeline_comprehensive = [
    {"sample_augmentation": {
        "transformers": [
            # Geometric transforms
            Rotate_Translate(p_range=2, y_factor=3),
            WavelengthShift(),
            SmoothMagnitudeWarp(),

            # Noise
            GaussianAdditiveNoise(sigma=0.005),
            MultiplicativeNoise(sigma_gain=0.02),

            # Baseline
            LinearBaselineDrift(),

            # Spline
            Spline_Y_Perturbations(perturbation_intensity=0.003),

            # Mixing
            MixupAugmenter(),
        ],
        "count": 4,
        "selection": "random",
        "random_state": 42,
    }},

    "augment_chart",
    ShuffleSplit(n_splits=2, random_state=42),
    {"model": PLSRegression(n_components=10)},
]

result_comprehensive = nirs4all.run(
    pipeline=pipeline_comprehensive,
    dataset="sample_data/regression",
    name="Comprehensive",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nComprehensive augmentation: RMSE = {result_comprehensive.best_rmse:.4f}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Sample Augmentation Configuration:

  Basic Usage:
    {"sample_augmentation": {
        "transformers": [Rotate_Translate, GaussianAdditiveNoise(sigma=0.01)],
        "count": 2,              # Augmented samples per original
        "selection": "random",   # 'random' or 'all'
        "random_state": 42,
    }}

  Balanced Augmentation:
    {"sample_augmentation": {
        "transformers": [Rotate_Translate],
        "balance": "y",           # Balance by target
        "ref_percentage": 1.0,    # Match majority class
        # or: "target_size": 50   # Fixed count per class
        # or: "max_factor": 2.0   # Max augmentation ratio
    }}

  Regression Balancing:
    {"sample_augmentation": {
        "transformers": [...],
        "balance": "y",
        "bins": 5,                    # Number of bins
        "binning_strategy": "quantile",  # or 'equal_width'
    }}

Visualization Charts:
  "augment_chart"         - Original vs augmented overlay
  "augment_details_chart" - Each transformer shown separately

Available Augmenters:
  Rotate_Translate, GaussianAdditiveNoise, MultiplicativeNoise,
  LinearBaselineDrift, PolynomialBaselineDrift, WavelengthShift,
  WavelengthStretch, SmoothMagnitudeWarp, Spline_Y_Perturbations,
  Spline_X_Perturbations, MixupAugmenter, ScatterSimulationMSC, ...

Next: U04_signal_conversion.py - Absorbance/Reflectance handling
""")
