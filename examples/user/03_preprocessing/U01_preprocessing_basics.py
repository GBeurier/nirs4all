"""
U01 - Preprocessing Basics: NIRS-Specific Transformations
==========================================================

Overview of standard NIRS preprocessing techniques.

This tutorial covers:

* Scatter correction: SNV, MSC
* Baseline correction: Detrend
* Derivatives: First, Second, Savitzky-Golay
* Smoothing: Gaussian, Savitzky-Golay
* Wavelet: Haar

Prerequisites
-------------
Complete the data handling examples first.

Next Steps
----------
See :ref:`U02_feature_augmentation` for preprocessing exploration.

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
    # Baseline correction
    Detrend,
    # Derivatives
    FirstDerivative,
    # Smoothing
    Gaussian,
    # Wavelet
    Haar,
    MultiplicativeScatterCorrection,
    SavitzkyGolay,
    SecondDerivative,
    # Scatter correction
    StandardNormalVariate,
)
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U01 Preprocessing Basics Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Section 1: Overview of NIRS Preprocessing
# =============================================================================
print("\n" + "=" * 60)
print("U01 - NIRS Preprocessing Techniques")
print("=" * 60)

print("""
NIRS preprocessing addresses common spectral issues:

  📊 SCATTER CORRECTION
     SNV (StandardNormalVariate)  - Per-sample mean-centering and scaling
     MSC (MultiplicativeScatterCorrection) - Regression-based correction

  📈 BASELINE CORRECTION
     Detrend - Remove polynomial baseline drift

  📉 DERIVATIVES
     FirstDerivative  - Enhance peaks, remove constant baseline
     SecondDerivative - Enhance peaks more, remove linear baseline
     SavitzkyGolay    - Smoothed derivatives using polynomial fitting

  🔊 SMOOTHING
     Gaussian     - Gaussian convolution smoothing
     SavitzkyGolay - Polynomial smoothing (also does derivatives)

  🌊 WAVELET
     Haar - Haar wavelet transform for denoising
""")

# =============================================================================
# Section 2: Scatter Correction
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Scatter Correction")
print("-" * 60)

# Pipeline with SNV
pipeline_snv = [
    "chart_2d",
    StandardNormalVariate(),
    "chart_2d",
    ShuffleSplit(n_splits=2),
    {"model": PLSRegression(n_components=10)},
]

result_snv = nirs4all.run(
    pipeline=pipeline_snv,
    dataset="sample_data/regression",
    name="SNV",
    verbose=0,
    plots_visible=args.plots
)
print(f"   SNV - RMSE: {result_snv.best_rmse:.4f}")

# Pipeline with MSC
pipeline_msc = [
    "chart_2d",
    MultiplicativeScatterCorrection(),
    "chart_2d",
    ShuffleSplit(n_splits=2),
    {"model": PLSRegression(n_components=10)},
]

result_msc = nirs4all.run(
    pipeline=pipeline_msc,
    dataset="sample_data/regression",
    name="MSC",
    verbose=0,
    plots_visible=args.plots
)
print(f"   MSC - RMSE: {result_msc.best_rmse:.4f}")

# =============================================================================
# Section 3: Derivatives
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Derivatives")
print("-" * 60)

# First derivative
pipeline_d1 = [
    FirstDerivative(),
    ShuffleSplit(n_splits=2),
    {"model": PLSRegression(n_components=10)},
]
result_d1 = nirs4all.run(
    pipeline=pipeline_d1,
    dataset="sample_data/regression",
    name="FirstDeriv",
    verbose=0
)
print(f"   FirstDerivative - RMSE: {result_d1.best_rmse:.4f}")

# Second derivative
pipeline_d2 = [
    SecondDerivative(),
    ShuffleSplit(n_splits=2),
    {"model": PLSRegression(n_components=10)},
]
result_d2 = nirs4all.run(
    pipeline=pipeline_d2,
    dataset="sample_data/regression",
    name="SecondDeriv",
    verbose=0
)
print(f"   SecondDerivative - RMSE: {result_d2.best_rmse:.4f}")

# Savitzky-Golay derivative
pipeline_sg = [
    SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
    ShuffleSplit(n_splits=2),
    {"model": PLSRegression(n_components=10)},
]
result_sg = nirs4all.run(
    pipeline=pipeline_sg,
    dataset="sample_data/regression",
    name="SavGol",
    verbose=0
)
print(f"   SavitzkyGolay (d1) - RMSE: {result_sg.best_rmse:.4f}")

# =============================================================================
# Section 4: Smoothing
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Smoothing")
print("-" * 60)

# Gaussian smoothing
pipeline_gauss = [
    Gaussian(sigma=2),
    ShuffleSplit(n_splits=2),
    {"model": PLSRegression(n_components=10)},
]
result_gauss = nirs4all.run(
    pipeline=pipeline_gauss,
    dataset="sample_data/regression",
    name="Gaussian",
    verbose=0
)
print(f"   Gaussian (sigma=2) - RMSE: {result_gauss.best_rmse:.4f}")

# Savitzky-Golay smoothing (deriv=0)
pipeline_sg_smooth = [
    SavitzkyGolay(window_length=11, polyorder=2, deriv=0),
    ShuffleSplit(n_splits=2),
    {"model": PLSRegression(n_components=10)},
]
result_sg_smooth = nirs4all.run(
    pipeline=pipeline_sg_smooth,
    dataset="sample_data/regression",
    name="SG_Smooth",
    verbose=0
)
print(f"   SavitzkyGolay (smooth) - RMSE: {result_sg_smooth.best_rmse:.4f}")

# =============================================================================
# Section 5: Combining Preprocessing Steps
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Combining Preprocessing Steps")
print("-" * 60)

# Common combination: SNV + First Derivative
pipeline_combined = [
    "chart_2d",
    StandardNormalVariate(),
    FirstDerivative(),
    "chart_2d",
    ShuffleSplit(n_splits=2),
    {"model": PLSRegression(n_components=10)},
]
result_combined = nirs4all.run(
    pipeline=pipeline_combined,
    dataset="sample_data/regression",
    name="SNV_D1",
    verbose=0,
    plots_visible=args.plots
)
print(f"   SNV + FirstDerivative - RMSE: {result_combined.best_rmse:.4f}")

# Detrend + MSC + Savitzky-Golay
pipeline_chain = [
    Detrend(),
    MultiplicativeScatterCorrection(),
    SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
    ShuffleSplit(n_splits=2),
    {"model": PLSRegression(n_components=10)},
]
result_chain = nirs4all.run(
    pipeline=pipeline_chain,
    dataset="sample_data/regression",
    name="Chain",
    verbose=0
)
print(f"   Detrend + MSC + SG(d1) - RMSE: {result_chain.best_rmse:.4f}")

# =============================================================================
# Section 6: Comparing All Methods
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Comparing All Methods")
print("-" * 60)

# Collect results
results = {
    'SNV': result_snv.best_rmse,
    'MSC': result_msc.best_rmse,
    'FirstDeriv': result_d1.best_rmse,
    'SecondDeriv': result_d2.best_rmse,
    'SavGol': result_sg.best_rmse,
    'Gaussian': result_gauss.best_rmse,
    'SG_Smooth': result_sg_smooth.best_rmse,
    'SNV+D1': result_combined.best_rmse,
    'Chain': result_chain.best_rmse,
}

# Sort by RMSE
sorted_results = sorted(results.items(), key=lambda x: x[1])

print("\nRanked by RMSE (best to worst):")
for i, (name, rmse) in enumerate(sorted_results, 1):
    print(f"   {i}. {name}: {rmse:.4f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
NIRS Preprocessing Operators:

  SCATTER CORRECTION (path length, scattering effects):
    StandardNormalVariate()      - SNV, per-sample normalization
    MultiplicativeScatterCorrection()  - MSC, regression-based

  BASELINE CORRECTION:
    Detrend()  - Remove polynomial drift

  DERIVATIVES (enhance peaks, remove baselines):
    FirstDerivative()   - d/dx, removes constant baseline
    SecondDerivative()  - d²/dx², removes linear baseline
    SavitzkyGolay(deriv=1)  - Smoothed derivative

  SMOOTHING (reduce noise):
    Gaussian(sigma=2)   - Gaussian convolution
    SavitzkyGolay(deriv=0, window_length=11)  - Polynomial smoothing

  WAVELET:
    Haar()  - Haar wavelet transform

Common Combinations:
  1. SNV + FirstDerivative (scatter + baseline)
  2. MSC + SavitzkyGolay (scatter + smooth derivative)
  3. Detrend + SNV + Gaussian (full preprocessing)

Tip: Use feature_augmentation to explore combinations automatically!

Next: U02_feature_augmentation.py - Automated preprocessing exploration
""")

if args.show:
    plt.show()
