"""
U04 - Wavelength Handling: Resampling and Unit Conversion
==========================================================

Handle wavelength grids: interpolation, downsampling, and unit conversion.

This tutorial covers:

* Resampler operator for wavelength interpolation
* Downsampling to fewer wavelengths
* Focusing on specific spectral regions
* Wavelength header units (nm, cm⁻¹)

Prerequisites
-------------
Complete :ref:`U03_multi_source` first.

Next Steps
----------
See :ref:`03_preprocessing/U01_preprocessing_basics` for NIRS preprocessing techniques.

Duration: ~3 minutes
Difficulty: ★★☆☆☆
"""

# Standard library imports
import argparse
import numpy as np

# Third-party imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression

# NIRS4All imports
import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import Resampler, StandardNormalVariate

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U04 Wavelength Handling Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Understanding Wavelength Resampling
# =============================================================================
print("\n" + "=" * 60)
print("U04 - Wavelength Handling")
print("=" * 60)

print("""
Wavelength resampling is useful for:

  • Combining data from different instruments (different wavelength grids)
  • Standardizing wavelength resolution across datasets
  • Focusing on specific spectral regions of interest
  • Reducing dimensionality while preserving spectral shape

The Resampler operator uses scipy interpolation to estimate spectral
values at new wavelengths.
""")


# =============================================================================
# Section 2: Get Reference Wavelengths
# =============================================================================
print("\n" + "-" * 60)
print("Loading Reference Dataset")
print("-" * 60)

# Get wavelengths from a reference dataset
ref_config = DatasetConfigs("sample_data/regression_2")
ref_dataset = list(ref_config.iter_datasets())[0]
target_wavelengths = ref_dataset.float_headers(0)

print(f"Reference dataset wavelengths:")
print(f"   Count: {len(target_wavelengths)} points")
print(f"   Range: {target_wavelengths[0]:.1f} to {target_wavelengths[-1]:.1f}")


# =============================================================================
# Section 3: Resample to Match Another Dataset
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Resample to Match Reference Dataset")
print("-" * 60)

pipeline_match = [
    # Show original spectra
    "chart_2d",

    # Resample to match reference wavelengths
    Resampler(target_wavelengths=target_wavelengths, method='linear'),

    # Show resampled spectra
    "chart_2d",
]

result1 = nirs4all.run(
    pipeline=pipeline_match,
    dataset="sample_data/regression_3",
    name="ResampleMatch",
    verbose=1,
    plots_visible=args.plots
)

print(f"   ✓ Resampled spectra to match reference: {len(target_wavelengths)} points")


# =============================================================================
# Section 4: Downsample to Fewer Points
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Downsample to 10 Wavelengths")
print("-" * 60)

# Create 10 evenly spaced points
# Note: Keep order consistent (descending if original is descending)
target_wl_downsample = np.linspace(11012, 5966, 10)  # Descending

pipeline_downsample = [
    "chart_2d",
    Resampler(target_wavelengths=target_wl_downsample, method='linear'),
    "chart_2d",
]

result2 = nirs4all.run(
    pipeline=pipeline_downsample,
    dataset="sample_data/regression_3",
    name="Downsample",
    verbose=1,
    plots_visible=args.plots
)

print(f"   ✓ Downsampled from original to 10 points")
print(f"   Wavelengths: {target_wl_downsample}")


# =============================================================================
# Section 5: Focus on Specific Spectral Region
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Focus on Fingerprint Region")
print("-" * 60)

# Focus on a specific region with higher resolution
target_wl_region = np.linspace(9500, 7000, 50)  # 50 points in fingerprint region

pipeline_region = [
    "chart_2d",
    Resampler(target_wavelengths=target_wl_region, method='linear'),
    "chart_2d",
]

result3 = nirs4all.run(
    pipeline=pipeline_region,
    dataset="sample_data/regression_3",
    name="FingerprintRegion",
    verbose=1,
    plots_visible=args.plots
)

print(f"   ✓ Focused on region: 9500-7000 cm⁻¹ with 50 points")


# =============================================================================
# Section 6: Full Pipeline with Resampling
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Full Pipeline with Resampling")
print("-" * 60)

# Downsample then train a model
target_wl_model = np.linspace(11012, 5966, 50)  # 50 evenly spaced

pipeline_full = [
    # Downsample
    Resampler(target_wavelengths=target_wl_model, method='linear'),

    # Standard preprocessing
    MinMaxScaler(),
    StandardNormalVariate(),

    # Cross-validation
    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),

    # Model
    {"model": PLSRegression(n_components=10), "name": "PLS-10"},
]

result4 = nirs4all.run(
    pipeline=pipeline_full,
    dataset="sample_data/regression_3",
    name="ResampledModel",
    verbose=1,
    save_artifacts=True,
    plots_visible=args.plots
)

print(f"\n📊 Model trained on resampled data:")
print(f"   Original features: ~125 wavelengths")
print(f"   Resampled features: 50 wavelengths")
print(f"   Best RMSE: {result4.best_rmse:.4f}")
r2_value = result4.best_r2
print(f"   Best R²: {r2_value:.4f}" if not np.isnan(r2_value) else "   Best R²: (see test metrics)")


# =============================================================================
# Section 7: Interpolation Methods
# =============================================================================
print("\n" + "-" * 60)
print("Interpolation Methods")
print("-" * 60)

print("""
The Resampler supports several interpolation methods:

  'linear'    - Linear interpolation (default, fast)
  'cubic'     - Cubic spline (smooth but slower)
  'quadratic' - Quadratic interpolation
  'nearest'   - Nearest neighbor (fast, preserves discrete features)

Choose based on your spectral data characteristics:
  • Smooth spectra: 'linear' or 'cubic'
  • Discrete/step features: 'nearest'
  • High-frequency content: 'cubic' to preserve details
""")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Wavelength Handling with Resampler:

  1. Match another dataset's wavelengths:
     Resampler(target_wavelengths=other_wavelengths)

  2. Downsample to fewer points:
     target_wl = np.linspace(start, end, n_points)
     Resampler(target_wavelengths=target_wl)

  3. Focus on specific region:
     region_wl = np.linspace(9500, 7000, 50)
     Resampler(target_wavelengths=region_wl)

  4. Choose interpolation method:
     Resampler(..., method='cubic')

Key parameters:
  target_wavelengths  - Array of desired wavelengths
  method              - 'linear', 'cubic', 'quadratic', 'nearest'

Header unit configuration (in DatasetConfigs):
  DatasetConfigs(path, params={'header_unit': 'nm'})
  DatasetConfigs(path, params={'header_unit': 'cm-1'})

Use cases:
  • Instrument standardization
  • Transfer learning between instruments
  • Feature reduction with spectral awareness
  • Region-of-interest analysis

Next: See 03_preprocessing/U01_preprocessing_basics.py for NIRS preprocessing
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
