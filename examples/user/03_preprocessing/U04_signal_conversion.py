"""
U04 - Signal Conversion: Absorbance, Reflectance, and Kubelka-Munk
==================================================================

Convert between spectral representations.

This tutorial covers:

* Signal types: absorbance, reflectance, transmittance
* ToAbsorbance and FromAbsorbance converters
* Kubelka-Munk transformation
* Automatic signal type detection
* Using converters in pipelines

Prerequisites
-------------
Basic understanding of spectroscopy concepts.

Next Steps
----------
See :ref:`04_models/U01_multi_model` for comparing multiple models.

Duration: ~4 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms.signal_conversion import (
    ToAbsorbance,
    FromAbsorbance,
    KubelkaMunk,
    PercentToFraction,
    FractionToPercent,
    SignalTypeConverter,
)
from nirs4all.data.signal_type import detect_signal_type, SignalType

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U04 Signal Conversion Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Signal Types Overview
# =============================================================================
print("\n" + "=" * 60)
print("U04 - Signal Conversion")
print("=" * 60)

print("""
Spectral data can be represented in different forms:

  📊 REFLECTANCE (R)
     Fraction of light reflected: [0, 1] or [0, 100]%
     Raw measurement from many NIR instruments

  📈 ABSORBANCE (A)
     A = -log₁₀(R) or A = -log₁₀(T)
     Proportional to concentration (Beer-Lambert law)
     Range: typically [0, 3+]

  📉 TRANSMITTANCE (T)
     Fraction of light transmitted: [0, 1] or [0, 100]%
     Used for liquid/thin samples

  🔬 KUBELKA-MUNK F(R)
     F(R) = (1-R)² / (2R)
     Theory for scattering media (powders, granules)

Why convert?
  • Some algorithms work better with absorbance
  • Kubelka-Munk linearizes relationship for scattering samples
  • Standardize data from different instruments
""")


# =============================================================================
# Section 2: Generate Sample Data
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Generate Sample Data")
print("-" * 60)

# Create synthetic reflectance data
np.random.seed(42)
wavelengths = np.linspace(900, 2500, 200)
n_samples = 50

# Base reflectance (organic materials typically 0.3-0.7)
R = np.random.uniform(0.3, 0.7, size=(n_samples, len(wavelengths)))

# Add realistic variation
slope = np.random.uniform(-0.0001, 0.0001, size=(n_samples, 1))
R = R + slope * (wavelengths - wavelengths.mean())

# Add absorption features
for band_center in [1450, 1940]:  # Water bands
    band_idx = np.argmin(np.abs(wavelengths - band_center))
    band_depth = np.random.uniform(0.05, 0.15, size=(n_samples, 1))
    band_shape = np.exp(-0.5 * ((wavelengths - band_center) / 30) ** 2)
    R = R - band_depth * band_shape

R = np.clip(R, 0.01, 0.99)

# Create target related to water content
y = R[:, np.argmin(np.abs(wavelengths - 1940))] + np.random.normal(0, 0.02, n_samples)

print(f"Generated {n_samples} reflectance spectra")
print(f"   R range: [{R.min():.3f}, {R.max():.3f}]")


# =============================================================================
# Section 3: Signal Type Detection
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Automatic Signal Type Detection")
print("-" * 60)

print("""
nirs4all can automatically detect the signal type from data values.
""")

# Detect reflectance
signal_type, confidence, reason = detect_signal_type(R)
print(f"\nDetection on reflectance data:")
print(f"   Type: {signal_type.value}")
print(f"   Confidence: {confidence:.1%}")
print(f"   Reason: {reason}")

# Detect absorbance
A = -np.log10(R)
signal_type, confidence, reason = detect_signal_type(A)
print(f"\nDetection on absorbance data:")
print(f"   Type: {signal_type.value}")
print(f"   Confidence: {confidence:.1%}")
print(f"   Reason: {reason}")


# =============================================================================
# Section 4: Basic Conversions
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Basic Conversions")
print("-" * 60)

# Reflectance to Absorbance
print("\n📐 Reflectance → Absorbance:")
to_abs = ToAbsorbance(source_type="reflectance")
A_converted = to_abs.fit_transform(R)
print(f"   Input R:  [{R.min():.4f}, {R.max():.4f}]")
print(f"   Output A: [{A_converted.min():.4f}, {A_converted.max():.4f}]")

# Verify
expected = -np.log10(R)
error = np.abs(A_converted - expected).max()
print(f"   Verification: error = {error:.2e} ✓")

# Absorbance to Reflectance
print("\n📐 Absorbance → Reflectance:")
from_abs = FromAbsorbance(target_type="reflectance")
R_recovered = from_abs.fit_transform(A_converted)
print(f"   Input A:  [{A_converted.min():.4f}, {A_converted.max():.4f}]")
print(f"   Output R: [{R_recovered.min():.4f}, {R_recovered.max():.4f}]")

# Verify round-trip
roundtrip_error = np.abs(R_recovered - R).max()
print(f"   Round-trip error: {roundtrip_error:.2e} ✓")

# Percent conversion
print("\n📐 Reflectance % → Reflectance fraction:")
R_percent = R * 100
pct_to_frac = PercentToFraction()
R_fraction = pct_to_frac.fit_transform(R_percent)
print(f"   Input %R:  [{R_percent.min():.1f}, {R_percent.max():.1f}]")
print(f"   Output R:  [{R_fraction.min():.4f}, {R_fraction.max():.4f}]")


# =============================================================================
# Section 5: Kubelka-Munk Transformation
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Kubelka-Munk Transformation")
print("-" * 60)

print("""
Kubelka-Munk is used for diffuse reflectance of scattering media.
Formula: F(R) = (1-R)² / (2R)
""")

km = KubelkaMunk(source_type="reflectance")
F_R = km.fit_transform(R)

print(f"\nReflectance → Kubelka-Munk:")
print(f"   Input R:    [{R.min():.4f}, {R.max():.4f}]")
print(f"   Output F(R): [{F_R.min():.4f}, {F_R.max():.4f}]")

# Verify formula
F_R_expected = np.square(1 - R) / (2 * R)
km_error = np.abs(F_R - F_R_expected).max()
print(f"   Formula verification: error = {km_error:.2e} ✓")

# Inverse transform
R_from_km = km.inverse_transform(F_R)
km_roundtrip = np.abs(R_from_km - R).max()
print(f"   Round-trip error: {km_roundtrip:.2e} ✓")


# =============================================================================
# Section 6: General-Purpose SignalTypeConverter
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: SignalTypeConverter")
print("-" * 60)

print("""
SignalTypeConverter automatically determines the conversion path.
Useful when you want to specify source and target types explicitly.
""")

conversions = [
    ("reflectance", "absorbance"),
    ("reflectance%", "absorbance"),
    ("absorbance", "reflectance"),
    ("reflectance", "kubelka_munk"),
]

for src, tgt in conversions:
    converter = SignalTypeConverter(source_type=src, target_type=tgt)

    # Prepare input
    if src == "reflectance":
        X_in = R
    elif src == "reflectance%":
        X_in = R * 100
    elif src == "absorbance":
        X_in = -np.log10(R)
    else:
        X_in = R

    X_out = converter.fit_transform(X_in)
    print(f"   {src:20s} → {tgt:15s}: [{X_out.min():.4f}, {X_out.max():.4f}]")


# =============================================================================
# Section 7: Using Converters in nirs4all Pipelines
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Converters in Pipelines")
print("-" * 60)

print("""
Signal converters are sklearn-compatible and work in nirs4all pipelines.
""")

# Pipeline 1: Raw reflectance
pipeline_r = [
    StandardScaler(),
    ShuffleSplit(n_splits=2, random_state=42),
    {"model": PLSRegression(n_components=5)},
]

# Pipeline 2: Convert to absorbance first
pipeline_a = [
    ToAbsorbance(source_type="reflectance"),
    StandardScaler(),
    ShuffleSplit(n_splits=2, random_state=42),
    {"model": PLSRegression(n_components=5)},
]

# Pipeline 3: Convert to Kubelka-Munk first
pipeline_km = [
    KubelkaMunk(source_type="reflectance"),
    StandardScaler(),
    ShuffleSplit(n_splits=2, random_state=42),
    {"model": PLSRegression(n_components=5)},
]

# Run with synthetic data
result_r = nirs4all.run(
    pipeline=pipeline_r,
    dataset=(R, y),
    name="Reflectance",
    verbose=0
)

result_a = nirs4all.run(
    pipeline=pipeline_a,
    dataset=(R, y),
    name="Absorbance",
    verbose=0
)

result_km = nirs4all.run(
    pipeline=pipeline_km,
    dataset=(R, y),
    name="KubelkaMunk",
    verbose=0
)

print("\nComparison of signal representations:")
print(f"   Reflectance:   RMSE = {result_r.best_rmse:.4f}")
print(f"   Absorbance:    RMSE = {result_a.best_rmse:.4f}")
print(f"   Kubelka-Munk:  RMSE = {result_km.best_rmse:.4f}")


# =============================================================================
# Section 8: Signal Type Aliases
# =============================================================================
print("\n" + "-" * 60)
print("Section 8: Signal Type Aliases")
print("-" * 60)

print("""
Various string aliases are supported for specifying signal types:
""")

from nirs4all.data.signal_type import normalize_signal_type

aliases = {
    'Absorbance': ['a', 'abs', 'absorbance', 'A'],
    'Reflectance': ['r', 'ref', 'refl', 'R'],
    'Reflectance %': ['%r', 'r%', 'reflectance%'],
    'Transmittance': ['t', 'trans', 'T'],
    'Kubelka-Munk': ['km', 'kubelka_munk', 'f(r)'],
}

for signal_name, alias_list in aliases.items():
    aliases_str = ", ".join([f"'{a}'" for a in alias_list])
    print(f"   {signal_name}: {aliases_str}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Signal Conversion Operators:

  Basic Converters:
    ToAbsorbance(source_type="reflectance")  # R → A
    FromAbsorbance(target_type="reflectance") # A → R
    KubelkaMunk(source_type="reflectance")   # R → F(R)

  Percent/Fraction:
    PercentToFraction()  # 0-100 → 0-1
    FractionToPercent()  # 0-1 → 0-100

  General Converter:
    SignalTypeConverter(source_type="reflectance%", target_type="absorbance")

  Signal Type Detection:
    from nirs4all.data.signal_type import detect_signal_type
    signal_type, confidence, reason = detect_signal_type(X)

Common Use Cases:
  • Convert reflectance instruments to absorbance scale
  • Apply Kubelka-Munk for powder/granule samples
  • Standardize data from mixed instrument sources

All converters are sklearn-compatible TransformerMixin.

Next: See 04_models/U01_multi_model.py - Comparing multiple models
""")
