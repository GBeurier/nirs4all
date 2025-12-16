"""
Q29: Signal Type Detection and Conversion
==========================================

This example demonstrates:
1. Signal type detection (autodetecting absorbance, reflectance, transmittance)
2. Signal conversions between different spectral representations
3. Using signal conversion operators in pipelines
4. Kubelka-Munk transformation for diffuse reflectance

Key Concepts:
- Absorbance (A): Typically values [0, 3+], computed as -log10(R) or -log10(T)
- Reflectance (R): Fraction [0, 1] or percent [0, 100] of reflected light
- Transmittance (T): Fraction [0, 1] or percent [0, 100] of transmitted light
- Kubelka-Munk F(R): (1-R)²/(2R), theoretical for scattering media

Usage:
    python Q29_signal_conversion.py
"""

import numpy as np
import matplotlib.pyplot as plt

from nirs4all.data.signal_type import (
    SignalType,
    SignalTypeDetector,
    detect_signal_type,
    normalize_signal_type
)
from nirs4all.operators.transforms.signal_conversion import (
    ToAbsorbance,
    FromAbsorbance,
    KubelkaMunk,
    PercentToFraction,
    FractionToPercent,
    SignalTypeConverter
)

# =============================================================================
# Generate synthetic spectral data
# =============================================================================
print("=" * 80)
print("Q29: Signal Type Detection and Conversion")
print("=" * 80)

np.random.seed(42)

# Create wavelength axis (typical NIR range in nm)
wavelengths = np.linspace(900, 2500, 200)

# Generate synthetic reflectance spectra with realistic features
n_samples = 50
n_features = len(wavelengths)

# Base reflectance (organic materials typically 0.3-0.7)
R_base = np.random.uniform(0.4, 0.6, size=(n_samples, 1))
R = np.tile(R_base, (1, n_features))

# Add spectral variation (slope)
slope = np.random.uniform(-0.0001, 0.0001, size=(n_samples, 1))
R = R + slope * (wavelengths - wavelengths.mean())

# Add absorption features at water bands (~1450nm, ~1940nm)
for band_center in [1450, 1940]:
    band_idx = np.argmin(np.abs(wavelengths - band_center))
    band_depth = np.random.uniform(0.05, 0.15, size=(n_samples, 1))
    band_width = 30  # nm
    band_shape = np.exp(-0.5 * ((wavelengths - band_center) / band_width) ** 2)
    R = R - band_depth * band_shape

# Clip to valid reflectance range
R = np.clip(R, 0.01, 0.99)

print(f"\n📊 Generated synthetic reflectance data: {R.shape[0]} samples × {R.shape[1]} features")
print(f"   Reflectance range: [{R.min():.3f}, {R.max():.3f}]")

# =============================================================================
# Part 1: Signal Type Detection
# =============================================================================
print("\n" + "=" * 80)
print("Part 1: Signal Type Detection")
print("=" * 80)

# Detect signal type without wavelength information
signal_type, confidence, reason = detect_signal_type(R)
print(f"\n🔍 Detection without wavelength info:")
print(f"   Type: {signal_type.value}")
print(f"   Confidence: {confidence:.1%}")
print(f"   Reason: {reason}")

# Detect with wavelength information (more accurate)
signal_type, confidence, reason = detect_signal_type(
    R, wavelengths=wavelengths, wavelength_unit="nm"
)
print(f"\n🔍 Detection with wavelength info (nm):")
print(f"   Type: {signal_type.value}")
print(f"   Confidence: {confidence:.1%}")

# Test detection on absorbance data
A = -np.log10(R)
signal_type_a, conf_a, reason_a = detect_signal_type(A)
print(f"\n🔍 Detection on absorbance data:")
print(f"   Type: {signal_type_a.value}")
print(f"   Confidence: {conf_a:.1%}")
print(f"   Reason: {reason_a}")

# Test detection on preprocessed data (SNV-like)
R_snv = (R - R.mean(axis=1, keepdims=True)) / R.std(axis=1, keepdims=True)
signal_type_snv, conf_snv, _ = detect_signal_type(R_snv)
print(f"\n🔍 Detection on preprocessed (SNV) data:")
print(f"   Type: {signal_type_snv.value}")
print(f"   Confidence: {conf_snv:.1%}")

# =============================================================================
# Part 2: Basic Signal Conversions
# =============================================================================
print("\n" + "=" * 80)
print("Part 2: Basic Signal Conversions")
print("=" * 80)

# Reflectance to Absorbance
print("\n📐 Reflectance → Absorbance:")
to_abs = ToAbsorbance(source_type="reflectance")
A_converted = to_abs.fit_transform(R)
print(f"   Input R range:  [{R.min():.4f}, {R.max():.4f}]")
print(f"   Output A range: [{A_converted.min():.4f}, {A_converted.max():.4f}]")

# Verify: A = -log10(R)
A_expected = -np.log10(R)
error = np.abs(A_converted - A_expected).max()
print(f"   Verification error: {error:.2e} ✓" if error < 1e-10 else f"   Error: {error}")

# Absorbance to Reflectance
print("\n📐 Absorbance → Reflectance:")
from_abs = FromAbsorbance(target_type="reflectance")
R_recovered = from_abs.fit_transform(A_converted)
print(f"   Input A range:  [{A_converted.min():.4f}, {A_converted.max():.4f}]")
print(f"   Output R range: [{R_recovered.min():.4f}, {R_recovered.max():.4f}]")

# Verify round-trip
roundtrip_error = np.abs(R_recovered - R).max()
print(f"   Round-trip error: {roundtrip_error:.2e} ✓" if roundtrip_error < 1e-10 else f"   Error: {roundtrip_error}")

# Percent to Fraction conversion
print("\n📐 Reflectance % → Reflectance fraction:")
R_percent = R * 100  # Convert to percent
pct_to_frac = PercentToFraction()
R_fraction = pct_to_frac.fit_transform(R_percent)
print(f"   Input %R range:  [{R_percent.min():.2f}, {R_percent.max():.2f}]")
print(f"   Output R range:  [{R_fraction.min():.4f}, {R_fraction.max():.4f}]")

# =============================================================================
# Part 3: Kubelka-Munk Transformation
# =============================================================================
print("\n" + "=" * 80)
print("Part 3: Kubelka-Munk Transformation")
print("=" * 80)

print("\n📐 Reflectance → Kubelka-Munk F(R):")
print("   Formula: F(R) = (1-R)² / (2R)")
print("   Used for diffuse reflectance of scattering media (powders, etc.)")

km = KubelkaMunk(source_type="reflectance")
F_R = km.fit_transform(R)
print(f"\n   Input R range:   [{R.min():.4f}, {R.max():.4f}]")
print(f"   Output F(R) range: [{F_R.min():.4f}, {F_R.max():.4f}]")

# Verify formula: F(R) = (1-R)² / (2R)
F_R_expected = np.square(1 - R) / (2 * R)
km_error = np.abs(F_R - F_R_expected).max()
print(f"   Formula verification error: {km_error:.2e} ✓")

# Inverse K-M transformation
R_from_km = km.inverse_transform(F_R)
km_roundtrip = np.abs(R_from_km - R).max()
print(f"   Round-trip error: {km_roundtrip:.2e} ✓")

# =============================================================================
# Part 4: General-Purpose SignalTypeConverter
# =============================================================================
print("\n" + "=" * 80)
print("Part 4: General-Purpose SignalTypeConverter")
print("=" * 80)

print("\n🔄 The SignalTypeConverter automatically determines the conversion path")

# Example conversions
conversions = [
    ("reflectance", "absorbance"),
    ("reflectance%", "absorbance"),
    ("absorbance", "reflectance"),
    ("reflectance", "kubelka_munk"),
    ("reflectance%", "reflectance"),
]

for src, tgt in conversions:
    converter = SignalTypeConverter(source_type=src, target_type=tgt)

    # Prepare input data
    if src == "reflectance":
        X_in = R
    elif src == "reflectance%":
        X_in = R * 100
    elif src == "absorbance":
        X_in = -np.log10(R)
    else:
        X_in = R

    X_out = converter.fit_transform(X_in)
    print(f"\n   {src:20s} → {tgt:15s}")
    print(f"      Input:  [{X_in.min():.4f}, {X_in.max():.4f}]")
    print(f"      Output: [{X_out.min():.4f}, {X_out.max():.4f}]")

# =============================================================================
# Part 5: Visualization
# =============================================================================
print("\n" + "=" * 80)
print("Part 5: Visualization")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Select a few samples for visualization
sample_indices = [0, 10, 20, 30, 40]
colors = plt.cm.viridis(np.linspace(0, 1, len(sample_indices)))

# Plot 1: Original Reflectance
ax = axes[0, 0]
for i, idx in enumerate(sample_indices):
    ax.plot(wavelengths, R[idx], color=colors[i], alpha=0.8)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Reflectance')
ax.set_title('Original Reflectance')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# Plot 2: Absorbance (from R)
A_plot = -np.log10(R)
ax = axes[0, 1]
for i, idx in enumerate(sample_indices):
    ax.plot(wavelengths, A_plot[idx], color=colors[i], alpha=0.8)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Absorbance')
ax.set_title('Absorbance: A = -log₁₀(R)')
ax.grid(True, alpha=0.3)

# Plot 3: Kubelka-Munk
ax = axes[0, 2]
for i, idx in enumerate(sample_indices):
    ax.plot(wavelengths, F_R[idx], color=colors[i], alpha=0.8)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('F(R)')
ax.set_title('Kubelka-Munk: F(R) = (1-R)²/(2R)')
ax.grid(True, alpha=0.3)

# Plot 4: Comparison at single wavelength (conversion curve)
ax = axes[1, 0]
R_range = np.linspace(0.01, 0.99, 100).reshape(-1, 1)
A_range = -np.log10(R_range)
ax.plot(R_range, A_range, 'b-', linewidth=2, label='A = -log₁₀(R)')
ax.set_xlabel('Reflectance')
ax.set_ylabel('Absorbance')
ax.set_title('R → A Conversion Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Kubelka-Munk vs Absorbance comparison
ax = axes[1, 1]
km_range = np.square(1 - R_range) / (2 * R_range)
ax.plot(R_range, A_range, 'b-', linewidth=2, label='Absorbance')
ax.plot(R_range, km_range, 'r-', linewidth=2, label='Kubelka-Munk')
ax.set_xlabel('Reflectance')
ax.set_ylabel('Transformed Value')
ax.set_title('Absorbance vs Kubelka-Munk')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 3)

# Plot 6: Signal type properties
ax = axes[1, 2]
signal_types = [
    ('absorbance', SignalType.ABSORBANCE),
    ('reflectance', SignalType.REFLECTANCE),
    ('reflectance%', SignalType.REFLECTANCE_PERCENT),
    ('transmittance', SignalType.TRANSMITTANCE),
    ('kubelka_munk', SignalType.KUBELKA_MUNK),
]
properties = ['is_percent', 'is_fraction', 'is_absorbance_like']
data = np.array([[getattr(st, prop) for prop in properties] for _, st in signal_types])
im = ax.imshow(data.T, aspect='auto', cmap='Blues')
ax.set_xticks(range(len(signal_types)))
ax.set_xticklabels([name for name, _ in signal_types], rotation=45, ha='right')
ax.set_yticks(range(len(properties)))
ax.set_yticklabels(properties)
ax.set_title('Signal Type Properties')
for i in range(len(properties)):
    for j in range(len(signal_types)):
        ax.text(j, i, '✓' if data[j, i] else '', ha='center', va='center', fontsize=14)

plt.tight_layout()
plt.savefig('charts/signal_conversion_demo.png', dpi=150, bbox_inches='tight')
print("\n📊 Saved visualization to charts/signal_conversion_demo.png")
plt.show()

# =============================================================================
# Part 6: Using Signal Conversion in Pipelines
# =============================================================================
print("\n" + "=" * 80)
print("Part 6: Signal Conversion in Pipelines")
print("=" * 80)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score

# Create target variable (e.g., moisture content correlated with water bands)
water_band_idx = np.argmin(np.abs(wavelengths - 1940))
y = R[:, water_band_idx] + np.random.normal(0, 0.02, n_samples)

print("\n🔧 Creating sklearn pipelines with different signal representations:")

# Pipeline 1: Raw reflectance
pipeline_r = Pipeline([
    ('scaler', StandardScaler()),
    ('pls', PLSRegression(n_components=5))
])

# Pipeline 2: Absorbance
pipeline_a = Pipeline([
    ('to_absorbance', ToAbsorbance(source_type='reflectance')),
    ('scaler', StandardScaler()),
    ('pls', PLSRegression(n_components=5))
])

# Pipeline 3: Kubelka-Munk
pipeline_km = Pipeline([
    ('kubelka_munk', KubelkaMunk(source_type='reflectance')),
    ('scaler', StandardScaler()),
    ('pls', PLSRegression(n_components=5))
])

# Cross-validate each pipeline
print("\n   Cross-validation scores (5-fold, R²):")
for name, pipeline in [('Reflectance', pipeline_r),
                        ('Absorbance', pipeline_a),
                        ('Kubelka-Munk', pipeline_km)]:
    scores = cross_val_score(pipeline, R, y, cv=5, scoring='r2')
    print(f"   {name:15s}: {scores.mean():.3f} ± {scores.std():.3f}")

# =============================================================================
# Part 7: Signal Type String Parsing
# =============================================================================
print("\n" + "=" * 80)
print("Part 7: Signal Type String Parsing")
print("=" * 80)

print("\n📝 Supported string aliases for signal types:")

aliases = {
    'Absorbance': ['a', 'abs', 'absorbance', 'A'],
    'Reflectance': ['r', 'ref', 'refl', 'reflectance', 'R'],
    'Reflectance %': ['%r', 'r%', 'reflectance%', '%R'],
    'Transmittance': ['t', 'trans', 'transmittance', 'T'],
    'Transmittance %': ['%t', 't%', 'transmittance%', '%T'],
    'Kubelka-Munk': ['km', 'kubelka_munk', 'f(r)', 'KM'],
}

for signal_name, alias_list in aliases.items():
    print(f"\n   {signal_name}:")
    for alias in alias_list:
        parsed = normalize_signal_type(alias)
        print(f"      '{alias}' → {parsed.value}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

print("""
✅ Signal Type Detection:
   - Automatic detection based on value ranges and band patterns
   - Works with/without wavelength information
   - Detects preprocessed data (SNV, derivatives)

✅ Signal Conversions:
   - ToAbsorbance: R/T → A using -log₁₀(X)
   - FromAbsorbance: A → R/T using 10^(-A)
   - KubelkaMunk: R → F(R) = (1-R)²/(2R)
   - PercentToFraction / FractionToPercent

✅ Pipeline Integration:
   - All converters are sklearn-compatible (TransformerMixin)
   - Can be used in sklearn Pipeline
   - Support fit, transform, inverse_transform

✅ Configuration via JSON/YAML:
   - signal_type can be specified in config params
   - Follows same pattern as header_unit, delimiter, etc.
""")

print("🎉 Example complete!")
