# Signal Types and Conversion

## Overview

NIRS data can be in different physical representations depending on the instrument and measurement technique. nirs4all automatically detects and converts between these signal types to ensure consistent analysis.

Understanding signal types is crucial because:
- Different instruments output different signal types
- Some algorithms perform better with specific representations
- Conversion affects the scale and interpretation of spectral features
- Proper signal type handling is essential for multi-source data fusion

## Supported Signal Types

| Type | Range | Common Use | Description |
|------|-------|------------|-------------|
| `absorbance` | [0, ∞) | Lab spectrometers | -log₁₀(I/I₀), proportional to concentration |
| `reflectance` | [0, 1] | Portable devices | I/I₀ as fraction |
| `reflectance%` | [0, 100] | Older instruments | 100×(I/I₀) as percentage |
| `transmittance` | [0, 1] | Thin samples | I/I₀ for transmitted light |
| `transmittance%` | [0, 100] | Older instruments | 100×(I/I₀) as percentage |
| `log_1_r` | (-∞, 0] | Pre-converted | log₁₀(1/R) - pseudo-absorbance |
| `log_1_t` | (-∞, 0] | Pre-converted | log₁₀(1/T) - absorbance from transmittance |
| `kubelka_munk` | [0, ∞) | Diffuse reflectance | (1-R)²/(2R) for scattering media |

### Special Signal Types

| Type | Description | Use Case |
|------|-------------|----------|
| `auto` | Automatic detection | Default for new datasets |
| `unknown` | Cannot be determined | Ambiguous data ranges |
| `preprocessed` | Already preprocessed | SNV, derivatives, mean-centered data |

## Automatic Detection

nirs4all can automatically detect the signal type using value range analysis and wavelength band characteristics:

```python
from nirs4all.data import DatasetConfigs

# Load data (signal type detected automatically)
dataset = DatasetConfigs("path/to/data")

# Check detected signal type
signal_type, confidence, reason = dataset.detect_signal_type(src=0)
print(f"Detected: {signal_type.value}")
print(f"Confidence: {confidence:.1%}")
print(f"Reason: {reason}")
```

### Detection Logic

The detection algorithm uses multiple heuristics:

1. **Value Range Analysis**: Examines min/max/mean values
   - [0, 1] → likely reflectance/transmittance
   - [0, 100] → likely percent reflectance/transmittance
   - [0, 3+] → likely absorbance
   - Negative values → likely preprocessed

2. **Water Band Direction** (if wavelengths available):
   - Peaks at 1450, 1940, 2500 nm → absorbance
   - Dips at water bands → reflectance/transmittance

3. **Statistical Indicators**:
   - Mean near 0, std near 1 → preprocessed (SNV)
   - Symmetric distribution around 0 → preprocessed (derivative)

## Manual Signal Type Setting

Override automatic detection when you know the signal type:

```python
from nirs4all.data.signal_type import SignalType

# Force signal type
dataset.set_signal_type("absorbance", src=0, forced=True)

# Using enum
dataset.set_signal_type(SignalType.REFLECTANCE, src=0, forced=True)

# Using common abbreviations
dataset.set_signal_type("R", src=0, forced=True)  # Reflectance
dataset.set_signal_type("A", src=0, forced=True)  # Absorbance
dataset.set_signal_type("%R", src=0, forced=True) # Reflectance percent
```

### Signal Type Aliases

Many string aliases are supported for convenience:

| Signal Type | Accepted Aliases |
|-------------|------------------|
| Absorbance | `"a"`, `"abs"`, `"absorbance"`, `"A"` |
| Reflectance | `"r"`, `"ref"`, `"refl"`, `"R"` |
| Reflectance % | `"%r"`, `"r%"`, `"reflectance%"` |
| Transmittance | `"t"`, `"trans"`, `"T"` |
| Kubelka-Munk | `"km"`, `"kubelka_munk"`, `"f(r)"` |

## Signal Conversion

Convert between signal types using sklearn-compatible transformers:

```python
from nirs4all.operators.transforms.signal_conversion import (
    ToAbsorbance,
    FromAbsorbance,
    KubelkaMunk,
    SignalTypeConverter
)
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

# Pipeline 1: Convert reflectance to absorbance
pipeline_abs = [
    ToAbsorbance(source_type="reflectance"),
    StandardScaler(),
    {"model": PLSRegression(n_components=10)}
]

# Pipeline 2: Convert to Kubelka-Munk (for powders)
pipeline_km = [
    KubelkaMunk(source_type="reflectance"),
    StandardScaler(),
    {"model": PLSRegression(n_components=10)}
]

# Pipeline 3: General converter
from nirs4all.operators.transforms.signal_conversion import SignalTypeConverter

pipeline_general = [
    SignalTypeConverter(source_type="reflectance%", target_type="absorbance"),
    StandardScaler(),
    {"model": PLSRegression(n_components=10)}
]
```

### Common Conversions

#### Reflectance to Absorbance

```python
from nirs4all.operators.transforms.signal_conversion import ToAbsorbance
import numpy as np

# Reflectance data (0-1 range)
R = np.array([[0.5, 0.4, 0.3], [0.6, 0.5, 0.4]])

# Convert to absorbance
converter = ToAbsorbance(source_type="reflectance")
A = converter.fit_transform(R)
# A ≈ [[0.301, 0.398, 0.523], [0.222, 0.301, 0.398]]

# Round-trip (verify conversion)
R_back = converter.inverse_transform(A)
print(f"Round-trip error: {np.abs(R - R_back).max():.2e}")
```

#### Absorbance to Reflectance

```python
from nirs4all.operators.transforms.signal_conversion import FromAbsorbance

# Absorbance data
A = np.array([[0.3, 0.4, 0.5], [0.2, 0.3, 0.4]])

# Convert to reflectance
converter = FromAbsorbance(target_type="reflectance")
R = converter.fit_transform(A)
```

#### Percent to Fraction

```python
from nirs4all.operators.transforms.signal_conversion import PercentToFraction

# Reflectance percent (0-100 range)
R_pct = np.array([[50, 40, 30], [60, 50, 40]])

# Convert to fraction (0-1 range)
converter = PercentToFraction()
R_frac = converter.fit_transform(R_pct)
# R_frac ≈ [[0.5, 0.4, 0.3], [0.6, 0.5, 0.4]]
```

#### Kubelka-Munk Transformation

```python
from nirs4all.operators.transforms.signal_conversion import KubelkaMunk

# Reflectance data for diffuse reflectance (powders, granules)
R = np.array([[0.5, 0.4, 0.3], [0.6, 0.5, 0.4]])

# Apply Kubelka-Munk transformation: F(R) = (1-R)² / (2R)
converter = KubelkaMunk(source_type="reflectance")
F_R = converter.fit_transform(R)

# Inverse transform
R_back = converter.inverse_transform(F_R)
```

### When to Use Each Conversion

**Use Absorbance when:**
- Working with liquid samples (Beer-Lambert law applies)
- You need proportionality to concentration
- Comparing with reference methods that use absorbance
- Using algorithms designed for absorbance spectra

**Use Reflectance when:**
- Raw instrument output is reflectance
- Working with solid samples
- Preserving original measurement scale

**Use Kubelka-Munk when:**
- Analyzing diffuse reflectance from scattering media
- Working with powders, granules, or tablets
- You need a linearized relationship for thick samples
- Sample thickness is effectively infinite

## Working with Multiple Sources

Different data sources can have different signal types:

```python
from nirs4all.data import DatasetConfigs

# Multi-source dataset
dataset = DatasetConfigs({
    "train_x": ["nir_spectra.csv", "vis_spectra.csv"],
    "train_y": "targets.csv"
})

# Check signal types for each source
for src_idx in range(2):
    signal_type, confidence, reason = dataset.detect_signal_type(src=src_idx)
    print(f"Source {src_idx}: {signal_type.value} ({confidence:.0%})")

# Set different signal types per source
dataset.set_signal_type("absorbance", src=0)  # NIR is absorbance
dataset.set_signal_type("reflectance", src=1)  # VIS is reflectance

# Access signal types
signal_types = dataset.signal_types
print(f"All signal types: {[st.value for st in signal_types]}")
```

## Complete Example

Here's a complete workflow demonstrating signal type detection and conversion:

```python
import nirs4all
import numpy as np
from nirs4all.data.signal_type import detect_signal_type, SignalType
from nirs4all.operators.transforms.signal_conversion import ToAbsorbance
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

# Generate synthetic reflectance data
np.random.seed(42)
n_samples, n_wavelengths = 100, 200
R = np.random.uniform(0.3, 0.7, size=(n_samples, n_wavelengths))
y = R[:, 100] + np.random.normal(0, 0.02, n_samples)  # Target related to 1 band

# Detect signal type
signal_type, confidence, reason = detect_signal_type(R)
print(f"Detected: {signal_type.value} ({confidence:.1%})")
print(f"Reason: {reason}")

# Compare pipelines with different signal representations
pipeline_r = [
    StandardScaler(),
    {"model": PLSRegression(n_components=10)}
]

pipeline_a = [
    ToAbsorbance(source_type="reflectance"),
    StandardScaler(),
    {"model": PLSRegression(n_components=10)}
]

# Run both
result_r = nirs4all.run(pipeline=pipeline_r, dataset=(R, y), name="Reflectance", verbose=0)
result_a = nirs4all.run(pipeline=pipeline_a, dataset=(R, y), name="Absorbance", verbose=0)

print(f"\nResults:")
print(f"Reflectance:  RMSE = {result_r.best_rmse:.4f}")
print(f"Absorbance:   RMSE = {result_a.best_rmse:.4f}")
```

## API Reference

### SpectroDataset Methods

- `dataset.detect_signal_type(src=0, force_redetect=False)` → Tuple[SignalType, float, str]

  Detect signal type using heuristics. Returns (SignalType, confidence, reason_string).

- `dataset.set_signal_type(signal_type, src=0, forced=True)` → None

  Set the signal type for a data source. Use `forced=True` to prevent auto-detection from overriding.

- `dataset.signal_type(src=0)` → SignalType

  Get the current signal type for a source.

- `dataset.signal_types` → List[SignalType]

  Get signal types for all sources.

### Signal Converters

All converters follow the sklearn `TransformerMixin` pattern with `fit()`, `transform()`, and `inverse_transform()` methods:

- `ToAbsorbance(source_type="reflectance", epsilon=1e-10, clip_negative=True)`

  Convert reflectance/transmittance to absorbance: A = -log₁₀(X)

- `FromAbsorbance(target_type="reflectance")`

  Convert absorbance to reflectance/transmittance: X = 10^(-A)

- `KubelkaMunk(source_type="reflectance", epsilon=1e-10)`

  Apply Kubelka-Munk transformation: F(R) = (1-R)² / (2R)

- `PercentToFraction()`

  Convert percentage (0-100) to fraction (0-1)

- `FractionToPercent()`

  Convert fraction (0-1) to percentage (0-100)

- `SignalTypeConverter(source_type, target_type)`

  General-purpose converter that automatically determines the conversion path

### Utility Functions

- `detect_signal_type(spectra, wavelengths=None, wavelength_unit="nm")` → Tuple[SignalType, float, str]

  Standalone function to detect signal type from numpy array

- `normalize_signal_type(signal_type)` → SignalType

  Convert string or enum to SignalType enum

## See Also

- {doc}`loading_data` - Data loading fundamentals
- {doc}`aggregation` - Sample repetition and prediction aggregation
- {doc}`/user_guide/preprocessing/handbook` - Signal-specific preprocessing strategies
- {doc}`/getting_started/index` - Quick start guide

```{seealso}
**Related Examples:**
- [U04: Signal Conversion](../../../examples/user/03_preprocessing/U04_signal_conversion.py) - Signal type detection and conversion operators
- [U01: Flexible Inputs](../../../examples/user/02_data_handling/U01_flexible_inputs.py) - Data loading with signal type handling
```
