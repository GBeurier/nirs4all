# Preprocessing Examples

This section covers NIRS-specific preprocessing techniques, from basic transformations to automated exploration of preprocessing combinations.

```{contents} On this page
:local:
:depth: 2
```

## Overview

| Example | Topic | Difficulty | Duration |
|---------|-------|------------|----------|
| [U01](#u01-preprocessing-basics) | Preprocessing Basics | ★★☆☆☆ | ~3 min |
| [U02](#u02-feature-augmentation) | Feature Augmentation | ★★☆☆☆ | ~3 min |
| [U03](#u03-sample-augmentation) | Sample Augmentation | ★★☆☆☆ | ~3 min |
| [U04](#u04-signal-conversion) | Signal Conversion | ★★☆☆☆ | ~2 min |

---

## U01: Preprocessing Basics

**Overview of standard NIRS preprocessing techniques.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/03_preprocessing/U01_preprocessing_basics.py)

### What You'll Learn

- Scatter correction: SNV, MSC
- Baseline correction: Detrend
- Derivatives: First, Second, Savitzky-Golay
- Smoothing: Gaussian, Savitzky-Golay
- Wavelet transforms: Haar

### Preprocessing Categories

NIRS preprocessing addresses common spectral issues:

#### 📊 Scatter Correction

Corrects for variations in light scattering due to sample structure:

```python
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    MultiplicativeScatterCorrection
)

# SNV: Per-sample mean-centering and scaling
StandardNormalVariate()

# MSC: Regression-based correction using reference spectrum
MultiplicativeScatterCorrection()
```

| Method | How it Works | When to Use |
|--------|--------------|-------------|
| **SNV** | Centers and scales each spectrum individually | Path length variations, quick scatter correction |
| **MSC** | Regresses each spectrum against a reference (mean) | More robust to baseline variations |

#### 📈 Baseline Correction

Removes baseline drift from spectra:

```python
from nirs4all.operators.transforms import Detrend

# Remove polynomial baseline drift
Detrend()  # Default: linear detrending
```

#### 📉 Derivatives

Enhance peaks and remove baselines:

```python
from nirs4all.operators.transforms import (
    FirstDerivative,
    SecondDerivative,
    SavitzkyGolay
)

# Simple derivatives
FirstDerivative()   # Removes constant baseline
SecondDerivative()  # Removes linear baseline

# Smoothed derivative (recommended for noisy data)
SavitzkyGolay(window_length=11, polyorder=2, deriv=1)
```

| Derivative | Effect | Use Case |
|------------|--------|----------|
| **First** | Enhances peaks, removes constant baseline | General baseline issues |
| **Second** | Stronger enhancement, removes linear baseline | Complex baselines |
| **Savitzky-Golay** | Smoothed derivatives | Noisy spectra |

#### 🔊 Smoothing

Reduce noise while preserving spectral features:

```python
from nirs4all.operators.transforms import Gaussian, SavitzkyGolay

# Gaussian convolution
Gaussian(sigma=2)

# Polynomial smoothing (no derivative)
SavitzkyGolay(window_length=11, polyorder=2, deriv=0)
```

#### 🌊 Wavelet Transforms

Multi-resolution analysis:

```python
from nirs4all.operators.transforms import Haar

Haar()  # Haar wavelet transform
```

### Combining Preprocessing Steps

Common combinations for NIRS data:

```python
# Combination 1: Scatter + Derivative
pipeline = [
    StandardNormalVariate(),
    FirstDerivative(),
    PLSRegression(n_components=10)
]

# Combination 2: Full preprocessing chain
pipeline = [
    Detrend(),
    MultiplicativeScatterCorrection(),
    SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
    PLSRegression(n_components=10)
]
```

### Comparing Methods

```python
# Run pipelines with different preprocessing
methods = {
    'SNV': StandardNormalVariate(),
    'MSC': MultiplicativeScatterCorrection(),
    'D1': FirstDerivative(),
    'SG': SavitzkyGolay(deriv=1),
}

for name, method in methods.items():
    pipeline = [method, ShuffleSplit(n_splits=3), PLSRegression(n_components=10)]
    result = nirs4all.run(pipeline=pipeline, dataset="sample_data/regression")
    print(f"{name}: RMSE = {result.best_rmse:.4f}")
```

---

## U02: Feature Augmentation

**Automatically explore preprocessing combinations.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/03_preprocessing/U02_feature_augmentation.py)

### What You'll Learn

- Using `feature_augmentation` to generate variants
- The `_or_` generator syntax
- Pick, count, and combination controls
- Actions: extend vs add vs replace

### The Feature Augmentation Step

Instead of manually testing every preprocessing combination:

```python
# Manual approach (tedious!)
pipeline_1 = [SNV(), FirstDerivative(), ...]
pipeline_2 = [MSC(), FirstDerivative(), ...]
pipeline_3 = [SNV(), SavitzkyGolay(), ...]
# ... many more
```

Use feature augmentation:

```python
pipeline = [
    MinMaxScaler(),

    # Automatically generate preprocessing variants
    {
        "feature_augmentation": {
            "_or_": [SNV, MSC, FirstDerivative, SavitzkyGolay, Gaussian],
            "pick": 2,      # Pick 2 methods at a time
            "count": 5      # Generate 5 random combinations
        }
    },

    PLSRegression(n_components=10)
]
```

### Generator Syntax Options

#### `_or_` - Alternatives

```python
{"_or_": [A, B, C]}  # Generates: A, B, C (3 variants)
```

#### `pick` - Combinations

```python
{"_or_": [A, B, C, D], "pick": 2}
# Generates: [A,B], [A,C], [A,D], [B,C], [B,D], [C,D] (6 variants)
```

#### `count` - Limit

```python
{"_or_": [A, B, C, D], "pick": 2, "count": 3}
# Generates: 3 random combinations (from the 6 possible)
```

### Augmentation Actions

| Action | Behavior |
|--------|----------|
| `"extend"` | Add generated variants to existing features |
| `"add"` | Stack the new transform on top of previous |
| `"replace"` | Replace current features with augmented versions |

```python
# Extend: try each option separately
{"feature_augmentation": [SNV, MSC, Detrend], "action": "extend"}

# Add: stack a derivative on top of current preprocessing
{"feature_augmentation": [FirstDerivative], "action": "add"}
```

### Practical Example

```python
pipeline = [
    # Base scaling
    MinMaxScaler(),

    # Explore scatter correction options
    {"feature_augmentation": [SNV, MSC, Detrend], "action": "extend"},

    # Add derivative on top
    {"feature_augmentation": [FirstDerivative], "action": "add"},

    # Cross-validation and model
    ShuffleSplit(n_splits=3),
    PLSRegression(n_components=10)
]
```

---

## U03: Sample Augmentation

**Data augmentation techniques for increasing sample diversity.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/03_preprocessing/U03_sample_augmentation.py)

### What You'll Learn

- Noise injection for robustness
- Spectral transformations
- Sample mixing strategies
- Augmentation during training

### Sample Augmentation Techniques

While feature augmentation creates different preprocessing pipelines, **sample augmentation** creates synthetic training samples.

```python
{
    "sample_augmentation": {
        "noise_injection": 0.01,      # Add Gaussian noise (1% std)
        "shift": 2,                    # Shift spectra by ±2 wavelengths
        "scale": 0.05,                 # Scale intensity by ±5%
        "mixup_alpha": 0.2,           # Mixup with alpha=0.2
        "augmentation_factor": 3       # Triple training set size
    }
}
```

### When to Use Sample Augmentation

| Technique | Purpose | Best For |
|-----------|---------|----------|
| **Noise injection** | Robustness to measurement noise | Small datasets |
| **Spectral shift** | Robustness to wavelength calibration | Instrument transfer |
| **Intensity scaling** | Robustness to concentration variations | Variable samples |
| **Mixup** | Regularization, interpolation | Deep learning |

---

## U04: Signal Conversion

**Convert between signal representations (absorbance, reflectance, etc.).**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/03_preprocessing/U04_signal_conversion.py)

### What You'll Learn

- Converting between absorbance and reflectance
- Log transformations
- Standard signal formats

### Common Conversions

```python
from nirs4all.operators.transforms import (
    AbsorbanceToReflectance,
    ReflectanceToAbsorbance,
    Log1p,
    Log10
)

# Convert representations
pipeline = [
    ReflectanceToAbsorbance(),  # If your data is in reflectance
    SNV(),                       # Preprocessing expects absorbance
    PLSRegression(n_components=10)
]
```

### Signal Representation Guidelines

| Representation | Formula | Typical Range |
|----------------|---------|---------------|
| Reflectance (R) | I/I₀ | 0-1 |
| Absorbance (A) | -log₁₀(R) | 0-3+ |
| Transmittance (T) | I/I₀ | 0-1 |

Most NIRS preprocessing methods expect **absorbance** data.

---

## Preprocessing Best Practices

### 1. Order Matters

```python
# Recommended order:
pipeline = [
    # 1. Signal conversion (if needed)
    ReflectanceToAbsorbance(),

    # 2. Scatter correction
    StandardNormalVariate(),

    # 3. Baseline correction (optional)
    Detrend(),

    # 4. Derivatives
    FirstDerivative(),

    # 5. Smoothing (if noisy)
    Gaussian(sigma=1),

    # 6. Feature scaling (before model)
    MinMaxScaler(),

    # 7. Model
    PLSRegression(n_components=10)
]
```

### 2. Don't Over-Process

More preprocessing isn't always better. Common mistakes:

- ❌ Applying SNV after derivatives (destroys derivative information)
- ❌ Multiple smoothing steps (over-smooths, loses peaks)
- ❌ Second derivative on noisy data (amplifies noise)

### 3. Use Visualization

```python
pipeline = [
    "chart_2d",           # Visualize raw spectra
    SNV(),
    "chart_2d",           # Visualize after SNV
    FirstDerivative(),
    "chart_2d",           # Visualize after derivative
    PLSRegression(n_components=10)
]
```

### 4. Let the Data Decide

Use feature augmentation to find the best combination:

```python
pipeline = [
    {
        "feature_augmentation": {
            "_or_": [SNV, MSC, Detrend, FirstDerivative, SavitzkyGolay],
            "pick": [1, 2, 3],  # Try 1, 2, or 3 methods
            "count": 10         # Generate 10 random combinations
        }
    },
    PLSRegression(n_components=10)
]
```

---

## Running These Examples

```bash
cd examples

# Run all preprocessing examples
./run.sh -n "U0*.py" -c user

# Run with visualization
python user/03_preprocessing/U01_preprocessing_basics.py --plots --show
```

## Next Steps

After mastering preprocessing:

- **Models**: Compare different model architectures
- **Cross-Validation**: Proper model evaluation
- **Explainability**: Understand which wavelengths matter
