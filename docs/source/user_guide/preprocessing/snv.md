# Standard Normal Variate (SNV) Transformation

## Overview

Standard Normal Variate (SNV) is a scatter correction technique commonly used in Near-Infrared Spectroscopy (NIRS) and other spectroscopic applications. It normalizes each spectrum (sample) individually to remove multiplicative scatter effects.

## Implementation

The `StandardNormalVariate` class in `nirs4all` provides a flexible implementation that can work in two modes:

### 1. Row-wise SNV (Default - Spectroscopy Standard)

By default, SNV operates row-wise (axis=1), which is the standard approach in spectroscopy. Each spectrum (row) is centered and scaled independently:

```python
from nirs4all.operators.transforms import StandardNormalVariate
import numpy as np

# Example spectral data (3 samples, 5 wavelengths)
X = np.array([[1, 2, 3, 4, 5],
              [10, 20, 30, 40, 50],
              [100, 200, 300, 400, 500]], dtype=float)

# Apply SNV (row-wise by default)
snv = StandardNormalVariate()
X_transformed = snv.fit_transform(X)

# Each row now has mean≈0 and std≈1
```

**Formula (per sample):**
```
SNV(x) = (x - mean(x)) / std(x)
```

### 2. Column-wise SNV (Like StandardScaler)

You can also apply SNV column-wise (axis=0), which makes it equivalent to sklearn's StandardScaler:

```python
# Apply SNV column-wise
snv_colwise = StandardNormalVariate(axis=0)
X_transformed = snv_colwise.fit_transform(X)

# Each column now has mean≈0 and std≈1
```

## Parameters

- **axis** (int, default=1): Axis along which to compute mean and standard deviation
  - `axis=1`: Row-wise (default, standard SNV for spectroscopy)
  - `axis=0`: Column-wise (equivalent to StandardScaler)

- **with_mean** (bool, default=True): If True, center the data before scaling

- **with_std** (bool, default=True): If True, scale the data to unit variance

- **ddof** (int, default=0): Delta Degrees of Freedom for standard deviation calculation

- **copy** (bool, default=True): If False, try to avoid a copy and do inplace scaling

## Use Cases

### Row-wise SNV (axis=1) - Spectroscopy
- **Purpose**: Remove multiplicative scatter effects from individual spectra
- **When to use**: Standard preprocessing for NIRS, Raman, and other spectroscopic data
- **Effect**: Each spectrum is normalized independently, removing baseline shifts and scaling differences

### Column-wise SNV (axis=0) - Feature Scaling
- **Purpose**: Standardize features across samples
- **When to use**: When you want to normalize features (wavelengths) rather than samples
- **Effect**: Equivalent to sklearn's StandardScaler

## Examples

### Example 1: Basic SNV for Spectroscopy

```python
from nirs4all.operators.transforms import StandardNormalVariate
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Create preprocessing pipeline
pipeline = Pipeline([
    ('snv', StandardNormalVariate()),  # Row-wise SNV
    ('pca', PCA(n_components=10))
])

# Apply to spectral data
X_processed = pipeline.fit_transform(X_spectra)
```

### Example 2: SNV with Other Preprocessing

```python
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    SavitzkyGolay,
    MultiplicativeScatterCorrection
)
from sklearn.pipeline import Pipeline

# Compare different scatter correction methods
pipelines = {
    'snv': Pipeline([('snv', StandardNormalVariate())]),
    'msc': Pipeline([('msc', MultiplicativeScatterCorrection())]),
    'snv+savgol': Pipeline([
        ('snv', StandardNormalVariate()),
        ('savgol', SavitzkyGolay())
    ])
}
```

### Example 3: Column-wise Standardization

```python
# If you need column-wise standardization (like StandardScaler)
snv_colwise = StandardNormalVariate(axis=0)
X_scaled = snv_colwise.fit_transform(X)
```

## Technical Details

### Mathematical Formulation

For each sample (row) when axis=1:
```
x_i,transformed = (x_i - μ_i) / σ_i
```

Where:
- `x_i` is the i-th sample (spectrum)
- `μ_i` is the mean of the i-th sample
- `σ_i` is the standard deviation of the i-th sample

### Handling Edge Cases

- **Zero standard deviation**: If a sample has zero standard deviation (all values are the same), the std is set to 1.0 to avoid division by zero
- **Sparse matrices**: Not supported (will raise TypeError)

## Comparison with StandardScaler

| Feature | StandardNormalVariate (axis=1) | StandardNormalVariate (axis=0) | sklearn.StandardScaler |
|---------|-------------------------------|-------------------------------|------------------------|
| Default behavior | Row-wise (per sample) | Column-wise (per feature) | Column-wise (per feature) |
| Typical use case | Spectroscopy | Feature scaling | Feature scaling |
| Fits parameters | No (stateless) | No (stateless) | Yes (stores mean/std) |
| Memory in pipeline | Minimal | Minimal | Stores statistics |

## Migration from Previous Implementation

If you were using the old alias (`StandardScaler` as `StandardNormalVariate`), the behavior has changed:

**Old behavior (column-wise):**
```python
# This used to be sklearn's StandardScaler (column-wise)
snv = StandardNormalVariate()
```

**New behavior (row-wise - proper SNV):**
```python
# Now it's true row-wise SNV by default
snv = StandardNormalVariate()  # Row-wise by default

# To get the old behavior (column-wise):
snv = StandardNormalVariate(axis=0)
```

## References

- Barnes, R. J., Dhanoa, M. S., & Lister, S. J. (1989). Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra. *Applied Spectroscopy*, 43(5), 772-777.
- Rinnan, Å., van den Berg, F., & Engelsen, S. B. (2009). Review of the most common pre-processing techniques for near-infrared spectra. *TrAC Trends in Analytical Chemistry*, 28(10), 1201-1222.

## See Also

- `MultiplicativeScatterCorrection`: Another scatter correction method
- `SavitzkyGolay`: Smoothing and derivative calculation
- `sklearn.preprocessing.StandardScaler`: Column-wise standardization
