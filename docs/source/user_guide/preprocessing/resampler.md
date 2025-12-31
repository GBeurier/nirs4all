# Resampler - Wavelength Grid Resampling

The `Resampler` operator allows you to resample spectral data to a different wavelength grid using various interpolation methods from scipy. This is particularly useful for:

- **Standardizing wavelength grids** across different instruments or datasets
- **Reducing dimensionality** by downsampling to fewer wavelengths
- **Increasing resolution** by upsampling with interpolation
- **Focusing on specific regions** by cropping and resampling
- **Preparing data for transfer learning** across different spectral resolutions

## Basic Usage

```python
from nirs4all.operators.transforms import Resampler
import numpy as np

# Define target wavelength grid
target_wavelengths = np.linspace(1000, 2500, 100)

# Create resampler
resampler = Resampler(
    target_wavelengths=target_wavelengths,
    method='linear'
)

# Use in pipeline
pipeline = [
    MinMaxScaler(),
    resampler,
    StandardNormalVariate(),
    # ... rest of pipeline
]
```

## Parameters

### `target_wavelengths` (required)
Array-like or list of target wavelength values. Can be:
- Single array: same grid for all sources
- List of arrays: different grid per source (for multi-source datasets)

```python
# Same grid for all sources
target_wl = np.linspace(1000, 2500, 100)
resampler = Resampler(target_wavelengths=target_wl)

# Different grid per source
target_wl_list = [
    np.linspace(1000, 2500, 100),  # Source 0
    np.linspace(1100, 2400, 120),  # Source 1
]
resampler = Resampler(target_wavelengths=target_wl_list)
```

### `method` (default: 'linear')
Interpolation method from scipy.interpolate.interp1d. Options include:
- `'linear'`: Linear interpolation (fast, good for most cases)
- `'cubic'`: Cubic spline interpolation (smoother)
- `'quadratic'`: Quadratic interpolation
- `'nearest'`: Nearest neighbor (no interpolation)
- `'slinear'`, `'zero'`, `'previous'`, `'next'`: Other scipy methods

```python
resampler = Resampler(
    target_wavelengths=target_wl,
    method='cubic'  # Smooth interpolation
)
```

### `crop_range` (optional)
Tuple `(min_wavelength, max_wavelength)` to crop the original data before resampling.
Useful for focusing on specific spectral regions.

```python
# Focus on mid-infrared region
resampler = Resampler(
    target_wavelengths=np.linspace(1200, 2200, 100),
    crop_range=(1100, 2300)  # Crop before resampling
)
```

### `fill_value` (default: 0)
Value to use for wavelengths outside the original range. Can be:
- A number (e.g., `0`, `np.nan`)
- `'extrapolate'`: Use scipy's extrapolation

```python
resampler = Resampler(
    target_wavelengths=target_wl,
    fill_value='extrapolate'  # Extrapolate beyond original range
)
```

### `bounds_error` (default: False)
If True, raise error when target wavelengths are outside original range.
If False, use `fill_value` instead.

```python
resampler = Resampler(
    target_wavelengths=target_wl,
    bounds_error=True  # Strict: raise error if out of bounds
)
```

### `copy` (default: True)
Whether to copy the input data before transformation.

## Examples

### Example 1: Downsampling for Faster Processing

```python
from nirs4all.operators.transforms import Resampler
import numpy as np

# Reduce from 200 to 50 wavelengths
target_wl = np.linspace(1000, 2500, 50)

pipeline = [
    MinMaxScaler(),
    Resampler(target_wavelengths=target_wl, method='linear'),
    StandardNormalVariate(),
    ShuffleSplit(n_splits=5),
    {"y_processing": MinMaxScaler()},
    {"model": PLSRegression(n_components=10)},
]
```

### Example 2: Upsampling for Higher Resolution

```python
# Increase from 200 to 500 wavelengths using cubic interpolation
target_wl = np.linspace(1000, 2500, 500)

pipeline = [
    Resampler(target_wavelengths=target_wl, method='cubic'),
    # ... rest of pipeline
]
```

### Example 3: Focusing on Specific Wavelength Range

```python
# Focus on fingerprint region (1300-1800 nm)
target_wl = np.linspace(1300, 1800, 100)

pipeline = [
    Resampler(
        target_wavelengths=target_wl,
        crop_range=(1250, 1850),  # Crop with buffer
        method='linear'
    ),
    # ... rest of pipeline
]
```

### Example 4: Multi-Source with Different Target Grids

```python
# Different sampling for each source
target_wl_list = [
    np.linspace(1000, 2500, 100),  # Source 0: standard resolution
    np.linspace(1100, 2300, 150),  # Source 1: higher resolution in narrower range
]

pipeline = [
    Resampler(target_wavelengths=target_wl_list),
    # ... rest of pipeline
]
```

### Example 5: Comparing Interpolation Methods

```python
# Test different interpolation methods
target_wl = np.linspace(1000, 2500, 100)

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3),
    {"y_processing": MinMaxScaler()},
]

# Add models with different resampling methods
for method in ['linear', 'cubic', 'quadratic']:
    pipeline.extend([
        {"model": Resampler(target_wavelengths=target_wl, method=method),
         "name": f"Resample_{method}"},
        {"model": PLSRegression(n_components=15),
         "name": f"PLS_with_{method}_resampling"}
    ])
```

## How It Works

1. **Wavelength Extraction**: The controller automatically extracts wavelength information from `dataset.headers(source_idx)` and converts to float.

2. **Validation**: Ensures headers are numeric wavelengths (raises error if not convertible to float).

3. **Cropping** (optional): If `crop_range` is specified, crops to that range first.

4. **Interpolation**: Uses scipy's `interp1d` to interpolate each spectrum to the target wavelength grid.

5. **Header Update**: Updates dataset headers with the new wavelength values.

6. **Preprocessing Name**: Adds a new preprocessing name like `"raw_Resampler_1"` following the standard naming convention.

## Controller Integration

The `ResamplerController` integrates the Resampler into the pipeline:

- **Multi-source support**: ✓ Yes
- **Prediction mode support**: ✓ Yes (stores interpolation parameters)
- **Priority**: 15 (runs before most transformers)
- **Matching**: Detects `Resampler` instances in pipeline steps

## Warnings and Errors

### ⚠️ Warnings

- **Extrapolation Warning**: If target wavelengths extend beyond the original range, a warning is issued (unless `fill_value='extrapolate'`).

```
UserWarning: Target wavelengths extend below 1000.0 and above 2500.0 original range.
Using fill_value=0 for extrapolation.
```

### ❌ Errors

- **No wavelength overlap**: If crop_range or target_wavelengths have no overlap with original wavelengths.
- **Invalid wavelengths**: If dataset headers cannot be converted to float.
- **Shape mismatch**: If transform data doesn't match fitted dimensions.

## Best Practices

1. **Choose appropriate method**:
   - Use `'linear'` for most cases (fast, reliable)
   - Use `'cubic'` for smoother spectra (slower)
   - Avoid `'nearest'` unless you specifically want no interpolation

2. **Consider computational cost**:
   - Downsampling reduces computation time in subsequent steps
   - Upsampling increases computation time but may preserve more information

3. **Validate wavelength ranges**:
   - Ensure target wavelengths are within or close to original range
   - Use `crop_range` to focus on regions of interest

4. **Multi-source datasets**:
   - Provide different target grids if sources have different spectral characteristics
   - Or use same grid to standardize across sources

5. **Save resampled data**:
   - The resampler parameters are automatically saved when using `save_artifacts=True`
   - This ensures consistent resampling in prediction mode

## Technical Details

### Serialization

The resampler stores these parameters for prediction mode:
- `target_wavelengths`: Target grid
- `method`: Interpolation method
- `fill_value`: Fill value for out-of-bounds
- `bounds_error`: Bounds error flag
- `crop_mask_`: Boolean mask for cropping (if used)

### Performance

- Time complexity: O(n_samples × n_features × n_target) for interpolation
- Memory: Stores only the interpolation parameters, not full interpolator objects
- Parallelization: Processes each sample independently (future: parallel processing)

## See Also

- {doc}`/user_guide/preprocessing/snv` - Spectral normalization
- {doc}`/user_guide/preprocessing/cheatsheet` - Preprocessing cheatsheet
- {doc}`/user_guide/preprocessing/index` - All preprocessing options

## References

- scipy.interpolate.interp1d documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
