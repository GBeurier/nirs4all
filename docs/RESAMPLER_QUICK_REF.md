# Resampler Quick Reference

## Import
```python
from nirs4all.operators.transformations import Resampler
import numpy as np
```

## Basic Usage
```python
# Define target wavelengths
target_wl = np.linspace(1000, 2500, 100)

# Create resampler
resampler = Resampler(target_wavelengths=target_wl)

# Add to pipeline
pipeline = [
    MinMaxScaler(),
    resampler,
    # ... rest of pipeline
]
```

## Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_wavelengths` | array or list | **required** | Target wavelength grid(s) |
| `method` | str | `'linear'` | Interpolation method |
| `crop_range` | tuple | `None` | (min, max) to crop before resampling |
| `fill_value` | float/str | `0` | Value for out-of-bounds wavelengths |
| `bounds_error` | bool | `False` | Raise error if out of bounds |
| `copy` | bool | `True` | Copy input data |

## Interpolation Methods

- `'linear'` - Linear (fast, recommended)
- `'cubic'` - Cubic spline (smooth)
- `'quadratic'` - Quadratic
- `'nearest'` - Nearest neighbor
- `'zero'`, `'slinear'`, `'previous'`, `'next'` - Other scipy methods

## Quick Examples

### Downsample
```python
Resampler(target_wavelengths=np.linspace(1000, 2500, 50))
```

### Upsample
```python
Resampler(target_wavelengths=np.linspace(1000, 2500, 500), method='cubic')
```

### Crop Region
```python
Resampler(
    target_wavelengths=np.linspace(1200, 2200, 100),
    crop_range=(1100, 2300)
)
```

### Multi-Source
```python
Resampler(target_wavelengths=[
    np.linspace(1000, 2500, 100),  # Source 0
    np.linspace(1100, 2400, 120),  # Source 1
])
```

### Allow Extrapolation
```python
Resampler(
    target_wavelengths=np.linspace(900, 2600, 150),
    fill_value='extrapolate'
)
```

## Warnings & Errors

| Issue | Behavior |
|-------|----------|
| Target extends beyond original range | ⚠️ Warning (uses fill_value) |
| No overlap with original range | ❌ ValueError |
| Headers not numeric | ❌ ValueError |
| Shape mismatch in transform | ❌ ValueError |
| Crop excludes all wavelengths | ❌ ValueError |

## Full Pipeline Example
```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression
from nirs4all.operators.transformations import Resampler, StandardNormalVariate
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.dataset import DatasetConfigs

# Target wavelengths
target_wl = np.linspace(1000, 2500, 100)

# Pipeline
pipeline = [
    MinMaxScaler(),
    Resampler(target_wavelengths=target_wl, method='linear'),
    StandardNormalVariate(),
    ShuffleSplit(n_splits=5, test_size=0.25),
    {"y_processing": MinMaxScaler()},
    {"model": PLSRegression(n_components=15)},
]

# Run
dataset_config = DatasetConfigs("path/to/data")
pipeline_config = PipelineConfigs(pipeline, name="Resampled_Pipeline")
runner = PipelineRunner(save_files=True, verbose=1)
predictions, _ = runner.run(pipeline_config, dataset_config)

# Results
top_models = predictions.top_k(5, 'rmse')
```

## Tips

💡 **Use linear for speed**: `method='linear'` is fastest and works well for most cases

💡 **Downsample to reduce computation**: Fewer wavelengths = faster training

💡 **Crop before resampling**: Focus on informative regions with `crop_range`

💡 **Check headers**: Ensure dataset headers are numeric wavelengths

💡 **Multi-source flexibility**: Use list of targets for different grids per source

## See Full Documentation
- `docs/RESAMPLER.md` - Complete guide
- `examples/Q10_resampler.py` - Working examples
- `tests/integration_tests/test_resampler.py` - Test cases
