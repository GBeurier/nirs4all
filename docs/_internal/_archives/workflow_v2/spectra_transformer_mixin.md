# SpectraTransformerMixin: Implementation Notes

**Status**: Implemented (Phase 1 of Operator Migration)
**Related**: [synthetic_generator_operator_migration_roadmap.md](./synthetic_generator_operator_migration_roadmap.md)

---

## Overview

The `SpectraTransformerMixin` is a base class that enables wavelength-aware transformations while maintaining full sklearn compatibility. It was created to allow operators from the synthetic generator (environmental effects, scattering simulations) to be used directly in nirs4all pipelines.

## Design Rationale

### Problem Statement

The synthetic data generator contains sophisticated spectral transformations (temperature effects, moisture effects, particle scattering) that require wavelength information to apply region-specific effects. These transformations could not be used in standard nirs4all pipelines because:

1. sklearn's `TransformerMixin.transform()` only accepts `X` and optionally `y`
2. Pipelines had no mechanism to pass wavelengths to operators
3. The controller system didn't detect wavelength requirements

### Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline Step                             │
├─────────────────────────────────────────────────────────────────┤
│  TransformerMixinController                                      │
│  ├─ Detects SpectraTransformerMixin instances                   │
│  ├─ Checks _requires_wavelengths flag                           │
│  ├─ Extracts wavelengths from dataset                           │
│  └─ Passes wavelengths to transform()                           │
├─────────────────────────────────────────────────────────────────┤
│  SpectraTransformerMixin Operator                               │
│  ├─ transform(X, wavelengths=...) → delegates                   │
│  └─ transform_with_wavelengths(X, wavelengths) → actual logic   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Backward Compatibility First

Standard `TransformerMixin` operators continue to work unchanged. The controller only passes wavelengths to operators that:
- Inherit from `SpectraTransformerMixin`
- Have `_requires_wavelengths = True` (the default)

```python
# Standard sklearn - works as before
from sklearn.preprocessing import StandardScaler
pipeline = [StandardScaler(), PLSRegression(10)]

# Wavelength-aware - automatically receives wavelengths
from nirs4all.operators.augmentation import TemperatureAugmenter
pipeline = [TemperatureAugmenter(temperature_delta=5.0), PLSRegression(10)]
```

### 2. Class-Level Flag for Wavelength Requirements

The `_requires_wavelengths` class attribute indicates whether wavelengths are mandatory:

```python
class TemperatureAugmenter(SpectraTransformerMixin):
    _requires_wavelengths = True  # Mandatory - error if not provided

class OptionalWavelengthOperator(SpectraTransformerMixin):
    _requires_wavelengths = False  # Optional - works with or without
```

### 3. Dual Interface Support

Operators can be called in two ways:

```python
# Via transform() with keyword argument (pipeline usage)
X_transformed = op.transform(X, wavelengths=wavelengths)

# Via transform_with_wavelengths() directly (explicit usage)
X_transformed = op.transform_with_wavelengths(X, wavelengths)
```

### 4. No Strict ABC Enforcement

The mixin uses `@abstractmethod` but does not use `ABCMeta` as metaclass. This follows sklearn conventions and allows:
- Instantiation for testing `get_params()`/`set_params()` without implementing abstract methods
- Clear `NotImplementedError` messages when abstract methods are called

## Controller Integration

### Detection Logic

The `TransformerMixinController` uses static helper methods for clean detection:

```python
@staticmethod
def _needs_wavelengths(operator: Any) -> bool:
    """Check if operator requires wavelengths."""
    return (
        isinstance(operator, SpectraTransformerMixin) and
        getattr(operator, '_requires_wavelengths', False)
    )
```

### Wavelength Extraction

Wavelengths are extracted from the dataset with a fallback mechanism:

```python
@staticmethod
def _extract_wavelengths(dataset: SpectroDataset, source_index: int) -> np.ndarray:
    """Extract wavelengths from dataset."""
    try:
        return dataset.wavelengths_nm(source_index)
    except (ValueError, AttributeError):
        # Fallback to numeric headers
        wavelengths = dataset.float_headers(source_index)
        if wavelengths is None:
            raise ValueError(
                "Operator requires wavelengths but dataset has no wavelength information."
            )
        return wavelengths
```

### Execution Paths Updated

Three execution paths in the controller were updated to support wavelength passing:
1. Main `execute()` loop for standard transforms
2. `_execute_for_sample_augmentation()` for batch augmentation
3. `_execute_for_sample_augmentation_sequential()` for fallback sequential augmentation

## Creating a Wavelength-Aware Operator

### Basic Template

```python
from nirs4all.operators.base import SpectraTransformerMixin
import numpy as np

class MyWavelengthAwareOperator(SpectraTransformerMixin):
    """Apply wavelength-dependent transformation."""

    _requires_wavelengths = True  # Set to False if optional

    def __init__(self, param1: float = 1.0, param2: str = "default"):
        self.param1 = param1
        self.param2 = param2

    def transform_with_wavelengths(
        self,
        X: np.ndarray,
        wavelengths: np.ndarray
    ) -> np.ndarray:
        """
        Apply the transformation.

        Args:
            X: Input spectra (n_samples, n_features)
            wavelengths: Wavelength array in nm (n_features,)

        Returns:
            Transformed spectra (n_samples, n_features)
        """
        # Example: apply wavelength-dependent scaling
        scale_factors = self._compute_scale_factors(wavelengths)
        return X * scale_factors

    def _compute_scale_factors(self, wavelengths: np.ndarray) -> np.ndarray:
        """Compute per-wavelength scale factors."""
        # Implementation details...
        return np.ones_like(wavelengths)
```

### sklearn Compatibility Checklist

1. **All parameters in `__init__`**: Every parameter must be stored as an instance attribute with the same name
2. **No required positional arguments**: Use keyword arguments with defaults
3. **fit() returns self**: The default implementation in `SpectraTransformerMixin` handles this
4. **Implement `_more_tags()` for sklearn introspection** (optional):

```python
def _more_tags(self):
    return {
        'requires_wavelengths': self._requires_wavelengths,
        'stateless': True,
    }
```

## Implemented Operators

### Environmental Effects (`nirs4all.operators.augmentation.environmental`)

| Operator | Description | Key Parameters |
|----------|-------------|----------------|
| `TemperatureAugmenter` | Temperature-induced spectral changes | `temperature_delta`, `temperature_range`, `enable_shift`, `enable_intensity`, `enable_broadening` |
| `MoistureAugmenter` | Moisture/water activity effects | `water_activity_delta`, `water_activity_range`, `free_water_fraction` |

### Scattering Effects (`nirs4all.operators.augmentation.scattering`)

| Operator | Description | Key Parameters |
|----------|-------------|----------------|
| `ParticleSizeAugmenter` | Particle size scattering | `mean_size_um`, `size_variation_um`, `wavelength_exponent` |
| `EMSCDistortionAugmenter` | EMSC-style scatter distortions | `multiplicative_range`, `additive_range`, `polynomial_order` |

## Testing Guidelines

### Unit Test Patterns

```python
import pytest
import numpy as np
from nirs4all.operators.base import SpectraTransformerMixin

class TestMyOperator:
    @pytest.fixture
    def operator(self):
        return MyWavelengthAwareOperator(param1=2.0)

    @pytest.fixture
    def sample_data(self):
        X = np.random.randn(10, 100)
        wavelengths = np.linspace(900, 2500, 100)
        return X, wavelengths

    def test_transform_preserves_shape(self, operator, sample_data):
        X, wavelengths = sample_data
        X_transformed = operator.transform(X, wavelengths=wavelengths)
        assert X_transformed.shape == X.shape

    def test_requires_wavelengths(self, operator, sample_data):
        X, _ = sample_data
        with pytest.raises(ValueError, match="requires wavelengths"):
            operator.transform(X)

    def test_sklearn_clone(self, operator):
        from sklearn.base import clone
        cloned = clone(operator)
        assert cloned.param1 == operator.param1
        assert cloned is not operator
```

### Integration Test Patterns

```python
import nirs4all
from nirs4all.operators.augmentation import TemperatureAugmenter
from sklearn.cross_decomposition import PLSRegression

def test_pipeline_with_spectra_transformer():
    # Generate synthetic data with wavelength headers
    dataset = nirs4all.generate(n_samples=100, complexity="simple")

    pipeline = [
        TemperatureAugmenter(temperature_delta=5.0),
        PLSRegression(n_components=5),
    ]

    result = nirs4all.run(pipeline=pipeline, dataset=dataset)
    assert result.best_rmse is not None
```

## File Locations

| File | Purpose |
|------|---------|
| `nirs4all/operators/base/__init__.py` | Package exports |
| `nirs4all/operators/base/spectra_mixin.py` | SpectraTransformerMixin implementation |
| `nirs4all/controllers/transforms/transformer.py` | Controller with wavelength support |
| `tests/unit/operators/base/test_spectra_mixin.py` | Unit tests |
| `tests/unit/controllers/test_transformer_wavelengths.py` | Controller unit tests |
| `tests/integration/pipeline/test_spectra_transformer_pipeline.py` | Integration tests |

## Migration from Legacy Code

The synthetic generator's internal simulators have been replaced by operators:

| Legacy Class | Replacement Operator |
|--------------|---------------------|
| `TemperatureEffectSimulator` | `TemperatureAugmenter` |
| `MoistureEffectSimulator` | `MoistureAugmenter` |
| `ParticleSizeSimulator` | `ParticleSizeAugmenter` |
| `EMSCTransformSimulator` | `EMSCDistortionAugmenter` |

The generator now always uses operators when environmental or scattering configs are provided.
