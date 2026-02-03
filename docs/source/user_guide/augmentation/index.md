# Data Augmentation

This section covers data augmentation techniques for NIRS data.

```{toctree}
:maxdepth: 2

augmentations
sample_augmentation_guide
synthetic_nirs_generator
```

## Overview

Data augmentation is a powerful technique to improve model robustness, especially when working with limited training data. NIRS4ALL supports both sample-level and feature-level augmentation.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🔄 Augmentation Overview
:link: augmentations
:link-type: doc

Overview of augmentation methods and strategies.

+++
{bdg-primary}`Start Here`
:::

:::{grid-item-card} 📈 Sample Augmentation Guide
:link: sample_augmentation_guide
:link-type: doc

Complete guide to augmenting training data with synthetic samples, including balanced mode for imbalanced datasets.

+++
{bdg-success}`Comprehensive`
:::

:::{grid-item-card} 🧪 Synthetic Data
:link: synthetic_nirs_generator
:link-type: doc

Generate synthetic NIRS spectra for testing and validation.

+++
{bdg-info}`Testing`
:::

::::

## Types of Augmentation

### Sample Augmentation
Create new training samples by applying transformations to existing spectra:
- Noise injection
- Baseline shifts
- Wavelength perturbations
- Mixup strategies

### Feature Augmentation
Generate multiple views of the same data through different preprocessing:
- Multiple preprocessing variants
- Different spectral regions
- Derivative combinations

## Quick Example

```python
from nirs4all.operators.augmentation import GaussianAdditiveNoise, WavelengthShift

pipeline = [
    {"sample_augmentation": {
        "transformers": [
            GaussianAdditiveNoise(sigma=0.01),
            WavelengthShift(shift_range=(-2.0, 2.0)),
        ],
        "count": 3,
    }},
    # ... rest of pipeline
]
```

## See Also

- {doc}`/user_guide/preprocessing/index` - Preprocessing techniques
- {doc}`/reference/operator_catalog` - All available operators
