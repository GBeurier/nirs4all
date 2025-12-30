# nirs4all.generate API

Synthetic NIRS data generation API.

```{eval-rst}
.. currentmodule:: nirs4all.api.generate
```

## Overview

The `nirs4all.generate` module provides a comprehensive API for generating synthetic Near-Infrared Spectroscopy (NIRS) data. The generator is based on Beer-Lambert law physics with realistic instrumental effects.

## Main Function

```{eval-rst}
.. autofunction:: generate
```

## Convenience Functions

### Regression

```{eval-rst}
.. autofunction:: regression
```

### Classification

```{eval-rst}
.. autofunction:: classification
```

### Multi-Source

```{eval-rst}
.. autofunction:: multi_source
```

### Builder Access

```{eval-rst}
.. autofunction:: builder
```

## Export Functions

### To Folder

```{eval-rst}
.. autofunction:: to_folder
```

### To CSV

```{eval-rst}
.. autofunction:: to_csv
```

### From Template

```{eval-rst}
.. autofunction:: from_template
```

## Quick Reference

### Basic Usage

```python
import nirs4all

# Generate SpectroDataset
dataset = nirs4all.generate(n_samples=1000, random_state=42)

# Generate numpy arrays
X, y = nirs4all.generate(n_samples=500, as_dataset=False)

# Regression with scaling
dataset = nirs4all.generate.regression(
    n_samples=1000,
    target_range=(0, 100),
    random_state=42
)

# Classification
dataset = nirs4all.generate.classification(
    n_samples=500,
    n_classes=3,
    random_state=42
)
```

### Builder Pattern

```python
from nirs4all.data.synthetic import SyntheticDatasetBuilder

dataset = (
    SyntheticDatasetBuilder(n_samples=1000, random_state=42)
    .with_features(complexity="realistic")
    .with_targets(distribution="lognormal", range=(0, 100))
    .with_partitions(train_ratio=0.8)
    .build()
)
```

## See Also

- {doc}`/user_guide/data/synthetic_data` - User guide for synthetic data
- {doc}`nirs4all.data.synthetic` - Low-level generator classes
