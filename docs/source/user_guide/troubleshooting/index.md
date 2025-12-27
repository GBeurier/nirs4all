# Troubleshooting

This section covers migration guides, common issues, and solutions.

```{toctree}
:maxdepth: 2

migration
dataset_troubleshooting
faq
```

## Overview

This section helps you resolve common issues and migrate from older versions of NIRS4ALL.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🔄 Migration Guide
:link: migration
:link-type: doc

Complete guide to upgrading from older NIRS4ALL versions, including API changes, dataset configuration updates, and prediction format migration.

+++
{bdg-primary}`Upgrading`
:::

:::{grid-item-card} 🔧 Dataset Issues
:link: dataset_troubleshooting
:link-type: doc

Common data loading problems and solutions.

+++
{bdg-success}`Debugging`
:::

:::{grid-item-card} ❓ FAQ
:link: faq
:link-type: doc

Frequently asked questions and solutions.

+++
{bdg-info}`Help`
:::

::::

## Quick Fixes

### ImportError: No module named 'nirs4all'

```bash
pip install nirs4all
# or for development
pip install -e .
```

### TensorFlow/PyTorch not found

Install the optional dependency:
```bash
pip install nirs4all[tensorflow]  # or [torch], [jax]
```

### Out of Memory

```python
# Reduce batch size in deep learning models
pipeline = [
    {"model": decon(batch_size=16)}  # Smaller batch
]

# Or reduce cross-validation folds
from sklearn.model_selection import KFold
pipeline = [
    KFold(n_splits=3),  # Instead of 10
    {"model": ...}
]
```

### NaN in Predictions

Common causes:
1. **Missing values in data** - Use imputation or filtering
2. **Incompatible preprocessing** - Check spectral range
3. **Numerical instability** - Add `MinMaxScaler()` before model

## See Also

- {doc}`/getting_started/index` - Installation guide
- {doc}`/reference/cli` - Diagnostic commands
