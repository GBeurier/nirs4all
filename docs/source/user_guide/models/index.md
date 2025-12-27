# Models

This section covers model training, comparison, and optimization in NIRS4ALL.

```{toctree}
:maxdepth: 2

training
hyperparameter_tuning
deep_learning
```

## Overview

NIRS4ALL supports a wide range of machine learning models from different frameworks. This section covers model training workflows.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🎯 Model Training
:link: training
:link-type: doc

Train models with cross-validation and evaluate performance.

+++
{bdg-primary}`Essential`
:::

:::{grid-item-card} ⚙️ Hyperparameter Tuning
:link: hyperparameter_tuning
:link-type: doc

Optimize models with Optuna integration and grid search.

+++
{bdg-success}`Optimization`
:::

:::{grid-item-card} 🧠 Deep Learning
:link: deep_learning
:link-type: doc

Use TensorFlow, PyTorch, and JAX models in pipelines.

+++
{bdg-warning}`Neural Networks`
:::

::::

## Supported Model Frameworks

| Framework | Import | Examples |
|-----------|--------|----------|
| **scikit-learn** | Direct | PLSRegression, RandomForest, SVM |
| **TensorFlow/Keras** | `nirs4all[tensorflow]` | Sequential, Custom architectures |
| **PyTorch** | `nirs4all[torch]` | nn.Module subclasses |
| **JAX** | `nirs4all[jax]` | Flax/Haiku models |

## Quick Example

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),
    {"model": PLSRegression(n_components=10)}
]

result = nirs4all.run(pipeline, dataset="data/")
print(f"RMSE: {result.best_rmse:.4f}")
```

## Built-in NIRS Models

NIRS4ALL includes specialized models optimized for spectroscopy:

- **nicon** - 1D CNN for NIR classification
- **decon** - 1D CNN for NIR regression

```python
from nirs4all.operators.models import nicon, decon

pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),
    {"model": decon(n_outputs=1)}
]
```

## See Also

- {doc}`/reference/operator_catalog` - All available operators
- {doc}`/user_guide/pipelines/stacking` - Model stacking and ensembles
- {doc}`/developer/index` - Custom model integration
