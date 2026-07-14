# Models

This section covers model training, comparison, and optimization in NIRS4ALL.

```{toctree}
:maxdepth: 2

training
hyperparameter_tuning
native_tuning_conformal
aom_models
deep_learning
tabpfn_nirs
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

:::{grid-item-card} Native Tuning + Conformal
:link: native_tuning_conformal
:link-type: doc

Use the native `run(tuning=...)` subset, persist tuning/conformal evidence, and apply calibrated intervals.
Includes the smoke-tested tune→calibrate→predict→robustness example.
Includes a native PLS example that tunes `model.n_components`.

+++
{bdg-info}`Experimental`
:::

:::{grid-item-card} 🧠 Deep Learning
:link: deep_learning
:link-type: doc

Use TensorFlow, PyTorch, and JAX models in pipelines.

+++
{bdg-warning}`Neural Networks`
:::

:::{grid-item-card} AOM Models
:link: aom_models
:link-type: doc

Use AOM-PLS, AOM-Ridge, AutoSelector, Blender, and FastAOM with pipeline folds.

+++
{bdg-primary}`Spectroscopy`
:::

:::{grid-item-card} 🤖 TabPFN for NIRS
:link: tabpfn_nirs
:link-type: doc

Fixed-recipe TabPFN regressor that matches per-dataset HPO without it.

+++
{bdg-info}`No HPO`
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

result = nirs4all.run([
    MinMaxScaler(),
    KFold(n_splits=5),
    {"model": PLSRegression(n_components=10)}
], dataset="data/")
print(f"RMSE: {result.best_rmse:.4f}")
```

## Built-in NIRS Models

NIRS4ALL includes specialized models optimized for spectroscopy:

### PLS Variants

| Model | Description |
|-------|-------------|
| **AOMPLSRegressor** | Adaptive Operator-Mixture PLS — auto-selects best preprocessing from operator bank |
| **AOMPLSClassifier** | AOM-PLS for classification with probability calibration |
| **AOMRidgeRegressor** | AOM-Ridge with operator-bank selection and internal CV |
| **AOMRidgeAutoSelector** | Selects the best AOM-Ridge variant under outer CV |
| **AOMRidgeBlender** | Convex blend of AOM-Ridge variants; strongest general AOM-Ridge recipe |
| **FastAOMPLSRidge** | Fast screened operator-chain AOM PLS/Ridge |
| **POPPLSRegressor** | Per-Operator-Per-component PLS — different operator per component via PRESS |
| **POPPLSClassifier** | POP-PLS for classification with probability calibration |
| **PLSDA** | PLS Discriminant Analysis |
| **OPLS** / **OPLSDA** | Orthogonal PLS / OPLS-DA |
| **MBPLS** | Multi-Block PLS |
| **DiPLS** | Domain-Invariant PLS |
| **IKPLS** | Improved Kernel PLS |
| **FCKPLS** | Fractional Convolution Kernel PLS |
| **LWPLS** | Locally Weighted PLS |
| **SparsePLS** | Sparse PLS |
| **SIMPLS** | SIMPLS algorithm |

```python
from nirs4all.operators.models import AOMPLSRegressor, AOMRidgeBlender, POPPLSRegressor

# AOM-PLS: auto-selects best preprocessing from operator bank
pipeline = [
    KFold(n_splits=5),
    {"model": AOMPLSRegressor(n_components="auto", operator_bank="compact")}
]

# AOM-Ridge Blender: strong split-aware operator-mixture Ridge recipe
pipeline = [
    KFold(n_splits=5),
    {
        "model": AOMRidgeBlender(outer_cv=5, inner_cv=5),
        "train_params": {"use_pipeline_folds_for_aom": "required"},
    },
]

# POP-PLS: selects different operator per component (no holdout needed)
pipeline = [
    KFold(n_splits=5),
    {"model": POPPLSRegressor(n_components=15, auto_select=True)}
]
```

### Deep Learning

- **nicon** - 1D CNN for NIR classification/regression
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
