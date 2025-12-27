# User Guide

This section contains step-by-step guides for common tasks and workflows with NIRS4ALL.

```{toctree}
:maxdepth: 2

preprocessing
branching_merging
stacking
export_deploy
```

## Overview

The User Guide provides practical how-to guides organized by topic. Each guide focuses on a specific task and includes working code examples.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🔬 Preprocessing
:link: preprocessing
:link-type: doc

Master spectral preprocessing techniques: SNV, MSC, derivatives, Savitzky-Golay, and more.

+++
See {doc}`preprocessing`
:::

:::{grid-item-card} 🌿 Branching & Merging
:link: branching_merging
:link-type: doc

Create parallel pipelines with different preprocessing strategies.

+++
See {doc}`branching_merging`
:::

:::{grid-item-card} 📚 Stacking
:link: stacking
:link-type: doc

Build meta-models that combine predictions from multiple base models.

+++
See {doc}`stacking`
:::

:::{grid-item-card} 📦 Export & Deploy
:link: export_deploy
:link-type: doc

Package trained pipelines as standalone bundles for production.

+++
See {doc}`export_deploy`
:::

::::

## Coming Soon

The following guides are planned:

- **Data Handling** - Loading and managing spectroscopic data from various formats
- **Models** - Training and comparing different machine learning models
- **Cross-Validation** - Proper validation strategies including group-based splitting
- **Hyperparameter Tuning** - Optimize pipelines with Optuna integration
- **Explainability** - SHAP values and feature importance analysis
- **Logging** - Configure logging for debugging and monitoring

:::{tip}
For now, see the {doc}`/examples/index` for working examples of these features.
:::

## See Also

- {doc}`/reference/pipeline_syntax` - Complete pipeline syntax reference
- {doc}`/reference/operator_catalog` - All available operators
- {doc}`/examples/index` - Working examples organized by topic
