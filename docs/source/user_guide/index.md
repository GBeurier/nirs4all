# User Guide

This section contains step-by-step guides for common tasks and workflows with NIRS4ALL.

```{toctree}
:maxdepth: 2

data/index
preprocessing/index
augmentation/index
pipelines/index
models/index
predictions/index
deployment/index
visualization/index
troubleshooting/index
logging
```

## Overview

The User Guide provides practical how-to guides organized by topic. Each guide focuses on a specific task and includes working code examples.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 📊 Data Handling
:link: data/index
:link-type: doc

Load and manage spectroscopic data from various formats. Filter samples and aggregate predictions.

+++
{bdg-primary}`Essential`
:::

:::{grid-item-card} 🔬 Preprocessing
:link: preprocessing/index
:link-type: doc

Master spectral preprocessing techniques: SNV, MSC, derivatives, Savitzky-Golay, and more.

+++
{bdg-success}`Spectral`
:::

:::{grid-item-card} 📈 Augmentation
:link: augmentation/index
:link-type: doc

Improve model robustness with sample and feature augmentation techniques.

+++
{bdg-info}`Training`
:::

:::{grid-item-card} 🔀 Pipelines
:link: pipelines/index
:link-type: doc

Advanced pipeline patterns: branching, merging, stacking, and generators.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} 🎯 Models
:link: models/index
:link-type: doc

Train and compare machine learning models with hyperparameter tuning.

+++
{bdg-primary}`ML`
:::

:::{grid-item-card} 📦 Deployment
:link: deployment/index
:link-type: doc

Export trained pipelines as standalone bundles for production use.

+++
{bdg-success}`Production`
:::

:::{grid-item-card} 📊 Visualization
:link: visualization/index
:link-type: doc

Analyze predictions, residuals, and explain model behavior with SHAP.

+++
{bdg-info}`Analysis`
:::

:::{grid-item-card} 🔧 Troubleshooting
:link: troubleshooting/index
:link-type: doc

Migration guides, common issues, and solutions.

+++
{bdg-warning}`Help`
:::

::::

## Additional Topics

- {doc}`logging` - Configure logging for debugging and monitoring

## See Also

- {doc}`/reference/pipeline_syntax` - Complete pipeline syntax reference
- {doc}`/reference/operator_catalog` - All available operators
- {doc}`/examples/index` - Working examples organized by topic
