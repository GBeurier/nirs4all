# Reference

Complete reference documentation for NIRS4ALL APIs, syntax, and operators.

```{toctree}
:maxdepth: 2

pipeline_syntax
operator_catalog
generator_keywords
combination_generator
configuration
metrics
predictions_api
cli
workspace
storage
```

## Overview

This section provides detailed reference documentation:

::::{grid} 2
:gutter: 3

:::{grid-item-card} 📄 Pipeline Syntax
:link: pipeline_syntax
:link-type: doc

Complete guide to writing pipelines including all step types, generator syntax, and options.

+++
{bdg-primary}`Comprehensive`
:::

:::{grid-item-card} 🧰 Operator Catalog
:link: operator_catalog
:link-type: doc

Complete listing of all available operators: preprocessors, splitters, models, and augmentations.

+++
{bdg-success}`All Operators`
:::

:::{grid-item-card} 🔧 Generator Keywords
:link: generator_keywords
:link-type: doc

Reference for `_or_`, `_range_`, `_choice_` and other generator keywords.

+++
{bdg-info}`Generators`
:::

:::{grid-item-card} ⚙️ Configuration
:link: configuration
:link-type: doc

Complete PipelineConfigs and DatasetConfigs specification.

+++
{bdg-info}`Config`
:::

::::

::::{grid} 2
:gutter: 3

:::{grid-item-card} 📊 Metrics
:link: metrics
:link-type: doc

All evaluation metrics for regression and classification.

+++
{bdg-primary}`Evaluation`
:::

:::{grid-item-card} 📈 Predictions API
:link: predictions_api
:link-type: doc

Complete guide to the Predictions and PredictionResultsList objects.

+++
{bdg-info}`Results`
:::

:::{grid-item-card} 💻 CLI Reference
:link: cli
:link-type: doc

Command-line interface documentation for all `nirs4all` workspace commands.

+++
{bdg-warning}`Commands`
:::

:::{grid-item-card} 📁 Workspace
:link: workspace
:link-type: doc

Workspace architecture and directory structure.

+++
{bdg-secondary}`Organization`
:::

::::

## API Documentation

Detailed API documentation is also available in the {doc}`/api/modules` section.

## See Also

- {doc}`/user_guide/index` - Step-by-step how-to guides
- {doc}`/developer/architecture` - Architecture overview for developers
- {doc}`/developer/artifacts` - Artifacts and storage developer guide
- {doc}`/examples/index` - Working examples organized by topic
