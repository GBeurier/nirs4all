# Reference

Complete reference documentation for NIRS4ALL APIs, syntax, and operators.

```{toctree}
:maxdepth: 2

pipeline_syntax
pipeline_keywords
operator_catalog
transforms
augmentations
models
splitters
filters
generator_keywords
combination_generator
configuration
configuration/index
metrics
predictions_api
api/session
cli
workspace
storage
```

## Pipeline & Syntax

::::{grid} 2
:gutter: 3

:::{grid-item-card} Pipeline Syntax
:link: pipeline_syntax
:link-type: doc

Step formats, execution order, serialization rules.

+++
{bdg-primary}`Syntax`
:::

:::{grid-item-card} Pipeline Keywords
:link: pipeline_keywords
:link-type: doc

All dict keywords: `model`, `y_processing`, `branch`, `merge`, `tag`, `exclude`, and more.

+++
{bdg-primary}`Keywords`
:::

:::{grid-item-card} Generator Keywords
:link: generator_keywords
:link-type: doc

`_or_`, `_range_`, `_grid_`, `_zip_`, `_sample_` and other expansion keywords.

+++
{bdg-info}`Generators`
:::

:::{grid-item-card} Configuration
:link: configuration
:link-type: doc

PipelineConfigs, DatasetConfigs, CacheConfig specification.

+++
{bdg-info}`Config`
:::

::::

## Operators by Category

::::{grid} 3
:gutter: 3

:::{grid-item-card} Transforms
:link: transforms
:link-type: doc

SNV, MSC, derivatives, baseline, wavelets, feature selection, and more.

+++
{bdg-success}`40+ operators`
:::

:::{grid-item-card} Augmentations
:link: augmentations
:link-type: doc

Noise, baseline drift, wavelength shifts, mixup, physical effects, and more.

+++
{bdg-success}`40+ operators`
:::

:::{grid-item-card} Models
:link: models
:link-type: doc

Built-in PLS variants (AOM-PLS, POP-PLS, OPLS, DiPLS, etc.) and deep learning.

+++
{bdg-success}`25+ models`
:::

:::{grid-item-card} Splitters
:link: splitters
:link-type: doc

Kennard-Stone, SPXY, K-means, and sklearn splitters.

+++
{bdg-success}`Splitters`
:::

:::{grid-item-card} Filters
:link: filters
:link-type: doc

Y-outlier, X-outlier, spectral quality, high leverage, metadata filters.

+++
{bdg-success}`Filters`
:::

:::{grid-item-card} Operator Catalog
:link: operator_catalog
:link-type: doc

Full catalog including sklearn and deep learning operators.

+++
{bdg-secondary}`Full List`
:::

::::

## Results & Deployment

::::{grid} 2
:gutter: 3

:::{grid-item-card} Metrics
:link: metrics
:link-type: doc

All evaluation metrics for regression and classification.

+++
{bdg-primary}`Evaluation`
:::

:::{grid-item-card} Predictions API
:link: predictions_api
:link-type: doc

Predictions and PredictionResultsList objects.

+++
{bdg-info}`Results`
:::

:::{grid-item-card} Session API
:link: api/session
:link-type: doc

Stateful Session workflows and model persistence.

+++
{bdg-info}`Stateful`
:::

:::{grid-item-card} CLI Reference
:link: cli
:link-type: doc

Command-line interface for workspace commands.

+++
{bdg-warning}`Commands`
:::

:::{grid-item-card} Workspace
:link: workspace
:link-type: doc

Workspace architecture and directory structure.

+++
{bdg-secondary}`Organization`
:::

:::{grid-item-card} Storage
:link: storage
:link-type: doc

SQLite metadata store, Parquet arrays, artifact storage.

+++
{bdg-secondary}`Storage`
:::

::::

## API Documentation

Detailed API documentation is also available in the {doc}`/api/modules` section.

## See Also

- {doc}`/user_guide/index` - Step-by-step how-to guides
- {doc}`/developer/architecture` - Architecture overview for developers
- {doc}`/developer/artifacts` - Artifacts and storage developer guide
- {doc}`/examples/index` - Working examples organized by topic
