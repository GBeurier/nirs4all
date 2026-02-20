# Pipelines

This section covers advanced pipeline patterns and techniques.

```{toctree}
:maxdepth: 2

writing_pipelines
branching
merging
stacking
multi_source
generators
cache_optimization
```

## Overview

NIRS4ALL pipelines are flexible and powerful. This section covers advanced patterns beyond basic sequential pipelines.

::::{grid} 2
:gutter: 3

:::{grid-item-card} ✍️ Writing Pipelines
:link: writing_pipelines
:link-type: doc

Complete guide to pipeline syntax and patterns.

+++
{bdg-primary}`Start Here`
:::

:::{grid-item-card} 🌿 Branching
:link: branching
:link-type: doc

Create parallel paths with different preprocessing strategies.

+++
{bdg-success}`Parallel`
:::

:::{grid-item-card} 🔀 Merging
:link: merging
:link-type: doc

Combine branch outputs: features, predictions, or both for stacking.

+++
{bdg-primary}`Combine`
:::

:::{grid-item-card} 📚 Stacking
:link: stacking
:link-type: doc

Build meta-models that combine predictions from multiple base models.

+++
{bdg-warning}`Ensembles`
:::

:::{grid-item-card} 🔀 Multi-Source Pipelines
:link: multi_source
:link-type: doc

Handle multiple data sources with source_branch and merge_sources.

+++
{bdg-success}`Fusion`
:::

:::{grid-item-card} ⚡ Cache Optimization
:link: cache_optimization
:link-type: doc

Speed up generator-heavy pipelines with step-level caching.

+++
{bdg-warning}`Performance`
:::

::::

## Pipeline Patterns

### Sequential Pipeline
```python
pipeline = [
    MinMaxScaler(),
    SNV(),
    KFold(n_splits=5),
    {"model": PLSRegression(10)}
]
```

### Branching Pipeline
```python
pipeline = [
    {"branch": [
        [SNV(), PLSRegression(10)],
        [MSC(), RandomForestRegressor()]
    ]},
    {"merge": "predictions"}  # Combine for stacking
]
```

### Generator Pipeline
```python
pipeline = [
    {"_or_": [SNV, MSC, Detrend]},  # Try each preprocessing
    {"_range_": [1, 20, 2], "model": PLSRegression}  # Sweep n_components
]
```

## See Also

- {doc}`/reference/pipeline_syntax` - Complete syntax reference
- {doc}`/reference/generator_keywords` - Generator syntax (`_or_`, `_range_`)
- {doc}`/reference/operator_catalog` - All available operators
