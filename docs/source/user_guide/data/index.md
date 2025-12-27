# Data Handling

This section covers loading and managing spectroscopic data in NIRS4ALL.

```{toctree}
:maxdepth: 2

loading_data
aggregation
sample_filtering
```

## Overview

NIRS4ALL provides flexible data handling capabilities for loading spectroscopic data from various formats, filtering samples, and aggregating predictions.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 📂 Loading Data
:link: loading_data
:link-type: doc

Load data from CSV, Excel, MATLAB, NumPy, and Parquet formats.

+++
{bdg-primary}`Essential`
:::

:::{grid-item-card} 📊 Aggregation
:link: aggregation
:link-type: doc

Combine predictions across multiple samples or replicates.

+++
{bdg-success}`Post-processing`
:::

:::{grid-item-card} 🔍 Sample Filtering
:link: sample_filtering
:link-type: doc

Filter samples based on metadata, outliers, or custom criteria.

+++
{bdg-info}`Preprocessing`
:::

::::

## Coming Soon

- **Loading Data** - Complete guide to DatasetConfigs and supported formats (CSV, Excel, MATLAB, NumPy, Parquet)

## See Also

- {doc}`/getting_started/index` - Quick start guide
- {doc}`/reference/pipeline_syntax` - Pipeline syntax reference
