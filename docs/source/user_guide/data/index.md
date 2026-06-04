# Data Handling

This section covers loading and managing spectroscopic data in NIRS4ALL.

```{toctree}
:maxdepth: 2

loading_data
synthetic_data
signal_types
aggregation
sample_filtering
```

## Overview

NIRS4ALL provides flexible data handling capabilities for loading spectroscopic data from various formats, generating synthetic data, filtering samples, and aggregating predictions.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 📂 Loading Data
:link: loading_data
:link-type: doc

Load data from CSV, Excel, MATLAB, NumPy, and Parquet formats.

+++
{bdg-primary}`Essential`
:::

:::{grid-item-card} 🧪 Synthetic Data
:link: synthetic_data
:link-type: doc

Generate realistic synthetic NIRS spectra for testing and prototyping.

+++
{bdg-info}`New in 0.6`
:::

:::{grid-item-card} 🔬 Signal Types
:link: signal_types
:link-type: doc

Detect and convert between absorbance, reflectance, and other spectral representations.

+++
{bdg-success}`Essential`
:::

:::{grid-item-card} 🔁 Repetitions & Aggregation
:link: aggregation
:link-type: doc

Handle repeated scans of the same physical sample: group replicates with
`DatasetConfigs(repetition=...)` and aggregate them (`aggregate`,
`aggregate_method`, outlier exclusion). Core to NIRS.

+++
{bdg-success}`NIRS`
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
