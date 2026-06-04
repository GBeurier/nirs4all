# Visualization

This section covers visualization tools for analysis and interpretation.

```{toctree}
:maxdepth: 2

in_pipeline_charts
prediction_charts
pipeline_diagram
shap
```

## Overview

NIRS4ALL provides comprehensive visualization tools for analyzing predictions, understanding model behavior, and interpreting results.

::::{grid} 2
:gutter: 3

:::{grid-item-card} � In-Pipeline Charts
:link: in_pipeline_charts
:link-type: doc

Visualize spectra, folds, targets, augmentation, and exclusions during pipeline execution.

+++
{bdg-info}`Pipeline`
:::

:::{grid-item-card} 📊 Prediction Charts
:link: prediction_charts
:link-type: doc

Visualize predictions, residuals, and model performance.

+++
{bdg-primary}`Analysis`
:::

:::{grid-item-card} 🔀 Pipeline Diagram
:link: pipeline_diagram
:link-type: doc

Visualize pipeline structure as interactive diagrams.

+++
{bdg-warning}`Structure`
:::

:::{grid-item-card} 🔍 SHAP Analysis
:link: shap
:link-type: doc

Explain model predictions with SHAP values.

+++
{bdg-success}`Explainability`
:::

::::

## Quick Example

```python
import nirs4all
from nirs4all.visualization import PredictionAnalyzer

# Run pipeline and get results
result = nirs4all.run(pipeline, dataset="data/")

# Create analyzer from the Predictions object
analyzer = PredictionAnalyzer(result.predictions)

# Generate charts
analyzer.plot_top_k(k=5)            # Top-K model comparison
analyzer.plot_histogram()           # Score distribution
analyzer.plot_candlestick(variable='model_name')  # Score ranges
```

## Available Charts

| Chart Type | Method | Use Case |
|------------|--------|----------|
| **Top-K Comparison** | `plot_top_k` | Compare best models |
| **Confusion Matrix** | `plot_confusion_matrix` | Classification accuracy |
| **Score Histogram** | `plot_histogram` | Score distribution |
| **Candlestick** | `plot_candlestick` | Score ranges per group |
| **Heatmap** | `plot_heatmap` | Performance across factors |

## See Also

- {doc}`/examples/index` - Example visualizations
- {doc}`/reference/operator_catalog` - Chart operators in pipelines
