# Quickstart

Get up and running with NIRS4ALL in 5 minutes. This guide walks you through your first complete pipeline.

## Prerequisites

- NIRS4ALL installed (see {doc}`installation`)
- Python 3.9+

## Your First Pipeline

### Step 1: Import Libraries

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
```

### Step 2: Define a Pipeline

A pipeline is a list of processing steps:

```python
pipeline = [
    MinMaxScaler(),                              # Scale features to [0, 1]
    {"y_processing": MinMaxScaler()},            # Scale targets
    ShuffleSplit(n_splits=3, test_size=0.25),    # 3-fold cross-validation
    {"model": PLSRegression(n_components=10)}    # PLS model
]
```

### Step 3: Run the Pipeline

Use `nirs4all.run()` to train with one function call:

```python
result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",   # Path to your data
    name="MyFirstPipeline",
    verbose=1
)
```

### Step 4: View Results

```python
# Check overall performance
print(f"Best RMSE: {result.best_score:.4f}")
print(f"Number of predictions: {result.num_predictions}")

# Get top 3 models
for pred in result.top(n=3, display_metrics=['rmse', 'r2']):
    print(f"  {pred['model_name']}: RMSE={pred['rmse']:.4f}, R²={pred['r2']:.4f}")
```

### Step 5: Export for Production

```python
# Export the best model for later use
result.export("exports/my_model.n4a")
```

## Complete Example

Here's the complete code you can copy and run:

```python
"""My first NIRS4ALL pipeline."""

import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

# Define pipeline
pipeline = [
    MinMaxScaler(),                              # Scale features
    {"y_processing": MinMaxScaler()},            # Scale targets
    ShuffleSplit(n_splits=3, test_size=0.25),    # Cross-validation
    {"model": PLSRegression(n_components=10)}    # Model
]

# Run pipeline
result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="MyFirstPipeline",
    verbose=1
)

# View results
print(f"\n📊 Results:")
print(f"   Best RMSE: {result.best_score:.4f}")
print(f"   Total predictions: {result.num_predictions}")

# Top models
print("\n🏆 Top 3 Models:")
for i, pred in enumerate(result.top(n=3, display_metrics=['rmse', 'r2']), 1):
    print(f"   {i}. {pred['model_name']}: RMSE={pred['rmse']:.4f}, R²={pred['r2']:.4f}")

# Export best model
result.export("exports/my_model.n4a")
print("\n✅ Model exported to exports/my_model.n4a")
```

## Add NIRS-Specific Preprocessing

NIRS data benefits from specialized preprocessing. Try this enhanced pipeline:

```python
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    FirstDerivative
)

pipeline = [
    MinMaxScaler(),                              # Feature scaling
    StandardNormalVariate(),                     # SNV: scatter correction
    FirstDerivative(),                           # Enhance spectral features
    {"y_processing": MinMaxScaler()},            # Target scaling
    ShuffleSplit(n_splits=3),                    # Cross-validation
    {"model": PLSRegression(n_components=10)}    # Model
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="NIRSPipeline",
    verbose=1
)
```

## Using Your Own Data

Replace the sample data with your own:

```python
# From a CSV file
result = nirs4all.run(pipeline, dataset="path/to/your/data.csv")

# From a folder
result = nirs4all.run(pipeline, dataset="path/to/data_folder/")

# With explicit configuration
from nirs4all.data import DatasetConfigs

dataset = DatasetConfigs({
    "train_x": "spectra.csv",
    "train_y": "targets.csv",
})
result = nirs4all.run(pipeline, dataset=dataset)
```

## Compare Multiple Models

Run and compare different models in one pipeline:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3),

    # Multiple models - each is evaluated
    {"model": PLSRegression(n_components=5)},
    {"model": PLSRegression(n_components=10)},
    {"model": PLSRegression(n_components=15)},
    {"model": Ridge(alpha=1.0)},
    {"model": RandomForestRegressor(n_estimators=100)},
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="MultiModel",
    verbose=1
)

# See which model performed best
for pred in result.top(n=5, display_metrics=['rmse', 'r2']):
    print(f"{pred['model_name']}: RMSE={pred['rmse']:.4f}")
```

## Visualize Results

Create publication-quality visualizations:

```python
from nirs4all.visualization.predictions import PredictionAnalyzer

analyzer = PredictionAnalyzer(result.predictions)

# Predicted vs actual plot for top models
fig1 = analyzer.plot_top_k(k=3, rank_metric='rmse')

# Compare models with candlestick chart
fig2 = analyzer.plot_candlestick(variable="model_name")

# Show all plots
import matplotlib.pyplot as plt
plt.show()
```

## What's Next?

::::{grid} 2
:gutter: 3

:::{grid-item-card} 📚 Core Concepts
:link: concepts
:link-type: doc

Understand pipelines, datasets, and execution flow.
:::

:::{grid-item-card} 📖 User Guide
:link: /user_guide/index
:link-type: doc

Learn preprocessing, stacking, and deployment.
:::

:::{grid-item-card} 📝 Examples
:link: /examples/index
:link-type: doc

50+ working examples organized by topic.
:::

:::{grid-item-card} 📋 Pipeline Syntax
:link: /reference/pipeline_syntax
:link-type: doc

Complete pipeline syntax reference.
:::

::::

## Key Takeaways

1. **Pipelines are lists** of processing steps
2. **One function** (`nirs4all.run()`) handles everything
3. **Results are accessible** via `result.best_score`, `result.top()`, etc.
4. **Export models** with `result.export()` for deployment
5. **NIRS preprocessing** (SNV, derivatives) improves spectral analysis

## See Also

- {doc}`concepts` - Understanding SpectroDataset and pipelines
- {doc}`/user_guide/preprocessing/overview` - NIRS preprocessing techniques
- {doc}`/reference/pipeline_syntax` - Complete syntax reference
