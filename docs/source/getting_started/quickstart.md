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
print(f"Best RMSE: {result.best_rmse:.4f}")
print(f"Best R²: {result.best_r2:.4f}")
print(f"Number of predictions: {result.num_predictions}")

# Get top 3 models
for pred in result.top(n=3, display_metrics=['rmse', 'r2']):
    print(f"  {pred['model_name']}: RMSE={pred['rmse']:.4f}, R²={pred['r2']:.4f}")
```

### Step 4b: Understand Prediction Entries

Each prediction returned by `top()` is a dictionary with detailed information:

```python
# Get the best prediction
best = result.best

# Core identification
print(f"Model: {best['model_name']}")
print(f"Dataset: {best['dataset_name']}")
print(f"Fold: {best['fold_id']}")
print(f"Preprocessing: {best.get('preprocessings', 'none')}")

# Scores by partition (primary metric, always available)
print(f"Primary metric: {best['metric']}")
print(f"Train: {best['train_score']:.6f}")
print(f"Val: {best['val_score']:.6f}")
print(f"Test: {best['test_score']:.6f}")

# Additional metrics (when using display_metrics)
print(f"RMSE: {best.get('rmse', 0):.4f}")
print(f"R²: {best.get('r2', 0):.4f}")
```

**Key fields in each prediction entry:**

| Field | Description |
|-------|-------------|
| `model_name` | Name of the model (e.g., "PLSRegression") |
| `model_classname` | Class name of the model |
| `dataset_name` | Dataset name |
| `fold_id` | Cross-validation fold index |
| `preprocessings` | Preprocessing steps applied |
| `metric` | Primary metric name (e.g., 'mse') |
| `train_score`, `val_score`, `test_score` | Scores by partition (primary metric) |
| `rmse`, `r2`, `mse`, `mae` | Metrics (when using `display_metrics`) |
| `n_samples`, `n_features` | Data shape info |
| `task_type` | 'regression' or 'classification' |

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

# Generate synthetic NIRS data (or use your own dataset path)
dataset = nirs4all.generate.regression(
    n_samples=200,
    target_component=0,
    random_state=42
)

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
    dataset=dataset,
    name="MyFirstPipeline",
    verbose=1
)

# View results
print(f"\n📊 Results:")
print(f"   Best RMSE: {result.best_rmse:.4f}")
print(f"   Best R²: {result.best_r2:.4f}")
print(f"   Total predictions: {result.num_predictions}")

# Top models with detailed metrics
print("\n🏆 Top 3 Models:")
for i, pred in enumerate(result.top(n=3, display_metrics=['rmse', 'r2']), 1):
    print(f"   {i}. {pred['model_name']}: RMSE={pred['rmse']:.4f}, R²={pred['r2']:.4f}")

# Explore the best prediction entry
print("\n📦 Best prediction details:")
best = result.best
print(f"   Model: {best['model_name']}")
print(f"   Dataset: {best['dataset_name']}")
print(f"   Fold: {best['fold_id']}")
print(f"   Metric: {best['metric']}")

# Access partition-specific scores (primary metric)
print(f"   Train: {best['train_score']:.6f}")
print(f"   Val: {best['val_score']:.6f}")
print(f"   Test: {best['test_score']:.6f}")

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

## No Data? Generate Synthetic NIRS Spectra

Get started immediately with realistic synthetic data:

```python
import nirs4all

# Generate synthetic NIRS data with known ground truth
dataset = nirs4all.generate.regression(
    n_samples=500,
    components=["water", "protein", "lipid"],
    complexity="realistic",
    random_state=42
)

# Use directly in pipelines
result = nirs4all.run(
    pipeline=[
        MinMaxScaler(),
        ShuffleSplit(n_splits=3),
        {"model": PLSRegression(n_components=10)}
    ],
    dataset=dataset
)

print(f"RMSE: {result.best_rmse:.4f}")
```

Synthetic data is perfect for:
- Learning and experimentation
- Testing preprocessing pipelines
- Prototyping before real data arrives
- Reproducible unit tests

See {doc}`/user_guide/data/synthetic_data` for full documentation.

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

## Optimize Performance with Caching

When testing many pipeline variants (generators, hyperparameter sweeps), enable caching to reuse preprocessing:

```python
from nirs4all.config.cache_config import CacheConfig
from nirs4all.operators.transforms import StandardNormalVariate, SavitzkyGolay

# Pipeline with generators creates many variants
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3),
    StandardNormalVariate(),  # This step is cached
    {"_or_": [None, SavitzkyGolay(window_length=11), SavitzkyGolay(window_length=15)]},
    {"_range_": [5, 20, 5], "param": "n_components", "model": PLSRegression},
]
# Creates 3 × 4 = 12 variants

# Enable caching to reuse SNV across all variants
result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    cache=CacheConfig(
        step_cache_enabled=True,        # Reuse preprocessing
        use_cow_snapshots=True,         # Reduce memory
        step_cache_max_mb=2048,         # 2 GB cache budget
    ),
    verbose=1
)

print(f"Tested {result.num_predictions} configurations")
print(f"Best RMSE: {result.best_rmse:.4f}")
```

**Performance tip**: Caching can speed up generator-heavy pipelines by 2-5×. See {doc}`/user_guide/pipelines/cache_optimization` for details.

## Run Multiple Pipelines at Once

Pass a list of pipelines to execute them all independently:

```python
# Define different pipeline strategies
pipeline_pls = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3),
    {"model": PLSRegression(n_components=10)}
]

pipeline_rf = [
    StandardScaler(),
    ShuffleSplit(n_splits=3),
    {"model": RandomForestRegressor(n_estimators=100)}
]

pipeline_ridge = [
    MinMaxScaler(),
    FirstDerivative(),
    ShuffleSplit(n_splits=3),
    {"model": Ridge(alpha=1.0)}
]

# Run all three pipelines with one call
result = nirs4all.run(
    pipeline=[pipeline_pls, pipeline_rf, pipeline_ridge],  # List of pipelines
    dataset="sample_data/regression",
    verbose=1
)

print(f"Total configurations tested: {result.num_predictions}")
print(f"Best RMSE: {result.best_rmse:.4f}")
```

## Run on Multiple Datasets

Test the same pipeline(s) across different datasets:

```python
# Cartesian product: each pipeline × each dataset
result = nirs4all.run(
    pipeline=[pipeline_pls, pipeline_rf],   # 2 pipelines
    dataset=["data/wheat", "data/corn"],    # 2 datasets
    verbose=1
)
# Runs 4 combinations: PLS×wheat, PLS×corn, RF×wheat, RF×corn

print(f"Tested {result.num_predictions} configurations")
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
3. **Results are accessible** via `result.best_rmse`, `result.best_r2`, `result.top()`, etc.
4. **Prediction entries are dicts** with model_name, dataset_name, fold_id, scores, and more
5. **Detailed scores** are available via `pred['scores']['train'/'val'/'test']['rmse'/'r2'/...]`
6. **Export models** with `result.export()` for deployment
7. **NIRS preprocessing** (SNV, derivatives) improves spectral analysis

## See Also

- {doc}`concepts` - Understanding SpectroDataset and pipelines
- {doc}`/user_guide/preprocessing/overview` - NIRS preprocessing techniques
- {doc}`/reference/pipeline_syntax` - Complete syntax reference

```{seealso}
**Related Examples:**
- [U01: Hello World](../../../examples/user/01_getting_started/U01_hello_world.py) - Your first nirs4all pipeline
- [U02: Basic Regression](../../../examples/user/01_getting_started/U02_basic_regression.py) - Complete workflow with preprocessing and visualization
- [Interactive Example Browser](../../examples/index.md) - Browse all 67 examples by topic
```
