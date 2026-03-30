# Quickstart

Get up and running with NIRS4ALL in 5 minutes. This guide walks you through your first complete pipeline.

## Prerequisites

- NIRS4ALL installed (see {doc}`installation`)
- Python 3.11+

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

## What's Next?

Now that you've run your first pipeline, continue with:

- {doc}`tutorial` — Progressive tutorial: add preprocessing, generators, branching, stacking, and export (20 minutes)
- {doc}`/concepts/index` — Understand the key concepts behind nirs4all
- {doc}`/reference/pipeline_keywords` — All pipeline keywords at a glance
- {doc}`/examples/index` — 60+ working examples organized by topic
