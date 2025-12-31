# Getting Started Examples

This section introduces the fundamentals of NIRS4ALL through a series of progressive examples. Each example builds upon the previous one, guiding you from your first pipeline to advanced visualization techniques.

```{contents} On this page
:local:
:depth: 2
```

## Overview

| Example | Topic | Difficulty | Duration |
|---------|-------|------------|----------|
| [U01](#u01-hello-world) | Hello World | ★☆☆☆☆ | ~1 min |
| [U02](#u02-basic-regression) | Basic Regression | ★★☆☆☆ | ~3 min |
| [U03](#u03-basic-classification) | Basic Classification | ★★☆☆☆ | ~2 min |
| [U04](#u04-visualization) | Visualization | ★★☆☆☆ | ~3 min |

---

## U01: Hello World

**Your first NIRS4ALL pipeline in about 20 lines of code.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/01_getting_started/U01_hello_world.py)

### What You'll Learn

- Using `nirs4all.run()` to train a pipeline
- The structure of a minimal pipeline
- Reading results from the `RunResult` object

### Key Concepts

A pipeline in NIRS4ALL is simply a **list of processing steps**:

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

import nirs4all

# Define the pipeline as a list of steps
pipeline = [
    MinMaxScaler(),                              # Feature scaling
    {"y_processing": MinMaxScaler()},            # Target scaling
    ShuffleSplit(n_splits=3, test_size=0.25),    # Cross-validation
    {"model": PLSRegression(n_components=10)}    # Model
]

# Run with one simple call
result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="HelloWorld",
    verbose=1
)

# Access results
print(f"Best Score (MSE): {result.best_score:.4f}")
```

### The RunResult Object

The `result` object provides convenient accessors:

| Accessor | Description |
|----------|-------------|
| `result.best_score` | Best model's primary score (MSE by default) |
| `result.best` | Best prediction entry as a dictionary |
| `result.top(n)` | Top N predictions ranked by score |
| `result.predictions` | Full Predictions object for analysis |

### Tips for Beginners

1. **Start simple**: Begin with a basic pipeline and add complexity gradually
2. **Use verbose=1**: See what's happening during training
3. **Check top models**: Use `result.top(n=5)` to compare performance

---

## U02: Basic Regression

**A complete regression pipeline with NIRS-specific preprocessing and visualization.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/01_getting_started/U02_basic_regression.py)

### What You'll Learn

- NIRS-specific preprocessing (SNV, Detrend, Derivatives, Gaussian)
- Feature augmentation to explore preprocessing combinations
- Using `PredictionAnalyzer` for result visualization
- Comparing models with different `n_components`

### NIRS Preprocessing Options

NIRS4ALL provides specialized transforms for spectral data:

| Transform | Purpose | When to Use |
|-----------|---------|-------------|
| `StandardNormalVariate` (SNV) | Scatter correction | Path length variations |
| `MultiplicativeScatterCorrection` (MSC) | Scatter correction | Reference-based correction |
| `Detrend` | Baseline correction | Polynomial drift removal |
| `FirstDerivative` | Enhance peaks, remove baseline | Constant baseline issues |
| `SavitzkyGolay` | Smoothing + derivatives | Noisy data |
| `Gaussian` | Smoothing | Noise reduction |
| `Haar` | Wavelet transform | Multi-resolution analysis |

### Feature Augmentation

Instead of manually defining multiple pipelines, use **feature augmentation** to explore combinations:

```python
pipeline = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},

    # Generate 3 preprocessing combinations from 5 options
    {
        "feature_augmentation": {
            "_or_": [Detrend, FirstDerivative, Gaussian, SavitzkyGolay, Haar],
            "pick": 2,      # Pick 2 at a time
            "count": 3      # Generate 3 combinations
        }
    },

    ShuffleSplit(n_splits=3, test_size=0.25),
    {"model": PLSRegression(n_components=10)}
]
```

### Visualization with PredictionAnalyzer

```python
from nirs4all.visualization.predictions import PredictionAnalyzer

analyzer = PredictionAnalyzer(result.predictions)

# Compare top K models
analyzer.plot_top_k(k=3, rank_metric='rmse')

# Heatmap: models vs preprocessing
analyzer.plot_heatmap(x_var="model_name", y_var="preprocessings")

# Performance distribution
analyzer.plot_candlestick(variable="model_name")
```

---

## U03: Basic Classification

**Classification pipeline with Random Forest, XGBoost, and confusion matrix visualization.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/01_getting_started/U03_basic_classification.py)

### What You'll Learn

- Setting up a classification pipeline
- Using multiple classifiers (Random Forest, XGBoost)
- Confusion matrix visualization
- Classification metrics (accuracy, balanced recall)

### Classification Pipeline Structure

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import StandardScaler

pipeline = [
    # Feature augmentation with preprocessing options
    {"feature_augmentation": [
        FirstDerivative,
        StandardNormalVariate,
        Haar,
        MultiplicativeScatterCorrection
    ]},

    StandardScaler(),
    ShuffleSplit(n_splits=3, test_size=0.25),

    # Classifier
    {"model": RandomForestClassifier(n_estimators=50, max_depth=8)}
]
```

### Classification Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `accuracy` | Overall correct predictions | Balanced classes |
| `balanced_recall` | Average recall per class | Imbalanced classes |
| `balanced_accuracy` | Average accuracy per class | Class imbalance |

### Confusion Matrix Visualization

```python
# Plot confusion matrices for top 4 classifiers
analyzer.plot_confusion_matrix(
    k=4,
    rank_metric='accuracy',
    rank_partition='val',
    display_partition='test'
)
```

---

## U04: Visualization

**A comprehensive tour of all visualization options in NIRS4ALL.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/01_getting_started/U04_visualization.py)

### What You'll Learn

- All PredictionAnalyzer methods and options
- Heatmaps, candlestick charts, histograms
- Top-k comparison plots
- Ranking vs display partition configuration

### Available Visualizations

#### Top-K Comparison

```python
# Basic top-k plot
analyzer.plot_top_k(k=3, rank_metric='rmse')

# Rank by test partition, display R²
analyzer.plot_top_k(k=3, rank_metric='r2', rank_partition='test')
```

#### Heatmaps

Create 2D comparisons between any two variables:

```python
# Model vs preprocessing
analyzer.plot_heatmap(x_var="model_name", y_var="preprocessings")

# Model vs dataset
analyzer.plot_heatmap(x_var="model_name", y_var="dataset_name", display_metric="r2")

# Model vs fold with counts
analyzer.plot_heatmap(x_var="model_name", y_var="fold_id", show_counts=True)
```

#### Candlestick Charts

Show performance distribution per category:

```python
analyzer.plot_candlestick(variable="model_name", display_metric='rmse')
analyzer.plot_candlestick(variable="dataset_name", display_metric='r2')
```

#### Histograms

```python
analyzer.plot_histogram(display_metric='rmse')
analyzer.plot_histogram(display_metric='r2')
```

### Ranking vs Display: A Key Concept

You can **separate ranking from display**:

| Parameter | Purpose |
|-----------|---------|
| `rank_metric` + `rank_partition` | Determines which models are "best" |
| `display_metric` + `display_partition` | What values to show |

```python
# Rank by validation RMSE, but display test R²
analyzer.plot_heatmap(
    x_var="model_name",
    y_var="preprocessings",
    rank_metric='rmse',
    rank_partition='val',
    display_metric='r2',
    display_partition='test'
)
```

### Aggregation Options

| Option | Description |
|--------|-------------|
| `'best'` | Show best score for each cell |
| `'mean'` | Show mean score |
| `'median'` | Show median score |

---

## Running These Examples

```bash
cd examples

# Run all getting started examples
./run.sh -n "U0*.py" -c user

# Run a specific example
python user/01_getting_started/U01_hello_world.py

# Enable plots
python user/01_getting_started/U02_basic_regression.py --plots --show
```

## Next Steps

After completing these examples:

- **Data Handling**: Learn different input formats and multi-dataset analysis
- **Preprocessing**: Deep dive into NIRS-specific transformations
- **Models**: Compare multiple models and hyperparameter tuning
