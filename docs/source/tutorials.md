# Tutorials

This section provides a progressive learning path through nirs4all examples, from your first pipeline to advanced features.

## Getting Started

Start here if you're new to nirs4all. These tutorials introduce the core concepts.

### U01 - Hello World

**Your first pipeline in 20 lines of code.**

[U01_hello_world.py](https://github.com/GBeurier/nirs4all/blob/main/examples/user/01_getting_started/U01_hello_world.py)

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

result = nirs4all.run(
    pipeline=[
        MinMaxScaler(),
        {"y_processing": MinMaxScaler()},
        ShuffleSplit(n_splits=3, test_size=0.25),
        {"model": PLSRegression(n_components=10)}
    ],
    dataset="sample_data/regression",
    name="HelloWorld",
    verbose=1
)

print(f"Best RMSE: {result.best_rmse:.4f}")
print(f"Best R²: {result.best_r2:.4f}")
```

Key concepts:
- `nirs4all.run()` is the main entry point
- Pipeline is a list of processing steps
- Results accessed via `result.best_rmse`, `result.top(n)`, etc.

### U02 - Basic Regression

**Regression with NIRS preprocessing and visualization.**

[U02_basic_regression.py](https://github.com/GBeurier/nirs4all/blob/main/examples/user/01_getting_started/U02_basic_regression.py)

Key concepts:
- NIRS-specific preprocessing (SNV, MSC, derivatives)
- PredictionAnalyzer for visualizations
- Comparing different preprocessing approaches

### U03 - Basic Classification

**Classification with Random Forest and XGBoost.**

[U03_basic_classification.py](https://github.com/GBeurier/nirs4all/blob/main/examples/user/01_getting_started/U03_basic_classification.py)

Key concepts:
- Classification vs regression
- Confusion matrix visualization
- Multi-class handling

### U04 - Visualization

**Tour of available visualization tools.**

[U04_visualization.py](https://github.com/GBeurier/nirs4all/blob/main/examples/user/01_getting_started/U04_visualization.py)

Key concepts:
- PredictionAnalyzer class
- Candlestick charts, top-K, heatmaps
- Customizing visualizations

---

## User Path Overview

The complete User Path covers these topics:

| Section | Examples | Description |
|---------|----------|-------------|
| **01_getting_started** | U01-U04 | First pipelines, basic concepts |
| **02_data_handling** | U05-U08 | Data loading, multi-source |
| **03_preprocessing** | U09-U12 | NIRS preprocessing techniques |
| **04_models** | U13-U16 | Model comparison, tuning, ensembles |
| **05_cross_validation** | U17-U20 | CV strategies, filtering |
| **06_deployment** | U21-U24 | Save, load, export, sklearn integration |
| **07_explainability** | U25-U27 | SHAP, feature selection |

---

## Developer Path Overview

For advanced users and contributors:

| Section | Examples | Description |
|---------|----------|-------------|
| **01_advanced_pipelines** | D01-D05 | Branching, merging, stacking |
| **02_generators** | D06-D09 | Generator syntax, constraints |
| **03_deep_learning** | D10-D13 | TensorFlow, PyTorch, JAX |
| **04_transfer_learning** | D14-D16 | Retrain modes, domain adaptation |
| **05_advanced_features** | D17-D20 | Outliers, metadata, repetitions |
| **06_internals** | D21-D22 | Sessions, custom controllers |

---

## Running Examples

```bash
cd examples

# Run all examples
./run.sh

# Run specific category
./run.sh -c user          # User path only
./run.sh -c developer     # Developer path only

# Run specific example
./run.sh -n "U01*.py"     # By name pattern
./run.sh -i 1             # By index

# With visualization
./run.sh -p -s            # Generate and show plots

# Quick mode (skip deep learning)
./run.sh -q
```

## Next Steps

After completing the tutorials:

1. **Reference Documentation**: See [Reference](reference.md) for complete syntax documentation
2. **API Documentation**: See [Module API](api/module_api.md) for function reference
3. **Examples README**: See [examples/README.md](https://github.com/GBeurier/nirs4all/blob/main/examples/README.md) for the complete index
