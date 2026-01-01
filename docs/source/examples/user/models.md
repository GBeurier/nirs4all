# Model Examples

This section covers model training, comparison, hyperparameter tuning, and ensemble methods in NIRS4ALL.

```{contents} On this page
:local:
:depth: 2
```

## Overview

| Example | Topic | Difficulty | Duration |
|---------|-------|------------|----------|
| [U01](#u01-multi-model) | Multi-Model Comparison | ★★☆☆☆ | ~4 min |
| [U02](#u02-hyperparameter-tuning) | Hyperparameter Tuning | ★★★☆☆ | ~5 min |
| [U03](#u03-stacking-ensembles) | Stacking Ensembles | ★★★☆☆ | ~4 min |
| [U04](#u04-pls-variants) | PLS Variants | ★★☆☆☆ | ~3 min |

---

## U01: Multi-Model Comparison

**Run and compare multiple models in a single pipeline.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/04_models/U01_multi_model.py)

### What You'll Learn

- Defining multiple models in one pipeline
- Using the `_or_` generator syntax
- Comparing model performance
- Model selection strategies

### Model Families for NIRS

Different models have different strengths:

| Family | Models | Strengths |
|--------|--------|-----------|
| **Linear** | PLS, Ridge, Lasso, ElasticNet | Handles collinearity, interpretable |
| **Tree-based** | RandomForest, GradientBoosting, ExtraTrees | Handles non-linearity, feature importance |
| **Other** | SVR, KNeighbors | Local patterns, non-parametric |

### Basic Multi-Model Pipeline

List multiple models in the pipeline—each is trained and evaluated:

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

pipeline = [
    StandardNormalVariate(),
    FirstDerivative(),
    StandardScaler(),

    ShuffleSplit(n_splits=3, random_state=42),

    # Multiple models - each is evaluated
    {"model": PLSRegression(n_components=10)},
    {"model": Ridge(alpha=1.0)},
    {"model": RandomForestRegressor(n_estimators=100)},
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="MultiModel"
)

print(f"Models tested: {result.num_predictions}")
print(f"Best RMSE: {result.best_score:.4f}")
```

### Using _or_ Generator Syntax

More compact syntax for model variants:

```python
pipeline = [
    SNV(),
    ShuffleSplit(n_splits=3),

    # Generate variants for each model
    {"_or_": [
        {"model": PLSRegression(n_components=10)},
        {"model": Ridge(alpha=1.0)},
        {"model": Lasso(alpha=0.1)},
        {"model": ElasticNet(alpha=0.1)},
    ]},
]
```

### Combined Preprocessing + Model Search

Find the best preprocessing + model combination:

```python
pipeline = [
    # Explore preprocessing options
    {"feature_augmentation": [SNV, MSC, Detrend], "action": "extend"},

    # Add derivative
    {"feature_augmentation": [FirstDerivative], "action": "add"},

    ShuffleSplit(n_splits=3),

    # Multiple models
    {"model": PLSRegression(n_components=10)},
    {"model": Ridge(alpha=1.0)},
    {"model": RandomForestRegressor(n_estimators=50)},
]
```

### Viewing Results

```python
# Top 10 configurations
for i, pred in enumerate(result.top(10, display_metrics=['rmse', 'r2']), 1):
    preproc = pred.get('preprocessings', 'N/A')
    model = pred.get('model_name', 'Unknown')
    print(f"{i}. {preproc} + {model}: RMSE={pred.get('rmse', 0):.4f}")
```

---

## U02: Hyperparameter Tuning

**Automated hyperparameter search with Optuna integration.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/04_models/U02_hyperparameter_tuning.py)

### What You'll Learn

- Grid search with `_range_` syntax
- Logarithmic ranges for regularization
- Optuna integration for smart search
- Early stopping and pruning

### Parameter Type Reference

NIRS4ALL supports all Optuna parameter types via flexible syntax:

**Tuple Format (most common):**
```python
"model_params": {
    "n_components": ('int', 1, 20),        # Integer uniform [1, 20]
    "n_layers": ('int_log', 1, 100),       # Integer log-uniform [1, 100]
    "tol": ('float', 1e-6, 1e-4),          # Float uniform [1e-6, 1e-4]
    "lr": ('float_log', 1e-5, 1e-1),       # Float log-uniform [1e-5, 1e-1]
    "kernel": ['rbf', 'linear', 'poly'],   # Categorical
}
```

**Dict Format (most flexible):**
```python
"model_params": {
    # Integer with step
    "n_estimators": {'type': 'int', 'min': 10, 'max': 100, 'step': 10},
    # Integer log-scale
    "max_iter": {'type': 'int', 'min': 100, 'max': 10000, 'log': True},
    # Float with step (discrete)
    "learning_rate": {'type': 'float', 'min': 0.1, 'max': 1.0, 'step': 0.1},
    # Float log-scale (recommended for regularization, learning rates)
    "alpha": {'type': 'float', 'min': 1e-5, 'max': 1e-1, 'log': True},
    # Categorical
    "solver": {'type': 'categorical', 'choices': ['lbfgs', 'sgd', 'adam']},
}
```

### When to Use Log-Scale

Use `float_log` or `int_log` for parameters spanning multiple orders of magnitude:
- **Learning rates**: `('float_log', 1e-5, 1e-1)`
- **Regularization (alpha, lambda)**: `('float_log', 1e-6, 1.0)`
- **Number of iterations**: `('int_log', 100, 10000)`

Log-uniform sampling ensures each order of magnitude gets equal exploration probability.

### Grid Search with _range_

```python
pipeline = [
    SNV(),
    ShuffleSplit(n_splits=3),

    # Sweep n_components from 5 to 25, step 5
    {"model": PLSRegression(),
     "n_components": {"_range_": [5, 25, 5]}}
]
# Generates: PLS(5), PLS(10), PLS(15), PLS(20), PLS(25)
```

### Logarithmic Ranges

For parameters like regularization strength:

```python
pipeline = [
    SNV(),
    ShuffleSplit(n_splits=3),

    # Log-spaced alpha values: 0.001, 0.01, 0.1, 1.0
    {"model": Ridge(),
     "alpha": {"_log_range_": [0.001, 1.0, 4]}}
]
```

### Combined Grid Search

```python
pipeline = [
    SNV(),
    ShuffleSplit(n_splits=3),

    # Grid over model type AND hyperparameters
    {"_grid_": {
        "model": [
            PLSRegression(n_components=5),
            PLSRegression(n_components=10),
            Ridge(alpha=0.1),
            Ridge(alpha=1.0),
        ]
    }}
]
```

### Optuna Integration

For smarter search (Bayesian optimization):

```python
pipeline = [
    SNV(),
    ShuffleSplit(n_splits=3),
    {
        "model": Ridge(),
        "finetune_params": {
            "n_trials": 50,
            "sample": "tpe",          # Bayesian optimization
            "verbose": 1,
            "approach": "single",
            "model_params": {
                "alpha": ('float_log', 1e-4, 1e2),  # Log-uniform sampling
            }
        }
    }
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression"
)
```

---

## U03: Stacking Ensembles

**Combine multiple models for improved predictions.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/04_models/U03_stacking_ensembles.py)

### What You'll Learn

- Prediction merging for stacking
- Out-of-fold prediction collection
- Meta-learner training
- Two-level stacking

### Stacking Concept

Stacking combines predictions from multiple base models:

```
Level 0 (Base Models):
  PLS  →  predictions_pls
  RF   →  predictions_rf
  Ridge → predictions_ridge

Level 1 (Meta-Learner):
  [predictions_pls, predictions_rf, predictions_ridge] → final_prediction
```

### Basic Stacking Pipeline

```python
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5),

    # Create branches for base models
    {"branch": {
        "pls": [PLSRegression(n_components=10)],
        "rf": [RandomForestRegressor(n_estimators=50)],
        "ridge": [Ridge(alpha=1.0)],
    }},

    # Merge predictions (OOF reconstruction)
    {"merge": "predictions"},

    # Meta-learner
    {"model": Ridge(alpha=0.1), "name": "MetaLearner"}
]
```

### Understanding Prediction Merging

When using `{"merge": "predictions"}`:

1. Each branch produces out-of-fold (OOF) predictions
2. OOF predictions are reconstructed to form features for the meta-learner
3. No data leakage: each sample's meta-features come from models that didn't see it

### Two-Level Stacking

```python
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5),

    # Level 0: Different preprocessing + model combinations
    {"branch": {
        "snv_pls": [SNV(), PLSRegression(n_components=10)],
        "msc_pls": [MSC(), PLSRegression(n_components=10)],
        "snv_rf": [SNV(), RandomForestRegressor(n_estimators=50)],
    }},

    {"merge": "predictions"},

    # Level 1: Meta-learner
    {"model": Ridge(alpha=0.1)}
]
```

---

## U04: PLS Variants

**Explore different Partial Least Squares implementations.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/04_models/U04_pls_variants.py)

### What You'll Learn

- Standard PLSRegression
- PLSCanonical
- Kernel PLS
- Selecting the right variant

### PLS Implementations

```python
from sklearn.cross_decomposition import (
    PLSRegression,
    PLSCanonical,
    CCA
)

# Standard PLS - most common for NIRS
PLSRegression(n_components=10)

# PLSCanonical - symmetric, multivariate Y
PLSCanonical(n_components=10)

# CCA - Canonical Correlation Analysis
CCA(n_components=10)
```

### When to Use Each

| Variant | Use Case |
|---------|----------|
| **PLSRegression** | Standard NIRS calibration, single target |
| **PLSCanonical** | Multi-output regression, balanced X/Y |
| **CCA** | Finding correlations, exploratory analysis |

### Comparing PLS Components

```python
pipeline = [
    SNV(),
    ShuffleSplit(n_splits=5),

    # Compare different n_components
    {"model": PLSRegression(n_components=5), "name": "PLS-5"},
    {"model": PLSRegression(n_components=10), "name": "PLS-10"},
    {"model": PLSRegression(n_components=15), "name": "PLS-15"},
    {"model": PLSRegression(n_components=20), "name": "PLS-20"},
]

result = nirs4all.run(pipeline=pipeline, dataset="sample_data/regression")

# Analyze component selection
analyzer = PredictionAnalyzer(result.predictions)
analyzer.plot_candlestick(variable="model_name", display_metric='rmse')
```

### Optimal Component Selection

Typically, optimal n_components:

- Too few: Underfitting, high bias
- Too many: Overfitting, high variance

Use cross-validation to find the sweet spot:

```python
pipeline = [
    SNV(),
    RepeatedKFold(n_splits=5, n_repeats=3),

    # Sweep components
    {"model": PLSRegression(),
     "n_components": {"_range_": [2, 30, 2]}}
]
```

---

## Model Selection Guidelines

### For NIRS Data

| Data Characteristic | Recommended Models |
|---------------------|-------------------|
| Linear relationships | PLS, Ridge |
| Non-linear relationships | Random Forest, Gradient Boosting |
| High collinearity | PLS (designed for this) |
| Small sample size | PLS, Ridge (regularized) |
| Large sample size | Random Forest, neural networks |
| Need interpretability | PLS, Ridge |

### Quick Comparison Strategy

```python
# Quick multi-model comparison
pipeline = [
    SNV(),
    ShuffleSplit(n_splits=3),

    {"model": PLSRegression(n_components=10), "name": "PLS"},
    {"model": Ridge(alpha=1.0), "name": "Ridge"},
    {"model": RandomForestRegressor(n_estimators=100), "name": "RF"},
    {"model": GradientBoostingRegressor(n_estimators=50), "name": "GBR"},
]

result = nirs4all.run(pipeline=pipeline, dataset="sample_data/regression")

# Visualize comparison
analyzer = PredictionAnalyzer(result.predictions)
analyzer.plot_candlestick(variable="model_name", display_metric='rmse')
```

---

## Running These Examples

```bash
cd examples

# Run all model examples
./run.sh -n "U0*.py" -c user

# Run hyperparameter tuning
python user/04_models/U02_hyperparameter_tuning.py --plots
```

## Next Steps

After mastering model training:

- **Cross-Validation**: Proper evaluation strategies
- **Deployment**: Save and deploy trained models
- **Explainability**: Understand model decisions with SHAP
