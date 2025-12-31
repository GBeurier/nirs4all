# Explainability Examples

This section covers model interpretation and explainability using SHAP (SHapley Additive exPlanations) for NIRS data.

```{contents} On this page
:local:
:depth: 2
```

## Overview

| Example | Topic | Difficulty | Duration |
|---------|-------|------------|----------|
| [U01](#u01-shap-basics) | SHAP Basics | ★★☆☆☆ | ~5 min |
| [U02](#u02-shap-sklearn) | SHAP with sklearn | ★★☆☆☆ | ~4 min |
| [U03](#u03-feature-selection) | Feature Selection | ★★★☆☆ | ~4 min |

---

## U01: SHAP Basics

**Understand model predictions using SHAP analysis.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/07_explainability/U01_shap_basics.py)

### What You'll Learn

- Why SHAP for model interpretation
- Running SHAP analysis with `runner.explain()`
- Spectral, waterfall, and beeswarm visualizations
- Binning and aggregation for spectral data

### Why SHAP?

SHAP provides model-agnostic explanations:

| What SHAP Tells You | Description |
|---------------------|-------------|
| **Which wavelengths matter** | Identify important spectral regions |
| **Direction of influence** | Positive or negative contribution |
| **Magnitude of effect** | Quantify importance |
| **Per-sample explanations** | Understand individual predictions |

### SHAP Key Concepts

| Concept | Meaning |
|---------|---------|
| **SHAP value** | Contribution of each feature to the prediction |
| **Positive value** | Increases the prediction |
| **Negative value** | Decreases the prediction |
| **Base value** | Average model output |

### Running SHAP Analysis

```python
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

# Train a model
runner = PipelineRunner(save_artifacts=True, verbose=0)
predictions, _ = runner.run(pipeline_config, dataset_config)

# Get best model
best = predictions.top(n=1, rank_metric='rmse')[0]

# Run SHAP analysis
shap_params = {
    'n_samples': 100,
    'explainer_type': 'auto',
    'visualizations': ['spectral', 'waterfall']
}

shap_results, output_dir = runner.explain(
    best,
    dataset_config,
    shap_params=shap_params,
    plots_visible=True
)
```

### SHAP Visualizations

#### Spectral Plot

Shows SHAP values across the spectrum:

```python
shap_params = {
    'visualizations': ['spectral'],
    'bin_size': {'spectral': 10},  # Group 10 wavelengths per bin
}
```

- **X-axis**: Wavelength/feature index
- **Y-axis**: SHAP value (importance)
- **Identifies**: Key spectral regions for prediction

#### Waterfall Plot

Explains a single prediction step-by-step:

```python
shap_params = {
    'visualizations': ['waterfall'],
    'bin_size': {'waterfall': 20},  # Coarser for readability
}
```

- Shows cumulative contribution from base value to final prediction
- Best for understanding individual samples

#### Beeswarm Plot

Shows SHAP distribution for all samples:

```python
shap_params = {
    'visualizations': ['beeswarm'],
}
```

- Each dot represents a sample
- Color indicates feature value (low/high)
- Reveals patterns in feature importance

### Binning for Spectral Data

For high-dimensional spectral data, binning groups features:

```python
shap_params = {
    'n_samples': 200,
    'visualizations': ['spectral', 'waterfall', 'beeswarm'],

    # Different bin sizes per visualization
    'bin_size': {
        'spectral': 10,      # Fine-grained overview
        'waterfall': 20,     # Fewer bars for readability
        'beeswarm': 20       # Medium granularity
    },

    # Stride (overlap) control
    'bin_stride': {
        'spectral': 5,       # 50% overlap
        'waterfall': 10,     # 50% overlap
        'beeswarm': 20       # No overlap
    },

    # How to aggregate SHAP values in bins
    'bin_aggregation': {
        'spectral': 'mean',   # Average importance
        'waterfall': 'mean',
        'beeswarm': 'mean'
    }
}
```

### Explainer Types

| Type | Models | Speed |
|------|--------|-------|
| `'auto'` | Auto-detect best explainer | Varies |
| `'tree'` | Tree-based (RF, GBR) | Fast |
| `'linear'` | Linear models (PLS, Ridge) | Fast |
| `'kernel'` | Universal (any model) | Slow |
| `'deep'` | Neural networks | Medium |

---

## U02: SHAP with sklearn Wrapper

**Use SHAP with sklearn-wrapped NIRS4ALL models.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/07_explainability/U02_shap_sklearn.py)

### What You'll Learn

- Direct SHAP with sklearn wrapper
- Custom explainer configuration
- Integration with SHAP library

### Direct SHAP Usage

```python
import shap
from nirs4all.sklearn import SklearnWrapper

# Wrap trained model
wrapper = SklearnWrapper(prediction_entry=best)

# Create SHAP explainer
explainer = shap.Explainer(wrapper.predict, X_train)

# Calculate SHAP values
shap_values = explainer(X_test[:100])

# Visualizations
shap.plots.beeswarm(shap_values)
shap.plots.waterfall(shap_values[0])
```

### Feature Names for Spectral Data

```python
# Create meaningful feature names
wavelengths = np.linspace(1000, 2500, X.shape[1])
feature_names = [f"{w:.0f}nm" for w in wavelengths]

# Use with SHAP
shap.plots.beeswarm(shap_values, feature_names=feature_names)
```

---

## U03: Feature Selection

**Use SHAP importance for feature selection.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/07_explainability/U03_feature_selection.py)

### What You'll Learn

- SHAP-based feature ranking
- Selecting top wavelengths
- Validation with reduced features

### SHAP Feature Importance

```python
import numpy as np

# Get absolute SHAP values
shap_importance = np.abs(shap_values.values).mean(axis=0)

# Rank features
ranking = np.argsort(shap_importance)[::-1]
top_features = ranking[:50]  # Top 50 wavelengths

print(f"Top wavelengths: {wavelengths[top_features]}")
```

### Feature Selection Pipeline

```python
# Select top features based on SHAP
def select_top_features(X, shap_values, n_features=50):
    importance = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(importance)[::-1][:n_features]
    return X[:, top_idx], top_idx

# Apply to data
X_selected, selected_idx = select_top_features(X, shap_values, n_features=50)

# Train on selected features
result_selected = nirs4all.run(
    pipeline=[PLSRegression(n_components=10)],
    dataset=(X_selected, y, {"train": 160}),
    name="SelectedFeatures"
)
```

### Wavelength Region Analysis

Identify important spectral regions:

```python
# Group SHAP values by wavelength regions
regions = {
    "1000-1300nm": (0, 100),
    "1300-1600nm": (100, 200),
    "1600-1900nm": (200, 300),
    # ...
}

region_importance = {}
for name, (start, end) in regions.items():
    region_importance[name] = np.abs(shap_values.values[:, start:end]).mean()

# Sort by importance
sorted_regions = sorted(region_importance.items(), key=lambda x: x[1], reverse=True)
for region, importance in sorted_regions:
    print(f"{region}: {importance:.4f}")
```

---

## Explainability Best Practices

### 1. Use Sufficient Samples

```python
# More samples = more reliable SHAP values
shap_params = {
    'n_samples': 200,  # At least 100-200 samples
}
```

### 2. Validate Explanations

Check if SHAP values make chemical sense:

```python
# If predicting sugar content, expect importance at:
# - ~1400-1500 nm (O-H bonds in sugars)
# - ~2100-2300 nm (C-H bonds)
```

### 3. Compare Models

```python
# Compare SHAP patterns across different models
for model_name in ['PLS', 'RF', 'Ridge']:
    model_pred = predictions.filter(model_name=model_name).top(1)[0]
    shap_results, _ = runner.explain(model_pred, dataset_config, shap_params)
    # Compare spectral patterns
```

### 4. Document Findings

```python
# Include SHAP insights in model documentation
metadata = {
    "important_regions": {
        "1400-1500nm": "O-H overtones",
        "2100-2300nm": "C-H combinations"
    },
    "shap_validation": "Patterns consistent with known sugar absorption"
}
```

---

## SHAP Parameter Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_samples` | Samples to explain | 100 |
| `explainer_type` | `'auto'`, `'tree'`, `'kernel'`, `'linear'`, `'deep'` | `'auto'` |
| `visualizations` | List of plots: `['spectral', 'waterfall', 'beeswarm']` | All |
| `bin_size` | Features per bin (int or dict per viz) | 10 |
| `bin_stride` | Step between bins | Same as bin_size |
| `bin_aggregation` | `'mean'`, `'sum'`, `'mean_abs'`, `'sum_abs'` | `'mean'` |

---

## Running These Examples

```bash
cd examples

# Run all explainability examples
./run.sh -n "U0*.py" -c user

# Run with plots
python user/07_explainability/U01_shap_basics.py --plots --show
```

## Requirements

SHAP analysis requires the `shap` library:

```bash
pip install shap
```

## Next Steps

After mastering explainability:

- **Developer Examples**: Advanced pipelines, deep learning
- **Transfer Learning**: Adapt models to new instruments
- **Custom Controllers**: Extend NIRS4ALL functionality
