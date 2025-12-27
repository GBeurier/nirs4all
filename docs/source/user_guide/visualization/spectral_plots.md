# Spectral Visualization

Visualize spectra and preprocessing effects within your pipeline.

## Overview

nirs4all provides built-in chart controllers for visualizing spectral data at any point in your pipeline. These visualizations help you:

- Inspect raw spectra before preprocessing
- Verify preprocessing effects
- Identify outliers and anomalies
- Compare train/test data distributions
- Understand model inputs

## Chart Keywords

Add visualization steps to your pipeline using these keywords:

| Keyword | Description |
|---------|-------------|
| `"chart_2d"` | 2D spectra plot (wavelength vs. intensity) |
| `"chart_3d"` | 3D spectra plot with target gradient |
| `"spectral_distribution"` | Envelope plot (min/max/mean/IQR) |

## Basic Usage

### 2D Spectra Chart

```python
pipeline = [
    "chart_2d",           # Visualize raw spectra
    SNV(),
    "chart_2d",           # Visualize after SNV
    ShuffleSplit(n_splits=3, random_state=42),
    PLSRegression(n_components=10),
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    plots_visible=True  # Display plots interactively
)
```

### 3D Spectra Chart

```python
pipeline = [
    "chart_3d",  # 3D view with target color gradient
    SNV(),
    FirstDerivative(),
    "chart_3d",  # 3D view after preprocessing
    ShuffleSplit(n_splits=3, random_state=42),
    PLSRegression(n_components=10),
]
```

### Spectral Distribution

```python
pipeline = [
    "spectral_distribution",  # Show envelope: min/max/mean
    SNV(),
    "spectral_distribution",
    ShuffleSplit(n_splits=3, random_state=42),
    PLSRegression(n_components=10),
]
```

## Controlling Plot Display

### Interactive Display

Show plots during execution:

```python
result = nirs4all.run(
    pipeline=pipeline,
    dataset="data/",
    plots_visible=True  # Display plots interactively
)
```

### Saving Charts as Artifacts

Charts are automatically saved as artifacts:

```python
runner = PipelineRunner(
    save_artifacts=True,
    workspace_path="workspace/"
)
predictions, _ = runner.run(pipeline, dataset)

# Charts saved in: workspace/runs/<run_id>/artifacts/
```

### Batch Mode (No Display)

For headless environments:

```python
result = nirs4all.run(
    pipeline=pipeline,
    dataset="data/",
    plots_visible=False  # Don't display, just save
)
```

## Chart Options

### Dictionary Syntax for Options

```python
pipeline = [
    {"chart_2d": {
        "include_excluded": True,    # Show excluded samples
        "highlight_excluded": True,  # Highlight with different style
    }},
    SNV(),
    ShuffleSplit(n_splits=3, random_state=42),
    PLSRegression(n_components=10),
]
```

### Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `include_excluded` | bool | False | Include excluded samples |
| `highlight_excluded` | bool | True | Highlight excluded with red dashed lines |

## Chart Types Explained

### 2D Spectra Plot

Displays all spectra as overlaid lines:
- **X-axis**: Wavelength or feature index
- **Y-axis**: Intensity/absorbance
- **Color**: Gradient based on target value

```
      ┌────────────────────────────────────┐
      │  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   │
      │   ~~~~~~~~~~~~~~~~~~~~~~~~~~~      │ ← Spectra colored
      │    ~~~~~~~~~~~~~~~~~~~~~~~~~       │   by target value
      │     ~~~~~~~~~~~~~~~~~~~~~~~        │
      │      ~~~~~~~~~~~~~~~~~~~~~         │
      │       ~~~~~~~~~~~~~~~~~~~          │
      └────────────────────────────────────┘
         Wavelength (nm or cm⁻¹)
```

**Use for**: Quick overview, outlier detection, preprocessing verification

### 3D Spectra Plot

Adds target value as Z-axis:
- **X-axis**: Wavelength
- **Y-axis**: Target value (sorted)
- **Z-axis**: Intensity

```
            Intensity
               │
               │    ╱──────╲
               │   ╱        ╲
               │  ╱          ╲──────
               │ ╱                  ╲
               │╱                    ╲
               └─────────────────────────── Wavelength
              ╱
             ╱ Target
            ╱
```

**Use for**: Understanding spectra-target relationship, identifying patterns

### Spectral Distribution

Shows statistical envelope:
- **Mean line**: Average spectrum
- **Shaded region**: Min-max or IQR range
- **Separate for train/test**: Compare distributions

```
      ┌────────────────────────────────────┐
      │  ════════════════════════════════  │ ← Max
      │  ┌───────────────────────────────┐ │
      │  │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│ │ ← Mean
      │  └───────────────────────────────┘ │
      │  ════════════════════════════════  │ ← Min
      └────────────────────────────────────┘
         Wavelength
```

**Use for**: Data quality assessment, train/test distribution comparison

## Multi-Processing Visualization

When using `feature_augmentation`, charts show each preprocessing:

```python
pipeline = [
    {"feature_augmentation": [SNV, Detrend, FirstDerivative], "action": "extend"},
    "chart_2d",  # Shows subplot for each preprocessing
    ShuffleSplit(n_splits=3, random_state=42),
    PLSRegression(n_components=10),
]
```

Output creates a multi-panel figure:

```
┌──────────────┬──────────────┬──────────────┐
│     Raw      │     SNV      │   Detrend    │
│   ~~~~~~~~   │   ~~~~~~~~   │   ~~~~~~~~   │
│  ~~~~~~~~~~  │  ~~~~~~~~~~  │  ~~~~~~~~~~  │
└──────────────┴──────────────┴──────────────┘
```

## Multi-Source Visualization

For multi-source datasets, charts are generated per source:

```python
dataset = DatasetConfigs([
    {"path": "nir.csv", "source_name": "NIR"},
    {"path": "raman.csv", "source_name": "Raman"},
])

pipeline = [
    "chart_2d",  # Generates separate chart per source
    {"source_branch": {
        "NIR": [SNV()],
        "Raman": [MSC()],
    }},
    "chart_2d",  # Charts after source-specific preprocessing
    PLSRegression(n_components=10),
]
```

## Programmatic Plotting

For custom visualizations outside the pipeline:

### Using SpectroDataset

```python
import matplotlib.pyplot as plt
import numpy as np
from nirs4all.data import DatasetConfigs

# Load data
config = DatasetConfigs("sample_data/regression")
dataset = config.load()

# Get data
X = dataset.x(layout="2d")
y = dataset.y()
wavelengths = dataset.headers(0)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
for i in range(min(50, len(X))):
    color = plt.cm.viridis(y[i] / y.max())
    ax.plot(wavelengths, X[i], alpha=0.5, color=color)

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Absorbance')
ax.set_title('Raw Spectra')
plt.show()
```

### Before/After Preprocessing

```python
from nirs4all.operators.transforms import SNV

# Original
X_raw = dataset.x(layout="2d")

# Apply preprocessing
snv = SNV()
X_snv = snv.fit_transform(X_raw)

# Compare
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw
axes[0].plot(X_raw.T, alpha=0.3)
axes[0].set_title('Raw Spectra')

# SNV
axes[1].plot(X_snv.T, alpha=0.3)
axes[1].set_title('After SNV')

plt.tight_layout()
plt.show()
```

## Using PredictionAnalyzer

For post-pipeline visualization of results:

```python
from nirs4all.visualization.predictions import PredictionAnalyzer

result = nirs4all.run(pipeline, dataset)
analyzer = PredictionAnalyzer(result.predictions)

# Prediction scatter plot
analyzer.plot_scatter(partition="test")

# Residuals
analyzer.plot_residuals()

# Top-k comparison
analyzer.plot_top_k(k=10, rank_metric='rmse')
```

## Best Practices

1. **Place strategically**: Add charts before and after key preprocessing steps
2. **Use sparingly in production**: Charts add execution time
3. **Check distributions**: Use `spectral_distribution` to verify train/test similarity
4. **Save artifacts**: Enable `save_artifacts=True` for reproducibility
5. **Headless mode**: Use `plots_visible=False` in automated pipelines
6. **Multi-panel for comparison**: Use `feature_augmentation` + chart for side-by-side comparison

## Troubleshooting

### "No module named 'matplotlib'"
```bash
pip install matplotlib
```

### Plots not displaying
- Check `plots_visible=True` is set
- In Jupyter: ensure `%matplotlib inline` is set
- In scripts: add `plt.show()` at the end

### Charts slow with large datasets
- Charts render all samples—reduce dataset size for quick checks
- Use `spectral_distribution` for summary statistics instead

## See Also

- {doc}`/user_guide/visualization/prediction_charts` - Post-prediction visualization
- {doc}`/user_guide/preprocessing/overview` - Preprocessing techniques
- {doc}`/user_guide/visualization/shap` - SHAP visualization
