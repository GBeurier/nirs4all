# Data Handling Examples

This section covers all the ways to load, configure, and work with data in NIRS4ALL. From simple numpy arrays to complex multi-source datasets, you'll learn the flexible input options available.

```{contents} On this page
:local:
:depth: 2
```

## Overview

| Example | Topic | Difficulty | Duration |
|---------|-------|------------|----------|
| [U01](#u01-flexible-inputs) | Flexible Inputs | ★☆☆☆☆ | ~2 min |
| [U02](#u02-multi-datasets) | Multi-Datasets | ★★☆☆☆ | ~3 min |
| [U03](#u03-multi-source) | Multi-Source Data | ★★★☆☆ | ~3 min |
| [U04](#u04-wavelength-handling) | Wavelength Handling | ★★☆☆☆ | ~3 min |
| [U05](#u05-synthetic-data) | Synthetic Data | ★★☆☆☆ | ~2 min |
| [U06](#u06-synthetic-advanced) | Advanced Synthetic Data | ★★★☆☆ | ~5 min |

---

## U01: Flexible Inputs

**Demonstrates all possible input formats for datasets and pipelines.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/02_data_handling/U01_flexible_inputs.py)

### What You'll Learn

- Direct numpy array input with `(X, y)` tuples
- Dictionary-based dataset configuration
- Partition info specification
- SpectroDataset object usage

### Input Format Overview

NIRS4ALL accepts datasets in multiple formats:

| Format | Example | Best For |
|--------|---------|----------|
| Folder path | `"sample_data/regression"` | File-based datasets |
| Tuple | `(X, y)` or `(X, y, partition_info)` | Quick experiments |
| Dictionary | `{"train_x": X_train, "test_x": X_test, ...}` | Explicit splits |
| DatasetConfigs | `DatasetConfigs("path")` | Full control |
| SpectroDataset | `SpectroDataset(name="my_data")` | Programmatic access |

### Simplest Approach: Direct Arrays

```python
import numpy as np
from sklearn.linear_model import Ridge

# Generate or load your data
X = np.random.randn(200, 100)
y = np.random.randn(200)

# Partition info: first 160 samples for training
partition_info = {"train": 160}

# Run directly with tuple
result = nirs4all.run(
    pipeline=[Ridge(alpha=1.0)],
    dataset=(X, y, partition_info),
    name="DirectArrays"
)
```

### Partition Info Options

```python
# Integer: first N samples = train
{"train": 160}

# Slice objects
{"train": slice(0, 150), "test": slice(150, 200)}

# Explicit indices
{"train": list(range(150)), "test": list(range(150, 200))}
```

### Dictionary Configuration

```python
dataset_dict = {
    "name": "my_dataset",
    "train_x": X_train,
    "train_y": y_train,
    "test_x": X_test,
    "test_y": y_test
}

result = nirs4all.run(pipeline=pipeline, dataset=dataset_dict)
```

### Using SpectroDataset Directly

For maximum control, create a SpectroDataset object:

```python
from nirs4all.data import SpectroDataset

dataset = SpectroDataset(name="custom")
dataset.add_samples(X_train, indexes={"partition": "train"})
dataset.add_targets(y_train)
dataset.add_samples(X_test, indexes={"partition": "test"})
dataset.add_targets(y_test)

result = nirs4all.run(pipeline=pipeline, dataset=dataset)
```

---

## U02: Multi-Datasets

**Run the same pipeline on multiple datasets and compare results.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/02_data_handling/U02_multi_datasets.py)

### What You'll Learn

- Specifying multiple datasets as a list
- Per-dataset result access
- Cross-dataset comparison visualizations

### Specifying Multiple Datasets

Simply pass a list of dataset paths:

```python
data_paths = [
    'sample_data/regression',
    'sample_data/regression_2',
    'sample_data/regression_3'
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset=data_paths,
    name="MultiDataset"
)
```

### Accessing Per-Dataset Results

```python
for dataset_name, dataset_info in result.per_dataset.items():
    dataset_predictions = dataset_info.get('run_predictions')

    if dataset_predictions:
        top_models = dataset_predictions.top(n=3, rank_metric='rmse')
        print(f"Dataset: {dataset_name}")
        for model in top_models:
            print(f"  {model['model_name']}: RMSE={model.get('rmse', 0):.4f}")
```

### Cross-Dataset Visualization

```python
analyzer = PredictionAnalyzer(result.predictions)

# Compare models across datasets
analyzer.plot_heatmap(
    x_var="model_name",
    y_var="dataset_name",
    display_metric='rmse'
)

# Dataset difficulty comparison
analyzer.plot_candlestick(
    variable="dataset_name",
    display_metric='rmse'
)
```

### Use Cases

- **Generalization testing**: Does a model work well on different samples?
- **Dataset comparison**: Which datasets are most challenging?
- **Robust model selection**: Find models that work everywhere

---

## U03: Multi-Source Data

**Work with datasets that have multiple feature sources (e.g., NIR + markers).**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/02_data_handling/U03_multi_source.py)

### What You'll Learn

- Loading multi-source datasets
- Feature augmentation with generator syntax
- Basic multi-source handling

### Understanding Multi-Source Data

Multi-source datasets contain features from different instruments or measurement types:

```
Example: NIR spectrometer + wet chemistry markers
├── Source 1: NIR spectra (1000 wavelengths)
└── Source 2: Lab markers (10 chemical values)
```

### Data Structure

Multi-source data has multiple X files per partition:

```
sample_data/multi/
├── Xtrain_1.csv  # Source 1 training features
├── Xtrain_2.csv  # Source 2 training features
├── Xval_1.csv    # Source 1 validation features
├── Xval_2.csv    # Source 2 validation features
└── Ytrain.csv    # Targets
```

### Loading Multi-Source Data

```python
from nirs4all.data import DatasetConfigs

# Automatic loading
dataset_config = DatasetConfigs('sample_data/multi')

# NIRS4ALL automatically detects and loads all sources
result = nirs4all.run(
    pipeline=pipeline,
    dataset='sample_data/multi',
    name="MultiSource"
)
```

### Advanced: Source-Specific Processing

For per-source preprocessing (covered in Developer examples):

```python
# Source branching: different preprocessing per source
{"source_branch": {
    "NIR": [SNV(), FirstDerivative()],
    "markers": [VarianceThreshold()],
}}

# Merge sources
{"merge_sources": "concat"}  # Horizontal concatenation
```

---

## U04: Wavelength Handling

**Handle wavelength grids: interpolation, downsampling, and unit conversion.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/02_data_handling/U04_wavelength_handling.py)

### What You'll Learn

- `Resampler` operator for wavelength interpolation
- Downsampling to fewer wavelengths
- Focusing on specific spectral regions

### Why Resample Wavelengths?

- **Instrument standardization**: Different spectrometers have different wavelength grids
- **Transfer learning**: Match wavelengths between training and inference instruments
- **Dimensionality reduction**: Reduce features while preserving spectral shape
- **Region focus**: Analyze specific spectral regions

### The Resampler Operator

```python
from nirs4all.operators.transforms import Resampler
import numpy as np

# Target wavelengths (e.g., from a reference instrument)
target_wavelengths = np.linspace(1000, 2500, 100)

pipeline = [
    Resampler(target_wavelengths=target_wavelengths, method='linear'),
    # ... rest of pipeline
]
```

### Common Resampling Scenarios

#### Match Another Dataset

```python
# Get wavelengths from reference dataset
ref_config = DatasetConfigs("reference_data")
ref_dataset = list(ref_config.iter_datasets())[0]
target_wl = ref_dataset.float_headers(0)

# Resample to match
Resampler(target_wavelengths=target_wl)
```

#### Downsample

```python
# Reduce to 50 evenly-spaced points
target_wl = np.linspace(start_wl, end_wl, 50)
Resampler(target_wavelengths=target_wl)
```

#### Focus on Region

```python
# Focus on fingerprint region (e.g., 1400-1800 nm)
region_wl = np.linspace(1400, 1800, 100)
Resampler(target_wavelengths=region_wl)
```

### Interpolation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `'linear'` | Linear interpolation | Default, fast |
| `'cubic'` | Cubic spline | Smooth spectra |
| `'quadratic'` | Quadratic interpolation | Balance speed/smoothness |
| `'nearest'` | Nearest neighbor | Discrete features |

---

## U05: Synthetic Data

**Generate synthetic NIRS spectra for testing and prototyping.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/02_data_handling/U05_synthetic_data.py)

### What You'll Learn

- Using `nirs4all.generate()` for quick dataset creation
- Convenience functions for regression and classification
- Configuring spectral complexity and components

### Basic Generation

```python
import nirs4all

# Generate a SpectroDataset
dataset = nirs4all.generate(n_samples=500, random_state=42)

# Or get raw numpy arrays
X, y = nirs4all.generate(n_samples=300, as_dataset=False)
```

### Regression Datasets

```python
dataset = nirs4all.generate.regression(
    n_samples=500,
    target_range=(0, 100),      # Scale targets
    target_component=0,          # Which component as target
    complexity="realistic",      # Noise level
    random_state=42
)
```

### Classification Datasets

```python
# Binary classification
dataset = nirs4all.generate.classification(
    n_samples=400,
    n_classes=2,
    class_separation=2.0,  # Well-separated classes
    random_state=42
)

# Imbalanced multiclass
dataset = nirs4all.generate.classification(
    n_samples=600,
    n_classes=3,
    class_weights=[0.5, 0.3, 0.2],
    random_state=42
)
```

### Complexity Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `"simple"` | Minimal noise | Unit tests, fast prototyping |
| `"realistic"` | Typical NIR noise/scatter | Development, validation |
| `"complex"` | High noise, artifacts | Robustness testing |

### Specifying Chemical Components

```python
dataset = nirs4all.generate(
    n_samples=400,
    components=["water", "protein", "lipid", "starch"],
    complexity="realistic"
)
```

Available components: `water`, `protein`, `lipid`, `starch`, `cellulose`, `chlorophyll`, `oil`, `nitrogen_compound`

### Direct Pipeline Integration

```python
# Generate and train in one call
result = nirs4all.run(
    pipeline=[StandardScaler(), PLSRegression(n_components=10)],
    dataset=nirs4all.generate.regression(n_samples=600, complexity="realistic"),
    name="SyntheticTest"
)
```

---

## U06: Synthetic Advanced

**Master the full synthetic data generation API for complex scenarios.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/02_data_handling/U06_synthetic_advanced.py)

### What You'll Learn

- Using `SyntheticDatasetBuilder` for full control
- Metadata generation (groups, repetitions)
- Multi-source datasets
- Batch effects simulation
- Non-linear target complexity for realistic benchmarks
- Exporting to files
- Matching real data characteristics

### SyntheticDatasetBuilder

For maximum control over synthetic data generation:

```python
from nirs4all.data.synthetic import SyntheticDatasetBuilder

# Full control over generation
builder = SyntheticDatasetBuilder(
    n_samples=500,
    wavelength_range=(1000, 2500),
    n_wavelengths=256,
    components=["water", "protein", "lipid"],
)

# Add batch effects
builder.add_batch_effect(n_batches=3, intensity=0.1)

# Add metadata
builder.add_group_metadata(n_groups=5)

# Generate dataset
dataset = builder.build()
```

### Multi-Source Synthetic Data

```python
# Create multi-source datasets
builder = SyntheticDatasetBuilder(n_samples=300)
builder.add_source("NIR", wavelength_range=(1000, 2500), n_wavelengths=256)
builder.add_source("markers", n_features=10, feature_type="numerical")

dataset = builder.build()
```

### Exporting to Files

```python
# Export for loader testing
dataset.to_csv("synthetic_data/")
# Creates: Xtrain.csv, Ytrain.csv, Xtest.csv, Ytest.csv
```

---

## Running These Examples

```bash
cd examples

# Run all data handling examples
./run.sh -n "U0*.py" -c user

# Run with plots
python user/02_data_handling/U05_synthetic_data.py --plots --show
```

## Next Steps

After mastering data handling:

- **Preprocessing**: Apply NIRS-specific transformations
- **Models**: Compare different model architectures
- **Cross-Validation**: Choose the right validation strategy
