# Loading Data

This guide covers how to load spectral data into NIRS4ALL using `DatasetConfigs`.

## Overview

NIRS4ALL loads data through `DatasetConfigs`, which handles:
- Multiple file formats (CSV, Excel, MATLAB, NumPy, Parquet)
- Automatic file detection in folders
- Multi-source datasets (e.g., NIR + markers)
- Train/test splits
- Metadata handling

## Quick Start

### From a Folder

The simplest approach - NIRS4ALL auto-detects your data files:

```python
from nirs4all.data import DatasetConfigs

# Auto-detect files in folder
dataset = DatasetConfigs("path/to/data/")
```

Expected folder structure:
```
data/
├── train_x.csv      # Training features (spectra)
├── train_y.csv      # Training targets
├── train_m.csv      # Training metadata (optional)
├── test_x.csv       # Test features (optional)
├── test_y.csv       # Test targets (optional)
└── test_m.csv       # Test metadata (optional)
```

### From a Single File

For a single file with features and targets combined:

```python
# CSV with last column as target
dataset = DatasetConfigs("data.csv")

# Explicit target column
dataset = DatasetConfigs({
    "train_x": "data.csv",
    "global_params": {"target_column": "protein"}
})
```

### From Explicit Files

Full control over file paths:

```python
dataset = DatasetConfigs({
    "train_x": "spectra_train.csv",
    "train_y": "targets_train.csv",
    "test_x": "spectra_test.csv",
    "test_y": "targets_test.csv"
})
```

## Supported Formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| CSV | `.csv` | Most common; configurable delimiter |
| Excel | `.xlsx`, `.xls` | Single sheet or specify sheet name |
| MATLAB | `.mat` | Reads first array variable |
| NumPy | `.npy`, `.npz` | Binary format, fast loading |
| Parquet | `.parquet` | Columnar format, efficient for large data |

### CSV Configuration

```python
dataset = DatasetConfigs({
    "train_x": "spectra.csv",
    "train_x_params": {
        "delimiter": ";",           # Column separator
        "decimal_separator": ",",   # Decimal point
        "has_header": True,         # First row is header
        "na_policy": "drop"         # Handle missing values
    }
})
```

### Excel Configuration

```python
dataset = DatasetConfigs({
    "train_x": "data.xlsx",
    "train_x_params": {
        "sheet_name": "Spectra",    # Specific sheet
        "header_row": 0             # Header row index
    }
})
```

## File Keys Reference

| Key | Description |
|-----|-------------|
| `train_x` | Training features (spectra) |
| `train_y` | Training targets |
| `train_m` | Training metadata |
| `test_x` | Test features |
| `test_y` | Test targets |
| `test_m` | Test metadata |
| `*_params` | Parameters for corresponding file (e.g., `train_x_params`) |
| `global_params` | Parameters applied to all files |

## Wavelength Headers

NIRS4ALL understands wavelength information from column headers:

```python
dataset = DatasetConfigs({
    "train_x": "spectra.csv",
    "train_x_params": {
        "header_unit": "nm"        # Headers are wavelengths in nm
        # Options: "nm", "cm-1", "none", "text", "index"
    }
})
```

| `header_unit` | Description | Example Headers |
|---------------|-------------|-----------------|
| `"nm"` | Wavelengths in nanometers | `900, 902, 904, ...` |
| `"cm-1"` | Wavenumbers in cm⁻¹ | `4000, 3998, 3996, ...` |
| `"none"` | No header row | - |
| `"text"` | Text labels (ignored) | `V1, V2, V3, ...` |
| `"index"` | Numeric indices | `1, 2, 3, ...` |

## Signal Type

Specify the type of spectral signal for proper handling:

```python
dataset = DatasetConfigs({
    "train_x": "spectra.csv",
    "train_x_params": {
        "signal_type": "reflectance"
    }
})

# Or as constructor parameter
dataset = DatasetConfigs("spectra.csv", signal_type="reflectance")
```

| Signal Type | Description |
|-------------|-------------|
| `"absorbance"` | -log₁₀(R) |
| `"reflectance"` | Raw reflectance (0-1) |
| `"reflectance%"` | Reflectance as percentage (0-100) |
| `"transmittance"` | Raw transmittance (0-1) |
| `"transmittance%"` | Transmittance as percentage |
| `"auto"` | Automatic detection (default) |

## Task Type

Force regression or classification mode:

```python
# Force regression
dataset = DatasetConfigs("data/", task_type="regression")

# Force classification
dataset = DatasetConfigs("data/", task_type="binary_classification")

# Valid options:
# - "auto" (default)
# - "regression"
# - "binary_classification"
# - "multiclass_classification"
```

## Multi-Source Datasets

Combine multiple data sources (e.g., NIR spectra + chemical markers):

```python
dataset = DatasetConfigs({
    "train_x": ["nir_spectra.csv", "markers.csv"],
    "train_y": "targets.csv",
    "train_x_params": [
        {"header_unit": "nm", "signal_type": "reflectance"},
        {"header_unit": "text"}  # Markers have text headers
    ]
})
```

### Processing Multi-Source Data

Use `source_branch` to apply different preprocessing to each source:

```python
pipeline = [
    {"source_branch": {
        0: [StandardNormalVariate(), FirstDerivative()],  # NIR source
        1: [StandardScaler()]                             # Markers source
    }},
    {"merge_sources": "concat"},  # Combine sources
    {"model": PLSRegression(n_components=10)}
]
```

## Sample Aggregation

Aggregate predictions from multiple measurements per sample:

```python
# Aggregate by sample ID column
dataset = DatasetConfigs(
    "data/",
    aggregate="sample_id"      # Metadata column name
)

# Aggregate by target values
dataset = DatasetConfigs(
    "data/",
    aggregate=True             # Group by y values
)

# With custom method
dataset = DatasetConfigs(
    "data/",
    aggregate="sample_id",
    aggregate_method="median",  # "mean", "median", or "vote"
    aggregate_exclude_outliers=True  # Remove outliers before aggregating
)
```

## Multiple Datasets

Run the same pipeline on multiple datasets:

```python
dataset = DatasetConfigs([
    "dataset1/",
    "dataset2/",
    {"train_x": "custom/spectra.csv", "train_y": "custom/targets.csv"}
])

# Results will include predictions for all datasets
result = nirs4all.run(pipeline, dataset)
```

## Using SpectroDataset Directly

For advanced use cases, you can pass `SpectroDataset` instances directly to `nirs4all.run()`:

```python
from nirs4all.data import SpectroDataset
import nirs4all

# Create a SpectroDataset manually
dataset = SpectroDataset(name="my_dataset")
dataset.add_samples(X_train, indexes={"partition": "train"})
dataset.add_targets(y_train)

# Use directly in run()
result = nirs4all.run(pipeline, dataset)
```

### Multiple SpectroDataset Instances

You can also pass a list of `SpectroDataset` instances:

```python
# Multiple SpectroDataset instances
datasets = [dataset1, dataset2, dataset3]
result = nirs4all.run(pipeline, datasets)
```

This is particularly useful when:
- Working with synthetic data generators that return `SpectroDataset`
- Programmatically creating datasets from different sources
- Chaining multiple pipeline runs with transformed data

## Complete Example

```python
from nirs4all.data import DatasetConfigs
import nirs4all

# Comprehensive configuration
dataset = DatasetConfigs({
    # Training data
    "train_x": "data/train_spectra.csv",
    "train_y": "data/train_targets.csv",
    "train_m": "data/train_metadata.csv",

    # Test data
    "test_x": "data/test_spectra.csv",
    "test_y": "data/test_targets.csv",

    # Training file parameters
    "train_x_params": {
        "header_unit": "nm",
        "signal_type": "reflectance",
        "delimiter": ","
    },

    # Force regression task
    "task_type": "regression"
})

# Run pipeline
result = nirs4all.run(
    pipeline=[
        MinMaxScaler(),
        ShuffleSplit(n_splits=3),
        {"model": PLSRegression(n_components=10)}
    ],
    dataset=dataset,
    verbose=1
)
```

## Common Patterns

### Load with Metadata

```python
dataset = DatasetConfigs({
    "train_x": "spectra.csv",
    "train_y": "targets.csv",
    "train_m": "metadata.csv"  # Sample IDs, dates, groups, etc.
})
```

### Specify Target Column

```python
# When features and target are in the same file
dataset = DatasetConfigs({
    "train_x": "combined_data.csv",
    "global_params": {
        "target_column": "protein"  # Column name for target
    }
})
```

### Handle Missing Values

```python
dataset = DatasetConfigs({
    "train_x": "spectra.csv",
    "global_params": {
        "na_policy": "drop"     # Drop rows with NaN
        # Options: "drop", "fill_mean", "fill_median", "fill_zero"
    }
})
```

## Troubleshooting

### File Not Found

```python
# Use absolute paths if relative paths fail
import os
path = os.path.abspath("data/spectra.csv")
dataset = DatasetConfigs(path)
```

### Wrong Delimiter

```python
# Check file manually, then specify
dataset = DatasetConfigs({
    "train_x": "spectra.csv",
    "train_x_params": {"delimiter": "\t"}  # Tab-separated
})
```

### Header Issues

```python
# No header row
dataset = DatasetConfigs({
    "train_x": "spectra.csv",
    "train_x_params": {"has_header": False}
})

# Skip header row
dataset = DatasetConfigs({
    "train_x": "spectra.csv",
    "train_x_params": {"header_unit": "none", "has_header": True}
})
```

## See Also

- {doc}`/getting_started/concepts` - Understanding SpectroDataset
- {doc}`/reference/configuration` - Full DatasetConfigs specification
- {doc}`sample_filtering` - Filter samples during loading
- {doc}`aggregation` - Aggregate multiple measurements
