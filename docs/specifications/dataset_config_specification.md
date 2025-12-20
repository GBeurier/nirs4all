# Dataset Configuration Specification

**Version**: 1.0.0-draft
**Status**: Draft Specification
**Date**: December 2025

---

> ⚠️ **DRAFT – NOT YET IMPLEMENTED**
>
> This specification describes the target configuration system for nirs4all dataset loading.
> **Most features documented here are aspirational and not yet supported by the current codebase.**
>
> ### Currently Supported Configuration
>
> The current implementation supports only the **legacy key format**:
>
> ```yaml
> # Supported keys (current implementation)
> train_x: path/to/Xcal.csv          # Training features (required for training)
> train_y: path/to/Ycal.csv          # Training targets
> train_group: path/to/Mcal.csv      # Training metadata
> test_x: path/to/Xval.csv           # Test features
> test_y: path/to/Yval.csv           # Test targets
> test_group: path/to/Mval.csv       # Test metadata
>
> # Supported global parameters
> global_params:
>   delimiter: ","
>   decimal_separator: "."
>   has_header: true
>   header_unit: nm | cm-1 | none | text | index
>   signal_type: absorbance | reflectance | ...
>
> # Per-partition parameters (e.g., train_x_params, test_y_params)
> train_x_params:
>   header_unit: nm
>   signal_type: reflectance
>
> # Task and aggregation settings
> task_type: regression | binary_classification | multiclass_classification | auto
> aggregate: sample_id | true | null
> aggregate_method: mean | median | vote
> ```
>
> **Supported file formats**: CSV (`.csv`), compressed CSV (`.csv.gz`, `.csv.zip`), NumPy arrays (in-memory only)
>
> **Not yet implemented**: `files` syntax, `sources` syntax, `variations` syntax, column/row selection, partition column, fold definitions, Parquet, Excel, MATLAB, archive member selection, key-based linking.
>
> See [Migration from Simple Format](#migration-from-simple-format) for the relationship between legacy and new syntax.

---

## Table of Contents

1. [Overview](#overview)
2. [Supported File Formats](#supported-file-formats)
3. [Core Concepts](#core-concepts)
4. [Currently Implemented: Simple Configuration](#currently-implemented-simple-configuration)
5. [Configuration Schema](#configuration-schema)
6. [Use Cases and Examples](#use-cases-and-examples)
7. [Column and Row Selection](#column-and-row-selection)
8. [Partition Assignment](#partition-assignment)
9. [Fold Definition](#fold-definition)
10. [Multi-Source Datasets](#multi-source-datasets)
11. [Feature Variations and Preprocessed Data](#feature-variations-and-preprocessed-data)
12. [Validation Rules](#validation-rules)

---

## Overview

The nirs4all dataset configuration system provides a flexible, declarative way to load spectral and tabular data from various file formats and structures. The configuration can be specified in:

- **Folder path** (string) - auto-scan for standard file naming conventions ✅ *implemented*
- **YAML files** (`.yaml`, `.yml`) - recommended for readability ✅ *implemented*
- **JSON files** (`.json`) - for programmatic generation ✅ *implemented*
- **Python dictionaries** - for inline API usage ✅ *implemented*
- **In-memory NumPy arrays** - for programmatic data ✅ *implemented*
- **Python dataclasses** (future) - for type-safe configuration

### Design Principles

1. **Flexibility**: Handle any combination of files, columns, and partitions
2. **Explicitness**: Clear declaration of data roles (features, targets, metadata)
3. **Composability**: Combine multiple files into unified datasets
4. **Backward Compatibility**: Existing simple configs continue to work
5. **Validation**: Early detection of configuration errors

---

## Supported File Formats

### Primary Formats (Current Implementation)

| Format | Extensions | Status | Description |
|--------|------------|--------|-------------|
| CSV | `.csv` | ✅ Implemented | Comma/delimiter-separated values |
| Compressed CSV | `.csv.gz`, `.csv.zip` | ✅ Implemented | Compressed CSV files |
| NumPy (in-memory) | - | ✅ Implemented | NumPy arrays passed directly |

### Primary Formats (Planned)

| Format | Extensions | Status | Description |
|--------|------------|--------|-------------|
| NumPy (files) | `.npy`, `.npz` | 🔜 Planned | NumPy array files |
| Parquet | `.parquet` | 🔜 Planned | Apache Parquet columnar format |
| Excel | `.xlsx`, `.xls` | 🔜 Planned | Excel spreadsheets |
| MATLAB | `.mat` | 🔜 Planned | MATLAB data files |

### Archive Formats (Planned)

| Format | Extensions | Status | Description |
|--------|------------|--------|-------------|
| Gzip | `.gz` | ✅ Implemented | Gzip-compressed single file |
| Zip | `.zip` | ✅ Implemented | Zip archive (single file extraction) |
| Tar | `.tar`, `.tar.gz`, `.tgz` | 🔜 Planned | Tar archive (optionally compressed) |

### Format-Specific Parameters

```yaml
# CSV parameters
csv_params:
  delimiter: ","              # Field delimiter (default: ",")
  decimal_separator: "."      # Decimal separator (default: ".")
  thousands_separator: null   # Thousands separator (default: null)
  encoding: "utf-8"           # File encoding (default: "utf-8")
  skip_rows: 0                # Rows to skip at start (default: 0)
  skip_footer: 0              # Rows to skip at end (default: 0)
  na_values: ["", "NA", "NaN", "null"]  # Values to treat as NA
  comment: null               # Comment character (lines starting with this are skipped)

# NumPy parameters
numpy_params:
  allow_pickle: false         # Allow loading pickled objects (default: false)
  key: null                   # Key for .npz files (default: first array)

# Excel parameters
excel_params:
  sheet_name: 0               # Sheet name or index (default: 0)
  skip_rows: 0                # Rows to skip
  skip_footer: 0              # Rows to skip at end

# Archive parameters
archive_params:
  member: null                # Specific file within archive (default: auto-detect)
  password: null              # Archive password if encrypted
```

---

## Core Concepts

### Data Roles

Every piece of data in a dataset has one of three roles:

| Role | Description | Required |
|------|-------------|----------|
| **features** | Input variables (X) - spectral data, measurements | Yes (train or test) |
| **targets** | Output variables (y) - values to predict | No (optional for prediction) |
| **metadata** | Auxiliary information - sample IDs, groups, dates | No |

### Partitions

Data is organized into partitions for training and evaluation:

| Partition | Description | Purpose |
|-----------|-------------|---------|
| `train` | Training data | Model fitting |
| `test` | Test/validation data | Model evaluation |
| `predict` | Prediction-only data | No targets, just features |

### Sources (Multi-Source)

For sensor fusion or multi-instrument data, multiple feature sources can be defined:

```yaml
sources:
  - name: "NIR"           # First spectrometer
    files: [...]
  - name: "MIR"           # Second spectrometer
    files: [...]
```

### Variations (Preprocessed Data / Feature Variants)

For datasets with pre-computed preprocessing or multiple feature representations of the same samples (e.g., time series variables, offline-computed derivatives):

```yaml
variations:
  - name: "raw"           # Original signal
    files: [...]
  - name: "snv"           # Pre-computed SNV normalization
    files: [...]
  - name: "derivative"    # Pre-computed derivative
    files: [...]

variation_mode: separate | concat | select | compare
```

See [Feature Variations and Preprocessed Data](#feature-variations-and-preprocessed-data) for details.

---

## Currently Implemented: Simple Configuration

This section documents the **currently working** configuration methods. All features described here are fully implemented and tested.

### Method 1: Folder Path with Auto-Scanning

The simplest way to load a dataset is to provide a folder path. nirs4all will automatically scan for files matching standard naming conventions.

```python
from nirs4all.data import DatasetConfigs

# Just pass a folder path - files are auto-detected
configs = DatasetConfigs("/path/to/my_dataset/")
```

#### Auto-Detected File Naming Conventions

The folder scanner looks for files matching these patterns (case-insensitive):

| Role | Recognized Patterns | Examples |
|------|---------------------|----------|
| **Training Features (X)** | `Xcal`, `X_cal`, `Cal_X`, `calX`, `train_X`, `trainX`, `X_train`, `Xtrain` | `Xcal.csv`, `X_train.csv.gz` |
| **Training Targets (Y)** | `Ycal`, `Y_cal`, `Cal_Y`, `calY`, `train_Y`, `trainY`, `Y_train`, `Ytrain` | `Ycal.csv`, `trainY.csv` |
| **Training Metadata (M)** | `Mcal`, `M_cal`, `Cal_M`, `calM`, `train_M`, `trainM`, `M_train`, `Mtrain`, `Metacal`, `Meta_cal`, `train_Meta`, `metadata_train`, etc. | `Mcal.csv`, `Meta_train.csv` |
| **Test Features (X)** | `Xval`, `X_val`, `val_X`, `valX`, `Xtest`, `X_test`, `test_X`, `testX` | `Xval.csv`, `X_test.csv` |
| **Test Targets (Y)** | `Ytest`, `Y_test`, `test_Y`, `testY`, `Yval`, `Y_val`, `val_Y`, `valY` | `Yval.csv`, `testY.csv` |
| **Test Metadata (M)** | `Mtest`, `M_test`, `test_M`, `testM`, `Mval`, `M_val`, `val_M`, `valM`, `Metaval`, `metadata_test`, etc. | `Mval.csv`, `metadata_val.csv` |

#### Typical Folder Structures

**6-file structure** (train + test with metadata):
```
my_dataset/
├── Xcal.csv          # Training features
├── Ycal.csv          # Training targets
├── Mcal.csv          # Training metadata
├── Xval.csv          # Test features
├── Yval.csv          # Test targets
└── Mval.csv          # Test metadata
```

**4-file structure** (train + test without metadata):
```
my_dataset/
├── X_train.csv
├── Y_train.csv
├── X_test.csv
└── Y_test.csv
```

**3-file structure** (train only):
```
my_dataset/
├── Xcal.csv
├── Ycal.csv
└── Mcal.csv
```

**2-file structure** (train only, no metadata):
```
my_dataset/
├── trainX.csv
└── trainY.csv
```

#### Multi-Source Auto-Detection

If multiple files match the same pattern (e.g., multiple `Xcal*.csv` files), they are automatically treated as **multi-source** data:

```
my_dataset/
├── Xcal_NIR.csv      # Source 1
├── Xcal_MIR.csv      # Source 2
├── Ycal.csv
└── ...
```

### Method 2: Folder Path with Parameters

Pass a dictionary with `folder` key to add loading parameters:

```python
configs = DatasetConfigs({
    "folder": "/path/to/my_dataset/",
    "params": {
        "delimiter": ";",
        "header_unit": "nm",
        "signal_type": "reflectance"
    }
})
```

### Method 3: JSON/YAML Configuration File

Load configuration from a file (detected by `.json`, `.yaml`, or `.yml` extension):

```python
# From YAML file
configs = DatasetConfigs("path/to/config.yaml")

# From JSON file
configs = DatasetConfigs("path/to/config.json")
```

**Example YAML configuration** (`config.yaml`):

```yaml
name: wheat_protein

train_x: data/Xcal.csv
train_y: data/Ycal.csv
train_group: data/Mcal.csv

test_x: data/Xval.csv
test_y: data/Yval.csv
test_group: data/Mval.csv

global_params:
  delimiter: ","
  header_unit: nm
  signal_type: absorbance

task_type: regression
```

### Method 4: Python Dictionary

Pass a configuration dictionary directly:

```python
configs = DatasetConfigs({
    "train_x": "data/Xcal.csv",
    "train_y": "data/Ycal.csv",
    "train_group": "data/Mcal.csv",
    "test_x": "data/Xval.csv",
    "test_y": "data/Yval.csv",
    "test_group": "data/Mval.csv",
    "global_params": {
        "header_unit": "nm"
    },
    "task_type": "regression"
})
```

#### Key Name Flexibility

Keys are normalized automatically. All these variations are accepted (case-insensitive):

| Standard Key | Accepted Variations |
|--------------|---------------------|
| `train_x` | `train_x`, `x_train`, `Xtrain`, `trainX`, `X_train`, `TrainX`, etc. |
| `train_y` | `train_y`, `y_train`, `Ytrain`, `trainY`, etc. |
| `test_x` | `test_x`, `x_test`, `val_x`, `x_val`, `Xtest`, `Xval`, etc. |
| `test_y` | `test_y`, `y_test`, `val_y`, `y_val`, `Ytest`, `Yval`, etc. |
| `train_group` | `train_group`, `train_metadata`, `train_meta`, `train_m`, `m_train`, `metadata_train`, etc. |
| `test_group` | `test_group`, `test_metadata`, `val_metadata`, `test_m`, `m_val`, etc. |

### Method 5: In-Memory NumPy Arrays

Pass NumPy arrays directly for programmatic use:

```python
import numpy as np

X_train = np.random.randn(100, 500)  # 100 samples, 500 features
y_train = np.random.randn(100)
X_test = np.random.randn(30, 500)
y_test = np.random.randn(30)

configs = DatasetConfigs({
    "name": "my_array_dataset",
    "train_x": X_train,
    "train_y": y_train,
    "test_x": X_test,
    "test_y": y_test,
    "task_type": "regression"
})
```

### Method 6: Multi-Source Data (Multiple X Files)

For sensor fusion or multi-instrument data, provide lists of paths:

```python
configs = DatasetConfigs({
    "train_x": [
        "data/Xcal_NIR.csv",      # Source 1: NIR spectrometer
        "data/Xcal_MIR.csv"       # Source 2: MIR spectrometer
    ],
    "train_y": "data/Ycal.csv",   # Shared targets
    "test_x": [
        "data/Xval_NIR.csv",
        "data/Xval_MIR.csv"
    ],
    "test_y": "data/Yval.csv",
    "train_x_params": {
        "header_unit": ["nm", "cm-1"],      # Per-source units
        "signal_type": ["reflectance", "absorbance"]  # Per-source signal types
    },
    "task_type": "regression"
})
```

### Supported Configuration Keys (Complete Reference)

#### Data Path Keys

| Key | Type | Description |
|-----|------|-------------|
| `train_x` | `str`, `List[str]`, `np.ndarray` | Training features (required for training) |
| `train_y` | `str`, `List[int]`, `np.ndarray` | Training targets (path, column indices, or array) |
| `train_group` | `str`, `np.ndarray`, `pd.DataFrame` | Training metadata |
| `test_x` | `str`, `List[str]`, `np.ndarray` | Test features |
| `test_y` | `str`, `List[int]`, `np.ndarray` | Test targets |
| `test_group` | `str`, `np.ndarray`, `pd.DataFrame` | Test metadata |

#### Parameter Keys

| Key | Description |
|-----|-------------|
| `global_params` | Parameters applied to all file loading |
| `train_params` | Parameters applied to all train files |
| `test_params` | Parameters applied to all test files |
| `train_x_params` | Parameters for train_x loading only |
| `train_y_params` | Parameters for train_y loading only |
| `train_group_params` | Parameters for train_group loading only |
| `test_x_params` | Parameters for test_x loading only |
| `test_y_params` | Parameters for test_y loading only |
| `test_group_params` | Parameters for test_group loading only |

**Parameter precedence**: `*_x_params` > `train/test_params` > `global_params`

#### Loading Parameters (for `global_params`, `*_params`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delimiter` | `str` | `","` | CSV field delimiter |
| `decimal_separator` | `str` | `"."` | Decimal point character |
| `has_header` | `bool` | `true` | Whether first row is header |
| `header_unit` | `str` | `"cm-1"` | Header unit: `"nm"`, `"cm-1"`, `"none"`, `"text"`, `"index"` |
| `signal_type` | `str` | `null` | Signal type: `"absorbance"`, `"reflectance"`, `"reflectance%"`, `"transmittance"`, etc. |
| `encoding` | `str` | `"utf-8"` | File encoding |
| `na_policy` | `str` | `"remove"` | How to handle NaN values |

#### Task and Aggregation Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | `str` | auto | Dataset name (derived from path if not specified) |
| `task_type` | `str` | `"auto"` | `"regression"`, `"binary_classification"`, `"multiclass_classification"`, `"auto"` |
| `aggregate` | `str`, `bool` | `null` | Aggregation column name, `true` for y-based, or `null` |
| `aggregate_method` | `str` | `"mean"` | `"mean"`, `"median"`, `"vote"` |
| `aggregate_exclude_outliers` | `bool` | `false` | Exclude outliers before aggregation |

### Prediction-Only Configuration

For prediction scenarios (no training data), only `test_x` is required:

```python
configs = DatasetConfigs({
    "test_x": "data/new_samples.csv",
    "test_x_params": {
        "header_unit": "nm"
    }
})
```

---

## Configuration Schema

### Top-Level Structure

```yaml
# Dataset Configuration Schema v1.0

# Dataset identification (optional)
name: "my_dataset"
description: "Dataset description"

# Task configuration
task_type: regression | binary_classification | multiclass_classification | auto
signal_type: absorbance | reflectance | reflectance% | transmittance | auto

# === DATA SOURCES ===
# Option 1: Simple paths (backward compatible)
train_x: "path/to/train_features.csv"
train_y: "path/to/train_targets.csv"
test_x: "path/to/test_features.csv"
test_y: "path/to/test_targets.csv"
train_group: "path/to/train_metadata.csv"
test_group: "path/to/test_metadata.csv"

# Option 2: Unified files with column/row selection
files: [...]  # See File Definition section

# Option 3: Multi-source configuration
sources: [...]  # See Multi-Source section

# Option 4: Feature variations / preprocessed data
variations: [...]  # See Feature Variations section
variation_mode: separate | concat | select | compare
variation_select: ["var1", "var2"]  # When mode is "select"

# === GLOBAL PARAMETERS ===
global_params:
  delimiter: ","
  header_unit: nm | cm-1 | none | text | index
  has_header: true
  # ... other loading parameters

# === AGGREGATION ===
aggregate: sample_id | true | null
aggregate_method: mean | median | vote
aggregate_exclude_outliers: false

# === CROSS-VALIDATION FOLDS ===
folds: [...]  # See Fold Definition section
```

### File Definition Schema

Each file entry defines how to load and interpret a data file:

```yaml
files:
  - path: "data/measurements.csv"

    # File format parameters (optional - auto-detected from extension)
    format: csv | numpy | parquet | excel
    format_params:
      delimiter: ";"
      encoding: "utf-8"

    # === COLUMN SELECTION ===
    # Define which columns serve which role
    columns:
      # Features: columns to use as X (input)
      features:
        include: "400:2500"     # Range of columns (wavelengths)
        # OR
        include: [0, 1, 2, 3]   # Column indices
        # OR
        include: ["col1", "col2", "col3"]  # Column names
        # OR
        include: "all"          # All columns not assigned elsewhere
        exclude: ["wavelength_700", "bad_column"]  # Columns to exclude

      # Targets: columns to use as y (output)
      targets:
        include: "protein"      # Single column name
        # OR
        include: ["protein", "moisture"]  # Multiple targets
        # OR
        include: -1             # Last column

      # Metadata: auxiliary columns
      metadata:
        include: ["sample_id", "batch", "date", "operator"]
        # OR
        include: [0, 1, 2]      # First 3 columns

    # === ROW SELECTION ===
    rows:
      # Select specific rows
      include: "all"            # All rows (default)
      # OR
      include: [0, 1, 2, 100, 101]  # Specific row indices
      # OR
      include: "0:100"          # Row range (0-99)
      # OR
      include:                  # Conditional selection
        column: "split"
        values: ["train", "cal"]
      exclude: [50, 51]         # Rows to exclude

    # === PARTITION ASSIGNMENT ===
    partition: train | test | predict
    # OR (for mixed partitions in single file)
    partition:
      column: "partition"       # Column containing partition labels
      mapping:                  # Optional: map values to standard names
        "calibration": train
        "validation": test
    # OR
    partition:
      train: "0:80%"            # First 80% of rows
      test: "80%:100%"          # Last 20% of rows
    # OR
    partition:
      train: [0, 1, 2, ..., 79]  # Explicit row indices
      test: [80, 81, ..., 99]

    # === HEADER CONFIGURATION ===
    header:
      row: 0                    # Row index containing headers (default: 0)
      unit: nm | cm-1 | none    # Unit of wavelength headers
      prefix: "wl_"             # Prefix to strip from feature headers

    # === MULTI-FILE LINKING ===
    # Link this file's rows to another file by key column
    link:
      to: "other_file.csv"      # File to link to
      on: "sample_id"           # Column to match on
      how: left | right | inner | outer
```

### Compact Column Specification

For simpler cases, columns can be specified compactly:

```yaml
files:
  - path: "data.csv"
    # Compact format: first 3 cols metadata, last col target, rest features
    columns:
      metadata: ":3"            # Columns 0, 1, 2
      targets: "-1"             # Last column
      features: "3:-1"          # Everything in between
    partition: train
```

---

## Use Cases and Examples

### Use Case 1: Single File with All Data

**Scenario**: One CSV file containing features, targets, metadata, and partition labels.

```yaml
# dataset.yaml
name: wheat_protein_single_file

files:
  - path: "data/wheat_measurements.csv"

    columns:
      # First 3 columns are metadata
      metadata:
        include: ["sample_id", "batch", "measurement_date"]

      # Last column is target (protein content)
      targets:
        include: "protein_content"

      # Wavelength columns as features (400nm to 2500nm)
      features:
        include: "400:2500"     # Auto-detected as wavelength range
        # Alternative: include: "4:"  # Column index 4 onwards
        # Alternative: include: ["wl_400", "wl_402", ...]

    # Partition based on column value
    partition:
      column: "dataset_split"
      mapping:
        "calibration": train
        "validation": test

task_type: regression
signal_type: absorbance
```

**Equivalent Python dict**:

```python
config = {
    "name": "wheat_protein_single_file",
    "files": [{
        "path": "data/wheat_measurements.csv",
        "columns": {
            "metadata": {"include": ["sample_id", "batch", "measurement_date"]},
            "targets": {"include": "protein_content"},
            "features": {"include": "400:2500"}
        },
        "partition": {
            "column": "dataset_split",
            "mapping": {"calibration": "train", "validation": "test"}
        }
    }],
    "task_type": "regression",
    "signal_type": "absorbance"
}
```

---

### Use Case 2: Separate Train and Test Files

**Scenario**: Traditional split with separate files for training and testing.

```yaml
# Simple format (backward compatible)
name: wheat_simple

train_x: data/Xcal.csv
train_y: data/Ycal.csv
train_group: data/Mcal.csv

test_x: data/Xval.csv
test_y: data/Yval.csv
test_group: data/Mval.csv

global_params:
  header_unit: nm
  delimiter: ","
  has_header: true

task_type: regression
```

**Or using the files syntax**:

```yaml
name: wheat_explicit_files

files:
  # Training features
  - path: data/Xcal.csv
    columns:
      features: all
    partition: train
    header:
      unit: nm

  # Training targets
  - path: data/Ycal.csv
    columns:
      targets: all
    partition: train
    link:
      to: data/Xcal.csv
      on: row_index          # Link by row position

  # Training metadata
  - path: data/Mcal.csv
    columns:
      metadata: all
    partition: train
    link:
      to: data/Xcal.csv
      on: row_index

  # Test features
  - path: data/Xval.csv
    columns:
      features: all
    partition: test
    header:
      unit: nm

  # Test targets
  - path: data/Yval.csv
    columns:
      targets: all
    partition: test
    link:
      to: data/Xval.csv
      on: row_index

task_type: regression
signal_type: absorbance
```

---

### Use Case 3: Multiple Files Combined into Single Partition

**Scenario**: Training data split across multiple files that should be concatenated.

```yaml
name: multi_batch_dataset

files:
  # Multiple training feature files to concatenate
  - path: data/batch_001_X.csv
    columns:
      features: all
    partition: train
    header:
      unit: nm

  - path: data/batch_002_X.csv
    columns:
      features: all
    partition: train
    header:
      unit: nm

  - path: data/batch_003_X.csv
    columns:
      features: all
    partition: train
    header:
      unit: nm

  # All targets in one file with batch identifier
  - path: data/all_targets.csv
    columns:
      targets:
        include: "protein"
      metadata:
        include: ["sample_id", "batch"]
    partition: train
    # Match samples across files by sample_id
    link:
      to: [data/batch_001_X.csv, data/batch_002_X.csv, data/batch_003_X.csv]
      on: sample_id

  # Test data
  - path: data/test_X.csv
    columns:
      features: all
    partition: test

  - path: data/test_Y.csv
    columns:
      targets: all
    partition: test

task_type: regression
```

---

### Use Case 4: Mixed Features/Targets/Metadata in Single File

**Scenario**: One file where columns need to be assigned to different roles.

```yaml
name: all_in_one

files:
  - path: data/complete_dataset.csv

    # Row 0 is header, data starts at row 1
    header:
      row: 0

    columns:
      # Columns by index
      metadata:
        include: [0, 1, 2]     # sample_id, batch, date

      targets:
        include: [-2, -1]       # Last two columns: protein, moisture

      features:
        include: "3:-2"         # Everything between metadata and targets

    # Split by percentage
    partition:
      train: "0:80%"
      test: "80%:100%"

task_type: regression
```

---

### Use Case 5: Features and Targets in Same Columns (Wide Format)

**Scenario**: Time series or repeated measurements where features and targets are interleaved.

```yaml
name: time_series_prediction

files:
  - path: data/sensor_readings.csv

    columns:
      metadata:
        include: ["timestamp", "sensor_id", "location"]

      # Features: readings from time t-10 to t-1
      features:
        include:
          pattern: "reading_t-*"  # Regex pattern
        # OR explicit
        include: ["reading_t-10", "reading_t-9", ..., "reading_t-1"]

      # Target: reading at time t
      targets:
        include: "reading_t0"

    partition:
      column: "dataset"
      mapping:
        "training": train
        "testing": test

task_type: regression
```

---

### Use Case 6: Compressed Files and Archives

**Scenario**: Data stored in compressed formats.

```yaml
name: compressed_data

files:
  # Gzipped CSV
  - path: data/large_dataset.csv.gz
    format: csv
    format_params:
      delimiter: ","
    columns:
      features: "2:"
      targets: 1
      metadata: 0
    partition: train

  # Zip archive with multiple files
  - path: data/archive.zip
    archive_params:
      member: "train_features.csv"   # Specific file in archive
    columns:
      features: all
    partition: train

  - path: data/archive.zip
    archive_params:
      member: "train_targets.csv"
    columns:
      targets: all
    partition: train

  # Tar.gz archive
  - path: data/backup.tar.gz
    archive_params:
      member: "measurements/test_data.csv"
    columns:
      features: "1:"
      targets: 0
    partition: test

task_type: regression
```

---

### Use Case 7: NumPy Arrays

**Scenario**: Data stored as NumPy .npy or .npz files.

```yaml
name: numpy_dataset

files:
  # Single .npy file (2D array)
  - path: data/X_train.npy
    format: numpy
    columns:
      features: all
    partition: train

  - path: data/y_train.npy
    format: numpy
    columns:
      targets: all
    partition: train
    link:
      to: data/X_train.npy
      on: row_index

  # .npz file with multiple arrays
  - path: data/dataset.npz
    format: numpy
    numpy_params:
      key: "X_test"            # Array name in .npz
    columns:
      features: all
    partition: test

  - path: data/dataset.npz
    format: numpy
    numpy_params:
      key: "y_test"
    columns:
      targets: all
    partition: test

task_type: regression
```

**Python dict equivalent**:

```python
config = {
    "name": "numpy_dataset",
    "files": [
        {
            "path": "data/X_train.npy",
            "format": "numpy",
            "columns": {"features": "all"},
            "partition": "train"
        },
        {
            "path": "data/y_train.npy",
            "format": "numpy",
            "columns": {"targets": "all"},
            "partition": "train",
            "link": {"to": "data/X_train.npy", "on": "row_index"}
        }
    ],
    "task_type": "regression"
}
```

---

### Use Case 8: Row-Based Partition from External File

**Scenario**: Partition assignments stored in a separate file or array.

```yaml
name: external_partition

files:
  - path: data/all_features.csv
    columns:
      features: all
    header:
      unit: nm

  - path: data/all_targets.csv
    columns:
      targets: all
    link:
      to: data/all_features.csv
      on: row_index

# Partition defined externally
partition_file: data/splits.csv
partition_config:
  column: "partition"          # Column in splits.csv
  mapping:
    "cal": train
    "val": test
  # OR provide row indices directly
  # train_rows: [0, 1, 2, ..., 99]
  # test_rows: [100, 101, ..., 129]

task_type: regression
```

---

## Column and Row Selection

### Column Selection Syntax

```yaml
columns:
  features:
    # By names
    include: ["col1", "col2", "col3"]

    # By indices (0-based)
    include: [0, 1, 2, 10, 11, 12]

    # By range (Python slice syntax)
    include: "5:50"           # Columns 5-49
    include: "5:"             # Column 5 to end
    include: ":10"            # First 10 columns
    include: ":-2"            # All except last 2

    # Wavelength range (auto-detected numeric headers)
    include: "400:2500"       # 400nm to 2500nm
    include: "4000:10000"     # 4000 cm⁻¹ to 10000 cm⁻¹

    # Regex pattern
    include:
      pattern: "^wl_\\d+$"    # Columns matching pattern

    # All columns (default for features if others defined)
    include: all

    # Exclude specific columns from selection
    exclude: ["bad_column", "ignore_me"]
    exclude: [5, 10, 15]      # By index
```

### Row Selection Syntax

```yaml
rows:
  # All rows (default)
  include: all

  # By indices
  include: [0, 1, 2, 50, 51, 52]

  # By range
  include: "0:100"            # First 100 rows
  include: "100:"             # Row 100 to end

  # By percentage
  include: "0:80%"            # First 80%
  include: "20%:100%"         # Last 80%

  # By condition on column value
  include:
    column: "quality"
    operator: "=="            # ==, !=, >, <, >=, <=, in, not_in
    value: "good"

  # Multiple conditions (AND)
  include:
    - column: "quality"
      operator: "=="
      value: "good"
    - column: "year"
      operator: ">="
      value: 2020

  # Exclude specific rows
  exclude: [50, 51, 52]
  exclude:
    column: "outlier"
    value: true
```

---

## Partition Assignment

### Static Partition

```yaml
# Entire file belongs to one partition
partition: train

# OR
partition: test
```

### Column-Based Partition

```yaml
partition:
  column: "split"             # Column containing partition info
  mapping:                    # Optional: map values to standard names
    "calibration": train
    "cal": train
    "validation": test
    "val": test
    "predict": predict
```

### Percentage-Based Partition

```yaml
partition:
  train: "0:80%"              # First 80% to train
  test: "80%:100%"            # Last 20% to test

# With shuffling
partition:
  train: "80%"
  test: "20%"
  shuffle: true
  random_state: 42
```

### Index-Based Partition

```yaml
partition:
  train: [0, 1, 2, 3, ..., 79]     # Explicit row indices
  test: [80, 81, 82, ..., 99]

# OR reference external file/array
partition:
  train_file: "data/train_indices.txt"
  test_file: "data/test_indices.txt"
```

### Stratified Partition

```yaml
partition:
  train: "80%"
  test: "20%"
  stratify: "class"           # Stratify by this column
  random_state: 42
```

---

## Fold Definition

Cross-validation folds can be defined in several ways:

### Pre-defined Fold Indices

```yaml
# Explicit fold definitions
folds:
  - train: [0, 1, 2, 3, 4]
    val: [5, 6, 7]
  - train: [0, 1, 5, 6, 7]
    val: [2, 3, 4]
  - train: [2, 3, 4, 5, 6, 7]
    val: [0, 1]
```

### Fold Column in Data

```yaml
# Folds defined by column value
folds:
  column: "fold"              # Column containing fold assignments
  # Values in column: 0, 1, 2, 3, 4
  # Each unique value becomes a validation fold
```

### External Fold File

```yaml
# Folds from external file
folds:
  file: "data/cv_splits.json"
  format: json | yaml | csv

# Expected file format (JSON):
# [
#   {"train": [0, 1, 2], "val": [3, 4]},
#   {"train": [0, 3, 4], "val": [1, 2]},
#   ...
# ]
```

### Generated Folds (Shorthand)

```yaml
# Generate k-fold splits (actual splitting done at runtime)
folds:
  method: kfold | stratified_kfold | group_kfold | shuffle_split
  n_splits: 5
  shuffle: true
  random_state: 42
  # For group_kfold:
  group_column: "batch"
  # For stratified:
  stratify_column: "class"
```

---

## Multi-Source Datasets

For sensor fusion or multi-instrument data:

```yaml
name: sensor_fusion

sources:
  - name: "NIR"
    files:
      - path: data/nir_train.csv
        columns:
          features: all
        partition: train
        header:
          unit: nm

      - path: data/nir_test.csv
        columns:
          features: all
        partition: test
        header:
          unit: nm

  - name: "MIR"
    files:
      - path: data/mir_train.csv
        columns:
          features: all
        partition: train
        header:
          unit: cm-1

      - path: data/mir_test.csv
        columns:
          features: all
        partition: test
        header:
          unit: cm-1

# Shared targets (linked by sample_id)
targets:
  - path: data/targets.csv
    columns:
      targets: "protein"
      metadata: ["sample_id", "batch"]
    link:
      on: "sample_id"

task_type: regression
```

---

## Feature Variations and Preprocessed Data

This section describes how to configure datasets where features represent **variations or transformations** of the same underlying signal. This is particularly useful for:

- **Time series data**: Weather data where temperature, humidity, pressure, etc. are different "views" of the same time index
- **Preprocessed spectral data**: Loading pre-computed derivatives, SNV, MSC, etc. as separate feature sets
- **Multi-modal signals**: Different physical measurements from the same samples
- **Feature engineering outputs**: Features computed offline that should be treated as preprocessing variants

### Core Concept: Feature Variants

Traditional nirs4all pipelines apply preprocessing transforms sequentially in the pipeline. However, some workflows require:

1. **Pre-computed preprocessing**: Transforms applied offline (e.g., vendor software, external tools)
2. **Variable-as-preprocessing analogy**: In 1D time series, each variable (temperature, humidity) can be conceptually treated as a "preprocessed view" of the time dimension
3. **Variant comparison**: Testing which preprocessing or variable combination works best

The `variations` key allows declaring multiple feature variants that share the same sample structure.

### Variation Definition Schema

```yaml
name: weather_forecast

# Feature variations represent different "views" of the same samples
variations:
  - name: "temperature"
    description: "Temperature time series"
    files:
      - path: data/temperature.csv
        columns:
          features: all
        partition: train

  - name: "humidity"
    description: "Relative humidity time series"
    files:
      - path: data/humidity.csv
        columns:
          features: all
        partition: train

  - name: "pressure"
    description: "Atmospheric pressure time series"
    files:
      - path: data/pressure.csv
        columns:
          features: all
        partition: train

# Shared targets
targets:
  - path: data/labels.csv
    columns:
      targets: "precipitation"

# How to combine variations in the pipeline
variation_mode: separate | concat | select
# - separate: Each variation creates an independent pipeline branch
# - concat: Concatenate all variations horizontally (sensor fusion style)
# - select: Use variation_select to pick specific ones

# When mode is "select", specify which variations to use
variation_select: ["temperature", "humidity"]

task_type: regression
```

### Use Case: 1D Time Series with Multiple Variables

**Scenario**: Weather prediction where each meteorological variable is a separate feature set.

```yaml
name: weather_prediction

description: |
  Predict daily precipitation from hourly measurements.
  Each variable (temp, humidity, etc.) is treated as a "preprocessing variant"
  of the underlying time series.

# Define each variable as a variation
variations:
  - name: "raw_temperature"
    files:
      - path: data/hourly_temp.csv
        columns:
          features: "1:"        # Hours 1-24
          metadata: 0           # Date column
        partition:
          column: "split"

  - name: "temp_derivative"
    description: "Pre-computed temperature rate of change"
    files:
      - path: data/temp_derivative.csv
        columns:
          features: "1:"
        partition:
          column: "split"

  - name: "humidity"
    files:
      - path: data/hourly_humidity.csv
        columns:
          features: "1:"
        partition:
          column: "split"

  - name: "pressure"
    files:
      - path: data/hourly_pressure.csv
        columns:
          features: "1:"
        partition:
          column: "split"

# Shared targets
files:
  - path: data/daily_precipitation.csv
    columns:
      targets: "precip_mm"
      metadata: ["date", "station_id"]
    partition:
      column: "split"

# Explore all variations separately, then optionally combine
variation_mode: separate

task_type: regression
```

### Use Case: Pre-computed Spectral Preprocessing

**Scenario**: Spectral data with preprocessing applied offline by vendor software.

```yaml
name: nir_preprocessed_study

description: |
  NIR spectra with multiple preprocessing variants computed offline.
  Compare raw vs SNV vs derivative without re-computing.

variations:
  - name: "raw"
    description: "Raw absorbance spectra"
    files:
      - path: data/spectra_raw.csv
        columns:
          features: "1100:2500"
        partition: train
        header:
          unit: nm

  - name: "snv"
    description: "Standard Normal Variate preprocessed"
    files:
      - path: data/spectra_snv.csv
        columns:
          features: "1100:2500"
        partition: train
        header:
          unit: nm

  - name: "sg_derivative"
    description: "Savitzky-Golay 1st derivative"
    files:
      - path: data/spectra_sg1d.csv
        columns:
          features: "1100:2500"
        partition: train
        header:
          unit: nm

  - name: "snv_sg"
    description: "SNV followed by SG derivative"
    files:
      - path: data/spectra_snv_sg1d.csv
        columns:
          features: "1100:2500"
        partition: train
        header:
          unit: nm

# Test data variations (same preprocessing applied)
  - name: "raw"
    files:
      - path: data/test_spectra_raw.csv
        partition: test
        # ... (similar structure)

# Shared reference values
targets:
  - path: data/protein_reference.csv
    columns:
      targets: "protein"
    link:
      on: "sample_id"

# Run each variation as separate pipeline
variation_mode: separate

task_type: regression
signal_type: absorbance
```

### Use Case: Combining Variations (Sensor Fusion Style)

**Scenario**: Use multiple preprocessed versions together as expanded feature set.

```yaml
name: combined_preprocessing

variations:
  - name: "raw"
    files:
      - path: data/spectra_raw.csv
        columns:
          features: all
        partition: train

  - name: "snv"
    files:
      - path: data/spectra_snv.csv
        columns:
          features: all
        partition: train

  - name: "derivative"
    files:
      - path: data/spectra_deriv.csv
        columns:
          features: all
        partition: train

targets:
  - path: data/targets.csv
    columns:
      targets: "protein"

# Concatenate all variations into one wide feature matrix
variation_mode: concat
# Result: [raw_features | snv_features | derivative_features]

# Optionally prefix column names to track origin
variation_prefix: true  # Creates columns like "raw_1100", "snv_1100", etc.

task_type: regression
```

### Variation Mode Details

| Mode | Description | Pipeline Behavior |
|------|-------------|-------------------|
| `separate` | Each variation runs as independent pipeline | Multiple predictions, one per variation |
| `concat` | Horizontal concatenation of all variations | Single wide feature matrix |
| `select` | Use only specified variations | Subset of variations used |
| `compare` | Run each variation and rank by performance | Automatic comparison report |

### Inline Preprocessing Declaration

For cases where you want to document what preprocessing was applied (even if computed offline):

```yaml
variations:
  - name: "snv_normalized"
    preprocessing_applied:
      - type: "SNV"
        description: "Standard Normal Variate"
        software: "OPUS 8.0"
      - type: "SG_smooth"
        params:
          window: 15
          polyorder: 2
    files:
      - path: data/preprocessed_snv.csv
        # ...
```

This metadata is informational only but helps track provenance.

### Legacy Format with Variations

The legacy format can also support variations via multiple X paths:

```yaml
# Legacy format with multiple train_x as variations
train_x:
  - path: data/X_raw.csv
    variation: "raw"
  - path: data/X_snv.csv
    variation: "snv"
  - path: data/X_deriv.csv
    variation: "derivative"

train_y: data/Y.csv
variation_mode: separate

task_type: regression
```

Or using the multi-source syntax with variation semantics:

```yaml
# Multi-source reinterpreted as variations
train_x:
  - data/X_temp.csv
  - data/X_humidity.csv
  - data/X_pressure.csv

train_x_params:
  variation_names: ["temperature", "humidity", "pressure"]
  variation_mode: separate  # or concat, select

train_y: data/Y.csv
task_type: regression
```

---

## Validation Rules

The configuration validator enforces these rules:

### Required Data
- At least one file with features must be defined
- Either `train` or `test` partition must have features

### Column Consistency
- Same columns cannot be assigned to multiple roles (features/targets/metadata)
- All features files in same partition must have matching column count
- All targets must have same dimensionality

### Partition Consistency
- Row indices in partition definitions must be valid
- Percentage-based partitions must sum to 100% or less
- Linked files must have compatible row counts

### Fold Consistency
- Fold indices must be within partition bounds
- Fold train/val sets must not overlap within a fold
- Union of all fold validation sets should cover all samples (optional warning)

### File Existence
- All referenced files must exist (warning if not, error on load)
- Archive members must exist within archives

### Variation Consistency
- All variations must have the same number of samples (rows)
- Variation names must be unique within a configuration
- When `variation_mode: concat`, all variations must be present for both train and test partitions
- When `variation_mode: select`, all selected variation names must exist in the `variations` list
- Variations with the same name across partitions (train/test) are automatically paired

---

## Complete Example

```yaml
# Complete dataset configuration example
# wheat_protein_study.yaml

name: wheat_protein_2024
description: |
  Wheat protein content prediction using NIR spectroscopy.
  Multi-batch study with 3 instruments.

task_type: regression
signal_type: absorbance

# Global loading parameters
global_params:
  delimiter: ","
  encoding: utf-8
  decimal_separator: "."

# Multi-source configuration
sources:
  - name: "FOSS_NIR"
    files:
      - path: data/foss/batch1_spectra.csv.gz
        columns:
          features: "1100:2500"     # NIR region
        partition: train
        header:
          unit: nm

      - path: data/foss/batch2_spectra.csv.gz
        columns:
          features: "1100:2500"
        partition: train
        header:
          unit: nm

      - path: data/foss/validation_spectra.csv
        columns:
          features: "1100:2500"
        partition: test
        header:
          unit: nm

  - name: "Bruker_FT"
    files:
      - path: data/bruker/all_samples.npz
        format: numpy
        numpy_params:
          key: "X_train"
        columns:
          features: all
        partition: train

      - path: data/bruker/all_samples.npz
        format: numpy
        numpy_params:
          key: "X_test"
        columns:
          features: all
        partition: test

# Shared targets and metadata
files:
  - path: data/reference_values.csv
    columns:
      targets:
        include: ["protein_NIR", "protein_Kjeldahl"]
      metadata:
        include: ["sample_id", "batch", "harvest_date", "variety", "moisture"]
    rows:
      include:
        column: "qc_pass"
        value: true
    partition:
      column: "split"
      mapping:
        "calibration": train
        "validation": test

# Aggregation for repeated measurements
aggregate: sample_id
aggregate_method: mean
aggregate_exclude_outliers: true

# Cross-validation using group-aware splitting
folds:
  method: group_kfold
  n_splits: 5
  group_column: batch
  random_state: 42
```

---

## Migration from Simple Format

The simple format remains fully supported:

```yaml
# Simple format (continues to work)
train_x: data/Xcal.csv
train_y: data/Ycal.csv
test_x: data/Xval.csv
test_y: data/Yval.csv
task_type: regression
```

Is equivalent to:

```yaml
# Expanded format
files:
  - path: data/Xcal.csv
    columns: {features: all}
    partition: train
  - path: data/Ycal.csv
    columns: {targets: all}
    partition: train
  - path: data/Xval.csv
    columns: {features: all}
    partition: test
  - path: data/Yval.csv
    columns: {targets: all}
    partition: test
task_type: regression
```

---

## Appendix: Python API

```python
from nirs4all.data import DatasetConfigs

# From YAML file
configs = DatasetConfigs("path/to/config.yaml")

# From dict
configs = DatasetConfigs({
    "name": "my_dataset",
    "files": [
        {
            "path": "data.csv",
            "columns": {"features": "1:", "targets": 0},
            "partition": {"train": "0:80%", "test": "80%:100%"}
        }
    ],
    "task_type": "regression"
})

# From dataclass (future)
from nirs4all.data.config import DatasetConfig, FileConfig

config = DatasetConfig(
    name="my_dataset",
    files=[
        FileConfig(
            path="data.csv",
            columns={"features": "1:", "targets": 0},
            partition={"train": "0:80%", "test": "80%:100%"}
        )
    ],
    task_type="regression"
)
configs = DatasetConfigs(config)
```

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.2-draft | Dec 2025 | Added comprehensive "Currently Implemented" section documenting folder auto-scanning, all input methods, key normalization, and complete parameter reference |
| 1.0.1-draft | Dec 2025 | Added implementation status notice; added Feature Variations section for preprocessed data and 1D time series support |
| 1.0.0-draft | Dec 2025 | Initial specification |