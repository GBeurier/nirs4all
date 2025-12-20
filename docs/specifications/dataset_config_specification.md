# Dataset Configuration Specification

**Version**: 1.0.0-draft
**Status**: Draft Specification
**Date**: December 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Supported File Formats](#supported-file-formats)
3. [Core Concepts](#core-concepts)
4. [Configuration Schema](#configuration-schema)
5. [Use Cases and Examples](#use-cases-and-examples)
6. [Column and Row Selection](#column-and-row-selection)
7. [Partition Assignment](#partition-assignment)
8. [Fold Definition](#fold-definition)
9. [Multi-Source Datasets](#multi-source-datasets)
10. [Validation Rules](#validation-rules)

---

## Overview

The nirs4all dataset configuration system provides a flexible, declarative way to load spectral and tabular data from various file formats and structures. The configuration can be specified in:

- **YAML files** (`.yaml`, `.yml`) - recommended for readability
- **JSON files** (`.json`) - for programmatic generation
- **Python dictionaries** - for inline API usage
- **Python dataclasses** (future) - for type-safe configuration

### Design Principles

1. **Flexibility**: Handle any combination of files, columns, and partitions
2. **Explicitness**: Clear declaration of data roles (features, targets, metadata)
3. **Composability**: Combine multiple files into unified datasets
4. **Backward Compatibility**: Existing simple configs continue to work
5. **Validation**: Early detection of configuration errors

---

## Supported File Formats

### Primary Formats

| Format | Extensions | Description |
|--------|------------|-------------|
| CSV | `.csv` | Comma/delimiter-separated values |
| Compressed CSV | `.csv.gz`, `.csv.zip`, `.csv.bz2` | Compressed CSV files |
| NumPy | `.npy`, `.npz` | NumPy array files |
| Parquet | `.parquet` | Apache Parquet columnar format |
| Excel | `.xlsx`, `.xls` | Excel spreadsheets |
| MATLAB | `.mat` | MATLAB data files |

### Archive Formats

| Format | Extensions | Description |
|--------|------------|-------------|
| Gzip | `.gz` | Gzip-compressed single file |
| Zip | `.zip` | Zip archive (single or multiple files) |
| Tar | `.tar`, `.tar.gz`, `.tgz` | Tar archive (optionally compressed) |

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
| 1.0.0-draft | Dec 2025 | Initial specification |
