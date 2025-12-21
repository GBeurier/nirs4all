# Migration Guide: Legacy to New Dataset Configuration

This guide helps you migrate from the legacy dataset configuration format to the new, more powerful configuration system introduced in nirs4all v0.5.

## Overview

The new configuration system provides:
- **Multiple file formats**: CSV, NumPy, Parquet, Excel, MATLAB
- **Flexible column/row selection**: Select data by name, index, or pattern
- **Multiple partition methods**: Static, column-based, percentage, or index-based
- **Multi-source support**: Sensor fusion with multiple feature sources
- **Feature variations**: Pre-computed preprocessing variants
- **Cross-validation folds**: Load pre-defined fold assignments

**Good news**: The legacy format continues to work unchanged. You can migrate gradually.

## Quick Comparison

### Legacy Format (Still Supported)

```yaml
# Simple train/test split
train_x: data/Xcal.csv
train_y: data/Ycal.csv
test_x: data/Xval.csv
test_y: data/Yval.csv

global_params:
  delimiter: ";"
  has_header: true
  header_unit: cm-1

task_type: regression
```

### New Sources Format

```yaml
# Multi-instrument data
sources:
  - name: "NIR"
    train_x: data/NIR_train.csv
    test_x: data/NIR_test.csv
    params:
      header_unit: nm

  - name: "MIR"
    train_x: data/MIR_train.csv
    test_x: data/MIR_test.csv
    params:
      header_unit: cm-1

targets:
  path: data/targets.csv

task_type: regression
```

### New Variations Format

```yaml
# Pre-computed preprocessing variants
variations:
  - name: raw
    train_x: data/X_raw_train.csv
    test_x: data/X_raw_test.csv

  - name: snv
    description: "SNV preprocessed"
    train_x: data/X_snv_train.csv
    test_x: data/X_snv_test.csv

  - name: msc
    description: "MSC preprocessed"
    train_x: data/X_msc_train.csv
    test_x: data/X_msc_test.csv

variation_mode: compare  # Run each and compare results

targets:
  path: data/Y.csv

task_type: regression
```

## Migration Steps

### Step 1: Validate Your Current Configuration

Use the CLI to validate your existing configuration:

```bash
nirs4all dataset validate path/to/config.yaml
```

### Step 2: Choose Your Target Format

| Use Case | Recommended Format |
|----------|-------------------|
| Simple train/test split | Legacy (no change needed) |
| Multiple instruments | Sources format |
| Pre-computed preprocessing | Variations format |
| Complex file layouts | Files format |

### Step 3: Convert Configuration

#### Converting to Sources Format

**Before (Legacy multi-source):**

```yaml
train_x:
  - data/sensor1_train.csv
  - data/sensor2_train.csv
test_x:
  - data/sensor1_test.csv
  - data/sensor2_test.csv
train_y: data/Y_train.csv
test_y: data/Y_test.csv
```

**After (Sources format):**

```yaml
sources:
  - name: "sensor1"
    files:
      - path: data/sensor1_train.csv
        partition: train
      - path: data/sensor1_test.csv
        partition: test

  - name: "sensor2"
    files:
      - path: data/sensor2_train.csv
        partition: train
      - path: data/sensor2_test.csv
        partition: test

targets:
  path: data/Y.csv  # Shared targets
```

#### Converting to Variations Format

**Before (Manual switching):**

```python
# In code, manually switching between preprocessing variants
configs_raw = DatasetConfigs({"train_x": "X_raw.csv", ...})
configs_snv = DatasetConfigs({"train_x": "X_snv.csv", ...})
```

**After (Variations format):**

```yaml
variations:
  - name: raw
    train_x: data/X_raw.csv
    test_x: data/X_raw_test.csv

  - name: snv
    train_x: data/X_snv.csv
    test_x: data/X_snv_test.csv
    preprocessing_applied:
      - type: SNV
        software: "nirs4all"

variation_mode: compare
targets:
  path: data/Y.csv
```

### Step 4: Use New Loading Parameters

The new format supports richer loading parameters:

```yaml
global_params:
  delimiter: ";"
  decimal_separator: "."
  has_header: true
  header_unit: cm-1
  signal_type: absorbance
  encoding: utf-8
  na_policy: remove

# Override per-partition
train_x_params:
  header_unit: nm  # Different unit for training data
```

### Step 5: Test Your Migration

```bash
# Validate new configuration
nirs4all dataset validate new_config.yaml

# Inspect configuration details
nirs4all dataset inspect new_config.yaml --detect

# Compare old and new
nirs4all dataset diff old_config.yaml new_config.yaml
```

## Key Changes Summary

| Feature | Legacy | New |
|---------|--------|-----|
| Multi-source | List of paths in `train_x` | Named `sources` with metadata |
| Preprocessing variants | Manual switching | `variations` with modes |
| Shared targets | Repeated in each config | Single `targets` block |
| Partition assignment | Inferred from key names | Explicit `partition` field |
| Fold definitions | Not in config | `folds` block or file reference |
| File formats | CSV only | CSV, NumPy, Parquet, Excel, MATLAB |

## Troubleshooting

### Error: "No data source specified"

Your configuration needs at least one of:
- `train_x` or `test_x` (legacy)
- `sources` (new multi-source)
- `variations` (new variations)
- `folder` (auto-scan)

### Error: "variation_mode='select' requires 'variation_select'"

When using `variation_mode: select`, you must specify which variations to use:

```yaml
variation_mode: select
variation_select:
  - raw
  - snv
```

### Warning: "Sample count mismatch across sources"

All sources must have the same number of samples. Check that your data files have consistent row counts.

## Getting Help

- Run `nirs4all dataset validate config.yaml -v` for verbose output
- Check the [Dataset Configuration Specification](./dataset_config_specification.md)
- See [example configurations](../../examples/configs/)

## Backward Compatibility

The legacy format will continue to be supported for the foreseeable future. You can:
- Mix legacy keys with new features
- Convert legacy to new format using `nirs4all dataset export`
- Use the validation CLI to check compatibility
