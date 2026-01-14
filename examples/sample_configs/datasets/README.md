# Test Dataset Configurations

This folder contains 30 YAML configuration files covering all possible dataset loading scenarios in nirs4all. Each config has a matching synthetic dataset in `../sample_datasets/`.

## Overview

| Category | Count | Description |
|----------|-------|-------------|
| A | 5 | File format & delimiter variations |
| B | 5 | Loading parameters (headers, signal types) |
| C | 5 | File structure variations |
| D | 5 | Partition & split strategies |
| E | 5 | Multi-source & aggregation |
| F | 5 | Task types & feature variations |

---

## Category A: File Format & Delimiters

Basic CSV parsing variations.

| Config | Key Options | Description |
|--------|-------------|-------------|
| `A01_csv_semicolon` | `delimiter: ";"` | Standard semicolon-delimited CSV (default) |
| `A02_csv_comma` | `delimiter: ","` | Comma-delimited CSV |
| `A03_csv_tab` | `delimiter: "\t"` | Tab-separated values (.tsv) |
| `A04_csv_pipe` | `delimiter: "\|"` | Pipe-delimited CSV |
| `A05_csv_european` | `delimiter: ";"`, `decimal_separator: ","` | European format (semicolon + comma decimal) |

---

## Category B: Loading Parameters

Header units, signal types, and encoding options.

| Config | Key Options | Description |
|--------|-------------|-------------|
| `B01_no_header` | `has_header: false`, `header_unit: "index"` | Headerless files with numeric column indices |
| `B02_wavenumber` | `header_unit: "cm-1"` | Wavenumber headers (cm⁻¹) |
| `B03_wavelength` | `header_unit: "nm"`, `signal_type: "absorbance"` | Wavelength headers with absorbance signal |
| `B04_reflectance` | `signal_type: "reflectance%"` | Reflectance percentage signal type |
| `B05_encoding_skiprows` | `encoding: "latin-1"`, `skip_rows: 2` | Latin-1 encoding with header rows to skip |

---

## Category C: File Structures

Different ways to organize dataset files.

| Config | Key Options | Description |
|--------|-------------|-------------|
| `C01_legacy_separate` | `train_x`, `train_y`, `test_x`, `test_y` | Legacy format with separate train/test X/Y files |
| `C02_combined_single` | Single file + `partition.column` | All data in one file with split column |
| `C03_standard_folder` | `Xcal.csv`, `Ycal.csv`, `Xval.csv`, `Yval.csv` | Standard nirs4all folder structure |
| `C04_with_metadata` | `train_group`, `test_group` | Includes metadata/group files |
| `C05_compressed` | `.csv.gz` extension | Gzip-compressed CSV files |

---

## Category D: Partition Strategies

Train/test splitting methods for single-file or combined datasets.

| Config | Key Options | Description |
|--------|-------------|-------------|
| `D01_column_partition` | `partition.method: "column"`, `partition.column: "split"` | Split based on column values (train/test) |
| `D02_percentage_partition` | `partition.method: "percentage"`, `partition.train_percent: 80` | Random 80/20 split |
| `D03_index_partition` | `partition.method: "index"`, `partition.train_indices: [...]` | Explicit row indices for train/test |
| `D04_stratified_partition` | `partition.method: "stratified"`, `partition.stratify_column` | Stratified split maintaining class balance |
| `D05_custom_folds` | `folds: [...]` | Predefined cross-validation fold assignments |

---

## Category E: Multi-Source & Aggregation

Multi-instrument datasets and sample aggregation.

| Config | Key Options | Description |
|--------|-------------|-------------|
| `E01_dual_source` | `sources: [NIR, MIR]` | Two spectral sources (NIR + MIR) |
| `E02_nir_markers` | `sources: [NIR, markers]` | NIR spectra + auxiliary marker features |
| `E03_shared_targets` | `shared_targets.path`, `shared_targets.link_by` | Multiple sources sharing one target file |
| `E04_aggregate_mean` | `aggregation.column`, `aggregation.method: "mean"` | Aggregate replicate spectra by sample ID |
| `E05_aggregate_outliers` | `aggregation.exclude_outliers: true` | Aggregation with Hotelling T² outlier removal |

### Multi-Source File Naming

Multi-source datasets use prefixed filenames:
```
E01_dual_source/
├── NIR_train.csv      # Source 1: NIR training spectra
├── NIR_test.csv       # Source 1: NIR test spectra
├── MIR_train.csv      # Source 2: MIR training spectra
├── MIR_test.csv       # Source 2: MIR test spectra
├── Y_train.csv        # Shared targets (training)
├── Y_test.csv         # Shared targets (test)
└── metadata_train.csv # Optional metadata
```

---

## Category F: Task Types & Variations

Different ML task types and feature variation modes.

| Config | Key Options | Description |
|--------|-------------|-------------|
| `F01_regression` | `task_type: "regression"` | Continuous target prediction |
| `F02_binary_class` | `task_type: "binary_classification"` | Two-class classification |
| `F03_multiclass` | `task_type: "multiclass_classification"` | Multi-class classification (5 classes) |
| `F04_variations_separate` | `variation_mode: "separate"` | Multiple preprocessed versions, run independently |
| `F05_variations_concat` | `variation_mode: "concat"` | Multiple versions, concatenate features |

### Feature Variations

Variations allow storing multiple preprocessed versions of the same data:

```yaml
variations:
  - name: raw
    description: Raw spectral data
    train_x: X_raw_train.csv
    test_x: X_raw_test.csv
  - name: snv
    description: SNV preprocessed spectra
    train_x: X_snv_train.csv
    test_x: X_snv_test.csv
    preprocessing_applied:
      - type: SNV
        description: Standard Normal Variate
variation_mode: separate  # or "concat", "select", "compare"
```

---

## Usage

### Loading with nirs4all

```python
from nirs4all.data import DatasetConfigs

# Load a specific config
configs = DatasetConfigs("sample_configs/datasets/A01_csv_semicolon.yaml")
dataset = configs.get_dataset_at(0)

print(f"Samples: {dataset.num_samples}")
print(f"Features: {dataset.num_features}")
```

### Linking to Webapp

Use the included script to link all datasets to the webapp:

```bash
cd examples
python scripts/link_test_datasets.py

# Or link specific patterns
python scripts/link_test_datasets.py --filter "A*"
```

### Verification

Verify all configs load correctly:

```bash
cd examples
python scripts/verify_test_datasets.py
```

---

## Loader Compatibility

| Category | Status | Notes |
|----------|--------|-------|
| A (Delimiters) | ✅ Fully supported | All 5 configs load correctly |
| B (Loading params) | ✅ Fully supported | All 5 configs load correctly |
| C (File structures) | ⚠️ Partial | C02 (single-file) needs partition support |
| D (Partitions) | ⚠️ Partial | D01-D04 need partition loader support |
| E (Multi-source) | ⚠️ Partial | E01-E03 need multi-source loader support |
| F (Task types) | ⚠️ Partial | F04-F05 need variations loader support |

**Current status: 20/30 configs load successfully.**

The remaining 10 require extended loader functionality for:
- Column-based partitioning
- Multi-source datasets
- Feature variations

---

## Schema Reference

See the full `DatasetConfig` schema in:
- Python: `nirs4all/data/config_parser.py`
- TypeScript: `nirs4all_webapp/src/types/datasets.ts`
