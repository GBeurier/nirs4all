# Dataset Configuration Troubleshooting Guide

This guide helps diagnose and resolve common issues when loading NIRS datasets.

## Error Code Reference

### Schema Errors (E1xx)

| Code | Description | Solution |
|------|-------------|----------|
| E101 | Missing required field | Add the missing field to your configuration |
| E102 | Invalid field type | Check expected type in schema documentation |
| E103 | Invalid enum value | Use one of the allowed values listed in error |
| E104 | Configuration validation failed | Check the detailed validation messages |

**Example - E101 Missing Required Field:**
```yaml
# ❌ Error: E101 - Missing required field 'train_x'
name: my_dataset

# ✅ Fixed: Add required train_x path
name: my_dataset
train_x: data/spectra.csv
```

### File Errors (E2xx)

| Code | Description | Solution |
|------|-------------|----------|
| E201 | File not found | Check file path exists and is accessible |
| E202 | Permission denied | Check file permissions |
| E203 | Invalid file format | Ensure file matches expected format (CSV, TSV, etc.) |
| E204 | File is empty | Verify file contains data |
| E205 | Encoding error | Specify correct encoding in params |

**Example - E201 File Not Found:**
```yaml
# Check paths are relative to config file or use absolute paths
train_x: ./data/spectra.csv  # Relative to config
train_x: /home/user/project/data/spectra.csv  # Absolute
```

### Data Errors (E3xx)

| Code | Description | Solution |
|------|-------------|----------|
| E301 | Missing values detected | Handle NaN/missing values in preprocessing |
| E302 | Shape mismatch | Ensure X and y have matching sample counts |
| E303 | Invalid numeric data | Check for non-numeric values in spectral data |
| E304 | Spectral range inconsistent | Verify wavelength headers match across files |
| E305 | Duplicate samples | Use aggregation or remove duplicates |

**Example - E302 Shape Mismatch:**
```yaml
# train_x has 100 samples, train_y has 95 samples
# Check for:
# 1. Header row counted as data
# 2. Missing samples in target file
# 3. Different sample ID formats
```

### Loading Errors (E4xx)

| Code | Description | Solution |
|------|-------------|----------|
| E401 | Delimiter detection failed | Explicitly set delimiter in params |
| E402 | Header parsing failed | Check header format matches header_unit |
| E403 | Data type conversion failed | Check data contains valid numbers |
| E404 | Memory error | Use lazy_loading or process in chunks |

**Example - E401 Delimiter Detection:**
```yaml
# If auto-detection fails, specify delimiter explicitly
params:
  delimiter: ","   # CSV
  delimiter: "\t"  # TSV
  delimiter: ";"   # European CSV
```

### Partition Errors (E5xx)

| Code | Description | Solution |
|------|-------------|----------|
| E501 | Partition overlap | Ensure train/val/test don't share samples |
| E502 | Empty partition | Check partition indices are valid |
| E503 | Invalid partition indices | Indices must be within sample count |
| E504 | Partition sum mismatch | Partition sizes should account for all samples |

### Aggregation Errors (E6xx)

| Code | Description | Solution |
|------|-------------|----------|
| E601 | Group column not found | Check group_by column exists in metadata |
| E602 | Aggregation method failed | Check method name is valid |
| E603 | Custom aggregation error | Verify custom function signature |
| E604 | Empty group after aggregation | Some groups may have all outliers |

**Example - E601 Group Column:**
```yaml
# Ensure the group column exists in your data
aggregation:
  group_by: sample_id  # Must match column name exactly (case-sensitive)
```

### Variation Errors (E7xx)

| Code | Description | Solution |
|------|-------------|----------|
| E701 | Variation definition error | Check variation syntax |
| E702 | Invalid spectral range | Range must be within data bounds |
| E703 | Resampling error | Check resample parameters |
| E704 | Noise application error | Verify noise level is valid (0-1) |

### Fold Errors (E8xx)

| Code | Description | Solution |
|------|-------------|----------|
| E801 | Fold definition error | Check fold indices syntax |
| E802 | Fold overlap | Ensure folds don't share test samples |
| E803 | Invalid fold indices | Indices must be within sample count |
| E804 | Inconsistent fold structure | All folds should have train and test |

### Runtime Errors (E9xx)

| Code | Description | Solution |
|------|-------------|----------|
| E901 | Cache error | Clear cache and retry |
| E902 | Lazy loading error | Try with lazy_loading: false |
| E903 | Timeout error | Increase timeout or reduce data size |
| E904 | Memory limit exceeded | Use lazy loading or reduce batch size |

---

## Common Scenarios

### Scenario 1: European CSV Format

European CSV files use semicolons as delimiters and commas as decimal separators.

```yaml
params:
  delimiter: ";"
  decimal: ","
```

### Scenario 2: Wavelength Headers with Units

If headers contain units (e.g., "1100 nm", "4000 cm-1"):

```yaml
params:
  header_unit: nm       # or cm-1, um
  header_regex: null    # Use default pattern
```

### Scenario 3: Large Dataset Memory Issues

For datasets that exceed available memory:

```yaml
performance:
  lazy_loading: true
  cache_enabled: true
  cache_max_size_mb: 1024  # Limit cache size
```

### Scenario 4: Sample Replicates

When each sample has multiple measurements:

```yaml
aggregation:
  group_by: sample_id
  method: mean
  exclude_outliers: true
  outlier_threshold: 2.5
```

### Scenario 5: Metadata Linking Issues

When sample IDs don't match between files:

```yaml
# Check for common issues:
# 1. Leading/trailing whitespace: "  sample1  " vs "sample1"
# 2. Case differences: "Sample1" vs "sample1"
# 3. Numeric formatting: "001" vs "1"

metadata:
  path: metadata.csv
  link_by: sample_id
  strip_whitespace: true  # Remove whitespace
  case_sensitive: false   # Ignore case
```

---

## Validation Workflow

Use the CLI to validate configurations before running pipelines:

```bash
# Validate configuration syntax
nirs4all dataset validate config.yaml

# Inspect data with auto-detection
nirs4all dataset inspect data.csv --detect

# Export normalized configuration
nirs4all dataset export config.yaml -o normalized.yaml

# Compare configurations
nirs4all dataset diff config1.yaml config2.yaml
```

---

## Getting Diagnostic Reports

For detailed diagnostics, enable verbose mode:

```python
from nirs4all.data import DatasetConfigs
from nirs4all.data.schema.validation import DiagnosticBuilder

# Create diagnostic builder
diagnostics = DiagnosticBuilder()

# Load with diagnostics
try:
    config = DatasetConfigs.from_yaml("config.yaml")
except Exception as e:
    # Get diagnostic report
    report = diagnostics.build()
    print(report.to_text())

    # Or save as JSON for analysis
    report.save_json("diagnostics.json")
```

---

## FAQ

**Q: My file loads but wavelengths are wrong**
A: Check `header_unit` matches your data. Use `nirs4all dataset inspect file.csv --detect` to see detected parameters.

**Q: Aggregation removes too many samples**
A: Lower `outlier_threshold` or set `exclude_outliers: false`.

**Q: Cache isn't being used**
A: Ensure `cache_enabled: true` and check cache size limits.

**Q: Getting OOM errors with large datasets**
A: Enable `lazy_loading: true` in performance settings.

**Q: Configuration works locally but fails in CI**
A: Use absolute paths or paths relative to config file location.
