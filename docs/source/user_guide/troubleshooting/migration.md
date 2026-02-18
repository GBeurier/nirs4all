# Migration Guide

This guide helps you migrate from older versions of nirs4all to the current version. It covers API changes, prediction format updates, and dataset configuration migrations.

## Table of Contents

1. [API Migration (v0.5 → v0.6+)](#api-migration-v05--v06)
2. [Storage Migration (v0.7 → v0.8+)](#storage-migration-v07--v08)
3. [Dataset Configuration Migration](#dataset-configuration-migration)
4. [Prediction Format Migration](#prediction-format-migration)
5. [Troubleshooting](#troubleshooting)

---

## API Migration (v0.5 → v0.6+)

nirs4all v0.6 introduced a simplified module-level API that reduces boilerplate while maintaining full functionality. The module-level API is now the only supported API.

### What Changed

| Aspect | Classic API | New API (v0.6+) |
|--------|-------------|-----------------|
| Entry point | `PipelineRunner.run()` | `nirs4all.run()` |
| Configuration | Explicit config objects | Inline parameters |
| Result access | `predictions.top(n=1)[0]` | `result.best` |
| Sessions | N/A | `nirs4all.session()` |
| sklearn integration | Manual | `NIRSPipeline` wrapper |

### Quick Comparison

#### Classic API (Deprecated)

```python
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

# Create configuration objects
pipeline_config = PipelineConfigs(
    [MinMaxScaler(), PLSRegression(n_components=10)],
    name="MyPipeline"
)
dataset_config = DatasetConfigs("sample_data/regression")

# Create runner and execute
runner = PipelineRunner(
    verbose=1,
    save_artifacts=True,
    save_charts=False
)
predictions, per_dataset = runner.run(pipeline_config, dataset_config)

# Access results
best = predictions.top(n=1)[0]
print(f"Best RMSE: {best.get('rmse', 'N/A')}")
```

#### New Module-Level API (Recommended)

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

# Direct execution with inline configuration
result = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(n_components=10)],
    dataset="sample_data/regression",
    name="MyPipeline",
    verbose=1,
    save_artifacts=True,
    save_charts=False
)

# Convenient result access
print(f"Best RMSE: {result.best_rmse:.4f}")
print(f"Best R²: {result.best_r2:.4f}")
```

### Migration Steps

#### 1. Basic Training

**Before:**
```python
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

runner = PipelineRunner(verbose=1, save_artifacts=True)
predictions, _ = runner.run(
    PipelineConfigs(pipeline, "name"),
    DatasetConfigs("path/to/data")
)
best = predictions.top(n=1)[0]
```

**After:**
```python
import nirs4all

result = nirs4all.run(
    pipeline=pipeline,
    dataset="path/to/data",
    name="name",
    verbose=1,
    save_artifacts=True
)
best = result.best
```

#### 2. Accessing Results

**Before:**
```python
top_5 = predictions.top(n=5)
best = predictions.top(n=1)[0]
rmse = best.get('rmse', float('nan'))
r2 = best.get('r2', float('nan'))
pls_preds = predictions.filter_predictions(model_name='PLSRegression')
```

**After:**
```python
top_5 = result.top(n=5)
rmse = result.best_rmse
r2 = result.best_r2
pls_preds = result.filter(model_name='PLSRegression')
print(result.num_predictions)
print(result.get_models())
```

#### 3. Prediction

**Before:**
```python
runner = PipelineRunner(verbose=0)
y_pred, metadata = runner.predict(source=best_prediction, dataset=new_data)
```

**After:**
```python
predict_result = nirs4all.predict(
    source=result.best,
    dataset=new_data,
    verbose=0
)
y_pred = predict_result.values
df = predict_result.to_dataframe()
```

#### 4. Model Export

**Before:**
```python
runner = PipelineRunner(save_artifacts=True)
predictions, _ = runner.run(pipeline_config, dataset_config)
best = predictions.top(n=1)[0]
runner.export(source=best, output_path="exports/model.n4a")
```

**After:**
```python
result = nirs4all.run(pipeline, dataset, save_artifacts=True)
result.export("exports/model.n4a")
```

### New Features in v0.6+

#### Sessions for Multiple Runs

```python
with nirs4all.session(verbose=1, save_artifacts=True) as s:
    result1 = nirs4all.run(pipeline1, data, name="PLS", session=s)
    result2 = nirs4all.run(pipeline2, data, name="RF", session=s)
    result3 = nirs4all.run(pipeline3, data, name="SVM", session=s)
```

#### sklearn Integration

```python
from nirs4all.sklearn import NIRSPipeline

result = nirs4all.run(pipeline, dataset, save_artifacts=True)
pipe = NIRSPipeline.from_result(result)
y_pred = pipe.predict(X_test)
score = pipe.score(X_test, y_test)
```

### Migration Checklist

- [ ] Replace `PipelineRunner(...)` with `nirs4all.run(...)`
- [ ] Remove explicit `PipelineConfigs` and `DatasetConfigs` wrappers
- [ ] Update result access from `predictions.top(n=1)[0]` to `result.best`
- [ ] Use `result.best_rmse`, `result.best_r2` for quick access
- [ ] Consider using `nirs4all.session()` for multiple related runs
- [ ] Use `NIRSPipeline.from_result()` for sklearn/SHAP integration
- [ ] Update exports from `runner.export(source=best, ...)` to `result.export(...)`

---

## Storage Migration (v0.7 → v0.8+)

nirs4all v0.8 moved prediction arrays from DuckDB to Parquet sidecar files for better performance and disk efficiency.

### What Changed

| Aspect | Before (v0.7) | After (v0.8+) |
|--------|---------------|----------------|
| Prediction arrays | `prediction_arrays` DuckDB table | `arrays/<dataset>.parquet` Parquet files |
| Array format | DuckDB `DOUBLE[]` columns | Zstd-compressed Parquet with list columns |
| Array management | `WorkspaceStore.save_prediction_arrays()` | `ArrayStore.save_batch()` |
| DuckDB tables | 7 (including `prediction_arrays`) | 7 (replaced with `projects`) |

### Automatic Migration

Migration is **fully automatic**. When you open a workspace with a legacy `prediction_arrays` table, nirs4all auto-migrates all rows to Parquet sidecar files and drops the DuckDB table. No manual action required.

### Manual Migration (Optional)

For explicit control, use the migration tool:

```python
from nirs4all.pipeline.storage import migrate_arrays_to_parquet, verify_migrated_store

# Run migration
report = migrate_arrays_to_parquet("workspace/")
print(f"Migrated {report.rows_migrated} rows across {report.datasets_migrated} datasets")

# Verify migration
verify_migrated_store("workspace/")
```

### Migration Checklist

- [ ] Backup your workspace before upgrading (optional but recommended)
- [ ] Open workspace with nirs4all v0.8+ (auto-migration happens automatically)
- [ ] Verify `workspace/arrays/` directory contains per-dataset `.parquet` files
- [ ] Verify `prediction_arrays` table no longer exists in DuckDB

---

## Dataset Configuration Migration

The new configuration system provides:
- **Multiple file formats**: CSV, NumPy, Parquet, Excel, MATLAB
- **Flexible column/row selection**: Select data by name, index, or pattern
- **Multiple partition methods**: Static, column-based, percentage, or index-based
- **Multi-source support**: Sensor fusion with multiple feature sources
- **Feature variations**: Pre-computed preprocessing variants
- **Cross-validation folds**: Load pre-defined fold assignments

:::{note}
The legacy format continues to work unchanged. You can migrate gradually.
:::

### Quick Comparison

#### Legacy Format (Still Supported)

```yaml
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

#### New Sources Format

```yaml
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

#### New Variations Format

```yaml
variations:
  - name: raw
    train_x: data/X_raw_train.csv
    test_x: data/X_raw_test.csv

  - name: snv
    description: "SNV preprocessed"
    train_x: data/X_snv_train.csv
    test_x: data/X_snv_test.csv

variation_mode: compare
targets:
  path: data/Y.csv
task_type: regression
```

### Converting Configurations

#### Multi-Source (Legacy → Sources Format)

**Before:**
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

**After:**
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
  path: data/Y.csv
```

### Validation Commands

```bash
# Validate configuration
nirs4all dataset validate path/to/config.yaml

# Inspect configuration details
nirs4all dataset inspect new_config.yaml --detect

# Compare configurations
nirs4all dataset diff old_config.yaml new_config.yaml
```

---

## Prediction Format Migration

### New Fields in Predictions (v0.9+)

| Field | Description |
|-------|-------------|
| `trace_id` | Unique identifier for the execution trace |
| `model_artifact_id` | Reference to the saved model artifact |
| `execution_hash` | Hash of the exact execution path |
| `step_artifacts` | List of artifact IDs for each pipeline step |

### Impact

Old predictions without the new fields will:
- ✅ Continue to work for basic operations
- ✅ Work with `predict()` if model folder still exists
- ⚠️ Not support `retrain()` with mode='transfer' or 'finetune'
- ⚠️ Not support bundle export with full artifact chain

### Migration Methods

#### Automatic Migration (Recommended)

```python
from nirs4all.database import PredictionsDB
from nirs4all.migration import migrate_predictions

db = PredictionsDB('runs/')
results = migrate_predictions(db, dry_run=False, verbose=1)
print(f"Migrated: {results['migrated']}")
```

#### Migration During Retrain

Old predictions are automatically migrated when used:

```python
runner = PipelineRunner(save_artifacts=True, verbose=0)
new_preds, _ = runner.retrain(
    source=old_prediction,  # Will be migrated automatically
    dataset=new_data,
    mode='full'
)
```

### Checking Migration Status

```python
from nirs4all.database import PredictionsDB

db = PredictionsDB('runs/')
old_format = sum(1 for p in db.all() if 'trace_id' not in p)
new_format = sum(1 for p in db.all() if 'trace_id' in p)

print(f"Old format: {old_format}")
print(f"New format: {new_format}")
```

---

## Troubleshooting

### Common API Migration Issues

#### Wrong Return Type

```python
# ❌ Wrong - will fail
predictions, per_dataset = nirs4all.run(pipeline, data)

# ✅ Correct
result = nirs4all.run(pipeline, data)
predictions = result.predictions
per_dataset = result.per_dataset
```

#### NIRSPipeline is for Prediction Only

```python
# ❌ NIRSPipeline doesn't train
pipe = NIRSPipeline(steps=[MinMaxScaler(), PLSRegression(10)])
pipe.fit(X, y)  # Raises NotImplementedError

# ✅ Train with nirs4all, then wrap
result = nirs4all.run(pipeline, dataset)
pipe = NIRSPipeline.from_result(result)
pipe.predict(X_new)  # Works
```

### Common Dataset Issues

#### "No data source specified"

Your configuration needs at least one of:
- `train_x` or `test_x` (legacy)
- `sources` (new multi-source)
- `variations` (new variations)
- `folder` (auto-scan)

#### "Sample count mismatch across sources"

All sources must have the same number of samples. Check that your data files have consistent row counts.

### Common Prediction Format Issues

#### Missing Model Folder

```python
# Error: Model folder not found
# Old predictions without saved folders cannot be fully migrated
from pathlib import Path
folder = Path(pred['folder'])
if not folder.exists():
    print("Original model folder missing - limited functionality")
```

#### Hash Mismatch

```
ValueError: Content hash mismatch for artifact 0001:3:all
```

**Cause**: Artifact file was modified after saving.
**Solution**: Delete the corrupted artifact and re-run training.

## See Also

- {doc}`dataset_troubleshooting` - Common data loading issues
- {doc}`/getting_started/index` - Installation guide
- {doc}`/reference/cli` - CLI command reference
