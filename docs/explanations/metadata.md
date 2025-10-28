# Metadata Usage Guide

## Overview

Metadata in `nirs4all` allows you to store and manage auxiliary sample-level information alongside your spectral data. This includes information like:
- Sample identifiers
- Batch numbers
- Collection locations
- Instrument types
- Environmental conditions (temperature, humidity)
- Any other sample-specific attributes

Metadata has **one row per sample**, aligning with features and targets, and can be easily filtered, retrieved, and converted to numeric format for use in machine learning pipelines.

---

## Key Concepts

### What is Metadata?

**Metadata** is distinct from features and targets:
- **Features (X)**: Spectral measurements or input variables for modeling
- **Targets (Y)**: Response variables you want to predict
- **Metadata**: Auxiliary information about each sample

### Metadata vs Features

- **Metadata** is typically:
  - Categorical or mixed-type data
  - Used for grouping, filtering, or as auxiliary features
  - Not transformed through complex preprocessing pipelines

- **Features** are:
  - Numerical spectral data
  - Subject to preprocessing (SNV, MSC, derivatives, etc.)
  - Primary input to models

---

## Loading Metadata

### From Folder (Auto-detection)

Place metadata files in your dataset folder with standard naming patterns:

```
dataset/
├── X_train.csv       # Training spectra
├── Y_train.csv       # Training targets
├── M_train.csv       # Training metadata ✓
├── X_test.csv        # Test spectra
├── Y_test.csv        # Test targets
└── M_test.csv        # Test metadata ✓
```

**Supported naming patterns:**
- `M_train.csv`, `M_test.csv`
- `Mcal.csv`, `Mtest.csv`, `Mval.csv`
- `Meta_train.csv`, `Metatest.csv`
- `metadata_train.csv`, `metadata_test.csv`
- And many more variations...

```python
from nirs4all.data.dataset_config import DatasetConfigs

configs = DatasetConfigs("path/to/dataset/folder")
dataset = configs.get_dataset_at(0)

print(f"Metadata columns: {dataset.metadata_columns}")
# Output: Metadata columns: ['batch', 'location', 'instrument']
```

### From Explicit Configuration

```python
config = {
    'train_x': 'data/X_train.csv',
    'train_y': 'data/Y_train.csv',
    'train_group': 'data/M_train.csv',  # Metadata
    'test_x': 'data/X_test.csv',
    'test_y': 'data/Y_test.csv',
    'test_group': 'data/M_test.csv',    # Metadata
}

configs = DatasetConfigs(config)
dataset = configs.get_dataset_at(0)
```

### Programmatic Addition

```python
import pandas as pd
from nirs4all.data.dataset import SpectroDataset

dataset = SpectroDataset(name="my_dataset")

# Add samples and targets
X_train = ...  # Your spectral data
y_train = ...  # Your targets
dataset.add_samples(X_train, {"partition": "train"})
dataset.add_targets(y_train)

# Add metadata as DataFrame
metadata_df = pd.DataFrame({
    'sample_id': [1, 2, 3, ...],
    'batch': [1, 1, 2, ...],
    'location': ['A', 'A', 'B', ...]
})
dataset.add_metadata(metadata_df)

# Or as numpy array with headers
import numpy as np
metadata_array = np.array([[1, 'A'], [1, 'A'], [2, 'B']], dtype=object)
dataset.add_metadata(metadata_array, headers=['batch', 'location'])
```

---

## Accessing Metadata

### Get All Metadata

```python
# Get all metadata as Polars DataFrame
all_meta = dataset.metadata()
print(all_meta)
```

### Filter by Partition

```python
# Get training metadata only
train_meta = dataset.metadata(selector={"partition": "train"})

# Get test metadata only
test_meta = dataset.metadata(selector={"partition": "test"})
```

### Get Specific Columns

```python
# Get only batch and location columns
batch_location = dataset.metadata(columns=['batch', 'location'])
```

### Get Single Column as Array

```python
# Get batch numbers as numpy array
batch_numbers = dataset.metadata_column('batch')
print(batch_numbers)  # array([1, 1, 2, 2, 3, ...])

# Get batch numbers for training samples only
train_batches = dataset.metadata_column('batch', selector={"partition": "train"})
```

---

## Converting Metadata to Numeric

Metadata is often categorical (strings, labels), but machine learning models require numeric input. Use `metadata_numeric()` to convert:

### Label Encoding

Converts categories to integers (0, 1, 2, ...):

```python
location_encoded, encoding_info = dataset.metadata_numeric(
    'location',
    method='label'
)

print(location_encoded)  # array([0, 0, 1, 1, 2, ...])
print(encoding_info)
# {
#   'method': 'label',
#   'classes': ['A', 'B', 'C', 'D']
# }
```

### One-Hot Encoding

Converts categories to binary vectors:

```python
location_onehot, encoding_info = dataset.metadata_numeric(
    'location',
    method='onehot'
)

print(location_onehot.shape)  # (n_samples, n_categories)
print(location_onehot[:3])
# array([[1, 0, 0, 0],   # Location A
#        [1, 0, 0, 0],   # Location A
#        [0, 1, 0, 0]])  # Location B
```

### Encoding Consistency

Encodings are **cached** to ensure consistency:

```python
# First call creates encoding
encoded1, info1 = dataset.metadata_numeric('instrument', method='label')

# Subsequent calls return the same encoding
encoded2, info2 = dataset.metadata_numeric('instrument', method='label')

assert np.array_equal(encoded1, encoded2)  # True
```

---

## Modifying Metadata

### Update Existing Values

```python
# Update location for first 5 training samples
dataset.update_metadata(
    column='location',
    values=['Updated', 'Updated', 'Updated', 'Updated', 'Updated'],
    selector={"partition": "train"}
)
```

### Add New Column

```python
import numpy as np

# Add quality scores for all samples
quality_scores = np.random.rand(dataset.num_samples)
dataset.add_metadata_column('quality', quality_scores)

print(dataset.metadata_columns)
# ['batch', 'location', 'instrument', 'quality']
```

---

## Using Metadata in Pipelines

### Combine Metadata with Spectral Features

```python
from sklearn.ensemble import RandomForestRegressor

# Get spectral data
X_spectra = dataset.x({"partition": "train"})
y = dataset.y({"partition": "train"})

# Get numeric metadata
instrument_encoded, _ = dataset.metadata_numeric('instrument', method='onehot')
temperature = dataset.metadata_column('temperature', selector={"partition": "train"})

# Combine features
import numpy as np
X_combined = np.hstack([
    X_spectra,
    instrument_encoded,
    temperature.reshape(-1, 1)
])

# Train model
model = RandomForestRegressor()
model.fit(X_combined, y)
```

### Filter Samples by Metadata

```python
# Get metadata for filtering
batch_col = dataset.metadata_column('batch', selector={"partition": "train"})

# Manually filter batch 1 samples
batch_1_mask = (batch_col == 1)

X_batch_1 = X_train[batch_1_mask]
y_batch_1 = y_train[batch_1_mask]
```

---

## Best Practices

### 1. Consistent Naming

Use clear, descriptive column names:
- ✅ `'instrument_id'`, `'batch_number'`, `'collection_date'`
- ❌ `'col1'`, `'x'`, `'data'`

### 2. Keep Metadata Aligned

Metadata must have the same number of rows as samples:

```python
# CORRECT: Same number of rows
dataset.add_samples(X_train, {"partition": "train"})  # 100 samples
metadata_df = pd.DataFrame({...})  # 100 rows
dataset.add_metadata(metadata_df)  # ✅

# INCORRECT: Mismatched rows
dataset.add_samples(X_train, {"partition": "train"})  # 100 samples
metadata_df = pd.DataFrame({...})  # 50 rows
dataset.add_metadata(metadata_df)  # ❌ Error!
```

### 3. Use Appropriate Encoding

- **Label encoding**: For ordinal categories or when number of categories is small
- **One-hot encoding**: For nominal categories, but watch out for high cardinality

```python
# Good: Few categories
instrument_encoded, _ = dataset.metadata_numeric('instrument', method='onehot')
# Result: 3 binary columns for 3 instruments

# Careful: Many categories
sample_id_encoded, _ = dataset.metadata_numeric('sample_id', method='onehot')
# Result: 1000 binary columns for 1000 unique IDs (sparse!)
```

### 4. Cache-Aware Operations

Modifying metadata clears the encoding cache:

```python
# Create encoding
encoded1, _ = dataset.metadata_numeric('location', method='label')

# Update metadata
dataset.update_metadata('location', ['New'], selector={"partition": "train"})

# Encoding is recalculated (cache was cleared)
encoded2, _ = dataset.metadata_numeric('location', method='label')
# encoded2 may differ from encoded1 due to new category 'New'
```

### 5. Documentation

Document your metadata columns:

```python
# Good practice: Document what each column means
metadata_df = pd.DataFrame({
    'batch': batch_numbers,        # Production batch (1-10)
    'instrument': instruments,      # Instrument ID (A, B, C)
    'temp_c': temperatures,         # Collection temperature (°C)
    'operator': operators,          # Lab technician name
})
```

---

## Common Patterns

### Pattern 1: Stratified Splitting by Metadata

```python
from sklearn.model_selection import StratifiedKFold

# Get batch information
batch_col = dataset.metadata_column('batch', selector={"partition": "train"})

# Use for stratified CV
skf = StratifiedKFold(n_splits=5)
for train_idx, val_idx in skf.split(X_train, batch_col):
    X_train_fold = X_train[train_idx]
    X_val_fold = X_train[val_idx]
    # Train and validate...
```

### Pattern 2: Cross-Instrument Validation

```python
# Get training data and metadata
X_train = dataset.x({"partition": "train"})
y_train = dataset.y({"partition": "train"})
instruments = dataset.metadata_column('instrument', selector={"partition": "train"})

# Train on instrument A, test on instrument B
mask_A = (instruments == 'A')
mask_B = (instruments == 'B')

model.fit(X_train[mask_A], y_train[mask_A])
score = model.score(X_train[mask_B], y_train[mask_B])
print(f"Cross-instrument R²: {score:.3f}")
```

### Pattern 3: Temporal Splits

```python
# Assuming metadata has 'date' column
dates = dataset.metadata_column('date', selector={"partition": "train"})

# Sort by date
sorted_indices = np.argsort(dates)
split_point = int(len(dates) * 0.8)

train_idx = sorted_indices[:split_point]
val_idx = sorted_indices[split_point:]

# Temporal train/validation split
X_train_temporal = X_train[train_idx]
X_val_temporal = X_train[val_idx]
```

---

## API Reference

### Dataset Methods

#### `add_metadata(data, headers=None)`
Add metadata rows.

**Parameters:**
- `data`: 2D array, pandas DataFrame, or polars DataFrame
- `headers`: Column names (required if data is ndarray)

#### `metadata(selector=None, columns=None)`
Get metadata as DataFrame.

**Parameters:**
- `selector`: Filter dict (e.g., `{"partition": "train"}`)
- `columns`: List of column names to return

**Returns:** Polars DataFrame

#### `metadata_column(column, selector=None)`
Get single metadata column as array.

**Parameters:**
- `column`: Column name
- `selector`: Filter dict

**Returns:** Numpy array

#### `metadata_numeric(column, selector=None, method='label')`
Convert metadata column to numeric.

**Parameters:**
- `column`: Column name
- `selector`: Filter dict
- `method`: `'label'` or `'onehot'`

**Returns:** Tuple of (numeric_array, encoding_info)

#### `update_metadata(column, values, selector=None)`
Update metadata values.

**Parameters:**
- `column`: Column name
- `values`: New values
- `selector`: Filter dict

#### `add_metadata_column(column, values)`
Add new metadata column.

**Parameters:**
- `column`: Column name
- `values`: Column values (must match number of samples)

#### `metadata_columns`
Property returning list of metadata column names.

---

## Troubleshooting

### Issue: Metadata not loading

**Check:**
1. File naming matches patterns (M_train, Mcal, metadata_train, etc.)
2. Files are in the same folder as X and Y files
3. CSV format is correct (check delimiter, headers)

```python
# Debug: Check what files were detected
from nirs4all.data.dataset_config_parser import browse_folder
config = browse_folder("path/to/folder")
print(config.get('train_group'))  # Should show metadata file path
```

### Issue: Row count mismatch

**Error:** `ValueError: Row count mismatch: X(100) Metadata(50)`

**Solution:** Ensure metadata has the same number of rows as X data:

```python
print(f"X rows: {len(X_train)}")
print(f"Metadata rows: {len(metadata_df)}")
# They must match!
```

### Issue: Missing metadata columns

**Error:** `ValueError: Column 'instrument' not found`

**Solution:** Check column names:

```python
print(dataset.metadata_columns)  # See what's available
```

---

## Examples

See `examples/metadata_usage.py` for complete working examples including:
- Loading datasets with metadata
- Filtering and accessing metadata
- Numeric encoding
- Using metadata in pipelines
- Cross-instrument validation

---

## Summary

Metadata in `nirs4all` provides a flexible way to:
- ✅ Store auxiliary sample information
- ✅ Filter and group samples
- ✅ Enhance models with contextual features
- ✅ Enable stratified validation strategies
- ✅ Support reproducible data management

Start by loading your metadata files alongside X and Y data, then explore the rich API for accessing, converting, and utilizing metadata in your spectroscopy workflows!
