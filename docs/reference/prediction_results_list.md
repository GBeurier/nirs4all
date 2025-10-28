# PredictionResultsList Documentation

## Overview

The `PredictionResultsList` class is a specialized list container that wraps lists of `PredictionResult` objects returned by the `top()` and `top_k()` methods of the `Predictions` class. It provides additional functionality while maintaining full compatibility with standard Python list operations.

## Key Features

### 1. Extended Functionality
- **`save(path, filename)`**: Save all predictions to a single structured CSV file
- **`get(id)`**: Fast retrieval of predictions by their unique ID
- Standard list operations: indexing, slicing, iteration, length, etc.

### 2. Enhanced PredictionResult
- **`summary()`**: Generate a formatted tab report with metrics for train/val/test partitions
- **`save_to_csv(path, force_path)`**: Save individual prediction to CSV (existing functionality)
- **`eval_score(metrics)`**: Calculate metrics for the prediction (existing functionality)

## Usage Examples

### Basic Usage

```python
from nirs4all.dataset import Predictions

# Create or load predictions
predictions = Predictions()

# Get top 5 models using top() method
top_models = predictions.top(
    n=5,
    rank_metric="mse",
    rank_partition="val",
    display_partition="test",
    aggregate_partitions=True
)

# Type: PredictionResultsList (extends list)
print(type(top_models))  # <class 'PredictionResultsList'>
print(len(top_models))   # 5
```

### Saving to CSV

The `save()` method creates a structured CSV with the following format:

```csv
Line 1: dataset_name
Line 2: model_classname + model_id
Line 3: fold_id
Line 4: partition
Line 5: column headers (y_true_partition, y_pred_partition, ...)
Lines 6+: prediction data
```

**Example:**

```python
# Save all predictions to a single CSV
top_models.save(
    path="results",
    filename="top_5_models.csv"
)
```

For aggregated results (multiple partitions), the CSV will have columns like:
- `y_true_train_fold0`, `y_pred_train_fold0`
- `y_true_val_fold0`, `y_pred_val_fold0`
- `y_true_test`, `y_pred_test`

### Getting Predictions by ID

```python
# Get the ID of the first prediction
first_id = top_models[0].id

# Retrieve the prediction by ID
prediction = top_models.get(first_id)

if prediction:
    print(f"Model: {prediction.model_name}")
    print(f"Dataset: {prediction.dataset_name}")
    print(f"Fold: {prediction.fold_id}")
```

### Generating Summary Reports

The `summary()` method generates a formatted tab report showing metrics for all partitions:

```python
# Get summary for the best model
best_model = top_models[0]
print(best_model.summary())
```

**Output Example:**

```
|----------|---------|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-------------|
|          | Nsample | Nfeature | Mean   | Median | Min    | Max    | SD     | CV     | R²     | RMSE   | MSE    | SEP    | MAE    | RPD    | Bias   | Consistency |
|----------|---------|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-------------|
| Cros Val | 50      | 100      | 0.069  | 0.162  | -2.405 | 2.417  | 1.057  | 15.210 | 0.966  | 0.195  | 0.038  | 0.192  | 0.160  | 5.49   | -0.031 | 100.0       |
| Train    | 50      | 100      | -0.119 | -0.163 | -2.477 | 2.546  | 0.977  | -8.186 | 0.944  | 0.231  | 0.053  | 0.230  | 0.191  | 4.24   | -0.018 | 100.0       |
| Test     | 50      | 100      | -0.104 | -0.118 | -2.182 | 1.937  | 0.907  | -8.697 | 0.962  | 0.176  | 0.031  | 0.169  | 0.141  | 5.36   | -0.049 | 100.0       |
|----------|---------|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-------------|
```

### Standard List Operations

PredictionResultsList maintains full compatibility with standard list operations:

```python
# Length
print(len(top_models))

# Indexing
first = top_models[0]
last = top_models[-1]

# Slicing
top_3 = top_models[:3]

# Iteration
for prediction in top_models:
    print(f"{prediction.model_name}: {prediction.get('rank_score')}")

# List comprehension
model_names = [p.model_name for p in top_models]

# Membership testing
if some_prediction in top_models:
    print("Found!")
```

## Complete Workflow Example

```python
from nirs4all.dataset import Predictions

# Load existing predictions
predictions = Predictions.load(
    dataset_name="my_dataset",
    path="results"
)

# Get top 10 models ranked by MSE on validation set
top_models = predictions.top(
    n=10,
    rank_metric="mse",
    rank_partition="val",
    display_partition="test",
    aggregate_partitions=True,  # Include train/val/test data
    ascending=True  # Lower MSE is better
)

# Save all predictions to CSV
top_models.save(
    path="results/analysis",
    filename="top_10_models.csv"
)

# Print summary for best model
print("=" * 80)
print("BEST MODEL SUMMARY")
print("=" * 80)
print(top_models[0].summary())

# Access specific prediction by ID
best_id = top_models[0].id
best_prediction = top_models.get(best_id)

# Iterate through predictions
for i, prediction in enumerate(top_models, 1):
    print(f"\n{i}. {prediction.model_name} (ID: {prediction.id})")
    print(f"   Fold: {prediction.fold_id}")
    print(f"   Rank Score: {prediction.get('rank_score'):.4f}")

    # Save individual prediction
    prediction.save_to_csv(
        path="results/individual",
        force_path=f"results/individual/model_{i}.csv"
    )
```

## API Reference

### PredictionResultsList

**Methods:**

- `__init__(predictions=None)`: Initialize with optional list of predictions
- `save(path="results", filename=None)`: Save all predictions to structured CSV
- `get(prediction_id)`: Retrieve prediction by ID (returns `PredictionResult` or `None`)
- All standard list methods: `append()`, `extend()`, `pop()`, `remove()`, etc.

**Properties:**

- `len(results)`: Number of predictions
- `results[i]`: Index access
- `results[i:j]`: Slice access
- `for pred in results:`: Iteration support

### PredictionResult

**New Method:**

- `summary()`: Generate formatted tab report with metrics

**Existing Methods:**

- `save_to_csv(path="results", force_path=None)`: Save to individual CSV file
- `eval_score(metrics=None)`: Calculate metrics for this prediction

**Properties:**

- `id`: Unique prediction identifier
- `dataset_name`: Dataset name
- `model_name`: Model name
- `model_classname`: Model class name
- `fold_id`: Cross-validation fold ID
- `config_name`: Configuration name
- `step_idx`: Pipeline step index
- `op_counter`: Operation counter

## Notes

### Aggregated vs Non-Aggregated Results

**Aggregated results** (when `aggregate_partitions=True`):
- Contains nested dictionaries for `train`, `val`, `test` partitions
- Each partition has `y_true`, `y_pred`, and score fields
- Summary shows metrics for all partitions

**Non-aggregated results** (single partition):
- Contains `y_true`, `y_pred` at the root level
- Summary shows metrics for that partition only

### CSV File Structure

The CSV structure adapts based on aggregation:

**With aggregation:**
```csv
dataset_name
model_classname_id
fold_id
partition
y_true_train_foldX,y_pred_train_foldX,y_true_val_foldX,y_pred_val_foldX,y_true_test,y_pred_test
0.5,0.52,0.6,0.58,0.55,0.54
...
```

**Without aggregation:**
```csv
dataset_name
model_classname_id
fold_id
partition
y_true,y_pred
0.5,0.52
...
```

## Implementation Details

- **Type:** `PredictionResultsList` extends Python's built-in `list` class
- **Compatibility:** Fully compatible with all list operations and duck typing
- **Performance:** `get()` method uses linear search (O(n)), suitable for small result sets
- **Dependencies:** Uses `TabReportManager` for summary generation
- **Return Type:** Both `top()` and `top_k()` now return `PredictionResultsList` instead of plain list

## See Also

- [Predictions API](./predictions.md)
- [Tab Report Manager](./tab_report_manager.md)
- [Evaluator Module](./evaluator.md)
