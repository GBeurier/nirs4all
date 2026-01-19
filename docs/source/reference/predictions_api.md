# PredictionResultsList Reference

The `PredictionResultsList` class is a specialized list container that wraps lists of `PredictionResult` objects returned by the `top()` method of the `Predictions` class. It provides additional functionality while maintaining full compatibility with standard Python list operations.

## Quick Reference

### Get Top Predictions

```python
# Get top predictions (returns PredictionResultsList)
top_models = predictions.top(n=5, rank_metric="mse", aggregate_partitions=True)
```

### Save All Predictions to CSV

```python
top_models.save(path="results", filename="top_5_models.csv")
```

### Get Prediction by ID

```python
prediction = top_models.get("abc123")
if prediction:
    print(f"Found: {prediction.model_name}")
```

### Print Summary Report

```python
print(top_models[0].summary())
```

**Output:**
```
|----------|---------|----------|--------|--------|--------|--------|
|          | Nsample | Nfeature | R²     | RMSE   | MSE    | MAE    |
|----------|---------|----------|--------|--------|--------|--------|
| Cros Val | 50      | 100      | 0.966  | 0.195  | 0.038  | 0.160  |
| Train    | 50      | 100      | 0.944  | 0.231  | 0.053  | 0.191  |
| Test     | 50      | 100      | 0.962  | 0.176  | 0.031  | 0.141  |
|----------|---------|----------|--------|--------|--------|--------|
```

### Standard List Operations

```python
len(top_models)           # Length
top_models[0]             # Indexing
top_models[:3]            # Slicing
for model in top_models:  # Iteration
    ...
```

## Key Features

### Extended Functionality

- **`save(path, filename)`**: Save all predictions to a single structured CSV file
- **`get(id)`**: Fast retrieval of predictions by their unique ID
- Standard list operations: indexing, slicing, iteration, length, etc.

### Enhanced PredictionResult

- **`summary()`**: Generate a formatted tab report with metrics for train/val/test partitions
- **`save_to_csv(path_or_file, filename)`**: Save individual prediction to CSV
- **`eval_score(metrics)`**: Calculate metrics for the prediction

## Usage Examples

### Basic Usage

```python
from nirs4all.data import Predictions

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

The `save()` method creates a structured CSV:

```text
Line 1: dataset_name
Line 2: model_classname + model_id
Line 3: fold_id
Line 4: partition
Line 5: column headers (y_true_partition, y_pred_partition, ...)
Lines 6+: prediction data
```

**Example:**

```python
top_models.save(
    path="results",
    filename="top_5_models.csv"
)
```

For aggregated results, the CSV has columns like:
- `y_true_train_fold0`, `y_pred_train_fold0`
- `y_true_val_fold0`, `y_pred_val_fold0`
- `y_true_test`, `y_pred_test`

### Common Workflows

#### Analyze Top Models

```python
# Get top 10 models
top_10 = predictions.top(
    n=10,
    rank_metric="mse",
    aggregate_partitions=True
)

# Save all to CSV
top_10.save(path="results/analysis")

# Print summaries
for i, model in enumerate(top_10, 1):
    print(f"\n{'='*80}")
    print(f"MODEL {i}: {model.model_name} (ID: {model.id})")
    print(f"{'='*80}")
    print(model.summary())
```

#### Export Best Model Details

```python
# Get best model
best = predictions.top(n=1, rank_metric="rmse")[0]

# Print summary
print("BEST MODEL PERFORMANCE:")
print(best.summary())

# Save individual prediction
best.save_to_csv("results/best_model.csv")

# Access details
print(f"Model: {best.model_name}")
print(f"Dataset: {best.dataset_name}")
print(f"Fold: {best.fold_id}")
print(f"Score: {best.get('rank_score')}")
```

#### Compare Multiple Models

```python
# Get top 5 models
top_5 = predictions.top(n=5, rank_metric="r2", ascending=False)

# Save all predictions to single file
top_5.save(filename="top_5_comparison.csv")

# Compare metrics
for model in top_5:
    scores = model.eval_score(metrics=["rmse", "mae", "r2"])
    print(f"{model.model_name}: {scores}")
```

#### Group By: Top N Per Group

The `group_by` parameter allows you to get top N results **per group** instead of N total.
This is useful when comparing models across multiple datasets or configurations.

```python
# Get top 3 models PER DATASET (flat list, sorted by global rank)
top_per_dataset = predictions.top(
    n=3,
    rank_metric="rmse",
    group_by="dataset_name"
)

# Each result includes 'group_key' for easy filtering
for pred in top_per_dataset:
    dataset = pred['group_key'][0]  # group_key is a tuple
    print(f"{dataset}: {pred.model_name} - RMSE: {pred.get('rmse', 0):.4f}")

# Filter results for a specific dataset
wheat_results = [r for r in top_per_dataset if r['group_key'] == ('wheat',)]
```

**Grouped dict output** with `return_grouped=True`:

```python
# Get top 3 models per dataset as a dictionary
grouped = predictions.top(
    n=3,
    rank_metric="rmse",
    group_by="dataset_name",
    return_grouped=True
)

# Result: {('dataset1',): [...], ('dataset2',): [...]}
for group_key, results in grouped.items():
    print(f"\n{group_key[0]}: {len(results)} best models")
    for i, pred in enumerate(results, 1):
        print(f"  {i}. {pred.model_name}: RMSE={pred.get('rmse', 0):.4f}")
```

**Multi-column grouping**:

```python
# Top 2 per (dataset, model_class) combination
per_combo = predictions.top(
    n=2,
    rank_metric="rmse",
    group_by=["dataset_name", "model_classname"]
)
# Each result has group_key like ('wheat', 'PLSRegression')
```

## Complete Workflow Example

```python
from nirs4all.data import Predictions

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
    prediction.save_to_csv(f"results/individual/model_{i}.csv")
```

## API Reference

### PredictionResultsList

```python
class PredictionResultsList(list):
    def save(self, path: str = "results", filename: Optional[str] = None) -> None
    def get(self, prediction_id: str) -> Optional[PredictionResult]
```

**Methods:**

- `__init__(predictions=None)`: Initialize with optional list of predictions
- `save(path="results", filename=None)`: Save all predictions to structured CSV
- `get(prediction_id)`: Retrieve prediction by ID (returns `PredictionResult` or `None`)
- All standard list methods: `append()`, `extend()`, `pop()`, `remove()`, etc.

### PredictionResult

```python
class PredictionResult(dict):
    def summary(self) -> str
    def save_to_csv(self, path_or_file: str = "results", filename: Optional[str] = None) -> None
    def eval_score(self, metrics: Optional[List[str]] = None) -> Dict[str, Any]

    @property
    def id(self) -> str
    @property
    def dataset_name(self) -> str
    @property
    def model_name(self) -> str
    @property
    def model_classname(self) -> str
    @property
    def fold_id(self) -> str
    @property
    def config_name(self) -> str
    @property
    def step_idx(self) -> int
    @property
    def op_counter(self) -> int
```

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

**With aggregation:**
```text
dataset_name
model_classname_id
fold_id
partition
y_true_train_foldX,y_pred_train_foldX,y_true_val_foldX,y_pred_val_foldX,y_true_test,y_pred_test
0.5,0.52,0.6,0.58,0.55,0.54
...
```

**Without aggregation:**
```text
dataset_name
model_classname_id
fold_id
partition
y_true,y_pred
0.5,0.52
...
```

### Implementation Details

- **Type:** `PredictionResultsList` extends Python's built-in `list` class
- **Compatibility:** Fully compatible with all list operations and duck typing
- **Performance:** `get()` method uses linear search (O(n)), suitable for small result sets
- **Dependencies:** Uses `TabReportManager` for summary generation
- **Return Type:** `top()` returns `PredictionResultsList` instead of plain list

## Key Points

- ✅ **Backward Compatible**: All existing code continues to work
- ✅ **List Compatible**: Standard list operations work normally
- ✅ **Flexible**: Works with aggregated and non-aggregated results
- ✅ **Type Safe**: Properly typed with Union types

## Prediction Entry Fields

When you call `result.top(n)` or access `result.best`, you get prediction entries (dictionaries) containing metadata about each prediction. Here are the commonly available fields:

### Core Identification Fields

| Field | Description | Example |
|-------|-------------|---------|
| `model_name` | Custom name or auto-generated model identifier | `"PLS-10"` |
| `model_classname` | Class name of the model | `"PLSRegression"` |
| `dataset_name` | Name of the dataset | `"regression"` |
| `fold_id` | Cross-validation fold identifier | `"fold_0"` |
| `preprocessings` | Full preprocessing chain applied | `"SNV\|FirstDerivative"` |

### Score Fields

| Field | Description | Example |
|-------|-------------|---------|
| `train_score` | Score on training set (primary metric) | `0.0234` |
| `val_score` | Score on validation set (primary metric) | `0.0312` |
| `test_score` | Score on test set (primary metric) | `0.0298` |
| `metric` | Name of the primary metric | `"mse"` |

### Data Information Fields

| Field | Description | Example |
|-------|-------------|---------|
| `n_samples` | Number of samples in the dataset | `100` |
| `n_features` | Number of features after preprocessing | `256` |
| `task_type` | Type of task | `"regression"` or `"classification"` |

### Additional Metrics via `display_metrics`

When using `display_metrics` parameter, additional computed metrics are added:

```python
# Request additional metrics
for pred in result.top(n=5, display_metrics=['rmse', 'r2', 'mae']):
    print(f"RMSE: {pred.get('rmse', 0):.4f}")
    print(f"R²: {pred.get('r2', 0):.4f}")
    print(f"MAE: {pred.get('mae', 0):.4f}")
```

### Example: Accessing Preprocessing Chains

A common use case is analyzing which preprocessing combination works best:

```python
# Get the best prediction
best = result.best

# Access the preprocessing chain
print(f"Best preprocessing: {best.get('preprocessings', 'N/A')}")
print(f"Model: {best.get('model_name', 'N/A')}")
print(f"Validation score: {best.get('val_score', 0):.6f}")

# Compare top preprocessing chains
for i, pred in enumerate(result.top(n=5, display_metrics=['rmse', 'r2']), 1):
    preproc = pred.get('preprocessings', 'N/A')
    rmse = pred.get('rmse', 0)
    r2 = pred.get('r2', 0)
    print(f"{i}. {preproc}: RMSE={rmse:.4f}, R²={r2:.4f}")
```

```{tip}
For a comprehensive example of analyzing preprocessing chains with feature augmentation, see the
**U02_feature_augmentation.py** example in `examples/user/03_preprocessing/`. It demonstrates:

- Using feature augmentation to explore preprocessing combinations
- Accessing the best preprocessing chain from results
- Comparing top preprocessing variants with detailed metrics
```

## See Also

- {doc}`/reference/pipeline_syntax` - Pipeline syntax reference
- {doc}`/user_guide/visualization/index` - Visualization and charts
- {doc}`/examples/user/preprocessing` - Preprocessing examples including feature augmentation
