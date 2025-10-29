# Quick Reference: PredictionResultsList

## New Features Overview

### 1. PredictionResultsList Class
Wrapper for lists of predictions returned by `top()` method.

```python
# Get top predictions (now returns PredictionResultsList)
top_models = predictions.top(n=5, rank_metric="mse", aggregate_partitions=True)
top_k_models = predictions.top(n=5, rank_metric="rmse")
```

### 2. Save All Predictions to CSV

```python
# Save with auto-generated filename
top_models.save(path="results")

# Save with custom filename
top_models.save(path="results", filename="my_top_models.csv")
```

**CSV Structure:**
```csv
dataset_name
model_classname_id
fold_id
partition
y_true_train_fold0,y_pred_train_fold0,y_true_val_fold0,y_pred_val_fold0,y_true_test,y_pred_test
0.5,0.52,0.6,0.58,0.55,0.54
...
```

### 3. Get Prediction by ID

```python
# Get prediction by ID
prediction = top_models.get("abc123")

if prediction:
    print(f"Found: {prediction.model_name}")
```

### 4. Print Summary Report

```python
# Print formatted summary for a prediction
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

### 5. Standard List Operations

```python
# All standard list operations work
len(top_models)           # Length
top_models[0]             # Indexing
top_models[:3]            # Slicing
for model in top_models:  # Iteration
    ...
```

## Common Workflows

### Workflow 1: Analyze Top Models

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

### Workflow 2: Export Best Model Details

```python
# Get best model
best = predictions.top(n=1, rank_metric="rmse")[0]

# Print summary
print("BEST MODEL PERFORMANCE:")
print(best.summary())

# Save individual prediction
best.save_to_csv(force_path="results/best_model.csv")

# Access details
print(f"Model: {best.model_name}")
print(f"Dataset: {best.dataset_name}")
print(f"Fold: {best.fold_id}")
print(f"Score: {best.get('rank_score')}")
```

### Workflow 3: Compare Multiple Models

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

## Method Signatures

### PredictionResultsList

```python
class PredictionResultsList(list):
    def save(self, path: str = "results", filename: Optional[str] = None) -> None
    def get(self, prediction_id: str) -> Optional[PredictionResult]
```

### PredictionResult

```python
class PredictionResult(dict):
    def summary(self) -> str
    def save_to_csv(self, path: str = "results", force_path: Optional[str] = None) -> None
    def eval_score(self, metrics: Optional[List[str]] = None) -> Dict[str, Any]

    # Properties
    @property
    def id(self) -> str
    @property
    def dataset_name(self) -> str
    @property
    def model_name(self) -> str
    @property
    def fold_id(self) -> str
    @property
    def config_name(self) -> str
```

## Key Points

1. ✅ **Backward Compatible**: All existing code continues to work
2. ✅ **List Compatible**: Standard list operations work normally
3. ✅ **Flexible**: Works with aggregated and non-aggregated results
4. ✅ **Reuses Code**: Leverages existing TabReportManager
5. ✅ **Type Safe**: Properly typed with Union types

## Examples

See `examples/generated/test_prediction_results_list.py` for complete working examples.
