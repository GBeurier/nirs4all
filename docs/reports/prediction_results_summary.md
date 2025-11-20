# Summary: PredictionResultsList Implementation

## What Was Implemented

### 1. New Class: `PredictionResultsList`
A specialized list wrapper that extends Python's built-in `list` class with additional functionality:

- **`save(path, filename)`**: Saves all predictions to a structured CSV file with headers containing:
  - Line 1: dataset_name
  - Line 2: model_classname + model_id
  - Line 3: fold_id
  - Line 4: partition
  - Lines 5+: column headers and prediction data

- **`get(prediction_id)`**: Fast retrieval of predictions by their unique ID

- Maintains full compatibility with standard list operations (indexing, slicing, iteration, etc.)

### 2. Enhanced `PredictionResult` Class
Added new method:

- **`summary()`**: Generates a formatted tab report showing metrics and statistics for all partitions (train/val/test)
  - Uses the existing `TabReportManager` for consistent formatting
  - Works with both aggregated (multiple partitions) and non-aggregated results

### 3. Updated Return Types
Modified `Predictions` class methods to return `PredictionResultsList`:

- **`top()`**: Now returns `PredictionResultsList` instead of plain list
- **`top_k()`**: Now returns `PredictionResultsList` instead of plain list

All 10 return statements across both methods were updated to wrap results in `PredictionResultsList`.

## Key Design Decisions

1. **Code Reuse**:
   - Leveraged existing `TabReportManager` for summary generation
   - Used existing `save_to_csv()` method as reference for CSV structure
   - No duplication of metric calculation logic

2. **Backwards Compatibility**:
   - `PredictionResultsList` extends `list`, so all existing code treating results as lists will continue to work
   - Added type hints with `# type: ignore` where needed to avoid type checker warnings

3. **Flexible CSV Structure**:
   - Adapts to aggregated vs non-aggregated results
   - Handles multiple folds and partitions automatically
   - Uses consistent column naming (e.g., `y_true_train_fold0`, `y_pred_val_fold1`)

4. **Clean API**:
   - Methods are intuitive and follow Python conventions
   - Property accessors for common fields (id, dataset_name, fold_id, etc.)
   - Consistent with existing codebase style

## Files Modified

1. **`nirs4all/dataset/predictions.py`**:
   - Added `PredictionResult.summary()` method
   - Added `PredictionResultsList` class
   - Updated all return statements in `top()` and `top_k()` methods
   - Added imports: `csv`, `io`

2. **`nirs4all/dataset/__init__.py`**:
   - Exported `PredictionResult` and `PredictionResultsList` classes

## Testing

Created comprehensive test file: `examples/generated/test_prediction_results_list.py`

Tests verify:
- ✅ `top()` returns `PredictionResultsList`
- ✅ `top_k()` returns `PredictionResultsList`
- ✅ `save()` method creates properly structured CSV
- ✅ `get()` method retrieves predictions by ID
- ✅ `summary()` method generates formatted reports
- ✅ Standard list operations work correctly

All tests passed successfully!

## Usage Example

```python
from nirs4all.data import Predictions

# Load predictions
predictions = Predictions.load(dataset_name="my_data")

# Get top 5 models (returns PredictionResultsList)
top_models = predictions.top(n=5, rank_metric="mse", aggregate_partitions=True)

# Save all to CSV
top_models.save(path="results", filename="top_5.csv")

# Get specific prediction
best = top_models.get(top_models[0].id)

# Print summary
print(best.summary())

# Standard list operations still work
for model in top_models:
    print(f"{model.model_name}: {model.get('rank_score')}")
```

## Documentation

Created comprehensive documentation: `docs/PREDICTION_RESULTS_LIST.md`

Includes:
- Overview and key features
- Usage examples
- API reference
- Complete workflow examples
- Notes on aggregated vs non-aggregated results
- CSV file structure documentation

## Benefits

1. **Convenience**: Single method call to save all predictions instead of manual iteration
2. **Fast Access**: `get()` method for quick ID-based retrieval
3. **Summary Reports**: Instant tab reports without manual calculation
4. **Consistency**: Uses existing TabReportManager for standardized formatting
5. **Flexibility**: Works with both aggregated and non-aggregated results
6. **Backward Compatible**: Existing code continues to work without changes
