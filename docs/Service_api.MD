# NIRS4All Service API (Alternative Design)

This document proposes an alternative, cleaner, and more uniform external API for nirs4all designed specifically for service integration. Instead of exposing internal classes directly, this API provides a consistent set of functions that services can call to control nirs4all operations.

## Design Principles

- **Uniform Interface**: All functions follow consistent naming and parameter patterns
- **Resource-Based**: Operations on datasets, pipelines, runs, and predictions as resources
- **Idempotent Operations**: Safe to retry operations
- **Clear Error Handling**: Structured error responses
- **Async-Friendly**: Designed for background processing
- **Minimal Dependencies**: Services only need to know function signatures

## Core Concepts

- **Resources**: Datasets, Pipelines, Runs, Predictions identified by UUIDs
- **Operations**: CRUD operations on resources plus specialized actions
- **Async Pattern**: Long operations return job IDs, status can be polled
- **Configuration**: All complex setup via configuration dictionaries

## API Functions

### Dataset Management

#### `create_dataset(config: Dict[str, Any], name: str = None) -> Dict[str, Any]`

Create a new dataset from configuration.

**Parameters:**
- `config`: Dataset configuration (paths, formats, options)
- `name`: Optional human-readable name

**Returns:**
```json
{
  "dataset_id": "uuid",
  "status": "created",
  "metadata": {
    "name": "string",
    "num_samples": 1000,
    "num_features": 500,
    "task_type": "regression",
    "created_at": "2025-10-11T10:00:00Z"
  }
}
```

**Example:**
```python
# Create from CSV files
dataset = create_dataset({
  "type": "csv",
  "train": {
    "X": "data/Xcal.csv",
    "y": "data/Ycal.csv"
  },
  "test": {
    "X": "data/Xval.csv",
    "y": "data/Yval.csv"
  }
}, name="My Dataset")

# Create from directory
dataset = create_dataset({
  "type": "directory",
  "path": "sample_data/regression"
}, name="Regression Dataset")
```

#### `get_dataset(dataset_id: str) -> Dict[str, Any]`

Retrieve dataset metadata and basic information.

**Parameters:**
- `dataset_id`: Dataset UUID

**Returns:**
```json
{
  "dataset_id": "uuid",
  "name": "string",
  "metadata": {
    "num_samples": 1000,
    "num_features": 500,
    "task_type": "regression",
    "feature_names": ["X1000", "X1001", ...],
    "created_at": "2025-10-11T10:00:00Z",
    "updated_at": "2025-10-11T10:00:00Z"
  },
  "status": "ready"
}
```

#### `list_datasets(filters: Dict[str, Any] = None) -> List[Dict[str, Any]]`

List available datasets with optional filtering.

**Parameters:**
- `filters`: Optional filters (name, task_type, etc.)

**Returns:**
```json
[
  {
    "dataset_id": "uuid1",
    "name": "Dataset 1",
    "task_type": "regression",
    "num_samples": 1000,
    "created_at": "2025-10-11T10:00:00Z"
  },
  {
    "dataset_id": "uuid2",
    "name": "Dataset 2",
    "task_type": "classification",
    "num_samples": 500,
    "created_at": "2025-10-11T10:30:00Z"
  }
]
```

#### `get_dataset_samples(dataset_id: str, partition: str = "train", start: int = 0, limit: int = 100) -> Dict[str, Any]`

Retrieve dataset samples for inspection.

**Parameters:**
- `dataset_id`: Dataset UUID
- `partition`: "train", "test", or "val"
- `start`: Starting index
- `limit`: Maximum samples to return

**Returns:**
```json
{
  "dataset_id": "uuid",
  "partition": "train",
  "total_samples": 1000,
  "returned_samples": 100,
  "features": [[0.1, 0.2, ...], [0.15, 0.25, ...], ...],
  "targets": [1.2, 3.4, ...],
  "feature_names": ["X1000", "X1001", ...]
}
```

#### `delete_dataset(dataset_id: str) -> Dict[str, Any]`

Delete a dataset and associated resources.

**Parameters:**
- `dataset_id`: Dataset UUID

**Returns:**
```json
{
  "dataset_id": "uuid",
  "status": "deleted"
}
```

### Pipeline Management

#### `create_pipeline(config: Dict[str, Any], name: str = None) -> Dict[str, Any]`

Create a new pipeline configuration.

**Parameters:**
- `config`: Pipeline configuration
- `name`: Optional human-readable name

**Returns:**
```json
{
  "pipeline_id": "uuid",
  "status": "created",
  "metadata": {
    "name": "string",
    "steps": 5,
    "has_models": true,
    "created_at": "2025-10-11T10:00:00Z"
  }
}
```

**Example:**
```python
pipeline = create_pipeline({
  "steps": [
    {
      "type": "preprocessor",
      "name": "MinMaxScaler",
      "params": {"feature_range": [0, 1]}
    },
    {
      "type": "preprocessor",
      "name": "StandardNormalVariate"
    },
    {
      "type": "cross_validator",
      "name": "ShuffleSplit",
      "params": {"n_splits": 3, "test_size": 0.25}
    },
    {
      "type": "model",
      "name": "PLSRegression",
      "params": {"n_components": 5}
    }
  ]
}, name="Basic PLS Pipeline")
```

#### `get_pipeline(pipeline_id: str) -> Dict[str, Any]`

Retrieve pipeline configuration and metadata.

#### `list_pipelines(filters: Dict[str, Any] = None) -> List[Dict[str, Any]]`

List available pipelines.

#### `update_pipeline(pipeline_id: str, config: Dict[str, Any]) -> Dict[str, Any]`

Update pipeline configuration.

#### `delete_pipeline(pipeline_id: str) -> Dict[str, Any]`

Delete a pipeline.

### Execution Management

#### `run_pipeline(pipeline_id: str, dataset_id: str, config: Dict[str, Any] = None) -> Dict[str, Any]`

Execute a pipeline on a dataset asynchronously.

**Parameters:**
- `pipeline_id`: Pipeline UUID
- `dataset_id`: Dataset UUID
- `config`: Optional execution configuration

**Returns:**
```json
{
  "run_id": "uuid",
  "status": "queued",
  "pipeline_id": "uuid",
  "dataset_id": "uuid",
  "started_at": "2025-10-11T10:00:00Z"
}
```

**Example:**
```python
run = run_pipeline(
  pipeline_id="pipeline-uuid",
  dataset_id="dataset-uuid",
  config={
    "save_files": true,
    "verbose": 1,
    "random_state": 42
  }
)
```

#### `get_run_status(run_id: str) -> Dict[str, Any]`

Get execution status and progress.

**Parameters:**
- `run_id`: Run UUID

**Returns:**
```json
{
  "run_id": "uuid",
  "status": "running",  // "queued", "running", "completed", "failed"
  "progress": {
    "current_step": 3,
    "total_steps": 5,
    "current_fold": 2,
    "total_folds": 3,
    "message": "Training PLS model..."
  },
  "started_at": "2025-10-11T10:00:00Z",
  "updated_at": "2025-10-11T10:05:00Z",
  "estimated_completion": "2025-10-11T10:15:00Z"
}
```

#### `cancel_run(run_id: str) -> Dict[str, Any]`

Cancel a running execution.

#### `get_run_results(run_id: str) -> Dict[str, Any]`

Get complete results from a finished run.

**Returns:**
```json
{
  "run_id": "uuid",
  "status": "completed",
  "predictions_id": "uuid",
  "metrics": {
    "best_model": {
      "model_name": "PLSRegression_5",
      "test_rmse": 0.123,
      "val_rmse": 0.145
    },
    "total_predictions": 15
  },
  "completed_at": "2025-10-11T10:10:00Z"
}
```

### Predictions Management

#### `get_predictions(predictions_id: str, filters: Dict[str, Any] = None) -> Dict[str, Any]`

Retrieve predictions with filtering.

**Parameters:**
- `predictions_id`: Predictions UUID
- `filters`: Optional filters (partition, model_name, etc.)

**Returns:**
```json
{
  "predictions_id": "uuid",
  "total_predictions": 15,
  "predictions": [
    {
      "id": "pred-uuid-1",
      "model_name": "PLSRegression_5",
      "partition": "test",
      "fold_id": 0,
      "metrics": {
        "rmse": 0.123,
        "r2": 0.945
      },
      "y_true": [1.2, 3.4, ...],
      "y_pred": [1.1, 3.5, ...]
    }
  ]
}
```

#### `list_predictions(filters: Dict[str, Any] = None) -> List[Dict[str, Any]]`

List available prediction sets.

#### `export_predictions(predictions_id: str, format: str = "csv", filters: Dict[str, Any] = None) -> Dict[str, Any]`

Export predictions to file.

**Parameters:**
- `predictions_id`: Predictions UUID
- `format`: Export format ("csv", "json", "parquet")
- `filters`: Optional filters

**Returns:**
```json
{
  "export_id": "uuid",
  "format": "csv",
  "file_path": "/exports/predictions_2025-10-11.csv",
  "num_rows": 1000,
  "created_at": "2025-10-11T10:00:00Z"
}
```

### Analysis and Visualization

#### `create_analysis(predictions_id: str, config: Dict[str, Any]) -> Dict[str, Any]`

Create an analysis task for predictions.

**Parameters:**
- `predictions_id`: Predictions UUID
- `config`: Analysis configuration

**Returns:**
```json
{
  "analysis_id": "uuid",
  "status": "queued",
  "type": "comparison_plot",
  "predictions_id": "uuid"
}
```

**Example:**
```python
# Create comparison plot
analysis = create_analysis("predictions-uuid", {
  "type": "top_k_comparison",
  "params": {
    "k": 5,
    "metric": "rmse",
    "partition": "test"
  }
})

# Create heatmap
analysis = create_analysis("predictions-uuid", {
  "type": "heatmap",
  "params": {
    "x_var": "model_name",
    "y_var": "preprocessings",
    "metric": "rmse"
  }
})
```

#### `get_analysis_status(analysis_id: str) -> Dict[str, Any]`

Get analysis task status.

#### `get_analysis_result(analysis_id: str) -> Dict[str, Any]`

Get analysis results (plots, statistics, etc.).

**Returns:**
```json
{
  "analysis_id": "uuid",
  "status": "completed",
  "type": "comparison_plot",
  "result": {
    "plot_url": "/plots/analysis_123.png",
    "statistics": {
      "best_model": "PLSRegression_5",
      "mean_rmse": 0.145,
      "std_rmse": 0.023
    }
  }
}
```

### Prediction on New Data

#### `predict_with_model(model_id: str, dataset_id: str, config: Dict[str, Any] = None) -> Dict[str, Any]`

Make predictions on new data using a trained model.

**Parameters:**
- `model_id`: Model identifier (from predictions)
- `dataset_id`: Dataset UUID for prediction
- `config`: Optional prediction configuration

**Returns:**
```json
{
  "prediction_id": "uuid",
  "status": "queued",
  "model_id": "model-uuid",
  "dataset_id": "dataset-uuid"
}
```

#### `get_prediction_results(prediction_id: str) -> Dict[str, Any]`

Get prediction results.

**Returns:**
```json
{
  "prediction_id": "uuid",
  "status": "completed",
  "predictions": [1.23, 4.56, ...],
  "metadata": {
    "num_samples": 100,
    "model_name": "PLSRegression_5"
  }
}
```

### SHAP Analysis

#### `create_shap_analysis(model_id: str, dataset_id: str, config: Dict[str, Any] = None) -> Dict[str, Any]`

Create SHAP explanation analysis.

**Parameters:**
- `model_id`: Model identifier
- `dataset_id`: Dataset UUID
- `config`: SHAP configuration

**Returns:**
```json
{
  "shap_id": "uuid",
  "status": "queued",
  "model_id": "model-uuid",
  "dataset_id": "dataset-uuid"
}
```

#### `get_shap_results(shap_id: str) -> Dict[str, Any]`

Get SHAP analysis results.

### System Management

#### `get_system_status() -> Dict[str, Any]`

Get system status and capabilities.

**Returns:**
```json
{
  "version": "0.2.1",
  "backends": {
    "tensorflow": true,
    "pytorch": false,
    "gpu": true
  },
  "resources": {
    "available_memory": "8GB",
    "cpu_cores": 8,
    "gpu_memory": "4GB"
  },
  "active_runs": 2,
  "queued_runs": 5
}
```

#### `get_available_transformations() -> List[Dict[str, Any]]`

Get list of available preprocessing transformations.

#### `get_available_models() -> List[Dict[str, Any]]`

Get list of available ML models.

#### `validate_config(resource_type: str, config: Dict[str, Any]) -> Dict[str, Any]`

Validate configuration for datasets, pipelines, etc.

**Parameters:**
- `resource_type`: "dataset", "pipeline", "analysis"
- `config`: Configuration to validate

**Returns:**
```json
{
  "valid": true,
  "errors": [],
  "warnings": ["Consider adding cross-validation"]
}
```

## Error Handling

All functions return structured error responses:

```json
{
  "error": {
    "code": "INVALID_CONFIG",
    "message": "Pipeline configuration is invalid",
    "details": {
      "field": "steps[0].params.n_components",
      "expected": "integer > 0",
      "received": -1
    }
  }
}
```

Common error codes:
- `INVALID_CONFIG`: Configuration validation failed
- `RESOURCE_NOT_FOUND`: Requested resource doesn't exist
- `OPERATION_FAILED`: Execution failed
- `RESOURCE_BUSY`: Resource is locked by another operation
- `SYSTEM_OVERLOAD`: Too many concurrent operations

## Async Operation Pattern

Long-running operations follow this pattern:

1. **Initiate**: Call function, get job ID
2. **Poll Status**: Check progress with status function
3. **Get Results**: Retrieve final results when complete

```python
# Start operation
result = run_pipeline(pipeline_id, dataset_id)
job_id = result["run_id"]

# Poll for completion
while True:
    status = get_run_status(job_id)
    if status["status"] == "completed":
        results = get_run_results(job_id)
        break
    elif status["status"] == "failed":
        handle_error(status["error"])
        break
    time.sleep(5)  # Wait before polling again
```

## Configuration Schemas

### Dataset Configuration

```json
{
  "type": "csv|directory|numpy",
  "name": "optional_name",
  "train": {
    "X": "path/to/X.csv",
    "y": "path/to/y.csv"
  },
  "test": {
    "X": "path/to/X_test.csv",
    "y": "path/to/y_test.csv"
  },
  "options": {
    "delimiter": ";",
    "header": true,
    "wavelengths": [1000, 1001, ...]
  }
}
```

### Pipeline Configuration

```json
{
  "name": "optional_name",
  "steps": [
    {
      "type": "preprocessor|model|cross_validator",
      "name": "ClassName",
      "params": {
        "param1": "value1",
        "param2": 42
      }
    }
  ]
}
```

This service-oriented API provides a clean, uniform interface for controlling nirs4all, making it easy for web applications and other services to integrate comprehensively.</content>
<parameter name="filePath">d:\Workspace\ML\NIRS\nirs4all\docs\Service_api.MD