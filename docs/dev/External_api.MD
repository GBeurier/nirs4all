# External API Documentation

This document describes the public API of nirs4all for external applications (e.g., webapps) to manage datasets, pipelines, predictions, and visualizations. It focuses on the signatures used in the examples and provides a comprehensive interface for controlling nirs4all programmatically.

## Overview

nirs4all provides a pipeline-based framework for Near-Infrared Spectroscopy (NIRS) data analysis. The main components are:

- **Datasets**: Spectroscopic data with samples, features, and targets
- **Pipelines**: Processing workflows combining preprocessing, models, and evaluation
- **Predictions**: Model outputs with performance metrics
- **Analysis**: Visualization and comparison tools

## Core Classes and Functions

### Dataset Management

#### `nirs4all.dataset.DatasetConfigs`

Configuration class for loading datasets from various sources.

```python
class DatasetConfigs:
    def __init__(self, configurations: Union[Dict[str, Any], List[Dict[str, Any]], str, List[str]])

    def get_dataset(self, config, name) -> SpectroDataset
    def get_datasets(self) -> List[SpectroDataset]
    def iter_datasets(self)
```

**Usage Examples:**
```python
# Load from directory with CSV files
dataset_config = DatasetConfigs('sample_data/regression')

# Load from specific files
dataset_config = DatasetConfigs({
    'X_train': 'data/Xcal.csv',
    'y_train': 'data/Ycal.csv',
    'X_test': 'data/Xval.csv',
    'y_test': 'data/Yval.csv'
})

# Multiple datasets
dataset_config = DatasetConfigs(['dataset1', 'dataset2'])
```

#### `nirs4all.dataset.SpectroDataset`

Main dataset class containing spectroscopic data.

```python
class SpectroDataset:
    def __init__(self, name: str = "Unknown_dataset")

    # Data access
    def x(self, selector: Selector, layout: Layout = "2d", concat_source: bool = True) -> OutputData
    def y(self, selector: Selector) -> np.ndarray

    # Metadata
    @property
    def num_samples(self) -> int
    @property
    def num_features(self) -> Union[List[int], int]
    @property
    def task_type(self) -> Optional[str]  # "regression", "binary_classification", "multiclass_classification"

    # Data manipulation
    def add_samples(self, data: InputData, indexes: Optional[IndexDict] = None, headers: Optional[Union[List[str], List[List[str]]]] = None) -> None
    def add_targets(self, y: np.ndarray) -> None
    def add_features(self, features: InputFeatures, processings: ProcessingList, source: int = -1) -> None

    # Utility
    def print_summary(self) -> None
    def __str__(self) -> str
```

**Common Selectors:**
```python
{"partition": "train"}  # Training data
{"partition": "test"}   # Test data
{"partition": "val"}    # Validation data
```

### Pipeline Management

#### `nirs4all.pipeline.PipelineConfigs`

Configuration class for defining processing pipelines.

```python
class PipelineConfigs:
    def __init__(self, definition: Union[Dict, List[Any], str], name: str = "", description: str = "No description provided", max_generation_count: int = 10000)

    # Properties
    @property
    def steps(self) -> List[Any]  # List of pipeline configurations
    @property
    def names(self) -> List[str]  # Configuration names
    @property
    def has_configurations(self) -> bool  # True if multiple configurations generated
```

**Pipeline Definition Examples:**
```python
# Basic regression pipeline
pipeline = [
    MinMaxScaler(),  # Feature scaling
    {"y_processing": MinMaxScaler()},  # Target scaling
    {"feature_augmentation": {"_or_": [Detrend, FirstDerivative, Gaussian], "size": 2}},  # Preprocessing combinations
    ShuffleSplit(n_splits=3, test_size=0.25),  # Cross-validation
    {"model": PLSRegression(n_components=5), "name": "PLS_5_components"}
]

pipeline_config = PipelineConfigs(pipeline, "basic_regression")
```

#### `nirs4all.pipeline.PipelineRunner`

Main execution engine for running pipelines.

```python
class PipelineRunner:
    def __init__(self,
                 max_workers: Optional[int] = None,
                 continue_on_error: bool = False,
                 backend: str = 'threading',
                 verbose: int = 0,
                 parallel: bool = False,
                 results_path: Optional[str] = None,
                 save_files: bool = True,
                 mode: str = "train",
                 load_existing_predictions: bool = True,
                 show_spinner: bool = True,
                 enable_tab_reports: bool = True,
                 random_state: Optional[int] = None,
                 plots_visible: bool = False,
                 keep_datasets: bool = True)

    # Main execution
    def run(self, pipeline_configs: PipelineConfigs, dataset_configs: DatasetConfigs) -> Tuple[Predictions, Dict]

    # Prediction on new data
    def predict(self, prediction_obj: Union[Dict[str, Any], str], dataset_config: DatasetConfigs, all_predictions: bool = False, verbose: int = 0) -> Tuple[np.ndarray, Predictions]

    # SHAP explanations
    def explain(self, prediction_obj: Union[Dict[str, Any], str], dataset_config: DatasetConfigs, shap_params: Optional[Dict[str, Any]] = None, verbose: int = 0) -> Tuple[Dict[str, Any], str]
```

**Usage Examples:**
```python
# Training
runner = PipelineRunner(save_files=True, verbose=1)
predictions, datasets_predictions = runner.run(pipeline_config, dataset_config)

# Prediction
y_pred, predictions = runner.predict(best_model_id, new_dataset_config)

# SHAP analysis
shap_results, output_dir = runner.explain(best_model_id, dataset_config,
                                         shap_params={'n_samples': 200, 'visualizations': ['summary']})
```

### Predictions Management

#### `nirs4all.dataset.Predictions`

Container for model predictions and performance metrics.

```python
class Predictions:
    def __init__(self, filepath: Optional[str] = None)

    # Loading
    @classmethod
    def load(cls, dataset_name: Optional[str] = None, path: str = "results", aggregate_partitions: bool = False, **filters) -> 'Predictions'
    def load_from_file(self, filepath: str) -> None

    # Saving
    def save_to_file(self, filepath: str) -> None
    @staticmethod
    def save_predictions_to_csv(y_true: Optional[Union[np.ndarray, List[float]]] = None, y_pred: Optional[Union[np.ndarray, List[float]]] = None, filepath: str = "", prefix: str = "", suffix: str = "")

    # Querying
    def filter_predictions(self, dataset_name: Optional[str] = None, partition: Optional[str] = None, config_name: Optional[str] = None, model_name: Optional[str] = None, fold_id: Optional[str] = None, step_idx: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]
    def get_similar(self, **filter_kwargs) -> Optional[Dict[str, Any]]

    # Ranking and selection
    def top(self, n: int, rank_metric: str = "", rank_partition: str = "val", display_metrics: list[str] = None, display_partition: str = "test", aggregate_partitions: bool = False, ascending: bool = True, group_by_fold: bool = False, **filters) -> PredictionResultsList
    def top_k(self, k: int = 5, metric: str = "", ascending: bool = True, aggregate_partitions: List[str] = [], **filters) -> List[Union[Dict[str, Any], 'PredictionResult']]
    def get_best(self, metric: str = "", ascending: bool = True, aggregate_partitions: List[str] = [], **filters) -> Optional[Union[Dict[str, Any], 'PredictionResult']]

    # Utility
    @property
    def num_predictions(self) -> int
    def get_unique_values(self, column: str) -> List[str]
    def get_datasets(self) -> List[str]
    def get_models(self) -> List[str]
    def merge_predictions(self, other: 'Predictions') -> None
```

**Usage Examples:**
```python
# Load predictions
predictions = Predictions.load(dataset_name="my_dataset")

# Get top models
top_models = predictions.top(5, rank_metric="rmse", rank_partition="val")
best_model = predictions.get_best(metric="rmse")

# Filter predictions
test_predictions = predictions.filter_predictions(partition="test", model_name="PLS_5")

# Save results
predictions.save_to_file("results/predictions.json")
best_model.save_to_csv("results/best_model.csv")
```

#### `nirs4all.dataset.PredictionResult`

Individual prediction result with convenience methods.

```python
class PredictionResult(dict):
    @property
    def id(self) -> str
    @property
    def model_name(self) -> str
    @property
    def dataset_name(self) -> str

    def save_to_csv(self, path: str = "results", force_path: Optional[str] = None) -> None
```

### Analysis and Visualization

#### `nirs4all.dataset.PredictionAnalyzer`

Analysis and plotting utilities for predictions.

```python
class PredictionAnalyzer:
    def __init__(self, predictions_obj: Predictions, dataset_name_override: str = None)

    # Plotting methods
    def plot_top_k_comparison(self, k: int = 5, rank_metric: str = 'rmse', rank_partition: str = 'val', display_partition: str = 'all', **filters) -> matplotlib.figure.Figure
    def plot_heatmap_v2(self, x_var: str = "model_name", y_var: str = "preprocessings", aggregation: str = 'best', rank_metric: str = "rmse", rank_partition: str = "val", display_metric: str = "rmse", display_partition: str = "test", show_counts: bool = True, **filters) -> matplotlib.figure.Figure
    def plot_variable_candlestick(self, variable: str = "model_name", metric: str = "rmse", partition: str = "test", **filters) -> matplotlib.figure.Figure
    def plot_score_histogram(self, partition: str = "test", metric: str = "rmse", **filters) -> matplotlib.figure.Figure
```

**Usage Examples:**
```python
analyzer = PredictionAnalyzer(predictions)

# Generate plots
fig1 = analyzer.plot_top_k_comparison(k=5, rank_metric='rmse')
fig2 = analyzer.plot_heatmap_v2(x_var="model_name", y_var="preprocessings")
fig3 = analyzer.plot_variable_candlestick(variable="model_name")
fig4 = analyzer.plot_score_histogram()

# Display plots
import matplotlib.pyplot as plt
plt.show()
```

### Transformations and Operators

#### Preprocessing Transformations

Located in `nirs4all.operators.transformations`

```python
# Scaling and normalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# NIRS-specific transformations
from nirs4all.operators.transformations import (
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection,
    LogTransform, Baseline, ResampleTransformer, CropTransformer
)
```

**Usage in Pipelines:**
```python
pipeline = [
    MinMaxScaler(),  # sklearn scaler
    StandardNormalVariate(),  # NIRS preprocessing
    SavitzkyGolay(window_length=11, polyorder=2),  # Smoothing
    FirstDerivative()  # Derivative preprocessing
]
```

#### Cross-Validation

```python
from sklearn.model_selection import ShuffleSplit, KFold, RepeatedKFold

# Usage in pipeline
pipeline = [
    # ... preprocessing ...
    ShuffleSplit(n_splits=5, test_size=0.2),
    # ... models ...
]
```

#### Models

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Usage in pipeline
pipeline = [
    # ... preprocessing ...
    {"model": PLSRegression(n_components=10), "name": "PLS_10"},
    {"model": RandomForestRegressor(n_estimators=100), "name": "RF_100"}
]
```

### Utility Functions

#### Backend Detection

```python
from nirs4all.utils import (
    is_tensorflow_available,
    is_torch_available,
    is_gpu_available,
    framework
)

# Check available backends
if is_tensorflow_available():
    print("TensorFlow available")

gpu_available = is_gpu_available()
current_framework = framework()
```

#### Controller System

```python
from nirs4all.controllers import register_controller, CONTROLLER_REGISTRY

# Access registered controllers
num_controllers = len(CONTROLLER_REGISTRY)
print(f"Available controllers: {num_controllers}")
```

## Complete Workflow Examples

### Basic Regression Pipeline

```python
from nirs4all.dataset import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

# Configure dataset
dataset_config = DatasetConfigs('sample_data/regression')

# Configure pipeline
pipeline = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    ShuffleSplit(n_splits=3, test_size=0.25),
    {"model": PLSRegression(n_components=5), "name": "PLS_5"}
]
pipeline_config = PipelineConfigs(pipeline, "basic_regression")

# Run pipeline
runner = PipelineRunner(verbose=1)
predictions, datasets_predictions = runner.run(pipeline_config, dataset_config)

# Analyze results
best_model = predictions.get_best(metric="rmse")
print(f"Best model: {best_model['model_name']} (RMSE: {best_model['test_score']:.4f})")

# Save results
best_model.save_to_csv("results/best_model.csv")
```

### Prediction on New Data

```python
# Get best model from training
best_model = predictions.get_best(metric="rmse")
model_id = best_model['id']

# Prepare new dataset
new_dataset_config = DatasetConfigs({
    'X_test': 'new_data/Xtest.csv'
})

# Run prediction
runner = PipelineRunner(verbose=0)
y_pred, pred_results = runner.predict(model_id, new_dataset_config)

# Save predictions
Predictions.save_predictions_to_csv(y_pred=y_pred, filepath="results/new_predictions.csv")
```

### Advanced Analysis with Visualization

```python
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer

# Create analyzer
analyzer = PredictionAnalyzer(predictions)

# Generate comparison plots
fig1 = analyzer.plot_top_k_comparison(k=5, rank_metric='rmse')
fig2 = analyzer.plot_heatmap_v2(
    x_var="model_name",
    y_var="preprocessings",
    rank_metric="rmse",
    display_metric="rmse"
)

# Save plots
fig1.savefig("results/model_comparison.png")
fig2.savefig("results/preprocessing_heatmap.png")
```

### Hyperparameter Optimization

```python
pipeline = [
    MinMaxScaler(),
    StandardNormalVariate(),
    ShuffleSplit(n_splits=3),
    {
        "model": PLSRegression(),
        "name": "PLS_Optimized",
        "finetune_params": {
            "n_trials": 20,
            "model_params": {
                'n_components': ('int', 1, 10)
            }
        }
    }
]

pipeline_config = PipelineConfigs(pipeline, "optimized")
runner = PipelineRunner()
predictions, _ = runner.run(pipeline_config, dataset_config)
```

## Error Handling

Most methods may raise exceptions for:
- File not found errors
- Invalid configuration
- Data format issues
- Model training failures

Use try-except blocks for robust webapp integration:

```python
try:
    predictions, _ = runner.run(pipeline_config, dataset_config)
except Exception as e:
    print(f"Pipeline execution failed: {e}")
    # Handle error in webapp
```

## Performance Considerations

- **Large datasets**: Use `save_files=False` and `keep_datasets=False` in PipelineRunner to reduce memory usage
- **Parallel execution**: Set `parallel=True` and adjust `max_workers` for multi-core systems
- **Background processing**: For webapps, run long pipelines asynchronously and poll status
- **Caching**: DatasetConfigs caches loaded data; reuse instances when possible

## File Formats and Data Sources

nirs4all supports:
- **CSV files**: Spectral data with wavelength columns, target files
- **Directories**: Auto-discovery of train/test splits
- **Custom loaders**: Extendable data loading system

See `examples/` directory for concrete usage patterns and data format examples.</content>
<parameter name="filePath">d:\Workspace\ML\NIRS\nirs4all\docs\External_api.MD