# Architecture & Hello World

NIRS4ALL is designed around a modular pipeline architecture that separates data definition, pipeline configuration, and execution logic. This separation allows for flexible experimentation and reproducible research.

## High-Level Architecture

The core of NIRS4ALL consists of three main components:

1.  **Data Management (`nirs4all.data`)**:
    *   **`DatasetConfigs`**: Defines how to load and structure your NIRS datasets. It handles reading from various file formats (CSV, Excel, etc.), managing metadata, and organizing data into partitions (train, test, validation).
    *   **`Predictions`**: A standardized container for model outputs, metrics, and metadata, facilitating comparison and analysis.

2.  **Pipeline Configuration (`nirs4all.pipeline`)**:
    *   **`PipelineConfigs`**: Describes the sequence of operations to be performed. A pipeline is a list of steps, which can include preprocessing transformers, data splitters, and machine learning models.
    *   **Operators**: The building blocks of a pipeline. These include:
        *   **Transforms**: Preprocessing steps like SNV, Derivatives, Smoothing (e.g., `StandardNormalVariate`, `SavitzkyGolay`).
        *   **Splitters**: Methods for dividing data (e.g., `KennardStone`, `SPXYSplitter`).
        *   **Models**: Wrappers for ML algorithms (Scikit-learn, TensorFlow, PyTorch, etc.).

3.  **Execution Engine (`nirs4all.pipeline`)**:
    *   **`PipelineRunner`**: The engine that executes the defined pipeline on the specified datasets. It handles the flow of data, training of models, generation of predictions, and logging of results.

## Hello World: A Basic Pipeline

Here is a simple "Hello World" example demonstrating a basic regression pipeline. This example loads data, applies preprocessing, splits the data, and trains a PLS regression model.

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import (
    StandardNormalVariate, SavitzkyGolay, MultiplicativeScatterCorrection
)

# 1. Define your processing pipeline
# The pipeline is a list of steps executed in order.
pipeline = [
    MinMaxScaler(),                    # Scale features to [0, 1]
    StandardNormalVariate(),           # Apply SNV transformation to spectra
    ShuffleSplit(n_splits=3),          # 3-fold cross-validation
    {"y_processing": MinMaxScaler()},  # Scale target values (y)
    {"model": PLSRegression(n_components=10)}, # Add a PLS model
    {"model": RandomForestRegressor(n_estimators=100)}, # Add a Random Forest model
]

# 2. Create configurations
# PipelineConfigs wraps the list of steps.
pipeline_config = PipelineConfigs(pipeline, name="MyFirstPipeline")

# DatasetConfigs points to your data.
# Replace "path/to/your/data" with your actual data path or a dictionary of paths.
dataset_config = DatasetConfigs("path/to/your/data")

# 3. Run the pipeline
# PipelineRunner executes the pipeline and returns predictions.
runner = PipelineRunner(save_files=False, verbose=1)
predictions, predictions_per_datasets = runner.run(pipeline_config, dataset_config)

# 4. Analyze results
# The 'predictions' object contains results for all models and folds.
top_models = predictions.top(n=5, rank_metric='rmse')
print("Top 5 models by RMSE:")
for i, model in enumerate(top_models):
    print(f"{i+1}. {model['model_name']}: RMSE = {model['rmse']:.4f}")
```

### Key Concepts in the Example

*   **Pipeline Definition**: The `pipeline` list mixes standard Scikit-learn objects (like `MinMaxScaler`, `PLSRegression`) with NIRS4ALL specific operators (like `StandardNormalVariate`).
*   **`y_processing`**: Special dictionary syntax `{ "y_processing": ... }` allows you to apply transformations specifically to the target variable.
*   **`model`**: The `{ "model": ... }` syntax explicitly marks a step as a model to be trained and evaluated.
*   **Runner**: The `PipelineRunner` abstracts away the complexity of cross-validation loops, data splitting, and result aggregation.
