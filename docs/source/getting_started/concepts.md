# Core Concepts

This page explains the fundamental concepts behind NIRS4ALL. Understanding these will help you build effective pipelines.

## Overview

NIRS4ALL is built around three core concepts:

1. **SpectroDataset** - A container for spectral data, targets, and metadata
2. **Pipeline** - A sequence of processing steps
3. **Controllers** - The execution engine that runs each step

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Dataset    │ -> │   Pipeline   │ -> │   Results    │
│  (your data) │    │   (steps)    │    │ (predictions)│
└──────────────┘    └──────────────┘    └──────────────┘
```

## SpectroDataset

`SpectroDataset` is the core data container. It holds:

| Component | Description |
|-----------|-------------|
| **X** (features) | Spectral data matrix (n_samples × n_wavelengths) |
| **y** (targets) | Target values for prediction (n_samples,) |
| **metadata** | Sample information (IDs, groups, dates, etc.) |
| **fold indices** | Cross-validation assignments |

### Creating a Dataset

Most often, NIRS4ALL creates datasets automatically from your files:

```python
from nirs4all.data import DatasetConfigs

# From a folder (auto-detects files)
dataset = DatasetConfigs("path/to/data/")

# From explicit files
dataset = DatasetConfigs({
    "train_x": "spectra.csv",
    "train_y": "targets.csv",
    "train_m": "metadata.csv"  # Optional metadata
})
```

You can also generate synthetic data for testing and prototyping:

```python
import nirs4all

# Generate realistic synthetic NIRS spectra
dataset = nirs4all.generate.regression(
    n_samples=500,
    components=["water", "protein", "lipid"],
    complexity="realistic"
)
```

See {doc}`/user_guide/data/synthetic_data` for more on synthetic data generation.
```

### Partitions

Data is organized into partitions:

| Partition | Purpose |
|-----------|---------|
| **train** | Used for model training and cross-validation |
| **test** | Held-out data for final evaluation |
| **val** | Validation set (often created from train via CV) |

:::{note}
During cross-validation, the train partition is automatically split into train/val folds. The test partition (if provided) remains untouched for final evaluation.
:::

## Pipeline

A **pipeline** is a list of processing steps applied sequentially:

```python
pipeline = [
    MinMaxScaler(),                              # Step 1: Scale features
    StandardNormalVariate(),                     # Step 2: SNV preprocessing
    {"y_processing": MinMaxScaler()},            # Step 3: Scale targets
    ShuffleSplit(n_splits=5),                    # Step 4: Cross-validation
    {"model": PLSRegression(n_components=10)}    # Step 5: Model
]
```

### Step Types

| Step Type | Syntax | Purpose |
|-----------|--------|---------|
| **Transformer** | `MinMaxScaler()` | Modify features (X) |
| **Y Processing** | `{"y_processing": ...}` | Modify targets (y) |
| **Splitter** | `ShuffleSplit(n_splits=5)` | Define cross-validation |
| **Model** | `{"model": PLSRegression()}` | Train predictive model |
| **Branch** | `{"branch": [...]}` | Parallel processing paths |
| **Merge** | `{"merge": "features"}` | Combine branch outputs |
| **Augmentation** | `{"feature_augmentation": ...}` | Generate preprocessing variants |

### Execution Flow

```
Input Data → [Preprocessing] → [CV Split] → [Training] → Predictions
     │              │               │            │
     ▼              ▼               ▼            ▼
SpectroDataset  Transformers    Splitter     Models
```

1. **Data Loading**: Your files are loaded into a SpectroDataset
2. **Preprocessing**: Transformers modify X (and optionally y)
3. **Cross-Validation**: Splitter defines train/val folds
4. **Training**: Each model is trained on each fold
5. **Prediction**: Out-of-fold predictions are collected

## Controllers

**Controllers** are the execution engine. They interpret each pipeline step and perform the appropriate action.

| Controller | Handles |
|------------|---------|
| `TransformController` | sklearn TransformerMixin (scalers, preprocessors) |
| `YProcessingController` | `{"y_processing": ...}` steps |
| `SplitterController` | Cross-validation splitters |
| `ModelController` | `{"model": ...}` steps |
| `BranchController` | `{"branch": ...}` parallel paths |
| `MergeController` | `{"merge": ...}` combining outputs |

:::{tip}
You rarely interact with controllers directly. They work behind the scenes to execute your pipeline.
:::

## Predictions and Results

When you run a pipeline, you get a `RunResult` object:

```python
result = nirs4all.run(pipeline, dataset)

# Access results
result.best_score        # Best model's primary score
result.best              # Best prediction entry (dict)
result.num_predictions   # Total prediction entries
result.predictions       # Full PredictionResultsList

# Get top performers
for pred in result.top(n=5, display_metrics=['rmse', 'r2']):
    print(f"{pred['model_name']}: RMSE={pred['rmse']:.4f}")
```

### Prediction Entry Structure

Each prediction entry contains:

| Field | Description |
|-------|-------------|
| `model_name` | Name of the model |
| `dataset_name` | Name of the dataset |
| `fold_id` | Cross-validation fold index |
| `y_true` | True target values |
| `y_pred` | Predicted values |
| `rmse`, `r2`, etc. | Computed metrics |
| `preprocessings` | Applied preprocessing chain |
| `partition` | Data partition (train/val/test) |

## Key Terminology

| Term | Definition |
|------|------------|
| **Spectral data** | Features from spectroscopy (reflectance, absorbance, etc.) |
| **Wavelength** | Individual feature/column in spectral data |
| **Fold** | One train/validation split in cross-validation |
| **OOF (Out-of-Fold)** | Predictions made on validation data during CV |
| **Operator** | A preprocessing or transformation class |
| **Transformer** | sklearn-compatible operator with `fit()` and `transform()` |
| **Pipeline variant** | One specific configuration when using generators |

## The nirs4all.run() Function

The simplest way to run a pipeline:

```python
result = nirs4all.run(
    pipeline=pipeline,           # List of steps
    dataset=dataset,             # See below for supported formats
    name="MyPipeline",           # Pipeline name
    verbose=1,                   # 0=silent, 1=progress, 2=debug
    save_artifacts=True,         # Save models and results
    save_charts=True,            # Save generated plots
    plots_visible=False          # Show plots interactively
)
```

### Supported Dataset Formats

The `dataset` parameter accepts multiple formats:

| Format | Example |
|--------|----------|
| Path to folder | `"sample_data/regression"` |
| Numpy arrays | `(X, y)` or `X` alone |
| Dict with arrays | `{"X": X, "y": y, "metadata": meta}` |
| `SpectroDataset` | Direct dataset instance |
| `List[SpectroDataset]` | Multiple datasets for multi-dataset runs |
| `DatasetConfigs` | Full configuration object |

For more control, use `PipelineRunner` directly:

```python
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

runner = PipelineRunner(
    verbose=1,
    save_artifacts=True,
    save_charts=True
)

predictions, per_dataset = runner.run(
    PipelineConfigs(pipeline, "MyPipeline"),
    DatasetConfigs("path/to/data")
)
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        nirs4all.run()                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │PipelineRunner│ --> │PipelineOrches│ --> │ Controllers  │    │
│  │              │     │    -trator   │     │  (registry)  │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │PipelineConfigs│    │ExecutionContext│   │SpectroDataset│    │
│  │  (pipeline)   │    │   (state)      │   │   (data)     │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                         RunResult                                │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │ Predictions  │     │   Metrics    │     │  Artifacts   │    │
│  │    List      │     │              │     │   (.n4a)     │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Next Steps

::::{grid} 2
:gutter: 3

:::{grid-item-card} 📖 Loading Data
:link: /user_guide/data/loading_data
:link-type: doc

Learn about DatasetConfigs and supported formats.
:::

:::{grid-item-card} 🔧 Preprocessing
:link: /user_guide/preprocessing/overview
:link-type: doc

NIRS-specific preprocessing techniques.
:::

:::{grid-item-card} 📋 Pipeline Syntax
:link: /reference/pipeline_syntax
:link-type: doc

Complete syntax reference.
:::

:::{grid-item-card} 📝 Examples
:link: /examples/index
:link-type: doc

Working examples organized by topic.
:::

::::

## See Also

- {doc}`quickstart` - Run your first pipeline
- {doc}`/reference/pipeline_syntax` - Complete pipeline syntax
- {doc}`/developer/architecture` - Detailed architecture for contributors
