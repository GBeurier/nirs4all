# Frequently Asked Questions

Common questions, errors, and solutions for nirs4all.

## Installation

### How do I install nirs4all?

```bash
pip install nirs4all
```

For GPU support with TensorFlow:
```bash
pip install nirs4all tensorflow[and-cuda]
```

### How do I verify my installation?

```bash
nirs4all --test-install
```

This checks all dependencies and reports available frameworks.

### Which Python versions are supported?

Python 3.11+ is required.

### Do I need TensorFlow, PyTorch, or JAX?

No. nirs4all works with scikit-learn only. Deep learning frameworks are optional and only needed if you want to use neural network models.

---

## Data Loading

### What file formats are supported?

- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- MATLAB (`.mat`)
- NumPy (`.npy`, `.npz`)
- Parquet (`.parquet`)

See {doc}`/user_guide/data/loading_data` for details.

### How do I specify which column is the target variable?

```python
from nirs4all.data import DatasetConfigs

dataset = DatasetConfigs(
    "data.csv",
    y_column="concentration",  # Target column name
)
```

### How do I handle multiple data sources?

```python
dataset = DatasetConfigs([
    {"path": "nir.csv", "source_name": "NIR"},
    {"path": "raman.csv", "source_name": "Raman"},
])
```

### Error: "Could not infer target column"

Your dataset doesn't have a clear target column. Specify it explicitly:

```python
dataset = DatasetConfigs("data.csv", y_column="my_target")
```

### Error: "Sample count mismatch between X and y"

Your feature matrix and target array have different numbers of samples. Check your data file for:
- Missing values
- Misaligned rows
- Header issues

---

## Pipeline Execution

### How do I run a basic pipeline?

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

pipeline = [
    ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
    PLSRegression(n_components=10),
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="path/to/data.csv"
)
```

### Why do I need cross-validation in my pipeline?

Cross-validation (e.g., `ShuffleSplit`, `KFold`) is required to:
- Split data into train/test sets
- Evaluate model generalization
- Generate out-of-fold predictions

### How do I save my results?

Results are automatically saved when using `PipelineRunner`:

```python
from nirs4all.pipeline import PipelineRunner

runner = PipelineRunner(
    save_artifacts=True,
    workspace_path="workspace/"
)
predictions, _ = runner.run(pipeline, dataset)
```

### Error: "No splitter found in pipeline"

Add a cross-validation splitter before your model:

```python
pipeline = [
    ShuffleSplit(n_splits=5, random_state=42),  # Add this
    PLSRegression(n_components=10),
]
```

### Error: "Pipeline must contain at least one model step"

Add a model to your pipeline:

```python
pipeline = [
    SNV(),
    ShuffleSplit(n_splits=5, random_state=42),
    PLSRegression(n_components=10),  # Model step
]
```

---

## Preprocessing

### Which preprocessing should I use?

| Data Issue | Recommended Preprocessing |
|------------|---------------------------|
| Baseline drift | `Detrend`, `BaselineCorrection` |
| Scatter effects | `SNV`, `MSC` |
| Noise | `SavitzkyGolay`, `Gaussian` |
| Scale differences | `StandardScaler`, `MinMaxScaler` |
| Derivatives | `FirstDerivative`, `SecondDerivative` |

See {doc}`/user_guide/preprocessing/cheatsheet` for model-specific recommendations.

### Can I combine multiple preprocessings?

Yes, chain them in your pipeline:

```python
pipeline = [
    SNV(),
    SavitzkyGolay(window_length=11, polyorder=2),
    FirstDerivative(),
    ShuffleSplit(n_splits=5, random_state=42),
    PLSRegression(n_components=10),
]
```

### How do I compare different preprocessings?

Use `feature_augmentation`:

```python
pipeline = [
    {"feature_augmentation": [SNV, Detrend, MSC], "action": "extend"},
    ShuffleSplit(n_splits=5, random_state=42),
    PLSRegression(n_components=10),
]

# Result will contain predictions for each preprocessing
```

---

## Models

### What models can I use?

Any scikit-learn compatible model:
- **Regression**: PLSRegression, RandomForestRegressor, SVR, etc.
- **Classification**: LogisticRegression, RandomForestClassifier, SVC, etc.
- **Deep Learning**: nicon, decon (with TensorFlow/PyTorch/JAX)

### How do I know if my task is regression or classification?

nirs4all auto-detects based on your target variable:
- **Continuous values** → Regression
- **Discrete categories** → Classification

Override with:
```python
dataset = DatasetConfigs("data.csv", task="classification")
```

### How do I tune hyperparameters?

Use `finetune_params`:

```python
{
    "model": PLSRegression(),
    "finetune_params": {
        "n_trials": 20,
        "sample": "tpe",
        "model_params": {
            "n_components": ('int', 1, 20),
        }
    }
}
```

See {doc}`/user_guide/models/hyperparameter_tuning` for details.

### Error: "Model does not support classification"

Some models are regression-only. For classification, use:
- `nicon_classification` instead of `nicon`
- `RandomForestClassifier` instead of `RandomForestRegressor`

---

## Deep Learning

### How do I use neural networks?

```python
from nirs4all.operators.models.tensorflow.nicon import nicon

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, random_state=42),
    {
        'model': nicon,
        'train_params': {'epochs': 50, 'verbose': 1}
    }
]
```

### Error: "TensorFlow not installed"

Install TensorFlow:
```bash
pip install tensorflow
```

### Error: "CUDA out of memory"

Reduce batch size or use a smaller model:

```python
{
    'model': thin_nicon,  # Smaller architecture
    'train_params': {'batch_size': 8}  # Smaller batches
}
```

### Neural network training is slow

- Enable GPU: Install `tensorflow[and-cuda]` or `torch` with CUDA
- Reduce epochs for quick tests
- Use `hyperband` for efficient hyperparameter search

---

## Results and Visualization

### How do I access prediction results?

```python
result = nirs4all.run(pipeline, dataset)

# Best score
print(result.best_score)

# All predictions
for pred in result.predictions:
    print(pred.get('rmse'))

# Top 5 configurations
for pred in result.top(5):
    print(pred)
```

### How do I visualize results?

```python
from nirs4all.visualization.predictions import PredictionAnalyzer

analyzer = PredictionAnalyzer(result.predictions)
analyzer.plot_scatter()
analyzer.plot_top_k(k=10)
analyzer.plot_heatmap(x_var="model_name", y_var="preprocessings")
```

### How do I export my model for production?

```python
from nirs4all.pipeline.bundle import BundleManager

manager = BundleManager()
manager.export(
    predictions=result.predictions,
    export_path="my_model.n4a"
)
```

---

## Performance

### Pipeline is slow. How do I speed it up?

1. **Reduce cross-validation folds**: `n_splits=3` instead of `n_splits=10`
2. **Use fewer trials**: Lower `n_trials` in `finetune_params`
3. **Enable parallelization**: `n_jobs=-1` for sklearn models
4. **Use GPU**: For neural networks
5. **Reduce preprocessing combinations**: Fewer items in `feature_augmentation`

### How much memory does nirs4all use?

Memory scales with:
- Dataset size (samples × features)
- Number of preprocessing variants
- Model complexity
- Cross-validation folds

For large datasets, process in batches or reduce `n_splits`.

### Can I run pipelines in parallel?

Sklearn models support `n_jobs=-1` for internal parallelization. Pipeline-level parallelism is planned for future releases.

---

## Troubleshooting

### Error: "No module named 'nirs4all'"

Install nirs4all:
```bash
pip install nirs4all
```

### Error: "AttributeError: module 'nirs4all' has no attribute..."

You may have an outdated version. Update:
```bash
pip install --upgrade nirs4all
```

### Plots don't display

- In scripts: Add `plt.show()` at the end
- In Jupyter: Use `%matplotlib inline`
- Set `plots_visible=True` in `nirs4all.run()`

### Results are NaN or infinite

Check your data for:
- Missing values
- Infinite values
- Division by zero in preprocessing
- Incompatible target scale

```python
import numpy as np

# Check data
print(np.isnan(X).any())  # NaN check
print(np.isinf(X).any())  # Inf check
```

### Memory error

Reduce memory usage:
```python
# Smaller cross-validation
ShuffleSplit(n_splits=3, test_size=0.2)  # Instead of 10 folds

# Process fewer variants
{"feature_augmentation": [SNV, Detrend]}  # Instead of 10 preprocessings
```

---

## Best Practices

### Preprocessing

1. **Always scale for neural networks**: Use `MinMaxScaler` or `StandardScaler`
2. **SNV before derivatives**: Apply scatter correction first
3. **Don't over-preprocess**: More isn't always better
4. **Match preprocessing to model**: See {doc}`/user_guide/preprocessing/cheatsheet`

### Cross-Validation

1. **Use enough folds**: Minimum 3, recommended 5-10
2. **Set random_state**: For reproducibility
3. **Use stratification for classification**: `StratifiedKFold`
4. **Consider group structure**: Use `GroupKFold` for grouped samples

### Model Selection

1. **Start with PLS**: Reliable baseline for NIRS
2. **Compare multiple models**: Use branching
3. **Don't overtune**: More trials ≠ better results
4. **Validate on held-out data**: Don't trust only CV scores

### Reproducibility

1. **Set random seeds**: `random_state=42` everywhere
2. **Save artifacts**: `save_artifacts=True`
3. **Version your data**: Track dataset versions
4. **Export configurations**: Save pipeline YAML

---

## Getting Help

### Where can I find more examples?

- `examples/user/` - User tutorials
- `examples/developer/` - Advanced examples
- `examples/reference/` - Reference implementations

### How do I report a bug?

Open an issue on GitHub with:
1. nirs4all version (`nirs4all --version`)
2. Python version
3. Error message and traceback
4. Minimal reproducible example

### Where can I ask questions?

- GitHub Discussions
- GitHub Issues (for bugs)

## See Also

- {doc}`/getting_started/installation` - Installation guide
- {doc}`/getting_started/quickstart` - Quick start tutorial
- {doc}`/user_guide/troubleshooting/migration` - Migration guides
- {doc}`/user_guide/troubleshooting/dataset_troubleshooting` - Dataset troubleshooting
