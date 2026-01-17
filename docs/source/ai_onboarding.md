# AI Coding Assistant Onboarding

**Quick reference for AI assistants working with nirs4all.** For detailed documentation, see [ReadTheDocs](https://nirs4all.readthedocs.io/).

---

## Core Concepts (30-Second Overview)

**nirs4all** is a Python library for Near-Infrared Spectroscopy (NIRS) data analysis with ML pipelines.

| Concept | Description |
|---------|-------------|
| **Pipeline** | List of steps (preprocessing → splitting → model) executed sequentially |
| **SpectroDataset** | Core container holding `X` (spectra), `y` (targets), `metadata`, `folds` |
| **Operators** | Transformers, models, splitters - anything sklearn-compatible + NIRS-specific |
| **Controllers** | Registry pattern that handles operator execution (extensible) |
| **Bundles (.n4a)** | Serialized pipelines for deployment with full preprocessing chain |

**Typical Workflow**: Load data → Define pipeline → `nirs4all.run()` → Analyze results → Export model

---

## Primary API (Module-Level)

All functions are directly on the `nirs4all` module:

```python
import nirs4all

# Train a pipeline
result = nirs4all.run(pipeline=[...], dataset="path/to/data", verbose=1)

# Make predictions with exported model
predictions = nirs4all.predict("model.n4a", new_data)

# SHAP explanations
explanations = nirs4all.explain("model.n4a", test_data)

# Retrain on new data
new_result = nirs4all.retrain("model.n4a", new_data, mode="transfer")

# Reusable session for resource sharing
with nirs4all.session() as s:
    r1 = nirs4all.run(pipeline1, data, session=s)
    r2 = nirs4all.run(pipeline2, data, session=s)

# Generate synthetic data
dataset = nirs4all.generate(n_samples=500, complexity="realistic")
dataset = nirs4all.generate.regression(n_samples=500)
dataset = nirs4all.generate.classification(n_samples=300, n_classes=3)
```

---

## Function Signatures

### `nirs4all.run()`

```python
nirs4all.run(
    pipeline,                    # List of steps or list of pipelines (batch)
    dataset,                     # Path, SpectroDataset, or list (batch)
    verbose=0,                   # 0=silent, 1=progress, 2=detailed
    session=None,                # Optional Session for resource reuse
    artifacts_path=None,         # Where to save run artifacts
    name=None,                   # Run name identifier
) -> RunResult
```

### `nirs4all.predict()`

```python
nirs4all.predict(
    model,                       # Path to .n4a bundle or loaded model
    data,                        # X array, DataFrame, or SpectroDataset
    verbose=0,
) -> PredictResult
```

### `nirs4all.explain()`

```python
nirs4all.explain(
    model,                       # Path to .n4a bundle
    data,                        # Test data for explanations
    explainer_type="auto",       # auto | tree | kernel | linear | deep
    max_samples=100,             # SHAP background samples
) -> ExplainResult
```

### `nirs4all.retrain()`

```python
nirs4all.retrain(
    source,                      # Path to .n4a bundle
    data,                        # New dataset
    mode="full",                 # full | transfer | finetune
    verbose=0,
) -> RunResult
```

### `nirs4all.generate()`

```python
nirs4all.generate(
    n_samples=1000,
    complexity="realistic",      # simple | realistic | complex
    wavelength_range=(1000, 2500),
    components=["water", "protein", "lipid"],
    target_range=(0, 100),
    train_ratio=0.8,
    as_dataset=True,             # False returns (X, y) tuple
    random_state=None,
) -> SpectroDataset
```

---

## Result Objects

### RunResult (from `run()`)

```python
result.best              # Best prediction entry (dict)
result.best_score        # Primary test score
result.best_rmse         # RMSE (regression)
result.best_r2           # R² (regression)
result.best_accuracy     # Accuracy (classification)
result.num_predictions   # Total predictions stored

result.top(n=5)          # Top N predictions
result.filter(model="PLS")  # Filter by criteria
result.export("model.n4a")  # Export best model
result.get_models()      # List unique model names
result.get_datasets()    # List unique dataset names
```

### PredictResult (from `predict()`)

```python
preds.values             # Predicted values array
preds.shape              # Shape of predictions
preds.model_name         # Model used
preds.to_dataframe()     # Convert to DataFrame
```

### ExplainResult (from `explain()`)

```python
exp.shap_values          # SHAP values array
exp.feature_names        # Feature labels
exp.base_value           # Expected value
exp.top_features         # Ranked by importance
exp.get_feature_importance(top_n=10)
exp.to_dataframe()
```

---

## Pipeline Syntax

Steps can be classes, instances, or dicts with keywords:

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

# Basic pipeline
pipeline = [
    MinMaxScaler(),              # Preprocessing
    KFold(n_splits=5),           # Cross-validation
    PLSRegression(n_components=10)  # Model (auto-detected)
]

# With explicit keywords
pipeline = [
    MinMaxScaler(),
    {"y_processing": StandardScaler()},     # Target scaling
    KFold(n_splits=5),
    {"model": PLSRegression(n_components=10)}
]
```

### Pipeline Keywords

| Keyword | Purpose | Example |
|---------|---------|---------|
| `model` | Explicit model definition | `{"model": PLSRegression(10)}` |
| `y_processing` | Target (y) scaling | `{"y_processing": MinMaxScaler()}` |
| `branch` | Parallel sub-pipelines | `{"branch": [[SNV(), PLS()], [MSC(), RF()]]}` |
| `merge` | Combine branch outputs | `{"merge": "predictions"}` |
| `source_branch` | Per-source preprocessing | `{"source_branch": {"NIR": [...], "VIS": [...]}}` |

### Generator Keywords (Pipeline Expansion)

```python
from nirs4all.operators import SNV, MSC, Detrend

# _or_: Creates N pipelines, one per option
pipeline = [
    {"_or_": [SNV, MSC, Detrend]},  # Expands to 3 pipelines
    PLSRegression(10)
]

# _range_: Parameter sweep
pipeline = [
    MinMaxScaler(),
    {"_range_": [1, 30, 5], "param": "n_components", "class": PLSRegression}
    # Creates pipelines with n_components=1,6,11,16,21,26
]

# _choice_: Named alternatives with full specs
pipeline = [
    {"_choice_": {
        "pls": PLSRegression(10),
        "rf": RandomForestRegressor(n_estimators=100)
    }}
]
```

---

## Stacking / Ensemble Patterns

```python
from sklearn.linear_model import Ridge

# Branch → Merge → Meta-model
pipeline = [
    {"branch": [
        [SNV(), PLSRegression(10)],           # Branch 1
        [MSC(), RandomForestRegressor()],     # Branch 2
    ]},
    {"merge": "predictions"},    # OOF predictions as features
    {"model": Ridge()},          # Meta-model
]
```

---

## Key Operators

### NIRS-Specific Preprocessing (`nirs4all.operators`)

```python
from nirs4all.operators import (
    SNV,                  # Standard Normal Variate
    MSC,                  # Multiplicative Scatter Correction
    MultiplicativeScatterCorrection,  # Alias for MSC
    SavitzkyGolay,        # Smoothing + derivatives
    Detrend,              # Baseline detrending
    Baseline,             # Baseline removal
    Derivate,             # First/second derivatives
    Gaussian,             # Gaussian filter
)
```

### Sample Augmentation

```python
from nirs4all.operators import (
    Spline_Smoothing,
    Rotate_Translate,
    Random_X_Operation,
)
```

### Splitting Strategies

```python
from nirs4all.operators import (
    KennardStone,         # Kennard-Stone sampling
    SPXY,                 # Sample Partitioning (X + Y)
)
# Plus all sklearn splitters: KFold, ShuffleSplit, StratifiedKFold, etc.
```

### Deep Learning Models

```python
# TensorFlow-based (lazy-loaded)
from nirs4all.operators.models.tensorflow import (
    NICON,                # 1D CNN for NIRS
    DECON,                # Deep CNN
    ResNet1D,
    Transformer1D,
)
```

---

## Dataset Loading

```python
import nirs4all
from nirs4all.data import SpectroDataset

# From path (auto-detects format)
result = nirs4all.run(pipeline, dataset="path/to/data.csv")
result = nirs4all.run(pipeline, dataset="path/to/folder/")

# From SpectroDataset
ds = SpectroDataset.from_csv("data.csv", x_col="spectrum", y_col="target")
ds = SpectroDataset.from_parquet("data.parquet")
result = nirs4all.run(pipeline, dataset=ds)

# Batch execution (Cartesian product)
result = nirs4all.run(
    pipeline=[pipeline_a, pipeline_b],
    dataset=[dataset_1, dataset_2]  # Runs 4 combinations
)
```

### SpectroDataset Properties

```python
ds.X                     # Feature matrix (n_samples, n_features)
ds.y                     # Target vector (n_samples,)
ds.metadata              # Sample-level metadata dict
ds.sources               # Multi-source tracking
ds.folds                 # CV fold assignments
ds.name                  # Dataset identifier
ds.wavelengths           # Wavelength labels (if available)
```

---

## Architecture Overview

```
nirs4all/
├── api/           # Module-level functions: run(), predict(), explain(), retrain(), session(), generate()
│   └── result.py  # RunResult, PredictResult, ExplainResult
├── pipeline/      # Execution engine
│   ├── runner.py  # PipelineRunner (main orchestrator)
│   ├── config.py  # PipelineConfigs (expands generators)
│   ├── bundle.py  # .n4a export/load
│   └── predictor.py
├── controllers/   # Registry for operator handlers
│   ├── registry.py
│   ├── transforms/
│   ├── models/    # sklearn, tensorflow, pytorch, jax
│   └── splitters/
├── operators/     # NIRS-specific operators
│   ├── transforms/  # SNV, MSC, SavitzkyGolay, etc.
│   ├── models/      # NICON, DECON, etc.
│   ├── augmentation/
│   └── splitters/   # KennardStone, SPXY
├── data/
│   ├── dataset.py   # SpectroDataset
│   ├── predictions.py
│   └── synthetic/   # Data generation
└── sklearn/
    └── pipeline.py  # NIRSPipeline (sklearn wrapper for SHAP)
```

---

## Controller Pattern (Extension Point)

Custom operators register via decorator:

```python
from nirs4all.controllers import register_controller, OperatorController

@register_controller
class MyController(OperatorController):
    priority = 50  # Lower = higher priority

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return isinstance(operator, MyOperatorType)

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True  # Execute during predict()

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        # Transform dataset; return (context, StepOutput)
        ...
```

---

## Common Patterns

### Full Example

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

# Generate or load data
dataset = nirs4all.generate.regression(n_samples=500)

# Define pipeline
pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),
    {"model": PLSRegression(n_components=10)}
]

# Train
result = nirs4all.run(pipeline, dataset, verbose=1)
print(f"Best RMSE: {result.best_rmse:.4f}")
print(f"Best R²: {result.best_r2:.4f}")

# Export
result.export("model.n4a")

# Predict on new data
new_data = nirs4all.generate.regression(n_samples=50)
predictions = nirs4all.predict("model.n4a", new_data.X)
print(predictions.values)

# Explain
explanations = nirs4all.explain("model.n4a", new_data.X)
print(explanations.top_features[:10])
```

### Hyperparameter Search

```python
pipeline = [
    MinMaxScaler(),
    {"_or_": [SNV(), MSC(), Detrend()]},  # Try 3 preprocessors
    {"_range_": [5, 25, 5], "param": "n_components", "class": PLSRegression}
]
# Runs: 3 preprocessors × 5 n_components values = 15 pipelines
result = nirs4all.run(pipeline, dataset)
```

### Multi-Source Data

```python
# Different preprocessing per source
pipeline = [
    {"source_branch": {
        "NIR": [SNV(), MinMaxScaler()],
        "VIS": [StandardScaler()],
    }},
    PLSRegression(10)
]
```

---

## Commands Reference

```bash
# Tests
pytest tests/                     # All tests
pytest tests/unit/                # Unit only
pytest tests/integration/         # Integration only
pytest -m sklearn                 # sklearn-only (fast)
pytest --cov=nirs4all             # With coverage

# Examples
cd examples && ./run.sh           # All examples
./run.sh -c user                  # User category only
./run.sh -q                       # Quick (skip deep learning)

# Verification
nirs4all --test-install
nirs4all --test-integration

# Code quality
ruff check .
mypy .
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `nirs4all/__init__.py` | Package entry, re-exports API |
| `nirs4all/api/run.py` | `run()` implementation |
| `nirs4all/api/result.py` | Result objects |
| `nirs4all/pipeline/runner.py` | Pipeline execution |
| `nirs4all/pipeline/config.py` | PipelineConfigs (generator expansion) |
| `nirs4all/data/dataset.py` | SpectroDataset |
| `nirs4all/controllers/registry.py` | Controller registration |
| `nirs4all/operators/transforms/` | NIRS transforms |

---

## Further Reading

- **[Getting Started](getting_started/index.md)** - Installation and first pipeline
- **[User Guide](user_guide/index.md)** - Task-oriented how-to guides
- **[Pipeline Syntax Reference](reference/pipeline_syntax.md)** - Complete syntax specification
- **[Operator Catalog](reference/operator_catalog.md)** - 270+ operators listed
- **[Generator Keywords](reference/generator_keywords.md)** - `_or_`, `_range_`, `_choice_`
- **[Developer Guide](developer/index.md)** - Architecture and internals
- **[API Reference](api/modules.rst)** - Auto-generated API docs

---

**Version**: 0.6.x | **Python**: 3.11+ | **License**: CeCILL-2.1
