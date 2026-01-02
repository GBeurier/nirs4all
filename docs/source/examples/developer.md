# Developer Examples

This section contains advanced examples for users who want to extend NIRS4ALL's capabilities or use its advanced features.

```{contents} On this page
:local:
:depth: 2
```

## Overview

Developer examples are organized into six sections, progressing from advanced pipeline patterns to internal customization:

| Section | Topics | Difficulty |
|---------|--------|------------|
| [Advanced Pipelines](#advanced-pipelines) | Branching, merging, stacking | ★★★☆☆ |
| [Generators](#generators) | Dynamic pipeline generation | ★★★☆☆ |
| [Synthetic Data](#synthetic-data) | Custom data generation | ★★★☆☆ |
| [Deep Learning](#deep-learning) | PyTorch, JAX, TensorFlow | ★★★★☆ |
| [Transfer Learning](#transfer-learning) | Instrument adaptation | ★★★★☆ |
| [Advanced Features](#advanced-features) | Metadata, transforms | ★★★★☆ |
| [Internals](#internals) | Custom controllers, sessions | ★★★★★ |

---

## Advanced Pipelines

Pipeline branching and merging enable sophisticated model comparison, ensemble methods, and multi-source data handling.

### D01: Branching Basics

**Introduction to pipeline branching for parallel experiments.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D01_branching_basics.py)

Pipeline branching enables running multiple parallel sub-pipelines ("branches"), each with its own preprocessing context while sharing common upstream state.

#### Key Concepts

```python
# List syntax: Simple parallel branches
{"branch": [
    [SNV()],              # Branch 0
    [MSC()],              # Branch 1
    [FirstDerivative()],  # Branch 2
]}

# Dict syntax: Named branches
{"branch": {
    "snv": [SNV()],
    "msc": [MSC()],
    "derivative": [FirstDerivative()],
}}

# Generator syntax: Dynamic branches
{"branch": {"_or_": [SNV(), MSC(), FirstDerivative()]}}
```

#### What Branches Share

- ✓ Data loading (no redundant I/O)
- ✓ Train/test splits
- ✓ Upstream preprocessing

#### What's Independent

- ✗ Branch-specific preprocessing
- ✗ Y processing per branch
- ✗ Models trained in-branch

### D02: Branching Advanced

**Statistical comparison and HTML reports.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D02_branching_advanced.py)

#### Branch Comparison

```python
analyzer = PredictionAnalyzer(result.predictions)

# Statistical summary
summary = analyzer.branch_summary(metrics=['rmse', 'r2'])

# Visualizations
analyzer.plot_branch_comparison(display_metric='rmse', show_ci=True)
analyzer.plot_branch_boxplot(display_metric='rmse')
analyzer.plot_branch_heatmap(y_var='fold_id', display_metric='rmse')
```

### D03: Merge Basics

**Stacking and ensemble methods through prediction merging.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D03_merge_basics.py)

```python
pipeline = [
    ShuffleSplit(n_splits=5),

    # Base models in branches
    {"branch": {
        "pls": [PLSRegression(n_components=10)],
        "rf": [RandomForestRegressor()],
        "ridge": [Ridge(alpha=1.0)],
    }},

    # Merge OOF predictions for stacking
    {"merge": "predictions"},

    # Meta-learner
    {"model": Ridge(alpha=0.1)}
]
```

### D04: Merge Sources

**Combine multi-source data with flexible merging.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D04_merge_sources.py)

```python
pipeline = [
    # Per-source preprocessing
    {"source_branch": {
        "NIR": [SNV(), FirstDerivative()],
        "markers": [StandardScaler()],
    }},

    # Merge strategies
    {"merge_sources": "concat"},  # Horizontal concatenation
    # or: "stack" for 3D stacking

    PLSRegression(n_components=10)
]
```

### D05: Meta-Stacking

**Multi-level stacking ensembles.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D05_meta_stacking.py)

---

## Generators

Generators enable dynamic pipeline generation for automated hyperparameter search and experiment design.

### D01: Generator Syntax

**Dynamic pipeline generation with `_or_`, `_range_`, `_grid_`.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/02_generators/D01_generator_syntax.py)

#### Generator Keywords

| Keyword | Purpose | Example |
|---------|---------|---------|
| `_or_` | Alternatives | `{"_or_": [A, B, C]}` → 3 variants |
| `_range_` | Numeric sweep | `{"_range_": [5, 20, 5]}` → [5, 10, 15, 20] |
| `_log_range_` | Log sweep | `{"_log_range_": [0.001, 1, 4]}` → [0.001, 0.01, 0.1, 1.0] |
| `_grid_` | Cartesian product | All combinations |
| `_zip_` | Parallel iteration | Paired values |

#### Combination Controls

```python
# pick: Select k items (combinations)
{"_or_": [A, B, C, D], "pick": 2}
# → [A,B], [A,C], [A,D], [B,C], [B,D], [C,D]

# arrange: Permutations (order matters)
{"_or_": [A, B, C], "arrange": 2}
# → [A,B], [A,C], [B,A], [B,C], [C,A], [C,B]

# count: Limit variants
{"_or_": [A, B, C, D, E], "count": 3}
# → 3 randomly selected
```

### D02: Generator Advanced

**Constraints, presets, and patterns.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/02_generators/D02_generator_advanced.py)

### D03: Generator Iterators

**Iterate over generated configurations.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/02_generators/D03_generator_iterators.py)

### D04: Nested Generators

**Complex nested generation patterns.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/02_generators/D04_nested_generators.py)

---

## Synthetic Data

The synthetic data generator allows creating realistic NIRS spectra for testing, validation, and development. These examples show advanced customization options.

### D05: Custom Components

**Create custom spectral components for synthetic data.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/02_generators/D05_synthetic_custom_components.py)

Learn how to define your own chemical components with specific absorption profiles.

### D06: Testing Integration

**Generate data for testing and benchmarking.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/02_generators/D06_synthetic_testing.py)

Create reproducible datasets for unit tests, benchmark different configurations, and compare real vs synthetic data.

### D07: Wavenumber & Procedural

**Wavenumber utilities and procedural component generation (Phase 1).**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/02_generators/D07_synthetic_wavenumber_procedural.py)

Advanced wavenumber-to-wavelength conversions, overtone calculations, and procedural spectral band generation.

### D08: Application Domains

**Domain-specific synthetic data (Phase 1).**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/02_generators/D08_synthetic_application_domains.py)

Generate spectra tailored to specific applications: agriculture, food, pharmaceutical, petrochemical, and more.

### D09: Instrument Simulation

**Simulate instrument-specific characteristics (Phase 2).**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/02_generators/D09_synthetic_instruments.py)

Model detector types, multi-sensor stitching, multi-scan averaging, and measurement mode effects.

```{note}
For advanced synthetic data features (environmental effects, validation, real data fitting), see the Reference Examples:
- R05: Environmental and Matrix Effects (Phase 3)
- R06: Validation and Quality Assessment (Phase 4)
- R07: Fitting to Real Data (Phase 4)
```

---

## Deep Learning

NIRS4ALL integrates with PyTorch, JAX, and TensorFlow for deep learning workflows.

### D01: PyTorch Models

**Integrate PyTorch neural networks.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/03_deep_learning/D01_pytorch_models.py)

```python
from nirs4all.operators.models.pytorch.nicon import nicon

pipeline = [
    MinMaxScaler(),
    SNV(),
    ShuffleSplit(n_splits=3),

    {"model": nicon(input_dim=2151, output_dim=1),
     "train_params": {
         "epochs": 100,
         "batch_size": 32,
         "learning_rate": 0.001,
         "device": "auto"  # Uses GPU if available
     }}
]
```

#### Built-in Architectures

| Model | Description |
|-------|-------------|
| `nicon` | Convolutional network for spectra |
| `decon` | Deconvolution architecture |
| `transformer` | Attention-based model |

#### Custom PyTorch Models

```python
import torch.nn as nn
from nirs4all.operators.models import framework

@framework("pytorch")
class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.layers(x)
```

### D02: JAX Models

**JAX/Flax integration.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/03_deep_learning/D02_jax_models.py)

### D03: TensorFlow Models

**TensorFlow/Keras integration.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/03_deep_learning/D03_tensorflow_models.py)

### D04: Framework Comparison

**Compare PyTorch, JAX, and TensorFlow.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/03_deep_learning/D04_framework_comparison.py)

---

## Transfer Learning

Adapt trained models to new instruments or conditions.

### D01: Transfer Analysis

**Analyze instrument transfer challenges.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/04_transfer_learning/D01_transfer_analysis.py)

### D02: Retrain Modes

**Strategies for model adaptation.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/04_transfer_learning/D02_retrain_modes.py)

```python
# Direct transfer: Apply model without adaptation
predictions = predictor.predict(model_id, new_instrument_data)

# Retrain last layers
predictor.retrain(
    model_id,
    new_data,
    mode="finetune",      # or "head_only", "full"
    epochs=10
)
```

### D03: PCA Geometry

**Analyze spectral space differences.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/04_transfer_learning/D03_pca_geometry.py)

---

## Advanced Features

Advanced data handling and transformation features.

### D01: Metadata Branching

**Branch based on sample metadata.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/05_advanced_features/D01_metadata_branching.py)

### D02: Concat Transform

**Concatenation transforms for multi-source data.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/05_advanced_features/D02_concat_transform.py)

### D03: Repetition Transform

**Repetition-based transforms.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/05_advanced_features/D03_repetition_transform.py)

### Creating Custom Transforms

```python
from sklearn.base import TransformerMixin, BaseEstimator

class MyTransform(TransformerMixin, BaseEstimator):
    def __init__(self, param=1.0):
        self.param = param

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.param

# Use in pipeline
pipeline = [MyTransform(param=2.0), PLSRegression()]
```

---

## Internals

Extend NIRS4ALL at the deepest level with custom controllers and session management.

### D01: Session Workflow

**Understanding execution flow.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/06_internals/D01_session_workflow.py)

```python
# Pipeline execution flow:
# 1. PipelineRunner.run() creates PipelineOrchestrator
# 2. Pipeline expands generators
# 3. For each variant:
#    a. Execute preprocessing steps
#    b. Execute splitter (CV)
#    c. For each fold:
#       - Execute model training
#       - Collect predictions
# 4. Aggregate results
```

### D02: Custom Controllers

**Extend NIRS4ALL with custom step handlers.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/06_internals/D02_custom_controllers.py)

```python
from nirs4all.controllers import register_controller, OperatorController

@register_controller
class MyController(OperatorController):
    priority = 50  # Lower = higher priority

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return keyword == "my_custom_step"

    @classmethod
    def use_multi_source(cls) -> bool:
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True  # Run during prediction

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        # Custom logic
        return context, output
```

---

## Running Developer Examples

```bash
cd examples

# Run all developer examples
./run.sh -c developer

# Run specific section
./run.sh -n "D01*.py" -c developer

# Run only generator examples (D01-D04)
./run.sh -n "D0[1-4]*.py" -c developer

# Run synthetic data examples (D05-D09)
./run.sh -n "D0[5-9]*.py" -c developer

# Skip deep learning (faster)
./run.sh -c developer -q
```

## Prerequisites

Developer examples assume familiarity with:

- All user examples
- Python advanced concepts (decorators, metaclasses)
- Machine learning theory

## Next Steps

- Read the {doc}`/developer/architecture` guide
- Explore the {doc}`/api/modules`
- Contribute to NIRS4ALL on [GitHub](https://github.com/GBeurier/nirs4all)
