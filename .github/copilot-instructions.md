# GitHub Copilot Instructions for nirs4all

## Project Overview

nirs4all is a Python library for Near-Infrared Spectroscopy (NIRS) data analysis. It provides ML/DL pipelines for classification and regression using scikit-learn, TensorFlow, PyTorch, and JAX backends. The library features a declarative pipeline syntax, automated hyperparameter tuning via Optuna, and comprehensive visualization tools.

**Status**: Pre-1.0 (v0.5.x) - APIs may change; actively remove deprecated/dead code.

## Architecture

### Core Components

```
nirs4all/
├── pipeline/          # Pipeline execution engine
│   ├── runner.py      # PipelineRunner - main entry point
│   ├── config/        # PipelineConfigs, ExecutionContext
│   ├── execution/     # PipelineOrchestrator, step execution
│   ├── bundle/        # Export/import trained pipelines (.n4a)
│   └── storage/       # Artifacts, manifests, library
├── controllers/       # Step handlers (registry pattern)
│   ├── registry.py    # @register_controller decorator
│   ├── transforms/    # TransformerMixin controllers
│   ├── models/        # Model training controllers
│   └── splitters/     # Cross-validation controllers
├── data/              # Dataset handling
│   ├── config.py      # DatasetConfigs
│   ├── dataset.py     # SpectroDataset
│   └── predictions.py # Predictions result container
├── operators/         # Pipeline operators
│   ├── transforms/    # NIRS-specific transformers (SNV, MSC, etc.)
│   ├── augmentation/  # Data augmentation operators
│   ├── models/        # Pre-built models (nicon, decon)
│   └── splitters/     # Data splitting methods (KS, SPXY)
└── visualization/     # PredictionAnalyzer, charts
```

### Key Patterns

**Controller Registry**: Pipeline steps are dispatched via a priority-based registry. Controllers inherit from `OperatorController` and use `@register_controller`:

```python
from nirs4all.controllers import register_controller, OperatorController

@register_controller
class MyController(OperatorController):
    priority = 50  # Lower = higher priority

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return isinstance(operator, MyOperatorType)

    def execute(self, step_info, dataset, context, runtime_context, ...):
        # Implementation
```

**Pipeline Syntax**: Multiple equivalent syntaxes normalize to canonical form during serialization:

```python
# All valid step formats:
MinMaxScaler                              # Class
MinMaxScaler()                            # Instance
{"preprocessing": MinMaxScaler()}         # Dict wrapper
{"class": "sklearn...MinMaxScaler"}       # Explicit class path
{"model": PLSRegression(n_components=10)} # Model step
{"_or_": [A, B, C], "count": 5}           # Generator (expands to variants)
```

**SpectroDataset**: Core data container holding X (features), y (targets), metadata, and fold indices. All controllers operate on this.

## Development Workflow

### Running Examples (Integration Tests)

```bash
cd examples
./run.sh                  # Run all examples
./run.sh -c user          # Run only user examples
./run.sh -c developer     # Run only developer examples
./run.sh -i 1             # Run single example by index
./run.sh -n "U01*.py"     # Run by name pattern (matches in any folder)
./run.sh -n "synthetic"   # Run examples containing "synthetic"
./run.sh -l               # Enable logging to log.txt
./run.sh -p -s            # Enable plots and show
./run.sh -q               # Quick mode: skip deep learning examples
```

Examples save outputs (plots, summaries) to `workspace/examples_output/`.

### Running Tests

```bash
pytest tests/                    # All tests
pytest tests/unit/               # Unit tests only
pytest tests/integration/        # Integration tests
pytest tests/unit/data/ -v       # Specific module
pytest --cov=nirs4all            # With coverage
```

Tests use `matplotlib.use('Agg')` backend (see `tests/conftest.py`).

### Key Test Markers

```bash
pytest -m sklearn      # sklearn-only tests
pytest -m tensorflow   # TensorFlow tests
pytest -m torch        # PyTorch tests
```

## Coding Conventions

- **Python 3.11+** required
- **Google Style Docstrings** for all public functions
- **PEP 8** with `max-line-length = 220` (see pyproject.toml)
- **TransformerMixin pattern** for custom transformers (sklearn-compatible)
- Prefer existing libraries; avoid reinventing

### Example Docstring

```python
def process_spectra(X: np.ndarray, method: str = "snv") -> np.ndarray:
    """Apply preprocessing to spectral data.

    Args:
        X: Spectral data matrix (n_samples, n_wavelengths).
        method: Preprocessing method name.

    Returns:
        Preprocessed spectral data.

    Raises:
        ValueError: If method is not supported.
    """
```

### Creating Custom Transformers

Follow sklearn's TransformerMixin pattern:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyTransform(TransformerMixin, BaseEstimator):
    def __init__(self, param=1.0, *, copy=True):
        self.param = param
        self.copy = copy

    def fit(self, X, y=None):
        # Fit logic (or just return self)
        return self

    def transform(self, X):
        # Transform logic
        return X_transformed
```

## Pipeline Configuration

### Basic Pipeline Structure

```python
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.data import DatasetConfigs

pipeline = [
    MinMaxScaler(),                           # Feature scaling
    {"y_processing": MinMaxScaler()},         # Target scaling
    {"feature_augmentation": {...}},          # Optional: generate variants
    ShuffleSplit(n_splits=3),                 # Cross-validation
    {"model": PLSRegression(n_components=10)} # Model
]

runner = PipelineRunner(verbose=1, save_artifacts=True)
predictions, per_dataset = runner.run(
    PipelineConfigs(pipeline, "MyPipeline"),
    DatasetConfigs("path/to/data")
)
```

### Generator Syntax for Hyperparameter Sweep

```python
# _or_ expands to multiple pipelines
{"_or_": [Detrend, SNV, Gaussian], "count": 5}

# _range_ sweeps parameter values
{"_range_": [1, 30, 5], "param": "n_components", "model": PLSRegression}
```

### Branching and Merging

```python
# Create parallel branches with different preprocessing
{"branch": [
    [SNV(), PLSRegression(n_components=10)],      # Branch 0
    [MSC(), RandomForestRegressor()],              # Branch 1
]}

# Merge: Exit branch mode and combine outputs
{"merge": "features"}      # Collect features from all branches
{"merge": "predictions"}   # Collect OOF predictions (stacking)
{"merge": {"features": [0], "predictions": [1]}}  # Mixed merge

# Source branching: Per-source preprocessing (multi-source datasets)
{"source_branch": {
    "NIR": [SNV(), FirstDerivative()],
    "markers": [VarianceThreshold()],
}}

# Source merging: Combine multi-source features
{"merge_sources": "concat"}   # Horizontal concatenation
{"merge_sources": "stack"}    # 3D stacking
```

**Key concepts**:
- `branch` creates parallel execution paths (N branches → each step runs N times)
- `merge` ALWAYS exits branch mode (returns to single-path execution)
- Prediction merging uses OOF reconstruction by default (prevents data leakage)
- `source_branch` processes each data source with its own pipeline
- `merge_sources` combines features from different data sources

See [docs/specifications/merge_syntax.md](docs/specifications/merge_syntax.md) for full reference.

## File Organization

- `examples/user/` - User-facing examples organized by topic
  - `01_getting_started/` - U01-U04: Hello world, regression, classification, visualization
  - `02_data_handling/` - U01-U06: Inputs, multi-datasets, multi-source, wavelengths, synthetic
  - `03_preprocessing/` - U01-U04: Basics, feature/sample augmentation, signal conversion
  - `04_models/` - U01-U04: Multi-model, tuning, stacking, PLS variants
  - `05_cross_validation/` - U01-U04: CV strategies, group splitting, filtering, aggregation
  - `06_deployment/` - U01-U04: Save/load, export bundles, workspace, sklearn integration
  - `07_explainability/` - U01-U03: SHAP basics, sklearn SHAP, feature selection
- `examples/developer/` - Advanced developer examples
  - `01_advanced_pipelines/` - D01-D05: Branching, merging, meta-stacking
  - `02_generators/` - D01-D06: Generator syntax, synthetic data customization
  - `03_deep_learning/` - D01-D04: PyTorch, JAX, TensorFlow, comparisons
  - `04_transfer_learning/` - D01-D03: Transfer analysis, retraining, PCA geometry
  - `05_advanced_features/` - D01-D03: Metadata branching, transforms
  - `06_internals/` - D01-D02: Session workflow, custom controllers
- `examples/reference/` - R01-R04: Comprehensive reference examples
- `examples/legacy/` - Q*/X* examples (deprecated, for transition)
- `docs/specifications/` - Pipeline syntax, config format specs
- `docs/user_guide/` - Preprocessing guides, cheatsheets
- `workspace/` - Default output directory (runs/, logs/, manifests, examples_output/)
- `exports/` - Exported model bundles (.n4a format)

## Key Files Reference

| File | Purpose |
|------|---------|
| [nirs4all/pipeline/runner.py](nirs4all/pipeline/runner.py) | Main `PipelineRunner` class |
| [nirs4all/controllers/registry.py](nirs4all/controllers/registry.py) | Controller registration |
| [nirs4all/controllers/data/merge.py](nirs4all/controllers/data/merge.py) | Merge controller (branch combination) |
| [nirs4all/controllers/data/source_branch.py](nirs4all/controllers/data/source_branch.py) | Source branch controller |
| [nirs4all/data/config.py](nirs4all/data/config.py) | `DatasetConfigs` class |
| [nirs4all/operators/transforms/nirs.py](nirs4all/operators/transforms/nirs.py) | NIRS-specific transforms |
| [docs/specifications/pipeline_syntax.md](docs/specifications/pipeline_syntax.md) | Full pipeline syntax reference |
| [docs/specifications/merge_syntax.md](docs/specifications/merge_syntax.md) | Merge and source branch syntax |

## Important Notes

- After any refactoring or API change, update: examples, docs, and tests
- The `.venv` contains all dependencies; work within this environment
- Use `nirs4all --test-install` to verify installation
- Tuples in configs are converted to lists during YAML serialization
- Controllers with `supports_prediction_mode() = True` run during prediction
