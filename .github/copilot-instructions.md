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
./run.sh              # Run all examples
./run.sh -i 1         # Run single example by index
./run.sh -n Q1*.py    # Run by name pattern
./run.sh -l           # Enable logging to log.txt
./run.sh -p -s        # Enable plots and show
```

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

## File Organization

- `examples/Q*.py` - Numbered examples (serve as docs + integration tests)
- `docs/specifications/` - Pipeline syntax, config format specs
- `docs/user_guide/` - Preprocessing guides, cheatsheets
- `workspace/` - Default output directory (runs/, logs/, manifests)
- `exports/` - Exported model bundles (.n4a format)

## Key Files Reference

| File | Purpose |
|------|---------|
| [nirs4all/pipeline/runner.py](nirs4all/pipeline/runner.py) | Main `PipelineRunner` class |
| [nirs4all/controllers/registry.py](nirs4all/controllers/registry.py) | Controller registration |
| [nirs4all/data/config.py](nirs4all/data/config.py) | `DatasetConfigs` class |
| [nirs4all/operators/transforms/nirs.py](nirs4all/operators/transforms/nirs.py) | NIRS-specific transforms |
| [docs/specifications/pipeline_syntax.md](docs/specifications/pipeline_syntax.md) | Full pipeline syntax reference |

## Important Notes

- After any refactoring or API change, update: examples, docs, and tests
- The `.venv` contains all dependencies; work within this environment
- Use `nirs4all --test-install` to verify installation
- Tuples in configs are converted to lists during YAML serialization
- Controllers with `supports_prediction_mode() = True` run during prediction
