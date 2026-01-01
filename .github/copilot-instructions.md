# GitHub Copilot Instructions for nirs4all

## Quick Reference

**Version**: 0.6.x | **Python**: 3.11+ | **License**: CeCILL-2.1

```python
# Minimal working example
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

result = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(10)],
    dataset="sample_data/regression",
    verbose=1
)
print(f"Best RMSE: {result.best_rmse:.4f}")
```

## Primary API

The **module-level API** is the primary interface. Use these functions:

| Function | Purpose |
|----------|---------|
| `nirs4all.run(pipeline, dataset, ...)` | Train a pipeline |
| `nirs4all.predict(model, data, ...)` | Make predictions |
| `nirs4all.explain(model, data, ...)` | SHAP explanations |
| `nirs4all.retrain(source, data, ...)` | Retrain on new data |
| `nirs4all.session(...)` | Create reusable session |
| `nirs4all.generate(...)` | Generate synthetic data |

**Result objects**: `RunResult`, `PredictResult`, `ExplainResult` provide `best_score`, `best_rmse`, `best_r2`, `top(n)`, `export()`.

## Architecture

```
nirs4all/
├── api/               # Module-level API (run, predict, explain, generate)
├── pipeline/          # Execution engine (PipelineRunner, PipelineOrchestrator)
├── controllers/       # Step handlers (registry pattern, @register_controller)
├── data/              # SpectroDataset, DatasetConfigs, Predictions
├── operators/         # Transforms (SNV, MSC), models (nicon), splitters (KS, SPXY)
├── sklearn/           # NIRSPipeline sklearn wrapper for SHAP compatibility
└── visualization/     # PredictionAnalyzer, charts
```

### Key Classes

- **`SpectroDataset`**: Core data container with X (features), y (targets), metadata, folds
- **`PipelineConfigs`**: Pipeline definition wrapper
- **`DatasetConfigs`**: Dataset path/configuration wrapper
- **`NIRSPipeline`**: sklearn-compatible wrapper for trained models

## Pipeline Syntax

```python
pipeline = [
    # Steps can be classes, instances, or wrapped in dicts
    MinMaxScaler(),                              # Transformer instance
    {"y_processing": MinMaxScaler()},            # Target scaling
    ShuffleSplit(n_splits=3),                    # Cross-validation splitter
    {"model": PLSRegression(n_components=10)},   # Model step
]
```

### Special Keywords

| Keyword | Purpose | Example |
|---------|---------|---------|
| `model` | Define model step | `{"model": PLSRegression(10)}` |
| `y_processing` | Target scaling | `{"y_processing": MinMaxScaler()}` |
| `branch` | Parallel pipelines | `{"branch": [[SNV(), PLS()], [MSC(), RF()]]}` |
| `merge` | Combine branches | `{"merge": "predictions"}` |
| `source_branch` | Per-source preprocessing | `{"source_branch": {"NIR": [...], "markers": [...]}}` |
| `_or_` | Generator (variants) | `{"_or_": [SNV, MSC, Detrend]}` |
| `_range_` | Parameter sweep | `{"_range_": [1, 30, 5], "param": "n_components"}` |

## Development Commands

```bash
# Examples (from examples/ directory)
./run.sh                  # All examples
./run.sh -c user          # User examples only
./run.sh -n "U01*"        # By pattern
./run.sh -q               # Quick (skip DL)

# Tests
pytest tests/             # All tests
pytest tests/unit/        # Unit only
pytest -m sklearn         # sklearn-only
pytest --cov=nirs4all     # With coverage

# Verify installation
nirs4all --test-install
```

## Code Style

- **Docstrings**: Google style
- **Line length**: 220 (see pyproject.toml)
- **Linting**: Ruff (`ruff check .`)
- **Type hints**: Required for public APIs

## Controller Pattern

Custom operators use the registry pattern:

```python
from nirs4all.controllers import register_controller, OperatorController

@register_controller
class MyController(OperatorController):
    priority = 50  # Lower = higher priority

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return isinstance(operator, MyOperatorType)

    @classmethod
    def use_multi_source(cls) -> bool:
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True  # Run during prediction

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        # Transform dataset, return (context, StepOutput)
        pass
```

## Common Tasks

### Generate Synthetic Data
```python
dataset = nirs4all.generate(n_samples=500, complexity="realistic")
# Or specialized generators:
dataset = nirs4all.generate.regression(n_samples=500)
dataset = nirs4all.generate.classification(n_samples=300, n_classes=3)
```

### Export/Load Models
```python
# Export
result.export("model.n4a")

# Load and predict
preds = nirs4all.predict("model.n4a", new_data)

# sklearn wrapper for SHAP
from nirs4all.sklearn import NIRSPipeline
model = NIRSPipeline.from_bundle("model.n4a")
```

### Stacking (Meta-models)
```python
pipeline = [
    {"branch": [
        [SNV(), PLSRegression(10)],
        [MSC(), RandomForestRegressor()],
    ]},
    {"merge": "predictions"},  # OOF predictions as features
    {"model": Ridge()},        # Meta-model
]
```

## File Layout

```
examples/
├── user/           # By topic: getting_started, data_handling, preprocessing, etc.
├── developer/      # Advanced: branching, generators, deep_learning, internals
└── reference/      # Comprehensive reference examples

tests/
├── unit/           # Fast isolated tests
└── integration/    # Full pipeline tests

docs/source/        # Sphinx documentation
```

## Reminders

- Run examples after API changes: `cd examples && ./run.sh -q`
- Update RTD after significant changes
- Tuples → lists during YAML serialization
- Deep learning backends are lazy-loaded
- Actively remove deprecated and dead code.
- Use `.venv` when launching python scripts and tests.