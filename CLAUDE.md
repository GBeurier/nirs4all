````markdown
# AI Instructions (Claude Code + GitHub Copilot) — nirs4all Workspace

This repository/workspace contains two related projects for Near-Infrared Spectroscopy (NIRS) analysis:

1. **nirs4all** (`/home/delete/nirs4all`) — Python library for NIRS data analysis with ML pipelines
2. **nirs4all_webapp** (`/home/delete/nirs_ui_workspace/nirs4all_webapp`) — Desktop/web app: React frontend + FastAPI backend

---

## nirs4all (Python Library)

**Version**: 0.6.x | **Python**: 3.11+ | **License**: CeCILL-2.1

### Quick Reference (Minimal Example)

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

result = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(10)],
    dataset="sample_data/regression",
    verbose=1,
)
print(f"Best RMSE: {result.best_rmse:.4f}")
````

### Primary API (Module-Level)

Prefer the module-level API functions:

| Function                               | Purpose                 |
| -------------------------------------- | ----------------------- |
| `nirs4all.run(pipeline, dataset, ...)` | Train a pipeline        |
| `nirs4all.predict(model, data, ...)`   | Make predictions        |
| `nirs4all.explain(model, data, ...)`   | SHAP explanations       |
| `nirs4all.retrain(source, data, ...)`  | Retrain on new data     |
| `nirs4all.session(...)`                | Create reusable session |
| `nirs4all.generate(...)`               | Generate synthetic data |

**Result objects**: `RunResult`, `PredictResult`, `ExplainResult` expose `best_score`, `best_rmse`, `best_r2`, `top(n)`, `export()`.

### Commands

```bash
# Tests
pytest tests/                     # All tests
pytest tests/unit/                # Unit tests only
pytest tests/integration/         # Integration tests
pytest -m sklearn                 # sklearn-only tests
pytest --cov=nirs4all             # With coverage

# Examples (from examples/ directory)
./run.sh                          # All examples
./run.sh -c user                  # User examples only
./run.sh -n "U01*"                # By pattern
./run.sh -q                       # Quick (skip deep learning)

# Code quality
ruff check .                      # Linting
mypy .                            # Type checking

# Installation verification
nirs4all --test-install
nirs4all --test-integration
```

### Architecture

```
nirs4all/
├── api/           # Primary interface: run(), predict(), explain(), retrain(), session(), generate()
├── pipeline/      # Execution engine (PipelineRunner/Orchestrator), bundle export (.n4a), prediction, retraining
├── controllers/   # Registry pattern for step handlers (@register_controller)
├── data/          # SpectroDataset (core container with X, y, metadata, folds)
├── operators/     # Transforms (SNV, MSC, SG), models (NICON), splitters (KS, SPXY), augmentation
├── sklearn/       # NIRSPipeline wrapper for SHAP compatibility
└── visualization/ # PredictionAnalyzer, heatmaps, candlestick charts
```

**Key classes**: `SpectroDataset`, `PipelineConfigs`, `DatasetConfigs`, `NIRSPipeline`

### Pipeline Syntax

Steps can be classes, instances, or wrapped in dicts:

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

pipeline = [
    MinMaxScaler(),                              # Transformer instance
    {"y_processing": MinMaxScaler()},            # Target scaling
    ShuffleSplit(n_splits=3),                    # Cross-validation splitter
    {"model": PLSRegression(n_components=10)},   # Model step
]
```

#### Special Keywords

| Keyword        | Purpose                     | Example                                                          |
| -------------- | --------------------------- | ---------------------------------------------------------------- |
| `model`        | Define model step           | `{"model": PLSRegression(10)}`                                   |
| `y_processing` | Target scaling              | `{"y_processing": MinMaxScaler()}`                               |
| `tag`          | Mark samples (non-removal)  | `{"tag": YOutlierFilter()}`                                      |
| `exclude`      | Exclude samples (training)  | `{"exclude": [YOutlierFilter(), XOutlierFilter()], "mode": "any"}` |
| `branch`       | Parallel/separation branches| `{"branch": [[SNV(), PLS()], [MSC(), RF()]]}`                    |
| `merge`        | Combine branches (stacking) | `{"merge": "predictions"}` or `{"merge": "concat"}`              |
| `_or_`         | Generator (variants)        | `{"_or_": [SNV, MSC, Detrend]}`                                  |
| `_range_`      | Parameter sweep             | `{"_range_": [1, 30, 5], "param": "n_components"}`               |

#### Tagging and Exclusion (v2)

Mark samples with tags (for analysis) or exclude them from training:

```python
from nirs4all.operators.filters import YOutlierFilter, XOutlierFilter

# Tag samples without removing (for downstream branching/analysis)
{"tag": YOutlierFilter(method="iqr")},

# Exclude outliers from training (mode: "any" or "all" for multiple)
{"exclude": YOutlierFilter(method="iqr")},
{"exclude": [YOutlierFilter(), XOutlierFilter(method="mahalanobis")], "mode": "any"},
```

#### Branching (v2)

Two branch modes: **duplication** (parallel pipelines) and **separation** (disjoint sample subsets):

```python
# Duplication branches - same samples, different pipelines
{"branch": [
    [SNV(), PLSRegression(10)],
    [MSC(), RandomForestRegressor()],
]},

# Separation branches - disjoint sample groups
{"branch": {"by_metadata": "site"}},                      # By metadata column
{"branch": {"by_tag": "y_outlier_iqr", "values": {       # By tag values
    "clean": False, "outliers": True
}}},
{"branch": {"by_source": True, "steps": {                # Per-source preprocessing
    "NIR": [SNV(), PLSRegression(10)],
    "markers": [MinMaxScaler(), Ridge()],
}}},
```

#### Merging (v2)

Merge strategies for different branch types:

```python
# For duplication branches (stacking)
{"merge": "predictions"},   # OOF predictions as features
{"merge": "features"},      # Transformed features as features

# For separation branches (reassembly)
{"merge": "concat"},        # Reassemble samples in original order

# For multi-source merging
{"merge": {"sources": "concat"}},  # Unified source merge syntax
```

### Controller Pattern (Registry)

Custom operators should follow the controller registry pattern:

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
        # Transform dataset; return (context, StepOutput)
        pass
```

### Common Tasks

#### Generate Synthetic Data

```python
import nirs4all

dataset = nirs4all.generate(n_samples=500, complexity="realistic")
# Or specialized generators:
dataset = nirs4all.generate.regression(n_samples=500)
dataset = nirs4all.generate.classification(n_samples=300, n_classes=3)
```

#### Export / Load Models

```python
import nirs4all
from nirs4all.sklearn import NIRSPipeline

# Export
result.export("model.n4a")

# Load and predict
preds = nirs4all.predict("model.n4a", new_data)

# sklearn wrapper for SHAP
model = NIRSPipeline.from_bundle("model.n4a")
```

#### Stacking (Meta-models)

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

pipeline = [
    {"branch": [
        [SNV(), PLSRegression(10)],
        [MSC(), RandomForestRegressor()],
    ]},
    {"merge": "predictions"},  # OOF predictions as features
    {"model": Ridge()},        # Meta-model
]
```

### File Layout

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

### Code Style / Conventions

* **Docstrings**: Google style
* **Line length**: 220 (see `pyproject.toml`)
* **Linting**: Ruff (`ruff check .`)
* **Type hints**: Required for public APIs
* Use `.venv` when launching Python scripts and tests
* Deep learning backends (TensorFlow, PyTorch, JAX) are **lazy-loaded**
* YAML note: tuples may convert to lists during serialization
* Actively remove deprecated and dead code
* After API changes, run: `cd examples && ./run.sh -q`

---

## nirs4all_webapp (Desktop/Web Application)

**Node**: 22 | **Python**: 3.11+
**Frontend**: React 19 + Vite + TypeScript
**Backend**: FastAPI

### Commands

```bash
# Frontend
npm run dev                       # Vite dev server (port 5173)
npm run build                     # Production build
npm run lint                      # ESLint
npm run test                      # Vitest tests
npm run validate:nodes            # Validate node registry
npm run storybook                 # Component docs (port 6006)

# Backend
python -m uvicorn main:app --reload --port 8000

# Desktop mode
python launcher.py                # Production
VITE_DEV=true python launcher.py  # Development with hot reload

# Unified launcher
./launch.sh web:dev               # Web development mode
./launch.sh desktop:prod          # Desktop production mode
```

### Architecture

```
nirs4all_webapp/
├── src/                          # React frontend
│   ├── components/
│   │   ├── pipeline-editor/      # Drag-and-drop pipeline builder
│   │   │   ├── config/           # Step configuration renderers
│   │   │   ├── validation/       # Multi-level validation system
│   │   │   └── custom-nodes/     # User-defined operators
│   │   ├── ui/                   # shadcn/ui base components
│   │   └── [feature]/            # dashboard, datasets, pipelines, runs, playground, settings
│   ├── data/nodes/               # Node registry system (NodeRegistry.ts)
│   ├── context/                  # Theme, Language, Settings, Developer Mode providers
│   ├── api/                      # API client functions
│   └── lib/                      # i18n, utilities
├── api/                          # FastAPI backend
│   ├── workspace.py              # Workspace management
│   ├── datasets.py               # Dataset operations
│   ├── pipelines.py              # Pipeline CRUD & execution
│   ├── training.py               # Model training
│   ├── nirs4all_adapter.py       # Bridge to nirs4all library
│   └── jobs/                     # Background job queue
└── websocket/                    # Real-time updates (training progress)
```

**Data flow**: Frontend (React) ←→ Backend (FastAPI) ←→ nirs4all library

### Key Frontend Patterns

* **State**: TanStack Query for server state, React Context for app state
* **UI**: shadcn/ui with Radix primitives, Tailwind CSS (teal/cyan theme)
* **Path aliases**: `@` → `./src` (tsconfig)
* **Validation**: Zod schemas, multi-level pipeline validation
* Webapp can run **without nirs4all installed** (UI development)

---

## Workspace Architecture (Phase 7)

Clear separation between:

| Term                   | Location              | Purpose                                                       |
| ---------------------- | --------------------- | ------------------------------------------------------------- |
| **App Settings**       | `~/.nirs4all-webapp/` | Webapp-specific data (favorites, UI prefs, linked workspaces) |
| **nirs4all Workspace** | User-defined paths    | Analysis data (runs, exports, predictions)                    |

### nirs4all Workspace Structure

```
/user/nirs4all-workspace/
  workspace/
    runs/<dataset>/0001_xxx/manifest.yaml  # Run manifests with dataset_info
    binaries/<dataset>/                    # Content-addressed artifacts
    exports/<dataset>/                     # Exported pipelines (json, n4a, csv)
    library/templates/                     # Pipeline templates
  <dataset>.meta.parquet                   # Predictions database
```

### Key Backend Endpoints (Webapp)

```
# Linked workspace management
GET  /workspaces
POST /workspaces/link
POST /workspaces/{id}/scan
GET  /workspaces/{id}/runs
GET  /workspaces/{id}/exports

# App settings
GET  /app/settings
GET  /app/favorites

# Dataset versioning
GET  /datasets/{id}/run-compatibility
```

### Dataset Versioning

* Each run manifest includes `dataset_info` with hash and version
* Datasets track `version_history` array of previous versions
* Run-compatibility endpoint shows warnings for outdated runs

---

## Constraints

* Avoid over-engineering. Only make changes that are directly requested or clearly necessary. Keep solutions simple and focused.
* Don't add features, refactor code, or make "improvements" beyond what was asked. A bug fix doesn't need surrounding code cleaned up. A simple feature doesn't need extra configurability.
* Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs). Don't use backwards-compatibility shims when you can just change the code.
* Don't create helpers, utilities, or abstractions for one-time operations. Don't design for hypothetical future requirements. The right amount of complexity is the minimum needed for the current task. Reuse existing abstractions where possible and follow the DRY principle.
* Please write a high-quality, general-purpose solution using the standard tools available. Do not create helper scripts or workarounds to accomplish the task more efficiently. Implement a solution that works correctly for all valid inputs, not just the test cases. Do not hard-code values or create solutions that only work for specific test inputs. Instead, implement the actual logic that solves the problem generally.
* Focus on understanding the problem requirements and implementing the correct algorithm. Tests are there to verify correctness, not to define the solution. Provide a principled implementation that follows best practices and software design principles.
* ALWAYS read and understand relevant files before proposing code edits. Do not speculate about code you have not inspected. If the user references a specific file/path, you MUST open and inspect it before explaining or proposing fixes. Be rigorous and persistent in searching code for key facts. Thoroughly review the style, conventions, and abstractions of the codebase before implementing new features or abstractions.
* Investigate before answering: never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make any claims about code before investigating unless you are certain of the correct answer - give grounded and hallucination-free answers.
* Never keep dead code, obsolete code or deprecated code. I want a clean repository (no backward compatibility)

```
