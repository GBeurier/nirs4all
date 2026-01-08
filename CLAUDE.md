# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace Overview

This workspace contains two related projects for Near-Infrared Spectroscopy (NIRS) analysis:

1. **nirs4all** (`/home/delete/nirs4all`) - Python library for NIRS data analysis with ML pipelines
2. **nirs4all_webapp** (`/home/delete/nirs_ui_workspace/nirs4all_webapp`) - Desktop/web app with React frontend + FastAPI backend

---

## nirs4all (Python Library)

**Version**: 0.6.x | **Python**: 3.11+ | **License**: CeCILL-2.1

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
├── pipeline/      # Execution engine (PipelineRunner), bundle export (.n4a), prediction, retraining
├── controllers/   # Registry pattern for step handlers (@register_controller)
├── data/          # SpectroDataset (core container with X, y, metadata, folds)
├── operators/     # Transforms (SNV, MSC, SG), models (NICON), splitters (KS, SPXY), augmentation
├── sklearn/       # NIRSPipeline wrapper for SHAP compatibility
└── visualization/ # PredictionAnalyzer, heatmaps, candlestick charts
```

**Key classes**: `SpectroDataset` (data container), `PipelineConfigs`, `DatasetConfigs`, `NIRSPipeline` (sklearn wrapper)

### Pipeline Syntax

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

result = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(10)],
    dataset="sample_data/regression",
    verbose=1
)
```

**Special keywords in pipeline dicts**:
- `model`: Define model step
- `y_processing`: Target scaling
- `branch`: Parallel pipelines
- `merge`: Combine branches (stacking)
- `source_branch`: Per-source preprocessing
- `_or_`: Generator for variants
- `_range_`: Parameter sweep

### Controller Pattern

```python
from nirs4all.controllers import register_controller, OperatorController

@register_controller
class MyController(OperatorController):
    priority = 50  # Lower = higher priority

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return isinstance(operator, MyOperatorType)

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        pass
```

### Code Style

- **Docstrings**: Google style
- **Line length**: 220 (configured in pyproject.toml)
- **Linting**: Ruff
- **Type hints**: Required for public APIs
- Use `.venv` when launching python scripts and tests

---

## nirs4all_webapp (Desktop/Web Application)

**Node**: 22 | **Python**: 3.11+ | **Frontend**: React 19 + Vite + TypeScript | **Backend**: FastAPI

### Commands

```bash
# Frontend
npm run dev                       # Vite dev server (port 5173)
npm run build                     # Production build
npm run lint                      # ESLint
npm run test                      # Vitest tests
npm run validate:nodes            # Validate node registry

# Backend
python -m uvicorn main:app --reload --port 8000

# Desktop mode
python launcher.py                # Production
VITE_DEV=true python launcher.py  # Development with hot reload

# Unified launcher
./launch.sh web:dev               # Web development mode
./launch.sh desktop:prod          # Desktop production mode

# Storybook (component docs)
npm run storybook                 # Port 6006
```

### Architecture

```
nirs4all_webapp/
├── src/                          # React frontend
│   ├── components/
│   │   ├── pipeline-editor/      # Drag-and-drop pipeline builder (24 files)
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

**Data flow**: Frontend (React) ←→ Backend (FastAPI) ←→ nirs4all Library

### Key Frontend Patterns

- **State**: TanStack Query for server state, React Context for app state
- **Components**: shadcn/ui with Radix primitives, Tailwind CSS (teal/cyan theme)
- **Path aliases**: `@` → `./src` (configured in tsconfig.json)
- **Validation**: Zod schemas, multi-level pipeline validation

---

## Workspace Architecture (Phase 7)

The system has a clear separation between:

| Term | Location | Purpose |
|------|----------|---------|
| **App Settings** | `~/.nirs4all-webapp/` | Webapp-specific data (favorites, UI prefs, linked workspaces) |
| **nirs4all Workspace** | User-defined paths | Analysis data (runs, exports, predictions) |

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

### Key API Endpoints (webapp)

```
# Linked workspace management
GET  /workspaces                    # List linked workspaces
POST /workspaces/link               # Link a nirs4all workspace
POST /workspaces/{id}/scan          # Discover runs/exports/predictions
GET  /workspaces/{id}/runs          # Get discovered runs
GET  /workspaces/{id}/exports       # Get discovered exports

# App settings
GET  /app/settings                  # Get app settings
GET  /app/favorites                 # Get favorite pipelines

# Dataset versioning
GET  /datasets/{id}/run-compatibility  # Check which runs used which dataset version
```

### Dataset Versioning

- Each run's manifest includes `dataset_info` with hash and version
- Datasets track `version_history` array of previous versions
- Run-compatibility endpoint shows warnings for outdated runs

---

## Development Notes

- Deep learning backends (TensorFlow, PyTorch, JAX) are lazy-loaded
- Run examples after API changes: `cd examples && ./run.sh -q`
- Tuples convert to lists during YAML serialization
- Actively remove deprecated and dead code
- The webapp can run without nirs4all installed (for UI development)
