# nirs4all — Python NIRS Analysis Library

**Version**: 0.8.3 | **Python**: 3.11+ | **License**: CeCILL-2.1

## Commands

```bash
pytest tests/                     # All tests
pytest tests/unit/                # Unit only
pytest tests/integration/         # Integration only
pytest -m sklearn                 # sklearn-only
pytest --cov=nirs4all             # With coverage
ruff check .                      # Lint
mypy .                            # Type check
cd examples && ./run.sh           # All examples
cd examples && ./run.sh -c user   # User examples only
```

## Architecture

```
nirs4all/
├── api/            # Public interface: run(), predict(), explain(), retrain(), session(), generate()
├── pipeline/
│   ├── execution/  # PipelineOrchestrator (parallel n_jobs), PipelineExecutor, refit
│   ├── storage/    # WorkspaceStore (SQLite metadata), ArrayStore (Parquet arrays)
│   ├── steps/      # StepParser → ControllerRouter → StepRunner
│   ├── bundle/     # BundleGenerator/BundleLoader (.n4a export)
│   ├── config/     # PipelineConfigs, PipelineGenerator, ExecutionContext
│   ├── trace/      # ExecutionTrace, TraceBasedExtractor, MinimalPipeline
│   └── runner.py   # PipelineRunner (main orchestration entry)
├── controllers/    # Registry pattern: @register_controller → OperatorController subclasses
│   ├── transforms/ # TransformerMixinController, YTransformerMixinController
│   ├── models/     # SklearnModelController, PyTorchModelController, TF, JAX
│   ├── data/       # BranchController, MergeController, ExcludeController, TagController
│   ├── flow/       # RepetitionController, ConcatTransformController, AutoTransferPreprocController
│   ├── shared/     # ModelSelector, PredictionAggregator
│   ├── splitters/  # CrossValidatorController
│   └── charts/     # Spectra, targets, folds, augmentation charts
├── data/           # SpectroDataset (X, y, metadata, folds, multi-source), Predictions
│   ├── loaders/    # CSV, Parquet, Excel, NumPy, MATLAB
│   ├── parsers/    # FolderParser, ConfigNormalizer, schema validation
│   └── signal_type.py # SignalType enum, detection, conversion
├── operators/
│   ├── transforms/ # SNV, MSC, SavitzkyGolay, Detrend, WaveletDenoise, OSC, EPO, CARS, MCUVE...
│   ├── models/     # AOM-PLS, POP-PLS, PLSDA, IKPLS, OPLS, DiPLS, SparsePLS, LWPLS, KOPLS...
│   ├── splitters/  # KennardStone, SPXY, SPXYFold, KMeans, KBinsStratified
│   ├── filters/    # YOutlierFilter, XOutlierFilter, SpectralQualityFilter
│   └── augmentation/ # Noise, baseline, wavelength, spectral, mixup, physical, scatter, spline
├── config/         # CacheConfig, DatasetConfigs
├── sklearn/        # NIRSPipeline (SHAP-compatible sklearn wrapper)
├── visualization/  # PredictionAnalyzer, heatmaps, candlestick charts
├── synthesis/      # Synthetic data generation
├── workspace/      # Workspace management
└── analysis/       # Pipeline topology analysis
```

## Execution Flow

```
nirs4all.run(pipeline, dataset)
  → api/run.py: normalize inputs, create PipelineRunner
    → PipelineRunner.run(): load dataset, build PipelineConfigs
      → PipelineOrchestrator.execute(): expand generators, init WorkspaceStore
        → For each variant (parallel via joblib if n_jobs>1):
          → PipelineExecutor.execute(): iterate steps
            → StepRunner → StepParser.parse() → ControllerRouter.route() → controller.execute()
        → Refit best variant on full data
        → Save run metadata + predictions
      → Return RunResult (best_score, best_rmse, top(n), export())
```

## Pipeline Syntax

Steps are classes, instances, or dicts with keywords:

```python
pipeline = [
    MinMaxScaler(),                            # Transformer
    {"y_processing": MinMaxScaler()},          # Target scaling
    ShuffleSplit(n_splits=3),                  # Cross-validator
    {"model": PLSRegression(n_components=10)}, # Model
]
```

### Keywords

| Keyword | Purpose |
|---------|---------|
| `model` | Define model step |
| `y_processing` | Target scaling |
| `tag` | Mark samples (non-removal) |
| `exclude` | Remove samples from training (`mode`: "any"/"all" for multiple) |
| `branch` | Duplication branches (parallel pipelines) or separation branches (by_metadata/by_tag/by_filter/by_source) |
| `merge` | Combine branches: `"predictions"` (stacking), `"features"`, `"all"`, `"concat"` (reassembly) |
| `sample_augmentation` | Data augmentation applied to training samples |
| `feature_augmentation` | Feature-level augmentation |
| `concat_transform` | Concatenate transformed features |
| `rep_to_sources` | Convert repetition groups to multi-source format |
| `rep_to_pp` | Convert repetition groups to preprocessing pipelines |

### Generator Syntax (Hyperparameter Sweeps)

| Keyword | Effect | Example |
|---------|--------|---------|
| `_or_` | Try alternatives | `[SNV, MSC, Detrend]` |
| `_range_` | Linear sweep | `[1, 30, 5]` → `[1, 6, 11, 16, 21, 26]` |
| `_log_range_` | Log sweep | `[1e-3, 1e0, 4]` → `[0.001, 0.01, 0.1, 1.0]` |
| `_grid_` | Cartesian product of params | `{"n_components": [5,10], "alpha": [0.1,1.0]}` → 4 combos |
| `_cartesian_` | Pipeline stage combinations | `[{"_or_": [A,B]}, {"_or_": [X,Y]}]` → 4 pipelines |
| `_zip_` | Paired iteration | `{"x": [1,2,3], "y": ["a","b","c"]}` → 3 pairs |
| `_chain_` | Sequential ordered | `[config1, config2, config3]` |
| `_sample_` | Random sampling | `{"distribution": "log_uniform", "from": 1e-4, "to": 1e-1, "num": 20}` |

## Controller Pattern

All pipeline operators are dispatched through a controller registry:

```python
from nirs4all.controllers import register_controller, OperatorController

@register_controller
class MyController(OperatorController):
    priority = 50  # Lower = higher priority

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return isinstance(operator, MyType)

    @classmethod
    def use_multi_source(cls) -> bool:
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        # return (context, StepOutput)
        pass
```

## Storage (Workspace)

Hybrid SQLite + Parquet model:

```
workspace/
  store.sqlite                          # Metadata: runs, pipelines, chains, logs, artifacts
  arrays/<dataset_name>.parquet         # Prediction arrays (Zstd-compressed)
  artifacts/<hash>.joblib               # Content-addressed model binaries
  runs/<dataset>/0001_xxx/manifest.yaml # Run manifests
  exports/<dataset>/                    # Exported models/predictions
  library/templates/                    # Pipeline templates
```

- `WorkspaceStore` — SQLite WAL-backed, thread-safe, retry-on-lock
- `ArrayStore` — Parquet-backed prediction arrays with tombstone deletes
- `Predictions` — Facade combining metadata queries + array retrieval

## Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `SpectroDataset` | `data.dataset` | Core data container (X, y, metadata, folds, multi-source) |
| `DatasetConfigs` | `data.config` | Dataset configuration and loading |
| `PipelineConfigs` | `pipeline.config` | Pipeline step configuration |
| `PipelineRunner` | `pipeline.runner` | Main training orchestration |
| `PipelineOrchestrator` | `pipeline.execution.orchestrator` | Parallel variant execution |
| `PipelineExecutor` | `pipeline.execution.executor` | Single variant execution |
| `WorkspaceStore` | `pipeline.storage.workspace_store` | SQLite metadata store |
| `ArrayStore` | `pipeline.storage.array_store` | Parquet prediction arrays |
| `RunResult` | `api.result` | Training results (best_score, export()) |
| `NIRSPipeline` | `sklearn` | SHAP-compatible sklearn wrapper |
| `BundleGenerator` | `pipeline.bundle` | .n4a export |
| `BundleLoader` | `pipeline.bundle` | .n4a load + predict |

## Code Conventions

- Google-style docstrings
- Line length: 220 (`pyproject.toml`)
- Ruff for linting, mypy for types
- Type hints required for public APIs
- Use `.venv` for running scripts/tests
- Deep learning backends (TF, PyTorch, JAX) are lazy-loaded
- YAML: tuples may convert to lists during serialization
- No dead code, no deprecated code, no backward compatibility shims

## Constraints

- Read and understand files before proposing edits. Never speculate about unread code.
- No over-engineering: minimum complexity for the current task. No hypothetical features.
- No unnecessary error handling for impossible scenarios. Validate only at system boundaries.
- No helpers/utilities/abstractions for one-time operations. Reuse existing abstractions (DRY).
- Write general-purpose solutions, not hard-coded workarounds for specific test inputs.
- After API changes, run: `cd examples && ./run.sh`
