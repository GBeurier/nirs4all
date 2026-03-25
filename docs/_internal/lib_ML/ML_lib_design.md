# ML Library Design — High-Level Specification

> Design for a general-purpose ML pipeline library extracted from nirs4all's core engine, enabling complex ML/DL workflows with any type of data.

---

## 1. Vision and Positioning

### Name Suggestion: **pipeforge** (or **mlforge**, **flowpipe**, **pipecraft**)

A Python library for building, executing, and managing complex ML/DL pipelines with:
- Declarative pipeline syntax (list-based, no boilerplate)
- Automatic cross-validation, refit, and out-of-fold patterns
- Branching, merging, and stacking as first-class concepts
- Built-in hyperparameter sweep generation
- Full lifecycle: train → predict → explain → retrain
- Persistent workspace with artifact tracking
- Plugin architecture for domain-specific operators

### Differentiators vs Existing Tools

| Feature | pipeforge | sklearn Pipeline | MLflow | Optuna | MetaFlow |
|---------|-----------|-----------------|--------|--------|----------|
| Declarative list syntax | Yes | Partial (named steps) | No | No | No |
| Auto CV + refit | Yes | No (manual) | No | No | No |
| OOF stacking | Yes (built-in) | No | No | No | No |
| Branch/merge/separate | Yes | No | No | No | No |
| Variant generation | Yes (built-in) | No | No | Yes (different API) | No |
| Artifact persistence | Yes (DuckDB+Parquet) | No | Yes | No | Yes |
| Bundle export | Yes (.n4a) | joblib only | Yes | No | No |
| Execution trace | Yes (minimal pipeline) | No | No | No | No |
| Multi-framework | Yes (sklearn, TF, PyTorch, JAX) | sklearn only | Yes | Framework-agnostic | Yes |
| Multi-source data | Yes | No | No | No | No |

### Target Users
- ML practitioners building complex pipelines with custom preprocessing
- Researchers running systematic experiments with many configurations
- Domain specialists (spectroscopy, medical imaging, NLP, etc.) via plugins
- Teams needing reproducible, auditable experiment tracking

---

## 2. Core Abstractions

### 2.1 Dataset — Generic Data Container

Replace `SpectroDataset` with a domain-agnostic `Dataset`:

```python
from pipeforge import Dataset

# Tabular data
ds = Dataset.from_arrays(X_train, y_train, X_test=X_test, y_test=y_test)

# From files
ds = Dataset.from_csv("data/train.csv", target_column="price")
ds = Dataset.from_parquet("data/features.parquet", targets="data/labels.parquet")

# Multi-source (sensor fusion, multimodal)
ds = Dataset()
ds.add_source("tabular", X_tabular, feature_names=["age", "weight", ...])
ds.add_source("images", X_images, shape=(224, 224, 3))
ds.add_source("text", X_text, tokenizer=my_tokenizer)

# Metadata
ds.add_metadata(metadata_df)
ds.set_repetition("patient_id")   # Group repeated measurements
ds.set_aggregate("patient_id", method="median")
```

#### Internal Architecture

```
Dataset
├── SourceRegistry          # Named feature sources (multi-modal support)
│   ├── Source("tabular")   # Tabular features (np.ndarray / pd.DataFrame)
│   ├── Source("images")    # Image features (np.ndarray with spatial dims)
│   └── Source("text")      # Text features (tokenized sequences)
├── Targets                 # Labels/regression targets
├── Metadata                # Sample-level auxiliary data
├── Indexer                 # Row index management
│   ├── partitions          # train/test/val splits
│   ├── folds               # CV fold assignments
│   ├── exclusions          # Excluded sample tracking
│   ├── tags                # Sample tags (non-destructive marking)
│   └── augmented_origins   # Augmentation tracking
└── Config                  # Task type, repetition, aggregation settings
```

**Key Design Decisions:**
- Sources are named and typed (tabular, image, sequence, graph, etc.)
- Each source declares its shape/schema independently
- Core `Dataset` is data-type agnostic — source plugins handle type-specific logic
- The `Indexer` manages all sample-level state (partitions, folds, exclusions, tags)
- `Targets` remain simple (numpy arrays) — no domain-specific semantics

### 2.2 Pipeline — Declarative Step List

```python
from pipeforge import run
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

result = run(
    pipeline=[
        StandardScaler(),
        KFold(n_splits=5),
        {"model": RandomForestRegressor(n_estimators=100)},
    ],
    dataset=ds,
    n_jobs=-1,
    verbose=1,
)
```

The pipeline syntax remains identical to nirs4all — this is a proven, ergonomic design.

### 2.3 Controller — Plugin Interface

```python
from pipeforge.controllers import OperatorController, register_controller

@register_controller
class ImageAugmentationController(OperatorController):
    priority = 15

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return isinstance(operator, ImageAugmentation)

    @classmethod
    def use_multi_source(cls) -> bool:
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return False  # Augmentation is training-only

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        source = dataset.get_source("images")
        augmented = operator.augment(source.data)
        dataset.add_augmented_samples(augmented, origin=operator.name)
        return context, StepOutput(...)
```

---

## 3. Package Structure

```
pipeforge/
├── __init__.py                    # Public API: run(), predict(), explain(), retrain(), session()
├── api/                           # High-level API functions
│   ├── run.py                     # run() entry point
│   ├── predict.py                 # predict() entry point
│   ├── explain.py                 # explain() entry point
│   ├── retrain.py                 # retrain() entry point
│   ├── session.py                 # session() / load_session()
│   └── result.py                  # RunResult, PredictResult, ExplainResult
│
├── data/                          # Data model
│   ├── dataset.py                 # Dataset (generic container)
│   ├── source.py                  # Source, SourceRegistry (multi-modal)
│   ├── targets.py                 # Targets
│   ├── metadata.py                # Metadata
│   ├── indexer.py                 # Indexer (partitions, folds, tags, exclusions)
│   ├── predictions.py             # Predictions (store-backed)
│   ├── config.py                  # DatasetConfig
│   └── loaders/                   # File loaders (CSV, Parquet, NumPy, etc.)
│
├── pipeline/                      # Execution engine
│   ├── runner.py                  # PipelineRunner (facade)
│   ├── execution/                 # Core execution
│   │   ├── orchestrator.py        # PipelineOrchestrator
│   │   ├── executor.py            # PipelineExecutor
│   │   ├── builder.py             # ExecutorBuilder
│   │   ├── step_cache.py          # Step caching
│   │   └── refit/                 # Refit strategies
│   │       ├── simple.py
│   │       ├── per_model.py
│   │       ├── stacking.py
│   │       ├── separation.py
│   │       └── competing.py
│   ├── steps/                     # Step execution
│   │   ├── step_runner.py
│   │   ├── parser.py
│   │   └── router.py
│   ├── config/                    # Pipeline configuration
│   │   ├── pipeline_config.py     # PipelineConfigs (variant expansion)
│   │   ├── context.py             # ExecutionContext
│   │   └── _generator/            # Variant generation strategies
│   │       └── strategies/
│   ├── trace/                     # Execution trace
│   │   ├── execution_trace.py
│   │   └── extractor.py
│   ├── storage/                   # Persistence
│   │   ├── workspace_store.py     # DuckDB metadata
│   │   ├── array_store.py         # Parquet prediction arrays
│   │   ├── artifacts/             # Artifact persistence
│   │   └── library.py             # Pipeline templates
│   ├── bundle/                    # Model export/import
│   │   ├── generator.py
│   │   └── loader.py
│   ├── predictor.py               # Prediction replay
│   ├── minimal_predictor.py       # Minimal pipeline prediction
│   ├── explainer.py               # SHAP explanations
│   └── retrainer.py               # Retraining system
│
├── controllers/                   # Built-in controllers
│   ├── controller.py              # OperatorController base class
│   ├── registry.py                # CONTROLLER_REGISTRY
│   ├── transforms/
│   │   ├── transformer.py         # TransformerMixinController
│   │   └── y_transformer.py       # YTransformerMixinController
│   ├── models/
│   │   ├── sklearn_model.py       # SklearnModelController
│   │   ├── tensorflow_model.py    # TensorFlowModelController
│   │   ├── torch_model.py         # PyTorchModelController
│   │   └── jax_model.py           # JaxModelController
│   ├── splitters/
│   │   ├── split.py               # CrossValidatorController
│   │   └── fold_file_loader.py    # External fold loading
│   ├── data/
│   │   ├── tag.py                 # TagController
│   │   ├── exclude.py             # ExcludeController
│   │   ├── branch.py              # BranchController
│   │   ├── merge.py               # MergeController
│   │   └── augmentation.py        # Generic augmentation controller
│   └── flow/
│       └── dummy.py               # DummyController (no-op)
│
├── operators/                     # Built-in generic operators
│   ├── filters/                   # Sample filters
│   │   ├── base.py                # SampleFilter ABC, CompositeFilter
│   │   ├── y_outlier.py           # YOutlierFilter
│   │   ├── x_outlier.py           # XOutlierFilter
│   │   └── metadata.py            # MetadataFilter
│   └── models/
│       └── meta.py                # MetaModel, StackingConfig
│
├── sklearn/                       # sklearn compatibility
│   ├── pipeline.py                # PipelineWrapper (generic)
│   └── classifier.py              # ClassifierWrapper
│
├── visualization/                 # Built-in visualization
│   ├── predictions.py             # PredictionAnalyzer
│   └── charts/                    # Chart implementations
│
├── config/                        # Global configuration
│   └── cache_config.py            # CacheConfig
│
└── plugins/                       # Plugin discovery and loading
    ├── __init__.py                # Plugin registry
    └── base.py                    # PluginBase (entry point spec)
```

---

## 4. Plugin Architecture

### Plugin Interface

```python
from pipeforge.plugins import PluginBase, register_plugin

class NIRSPlugin(PluginBase):
    """NIRS-specific operators and controllers for pipeforge."""

    name = "nirs"
    version = "1.0.0"

    def register(self, registry):
        """Called on plugin load — register controllers, operators, data types."""
        # Register NIRS-specific controllers
        registry.register_controller(SpectraChartController)
        registry.register_controller(ResamplerController)

        # Register source type
        registry.register_source_type("spectra", SpectralSource)

        # Register operators (for discoverability)
        registry.register_operators("nirs.transforms", [SNV, MSC, SavitzkyGolay, ...])
        registry.register_operators("nirs.augmentation", [WavelengthShift, ...])
        registry.register_operators("nirs.models", [AOMPLSRegressor, ...])
        registry.register_operators("nirs.splitters", [KennardStoneSplitter, ...])
```

### Plugin Discovery
- Entry point-based: `[project.entry-points."pipeforge.plugins"]` in pyproject.toml
- Explicit registration: `pipeforge.register_plugin(NIRSPlugin())`
- Plugin operators are usable in pipelines like any built-in operator

### Plugin Capabilities

| Capability | Interface | Example |
|-----------|-----------|---------|
| **Controllers** | `OperatorController` subclass | NIRS chart controllers, NLP tokenization controller |
| **Source Types** | `SourceType` subclass | Spectral data with wavelengths, image data with spatial dims |
| **Operators** | Any class with fit/transform or fit/predict | SNV, MSC, ImageResize, TextTokenizer |
| **Data Loaders** | `DataLoader` subclass | MATLAB loader, DICOM loader, HDF5 loader |
| **Metrics** | `Metric` callable | RPD, SEP (NIRS-specific), BLEU (NLP-specific) |
| **Visualizations** | Chart subclass | Spectral plots, attention maps, saliency maps |

---

## 5. Data Model Design

### 5.1 Source Types

```python
class Source:
    """Base source — holds feature data for one modality."""
    name: str
    data: np.ndarray          # Raw feature data
    feature_names: list[str]  # Optional column names
    shape_hint: tuple | None  # Per-sample shape (e.g., (224, 224, 3) for images)
    dtype: str                # "tabular", "image", "sequence", "graph", "custom"
    metadata: dict            # Source-specific metadata (extensible)
```

Plugins extend `Source` for domain-specific metadata:

```python
class SpectralSource(Source):
    """NIRS plugin — adds wavelength and signal type tracking."""
    dtype = "spectral"
    wavelengths: np.ndarray
    wavelength_unit: str       # "nm", "cm-1"
    signal_type: SignalType    # absorbance, reflectance, etc.
```

### 5.2 Data Access Pattern

```python
# Single source (most common)
X = dataset.X()                           # Default source
y = dataset.y()

# Multi-source
X_tab = dataset.X(source="tabular")
X_img = dataset.X(source="images")

# With selector (used by engine internally)
X = dataset.X(selector)                   # Respects partition, fold, exclusions, tags
```

### 5.3 Dataset Configuration

```python
from pipeforge import Dataset

ds = Dataset.from_config({
    "train_x": "data/train_features.csv",
    "train_y": "data/train_labels.csv",
    "test_x": "data/test_features.csv",
    "test_y": "data/test_labels.csv",
    "task_type": "regression",               # or "binary_classification", "multiclass_classification", "auto"
    "repetition": "sample_id",               # Group repeated measurements
    "aggregate": "sample_id",                # Aggregate predictions by group
    "aggregate_method": "median",            # mean, median, vote
})
```

---

## 6. API Design

### 6.1 Module-Level API (Primary Interface)

```python
import pipeforge

# Train
result = pipeforge.run(
    pipeline=[...],
    dataset=ds,               # Dataset, path, or config dict
    workspace="workspace/",   # Optional persistence directory
    n_jobs=-1,                # Parallel variant execution
    verbose=1,
    cache=CacheConfig(...),   # Optional caching
)

# Access results
print(result.best_score, result.best_rmse, result.best_r2)
result.top(5)                # Top 5 configurations
result.export("model.pfb")   # Export best model bundle (.pfb = pipeforge bundle)

# Predict
preds = pipeforge.predict("model.pfb", new_data)

# Explain
expl = pipeforge.explain("model.pfb", data)

# Retrain
result = pipeforge.retrain("model.pfb", new_data, mode="transfer")

# Session
session = pipeforge.session(pipeline=[...], workspace="workspace/")
result = session.run(ds)
preds = session.predict(new_data)
session.save("exports/session.pfb")
```

### 6.2 Pipeline Syntax (Unchanged from nirs4all)

```python
pipeline = [
    StandardScaler(),
    {"y_processing": RobustScaler()},
    KFold(n_splits=5),
    {"exclude": YOutlierFilter(method="iqr")},
    {"_or_": [PCA(n_components=10), SelectKBest(k=20)]},
    {"model": RandomForestRegressor(n_estimators={"_range_": [50, 500, 5]})},
]
```

### 6.3 Branching and Stacking

```python
# Parallel branches with stacking
pipeline = [
    StandardScaler(),
    KFold(n_splits=5),
    {"branch": [
        [PCA(10), Ridge()],
        [SelectKBest(20), SVR()],
        [{"model": RandomForestRegressor()}],
    ]},
    {"merge": "predictions"},       # OOF stacking
    {"model": LinearRegression()},  # Meta-model
]

# Separation branches
pipeline = [
    {"tag": YOutlierFilter()},
    {"branch": {"by_tag": "y_outlier_iqr", "values": {"clean": False, "outliers": True}}},
    # ... per-group processing ...
    {"merge": "concat"},
]
```

---

## 7. Storage Architecture

Reuse the proven DuckDB + Parquet architecture from nirs4all:

```
workspace/
  store.duckdb                           # Run metadata, pipeline configs, chains, logs
  arrays/<dataset_name>.parquet          # Prediction arrays (Zstd-compressed)
  artifacts/<ab>/<sha256>.joblib         # Content-addressed model binaries
  exports/                               # Exported bundles (.pfb)
  library/templates/                     # Reusable pipeline templates
```

No changes needed — the storage layer is already domain-agnostic.

---

## 8. Configuration System

### Runtime Config

```python
from pipeforge.config import CacheConfig

cache = CacheConfig(
    step_cache_enabled=True,
    step_cache_max_mb=2048,
    use_cow_snapshots=True,
    log_cache_stats=True,
    memory_warning_threshold_mb=3072,
)
```

### Environment Config

```python
# pipeforge.toml or pyproject.toml [tool.pipeforge]
[pipeforge]
default_workspace = "~/pipeforge-workspace"
default_n_jobs = -1
default_verbose = 1
plugin_paths = ["./my_plugins"]
```

---

## 9. Execution Engine Changes from nirs4all

### What stays identical
- `PipelineOrchestrator` / `PipelineExecutor` / `StepRunner` — unchanged
- Controller registry and dispatch — unchanged
- Variant generation (`_or_`, `_range_`, `_grid_`, etc.) — unchanged
- Refit strategies — unchanged
- Step caching and CoW — unchanged
- ExecutionTrace and minimal pipeline — unchanged
- Bundle system — unchanged (different extension: `.pfb`)
- Storage (WorkspaceStore, ArrayStore) — unchanged
- Prediction management — unchanged

### What changes
- `SpectroDataset` → `Dataset` (generic source registry, no wavelengths/signal types in core)
- Remove NIRS-specific controllers from core (spectral charts, resampler, etc.)
- Remove NIRS-specific operators from core (SNV, MSC, PLS variants, spectral augmentation)
- Generalize `DataSelector.processing` to work with named sources instead of indexed sources
- Remove spectroscopy terminology from report/metric labels
- Add plugin discovery mechanism
- Rename `NIRSPipeline` → `PipelineWrapper` in sklearn module

### Estimated code movement
- **~85% of pipeline/ code** stays in core (execution engine, storage, trace, bundle, refit)
- **~60% of data/ code** stays in core (Indexer, Targets, Metadata, Predictions, loaders)
- **~40% of data/ code** moves to NIRS plugin (SignalType, wavelength handling, signal conversion)
- **~100% of controllers/ code** stays in core (all generic controllers)
- **~0% of operators/ code** stays in core (filters stay; NIRS transforms, models, augmentation move to plugin)
- **~80% of visualization/ code** stays in core (PredictionAnalyzer, generic charts)

---

## 10. Future Enhancements (Post-MVP)

### Phase 1 — Core Library
- Extract engine from nirs4all
- Generic Dataset with plugin source types
- Plugin architecture with entry point discovery
- All built-in controllers
- Storage, bundles, traces, refit

### Phase 2 — Integrations
- Callbacks/hooks system at step/fold/variant/run boundaries
- Experiment tracking adapters (MLflow, W&B, Neptune)
- Pipeline DAG visualization (pre-execution structure rendering)

### Phase 3 — Scaling
- Distributed backend option (Ray, Dask) alongside joblib
- Optuna/Ray Tune integration for HPO (alternative to built-in generators)
- Streaming/incremental learning support

### Phase 4 — Ecosystem
- Domain plugin packages: `pipeforge-nirs`, `pipeforge-nlp`, `pipeforge-vision`, `pipeforge-tabular`
- Community plugin registry
- Pipeline sharing/marketplace
- AutoML plugin (automatic pipeline construction)

---

## 11. Dependencies

### Core (minimal)
- `numpy` — array operations
- `scikit-learn` — base interfaces (TransformerMixin, BaseEstimator, cross-validators)
- `duckdb` — metadata storage
- `pyarrow` / `polars` — Parquet I/O
- `joblib` — parallel execution and artifact serialization
- `pyyaml` — configuration files

### Optional (lazy-loaded)
- `tensorflow` — TF model controller
- `torch` — PyTorch model controller
- `jax` — JAX model controller
- `shap` — explanations
- `matplotlib` — visualization

---

## 12. Naming Conventions

| nirs4all | pipeforge (proposed) |
|----------|---------------------|
| `SpectroDataset` | `Dataset` |
| `NIRSPipeline` | `PipelineWrapper` |
| `.n4a` | `.pfb` (pipeforge bundle) |
| `nirs4all.run()` | `pipeforge.run()` |
| `nirs4all.predict()` | `pipeforge.predict()` |
| signal_type, wavelengths | Plugin-specific source metadata |
| RMSE/RPD/bias reports | Generic metric names (configurable via plugins) |
