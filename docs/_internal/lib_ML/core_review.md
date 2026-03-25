# nirs4all Core Capabilities Review

> Review of all pipeline/ML capabilities in nirs4all that are domain-agnostic and could be extracted into a general-purpose ML pipeline library.

---

## 1. Pipeline Definition Syntax

nirs4all uses a **list-based declarative pipeline syntax** where steps can be classes, instances, or dicts with special keywords:

```python
pipeline = [
    MinMaxScaler(),                              # sklearn TransformerMixin instance
    {"y_processing": MinMaxScaler()},            # Target scaling (keyword wrapper)
    ShuffleSplit(n_splits=3),                    # Cross-validation splitter
    {"model": PLSRegression(n_components=10)},   # Model step
]
```

### Step Types

| Syntax | Handled By |
|--------|-----------|
| `OperatorInstance()` | Auto-detected via controller matching (e.g., TransformerMixin, fit/predict) |
| `OperatorClass` (uninstantiated) | Instantiated with defaults |
| `{"keyword": operator}` | Routed by keyword (model, y_processing, tag, exclude, branch, merge) |
| `"module.path.ClassName"` | Deserialized from string |
| `{"class": "...", "params": {...}}` | Deserialized from dict |
| `[[step1, step2], [step3, step4]]` | Subpipeline (nested list) |

### Key Observations
- **No boilerplate** — users express intent directly; the engine handles routing, CV, artifact persistence
- **sklearn interoperable** — any object with `fit()`/`transform()` or `fit()`/`predict()` works out of the box
- **Extensible** — custom operators work if a matching controller is registered

**Files:** `pipeline/steps/parser.py` (StepParser), `pipeline/steps/router.py` (ControllerRouter), `pipeline/steps/step_runner.py` (StepRunner)

---

## 2. Pipeline Variant Generation (Hyperparameter Sweeps)

A **strategy-based generator system** expands compact specifications into multiple pipeline variants:

| Keyword | Purpose | Example |
|---------|---------|---------|
| `_or_` | Alternative operators/params | `{"_or_": [SNV, MSC, Detrend]}` → 3 variants |
| `_range_` | Linear parameter sweep | `{"_range_": [1, 30, 5]}` → `[1, 6, 11, 16, 21, 26]` |
| `_log_range_` | Logarithmic sweep | `{"_log_range_": [1e-3, 1e0, 4]}` → `[0.001, 0.01, 0.1, 1.0]` |
| `_grid_` | Full Cartesian grid search | `{"_grid_": {"n_components": [5,10], "alpha": [0.1,1.0]}}` → 4 combos |
| `_cartesian_` | Cross-stage combinations | `{"_cartesian_": [{"_or_": [A,B]}, {"_or_": [X,Y]}]}` → 4 pipelines |
| `_zip_` | Parallel paired iteration | `{"_zip_": {"x": [1,2,3], "y": ["A","B","C"]}}` → 3 pairs |
| `_chain_` | Sequential ordered choices | `{"_chain_": [c1, c2, c3]}` → preserves order |
| `_sample_` | Random distribution sampling | `{"_sample_": {"distribution": "log_uniform", ...}}` |

### Constraints (Phase 4)
- `_mutex_` — mutually exclusive combinations
- `_requires_` — dependency constraints
- `_exclude_` — filter out specific combos
- Lazy iteration via `expand_spec_iter()` for memory-efficient large spaces

### Architecture
Each keyword maps to a **strategy class** in `pipeline/config/_generator/strategies/`. The `PipelineConfigs` class orchestrates expansion, producing:
- `steps: list[list[step]]` — expanded pipeline configurations
- `names: list[str]` — hash-based unique names
- `generator_choices: list[list[dict]]` — recorded choices for reproducibility

**Files:** `pipeline/config/pipeline_config.py`, `pipeline/config/_generator/`

---

## 3. Controller Registry Pattern

The **controller pattern** decouples step interpretation from pipeline execution:

```python
@register_controller
class MyController(OperatorController):
    priority = 50  # Lower = higher priority

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return isinstance(operator, MyOperatorType)

    @classmethod
    def use_multi_source(cls) -> bool: return False

    @classmethod
    def supports_prediction_mode(cls) -> bool: return True

    @classmethod
    def supports_step_cache(cls) -> bool: return False

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        # Return (context, StepOutput)
        pass
```

### How It Works
1. Controllers are auto-registered via `@register_controller` into `CONTROLLER_REGISTRY`
2. Sorted by priority (lower = higher priority)
3. `ControllerRouter.route()` iterates registry, returns first match
4. No hardcoded operator types in the engine — purely registry-driven

### Built-in Controllers (generic)

| Controller | Priority | Matches | Prediction Mode | Cache |
|-----------|----------|---------|-----------------|-------|
| `TransformerMixinController` | 10 | `TransformerMixin` instances | Yes | Yes |
| `SklearnModelController` | 6 | Objects with `fit()` + `predict()` | Yes | No |
| `BranchController` | 5 | `{"branch": ...}` keyword | Yes | No |
| `MergeController` | 5 | `{"merge": ...}` keyword | Yes | No |
| `TagController` | 5 | `{"tag": ...}` keyword | No | No |
| `ExcludeController` | 5 | `{"exclude": ...}` keyword | No | No |
| `YTransformerMixinController` | — | `{"y_processing": ...}` | Yes | No |
| `CrossValidatorController` | — | sklearn CV splitter objects | No | No |
| `TensorFlowModelController` | — | TF/Keras models | Yes | No |
| `PyTorchModelController` | — | PyTorch nn.Module | Yes | No |
| `JaxModelController` | — | JAX models | Yes | No |

**Files:** `controllers/controller.py`, `controllers/registry.py`, `controllers/*/`

---

## 4. Execution Engine

### Architecture

```
API (run/predict/explain/retrain)
  └→ PipelineRunner (facade: setup, config, state coordination)
       └→ PipelineOrchestrator (central coordinator)
            ├→ PipelineConfigs (variant expansion)
            ├→ PipelineExecutor (single pipeline × single dataset)
            │    └→ StepRunner → StepParser → ControllerRouter → Controller.execute()
            ├→ joblib.Parallel (parallel variant execution)
            └→ Refit strategies (post-CV full-data training)
```

### PipelineOrchestrator (`pipeline/execution/orchestrator.py`, ~2100 lines)
- Creates `WorkspaceStore` (DuckDB) for persistence
- Normalizes pipeline/dataset inputs
- Manages per-dataset artifact lifecycle via `ArtifactRegistry`
- Routes to sequential or parallel execution
- Tracks best results across datasets
- Orchestrates refit passes

### PipelineExecutor (`pipeline/execution/executor.py`, ~400 lines)
- Executes a single pipeline on a single dataset
- Creates `ExecutionContext` (selector, state, metadata)
- Loops through steps via `StepRunner.execute()`
- Records `ExecutionTrace` for minimal pipeline extraction
- Monitors memory (RSS) and warns on thresholds
- Syncs artifacts to `WorkspaceStore` after each step

### ExecutionContext (`pipeline/config/context.py`)
Three-part typed context:

```
ExecutionContext
├── DataSelector    — partition, processing chains, layout, fold_id, branch_path, tag_filters
├── PipelineState   — y_processing version, step_number, mode (train/predict)
└── StepMetadata    — keyword, source index, fold_id, branch_path, custom fields
```

**Files:** `pipeline/runner.py`, `pipeline/execution/orchestrator.py`, `pipeline/execution/executor.py`

---

## 5. Cross-Validation and Fold Management

- Any sklearn-compatible `BaseCrossValidator` is recognized by `CrossValidatorController`
- Folds stored as tuples of `(train_indices, test_indices)` in the dataset
- Repetition-aware splitting: `dataset.set_repetition('Sample_ID')` groups by entity before splitting
- External fold loading from files supported
- Fold weights tracked for ensemble averaging during prediction

**Files:** `controllers/splitters/split.py`, `controllers/splitters/fold_file_loader.py`

---

## 6. Branching and Merging

### Duplication Branches (parallel pipelines, same data)

```python
{"branch": [
    [Preprocessor_A(), Model_A()],
    [Preprocessor_B(), Model_B()],
]}
```
- Creates independent context copies
- Each branch sees all samples
- Typically followed by a merge step

### Separation Branches (disjoint sample subsets)

```python
{"branch": {"by_metadata": "site"}}           # Per-metadata-value subsets
{"branch": {"by_tag": "outlier", "values": {...}}}  # By tag values
{"branch": {"by_source": True, "steps": {...}}}     # Per-source preprocessing
```
- Partitions dataset into non-overlapping subsets
- Each branch operates on its partition
- Requires `{"merge": "concat"}` to reassemble

### Merge Strategies

| Strategy | Use Case |
|----------|----------|
| `"features"` | Concatenate preprocessed features horizontally (late feature fusion) |
| `"predictions"` | OOF predictions as meta-features (stacking) |
| `"concat"` | Reassemble samples from separation branches |

**Files:** `controllers/data/branch.py`, `controllers/data/merge.py`

---

## 7. Out-of-Fold (OOF) Stacking

When `{"merge": "predictions"}` is used:
1. Each branch model produces OOF predictions during CV
2. `TrainingSetReconstructor` deterministically replays OOF predictions
3. OOF predictions become meta-features for the meta-model
4. `MetaModel` coordinates meta-model training

### Stacking Depth
- `DEFAULT_MAX_STACKING_DEPTH = 3` to prevent exponential complexity
- GPU models auto-serialized before fork to avoid OOM

**Files:** `pipeline/execution/refit/stacking_refit.py`, `operators/models/meta.py`

---

## 8. Refit Strategies

After CV selects the best variant, models are retrained on all data:

| Strategy | When Used |
|----------|-----------|
| **Simple Refit** | Standard pipeline — retrain winning model on full training data |
| **Per-Model Refit** | Pipelines with multiple models (subpipelines) — refit each separately |
| **Stacking Refit** | `{"merge": "predictions"}` — two-step: base OOF → meta-model refit |
| **Separation Refit** | Separation branches — refit each branch's model on its partition |
| **Competing Branches Refit** | Duplication branches with models — select best branch and refit |

All refit strategies produce artifacts with `fold_id="final"`.

**Files:** `pipeline/execution/refit/`

---

## 9. Tag and Exclude System

### Tags (non-destructive marking)
```python
{"tag": YOutlierFilter(method="iqr")}
```
- Creates metadata columns for downstream use (branching, analysis)
- Samples are NOT removed

### Exclusion (training-only removal)
```python
{"exclude": [YOutlierFilter(), XOutlierFilter(method="mahalanobis")], "mode": "any"}
```
- Marks samples as excluded during training
- Excluded samples still processed during prediction
- Modes: `"any"` (exclude if ANY filter flags) or `"all"` (exclude if ALL flag)

Both use `SampleFilter` interface: `fit(X, y)` + `get_mask(X, y) -> bool_array`

**Files:** `controllers/data/tag.py`, `controllers/data/exclude.py`, `operators/filters/`

---

## 10. Target Processing (y_processing)

```python
{"y_processing": StandardScaler()}
{"y_processing": [StandardScaler(), QuantileTransformer(n_quantiles=30)]}
```

- Applies sklearn `TransformerMixin` to targets instead of features
- Supports single transformer or chained list
- Fitted on training targets, applied to all
- Artifacts persisted for prediction mode inverse transform

**Files:** `controllers/transforms/y_transformer.py`

---

## 11. Multi-Source Dataset Support

Datasets can hold multiple feature sources (e.g., different sensor modalities):

```python
dataset.add_samples(X_source1, source_name="sensor_A", source_index=0)
dataset.add_samples(X_source2, source_name="sensor_B", source_index=1)
```

- Per-source preprocessing chains tracked in `DataSelector.processing`
- Branch by source: `{"branch": {"by_source": True, "steps": {...}}}`
- Merge sources: `{"merge": {"sources": "concat"}}`
- Controllers declare multi-source support via `use_multi_source() -> bool`

**Files:** `data/dataset.py` (Features block), `controllers/data/branch.py`, `controllers/data/merge.py`

---

## 12. Parallel Execution

- `joblib.Parallel(backend='loky')` for process-based parallelism
- Workers execute pipeline variants independently
- Workers run with `store=None` to avoid DuckDB lock conflicts
- Results merged back in the main process
- Controlled via `n_jobs` parameter (`-1` = all cores)

**Files:** `pipeline/execution/orchestrator.py`

---

## 13. Step Caching and CoW Snapshots

### Step Cache
- When multiple variants share preprocessing, cache step outputs for reuse
- Only `TransformerMixin` controllers with `supports_step_cache() = True` are cached
- Key: input data hash + operator config
- Thread-safe with read-write lock
- Configurable budget: `step_cache_max_mb`, `step_cache_max_entries`

### Copy-on-Write Snapshots
- Branch contexts use CoW snapshots to reduce memory
- `use_cow_snapshots=True` (default)

**Files:** `pipeline/execution/step_cache.py`, `config/cache_config.py`

---

## 14. Execution Trace and Minimal Pipeline

### ExecutionTrace
Records the exact path through a pipeline that produced a result:

```
ExecutionTrace
├── pipeline_uid
├── trace_id
├── steps: list[ExecutionStep]
│   ├── step_index, operator_type, chain_path
│   └── artifacts: StepArtifacts (per-fold, per-branch, per-source)
└── fold strategy and weights
```

### Minimal Pipeline Extraction
- During prediction, the trace is analyzed to extract only the steps needed
- Unused branches, CV folds, and chart steps are skipped
- Significantly faster for complex pipelines with many branches

**Files:** `pipeline/trace/execution_trace.py`, `pipeline/trace/extractor.py`, `pipeline/minimal_predictor.py`

---

## 15. Bundle Export/Import

### Export (`.n4a` ZIP archives)
```
model.n4a (ZIP)
├── manifest.json          # Bundle metadata, version, pipeline_uid
├── pipeline.json          # Minimal pipeline config
├── chain.json             # OperatorChain for deterministic replay
├── trace.json             # ExecutionTrace for artifact lookup
├── fold_weights.json      # CV fold weights
└── artifacts/             # Content-addressed artifact binaries
```

### Portable Export (`.n4a.py`)
- Self-contained Python script with embedded artifacts
- No library dependency required for prediction

### Import
- `BundleLoader.load(path)` → `BundleMetadata` + lazy artifact loading
- `BundleArtifactProvider` provides artifacts by step/fold/branch

**Files:** `pipeline/bundle/generator.py`, `pipeline/bundle/loader.py`

---

## 16. Storage Architecture

### DuckDB Metadata (`WorkspaceStore`)
- Tables: `runs`, `pipelines`, `chains`, `predictions`, `artifacts`, `logs`
- Content-addressed artifact storage with deduplication (SHA-256)
- Thread-safe with exponential backoff retry on lock conflicts
- Full run lifecycle management: `begin_run()`, `complete_run()`, `fail_run()`

### Parquet Prediction Arrays (`ArrayStore`)
- One Parquet file per dataset with Zstd compression
- Self-describing schema: prediction_id, model_name, fold_id, partition, metric, y_true, y_pred, y_proba, etc.
- Tombstone-based lazy deletes with compaction

### Workspace Layout
```
workspace/
  store.duckdb                           # All metadata
  arrays/<dataset_name>.parquet          # Prediction arrays
  artifacts/<ab>/<hash>.joblib           # Content-addressed binaries
  exports/                               # Exported bundles
  library/templates/                     # Pipeline templates
```

**Files:** `pipeline/storage/workspace_store.py`, `pipeline/storage/array_store.py`

---

## 17. Prediction Management and Ranking

### Predictions Object
- Buffers predictions in memory, flushes to `WorkspaceStore`/`ArrayStore`
- Each prediction row: y_true, y_pred, y_proba, fold_id, partition, metric, val_score, task_type, sample_indices, trace_id, model_artifact_id

### Ranking and Querying
- `top(k)` — best K predictions by metric
- `filter_predictions(filters)` — column-based filtering
- `get_best(metric, partition)` — single best
- `export_csv()`, `export()` — multiple export formats

### Result Objects
- `RunResult` — exposes `best_score`, `best_rmse`, `best_r2`, `top(n)`, `export()`
- `PredictResult`, `ExplainResult` — prediction/explanation results

**Files:** `data/predictions.py`, `api/result.py`

---

## 18. Session API

```python
session = nirs4all.session(pipeline=pipeline, workspace="workspace/", name="Experiment")
result = session.run(dataset)
predictions = session.predict(new_data)
session.save("exports/session.n4a")
loaded = nirs4all.load_session("exports/session.n4a")
```

Two usage patterns:
1. **Resource sharing** — context manager for shared `PipelineRunner` across multiple runs
2. **Stateful pipeline** — manage single pipeline lifecycle (train, predict, save, load)

**Files:** `api/session.py`

---

## 19. Retraining System

```python
result = nirs4all.retrain(source="model.n4a", data=new_dataset, mode="transfer")
```

- Modes: `FULL` (retrain everything), `TRANSFER` (freeze preprocessing, retrain model), `FINETUNE` (warm-start model)
- Per-step mode control via `StepMode` dataclass
- Controller-agnostic — works with any registered controller

**Files:** `pipeline/retrainer.py`

---

## 20. sklearn Compatibility Wrapper

```python
from nirs4all.sklearn import NIRSPipeline
model = NIRSPipeline.from_bundle("model.n4a")
model.predict(X_new)
model.score(X, y)
```

- Wraps trained pipelines in sklearn-compatible interface (prediction only, not training)
- SHAP compatibility for explainability
- CV ensemble averaging support
- Classification variant: `NIRSPipelineClassifier`

**Files:** `sklearn/pipeline.py`, `sklearn/classifier.py`

---

## 21. SHAP Explanations

```python
result = nirs4all.explain(model="model.n4a", data=new_data)
```

- Loads trained model and preprocessing chain
- Computes SHAP values via the sklearn wrapper
- Generates visualization plots

**Files:** `pipeline/explainer.py`

---

## 22. Data Augmentation Framework

A controller-based augmentation system that integrates with the pipeline:

- `SampleAugmentationController` — generates new samples (noise, mixup, etc.)
- `FeatureAugmentationController` — transforms existing features (baseline removal, etc.)
- Augmented samples tracked via `origin` column in the indexer
- Training-only: augmented samples excluded during prediction

The framework itself is generic; the operators are domain-specific.

**Files:** `controllers/data/sample_augmentation.py`, `controllers/data/feature_augmentation.py`

---

## 23. Visualization Infrastructure

- `PredictionAnalyzer` — unified interface for prediction visualization
- Chart types: TopKComparison, ConfusionMatrix, Heatmap, Candlestick, ScoreHistogram
- `PredictionCache` — caches aggregated results for multiple charts
- Filtering, grouping, ranking capabilities

**Files:** `visualization/predictions.py`, `visualization/charts/`

---

## 24. Deep Learning Backend Integration

- Lazy-loaded backends: TensorFlow, PyTorch, JAX
- Per-backend model controllers with framework-specific handling
- GPU serialization for parallel execution (detect GPU → serialize before fork)
- Automatic framework detection via `ModelFactory.detect_framework()`

**Files:** `controllers/models/tensorflow_model.py`, `controllers/models/torch_model.py`, `controllers/models/jax_model.py`

---

## Strengths (Pros)

1. **Declarative pipeline syntax** — minimal boilerplate, highly expressive
2. **Plugin architecture** — controller registry allows extending without modifying engine code
3. **Comprehensive variant generation** — from simple alternatives to full grid search, random sampling, and constraints
4. **Sophisticated OOF patterns** — stacking, branch merging, separation branches with automatic OOF reconstruction
5. **Full lifecycle management** — training → prediction → explanation → retraining in one system
6. **Robust persistence** — DuckDB metadata + Parquet arrays + content-addressed artifacts
7. **Bundle portability** — self-contained exports for deployment
8. **Execution trace** — deterministic prediction replay via minimal pipeline extraction
9. **Parallel execution** — joblib-based with proper isolation (no DB lock conflicts)
10. **Step caching** — cross-variant preprocessing reuse
11. **Multi-framework support** — sklearn, TensorFlow, PyTorch, JAX through uniform controller interface
12. **Production-ready** — session API, retraining, workspace management

## Weaknesses (Cons)

1. **Tight coupling to SpectroDataset** — the data model assumes tabular features with optional wavelength metadata; switching to a generic container requires touching many files
2. **Monolithic data container** — `SpectroDataset` manages features, targets, metadata, folds, exclusions, signal types, augmentation tracking all in one class (~large file)
3. **NIRS terminology leak** — some variable names, metrics, and report labels use spectroscopy terminology (RMSE, RPD, bias/slope naming conventions)
4. **Orchestrator complexity** — `orchestrator.py` at ~2100 lines handles too many responsibilities
5. **No first-class support for non-tabular data** — images, text, graphs, time series would require significant data model changes
6. **Limited distributed execution** — joblib/loky works for single-machine parallelism but not multi-node clusters
7. **DuckDB single-file constraint** — single DuckDB file limits concurrent writers from different processes
8. **No pipeline composition** — pipelines are flat lists; no reusable sub-pipeline definitions or pipeline-of-pipelines
9. **Generator syntax is custom** — not aligned with Optuna, Ray Tune, or other established hyperparameter frameworks
10. **No callbacks/hooks system** — no mechanism for user-defined callbacks at step/fold/variant boundaries

## Missing Features for a General ML Library

1. **Experiment tracking integration** — MLflow, W&B, Neptune, TensorBoard
2. **Distributed execution** — Ray, Dask, Spark backends for multi-node parallelism
3. **Pipeline composition** — reusable named sub-pipelines, pipeline templates
4. **Callbacks/hooks** — user-defined hooks at step, fold, variant, and run boundaries
5. **Non-tabular data support** — images (channels, spatial dims), text (tokenization), graphs, time series (temporal axis)
6. **Optuna/Ray Tune integration** — leverage established HPO frameworks instead of custom generators
7. **Feature store integration** — Feast, Tecton for production feature management
8. **Model registry** — MLflow Model Registry, BentoML for model versioning and deployment
9. **Data versioning** — DVC, LakeFS integration for dataset lineage
10. **Streaming/online learning** — incremental model updates with new data
11. **AutoML capabilities** — automatic pipeline construction (TPOT, auto-sklearn style)
12. **Pipeline visualization** — DAG rendering of pipeline structure before execution
13. **Resource management** — GPU allocation, memory budgets per step, distributed resource scheduling
