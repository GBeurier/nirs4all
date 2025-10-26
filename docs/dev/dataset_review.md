# Dataset Module Review

## Analysis

### Architectural overview
- `SpectroDataset` (`nirs4all/dataset/dataset.py:13`) is the entry point exposed to operators. It orchestrates:
  - `Features` (`features.py`) → multi-source feature arrays managed via `FeatureSource`.
  - `Targets` (`targets.py`) → multi-processing target store with automatic task-type detection.
  - `Indexer` (`indexer.py`) → Polars DataFrame that maps rows to partitions, groups, branches, and processing chains.
  - Optional blocks (`Metadata`, `Predictions`, folds) that are currently stubbed or loosely integrated.
- Dataset ingestion flows: `DatasetConfigs` → `parse_config` → `handle_data` → `load_csv`. Each stage performs lightweight key normalization and delegates to CSV parsing utilities. There is no central schema definition: arrays, headers, partitions, and target metadata rely on positional alignment and side effects.
- Augmentation pipeline: `SpectroDataset.augment_samples` (facade) → `Indexer.augment_rows` (row duplication) → `FeatureSource.augment_samples` (array manipulation) while `processings` metadata is mutated in-place. There is no single source of truth for augmentation lineage, and invariants (same number of augmented rows per source) are enforced manually.
- Predictions lifecycle: pipelines construct dictionaries that flow into `Predictions.add_prediction(s)`, storing JSON strings in a Polars DataFrame. `PredictionResultsList` and `PredictionAnalyzer` sit on top, consuming the same fields but reimplementing serialization, filtering, and ranking logic.

### Identified issues
- **Data contract opacity**: The relationships between samples, index rows, feature matrices, target matrices, and prediction partitions are implicit. No module exposes a canonical schema, so onboarding and debugging rely on reading multiple files.
- **Indexer schema**: `Indexer` persists arrays as Python stringified lists (e.g., `processings`, `sample_indices`) and deserializes with `eval`/`json.loads`. This is brittle, unsafe, and prevents zero-copy persistence. Column definitions use basic integer types even when logical values demand categorical or nested structures.
- **API duplication**: `_append` is the core writer, yet `add_samples`, `add_rows`, `register_samples`, and their dict variants all wrap it, duplicating normalization logic. Similarly, predictions have three layers (`PredictionResult`, `PredictionResultsList`, `Predictions`) performing overlapping responsibilities.
- **Feature/target coupling**: `FeatureSource.update_features` handles multiple input shapes, replacements, additions, padding, and header resets in one method. `Targets.add_targets` appends new samples only when no additional processings exist, forcing call-order constraints rather than enforcing them through dedicated lifecycle hooks.
- **Persistence gaps**: `io.py` is entirely commented out. `Metadata` is a stub. Saving/loading datasets currently depends on ad-hoc scripts, and there is no guarantee that predictions, metadata, or processed targets survive round-trips.
- **Logging and observability**: Modules rely on `print` statements with emojis. There is no structured logging strategy or ability to plug into pipeline-level loggers.

## Detailed recommendations

### 1. Canonical data model and type safety
- Introduce schema-bearing dataclasses or typed Polars schemas to describe row contracts. Suggested core types:
  ```python
  @dataclass(frozen=True)
  class SampleKey:
      sample_id: int
      origin_id: int | None
      partition: Partition
      branch: int
      group: int | None

  @dataclass(frozen=True)
  class ProcessingChain:
      inputs: tuple[str, ...]  # e.g. ("raw", "SavitzkyGolay", "StandardScaler")
      source: int              # feature source index
  ```
- Define explicit enums for `Partition` (`train`, `val`, `test`, `inference`) and `TaskType` to replace free-form strings across `Indexer`, `Targets`, and `Predictions`.
- Centralize schema definitions in `dataset/schema.py` with helper constructors returning Polars `StructType` objects. Consumers should import the schema instead of re-declaring column names.

### 2. Indexer refactor (Repository + Builder pattern)
- Replace stringified lists with native Polars `List` columns. Example Polars schema snippet:
  ```python
  INDEX_DF_SCHEMA = {
      "row": pl.Int32,
      "sample": pl.Int32,
      "origin": pl.Int32,
      "partition": pl.Categorical,
      "branch": pl.Int16,
      "group": pl.Int16,
      "processings": pl.List(pl.Utf8),
      "augmentation": pl.Categorical,
      "tags": pl.List(pl.Utf8),  # open column for downstream labeling
  }
  ```
- Provide a dedicated builder for bulk inserts:
  ```python
  class IndexMutation:
      def __init__(self, *, partition: Partition, processings: Sequence[str], ...):
          ...

  class Indexer:
      def append(self, mutations: Iterable[IndexMutation]) -> list[int]:
          ...
  ```
  This isolates normalization, validation, and type casting. Consumers (features, targets, augmentation) submit declarative mutations instead of pre-normalising dictionaries.
- Encapsulate all filtering/ranking in query objects (`IndexQuery`) to avoid exposing Polars expressions throughout the codebase.

### 3. Feature and target lifecycle tightening
- Split `FeatureSource.update_features` into composable operations:
  - `replace_processing(processing_id: str, data: np.ndarray, headers: Sequence[str] | None)`
  - `add_processing(processing_id: str, data: np.ndarray, headers: Sequence[str] | None)`
  - `resize_features(n_features: int, strategy: PaddingStrategy)`
- Introduce invariants enforced at the `SpectroDataset` layer:
  ```python
  class SpectroDataset:
      def append_samples(self, samples: FeatureBatch, targets: TargetBatch | None, *, partition: Partition) -> None:
          # orchestrates indexer + features + targets with validation hooks
  ```
  where `FeatureBatch` encapsulates `np.ndarray` plus headers, and `TargetBatch` contains both raw and numeric forms. The method should verify that counts align before mutating state.
- Provide lifecycle hooks such as `before_targets_update` / `after_targets_update` to ensure augmentations or replacements cannot silently desynchronise targets from index rows.

### 4. Persistence and serialization strategy
- Adopt dual storage for predictions:
  1. **Hot store**: Polars-backed COFFEE file (`.parquet`) optimised for random access and filtering. Schema example:
     ```python
     PREDICTIONS_SCHEMA = {
         "id": pl.Utf8,
         "dataset": pl.Utf8,
         "partition": pl.Categorical,
         "fold_id": pl.Utf8,
         "metrics": pl.Struct([
             pl.Field("rmse", pl.Float32),
             pl.Field("r2", pl.Float32),
             ...
         ]),
         "y_true": pl.List(pl.Float32),
         "y_pred": pl.List(pl.Float32),
         "sample_indices": pl.List(pl.Int32),
         "processings": pl.List(pl.Utf8),
         "metadata": pl.Struct([]),  # extensible
     }
     ```
  2. **Archive**: Parquet/Arrow file per dataset stored on disk with metadata (creation time, task type, pipeline step). Provide `PredictionsRepository.save(dataset_name: str, mode: SaveMode)`.
- Revive `io.py` with a `DatasetSnapshot` abstraction that materializes features (per source `.npy` or `.parquet`), indexer (`.parquet`), targets (`.parquet` with processing ancestry), metadata, and predictions. Define signatures:
  ```python
  class DatasetSnapshot(Protocol):
      def save(self, dataset: SpectroDataset, path: Path, *, include_predictions: bool = True) -> None
      def load(self, path: Path) -> SpectroDataset
  ```
- Provide `MetadataStore` with typed accessors instead of a bare Polars table. Consider `Dictionary[str, Series]` to decouple from Polars where not needed.

### 5. Predictions stack separation
- Split the 600+ lines `predictions.py` into modules:
  1. `prediction_store.py` → Persistence (Parquet/Polars I/O, hashing, schema validation).
  2. `prediction_metrics.py` → Metric computation utilities, with pluggable metric registry and caching.
  3. `prediction_views.py` → Query/aggregation (top-k, filter, partition grouping) returning typed view objects.
  4. `prediction_exports.py` → CSV/JSON export pipeline with streaming writers.
- Introduce interfaces:
  ```python
  class PredictionRecord(NamedTuple):
      id: str
      dataset: str
      partition: Partition
      fold_id: str | None
      y_true: NDArray[np.float32]
      y_pred: NDArray[np.float32]
      metrics: Mapping[str, float]
      metadata: Mapping[str, Any]
  ```
  Consumers should no longer interact with raw Polars rows or JSON strings.

### 6. Logging and diagnostics
- Replace `print`/emoji statements with injected logger dependencies (`logging.Logger` or `structlog`). Provide module-level `logger = logging.getLogger(__name__)` and ensure all diagnostic information is structured for pipeline-level observability.
- Add trace utilities that can dump mismatched shapes or schema violations when validation hooks fail. Consider a `DatasetAudit` utility that surfaces sample counts, processing coverage, and missing metadata.

## Implementation roadmap

1. **Codify schemas and invariants**
   - Produce formal schema documentation (`docs/dataset_schema.md`) derived from new dataclasses.
   - Implement unit tests that assert alignment invariants (feature samples == index rows == targets). Include augmentation scenarios and multi-source replacements.
   - Refactor logging to remove direct prints.

2. **Indexer modernization**
   - Implement `IndexMutation` + `IndexQuery`.
   - Migrate storage to the new Polars schema and provide migration tooling to convert older datasets (stringified processings) to list-based columns.
   - Deprecate legacy wrapper methods with warnings, guiding consumers to the unified API.

3. **Feature/target API overhaul**
   - Introduce `FeatureBatch` / `TargetBatch` wrappers and rework `SpectroDataset.append_samples`.
   - Extract padding/resizing strategies into reusable helpers (`PaddingStrategy.NORMALIZE`, `PaddingStrategy.STRICT`).
   - Add validation hooks and ensure they are invoked in every mutation path (add, replace, augment, update).

4. **Persistence revamp**
   - Implement `DatasetSnapshot` with configurable backends (default: Polars Parquet). Ensure dual predictions storage (hot store vs archive) with checksum validation.
   - Fill out `MetadataStore`, enabling typed access and schema evolution (versioned metadata schema stored alongside snapshots).
   - Provide CLI utilities (or Make targets) for `dataset save` / `dataset load` that developers can run locally.

5. **Predictions modularization**
   - Split `predictions.py` into dedicated modules. Introduce `PredictionRecord` interface and typed repositories.
   - Implement metric registry with lazy evaluation (cache computed metrics, allow injection of custom metrics).
   - Add exporters that support chunked writing to Parquet/CSV and optional evaluation reports (e.g. aggregated metrics per partition).

6. **Configuration and documentation**
   - Unify dataset ingestion (`handle_data`) by extracting train/test logic into reusable helpers and enforcing required keys. Provide descriptive exceptions when configurations are incomplete or inconsistent.
   - Update developer docs covering ingestion flow, augmentation lifecycle, indexer schema, persistence formats, and prediction storage patterns.
   - Deliver internal ADR(s) summarising architectural decisions (schemas, storage formats, logging).

7. **QA and migration**
   - Build migration scripts for existing prediction archives and dataset caches (convert JSON strings → structured columns, create archived parquet files).
   - Add regression tests focused on round-tripping: create dataset → augment → save → load → compare features/index/targets/predictions bitwise.
   - Schedule a beta phase where new APIs run in parallel with legacy ones, logging canonical vs legacy divergences before full switch-over.

By tackling these steps incrementally, the dataset subsystem gains explicit contracts, safer serialization, modular responsibilities, and production-grade persistence—reducing maintenance cost and increasing developer confidence.

