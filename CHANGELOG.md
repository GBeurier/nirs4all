# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.8.2] - DuckDB Stability, Branch/Merge Fixes & Parallel Tests - 2026-02-25

### 🐛 Bug Fixes

#### DuckDB / Storage Stability
- **`WorkspaceStore` atexit handler**: Registers an `atexit` callback to close the DuckDB connection on interpreter shutdown, preventing segfaults when the process exits without an explicit `close()` call
- **`_jittered_delay` return type**: Fixed return type to always be `float`, removing potential `int` return that caused type errors in retry logic

#### Branch / Merge Pipeline Correctness
- **Preserve preprocessing chains after merge** (closes #24): `MergeController.add_merged_features()` was calling `reset_features()` which wiped per-branch preprocessing history from the run summary `Preprocessing` column. Fix builds composite processing names from branch contexts before the reset; added 3 helper methods to `MergeController`, fixed 6 call sites, and removed dead code in `dataset.py`
- **Runner ownership in `RunResult`**: `RunResult` now tracks `WorkspaceStore` ownership so it is closed deterministically when the result object is garbage-collected or used in a `with` block, preventing premature closure when the store is shared across `retrain()` and `session` workflows

### 🧪 Tests

#### Merge Auto-Detection
- **Integration tests** (`test_merge_auto_detect.py`): Validate auto-detect merge strategy selection for duplication and separation branches across common configurations
- **Unit tests** (`test_merge_auto_detect.py`): Cover `MergeController` auto-detect logic for all merge modes

#### Separation Branch Generators
- **Integration tests** (`test_separation_branch_generators.py`): Extensive coverage for generator keywords (`_or_`, `_range_`, `_grid_`, etc.) inside separation branches, including parallelisation scenarios

#### Preprocessing Chain Regression
- **Unit tests** (`test_merge_preprocessing_chain.py`): 13 regression tests ensuring per-branch preprocessing chains survive merge operations

#### Pipeline Config Expansion
- **Unit tests** (`test_pipeline_config_separation_expansion.py`): Verify correct generator expansion for separation branch configs
- **Unit tests** (`test_pipeline_config_separation_gen_keys.py`): Confirm accurate detection of generator usage in separation branches

### 🔧 Improvements

#### Pre-publish Script
- **Parallel example categories** (`pre-publish.sh`): Example categories now run in parallel with per-category log files, reducing total pre-publish validation time

---

## [0.8.1] - DuckDB Resilience, PCA Projections & Sampling - 2026-02-25

### ✨ New Features

#### PCA Projection Utility
- **`nirs4all.analysis.projections`**: New PCA projection module for quick dataset visualization and dimensionality analysis

#### Sampling Functions
- **`nirs4all.data.selection.sampling`**: New sampling utilities for dataset subsampling and selection strategies

#### AOM-PLS Benchmarking Framework
- **`bench/AOM/`**: Next-generation benchmarking framework for AOM-PLS models including DARTS-PLS, MoE-PLS, zero-shot router, and enhanced AOM variants

### 🔧 Improvements

#### DuckDB Concurrency & Crash Recovery
- **Context manager support**: `WorkspaceStore` now supports `with` statement for deterministic resource management
- **Safety net in `__del__`**: Ensures connections are closed if not explicitly done by the user
- **Per-operation retry with jitter**: Replaced irreversible degraded mode with per-operation retries and jitter to avoid thundering herd
- **Transaction batching**: New transaction management to batch multiple writes, reducing lock contention
- **Orphaned file cleanup**: `ArrayStore` cleans up orphaned temporary files during initialization
- **Explicit connection closing**: CLI commands and `api/run.py` now close `WorkspaceStore` after use

#### CI/CD
- **Version consistency check**: CI workflow verifies version consistency across `pyproject.toml`, `__init__.py`, and `conda-forge/meta.yaml`
- **Conda-forge update notification**: CI alerts when conda-forge recipe needs updating

### 🧪 Testing

- **PCA projection tests**: Unit tests for the new projections module
- **Sampling function tests**: Unit tests for the new sampling utilities

---

## [0.8.0] - AOM*-PLS, Parquet Storage & Scoring Overhaul - 2026-02-20

### ✨ New Features

#### Docker Support
- **Multi-stage Dockerfile**: Lightweight production image based on `python:3.11-slim` with build-stage compilation
- **`.dockerignore`**: Optimized Docker build context

#### Conda-Forge Distribution
- **`conda-forge/meta.yaml`**: Recipe for nirs4all package on conda-forge
- **Staged recipes**: Conda-forge recipes for missing dependencies (`cvmatrix`, `ikpls`, `pyopls`, `trendfitter`)
- **Conda-forge setup guide**: Internal documentation for the submission process

### 🔧 Improvements

#### Dependency Management
- **Twinning reimplemented natively**: Replaced `twinning` external dependency with a pure NumPy implementation of the data twinning algorithm (Vakayil & Joseph 2022) in `SPlitSplitter`
- **PLS variants promoted to core dependencies**: `ikpls`, `pyopls`, `trendfitter` moved from optional `[pls]` extra to core dependencies
- **Removed `twinning` from all requirements files**

#### CI/CD
- **macOS compatibility**: Improved CI workflows to avoid deadlocks, handle test coverage, and skip problematic tests on macOS
- **Pre-publish validation**: Enhanced pre-publish script with macOS support and timeout handling
- **Reusable disk cleanup action**: Replaced manual disk cleanup with a reusable GitHub Action


#### AOM-PLS & POP-PLS Models
- **`AOMPLSRegressor`**: Adaptive Operator-selection Meta-PLS — automatic preprocessing selection using a bank of linear operators with sparsemax gating during PLS component extraction
- **`AOMPLSClassifier`**: Classification variant with probability calibration and sklearn compatibility
- **`POPPLSRegressor` / `POPPLSClassifier`**: Penalized Orthogonal Projections PLS for operator selection and validation
- **Linear operator bank**: Identity, Savitzky-Golay filter, Detrend projection, Composed operator, and additional SG/composed variants
- **PyTorch backend**: Optional Torch-based AOM-PLS implementation for GPU acceleration

#### New Preprocessing Operators
- **`NorrisWilliams`**: Norris-Williams smoothing and derivative transform with both function and transformer APIs
- **`WaveletDenoise`**: Wavelet-based denoising transform with configurable wavelet families and threshold modes
- **Orthogonalization transforms**: New orthogonalization module for spectral data

#### Prediction Storage Migration (DuckDB → Parquet)
- **Hybrid DuckDB + Parquet storage**: Structured metadata stays in DuckDB, dense prediction arrays moved to Parquet sidecar files
- **`ArrayStore`**: New module for saving, loading, and verifying prediction arrays in Parquet format
- **Migration utilities**: Automatic migration of legacy DuckDB prediction arrays to Parquet with data integrity verification
- **Tombstone-aware deletion**: Proper handling of soft-deleted data in the new storage format

#### SPXYFold Splitter
- **`SPXYFold`**: K-Fold cross-validation splitter based on the SPXY (Sample set Partitioning based on joint X-Y distances) algorithm

#### Chain Summary System
- **New `v_chain_summary` view**: Aggregate chain summaries with model metadata, CV scores, and final scores
- **Enhanced chain query methods**: `query_chain_summaries`, `top_chains` with list-based filter parameters supporting SQL `IN` clauses
- **Backfill logic**: Populate chain summary columns from existing predictions
- **`task_type` filter**: Additional filter parameter for query specificity

#### Project Management
- **Project CRUD operations**: Create, list, update, and delete projects in the database
- **`projects` table**: New SQL schema for project metadata
- **Run-project association**: Link runs to projects for experiment organization

#### Pipeline Metrics & Reporting
- **Ensemble test scores**: New `Ens_Test` and `W_Ens_Test` metrics for ensemble evaluation
- **Mean fold validation**: New `MF_Val` metric for cross-validation reporting
- **RMSEP-based sorting**: Tab report manager now sorts by RMSEP instead of RMSECV
- **Score scope filtering**: `build_aggregated_query` and `build_top_aggregated_query` support filtering by scope (`CV`, `all`, `final`)

#### Stacking & Model Helpers
- **`stack_params` helper**: New utility for fine-tuning stacking model parameters with enhanced model parameter handling

#### Data Loading
- **Gzip and tar file support**: Enhanced CSV and folder parsing for compressed file formats

### 🔧 Improvements

#### Pipeline Execution
- **Memory cleanup between datasets**: `PipelineOrchestrator.cleanup()` releases memory between dataset iterations
- **Graceful dataset failure handling**: Orchestrator logs errors and cleans up resources on dataset failures
- **Parallel execution**: Improved joblib/loky backend compatibility by removing unpicklable objects
- **Deferred artifact persistence**: `ArtifactRegistry` supports deferred persistence and enhanced generator keyword handling
- **Random state propagation**: Consistent random state throughout pipeline for reproducibility

#### Refit System
- **Multi-criteria refit**: Enhanced handling with independent model selection across multiple criteria and improved error diagnostics
- **`selection_score` rename**: `best_score` → `selection_score` in `LazyModelRefitResult` for clarity
- **Per-model config extraction**: New `extract_per_model_configs` function for extracting best configurations per model class
- **Competing branches refit**: `execute_competing_branches_refit` refits all branches with average CV scores in predictions
- **List-based refit parameters**: Support for list-based refit parameters with aggregation reporting

#### DuckDB Resilience
- **Degraded mode**: Automatic fallback when DuckDB encounters persistent lock failures
- **Retry logic**: Enhanced error handling and retry for DuckDB lock conflicts in pipeline execution and storage

#### Scoring & Validation
- **Scoring computation invariants**: Correct RMSECV calculation from pooled OOF predictions, proper None score preservation
- **NIRS/ML naming conventions**: Consistent metric naming conventions across contexts
- **`v_aggregated_predictions_all` view**: Supports querying both CV and refit entries

#### Configuration & Generators
- **Generator count limits**: `log_range_strategy`, `or_strategy`, `range_strategy`, `sample_strategy`, and `zip_strategy` now allow no limit when count ≤ 0
- **`BestChainEntry` dataclass**: Track best preprocessing chain per model during cross-validation for more efficient refit

#### Code Quality
- **2000+ mypy errors fixed**: Comprehensive type-checking cleanup across the codebase
- **Type aliases**: Added type aliases for clarity in multiple modules
- **Ruff and mypy CI integration**: Enhanced CI with ruff and mypy checks
- **Polars version**: Bumped minimum `polars` requirement to 1.0.0

### 📚 Documentation

- **Workspace architecture docs**: Updated for hybrid DuckDB + Parquet storage system
- **Operator catalog**: New spectral augmentation and advanced PLS variants documented
- **Prediction lifecycle**: Clarified scalar scores (DuckDB) vs. arrays (Parquet) storage
- **Core audit**: Pre-webapp core audit notes

### 🧪 Testing

- **AOM-PLS test suite**: Regressor, classifier, operator adjoint identity, sparsemax, sklearn compatibility, custom operator banks, Torch backend parity
- **POP-PLS test suite**: Regressor and classifier, operator selection and validation
- **New operator tests**: NorrisWilliams, FiniteDifference, WaveletProjection, FFTBandpass, wavelet denoising
- **Parquet storage tests**: ArrayStore save/load/integrity, migration from DuckDB, tombstone handling
- **Workspace store tests**: Chain replay, chain summaries, bulk update, API inventory
- **Scoring invariant tests**: RMSECV pooling, None preservation, metric naming, config deduplication
- **OptunaManager tests**: Aggregation (BUG-2), grid search (ISSUE-17), config validation, refit skip (BUG-4), single-path holdout (BUG-3), train_params sampling (ISSUE-4)
- **Parallel execution tests**: No pickling errors, result consistency between parallel and sequential runs
- **Refit tests**: Lazy refit, model selector, advanced refit, warm start, stacking refit, infrastructure
- **Prediction analyzer tests**: Comprehensive visualization and analysis coverage
- **Step cache tests**: Correctness, copy-on-write, cacheability
- **Classifier sklearn wrapper tests**: New comprehensive test suite

### 🐛 Bug Fixes

- **OptunaManager `_aggregate_scores`**: Fixed incorrect aggregation behavior (BUG-2 regression)
- **Grid search suitability**: Fixed `_is_grid_search_suitable` and `_create_grid_search_space` (ISSUE-17)
- **Single-path optimization**: Now uses holdout split to prevent overfitting (BUG-3 regression)
- **Refit phase finetuning**: Refit phase correctly skips finetuning; `finetune_params` stripped from steps (BUG-4 regression)
- **NaN checks in `RunResult`**: Refactored NaN validation and error handling
- **Polars DataFrame inference**: Set `infer_schema_length` to None for prediction DataFrames

### 🗑️ Removed

- **`csv_loader.py`**: Removed deprecated CSV loader
- **`lazy_loader.py`**: Removed deprecated lazy loading module
- **`io.py`** (data): Removed deprecated data I/O module
- **`legacy_parser.py`**: Removed deprecated legacy parser
- **Prediction component modules**: Removed `aggregator.py`, `array_registry.py`, `indexer.py`, `query.py`, `ranker.py`, `schemas.py`, `serializer.py`, `storage.py` (replaced by Parquet-based storage)
- **Storage I/O modules**: Removed `io.py`, `io_exporter.py`, `io_resolver.py`, `io_writer.py`, `manifest_manager.py` (replaced by new workspace store)
- **`reproducibility.py`**: Removed deprecated utilities; functionality integrated into runner and orchestrator
- **`branch_diagram.py`**: Removed deprecated visualization module
- **`library_manager.py`**: Removed deprecated workspace library manager
- **CI quick mode**: Removed quick mode from example verification workflows; all examples now execute fully

---

## [0.7.1] - Caching, Workspace Store & Refit Improvements - 2026-02-08

### ✨ New Features

#### Copy-on-Write Caching Mechanism
- **Step-level cache** (`StepCache`): Cache and restore dataset state between pipeline steps to avoid redundant recomputation
- **Copy-on-write `ArrayStorage`**: Block-based shared memory with automatic detach-on-write for efficient dataset cloning
- **Cache configuration** (`CacheConfig`): New centralized cache configuration dataclass
- **Memory estimation utilities** (`nirs4all.utils.memory`): Accurate byte-level memory estimation for datasets and cache entries

#### Pipeline Refit System
- **`RefitExecutor`**: Full refit pipeline for retraining best models on the entire dataset
- **`ModelSelector`**: Select best model configurations from completed runs
- **`ConfigExtractor`**: Extract pipeline configurations for refit execution
- **`StackingRefitExecutor`**: Specialized refit support for stacking/meta-model pipelines
- **`RefitParams`**: Configuration dataclass for refit behavior

#### Pipeline Topology Analysis
- **New `nirs4all.pipeline.analysis.topology` module**: Analyze pipeline structure including stacking detection, branch separation, and feature merge identification
- **Pattern detection**: Identify stacking, separation branches, and feature merges in pipeline definitions

#### WorkspaceStore Enhancements
- **Thread-safe database access**: Concurrent DuckDB access with proper locking
- **Aggregated predictions view** (`v_aggregated_predictions`): New database view aggregating prediction metrics across folds
- **Enhanced query system**: New queries for aggregated predictions, chain predictions, and top aggregated predictions with metric-aware ranking
- **Prediction array retrieval**: Direct access to prediction arrays from the store
- **Artifact query service** (`QueryService`): Centralized artifact querying with filtering and sorting

#### Hashing Utilities
- **`nirs4all.utils.hashing`**: New module for deterministic data content hashing
- **`SpectroDataset.content_hash()`**: Compute content-based hash for dataset change detection without unnecessary materialization

### 🔧 Improvements

#### Pipeline Execution
- **Enhanced `PipelineExecutor`**: Improved artifact management and prediction safety
- **Enhanced `PipelineOrchestrator`**: Explicit workspace path requirements, improved run tracking
- **Prediction resolver**: Deterministic resolution modes for consistent prediction handling

#### Model Training & Validation
- **Enhanced model training logic**: Improved validation score calculation and dataset handling
- **Refined splitter functionality**: Better cross-validation splitting behavior
- **Controller improvements**: Enhanced `TransformerMixinController` with extended step cache support

#### Storage & Artifacts
- **Enhanced `ArtifactRegistry`**: Content verification against full hashes
- **Enhanced `ArtifactLoader`**: Improved step index handling and artifact loading behavior
- **Schema improvements**: Updated store schema with new views and idempotent creation

#### CI/CD
- **Parallel pytest execution**: Enhanced CI workflows with parallel test execution support
- **Parallel CI example runner**: Job control and improved output validation in `run_ci_examples.sh`
- **Optimized example parameters**: Reduced computational load in CI examples for faster execution

### 📚 Documentation

- **Cache optimization guide**: New section in pipelines user guide
- **Session API documentation**: Detailed guide for stateful workflows
- **SpectroDataset cache investigation**: Comprehensive analysis of caching mechanisms and memory management strategy
- **Technical debt review**: Prioritized debt analysis for workspace/predictions/artifacts
- **Enhanced user guides**: Added related examples across preprocessing, visualization, merging, multi-source, and stacking docs

### 🧪 Testing

- **Pipeline executor regression tests**: Execution and prediction flushing coverage
- **Pipeline orchestrator tests**: Explicit workspace path requirement enforcement
- **Pipeline topology analysis tests**: Stacking, separation branches, and feature merge patterns
- **Execution phase and hashing tests**: Deterministic behavior verification
- **OOF prediction accumulation tests**: Correct averaging across validation folds
- **WorkspaceStore tests**: Chain replay, prediction upsert, artifact registration, method signature validation
- **Prediction resolver tests**: Determinism and resolution mode coverage
- **Step cache tests**: Cached state restoration, data integrity, statistics accuracy
- **Content hash tests**: Hash consistency on mutations
- **Memory estimation tests**: Accurate byte calculations for datasets and cache entries
- **Refit and run entity tests**: Transition validation and metric comparison

### 🗑️ Removed

- **`csv_loader.py`**: Removed deprecated CSV loader
- **`lazy_loader.py`**: Removed deprecated lazy loading module
- **`io.py`**: Removed deprecated data I/O module
- **Generator constraint/strategy files**: Removed unused generator constraint and strategy registry modules

---

## [0.7.0] - Major Architecture & Operator Overhaul - 2026-02-05

> **⚠️ Documentation Notice:** Due to the extensive scope of this release, some documentation may be temporarily incomplete or out of sync. Updates are in progress.

### ⚠ BREAKING CHANGES

#### Synthesis Module Relocated
- **`nirs4all.data.synthetic`** → **`nirs4all.synthesis`** — update all imports
- Generator now delegates to operators for path length, instrumental broadening, and noise effects

#### Augmenter Base Class Removed
- **`Augmenter` base class deleted** (`abc_augmenter.py` removed entirely)
- **`IdentityAugmenter` deleted** — remove from all configs
- All augmentation operators now inherit from `TransformerMixin + BaseEstimator` or `SpectraTransformerMixin`
- **Migration**: Replace `Augmenter` subclasses with `TransformerMixin, BaseEstimator` or `SpectraTransformerMixin`

#### Operator API Changes
- **`apply_on` parameter removed** from all augmentation operators — replaced by step-level `variation_scope`
- **`copy` parameter removed** from all augmentation operators
- **`lambda_axis` parameter removed** from all augmentation operators — wavelengths are now auto-injected by the controller
- **`augment()` method removed** — use standard `transform()` instead
- **`transform_with_wavelengths()` removed** — use `transform(X, wavelengths=wl)` or let the controller handle it

#### SpectraTransformerMixin Simplified
- `transform_with_wavelengths(X, wavelengths)` replaced by `_transform_impl(X, wavelengths)` (internal abstract method)
- Public API is now standard `transform(X, **kwargs)` — wavelengths passed via kwargs
- `_requires_wavelengths` attribute: `True`, `False`, or `"optional"`
- `_validate_wavelengths()` helper for wavelength validation

### ✨ New Features

#### NA Handling System
- **Centralized NA policy**: New `apply_na_policy` utility for consistent NA handling across all loaders
- **NA policies**: `remove_sample`, `remove_feature`, `replace`, `ignore`, `abort`
- **`NAFillConfig`**: Configurable NA replacement strategies (mean, median, constant, interpolate)
- **Enhanced error reporting**: Detailed messages for NA detection including affected rows/columns
- **Loader support**: MatlabLoader, NumpyLoader, ParquetLoader all support new NA policies

#### Controller-Managed Variation (`variation_scope`)
- New `variation_scope` parameter at the `sample_augmentation` step level: `"sample"` (default), `"batch"`
- Per-transformer override via dict spec: `{"transformer": ..., "variation_scope": "batch"}`
- Hybrid performance model: operators with `_supports_variation_scope = True` handle variation internally; others get per-sample cloning from controller

#### SpectraTransformerMixin Foundation
- **New `SpectraTransformerMixin` base class**: Enables wavelength-aware transformations with full sklearn compatibility
- **Automatic wavelength passing**: Controller detects operators that require wavelengths and extracts them from the dataset
- **`_requires_wavelengths` class flag**: Operators can declare mandatory or optional wavelength requirements

#### New Augmentation Operators
- **`PathLengthAugmenter`**: Multiplicative path length variation
- **`BatchEffectAugmenter`**: Wavelength-dependent batch effects (offset + gain)
- **`InstrumentalBroadeningAugmenter`**: Gaussian convolution broadening (FWHM-based)
- **`HeteroscedasticNoiseAugmenter`**: Signal-dependent noise
- **`DeadBandAugmenter`**: Random dead band (non-responsive region) simulation

#### Environmental Effect Operators (`nirs4all.operators.augmentation.environmental`)
- **`TemperatureAugmenter`**: Simulates temperature-induced spectral changes with region-specific effects for O-H, N-H, and C-H bands
- **`MoistureAugmenter`**: Simulates moisture/water activity effects on spectra

#### Scattering Effect Operators (`nirs4all.operators.augmentation.scattering`)
- **`ParticleSizeAugmenter`**: Simulates particle size effects on light scattering
- **`EMSCDistortionAugmenter`**: Applies EMSC-style scatter distortions

#### Spectral Components Expansion
- **111 predefined spectral components** (expanded from 48): Added petroleum/hydrocarbon components (crude oil, diesel, gasoline, kerosene, PAH)
- Enhanced metadata with synonyms and tags for better categorization

#### Run Management System
- **New `Run` module**: Manage experiment sessions with run configurations and status transitions
- **`RunConfig`, `RunSummary`, `TemplateInfo`, `DatasetInfo`**: Data classes for run-related data
- **Manifest management**: Create, update, serialize run manifests with checkpoints

#### Feature Selection Enhancement
- **`FlexiblePCA` and `FlexibleSVD` classes**: New flexible dimensionality reduction
- Enhanced feature selection module documentation

#### AutoDetector Improvements
- **Improved header detection**: Handle cases with and without headers
- **Wavelength header detection**: Based on value characteristics and spacing
- **Signal type detection**: Check both header and data values
- **Word-boundary-aware pattern matching**: For filename detection in FolderParser

### 🔧 Improvements

#### Storage & Infrastructure
- **DuckDB storage migration**: Refactored artifact storage from `binaries/` to `artifacts/` directory
- **Centralized workspace path**: `get_active_workspace()` function for consistent path management

#### RunResult Enhancements
- Simplified metrics retrieval (removed unnecessary parameters)
- Enhanced prediction metrics handling

#### Documentation Structure
- New modules: `branch_utils`, `exclude`, reconstruction submodules
- Enhanced augmentation module with new submodules: `edge_artifacts`, `random`, `spectral`, `splines`
- Pipeline documentation with branching and merging keywords

#### Tag System
- Unit tests for QueryBuilder tag filtering (boolean, numeric, range, list membership, callable, null handling)
- Tag serialization in IndexStore
- SpectroDataset tag operations (add, set, get, remove)

### 📚 Documentation

- **Pipeline samples**: 10 new pipeline samples (JSON/YAML) demonstrating branching, stacking, filtering, model tuning
- **Filtering guide**: Comprehensive documentation for non-destructive filtering system
- **Merging guide**: Detailed examples for feature and prediction merging strategies
- **ViT-NIRS roadmap**: Integration strategy for Universal Spectral Embedding project
- **Academic paper**: LaTeX document for nirs4all framework

### 🧪 Testing

- **Reconstruction module tests**: Forward model, calibration, inversion, distributions, generator, validation
- **Scattering operators tests**: ParticleSizeAugmenter, EMSCDistortionAugmenter
- **SpectraTransformerMixin tests**: 23 tests for base class behavior
- **Environmental operators tests**: With environmental parameter handling
- **Bundle export tests**: Integration tests for special operator types
- **DuckDB storage tests**: Updated assertions for `store.duckdb` persistence
- Removed obsolete tests: `test_catalog_export.py`, `test_library_manager.py`, `test_query_reporting.py`

### 🐛 Bug Fixes

- CI example scripts file permissions
- Fixture reproducibility for standard regression dataset

### Configuration Migration

**Before:**
```yaml
sample_augmentation:
  transformers:
    - GaussianAdditiveNoise(apply_on="samples", sigma=0.01)
    - LinearBaselineDrift(apply_on="global", lambda_axis=[...])
  count: 5
```

**After:**
```yaml
sample_augmentation:
  variation_scope: "sample"
  transformers:
    - GaussianAdditiveNoise(sigma=0.01)
    - transformer: LinearBaselineDrift()
      variation_scope: "batch"
  count: 5
```

---

## [0.6.3] - Wavelength-Aware Operators & Generator Migration - 2026-01-17

### New Features

#### SpectraTransformerMixin Foundation
- **New `SpectraTransformerMixin` base class**: Enables wavelength-aware transformations while maintaining full sklearn compatibility
- **Automatic wavelength passing**: Controller detects operators that require wavelengths and extracts them from the dataset
- **`_requires_wavelengths` class flag**: Operators can declare mandatory or optional wavelength requirements
- **Dual interface support**: Both `transform(X, wavelengths=...)` and `transform_with_wavelengths(X, wl)` supported

#### Environmental Effect Operators (`nirs4all.operators.augmentation.environmental`)
- **`TemperatureAugmenter`**: Simulates temperature-induced spectral changes with region-specific effects for O-H, N-H, and C-H bands
  - Configurable shift, intensity, and broadening effects
  - Literature-based parameters from Maeda et al. (1995), Segtnan et al. (2001)
- **`MoistureAugmenter`**: Simulates moisture/water activity effects on spectra
  - Models free vs. bound water state transitions
  - Affects 1st overtone (1400-1500nm) and combination (1900-2000nm) water bands

#### Scattering Effect Operators (`nirs4all.operators.augmentation.scattering`)
- **`ParticleSizeAugmenter`**: Simulates particle size effects on light scattering
  - Wavelength-dependent baseline (lambda^(-n) relationship)
  - Configurable path length effects
- **`EMSCDistortionAugmenter`**: Applies EMSC-style scatter distortions
  - Multiplicative and additive components
  - Configurable polynomial order for wavelength-dependent baseline

#### Generator Integration
- **Operators-first architecture**: Synthetic data generator now uses operators exclusively for environmental and scattering effects
- **Simplified generator API**: Removed `use_operators` flag - operators are always used when configs are provided
- **Consistent augmentation**: Same operators used in both data generation and pipeline augmentation

### Improvements

#### Controller Enhancement
- **`TransformerMixinController`**: Updated to detect `SpectraTransformerMixin` instances and pass wavelengths automatically
- **Wavelength extraction fallback**: Primary via `dataset.wavelengths_nm()`, fallback to numeric headers
- **All execution paths updated**: Main transform, batch augmentation, and sequential augmentation paths all support wavelength passing

### Code Cleanup

#### Dead Code Removal
- Removed `TemperatureEffectSimulator`, `MoistureEffectSimulator`, `EnvironmentalEffectsSimulator` classes
- Removed `ParticleSizeSimulator`, `EMSCTransformSimulator`, `ScatteringCoefficientGenerator`, `ScatteringEffectsSimulator` classes
- Removed legacy convenience functions (`apply_temperature_effects`, `apply_moisture_effects`, etc.)
- Retained configuration dataclasses used by operators

### Documentation

- **Developer guide**: New `docs/_internals/spectra_transformer_mixin.md` with implementation notes
- **User guide**: Updated augmentation guide with wavelength-aware operators section
- **API docs**: Added documentation for `operators.augmentation` and `operators.base` packages

### Testing

- **Unit tests**: 23 tests for SpectraTransformerMixin, 42 tests each for environmental and scattering operators
- **Controller tests**: 21 unit tests for wavelength passing logic
- **Integration tests**: 7 pipeline tests for spectra transformers, 11 generator parity tests
- **Configuration tests**: Updated tests for retained configuration classes

---

## [0.6.2] - Synthetic Data Enhancement & Pipeline Improvements - 2026-01-02

### ✨ New Features

#### Synthetic Data Generation (Phase 4)
- **Wavenumber utilities**: Conversions, NIR zone classification, overtone/combination band calculations, hydrogen bonding shifts
- **Procedural component generation**: Functional group types, properties dictionary, and procedural generator
- **Application domain priors**: 20 predefined domains across agriculture, food, pharmaceutical, and industrial categories
- **Instrument simulation**: Instrument archetypes (FOSS, Bruker, SCiO), detector types, multi-sensor stitching, measurement modes
- **Environmental/matrix effects**: Temperature, moisture, particle size effects simulation
- **Spectral realism validation**: Scorecard with correlation length, derivative stats, peak density, SNR metrics, adversarial validation
- **Benchmark dataset matching**: Generate synthetic data matching published dataset characteristics
- **GPU acceleration**: JAX/CuPy backends for fast generation
- **Real data fitting**: Analyze real spectra and create matching generators
- **48 predefined spectral components** (expanded from 31): Added casein, gluten, dietary fiber, glycerol, malic acid, tartaric acid, polymers, plastics

#### Pipeline Improvements
- **Run tracking in logs**: PipelineOrchestrator now logs current run and total runs for better progress visibility
- **Batch execution**: Enhanced support for multiple pipelines and datasets in `nirs4all.run()`
- **Group-by in `top()`**: Get top N results per group with `return_grouped` option

#### Optuna Integration
- **Sorted tuple parameter type**: New parameter type for OptunaManager
- **Detailed configuration support**: Enhanced parameter sampling with log-scale and advanced options

#### Deep Learning
- **FCK-PLS Torch**: End-to-end learnable Fractional Convolutional Kernel PLS prototype (in bench/)
- **PyTorch model improvements**: Better target handling and custom regularization support

### 🔧 Improvements

- **Dataset handling**: Support for lists of SpectroDataset instances, deep copies to prevent mutation
- **Non-linear target complexity**: NonLinearConfig, ConfounderConfig, MultiRegimeConfig for complex scenarios
- **CI/CD**: Aggressive disk cleanup, improved Windows test stability, logging reset for file handle issues

### 🐛 Bug Fixes

- **OptunaManager**: Support both tuple and list formats for length configuration
- **ConcentrationPrior tests**: Use unified params dictionary structure
- **Procedural generator tests**: Corrected parameter names for consistency

### 📚 Documentation

- **New examples**: D07-D09 synthetic generator examples (wavenumber, domains, instruments)
- **Reference examples**: R05-R07 for environmental effects, validation, and fitting
- **Updated developer path**: Reflects new synthetic data examples
- **pybaselines dependency**: Added to project requirements

## [0.6.1] - GitHub Actions CI/CD Update - 2025-12-31

### 🔧 Infrastructure

- **GitHub Actions**: Updated CI/CD workflows for improved build and deployment processes
- **Minor Fixes**: Removed sparse-pls and mbpls dependencies. Implemented natively the numpy versions.

## [0.6.0] - Major API Overhaul and Architecture Improvements - 2025-12-27

This release introduces a new module-level API, complete documentation overhaul, sklearn integration, branching/merging pipelines, synthetic data generation, and extensive architectural improvements.

### ✨ New Features

#### Synthetic Data Generation
- **New `nirs4all.generate()` API**: Generate realistic synthetic NIRS spectra for testing and prototyping
- **Convenience functions**: `nirs4all.generate.regression()`, `nirs4all.generate.classification()`, `nirs4all.generate.multi_source()`
- **Builder pattern**: `SyntheticDatasetBuilder` with fluent interface for full control
- **Physically-motivated generation**: Beer-Lambert law with Voigt profile peaks, realistic noise and scatter
- **Predefined components**: 8 spectral components (water, protein, lipid, starch, cellulose, chlorophyll, oil, nitrogen_compound)
- **Configurable complexity**: `"simple"` (fast tests), `"realistic"` (typical NIR), `"complex"` (challenging scenarios)
- **Classification support**: Controllable class separation and imbalanced class weights
- **Multi-source generation**: Combine NIR spectra with auxiliary data (markers, sensors)
- **Metadata generation**: Sample IDs, groups for GroupKFold, repetitions
- **Batch effects simulation**: For domain adaptation research
- **Export capabilities**: `to_folder()`, `to_csv()` compatible with DatasetConfigs
- **Real data fitting**: `from_template()` to generate data matching real dataset characteristics
- **Pytest fixtures**: Comprehensive test fixtures in `tests/conftest.py` for reproducible testing
- **CSV variation generator**: Test loaders with different formats (delimiters, headers, decimals)

#### Module-Level API (Primary Interface)
- **New `nirs4all.run()` function**: Simplified entry point for training pipelines with intuitive parameters
- **New `nirs4all.predict()` function**: Make predictions on new data using trained models
- **New `nirs4all.explain()` function**: Generate SHAP explanations for model interpretability
- **New `nirs4all.retrain()` function**: Retrain pipelines on new data
- **New `nirs4all.session()` context manager**: Resource reuse across multiple pipeline runs
- **New `nirs4all.load_session()` function**: Load saved sessions from bundle files
- **Result wrapper classes**: `RunResult`, `PredictResult`, `ExplainResult` with convenient properties like `best_rmse`, `best_r2`, `top(n)`

#### sklearn Integration
- **New `NIRSPipeline` class**: sklearn-compatible wrapper for trained pipelines
- **New `NIRSPipelineClassifier` class**: Classification-specific sklearn wrapper
- **SHAP compatibility**: Use `NIRSPipeline` with SHAP explainers directly
- **`from_result()` and `from_bundle()` factory methods**: Easy creation from training results or exported bundles

#### Branching and Merging
- **`branch` keyword**: Create parallel execution paths with different preprocessing/models
- **`merge` keyword**: Combine branch outputs (features or predictions for stacking)
- **`source_branch` keyword**: Per-source preprocessing for multi-source datasets
- **`merge_sources` keyword**: Combine features from different data sources
- **Meta-model stacking**: Train meta-models on OOF predictions from base models
- **Branch disambiguation**: Unique naming for models across branches

#### Concat-Transform Pipelines
- **`concat_transform` keyword**: Concatenate outputs from multiple transformers
- **Action modes**: `extend`, `add`, `replace` for feature augmentation
- **Cartesian product generation**: `_cartesian_` keyword for multi-stage preprocessing combinations

#### Data Handling
- **Repetition transformation**: Convert spectral repetitions to sources or preprocessings
- **`RepetitionConfig` class**: Configure handling of unequal repetition counts (error, drop, pad, truncate)
- **3D array support**: Handle preprocessing dimensions in dataset merging
- **Aggregation methods**: Mean, median, vote with outlier exclusion using Hotelling's T²
- **Signal type detection**: Automatic detection and handling of reflectance, absorbance, Kubelka-Munk

#### Filtering and Quality Control
- **`SpectralQualityFilter`**: Filter samples by NaN ratio, Inf values, zero ratio, variance, value range
- **`XOutlierFilter`**: Detect outliers using Mahalanobis distance, PCA residual, Isolation Forest
- **`YOutlierFilter`**: Detect target outliers using IQR, Z-score, percentile, MAD methods

#### Preprocessing
- **`ReflectanceToAbsorbance` transformer**: Convert reflectance spectra using Beer-Lambert law
- **`PyBaselineCorrection` class**: Integration with pybaselines library (ASLS, AirPLS, SNIP, etc.)
- **Wavelet feature extraction**: `WaveletFeatures`, `WaveletPCA`, `WaveletSVD` classes
- **CARS and MC-UVE**: Feature selection methods for wavelength selection

#### Cross-Validation
- **`force_group` parameter**: Use any splitter with group-based splitting
- **`GroupedSplitterWrapper`**: Wrap non-group splitters for group-aware splitting
- **Fold sample ID remapping**: Correct handling of sample IDs across folds

#### Model Controllers
- **AutoGluon support**: `AutoGluonModelController` with `random_state` for reproducibility
- **GPU memory management**: Automatic memory reset before/after training for CatBoost and other GPU models
- **Customizable NN architecture**: Parameters for neural network architecture in model training

### 🔧 Improvements

#### Architecture
- **Artifact ID system V3**: Chain-based artifact IDs for better traceability in complex pipelines
- **Lazy backend loading**: TensorFlow and PyTorch loaded only when needed
- **`require_backend` utility**: Clear error messages when optional backends are missing
- **Centralized logging system**: Replace print statements with structured logging throughout
- **Exception hierarchy**: Centralized error management with error codes and context
- **`PredictionCache`**: Cache expensive prediction computations for performance

#### Pipeline Execution
- **Validation score calculation**: Prevent data leakage in BaseModelController
- **Step number tracking**: `RuntimeContext` tracks execution progress
- **Substep indexing**: TraceRecorder supports substep configurations within branches
- **Branch state restoration**: PipelineExecutor restores chain state from branch snapshots

#### Configuration
- **JSON/YAML file loading**: Load dataset configurations from external files
- **Detailed error handling**: Line and column numbers for invalid JSON/YAML syntax
- **Key normalization**: Standardize configuration keys across formats
- **Generator expansion**: Always expand generator syntax during configuration processing

#### Workspace
- **Simplified directory structure**: Remove date prefixes from artifact paths
- **Lazy directory creation**: Directories created only when needed
- **Model export functionality**: `PipelineRunner.export()` method for model bundles

#### Visualization
- **Branch diagram improvements**: Branch-specific metadata and improved clarity
- **NaN value handling**: ScoreHistogramChart handles missing values gracefully
- **Best per model option**: Filter to show only best result per model type

### 📚 Documentation

- **Complete documentation refactor**: Restructured docs with Sphinx, RTD theme
- **New API reference**: Module-level API, sklearn integration, data handling, synthetic generation
- **User guides**: Preprocessing guide, API migration guide, augmentation guide, synthetic data guide
- **Specifications**: Pipeline syntax, config format, metrics, nested CV
- **40+ pipeline examples**: Comprehensive catalog for branching, merging, multi-source
- **Reorganized examples**: User examples by topic, reference examples for syntax
- **Developer guide**: Synthetic data generator internals and extension
- **Synthetic data examples**: U09, U10 (user), D10, D11 (developer)

### 🧪 Testing

- **Comprehensive unit tests**: API module, sklearn wrappers, result classes
- **Integration tests**: Source branching, merging, multi-source, stacking
- **pytest-xdist support**: Parallel test execution with GPU markers
- **Performance benchmarks**: pytest-benchmark for regression detection

### 🐛 Bug Fixes

- **Fold sample ID handling**: Correct remapping of sample IDs to positional indices
- **Aggregation sorting**: Fixed sort order for aggregated predictions
- **Missing RMSE handling**: Use `.get()` to avoid KeyErrors for missing metrics
- **DiPLS prediction handling**: Fixed prediction logic for DiPLS models
- **Whitespace cleanup**: Fixed trailing whitespace in docstrings

### 🗑️ Removed

- **Deprecated example scripts**: Removed outdated JAX and LightGBM examples
- **Emoji utility module**: Replaced by structured logging system
- **Outdated documentation**: Removed obsolete reports and proposals

### ⚠️ Breaking Changes

- **Minimum Python version**: Now requires Python 3.11+
- **Dependency versions updated**: numpy>=1.24, scikit-learn>=1.2, pandas>=2.0
- **`save_files` parameter renamed**: Now `save_artifacts` and `save_charts`
- **Metric naming**: Some metrics renamed for consistency (e.g., RMSE to MSE in certain contexts)

### 📦 Dependencies

- **Updated minimum versions**: numpy, pandas, scipy, scikit-learn, and all optional dependencies
- **New optional dependencies**: ruff for linting, mypy for type checking, sphinx for docs
- **kennard-stone**: Updated to >=2.2.0
- **flax**: Added to JAX optional dependencies

---

## [0.5.1] - Charts performance fix - 2025-11-25

### Fixed
- **Charts**: Applied Polars optimization fix to all visualization charts in the library

## [0.5.0] - Enhanced Metrics and Visualization - 2025-11-24

### Added
- **Metrics**: Added new metrics including consistency, NRMSE (Normalized Root Mean Squared Error), NMSE (Normalized Mean Squared Error), and NMAE (Normalized Mean Absolute Error).
- **Metrics Management**: Implemented full metrics calculation for all partitions in `BaseModelController`.
- **Predictions**: Added scores management in `Predictions` class with serialization and retrieval support.
- **Visualization**: Enhanced heatmap and top-k comparison charts to display scores with local scaling options.
- **Documentation**: Added initial Sphinx documentation with project overview, features, and installation instructions.
- **Examples**: Changed binary dataset for improved testing scenarios.

### Changed
- **Architecture**: Refactored chart classes to standardize signatures and support optional metrics validation.
- **Controllers**: Updated `PredictionRanker` to utilize pre-computed scores with fallback to legacy methods.
- **Controllers**: Enhanced `PredictionSerializer` to handle serialization of scores.
- **Balancing**: Updated `BalancingCalculator` methods to fix `ref_percentage` that was equivalent to `max_factor`.
- **Charts**: Improved `FoldChartController` with better debugging and documentation.
- **CSV Export**: Updated `save_to_csv` method to accept `path_or_file` and `filename` parameters for better flexibility.

### Fixed
- **Pipeline**: Plot charts layout and display during pipeline execution.
- **Tests**: Updated balancing calculator tests to reflect new method signatures.

## [0.4.2] - Torch, Jax and sklearn style models - 2025-11-21

### Added
- **Model Support**: Fixed support for XGBoost, LightGBM, CatBoost models and added support sklearn style models.
- **Deep Learning**: Added JAX and PyTorch model controllers with data preparation utilities and `JaxModelWrapper` for state management.
- **Metrics**: Added balanced accuracy metric.
- **Pipeline**: Reintroduced parallel execution in pipeline steps (ongoing development).
- **Installation**: Enhanced backend detection and installation instructions (TensorFlow, PyTorch, JAX with GPU).
- **Inference**: Automatic inference for ranking logic (ascending parameter can be None).

### Changed
- **Architecture**: Refactored file saving architecture to implement "Return, Don't Save" pattern.
- **Controllers**: Refactored `BaseModelController` for improved execution flow and parallel training.
- **Controllers**: Updated `SklearnModelController` to enhance framework detection.
- **Examples**: Updated JAX and PyTorch model examples for prediction reuse.
- **Tests**: Refactored tests to use `RuntimeContext`.

### Fixed
- **Regression**: Ensure fold averages are only created for regression tasks with multiple folds.
- **Pipeline**: Correct pipeline definition by adding missing ShuffleSplit instance for regression comparison.

### Removed
- **Examples**: Removed deprecated example scripts for JAX and LightGBM.

## [0.4.1] - Folder/File structure rc - 2025-11-20

### Major Refactoring and Architecture Improvements
This release introduces significant architectural changes, refactoring the codebase for better modularity, type safety, and maintainability.

### Added
- **Core Architecture**:
  - **Folder Structure**: Complete reorganization of `controllers`, `core`, `data`, `tests`, `examples`, and `docs`.
  - **Context Handling**: Typed `ExecutionContext` and mutable `DataSelector`.
  - **Dataset**: `SpectroDataset` refactored to use `ArrayRegistry` and split-parquet storage.
  - **Pipeline**: Refactored execution, step handling, and artifact management.
  - **Models**: Modularized `BaseModelController`.
- **Features & Tools**:
  - `run.ps1` script for unified example execution.
  - `--show-plots` CLI argument.
  - `StratifiedKFold` and `StratifiedShuffleSplit` support.
  - New storage modules for pipeline management.
- **Feature Components Architecture**:
  - New modular component-based architecture for `FeatureSource`.
  - Type-safe enums for layouts and header units.
  - Six specialized components: `ArrayStorage`, `ProcessingManager`, `HeaderManager`, `LayoutTransformer`, `UpdateStrategy`, `AugmentationHandler`.
- **Predictions Components Architecture**:
  - New modular component-based architecture for `Predictions` class.
  - Six specialized components: `PredictionStorage`, `PredictionSerializer`, `PredictionIndexer`, `PredictionRanker`, `PartitionAggregator`, `CatalogQueryEngine`.

### Changed
- **Visualization**:
  - Refactored `FoldChart`, `SpectraChart`, `ConfusionMatrix`.
  - Improved classification visualization (discrete color mapping).
  - Reorganized SHAP and PCA analyzers.
- **Internal**:
  - Migrated component imports to internal `_*` modules.
  - Centralized evaluator and serialization logic.
  - Hardened Optuna sampling and logging.
- `FeatureSource` class moved to `nirs4all/data/feature_components/feature_source.py`.

### Fixed
- **Critical**: Missing `pipeline_uid` in prediction ranker results.
- **Critical**: NumPy array weights handling in predictions.
- Evaluator import path issues.
- Header unit preservation when adding samples.

### Documentation & Tests
- Restructured tests to mirror source code.
- Added a few architecture reviews, roadmap updates, and developer guides.
- Removed obsolete or review documents.

