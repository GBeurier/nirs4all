# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - Operator Refactoring & Controller-Managed Variation - 2026-02-05

> **⚠️ Documentation Notice:** Due to the extensive scope of this release, some documentation may be temporarily incomplete or out of sync. Updates are in progress.

### ⚠ BREAKING CHANGES

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

#### Synthesis Module Moved
- **`nirs4all.data.synthetic`** → **`nirs4all.synthesis`** — update all imports
- Generator now delegates to operators for path length, instrumental broadening, and noise effects

### New Features

#### Controller-Managed Variation (`variation_scope`)
- New `variation_scope` parameter at the `sample_augmentation` step level: `"sample"` (default), `"batch"`
- Per-transformer override via dict spec: `{"transformer": ..., "variation_scope": "batch"}`
- Hybrid performance model: operators with `_supports_variation_scope = True` handle variation internally; others get per-sample cloning from controller

#### New Augmentation Operators
- **`PathLengthAugmenter`**: Multiplicative path length variation
- **`BatchEffectAugmenter`**: Wavelength-dependent batch effects (offset + gain)
- **`InstrumentalBroadeningAugmenter`**: Gaussian convolution broadening (FWHM-based)
- **`HeteroscedasticNoiseAugmenter`**: Signal-dependent noise
- **`DeadBandAugmenter`**: Random dead band (non-responsive region) simulation

#### Webapp Updates
- 39 augmentation nodes in the node registry (was 8)
- New subcategories: spectral-noise, spectral-baseline, spectral-wavelength, spectral-smoothing, spectral-masking, spectral-mixing, environmental, scattering, edge-artifacts, synthesis
- `variation_scope` parameter added to SampleAugmentation container
- Removed `apply_on`, `copy` from all webapp node definitions

### Improvements

#### Feature Selection Enhancement
- **`FlexiblePCA` and `FlexibleSVD` classes**: New flexible dimensionality reduction with enhanced documentation
- Enhanced feature selection module documentation

#### Storage & Infrastructure
- **DuckDB storage migration**: Refactored artifact storage from `binaries/` to `artifacts/` directory
- Updated tests to support DuckDB storage and new directory structure
- Removed deprecated directory checks in test suite

#### Testing
- Added integration tests for bundle export and prediction with special operator types
- Comprehensive test coverage for new storage structure
- Updated test assertions for `store.duckdb` persistence

#### Misc
- Changed file permissions for CI example scripts
- Removed obsolete tests: `test_catalog_export.py`, `test_library_manager.py`, `test_query_reporting.py`

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

