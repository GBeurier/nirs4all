# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.1] - GitHub Actions CI/CD Update - 2025-12-31

### 🔧 Infrastructure

- **GitHub Actions**: Updated CI/CD workflows for improved build and deployment processes

## [0.6.0] - Major API Overhaul and Architecture Improvements - 2025-12-27

This release introduces a new module-level API, complete documentation overhaul, sklearn integration, branching/merging pipelines, synthetic data generation, and extensive architectural improvements.

### ✨ New Features

#### Synthetic Data Generation (NEW)
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

