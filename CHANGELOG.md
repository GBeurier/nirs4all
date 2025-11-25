# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

