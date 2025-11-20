# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2025-11-20

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

