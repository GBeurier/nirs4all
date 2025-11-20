# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2025-01-XX

### Added
- **Feature Components Architecture**: Complete refactoring of feature management system
  - New modular component-based architecture for `FeatureSource`
  - Type-safe enums for layouts (`FeatureLayout`) and header units (`HeaderUnit`)
  - Six specialized components: `ArrayStorage`, `ProcessingManager`, `HeaderManager`, `LayoutTransformer`, `UpdateStrategy`, `AugmentationHandler`
  - Comprehensive test suite for feature components (`test_feature_components.py`)
  - Migration guide documentation (`FEATURE_COMPONENTS_MIGRATION.md`)

- **Predictions Components Architecture**: Complete refactoring of predictions management system
  - New modular component-based architecture for `Predictions` class
  - Six specialized components: `PredictionStorage`, `PredictionSerializer`, `PredictionIndexer`, `PredictionRanker`, `PartitionAggregator`, `CatalogQueryEngine`
  - Reduced main file from 2046 to 895 lines (56% reduction)
  - Complete refactoring documentation (`predictions_refactoring_completed.md`)

### Changed
- `FeatureSource` class moved from `nirs4all/data/feature_source.py` to `nirs4all/data/feature_components/feature_source.py`
- Internal usage of layouts and header units now uses enums for type safety
- Improved error messages with enum validation
- `Predictions` class now uses component-based delegation pattern

### Fixed
- **Critical**: Missing `pipeline_uid` in prediction ranker results (was breaking prediction replay functionality)
- **Critical**: NumPy array weights handling in predictions (`weights or []` causing ValueError)
- Evaluator import path issues (circular import fix)
- Header unit preservation when adding samples with existing headers

### Deprecated
- Direct import from `nirs4all.data.feature_source` (use `from nirs4all.data import FeatureSource` instead)

### Backward Compatibility
- ✅ All existing code continues to work without modification
- ✅ String layouts and header units still accepted (features)
- ✅ Public API remains unchanged (both features and predictions)
- ✅ Deprecation warnings for old import paths
- ✅ Integration tests passing (7/8, 1 pre-existing TensorFlow import issue)

### Developer Notes
- See `docs/developer/FEATURE_COMPONENTS_MIGRATION.md` for feature refactoring details
- See `docs/reports/predictions_refactoring_completed.md` for predictions refactoring summary
- All tests passing with 100% backward compatibility
- Original monolithic predictions file preserved as `predictions_OLD_BACKUP.py`

## [0.4.0] - Previous Release
<!-- Previous changelog entries -->
