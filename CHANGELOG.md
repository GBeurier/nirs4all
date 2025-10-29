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

### Changed
- `FeatureSource` class moved from `nirs4all/data/feature_source.py` to `nirs4all/data/feature_components/feature_source.py`
- Internal usage of layouts and header units now uses enums for type safety
- Improved error messages with enum validation

### Deprecated
- Direct import from `nirs4all.data.feature_source` (use `from nirs4all.data import FeatureSource` instead)

### Fixed
- Header unit preservation when adding samples with existing headers

### Backward Compatibility
- ✅ All existing code continues to work without modification
- ✅ String layouts and header units still accepted
- ✅ Public API remains unchanged
- ✅ Deprecation warnings for old import paths

### Developer Notes
- See `docs/developer/FEATURE_COMPONENTS_MIGRATION.md` for detailed migration guide
- See `docs/developer/FEATURES_REFACTORING_PROPOSAL.md` for complete refactoring documentation
- All tests passing with 100% backward compatibility

## [0.4.0] - Previous Release
<!-- Previous changelog entries -->
