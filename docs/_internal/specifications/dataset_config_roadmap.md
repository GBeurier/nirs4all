# Dataset Configuration Implementation Roadmap

**Version**: 1.7.0
**Status**: Complete
**Date**: December 2025
**Last Updated**: December 22, 2025
**Related Spec**: [dataset_config_specification.md](dataset_config_specification.md)

---

## ✅ Refactoring Summary (Completed)

The following modules have been refactored as part of Phase 1 through Phase 7:

| Module | Status | Changes Implemented |
|--------|--------|---------------------|
| [nirs4all/data/schema/](../../nirs4all/data/schema/) | ✅ **Complete** | Pydantic-based schema with `DatasetConfigSchema`, `FileConfig`, `ColumnConfig`, `PartitionConfig`, `LoadingParams`, `FoldConfig`, `SourceConfig`, `SharedTargetsConfig`, `SharedMetadataConfig`, `VariationConfig`, `VariationFileConfig`, `PreprocessingApplied` |
| [nirs4all/data/schema/validation/](../../nirs4all/data/schema/validation/) | ✅ **Complete** | `ConfigValidator` with comprehensive validation rules, error/warning handling |
| [nirs4all/data/parsers/](../../nirs4all/data/parsers/) | ✅ **Complete** | Modular parsers: `LegacyParser`, `FilesParser`, `SourcesParser`, `VariationsParser`, `FolderParser`, `ConfigNormalizer` |
| [nirs4all/data/loaders/base.py](../../nirs4all/data/loaders/base.py) | ✅ **Complete** | `FileLoader` base class, `LoaderRegistry`, `ArchiveHandler` |
| [nirs4all/data/loaders/](../../nirs4all/data/loaders/) | ✅ **Complete** | Format-specific loaders: CSV, NumPy, Parquet, Excel, MATLAB, Tar, Zip |
| [nirs4all/data/selection/](../../nirs4all/data/selection/) | ✅ **Complete** | Column/row selection: `ColumnSelector`, `RowSelector`, `RoleAssigner`, `SampleLinker` |
| [nirs4all/data/partition/](../../nirs4all/data/partition/) | ✅ **Complete** | Partition assignment: `PartitionAssigner` with static, column, percentage, and index methods |
| [nirs4all/controllers/splitters/fold_file_loader.py](../../nirs4all/controllers/splitters/fold_file_loader.py) | ✅ **Complete** | `FoldFileLoaderController` for loading saved fold files, `FoldFileParser` utility |

### Design Principles Applied

1. ✅ **Abstraction layers** - Pluggable loader system with registry pattern
2. ✅ **Schema-first approach** - Pydantic models for configuration validation
3. ✅ **Loader plugin system** - `FileLoader` base class with `load()`, `supports()`, `detect_format()` methods
4. ✅ **Backward compatibility** - Legacy syntax continues to work unchanged

---

## Implementation Phases

### Phase 1: Foundation & Schema ✅ COMPLETE

**Goal**: Establish a robust, extensible configuration system that supports the legacy format and provides the foundation for new features.

**Status**: ✅ Complete (December 2025)

#### 1.1 Configuration Schema Definition
- [x] Create `nirs4all/data/schema/` module
- [x] Define Pydantic/dataclass models for:
  - `FileConfig` (single file definition)
  - `ColumnConfig` (column selection/role assignment)
  - `PartitionConfig` (train/test/predict assignment)
  - `DatasetConfigSchema` (top-level configuration)
- [x] Add validation rules as model validators
- [x] Write comprehensive unit tests for schema validation (`tests/unit/data/test_schema.py`)

#### 1.2 Parser Refactoring
- [x] Refactor `config_parser.py` into:
  - `parsers/legacy_parser.py` - Handle current `train_x`/`test_x` format
  - `parsers/files_parser.py` - Handle new `files` syntax (stub)
  - `parsers/base.py` - Abstract base parser interface
- [x] Create `ConfigNormalizer` class to convert all input formats to canonical internal representation
- [x] Maintain 100% backward compatibility with existing tests
- [x] Add config format detection (legacy vs. new)

#### 1.3 Validation Layer
- [x] Create `nirs4all/data/schema/validation/` module
- [x] Implement validation rules from spec:
  - Required data checks
  - Column consistency
  - Partition consistency
  - File existence (warning on parse, error on load)
- [x] Provide clear, actionable error messages
- [x] Add validation hooks for custom rules

#### 1.4 Documentation & Tests
- [x] Update existing tests to use new internal APIs
- [x] Add schema validation test suite
- [x] Document migration path for any internal API changes

---

### Phase 2: File Loaders & Formats ✅ COMPLETE

**Goal**: Implement a pluggable file loading system supporting multiple formats.

**Status**: ✅ Complete (December 2025)

#### 2.1 Loader Architecture
- [x] Create `nirs4all/data/loaders/base.py`:
  ```python
  class FileLoader(ABC):
      @abstractmethod
      def load(self, path, params) -> LoaderResult

      @classmethod
      @abstractmethod
      def supports(cls, path: Path) -> bool

      @classmethod
      def detect_format(cls, path: Path) -> Optional[str]
  ```
- [x] Create `LoaderRegistry` for format-to-loader mapping
- [x] Refactor `csv_loader.py` to extend `FileLoader` (`csv_loader_new.py`)

#### 2.2 Archive Support Enhancement
- [x] Create `nirs4all/data/loaders/archive_loader.py`
- [x] Implement:
  - [x] Gzip support (`.gz`)
  - [x] Zip support (`.zip` single file)
  - [x] Zip multi-file with `member` selection
  - [x] Tar/TarGz support (`.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar.xz`)
- [x] Add `password` parameter for encrypted zip archives
- [x] Write integration tests for each archive format (`tests/unit/data/loaders/test_archive_loader.py`)

#### 2.3 New File Format Loaders
- [x] NumPy loader (`numpy_loader.py`):
  - [x] `.npy` single array files
  - [x] `.npz` multi-array files with `key` parameter
  - [x] `allow_pickle` parameter (default: False)
- [x] Parquet loader (`parquet_loader.py`):
  - [x] Basic Parquet support via `pyarrow` or `fastparquet`
  - [x] Column pruning (read only needed columns)
  - [x] Row group filtering (if applicable)
- [x] Excel loader (`excel_loader.py`):
  - [x] `.xlsx` and `.xls` support via `openpyxl`/`xlrd`
  - [x] Sheet selection (`sheet_name` parameter)
  - [x] Skip rows/footer parameters
- [x] MATLAB loader (`matlab_loader.py`):
  - [x] `.mat` file support via `scipy.io`
  - [x] Variable selection
  - [x] v7.3 (HDF5) support via `h5py`

#### 2.4 Format-Specific Parameters
- [x] Implement format-specific parameters in loaders
- [x] Add parameter inheritance: file-level > partition-level > global-level
- [x] Document all format-specific parameters

#### 2.5 Testing
- [x] Create test fixtures for each file format
- [x] Test format auto-detection
- [x] Test parameter merging
- [x] Test error handling for corrupt/missing files

---

### Phase 3: Column & Row Selection ✅ COMPLETE

**Goal**: Implement flexible column and row selection syntax.

**Status**: ✅ Complete (December 2025)

#### 3.1 Column Selection Syntax
- [x] Implement column selection methods:
  - [x] By name: `["col1", "col2"]`
  - [x] By index: `[0, 1, 2]` or `"0:10"`
  - [x] By range: `"2:-1"` (all except first 2 and last)
  - [x] By pattern: `{"regex": "^feature_.*"}`
  - [x] By exclusion: `{"exclude": ["id", "date"]}`
- [x] Create `ColumnSelector` class with chainable operations
- [x] Handle header vs. headerless files

#### 3.2 Row Selection Syntax
- [x] Implement row selection methods:
  - [x] All rows (default)
  - [x] By index: `[0, 1, 2]` or `"0:100"`
  - [x] By percentage: `"0:80%"`
  - [x] By condition: `{"where": {"column": "quality", "op": ">", "value": 0.5}}`
- [x] Create `RowSelector` class
- [x] Support random sampling with seed

#### 3.3 Role Assignment
- [x] Implement column role syntax:
  ```yaml
  columns:
    features: "2:-1"
    targets: -1
    metadata: [0, 1]
  ```
- [x] Validate no column overlap between roles
- [x] Support extracting Y from X columns

#### 3.4 Key-Based Sample Linking
- [x] Implement `link_by` parameter for multi-file datasets
- [x] Create `SampleLinker` class to join files by key column
- [x] Handle missing keys with configurable policy

---

### Phase 4: Partition System ✅ COMPLETE

**Goal**: Implement flexible partition assignment for train/test/predict splits.

**Status**: ✅ Complete (December 2025)

#### 4.1 Static Partition Assignment
- [x] Implement file-level `partition: train | test | predict`
- [x] Support partition in `files` syntax
- [x] Auto-concatenate files with same partition

#### 4.2 Column-Based Partition
- [x] Implement partition from data column:
  ```yaml
  partition:
    column: "split"
    train_values: ["train", "training"]
    test_values: ["test", "val", "validation"]
  ```
- [x] Support custom value mappings
- [x] Handle unknown values (error vs. ignore vs. assign to train)

#### 4.3 Percentage-Based Partition
- [x] Implement percentage splits:
  ```yaml
  partition:
    train: "0:80%"
    test: "80%:100%"
    shuffle: true
    random_state: 42
  ```
- [x] Add stratification option
- [x] Support range syntax ("0:50", "50:100")

#### 4.4 Index-Based Partition
- [x] Implement explicit index lists
- [x] Support loading indices from external files (TXT, JSON, YAML, CSV)
- [x] Validate no index overlap

#### 4.5 Backward Compatibility
- [x] Ensure legacy `train_x`/`test_x` still works
- [x] Auto-detect partition from file naming (current behavior)

#### 4.6 Implementation Details
- [x] Created `nirs4all/data/partition/` module with:
  - `PartitionAssigner` class for all partition methods
  - `PartitionResult` dataclass for results
  - `PartitionError` exception class
- [x] Updated `PartitionConfig` schema with full configuration options
- [x] Updated `FilesParser` to handle partition in files syntax
- [x] 36 passing tests in `tests/unit/data/partition/test_partition_assigner.py`

---

### Phase 5: Fold Definition ✅ COMPLETE

**Goal**: Support loading pre-computed cross-validation folds from saved files.

**Status**: ✅ Complete (December 2025)

> **Important Clarification**: Fold management in nirs4all follows a **two-tier approach**:
>
> 1. **Pipeline-level (Primary)**: Folds are generated by splitter operators (e.g., `KFold`, `ShuffleSplit`) during pipeline execution. These folds are saved as CSV artifacts in the workspace. To **reuse saved folds** in a new pipeline run, use the `split` keyword with a file path:
>    ```python
>    {"split": "path/to/folds_KFold_seed42.csv"}
>    ```
>
> 2. **Dataset-level (Secondary)**: For datasets that come with pre-defined fold assignments (e.g., benchmark datasets), fold information can be specified in the dataset configuration. This is less common but useful for reproducibility across different ML frameworks.
>
> The **primary use case** is loading fold files generated by previous pipeline runs, which is handled by the `FoldFileLoaderController`.

#### 5.1 Pipeline-Level Fold Loading (Primary Use Case) ✅
- [x] Create `FoldFileLoaderController` for loading saved fold files via pipeline syntax:
  ```python
  # Load pre-computed folds from a previous run
  pipeline = [
      MinMaxScaler(),
      {"split": "workspace/runs/my_run/folds_KFold_seed42.csv"},
      {"model": PLSRegression()}
  ]
  ```
- [x] Support nirs4all fold file format (CSV with `fold_0`, `fold_1`, ... columns containing sample IDs)
- [x] Support alternative fold file formats (JSON, YAML, TXT with index lists)
- [x] Validate fold sample IDs match current dataset samples
- [x] Create `FoldFileParser` utility for parsing various fold file formats

#### 5.2 Fold File Format Specification
- [x] Define standard nirs4all fold file format:
  ```csv
  fold_0,fold_1,fold_2
  0,1,2
  3,4,5
  6,7,8
  ...
  ```
  Each column represents a fold's train indices (sample IDs). Validation indices are computed as complement.
- [x] Support alternate single-column format with fold assignment per sample:
  ```csv
  sample_id,fold
  0,0
  1,1
  2,2
  ...
  ```
- [x] Document fold file format in specifications

#### 5.3 Dataset-Level Fold Configuration (Secondary Use Case)
- [x] Add `folds` field to `DatasetConfigSchema` for pre-defined fold definitions
- [x] Support inline fold definitions in config:
  ```yaml
  folds:
    - train: [0, 1, 2, 3, 4]
      val: [5, 6, 7, 8, 9]
    - train: [5, 6, 7, 8, 9]
      val: [0, 1, 2, 3, 4]
  ```
- [x] Support fold file reference in config:
  ```yaml
  folds:
    file: "path/to/folds.csv"
    format: auto  # auto-detect: csv, json, yaml
  ```
- [x] Support fold column in dataset:
  ```yaml
  folds:
    column: "cv_fold"  # Column in metadata containing fold assignments
  ```

#### 5.4 Integration with Existing Splitter System
- [x] `FoldFileLoaderController` uses same `dataset.set_folds()` API as `CrossValidatorController`
- [x] Fold files from previous runs can be directly loaded
- [x] Warning when loaded folds have mismatched sample counts

#### 5.5 Implementation Details
- [x] Created `nirs4all/controllers/splitters/fold_file_loader.py`:
  - `FoldFileLoaderController` - Pipeline controller for `{"split": "path/to/file"}`
  - `FoldFileParser` - Utility class for parsing various fold file formats
- [x] Added `FoldConfig` schema to `nirs4all/data/schema/config.py`
- [x] Unit tests in `tests/unit/controllers/test_fold_file_loader.py`
- [x] Integration tests demonstrating fold reuse across pipeline runs

---

### Phase 6: Multi-Source Datasets ✅ COMPLETE

**Goal**: Enhance sensor fusion / multi-instrument support.

**Status**: ✅ Complete (December 2025)

#### 6.1 Sources Syntax
- [x] Implement `sources` configuration:
  ```yaml
  sources:
    - name: "NIR"
      files:
        - path: data/NIR_train.csv
          partition: train
        - path: data/NIR_test.csv
          partition: test
    - name: "MIR"
      files:
        - path: data/MIR_train.csv
          partition: train
  ```
- [x] Each source becomes a separate feature matrix
- [x] Validate sample count consistency across sources
- [x] Support both `files` list and direct `train_x`/`test_x` paths per source

#### 6.2 Source-Level Parameters
- [x] Per-source loading parameters via `params` field
- [x] Per-source header units and signal types
- [x] Parameters merged with global_params (source-level takes precedence)

#### 6.3 Shared Targets and Metadata
- [x] Support shared `targets` block:
  ```yaml
  targets:
    path: data/targets.csv
    link_by: sample_id
  ```
- [x] Support shared `metadata` block:
  ```yaml
  metadata:
    path: data/metadata.csv
    link_by: sample_id
  ```
- [x] Support simple string paths for targets/metadata
- [x] Support partition-specific targets/metadata

#### 6.4 Multi-Source Pipeline Integration
- [x] Automatic conversion to legacy format for backward compatibility
- [x] `DatasetConfigSchema.to_legacy_format()` converts sources to `train_x`/`test_x` arrays
- [x] Source metadata preserved in `_sources` field for advanced processing
- [x] `ConfigNormalizer` automatically converts sources format to legacy format

#### 6.5 Implementation Details
- [x] Created schema models in `nirs4all/data/schema/config.py`:
  - `SourceConfig` - Configuration for a single feature source
  - `SourceFileConfig` - Configuration for files within a source
  - `SharedTargetsConfig` - Shared targets configuration
  - `SharedMetadataConfig` - Shared metadata configuration
- [x] Updated `DatasetConfigSchema` with:
  - `sources` field for multi-source configuration
  - `shared_targets` and `shared_metadata` fields
  - `is_sources_format()`, `get_source_names()`, `get_source_count()` methods
  - `to_legacy_format()` for backward compatibility
- [x] Implemented `SourcesParser` in `nirs4all/data/parsers/files_parser.py`
- [x] Updated `ConfigNormalizer` to handle sources format
- [x] 45 unit tests in `tests/unit/data/test_sources_parser.py`

---

### Phase 7: Feature Variations ✅ COMPLETE

**Goal**: Support pre-computed preprocessing and multiple feature representations.

**Status**: ✅ Complete (December 2025)

#### 7.1 Variations Syntax
- [x] Implement `variations` configuration:
  ```yaml
  variations:
    - name: "raw"
      files:
        - path: data/spectra_raw.csv
          partition: train
    - name: "snv"
      train_x: data/spectra_snv_train.csv
      test_x: data/spectra_snv_test.csv
    - name: "derivative"
      files: [...]
  ```
- [x] Parse and validate variation definitions
- [x] Ensure variation names are unique

#### 7.2 Variation Modes
- [x] Implement `variation_mode` options:
  - [x] `separate`: Each variation as independent pipeline (first variation returned, caller handles multiple)
  - [x] `concat`: Horizontal concatenation (wide features) - all variations combined as multi-source
  - [x] `select`: Use only specified variations via `variation_select`
  - [x] `compare`: Run all and rank by performance (same as separate, caller collects metrics)
- [x] Add `variation_select` for mode=select (with validation)
- [x] Add `variation_prefix` for mode=concat

#### 7.3 Variation Metadata
- [x] Store preprocessing provenance in variations via `PreprocessingApplied` schema:
  ```yaml
  variations:
    - name: "snv_sg"
      description: "SNV followed by SG derivative"
      preprocessing_applied:
        - type: "SNV"
          software: "OPUS 8.0"
        - type: "SG_derivative"
          params:
            window: 15
            polyorder: 2
  ```
- [x] Variation metadata preserved in `_variations` field of legacy format

#### 7.4 Schema and Parser Implementation
- [x] Created `VariationConfig` schema model with:
  - `name`, `description` fields
  - `files` list or `train_x`/`test_x` direct paths
  - `params` for loading parameters
  - `preprocessing_applied` for provenance tracking
  - `get_train_paths()` and `get_test_paths()` methods
- [x] Created `VariationFileConfig` for files within variations
- [x] Created `PreprocessingApplied` model for preprocessing metadata
- [x] Created `VariationMode` enum (SEPARATE, CONCAT, SELECT, COMPARE)
- [x] Created `VariationsParser` for parsing variations syntax
- [x] Updated `ConfigNormalizer` to handle variations format

#### 7.5 Legacy Format Conversion
- [x] `variations_to_legacy_format()` method converts variations to train_x/test_x format
- [x] Mode-aware conversion:
  - `separate`/`compare`: Returns first variation (caller handles multiple runs)
  - `concat`: Returns all variation paths as multi-source array
  - `select`: Returns only selected variations
- [x] Preserved variation metadata in `_variations` and `_variation_mode` fields

#### 7.6 Validation
- [x] Variation names must be unique
- [x] `variation_mode=select` requires `variation_select`
- [x] Unknown variation names in `variation_select` are rejected

#### 7.7 Implementation Details
- [x] Updated `nirs4all/data/schema/config.py`:
  - `VariationConfig` - Configuration for a single feature variation
  - `VariationFileConfig` - Configuration for files within a variation
  - `PreprocessingApplied` - Preprocessing provenance metadata
  - `VariationMode` enum
  - `is_variations_format()`, `get_variation_names()`, `get_variation_count()` methods
  - `get_selected_variations()` for mode=select filtering
  - `variations_to_legacy_format()` for backward compatibility
- [x] Created `VariationsParser` in `nirs4all/data/parsers/files_parser.py`
- [x] Updated `nirs4all/data/parsers/normalizer.py` to handle variations
- [x] Updated `nirs4all/data/schema/__init__.py` with new exports
- [x] Updated `nirs4all/data/parsers/__init__.py` with new exports
- [x] 40 unit tests in `tests/unit/data/test_variations_parser.py`

---

### Phase 8: Advanced Features ✅ COMPLETE

**Goal**: Implement remaining specification features and polish.

**Status**: ✅ Complete (December 2025)

#### 8.1 Sample Aggregation Enhancements
- [x] Aggregation column detection from config
- [x] Aggregation during loading (not just prediction)
- [x] Custom aggregation functions
- [x] Outlier exclusion with configurable threshold
- [x] Multiple aggregation methods (mean, median, vote, min, max, sum)

#### 8.2 Auto-Detection Improvements
- [x] Improve delimiter detection (comma, tab, semicolon, pipe, space)
- [x] Improve decimal separator detection (period vs comma)
- [x] Header detection heuristics (numeric vs text first row)
- [x] Signal type auto-detection from header patterns (nm, cm-1, wavelength keywords)
- [x] Confidence scoring for detection results

#### 8.3 Configuration Serialization
- [x] Serialize loaded configs back to YAML/JSON
- [x] Normalize configs before serialization
- [x] Support config diffing for reproducibility
- [x] Handle numpy arrays, enums, Path objects in serialization

#### 8.4 Error Handling & Diagnostics
- [x] Comprehensive error codes (E1xx-E9xx categories)
- [x] Config validation CLI command (`nirs4all dataset validate`)
- [x] Suggestion system for common mistakes
- [x] Verbose mode with loading diagnostics
- [x] DiagnosticBuilder and DiagnosticReport classes
- [x] CLI commands: validate, inspect, export, diff

#### 8.5 Performance Optimization
- [x] Lazy loading for large datasets (LazyArray, LazyDataset)
- [x] LRU cache with size limits and TTL
- [x] File modification detection for cache invalidation
- [x] Thread-safe cache access
- [x] Deferred loading until data access

#### 8.6 Documentation & Examples
- [x] Migration guide from legacy format (`docs/user_guide/dataset_migration_guide.md`)
- [x] Example configs for each use case (`examples/configs/`)
- [x] Troubleshooting guide (`docs/user_guide/dataset_troubleshooting.md`)
- [x] Error code reference documentation

#### 8.7 Implementation Details
- [x] Created `nirs4all/data/aggregation/` module:
  - `Aggregator` class with configurable aggregation methods
  - `AggregationConfig` dataclass for configuration
  - `AggregationMethod` enum (MEAN, MEDIAN, VOTE, MIN, MAX, SUM, CUSTOM)
  - Outlier exclusion with Z-score threshold
- [x] Created `nirs4all/data/detection/` module:
  - `AutoDetector` class for file parameter detection
  - `DetectionResult` dataclass with confidence scores
  - Delimiter, decimal, header, and signal type detection
- [x] Created `nirs4all/data/serialization/` module:
  - `ConfigSerializer` class for YAML/JSON serialization
  - `ConfigDiff` dataclass for configuration comparison
  - `SerializationFormat` enum
- [x] Created `nirs4all/data/performance/` module:
  - `DataCache` class with LRU eviction and TTL
  - `LazyArray` class with numpy-compatible interface
  - `LazyDataset` class for deferred dataset loading
- [x] Created `nirs4all/data/schema/validation/error_codes.py`:
  - `ErrorRegistry` with E1xx-E9xx error categories
  - `DiagnosticMessage`, `DiagnosticBuilder`, `DiagnosticReport` classes
- [x] Created `nirs4all/cli/commands/dataset.py`:
  - `dataset validate` - Validate configuration files
  - `dataset inspect` - Inspect data files with auto-detection
  - `dataset export` - Export normalized configurations
  - `dataset diff` - Compare configuration files

---

## Dependency Graph

```
Phase 1 (Foundation) ✅
    │
    ├──► Phase 2 (File Loaders) ✅
    │        │
    │        └──► Phase 3 (Column/Row Selection) ✅
    │                 │
    │                 └──► Phase 4 (Partitions) ✅
    │                          │
    │                          ├──► Phase 5 (Folds) ✅
    │                          │
    │                          └──► Phase 6 (Multi-Source) ✅
    │                                   │
    │                                   └──► Phase 7 (Variations) ✅
    │
    └──► Phase 8 (Advanced) ✅ ◄── All phases complete
```

---

## Testing Strategy

### Unit Tests
- Schema validation tests
- Parser tests for each input format
- Loader tests for each file format
- Column/row selection tests
- Partition logic tests

### Integration Tests
- End-to-end config loading
- Multi-file dataset loading
- Multi-source data handling
- Variation mode workflows

### Regression Tests
- All existing examples must pass
- Legacy config format must work unchanged
- Performance benchmarks

### Test Data
- Create `tests/data/configs/` with sample configs
- Create `tests/data/samples/` with sample data files
- Include edge cases (empty files, malformed data, etc.)

---

## Migration Path

### For Users

1. **No action required for existing configs** - Legacy format continues to work
2. **Gradual adoption** - New features can be adopted incrementally
3. **Deprecation warnings** - Will be added before any breaking changes (v2.0+)

### For Internal Code

1. Phase 1 refactoring may change internal APIs
2. All internal consumers will be updated during refactoring
3. Test coverage ensures no regressions

---

## Milestones & Deliverables

| Milestone | Target | Deliverables |
|-----------|--------|--------------|
| M1: Foundation Complete | Week 3 | Schema, refactored parser, validation layer |
| M2: Multi-Format Support | Week 6 | All file format loaders, archive support |
| M3: Selection Syntax | Week 9 | Column/row selection, role assignment |
| M4: Partition System | Week 12 | All partition modes, backward compat verified |
| M5: CV Folds | Week 14 | Fold definition, external fold files |
| M6: Multi-Source | Week 17 | Sources syntax, sensor fusion support |
| M7: Variations | Week 20 | Variation modes, pipeline integration |
| M8: Release Ready | Week 24 | Documentation, examples, polish |

---

## 🎉 Core Implementation Complete

Phases 1-8 of the Dataset Configuration Implementation Roadmap have been completed. The nirs4all library now supports:

- **Flexible file formats**: CSV, NumPy, Parquet, Excel, MATLAB, plus archive support (zip, tar, gzip)
- **Advanced column/row selection**: By name, index, range, pattern, exclusion, and conditions
- **Multiple partition methods**: Static, column-based, percentage-based, and index-based
- **Cross-validation fold support**: Pipeline-level and dataset-level fold definitions
- **Multi-source datasets**: Sensor fusion with shared targets and metadata
- **Feature variations**: Pre-computed preprocessing with multiple modes (separate, concat, select, compare)
- **Sample aggregation**: Mean, median, vote, min, max, sum with outlier exclusion
- **Auto-detection**: Delimiter, decimal separator, headers, and signal types
- **Performance optimization**: Lazy loading, LRU caching with TTL
- **Comprehensive error handling**: Error codes E1xx-E9xx with diagnostic reports
- **CLI tools**: validate, inspect, export, diff commands

Full backward compatibility with legacy format is maintained.

**Future Work**: Phase 9 (Asymmetric Multi-Source Support) is planned to enable sources with different feature dimensions and preprocessing counts.

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing configs | High | Extensive regression testing, deprecation warnings |
| Performance degradation | Medium | Benchmarking, lazy loading, caching |
| Scope creep | Medium | Strict phase boundaries, MVP focus |
| Complex edge cases | Medium | Comprehensive test suite, user feedback |

---

## Open Questions

1. **Schema validation library**: Pydantic vs. dataclasses + manual validation?
   - Recommendation: Pydantic for richer validation, but keep dependencies minimal

2. **Lazy loading strategy**: When to load file contents?
   - Recommendation: Load on first access, cache results

3. **Multi-source memory management**: How to handle very large multi-source datasets?
   - Recommendation: Support memory-mapped arrays, chunked loading

4. **Variation mode=compare**: How to surface comparison results?
   - Recommendation: Integrate with existing visualization/reporting system

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | Dec 2025 | - | Initial roadmap based on specification v1.0.2-draft |
| 1.1.0 | Dec 20, 2025 | - | Updated Phase 1 and 2 as complete; Phase 3 in progress |
| 1.2.0 | Dec 21, 2025 | - | Phase 3 complete: ColumnSelector, RowSelector, RoleAssigner, SampleLinker implemented with 101 passing tests |
| 1.3.0 | Dec 21, 2025 | - | Phase 4 complete: PartitionAssigner with static, column, percentage, and index methods. FilesParser now functional. 114 total tests passing. |
| 1.4.0 | Dec 21, 2025 | - | Phase 5 complete: FoldFileLoaderController for pipeline-level fold loading, FoldFileParser for parsing fold files, FoldConfig schema for dataset-level fold configuration. |
| 1.5.0 | Dec 21, 2025 | - | Phase 6 complete: Multi-source datasets with SourceConfig, SourcesParser, SharedTargetsConfig, SharedMetadataConfig. Automatic conversion to legacy format for backward compatibility. 45 new tests, 889 total tests passing. |
| 1.6.0 | Dec 21, 2025 | - | Phase 7 complete: Feature variations with VariationConfig, VariationFileConfig, PreprocessingApplied, VariationMode, VariationsParser. Support for separate, concat, select, compare modes. Automatic conversion to legacy format. 40 new tests, 924 total tests passing. |
| 1.7.0 | Dec 22, 2025 | - | Phase 8 complete: Advanced features including sample aggregation (Aggregator, AggregationConfig), auto-detection (AutoDetector, DetectionResult), config serialization (ConfigSerializer, ConfigDiff), error handling (ErrorRegistry E1xx-E9xx, DiagnosticBuilder), performance (DataCache, LazyArray, LazyDataset), and documentation (migration guide, example configs, troubleshooting guide). CLI commands: validate, inspect, export, diff. |

---

## Future Phases

### Phase 9: Asymmetric Multi-Source Data Access 🔜 PLANNED

**Goal**: Improve data access for multi-source datasets where sources have different feature dimensions and/or different numbers of preprocessings.

**Status**: 🔜 Planned

> ⚠️ **Scope Clarification**
>
> This phase focuses on **data loading and access** improvements only:
> - Better error messages when asymmetric sources cause concat failures
> - Source selection by name or index in data queries
> - Introspection APIs for source metadata
>
> **Runtime pipeline features** (source branching, merge controllers) are documented separately in:
> - [branching.md](../reference/branching.md) — Pipeline branching reference
> - [branching_merging_analysis.md](../report/branching_merging_analysis.md) — Analysis and roadmap for merge features
>
> Dataset configuration describes **data structure**, not **processing strategy**.

#### Problem Summary

When sources have asymmetric preprocessing counts (e.g., NIR: 3 preprocessings, Timeseries: 1 preprocessing), requesting 3D layout with source concatenation fails:

```python
# Source 1: (500 samples, 3 processings, 500 features)
# Source 2: (500 samples, 1 processing, 300 features)

X = dataset.x(selector, layout="3d", concat_source=True)
# FAILS: Cannot concatenate when processing dimension differs
```

#### In-Scope Deliverables (Data Access)

| Phase | Title | Key Deliverables |
|-------|-------|------------------|
| 9A | Validation & Error Handling | `sources_compatible()`, `SourceConcatError`, `on_incompatible` parameter |
| 9B | Selective Source Retrieval | `sources` parameter (by name or index), `source_info()` introspection |

#### Out of Scope (See Pipeline Documentation)

The following features are **pipeline runtime concerns** and will be documented/implemented separately:

- **Source Branching** (`source_branch` keyword, `SourceBranchController`) — Route each source through its own pipeline branch
- **Merge Controller** (`merge_sources`, `merge_predictions`) — Combine branch outputs
- **Multi-Head Model Support** (`MultiInputModel`, `source_inputs`) — Models accepting multiple input tensors

See [branching_merging_analysis.md](../report/branching_merging_analysis.md) for the comprehensive analysis and implementation roadmap for these features.

#### Resolution Strategies for Asymmetric Concat

| Strategy | Behavior | When to Use |
|----------|----------|-------------|
| **Error** (default) | Raise `SourceConcatError` with resolution options | Explicit user intervention |
| **Flatten** (`on_incompatible="flatten"`) | Flatten each source to 2D, then concat | When processing structure not needed |
| **Separate** (`on_incompatible="separate"`) | Return list instead of array | When sources need different handling |
