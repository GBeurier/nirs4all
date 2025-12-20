# Dataset Configuration Implementation Roadmap

**Version**: 1.0.0
**Status**: Planning Document
**Date**: December 2025
**Related Spec**: [dataset_config_specification.md](dataset_config_specification.md)

---

## ⚠️ Required Refactoring Summary

Before or during implementation, the following modules require refactoring:

| Module | Current State | Required Changes | Priority |
|--------|--------------|------------------|----------|
| [nirs4all/data/config_parser.py](../../nirs4all/data/config_parser.py) | Monolithic parsing with hardcoded key mappings | Extract into modular schema-based parser with validation layer | **Phase 1** |
| [nirs4all/data/loaders/loader.py](../../nirs4all/data/loaders/loader.py) | Tightly coupled to legacy key format (`train_x`, `test_x`) | Abstract into pluggable loader interface supporting `files` and `sources` syntax | **Phase 2** |
| [nirs4all/data/loaders/csv_loader.py](../../nirs4all/data/loaders/csv_loader.py) | CSV-only with embedded decompression | Extract format detection; create base `FileLoader` class with format-specific subclasses | **Phase 2** |
| [nirs4all/data/config.py](../../nirs4all/data/config.py) | `DatasetConfigs` handles both parsing and caching | Separate concerns: parsing → validation → loading → caching | **Phase 1** |

### Refactoring Recommendations

1. **Create abstraction layers before adding features** - The current implementation is functional but tightly coupled. Adding `files`, `sources`, and `variations` syntax directly would create technical debt.

2. **Schema-first approach** - Implement a formal configuration schema (Pydantic or dataclass-based) that can validate configs before processing.

3. **Loader plugin system** - Create a `FileLoader` base class with `load()`, `supports()`, and `detect_format()` methods. Register loaders for each format.

4. **Preserve backward compatibility** - All legacy syntax must continue working. New syntax should be normalized to an internal canonical representation.

---

## Implementation Phases

### Phase 1: Foundation & Schema (Weeks 1-3)

**Goal**: Establish a robust, extensible configuration system that supports the legacy format and provides the foundation for new features.

#### 1.1 Configuration Schema Definition
- [ ] Create `nirs4all/data/schema/` module
- [ ] Define Pydantic/dataclass models for:
  - `FileConfig` (single file definition)
  - `ColumnConfig` (column selection/role assignment)
  - `PartitionConfig` (train/test/predict assignment)
  - `DatasetConfig` (top-level configuration)
- [ ] Add validation rules as model validators
- [ ] Write comprehensive unit tests for schema validation

#### 1.2 Parser Refactoring
- [ ] Refactor `config_parser.py` into:
  - `parsers/legacy_parser.py` - Handle current `train_x`/`test_x` format
  - `parsers/files_parser.py` - Handle new `files` syntax (stub)
  - `parsers/base.py` - Abstract base parser interface
- [ ] Create `ConfigNormalizer` class to convert all input formats to canonical internal representation
- [ ] Maintain 100% backward compatibility with existing tests
- [ ] Add config format detection (legacy vs. new)

#### 1.3 Validation Layer
- [ ] Create `nirs4all/data/validation/` module
- [ ] Implement validation rules from spec:
  - Required data checks
  - Column consistency
  - Partition consistency
  - File existence (warning on parse, error on load)
- [ ] Provide clear, actionable error messages
- [ ] Add validation hooks for custom rules

#### 1.4 Documentation & Tests
- [ ] Update existing tests to use new internal APIs
- [ ] Add schema validation test suite
- [ ] Document migration path for any internal API changes

---

### Phase 2: File Loaders & Formats (Weeks 4-6)

**Goal**: Implement a pluggable file loading system supporting multiple formats.

#### 2.1 Loader Architecture
- [ ] Create `nirs4all/data/loaders/base.py`:
  ```python
  class FileLoader(ABC):
      @abstractmethod
      def load(self, path, params) -> pd.DataFrame

      @classmethod
      @abstractmethod
      def supports(cls, path: Path) -> bool

      @classmethod
      def detect_format(cls, path: Path) -> Optional[str]
  ```
- [ ] Create `LoaderRegistry` for format-to-loader mapping
- [ ] Refactor `csv_loader.py` to extend `FileLoader`

#### 2.2 Archive Support Enhancement
- [ ] Create `nirs4all/data/loaders/archive_loader.py`
- [ ] Implement:
  - [x] Gzip support (`.gz`) - *already implemented*
  - [x] Zip support (`.zip` single file) - *already implemented*
  - [ ] Zip multi-file with `archive_params.member` selection
  - [ ] Tar/TarGz support (`.tar`, `.tar.gz`, `.tgz`)
- [ ] Add `password` parameter for encrypted archives
- [ ] Write integration tests for each archive format

#### 2.3 New File Format Loaders
- [ ] NumPy loader (`numpy_loader.py`):
  - [ ] `.npy` single array files
  - [ ] `.npz` multi-array files with `key` parameter
  - [ ] `allow_pickle` parameter (default: False)
- [ ] Parquet loader (`parquet_loader.py`):
  - [ ] Basic Parquet support via `pyarrow` or `fastparquet`
  - [ ] Column pruning (read only needed columns)
  - [ ] Row group filtering (if applicable)
- [ ] Excel loader (`excel_loader.py`):
  - [ ] `.xlsx` and `.xls` support via `openpyxl`/`xlrd`
  - [ ] Sheet selection (`sheet_name` parameter)
  - [ ] Skip rows/footer parameters
- [ ] MATLAB loader (`matlab_loader.py`):
  - [ ] `.mat` file support via `scipy.io`
  - [ ] Variable selection

#### 2.4 Format-Specific Parameters
- [ ] Implement `csv_params`, `numpy_params`, `excel_params`, `archive_params` in schema
- [ ] Add parameter inheritance: file-level > partition-level > global-level
- [ ] Document all format-specific parameters

#### 2.5 Testing
- [ ] Create test fixtures for each file format
- [ ] Test format auto-detection
- [ ] Test parameter merging
- [ ] Test error handling for corrupt/missing files

---

### Phase 3: Column & Row Selection (Weeks 7-9)

**Goal**: Implement flexible column and row selection syntax.

#### 3.1 Column Selection Syntax
- [ ] Implement column selection methods:
  - [ ] By name: `["col1", "col2"]`
  - [ ] By index: `[0, 1, 2]` or `"0:10"`
  - [ ] By range: `"2:-1"` (all except first 2 and last)
  - [ ] By pattern: `{"regex": "^feature_.*"}`
  - [ ] By exclusion: `{"exclude": ["id", "date"]}`
- [ ] Create `ColumnSelector` class with chainable operations
- [ ] Handle header vs. headerless files

#### 3.2 Row Selection Syntax
- [ ] Implement row selection methods:
  - [ ] All rows (default)
  - [ ] By index: `[0, 1, 2]` or `"0:100"`
  - [ ] By percentage: `"0:80%"`
  - [ ] By condition: `{"where": {"column": "quality", "op": ">", "value": 0.5}}`
- [ ] Create `RowSelector` class
- [ ] Support random sampling with seed

#### 3.3 Role Assignment
- [ ] Implement column role syntax:
  ```yaml
  columns:
    features: "2:-1"
    targets: -1
    metadata: [0, 1]
  ```
- [ ] Validate no column overlap between roles
- [ ] Support extracting Y from X columns

#### 3.4 Key-Based Sample Linking
- [ ] Implement `link_by` parameter for multi-file datasets
- [ ] Create `SampleLinker` class to join files by key column
- [ ] Handle missing keys with configurable policy

---

### Phase 4: Partition System (Weeks 10-12)

**Goal**: Implement flexible partition assignment for train/test/predict splits.

#### 4.1 Static Partition Assignment
- [ ] Implement file-level `partition: train | test | predict`
- [ ] Support partition in `files` syntax
- [ ] Auto-concatenate files with same partition

#### 4.2 Column-Based Partition
- [ ] Implement partition from data column:
  ```yaml
  partition:
    column: "split"
    train_values: ["train", "training"]
    test_values: ["test", "val", "validation"]
  ```
- [ ] Support custom value mappings
- [ ] Handle unknown values (error vs. ignore)

#### 4.3 Percentage-Based Partition
- [ ] Implement percentage splits:
  ```yaml
  partition:
    train: "0:80%"
    test: "80%:100%"
    shuffle: true
    random_state: 42
  ```
- [ ] Add stratification option
- [ ] Support group-aware splitting

#### 4.4 Index-Based Partition
- [ ] Implement explicit index lists
- [ ] Support loading indices from external files
- [ ] Validate no index overlap

#### 4.5 Backward Compatibility
- [ ] Ensure legacy `train_x`/`test_x` still works
- [ ] Auto-detect partition from file naming (current behavior)

---

### Phase 5: Fold Definition (Weeks 13-14)

**Goal**: Support pre-defined cross-validation folds in configuration.

#### 5.1 Pre-defined Fold Indices
- [ ] Implement fold syntax:
  ```yaml
  folds:
    - train: [0, 1, 2, 3, 4]
      val: [5, 6, 7, 8, 9]
    - train: [5, 6, 7, 8, 9]
      val: [0, 1, 2, 3, 4]
  ```
- [ ] Validate fold indices within partition bounds
- [ ] Ensure train/val don't overlap within fold

#### 5.2 Fold Column in Data
- [ ] Support fold assignment from data column:
  ```yaml
  folds:
    column: "cv_fold"
  ```
- [ ] Map column values to fold indices

#### 5.3 External Fold Files
- [ ] Load folds from JSON/YAML/CSV files
- [ ] Support multiple file formats for fold definitions

#### 5.4 Generated Folds (Shorthand)
- [ ] Implement fold generation hints:
  ```yaml
  folds:
    method: kfold | stratified_kfold | group_kfold
    n_splits: 5
    group_column: "patient_id"
  ```
- [ ] These act as hints to pipeline splitters
- [ ] Document interaction with pipeline-level CV

---

### Phase 6: Multi-Source Datasets (Weeks 15-17)

**Goal**: Enhance sensor fusion / multi-instrument support.

#### 6.1 Sources Syntax
- [ ] Implement `sources` configuration:
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
- [ ] Each source becomes a separate feature matrix
- [ ] Validate sample count consistency across sources

#### 6.2 Source-Level Parameters
- [ ] Per-source loading parameters
- [ ] Per-source header units and signal types
- [ ] Source-specific preprocessing hints

#### 6.3 Shared Targets and Metadata
- [ ] Support shared `targets` block:
  ```yaml
  targets:
    - path: data/targets.csv
      link_by: sample_id
  ```
- [ ] Link targets to all sources by key
- [ ] Handle missing samples across sources

#### 6.4 Multi-Source Pipeline Integration
- [ ] Ensure pipeline operators receive multi-source data correctly
- [ ] Document multi-source pipeline patterns
- [ ] Add examples for sensor fusion workflows

---

### Phase 7: Feature Variations (Weeks 18-20)

**Goal**: Support pre-computed preprocessing and multiple feature representations.

#### 7.1 Variations Syntax
- [ ] Implement `variations` configuration:
  ```yaml
  variations:
    - name: "raw"
      files: [...]
    - name: "snv"
      files: [...]
    - name: "derivative"
      files: [...]
  ```
- [ ] Parse and validate variation definitions
- [ ] Ensure consistent sample counts across variations

#### 7.2 Variation Modes
- [ ] Implement `variation_mode` options:
  - [ ] `separate`: Each variation as independent pipeline
  - [ ] `concat`: Horizontal concatenation (wide features)
  - [ ] `select`: Use only specified variations
  - [ ] `compare`: Run all and rank by performance
- [ ] Add `variation_select` for mode=select
- [ ] Add `variation_prefix` for mode=concat

#### 7.3 Variation Metadata
- [ ] Store preprocessing provenance in variations:
  ```yaml
  variations:
    - name: "snv"
      preprocessing_applied: "SNV normalization"
  ```
- [ ] Make metadata available to pipeline for logging

#### 7.4 Pipeline Integration
- [ ] Modify pipeline runner to handle variation modes
- [ ] For `separate`: spawn multiple pipeline runs
- [ ] For `compare`: collect metrics and generate comparison report
- [ ] Add variation info to predictions/artifacts

#### 7.5 Legacy Integration
- [ ] Support variations via multiple `train_x` paths:
  ```yaml
  train_x:
    - path: data/X_raw.csv
      variation_name: "raw"
    - path: data/X_snv.csv
      variation_name: "snv"
  variation_mode: separate
  ```

---

### Phase 8: Advanced Features (Weeks 21-24)

**Goal**: Implement remaining specification features and polish.

#### 8.1 Sample Aggregation Enhancements
- [ ] Aggregation column detection from config
- [ ] Aggregation during loading (not just prediction)
- [ ] Custom aggregation functions

#### 8.2 Auto-Detection Improvements
- [ ] Improve delimiter detection (currently disabled)
- [ ] Improve decimal separator detection
- [ ] Header detection heuristics
- [ ] Signal type auto-detection from header patterns

#### 8.3 Configuration Serialization
- [ ] Serialize loaded configs back to YAML/JSON
- [ ] Normalize configs before serialization
- [ ] Support config diffing for reproducibility

#### 8.4 Error Handling & Diagnostics
- [ ] Comprehensive error codes
- [ ] Config validation CLI command
- [ ] Suggestion system for common mistakes
- [ ] Verbose mode with loading diagnostics

#### 8.5 Performance Optimization
- [ ] Lazy loading for large datasets
- [ ] Memory-mapped file support
- [ ] Parallel file loading
- [ ] Caching improvements

#### 8.6 Documentation & Examples
- [ ] Complete API documentation
- [ ] Migration guide from legacy format
- [ ] Example configs for each use case
- [ ] Troubleshooting guide

---

## Dependency Graph

```
Phase 1 (Foundation)
    │
    ├──► Phase 2 (File Loaders)
    │        │
    │        └──► Phase 3 (Column/Row Selection)
    │                 │
    │                 └──► Phase 4 (Partitions)
    │                          │
    │                          ├──► Phase 5 (Folds)
    │                          │
    │                          └──► Phase 6 (Multi-Source)
    │                                   │
    │                                   └──► Phase 7 (Variations)
    │
    └──► Phase 8 (Advanced) ◄── All phases
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
