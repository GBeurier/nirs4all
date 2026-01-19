# Workflow Operators v2 - Implementation Roadmap

This document provides a detailed implementation roadmap for the workflow operators redesign described in `workflows_operator_design_v2.md`.

**Estimated total effort:** 8-12 weeks (depending on team size)

---

## Overview

### Phase Summary

| Phase | Name | Focus | Estimated Effort |
|-------|------|-------|------------------|
| 1 | Foundation | Indexer tag infrastructure | 1-2 weeks |
| 2 | Tag Controller | `tag` keyword implementation | 1 week |
| 3 | Exclude Controller | `exclude` keyword implementation | 1 week |
| 4 | Unified Branch Controller | Branch refactoring + value mapping | 2-3 weeks |
| 5 | Merge Enhancements | Unified merge strategies | 1 week |
| 6 | Test Migration | Update/remove old tests, create new tests | 1-2 weeks |
| 7 | Examples Migration | Update all examples | 1 week |
| 8 | Documentation | Sphinx docs, migration guide, CLAUDE.md | 1 week |

---

## Phase 1: Foundation - Indexer Tag Infrastructure

**Goal:** Enable storing arbitrary tags as dynamic columns in the indexer.

### 1.1 Core Implementation

#### Task 1.1.1: Extend IndexStore with Tag Column Support
**File:** `nirs4all/data/_indexer/index_store.py`

**Changes:**
- Add `_tag_columns: Dict[str, pl.DataType]` attribute to track dynamic tag columns
- Implement `add_tag_column(name: str, dtype: pl.DataType)` method
- Implement `set_tags(indices: List[int], tag_name: str, values: Any)` method
- Implement `get_tags(tag_name: str, condition: Optional[pl.Expr]) -> List[Any]` method
- Implement `get_tag_column_names() -> List[str]` method
- Implement `remove_tag_column(name: str)` method
- Handle tag column serialization in `to_dict()` / `from_dict()`

**Acceptance Criteria:**
- [ ] Can create boolean, string, and numeric tag columns
- [ ] Can set/get tag values for specific sample indices
- [ ] Tag columns persist through serialization/deserialization
- [ ] Duplicate column names are handled gracefully (error or overwrite with warning)

#### Task 1.1.2: Add Tag Filtering to QueryBuilder
**File:** `nirs4all/data/_indexer/query_builder.py`

**Changes:**
- Add `with_tag_filter(tag_name: str, condition: Any)` method
- Support various condition types:
  - Boolean: `True`, `False`
  - Comparison strings: `"> 0.8"`, `"<= 50"`
  - Range strings: `"0..50"`, `"50.."`
  - List of values: `["a", "b", "c"]`
  - Lambda functions: `lambda x: x > 0.8`
- Implement `_parse_tag_condition(condition: Any) -> pl.Expr` helper

**Acceptance Criteria:**
- [ ] Can filter samples by tag values using all supported formats
- [ ] String conditions are parsed correctly
- [ ] Range conditions handle open-ended ranges
- [ ] Lambda conditions are applied correctly

#### Task 1.1.3: Expose Tag Operations on SpectroDataset
**File:** `nirs4all/data/dataset.py`

**Changes:**
- Add `add_tag(name: str, dtype: str = "bool")` method
- Add `set_tag(name: str, indices: List[int], values: Any)` method
- Add `get_tag(name: str, selector: Optional[Dict] = None) -> np.ndarray` method
- Add `tags` property returning list of tag column names
- Add `tag_info() -> Dict[str, Dict]` returning tag metadata

**Acceptance Criteria:**
- [ ] Dataset exposes tag operations through clean API
- [ ] Tags can be queried by selector (partition, branch, etc.)

### 1.2 Tests

#### Task 1.2.1: Unit Tests for Tag Columns
**File:** `tests/unit/data/indexer/test_tag_columns.py` (NEW)

**Test Cases:**
- `test_add_boolean_tag_column`
- `test_add_string_tag_column`
- `test_add_numeric_tag_column`
- `test_set_and_get_tags`
- `test_tag_column_serialization`
- `test_tag_column_deserialization`
- `test_duplicate_tag_column_error`
- `test_remove_tag_column`
- `test_get_tag_column_names`

#### Task 1.2.2: Unit Tests for Tag Filtering
**File:** `tests/unit/data/indexer/test_tag_filtering.py` (NEW)

**Test Cases:**
- `test_filter_by_boolean_tag`
- `test_filter_by_string_comparison`
- `test_filter_by_range`
- `test_filter_by_value_list`
- `test_filter_by_lambda`
- `test_filter_combined_conditions`
- `test_filter_open_ended_range`
- `test_invalid_condition_format_error`

#### Task 1.2.3: Unit Tests for Dataset Tag API
**File:** `tests/unit/data/test_dataset_tags.py` (NEW)

**Test Cases:**
- `test_dataset_add_tag`
- `test_dataset_set_tag`
- `test_dataset_get_tag`
- `test_dataset_tags_property`
- `test_dataset_tag_info`
- `test_dataset_tag_with_selector`

### 1.3 Deliverables Checklist

- [ ] IndexStore tag column support implemented
- [ ] QueryBuilder tag filtering implemented
- [ ] SpectroDataset tag API implemented
- [ ] 20+ unit tests passing
- [ ] Code reviewed

---

## Phase 2: Tag Controller

**Goal:** Implement the `tag` keyword for general sample tagging.

### 2.1 Core Implementation

#### Task 2.1.1: Create TagController
**File:** `nirs4all/controllers/data/tag.py` (NEW)

**Implementation:**
```python
@register_controller
class TagController(OperatorController):
    priority = 5

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return keyword == "tag"

    @classmethod
    def use_multi_source(cls) -> bool:
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True  # Tags computed fresh on prediction data

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        # 1. Parse tag configuration (single filter, list, or dict)
        # 2. For each tagger:
        #    a. Fit on training data
        #    b. Apply to get mask/values
        #    c. Store as tag column in indexer
        # 3. Return updated context
```

**Supported Syntax:**
- Single: `{"tag": Filter()}`
- Multiple: `{"tag": [Filter1(), Filter2()]}`
- Named: `{"tag": {"name1": Filter1(), "name2": Filter2()}}`

**Acceptance Criteria:**
- [ ] Handles single, list, and dict syntax
- [ ] Tags stored correctly in indexer
- [ ] Tags computed during prediction mode
- [ ] Artifacts saved for reproducibility

#### Task 2.1.2: Add tag_name Parameter to SampleFilter Base Class
**File:** `nirs4all/operators/filters/base.py`

**Changes:**
- Add `tag_name: Optional[str] = None` parameter to `__init__`
- Default tag name to class name if not provided
- Update all subclasses to pass through `tag_name`

**Files to update:**
- `nirs4all/operators/filters/y_outlier.py`
- `nirs4all/operators/filters/x_outlier.py`
- `nirs4all/operators/filters/spectral_quality.py`
- `nirs4all/operators/filters/high_leverage.py`
- `nirs4all/operators/filters/metadata.py`

#### Task 2.1.3: Add tag_filters to DataSelector
**File:** `nirs4all/pipeline/config/context.py`

**Changes:**
- Add `tag_filters: Dict[str, Any] = field(default_factory=dict)` to DataSelector
- Implement `with_tag_filter(tag_name: str, condition: Any) -> DataSelector` method
- Update `_build_query()` to include tag filters

### 2.2 Tests

#### Task 2.2.1: Unit Tests for TagController
**File:** `tests/unit/controllers/test_tag_controller.py` (NEW)

**Test Cases:**
- `test_tag_controller_matches_keyword`
- `test_tag_controller_single_filter`
- `test_tag_controller_multiple_filters`
- `test_tag_controller_named_filters`
- `test_tag_controller_prediction_mode`
- `test_tag_controller_saves_artifacts`
- `test_tag_controller_multi_source`

#### Task 2.2.2: Integration Tests for Tag Workflow
**File:** `tests/integration/pipeline/test_tag_workflow.py` (NEW)

**Test Cases:**
- `test_tag_then_branch_by_tag`
- `test_tag_persists_through_pipeline`
- `test_tag_prediction_computes_fresh`
- `test_tag_with_cross_validation`
- `test_multiple_tags_combined`

### 2.3 Deliverables Checklist

- [ ] TagController implemented and registered
- [ ] SampleFilter base class updated with tag_name
- [ ] All filter subclasses updated
- [ ] DataSelector tag_filters implemented
- [ ] 10+ unit tests passing
- [ ] 5+ integration tests passing
- [ ] Code reviewed

---

## Phase 3: Exclude Controller

**Goal:** Implement the `exclude` keyword for sample exclusion.

### 3.1 Core Implementation

#### Task 3.1.1: Create ExcludeController
**File:** `nirs4all/controllers/data/exclude.py` (NEW)

**Implementation:**
```python
@register_controller
class ExcludeController(OperatorController):
    priority = 5

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return keyword == "exclude"

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return False  # Never exclude during prediction

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        # 1. Parse configuration
        # 2. Extract filters and options (mode, remove)
        # 3. Fit filters on training data
        # 4. Get exclusion mask
        # 5. If remove=True (default), mark samples as excluded
        # 6. Always store tags for analysis
```

**Supported Syntax:**
- Basic: `{"exclude": Filter()}`
- Multiple: `{"exclude": [Filter1(), Filter2()], "mode": "any"|"all"}`
- Tag-only: `{"exclude": Filter(), "remove": False}`

#### Task 3.1.2: Delete SampleFilterController
**File:** `nirs4all/controllers/data/sample_filter.py` (DELETE)

**Actions:**
- Remove file completely
- Remove from `__init__.py` exports
- Remove any imports in other files

### 3.2 Tests

#### Task 3.2.1: Unit Tests for ExcludeController
**File:** `tests/unit/controllers/test_exclude_controller.py` (NEW)

**Test Cases:**
- `test_exclude_controller_matches_keyword`
- `test_exclude_single_filter`
- `test_exclude_multiple_filters_mode_any`
- `test_exclude_multiple_filters_mode_all`
- `test_exclude_tag_only_mode`
- `test_exclude_not_applied_in_prediction`
- `test_exclude_cascades_to_augmented`

#### Task 3.2.2: Migration Tests
**File:** `tests/integration/pipeline/test_exclude_migration.py` (NEW)

**Test Cases:**
- `test_exclude_replaces_sample_filter_basic`
- `test_exclude_replaces_sample_filter_multiple`
- `test_exclude_produces_same_results_as_old`

### 3.3 Tests to Remove/Update

#### Files to DELETE:
- `tests/unit/data/indexer/test_sample_filtering.py` (if only tests old keyword)

#### Files to UPDATE:
- `tests/integration/data/test_sample_filtering_integration.py` -> rename to `test_exclude_integration.py`
  - Update all `sample_filter` references to `exclude`
  - Update test names

### 3.4 Deliverables Checklist

- [ ] ExcludeController implemented and registered
- [ ] SampleFilterController deleted
- [ ] Old sample_filter tests removed/updated
- [ ] 10+ unit tests for ExcludeController passing
- [ ] Migration tests passing
- [ ] Code reviewed

---

## Phase 4: Unified Branch Controller

**Goal:** Refactor BranchController to handle all branch types with user-friendly value mapping.

### 4.1 Core Implementation

#### Task 4.1.1: Implement Value Mapping Parser
**File:** `nirs4all/controllers/data/branch_utils.py` (NEW)

**Implementation:**
```python
def parse_value_condition(condition: Any) -> Callable[[Any], bool]:
    """Parse user-friendly value conditions into callable predicates."""
    if callable(condition):
        return condition
    if isinstance(condition, bool):
        return lambda x: x == condition
    if isinstance(condition, list):
        return lambda x: x in condition
    if isinstance(condition, str):
        return _parse_string_condition(condition)
    raise ValueError(f"Unknown condition format: {condition}")

def _parse_string_condition(s: str) -> Callable[[Any], bool]:
    """Parse string conditions like '> 0.8', '0..50', etc."""
    # Handle comparison operators: >, >=, <, <=, ==, !=
    # Handle range syntax: 0..50, 50.., ..50
    pass
```

#### Task 4.1.2: Refactor BranchController for Mode Detection
**File:** `nirs4all/controllers/data/branch.py` (MAJOR REFACTOR)

**Changes:**
- Add `_detect_branch_mode(branch_def)` method returning "duplication" or "separation"
- Add `_execute_duplication_branch()` method (existing logic)
- Add `_execute_separation_branch()` method (new logic)
- Add handlers for each separation type:
  - `_execute_by_tag(tag_name, values, steps)`
  - `_execute_by_metadata(column, values, steps)`
  - `_execute_by_filter(filter_op, steps)`
  - `_execute_by_source(steps)`

**Separation Branch Logic:**
1. Get unique values from tag/metadata/filter
2. If `values` provided, group by value mapping
3. Create branch context for each group with filtered samples
4. Execute steps (shared or per-branch)
5. Store branch contexts with sample indices

#### Task 4.1.3: Absorb SourceBranchController
**Files:**
- `nirs4all/controllers/data/source_branch.py` (DELETE)
- `nirs4all/controllers/data/branch.py` (UPDATE)

**Actions:**
- Move source branch logic into `_execute_by_source()` in BranchController
- Delete SourceBranchController file
- Update imports

#### Task 4.1.4: Delete Legacy Branch Controllers
**Files to DELETE:**
- `nirs4all/controllers/data/outlier_excluder.py`
- `nirs4all/controllers/data/sample_partitioner.py` (if exists)
- `nirs4all/controllers/data/metadata_partitioner.py` (if exists)

### 4.2 Tests

#### Task 4.2.1: Unit Tests for Value Mapping
**File:** `tests/unit/controllers/data/test_branch_value_mapping.py` (NEW)

**Test Cases:**
- `test_parse_boolean_condition`
- `test_parse_list_condition`
- `test_parse_comparison_greater_than`
- `test_parse_comparison_less_equal`
- `test_parse_range_closed`
- `test_parse_range_open_start`
- `test_parse_range_open_end`
- `test_parse_lambda_condition`
- `test_invalid_condition_raises_error`

#### Task 4.2.2: Unit Tests for Separation Branches
**File:** `tests/unit/controllers/data/test_branch_separation.py` (NEW)

**Test Cases:**
- `test_branch_by_tag_auto_values`
- `test_branch_by_tag_explicit_values`
- `test_branch_by_metadata`
- `test_branch_by_filter`
- `test_branch_by_source`
- `test_separation_branch_shared_steps`
- `test_separation_branch_per_branch_steps`

#### Task 4.2.3: Integration Tests for New Branch Modes
**File:** `tests/integration/pipeline/test_new_branch_modes.py` (NEW)

**Test Cases:**
- `test_tag_then_branch_by_tag_full_pipeline`
- `test_branch_by_metadata_with_model`
- `test_branch_by_filter_clustering`
- `test_branch_by_source_unified_syntax`
- `test_separation_branch_with_merge_concat`
- `test_value_mapping_string_conditions`
- `test_value_mapping_range_conditions`

### 4.3 Tests to Remove/Update

#### Files to DELETE:
- `tests/unit/controllers/test_outlier_excluder.py`
- `tests/unit/controllers/data/test_source_branch.py`
- `tests/integration/pipeline/test_source_branch.py` (merge into test_new_branch_modes.py)
- `tests/integration/pipeline/test_branch_outlier.py` (update to use new syntax)

#### Files to UPDATE:
- `tests/unit/controllers/test_branch_controller.py`
  - Add tests for mode detection
  - Update any old syntax tests
- `tests/integration/pipeline/test_stacking_branching.py`
  - Verify works with new branch syntax
- `tests/integration/pipeline/test_branch_nested.py`
  - Verify nested branches work with new modes

### 4.4 Deliverables Checklist

- [ ] Value mapping parser implemented
- [ ] BranchController refactored with mode detection
- [ ] by_tag separation mode working
- [ ] by_metadata separation mode working
- [ ] by_filter separation mode working
- [ ] by_source mode absorbed from SourceBranchController
- [ ] Legacy controllers deleted
- [ ] 15+ unit tests passing
- [ ] 10+ integration tests passing
- [ ] Old tests removed/updated
- [ ] Code reviewed

---

## Phase 5: Merge Enhancements

**Goal:** Unify merge strategies and add `concat` for separation branches.

### 5.1 Core Implementation

#### Task 5.1.1: Add concat Strategy to MergeController
**File:** `nirs4all/controllers/data/merge.py`

**Changes:**
- Add `_execute_concat_merge()` method for reassembling separated samples
- Auto-detect branch type (duplication vs separation) from context
- Validate merge strategy matches branch type (warn or error)
- Consolidate `merge_sources` into main merge with `{"sources": "concat"}` syntax

**concat Merge Logic:**
1. Get branch contexts with their sample indices
2. For each branch, collect predictions/features for its samples
3. Reassemble into original sample order
4. Handle samples that appear in no branch (error) or multiple branches (error)

#### Task 5.1.2: Consolidate merge_sources
**File:** `nirs4all/controllers/data/merge.py`

**Changes:**
- Support `{"merge": {"sources": "concat"|"stack"|"dict"}}` syntax
- Deprecate standalone `merge_sources` keyword (or remove directly)

### 5.2 Tests

#### Task 5.2.1: Unit Tests for concat Merge
**File:** `tests/unit/controllers/data/test_merge_concat.py` (NEW)

**Test Cases:**
- `test_merge_concat_reassembles_samples`
- `test_merge_concat_preserves_order`
- `test_merge_concat_with_predictions`
- `test_merge_concat_with_features`
- `test_merge_concat_missing_samples_error`
- `test_merge_concat_duplicate_samples_error`

#### Task 5.2.2: Integration Tests for Merge Strategies
**File:** `tests/integration/pipeline/test_merge_strategies.py` (NEW or UPDATE)

**Test Cases:**
- `test_separation_branch_with_concat_merge`
- `test_duplication_branch_with_features_merge`
- `test_duplication_branch_with_predictions_merge`
- `test_merge_auto_detects_branch_type`
- `test_merge_warns_on_strategy_mismatch`
- `test_merge_sources_unified_syntax`

### 5.3 Tests to Update

- `tests/integration/pipeline/test_merge_per_branch.py` - verify still works
- Any tests using `merge_sources` keyword - update to new syntax

### 5.4 Deliverables Checklist

- [ ] concat merge strategy implemented
- [ ] Branch type auto-detection working
- [ ] merge_sources consolidated
- [ ] 10+ unit tests passing
- [ ] 5+ integration tests passing
- [ ] Code reviewed

---

## Phase 6: Test Migration

**Goal:** Systematically update all tests to use new syntax.

### 6.1 Tests to DELETE (Old Keywords)

| File | Reason |
|------|--------|
| `tests/unit/controllers/test_outlier_excluder.py` | Controller deleted |
| `tests/unit/controllers/data/test_source_branch.py` | Controller absorbed |
| `tests/integration/pipeline/test_source_branch.py` | Replaced by test_new_branch_modes |
| `tests/unit/data/indexer/test_sample_filtering.py` | If only tests old keyword |

### 6.2 Tests to UPDATE (Syntax Changes)

| File | Changes Needed |
|------|----------------|
| `tests/integration/data/test_sample_filtering_integration.py` | `sample_filter` -> `exclude` |
| `tests/integration/pipeline/test_branch_outlier.py` | `branch: {by: "outlier_excluder"}` -> `branch: [...]` with `exclude` |
| `tests/integration/pipeline/test_stacking_branching.py` | Verify new syntax works |
| `tests/integration/pipeline/test_branch_nested.py` | Test with new modes |
| `tests/integration/pipeline/test_multisource_branching_stacking.py` | `source_branch` -> `branch: {by_source: ...}` |

### 6.3 Tests to CREATE (New Features)

| File | Purpose |
|------|---------|
| `tests/unit/data/indexer/test_tag_columns.py` | Tag column CRUD |
| `tests/unit/data/indexer/test_tag_filtering.py` | Tag condition parsing |
| `tests/unit/data/test_dataset_tags.py` | Dataset tag API |
| `tests/unit/controllers/test_tag_controller.py` | TagController |
| `tests/unit/controllers/test_exclude_controller.py` | ExcludeController |
| `tests/unit/controllers/data/test_branch_value_mapping.py` | Value condition parser |
| `tests/unit/controllers/data/test_branch_separation.py` | Separation modes |
| `tests/unit/controllers/data/test_merge_concat.py` | concat strategy |
| `tests/integration/pipeline/test_tag_workflow.py` | Tag end-to-end |
| `tests/integration/pipeline/test_exclude_migration.py` | Exclude migration |
| `tests/integration/pipeline/test_new_branch_modes.py` | All new branch modes |
| `tests/integration/pipeline/test_merge_strategies.py` | All merge strategies |

### 6.4 Test Verification Checklist

Run full test suite after each change:
```bash
pytest tests/ -v --tb=short
```

Ensure coverage doesn't regress:
```bash
pytest tests/ --cov=nirs4all --cov-report=term-missing
```

### 6.5 Deliverables Checklist

- [ ] All deprecated test files deleted
- [ ] All syntax-change tests updated
- [ ] All new feature tests created
- [ ] Full test suite passing
- [ ] Coverage maintained or improved

---

## Phase 7: Examples Migration

**Goal:** Update all examples to use new syntax.

### 7.1 Examples to UPDATE

#### User Examples (Priority: HIGH)

| File | Changes Needed |
|------|----------------|
| `examples/user/05_cross_validation/U03_sample_filtering.py` | `sample_filter` -> `exclude`, update explanations |

#### Developer Examples (Priority: HIGH)

| File | Changes Needed |
|------|----------------|
| `examples/developer/01_advanced_pipelines/D01_branching_basics.py` | Add separation branch examples |
| `examples/developer/01_advanced_pipelines/D02_branching_advanced.py` | Update to new syntax |
| `examples/developer/01_advanced_pipelines/D03_merge_basics.py` | Add concat merge, update syntax |
| `examples/developer/01_advanced_pipelines/D04_merge_sources.py` | Update to unified merge syntax |
| `examples/developer/05_advanced_features/D01_metadata_branching.py` | `branch: {by: "metadata_partitioner"}` -> `branch: {by_metadata: ...}` |

#### Reference Examples (Priority: HIGH)

| File | Changes Needed |
|------|----------------|
| `examples/reference/R01_pipeline_syntax.py` | Update keyword reference |
| `examples/reference/R03_all_keywords.py` | Update all keywords |

#### Legacy Examples (Priority: LOW - Consider Deletion)

| File | Action |
|------|--------|
| `examples/legacy/Q28_sample_filtering.py` | DELETE or update |
| `examples/legacy/Q30_branching.py` | DELETE or update |
| `examples/legacy/Q31_outlier_branching.py` | DELETE or update |
| `examples/legacy/Q35_metadata_branching.py` | DELETE or update |
| `examples/legacy/Q_merge_branches.py` | DELETE or update |
| `examples/legacy/Q_merge_sources.py` | DELETE or update |

### 7.2 Examples to CREATE

| File | Purpose |
|------|---------|
| `examples/user/05_cross_validation/U05_tagging_analysis.py` | Show tag keyword for analysis |
| `examples/user/05_cross_validation/U06_exclusion_strategies.py` | Compare different exclusion methods |
| `examples/developer/01_advanced_pipelines/D06_separation_branches.py` | All separation branch modes |
| `examples/developer/01_advanced_pipelines/D07_value_mapping.py` | User-friendly value mapping syntax |

### 7.3 Example Verification

Run all examples to verify they work:
```bash
cd examples && ./run.sh -q
```

### 7.4 Deliverables Checklist

- [ ] All user examples updated
- [ ] All developer examples updated
- [ ] All reference examples updated
- [ ] Legacy examples deleted or updated
- [ ] New examples created
- [ ] All examples run successfully

---

## Phase 8: Documentation

**Goal:** Update all documentation to reflect new design.

### 8.1 CLAUDE.md Updates

**File:** `/home/delete/nirs4all/CLAUDE.md`

**Changes:**
- Update "Special Keywords" table
- Update pipeline syntax examples
- Add `tag` and `exclude` to keyword reference
- Update branch syntax documentation
- Update merge syntax documentation

### 8.2 Sphinx Documentation Updates

**Files in `docs/source/`:**

| File | Changes |
|------|---------|
| `api/pipeline.rst` | Update keyword documentation |
| `tutorials/filtering.rst` | Rewrite for `exclude` keyword |
| `tutorials/branching.rst` | Add separation branches |
| `tutorials/advanced.rst` | Update merge documentation |
| `reference/keywords.rst` | Complete keyword reference |

### 8.3 Migration Guide

**File:** `docs/_internal/migration_guide_v2.md` (NEW)

**Contents:**
1. **Overview of Changes**
   - Keywords renamed/removed
   - New keywords added
   - Syntax changes

2. **Quick Reference Table**
   | Old Syntax | New Syntax |
   |------------|------------|
   | `{"sample_filter": {...}}` | `{"exclude": ...}` |
   | `{"branch": {"by": "outlier_excluder", ...}}` | `{"branch": [...]}` with `{"exclude": ...}` |
   | `{"branch": {"by": "metadata_partitioner", "column": "x"}}` | `{"branch": {"by_metadata": "x"}}` |
   | `{"source_branch": {...}}` | `{"branch": {"by_source": True, "steps": {...}}}` |
   | `{"merge_sources": "concat"}` | `{"merge": {"sources": "concat"}}` |

3. **Detailed Migration Examples**
   - Before/after for each common pattern

4. **Breaking Changes**
   - No deprecation period
   - Old keywords will error immediately

### 8.4 Internal Documentation Updates

**Files:**
- `docs/_internal/workflows_operator_design_v2.md` - Already done
- `docs/_internal/filters_outliers_overview.md` - Update if exists

### 8.5 Deliverables Checklist

- [ ] CLAUDE.md updated
- [ ] Sphinx docs updated
- [ ] Migration guide created
- [ ] Internal docs updated
- [ ] Documentation builds without errors
- [ ] All code examples in docs tested

---

## Appendix A: File Change Summary

### Files to CREATE

| File | Phase |
|------|-------|
| `nirs4all/controllers/data/tag.py` | 2 |
| `nirs4all/controllers/data/exclude.py` | 3 |
| `nirs4all/controllers/data/branch_utils.py` | 4 |
| `tests/unit/data/indexer/test_tag_columns.py` | 1 |
| `tests/unit/data/indexer/test_tag_filtering.py` | 1 |
| `tests/unit/data/test_dataset_tags.py` | 1 |
| `tests/unit/controllers/test_tag_controller.py` | 2 |
| `tests/unit/controllers/test_exclude_controller.py` | 3 |
| `tests/unit/controllers/data/test_branch_value_mapping.py` | 4 |
| `tests/unit/controllers/data/test_branch_separation.py` | 4 |
| `tests/unit/controllers/data/test_merge_concat.py` | 5 |
| `tests/integration/pipeline/test_tag_workflow.py` | 2 |
| `tests/integration/pipeline/test_exclude_migration.py` | 3 |
| `tests/integration/pipeline/test_new_branch_modes.py` | 4 |
| `tests/integration/pipeline/test_merge_strategies.py` | 5 |
| `examples/user/05_cross_validation/U05_tagging_analysis.py` | 7 |
| `examples/user/05_cross_validation/U06_exclusion_strategies.py` | 7 |
| `examples/developer/01_advanced_pipelines/D06_separation_branches.py` | 7 |
| `examples/developer/01_advanced_pipelines/D07_value_mapping.py` | 7 |
| `docs/_internal/migration_guide_v2.md` | 8 |

### Files to DELETE

| File | Phase |
|------|-------|
| `nirs4all/controllers/data/sample_filter.py` | 3 |
| `nirs4all/controllers/data/source_branch.py` | 4 |
| `nirs4all/controllers/data/outlier_excluder.py` | 4 |
| `tests/unit/controllers/test_outlier_excluder.py` | 6 |
| `tests/unit/controllers/data/test_source_branch.py` | 6 |
| `tests/integration/pipeline/test_source_branch.py` | 6 |
| `examples/legacy/Q28_sample_filtering.py` | 7 |
| `examples/legacy/Q31_outlier_branching.py` | 7 |
| `examples/legacy/Q35_metadata_branching.py` | 7 |

### Files to MODIFY

| File | Phase | Changes |
|------|-------|---------|
| `nirs4all/data/_indexer/index_store.py` | 1 | Tag columns |
| `nirs4all/data/_indexer/query_builder.py` | 1 | Tag filtering |
| `nirs4all/data/dataset.py` | 1 | Tag API |
| `nirs4all/operators/filters/base.py` | 2 | tag_name param |
| `nirs4all/pipeline/config/context.py` | 2 | tag_filters |
| `nirs4all/controllers/data/branch.py` | 4 | Major refactor |
| `nirs4all/controllers/data/merge.py` | 5 | concat strategy |
| `CLAUDE.md` | 8 | Keywords update |

---

## Appendix B: Risk Mitigation

### Risk 1: Breaking User Pipelines
**Mitigation:**
- Clear migration guide with before/after examples
- Error messages that suggest new syntax
- Announce breaking changes prominently in release notes

### Risk 2: Performance Regression
**Mitigation:**
- Benchmark tag column operations before/after
- Profile separation branch memory usage
- Add performance tests for large datasets

### Risk 3: Incomplete Test Coverage
**Mitigation:**
- Track coverage per phase
- Require 80%+ coverage for new code
- Review edge cases explicitly

### Risk 4: Documentation Lag
**Mitigation:**
- Update docs in same PR as code changes
- Require docs review for API changes
- Automated doc build in CI

---

*Roadmap version: 1.0*
*Created: 2026-01-19*
*Associated design: workflows_operator_design_v2.md*
