# Test Suite Reorganization Plan for nirs4all

**Date**: November 18, 2025
**Branch**: 0.4.1
**Status**: PLANNING PHASE - DO NOT IMPLEMENT YET

## Executive Summary

The test suite needs comprehensive reorganization to mirror the source code structure in `nirs4all/`. This document provides a detailed migration plan to create a clean, maintainable, and well-organized test hierarchy.

## Current State Analysis

### Current Test Structure Problems

1. **Inconsistent organization**: Tests are split across `unit/`, `integration/`, and `integration_tests/` with unclear boundaries
2. **Missing module coverage**: Many source modules lack corresponding test folders
3. **Misplaced tests**: Some tests are in wrong directories (e.g., `pipeline/test_context.py` should be in `pipeline/config/`)
4. **Duplicate integration folders**: Both `integration/` and `integration_tests/` exist with overlapping purposes
5. **Poor naming**: Some test files have unclear names (e.g., header_units tests scattered everywhere)
6. **Flat structure**: Unit tests don't mirror the nested structure of source code

### Current Test Directory Structure

```
tests/
├── conftest.py
├── README.md
├── run_tests.py
├── run_runner_tests.py
├── fixtures/
│   ├── __init__.py
│   └── README.md
├── unit/
│   ├── controllers/
│   │   ├── test_indexer_augmentation.py
│   │   ├── test_sample_augmentation.py (actually named test_transformer_augmentation.py)
│   │   └── test_split_augmentation.py
│   ├── data/
│   │   ├── targets/
│   │   │   ├── test_converters.py
│   │   │   ├── test_encoders.py
│   │   │   ├── test_processing_chain.py
│   │   │   ├── test_targets_refactored.py
│   │   │   └── test_transformers.py
│   │   ├── test_config.py
│   │   ├── test_csv_loader_header_units.py
│   │   ├── test_dataset.py
│   │   ├── test_dataset_save_load_header_units.py
│   │   ├── test_dataset_wavelength_conversion.py
│   │   ├── test_feature_components.py
│   │   ├── test_feature_source_header_units.py
│   │   ├── test_group_split_validation.py
│   │   ├── test_indexer_header_units.py
│   │   ├── test_loaders.py
│   │   ├── test_metadata.py
│   │   ├── test_metadata_loading.py
│   │   ├── test_predictions.py
│   │   ├── test_predictions_header_units.py
│   │   ├── test_preprocessing_header_units.py
│   │   └── test_resampler_header_units.py
│   ├── models/
│   │   ├── test_pytorch.py
│   │   └── test_tensorflow.py
│   ├── pipeline/
│   │   ├── test_binary_loader.py
│   │   ├── test_config.py
│   │   ├── test_generator.py
│   │   ├── test_manifest_manager.py
│   │   ├── test_runner.py
│   │   ├── test_runner_comprehensive.py
│   │   ├── test_runner_normalization.py
│   │   ├── test_runner_predict.py
│   │   ├── test_runner_regression_prevention.py
│   │   ├── test_runner_state.py
│   │   ├── test_serialization.py
│   │   └── test_serializer.py
│   ├── transforms/
│   │   ├── test_augmentation.py
│   │   ├── test_nirs.py
│   │   ├── test_signal.py
│   │   └── test_splitters.py
│   └── utils/
│       ├── test_balancing.py
│       ├── test_balancing_value_aware.py
│       ├── test_binning.py
│       └── test_data_generator.py
├── data/
│   ├── predictions_components/
│   │   └── test_array_registry.py
│   ├── test_array_registry_integration.py
│   └── test_dataset_with_context.py
├── pipeline/
│   └── test_context.py
├── integration/
│   ├── test_augmentation_end_to_end.py
│   ├── test_augmentation_integration.py
│   └── test_dataset_augmentation.py
├── integration_tests/
│   ├── INTEGRATION_TEST_SUMMARY.md
│   ├── QUICK_REFERENCE.txt
│   ├── README.md
│   ├── run_integration_tests.py
│   ├── test_basic_pipeline.py
│   ├── test_classification_integration.py
│   ├── test_comprehensive_integration.py
│   ├── test_finetune_integration.py
│   ├── test_flexible_inputs_integration.py
│   ├── test_groupsplit_integration.py
│   ├── test_multisource_integration.py
│   ├── test_pca_analysis_integration.py
│   ├── test_prediction_reuse_integration.py
│   ├── test_resampler_integration.py
│   ├── test_sample_augmentation_integration.py
│   └── test_shap_integration.py
├── workspace/
│   ├── test_catalog_export.py
│   ├── test_library_manager.py
│   └── test_query_reporting.py
├── dataset/  (empty)
├── pipeline_runner/  (empty)
├── serialization/  (empty)
└── utils/  (empty)
```

### Source Code Structure (Target)

```
nirs4all/
├── cli/
│   ├── commands/
│   ├── installation_test.py
│   └── main.py
├── controllers/
│   ├── base.py
│   ├── chart/
│   ├── charts/
│   ├── controller.py
│   ├── data/
│   ├── dataset/
│   ├── flow/
│   ├── log/
│   ├── models/
│   ├── presets.py
│   ├── registry.py
│   ├── sklearn/
│   ├── splitters/
│   ├── tensorflow/
│   └── transforms/
├── core/
│   ├── config.py
│   ├── exceptions.py
│   ├── metrics.py
│   ├── task_detection.py
│   ├── task_type.py
│   └── validation.py
├── data/
│   ├── loaders/
│   │   ├── csv_loader.py
│   │   └── loader.py
│   ├── _dataset/
│   │   ├── feature_accessor.py
│   │   ├── metadata_accessor.py
│   │   └── target_accessor.py
│   ├── _features/
│   │   ├── array_storage.py
│   │   ├── augmentation_handler.py
│   │   ├── feature_constants.py
│   │   ├── feature_source.py
│   │   ├── header_manager.py
│   │   ├── layout_transformer.py
│   │   ├── processing_manager.py
│   │   └── update_strategy.py
│   ├── _indexer/
│   │   ├── augmentation_tracker.py
│   │   ├── index_store.py
│   │   ├── parameter_normalizer.py
│   │   ├── processing_manager.py
│   │   ├── query_builder.py
│   │   └── sample_manager.py
│   ├── _predictions/
│   │   ├── aggregator.py
│   │   ├── array_registry.py
│   │   ├── indexer.py
│   │   ├── query.py
│   │   ├── ranker.py
│   │   ├── result.py
│   │   ├── schemas.py
│   │   ├── serializer.py
│   │   └── storage.py
│   ├── _targets/
│   │   ├── converters.py
│   │   ├── encoders.py
│   │   ├── processing_chain.py
│   │   └── transformers.py
│   ├── binning.py
│   ├── config.py
│   ├── config_parser.py
│   ├── dataset.py
│   ├── ensemble_utils.py
│   ├── features.py
│   ├── indexer.py
│   ├── io.py
│   ├── metadata.py
│   ├── predictions.py
│   ├── targets.py
│   └── types.py
├── operators/
│   ├── augmentation/
│   │   ├── abc_augmenter.py
│   │   ├── random.py
│   │   └── splines.py
│   ├── models/
│   │   ├── base.py
│   │   ├── legacy_tf/
│   │   ├── pytorch/
│   │   ├── sklearn/
│   │   └── tensorflow/
│   ├── splitters/
│   │   └── splitters.py
│   └── transforms/
│       ├── features.py
│       ├── nirs.py
│       ├── presets.py
│       ├── resampler.py
│       ├── scalers.py
│       ├── signal.py
│       └── targets.py
├── optimization/
│   └── optuna.py
├── pipeline/
│   ├── config/
│   │   ├── component_serialization.py
│   │   ├── context.py
│   │   ├── generator.py
│   │   └── pipeline_config.py
│   ├── execution/
│   │   ├── builder.py
│   │   ├── executor.py
│   │   ├── orchestrator.py
│   │   └── result.py
│   ├── steps/
│   │   ├── parser.py
│   │   ├── router.py
│   │   └── step_runner.py
│   ├── storage/
│   │   ├── artifacts/
│   │   ├── io.py
│   │   ├── io_exporter.py
│   │   ├── io_resolver.py
│   │   ├── io_writer.py
│   │   ├── library.py
│   │   └── manifest_manager.py
│   ├── explainer.py
│   ├── predictor.py
│   └── runner.py
├── utils/
│   ├── backend.py
│   ├── emoji.py
│   └── spinner.py
├── visualization/
│   ├── analysis/
│   ├── charts/
│   ├── chart_utils/
│   ├── predictions.py
│   └── reports.py
└── workspace/
    └── library_manager.py
```

---

## Proposed New Test Structure

### Target Directory Organization

```
tests/
├── conftest.py                          # Global fixtures and pytest configuration
├── README.md                            # Updated test suite documentation
├── run_tests.py                         # Test runner scripts
├── run_runner_tests.py                  # (Consider consolidating into run_tests.py)
│
├── fixtures/                            # Shared test fixtures and data
│   ├── __init__.py
│   ├── README.md
│   ├── data_generators.py               # FROM: unit/utils/test_data_generator.py (refactor)
│   ├── sample_datasets/                 # Fixture datasets
│   └── sample_pipelines/                # Fixture pipeline configs
│
├── unit/                                # Unit tests (mirrors nirs4all/ structure)
│   ├── __init__.py
│   │
│   ├── cli/                             # NEW: Tests for CLI
│   │   ├── __init__.py
│   │   ├── test_main.py
│   │   └── commands/
│   │       ├── __init__.py
│   │       └── test_*.py
│   │
│   ├── controllers/                     # Tests for controllers
│   │   ├── __init__.py
│   │   ├── test_base.py                 # NEW
│   │   ├── test_controller.py           # NEW
│   │   ├── test_presets.py              # NEW
│   │   ├── test_registry.py             # NEW
│   │   ├── chart/                       # NEW
│   │   │   └── test_*.py
│   │   ├── charts/                      # NEW
│   │   │   └── test_*.py
│   │   ├── data/                        # NEW
│   │   │   └── test_*.py
│   │   ├── dataset/                     # NEW
│   │   │   ├── test_indexer_augmentation.py      # FROM: unit/controllers/
│   │   │   ├── test_sample_augmentation.py       # FROM: unit/controllers/test_transformer_augmentation.py
│   │   │   └── test_split_augmentation.py        # FROM: unit/controllers/
│   │   ├── flow/                        # NEW
│   │   │   └── test_*.py
│   │   ├── log/                         # NEW
│   │   │   └── test_*.py
│   │   ├── models/                      # NEW
│   │   │   └── test_*.py
│   │   ├── sklearn/                     # NEW
│   │   │   └── test_*.py
│   │   ├── splitters/                   # NEW
│   │   │   └── test_*.py
│   │   └── transforms/                  # NEW
│   │       └── test_*.py
│   │
│   ├── core/                            # NEW: Tests for core utilities
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_exceptions.py
│   │   ├── test_metrics.py
│   │   ├── test_task_detection.py
│   │   ├── test_task_type.py
│   │   └── test_validation.py
│   │
│   ├── data/                            # Tests for data module
│   │   ├── __init__.py
│   │   ├── loaders/                     # NEW: Organized loader tests
│   │   │   ├── __init__.py
│   │   │   ├── test_csv_loader.py       # FROM: test_csv_loader_header_units.py + test_loaders.py
│   │   │   └── test_loader.py           # FROM: test_loaders.py
│   │   ├── dataset/                     # NEW: Dataset accessor tests
│   │   │   ├── __init__.py
│   │   │   ├── test_feature_accessor.py
│   │   │   ├── test_metadata_accessor.py
│   │   │   └── test_target_accessor.py
│   │   ├── features/                    # NEW: Feature module tests
│   │   │   ├── __init__.py
│   │   │   ├── test_array_storage.py           # FROM: test_feature_components.py (split)
│   │   │   ├── test_augmentation_handler.py    # NEW or FROM test_preprocessing_header_units.py
│   │   │   ├── test_feature_constants.py       # FROM: test_feature_components.py (split)
│   │   │   ├── test_feature_source.py          # FROM: test_feature_source_header_units.py
│   │   │   ├── test_header_manager.py          # FROM: test_feature_components.py (split)
│   │   │   ├── test_layout_transformer.py      # NEW
│   │   │   ├── test_processing_manager.py      # FROM: test_feature_components.py (split)
│   │   │   └── test_update_strategy.py         # NEW
│   │   ├── indexer/                     # NEW: Indexer module tests
│   │   │   ├── __init__.py
│   │   │   ├── test_augmentation_tracker.py    # FROM: test_indexer_header_units.py (split)
│   │   │   ├── test_index_store.py
│   │   │   ├── test_parameter_normalizer.py
│   │   │   ├── test_processing_manager.py
│   │   │   ├── test_query_builder.py
│   │   │   └── test_sample_manager.py
│   │   ├── predictions/                 # NEW: Predictions module tests
│   │   │   ├── __init__.py
│   │   │   ├── test_aggregator.py
│   │   │   ├── test_array_registry.py          # FROM: data/predictions_components/
│   │   │   ├── test_indexer.py
│   │   │   ├── test_query.py
│   │   │   ├── test_ranker.py
│   │   │   ├── test_result.py
│   │   │   ├── test_schemas.py
│   │   │   ├── test_serializer.py
│   │   │   └── test_storage.py
│   │   ├── targets/                     # Existing - Keep as is
│   │   │   ├── __init__.py
│   │   │   ├── test_converters.py
│   │   │   ├── test_encoders.py
│   │   │   ├── test_processing_chain.py
│   │   │   ├── test_targets_refactored.py
│   │   │   └── test_transformers.py
│   │   ├── test_binning.py              # FROM: unit/utils/test_binning.py
│   │   ├── test_config.py               # Keep
│   │   ├── test_config_parser.py        # NEW
│   │   ├── test_dataset.py              # Keep + merge test_dataset_save_load_header_units.py
│   │   ├── test_dataset_wavelength_conversion.py  # Keep
│   │   ├── test_ensemble_utils.py       # NEW
│   │   ├── test_features.py             # NEW (high-level Features class tests)
│   │   ├── test_group_split_validation.py  # Keep
│   │   ├── test_indexer.py              # FROM: test_indexer_header_units.py (refactor)
│   │   ├── test_io.py                   # NEW
│   │   ├── test_metadata.py             # Keep + merge test_metadata_loading.py
│   │   ├── test_predictions.py          # Keep + FROM: test_predictions_header_units.py
│   │   ├── test_targets.py              # NEW (high-level Targets class tests)
│   │   └── test_types.py                # NEW
│   │
│   ├── operators/                       # NEW: Tests for operators
│   │   ├── __init__.py
│   │   ├── augmentation/
│   │   │   ├── __init__.py
│   │   │   ├── test_abc_augmenter.py
│   │   │   ├── test_random.py            # FROM: transforms/test_augmentation.py (split)
│   │   │   └── test_splines.py
│   │   ├── models/                      # FROM: unit/models/
│   │   │   ├── __init__.py
│   │   │   ├── test_base.py
│   │   │   ├── test_pytorch.py          # FROM: unit/models/
│   │   │   ├── test_sklearn.py          # NEW
│   │   │   └── test_tensorflow.py       # FROM: unit/models/
│   │   ├── splitters/
│   │   │   ├── __init__.py
│   │   │   └── test_splitters.py        # FROM: transforms/test_splitters.py
│   │   └── transforms/                  # FROM: unit/transforms/
│   │       ├── __init__.py
│   │       ├── test_features.py         # NEW
│   │       ├── test_nirs.py             # FROM: transforms/test_nirs.py
│   │       ├── test_presets.py          # NEW
│   │       ├── test_resampler.py        # FROM: data/test_resampler_header_units.py
│   │       ├── test_scalers.py          # NEW
│   │       ├── test_signal.py           # FROM: transforms/test_signal.py
│   │       └── test_targets.py          # NEW
│   │
│   ├── optimization/                    # NEW: Tests for optimization
│   │   ├── __init__.py
│   │   └── test_optuna.py
│   │
│   ├── pipeline/                        # Tests for pipeline (restructured)
│   │   ├── __init__.py
│   │   ├── config/                      # NEW: Config submodule tests
│   │   │   ├── __init__.py
│   │   │   ├── test_component_serialization.py  # FROM: test_serializer.py (split)
│   │   │   ├── test_context.py                  # FROM: tests/pipeline/test_context.py
│   │   │   ├── test_generator.py                # FROM: test_generator.py
│   │   │   └── test_pipeline_config.py          # FROM: test_config.py
│   │   ├── execution/                   # NEW: Execution submodule tests
│   │   │   ├── __init__.py
│   │   │   ├── test_builder.py
│   │   │   ├── test_executor.py
│   │   │   ├── test_orchestrator.py
│   │   │   └── test_result.py
│   │   ├── steps/                       # NEW: Steps submodule tests
│   │   │   ├── __init__.py
│   │   │   ├── test_parser.py
│   │   │   ├── test_router.py
│   │   │   └── test_step_runner.py
│   │   ├── storage/                     # NEW: Storage submodule tests
│   │   │   ├── __init__.py
│   │   │   ├── artifacts/
│   │   │   │   ├── __init__.py
│   │   │   │   └── test_*.py
│   │   │   ├── test_io.py
│   │   │   ├── test_io_exporter.py
│   │   │   ├── test_io_resolver.py
│   │   │   ├── test_io_writer.py
│   │   │   ├── test_library.py
│   │   │   └── test_manifest_manager.py         # FROM: test_manifest_manager.py
│   │   ├── test_explainer.py            # NEW
│   │   ├── test_predictor.py            # NEW
│   │   ├── test_runner.py               # Keep + comprehensive tests
│   │   ├── test_runner_comprehensive.py # Keep (or merge into test_runner.py)
│   │   ├── test_runner_normalization.py # Keep
│   │   ├── test_runner_predict.py       # Keep
│   │   ├── test_runner_regression_prevention.py  # Keep
│   │   ├── test_runner_state.py         # Keep
│   │   ├── test_serialization.py        # Keep (pipeline-level serialization)
│   │   └── test_binary_loader.py        # MOVE to storage/artifacts/
│   │
│   ├── utils/                           # Tests for utils
│   │   ├── __init__.py
│   │   ├── test_backend.py              # NEW
│   │   ├── test_balancing.py            # FROM: unit/utils/
│   │   ├── test_balancing_value_aware.py  # FROM: unit/utils/
│   │   ├── test_emoji.py                # NEW
│   │   └── test_spinner.py              # NEW
│   │
│   ├── visualization/                   # NEW: Tests for visualization
│   │   ├── __init__.py
│   │   ├── analysis/
│   │   │   └── test_*.py
│   │   ├── charts/
│   │   │   └── test_*.py
│   │   ├── chart_utils/
│   │   │   └── test_*.py
│   │   ├── test_predictions.py
│   │   └── test_reports.py
│   │
│   └── workspace/                       # FROM: tests/workspace/
│       ├── __init__.py
│       ├── test_catalog_export.py       # FROM: workspace/
│       ├── test_library_manager.py      # FROM: workspace/
│       └── test_query_reporting.py      # FROM: workspace/
│
└── integration/                         # Integration tests (consolidated)
    ├── __init__.py
    ├── README.md                        # Consolidated from integration_tests/README.md
    ├── INTEGRATION_TEST_SUMMARY.md      # FROM: integration_tests/
    ├── QUICK_REFERENCE.txt              # FROM: integration_tests/
    ├── run_integration_tests.py         # FROM: integration_tests/
    │
    ├── pipeline/                        # NEW: Pipeline-level integration tests
    │   ├── __init__.py
    │   ├── test_basic_pipeline.py               # FROM: integration_tests/
    │   ├── test_classification_integration.py   # FROM: integration_tests/
    │   ├── test_comprehensive_integration.py    # FROM: integration_tests/
    │   ├── test_finetune_integration.py         # FROM: integration_tests/
    │   ├── test_flexible_inputs_integration.py  # FROM: integration_tests/
    │   ├── test_groupsplit_integration.py       # FROM: integration_tests/
    │   ├── test_multisource_integration.py      # FROM: integration_tests/
    │   ├── test_pca_analysis_integration.py     # FROM: integration_tests/
    │   ├── test_prediction_reuse_integration.py # FROM: integration_tests/
    │   └── test_resampler_integration.py        # FROM: integration_tests/
    │
    ├── data/                            # NEW: Data module integration tests
    │   ├── __init__.py
    │   ├── test_array_registry_integration.py   # FROM: tests/data/
    │   ├── test_dataset_with_context.py         # FROM: tests/data/
    │   └── test_predictions_workflow.py         # NEW
    │
    ├── augmentation/                    # NEW: Augmentation integration tests
    │   ├── __init__.py
    │   ├── test_augmentation_end_to_end.py      # FROM: integration/
    │   ├── test_augmentation_integration.py     # FROM: integration/
    │   ├── test_dataset_augmentation.py         # FROM: integration/
    │   └── test_sample_augmentation_integration.py  # FROM: integration_tests/
    │
    ├── explainability/                  # NEW: Explainability integration tests
    │   ├── __init__.py
    │   └── test_shap_integration.py             # FROM: integration_tests/
    │
    └── end_to_end/                      # NEW: Full end-to-end workflow tests
        ├── __init__.py
        └── test_complete_workflows.py           # Combine some comprehensive tests
```

---

## Detailed Migration Plan

### Phase 1: Preparation (Pre-Migration)

**Objective**: Ensure we don't break anything during the migration.

#### Actions:
1. **Baseline Test Run**
   ```bash
   pytest tests/ --tb=no -q > pre_migration_results.txt
   ```
   - Document which tests pass/fail currently
   - This is our reference point

2. **Create Branch**
   ```bash
   git checkout -b test-reorganization
   ```

3. **Backup Current State**
   - Commit all current changes
   - Tag the commit: `git tag test-reorg-baseline`

4. **Create Migration Tracking**
   - Use this document to track completion
   - Mark each file migration as completed

### Phase 2: Create New Directory Structure

**Objective**: Create all new directories without moving files yet.

#### Actions:
1. **Create new directories** (in `tests/`):
   ```bash
   # Unit test structure
   mkdir -p unit/cli/commands
   mkdir -p unit/core
   mkdir -p unit/data/{loaders,dataset,features,indexer,predictions,targets}
   mkdir -p unit/operators/{augmentation,models,splitters,transforms}
   mkdir -p unit/optimization
   mkdir -p unit/pipeline/{config,execution,steps,storage/artifacts}
   mkdir -p unit/visualization/{analysis,charts,chart_utils}

   # Integration test structure
   mkdir -p integration/{pipeline,data,augmentation,explainability,end_to_end}

   # Fixtures
   mkdir -p fixtures/{sample_datasets,sample_pipelines}
   ```

2. **Create `__init__.py` files** for all new directories

3. **Update `fixtures/` directory**:
   - Move/refactor `unit/utils/test_data_generator.py` → `fixtures/data_generators.py`
   - This becomes a shared fixture, not a test file

### Phase 3: Migrate Unit Tests (by module)

**Objective**: Move and reorganize unit tests module by module.

#### Strategy:
- One module at a time
- Run tests after each module migration
- Update imports as needed
- Fix any broken tests immediately

#### Migration Order:

##### 3.1. Core Module (NEW)
- [ ] Create `unit/core/test_*.py` for each core module
- [ ] Write placeholder tests if none exist
- [ ] Verify: `pytest tests/unit/core/ -v`

##### 3.2. Utils Module
- [ ] Move `unit/utils/test_balancing.py` → `unit/utils/test_balancing.py` (no change)
- [ ] Move `unit/utils/test_balancing_value_aware.py` → `unit/utils/test_balancing_value_aware.py` (no change)
- [ ] Move `unit/utils/test_binning.py` → `unit/data/test_binning.py` (belongs in data module)
- [ ] Create `unit/utils/test_backend.py`, `test_emoji.py`, `test_spinner.py` (NEW)
- [ ] Remove `unit/utils/test_data_generator.py` (moved to fixtures/)
- [ ] Verify: `pytest tests/unit/utils/ -v`

##### 3.3. Data Module (Complex - Many files)

**Loaders:**
- [ ] Split `unit/data/test_loaders.py` into:
  - `unit/data/loaders/test_loader.py`
  - `unit/data/loaders/test_csv_loader.py`
- [ ] Merge `unit/data/test_csv_loader_header_units.py` into `unit/data/loaders/test_csv_loader.py`
- [ ] Verify: `pytest tests/unit/data/loaders/ -v`

**Dataset Accessors:**
- [ ] Create `unit/data/dataset/test_feature_accessor.py` (NEW)
- [ ] Create `unit/data/dataset/test_metadata_accessor.py` (NEW)
- [ ] Create `unit/data/dataset/test_target_accessor.py` (NEW)
- [ ] Verify: `pytest tests/unit/data/dataset/ -v`

**Features Module:**
- [ ] Split `unit/data/test_feature_components.py` into:
  - `unit/data/features/test_array_storage.py`
  - `unit/data/features/test_feature_constants.py`
  - `unit/data/features/test_header_manager.py`
  - `unit/data/features/test_processing_manager.py`
- [ ] Move `unit/data/test_feature_source_header_units.py` → `unit/data/features/test_feature_source.py`
- [ ] Refactor `unit/data/test_preprocessing_header_units.py` → `unit/data/features/test_augmentation_handler.py`
- [ ] Create `unit/data/features/test_layout_transformer.py` (NEW)
- [ ] Create `unit/data/features/test_update_strategy.py` (NEW)
- [ ] Verify: `pytest tests/unit/data/features/ -v`

**Indexer Module:**
- [ ] Split `unit/data/test_indexer_header_units.py` into:
  - `unit/data/indexer/test_augmentation_tracker.py`
  - `unit/data/indexer/test_index_store.py`
  - `unit/data/indexer/test_parameter_normalizer.py`
  - `unit/data/indexer/test_processing_manager.py`
  - `unit/data/indexer/test_query_builder.py`
  - `unit/data/indexer/test_sample_manager.py`
- [ ] Refactor `unit/data/test_indexer_header_units.py` → `unit/data/test_indexer.py`
- [ ] Verify: `pytest tests/unit/data/indexer/ -v`

**Predictions Module:**
- [ ] Move `data/predictions_components/test_array_registry.py` → `unit/data/predictions/test_array_registry.py`
- [ ] Create remaining predictions tests:
  - `test_aggregator.py`
  - `test_indexer.py`
  - `test_query.py`
  - `test_ranker.py`
  - `test_result.py`
  - `test_schemas.py`
  - `test_serializer.py`
  - `test_storage.py`
- [ ] Merge `unit/data/test_predictions_header_units.py` into `unit/data/test_predictions.py`
- [ ] Verify: `pytest tests/unit/data/predictions/ -v`

**Targets Module:**
- [ ] Keep existing `unit/data/targets/` as is (already well organized)
- [ ] Verify: `pytest tests/unit/data/targets/ -v`

**Top-level Data Tests:**
- [ ] Keep `unit/data/test_config.py`
- [ ] Create `unit/data/test_config_parser.py` (NEW)
- [ ] Merge `unit/data/test_dataset_save_load_header_units.py` into `unit/data/test_dataset.py`
- [ ] Keep `unit/data/test_dataset_wavelength_conversion.py`
- [ ] Create `unit/data/test_ensemble_utils.py` (NEW)
- [ ] Create `unit/data/test_features.py` (NEW - high-level)
- [ ] Keep `unit/data/test_group_split_validation.py`
- [ ] Create `unit/data/test_io.py` (NEW)
- [ ] Merge `unit/data/test_metadata_loading.py` into `unit/data/test_metadata.py`
- [ ] Create `unit/data/test_targets.py` (NEW - high-level)
- [ ] Create `unit/data/test_types.py` (NEW)
- [ ] Verify: `pytest tests/unit/data/ -v`

##### 3.4. Operators Module (Transform unit/transforms/ → unit/operators/)

**Augmentation:**
- [ ] Split `unit/transforms/test_augmentation.py` into:
  - `unit/operators/augmentation/test_abc_augmenter.py`
  - `unit/operators/augmentation/test_random.py`
  - `unit/operators/augmentation/test_splines.py`
- [ ] Verify: `pytest tests/unit/operators/augmentation/ -v`

**Models:**
- [ ] Move `unit/models/` → `unit/operators/models/`
- [ ] Keep `test_pytorch.py` and `test_tensorflow.py`
- [ ] Create `test_base.py` and `test_sklearn.py` (NEW)
- [ ] Verify: `pytest tests/unit/operators/models/ -v`

**Splitters:**
- [ ] Move `unit/transforms/test_splitters.py` → `unit/operators/splitters/test_splitters.py`
- [ ] Verify: `pytest tests/unit/operators/splitters/ -v`

**Transforms:**
- [ ] Move `unit/transforms/test_nirs.py` → `unit/operators/transforms/test_nirs.py`
- [ ] Move `unit/transforms/test_signal.py` → `unit/operators/transforms/test_signal.py`
- [ ] Move `unit/data/test_resampler_header_units.py` → `unit/operators/transforms/test_resampler.py`
- [ ] Create NEW tests:
  - `test_features.py`
  - `test_presets.py`
  - `test_scalers.py`
  - `test_targets.py`
- [ ] Verify: `pytest tests/unit/operators/transforms/ -v`

##### 3.5. Controllers Module

**Dataset Controllers:**
- [ ] Move `unit/controllers/test_indexer_augmentation.py` → `unit/controllers/dataset/test_indexer_augmentation.py`
- [ ] Move `unit/controllers/test_transformer_augmentation.py` → `unit/controllers/dataset/test_sample_augmentation.py`
- [ ] Move `unit/controllers/test_split_augmentation.py` → `unit/controllers/dataset/test_split_augmentation.py`
- [ ] Verify: `pytest tests/unit/controllers/dataset/ -v`

**Other Controllers:**
- [ ] Create tests for other controller modules (NEW):
  - `test_base.py`
  - `test_controller.py`
  - `test_presets.py`
  - `test_registry.py`
  - chart/, charts/, data/, flow/, log/, models/, sklearn/, splitters/, transforms/
- [ ] Verify: `pytest tests/unit/controllers/ -v`

##### 3.6. Pipeline Module (Restructure)

**Config Submodule:**
- [ ] Move `tests/pipeline/test_context.py` → `unit/pipeline/config/test_context.py`
- [ ] Move `unit/pipeline/test_config.py` → `unit/pipeline/config/test_pipeline_config.py`
- [ ] Move `unit/pipeline/test_generator.py` → `unit/pipeline/config/test_generator.py`
- [ ] Split `unit/pipeline/test_serializer.py` → `unit/pipeline/config/test_component_serialization.py`
- [ ] Verify: `pytest tests/unit/pipeline/config/ -v`

**Execution Submodule:**
- [ ] Create NEW tests:
  - `test_builder.py`
  - `test_executor.py`
  - `test_orchestrator.py`
  - `test_result.py`
- [ ] Verify: `pytest tests/unit/pipeline/execution/ -v`

**Steps Submodule:**
- [ ] Create NEW tests:
  - `test_parser.py`
  - `test_router.py`
  - `test_step_runner.py`
- [ ] Verify: `pytest tests/unit/pipeline/steps/ -v`

**Storage Submodule:**
- [ ] Keep `unit/pipeline/test_manifest_manager.py` → `unit/pipeline/storage/test_manifest_manager.py`
- [ ] Move `unit/pipeline/test_binary_loader.py` → `unit/pipeline/storage/artifacts/test_binary_loader.py`
- [ ] Create NEW tests:
  - `test_io.py`
  - `test_io_exporter.py`
  - `test_io_resolver.py`
  - `test_io_writer.py`
  - `test_library.py`
- [ ] Verify: `pytest tests/unit/pipeline/storage/ -v`

**Top-level Pipeline Tests:**
- [ ] Keep runner tests as is:
  - `test_runner.py`
  - `test_runner_comprehensive.py`
  - `test_runner_normalization.py`
  - `test_runner_predict.py`
  - `test_runner_regression_prevention.py`
  - `test_runner_state.py`
  - `test_serialization.py`
- [ ] Create NEW:
  - `test_explainer.py`
  - `test_predictor.py`
- [ ] Verify: `pytest tests/unit/pipeline/ -v`

##### 3.7. Optimization Module (NEW)
- [ ] Create `unit/optimization/test_optuna.py`
- [ ] Verify: `pytest tests/unit/optimization/ -v`

##### 3.8. Visualization Module (NEW)
- [ ] Create tests for visualization components (NEW):
  - analysis/, charts/, chart_utils/
  - `test_predictions.py`
  - `test_reports.py`
- [ ] Verify: `pytest tests/unit/visualization/ -v`

##### 3.9. Workspace Module
- [ ] Move `tests/workspace/` → `unit/workspace/`
- [ ] Keep all existing tests
- [ ] Verify: `pytest tests/unit/workspace/ -v`

##### 3.10. CLI Module (NEW)
- [ ] Create `unit/cli/test_main.py` (NEW)
- [ ] Create `unit/cli/commands/test_*.py` (NEW)
- [ ] Verify: `pytest tests/unit/cli/ -v`

### Phase 4: Consolidate Integration Tests

**Objective**: Merge `integration/` and `integration_tests/` into single organized structure.

#### Actions:

##### 4.1. Pipeline Integration Tests
- [ ] Move all from `integration_tests/`:
  - `test_basic_pipeline.py` → `integration/pipeline/`
  - `test_classification_integration.py` → `integration/pipeline/`
  - `test_comprehensive_integration.py` → `integration/pipeline/`
  - `test_finetune_integration.py` → `integration/pipeline/`
  - `test_flexible_inputs_integration.py` → `integration/pipeline/`
  - `test_groupsplit_integration.py` → `integration/pipeline/`
  - `test_multisource_integration.py` → `integration/pipeline/`
  - `test_pca_analysis_integration.py` → `integration/pipeline/`
  - `test_prediction_reuse_integration.py` → `integration/pipeline/`
  - `test_resampler_integration.py` → `integration/pipeline/`
- [ ] Verify: `pytest tests/integration/pipeline/ -v`

##### 4.2. Data Integration Tests
- [ ] Move from `tests/data/`:
  - `test_array_registry_integration.py` → `integration/data/`
  - `test_dataset_with_context.py` → `integration/data/`
- [ ] Create NEW:
  - `test_predictions_workflow.py`
- [ ] Verify: `pytest tests/integration/data/ -v`

##### 4.3. Augmentation Integration Tests
- [ ] Move from `integration/`:
  - `test_augmentation_end_to_end.py` → `integration/augmentation/`
  - `test_augmentation_integration.py` → `integration/augmentation/`
  - `test_dataset_augmentation.py` → `integration/augmentation/`
- [ ] Move from `integration_tests/`:
  - `test_sample_augmentation_integration.py` → `integration/augmentation/`
- [ ] Verify: `pytest tests/integration/augmentation/ -v`

##### 4.4. Explainability Integration Tests
- [ ] Move from `integration_tests/`:
  - `test_shap_integration.py` → `integration/explainability/`
- [ ] Verify: `pytest tests/integration/explainability/ -v`

##### 4.5. End-to-End Tests
- [ ] Consider creating `integration/end_to_end/test_complete_workflows.py`
- [ ] Consolidate some comprehensive tests here
- [ ] Verify: `pytest tests/integration/end_to_end/ -v`

##### 4.6. Documentation
- [ ] Merge documentation:
  - Consolidate `integration/` and `integration_tests/README.md`
  - Move to `integration/README.md`
  - Keep `INTEGRATION_TEST_SUMMARY.md` and `QUICK_REFERENCE.txt`
- [ ] Move `run_integration_tests.py` to `integration/`

### Phase 5: Clean Up

**Objective**: Remove old directories and update documentation.

#### Actions:

##### 5.1. Remove Old Directories
- [ ] Delete empty directories:
  ```bash
  rmdir tests/dataset
  rmdir tests/pipeline_runner
  rmdir tests/serialization
  rmdir tests/utils
  rmdir tests/integration_tests
  rmdir tests/unit/transforms
  rmdir tests/unit/models
  rmdir tests/data/predictions_components
  rmdir tests/data
  rmdir tests/pipeline
  ```

##### 5.2. Update Documentation
- [ ] Update `tests/README.md` with new structure
- [ ] Update import documentation
- [ ] Update CI/CD scripts if needed
- [ ] Update CONTRIBUTING.md if it references test structure

##### 5.3. Update Test Runners
- [ ] Review `run_tests.py`
- [ ] Consider consolidating `run_runner_tests.py` into `run_tests.py`
- [ ] Update any pytest.ini configurations

##### 5.4. Final Verification
- [ ] Run full test suite:
  ```bash
  pytest tests/ -v > post_migration_results.txt
  ```
- [ ] Compare with pre-migration results
- [ ] Document any new failures (should be none if migration done correctly)
- [ ] Fix any import issues
- [ ] Ensure all test markers still work

### Phase 6: Post-Migration Tasks

**Objective**: Finalize and document the new structure.

#### Actions:

##### 6.1. Create New Tests for Missing Coverage
Based on the reorganization, identify modules without tests and create placeholders:
- [ ] `unit/core/` - All modules
- [ ] `unit/cli/` - All modules
- [ ] `unit/operators/augmentation/test_splines.py`
- [ ] `unit/operators/models/test_sklearn.py`
- [ ] `unit/operators/transforms/` - Several missing
- [ ] `unit/pipeline/execution/` - All modules
- [ ] `unit/pipeline/steps/` - All modules
- [ ] `unit/pipeline/storage/` - Several missing
- [ ] `unit/visualization/` - All modules
- [ ] `unit/optimization/test_optuna.py`

##### 6.2. Update CI/CD
- [ ] Update GitHub Actions workflows
- [ ] Update test coverage reporting
- [ ] Update any pre-commit hooks

##### 6.3. Update Examples
- [ ] Check if examples reference test files
- [ ] Update any test-related documentation in examples/

##### 6.4. Create Migration Summary
- [ ] Document what was moved where
- [ ] Document any tests that were split or merged
- [ ] Document any new tests created

---

## Migration Checklist Template

For each file migration, use this checklist:

```markdown
### File: [original_path] → [new_path]

- [ ] Created target directory
- [ ] Moved/copied file
- [ ] Updated imports in file
- [ ] Updated imports in other files referencing this
- [ ] Ran tests: `pytest [new_path] -v`
- [ ] Tests pass ✓ / Tests fail ✗
- [ ] If failed: documented issue and fix
- [ ] Committed change
```

---

## Key Principles for Reorganization

1. **Mirror Source Structure**: Test structure should exactly mirror `nirs4all/` structure
2. **One Module = One Test Directory**: Each source module gets its own test directory
3. **Clear Boundaries**:
   - **Unit tests**: Test single classes/functions in isolation
   - **Integration tests**: Test multiple components working together
   - **End-to-end tests**: Test complete workflows
4. **Meaningful Names**: Test files should clearly indicate what they test
5. **No Dead Code**: Remove empty directories immediately
6. **Documentation**: Keep README files up to date
7. **Fixtures**: Centralize reusable test data and fixtures

---

## Expected Benefits

After reorganization:

1. **Easier Navigation**: Developers can find tests matching source code structure
2. **Better Organization**: Clear separation of unit vs integration tests
3. **Improved Maintainability**: Tests grouped by module responsibility
4. **Better Coverage Tracking**: Easy to see which modules lack tests
5. **Clearer Test Purpose**: Test location indicates scope (unit/integration/e2e)
6. **Reduced Confusion**: No more duplicate directories or unclear test placement
7. **Scalability**: New modules can follow established pattern

---

## Risk Mitigation

### Potential Issues:

1. **Import Breakage**: Moving files will break imports
   - **Mitigation**: Use search/replace carefully, run tests frequently

2. **Lost Test Coverage**: Accidentally skipping tests during move
   - **Mitigation**: Compare pre/post test counts, use git to track moves

3. **CI/CD Failures**: Workflows may reference old paths
   - **Mitigation**: Review and update CI/CD configs early

4. **Merge Conflicts**: Other branches may conflict
   - **Mitigation**: Do migration in dedicated branch, communicate with team

5. **Time Investment**: Large refactoring takes time
   - **Mitigation**: Migrate incrementally, one module at a time

---

## Success Metrics

Migration is successful when:

- [ ] All tests run in new locations
- [ ] Test count unchanged (or increased if new tests added)
- [ ] No empty directories remain
- [ ] Documentation is updated
- [ ] CI/CD passes
- [ ] Structure mirrors `nirs4all/` exactly
- [ ] All imports work correctly
- [ ] Code review approval

---

## Timeline Estimate

- **Phase 1 (Preparation)**: 1 hour
- **Phase 2 (Directory Creation)**: 1 hour
- **Phase 3 (Unit Test Migration)**: 8-12 hours
  - Core: 1 hour
  - Utils: 0.5 hour
  - Data: 3-4 hours (most complex)
  - Operators: 2 hours
  - Controllers: 1 hour
  - Pipeline: 2-3 hours (restructuring needed)
  - Others: 1-2 hours
- **Phase 4 (Integration Tests)**: 2-3 hours
- **Phase 5 (Clean Up)**: 1-2 hours
- **Phase 6 (Post-Migration)**: 2-3 hours

**Total Estimated Time**: 15-22 hours (2-3 days of focused work)

---

## Questions for Review

Before implementing, consider:

1. Should we consolidate `test_runner_*.py` files into a single file or keep them separate?
2. Should `run_runner_tests.py` be merged into `run_tests.py`?
3. Are there any tests that should be moved to integration that are currently in unit?
4. Should we create placeholder tests for all missing coverage now, or later?
5. Any special considerations for CI/CD that need addressing?

---

## Notes

- This is a **planning document only** - do not implement without approval
- Some test files may be failing already - focus is on organization, not fixing tests yet
- After reorganization, a separate effort should focus on fixing failing tests
- Consider doing this migration in stages (module by module) rather than all at once
- Use `git mv` to preserve file history when moving files

---

## Approval Required

- [ ] Reviewed by: ___________
- [ ] Approved by: ___________
- [ ] Start Date: ___________
- [ ] Target Completion: ___________

---

*END OF DOCUMENT*
