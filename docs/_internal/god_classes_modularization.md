    # God Classes Modularization Guide

> Generated 2026-04-01 — Analysis of all Python source files exceeding 2,000 lines in the nirs4all library.

---

## Overview

| # | File | Lines | Classes | Top Issue |
|---|------|-------|---------|-----------|
| 1 | [controllers/data/merge.py](#1-controllersdatamergepy) | 5,402 | 7 | MergeController = 4,520 lines, 48 methods, ~770 lines duplicated |
| 2 | [synthesis/fitter.py](#2-synthesisfitterpy) | 5,203 | 24 | 8 distinct functional domains in one file |
| 3 | [synthesis/_constants.py](#3-synthesis_constantspy) | 2,844 | 0 | 126 spectral components in one monolithic function |
| 4 | [pipeline/storage/workspace_store.py](#4-pipelinestorageworlkspace_storepy) | 2,833 | 1 | WorkspaceStore = 71 methods, 15+ responsibilities |
| 5 | [data/predictions.py](#5-datapredictionspy) | 2,503 | 2 | Predictions = 57 methods, top() = 281 lines |
| 6 | [controllers/data/branch.py](#6-controllersdatabranchpy) | 2,320 | 1 | BranchController = 31 methods, parallel+sequential duplication |
| 7 | [controllers/models/base_model.py](#7-controllersmodelsbase_modelpy) | 2,272 | 1 | BaseModelController = 40 methods, launch_training() = 251 lines |
| 8 | [operators/transforms/nirs.py](#8-operatorstransformsnirspy) | 2,229 | 22 | 9 baseline aliases with near-identical structure |
| 9 | [visualization/pipeline_diagram.py](#9-visualizationpipeline_diagrampy) | 2,184 | 2 | PipelineDiagram = 2,067 lines, _build_dag_from_trace() = 322 lines |
| 10 | [controllers/models/meta_model.py](#10-controllersmodelsmeta_modelpy) | 2,128 | 1 | MetaModelController = 34 methods, _persist_meta_model() = 163 lines |
| 11 | [pipeline/execution/orchestrator.py](#11-pipelineexecutionorchestratorpy) | 2,117 | 1 | execute() = 613 lines, 4 mixed concerns |
| 12 | [data/dataset.py](#12-datadatasetpy) | 2,083 | 1 | SpectroDataset = 85+ methods, 14 functional domains |

---

## 1. controllers/data/merge.py

**Severity: CRITICAL** — 5,402 lines, MergeController alone is 4,520 lines (83%).

### Key problems

- **48 methods** in a single class; 15 methods exceed 100 lines.
- **~770 lines of code duplication** across 3 OOF reconstruction implementations, 3 feature collection variants, and 2 partition prediction collectors.
- Largest method: `_execute_branch_merge()` at 355 lines.

### Proposed split

```
controllers/data/merge/
├── __init__.py              # Re-exports MergeController
├── controller.py            # Main MergeController (~800 lines, orchestration only)
├── types.py                 # 4 dataclasses: BranchAnalysisResult, AsymmetryReport, etc. (~90 lines)
├── parser.py                # MergeConfigParser (~310 lines)
├── analyzer.py              # AsymmetricBranchAnalyzer (~215 lines)
├── features.py              # Unified feature collection (replaces 3 variants, ~450 lines)
├── predictions.py           # Unified prediction collection + OOF reconstructor (~1,100 lines)
├── sources.py               # Source merge strategies: concat/stack/dict (~420 lines)
├── validators.py            # All validation methods (~150 lines)
└── utils.py                 # Helpers, naming, logging (~280 lines)
```

### Key simplifications

| Method | Current | Action |
|--------|---------|--------|
| `_execute_branch_merge()` | 355 lines | Split into regular vs disjoint dispatchers (~100 lines each) |
| `_collect_predictions()` | 234 lines | Extract `PredictionCollector` class, separate model selection from aggregation |
| `_collect_disjoint_predictions()` | 280 lines | Reuse base prediction collection, extract model ranking |
| OOF reconstruction | 3 implementations (~340 lines) | Consolidate into single `OOFReconstructor` |
| Feature collection | 3 variants (~280 lines) | Consolidate into single parameterized `collect_features()` |

### Priority

1. **Low risk**: Extract `types.py`, `parser.py`, `analyzer.py`
2. **Medium risk**: Extract `features.py`, `sources.py`, `validators.py`, `utils.py`
3. **High risk/high value**: Extract `predictions.py`, simplify core controller

---

## 2. synthesis/fitter.py

**Severity: CRITICAL** — 5,203 lines, 24 classes, 8 distinct functional domains.

### Key problems

- 8 unrelated fitter classes in one file (RealDataFitter, ComponentFitter, OptimizedComponentFitter, RealBandFitter, VarianceFitter, ForwardModelFitter, DerivativeAwareForwardModelFitter).
- Preprocessing detection logic duplicated 3 times (~192 + 73 + 25 lines).
- Chebyshev baseline construction duplicated 4 times.
- `RealDataFitter.fit()` = 175 lines, `_infer_preprocessing()` = 192 lines.

### Proposed split

```
synthesis/
├── fitter.py                    # Thin re-export module (~30 lines)
├── _inference_results.py        # 10 dataclasses + SpectralProperties + FittedParameters (~700 lines)
├── _spectral_analysis.py        # compute_spectral_properties() + 9 helpers (~600 lines)
├── _preprocessing_utils.py      # Unified detection + application (replaces 3 duplicates, ~200 lines)
├── _real_data_fitter.py         # RealDataFitter class (~1,300 lines)
├── _component_fitting.py        # ComponentFitter + fit_components() (~500 lines)
├── _optimized_fitting.py        # OptimizedComponentFitter + greedy selection (~500 lines)
├── _band_fitting.py             # RealBandFitter (~400 lines)
├── _variance_fitting.py         # VarianceFitter (~300 lines)
├── _forward_model_fitting.py    # ForwardModelFitter + DerivativeAware variant (~400 lines)
└── _fitter_utils.py             # Shared: chebyshev_baseline(), gaussian_band(), fit_metrics() (~300 lines)
```

### Key simplifications

| Item | Current | Action |
|------|---------|--------|
| Preprocessing detection | 3 implementations | Single `detect_preprocessing_type()` in `_preprocessing_utils.py` |
| Chebyshev baseline | 4 implementations | Single `chebyshev_baseline()` in `_fitter_utils.py` |
| `RealDataFitter.fit()` | 175 lines | Extract `_run_all_inferences()` + builder pattern for configs |
| `_infer_preprocessing()` | 192 lines | Table-driven heuristics with threshold dictionaries |
| `RealBandFitter.fit()` | 211 lines | Strategy pattern for variable vs fixed sigma, extract retry logic |

### Priority

1. **Phase 1**: Extract `_inference_results.py`, `_preprocessing_utils.py`, `_fitter_utils.py`
2. **Phase 2**: Extract `_spectral_analysis.py`, `_real_data_fitter.py`
3. **Phase 3**: Extract individual fitter modules (component, optimized, band, variance, forward)

---

## 3. synthesis/_constants.py

**Severity: MEDIUM** — 2,844 lines of spectral component data, not really a "god class" but a god data file.

### Key problems

- 126 predefined spectral components in a single `get_predefined_components()` function (2,400+ lines).
- Metadata dictionary (`_COMPONENT_METADATA`, ~150 entries) separate from component definitions creates sync issues.
- Mixes data constants, configuration parameters, and enrichment logic.

### Proposed split (category-based)

```
synthesis/
├── _constants.py                    # Entry point with get_predefined_components() (~100 lines)
├── _constants/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py              # ALL_COMPONENTS dict, lazy loading
│   │   ├── water.py                 # 2 components (~100 lines)
│   │   ├── proteins.py              # 12 components (~350 lines)
│   │   ├── lipids.py                # 20 components (~450 lines)
│   │   ├── carbohydrates.py         # 18 components (~380 lines)
│   │   ├── alcohols.py              # 9 components (~250 lines)
│   │   ├── organic_acids.py         # 12 components (~320 lines)
│   │   ├── pigments.py              # 27 components (~580 lines)
│   │   ├── pharmaceuticals.py       # 10 components (~280 lines)
│   │   ├── polymers.py              # 11 components (~320 lines)
│   │   ├── solvents.py              # 6 components (~200 lines)
│   │   └── minerals.py              # 8 components (~250 lines)
│   ├── metadata.py                  # _COMPONENT_METADATA + enrichment (~200 lines)
│   └── config.py                    # Wavelength params, complexity presets (~100 lines)
```

### Key simplification

- Merge metadata into component definitions (inline) to eliminate sync issues.
- Lazy-load by category for reduced import overhead.
- Backward compatible via re-exports.

---

## 4. pipeline/storage/workspace_store.py

**Severity: HIGH** — 2,833 lines, 1 class, 71 methods, 15+ responsibilities.

### Key problems

- God class pattern: DB lifecycle, runs CRUD, pipelines CRUD, predictions persistence, prediction analytics, chain storage, chain summaries, exports, artifact management, artifact cleanup, project management, logging.
- 3 methods > 100 lines: `cleanup_transient_artifacts()` (222), `bulk_update_chain_summaries()` (205), `update_chain_summary()` (145).
- ~150 lines of duplicated patterns (JSON deserialization, lock+ensure-open, dynamic SQL WHERE, artifact ID collection).

### Proposed split

```
pipeline/storage/
├── workspace_store.py               # Facade with delegation (~400 lines)
├── workspace_connection.py          # DB lifecycle, retry logic, transactions (~180 lines)
├── workspace_run_pipeline.py        # Run + pipeline CRUD (~180 lines)
├── workspace_prediction.py          # Prediction persistence (~180 lines)
├── workspace_prediction_queries.py  # Prediction analytics (~350 lines)
├── workspace_chain.py               # Chain storage & replay (~200 lines)
├── workspace_chain_summary.py       # Chain summary computation (~450 lines)
├── workspace_artifact.py            # Basic artifact I/O (~150 lines)
├── workspace_artifact_cache.py      # Artifact cleanup (~350 lines)
├── workspace_export.py              # Export operations (~250 lines)
├── workspace_deletion.py            # Deletion & cleanup (~100 lines)
├── workspace_project.py             # Project management (~80 lines)
├── workspace_logging.py             # Step logging (~70 lines)
└── workspace_utils.py               # JSON helpers, SQL builders (~80 lines)
```

### Key simplifications

| Method | Current | Action |
|--------|---------|--------|
| `cleanup_transient_artifacts()` | 222 lines | Split multi-stage logic into `_identify_transient()`, `_check_fold_protection()`, `_delete_artifacts()` |
| `bulk_update_chain_summaries()` | 205 lines | Extract temp table logic, deduplicate with `update_chain_summary()` |
| JSON deserialization | 4 duplicates | Single `_deserialize_row()` helper |
| Lock + ensure open | 25+ occurrences | Context manager or decorator |

---

## 5. data/predictions.py

**Severity: HIGH** — 2,503 lines, Predictions class = 57 methods.

### Key problems

- `top()` = 281 lines handling 14+ parameters and 5 distinct concerns.
- `merge_stores()` = 190 lines with nested exception handling.
- `aggregate()` = 131 lines with if/elif for aggregation methods.
- JSON deserialization pattern duplicated 8 times.
- Array validity checking duplicated ~20 times.

### Proposed split

```
data/
├── predictions.py                   # Facade (~400 lines)
├── _predictions/
│   ├── __init__.py
│   ├── ranking.py                   # RankingEngine: top(), get_best(), scoring (~300 lines)
│   ├── aggregation.py               # AggregationEngine + strategy classes (~200 lines)
│   ├── filtering.py                 # FilteringEngine (~150 lines)
│   ├── storage.py                   # StorageEngine: add, flush, load (~250 lines)
│   ├── maintenance.py               # MaintenanceEngine: cleanup, stats (~200 lines)
│   ├── merge_engine.py              # StoreMergeEngine (~150 lines)
│   ├── utilities.py                 # JSON parsing, array validation, group keys (~150 lines)
│   └── conversion.py               # to_dataframe, to_dicts, to_pandas (~100 lines)
```

### Key simplifications

| Item | Current | Action |
|------|---------|--------|
| `top()` | 281 lines | Split into `_compute_scores()`, `_deduplicate()`, `_group_candidates()`, `_sort_candidates()` |
| `aggregate()` | 131 lines | Strategy pattern: `MeanAggregator`, `MedianAggregator`, `VoteAggregator` |
| JSON deserialization | 8 duplicates | Single `_parse_json(value, default)` |
| Array validity | ~20 duplicates | Single `_is_array_valid(arr)` |

---

## 6. controllers/data/branch.py

**Severity: HIGH** — 2,320 lines, 1 class, 31 methods.

### Key problems

- Mixes 4 separation strategies (by_tag, by_metadata, by_filter, by_source) + duplication branches + parallel execution.
- `_execute_branches_parallel()` = 237 lines with nested worker function.
- `_execute_branches_sequential()` = 169 lines with mixed concerns.
- Selector/partition handling duplicated 4+ times, branch context creation duplicated 3+ times.

### Proposed split

```
controllers/data/branch/
├── __init__.py
├── controller.py                    # Main BranchController (~300 lines)
├── dispatcher.py                    # Mode detection (~130 lines)
├── duplication_branch.py            # Duplication logic (~180 lines)
├── separation/
│   ├── __init__.py
│   ├── by_tag.py                    # Tag-based (~100 lines)
│   ├── by_metadata.py               # Metadata-based (~120 lines)
│   ├── by_filter.py                 # Filter-based (~120 lines)
│   └── by_source.py                 # Source-based (~220 lines)
├── parser.py                        # Branch definition parsing (~250 lines)
├── parallel_execution.py            # Parallel + sequential executors (~500 lines)
├── feature_snapshots.py             # CoW snapshot management (~100 lines)
├── naming.py                        # Step/branch naming (~120 lines)
└── utils.py                         # Shared utilities (~150 lines)
```

### Key simplifications

- Extract `_get_sample_indices_for_mode()` to eliminate 4 duplicates.
- Extract `_create_branch_context()` to eliminate 3 duplicates.
- Use context manager `trace_branch_execution()` to eliminate 5 recorder entry/exit duplicates.
- Extract `_execute_branch_chunk_worker` from nested function to proper class.

---

## 7. controllers/models/base_model.py

**Severity: HIGH** — 2,272 lines, 1 abstract class, 40 methods.

### Key problems

- `launch_training()` = 251 lines handling 8 sub-tasks.
- `train()` = 216 lines mixing fold iteration, ensemble logic, artifact management.
- Fold averaging (6 methods, 419 lines) = separable domain.
- Partition iteration pattern duplicated 5+ times.

### Proposed split

```
controllers/models/
├── base_model.py                    # Core orchestration + abstract interface (~450 lines)
├── data_handler.py                  # DataExtractionHelper: get_xy, fold remapping (~200 lines)
├── fold_averager.py                 # FoldAverager + EnsembleVoter (~350 lines)
├── prediction_handler.py            # PredictionHandler + ScoreFormatter (~200 lines)
└── model_persistence.py             # ModelPersistenceManager + ModelLoader (~150 lines)
```

### Key simplifications

| Method | Current | Action |
|--------|---------|--------|
| `launch_training()` | 251 lines | Split into `_obtain_model()`, `_train_model_step()`, `_execute_prediction_workflow()` |
| `train()` | 216 lines | Extract predict mode to `_predict_mode_workflow()`, fold loop to `_train_fold_models()` |
| `_create_fold_averages()` | 129 lines | Delegate to `FoldAverager` with regression/classification strategy |
| Partition iteration | 5 duplicates | `PartitionProcessor` utility class |

---

## 8. operators/transforms/nirs.py

**Severity: MEDIUM** — 2,229 lines, 22 classes.

### Key problems

- 9 baseline correction aliases (AirPLS, ArPLS, IASLS, etc.) with near-identical ~35-line structure = 315 lines of boilerplate.
- Sparse matrix validation duplicated 24 times across fit/transform methods.
- 22 unrelated transform classes in one file.

### Proposed split

```
operators/transforms/
├── nirs.py                          # Re-exports for backward compat (~40 lines)
├── _nirs/
│   ├── base.py                      # NIRSTransformer base + _validate_not_sparse() (~80 lines)
│   ├── scatter_correction.py        # MSC, EMSC (~210 lines)
│   ├── baseline_correction.py       # PyBaselineCorrection + factory-generated aliases (~240 lines)
│   ├── spectral_math.py             # Derivatives, Wavelet, Haar, LogTransform (~150 lines)
│   ├── wavelet_features.py          # WaveletFeatures, WaveletPCA, WaveletSVD (~400 lines)
│   ├── normalization.py             # AreaNormalization, SavitzkyGolay (~130 lines)
│   └── signal_conversion.py         # ReflectanceToAbsorbance (~120 lines)
```

### Key simplifications

- **Baseline alias factory**: Replace 9 classes (315 lines) with a factory function + parameter dictionary (~55 lines) = **260 lines saved**.
- **NIRSTransformer base class**: Consolidate `_reset()`, `_more_tags()`, sparse validation = **~100 lines saved**.
- **`_validate_not_sparse()`**: Single utility replacing 24 inline checks = **~35 lines saved**.
- Total savings: ~540 lines (24%).

---

## 9. visualization/pipeline_diagram.py

**Severity: HIGH** — 2,184 lines, PipelineDiagram = 2,067 lines with 28 methods.

### Key problems

- `_build_dag_from_trace()` = 322 lines, cyclomatic complexity ~25.
- `_parse_keyword_step()` = 120 lines as giant if-elif chain.
- 5 code duplication patterns (prediction filtering, branch path handling, shape fallback).

### Proposed split

```
visualization/pipeline_diagram/
├── __init__.py
├── nodes.py                         # PipelineNode dataclass (~40 lines)
├── dag_builder.py                   # TraceDAGBuilder + SourceBranchHandler (~550 lines)
├── step_parser.py                   # StepParser + dispatch dict (~200 lines)
├── shape_estimator.py               # ShapeEstimator (~100 lines)
├── layout_engine.py                 # LayoutEngine + topological sort (~120 lines)
├── renderer.py                      # NodeDrawer + EdgeDrawer + ShapeFormatter (~280 lines)
├── labels.py                        # LabelFormatter (~85 lines)
└── pipeline_diagram.py              # PipelineDiagram facade (~150 lines)
```

### Key simplifications

- Replace `_parse_keyword_step()` if-elif chain with dispatch dict.
- Split `_build_dag_from_trace()` into source branch handling, step expansion, parent resolution.
- Extract `NodeIDGenerator` to eliminate inconsistent ID formatting.

---

## 10. controllers/models/meta_model.py

**Severity: MEDIUM-HIGH** — 2,128 lines, MetaModelController = 34 methods.

### Key problems

- `_persist_meta_model()` = 163 lines with 6 responsibilities.
- Prediction store filtering duplicated 4 times.
- Probability extraction logic duplicated 2 times.
- `_handle_coverage()` uses 5-branch if-elif instead of strategy pattern.

### Proposed split

```
controllers/models/
├── meta_model.py                    # Main orchestration (~450 lines)
├── meta_model_features.py           # FeatureBuilder + PredictionExtractor (~380 lines)
├── meta_model_validators.py         # ValidationOrchestrator (~280 lines)
├── meta_model_selection.py          # SourceModelSelector (~150 lines)
├── meta_model_persistence.py        # MetaModelPersistence (~330 lines)
├── meta_model_predict.py            # PredictionModeExecutor (~300 lines)
└── meta_model_utils.py              # CoverageHandler strategies + utilities (~150 lines)
```

### Key simplifications

| Method | Current | Action |
|--------|---------|--------|
| `_persist_meta_model()` | 163 lines | Split into `_build_meta_artifact()`, `_generate_artifact_id()`, `_register_meta_artifact()`, `_record_artifact_in_trace()` |
| `_handle_coverage()` | 66 lines | Strategy pattern: CoverageHandler subclasses |
| Prediction filtering | 4 duplicates | Single `PredictionFilter` utility |
| Probability extraction | 2 duplicates | Single `_extract_prediction_values()` |

---

## 11. pipeline/execution/orchestrator.py

**Severity: HIGH** — 2,117 lines, `execute()` = 613 lines.

### Key problems

- `execute()` mixes normalization, parallel/sequential execution, store lifecycle, error handling, reporting.
- `_print_refit_report()` = 236 lines with table formatting mixed into orchestrator.
- Report generation = 486 lines (23% of file).
- Refit orchestration = 483 lines (23% of file).
- Parallel and sequential paths have overlapping logic.

### Proposed split

```
pipeline/execution/
├── orchestrator.py                  # PipelineOrchestrator facade (~300 lines)
├── dataset_normalizer.py            # DatasetNormalizer (~160 lines)
├── report_generator.py              # ReportGenerator (~490 lines)
├── refit_orchestrator.py            # RefitOrchestrator (~485 lines)
├── execution_dispatcher.py          # ExecutionDispatcher: parallel/sequential (~400 lines)
├── parallel_worker.py               # _execute_single_variant() (~135 lines)
```

### Key simplifications

| Method | Current | Action |
|--------|---------|--------|
| `execute()` | 613 lines | Delegate to `_normalize_inputs()`, `_execute_on_all_datasets()`, `_print_results_summary()`, `_complete_run()` |
| `_print_refit_report()` | 236 lines | Split into `_print_per_model_summary()`, `_print_top_cv_chains()`, `_print_cv_selection_summary()` |
| `_execute_refit_pass()` | 206 lines | Split into `_execute_single_config_refit()` vs `_execute_multi_config_refit()` |

---

## 12. data/dataset.py

**Severity: MEDIUM-HIGH** — 2,083 lines, SpectroDataset = 85+ methods, 14 functional domains.

### Key problems

- 14 logically independent domains crammed into one class.
- Repetition transformation = 480 lines of highest complexity with `_rebuild_dataset_from_sources()` = 139 lines.
- `reshape_reps_to_sources()` and `reshape_reps_to_preprocessings()` share 95% structure.
- Manager delegation partially implemented (`_feature_accessor`, `_target_accessor`, `_metadata_accessor`) but not extended.

### Proposed split

```
data/
├── dataset.py                       # SpectroDataset facade (~600 lines)
├── dataset_repetition.py            # RepetitionManager (~480 lines)
├── dataset_signal_type.py           # SignalTypeManager (~130 lines)
├── dataset_nan_tracking.py          # NaNTracker (~115 lines)
├── dataset_tags.py                  # TagManager (~160 lines)
├── dataset_aggregation.py           # AggregationConfig (~235 lines)
├── dataset_manifest.py              # DatasetManifest: metadata export, print_summary (~200 lines)
└── dataset_utils.py                 # Hashing, fold management, size properties (~100 lines)
```

### Key simplifications

| Item | Current | Action |
|------|---------|--------|
| `reshape_reps_to_*` | 2 methods, 195 lines, 95% identical | Extract `_reshape_repetitions_base()`, thin wrappers |
| `_rebuild_dataset_from_sources()` | 139 lines | Split into `_reset_features()`, `_create_indexer()`, `_rebuild_metadata()`, `_rebuild_targets()`, `_finalize_rebuild()` |
| `get_dataset_metadata()` | 120 lines | Extract per-category collectors |
| Signal type | Scattered with defensive init calls | `SignalTypeManager` with lazy init |

---

## Global Recommendations

### Cross-cutting patterns to address first

1. **JSON deserialization**: Duplicated across `workspace_store.py`, `predictions.py`, and others. Create a shared `_parse_json(value, default)` utility.

2. **Sparse matrix validation**: 24 duplicates in `nirs.py`. Single `_validate_not_sparse()`.

3. **Prediction store filtering**: Duplicated in `merge.py`, `meta_model.py`, `predictions.py`. Standardize filter builder.

4. **Branch context creation**: Duplicated in `branch.py`, `merge.py`. Extract shared utility.

### Suggested implementation order

| Priority | Files | Rationale |
|----------|-------|-----------|
| **P0** | `merge.py`, `fitter.py` | >5k lines, critical complexity, highest ROI |
| **P1** | `orchestrator.py`, `workspace_store.py` | Core execution path, high method count |
| **P2** | `branch.py`, `base_model.py`, `predictions.py` | 2.2-2.5k lines, clear domain boundaries |
| **P3** | `dataset.py`, `meta_model.py`, `pipeline_diagram.py` | Medium severity, well-contained domains |
| **P4** | `nirs.py`, `_constants.py` | Mostly mechanical refactoring, lower risk |

### Refactoring principles

- **Each extraction should be a separate commit** with no behavioral changes.
- **Maintain backward compatibility** via re-exports in `__init__.py`.
- **Run the full test suite** after each extraction.
- **Extract data classes and utilities first** (lowest risk), then domain logic.
- **Never extract and simplify in the same commit** - separate structural moves from logic changes.
