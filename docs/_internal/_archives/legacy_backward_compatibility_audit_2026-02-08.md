# Legacy / Backward-Compatibility / Deprecated / Dead-Code Audit

Date: 2026-02-08 (original) — **Reviewed & corrected: 2026-02-08**
Scope: full repository scan (`nirs4all/`, `tests/`, `examples/`, `docs/source/`) with focus on explicit legacy/deprecated/backward-compatibility paths and dead references.

## Action Legend

- `REMOVE`: delete code/docs/tests/examples that only exist for legacy/backward compatibility or are dead.
- `UPDATE`: keep component but rewrite to current syntax/behavior and remove legacy/deprecated branches.
- `KEEP`: item was incorrectly flagged — no action needed.

## Review Corrections (Critical)

### Correction 1: `validators.py` — Mixed-format warning is wrong, not the validator itself

The MIXED_FORMAT warning (line 189) says *"Legacy format will take precedence"* but the normalizer (`normalizer.py:75-81`) actually checks parsers in order: `VariationsParser → SourcesParser → FilesParser → FolderParser → LegacyParser`. **New formats take precedence, not legacy.** The warning message is factually incorrect and must be fixed.

The `train_x/test_x` format is the **canonical internal format** — all new formats (sources, variations) get converted TO it via `to_legacy_format()` / `variations_to_legacy_format()`. The naming is misleading but the format itself is current, not legacy.

The tests in `test_schema.py` are **NOT wrong** — they correctly use `train_x` as the current canonical format. Only `test_mixed_format_warning` needs its assertion updated to match the corrected message.

### Correction 2: `tensorflow/legacy/` — Must NOT be removed

The `legacy/` directory contains **classic neural network architectures** (ResNet, VGG, Inception, U-Net, SE-ResNet, etc.) adapted for 1D spectroscopy. "Legacy" here means "historically established architectures" not "deprecated nirs4all code." Evidence:
- Zero deprecation warnings or backward-compat shims in any file
- Actively imported in `generic.py:35-38` (Inception, ResNetv2, SEResNet, VGG)
- Each file implements peer-reviewed architectures with academic citations
- Well-maintained, configurable, using current `tf.keras` API

**Action changed from REMOVE to KEEP. The audit entry was wrong.**

---

## Incremental Execution Plan

Each batch is self-contained: apply changes, run tests, verify only expected failures occur, fix/remove those tests, then proceed to next batch.

---

### Batch 1: Validator fix + dead code + deprecated wrappers

**Scope**: Fix the wrong validator warning, remove dead code and explicitly deprecated wrapper modules.

**Risk**: Low — these are isolated changes with no downstream dependencies.

#### Source code changes

| Location | Action | Detail |
|---|---|---|
| `nirs4all/data/schema/validation/validators.py` | UPDATE | Fix MIXED_FORMAT warning: remove "Legacy format will take precedence" — replace with neutral "Both train_x/test_x and files/sources detected. Use one format consistently." |
| `nirs4all/pipeline/steps/router.py` (`route_from_raw`) | REMOVE | Dead code: no callsites found in repo. |
| `nirs4all/visualization/branch_diagram.py` | REMOVE | Explicitly deprecated wrapper module; kept only for backward compatibility. |
| `nirs4all/visualization/__init__.py` | UPDATE | Remove backward-compatible aliases to deleted `branch_diagram` module. |
| `nirs4all/visualization/pipeline_diagram.py` | UPDATE | Remove backward-compat aliases (`BranchDiagram`, `plot_branch_diagram`). |

#### Test changes

| Location | Action | Detail |
|---|---|---|
| `tests/unit/data/test_schema.py` (`test_mixed_format_warning`) | UPDATE | Update assertion to match corrected MIXED_FORMAT message. |

#### Verification

```bash
pytest tests/unit/data/test_schema.py -v          # All pass after message update
pytest tests/unit/ -v                              # No unexpected failures
pytest tests/integration/ -v                       # No unexpected failures
```

---

### Batch 2: Legacy key normalization + LegacyParser

**Scope**: Remove the `LegacyParser` (flexible key normalization: `x_train → train_x`, etc.) and its integration points. The canonical `train_x/test_x` format remains — only alternate key spellings are removed.

**Risk**: Medium — the normalizer and config_parser depend on `normalize_config_keys`. Must verify no YAML/JSON configs use alternate key spellings.

#### Source code changes

| Location | Action | Detail |
|---|---|---|
| `nirs4all/data/parsers/legacy_parser.py` | REMOVE | Remove `KEY_MAPPINGS`, `normalize_config_keys()`, and `LegacyParser` class. Move minimal `_infer_dataset_name` logic into normalizer if needed. |
| `nirs4all/data/parsers/__init__.py` | UPDATE | Remove `LegacyParser` export. |
| `nirs4all/data/parsers/normalizer.py` | UPDATE | Remove `LegacyParser` from default parser chain (line 80). Remove `normalize_config_keys()` calls (lines 198, 261). Dict configs with `train_x` keys will be handled by remaining parsers or direct dict passthrough. |
| `nirs4all/data/config_parser.py` | UPDATE | Remove imports/uses of `normalize_config_keys` and legacy key normalization. |

#### Test changes

| Location | Action | Detail |
|---|---|---|
| `tests/unit/data/test_parsers.py` | UPDATE | Remove `LegacyParser` test suite and legacy key-normalization expectations. Keep tests for canonical key format. |

#### Verification

```bash
pytest tests/unit/data/test_parsers.py -v          # Legacy parser tests removed, rest pass
pytest tests/unit/data/ -v                         # No unexpected failures
pytest tests/integration/ -v                       # No unexpected failures
# Also verify examples still work:
cd examples && ./run.sh -q
```

---

### Batch 3: Deprecated aliases and parameters (data layer)

**Scope**: Remove deprecated parameter aliases in the data layer: `aggregate` alias in `DatasetConfigs`, deprecated prediction aliases, `labelizer` param, `float_headers` compat.

**Risk**: Medium — must check that examples and integration tests don't rely on deprecated aliases.

#### Source code changes

| Location | Action | Detail |
|---|---|---|
| `nirs4all/data/config.py` (`DatasetConfigs`) | UPDATE | Remove deprecated `aggregate` alias; only accept canonical parameter name. Remove deprecation warnings — just remove the alias. |
| `nirs4all/data/predictions.py` | UPDATE | Remove deprecated aliases (`aggregate*`, `best_per_model`, legacy metric kwarg path). |
| `nirs4all/data/targets.py` | UPDATE | Remove unused `labelizer` parameter. |
| `nirs4all/data/schema/config.py` | UPDATE | Remove `is_legacy_format()` method (misleading name — `train_x` is current). Remove `to_legacy_format()` and `variations_to_legacy_format()` — rename to canonical internal methods if they serve a real purpose, or inline. |
| `nirs4all/data/schema/__init__.py` | UPDATE | Update module docs to remove "legacy format is fully supported" language. |
| `nirs4all/data/loaders/csv_loader_new.py` | REMOVE | Backward-compat wrapper for old loader API. |
| `nirs4all/data/loaders/loader.py` | UPDATE | Remove routing through compatibility `load_csv` API. |

#### Test changes

| Location | Action | Detail |
|---|---|---|
| `tests/unit/data/test_config.py` | UPDATE | Remove tests for deprecated `aggregate` alias behavior. |
| `tests/unit/data/test_prediction_ranking.py` | UPDATE | Remove tests for deprecated ranking/aggregation aliases. |
| `tests/unit/data/test_dataset_wavelength_conversion.py` | UPDATE | Remove coverage for legacy `float_headers` compatibility API. |
| `tests/unit/data/test_sources_parser.py` | UPDATE | Remove tests for `to_legacy_format` conversion paths. |
| `tests/unit/data/test_variations_parser.py` | UPDATE | Remove tests for `variations_to_legacy_format` conversion paths. |
| `tests/unit/data/test_schema.py` | UPDATE | Remove `is_legacy_format` assertions. |

#### Verification

```bash
pytest tests/unit/data/ -v                         # All pass after updates
pytest tests/integration/pipeline/test_aggregation_integration.py -v  # Check aggregate usage
pytest tests/integration/ -v                       # No unexpected failures
```

---

### Batch 4: Pipeline `size` keyword + `source_branch` + `force_group` compat

**Scope**: Remove deprecated pipeline keywords and their compatibility shims.

**Risk**: Medium-High — these are used in pipeline configuration and controllers. Must verify no examples use the old keywords.

#### Source code changes

| Location | Action | Detail |
|---|---|---|
| `nirs4all/pipeline/config/_generator/strategies/or_strategy.py` | UPDATE | Remove legacy `size` alias for `pick`. |
| `nirs4all/pipeline/config/generator.py` | UPDATE | Remove `size` from public doc/type surface. |
| `nirs4all/controllers/data/sample_augmentation.py` | UPDATE | Remove `size → pick` conversion. |
| `nirs4all/controllers/data/feature_augmentation.py` | UPDATE | Remove `size → pick` conversion and backward-compat behavior branch. |
| `nirs4all/controllers/data/concat_transform.py` | UPDATE | Remove `size → pick` conversion. |
| `nirs4all/controllers/data/branch.py` | UPDATE | Remove `source_branch` compatibility matching/conversion and legacy pattern support. |
| `nirs4all/controllers/splitters/split.py` | UPDATE | Remove deprecated `force_group` support. |
| `nirs4all/pipeline/config/context.py` | UPDATE | Remove deprecated `branch_id` compatibility. |
| `nirs4all/operators/data/merge.py` | UPDATE | Remove legacy merge syntax normalization. |
| `nirs4all/controllers/data/merge.py` | UPDATE | Remove legacy merge-format branches and legacy info fields. |

#### Test changes

| Location | Action | Detail |
|---|---|---|
| `tests/unit/pipeline/config/test_generator_pick_arrange.py` | UPDATE | Remove legacy `size` keyword compatibility tests. |
| `tests/unit/controllers/data/test_branch_separation.py` | UPDATE | Remove `source_branch` compatibility conversion tests. |
| `tests/unit/controllers/data/test_feature_augmentation.py` | UPDATE | Remove legacy `size` conversion behavior tests. |
| `tests/unit/data/test_group_split_validation.py` | UPDATE | Remove `force_group` warnings/behavior compatibility tests. |
| `tests/integration/pipeline/test_new_branch_modes.py` | UPDATE | Remove backward-compat tests for `source_branch` and legacy patterns. |
| `tests/integration/pipeline/test_merge_per_branch.py` | UPDATE | Remove `test_legacy_syntax_still_works`. |
| `tests/integration/pipeline/test_force_group_integration.py` | UPDATE | Remove `force_group` compatibility suite. |
| `tests/integration/pipeline/test_groupsplit_integration.py` | UPDATE | Remove `force_group` coverage. |
| `tests/integration/pipeline/test_flexible_inputs_integration.py` | UPDATE | Remove backward-compat test for traditional config objects. |

#### Verification

```bash
pytest tests/unit/pipeline/config/ -v              # Pass after size removal
pytest tests/unit/controllers/ -v                  # Pass after branch/merge cleanup
pytest tests/integration/pipeline/ -v              # Pass after compat test removal
cd examples && ./run.sh -q                         # Verify no example uses old keywords
```

---

### Batch 5: Pipeline storage + bundle legacy paths

**Scope**: Remove legacy fold-key fallbacks, legacy artifact naming, legacy bundle loading paths.

**Risk**: Medium — affects stored artifacts. Old workspace runs may become unreadable. Acceptable if this is a breaking version.

#### Source code changes

| Location | Action | Detail |
|---|---|---|
| `nirs4all/pipeline/trace/execution_trace.py` | UPDATE | Remove `fold_key_candidates()` legacy fold key support — keep only canonical format. |
| `nirs4all/pipeline/storage/workspace_store.py` | UPDATE | Remove canonical/legacy fold-key fallback and legacy CV averaging replay path. |
| `nirs4all/pipeline/storage/artifacts/artifact_loader.py` | UPDATE | Remove legacy shims: `get_step_binaries`, `has_binaries_for_step`, compatibility naming lookup. |
| `nirs4all/pipeline/storage/artifacts/utils.py` | UPDATE | Remove legacy artifact filename parsing and compatibility arguments. |
| `nirs4all/pipeline/storage/artifacts/artifact_registry.py` | UPDATE | Remove pre-generated ID allowance and legacy v1 list manifest imports. |
| `nirs4all/pipeline/bundle/generator.py` | UPDATE | Remove resolver-based legacy export path. |
| `nirs4all/pipeline/bundle/loader.py` | UPDATE | Remove legacy refit key fallback logic. |
| `nirs4all/pipeline/steps/step_runner.py` | UPDATE | Remove legacy `(context, artifacts)` step return format handling. |
| `nirs4all/pipeline/runner.py` (`get_runs_dir`) | UPDATE | Remove explicit legacy compatibility method. |
| `nirs4all/pipeline/execution/orchestrator.py` | UPDATE | Remove legacy runs-dir compatibility branch. |
| `nirs4all/api/predict.py` | UPDATE | Remove legacy model-based prediction path — keep only store-based path. |
| `nirs4all/api/result.py` (`RunResult.export`) | UPDATE | Remove legacy resolver export fallback. |

#### Test changes

| Location | Action | Detail |
|---|---|---|
| `tests/unit/pipeline/storage/artifacts/test_artifact_loader.py` | UPDATE | Remove `TestArtifactLoaderLegacyCompatibility`. |
| `tests/unit/pipeline/storage/artifacts/test_utils.py` | UPDATE | Remove legacy artifact naming/compat param tests. |
| `tests/unit/pipeline/storage/test_cross_run_cache.py` | UPDATE | Remove artifact loading backward-compat coverage. |
| `tests/unit/test_oof_accumulation.py` | UPDATE | Remove legacy OOF collection behavior note/path. |
| `tests/integration/storage/test_fold_key_contract.py` | UPDATE | Remove legacy fold key normalization assertions. |
| `tests/integration/pipeline/test_branch_predict_mode.py` | UPDATE | Remove legacy manifest compatibility expectations. |

#### Verification

```bash
pytest tests/unit/pipeline/ -v                     # Pass after storage cleanup
pytest tests/integration/storage/ -v               # Pass after fold-key cleanup
pytest tests/integration/pipeline/ -v              # Pass after predict/bundle cleanup
```

---

### Batch 6: Remaining aliases + controller/model compat

**Scope**: Remove remaining backward-compat aliases scattered across controllers, models, utils, visualization, and other modules.

**Risk**: Low-Medium — these are isolated aliases and shims.

#### Source code changes

| Location | Action | Detail |
|---|---|---|
| `nirs4all/controllers/models/components/identifier_generator.py` | UPDATE | Remove deprecated `helper` parameter. |
| `nirs4all/controllers/models/utilities.py` | UPDATE | Remove `ModelUtils` alias. |
| `nirs4all/controllers/models/stacking/branch_validator.py` | UPDATE | Remove `LEGACY` detection paths for old controller patterns. |
| `nirs4all/utils/backend.py` | UPDATE | Remove legacy compatibility constants/wrapper. |
| `nirs4all/utils/__init__.py` | UPDATE | Remove legacy backend constant exports. |
| `nirs4all/visualization/charts/heatmap.py` | UPDATE | Remove deprecated `sort_by_value` compatibility path. |
| `nirs4all/visualization/charts/base.py` | UPDATE | Remove legacy chart fallback path (predictions-only). |
| `nirs4all/visualization/predictions.py` | UPDATE | Remove backward-compat mapping for old `aggregation` kwarg; remove dead commented alias block. |
| `nirs4all/operators/transforms/nirs.py` | UPDATE | Remove `asls_baseline` alias function. |
| `nirs4all/synthesis/__init__.py` | UPDATE | Remove `get_nir_zone` backward-compat alias export. |
| `nirs4all/optimization/optuna.py` | UPDATE | Remove legacy parameter-name compatibility paths. |
| `nirs4all/analysis/transfer_utils.py` | UPDATE | Remove string-based legacy pipeline spec support. |
| `nirs4all/analysis/selector.py` | UPDATE | Remove legacy string preprocessing support. |
| `nirs4all/api/run.py` | UPDATE | Remove `PipelineConfigs`/`DatasetConfigs` backward-compat input handling. |

#### Test changes

| Location | Action | Detail |
|---|---|---|
| `tests/fixtures/data_generators.py` | REMOVE | File is explicitly marked deprecated. |
| `tests/conftest.py` | UPDATE | Remove `legacy_data_generator`, `legacy_test_data_manager` deprecated fixtures. |
| `tests/integration/pipeline/test_aggregation_integration.py` | UPDATE | Replace deprecated `aggregate` dataset parameter usage with canonical parameter. |

#### Verification

```bash
pytest tests/unit/ -v                              # Full unit pass
pytest tests/integration/ -v                       # Full integration pass
```

---

### Batch 7: Legacy examples + stale references

**Scope**: Remove legacy examples, sample configs, and fix broken file references in run scripts.

**Risk**: Low — examples are independent of library code.

#### Changes

| Location | Action | Detail |
|---|---|---|
| `examples/legacy/` (entire directory) | REMOVE | Legacy-only training/migration content. |
| `examples/reference/R04_legacy_api.py` | REMOVE | Dedicated legacy API tutorial. |
| `examples/sample_configs/datasets/C01_legacy_separate.yaml` | REMOVE | Pure legacy dataset schema fixture. |
| `examples/run.sh` | UPDATE | Remove `legacy` category, remove references to 9 missing files. |
| `examples/run.ps1` | UPDATE | Same fixes as `run.sh`. |
| `examples/run_ci_examples.sh` | UPDATE | Remove `reference/R04_legacy_api.py` entry. |
| `examples/run_ci_examples.ps1` | UPDATE | Remove `reference/R04_legacy_api.py` entry. |
| `examples/README.md` | UPDATE | Remove `legacy/` folder documentation. |
| `examples/sample_configs/datasets/README.md` | UPDATE | Remove `C01_legacy_separate` references. |
| `examples/scripts/generate_test_datasets.py` | UPDATE | Remove legacy structure generation (`structure="legacy"`, `_export_legacy`). |
| `examples/pipeline_samples/README.md` | UPDATE | Remove `source_branch` pattern documentation. |
| `examples/user/02_data_handling/U01_flexible_inputs.py` | UPDATE | Remove "traditional" config-object path if classic API removed. |

#### Verification

```bash
cd examples && ./run.sh -q                         # All examples pass
cd examples && ./run.sh -c reference               # Reference examples pass
```

---

### Batch 8: Documentation cleanup

**Scope**: Update all documentation to reflect current-only syntax. No code changes.

**Risk**: None — docs only.

#### Changes

| Location | Action | Detail |
|---|---|---|
| `docs/source/user_guide/troubleshooting/migration.md` | UPDATE | Remove claims that legacy/classic API is fully supported. |
| `docs/source/api/nirs4all.data.parsers.rst` | UPDATE | Remove legacy parser from toctree. |
| `docs/source/api/nirs4all.data.parsers.legacy_parser.rst` | REMOVE | Legacy parser API page. |
| `docs/source/api/nirs4all.visualization.rst` | UPDATE | Remove deprecated `branch_diagram` from toctree. |
| `docs/source/api/nirs4all.visualization.branch_diagram.rst` | REMOVE | API page for deleted module. |
| `docs/source/api/module_api.md` | UPDATE | Remove `PipelineConfigs`/`DatasetConfigs` as backward-compat inputs. |
| `docs/source/examples/index.md` | UPDATE | Remove links to `R04_legacy_api.py`. |
| `docs/source/reference/combination_generator.md` | UPDATE | Remove `size` keyword documentation. |
| `docs/source/reference/generator_keywords.md` | UPDATE | Remove `size` as generator modifier. |
| `docs/source/reference/pipeline_syntax.md` | UPDATE | Remove `size` and `source_branch` syntax. |
| `docs/source/user_guide/pipelines/writing_pipelines.md` | UPDATE | Remove `size` syntax sections. |
| `docs/source/user_guide/pipelines/force_group_splitting.md` | REMOVE | Dedicated guide for removed `force_group` feature. |
| `docs/source/user_guide/pipelines/index.md` | UPDATE | Remove force-group page link and `source_branch` mentions. |
| `docs/source/user_guide/pipelines/multi_source.md` | UPDATE | Remove `source_branch` deprecated behavior docs. |
| `docs/source/user_guide/pipelines/merging.md` | UPDATE | Remove `source_branch`-based flow docs. |
| `docs/source/user_guide/data/loading_data.md` | UPDATE | Remove `source_branch` and deprecated `aggregate` examples. |
| `docs/source/user_guide/data/aggregation.md` | UPDATE | Remove deprecated `aggregate` argument and old analyzer aliases. |
| `docs/source/reference/configuration.md` | UPDATE | Remove deprecated `aggregate` parameter docs. |
| `docs/source/user_guide/visualization/prediction_charts.md` | UPDATE | Remove "old parameter names still work" statements. |
| `docs/source/examples/user/data_handling.md` | UPDATE | Remove `source_branch` pipeline examples. |
| `docs/source/examples/developer.md` | UPDATE | Remove `source_branch` examples and legacy pipeline flow references. |
| `docs/source/ai_onboarding.md` | UPDATE | Remove `source_branch` from onboarding tables/examples. |
| `docs/source/developer/controllers.md` | UPDATE | Remove `SourceBranchController` for `source_branch` keyword. |
| `tests/integration/README.md` | UPDATE | Remove backward compatibility focus description. |

#### TensorFlow docs — KEEP (not legacy nirs4all code)

| Location | Action | Detail |
|---|---|---|
| `docs/source/api/nirs4all.operators.models.tensorflow.rst` | KEEP | TF legacy package contains classic NN architectures, not deprecated code. |
| `docs/source/api/nirs4all.operators.models.tensorflow.legacy.rst` + 23 files | KEEP | API docs for active classic NN architectures. Consider renaming `legacy` → `classic` in a future rename batch. |

#### Verification

```bash
cd docs && make html                               # Docs build without warnings
```

---

## Items NOT in scope (KEEP as-is)

| Location | Reason |
|---|---|
| `nirs4all/operators/models/tensorflow/legacy/` | Classic NN architectures (ResNet, VGG, Inception, U-Net, SE-ResNet). "Legacy" = historically established, NOT deprecated nirs4all code. Actively imported in `generic.py`. |
| `nirs4all/operators/models/tensorflow/generic.py` | Active code importing classic architectures — not a backward-compat shim. |
| `nirs4all/operators/models/tensorflow/nicon.py` | Review separately — the "backward compat alias" claim needs verification in context of the classic NN package. |
| `train_x/test_x` format in `DatasetConfigSchema` | This is the **canonical internal format**, not legacy. All new formats convert TO it. The naming (`is_legacy_format`, `to_legacy_format`) is misleading but the format itself is current. Consider renaming methods in a future cleanup. |

---

## Cross-Cutting Dead / Stale References

| Location | Action | Reason |
|---|---|---|
| `examples/run.sh:166,176,190,192,193,194,197,205,206` | UPDATE (Batch 7) | References files that do not exist anymore. |
| `examples/run.ps1:177,187,201,203,204,205,208,216,217` | UPDATE (Batch 7) | Same broken missing-file references. |
