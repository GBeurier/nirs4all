# nirs4all — Pre-v1 Dead/Deprecated/Legacy/BC Removal Worklist

**Date:** 2026-06-04 · Produced by a 7-region discovery sweep + per-candidate adversarial verification (114 agents). Pending Codex review before execution.
**Directive:** pre-v1, no backward compatibility, no compat shims, no dead/deprecated/legacy code, remove BC tests. **Hard constraint: zero LIVE-behavior regression** (baseline = `debt-baseline-2026-06-04.md`, 7313 pass / 6 known fail).

## ✅ STATUS: EXECUTED (2026-06-04) — ZERO REGRESSION
Codex GO/NO-GO obtained, then executed via 1 manual batch + 5 parallel subtree agents + 1 careful metadata_partition agent (with inline branch/merge test verification).
- **121 files changed · 64 deleted · 57 modified · −17,245 LOC net.**
- Gate: `ruff` clean · `mypy` clean (396 files, 0 issues).
- Full suite: **7099 passed · 6 failed · 129 skipped** — the 6 failures are the **identical pre-existing baseline** (4 AOM `test_estimator_type` + `generator_grid_n_components_scale` + `explain_path`). **No new failure.** (passed 7313→7099 = removed dead tests only.)
- **Deferred / kept (Codex overrides):** `PREDEFINED_COMPONENTS` KEPT (imported by nirs4all-lab — coordinate before removing); refit cluster RELABEL-ONLY (live via orchestrator path, NOT deleted); `load_parquet/excel/matlab` convenience wrappers left (symmetry decision — only `load_numpy` removed); `csv_loader_new` module rename left (naming-debt, not a removal).

## Summary
75 deduplicated findings: **50 delete_safe · 15 relabel · 9 keep · 1 consolidate**. ~**7,000–7,500 LOC** removable (actual deletions incl. dead test suites: ~17k LOC).

Biggest wins: `synthesis/reconstruction/` (~4,353 LOC, 9 files + test) · 18 dead TF `legacy/` modules (~3,500 LOC) · `data/partition/`+`data/selection/` Phase-3/4 layer (~2,700 LOC) · `pytorch/generic.py` (783 LOC, fully commented) · `data/aggregation/` (~547 LOC).

---

## 2. DELETE NOW (delete_safe)

### operators — TF legacy dead modules (18, zero caller edits; `legacy/__init__.py __all__=[]`, `generic.py` imports only the 4 live ones)
ternausnet · dense_inception_unet · unet_plus_plus · **resnet_1d_cnn** (NOT resnet_v2) · **seresnext_1d_cnn** (NOT se_resnet) · resnext_1d_cnn · sedunet · unet_plus · unet3_plus · unet · multires_unet · pspnet · ibaunet · fpn · ensembled_unet · bcdunet · albunet · **vgg_2d_cnn** (NOT vgg_1d_cnn). All under `operators/models/tensorflow/legacy/`. **KEEP `legacy/__init__.py`.**

### operators — other
- `operators/models/pytorch/generic.py` (1-783, fully commented) → delete file + remove `from .generic import *` in `pytorch/__init__.py:6`.
- `operators/models/sklearn/fckpls.py:862-863` `FractionalPLS` alias → remove + `__all__` in `models/sklearn/__init__.py:47,89` & `models/__init__.py:69,125`; delete `test_alias_fractional_pls`; fix `docs/source/reference/models.md:92`.
- `operators/transforms/nirs.py:2162-2185` `asls_baseline` BC wrapper → remove + `transforms/__init__.py:68,202`. Do NOT touch `_asls_baseline`/`ASLSBaseline`.
- `operators/models/tensorflow/nicon.py:889-905` `decon_layer_classification` alias → remove (no `__all__`).

### controllers
- `controllers/flow/{condition,scope,sequential}.py` (0-byte empty modules) → delete.
- `controllers/models/utilities.py:203-204` `ModelUtils` alias → remove (importers use `ModelControllerUtils as ModelUtils`).
- `controllers/models/utilities.py:133-164` `validate_loss_compatibility` → remove (dead).
- `utils/model_utils.py:1-22` BC shim module → delete whole file.
- `controllers/flow/dummy.py:150-162` legacy dict-context `elif` → remove block (falls through to `else`).
- **`metadata_partition` cluster** (branch_validator.py:1015-1027,951-953,222-239,861-862; merge.py:245-247,300-305) → remove dead reads only, keep `sample_partition` + `BranchType.METADATA_PARTITIONER`. **Atomic; see Risk 2.**

### pipeline
- `store_queries.py:310-337` `build_aggregated_query` shim → delete; repoint `workspace_store.py:1850` to `build_chain_summary_query`, drop import L106. Keep public method sig (studio).
- `store_queries.py:370-404` `build_top_aggregated_query` shim → delete; repoint `workspace_store.py:1926` to `build_top_chains_query`, drop import L110. Keep `query_top_aggregated_predictions` sig (studio).
- `runner.py:441-448` `runs_dir` alias → delete; callers `runner.py:629`, `retrainer.py:332` → `workspace_path`. Not `PredictionResolver.runs_dir`.
- `artifacts/artifact_registry.py:977-979` legacy v1 manifest `list` branch → remove (if/else collapse). Not the loader copy.
- **Batch-7 refit chain — DO NOT DELETE blindly (see Risk 1).**

### data
- `data/aggregation/` (aggregator.py + __init__) → delete dir (not re-exported). Not `config.py _aggregates`.
- `data/partition/` (partition_assigner 797 + __init__) → delete dir; edit `data/__init__.py:27-32,112-115`; remove `PartitionConfig.to_assigner_spec` (schema/config.py:449-499); delete `tests/unit/data/partition/`.
- `data/selection/` (column_selector/row_selector/role_assigner/sample_linker/sampling + __init__) → delete dir; edit `data/__init__.py:49-59,103-111`; delete `tests/unit/data/selection/`. **column_selector↔role_assigner atomic (Risk 5).**
- `data/schema/config.py:1542-1549` `is_legacy_format` + `:1551-1553` `is_files_format` → remove; drop `test_schema.py:107,108` asserts.
- `data/loaders/__init__.py:76-77,185` `load_csv_new` alias → remove + `__all__`.
- `data/loaders/loader.py` `create_synthetic_dataset` → remove; delete `test_loader_random_state.py`.
- `data/loaders/numpy_loader.py:255-291` `load_numpy` BC wrapper → remove + `loaders/__init__.py:73,186` + test edits. **Decide parquet/excel/matlab symmetry (Risk 6).**
- `data/_features/feature_constants.py:32,55` `HeaderUnitType` alias → inline + `_features/__init__.py:8,26`.

### synthesis
- `synthesis/reconstruction/` (9 files ~4,353 LOC) → delete dir; delete `tests/unit/test_reconstruction.py`. Not re-exported.
- `synthesis/__init__.py:467-469,624` `get_nir_zone` alias → remove + `__all__`.
- `synthesis/__init__.py:472-476` `_get_predefined_components` shim → remove (not in `__all__`).
- `synthesis/__init__.py:478-507,518` `PREDEFINED_COMPONENTS`/`_PredefinedComponentsProxy` → remove + `__all__` + `test_components.py`. **nirs4all-lab imports it — coordinate (Risk 3).**

### core_misc
- `utils/backend.py:190-202` `check_backend_available` alias → remove + `utils/__init__.py:21,62`.
- `visualization/display.py:117-119` `figure_list` alias → remove (not re-exported).
- `cli/__init__.py:9-16` commented imports/`__all__` → remove.
- `cli/installation_test.py:175-181` commented block → remove.
- `core/validation.py` (0-byte) → delete.
- `visualization/charts/heatmap.py:358,390-392,434,455-458` + `visualization/predictions.py:865,903-905,981` `sort_by_value` deprecated param → remove (pair together).
- `optimization/optuna.py:1166-1171,1555-1562,1598-1605` 3 flat-finetune "Legacy support" blocks → remove (keep `.get('model_params',{})` line above each).
- `config/config.py` → phantom, NO-OP (verify `git ls-files`).

### tests (dead runners + placeholder stubs)
- `tests/run_tests.py`, `tests/run_runner_tests.py`, `tests/integration/run_integration_tests.py` → delete (hand-rolled runners).
- Placeholder stub files (delete whole): `tests/unit/operators/augmentation/test_augmentation.py`, `.../models/test_tensorflow.py`, `.../transforms/test_nirs.py`, `tests/unit/data/loaders/test_loader.py`, `tests/unit/pipeline/test_runner.py`, `tests/unit/pipeline/test_config.py`, `tests/unit/pipeline/config/test_generator.py`.
- `tests/unit/operators/transforms/test_signal.py:519-525` placeholder trailing class only (keep 1-517).

---

## 3. RELABEL (comment/docstring only — NO deletion, behavior identical)
merge.py:539-545,609,695-697 · stacking_refit.py:1369-1373 `_expand_branch_steps` docstring (only if Batch-7 keeps it) · config/context.py:75,245 `branch_id` docstrings · bundle/generator.py:20-26,35,246,253-255 · steps/step_runner.py:189,214 · artifacts/artifact_loader.py:645-648 · data/loaders/csv_loader_new.py:529,541 · data/dataset.py:422 + _dataset/feature_accessor.py:576 · data/config.py:493-496 · synthesis/wavenumber.py:68-69 · visualization/display.py:128 · api/predict.py:12,80,180,246 · api/result.py:352,386,413,466 · tests/fixtures/data_generators.py:4-55 (banner + import-time DeprecationWarning) · READMEs stale paths.

## 4. CONSOLIDATE
- `artifacts/utils.py:209` `get_binaries_path` — drop dead `dataset` param; update `artifact_registry.py:246`, `artifact_loader.py:182`, `test_utils.py:324`. Keep constructors' own `dataset`.
- (optional) `visualization/display.py` `finalize_figures` → inline into `keep_or_close_figures`.

## 5. KEEP (rejected — live caller)
`legacy/__init__.py` (package marker) · `sklearn/nlpls.py:1140-1141` NLPLS/KPLS (studio getattr + frontend JSON — 0.9.x contract) · `splitters/split.py:351-374` `_normalize_group_alias` (shipped examples) · `branch.py:215-224` legacy `"by"` guard (integration tests) · `store_queries.py:661-677` min/max/avg score aliases (studio /top default) · `stacking_refit.py:1409-1418` `_model_class_name` (reachable from orchestrator) · `migration.py:495-522` `migrate_duckdb_to_sqlite` (live in WorkspaceStore.__init__) · `generator.py:1385-1386` `_add_noise` (live fallback) · `predictions.py:942-948` `aggregation` kwarg (4 CI examples) · group-alias tests · `csv_loader_new` import (sole live CSV loader).

---

## 6. Sequencing (one Codex review per batch; run verify after each)
1. **Leaf dead files, zero caller edits** — 18 TF legacy + `core/validation.py` + 3 flow stubs + 3 dead runners + 8 placeholder tests + cli comment blocks. Verify: `ruff check nirs4all/ tests/ && pytest tests/unit -q`.
2. **pytorch generic + model_utils shim + simple aliases** — generic.py, model_utils.py, backend alias, figure_list, utilities aliases, load_csv_new. Verify: `ruff + mypy nirs4all/utils … + pytest tests/unit/operators tests/unit/data/loaders`.
3. **synthesis dead** — reconstruction/ + synthesis aliases/shims. **Flag nirs4all-lab (Risk 3).** Verify: `ruff + pytest tests/unit/data/synthetic`.
4. **data Phase 3/4 layer** — aggregation/ + partition/ + selection/ + is_legacy/is_files_format. Verify: `ruff + mypy nirs4all/data + pytest tests/unit/data`.
5. **data loaders + features** — create_synthetic_dataset, load_numpy, HeaderUnitType. Verify: as data.
6. **pipeline storage shims** — query builders, manifest branch, runs_dir, get_binaries_path consolidate. Verify: `pytest tests/unit/pipeline/storage`.
7. **refit dead chain + metadata_partition — HIGH RISK.** Treat refit as KEEP+RELABEL unless golden proves dead (Risk 1). Verify: refit + branch integration tests + `cd examples && ./run.sh`.
8. **viz + optuna + sklearn/transforms aliases** — sort_by_value pair, optuna blocks, FractionalPLS, asls_baseline, decon alias. Verify: viz/optimization/operators tests.
9. **RELABEL-only** — section 3 comment/docstring edits. Verify: `ruff + pytest tests/unit + ./run.sh`.

## 7. Risk callouts (carried verbatim into execution)
1. **Refit cluster (Batch 7) — highest risk, internal contradiction.** `_execute_per_model_competing_refit` + `_select_best_per_model` flagged delete_safe, but adjacent `_expand_branch_steps` (relabel, live via `test_branch_artifacts.py::test_named_branches_produce_predictions`, `test_branch_roundtrip.py`, `test_branch_predict_mode.py`) and `_model_class_name` (keep, reachable `orchestrator.py:1542`) are in the SAME chain. **Default: KEEP+RELABEL, do not delete** unless the named-branch-without-merge path is proven dead by golden + integration tests + `./run.sh`.
2. **metadata_partition** — discoverer named 2 of ~6 sites; remove all 6 as one unit, keep sample_partition + enum.
3. **PREDEFINED_COMPONENTS** — imported by `nirs4all-lab/synthesis/synthetic/__init__.py:52-59`; flag/coordinate, don't break.
4. **Studio-check** — re-grep `nirs4all-studio/api` before each `__all__`-touching delete (FractionalPLS, load_csv_new, load_numpy, HeaderUnitType, get_nir_zone, check_backend_available, asls_baseline).
5. **selection/partition coupling** — column_selector↔role_assigner atomic; whole packages + bridge + tests together.
6. **load_numpy symmetry** — decide: just load_numpy or all of parquet/excel/matlab too.
7. **csv_loader_new** — rename is naming-debt, NOT a removal; keep.
8. **config/config.py** — phantom, no-op.
9. Docs artifacts (`docs/_build`, generated API, egg-info) regenerate — non-blocking.
