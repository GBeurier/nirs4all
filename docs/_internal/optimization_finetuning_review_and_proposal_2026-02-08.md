# nirs4all Optimization and Finetuning Review and Proposal

Date: 2026-02-08
Author: Codex (code audit driven)
Scope: `nirs4all` library finetuning and optimization stack, with emphasis on `finetune_params` and Optuna integration.
Repository areas audited:
- `nirs4all/nirs4all/optimization/optuna.py`
- `nirs4all/nirs4all/controllers/models/base_model.py`
- `nirs4all/nirs4all/controllers/models/sklearn_model.py`
- `nirs4all/nirs4all/controllers/models/tensorflow_model.py`
- `nirs4all/nirs4all/controllers/models/torch_model.py`
- `nirs4all/nirs4all/controllers/models/jax_model.py`
- `nirs4all/nirs4all/controllers/models/meta_model.py`
- `nirs4all/nirs4all/operators/models/meta.py`
- `nirs4all/nirs4all/pipeline/config/component_serialization.py`
- `nirs4all/examples/user/04_models/U02_hyperparameter_tuning.py`
- `nirs4all/examples/pipeline_samples/08_complex_finetune.json`
- `nirs4all/examples/reference/R01_pipeline_syntax.py`
- `nirs4all/docs/source/examples/user/models.md`
- `nirs4all/tests/integration/pipeline/test_finetune_integration.py`

## 1. Executive Summary

This review confirms the user concern.
Optimization in `nirs4all` is currently a partial integration around Optuna, not a full optimization subsystem.
The implementation has solid foundations for parameter sampling formats.
The implementation does not fully support advertised search methods or tuning semantics.
There are concrete functional gaps.
There are also correctness bugs that can invalidate optimization outcomes.

Top conclusions:
- The library currently implements one optimization backend: Optuna.
- The Optuna wrapper is centralized but narrow in scope.
- Search strategy values documented in examples are broader than implemented behavior.
- Training-parameter optimization (`finetune_params.train_params` ranges) is not actually supported.
- A critical control-flow bug prevents tuned parameters from being applied in final training in the standard finetune path.
- Evaluation semantics differ by framework and are inconsistent.
- Sklearn optimization objective currently uses `cross_val_score` on validation data only, causing double-fit inefficiency and semantic mismatch.
- Meta-model finetuning appears broken when using `model__` prefixed parameter names from `finetune_space`.
- Observability, trial persistence, reproducibility controls, and pruning support are limited.

Strategic recommendation:
Treat optimization as a first-class subsystem with clear contracts.
Split concerns into:
- search-space parsing,
- trial execution,
- evaluation protocol,
- framework adapters,
- study lifecycle,
- trial/result persistence,
- and compatibility shims.

## 2. Methodology

This report is code-first.
Every core claim maps to concrete files and lines.
No speculative runtime behavior is used where static evidence is sufficient.

Audit steps:
1. Locate all optimization entrypoints and references to `finetune_params`, Optuna, samplers, and trial logic.
2. Trace execution from pipeline step parsing through controller execution and optimization manager.
3. Compare documented capabilities to actual implementations.
4. Identify semantic mismatches, bugs, and missing interfaces.
5. Propose a phased architecture and migration path.

## 3. Current Architecture Map

### 3.1 Main finetuning execution path

Primary flow:
1. Pipeline model step includes `finetune_params`.
2. `BaseModelController.execute` checks mode and `finetune_params`.
3. It enters `_execute_finetune`.
4. `_execute_finetune` calls `self.finetune(...)`.
5. `BaseModelController.finetune` delegates to `OptunaManager.finetune(...)`.
6. `OptunaManager` runs study and returns `best_params`.
7. Controller then calls `self.train(...)` with `best_params`.

Key references:
- `nirs4all/nirs4all/controllers/models/base_model.py:602`
- `nirs4all/nirs4all/controllers/models/base_model.py:628`
- `nirs4all/nirs4all/controllers/models/base_model.py:684`
- `nirs4all/nirs4all/controllers/models/base_model.py:723`
- `nirs4all/nirs4all/optimization/optuna.py:52`

### 3.2 Strategy routing in OptunaManager

`OptunaManager.finetune` strategy dispatch:
- `individual` for per-fold optimization.
- `grouped` for a single study across all folds.
- fallback `single` logic when no supported fold strategy branch is selected.

Key reference:
- `nirs4all/nirs4all/optimization/optuna.py:87`
- `nirs4all/nirs4all/optimization/optuna.py:100`
- `nirs4all/nirs4all/optimization/optuna.py:108`
- `nirs4all/nirs4all/optimization/optuna.py:116`

### 3.3 Study creation and sampler selection

Sampler routing in `_create_study`:
- Supports `'auto'`.
- Supports `'grid'`.
- All other values map to TPE.

There is no explicit implementation for:
- random sampler,
- CMA-ES sampler,
- Hyperband pruner integration,
- or user-provided sampler object.

Key references:
- `nirs4all/nirs4all/optimization/optuna.py:355`
- `nirs4all/nirs4all/optimization/optuna.py:357`
- `nirs4all/nirs4all/optimization/optuna.py:369`
- `nirs4all/nirs4all/optimization/optuna.py:373`

### 3.4 Hyperparameter sampling

Strengths:
- Supports tuple and list representations.
- Supports dict-based type definitions.
- Supports log scales.
- Supports sorted tuple generation.
- Handles tuple-to-list serialization conversion.

Key references:
- `nirs4all/nirs4all/optimization/optuna.py:409`
- `nirs4all/nirs4all/optimization/optuna.py:490`
- `nirs4all/nirs4all/optimization/optuna.py:504`
- `nirs4all/nirs4all/optimization/optuna.py:568`
- `nirs4all/nirs4all/optimization/optuna.py:626`
- `nirs4all/nirs4all/pipeline/config/component_serialization.py:162`

### 3.5 Controller integration

Controllers integrate in two ways:
- `_get_model_instance(..., force_params=...)`
- `process_hyperparameters(...)`

This allows frameworks to reshape sampled parameter dicts.
TensorFlow and PyTorch define custom `process_hyperparameters` implementations.
JAX returns params as-is.

Key references:
- `nirs4all/nirs4all/controllers/models/tensorflow_model.py:597`
- `nirs4all/nirs4all/controllers/models/torch_model.py:469`
- `nirs4all/nirs4all/controllers/models/jax_model.py:484`

## 4. Confirmed Gaps and Issues

This section is deliberately direct.
Severity is from optimization correctness and user trust perspective.

### 4.1 Critical: tuned parameters are not applied in final training flow

Observed behavior:
- `_execute_finetune` obtains `best_model_params`.
- It then calls `train(..., mode="train", best_params=best_model_params)`.
- `launch_training` only applies `best_params` when `mode == "finetune"`.
- Therefore, standard finetune execution with mode forced to `train` does not use tuned params.

References:
- `nirs4all/nirs4all/controllers/models/base_model.py:648`
- `nirs4all/nirs4all/controllers/models/base_model.py:1146`

Impact:
- Optimization may run and report best parameters.
- Final persisted models can still be trained with non-tuned defaults.
- Reported `best_params` becomes metadata, not behavior.

### 4.2 Critical: single strategy uses training data as validation data

Observed behavior:
- In fallback single path, `X_val, y_val` are set to `X_train, y_train`.
- Comment indicates prior alternative using test data was disabled.

Reference:
- `nirs4all/nirs4all/optimization/optuna.py:118`
- `nirs4all/nirs4all/optimization/optuna.py:119`

Impact:
- Hyperparameter selection can overfit to training distribution.
- No holdout semantics when folds are unavailable.
- Results are likely optimistic and unstable.

### 4.3 High: documented samplers are not implemented

Documentation/examples claim support for:
- `random`
- `tpe`
- `cmaes`
- `hyperband`

Actual implementation:
- only `grid` and `tpe` are explicitly built.
- unknown sampler values silently fall through to TPE.

References for claim surface:
- `nirs4all/examples/user/04_models/U02_hyperparameter_tuning.py:65`
- `nirs4all/examples/user/04_models/U02_hyperparameter_tuning.py:69`
- `nirs4all/examples/user/04_models/U02_hyperparameter_tuning.py:70`
- `nirs4all/examples/reference/R01_pipeline_syntax.py:350`
- `nirs4all/examples/pipeline_samples/README.md:77`

References for implementation:
- `nirs4all/nirs4all/optimization/optuna.py:355`
- `nirs4all/nirs4all/optimization/optuna.py:369`
- `nirs4all/nirs4all/optimization/optuna.py:373`

Impact:
- User expectation mismatch.
- Misleading configs can run “successfully” with unintended sampler.
- Hard to reason about experiments.

### 4.4 High: `finetune_params.train_params` search is not implemented

Observed behavior:
- Sampling routine reads only `model_params`.
- `train_params` in `finetune_params` are copied into train call as fixed values.
- If `train_params` contains range dicts, they are not sampled.
- Such dict values can break training calls (e.g., `epochs` expecting int).

References:
- `nirs4all/nirs4all/optimization/optuna.py:433`
- `nirs4all/nirs4all/optimization/optuna.py:438`
- `nirs4all/nirs4all/optimization/optuna.py:311`
- `nirs4all/nirs4all/optimization/optuna.py:314`
- `nirs4all/examples/pipeline_samples/08_complex_finetune.json:159`

Impact:
- Advertised capability exists only as schema shape, not functionality.
- Deep-learning tuning scenarios are constrained.

### 4.5 High: approach values in samples/docs do not match implemented router

Observed examples include `approach: cross` and other values.
Implementation recognizes only:
- `individual`
- `grouped`
Else branch becomes single optimization fallback.

References:
- `nirs4all/examples/pipeline_samples/08_complex_finetune.json:92`
- `nirs4all/examples/reference/R01_pipeline_syntax.py:348`
- `nirs4all/examples/pipeline_samples/README.md:75`
- `nirs4all/nirs4all/optimization/optuna.py:100`
- `nirs4all/nirs4all/optimization/optuna.py:108`
- `nirs4all/nirs4all/optimization/optuna.py:116`

Impact:
- User-provided strategy may silently run as single fallback.
- Experiments can be mislabeled.

### 4.6 High: eval mode values in docs/tests drift from implementation

Observed values in tests/docs:
- `mean`
Implementation supports:
- `best`
- `avg`
- `robust_best`
Other values fallback to sum.

References:
- `nirs4all/tests/integration/pipeline/test_finetune_integration.py:254`
- `nirs4all/examples/pipeline_samples/08_complex_finetune.json:55`
- `nirs4all/nirs4all/optimization/optuna.py:392`
- `nirs4all/nirs4all/optimization/optuna.py:399`
- `nirs4all/nirs4all/optimization/optuna.py:406`

Impact:
- Semantic drift.
- Non-obvious behavior.

### 4.7 High: sklearn evaluation path is semantically inconsistent and inefficient

Observed behavior:
- Objective trains model on train fold.
- Then `SklearnModelController._evaluate_model` runs `cross_val_score` on validation data.
- `cross_val_score` clones/refits estimator, ignoring the prior fit.

References:
- `nirs4all/nirs4all/optimization/optuna.py:225`
- `nirs4all/nirs4all/controllers/models/sklearn_model.py:445`
- `nirs4all/nirs4all/controllers/models/sklearn_model.py:453`
- `nirs4all/nirs4all/controllers/models/sklearn_model.py:457`

Impact:
- Double-fitting overhead per trial.
- Objective semantics differ from other frameworks.
- Selection criteria may not represent intended train/val procedure.

### 4.8 High: meta-model finetuning likely fails with `model__` prefixed params

Observed behavior:
- `MetaModel.get_finetune_params` returns `model_params` directly from `finetune_space`.
- Tests/examples use keys like `model__alpha`.
- `MetaModelController._get_model_instance` calls underlying sklearn model `set_params(**force_params)`.
- For simple estimators, `model__alpha` is invalid.

References:
- `nirs4all/nirs4all/operators/models/meta.py:338`
- `nirs4all/nirs4all/operators/models/meta.py:339`
- `nirs4all/tests/unit/controllers/models/stacking/test_finetune_integration.py:38`
- `nirs4all/nirs4all/controllers/models/meta_model.py:287`

Impact:
- trials may fail and return `inf`.
- meta-model tuning feature appears incomplete.

### 4.9 Medium: no explicit Optuna pruner integration

Observed behavior:
- `study.optimize` is used without pruner configuration.
- `sample="hyperband"` does not configure Hyperband pruner.

References:
- `nirs4all/nirs4all/optimization/optuna.py:244`
- `nirs4all/nirs4all/optimization/optuna.py:336`

Impact:
- expensive deep-learning trials cannot stop early via standardized pruner path.

### 4.10 Medium: no study storage or resume support

Observed behavior:
- `optuna.create_study` called without storage URL and without study name.

Reference:
- `nirs4all/nirs4all/optimization/optuna.py:377`

Impact:
- no resume.
- no multi-run trace continuity.
- no distributed tuning.

### 4.11 Medium: reproducibility controls are incomplete

Observed behavior:
- TPESampler created without seed.
- trial seed coordination across frameworks is absent.

Reference:
- `nirs4all/nirs4all/optimization/optuna.py:373`

Impact:
- repeated runs can diverge under identical config.

### 4.12 Medium: limited trial observability

Observed behavior:
- return value is only `study.best_params`.
- trial-level metrics and failure reasons are not persisted in the run result.

References:
- `nirs4all/nirs4all/optimization/optuna.py:342`
- `nirs4all/nirs4all/optimization/optuna.py:250`

Impact:
- post-mortem analysis is weak.

### 4.13 Medium: no explicit metric selection interface for finetune

Observed behavior:
- objective metric is implicit in controller `_evaluate_model` implementation.
- no `metric` field in finetune contract.

References:
- `nirs4all/nirs4all/controllers/models/sklearn_model.py:419`
- `nirs4all/nirs4all/controllers/models/tensorflow_model.py:489`
- `nirs4all/nirs4all/controllers/models/torch_model.py:397`
- `nirs4all/nirs4all/controllers/models/jax_model.py:423`

Impact:
- users cannot directly optimize a chosen business metric.

### 4.14 Medium: grid suitability logic excludes dict categorical parameters

Observed behavior:
- grid suitability requires all params to be list-type categorical.
- dict categorical params (with `{'type': 'categorical'}`) are treated as non-grid.

References:
- `nirs4all/nirs4all/optimization/optuna.py:688`
- `nirs4all/nirs4all/optimization/optuna.py:723`

Impact:
- user-specified categorical dict config can unexpectedly fallback to TPE.

### 4.15 Medium: dead extension hooks in controllers

Observed behavior:
- `SklearnModelController._sample_hyperparameters` calls `super()._sample_hyperparameters`.
- base controller does not define this method.
- OptunaManager currently bypasses this hook anyway.

References:
- `nirs4all/nirs4all/controllers/models/sklearn_model.py:555`
- `nirs4all/nirs4all/controllers/models/autogluon_model.py:523`

Impact:
- API drift and confusion for future contributors.

### 4.16 Medium: tests validate “runs complete” more than “optimization correctness”

Observed behavior:
- integration tests largely assert non-empty predictions and finite scores.
- they do not assert sampler semantics, parameter application correctness, or pruner behavior.

Reference:
- `nirs4all/tests/integration/pipeline/test_finetune_integration.py`

Impact:
- regressions in optimization semantics can pass unnoticed.

## 5. Claimed vs Implemented Capability Matrix

Legend:
- `Implemented`: behavior exists and aligns with expectations.
- `Partial`: behavior exists but limited or drifted.
- `Missing`: no implementation.
- `Misleading`: config accepted but behavior differs substantially.

| Capability | Surface Claim | Implementation Status | Notes |
|---|---|---|---|
| Optuna-based finetuning | Yes | Implemented | Core manager exists. |
| Tuple/list/dict search spaces | Yes | Implemented | Strong part of current stack. |
| Sorted tuple search parameter type | Yes | Implemented | Works in sampler parser. |
| `sample=grid` | Yes | Partial | Works only for list categorical spaces. |
| `sample=random` | Yes | Misleading | Falls to TPE. |
| `sample=tpe` | Yes | Implemented | Uses `TPESampler`. |
| `sample=cmaes` | Yes in docs/examples | Missing | Not wired. |
| `sample=hyperband` | Yes in docs/examples | Missing | No Hyperband pruner wiring. |
| `approach=grouped` | Yes | Implemented | Works. |
| `approach=individual` | Yes in some surfaces | Implemented | Works when folds exist. |
| `approach=cross` | Yes in examples | Misleading | Falls to single branch. |
| `eval_mode=best` | Yes | Implemented | Uses `min(scores)`. |
| `eval_mode=avg` | Internal | Implemented | Uses sum, effectively mean ranking if fold count fixed. |
| `eval_mode=mean` | Examples/tests | Misleading | Falls to default sum path. |
| Tune `model_params` | Yes | Implemented | Main supported flow. |
| Tune `finetune_params.train_params` ranges | Claimed in samples | Missing | Not sampled. |
| Final model trained with tuned params | Expected | Missing/Critical bug | Mode gate prevents apply. |
| Trial pruning | Claimed indirectly | Missing | No pruner setup/report calls. |
| Study persistence/resume | Common Optuna expectation | Missing | No storage config. |
| Seeded sampler reproducibility | Common requirement | Partial | Not exposed for Optuna. |
| Per-trial artifact/metric trace | Desired for analysis | Missing | Only best params returned. |

## 6. Root Cause Synthesis

Primary root causes appear structural, not isolated.

Root cause A: optimization is treated as a helper, not a subsystem.
- centralized class exists,
- but it owns too many responsibilities,
- and lacks explicit contracts between layers.

Root cause B: documentation/examples evolved faster than implementation.
- sample strategies and approach values drifted.
- test suite followed surface configs but not semantic assertions.

Root cause C: controller evaluation contracts are not normalized.
- each framework defines `_evaluate_model` differently.
- objective code assumes equivalent semantics.

Root cause D: compatibility shortcuts introduced silent fallback behavior.
- unsupported values often degrade to default branch.
- this preserves run completion but damages trust.

Root cause E: no strong typed schema for finetune config.
- values like `cross`, `mean`, or dict ranges in train_params are accepted until runtime.
- early validation is weak.

## 7. Design Goals for the Next Optimization Layer

Mandatory goals:
1. Correctness over convenience.
2. Explicitness over silent fallbacks.
3. Reproducibility by default.
4. Common evaluation semantics across frameworks.
5. Pluggable samplers and pruners.
6. First-class trial telemetry.
7. Backward-compatible migration with strict warnings.

Practical goals:
1. Keep tuple/list/dict search space ergonomics.
2. Keep integration footprint low for existing controllers.
3. Make failure diagnostics precise and fast.

## 8. Proposed Target Architecture

Introduce a dedicated `optimization` package structure:

- `optimization/config.py`
- `optimization/schema.py`
- `optimization/search_space.py`
- `optimization/study_manager.py`
- `optimization/evaluator.py`
- `optimization/trial_runner.py`
- `optimization/adapters/`
- `optimization/reporting.py`
- `optimization/compat.py`

### 8.1 Proposed contracts

#### 8.1.1 FinetuneConfig (typed)

Core fields:
- `n_trials: int`
- `approach: Literal['single', 'grouped', 'individual']`
- `sampler: Literal['auto', 'grid', 'random', 'tpe', 'cmaes']`
- `pruner: Literal['none', 'median', 'successive_halving', 'hyperband']`
- `eval_mode: Literal['mean', 'best', 'median', 'robust_best']`
- `metric: Optional[str]`
- `direction: Optional[Literal['minimize', 'maximize']]`
- `seed: Optional[int]`
- `timeout_seconds: Optional[int]`
- `n_jobs: int`
- `model_params: Dict[str, ParamSpec]`
- `train_params: Dict[str, ParamSpecOrLiteral]`
- `storage: Optional[StorageConfig]`
- `early_fail_threshold: Optional[int]`

#### 8.1.2 ParamSpec

Supported forms (normalized internally):
- `categorical`
- `int`
- `float`
- `log_int`
- `log_float`
- `sorted_tuple`
- `conditional`

#### 8.1.3 TrialResult

Capture per trial:
- sampled params,
- effective merged params,
- per-fold scores,
- aggregate score,
- status,
- failure reason,
- duration,
- framework backend metadata.

### 8.2 Layer responsibilities

`search_space.py`
- Parse and normalize all model/train param spaces.
- Convert legacy tuple/list forms to canonical ParamSpec objects.
- Validate unsupported combinations early.

`study_manager.py`
- Build sampler and pruner explicitly.
- Build Optuna study with optional storage and resume semantics.
- Coordinate seed and reproducibility controls.

`evaluator.py`
- Provide framework-neutral trial scoring contract.
- Manage fold iteration semantics.
- Aggregate fold scores by selected `eval_mode`.

`trial_runner.py`
- Execute one trial with robust error capture.
- Report metrics and intermediate values for pruning.

`adapters/*`
- Map trial params into framework-specific model/train parameter partitions.
- Optionally include callback hooks for training-level reporting.

`reporting.py`
- Persist trial history into run artifacts.
- Provide JSON export for dashboards and diagnostics.

`compat.py`
- Translate legacy fields (`sample`, `approach=cross`, `eval_mode=avg/mean`) with deprecation warnings.

## 9. Proposed Semantic Rules

### 9.1 Approach semantics

`single`:
- one train/validation split.
- if no folds supplied, create internal holdout from train partition.

`grouped`:
- one study.
- each trial evaluated across all folds.
- aggregation controlled by `eval_mode`.

`individual`:
- one study per fold.
- returns list of best params by fold.

### 9.2 Eval modes

`mean`:
- arithmetic mean of fold metric.

`best`:
- best fold value (min or max by direction).

`median`:
- median fold metric.

`robust_best`:
- percentile-based robust selector (e.g., 25th/75th depending direction).

### 9.3 Metric direction

Direction resolution order:
1. explicit `direction` in finetune config,
2. metric registry mapping,
3. controller default.

No silent inversion hacks should be required.

### 9.4 Train-parameter optimization

`train_params` should support sampling specs just like `model_params`.
Sampling output is split into:
- `sampled_model_params`
- `sampled_train_params`

These are merged with fixed train params deterministically.

### 9.5 Failure policy

Trial failures should be visible.
Policy fields:
- `fail_fast: bool`
- `max_consecutive_failures`
- `max_failure_ratio`

## 10. Concrete Fixes for Current Code (Immediate)

This is a near-term hotfix set before deeper refactor.

### 10.1 Apply tuned params in final training

Current issue:
- `best_params` ignored due mode gate.

Minimal fix options:
- Option A: pass `mode="finetune"` in `_execute_finetune` train call.
- Option B: remove mode gate in `launch_training` and apply `best_params` whenever provided.

Recommended:
- Option B plus explicit assertion tests.

### 10.2 Restore valid single-mode validation behavior

Current issue:
- single mode uses train as validation.

Fix:
- create internal holdout split from train partition when no folds exist.
- support `validation_split` and `validation_seed` in finetune config.

### 10.3 Implement explicit sampler mapping and validation

Current issue:
- unsupported sampler values silently map to TPE.

Fix:
- map values explicitly.
- throw validation error for unsupported values unless compatibility mode enabled.

### 10.4 Implement pruner mapping

Current issue:
- no pruner despite documented hyperband surface.

Fix:
- add Optuna pruner creation with mapping:
  - `hyperband -> HyperbandPruner`
  - `successive_halving -> SuccessiveHalvingPruner`
  - `median -> MedianPruner`

### 10.5 Implement train_params sampling

Current issue:
- train params ranges are passed as raw dicts.

Fix:
- parse and sample train param specs.
- keep fixed literals unchanged.

### 10.6 Normalize eval mode aliases

Current issue:
- `mean`/`avg` drift.

Fix:
- canonicalize aliases during config normalization.
- deprecate non-canonical values with warning.

### 10.7 Fix meta-model parameter names

Current issue:
- `model__` prefix likely invalid for underlying estimator set_params in MetaModelController path.

Fix:
- strip `model__` prefix for MetaModel internal estimator tuning path,
- or enforce plain estimator param names in `finetune_space` for MetaModel.

### 10.8 Remove dead `_sample_hyperparameters` hooks or wire them properly

Current issue:
- controller-specific `_sample_hyperparameters` methods are effectively dead.

Fix:
- either delete dead hooks,
- or make OptunaManager delegate sampling to controller adapter contract.

## 11. Mid-Term Refactor Plan (Phased)

### Phase 1: Stabilization (1-2 sprints)

Goals:
- fix correctness bugs,
- remove misleading silent fallbacks,
- improve runtime diagnostics.

Deliverables:
- hotfixes 10.1 to 10.8,
- config alias normalization,
- upgraded integration tests asserting semantics.

### Phase 2: Typed configuration and validation (1 sprint)

Goals:
- validate before optimization begins,
- produce deterministic behavior and explicit errors.

Deliverables:
- `FinetuneConfig` model,
- parsing and migration warnings,
- schema docs update.

### Phase 3: Study lifecycle and telemetry (1-2 sprints)

Goals:
- expose trial histories and study metadata.

Deliverables:
- persistent trial logs,
- per-trial status and error reporting,
- optional storage backend for resume.

### Phase 4: Advanced samplers/pruners (1 sprint)

Goals:
- align implementation with advertised capabilities.

Deliverables:
- random sampler,
- CMA-ES sampler,
- Hyperband and median pruners,
- intermediate metric reporting hooks.

### Phase 5: Unification and cleanup (1 sprint)

Goals:
- remove legacy aliases,
- simplify controller optimization contracts.

Deliverables:
- final compatibility mode toggle,
- docs and examples aligned to canonical behavior only.

## 12. Test Strategy Proposal

### 12.1 Unit tests

Needed for parser and config layer:
- sampler value mapping,
- eval mode alias mapping,
- approach mapping,
- train_params sampling,
- error handling for invalid ranges.

Needed for Optuna manager behavior:
- study creation includes selected sampler/pruner,
- single/grouped/individual semantics,
- direction handling,
- reproducible seed behavior.

Needed for controllers:
- tuned params application in final training,
- framework-specific evaluation contract compliance.

### 12.2 Integration tests

Must assert semantics, not just run completion.
Examples:
- A known convex toy problem where tuned params must beat baseline.
- sampler-specific trace checks (`random` vs `tpe`).
- hyperband pruner actually pruning trials.
- train_params ranges produce sampled numeric values, not raw dicts.

### 12.3 Regression tests for current critical bugs

Must include:
- Test that `best_params` affect final model instance config.
- Test that single mode with no folds does not use identical train==val.
- Test that unsupported sampler values raise explicit validation errors when strict mode is enabled.

## 13. Documentation and Example Cleanup Plan

Documentation updates required:
- Update `U02_hyperparameter_tuning.py` text to match real features or implement missing features first.
- Update `examples/pipeline_samples/README.md` approach/eval/sampler descriptions.
- Update `R01_pipeline_syntax.py` comments for canonical values.
- Update `docs/source/examples/user/models.md` pruning claims only when implemented.

Policy recommendation:
- No example should advertise a value that is not validated and supported in code.
- No test should treat “pipeline completed” as sufficient for optimization features.

## 14. Proposed `finetune_params` v2 Schema

Canonical example:

```json
{
  "finetune_params": {
    "n_trials": 40,
    "approach": "grouped",
    "sampler": "tpe",
    "pruner": "median",
    "eval_mode": "mean",
    "metric": "rmse",
    "direction": "minimize",
    "seed": 42,
    "timeout_seconds": 1800,
    "model_params": {
      "n_components": ["int", 1, 30],
      "alpha": ["float_log", 0.0001, 10.0]
    },
    "train_params": {
      "epochs": ["int", 10, 150],
      "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
      "learning_rate": ["float_log", 0.00001, 0.01]
    }
  }
}
```

Compatibility mapping examples:
- `sample` -> `sampler`
- `eval_mode=avg` -> `eval_mode=mean`
- `approach=cross` -> `approach=grouped` (warning)

## 15. Risk Register

R1: Behavior changes can affect existing experiments.
Mitigation: compatibility mode with structured warnings and telemetry.

R2: Adding strict validation could break permissive pipelines.
Mitigation: staged rollout with `strict_finetune_config=false` default first.

R3: Trial telemetry may increase storage size.
Mitigation: configurable retention policy and compact JSON summary mode.

R4: Framework adapters can diverge again.
Mitigation: shared protocol tests that every adapter must pass.

R5: Performance regression from richer bookkeeping.
Mitigation: benchmark gates and opt-out debug detail levels.

## 16. Suggested Priority Backlog (human-readable summary)

Priority P0:
- Fix tuned parameter application bug.
- Fix single strategy validation split semantics.
- Add explicit sampler validation and clear errors.

Priority P1:
- Implement train_params sampling.
- Add random and hyperband support.
- Normalize approach/eval aliases with deprecation warnings.

Priority P2:
- Add trial telemetry and persistent study support.
- Add metric/direction explicit controls.
- Add reproducibility seed wiring across sampler and framework.

Priority P3:
- Refactor to adapter-based optimization interfaces.
- Remove dead hooks and cleanup old docs/examples.

## 17. Acceptance Criteria for “Optimization V2”

Functional acceptance:
1. `best_params` must always influence final model training when finetuning is requested.
2. `single` must use a real validation split (never identical train and val arrays).
3. `sampler` values (`grid`, `random`, `tpe`, `cmaes`) must map to concrete behavior or fail validation.
4. `pruner=hyperband` must produce observable pruning events.
5. `train_params` ranges must be sampled and typed before model training call.
6. `approach` and `eval_mode` accepted values must be canonicalized and documented.
7. Trial history must be queryable after run completion.

Quality acceptance:
1. Unit tests cover all parser aliases and rejection paths.
2. Integration tests verify behavioral outcomes, not only successful execution.
3. Reproducibility test passes for seeded runs.
4. Documentation examples execute successfully in CI with declared semantics.

## 18. Appendix A: Detailed Evidence Notes

A-001: Optuna import guard and availability flag are centralized in one file.
A-002: `OptunaManager` constructor warns and skips finetune when Optuna is unavailable.
A-003: Current skip behavior returns empty dict instead of hard failure.
A-004: `finetune` strategy default is `grouped`.
A-005: `eval_mode` default is `best`.
A-006: `n_trials` default is `50`.
A-007: Single fallback path is selected for unknown strategy values.
A-008: Single fallback path currently assigns validation to training arrays.
A-009: Grouped objective loops folds and aggregates scores.
A-010: Grouped path catches all exceptions and appends `inf`.
A-011: Single objective catches all exceptions and returns `inf`.
A-012: Study is always created with `direction="minimize"`.
A-013: No explicit direction mapping from metric exists.
A-014: Sampler key allows both `sampler` and `sample` aliases.
A-015: Auto sampler picks grid only when every param is list-categorical.
A-016: Explicit `grid` request can downgrade to TPE on unsupported spaces.
A-017: Dict categorical parameters do not currently qualify for grid suitability.
A-018: Grid search space builder only includes list values.
A-019: `sample_hyperparameters` focuses only on `model_params`.
A-020: Legacy mode extracts params from top-level finetune dict excluding known keys.
A-021: `train_params` excluded from sampling map in legacy extraction.
A-022: `_sample_single_parameter` supports tuple and list patterns.
A-023: tuple-to-list conversion is intentionally handled.
A-024: `_suggest_from_type` supports int/float and log variants.
A-025: `_suggest_from_dict` supports categorical/int/float/sorted_tuple.
A-026: sorted_tuple supports dynamic length config.
A-027: `avg` mode currently computes sum of scores.
A-028: fallback eval_mode path also computes sum.
A-029: no explicit mean mode symbol in implementation.
A-030: no percentile or median eval mode.
A-031: no trial callbacks or intermediate reporting to pruner.
A-032: no timeout passed to study.optimize from finetune params.
A-033: no study name or storage backend passed.
A-034: no sampler seed wiring.
A-035: no `n_jobs` for Optuna optimizer call.
A-036: controller receives optimization params via `force_params`.
A-037: controller process hook exists (`process_hyperparameters`).
A-038: TensorFlow process hook maps `compile_` and `fit_` prefixes.
A-039: PyTorch process hook maps `optimizer_` prefix.
A-040: JAX process hook is identity.
A-041: final model training path from finetune calls `train(... mode='train')`.
A-042: launch path applies best params only when mode is `finetune`.
A-043: this mode mismatch makes tuned params ineffective in final training.
A-044: `train_params` for normal training is taken from model step `train_params`, not finetune sample.
A-045: best params are carried in prediction metadata regardless of application correctness.
A-046: fold training can run parallel in normal train mode only.
A-047: finetuning optimization is always sequential.
A-048: prediction mode and explain mode bypass optimization.
A-049: sklearn evaluation uses `cross_val_score` on validation partition.
A-050: `cross_val_score` refits estimator and can ignore previous fit state.
A-051: regression scoring in sklearn evaluator uses `neg_mean_squared_error` and negates sign.
A-052: classification scoring in sklearn evaluator uses balanced_accuracy and negates.
A-053: TF evaluator uses model.evaluate loss.
A-054: PyTorch evaluator uses MSE loss directly.
A-055: JAX evaluator uses MSE from predictions.
A-056: per-framework evaluator metrics are inconsistent by design.
A-057: `MetaModel.get_finetune_params` copies full finetune_space into model_params.
A-058: `n_trials` and `approach` can leak into model_params if user includes them in finetune_space.
A-059: stack tests encourage `model__alpha` format.
A-060: MetaModelController applies force params directly through underlying model.set_params.
A-061: `model__alpha` likely invalid for plain underlying estimator.
A-062: integration tests for finetune do not validate tuned-parameter application.
A-063: tests include sampler values `random` and eval mode `mean` without asserting semantics.
A-064: docs claim early stopping and pruning in user models guide.
A-065: current optimization code has no pruning integration.
A-066: examples claim hyperband and cmaes support.
A-067: code has no cmaes branch.
A-068: hyperband string currently routes to TPE by fallback behavior.
A-069: pipeline samples include `approach: cross`.
A-070: code does not have cross strategy symbol.
A-071: component serialization converts tuples to lists globally.
A-072: Optuna sampling parser compensates for this conversion pattern.
A-073: undefined sampler values are not rejected with strong errors.
A-074: optimization failures are mostly compressed into `inf` scores.
A-075: user can complete run with many failed trials but limited diagnostics.
A-076: there is no built-in min successful trials threshold.
A-077: no explicit trial status export into run result object.
A-078: no unified “optimization report” artifact.
A-079: no library-level API for resuming existing studies.
A-080: no distributed worker support pattern yet.

## 19. Appendix B: Proposed Compatibility Policy

Policy B-001: Introduce compatibility mode flag `finetune_compat_mode` default `true` in first release.
Policy B-002: In compat mode, map `sample` to `sampler` and log deprecation warning.
Policy B-003: In compat mode, map `eval_mode=avg` and `eval_mode=mean` to canonical `mean`.
Policy B-004: In compat mode, map `approach=cross` to `grouped` with warning.
Policy B-005: In strict mode, reject unsupported values with validation error before training.
Policy B-006: Track deprecation usage counts in runtime metrics.
Policy B-007: Publish migration examples for old to new schema.
Policy B-008: Remove compat aliases after two minor release cycles.

## 20. Appendix C: Immediate Patch Checklist (P0)

C-001: Update `launch_training` to apply `best_params` regardless of mode or when explicitly provided.
C-002: Add regression test ensuring tuned parameter reaches estimator instance.
C-003: Replace single-mode train-as-val assignment with holdout split.
C-004: Add configurable `validation_split` for single mode.
C-005: Add deterministic split seed support.
C-006: Add explicit sampler parser with allowed enum and alias table.
C-007: Implement random sampler branch.
C-008: Implement cmaes sampler branch behind optional dependency checks.
C-009: Implement hyperband pruner branch.
C-010: Add warning/error path for unsupported pruner values.
C-011: Parse train_params search specs in sampler path.
C-012: Keep fixed train_params literals unchanged.
C-013: Add test for train_params dict-range conversion and sampling.
C-014: Normalize eval_mode aliases and canonical enum.
C-015: Normalize approach aliases and canonical enum.
C-016: Add strict validation toggle.
C-017: Add failure ratio threshold to abort invalid studies early.
C-018: Add trial status and exception summary capture.
C-019: Add minimal optimization report artifact in workspace.
C-020: Document changed behavior in changelog and examples.

## 21. Appendix D: Proposed Public API Shape

New helper API idea:

```python
from nirs4all.optimization import FinetuneConfig, optimize

config = FinetuneConfig(
    n_trials=50,
    approach="grouped",
    sampler="tpe",
    pruner="median",
    eval_mode="mean",
    metric="rmse",
    direction="minimize",
    seed=42,
)

best, report = optimize(
    controller=controller,
    dataset=dataset,
    model_config=model_config,
    folds=folds,
    finetune_config=config,
)
```

Benefits:
- clearer separation between execution and optimization.
- reusable optimization entrypoint for CLI and API layers.
- easier telemetry and testing.

## 22. Appendix E: Migration Examples

### E.1 Old config (currently accepted)

```json
{
  "finetune_params": {
    "n_trials": 20,
    "sample": "random",
    "approach": "cross",
    "eval_mode": "mean",
    "model_params": {
      "n_components": ["int", 1, 20]
    }
  }
}
```

### E.2 Canonical migrated config

```json
{
  "finetune_params": {
    "n_trials": 20,
    "sampler": "random",
    "approach": "grouped",
    "eval_mode": "mean",
    "model_params": {
      "n_components": ["int", 1, 20]
    }
  }
}
```

### E.3 Train-parameter tuning migrated config

```json
{
  "finetune_params": {
    "n_trials": 30,
    "sampler": "tpe",
    "approach": "single",
    "model_params": {
      "filters_1": [8, 16, 32]
    },
    "train_params": {
      "epochs": ["int", 5, 50],
      "batch_size": {"type": "categorical", "choices": [16, 32, 64]}
    }
  }
}
```

## 23. Appendix F: Proposed CI Gates


## 28. Final Recommendation Snapshot

If only five actions are funded immediately, do these in order:
1. Fix best-params application bug.
2. Fix single-mode validation semantics.
3. Add strict sampler/approach/eval validation with compatibility warnings.
4. Implement train_params sampling and wire pruner support.
5. Add semantic integration tests that assert optimization correctness outcomes.

Expected outcome after these five:
- optimization behavior becomes trustworthy,
- docs/examples become aligned with runtime behavior,
- and future extension cost drops due cleaner contracts.

## 29. Closeout

This audit confirms that current finetuning is useful but incomplete.
The next iteration should prioritize correctness and transparency before adding more surface area.
Once the P0/P1 changes land, `nirs4all` can safely expand to a more complete optimization platform.
