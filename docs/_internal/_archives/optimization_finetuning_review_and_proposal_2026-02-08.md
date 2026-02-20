# nirs4all Optimization: Review, Technical Debt, and Roadmap

Date: 2026-02-09
Scope: Full audit of Optuna integration across optimization, controllers, tests, examples, and documentation.

Files audited:
- `nirs4all/optimization/optuna.py` (OptunaManager, 754 lines)
- `nirs4all/controllers/models/base_model.py` (_execute_finetune, launch_training, train, finetune, process_hyperparameters)
- `nirs4all/controllers/models/sklearn_model.py` (_evaluate_model, _get_model_instance)
- `nirs4all/controllers/models/tensorflow_model.py` (_evaluate_model, process_hyperparameters)
- `nirs4all/controllers/models/torch_model.py` (_evaluate_model, process_hyperparameters)
- `nirs4all/controllers/models/jax_model.py` (_evaluate_model, process_hyperparameters)
- `nirs4all/controllers/models/meta_model.py` (_get_model_instance, force_params)
- `nirs4all/operators/models/meta.py` (get_finetune_params, finetune_space)
- `tests/integration/pipeline/test_finetune_integration.py`
- `tests/unit/controllers/models/stacking/test_finetune_integration.py`
- `examples/user/04_models/U02_hyperparameter_tuning.py`
- `examples/reference/R01_pipeline_syntax.py`
- `examples/pipeline_samples/08_complex_finetune.json`
- `examples/pipeline_samples/README.md`
- `bench/_tabpfn/search_space.py` (nested dict parameter pattern reference)

---

## 1. Objectives with Optimization in nirs4all

### 1.1 What optimization should achieve

The optimization system in nirs4all exists to answer one question for the user: **given a pipeline and dataset, what are the best hyperparameters for the model step?** This must be:

- **Correct**: Optimized parameters must actually be used in the final trained model.
- **Honest**: The system must only advertise features it implements. Silent fallbacks are bugs.
- **Flexible**: Users should be able to tune model constructor params (e.g. `n_components`), training params (e.g. `epochs`, `batch_size`), and complex nested params (e.g. TabPFN `inference_config`, sklearn `StackingRegressor` estimators).
- **Efficient**: Smart search strategies (Bayesian, pruning, multi-phase) should reduce wasted trials.
- **Reproducible**: Same seed, same data, same results.
- **Observable**: Users should see what happened during optimization, not just the final answer.

### 1.2 Design principles

| Principle | Meaning |
|-----------|---------|
| **Correctness over features** | A working `tpe` sampler is worth more than a broken `hyperband` |
| **Fail fast, never fail silent** | Unknown `approach`, `sampler`, or `eval_mode` values must raise, never silently fallback |
| **Orthogonal to generators** | Pipeline generators (`_or_`, `_range_`, `_cartesian_`) explore pipeline structure. Optuna explores parameter values within a fixed pipeline variant. They compose, never conflict |
| **Framework-agnostic evaluation** | The same `metric`/`direction` config must produce comparable objectives whether the model is sklearn, TensorFlow, PyTorch, or JAX |
| **No dead code** | Every advertised feature is implemented and tested. Every code path is reachable |

### 1.3 Target user experience

```python
# Simple: tune 1 parameter
{"model": PLSRegression(), "finetune_params": {
    "n_trials": 20,
    "model_params": {"n_components": ("int", 1, 30)},
}}

# Intermediate: control sampler, eval, pruning
{"model": RandomForestRegressor(), "finetune_params": {
    "n_trials": 100,
    "sampler": "tpe",
    "pruner": "successive_halving",
    "approach": "grouped",
    "eval_mode": "mean",
    "seed": 42,
    "model_params": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 3, 20),
        "min_samples_leaf": [1, 2, 4, 8],
    },
}}

# Advanced: multi-phase search, train_params tuning, nested params
{"model": NICONModel(), "finetune_params": {
    "phases": [
        {"sampler": "random", "n_trials": 30},
        {"sampler": "tpe", "n_trials": 70},
    ],
    "model_params": {
        "filters_1": [8, 16, 32, 64],
        "dropout_rate": ("float", 0.1, 0.5),
    },
    "train_params": {
        "epochs": ("int", 50, 300),
        "batch_size": [16, 32, 64],
    },
}}

# Expert: reuse best_params from older run, nested dict for TabPFN
{"model": TabPFNRegressor(), "finetune_params": {
    "n_trials": 50,
    "force_params": {"n_components": 12},  # Seed from prior run
    "model_params": {
        "inference_config": {
            "FINGERPRINT_FEATURE": [True, False],
            "OUTLIER_REMOVAL_STD": [None, 7.0, 12.0],
        },
    },
}}
```

---

## 2. Technical Debt and Discrepancy Between Optuna Capabilities and Its Usage

### 2.1 Critical bugs (break correctness)

#### BUG-1: Best parameters are NOT applied in final training

**Severity**: Critical
**Impact**: Finetuning runs Optuna, finds best params, then trains the final model with DEFAULT params. The entire optimization is wasted.

**Root cause**: Mode mismatch between `_execute_finetune` and `launch_training`.

```
_execute_finetune() [base_model.py:645-648]:
    self.train(..., mode="train", best_params=best_model_params)
                     ^^^^^^^^^^^^
                     mode is "train", NOT "finetune"

launch_training() [base_model.py:1146]:
    if mode == "finetune" and best_params is not None:
       ^^^^^^^^^^^^^^^^^
       This check FAILS because mode == "train"
       → Falls through to line 1150-1157: creates model with DEFAULT params
```

**Evidence**: `base_model.py:648` passes `mode="train"` but `base_model.py:1146` gates on `mode == "finetune"`.

**Fix**: Change condition at line 1146 to `if best_params is not None:` (mode is irrelevant when best_params are provided).

---

#### BUG-2: `eval_mode="avg"` computes sum, not mean

**Severity**: Critical
**Impact**: Grouped fold optimization with `eval_mode="avg"` optimizes for total loss across folds, not average loss. A 5-fold setup penalizes trial scores by 5x compared to 3-fold, making comparisons nonsensical.

```python
# optuna.py:399-400
elif eval_mode == 'avg':
    return np.sum(scores)  # BUG: should be np.mean(scores)

# optuna.py:406-407 (default fallback)
else:
    return np.sum(scores)  # Same bug in fallback
```

**Fix**: Replace `np.sum` with `np.mean` at lines 400 and 407.

---

#### BUG-3: Single-path optimization uses training data as validation

**Severity**: High
**Impact**: When no folds are available and approach falls through to single path, the model is evaluated on the same data it trained on. This guarantees overfitting — the optimizer will select parameters that memorize training data.

```python
# optuna.py:119
X_val, y_val = X_train, y_train  # Evaluates on training data!
```

Note the commented-out correct line at 118: `# X_val, y_val = X_test, y_test`. The previous implementation used test data but was changed, breaking the validation semantics.

**Fix**: Use a holdout split from X_train (e.g. 80/20) or require folds. Never use X_train == X_val for optimization.

---

#### BUG-4: Finetuning is re-triggered during refit instead of using Pass 1 best_params

**Severity**: Critical
**Impact**: When the refit phase re-executes the winning pipeline on the full training set, it re-runs Optuna from scratch instead of simply training with the `best_params` found during Pass 1 (CV). This wastes time, may find different params (non-deterministic samplers), and compounds with BUG-1 (the re-found params aren't applied either).

**Root cause**: Three compounding issues in the refit path:

1. **`_inject_best_params`** (`refit/executor.py:274-312`) applies `best_params` via `set_params()` on the model instance, but **does not remove `finetune_params` from the step dict**. The step dict still contains the full finetune configuration.

2. The executor reused from Pass 1 has **`mode="train"`** (set at init, `executor.py:63`). The `runtime_context.phase = ExecutionPhase.REFIT` is set (`refit/executor.py:127`), but the model controller **never checks the execution phase** before the finetune dispatch.

3. In `BaseModelController.execute()` (`base_model.py:602-604`):
   ```python
   finetune_params = model_config.get('finetune_params')  # Still present!
   if mode == "finetune" or (mode == "train" and finetune_params):
       return self._execute_finetune(...)  # Re-runs Optuna!
   ```
   Since `mode == "train"` and `finetune_params` is still in the step dict, finetuning is re-triggered.

**What should happen**: During refit, the model controller should skip finetuning entirely and train with `best_params` from `RefitConfig` (already extracted by `config_extractor.py`).

**Fix** (both layers):

a) Guard in model controller (`base_model.py:602-604`):
```python
finetune_params = model_config.get('finetune_params')
is_refit = runtime_context.phase == ExecutionPhase.REFIT

if not is_refit and (mode == "finetune" or (mode == "train" and finetune_params)):
    return self._execute_finetune(...)
```

b) Strip `finetune_params` in `_inject_best_params` (`refit/executor.py`):
```python
step.pop("finetune_params", None)  # Prevent re-triggering during refit
```

---

### 2.2 High-severity issues (produce incorrect or misleading behavior)

#### ISSUE-4: `train_params` ranges are not sampled by Optuna

**Current behavior**: `sample_hyperparameters()` (optuna.py:433) only iterates `model_params`. The `train_params` dict is merged directly into trial kwargs at line 311 without any Optuna sampling.

```python
# optuna.py:311
train_params_for_trial = finetune_params.get('train_params', {}).copy()

# optuna.py:314 — merges model_params (sampled) into train_params (raw)
train_params_for_trial.update(sampled_params)
```

If a user writes `"train_params": {"epochs": ("int", 50, 300)}`, Optuna will NOT sample it. The tuple `("int", 50, 300)` is passed directly to `_train_model` as a raw value, causing a crash or silent misuse.

**Impact**: Users cannot tune training hyperparameters (epochs, learning rate, batch size) through Optuna — only model constructor params.

---

#### ISSUE-5: Documented samplers exceed implemented samplers

**Implemented** in `_create_study()` (optuna.py:368-373):
- `grid` → `GridSampler`
- Everything else → `TPESampler()` (no distinction)

**Documented** in `U02_hyperparameter_tuning.py:65-70` and `08_complex_finetune.json`:
- `grid`, `random`, `tpe`, `cmaes`, `hyperband`

Users who set `"sampler": "random"` or `"sampler": "cmaes"` silently get TPE. There is no warning, no error, no indication that their request was ignored.

---

#### ISSUE-6: `approach` values drift between docs and implementation

**Router** (optuna.py:100-124) recognizes: `"individual"`, `"grouped"`. Everything else falls through to `_optimize_single`.

**Examples** use: `"cross"` (08_complex_finetune.json:92), which silently becomes single-path optimization with train-as-validation (BUG-3).

---

#### ISSUE-7: `eval_mode` values drift between docs and implementation

**`_aggregate_scores`** (optuna.py:397) recognizes: `"best"`, `"avg"`, `"robust_best"`.

**Tests/examples** use: `"mean"` (test_finetune_integration.py:254, 08_complex_finetune.json:55), which silently falls through to the default case (`np.sum` — BUG-2).

---

#### ISSUE-8: Sklearn `_evaluate_model` runs inner cross-validation

**Current behavior**: `sklearn_model.py:453-458` calls `cross_val_score(model, X_val, y_val, cv=3)` inside the Optuna objective. This means:

1. Optuna trial samples params → creates model
2. `_train_model` fits model on X_train_fold → y_train_fold
3. `_evaluate_model` runs 3-fold CV on X_val → y_val (re-fitting the model 3 more times)

This is a double cross-validation: the outer loop is Optuna's fold strategy, the inner loop is sklearn's `cross_val_score`. The model trained in step 2 is discarded; evaluation uses entirely new fits.

**Semantic issues**:
- The trained model's quality is never measured — evaluation trains new models
- 3 extra fits per trial per fold (expensive for large models)
- Results don't reflect the actual training procedure

**All other frameworks** (TF, PyTorch, JAX) evaluate the trained model directly with forward pass / `model.evaluate()`, which is semantically correct.

---

#### ISSUE-9: Meta-model `model__` prefix parameter handling

**`meta_model.py:286-287`**: `force_params` is passed directly to `model.set_params(**force_params)`.

**Test** `test_finetune_integration.py:91` uses `model__alpha`, expecting sklearn's nested parameter syntax. But `set_params(model__alpha=0.1)` on a bare `Ridge()` model will fail — the `model__` prefix is only valid for sklearn's `Pipeline` or meta-estimator `set_params`.

If the MetaModel wraps a `StackingRegressor(final_estimator=Ridge())`, the user would need `final_estimator__alpha` — but the system doesn't translate between nirs4all's finetune_space keys and sklearn's nested parameter paths.

---

### 2.3 Medium-severity issues (missing features, dead code, observability gaps)

#### ISSUE-10: No pruner support

Optuna offers pruners (`MedianPruner`, `SuccessiveHalvingPruner`, `HyperbandPruner`) that terminate unpromising trials early. This is especially valuable for neural networks where each trial costs minutes.

**Current state**: `study.optimize()` (optuna.py:244, 336) has no pruner. No `trial.report()` or `trial.should_prune()` calls exist in any objective function. The BOHB (Bayesian Optimization + HyperBand) workflow is entirely missing.

---

#### ISSUE-11: No multi-phase search support

Users want scenarios like: "random 30 trials for broad exploration, then TPE 70 trials for refinement." Optuna supports this natively (enqueue trials from phase 1 into phase 2's study), but nirs4all has no `phases` config.

---

#### ISSUE-12: No `force_params` / seed-from-prior-run support

Users want to inject known-good parameters as starting points for optimization (e.g. reuse `best_params` from a previous run). Optuna's `study.enqueue_trial()` supports this, but the current system has no way to accept initial parameter suggestions.

---

#### ISSUE-13: No storage/resume lifecycle

`_create_study` (optuna.py:377) creates in-memory studies with no `storage` or `study_name`. This means:
- No resume after interruption
- No cross-run comparison
- No Optuna Dashboard integration

---

#### ISSUE-14: No reproducibility controls

`TPESampler()` (optuna.py:373) is created without a `seed` parameter. There is no `seed` field in `finetune_params`. Runs are non-deterministic.

---

#### ISSUE-15: Trial observability limited to `best_params`

`finetune()` returns only `study.best_params` (optuna.py:250, 342). The full trial history (all trials' params, scores, durations, failure reasons) is discarded.

The prediction payload (base_model.py:2019) stores `best_params` but nothing about the optimization process.

---

#### ISSUE-16: No complex/nested parameter sampling

Two real-world patterns are unsupported:

**a) Nested dict params (TabPFN pattern)**: TabPFN's search space uses slash-separated paths (`"inference_config/FINGERPRINT_FEATURE"`) to represent nested configuration. nirs4all's `_sample_single_parameter` doesn't handle dict-valued params as nested structures — a dict is treated as a single parameter spec (`{"type": "int", "min": ...}`), not as a group of sub-parameters.

**b) Sklearn meta-estimator params (Stacking pattern)**: `StackingRegressor(final_estimator=Ridge(alpha=0.1), estimators=[...])` needs parameters like `final_estimator__alpha`. The double-underscore convention is sklearn's and must be mapped to nirs4all's finetune_space structure.

---

#### ISSUE-17: Grid suitability excludes dict-categorical params

`_is_grid_search_suitable()` (optuna.py:722-725) requires all parameters to be `list` type. A parameter like `{"type": "categorical", "choices": [1, 2, 3]}` is a dict, so grid suitability returns `False` even though the parameter is categorical. The user must use list syntax for grid search to work.

---

#### ISSUE-18: Dead code and debug remnants

- `ModelFactory` imported but never used (optuna.py:31)
- Commented-out debug prints (optuna.py:705, 719, 724, 728)
- Legacy `sample` key (optuna.py:355) alongside `sampler` — should be normalized, not dual-supported
- `SklearnModelController._sample_hyperparameters` (sklearn_model.py:569) calls a parent method that doesn't exist

---

#### ISSUE-19: Tests validate execution, not semantics

Integration tests assert:
- `predictions.num_predictions > 0` (runs without crash)
- `np.isfinite(best_pred['val_score'])` (score is a number)
- `len(model_names) >= N` (multiple models produced)

No test asserts:
- That `best_params` were actually applied to the final model
- That different samplers produce different search behavior
- That `eval_mode` aggregation is mathematically correct
- That pruning terminates trials early
- That multi-phase search transitions between samplers

---

### 2.4 Summary table

| ID | Issue | Severity | Category |
|----|-------|----------|----------|
| BUG-1 | best_params not applied in final training | Critical | Correctness |
| BUG-2 | eval_mode="avg" computes sum not mean | Critical | Correctness |
| BUG-3 | Single-path uses X_train as validation | High | Correctness |
| BUG-4 | Finetuning re-triggered during refit (should use Pass 1 best_params) | Critical | Correctness |
| ISSUE-4 | train_params ranges not sampled | High | Missing feature |
| ISSUE-5 | Documented samplers not implemented | High | Contract drift |
| ISSUE-6 | approach values drift | High | Contract drift |
| ISSUE-7 | eval_mode values drift | High | Contract drift |
| ISSUE-8 | Sklearn double cross-validation | High | Semantic error |
| ISSUE-9 | Meta-model model__ prefix broken | High | Correctness |
| ISSUE-10 | No pruner support | Medium | Missing feature |
| ISSUE-11 | No multi-phase search | Medium | Missing feature |
| ISSUE-12 | No force_params/seed-from-prior | Medium | Missing feature |
| ISSUE-13 | No storage/resume | Medium | Missing feature |
| ISSUE-14 | No reproducibility (seed) | Medium | Missing feature |
| ISSUE-15 | Trial observability limited | Medium | Missing feature |
| ISSUE-16 | No nested/complex param sampling | Medium | Missing feature |
| ISSUE-17 | Grid excludes dict-categorical | Medium | Design limitation |
| ISSUE-18 | Dead code and debug remnants | Low | Cleanup |
| ISSUE-19 | Tests validate execution not semantics | Medium | Test quality |

---

## 3. Roadmap

### Overview

The roadmap is split into 7 phases, ordered by dependency and severity. Each phase is self-contained with clear deliverables, tests, and exit criteria. No phase ships features without corresponding tests and documentation updates.

```
Phase 1: Fix critical bugs (BUG-1, BUG-2, BUG-3, BUG-4)
Phase 2: Contract validation and cleanup (ISSUE-5/6/7/17/18)
Phase 3: Evaluation unification (ISSUE-8) + train_params sampling (ISSUE-4)
Phase 4: Samplers, pruners, and Optuna lifecycle (ISSUE-5/10/13/14)
Phase 5: Advanced features (ISSUE-11/12/16) + meta-model fixes (ISSUE-9)
Phase 6: Trial observability (ISSUE-15)
Phase 7: Documentation, examples, and test hardening (ISSUE-19)
```

---

### Phase 1: Critical Bug Fixes

**Goal**: Restore correctness. After this phase, finetuning actually works.

#### 1.1 Fix best_params application (BUG-1)

**File**: `nirs4all/controllers/models/base_model.py`

Change `launch_training()` line 1146 from:
```python
if mode == "finetune" and best_params is not None:
```
to:
```python
if best_params is not None:
```

This ensures best_params are applied regardless of mode, since `_execute_finetune` already sets `mode="train"` when calling `train()`.

**Test**: Unit test that runs a pipeline with `finetune_params`, intercepts `_get_model_instance` calls in `launch_training`, and asserts `force_params == best_params` (not None, not default).

#### 1.2 Fix eval_mode aggregation (BUG-2)

**File**: `nirs4all/optimization/optuna.py`

Replace `np.sum(scores)` with `np.mean(scores)` at lines 400 and 407.

**Test**: Unit test for `_aggregate_scores` that asserts:
- `_aggregate_scores([0.5, 0.6, 0.7], "avg")` returns `0.6`, not `1.8`
- `_aggregate_scores([0.5, 0.6, 0.7], "best")` returns `0.5`
- `_aggregate_scores([float('inf'), 0.5, 0.6], "robust_best")` returns `0.5`

#### 1.3 Fix single-path validation (BUG-3)

**File**: `nirs4all/optimization/optuna.py`

Replace line 119 with a proper holdout split:
```python
from sklearn.model_selection import train_test_split
X_opt_train, X_val, y_opt_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
```

Pass `X_opt_train, y_opt_train` as training data and `X_val, y_val` as validation in the single-path call.

**Test**: Integration test that verifies single-path optimization uses different data for training and evaluation (check shapes or indices).

#### 1.4 Fix finetuning re-triggered during refit (BUG-4)

**Files**: `nirs4all/controllers/models/base_model.py`, `nirs4all/pipeline/execution/refit/executor.py`

Two complementary fixes:

a) Guard in model controller — skip finetuning when in refit phase (`base_model.py:602-604`):
```python
finetune_params = model_config.get('finetune_params')
is_refit = runtime_context.phase == ExecutionPhase.REFIT

if not is_refit and (mode == "finetune" or (mode == "train" and finetune_params)):
    return self._execute_finetune(...)
```

b) Strip `finetune_params` in `_inject_best_params` (`refit/executor.py`) as defense in depth:
```python
# After applying best_params, remove finetune_params to prevent re-triggering
step.pop("finetune_params", None)
```

**Test**: Integration test that runs a pipeline with `finetune_params` through full CV + refit. Assert that:
- During Pass 1, Optuna runs (finetune triggered)
- During refit, Optuna does NOT run (finetune skipped)
- The refit model uses best_params from Pass 1

#### 1.5 Deliverables

| Deliverable | File(s) |
|-------------|---------|
| Patch: best_params condition | `base_model.py` |
| Patch: eval_mode mean | `optuna.py` |
| Patch: single-path holdout | `optuna.py` |
| Patch: refit skips finetuning (phase guard) | `base_model.py` |
| Patch: strip finetune_params in refit | `refit/executor.py` |
| Unit tests: _aggregate_scores | `tests/unit/optimization/test_optuna_aggregation.py` |
| Unit test: best_params application | `tests/unit/controllers/models/test_best_params_application.py` |
| Integration test: single-path holdout | `tests/integration/pipeline/test_finetune_integration.py` |
| Integration test: refit skips finetuning | `tests/integration/pipeline/test_refit_finetune_skip.py` |

**Exit criteria**: All 4 bugs have regression tests. Tests fail on old code, pass on patched code.

---

### Phase 2: Contract Validation and Cleanup

**Goal**: Eliminate silent fallbacks. Every invalid config value raises a clear error.

#### 2.1 Config validation layer

**File**: `nirs4all/optimization/optuna.py`

Add a `_validate_finetune_params()` method called at the top of `finetune()`:

```python
VALID_SAMPLERS = {"auto", "grid", "tpe", "random"}
VALID_APPROACHES = {"single", "grouped", "individual"}
VALID_EVAL_MODES = {"best", "mean", "robust_best"}

def _validate_finetune_params(self, finetune_params):
    sampler = finetune_params.get("sampler", finetune_params.get("sample", "auto"))
    approach = finetune_params.get("approach", "grouped")
    eval_mode = finetune_params.get("eval_mode", "best")

    if sampler not in VALID_SAMPLERS:
        raise ValueError(f"Unknown sampler '{sampler}'. Valid: {VALID_SAMPLERS}")
    if approach not in VALID_APPROACHES:
        raise ValueError(f"Unknown approach '{approach}'. Valid: {VALID_APPROACHES}")
    if eval_mode not in VALID_EVAL_MODES:
        raise ValueError(f"Unknown eval_mode '{eval_mode}'. Valid: {VALID_EVAL_MODES}")
```

Note: `VALID_SAMPLERS` will be extended in Phase 4 when `cmaes`, `successive_halving`, `hyperband` are implemented.

#### 2.2 Normalize aliases

- `"sample"` → `"sampler"` (normalize at entry, drop the `sample` key)
- `"avg"` → `"mean"` (standardize on `"mean"`)
- `"cross"` → remove (invalid, was never implemented)

#### 2.3 Extend grid suitability for dict-categorical (ISSUE-17)

In `_is_grid_search_suitable()`, add:
```python
if isinstance(param_config, dict):
    if param_config.get("type") == "categorical":
        continue  # Dict-categorical is grid-compatible
    else:
        return False
```

#### 2.4 Remove dead code (ISSUE-18)

- Remove unused `ModelFactory` import from `optuna.py:31`
- Remove commented-out debug prints (lines 705, 719, 724, 728)
- Remove `SklearnModelController._sample_hyperparameters` dead method

#### 2.5 Deliverables

| Deliverable | File(s) |
|-------------|---------|
| Config validation method | `optuna.py` |
| Alias normalization | `optuna.py` |
| Grid suitability for dict-categorical | `optuna.py` |
| Dead code removal | `optuna.py`, `sklearn_model.py` |
| Unit tests: validation rejects unknown values | `tests/unit/optimization/test_optuna_validation.py` |
| Unit tests: grid suitability with dict-categorical | `tests/unit/optimization/test_optuna_grid.py` |

**Exit criteria**: `"sampler": "hyperband"` raises ValueError (until Phase 4 implements it). `"approach": "cross"` raises ValueError. `"eval_mode": "mean"` works, `"avg"` is normalized to `"mean"`.

---

### Phase 3: Evaluation Unification and train_params Sampling

**Goal**: All frameworks evaluate the same way. Users can tune training hyperparameters.

#### 3.1 Fix sklearn _evaluate_model (ISSUE-8)

Replace `cross_val_score` with direct evaluation on the already-trained model:

```python
def _evaluate_model(self, model, X_val, y_val):
    y_val_1d = y_val.ravel() if y_val.ndim > 1 else y_val
    y_pred = model.predict(X_val)

    if is_classifier(model):
        from sklearn.metrics import balanced_accuracy_score
        return -balanced_accuracy_score(y_val_1d, y_pred)
    else:
        return mean_squared_error(y_val_1d, y_pred)
```

This makes sklearn evaluation consistent with TF/PyTorch/JAX: evaluate the trained model's predictions, don't re-fit.

#### 3.2 Unified evaluation contract

Add `metric` and `direction` to `finetune_params`:

```python
"finetune_params": {
    "metric": "mse",       # or "rmse", "mae", "r2", "accuracy", "balanced_accuracy"
    "direction": "minimize", # or "maximize"
    ...
}
```

Defaults: `metric="mse"`, `direction="minimize"` for regression; `metric="balanced_accuracy"`, `direction="maximize"` for classification. Auto-detected from `dataset.task_type`.

Each controller's `_evaluate_model` receives the metric name and computes accordingly, removing the current ad-hoc metric choices.

#### 3.3 Implement train_params sampling (ISSUE-4)

Extend `sample_hyperparameters()` to also sample from `train_params`:

```python
def sample_hyperparameters(self, trial, finetune_params):
    params = {}
    sampled_train_params = {}

    # Sample model_params (existing logic)
    model_params = finetune_params.get("model_params", {})
    for name, config in model_params.items():
        params[name] = self._sample_single_parameter(trial, name, config)

    # Sample train_params (NEW)
    train_params_spec = finetune_params.get("train_params", {})
    for name, config in train_params_spec.items():
        if self._is_sampable(config):
            sampled_train_params[name] = self._sample_single_parameter(
                trial, f"train_{name}", config
            )
        else:
            sampled_train_params[name] = config  # Pass through static values

    return params, sampled_train_params
```

Update objective functions to use the two-dict return: `model_params` for `_get_model_instance(force_params=...)` and `sampled_train_params` for `_train_model(**train_params)`.

The `_is_sampable()` helper checks if a value is a range spec (tuple, range-list, or dict with type/min/max) vs. a static value.

#### 3.4 Deliverables

| Deliverable | File(s) |
|-------------|---------|
| Unified _evaluate_model for sklearn | `sklearn_model.py` |
| metric/direction support in finetune_params | `optuna.py`, all controller files |
| train_params sampling | `optuna.py` |
| _is_sampable helper | `optuna.py` |
| Unit tests: sklearn evaluator consistency | `tests/unit/controllers/models/test_evaluate_model.py` |
| Integration test: train_params actually sampled | `tests/integration/pipeline/test_finetune_train_params.py` |
| Integration test: metric/direction config | `tests/integration/pipeline/test_finetune_metrics.py` |

**Exit criteria**:
- Sklearn evaluation matches TF/PyTorch/JAX semantics (direct prediction, no re-fit).
- `"train_params": {"epochs": ("int", 50, 300)}` is sampled by Optuna across trials.
- `"metric": "rmse"` works consistently across all frameworks.

---

### Phase 4: Samplers, Pruners, and Optuna Lifecycle

**Goal**: Expose Optuna's full power. Support all major samplers, pruning, storage, and seed.

#### 4.1 Implement sampler mapping (ISSUE-5)

Extend `_create_study()`:

```python
SAMPLER_MAP = {
    "grid": lambda fp, seed: GridSampler(self._create_grid_search_space(fp)),
    "random": lambda fp, seed: RandomSampler(seed=seed),
    "tpe": lambda fp, seed: TPESampler(seed=seed),
    "cmaes": lambda fp, seed: CmaEsSampler(seed=seed),
}
```

Update `VALID_SAMPLERS` from Phase 2 to include `"cmaes"`.

#### 4.2 Implement pruner support (ISSUE-10)

Add `pruner` field to `finetune_params`:

```python
PRUNER_MAP = {
    "none": lambda: None,
    "median": lambda: MedianPruner(),
    "successive_halving": lambda: SuccessiveHalvingPruner(),
    "hyperband": lambda: HyperbandPruner(),
}
```

Wire pruner into study creation:
```python
pruner = PRUNER_MAP[pruner_type]()
study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
```

For pruning to work, objectives must call `trial.report(score, step)` and check `trial.should_prune()`. This is most useful for iterative models (neural networks) where `step` maps to epoch.

For sklearn (non-iterative), pruning works across folds: report after each fold and prune if early folds are already worse than the median.

```python
# In grouped objective:
for fold_idx, (train_indices, val_indices) in enumerate(folds):
    ...
    score = controller._evaluate_model(trained_model, X_val_prep, y_val_prep)
    scores.append(score)
    trial.report(self._aggregate_scores(scores, eval_mode), fold_idx)
    if trial.should_prune():
        raise optuna.TrialPruned()
```

#### 4.3 Implement storage/resume (ISSUE-13)

Add optional `storage` and `study_name` to `finetune_params`:

```python
"finetune_params": {
    "storage": "sqlite:///optuna.db",  # or None for in-memory
    "study_name": "pls_tuning_v1",
    "resume": True,  # load_if_exists=True
}
```

Wire into `_create_study`:
```python
study = optuna.create_study(
    direction=direction,
    sampler=sampler,
    pruner=pruner,
    storage=storage,
    study_name=study_name,
    load_if_exists=resume,
)
```

#### 4.4 Implement seed support (ISSUE-14)

Add `seed` to `finetune_params`. Pass to sampler constructors:
```python
seed = finetune_params.get("seed", None)
sampler = SAMPLER_MAP[sampler_type](finetune_params, seed)
```

#### 4.5 Deliverables

| Deliverable | File(s) |
|-------------|---------|
| Sampler mapping (random, tpe, cmaes, grid) | `optuna.py` |
| Pruner mapping + trial.report/should_prune | `optuna.py` |
| Storage/resume support | `optuna.py` |
| Seed support | `optuna.py` |
| Unit tests: each sampler instantiated correctly | `tests/unit/optimization/test_optuna_samplers.py` |
| Unit tests: pruner instantiation | `tests/unit/optimization/test_optuna_pruners.py` |
| Integration test: seeded runs produce same results | `tests/integration/pipeline/test_finetune_reproducibility.py` |
| Integration test: pruning terminates early | `tests/integration/pipeline/test_finetune_pruning.py` |
| Update VALID_SAMPLERS to include new values | `optuna.py` |

**Exit criteria**:
- `"sampler": "random"` uses `RandomSampler`, not TPE.
- `"pruner": "successive_halving"` prunes bad trials.
- `"seed": 42` produces deterministic results.
- `"storage": "sqlite:///..."` persists study.

---

### Phase 5: Advanced Features

**Goal**: Multi-phase search, force_params, and complex/nested parameter structures.

#### 5.1 Multi-phase search (ISSUE-11)

Add `phases` config to `finetune_params`:

```python
"finetune_params": {
    "phases": [
        {"sampler": "random", "n_trials": 30},
        {"sampler": "tpe", "n_trials": 70},
    ],
    "model_params": {...},
}
```

Implementation: run phases sequentially on the same study. After phase 1 completes, the TPE sampler in phase 2 benefits from phase 1's trial history.

```python
def _optimize_multiphase(self, ...):
    # Create study with first phase's sampler
    for phase_idx, phase_config in enumerate(phases):
        sampler = self._create_sampler(phase_config)
        study.sampler = sampler  # Optuna allows changing sampler mid-study
        study.optimize(objective, n_trials=phase_config["n_trials"])
    return study.best_params
```

When `phases` is present, `n_trials` and `sampler` at the top level are ignored (or raise if both are specified).

#### 5.2 Force params / seed from prior run (ISSUE-12)

Add `force_params` to `finetune_params`:

```python
"finetune_params": {
    "force_params": {"n_components": 12, "alpha": 0.01},
    "n_trials": 50,
    ...
}
```

Implementation: enqueue the forced params as trial 0, then run remaining trials normally:

```python
if force_params:
    study.enqueue_trial(force_params)
```

This lets Optuna evaluate the known-good point first, then explore around it. The user's prior knowledge becomes the starting point for Bayesian optimization.

#### 5.3 Nested dict parameter sampling (ISSUE-16a — TabPFN pattern)

Support nested parameter dicts using a separator convention:

```python
"model_params": {
    "inference_config": {
        "FINGERPRINT_FEATURE": [True, False],
        "OUTLIER_REMOVAL_STD": [None, 7.0, 12.0],
    },
    "softmax_temperature": ("float", 0.7, 1.1),
}
```

Implementation in `sample_hyperparameters`:
- Detect dict values that are NOT parameter specs (no `type`/`min`/`max` keys and not a list/tuple)
- Recursively sample sub-parameters with `"parent__child"` naming for Optuna
- Reconstruct nested dict structure from flat sampled params before passing to controller

```python
def _flatten_nested_params(self, params, prefix=""):
    """Flatten nested param dicts into flat keys with __ separator."""
    flat = {}
    for key, value in params.items():
        full_key = f"{prefix}__{key}" if prefix else key
        if isinstance(value, dict) and not self._is_param_spec(value):
            flat.update(self._flatten_nested_params(value, full_key))
        else:
            flat[full_key] = value
    return flat

def _unflatten_params(self, flat_params):
    """Reconstruct nested dict from flat __ separated keys."""
    nested = {}
    for key, value in flat_params.items():
        parts = key.split("__")
        current = nested
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return nested
```

#### 5.4 Sklearn meta-estimator parameter mapping (ISSUE-16b + ISSUE-9)

For sklearn `StackingRegressor`, `VotingRegressor`, and similar meta-estimators, support the sklearn `__` parameter path:

```python
"model_params": {
    "final_estimator__alpha": ("float_log", 1e-4, 1e-1),
    "final_estimator__fit_intercept": [True, False],
}
```

This already works with `set_params(**force_params)` on sklearn meta-estimators. The fix is to ensure:
1. `_get_model_instance` in `meta_model.py` passes params to the correct level
2. The `finetune_space` in `MetaModel` operator is properly mapped to the underlying estimator's `set_params` namespace

For MetaModel wrapping a `StackingRegressor(final_estimator=Ridge())`:
- `model.set_params(final_estimator__alpha=0.1)` — works via sklearn
- The finetune_space must declare `final_estimator__alpha`, not `model__alpha`

#### 5.5 Deliverables

| Deliverable | File(s) |
|-------------|---------|
| Multi-phase optimizer | `optuna.py` |
| force_params / enqueue_trial | `optuna.py` |
| Nested dict flatten/unflatten | `optuna.py` |
| Meta-estimator param mapping | `meta_model.py`, `meta.py` |
| Unit tests: multi-phase runs both phases | `tests/unit/optimization/test_optuna_multiphase.py` |
| Unit tests: force_params enqueued | `tests/unit/optimization/test_optuna_force_params.py` |
| Unit tests: nested param flatten/unflatten | `tests/unit/optimization/test_optuna_nested_params.py` |
| Integration test: TabPFN-style nested tuning | `tests/integration/pipeline/test_finetune_nested_params.py` |
| Integration test: StackingRegressor finetune | `tests/integration/pipeline/test_finetune_stacking.py` |

**Exit criteria**:
- Multi-phase search transitions samplers within one study.
- `force_params` appear as trial 0 in the study.
- Nested dict params (TabPFN-style) are correctly flattened for Optuna and reconstructed for the model.
- `StackingRegressor(final_estimator=Ridge())` can be finetuned via `final_estimator__alpha`.

---

### Phase 6: Trial Observability

**Goal**: Users can inspect and analyze optimization history, not just the final answer.

#### 6.1 Return trial history from finetune (ISSUE-15)

Change `finetune()` return type from `Dict` to a structured `FinetuneResult`:

```python
@dataclass
class FinetuneResult:
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    n_pruned: int
    trials: List[TrialSummary]  # All trials with params, value, duration, state
    study_name: Optional[str]

@dataclass
class TrialSummary:
    number: int
    params: Dict[str, Any]
    value: float
    duration_seconds: float
    state: str  # "COMPLETE", "PRUNED", "FAIL"
```

#### 6.2 Persist trial history in prediction payload

Extend `base_model.py:2019` to store:
```python
'best_params': finetune_result.best_params,
'optimization_summary': {
    'n_trials': finetune_result.n_trials,
    'n_pruned': finetune_result.n_pruned,
    'best_value': finetune_result.best_value,
    'study_name': finetune_result.study_name,
}
```

Full trial history is optionally saved to workspace artifacts (too large for prediction payload).

#### 6.3 Deliverables

| Deliverable | File(s) |
|-------------|---------|
| FinetuneResult/TrialSummary dataclasses | `optimization/optuna.py` |
| Return FinetuneResult from all optimization paths | `optuna.py` |
| Persist optimization_summary in predictions | `base_model.py` |
| Unit tests: FinetuneResult structure | `tests/unit/optimization/test_optuna_result.py` |
| Integration test: trial history accessible post-run | `tests/integration/pipeline/test_finetune_observability.py` |

**Exit criteria**: After a finetune run, users can access per-trial params, scores, and pruning info.

---

### Phase 7: Documentation, Examples, Unified Metric, and Test Hardening

**Goal**: Every feature is documented, every example runs, every semantic contract has a test. The `metric` field unifies evaluation across frameworks.

#### 7.1 Unified `metric` field in finetune_params (deferred from Phase 3.2)

Currently each controller hardcodes its evaluation metric in `_evaluate_model`: sklearn uses MSE/balanced_accuracy, TensorFlow/PyTorch/JAX use their compiled loss. This works but doesn't let users choose the optimization target.

Add `metric` to `finetune_params`:

```python
"finetune_params": {
    "metric": "rmse",          # or "mse", "mae", "r2", "accuracy", "balanced_accuracy"
    "direction": "minimize",   # auto-inferred from metric if omitted
    ...
}
```

**Implementation**:

1. Add a shared metric registry in `optuna.py`: !! WARNING. Nirs4all has already all the necessary classes to manage metrics and directions. Search for it. !!
```python
METRIC_REGISTRY = {
    "mse":               {"fn": mean_squared_error,      "direction": "minimize"},
    "rmse":              {"fn": root_mean_squared_error,  "direction": "minimize"},
    "mae":               {"fn": mean_absolute_error,      "direction": "minimize"},
    "r2":                {"fn": r2_score,                  "direction": "maximize"},
    "accuracy":          {"fn": accuracy_score,            "direction": "maximize"},
    "balanced_accuracy": {"fn": balanced_accuracy_score,   "direction": "maximize"},
}
```

2. Pass `metric` name down to each controller's `_evaluate_model(model, X_val, y_val, metric=...)`. Each controller computes predictions then delegates to the shared registry function.

3. Auto-infer `direction` from metric when not explicitly set. If both `metric` and `direction` are provided, use the explicit `direction`.

4. Defaults: `metric="mse"` for regression (`dataset.task_type == "regression"`), `metric="balanced_accuracy"` for classification. Auto-detected from `dataset.task_type` when `metric` is omitted (preserves current behavior).

**Files**: `optuna.py` (registry + direction inference), `sklearn_model.py`, `tensorflow_model.py`, `torch_model.py`, `jax_model.py` (all `_evaluate_model` signatures).

**Tests**:
- Unit test: each metric function produces expected value
- Unit test: direction auto-inferred correctly per metric
- Integration test: `"metric": "r2", "direction": "maximize"` works end-to-end

#### 7.2 Update examples

| File | Changes |
|------|---------|
| `examples/user/04_models/U02_hyperparameter_tuning.py` | Update sampler list to match implementation. Add examples for pruner, seed, metric, train_params tuning, multi-phase, force_params. Remove references to unimplemented features |
| `examples/user/04_models/U03_stacking_ensembles.py` | Add finetune example for StackingRegressor meta-estimator params |
| `examples/user/04_models/U04_pls_variants.py` | Update any finetune references to use canonical syntax |
| `examples/reference/R01_pipeline_syntax.py` | Update finetune_params section with full supported syntax |
| `examples/reference/R03_all_keywords.py` | Add finetune_params keyword documentation |
| `examples/developer/01_advanced_pipelines/D05_meta_stacking.py` | Add finetune example for meta-model tuning |
| `examples/pipeline_samples/08_complex_finetune.json` | Rewrite with valid approach, eval_mode, sampler values |
| `examples/pipeline_samples/README.md` | Update finetune section with canonical values |

#### 7.3 Update documentation

| File | Changes |
|------|---------|
| `docs/source/user_guide/` | Create or update optimization guide with finetune_params reference |
| CLAUDE.md | Update finetune_params section in pipeline syntax table |

#### 7.4 Semantic test hardening (ISSUE-19)

Add tests that verify optimization semantics, not just execution:

| Test | Assertion |
|------|-----------|
| `test_best_params_actually_applied` | Intercept final model creation, assert params match Optuna's best |
| `test_sampler_type_affects_search` | Grid sampler exhausts all combinations; random doesn't |
| `test_eval_mode_mean_vs_best` | Different eval_modes produce different best_params |
| `test_pruner_reduces_completed_trials` | With pruning, `n_complete < n_trials` |
| `test_seed_determinism` | Same seed + data → same best_params |
| `test_force_params_is_trial_zero` | Forced params appear in trial 0 |
| `test_train_params_vary_across_trials` | Sampled train_params differ between trials |
| `test_nested_params_reconstructed` | Flat Optuna params → correct nested dict for model |
| `test_invalid_sampler_raises` | `"sampler": "invalid"` raises ValueError |
| `test_invalid_approach_raises` | `"approach": "cross"` raises ValueError |

#### 7.5 Run all examples

After all changes, run the full example suite:
```bash
cd nirs4all/examples && ./run.sh
```

Verify all examples pass without errors or deprecation warnings.

#### 7.6 Deliverables

| Deliverable | File(s) |
|-------------|---------|
| Unified metric registry + direction inference | `optuna.py` |
| Metric-aware `_evaluate_model` in all controllers | `sklearn_model.py`, `tensorflow_model.py`, `torch_model.py`, `jax_model.py` |
| Unit tests: metric registry + direction inference | `tests/unit/optimization/test_optuna_metrics.py` |
| Integration test: custom metric end-to-end | `tests/integration/pipeline/test_finetune_metrics.py` |
| Updated user examples | `examples/user/04_models/U02_*.py`, `U03_*.py`, `U04_*.py` |
| Updated reference examples | `examples/reference/R01_*.py`, `R03_*.py` |
| Updated developer examples | `examples/developer/01_*/D05_*.py` |
| Updated pipeline samples | `examples/pipeline_samples/08_*.json`, `README.md` |
| Updated documentation | `docs/source/user_guide/`, CLAUDE.md |
| Semantic integration tests | `tests/integration/pipeline/test_finetune_*.py` |
| Full example suite validation | `./run.sh` pass |

**Exit criteria**: `"metric": "rmse"` works consistently across all frameworks. All examples run. All tests pass. Documentation matches implementation. No silent fallbacks remain. No dead code. No deprecated references.

---

### Roadmap Summary

| Phase | Scope | Issues Resolved | Dependencies |
|-------|-------|----------------|--------------|
| 1 | Critical bug fixes | BUG-1, BUG-2, BUG-3, BUG-4 | None |
| 2 | Contract validation + cleanup | ISSUE-5/6/7/17/18 | Phase 1 |
| 3 | Evaluation unification + train_params | ISSUE-4, ISSUE-8 | Phase 2 |
| 4 | Samplers, pruners, storage, seed | ISSUE-5 (full), ISSUE-10/13/14 | Phase 2 |
| 5 | Advanced features | ISSUE-9/11/12/16 | Phase 3, 4 |
| 6 | Trial observability | ISSUE-15 | Phase 4 |
| 7 | Docs, examples, unified metric, test hardening | ISSUE-19, Phase 3.2 metric field | All previous |

### Files Modified Across All Phases

| File | Phases |
|------|--------|
| `nirs4all/optimization/optuna.py` | 1, 2, 3, 4, 5, 6, 7 |
| `nirs4all/controllers/models/base_model.py` | 1, 6 |
| `nirs4all/pipeline/execution/refit/executor.py` | 1 |
| `nirs4all/controllers/models/sklearn_model.py` | 2, 3, 7 |
| `nirs4all/controllers/models/tensorflow_model.py` | 3, 7 |
| `nirs4all/controllers/models/torch_model.py` | 3, 7 |
| `nirs4all/controllers/models/jax_model.py` | 3, 7 |
| `nirs4all/controllers/models/meta_model.py` | 5 |
| `nirs4all/operators/models/meta.py` | 5 |
| `tests/unit/optimization/` (new) | 1, 2, 3, 4, 5, 6 |
| `tests/integration/pipeline/test_finetune_*.py` | 1, 3, 4, 5, 6, 7 |
| `examples/user/04_models/U02_*.py` | 7 |
| `examples/user/04_models/U03_*.py` | 7 |
| `examples/user/04_models/U04_*.py` | 7 |
| `examples/reference/R01_*.py` | 7 |
| `examples/pipeline_samples/08_*.json` | 7 |
| `examples/pipeline_samples/README.md` | 7 |
| CLAUDE.md | 7 |
