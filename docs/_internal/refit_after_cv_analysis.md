# Refit-After-CV: Two-Pass Architecture Design

## Table of Contents

1. [Problem Reformulation and Objectives](#1-problem-reformulation-and-objectives)
2. [Current State of nirs4all](#2-current-state-of-nirs4all)
3. [The Two-Pass Architecture](#3-the-two-pass-architecture)
4. [Edge Case Analysis](#4-edge-case-analysis)
5. [Selection Semantics: What Gets Selected at Each Level](#5-selection-semantics-what-gets-selected-at-each-level)
6. [Open Questions](#6-open-questions)
7. [Remarks](#7-remarks)
8. [Design Review: Issues, Corrections, and Solutions](#8-design-review-issues-corrections-and-solutions)
   - 8.1 [Factual Inaccuracies](#81-factual-inaccuracies-corrections-to-sections-27)
   - 8.2 [Hidden Flaws and Bugs](#82-hidden-flaws-and-bugs)
   - 8.3 [Redundancies](#83-redundancies-sections-to-consolidate)
   - 8.4 [Missing Features and Gaps](#84-missing-features-and-gaps)
   - 8.5 [Optimizations](#85-optimizations)
   - 8.6 [Potential Deadlocks and Race Conditions](#86-potential-deadlocks-and-race-conditions)
   - 8.7 [Architecture Recommendations Summary](#87-architecture-recommendations-summary)
   - 8.8 [Revised Stacking Refit](#88-revised-stacking-refit-incorporating-solutions)
   - 8.9 [Revised Implementation Phases](#89-revised-implementation-phases)
   - 8.10 [Revised Open Questions](#810-revised-open-questions-replacing-section-6)
   - 8.11 [Refit-Specific Training Parameters (`refit_params`)](#811-refit-specific-training-parameters-refit_params)
   - 8.12 [Caching Architecture for Refit Acceleration](#812-caching-architecture-for-refit-acceleration)
   - 8.13 [Generators Varying the Cross-Validation Splitter](#813-generators-varying-the-cross-validation-splitter)

---

## 1. Problem Reformulation and Objectives

### 1.1 The Canonical ML Training Pattern

The standard, scientifically established pattern for training predictive models is:

1. **Split** data into train and test sets (held-out evaluation).
2. **Cross-validate** on the training set: split training data into K folds, train K models, evaluate on each held-out fold. This serves to (a) estimate generalization performance, and (b) select the best hyperparameters/preprocessing.
3. **Select** the best configuration based on averaged CV scores.
4. **Refit** a single final model on the **entire training set** using the selected configuration.
5. **Evaluate** the final model on the held-out test set for an unbiased performance estimate.
6. **Deploy** the single refitted model.

**Rationale**: Cross-validation is a *selection procedure*, not a *training procedure*. The K fold models are disposable artifacts whose purpose is to estimate out-of-sample performance and guide selection. The production model should leverage ALL available training data, because more data generally produces a better model. Reporting the average of fold validation scores is an estimate of generalization performance, but the deployed model should be trained on more data than any individual fold model saw.

### 1.2 What nirs4all Does Instead

nirs4all currently does **not** perform step 4 (refit). After cross-validation:

- Per-fold models are persisted individually as artifacts.
- Two virtual prediction entries are created: `avg` (simple average of fold predictions) and `w_avg` (weighted average based on fold val scores).
- At prediction time, ALL fold models predict independently, and results are averaged (fold ensemble).
- The "best score" reported to the user is either a single fold's score, `avg`'s score, or `w_avg`'s score, depending on which has the best val score.

**This means nirs4all treats cross-validation as a training procedure that produces an ensemble, not as a selection procedure that guides a final refit.**

### 1.3 Why This Matters

#### 1.3.1 Scientific Rigor

In the chemometrics/NIRS literature, the standard practice is to report the performance of a model trained on the full calibration set and evaluated on an independent validation set. Reporting fold-averaged metrics conflates model selection with model evaluation. When a user reports "RMSE = 0.32" from nirs4all, that number is an average of fold validation scores -- not the performance of the model they would actually deploy.

#### 1.3.2 Suboptimal Use of Training Data

Each fold model is trained on (K-1)/K of the training data. By keeping fold models rather than refitting on all training data, the deployed model has effectively seen less data than it could have.

For spectroscopic applications where sample collection is expensive, wasting 1/K of the training data (20% for 5-fold CV) is significant. A model trained on 100% of the training data will, on average, outperform one trained on 80%.

#### 1.3.3 Deployment Complexity

Deploying K models and averaging their predictions is more complex than deploying a single model. The `.n4a` bundle must store K model artifacts instead of one. Prediction requires K forward passes instead of one.

#### 1.3.4 The Problem Multiplied by Pipeline Complexity

The problem compounds with nirs4all's advanced features:

- **Generators** (`_or_`, `_range_`, `_cartesian_`): These generate N pipeline variants, each running its own CV. The best variant is selected by val score. But after selection, there is no refit -- the user gets the fold ensemble of the selected variant, not a single model trained on all data with the winning configuration.

- **Finetuning** (Optuna): Finds best hyperparameters via search, then trains with those params on CV folds. Again, no final refit on all data. The `nested_cv.md` specification describes a `use_full_train_for_final` option that is **not implemented**.

- **Branching and Stacking**: This is the hardest case. In a stacking pipeline `[branch([SNV+PLS, MSC+RF]), merge("predictions"), Ridge()]`:
  - Base models (PLS, RF) are trained per fold, producing OOF predictions.
  - OOF predictions become training features for the meta-model (Ridge).
  - The meta-model is itself trained per fold.
  - Refit cannot simply retrain everything on all data because the meta-model's features (OOF predictions) would not exist without CV folds. This is a circular dependency.

### 1.4 Objectives

The goal is to implement a **refit-after-CV** mechanism that:

1. After CV-based selection, retrains the winning configuration on ALL training data to produce a single final model.
2. Reports both the CV estimate (for model selection) and the final model's test score (for deployment evaluation).
3. Works correctly for all pipeline topologies: simple, generators, finetuning, branching, and stacking.
4. Is the default behavior (opt-out rather than opt-in), since it is the scientifically correct pattern.
5. Handles the stacking circular dependency correctly.
6. Is backward-compatible with existing pipelines (the fold ensemble behavior should remain available as an option).

---

## 2. Current State of nirs4all

### 2.1 What Exists and Works

#### Cross-Validation Fold Training
- **`CrossValidatorController`** (`controllers/splitters/split.py:168-828`): Correctly handles fold splitting with support for group-aware splitting, absolute sample IDs, and augmentation-safe indexing.
- **`BaseModelController.train()`** (`controllers/models/base_model.py:737-926`): Per-fold training loop with parallel execution support (`joblib`). Each fold model is cloned, trained, evaluated, and persisted with `fold_id`.

#### Fold Averaging
- **`_create_fold_averages()`** (`base_model.py:1347-1509`): Produces `avg` and `w_avg` predictions by having each fold model predict on all data and averaging. Val scores are correctly overridden to prevent leakage (lines 1423-1449).
- **`EnsembleUtils._scores_to_weights()`** (`data/ensemble_utils.py:266-313`): Proper weight computation from fold scores.

#### Prediction Mode (Fold Ensemble)
- **`BaseModelController.train()` in predict mode** (`base_model.py:787-811`): Loads all fold models, generates predictions per fold, averages. This IS the current deployment strategy: fold ensemble averaging.

#### Generator System
- **`PipelineConfigs`** (`pipeline/config/pipeline_config.py:33-93`): Expands `_or_`, `_range_`, `_cartesian_`, etc. into concrete variants before execution. Critically, `expand_spec_with_choices()` tracks which generator decision was made for each variant.
- **`PipelineOrchestrator`** (`pipeline/execution/orchestrator.py:222-268`): Iterates variants sequentially, merges predictions. Selection is done by `Predictions.get_best()` which ranks by val score.

#### Finetuning (Optuna)
- **`OptunaManager`** (`optimization/optuna.py`): Three approaches: `grouped` (evaluates across all folds), `individual` (per-fold optimization), `single` (no folds). Returns `best_params` as a dict (or list of dicts for individual mode).
- **`_execute_finetune()`** (`base_model.py:627-650`): Two-phase: search (Optuna) then train (normal CV with best params). The optimized params are passed via `force_params` to model instantiation. No refit on full data after finding best params.

#### Branching and Stacking
- **`BranchController`** (`controllers/data/branch.py:92-1633`): Creates branch contexts (duplication or separation). Stores `features_snapshot`, `chain_snapshot`, `branch_path` per branch for isolation. Nested branches produce Cartesian products via `_multiply_branch_contexts()`.
- **`MergeController`** (`controllers/data/merge.py:878+`): Dispatches based on merge mode: `"features"` (concatenate transformed features), `"predictions"` (collect OOF predictions), `"concat"` (reassemble disjoint samples), or mixed `{"features": [...], "predictions": [...]}`.
- **`TrainingSetReconstructor`** (`controllers/models/stacking/reconstructor.py:378-644`): Properly reconstructs OOF predictions with no data leakage. Uses sample ID mapping for precise feature matrix assembly. Handles cross-branch prediction collection.

#### Execution Trace
- **`ExecutionTrace`** (`pipeline/trace/execution_trace.py`): Records every step with `step_index`, `operator_type`, `operator_class`, `artifacts`, `branch_path`, and `chain_path`. This is the complete record of what happened during execution.
- **`MinimalPipelineExtractor`** (`pipeline/trace/extractor.py`): Extracts the minimal reproducible pipeline from a trace, used by `BundleLoader` for prediction replay.

#### Retrainer (Explicit, User-Initiated)
- **`Retrainer`** (`pipeline/retrainer.py:299-698`): Three modes (FULL, TRANSFER, FINETUNE). FULL re-runs the entire pipeline from scratch (with CV). TRANSFER reuses preprocessing, trains new model. FINETUNE continues training. **None of these modes implement "refit on all data after CV selection".**

### 2.2 What's Missing

| Gap | Description | Impact |
|-----|-------------|--------|
| **No refit step** | After CV completes, no model is trained on full training data | Deployed models use less data than available |
| **No "final model" concept** | The system has no notion of a single selected-and-retrained model | Users get fold ensembles, not single deployable models |
| **No final test evaluation** | The test score reported is from fold models, not from a model trained on all training data | Reported metrics don't match deployment reality |
| **Generators don't trigger refit** | After selecting best variant via generators, no refit occurs | Best preprocessing+params found but not properly exploited |
| **Finetuning doesn't trigger refit** | `use_full_train_for_final` is specified in `nested_cv.md` but not implemented | Best hyperparams found but final model still uses fold training |
| **Stacking refit not addressed** | No mechanism to handle the OOF dependency when refitting stacking pipelines | The hardest case is completely unhandled |
| **Export produces fold ensemble** | `.n4a` bundles contain K models | Larger bundles, slower prediction, no single-model export option |

### 2.3 Technical Debt

1. **Prediction store conflation**: Individual fold predictions, `avg`, `w_avg`, and (future) refit predictions all live in the same `Predictions` buffer with no clear semantic distinction. The `fold_id` field is overloaded (`0, 1, ..., "avg", "w_avg"`) to distinguish them.

2. **No execution phase concept**: The executor runs steps linearly with no notion of "CV phase" vs "refit phase". Adding refit requires either a new phase concept or a clever re-execution mechanism.

3. **Artifact persistence is fold-coupled**: Models are persisted as `(artifact_id, fold_id)`. There's no slot for a "final" or "refit" model that isn't associated with a fold.

4. **Prediction mode assumes fold ensemble**: `BaseModelController.train()` in predict mode (lines 787-811) hardcodes the pattern of loading per-fold models. Supporting both fold ensemble and single-model prediction requires branching here.

5. **`nested_cv.md` specification drift**: The spec describes `cv_mode`, `param_strategy`, `use_full_train_for_final` -- none of which are implemented. Only `approach` (`grouped`/`individual`/`single`) and `eval_mode` (`best`/`avg`/`robust_best`) exist in code.

6. **Branch-level refit complexity**: The executor's branch iteration (`executor.py:414-599`) dispatches post-branch steps across branch contexts. A refit phase would need to either bypass branches entirely (use winning branch only) or handle them specially.

### 2.4 Partial Existing Mechanisms

- **`fit_on_all` reserved keyword** (`pipeline/steps/parser.py:54`): The parser recognizes `"fit_on_all"` as a reserved keyword in step dicts. However, grepping the codebase shows **no controller or executor logic that acts on it**. This appears to be a placeholder for the refit feature that was never implemented.

- **Prediction mode's dummy folds** (`split.py:299-316`): In predict mode, the splitter creates `[(all_indices, [])]` -- training on all data with no validation. This demonstrates the system CAN produce a model trained on all data; it just only does it during prediction, not during training.

---

## 3. The Two-Pass Architecture

### 3.1 Core Concept

Redesign the execution model to explicitly support two passes:

- **Pass 1 (Selection)**: Current behavior. Run CV, generators, finetuning. Produce fold models, averages, rankings. Purpose: **select** the best configuration.
- **Pass 2 (Refit)**: Automatically triggered after Pass 1. Re-execute the winning configuration on full training data. Purpose: **produce the deployment model**.

The two passes are conceptually separate but execute within the same `nirs4all.run()` call.

### 3.2 The Key Insight: Chain Replay

The refit mechanism leverages a pattern that **already exists** in nirs4all: the prediction mode's chain replay.

When a trained model is loaded for prediction (`BundleLoader`, `BaseModelController` in predict mode), the system:

1. Reads the execution trace to identify the preprocessing chain.
2. Loads stored transformer artifacts.
3. Creates dummy folds: `[(all_indices, [])]` -- one "fold" with all samples, no validation set.
4. Replays each preprocessing step using the stored artifacts (transform, not fit).
5. Loads fold model artifacts and predicts.

**Refit uses the same logic, but in training mode**: instead of loading stored artifacts (transform-only), it **re-fits** transformers on all training data and **re-trains** the model on all training data.

The mechanism:

```
Prediction mode (existing):     Load artifact → transform(X)     → Load fold model → predict(X)
Refit mode (new):               Clone operator → fit_transform(X_train_all) → Clone model → fit(X_train_all, y_train_all)
```

Both follow the same pipeline path (same steps, same order). The difference is whether operators are loaded (predict) or freshly trained (refit).

### 3.3 Orchestrator-Level Design

In `PipelineOrchestrator.execute()`, after the main variant loop (line 268):

```
# Pass 1 complete. All variants have run with CV.
best_variant = run_predictions.get_best()
best_config = extract_winning_config(best_variant)

# Pass 2: Refit
if refit_enabled:
    refit_result = execute_refit(best_config, dataset, context)
    run_predictions.add(refit_result, fold_id="final")
```

### 3.4 The `execute_refit()` Function

A new function (or method on `PipelineOrchestrator`) that:

1. Takes the winning pipeline configuration (steps + params + preprocessing chain).
2. Creates a modified execution where:
   - Splitter steps are replaced with a dummy `[(all_train_indices, [])]` split (train on everything, no validation fold).
   - Model steps use the winning params (finetuned params if applicable).
   - All other steps (transformers, filters, etc.) execute normally in training mode (fit + transform).
3. Evaluates the refitted model on the test set.
4. Returns a prediction entry with `fold_id="final"`.

The function detects the pipeline topology and dispatches to the appropriate refit strategy:

```
execute_refit(best_config, dataset, context):
    topology = analyze_topology(best_config)

    if topology.has_stacking:
        return execute_stacking_refit(best_config, dataset, context, topology)
    elif topology.has_mixed_merge:
        return execute_mixed_merge_refit(best_config, dataset, context, topology)
    else:
        return execute_simple_refit(best_config, dataset, context)
```

### 3.5 Simple Refit (Non-Stacking Pipelines)

For pipelines without `merge: "predictions"`, the refit is straightforward:

1. Take the winning variant's expanded config.
2. Replace the splitter with a single dummy fold: `[(all_train_indices, [])]`.
3. Inject finetuned params (if any) into the model step.
4. Execute the pipeline in training mode.
5. The single "fold" trains on all training data.
6. Evaluate on the test set.
7. Persist as `fold_id="final"`.

This covers: simple pipelines, pipelines with generators, pipelines with finetuning, and branch+merge("features") topologies.

### 3.6 Stacking-Specific Refit (Two-Pass)

For pipelines containing `merge: "predictions"` (stacking), simple refit breaks because:

- The meta-model's training features are **OOF predictions** from base models.
- OOF predictions only exist when base models are trained with CV folds.
- If we train base models on all data (no folds), there are no OOF predictions to feed the meta-model.

The solution is a three-sub-pass refit:

**Pass 2a (OOF generation)**:
- Re-run base models with CV (same fold scheme as Pass 1) to generate fresh OOF predictions for meta-model training.
- This is necessary because the meta-model REQUIRES OOF features.

**Pass 2b (Meta-model refit)**:
- Train the meta-model on the OOF predictions from Pass 2a using ALL training samples.
- The meta-model is now fitted on all available OOF features.

**Pass 2c (Base model refit)**:
- Retrain base models on ALL training data (no CV).
- These are the deployment base models.
- The meta-model from Pass 2b is kept as-is.

**At prediction time**:
- New data goes through deployment base models (from Pass 2c).
- Their predictions become features for the meta-model (from Pass 2b).

This is scientifically sound: the meta-model was trained on OOF predictions (no leakage), and the deployment base models use all available training data.

**Distribution shift caveat**: The meta-model was trained on OOF predictions (which have higher variance than in-sample predictions), but at deployment it receives predictions from all-data base models (slightly different distribution). In practice, this shift is small and well-documented in the ML literature (Wolpert's stacked generalization, Kaggle best practices). For NIRS applications with typical dataset sizes (100-2000 samples), this is the right compromise.

### 3.7 Configuration

```python
# Global default (recommended: on)
nirs4all.run(pipeline, dataset, refit=True)

# Disable for research/comparison
nirs4all.run(pipeline, dataset, refit=False)

# Fine-grained control
nirs4all.run(pipeline, dataset, refit={
    "enabled": True,
    "stacking_meta_training": "in_sample",  # default: base model predictions on training data
    # Alternative: "oof" — use Pass 1 OOF predictions for meta-model training
    "report_both": True,                    # report CV score AND refit score
    "default_refit_params": {               # global refit param overrides for all models
        "verbose": 1
    },
})

# Per-model refit training params (see Section 8.11)
pipeline = [
    SNV(),
    {
        "model": MyNeuralNet(),
        "train_params": {"epochs": 100, "lr": 0.001},      # CV training
        "refit_params": {"epochs": 1000, "lr": 0.0001,     # Refit: more epochs, lower LR
                         "warm_start": True},                # Initialize from best fold weights
    }
]
```

### 3.8 Result Object Changes

```python
result = nirs4all.run(pipeline, dataset)

# Current behavior (unchanged)
result.best_score      # Best test score across all entries
result.best_rmse       # Best RMSE
result.top(5)          # Top 5 predictions (includes refit models)

# New accessors -- outermost model
result.final           # The outermost refit model's prediction entry
result.final_score     # The outermost refit model's test score
result.cv_best         # The best CV entry (for comparison)
result.cv_best_score   # CV-estimated performance

# New accessors -- per-model independent refits (Section 3.10)
result.models                 # Dict of all independently refit models
result.models["PLS"].score    # PLS standalone test score (with its best chain)
result.models["PLS"].chain    # "SNV → PLS(10)" -- the winning chain for PLS
result.models["RF"].score     # RF standalone test score (with its best chain)

# Export exports the outermost refit model
result.export("model.n4a")              # Exports the final/meta model
result.models["PLS"].export("pls.n4a")  # Export a specific model's standalone refit
```

### 3.9 Bundle Changes

The `.n4a` bundle format needs to support:
- **Single model** (from refit): one artifact, direct prediction.
- **Fold ensemble** (current): K artifacts, averaged prediction.
- **Stacking with refit**: base model artifacts (single per branch) + meta-model artifact.

The `BundleGenerator` (`pipeline/bundle/generator.py`) uses the `fold_id="final"` artifact exclusively. Fold ensemble export is no longer supported (see Section 3.11).

### 3.10 Per-Model Independent Refit

Every model node in the pipeline -- not just the outermost/final model -- gets its own independent refit. After Pass 1, the system identifies, for each model node, the variant where that model performed best and refits it with that variant's complete chain.

#### Why Per-Model Refit?

In a stacking pipeline with generators:

```python
pipeline = [
    {"_or_": [SNV(), MSC()]},
    KFold(n_splits=5),
    {"branch": [
        [PLSRegression(n_components=10)],
        [RandomForestRegressor()],
    ]},
    {"merge": "predictions"},
    {"model": Ridge()},
]
```

After Pass 1, suppose:
```
Variant 0 (SNV): PLS val=0.85, RF val=0.82, Ridge val=0.90
Variant 1 (MSC): PLS val=0.80, RF val=0.88, Ridge val=0.92
```

Three different models exist, and each has its own "best" variant:
- **PLS** performs best with SNV (val=0.85 vs 0.80).
- **RF** performs best with MSC (val=0.88 vs 0.82).
- **Ridge** (stacking) performs best with MSC (val=0.92 vs 0.90).

Per-model independent refit produces **three independently deployable models**:

| Model | Best Local Chain | Refit |
|-------|-----------------|-------|
| PLS | Variant 0: SNV → PLS(10) | SNV fit all → PLS fit all |
| RF | Variant 1: MSC → RF | MSC fit all → RF fit all |
| Ridge (stacking) | Variant 1: MSC → {PLS, RF} → Ridge | Stacking refit (2a/2b/2c) using MSC chain |

Notice that PLS is refit **twice**:
1. **Standalone refit**: with its best local chain (SNV → PLS). This is the best PLS model.
2. **Stacking context refit**: within the stacking refit of Ridge (Variant 1 uses MSC, so PLS is retrained with MSC in Pass 2c). This PLS model is a component of the stacking deployment.

These are two different models with two different preprocessing chains and two different purposes.

#### How It Works

After Pass 1 completes:

1. **Enumerate all model nodes** in the pipeline (across all variants and branches).
2. **For each model node**, identify the variant where that model achieved the best val score.
3. **Refit each model** with its own best chain (simple refit -- no folds, all training data).
4. **Then**, perform the stacking refit for the outermost model (which re-runs base models in the stacking context).

The result object provides access to every model's independent refit:

```python
result = nirs4all.run(pipeline, dataset)

# The stacking meta-model's refit (outermost)
result.final                  # Ridge refit with full stacking pipeline
result.final_score            # Ridge test score

# Per-model independent refits
result.models                 # Dict of all independently refit models
result.models["PLS"].score    # PLS standalone test score (with its best chain)
result.models["RF"].score     # RF standalone test score (with its best chain)
result.models["Ridge"].score  # Same as result.final_score
```

#### When a Pipeline Has No Generators

If there are no generators (only one variant), all models share the same chain. Per-model independent refit still runs: each model is simply refit with the single chain on all data. There is no divergence between models.

#### When a Pipeline Has No Stacking

If there is no stacking (no `merge: "predictions"`), the per-model refit is straightforward: there is only one model (or one winning branch's model). It gets refit once.

### 3.11 Artifact Strategy: Refit-Only Persistence

During Pass 1 (CV), fold models and fold transformers are trained in memory for scoring purposes but are **NOT persisted** as artifacts to the workspace store. Only refit artifacts (`fold_id="final"`) are persisted.

#### What Changes

| Aspect | Current Behavior | New Behavior |
|--------|-----------------|-------------|
| **Fold model artifacts** | Persisted to workspace store (K artifacts per model step) | Transient -- exist in memory during Pass 1 only |
| **Fold transformer artifacts** | Persisted | Transient |
| **Refit model artifact** | Does not exist | Persisted as `fold_id="final"` (one per model step) |
| **`avg`/`w_avg` predictions** | Computed from persisted fold models | Still computed during Pass 1 (in-memory fold models predict before being discarded) |
| **Fold prediction entries** | Persisted in prediction store | Still persisted (prediction records are lightweight -- they store scores and y_pred arrays, not model binaries) |
| **Bundle export** | K fold model binaries | Single refit model binary |

#### Why This Is Better

1. **Storage**: A 5-fold PLS pipeline currently stores 6 artifacts (5 fold models + 1 transformer). With refit-only persistence, it stores 2 (1 refit model + 1 refit transformer). For stacking with 5 folds and 3 branch models: 18 artifacts → 4 artifacts.

2. **Simplicity**: No ambiguity about which model to use for prediction. The refit model is the only model. No fold ensemble logic needed at prediction time.

3. **Consistency**: The persisted model IS the deployment model. No gap between what's stored and what's deployed.

4. **Bundle size**: `.n4a` bundles are K times smaller. A single model artifact instead of K fold artifacts.

#### What Remains in Memory During Pass 1

Fold models still need to exist temporarily during Pass 1 for:
- Computing per-fold validation scores (model selection).
- Computing `avg` and `w_avg` predictions (all fold models predict on all data, then averaged).
- Generating OOF predictions for stacking (`TrainingSetReconstructor`).

After Pass 1 completes and the refit phase begins, fold models are discarded.

#### Impact on `avg`/`w_avg`

The `avg` and `w_avg` prediction entries remain as CV-phase scoring entries. They still appear in `result.top()` and `result.cv_best`. They are computed from in-memory fold models during Pass 1. They simply don't link to persisted model artifacts anymore.

#### Impact on Prediction Mode

Current prediction mode loads per-fold model artifacts and ensembles them. With refit-only persistence:
- Prediction mode loads the single refit model artifact.
- No ensemble logic needed.
- Simpler, faster, smaller bundles.

---

## 4. Edge Case Analysis

### 4.1 Taxonomy of Pipeline Topologies

Before analyzing edge cases, we establish a taxonomy of pipeline topologies that the refit mechanism must handle. Each topology introduces different constraints:

| Topology | Key Characteristic | Refit Strategy |
|----------|-------------------|----------------|
| **Simple** | No branches, no generators | Simple refit |
| **Generators** | Multiple variants, one winner | Simple refit on winning variant |
| **Finetuning** | Optuna-optimized params | Simple refit with optimized params |
| **Generators + Finetuning** | Variants × optimized params | Simple refit (params embedded in variant) |
| **Branch + merge("features")** | Parallel transforms, concatenated | Simple refit (no OOF dependency) |
| **Branch + merge("predictions")** | Stacking, OOF dependency | Stacking refit (2a/2b/2c) |
| **Branch + merge(mixed)** | Some features, some predictions | Hybrid refit |
| **Branch (no merge)** | Independent branch predictions | Select winning branch, simple refit |
| **Nested branches** | Branches inside branches | Depends on inner/outer merge modes |
| **Separation branches** | Disjoint sample subsets | Per-branch refit |
| **Multi-source** | Per-source preprocessing | Per-source refit |
| **Generators + stacking** | Variants × stacking | Stacking refit on winning variant |
| **Generators + nested + finetuning + mixed merge** | All combined | Full topology analysis required |

The following sections analyze each topology in detail.

### 4.2 Simple Pipeline

```python
pipeline = [SNV(), KFold(n_splits=5), PLSRegression(n_components=10)]
```

**Pass 1 execution**:
- SNV fits on training data, transforms.
- KFold creates 5 folds.
- PLS trains per fold: 5 fold models + avg + w_avg.

**Selection**: Best of {fold_0, ..., fold_4, avg, w_avg} by val score.

**Refit**:
1. Clone SNV → `fit_transform(X_train_all)`.
2. Skip splitter (or use dummy single fold).
3. Clone PLS(n_components=10) → `fit(X_train_all_preprocessed, y_train_all)`.
4. Predict on X_test → evaluate → `fold_id="final"`.

**Complexity**: Trivial. One preprocessing step, one model, no branching.

**What needs to be stored for refit**: The original pipeline config (steps + params). No special extraction needed -- the pipeline is already fully specified.

### 4.3 Pipeline with Generators

```python
pipeline = [
    {"_or_": [SNV(), MSC(), Detrend()]},
    KFold(n_splits=5),
    PLSRegression(n_components=10),
]
```

**Pass 1 execution**:
- `PipelineConfigs` expands to 3 variants:
  - Variant 0: `[SNV(), KFold(5), PLS(10)]`
  - Variant 1: `[MSC(), KFold(5), PLS(10)]`
  - Variant 2: `[Detrend(), KFold(5), PLS(10)]`
- Orchestrator runs all 3 sequentially. Each produces fold models + avg + w_avg.
- `generator_choices` stored per variant: `[{"_or_": SNV}]`, `[{"_or_": MSC}]`, `[{"_or_": Detrend}]`.

**Selection**: Best across all 3 variants by val score → say Variant 1 (MSC).

**Refit**:
1. Retrieve Variant 1's `expanded_config`: `[MSC(), KFold(5), PLS(10)]`.
2. Execute refit using that config: MSC fit on all train → PLS(10) fit on all preprocessed train.
3. Evaluate on test → `fold_id="final"`.

**Complexity**: Low. The generator expansion is already done; the winning variant's config is fully specified. Refit simply replays one variant without folds.

**What needs to be stored for refit**: The winning variant's `expanded_config` (already stored in `WorkspaceStore` by `executor.py:148-158`).

### 4.4 Pipeline with Finetuning

```python
pipeline = [
    SNV(),
    KFold(n_splits=5),
    {"model": PLSRegression(), "finetune_params": {
        "n_trials": 50,
        "model_params": {"n_components": ("int", 1, 20)}
    }},
]
```

**Pass 1 execution**:
1. Phase 1 (Optuna): `OptunaManager.finetune()` → `best_params = {"n_components": 8}`.
2. Phase 2 (Train): `train(best_params={"n_components": 8})` → 5 fold models with PLS(8).

**Selection**: Best of fold models / avg / w_avg by val score.

**Refit**:
1. SNV fit on all train data.
2. PLS(n_components=8) fit on all preprocessed train data.
3. Evaluate on test → `fold_id="final"`.

**Key question**: Where are the finetuned params stored?

Currently, `best_params` is passed to `train()` → `launch_training()` → `_get_model_instance(force_params=best_params)`. The resulting model artifact IS the trained model with those params. But the `best_params` dict itself is not explicitly persisted in a retrievable location.

**Requirement**: The refit function needs access to `best_params`. Options:
- **Store in execution trace**: Add a `best_params` field to `ExecutionStep` for model steps.
- **Store in prediction metadata**: Add `best_params` to the prediction record.
- **Re-extract from Optuna study**: If the Optuna study is stored, query `study.best_params`. But this requires persisting the study.
- **Store in WorkspaceStore pipeline record**: Add a `best_params` field alongside `expanded_config`.

The cleanest option is storing `best_params` in the execution trace at the model step level. This keeps the information close to its source and is naturally available during refit.

### 4.5 Generators + Finetuning (The Double Constraint)

```python
pipeline = [
    {"_or_": [SNV(), MSC()]},
    KFold(n_splits=5),
    {"model": PLSRegression(), "finetune_params": {
        "n_trials": 50,
        "model_params": {"n_components": ("int", 1, 20)}
    }},
]
```

**Pass 1 execution**:
- Variant 0 (SNV): Optuna finds `n_components=8`. CV trains PLS(8) per fold.
- Variant 1 (MSC): Optuna finds `n_components=12`. CV trains PLS(12) per fold.

**Selection**: Best variant by val score → say Variant 0 (SNV + PLS(8)).

**The "double constraint"**: The refit must use BOTH the winning preprocessing (SNV, not MSC) AND the winning params (n_components=8, not 12).

**Why this is NOT actually complex**: These two constraints are not independent. They are nested:

1. Generator expansion creates variants. Each variant is a complete pipeline.
2. Finetuning runs WITHIN each variant, independently.
3. Variant 0's Optuna finds params optimal for the SNV preprocessing chain.
4. Variant 1's Optuna finds params optimal for the MSC preprocessing chain.
5. Selection picks the best **variant** (which already embeds its finetuned params).

The winning variant IS the winning chain AND the winning params, inseparably. There is no separate "select the chain, then separately select the params" step. The params are specific to the chain because Optuna optimized them within that chain's context.

**Refit**:
1. Retrieve Variant 0's expanded config + its `best_params = {"n_components": 8}`.
2. SNV fit on all train → PLS(n_components=8) fit on all preprocessed train.
3. Evaluate on test → `fold_id="final"`.

**The double constraint resolves itself because the unit of selection is the variant, not the individual step.**

### 4.6 Duplication Branches with Feature Merge

```python
pipeline = [
    KFold(n_splits=5),
    {"branch": [
        [SNV(), PCA(n_components=10)],       # Branch 0: SNV features
        [MSC(), PCA(n_components=10)],       # Branch 1: MSC features
    ]},
    {"merge": "features"},                    # Concatenate transformed features
    {"model": Ridge()},                       # Train on merged features
]
```

**Pass 1 execution**:
- Branch 0: SNV → PCA → 10-dim features.
- Branch 1: MSC → PCA → 10-dim features.
- Merge: concatenate → 20-dim features.
- Ridge trains per fold on 20-dim features.

**Selection**: Best of Ridge's fold models / avg / w_avg.

**Refit**:
1. Branch 0: SNV fit on all train → PCA fit on all train → features_0.
2. Branch 1: MSC fit on all train → PCA fit on all train → features_1.
3. Merge: concatenate (features_0, features_1) → 20-dim features.
4. Ridge fit on all train → evaluate on test → `fold_id="final"`.

**No OOF dependency**: The merge is feature-based, not prediction-based. Transformers produce features deterministically -- there's no leakage risk in fitting transformers on all data and transforming all data.

**Complexity**: Moderate. Must replay ALL branches (not just the winning one), because all branches contribute features to the merge. The refit mechanism must:
- Detect that `merge: "features"` does not introduce an OOF dependency.
- Replay each branch's preprocessing chain independently.
- Re-execute the merge step.
- Train the final model on merged features.

### 4.7 Stacking (Branch + Merge Predictions)

```python
pipeline = [
    KFold(n_splits=5),
    {"branch": [
        [SNV(), PLSRegression(n_components=10)],
        [MSC(), RandomForestRegressor()],
    ]},
    {"merge": "predictions"},
    {"model": Ridge()},
]
```

**Pass 1 execution**:
- Branch 0: SNV → PLS trains per fold → 5 fold models → OOF predictions.
- Branch 1: MSC → RF trains per fold → 5 fold models → OOF predictions.
- Merge: `TrainingSetReconstructor` assembles OOF predictions → 2-column meta-features.
- Ridge trains per fold on meta-features.

**Selection**: Best of Ridge's fold models / avg / w_avg.

**Why simple refit fails**: If we train PLS and RF on all data (no folds), they predict in-sample on training data. These in-sample predictions are optimistically biased and cannot be used as meta-model training features (leakage).

**Stacking refit (three sub-passes)**:

Pass 2a: Re-run PLS and RF with CV (same fold scheme) on all training data → generate fresh OOF predictions.

Pass 2b: Collect OOF predictions → 2-column meta-features. Train Ridge on ALL meta-features (single model, no folds). This is the deployment meta-model.

Pass 2c: Retrain PLS on all train data (SNV fit on all → PLS fit on all). Retrain RF on all train data (MSC fit on all → RF fit on all). These are the deployment base models.

**Deployment prediction flow**:
```
new_data → SNV.transform() → PLS(from 2c).predict() → pred_0
new_data → MSC.transform() → RF(from 2c).predict()  → pred_1
[pred_0, pred_1] → Ridge(from 2b).predict() → final_prediction
```

**Per-model independent refit** (Section 3.10): In addition to the stacking refit above, PLS and RF each get their own independent refit:
- PLS standalone refit: SNV fit all → PLS fit all (best PLS model, independently deployable).
- RF standalone refit: MSC fit all → RF fit all (best RF model, independently deployable).

These standalone refits happen BEFORE the stacking refit. The stacking refit (2a/2b/2c) then produces a SEPARATE set of base models specifically for the stacking deployment context.

**Persisted artifacts** (Section 3.11 -- refit-only):
- PLS standalone refit model + SNV standalone refit transformer
- RF standalone refit model + MSC standalone refit transformer
- PLS stacking-context refit model (from 2c) + SNV stacking-context transformer (from 2c)
- RF stacking-context refit model (from 2c) + MSC stacking-context transformer (from 2c)
- Ridge all-data model (from 2b, trained on OOF features)
- Fold models from Pass 1 are NOT persisted.

### 4.8 Generators + Stacking

```python
pipeline = [
    {"_or_": [SNV(), MSC()]},                     # 2 outer variants
    KFold(n_splits=5),
    {"branch": [
        [PLSRegression(n_components=10)],          # Branch 0
        [RandomForestRegressor()],                 # Branch 1
    ]},
    {"merge": "predictions"},
    {"model": Ridge()},
]
```

**Pass 1 execution**:
- Variant 0 (SNV): SNV → {PLS, RF branches} → OOF merge → Ridge per fold.
- Variant 1 (MSC): MSC → {PLS, RF branches} → OOF merge → Ridge per fold.

Suppose the scores are:
```
Variant 0 (SNV): PLS val=0.85, RF val=0.82, Ridge val=0.90
Variant 1 (MSC): PLS val=0.80, RF val=0.88, Ridge val=0.92
```

**Per-model selection** (Section 3.10): Each model independently selects its best variant:
- PLS best: Variant 0 (SNV) → val=0.85
- RF best: Variant 1 (MSC) → val=0.88
- Ridge best: Variant 1 (MSC) → val=0.92

**Refit produces three independently deployable models**:

1. **PLS standalone refit**: SNV fit all → PLS fit all. Deployable as a standalone model.
2. **RF standalone refit**: MSC fit all → RF fit all. Deployable as a standalone model.
3. **Ridge stacking refit**: Uses Variant 1 (MSC, the best for Ridge):
   - 2a: Re-run MSC → {PLS, RF} with CV → OOF predictions.
   - 2b: Train Ridge on OOF features (all samples).
   - 2c: Retrain MSC (fit all), PLS (fit all), RF (fit all) for stacking deployment.

Notice: PLS's standalone refit uses SNV (its best chain), but PLS's stacking-context refit uses MSC (Ridge's best chain). These are two different PLS models for two different purposes.

**The generator selects per-model, not globally.** Each model identifies its own winning variant. The stacking refit then uses the meta-model's winning variant for the stacking context.

### 4.9 Mixed Merge (Features + Predictions)

```python
pipeline = [
    KFold(n_splits=5),
    {"branch": [
        [SNV(), PCA(n_components=10)],                    # Branch 0: produces features
        [MSC(), PLSRegression(n_components=10)],          # Branch 1: produces predictions
    ]},
    {"merge": {"features": [0], "predictions": [1]}},    # Mixed merge
    {"model": Ridge()},
]
```

**Pass 1 execution**:
- Branch 0: SNV → PCA → 10-dim features (no model, no OOF).
- Branch 1: MSC → PLS trains per fold → OOF predictions.
- Merge: concatenate features_0 (10 cols) and oof_predictions_1 (1 col) → 11-dim meta-features.
- Ridge trains per fold on 11-dim meta-features.

**Selection**: Best of Ridge's fold models / avg / w_avg.

**Refit (hybrid)**:

The refit must distinguish between branches contributing features vs predictions:

- **Branch 0 (features, no OOF dependency)**: Simple refit. SNV fit on all → PCA fit on all → features_0.
- **Branch 1 (predictions, OOF dependency)**: Stacking refit.
  - 2a: Re-run PLS with CV → OOF predictions.
  - 2c: Retrain MSC (fit all) → PLS (fit all) for deployment.

Then:
- 2b: Merge (features_0, oof_predictions_1) → train Ridge on all meta-features.

**How to detect the hybrid case**: The merge config `{"features": [0], "predictions": [1]}` explicitly identifies which branches contribute features vs predictions. The refit mechanism can inspect this:
- Branches in the `"features"` list → simple refit (no OOF dependency).
- Branches in the `"predictions"` list → stacking refit.

### 4.10 Nested Branches with Inner Models

```python
pipeline = [
    {"_cartesian_": [[SNV(), MSC()], [MinMaxScaler(), StandardScaler()]]},
    # Produces 4 outer variants: SNV+MinMax, SNV+Standard, MSC+MinMax, MSC+Standard
    KFold(n_splits=5),
    {"branch": [
        [Transform_A(), {"model": PLSRegression(), "finetune_params": {
            "model_params": {"n_components": ("int", 1, 20)}
        }}],
        [Transform_B(), {"model": RandomForestRegressor(), "finetune_params": {
            "model_params": {"n_estimators": ("int", 50, 500)}
        }}],
    ]},
    {"merge": "predictions"},
    {"model": Ridge()},
]
```

**Pass 1 execution** (for each of the 4 outer variants):
1. Outer preprocessing: e.g., SNV → MinMaxScaler.
2. Fold splitting.
3. Branch 0: Transform_A → PLS. Optuna finds `n_components=8`. CV trains PLS(8) per fold → OOF predictions.
4. Branch 1: Transform_B → RF. Optuna finds `n_estimators=200`. CV trains RF(200) per fold → OOF predictions.
5. Merge: OOF predictions from PLS and RF → 2-col meta-features.
6. Ridge trains per fold on meta-features.

Total: 4 outer variants × (2 inner branches each with finetuning) = 4 Ridge models to compare.

**Selection**: Best of the 4 outer variants' Ridge val scores → say Variant 2 (MSC + MinMaxScaler).

**What gets selected and what gets preserved**:
- **Selected**: The outer preprocessing (MSC + MinMaxScaler). This is a generator selection.
- **Preserved (not selected against each other)**: BOTH inner branches. Because `merge: "predictions"` means both branches contribute to the meta-model. The inner branches are not alternatives -- they are cooperating components of the stacking architecture.
- **Preserved (from finetuning)**: Each inner branch's finetuned params. For the winning variant (Variant 2):
  - Branch 0: PLS with `n_components` optimized for MSC+MinMax+Transform_A chain.
  - Branch 1: RF with `n_estimators` optimized for MSC+MinMax+Transform_B chain.

**Refit** (stacking refit on winning variant):
1. 2a: MSC fit all → MinMaxScaler fit all → {Branch 0: Transform_A → PLS(n_components=8), Branch 1: Transform_B → RF(n_estimators=200)} with CV → OOF predictions.
2. 2b: Merge OOF → train Ridge on all meta-features.
3. 2c: Retrain MSC → MinMaxScaler → {Branch 0: Transform_A → PLS(8), Branch 1: Transform_B → RF(200)} on all training data.

**The triple constraint**: outer preprocessing (generator) × inner models (all preserved for stacking) × inner model params (finetuning). Resolves as:
- Generator selects the outer variant → the inner pipeline is fixed.
- Inner branches are ALL kept (stacking requires all).
- Inner finetuned params are specific to the winning variant (already optimized in that context).

### 4.11 The Monster: Stacking with Nested Branches Containing Stacking

```python
pipeline = [
    KFold(n_splits=5),
    {"branch": [
        # Branch 0: a sub-stacking pipeline
        [
            {"branch": [
                [SNV(), PLSRegression(n_components=5)],
                [MSC(), PLSRegression(n_components=10)],
            ]},
            {"merge": "predictions"},
            {"model": Ridge()},           # Inner meta-model
        ],
        # Branch 1: a simple pipeline
        [Detrend(), RandomForestRegressor()],
    ]},
    {"merge": "predictions"},             # Outer stacking merge
    {"model": GradientBoostingRegressor()},  # Outer meta-model
]
```

**Pipeline topology**:
- Outer stacking: two branches → OOF merge → GBR.
- Branch 0 is ITSELF a stacking pipeline: two sub-branches → OOF merge → Ridge.
- Branch 1 is simple: Detrend → RF.

**Pass 1 execution**:
1. Outer CV splits into 5 folds.
2. Branch 0: for each outer fold:
   - Inner branch: SNV → PLS(5) per fold → OOF preds.
   - Inner branch: MSC → PLS(10) per fold → OOF preds.
   - Inner merge: OOF predictions → Ridge input.
   - Inner Ridge trains per fold → OOF predictions for outer merge.
3. Branch 1: Detrend → RF per fold → OOF predictions for outer merge.
4. Outer merge: OOF predictions from [Ridge (Branch 0), RF (Branch 1)] → 2-col meta-features.
5. GBR trains per fold.

**Selection**: Best of GBR's fold models / avg / w_avg.

**Refit (nested stacking)**:

This is the most complex case because there are TWO levels of OOF dependency:
- The inner merge (`merge: "predictions"` in Branch 0) requires OOF predictions from inner base models (SNV+PLS, MSC+PLS).
- The outer merge (`merge: "predictions"`) requires OOF predictions from Branch 0's Ridge and Branch 1's RF.

**Refit strategy**:

Pass 2a (outer level): Re-run both branches with CV to generate OOF predictions for the outer meta-model.
- For Branch 0: re-run the inner stacking pipeline (which itself requires inner CV for OOF). The inner stacking runs its own Pass 2a/2b/2c within the outer fold context.
- For Branch 1: re-run Detrend → RF with CV.
- Collect outer OOF predictions from Ridge in Branch 0 and RF in Branch 1.

Pass 2b (outer level): Train GBR on outer OOF features (all training samples).

Pass 2c (outer level): Retrain deployment models on all data:
- Branch 0: inner stacking refit → deployment Ridge + deployment PLS models.
- Branch 1: Detrend → RF on all data.

**This is recursive**: the refit of a stacking pipeline that contains a stacking sub-pipeline requires a recursive application of the stacking refit strategy. The inner stacking refit happens WITHIN the outer Pass 2a and Pass 2c.

**Depth of recursion**: In practice, nesting is typically 1-2 levels deep. But the algorithm must handle arbitrary depth:

```
refit(pipeline, data):
    if pipeline has merge("predictions"):
        # Stacking refit
        Pass 2a: for each branch:
            refit(branch_pipeline, data)  # RECURSIVE
            collect OOF predictions
        Pass 2b: train meta-model on OOF
        Pass 2c: for each branch:
            refit(branch_pipeline, data)  # RECURSIVE (all-data version)
    else:
        # Simple refit
        replace splitter with dummy
        train all steps on all data
```

### 4.12 Branches Without Merge (Independent Predictions)

```python
pipeline = [
    KFold(n_splits=5),
    {"branch": [
        [SNV(), PLSRegression(n_components=10)],
        [MSC(), RandomForestRegressor()],
    ]},
    # No merge step → branches produce independent prediction entries
]
```

**Pass 1 execution**:
- Branch 0: SNV → PLS per fold → predictions with `branch_id=0`.
- Branch 1: MSC → RF per fold → predictions with `branch_id=1`.
- No merge → each branch's predictions are separate entries in the prediction store.

**Selection**: Best across all branches and folds by val score → say Branch 1 (MSC+RF).

**Refit**: Since branches are independent (no merge), the refit only trains the **winning branch**:
1. MSC fit on all train → RF fit on all preprocessed train.
2. Evaluate on test → `fold_id="final"`.

Branch 0's models are discarded for deployment (they were exploration candidates).

**This is the only case where branch selection happens**: when branches have no merge step, they are alternatives (like generator variants), and the best one is selected. When branches have a merge step, ALL branches are preserved because they contribute to the merged result.

### 4.13 Separation Branches (by_metadata, by_tag)

```python
pipeline = [
    KFold(n_splits=5),
    {"branch": {"by_metadata": "site"}},  # Split by site: "lab_A", "lab_B", "lab_C"
    PLSRegression(n_components=10),
    {"merge": "concat"},
]
```

**Pass 1 execution**:
- Samples split by `site` metadata → 3 disjoint groups.
- Branch lab_A: PLS trains per fold on lab_A samples only.
- Branch lab_B: PLS trains per fold on lab_B samples only.
- Branch lab_C: PLS trains per fold on lab_C samples only.
- Merge "concat": predictions from all branches reassembled in original sample order.

**Selection**: Per-branch selection (each branch has its own model quality).

**Refit**: Each branch is refitted independently on its own full training subset:
1. Branch lab_A: PLS fit on all lab_A train samples.
2. Branch lab_B: PLS fit on all lab_B train samples.
3. Branch lab_C: PLS fit on all lab_C train samples.

**Key insight**: Separation branches require **per-branch refit**, not a single global refit. Each branch has its own data subset and its own model.

**At prediction time**: New samples are routed by their `site` metadata to the corresponding branch model.

### 4.14 Multi-Source Pipelines (by_source)

```python
pipeline = [
    {"branch": {"by_source": True, "steps": {
        "NIR": [SNV(), PLSRegression(n_components=10)],
        "markers": [MinMaxScaler(), Ridge()],
    }}},
    {"merge": {"sources": "concat"}},
    {"model": GradientBoostingRegressor()},
]
```

**Pass 1 execution**:
- Source "NIR": SNV → PLS per fold → predictions.
- Source "markers": MinMaxScaler → Ridge per fold → predictions.
- Merge: source predictions concatenated → 2-col meta-features.
- GBR trains per fold on meta-features.

This is structurally equivalent to stacking (Section 4.7) because each source's model produces predictions that become meta-features.

**Refit**: Stacking refit with per-source base models:
- 2a: Re-run {NIR: SNV→PLS, markers: MinMax→Ridge} with CV → OOF predictions.
- 2b: Train GBR on OOF features (all samples).
- 2c: Retrain NIR: SNV(fit all)→PLS(fit all), markers: MinMax(fit all)→Ridge(fit all).

### 4.15 Edge Case: Generators Inside Branches

```python
pipeline = [
    KFold(n_splits=5),
    {"branch": [
        [SNV(), {"_or_": [PLSRegression(5), PLSRegression(10), PLSRegression(15)]}],
        [MSC(), RandomForestRegressor()],
    ]},
    {"merge": "predictions"},
    {"model": Ridge()},
]
```

**Question**: Does the generator inside a branch expand to 3 variants of the entire pipeline, or does it create sub-variants within the branch?

**Current behavior**: Generators are expanded by `PipelineConfigs` BEFORE execution. The branch definition is part of the pipeline, so the generator inside the branch would expand the entire pipeline to 3 variants:
- Variant 0: `[KFold, branch([[SNV, PLS(5)], [MSC, RF]]), merge, Ridge]`
- Variant 1: `[KFold, branch([[SNV, PLS(10)], [MSC, RF]]), merge, Ridge]`
- Variant 2: `[KFold, branch([[SNV, PLS(15)], [MSC, RF]]), merge, Ridge]`

Each variant runs independently as a full stacking pipeline.

**Selection**: Best of the 3 variants' Ridge val scores.

**Refit**: Stacking refit on the winning variant. The generator selected which PLS n_components is used in Branch 0. Both branches are preserved for stacking.

**This is equivalent to Section 4.8** (Generators + Stacking) -- the generator just happens to be inside a branch definition rather than outside it. The expansion still creates full pipeline variants.

### 4.16 Edge Case: Finetuning Inside Stacking Branches

```python
pipeline = [
    KFold(n_splits=5),
    {"branch": [
        [SNV(), {"model": PLSRegression(), "finetune_params": {
            "model_params": {"n_components": ("int", 1, 20)}
        }}],
        [MSC(), {"model": RandomForestRegressor(), "finetune_params": {
            "model_params": {"n_estimators": ("int", 50, 500)}
        }}],
    ]},
    {"merge": "predictions"},
    {"model": Ridge()},
]
```

**Pass 1 execution**:
- Branch 0: SNV → Optuna finds PLS(n_components=8) → CV trains PLS(8) → OOF predictions.
- Branch 1: MSC → Optuna finds RF(n_estimators=200) → CV trains RF(200) → OOF predictions.
- Merge: OOF predictions → 2-col meta-features.
- Ridge trains per fold.

**Selection**: Best of Ridge's fold models / avg / w_avg.

**Key question**: For the stacking refit, do we re-run Optuna or reuse the finetuned params?

**Answer: Reuse the finetuned params.** Reasons:
1. The finetuned params were optimized for the specific preprocessing chain within each branch. They are valid.
2. Re-running Optuna would be expensive and might produce different params (non-determinism).
3. The purpose of refit is to train on all data with the SAME configuration, not to re-optimize.

**Refit**:
- 2a: Re-run {Branch 0: SNV→PLS(n_components=8), Branch 1: MSC→RF(n_estimators=200)} with CV → OOF predictions. Optuna is NOT re-run; the params are injected.
- 2b: Train Ridge on OOF (all samples).
- 2c: Retrain deployment models with same params on all data.

**Requirement**: The stacking refit must be able to inject `best_params` into the inner model steps during Pass 2a and Pass 2c. This requires storing the finetuned params somewhere accessible (see Section 4.4).

### 4.17 Edge Case Summary Matrix

| Case | Generator Selection | Branch Selection | Finetuned Params | OOF Dependency | Refit Strategy |
|------|:---:|:---:|:---:|:---:|---|
| **Simple** | - | - | - | No | Simple refit |
| **Generators** | Yes | - | - | No | Simple refit on winner |
| **Finetuning** | - | - | Reuse | No | Simple refit with params |
| **Gen + Finetune** | Yes | - | Reuse (variant-specific) | No | Simple refit on winner |
| **Branch + features** | - | No (all kept) | - | No | Simple refit all branches |
| **Branch + predictions** | - | No (all kept) | - | Yes | Stacking refit |
| **Branch + mixed** | - | No (all kept) | - | Partial | Hybrid refit |
| **Branch (no merge)** | - | Yes (best branch) | - | No | Simple refit winner branch |
| **Gen + stacking** | Yes | No (all kept) | - | Yes | Stacking refit on winner variant |
| **Nested + finetune + stacking** | Yes | No (all kept) | Reuse | Yes | Stacking refit, recursive if nested stacking |
| **Separation** | - | Per-branch | - | No | Per-branch refit |
| **Multi-source** | - | No (all kept) | - | Yes (if predictions) | Stacking refit per source |
| **Gen inside branch** | Yes (expands to variants) | No (all kept) | - | Yes (if predictions) | Stacking refit on winner variant |
| **Finetune inside stacking** | - | No (all kept) | Reuse (per-branch) | Yes | Stacking refit with injected params |

---

## 5. Selection Semantics: What Gets Selected at Each Level

### 5.1 The Core Question

The user writes a pipeline. After Pass 1, the system must determine what to retrain on all data. This involves two levels of selection:

1. **Per-model selection**: For each model node, which variant produced the best version of THAT model?
2. **Stacking context selection**: For the meta-model, which variant produced the best stacking result?

The fundamental principles:

> **Every model gets its own refit with its own best chain.** Generators select per-model, not globally. Branches with merge preserve all branches. Branches without merge select the best branch. Finetuning params are fixed at the variant level. Fold artifacts are not persisted -- only refit artifacts.

### 5.2 Selection Granularity

Different pipeline elements create different levels of selection:

| Element | Creates | Selection | What Happens at Refit |
|---------|---------|-----------|----------------------|
| **Generator** (`_or_`, `_range_`, `_cartesian_`) | N variants (entire pipeline copies) | Per-model: each model finds its best variant independently | Each model refit with its own best variant's chain |
| **Branch + merge** | N branches (parallel pipelines) | No selection -- all branches kept | All branches refitted (standalone + stacking context) |
| **Branch (no merge)** | N branches (parallel pipelines) | Best branch (by model val score) | Only winning branch refitted |
| **Finetuning** | N Optuna trials | Best params (by Optuna objective) | Winning params reused (not re-optimized) |
| **Separation branch** | N data subsets | Per-subset (each branch has its own model) | Each subset refitted independently |

### 5.3 The Double Constraint: Chain x Params

When generators and finetuning coexist:

```
Variant 0: [SNV → PLS(n=8)]    ← Optuna found n=8 within SNV context
Variant 1: [MSC → PLS(n=12)]   ← Optuna found n=12 within MSC context
```

The "double constraint" (which chain? which params?) resolves automatically because:
1. Each variant runs its own finetuning independently.
2. The params are optimized for that specific variant's preprocessing chain.
3. Selection picks the best VARIANT, which already embeds its params.

**There is no "select chain first, then select params" two-step decision.** The variant is the atomic unit of selection.

### 5.4 Branch Selection vs Branch Preservation

This is the most important distinction for the refit mechanism:

**Branches with merge ("features", "predictions", "concat")**: ALL branches are preserved. The merge combines their outputs. For refit, all branches must be retrained (they are cooperative components).

**Branches without merge**: Branches produce independent predictions. The best branch is selected. Only the winning branch is refitted (they are competing alternatives).

**Why this matters for complex pipelines**: Consider:

```python
pipeline = [
    {"_or_": [SNV(), MSC()]},              # Generator: competing
    {"branch": [                            # Branches: cooperating (merged)
        [PLS(10)],
        [RF()],
    ]},
    {"merge": "predictions"},
    {"model": Ridge()},
]
```

The `_or_` creates 2 **competing** variants. The `branch` creates 2 **cooperating** branches within each variant. At refit:
- Generator selects ONE variant (competition resolved).
- Both branches within the winning variant are preserved (cooperation maintained).

### 5.5 Selection in Nested Topologies

For deeply nested topologies, selection propagates from the outermost level inward:

```
Level 0: Generator selects outer variant (competing)
  Level 1: Branches cooperate (all kept for merge)
    Level 2: Inner generator selects inner variant (competing within each branch)
      Level 3: Inner branches cooperate (all kept for inner merge)
```

At each level, the rule is the same:
- **Generators**: select one variant → discard the rest.
- **Branches with merge**: keep all → refit all.
- **Branches without merge**: select one → discard the rest.

The refit mechanism walks the winning path through the topology tree:
1. Start at outermost level.
2. If it's a generator level: take the winning variant's config.
3. If it's a branch-with-merge level: recurse into ALL branches.
4. If it's a branch-without-merge level: recurse into the winning branch only.
5. At model level: refit with stored params.

### 5.6 Per-Model Scoring and Selection

In nested topologies, **each model is scored independently** for its own standalone refit:

```python
pipeline = [
    {"_or_": [SNV(), MSC()]},
    {"branch": [
        [PLS(10)],          # Inner model 1
        [RF()],             # Inner model 2
    ]},
    {"merge": "predictions"},
    {"model": Ridge()},     # Outer model (meta-model)
]
```

After Pass 1:
```
Variant 0 (SNV): PLS val=0.85, RF val=0.82, Ridge val=0.90
Variant 1 (MSC): PLS val=0.80, RF val=0.88, Ridge val=0.92
```

**Per-model selection**: Each model finds its own best variant:
- PLS best variant: Variant 0 (SNV) → standalone refit: SNV → PLS
- RF best variant: Variant 1 (MSC) → standalone refit: MSC → RF
- Ridge best variant: Variant 1 (MSC) → stacking refit using MSC chain

**For the stacking context**, the meta-model's best variant determines the chain for base models within the stacking deployment. So the stacking deployment uses MSC for both PLS and RF (because Ridge's best variant is Variant 1), even though PLS's standalone best is Variant 0.

This is important because:
- The inner models' individual scores might disagree with the meta-model's score.
- PLS has a better standalone model with SNV, but when PLS and RF work together with Ridge, MSC is better overall.
- Each model gets its own standalone refit AND participates in the stacking refit under the meta-model's winning chain.

### 5.7 When Inner Models SHOULD Be Selected

There is one case where inner models are genuinely selected: **generators inside branches without merge**.

```python
pipeline = [
    {"branch": [
        [SNV(), {"_or_": [PLS(5), PLS(10), PLS(15)]}],
        [MSC(), RF()],
    ]},
    # No merge
]
```

After expansion:
- Variant 0: Branch 0 = SNV+PLS(5), Branch 1 = MSC+RF
- Variant 1: Branch 0 = SNV+PLS(10), Branch 1 = MSC+RF
- Variant 2: Branch 0 = SNV+PLS(15), Branch 1 = MSC+RF

Without merge, each branch produces independent predictions. Selection compares across ALL variants AND ALL branches: 6 candidates (3 variants × 2 branches).

The winner might be Variant 1, Branch 0 (SNV+PLS(10)). Refit trains only that one pipeline on all data.

---

## 6. Open Questions

### 6.1 How to Store Finetuned Params for Refit?

Finetuned params (`best_params` from Optuna) are currently passed in memory from `_execute_finetune()` to `train()` but are not persisted in a structured, retrievable location. The refit function needs to access these params without re-running Optuna.

**Options**:
1. Store in `ExecutionTrace.steps[model_step].best_params` → cleanest, closest to source.
2. Store in `WorkspaceStore.pipelines` record → accessible via database query.
3. Store in prediction metadata → pollutes prediction records with model configuration details.
4. Extract from the trained model artifact → fragile, model-dependent.

**Recommendation**: Option 1 (ExecutionTrace). The trace already records per-step metadata; adding `best_params` is a natural extension.

### 6.2 Should Refit Re-Run Optuna in Pass 2a (Stacking)?

In stacking refit Pass 2a, base models are re-run with CV to generate OOF predictions. If the base models have finetuning, should Optuna be re-run?

**Arguments for re-running**: The full training data is different from the fold-reduced data in Pass 1. Different data might yield different optimal params.

**Arguments against re-running**: (a) Expensive. (b) Non-deterministic -- might get different params, making the refit inconsistent with what was selected. (c) The purpose of Pass 2a is to generate OOF features, not to re-optimize.

**Recommendation**: Do NOT re-run Optuna. Use the params from Pass 1. If the user wants to re-optimize with full data, they can run a new experiment.

### 6.3 What Happens When Refit Fails?

The refit model might fail to train (convergence failure, memory error, timeout). Since fold artifacts are not persisted (Section 3.11), there is no fold ensemble to fall back to. Options:
- Report the failure and return `result.final = None`?
- Retry with different settings?
- Re-run Pass 1 for the winning variant only, persisting fold models as a fallback?

**Recommendation**: Report the failure clearly (`result.final = None`, `result.final_score = None`). The CV-phase prediction entries (`avg`, `w_avg`) still exist as score references but cannot be deployed without model artifacts. Log a warning. If the user needs a deployable model, they re-run with `refit=False` which falls back to the legacy fold ensemble behavior (persisting fold artifacts).

### 6.4 How to Handle Augmented Samples in Refit?

If the pipeline includes augmentation steps (e.g., `GaussianNoise`, `Mixup`), should augmented samples be regenerated for the refit training set?

- During Pass 1, augmentation is applied per fold. The augmented samples are different for each fold.
- During refit, there are no folds. Should augmentation be applied to all training data?

**Recommendation**: Yes. Augmentation should run normally during refit. The augmentation controller already handles the all-data case. The refit simply operates like a single fold that includes all training data.

### 6.5 What About Filters/Exclusions in Refit?

If the pipeline includes exclusion steps (`{"exclude": YOutlierFilter()}`), should they re-run during refit?

- If an outlier is excluded during Pass 1, it should also be excluded during refit (consistency).
- But the outlier detection might produce slightly different results when fitted on all data vs on fold subsets.

**Recommendation**: Re-run filters during refit (fit on all training data). This ensures the refit model sees the same quality of data, adapted to the full training set. The slightly different outlier detection on the full set is acceptable -- it's more representative of the deployment scenario.

### 6.6 How to Handle the `nested_cv.md` Spec?

The `nested_cv.md` specification describes `cv_mode`, `param_strategy`, `use_full_train_for_final` -- none of which are implemented. Should the refit implementation align with this spec, replace it, or extend it?

**Recommendation**: The refit implementation supersedes `use_full_train_for_final`. The rest of `nested_cv.md` (nested CV modes) is a separate concern and can be implemented later. Don't try to solve both at once.

### 6.7 How to Handle Separation Branches in Refit with Stacking?

Consider:

```python
pipeline = [
    KFold(n_splits=5),
    {"branch": {"by_metadata": "site"}},
    {"model": PLSRegression(n_components=10)},
    {"merge": "concat"},
    {"model": Ridge()},
]
```

After separation branch + merge("concat") + outer model: is this stacking?

The merge is "concat" (reassembling disjoint samples), not "predictions" (stacking). The predictions from each site's PLS are concatenated in the original sample order, and Ridge trains on these. This is NOT stacking in the OOF sense -- the concat merge doesn't create meta-features from OOF predictions.

However, if the inner PLS predictions are used as features for Ridge, there IS a leakage concern: PLS predicted on its own training data within each site.

**This needs clarification**: Does `merge: "concat"` after separation branches feed raw predictions to the next model? If so, it has the same leakage issue as stacking. If it feeds the ORIGINAL features (just reassembled), there's no leakage.

### 6.8 How to Handle Multiple Models at Different Pipeline Levels?

```python
pipeline = [
    SNV(),
    PLSRegression(n_components=10),   # Model 1: intermediate predictions
    Ridge(),                            # Model 2: final model
]
```

If a pipeline has multiple model steps (without explicit branching/merging), how does refit handle them?

Currently, the executor treats each model step independently. Model 1 trains per fold, Model 2 trains per fold using Model 1's output.

**Question**: Should refit train BOTH models on all data? Or only the last model? What about Model 2's features -- they come from Model 1's predictions, which have an OOF-like dependency.

**Recommendation**: This is effectively an implicit stacking pipeline. If Model 2 receives features derived from Model 1's predictions, the same OOF dependency applies. The refit mechanism should detect this "sequential model" pattern and treat it like stacking.

### 6.9 How Deep Can Nesting Go? Is Recursion Safe?

Section 4.11 showed that nested stacking requires recursive refit. How deep can this go?

In practice, 1-2 levels of stacking is the maximum useful depth (beyond that, returns diminish and overfitting increases). But the algorithm should handle arbitrary depth.

**Concern**: Recursive stacking refit is expensive. Each level of nesting multiplies the number of training passes. A 2-level stacking with 5-fold CV requires:
- Level 2 (inner): 5 base model trainings per fold × 5 folds = 25 trainings
- Level 1 (outer): 5 inner stacking runs × 5 folds = 125 trainings
- Refit adds: inner stacking refit + outer stacking refit

**Recommendation**: Limit recursion depth. Warn the user if the refit pass count exceeds a threshold (e.g., 100 model trainings).

### 6.10 Preprocessing Refit: Fit on All Data or Use Stored Artifacts?

When refitting, should preprocessing transformers (SNV, MSC, MinMaxScaler) be:

(a) **Re-fitted on all training data** (freshly trained), or
(b) **Reused from Pass 1** (loaded from stored artifacts)?

**Arguments for (a)**: The transformer fitted on 100% of training data is more representative than one fitted on 80% (one fold's training set). For transformers like MinMaxScaler, the min/max values from all data are better estimates. For MSC, the mean spectrum from all data is a better reference.

**Arguments for (b)**: Simpler. Guarantees exact same preprocessing as CV. No risk of subtle differences.

**Recommendation**: (a) Re-fit on all training data. This is consistent with the refit philosophy: use ALL available data. The preprocessing is part of the pipeline; if the model is trained on all data, the preprocessing should be too. This is also what the `Retrainer` does in FULL mode.

---

## 7. Remarks

### 7.1 The Topology Detection Problem

The refit mechanism must inspect the pipeline topology to determine the correct strategy. This requires a **topology analyzer** that can identify:

- Presence of `merge: "predictions"` (stacking).
- Presence of `merge: "features"` or `merge: "concat"` (non-stacking merge).
- Presence of mixed merge (hybrid).
- Nesting depth (stacking within stacking).
- Separation branches (per-branch refit).
- Branches without merge (branch selection).

This analyzer must work on the **expanded config** (after generator expansion), because generators change the topology. It should produce a `PipelineTopology` descriptor that the refit engine can dispatch on.

### 7.2 The Stacking Refit is Expensive

For a stacking pipeline with 5-fold CV and 2 base models:
- Pass 1: 2 base models × 5 folds + 1 meta-model × 5 folds = 15 training runs.
- Pass 2a: 2 base models × 5 folds = 10 training runs (re-generating OOF).
- Pass 2b: 1 meta-model training run.
- Pass 2c: 2 base models × 1 = 2 training runs.
- Total refit cost: 13 additional training runs (87% overhead).

For deep learning models or large random forests, this is significant. The refit should be:
- **Interruptible**: User can cancel the refit without losing Pass 1 results.
- **Skippable**: `refit=False` disables it entirely.
- **Visible**: The webapp should show refit as a distinct progress phase with estimated time.

### 7.3 The Refit Does NOT Change Selection

An important invariant: **the refit pass never changes which configuration was selected**. Pass 1 determines the winner. Pass 2 retrains the winner. Even if the refitted model has a worse test score than expected (due to training set noise, different convergence, etc.), it remains the deployed model.

The CV estimate from Pass 1 is the selection criterion. The refit test score is the deployment evaluation. These are two different things and should be reported separately.

### 7.4 Relationship to Existing Retrainer

The `Retrainer` class already implements pipeline replay in several modes. The refit mechanism is conceptually a new mode:

```
Retrainer modes:
- FULL:     Re-run entire pipeline from scratch (with CV)
- TRANSFER: Reuse preprocessing, train new model
- FINETUNE: Continue training existing model
- REFIT:    Re-run winning variant on all data (new mode, no CV)
```

The refit mechanism could be implemented as a new `RetrainMode.REFIT` that:
- Uses the execution trace to identify the winning pipeline.
- Replays with dummy single fold.
- Handles stacking specially (recursive stacking refit).

This avoids duplicating the trace-based replay infrastructure.

### 7.5 Backward Compatibility

The refit changes include a **breaking change** in artifact persistence:

- **Fold model artifacts are no longer persisted** by default. This eliminates fold ensemble deployment. Users who need the legacy behavior (fold ensemble with persisted fold models) can use `refit=False`, which reverts to the current behavior entirely.
- Existing `avg` and `w_avg` prediction entries continue to exist as CV-phase scoring references.
- A new `fold_id="final"` entry is added for each refit model.
- `result.best_score` returns the best refit model's test score.
- `result.export()` exports the refit model (single model per model node, not fold ensemble).
- `result.models` provides access to every model's independent refit.
- Old `.n4a` bundles (containing fold artifacts) remain loadable for prediction -- the `BundleLoader` supports both formats.

### 7.6 The `fold_id="final"` Design

Adding `fold_id="final"` as a new entry type is cleaner than overloading existing fold IDs. This entry:
- Has `partition="test"` (always evaluated on the held-out test set).
- Has no `val_score` (there's no validation fold in refit).
- Has `test_score` (the deployment evaluation metric).
- Links to a single model artifact (the only persisted artifact for that model step).
- Is produced by the refit pass, never by Pass 1.
- Multiple `fold_id="final"` entries may exist per model node if the model has both a standalone refit and a stacking-context refit (distinguished by a `refit_context` field: `"standalone"` vs `"stacking"`).

### 7.7 Implementation Priority

Based on the edge case analysis, the implementation can be phased:

**Phase 1** (covers 80% of use cases):
- Simple refit for non-stacking pipelines.
- Handles: simple, generators, finetuning, generators+finetuning, branch+features merge.
- Topology detection: if `merge: "predictions"` not present → simple refit.
- Estimated scope: changes to orchestrator, model controller, predictions, result.

**Phase 2** (covers 15% of use cases):
- Stacking refit (Pass 2a/2b/2c).
- Handles: branch+predictions merge, generators+stacking, mixed merge.
- Topology detection: if `merge: "predictions"` present → stacking refit.
- Estimated scope: new stacking refit logic, integration with merge controller.

**Phase 3** (covers 5% of use cases):
- Nested stacking refit (recursive).
- Separation branch refit (per-branch).
- Handles: stacking within stacking, by_metadata/by_tag separation.
- Estimated scope: recursive refit, per-branch dispatch.

### 7.8 What Not To Do

- **Don't re-run Optuna during refit**. Use the params from Pass 1.
- **Don't make refit a separate API call** (`nirs4all.refit()`). It should be automatic within `nirs4all.run()`.
- **Don't persist fold model artifacts**. They are transient scoring tools, not deployment artifacts. Only refit models are persisted.
- **Don't change the semantics of `avg`/`w_avg` prediction entries**. These continue to work as CV-phase scoring references. They simply no longer link to persisted model artifacts.
- **Don't try to solve nested CV and refit simultaneously**. Nested CV is a separate concern (described in `nested_cv.md`). Implement refit first, then layer nested CV on top.

---

## 8. Design Review: Issues, Corrections, and Solutions

### 8.1 Factual Inaccuracies (Corrections to Sections 2–7)

#### 8.1.1 `fit_on_all` Is NOT a Placeholder (Section 2.4)

**Document claim**: *"The parser recognizes 'fit_on_all' as a reserved keyword... no controller or executor logic that acts on it. This appears to be a placeholder for the refit feature that was never implemented."*

**Actual state**: `fit_on_all` is **fully implemented** in `controllers/transforms/transformer.py` (lines 125–170). It controls whether *transformers* (not models) fit on all data (train+test) vs train-only. This is an unsupervised preprocessing option (e.g., fitting `StandardScaler` on the full data distribution), **not** a model refit mechanism. The document's claim that it's an unimplemented placeholder is incorrect.

**Impact on design**: `fit_on_all` for transformers is orthogonal to the refit-after-CV feature. The refit mechanism should respect the existing `fit_on_all` setting for transformers during the refit pass: if a transformer has `fit_on_all=True`, it should be fitted on all data (train+test) during refit, matching Pass 1 behavior.

#### 8.1.2 `fold_id` Is String-Based, Not Integer (Section 2.3)

**Document claim (Section 2.3.1)**: *"The `fold_id` field is overloaded (`0, 1, ..., "avg", "w_avg"`)"*

**Actual state**: `fold_id` values are **strings throughout**: `"fold_0"`, `"fold_1"`, ..., `"avg"`, `"w_avg"`. The prediction creation function (`predictions.py:145`) explicitly converts: `"fold_id": str(fold_id) if fold_id is not None else ""`. The integer `fold_idx` used during iteration is a different variable from the stored `fold_id` string.

**Impact on design**: The proposed `fold_id="final"` fits naturally into the string-based naming convention. No schema change needed — but the document's description of the current overloading is misleading. The actual issue is not type overloading (integers mixed with strings) but semantic overloading (CV folds vs aggregation entries vs refit entries sharing the same field).

#### 8.1.3 `best_params` IS Already Stored in Predictions (Sections 4.4, 6.1)

**Document claim (Section 6.1)**: *"Finetuned params... are not persisted in a structured, retrievable location"* and recommends storing them in ExecutionTrace.

**Actual state**: The prediction record schema already includes a `best_params` field (`predictions.py:122`), and `base_model.py:1963` stores it: `best_params=prediction_data['best_params']`. The `WorkspaceStore` schema also has `best_params (JSON)` in the predictions table.

**Impact on design**: The `best_params` storage problem is **already solved**. The refit function can retrieve `best_params` by querying the prediction store for the winning variant's prediction records. The recommended Option 1 (ExecutionTrace) is unnecessary — use the existing prediction store instead. This simplifies implementation.

#### 8.1.4 `result.best_score` Uses Val Score, Not Test Score (Section 3.8)

**Document claim**: *"`result.best_score` — Best test score across all entries"*

**Actual state**: The `RunResult` ranking (in `api/result.py`) uses **validation partition** scores for ranking, falling back to test scores. `best_score` returns the best-ranked entry's primary score, which is val-based.

**Impact on design**: After refit, the `fold_id="final"` entry has no validation fold (it's trained on all train data). It has `val_score=None` and a `test_score`. The ranking logic must be updated to handle entries without val scores. Either:
- (a) `fold_id="final"` entries are ranked by `test_score` while CV entries are ranked by `val_score`, or
- (b) `result.best_score` and `result.final_score` are explicitly decoupled — `best_score` remains CV-based (for selection), `final_score` is refit-based (for deployment).

**Recommendation**: Option (b). Keep `result.best_score` as the CV selection metric (unchanged). Add `result.final_score` as the deployment metric. Don't mix selection and evaluation in the same ranking.

### 8.2 Hidden Flaws and Bugs

#### 8.2.1 Pass 2a Is Redundant — OOF Predictions Already Exist from Pass 1

**The flaw**: Section 3.6 describes Pass 2a as: *"Re-run base models with CV (same fold scheme as Pass 1) to generate fresh OOF predictions for meta-model training."* This re-runs the ENTIRE base model CV — `N_base_models × K_folds` training runs — to produce OOF predictions that are **identical** to what Pass 1 already produced (assuming same data, same folds, same params, same preprocessing).

**Why it's wasteful**: For a pipeline with 3 base models and 5-fold CV, Pass 2a costs 15 additional model trainings. The document itself (Section 7.2) acknowledges this is 87% overhead, but presents it as inevitable.

**It is NOT inevitable**: Pass 1 already produces the exact OOF predictions needed. The `TrainingSetReconstructor` collects them from the prediction store. They are persisted (prediction records are lightweight). The refit can simply read them.

**One caveat**: The document's Pass 2a re-runs with the "winning variant's preprocessing" (which might differ from what some base models saw in Pass 1 if generators selected different variants for different models). But for the stacking refit context, ALL base models use the **meta-model's winning variant chain** — so if the meta-model's winning variant was Variant 1 (MSC), Pass 1 already ran MSC→PLS and MSC→RF with CV in Variant 1's execution, producing the OOF predictions for that variant.

**Solution**: **Eliminate Pass 2a entirely.** The stacking refit becomes a two-sub-pass operation:

```
Step 1 (Base model refit):   Retrain each base model on ALL training data (no CV).
Step 2 (Meta-model refit):   Base models predict on training data → meta-model trains on those predictions.
```

The ordering matters: base models are retrained FIRST, then their predictions on training data become the meta-model's training features. This ensures the deployment prediction flow is consistent — at deployment, the meta-model receives predictions from the same refit base models it was trained on.

This reduces the stacking refit cost from `N×K + 1 + N` to `N + 1` training runs — a massive saving.

**Note on in-sample predictions**: The meta-model trains on predictions that base models made on their own training data (in-sample). This is acceptable because:
1. **Model selection already happened** in Pass 1 using proper OOF predictions (no leakage in the selection phase).
2. The refit phase is about building the best deployment model, not about evaluation.
3. For typical NIRS models (PLS, Ridge, RF), in-sample predictions are not severely overfit.
4. The test set evaluation (separate from refit training) provides an honest performance estimate.
5. This eliminates the distribution shift between training (OOF predictions with higher variance) and deployment (full-data predictions with lower variance).

**When Pass 2a IS needed**: Only if the user explicitly wants OOF-based meta-model training during refit (e.g., for highly flexible base models where in-sample predictions would be overfit). This should be an explicit opt-in via `refit={"stacking_meta_training": "oof"}`, not the default.

#### 8.2.2 Memory Pressure from Transient Fold Models (Section 3.11)

**The flaw**: The document states fold models are "transient — exist in memory during Pass 1 only." Currently, fold models are persisted to disk immediately after training (`_persist_model()` at line 868). The document proposes keeping them in memory instead.

**Problem**: For large models (scikit-learn random forests with 500 trees, deep learning models, large ensembles), keeping K fold models in memory simultaneously can exhaust RAM. A 5-fold RF pipeline with 100MB per model requires 500MB+ in memory. The current system persists to disk and loads on demand.

**Solution**: Keep the current persistence behavior during Pass 1 (fold models saved to disk as transient artifacts). After refit completes successfully, **delete** fold model artifacts from the workspace store. This provides:
- Same memory footprint as current system during Pass 1
- Clean final state (only refit artifacts remain)
- A natural rollback point: if refit fails, fold artifacts still exist

Add a cleanup step at the end of `execute_refit()`:
```
if refit_succeeded:
    workspace_store.delete_transient_fold_artifacts(run_id)
```

#### 8.2.3 Individual Optuna Approach Returns Per-Fold Params (Section 4.4)

**The flaw**: The document assumes `best_params` is always a single dict. But the `individual` Optuna approach returns a **list** of dicts — one per fold. The code handles this at line 832-835:
```python
if isinstance(best_params, list):
    best_params_fold = best_params[fold_idx]
else:
    best_params_fold = best_params
```

During refit (single model, no folds), which params from the list should be used?

**Solution**: For refit after `individual` Optuna:
- **Default**: Use the params from the fold that had the best val score. This is the "best individual configuration" — the params that were optimal for the most representative data subset.
- **Alternative**: Average numeric params across folds (for continuous hyperparameters only). This is more robust but less interpretable.

Store the selected refit params explicitly:
```python
if isinstance(best_params, list):
    # For refit: select params from best-performing fold
    best_fold_idx = np.argmin(fold_scores) if lower_is_better else np.argmax(fold_scores)
    refit_params = best_params[best_fold_idx]
```

#### 8.2.4 No Test Set Scenario (Section 3.5)

**The flaw**: The document assumes a train/test split always exists ("Evaluate the refitted model on the test set"). But nirs4all supports pipelines where **all data is used for CV** with no held-out test set (e.g., Leave-One-Out CV on small datasets).

**Impact**: If there's no test set:
- Step 5 ("Evaluate on test") is impossible.
- `result.final_score` would be undefined.
- The refit model would have no independent evaluation metric.

**Solution**: When no test set exists:
- The refit still trains on all data (the entire dataset, since there's no test partition).
- `result.final_score = None` (no independent evaluation possible).
- `result.cv_best_score` remains the only performance estimate.
- Log a warning: "No test set available. The refit model has no independent evaluation. CV score is the only performance estimate."
- The refit model is still useful for deployment — the user just doesn't get a final test metric.

#### 8.2.5 Augmentation Non-Determinism in Pass 2a (If Kept)

**The flaw** (relevant only if Pass 2a is retained, see 8.2.1): Section 6.4 says augmentation should "run normally during refit." But augmentation is stochastic (random noise, random mixup pairs). If Pass 2a re-runs augmentation, the augmented samples differ from Pass 1, producing different OOF predictions. The meta-model is then trained on OOF predictions from a different data distribution than what was evaluated in Pass 1.

**Solution**: If Pass 2a is eliminated (per 8.2.1), this issue disappears. If Pass 2a is retained, either:
- (a) Fix random seeds per fold to ensure reproducibility, or
- (b) Accept the non-determinism (the OOF predictions will be similar but not identical).

Recommendation: Eliminate Pass 2a.

#### 8.2.6 Separation Branches + Merge("concat") + Downstream Model Leakage (Section 6.7)

**The flaw**: Section 6.7 raises the question of whether `merge: "concat"` after separation branches feeds predictions to the next model. The answer from the codebase is: **yes, it can**. The MergeController with `is_separation_merge=True` reassembles features/predictions in original sample order. If an inner model predicted on its own training data (within-site), those predictions become input to the outer model — **data leakage**.

**This is a pre-existing architectural issue, not introduced by the refit design.** But the refit design should not make it worse.

**Solution**: Document this as a known limitation. For refit:
- If `merge: "concat"` follows separation branches with inner models, treat it as a stacking topology (OOF dependency exists).
- The topology analyzer should detect: `separation_branch → inner_model → merge("concat") → outer_model` and flag it for stacking refit treatment.

### 8.3 Redundancies (Sections to Consolidate)

#### 8.3.1 Double Constraint Explained Three Times

The "double constraint" (generators × finetuning) is explained in:
- **Section 4.5** (full analysis with example)
- **Section 5.3** (same analysis, same conclusion)
- Partially in **Section 4.10** (triple constraint — generators × branches × finetuning)

**Solution**: Keep the full analysis in Section 4.5. Replace Section 5.3 with a single-line cross-reference: *"See Section 4.5 for the full analysis of the double constraint."*

#### 8.3.2 Generators + Stacking Example Repeated

The same SNV/MSC × PLS/RF × Ridge example appears in:
- **Section 3.10** (per-model independent refit motivation)
- **Section 4.8** (generators + stacking edge case)
- **Section 5.6** (per-model scoring and selection)

All three use identical variant scores (PLS=0.85/0.80, RF=0.82/0.88, Ridge=0.90/0.92) and reach the same conclusions.

**Solution**: Introduce the example ONCE in Section 3.10. In Sections 4.8 and 5.6, reference: *"Using the example from Section 3.10..."* and present only the new insights specific to each section.

#### 8.3.3 Section 4.15 Is Explicitly Redundant

Section 4.15 (Generators Inside Branches) ends with: *"This is equivalent to Section 4.8."*

**Solution**: Collapse Section 4.15 to a single paragraph: *"Generators inside branches expand to full pipeline variants before execution, making this case structurally equivalent to Section 4.8 (Generators + Stacking). The generator selects the variant; all branches within the winning variant are preserved for stacking."*

### 8.4 Missing Features and Gaps

#### 8.4.1 Classification-Specific Refit Behavior

**Gap**: All examples use regression. Classification introduces:
- **Fold averaging**: Currently averages class probabilities across folds (soft voting). Refit produces a single model's probabilities — no averaging needed.
- **Metric semantics**: Accuracy, F1, AUC have different directionality and scale than RMSE/R².
- **Class balance**: If training data has class imbalance, fold stratification ensures balanced folds. Refit on all data naturally handles this (no fold stratification needed). But if augmentation was stratification-aware in CV, the refit should maintain similar class proportions.
- **Multiclass stacking**: OOF predictions for multiclass have K probability columns per base model. The meta-model's feature space scales with `N_base_models × N_classes`.

**Solution**: Add a subsection "4.18 Classification Pipelines":
- For simple refit: no special handling needed. Train on all data, evaluate test accuracy/F1.
- For stacking refit: OOF predictions carry class probability columns. Pass 2b's meta-model training features have shape `(N_train, N_base_models × N_classes)`. This is already handled by the existing `TrainingSetReconstructor` (it supports `use_proba=True`).
- For `result.final_score`: report the classification metric (accuracy, F1) from the test evaluation.

#### 8.4.2 `y_processing` (Target Scaling) During Refit

**Gap**: The document never mentions `{"y_processing": MinMaxScaler()}`. Target scalers are fitted per fold during CV. During refit, the target scaler should be refitted on all training targets.

**Solution**: During refit, `y_processing` transformers are treated like any other preprocessing step: cloned and refitted on all training data. The inverse transform is applied to predictions before evaluation. This is consistent with the "refit everything" philosophy. Add this to Section 3.5 (Simple Refit):
- Step 2.5: If `y_processing` exists, clone the target scaler → `fit(y_train_all)` → use for `transform(y_train_all)` during training and `inverse_transform(y_pred)` during evaluation.

#### 8.4.3 Pipelines Without a CV Splitter

**Gap**: If the pipeline has no splitter (e.g., `[SNV(), PLSRegression()]`), there's no CV phase. There are no folds, no fold models, no avg/w_avg. The current system trains a single model on all training data (with train/test split from the dataset).

**Impact**: In this case, the model is already trained on all training data. Refit would be a no-op.

**Solution**: Add detection: if no CV splitter is present, skip the refit phase entirely. The model from Pass 1 IS the final model. Set `fold_id="final"` on the existing prediction entry. Log: "No cross-validation detected. The trained model is already the final model."

#### 8.4.4 Determinism and Fold Reproducibility

**Gap**: If Pass 2a is retained (see 8.2.1), it needs the **same fold split** as Pass 1 for consistent OOF predictions. Random splitters (ShuffleSplit, StratifiedKFold with shuffle) need seed management.

**Solution** (if Pass 2a is retained): Store the fold indices from Pass 1 in the execution trace or chain record. During Pass 2a, reuse the stored fold indices instead of re-computing from the splitter. The `CrossValidatorController` already stores fold information in the dataset via `dataset.set_folds()` — these can be serialized and replayed.

**Better solution**: Eliminate Pass 2a entirely (per 8.2.1), making this moot.

#### 8.4.5 Webapp Integration

**Gap**: No discussion of how the two-pass architecture integrates with the webapp's job system, WebSocket progress events, and UI.

**Solution**: Add a subsection "7.9 Webapp Integration":
- The refit phase should emit distinct WebSocket events: `REFIT_STARTED`, `REFIT_PROGRESS`, `REFIT_COMPLETED`, `REFIT_FAILED`.
- The RunProgress page should show a "Refit" phase indicator after the CV phase completes.
- The job manager should track refit as a sub-phase of the training job, not a separate job.
- The Results page should display both CV scores and refit scores, clearly labeled.
- The `result.final` model should be the default export target in the UI.

#### 8.4.6 Partial Refit Failure and Rollback

**Gap**: Section 6.3 discusses what happens when refit fails but doesn't address partial failures in stacking refit (e.g., base model refit succeeds, meta-model refit fails).

**Solution**: Adopt transactional semantics for the refit phase:
1. Refit artifacts are written to a staging area (or marked as `refit_pending`).
2. Only after ALL refit steps succeed, promote artifacts to `fold_id="final"`.
3. If any step fails:
   - Roll back all refit artifacts for that model.
   - Keep fold artifacts from Pass 1 (per 8.2.2, they're still on disk).
   - Set `result.final = None` for the failed model but keep successful standalone refits.
   - Log which model's refit failed and why.

#### 8.4.7 Parallel Execution During Refit and GPU Resource Contention

**Gap**: No discussion of parallelism in the refit phase, and no handling of GPU resource contention.

**Parallelism opportunities**:
- **Base model refits** (stacking): Each base model's refit is independent → parallelize with joblib.
- **Per-model standalone refits** (Section 3.10): Each model's standalone refit is independent → parallelize.
- **Meta-model refit**: Single model, no parallelism.

Reuse the existing `n_jobs` parameter from model config. The refit inherits the same parallelism settings as Pass 1.

**GPU contention problem**: When parallel base model refits target GPU-backed models (TensorFlow, PyTorch, JAX), multiple models competing for GPU memory will cause OOM crashes or severe slowdown. This is a real risk because:
- GPU memory is finite and non-preemptible — two models each wanting 4GB on an 8GB GPU will crash.
- CUDA contexts have per-process overhead. Multiple concurrent contexts waste memory.
- Even with memory management (e.g., `tf.config.experimental.set_memory_growth`), concurrent GPU training is fragile.

**Solution — GPU-aware serialization**:
1. **Detection**: At the start of the refit phase, detect if any model step uses a GPU-backed framework. The existing `ModelFactory.detect_framework()` and GPU detection in `api/system.py` can provide this information.
2. **Serial GPU, parallel CPU**: If GPU models are present, serialize ALL model refits (no parallelism). If all models are CPU-only (sklearn, CPU-bound RF, PLS), parallelize freely.
3. **Implementation**: Add a `_is_gpu_model(model_config)` check in the refit orchestration logic:

```
refit_models = [base_1_config, base_2_config, ...]

if any(_is_gpu_model(m) for m in refit_models):
    # Sequential refit — safe for GPU
    for model_config in refit_models:
        execute_single_refit(model_config, dataset, context)
else:
    # Parallel refit — CPU models only
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(execute_single_refit)(m, dataset, context)
        for m in refit_models
    )
```

4. **GPU memory cleanup**: After each GPU model refit, explicitly release GPU memory (`torch.cuda.empty_cache()`, `tf.keras.backend.clear_session()`, etc.) before the next model trains. The existing deep learning controllers already handle this in their `_cleanup_after_training()` methods.
5. **Future optimization**: A more sophisticated approach could use GPU memory estimation to pack multiple small models concurrently, but for Phase 1, serial GPU execution is the safe default.

### 8.5 Optimizations

#### 8.5.1 Lazy Per-Model Standalone Refit

**Current proposal**: Refit EVERY model node independently after Pass 1 (Section 3.10).

**Problem**: For a pipeline with 5 base models in a stacking setup, this produces 5 standalone refits + 5 stacking-context refits + 1 meta-model refit = 11 training runs, even though most users only want the stacking model.

**Solution**: Make standalone per-model refits **lazy** (computed on demand):
- By default, only refit the **outermost** model (the meta-model or the single final model).
- Per-model standalone refits are triggered only when accessed: `result.models["PLS"].export()` triggers the PLS standalone refit.
- Store enough metadata (winning variant, best_params, chain config) to execute the deferred refit.

This reduces the default refit cost to the minimum (1 stacking refit) while preserving access to standalone models.

#### 8.5.2 Shared Preprocessing Across Refit Contexts

**Observation**: When the stacking variant's chain matches a base model's standalone best chain, preprocessing artifacts are identical. Example: if Ridge's winning variant is MSC, and RF's standalone best is also MSC, both refits produce the same MSC artifact.

**Solution**: Cache preprocessor artifacts by (operator_class, operator_params, training_data_hash). Reuse cached artifacts across refits. The existing content-addressed artifact store (`SHA-256` hashing) already deduplicates at the storage level — but the training runs themselves can be skipped by checking the cache before fitting.

#### 8.5.3 Warm-Start Refit for Deep Learning

**Gap**: For neural network models (TensorFlow, PyTorch, JAX), training from scratch on all data is expensive.

**What warm-start means**: Instead of initializing a new model with random weights and training from epoch 0, *warm-start* means loading the trained weights from an existing model (typically the best fold model from CV) and continuing training from there on the full dataset. The network starts from an already-good solution and only needs to fine-tune to the slightly larger dataset — this converges much faster than training from scratch.

**Concrete example**:
- During CV, the best fold model trained for 100 epochs on 80% of training data, reaching loss=0.02.
- A cold-start refit on 100% of data would need ~100 epochs from random init to reach a similar loss.
- A warm-start refit loads the fold model's weights, then continues training on 100% of data. Because the weights are already near the optimum, 20–30 additional epochs typically suffice — a 3–5× speedup.

**How it works for each framework**:
- **PyTorch**: `refit_model.load_state_dict(fold_model.state_dict())` → continue `optimizer.step()` on full data.
- **TensorFlow/Keras**: `refit_model.set_weights(fold_model.get_weights())` → continue `model.fit()` on full data.
- **JAX**: Copy the parameter pytree from the fold model → continue training loop.
- **Scikit-learn**: Some estimators support `warm_start=True` (e.g., `MLPRegressor`, `GradientBoosting`). Set `model.set_params(warm_start=True)` and call `fit()` again on the full data — the model continues from its current state rather than reinitializing.

**Implementation**:
1. After CV, select the fold model with the best validation score (already identified for param selection).
2. Load its weights/state (the fold model artifact is still on disk at this point, not yet cleaned up).
3. Create the refit model instance, inject the fold model's weights.
4. Train on ALL training data using `refit_params` (which may specify fewer epochs, lower learning rate — see Section 8.11).
5. The warm-started model converges faster because it starts near the optimum.

**When NOT to warm-start**: If the fold was trained on 80% of data and the refit trains on 100%, the distribution is similar enough for warm-start to work. But if the refit data is substantially different (e.g., new data added via `retrain()`), cold-start may be more appropriate. The `refit_params` config (Section 8.11) provides control: `"warm_start": True/False`.

**Integration with `refit_params`**: Warm-start naturally pairs with refit-specific training parameters. For example, CV trains with `epochs=100, lr=0.001` (exploration), while refit uses `epochs=30, lr=0.0001` (fine-tuning from warm-start). See Section 8.11.

### 8.6 Potential Deadlocks and Race Conditions

#### 8.6.1 Workspace Store Concurrent Access During Refit

**Risk**: If Pass 1 and the refit phase write to the same DuckDB store concurrently (e.g., prediction records, artifact metadata), DuckDB's single-writer constraint could cause blocking.

**Mitigation**: The current architecture is single-threaded at the orchestrator level (variants run sequentially). The refit phase runs AFTER Pass 1 completes. No concurrent writes. However, if future optimizations introduce parallel variant execution, this becomes a concern.

**Solution**: Keep the refit phase as a strictly sequential post-Pass-1 operation. Do not parallelize at the orchestrator level. Parallelism is safe within a single refit (parallel base model refits via joblib) because they write to different artifact slots.

#### 8.6.2 Prediction Store Collision with `fold_id="final"`

**Risk**: Multiple refit entries with `fold_id="final"` for the same model (standalone + stacking context, Section 7.6). The uniqueness constraint on `(pipeline_id, model_name, fold_id, partition)` may conflict.

**Mitigation**: The `refit_context` field distinguishes them. But the existing prediction store doesn't have this field, and the uniqueness constraint (if any) doesn't account for it.

**Solution**: Add `refit_context` to the prediction record schema. Update the DuckDB schema to include this column. The uniqueness key becomes `(pipeline_id, model_name, fold_id, partition, refit_context)`. Default value for non-refit entries: `NULL` or `"cv"`.

### 8.7 Architecture Recommendations Summary

| # | Issue | Severity | Solution | Impact |
|---|-------|----------|----------|--------|
| 1 | `fit_on_all` is implemented, not a placeholder | Factual error | Correct Section 2.4 | Documentation |
| 2 | `fold_id` is string-based | Factual error | Correct Section 2.3 | Documentation |
| 3 | `best_params` already in predictions | Factual error | Use existing storage; drop Section 6.1 | Simplifies implementation |
| 4 | Pass 2a is redundant | Major optimization | Reuse Pass 1 OOF predictions | Saves N×K training runs |
| 5 | Memory pressure from in-memory fold models | Design flaw | Persist then cleanup after refit | Prevents OOM |
| 6 | Individual Optuna returns per-fold params | Missing handling | Select best-fold params for refit | Correctness |
| 7 | No test set scenario | Missing handling | Skip test evaluation, warn user | Robustness |
| 8 | Classification not addressed | Missing feature | Add classification subsection | Completeness |
| 9 | `y_processing` not addressed | Missing feature | Refit target scaler on all train data | Correctness |
| 10 | Pipelines without CV splitter | Missing handling | Detect and skip refit (already final) | Edge case |
| 11 | Per-model standalone refit is expensive | Over-engineering | Make lazy (on-demand) | Performance |
| 12 | Redundant sections (4.5/5.3, 3.10/4.8/5.6, 4.15) | Redundancy | Consolidate with cross-references | Readability |
| 13 | `result.best_score` semantics | Design conflict | Decouple `best_score` (CV) from `final_score` (refit) | Clarity |
| 14 | Separation + concat + outer model leakage | Pre-existing bug | Detect and treat as stacking topology | Correctness |
| 15 | Webapp integration absent | Missing feature | Add WebSocket events and UI phases | Completeness |
| 16 | Partial refit failure | Missing handling | Transactional semantics with rollback | Robustness |
| 17 | `refit_context` field collision | Potential bug | Add to schema, update uniqueness | Correctness |
| 18 | GPU contention during parallel refit | Missing handling | GPU-aware serialization: serialize GPU models, parallelize CPU-only | Prevents GPU OOM |
| 19 | No refit-specific training params | Missing feature | `refit_params` key in model config (Section 8.11) | Flexibility |
| 20 | Warm-start not explained or designed | Missing feature | Load fold weights → continue training (Section 8.5.3) | Performance for DL |
| 21 | No caching for preprocessing/predictions | Missing feature | Content-addressed step-level cache (Section 8.12) | Performance |
| 22 | RepeatedKFold OOF overwrite in TrainingSetReconstructor | Pre-existing bug | Accumulate and average OOF predictions per sample (Section 8.13) | Correctness |
| 23 | Generators varying splitter: no special refit handling needed | Confirmed OK | Splitter stripped during refit; generator_choices records selection | Completeness |

### 8.8 Revised Stacking Refit (Incorporating Solutions)

The original three-sub-pass design (2a/2b/2c) should be simplified to a **two-step** design. The key insight: base models are retrained FIRST, then the meta-model trains on predictions from the refit base models. This ensures training-time and deployment-time prediction distributions are identical.

```
STACKING REFIT (revised):

Step 1 — Base model refit:
  For each base model in the winning variant's branch:
     a. Clone the preprocessing chain.
     b. Fit preprocessing on ALL training data.
     c. Clone the model with winning params (from best_params in prediction store).
        If refit_params exist, apply refit-specific overrides (see Section 8.11).
     d. Train model on all preprocessed training data.
     e. Generate predictions on the training data (in-sample).
     f. Persist model as fold_id="final" with refit_context="stacking".
  GPU-aware serialization: if any base model is GPU-backed, run sequentially (see 8.4.7).
  Otherwise: parallelize across base models (joblib), then collect predictions after all complete.

Step 2 — Meta-model refit:
  1. Collect base model predictions from Step 1 into a feature matrix (N_train × N_base_models).
     For classification: (N_train × N_base_models × N_classes) probability columns.
  2. Train meta-model on base model prediction features with training targets.
     Apply refit_params overrides if specified.
  3. Evaluate meta-model on test set (base models predict on test → meta-model predicts → score).
  4. Persist meta-model as fold_id="final" with refit_context="stacking".

Deployment prediction flow (consistent with training):
  new_data → Branch_i preprocessing → Base_model_i(Step 1).predict() → predictions
  [all branch predictions] → Meta-model(Step 2).predict() → final
```

**Why this ordering is correct**:
- The meta-model is trained on predictions from the SAME base models used at deployment.
- No distribution shift between training and deployment (unlike OOF-based meta-training where OOF predictions have higher variance than full-data predictions).
- Model selection integrity is preserved: the stacking architecture was validated in Pass 1 using proper OOF predictions with no leakage. The refit phase only rebuilds the deployment models.

**Cost comparison**:
- Original (2a/2b/2c): `N×K + 1 + N` training runs for refit
- Revised (Step 1 + Step 2): `N + 1` training runs for refit
- For 3 base models, 5-fold CV: 16 → 4 training runs (75% reduction)

**Optional OOF-based meta-training**: For users with highly flexible base models (deep learning, large ensembles) where in-sample predictions risk severe overfitting, provide an opt-in flag `refit={"stacking_meta_training": "oof"}` that reverts to training the meta-model on Pass 1's OOF predictions instead. In that case, the meta-model is trained before base model refit (original 2b→2c order).

### 8.9 Revised Implementation Phases

**Phase 1** (covers 80% of use cases):
- Simple refit for non-stacking pipelines.
- Topology detection: check for `merge: "predictions"`.
- Handle: no-splitter pipelines (skip refit), generators, finetuning, branch+features.
- Handle: classification and `y_processing`.
- Persist-then-cleanup strategy for fold artifacts.
- `refit_params` support in model config (Section 8.11).
- `fit_on_all` artifact reuse and stateless transform skip (Level 1–2 caching, Section 8.12).
- GPU-aware serialization for parallel refits (Section 8.4.7).
- Changes: orchestrator, model controller, predictions, result object, parser (add `refit_params` keyword).

**Phase 2** (covers 15% of use cases):
- Stacking refit: base models refit → base predictions → meta-model refit (Section 8.8).
- Handle: mixed merge, generators+stacking.
- Handle: per-fold Optuna params → best-fold selection.
- Warm-start support for deep learning models (Section 8.5.3).
- In-memory preprocessed data snapshot cache (Level 3 caching, Section 8.12).
- Changes: new stacking refit logic, integration with merge controller and prediction store.

**Phase 3** (covers 5% of use cases):
- Nested stacking refit (recursive).
- Separation branch refit (per-branch).
- Lazy per-model standalone refits.
- Webapp integration (WebSocket events, UI phases).
- Changes: recursive refit, per-branch dispatch, webapp backend.

### 8.10 Revised Open Questions (Replacing Section 6)

| # | Question | Status | Resolution |
|---|----------|--------|------------|
| 6.1 | How to store finetuned params? | **Resolved** | Already stored in `best_params` field of prediction records |
| 6.2 | Re-run Optuna in Pass 2a? | **Resolved** | Pass 2a eliminated; no re-run needed |
| 6.3 | What if refit fails? | **Resolved** | Transactional semantics; keep fold artifacts until refit succeeds |
| 6.4 | Augmentation in refit? | **Resolved** | Run normally; refit is a single "fold" with all training data |
| 6.5 | Filters/exclusions in refit? | Unchanged | Re-run filters on all training data |
| 6.6 | `nested_cv.md` alignment? | Unchanged | Refit supersedes `use_full_train_for_final`; rest is orthogonal |
| 6.7 | Separation + concat + outer model? | **Resolved** | Detect as stacking topology if inner model predictions feed outer model |
| 6.8 | Sequential models without branches? | **Clarified** | Treat as implicit stacking; apply stacking refit |
| 6.9 | Recursion depth? | Unchanged | Warn at threshold; practical limit is 2 levels |
| 6.10 | Preprocessing: refit or reuse? | Unchanged | Refit on all training data |
| **NEW** | Per-fold Optuna params in refit? | **Resolved** | Use params from best-performing fold |
| **NEW** | No test set scenario? | **Resolved** | Skip test eval, warn user, `final_score=None` |
| **NEW** | No CV splitter in pipeline? | **Resolved** | Skip refit entirely; existing model is final |
| **NEW** | Classification-specific behavior? | **Resolved** | Handled by existing infrastructure; add subsection |
| **NEW** | `y_processing` in refit? | **Resolved** | Refit target scaler on all training targets |
| **NEW** | Refit-specific training params? | **Resolved** | `refit_params` in model config (Section 8.11) |
| **NEW** | GPU contention during parallel refit? | **Resolved** | GPU-aware serialization (Section 8.4.7) |
| **NEW** | Warm-start for deep learning refit? | **Resolved** | Load fold model weights, continue training (Section 8.5.3) |
| **NEW** | Caching to accelerate refit? | **Resolved** | Step-level caching with content-addressed keys (Section 8.12) |
| **NEW** | Generator varying CV splitter? | **Resolved** | No special handling for refit; splitter stripped. RepeatedKFold OOF bug found and fixed (Section 8.13) |

### 8.11 Refit-Specific Training Parameters (`refit_params`)

**Problem**: Training parameters optimal for CV may not be optimal for the refit phase. Common scenarios:
- **More epochs**: CV uses 100 epochs (fast iteration across folds/variants), refit uses 1000 epochs (final model quality matters more than speed).
- **Lower learning rate**: CV uses `lr=0.001` for fast convergence, refit uses `lr=0.0001` for fine-grained optimization on the full dataset.
- **Larger batch size**: With all training data available, a larger batch size may improve gradient estimates.
- **No early stopping**: CV uses `patience=10` to avoid wasting time on bad folds, refit can train longer without early stopping.
- **Warm-start control**: Enable/disable warm-starting from the best fold model (see Section 8.5.3).

**Design**: Add an optional `refit_params` key to model step configuration, parallel to the existing `train_params` and `finetune_params` keys.

#### Pipeline Syntax

```python
# Model step with refit-specific overrides
{
    "model": MyNeuralNet(),
    "train_params": {
        "epochs": 100,        # CV training: fast iteration
        "batch_size": 32,
        "learning_rate": 0.001,
        "patience": 10,       # Early stopping for CV
        "verbose": 0
    },
    "refit_params": {
        "epochs": 1000,       # Refit: train longer for deployment quality
        "learning_rate": 0.0001,  # Fine-grained optimization
        "batch_size": 64,     # Larger batch with full dataset
        "patience": None,     # Disable early stopping
        "warm_start": True    # Initialize from best fold model weights
    }
}

# Minimal: only override what differs
{
    "model": PLSRegression(),
    "train_params": {"verbose": 0},
    "refit_params": {"verbose": 1}  # See refit training output
}

# No refit_params → refit uses train_params (default behavior)
{
    "model": RandomForestRegressor(n_estimators=100),
    "train_params": {"verbose": 0}
    # Refit uses the same train_params — no override needed for sklearn models
}
```

#### Parameter Resolution

During refit, training parameters are resolved by merging `refit_params` on top of `train_params`:

```python
def resolve_refit_params(model_config: dict) -> dict:
    """Merge refit_params on top of train_params. refit_params wins on conflicts."""
    base = model_config.get("train_params", {}).copy()
    overrides = model_config.get("refit_params", {})
    base.update(overrides)
    return base
```

This means `refit_params` only needs to specify the parameters that differ from `train_params`. Unspecified parameters inherit from `train_params`.

#### Special `refit_params` Keys

Beyond standard training parameters, `refit_params` supports these refit-specific keys:

| Key | Type | Default | Purpose |
|-----|------|---------|---------|
| `warm_start` | bool | `False` | Initialize from best fold model weights (deep learning only) |
| `warm_start_fold` | str | `"best"` | Which fold model to warm-start from: `"best"` (best val score), `"last"` (last fold), or `"fold_N"` |

#### Integration Points

1. **Parser**: Add `"refit_params"` to `RESERVED_KEYWORDS` in `pipeline/steps/parser.py`.
2. **Config preprocessing**: `PipelineConfigs._preprocess_steps()` already handles `XX_params` key merging. Adding `refit_params` follows the existing pattern — `model_refit_params` normalizes to `{"model": {..., "refit_params": {...}}}`.
3. **Model controller**: In `BaseModelController`, during the refit execution path, call `resolve_refit_params(model_config)` instead of `model_config.get("train_params", {})`.
4. **Deep learning controllers**: In `TensorFlowModelController`, `PyTorchModelController`, `JAXModelController`, check for `warm_start` in the resolved params and load fold model weights before training.
5. **Optuna**: `finetune_params` is NOT re-run during refit (Section 7.8). The `best_params` from CV are used directly, with `refit_params` applied as additional training parameter overrides.

#### Global Refit Config (from `nirs4all.run()`)

The `refit` parameter at the `nirs4all.run()` level can also specify global defaults:

```python
nirs4all.run(pipeline, dataset, refit={
    "enabled": True,
    "default_refit_params": {       # Applied to ALL models unless overridden per-model
        "verbose": 1
    },
    "stacking_meta_training": "in_sample",  # Default: base model predictions on training data
    # Alternative: "oof" — use Pass 1 OOF predictions for meta-model training
})
```

Per-model `refit_params` in the pipeline step take precedence over global `default_refit_params`.

### 8.12 Caching Architecture for Refit Acceleration

> Full analysis: `docs/_internal/caching_analysis.md`

The refit phase re-executes the winning pipeline's preprocessing chain on the full training data. Without caching, every transform step is cloned and refitted from scratch, even when some steps would produce identical results. A caching layer eliminates this redundancy.

#### Current State (What Exists)

| Mechanism | Type | Scope | Relevant for Refit? |
|-----------|------|-------|---------------------|
| Artifact content-addressed dedup (SHA-256) | Binary storage dedup | Cross-pipeline | Partially — deduplicates identical artifacts at rest, but does NOT skip fitting |
| `DatasetConfigs.cache` | Parsed data tuple cache | Single run | Yes — prevents re-parsing source files |
| Dataset content hash (`SpectroDataset.metadata()`) | Provenance hash | Metadata | Yes — can be part of cache keys |
| Pipeline config hash (`PipelineConfigs.get_hash()`) | Pipeline identification | Naming | Yes — identifies step configurations |
| `DataCache` (LRU, thread-safe, size-limited) | Data loading cache | Session | Not yet integrated — but infrastructure is ready |
| `OperatorChain` path hashing | Deterministic step identity | Artifact naming | Yes — natural cache key component |

**Key gap**: No step-level computation cache exists. The existing artifact dedup only prevents writing duplicate files — it does NOT skip the fitting step that produced the duplicate.

#### Caching Strategy for the Refit Pass

Three levels of caching, ordered by implementation priority:

**Level 1 — `fit_on_all` artifact reuse (trivial, Phase 1)**:
If a transformer step has `fit_on_all: True`, it was fitted on ALL data during CV (train+test). During refit, the training data is the same → the fitted artifact is identical. **Skip fitting entirely**; load the CV artifact and use it for transform-only.

Detection: check `step_info.original_step.get("fit_on_all", False)` in the refit execution path.

**Level 2 — Stateless transform skip (low effort, Phase 1)**:
Some transforms are stateless (no learned parameters): fixed-parameter spectral derivatives, Detrend with fixed polynomial order, CropTransformer, ResampleTransformer. Their output depends only on input data and fixed parameters, not on what data was used for fitting.

Detection: add a `_stateless = True` class attribute to stateless operators. During refit, if `_stateless` is True and the input data is the same, reuse the CV output.

**Level 3 — In-memory preprocessed data snapshot cache (medium effort, Phase 2)**:
Cache the full dataset state after each preprocessing step, keyed by `(chain_path_hash, input_data_hash)`. This benefits both:
- **Refit**: The winning pipeline's preprocessing chain is replayed; cached snapshots avoid redundant computation.
- **Generator sweeps**: Multiple variants sharing a preprocessing prefix (e.g., `[SNV, _or_: [PLS(5), PLS(10), PLS(15)]]`) compute SNV only once.

Implementation:
```python
class StepCache:
    """In-memory cache of preprocessed dataset states within a single run."""

    def __init__(self, max_size_mb: int = 2048):
        self._cache: dict[str, CachedSnapshot] = {}
        self._max_size = max_size_mb * 1024 * 1024

    def cache_key(self, chain_path_hash: str, data_hash: str) -> str:
        return f"{chain_path_hash}:{data_hash}"

    def get(self, key: str) -> SpectroDataset | None:
        entry = self._cache.get(key)
        return entry.dataset.copy() if entry else None

    def put(self, key: str, dataset: SpectroDataset):
        # LRU eviction if over memory limit
        ...
```

Integration: `PipelineExecutor._execute_single_step()` checks the cache before executing, stores the result after executing.

#### What Should NOT Be Cached

- **Model training**: The refit model trains on the full training set — a genuinely new computation. Caching cannot help (except warm-start, which is a separate mechanism — see Section 8.5.3).
- **Cross-run caching**: For Phase 1, keep caching in-memory within a single `nirs4all.run()` call. Cross-run caching (DuckDB-backed) adds complexity (invalidation, versioning) and should be deferred to later phases.

#### Cache Invalidation

The `OperatorChain` hash naturally handles most invalidation: if any upstream step changes (different operator, different params), the chain hash changes, invalidating all downstream cache entries. The remaining risk is stale data (dataset modified between CV and refit), which cannot happen in the normal execution flow since both phases share the same in-memory `SpectroDataset` instance.

#### Performance Impact Estimates

| Scenario | Without Cache | With Cache | Savings |
|----------|--------------|------------|---------|
| Simple refit (5 preprocessing steps, PLS) | 5 fits + 1 model | 0 fits* + 1 model | ~50% of refit time |
| Generator sweep (100 variants, shared 3-step prefix) | 300 fits + 100 models | 3 fits + 100 models | ~30% of CV time |
| Stacking refit (3 base models, 5 preprocessing steps) | 15 fits + 4 models | 5 fits** + 4 models | ~20% of refit time |

\* With `fit_on_all` or stateless transforms.
\** Preprocessing refitted once per base model branch; each branch's chain is different.

### 8.13 Generators Varying the Cross-Validation Splitter

#### The Scenario

Users can use generators to search over CV strategies themselves:

```python
from sklearn.model_selection import RepeatedKFold
from nirs4all.operators.splitters import KennardStoneSplit, SPXYSplit

pipeline = [
    SNV(),
    {"_or_": [RepeatedKFold(n_splits=5, n_repeats=3), KennardStoneSplit(5), SPXYSplit(5)]},
    PLSRegression(10),
]
```

This expands to 3 variants:
- **Variant 1**: SNV → RepeatedKFold(5,3) → PLS(10) — 15 folds
- **Variant 2**: SNV → KennardStoneSplit(5) → PLS(10) — 5 folds
- **Variant 3**: SNV → SPXYSplit(5) → PLS(10) — 5 folds

The generator expansion system has **no restrictions** on splitters — they are treated like any other operator. Each variant runs independently with its own fresh `dataset` and `context`, producing its own fold structure, fold models, and prediction records. The winning variant is selected by comparing `avg`/`w_avg` val_scores across variants.

Combined generators (splitter + other variations) also work:

```python
pipeline = [
    {"_or_": [SNV(), MSC()]},
    {"_or_": [KFold(5), RepeatedKFold(3, 5)]},
    PLSRegression(10),
]
# → 4 variants: SNV+KFold5, SNV+RKFold, MSC+KFold5, MSC+RKFold
```

#### Impact on Simple Refit: None

For non-stacking pipelines, the splitter is irrelevant during refit. The refit replaces the splitter step with a dummy single fold `[(all_train_indices, [])]` (Section 3.5). This works regardless of whether the splitter was a fixed step or came from a generator:

1. The winning variant's expanded config includes the concrete splitter instance (e.g., `KennardStoneSplit(5)`).
2. The `execute_refit()` function identifies the splitter step via `CrossValidatorController.matches()`.
3. The splitter step is replaced/skipped. Preprocessing and model from the winning variant are used for refit.

No special handling needed.

#### Impact on Stacking Refit with In-Sample Meta-Training (Default): None

Since the default stacking refit retrains base models on full data and trains the meta-model on in-sample predictions (Section 8.8), the splitter is completely irrelevant. The fold structure from Pass 1 is not used during refit.

#### Impact on Stacking Refit with OOF Meta-Training (Opt-In): Requires Attention

When `refit={"stacking_meta_training": "oof"}` is used, the meta-model trains on OOF predictions from Pass 1. These OOF predictions are tied to the winning variant's specific splitter.

**Issue 1 — RepeatedKFold OOF overwrite bug (pre-existing)**:

If the winning variant uses `RepeatedKFold(n_splits=5, n_repeats=3)`, each training sample appears in 3 different validation folds (once per repeat). The `TrainingSetReconstructor._collect_oof_predictions()` (at `reconstructor.py:771-775`) places OOF values at sample positions using:

```python
oof_preds[pos] = y_vals[i]
```

This **overwrites** previous values. For samples with 3 OOF predictions, only the last fold's prediction is kept — the first two are silently discarded. The `FoldAlignmentValidator` detects this (issues a `DUPLICATE_VAL_SAMPLES` warning), but the reconstructor doesn't handle it.

**This is a pre-existing bug, not introduced by the refit design.** But it affects OOF meta-training quality if RepeatedKFold wins the generator search.

**Fix**: In `_collect_oof_predictions()`, accumulate predictions per sample position and average them:

```python
# Replace simple assignment with accumulation
oof_preds_sum = np.zeros(n_samples)
oof_preds_count = np.zeros(n_samples)

for pred in val_predictions:
    for i, sample_idx in enumerate(sample_indices):
        if i < len(y_vals):
            pos = id_to_pos.get(int(sample_idx))
            if pos is not None:
                oof_preds_sum[pos] += y_vals[i]
                oof_preds_count[pos] += 1

# Average where multiple predictions exist
mask = oof_preds_count > 0
oof_preds[mask] = oof_preds_sum[mask] / oof_preds_count[mask]
```

**Issue 2 — Fold count affects OOF prediction quality**: With 5-fold CV, each OOF prediction comes from a model trained on 80% of data. With 3-fold CV, each OOF prediction comes from a model trained on 67% of data. OOF predictions from higher fold counts are less noisy (models trained on more data). If the winning splitter has few folds, the OOF predictions fed to the meta-model are noisier. This is not a bug — it's the expected statistical property of the chosen CV strategy. The user selected it; the refit should respect it.

#### Score Comparability Across Splitter Variants

When comparing variants with different splitters, the `avg` val_scores have different statistical properties:

| Splitter | Folds | `avg` val_score is... | Variance of estimate |
|----------|-------|----------------------|---------------------|
| KFold(5) | 5 | Mean of 5 fold val_scores | Higher |
| KFold(10) | 10 | Mean of 10 fold val_scores | Lower |
| RepeatedKFold(5, 3) | 15 | Mean of 15 fold val_scores | Much lower |
| KennardStoneSplit(5) | 5 | Mean of 5 structured val_scores | Depends on data |
| SPXYSplit(5) | 5 | Mean of 5 structured val_scores | Depends on data |

**Implications**:
- Random splitters (KFold, ShuffleSplit) vs. structured splitters (KennardStone, SPXY) test fundamentally different properties. Random splits estimate generalization on similar data. Structured splits estimate generalization on dissimilar data (extrapolation). Comparing them is comparing different evaluation philosophies.
- RepeatedKFold with more repeats has lower variance, which means its `avg` score is more stable but not necessarily better.
- A variant might "win" because its splitter produced optimistic estimates (e.g., random splits on clustered data), not because the model is truly better.

**This is NOT a refit problem** — it's a model selection concern that exists before any refit. But refit amplifies its importance: the winner gets retrained as the deployment model.

**Recommendation**: This does not require changes to the refit design. However, when a generator varies the splitter, the result should include metadata about which splitter was selected. Add the winning variant's splitter info to the `fold_id="final"` prediction record and to `result.final`:

```python
result.final.cv_strategy = "KennardStoneSplit(5)"  # From generator_choices
result.final.cv_n_folds = 5
```

This allows users to understand which CV strategy was used for selection.

#### Config Reconstruction and `generator_choices`

The `generator_choices` field already records which option was selected at each generator position. For a splitter generator:

```python
# After winning variant is identified:
generator_choices = [
    {"step": 1, "choice": "KennardStoneSplit(5)", "index": 1}
]
```

This is sufficient for refit config reconstruction. The expanded variant's steps contain the concrete splitter instance. The refit logic strips it.

#### Caching Benefit

When the splitter varies but preprocessing doesn't (all 3 variants share `SNV()`), the preprocessing cache (Section 8.12) ensures SNV is fitted only once. All 3 splitter variants benefit from the cached preprocessed data. The cache key is based on the chain path hash, which is identical up to the splitter step.

#### Summary

| Concern | Severity | Resolution |
|---------|----------|------------|
| Simple refit with splitter generators | None | Splitter stripped; works already |
| Stacking refit (in-sample, default) | None | Splitter irrelevant for refit |
| Stacking refit (OOF, opt-in) with RepeatedKFold | Pre-existing bug | Fix OOF accumulation to average duplicates |
| Score comparability across splitters | Caveat (pre-existing) | Document; add splitter metadata to result |
| Config reconstruction | None | `generator_choices` already records selection |
| Caching interaction | Benefit | Shared preprocessing cached across splitter variants |
