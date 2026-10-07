# 3. Choose the operation you need

**You will learn:** when to run, generate, predict, retrain, calibrate or inspect. Start from what you already have.

## I have measurements and want a model

Use **run**. It fits candidates, evaluates them, selects one and creates the final predictor. Start with {doc}`start`, then add nodes in {doc}`pipelines`.

Python SDK calls `nirs4all.run(pipeline, dataset, ...)`. The shared finite native recipe uses Python Core `run_pipeline(data, recipe)`, R `nirs4all_run_pipeline`, Octave `nirs4all.runPipeline`, or WASM `runBrowserPipeline`. Real calls appear together in {doc}`languages`.

## I want to see which recipes will be tried

Use a **generator**, which expands choices without fitting. Two preprocessing choices and three component counts make six recipes. Inspect the list before launching 18 fold fits. See the {doc}`generator node </reference/nodes/generators>`.

A candidate list handles a small comparison. **Tuning** asks an optimizer to propose settings within a budget. Resume restores a saved compatible search; changing data or search settings starts a different experiment. Begin with fixed alternatives before {doc}`structural tuning </user_guide/models/structural_hpo>`.

## I have a saved model and new measurements

Use **predict**. Reload the predictor, pass raw measurements in the same feature/source order, and keep IDs beside the predicted values. New measurements need no labels. Follow {doc}`deployment`.

## I have new labeled data

Use **retrain** to build another predictor. Refit uses the selected recipe on its complete original training data; retraining starts a new campaign with new training data. Continuing old neural-network weights is a separate, model-specific option.

## I need uncertainty intervals

Use **calibrate** after fixing the predictor. Predict a separate labeled calibration cohort and learn an allowance from its errors. Future predictions can include intervals. See {doc}`evaluation`.

## I want to compare completed experiments

Open **results** or a **workspace**, without starting another fit. Compare held-out scores only when target, units, data and validation protocol match. See {doc}`results`.

| Operation | Needed input | Output | Example or explanation |
|---|---|---|---|
| Generate | Recipe with choices | Candidate recipes | {doc}`pipelines` |
| Run | Training data and recipe | Scores, winner and predictor | {doc}`start` |
| Tune/resume | Data, space and budget/checkpoint | Trials and selected predictor | {doc}`tutorials` |
| Predict | Fitted predictor and compatible X | Estimates and sample IDs | {doc}`deployment` |
| Retrain | New labeled data and recipe/source model | New experiment | {doc}`/user_guide/deployment/retrain_transfer` |
| Calibrate | Fixed predictor and separate labels | Calibration state and intervals | {doc}`evaluation` |
| Export/load | Completed model or experiment | Durable artifact/reloaded object | {doc}`languages` |
| Inspect/compare | Saved scores and predictions | Tables and diagnostic figures | {doc}`results` |
| Audit robustness | Fixed predictor and defined perturbations | Measured performance changes | {doc}`/user_guide/models/native_tuning_conformal` |

## Which settings change the scientific answer?

| Setting family | Examples | Why it matters |
|---|---|---|
| Data | Targets, sources, groups, partitions | Defines the scientific question |
| Recipe | Transform order, model, branch/merge | Defines the relation you learn |
| Evaluation | Folds, metric and aggregation | Defines what a score means |
| Search | Alternatives, budget, seed | Defines competing recipes |
| Execution | Engine and installed runtime | Defines executable operations |
| Storage | Workspace/export destination | Defines what can be inspected or reloaded |
| Display | Figure visibility and labels | Helps interpretation |

The CLI uses groups such as `workflow`, `results`, `tuning`, `dataset` and `workspace`; exact flags are in {doc}`/reference/cli`.

**Checkpoint:** “saved model and unlabeled measurements” means load → predict. “Two possible recipes” means compare on shared folds. Continue to {doc}`datasets`.
