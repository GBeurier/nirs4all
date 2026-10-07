# 6. Measure whether your model answers the question

**Your goal:** choose a validation split, calculate its error, and distinguish the score used to select a recipe from the score used to assess the final predictor.

Ask first: **what will the next unknown observation be?** Another scan of a known specimen, a new specimen, a new batch or a new instrument? The split should imitate that situation. If specimens have repeated scans, separate specimens rather than random rows.

## Keep partitions distinct

| Evidence | Use | Avoid |
|---|---|---|
| Training predictions | Diagnose fitted behavior | Treating them as held-out accuracy |
| Validation/OOF predictions | Candidate selection and stacking | Fitting preprocessing on validation rows |
| Test predictions | Final external evaluation | Selecting repeated candidates on test scores |
| Refit state | Deploy the selected recipe | Reusing a fold model as the complete predictor |
| Calibration predictions | Fit a conformal calibrator | Training the point model on the calibration labels |
| New-data prediction | Apply fixed learned state | Reconstructing or refitting hidden encoders |

Specify the objective and direction together: minimize error/loss, maximize the chosen score where appropriate. Report per-target results and validity masks for partial/multiple targets. Aggregating groups, repetitions or folds requires a declared aggregation policy and must preserve the original sample/unit identities.

## Add intervals only after the point predictor is fixed

Suppose a held-out calibration set gives a supported 90% residual allowance of 2 concentration units. A new point prediction of 10 receives interval `[8, 12]`. The allowance is learned from calibration errors using the conformal method's finite-sample rule; it is not the training RMSE copied into a band. Assess actual coverage on another labeled cohort.

### Conformal intervals

Split-conformal calibration uses a fitted point predictor and an explicitly separate observed calibration cohort. Export includes the calibrator's method, exchangeability unit, supported coverage levels and cohort provenance. `predict_calibrated` applies it to future point predictions. `conformal_metrics` reports observed coverage and interval widths where truth is available.

Calibration support is profile-specific. A browser consuming a CPU archive already calibrated is different from calibrating a previously uncalibrated CPU archive in the browser. Consult the current transport matrix in {doc}`deployment`. Do not infer one from the other.

## Robustness and visual reports

A frozen robustness audit evaluates declared slices or perturbation scenarios against a fixed clean predictor. It does not replace the model or claim robustness to every distribution change. Retain the scenario definition, seed where applicable, reference cohort and metrics.

Use a chart when it helps explain a result, and accompany it with a visible conclusion and the values needed to check it. Identify units, partitions, target names and missing values. Color alone must not distinguish models or uncertainty. Exported figures need their own text alternative or adjacent data table.

For examples, see {doc}`/user_guide/scoring_and_refit`, {doc}`/reference/metrics`, {doc}`/user_guide/models/native_tuning_conformal` and {doc}`/user_guide/visualization/prediction_charts`. Multimodal SHAP for the complete refit predictor remains a separate deferred work item; explanations of one feature branch are not complete-model explanations.


## Select a splitter for the prediction question

| Intended prediction | Starting splitter | Why | Check |
|---|---|---|---|
| New independent observations from the same population | KFold or ShuffleSplit | Random held-out evaluation | Independence and sufficient training size |
| New specimens with repeated scans | GroupKFold | All scans of a specimen stay together | Group IDs really identify specimens |
| Imbalanced categorical outcome | StratifiedKFold | Keeps classes represented | Every fold has enough class observations |
| Imbalanced outcome with specimen groups | StratifiedGroupKFold | Combines class balance and group separation | Balance may be constrained by groups |
| Future measurements | TimeSeriesSplit or explicit chronological holdout | Respects the direction of time | No preprocessing or labels from the future |
| New laboratory/instrument/batch | Explicit held-out groups | Tests the acquisition shift of interest | Enough independent groups for the conclusion |

Splitter availability and grouping syntax depend on the execution profile.
The SDK
[U01_cv_strategies.py](https://github.com/GBeurier/nirs4all/blob/main/examples/user/05_cross_validation/U01_cv_strategies.py)
and
[U02_group_splitting.py](https://github.com/GBeurier/nirs4all/blob/main/examples/user/05_cross_validation/U02_group_splitting.py)
provide worked examples. The latter explicitly uses the legacy profile for
its automatic repetition-grouping demonstrations; do not transplant those
settings to native Core without checking its profile.

```{figure} /assets/guide/evaluation.svg
:alt: Repeated scans of the same specimen stay together in a fold so validation tests new specimens.

**Match validation to deployment.** All scans of a held-out specimen remain outside training. The scaler and model learn only from training specimens; held-out scans use the fixed learned state.
```

Explanation: Every scan of specimen B belongs to the
validation side; specimens A and C belong to training. The figure illustrates
one fold, not a claim that three specimens are enough for a scientific study.
This protocol asks about a new specimen rather than another scan of a known
specimen.

## Work through pooled error by hand

For regression, with residual `r = prediction - truth`, RMSE is the square root
of the mean squared residual. Its unit is the target unit. MAE is the mean
absolute residual; it weights large deviations less strongly. R² compares the
squared residuals with target variability and can be negative. A favorable R²
alone does not demonstrate sufficiently small error for your application.

Here is a deliberately small **illustrative calculation**, not an experiment
result:

| Fold | Held-out residuals | Count | Sum of squared residuals | Fold RMSE |
|---|---|---:|---:|---:|
| 0 | 1, 1 | 2 | 2 | 1 |
| 1 | 3, 3 | 2 | 18 | 3 |
| Pooled | 1, 1, 3, 3 | 4 | 20 | √5 ≈ 2.236 |

```{figure} /assets/guide/pooled_rmse.svg
:alt: Fold RMSEs of 1 and 3 average to 2, while pooling their four residuals gives RMSECV of approximately 2.236.

Illustrative RMSE aggregation, in arbitrary target units. The two folds have
the same size; their pooled RMSE still differs from the mean fold RMSE.
The table above supplies the full residuals and values. These are teaching
values, not measured nirs4all performance.
```

The mean fold RMSE is `(1 + 3) / 2 = 2`. The pooled RMSE is
`sqrt((2 + 18) / 4) ≈ 2.236`. They answer different aggregation questions;
unequal fold sizes introduce an additional weighting difference.


Explanation: Pooling residuals before taking
the square root produces approximately 2.236; averaging fold RMSEs produces
2. The table above contains all values needed to reproduce the diagram.

KFold gives each observation one validation prediction per complete CV pass.
ShuffleSplit and repeated CV can produce several held-out predictions for one
observation or leave some observations unvalidated. Keep fold membership and
prediction multiplicity visible; choose the supported aggregation policy
rather than assuming all CV evidence is one row per observation.

### Calculate the error in your language

The inputs below give residuals 1, 1, 3 and 3. JSON/YAML show the same values and expected answer. The executable tabs use the definition directly, so the error calculation does not depend on a particular model API.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "truth": [
    10,
    12,
    14,
    16
  ],
  "prediction": [
    11,
    13,
    17,
    19
  ],
  "residual": [
    1,
    1,
    3,
    3
  ],
  "rmse": 2.23606797749979,
  "mae": 2
}
```

:::

:::{tab-item} YAML
:sync: yaml

```yaml
truth: [10, 12, 14, 16]
prediction: [11, 13, 17, 19]
residual: [1, 1, 3, 3]
rmse: 2.23606797749979
mae: 2.0
```

:::

:::{tab-item} Python
:sync: python

```python
import numpy as np
truth = np.array([10, 12, 14, 16])
prediction = np.array([11, 13, 17, 19])
residual = prediction - truth
print("RMSE:", np.sqrt(np.mean(residual ** 2)))
print("MAE:", np.mean(np.abs(residual)))
```

:::

:::{tab-item} R
:sync: r

```r
truth <- c(10, 12, 14, 16)
prediction <- c(11, 13, 17, 19)
residual <- prediction - truth
print(sqrt(mean(residual^2))) # RMSE = 2.236068
print(mean(abs(residual)))   # MAE = 2
```

:::

:::{tab-item} Octave / MATLAB
:sync: matlab

```matlab
truth = [10, 12, 14, 16];
prediction = [11, 13, 17, 19];
residual = prediction - truth;
disp(sqrt(mean(residual.^2))); % RMSE = 2.236068
disp(mean(abs(residual)));     % MAE = 2
```

:::

:::{tab-item} WASM / JavaScript
:sync: javascript

```javascript
const truth = [10, 12, 14, 16];
const prediction = [11, 13, 17, 19];
const residual = prediction.map((p, i) => p - truth[i]);
const rmse = Math.sqrt(residual.reduce((s, r) => s + r*r, 0) / residual.length);
const mae = residual.reduce((s, r) => s + Math.abs(r), 0) / residual.length;
console.log({rmse, mae}); // 2.23606797749979, 2
```

:::

::::

**Expected result:** RMSE ≈ 2.236 and MAE = 2, in the target's unit. These values are illustrative and were chosen to make the pooled calculation visible.

## Separate selection from the final score

`cv_best_score` identifies CV selection evidence in the SDK result.
`final_score` refers to the refit predictor's test evidence where that cohort
exists. `best_score` is a convenience accessor; inspect its associated entry
and metric rather than treating the name as a complete definition of the
protocol. The detailed {doc}`/user_guide/scoring_and_refit` distinguishes pooled
CV score, mean fold validation score, fold-ensemble test score and refit test
score.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} YAML
:sync: yaml

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} Python
:sync: python



```python
# After result = nirs4all.run(...):
print("CV winner:", result.cv_best)
print("CV selection score:", result.cv_best_score)
print("Final refit entry:", result.final)
print("Refit test score:", result.final_score)
```

:::

:::{tab-item} R
:sync: r

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} Octave / MATLAB
:sync: matlab

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} WASM / JavaScript
:sync: javascript

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

::::

For classification, overall accuracy can hide a minority class failure.
Inspect balanced accuracy, per-class recall and a confusion matrix with class
names and counts. For continuous targets, inspect prediction-versus-truth and
residual plots by relevant acquisition groups. For multiple targets, report
each target's valid observation count and unit before any aggregate score.

Choosing repeatedly among many recipes using the same CV cohort also adapts
to that cohort. Use a separately preserved final test set, or a supported
nested evaluation protocol, when the purpose is an estimate after substantial
model search. A large candidate count is not additional independent evidence.

## What a calibrated interval says

A split-conformal interval adds an estimated residual allowance to the point
prediction, using a held-out calibration cohort. The requested coverage is a
population-level statement under the method's assumptions, not a guarantee
for every particular sample, batch or future distribution.


Explanation: Development data create the frozen
point predictor; separate labels estimate its residual distribution. New
observations receive predictions and intervals without fitting the point
predictor again.

Report achieved coverage and widths on an observed evaluation cohort, along
with its sample count and any group slices. A wider interval can increase
coverage while reducing practical usefulness. Follow the executable
[U09_native_tuning_conformal.py](https://github.com/GBeurier/nirs4all/blob/main/examples/user/04_models/U09_native_tuning_conformal.py)
for tuning, calibration, prediction and robustness persistence; use
[U10_native_pls_conformal_robustness.py](https://github.com/GBeurier/nirs4all/blob/main/examples/user/04_models/U10_native_pls_conformal_robustness.py)
for the PLS version. Host transport and browser calibration remain bounded by
{doc}`deployment`, even when a language exposes a calibrate symbol.

## Choose diagnostic figures with a stated question

| Figure | Question | Include in its text alternative |
|---|---|---|
| Predicted versus observed | Does the model track the target range? | Partition, unit, count, error and major bias |
| Residual versus target | Does error change with concentration? | Residual sign convention and range-dependent pattern |
| Fold score table/plot | Is selection stable across folds? | Fold sizes, score values and aggregation |
| Confusion matrix | Which classes are confused? | Labeled row/column counts and per-class recall |
| Source/model comparison | Which recipe helps on the same protocol? | Candidate identities, shared cohort and metric direction |
| Coverage/width report | Are intervals useful at the requested coverage? | Requested/observed coverage, widths and sample count |

Keep plotted values beside a summary or data table. Separate measured results
from the illustrative figures on this page, and identify masked/missing
observations rather than silently dropping them from a performance claim.

**Checkpoint:** explain why splitting three scans of the same specimen across train/validation can inflate performance. State which score selected the recipe, and whether you have an untouched external test score. Continue to {doc}`results`.
