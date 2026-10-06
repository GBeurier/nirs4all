# 6. Evaluate, select and calibrate

Choose the evaluation protocol before choosing the model. The unit of independence may be a specimen, instrument batch, subject or acquisition group rather than an individual row. Use folds and scoring that reflect the intended prediction task.

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

## Conformal intervals

Split-conformal calibration uses a fitted point predictor and an explicitly separate observed calibration cohort. Export includes the calibrator's method, exchangeability unit, supported coverage levels and cohort provenance. `predict_calibrated` applies it to future point predictions. `conformal_metrics` reports observed coverage and interval widths where truth is available.

Calibration support is profile-specific. A browser consuming a CPU archive already calibrated is different from calibrating a previously uncalibrated CPU archive in the browser. Consult the current transport matrix in {doc}`deployment`. Do not infer one from the other.

## Robustness and visual reports

A frozen robustness audit evaluates declared slices or perturbation scenarios against a fixed clean predictor. It does not replace the model or claim robustness to every distribution change. Retain the scenario definition, seed where applicable, reference cohort and metrics.

Use a chart when it helps explain a result, and accompany it with a visible conclusion and the values needed to check it. Identify units, partitions, target names and missing values. Color alone must not distinguish models or uncertainty. Exported figures need their own text alternative or adjacent data table.

For examples, see {doc}`/user_guide/scoring_and_refit`, {doc}`/reference/metrics`, {doc}`/user_guide/models/native_tuning_conformal` and {doc}`/user_guide/visualization/prediction_charts`. Multimodal SHAP for the complete refit predictor remains a separate deferred work item; explanations of one feature branch are not complete-model explanations.
