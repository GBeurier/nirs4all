# 2. Understand the workflow

A dataset describes the observations and their sources; a pipeline describes how features and targets are transformed; an experiment records evaluated variants; a run has its own identity and provenance; a workspace stores runs, results and artifacts; a session provides a reusable execution or prediction context.

## Follow identity through execution

Rows are joined by stable sample identity. A physical sample can have several measurements or representations. Source order, row position, repetition identity and statistical independence are different concepts. Keep sample, observation, group, repetition and origin identities explicit when they affect validation.

| Stage | What is learned or selected | Evidence to retain |
|---|---|---|
| Declare | Input representations and recipe | Dataset schema, source IDs, units and pipeline parameters |
| Plan | Variants, folds, fitting scope and dependency order | Graph/plan identity, fold membership and leakage policy |
| FIT_CV | Preprocessors and models fitted separately on training folds | Per-fold fitted state and held-out predictions |
| Select | Candidate ranked by the chosen validation objective | Scores, direction, tie policy and winner |
| REFIT | Selected recipe fitted on its permitted complete training scope | Final learned state, source binding and target names |
| PREDICT | Saved state applied to a new cohort | New sample IDs, predictions and replay provenance |
| Calibrate | Residual distribution on a separate calibration cohort | Calibrator, cohort identity, method and coverage |

OOF predictions are held-out predictions for training-cohort observations. They are useful for evaluation and stacking because their fitted ancestor did not train on the observation being predicted. A full-data fit prediction is not a substitute for OOF evidence. Native DAG validation owns this distinction across hosts.

## Distinguish four operations

Refit follows candidate selection and fits its recipe on the allowed training scope. Retraining starts a new fitting campaign on a new dataset. Hyperparameter search evaluates candidate settings. Continuing learned weights requires an explicitly supported model-specific warm-start contract. A function named `retrain` does not imply all four operations.

## Know where computation happens

DAG-ML coordinates phases, identities, OOF, scoring and selection. Methods owns numerical estimators and learned native state. IO assembles and validates datasets; Formats reads vendor payloads. Core composes those services. Host controllers support their own operator/model families. Host-specific fitted weights remain host-specific unless their declared artifact profile is portable.

Read {doc}`/concepts/mental_models`, {doc}`/concepts/cross_validation`, {doc}`/concepts/branching_and_merging` and {doc}`/migration/native_v1` for the detailed execution contracts.
