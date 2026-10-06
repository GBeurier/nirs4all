# 8. Export, replay and deploy

Deploy a selected fitted predictor together with its required input schema. Replay loads the saved preprocessing and model state, checks the new cohort and predicts without fitting. It should work after the training workspace has been removed or moved.

## Decide what is portable

A portable definition describes a recipe. A host can execute that recipe only when its operators/controllers exist. A fitted host model may still be tied to Python, R or a JavaScript library. Methods native artifacts use shared numerical implementations and declared portable state profiles. Core `.n4a` containers validate inventories and storage; DAG validates model/recipe/replay closure. Neither layer converts arbitrary host weights.

| Artifact path | Established contract | Check before reuse |
|---|---|---|
| CPU native Archive V2 | Native selected predictor with frozen raw schema | Supported N4MM/N4ME/N4MF profile, Methods ABI, IDs and target names |
| Browser role pipeline | WASM Methods state and native initial-full-refit package where documented | Actual producer/consumer package schema and replay implementation |
| CPU archive consumed by browser | Qualified C-native replay profiles | Input columns, axes, units and artifact profile |
| Host estimator export | The estimator library's serialization contract | Same compatible host runtime and model family |
| Workflow export directory/record | User choices plus model and outcome | Native-validated outcome and matching config |

Archive direction matters. CPU → browser replay does not by itself prove browser → CPU transport. Prediction support does not imply retraining or new calibration support on that host. Test each public consumer through the corresponding profile, not by manually renaming bytes.

## New-data contract

New sample IDs differ from training IDs but must be unique and aligned across sources. Source IDs, representation, non-sample dimensions, feature order, dtype policy, coordinate units and target ordering follow the saved schema. Target-free inference cannot request labels. Presence and target masks retain their separate meanings.

## Test the deployed consumer

1. Export the selected refit state without another fit.
2. Remove access to the training workspace and fitting/HPO callbacks.
3. Start a fresh process or worker with the installed consumer packages.
4. Load the exported artifact and independently declared new data.
5. Compare predictions and sample/target ordering to the producer's reference.
6. Check rejection of wrong units, reordered columns, forged inventories and unsupported profiles.

Persist results and provenance when predictions must be audited. Use {doc}`/user_guide/deployment/export_bundles`, {doc}`/user_guide/deployment/prediction_model_reuse`, {doc}`/user_guide/deployment/retrain_transfer` and {doc}`/user_guide/predictions/exporting_models` for detailed tasks.

## Browser and CPU state transport (candidate cohort)

The bounded browser SNV/Savitzky–Golay/PLS tuning producer exports
`nirs4all.browser-tuning.v1` with an initial-full-refit package. Python Core
`load_browser_tuning(record_or_path).predict(prediction_dataset)` validates its
native package, HPO request/search/checkpoint provenance and frozen source
binding, then rehydrates Methods state and predicts on CPU without FIT.
Target-free prediction rows must retain the trained source schema. Loading this
record does not resume its browser optimizer on CPU.

In the reverse direction, a supported CPU Archive V2 with C-native Methods
state can be loaded by the browser consumer. The `methods.pls` replay profile
also supports browser conformal calibration from real calibration inputs via
DAG's controller replay. Calibration predicts with the frozen model and fits a
calibrator; it does not refit the predictor. Use disjoint calibration and test
cohorts with explicit IDs. The initial browser HPO role-pipeline package and a
CPU Archive V2 remain separate contracts.

Generic N4ME operator replay and these bounded tuning/calibration profiles have
separate qualification evidence. Neither implies support for arbitrary host
weights, every optimizer, every operator or every cross-language archive path.

For the native PLS-LDA classification profile, the independent reference uses
sklearn PLS scores followed by NumPy class statistics and pooled covariance
normalized by `(n - k)` (observations minus classes). It is not exact parity
with sklearn LDA's SVD solver. This profile does not expose `predict_proba`.
Class identity, masks and predictions require their own classification checks;
regression parity alone does not qualify them.
