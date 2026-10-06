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
