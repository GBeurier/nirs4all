# 0. Public interfaces and availability

Choose the product before choosing the function. `nirs4all` on PyPI is the full Python SDK; `nirs4all-core` exposes the portable Python facade. The R product is maintained in its own repository. npm and Rust use the package name `nirs4all`; MATLAB/Octave uses the `+nirs4all` namespace.

| Product | Import or namespace | Runtime |
|---|---|---|
| Python SDK | `import nirs4all` | Python host controllers and native DAG/Methods |
| Python Core | `import nirs4all_core` or `import n4a` | Native Core CLI and optional upstream bindings |
| R | `library(nirs4all)` | R bindings; Core CLI for the native workflow profile |
| JavaScript/TypeScript | `import … from 'nirs4all'` | Core, DAG, IO and Methods WASM |
| MATLAB/Octave | `nirs4all.*` | Core CLI and Methods native bindings/MEX |
| Rust | `nirs4all` crate | Native Rust coordination and dynamic Methods library |
| CLI | `nirs4all workflow …` | SDK command frontend to the native workflow |

## Which function performs my task?

| Task | Python Core | R native product | JavaScript/WASM | MATLAB/Octave |
|---|---|---|---|---|
| Declare a dataset | `dataset`, `Dataset` | `nirs4all_dataset` | `dataset` | `nirs4all.dataset` |
| Evaluate, select and refit | `run` | `nirs4all_native_run` | `run` | `nirs4all.run` |
| Predict | `predict` | `nirs4all_native_predict` | `predict` | `nirs4all.predict` |
| Export and reload workflow | `export`, `load` | `nirs4all_native_export`, `nirs4all_native_load` | `exportWorkflow`, `load` | `nirs4all.export`, `nirs4all.load` |
| Retrain | `retrain` | `nirs4all_native_retrain` | `retrain` | `nirs4all.retrain` |
| Native tuning | `tune`, `resume_tuning` | `nirs4all_tune`, `nirs4all_resume_tuning` | `tune`, `tuneBrowser` | `nirs4all.tune`, `nirs4all.resumeTuning` |
| Open saved results | `open_experiment` | `nirs4all_open_experiment` | `openExperiment` | `nirs4all.resultView` |
| Calibrate and audit | `calibrate`, `robustness` | `nirs4all_calibrate`, `nirs4all_robustness` | `calibrate`, `robustness` | `nirs4all.calibrate`, `nirs4all.robustness` |
| Raw multimodal predictor | `MultimodalPredictor` | `nirs4all_multimodal_*` | `MultimodalPredictor` | `nirs4all.MultimodalPredictor` |

R also has a distinct local `nirs4all_run(X, y, …)` workflow. Its inputs and fitted object are not interchangeable with `nirs4all_native_run(dataset, archive, …)`. JavaScript `tuneBrowser` uses a browser-native initial-full-refit package; native CPU tuning uses its own archive contract. Consult {doc}`deployment` before transporting either.

## Read a capability claim

A declaration describes shape and identity. Execution means an actual runtime can consume it. Qualified replay additionally verifies the persisted model, current input schema and a fresh process. Retraining performs new fitting and has its own support contract. Do not infer any of these from the mere presence of an exported symbol.

The released common dense workflow has one complete numeric source, one regression target and SNV/Savitzky–Golay/PLS selection. The raw multimodal profile has separately qualified encoders and state. General pipelines, grouped folds, ragged inputs and partial targets have their own profiles; support in the SDK does not imply support in every facade.

See {doc}`/reference/public_interfaces` for the SDK API and runtime contracts, {doc}`/reference/multimodal_execution_matrix` for multimodal routes, and {doc}`/reference/native_capability_preflight` for execution preflight. The [Core capability matrix](https://github.com/GBeurier/nirs4all-core/blob/main/docs/CAPABILITIES.md) records the portable product surfaces. Versions and actual qualified profiles must be checked together.
