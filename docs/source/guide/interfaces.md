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
| Native CPU multimodal (candidate) | `run_multimodal`, `NativeMultimodal` | `nirs4all_run_multimodal`, `nirs4all_native_multimodal_export/load` | Node `runMultimodal`, `NativeMultimodal` | `nirs4all.runMultimodal`, `nirs4all.NativeMultimodal` |
| SDK workspace snapshot | `save_workspace`, `open_workspace`, `import_workspace` | `nirs4all_open_workspace`, `nirs4all_import_workspace` | `openWorkspace` (hashed bytes and native experiments) | `nirs4all.Workspace`, `Workspace.importSnapshot` |

R also has a distinct local `nirs4all_run(X, y, …)` workflow. Its inputs and fitted object are not interchangeable with `nirs4all_native_run(dataset, archive, …)`. JavaScript `tuneBrowser` uses a browser-native initial-full-refit package; native CPU tuning uses its own archive contract. Consult {doc}`deployment` before transporting either.

## Read a capability claim

A declaration describes shape and identity. Execution means an actual runtime can consume it. Qualified replay additionally verifies the persisted model, current input schema and a fresh process. Retraining performs new fitting and has its own support contract. Do not infer any of these from the mere presence of an exported symbol.

The released common dense workflow has one complete numeric source, one regression target and SNV/Savitzky–Golay/PLS selection. The raw multimodal profile has separately qualified encoders and state. General pipelines, grouped folds, ragged inputs and partial targets have their own profiles; support in the SDK does not imply support in every facade.

See {doc}`/reference/public_interfaces` for the SDK API and runtime contracts, {doc}`/reference/multimodal_execution_matrix` for multimodal routes, and {doc}`/reference/native_capability_preflight` for execution preflight. The [Core capability matrix](https://github.com/GBeurier/nirs4all-core/blob/main/docs/CAPABILITIES.md) records the portable product surfaces. Versions and actual qualified profiles must be checked together.

## Candidate cohort: Core 0.4.4 and R 0.7.1

The additions below describe the reviewed release candidate. Core 0.4.4 and
R 0.7.1 publication is still pending; this draft does not establish installed
availability. IO 0.2.6 is already public. The candidate pairs it with DAG 0.3.39
and Methods 1.3.4. See {doc}`interop` for input and runtime requirements.

The native pipeline facade is Python `run_pipeline` / `NativePipeline`, R
`nirs4all_run_pipeline`, Node `runPipeline` / `NativePipeline`, and
MATLAB/Octave `nirs4all.runPipeline` / `nirs4all.NativePipeline`. Browser
`runBrowserPipeline` / `BrowserNativePipeline` has its own WASM consumer, with
`loadBrowserPipeline` and `predictBrowserPipeline` for persisted replay. These routes use upstream
Methods operators and DAG fitting, scoring and replay. Independently compared
finite regression profiles include StandardScaler → Ridge and raw PLS. They
do not establish numerical parity for the whole Methods catalog.

IO's explicit masked matrix projection accepts multiple int64 classification
target columns, with one model and observed mask per target. False cells become
zero before float32 conversion; observed labels must be exactly representable
in float32. The complete matrix API retains its single-vector classification
contract. Ragged sources require an explicit native projection before matrix
execution; an IO record alone does not execute an encoder.

Multimodal SHAP, licensed MATLAB qualification and Windows ARM64 remain deferred.
Octave qualification is distinct from running licensed MATLAB.
