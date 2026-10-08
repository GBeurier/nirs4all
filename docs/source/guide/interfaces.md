# 0. Choose the tools for your language

**Your goal:** install the package that runs your examples and understand which recipe style it accepts. If you just want to begin, choose your language in {doc}`start` and follow its shared StandardScaler → Ridge exercise.

There are two useful entry points: the **full Python SDK** accepts Python objects and a broad range of workflows; the **portable native tools** run supported shared numerical methods from Python, R, Octave and WASM. A recipe written as a Python class path needs Python. A portable recipe uses method IDs such as `models.regularized.ridge`.

Choose the product before choosing the function. `nirs4all` on PyPI is the full Python SDK; `nirs4all-core` exposes the portable Python facade. The R product is maintained in its own repository. npm and Rust use the package name `nirs4all`; MATLAB/Octave uses the `+nirs4all` namespace.

| Product | Import or namespace | Runtime |
|---|---|---|
| Python SDK | `import nirs4all` | Python host controllers and native DAG/Methods |
| Python Core | `import nirs4all_core` or `import n4a` | Current wheel's embedded native dispatcher, with explicit/source-only CLI route available |
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
| Native CPU multimodal | `run_multimodal`, `NativeMultimodal` | `nirs4all_run_multimodal`, `nirs4all_native_multimodal_export/load` | Node `runMultimodal`, `NativeMultimodal` | `nirs4all.runMultimodal`, `nirs4all.NativeMultimodal` |
| SDK workspace snapshot | `save_workspace`, `open_workspace`, `import_workspace` | `nirs4all_open_workspace`, `nirs4all_import_workspace` | `openWorkspace` (hashed bytes and native experiments) | `nirs4all.Workspace`, `Workspace.importSnapshot` |

R also has a distinct local `nirs4all_run(X, y, …)` workflow. Its inputs and fitted object are not interchangeable with `nirs4all_native_run(dataset, archive, …)`. JavaScript `tuneBrowser` uses a browser-native initial-full-refit package; native CPU tuning uses its own archive contract. Consult {doc}`deployment` before transporting either.

## What does “supported” mean for an example?

Four separate questions matter:

1. Can the package **load your input** with its shape, units and missing values?
2. Can its runtime **fit this recipe** with the requested nodes and splits?
3. Can it **save and reload the learned predictor**?
4. Can your intended **other language** reload that export and predict?

Answer them in order. A JSON parser accepting an image does not mean a spectral model can fit it. A Python model saved successfully does not automatically become a browser model.

### Read the available workflow profiles

A declaration describes shape and identity. Execution means an actual runtime can consume it. Qualified replay additionally verifies the persisted model, current input schema and a fresh process. Retraining performs new fitting and has its own support contract. Do not infer any of these from the mere presence of an exported symbol.

The released common dense workflow has one complete numeric source, one regression target and SNV/Savitzky–Golay/PLS selection. The raw multimodal profile has separately qualified encoders and state. General pipelines, grouped folds, ragged inputs and partial targets have their own profiles; support in the SDK does not imply support in every facade.

See {doc}`/reference/public_interfaces` for the SDK API and runtime contracts, {doc}`/reference/multimodal_execution_matrix` for multimodal routes, and {doc}`/reference/native_capability_preflight` for execution preflight. The [Core capability matrix](https://github.com/GBeurier/nirs4all-core/blob/main/docs/CAPABILITIES.md) records the portable product surfaces. Versions and actual qualified profiles must be checked together.

## Current published packages

The following package and source releases are published as of 8 October 2026.
Distribution status is explicit where a registry build is pending. Each package
retains its own supported profiles; this inventory does not establish that every
cross-language combination has been qualified.

| Product | Public version | Runtime boundary |
|---|---|---|
| Python SDK | 1.4.7 | Python execution with the supported native/DAG routes |
| DAG | 0.3.41 | Public Python, WASM, Rust and R bindings |
| Core | 0.4.5 | Portable host facades; its compiled Rust dependency is DAG 0.3.41 |
| IO | 0.2.6 | Dataset assembly and explicit matrix/mask projections |
| Methods | 1.3.4 | Numerical engine, ABI 2.17.0; the `pls4all` companion is also 1.3.4 |
| Formats | 0.2.11 | Reader bindings; both Python and Rust use 0.2.11 |
| R product | 0.7.2 | R native workflow and workspace facades; GitHub source and R-universe distributions published |
| UI | 0.1.15 | Shared React components |
| Cluster | 0.1.5 | Distributed orchestration and prediction evidence |
| Studio | 0.15.2 | Public desktop installers and Docker using SDK 1.4.7, DAG 0.3.41 and Core 0.4.5; unpublished 0.15.1 is superseded |

The standalone DAG package and Core's compiled DAG dependency are separate
runtime boundaries. Both are now 0.3.41; installing another standalone DAG
version does not replace Core 0.4.5's compiled implementation. Record both
versions when describing a runtime cohort.

## Versions used by the shared native examples

The historical shared-example qualification uses Core 0.4.4 and R 0.7.1 with IO 0.2.6,
DAG 0.3.39 and Methods 1.3.4. These recorded results retain their original cohort.
Their qualification covers the explicit profiles described here;
other operator combinations require their own checks. See {doc}`interop` for
input and runtime requirements.

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


## When the full Python SDK asks for an engine

An **engine** is the runtime executing a Python recipe. General `run` selects a supported native or DAG route before fitting. An explicit engine request is strict and can reject unsupported recipes:

- `native` runs its supported portable model/data combinations.
- `dag-ml` schedules broader graph workflows, including the advanced Python examples in this guide.
- `legacy` runs the direct Python controller workflow when explicitly requested.

For the first portable recipe, you do not need to choose an SDK engine. For an advanced Python example, use its documented engine so you reproduce its behavior.


Explanation: general `run` chooses a supported native or DAG path before
execution. An explicit selector is strict. The direct Python legacy path is
requested explicitly; a runtime failure does not authorize silently changing
engines and continuing with different semantics.

| Question | Portable facade | Full SDK / host workflow |
|---|---|---|
| What is a step? | A method ID, role and typed parameter map in a supported recipe | An operator object or supported configuration keyword |
| Where is it fitted? | Methods, with DAG phase scheduling | The selected native/DAG controller or explicit legacy host controller |
| What can reload? | The declared native state profile and signed recipe | The artifact's compatible native or host runtime |
| Can any Python object cross languages? | No; its portable method/state must exist | Host serialization usually needs its original libraries |
| Is every source executable? | Only accepted representations and projections | Runtime/controller support is checked separately from data parsing |

Start with the common workflow when it matches your task. Use the finite
`run_pipeline` recipe when you need individually fitted catalog-native nodes.
Use a multimodal task for typed encoders/fusion, and the full SDK for broader
Python workflows. The choice should follow required operators, masks, folds,
artifact consumer and scientific evaluation protocol.

## Operator, controller, node, role and phase

An **operator** is the computation, such as SNV, standardization or Ridge.
A **controller** adapts that operation to execution: accepted representations,
fit/predict calls, serialization and resource ownership. A **node** is one
configured operation in a graph; two nodes may use the same operator with
different parameters or source bindings. A **role** declares its task:
transformer/selector before a regressor/classifier, for the finite recipe.
A **phase** says why it runs: fold fitting, validation prediction, selection,
full refit or new-data replay.

Consequently, `SNV` being present in a method catalog does not establish that a
specific controller can apply it to an image tensor. A transform with no learned
parameters still needs a schema-preserving replay path. A model accepting
multiple targets must have a declared target ordering and per-target mask
policy. These distinctions explain many preflight errors before any fitting
starts; consult {doc}`pipelines` and {doc}`/reference/native_capability_preflight`.

## Recognize the returned object

| Operation | What you inspect | What you persist |
|---|---|---|
| Common workflow `run` | Candidate outcomes and selected predictor | Workflow export plus its native model contract |
| Finite native `run_pipeline` | `outcome`, effective plan and per-node refit artifacts | `NativePipeline.export` package JSON |
| Browser `runBrowserPipeline` | Training outcome, native role artifacts and replay lineage | Exact `model.export()` text |
| SDK `nirs4all.run` | `RunResult`, ranked predictions and selected models | Documented SDK export/session artifacts |
| Workspace query | Run metadata, arrays, experiments and saved models | `.n4w` snapshot or the workspace directory |

Do not infer a reader from a filename extension. `ridge.native.json` is a
pipeline package, `model.n4a` is a model archive and `snapshot.n4w` is a workspace
snapshot. Their load functions validate different envelopes. See
{doc}`deployment` for the lifecycle and {doc}`languages` for exact call signatures.
