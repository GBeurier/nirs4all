# Octave models in a Python multimodal workflow

The public Python SDK can schedule native DAG training and host HPO while
Methods models execute in an Octave process. The shipped example uses the
existing twelve-sample numeric NIR, image, series and metadata fixture, three
recorded parameter proposals, and a meta-model trained on out-of-fold branch
predictions. It captures all four source Ridge models and the meta Ridge in one
portable `.n4a` archive. The selected fixture recipe uses `alpha=0.05`.

This workflow requires Octave 7.1 or newer with built-in `jsonencode` and
`jsondecode`, plus compiled Methods MEX bindings. Missing runtime prerequisites
are failures in the canonical example runner. The inputs are numeric
projections; raw N-D encoders and licensed MATLAB are separate profiles.

## Prepare the runtime

Install matching `nirs4all`, `dag_ml` and `nirs4all_core` Python wheels. Keep the
DAG-ML checkout containing the shipped Octave adapter and the Methods checkout
containing `bindings/matlab/+n4m/RolePipeline.m`. The current release cohort is
SDK 1.6.0, DAG-ML 0.5.0, Core 0.6.0 and Methods 1.5.1 with ABI 2.24.0. Build the
Methods MEX against the actual released library and headers from that same
release. The generated export header must match its platform and ABI:

```bash
export N4M_INCLUDE_DIR=/absolute/path/to/methods-1.5.1/cpp/include
export N4M_GENERATED_DIR=/absolute/path/to/methods-1.5.1/build/dev-release/generated
export N4M_LIB_DIR=/absolute/path/to/public-methods-1.5.1/lib
cd /absolute/path/to/nirs4all-methods/bindings/matlab
octave --no-gui --no-history --eval "build_mex({'n4m_role_pipeline_mex','n4m_version_mex'})"
```

Select the installed binding locations for the external controller:

```bash
export DAG_ML_OCTAVE_METHODS_PATH=/absolute/path/to/nirs4all-methods/bindings/matlab
export DAG_ML_OCTAVE_MEX_PATH=/absolute/path/to/compiled-mex-directory
export LD_LIBRARY_PATH=/absolute/path/to/public-methods-1.5.1/lib:${LD_LIBRARY_PATH:-}
```

Use the fresh capture emitted by DAG-ML's
`scripts/smoke_wasm_multimodal_methods_hpo.mjs`, which is also the R
qualification input. It already contains the data identities, folds, three
proposals and reference results; this example does not create a new dataset.

## Train and save all five models

From the SDK checkout, run the example with your installed Python interpreter.
Choose a new output directory:

```bash
python examples/user/02_data_handling/U15_octave_multimodal_archive.py \
  --dag-ml-root /absolute/path/to/dag-ml \
  --octave /absolute/path/to/octave \
  --workdir /absolute/path/to/new-training-output \
  train --node-capture /absolute/path/to/fresh-node-capture.json
```

The example calls `nirs4all.run_host_hpo_search(...)` for all three proposals
and `nirs4all.execute_training(...)` for native CV, SELECT and REFIT. DAG-ML
owns the folds, scores and selection, and Octave uses the public native
`n4m.RolePipeline` binding for the models. The example exports the captured
portable package with `nirs4all.write_portable_predictor_archive_v2(...)`.
Outputs include `five-octave-models.n4a` and `heldout-replay.json`, which contains
only current heldout sources and signed prediction contracts.

The archive is a Core V1 container with a DAG-ML Package V2 root. Container and
package versions are independent. Core's `read_archive_v2_payloads` API returns
the validated V1 manifest, selected physical members, exact package/outcome
bytes and native view. Fitted-state members follow the selected model's declared
facet paths; they are not an old physical Archive V2 inventory.

## Predict in a new process

Start a separate installed Python process. The prediction command opens a new
Octave worker with fitting disabled and supplies no targets:

```bash
python -I examples/user/02_data_handling/U15_octave_multimodal_archive.py \
  --dag-ml-root /absolute/path/to/dag-ml \
  --octave /absolute/path/to/octave \
  --workdir /absolute/path/to/new-prediction-output \
  predict \
  --archive /absolute/path/to/new-training-output/five-octave-models.n4a \
  --replay-inputs /absolute/path/to/new-training-output/heldout-replay.json
```

`nirs4all.read_portable_predictor_archive_v2(...)` validates the archive before
loading states. `nirs4all.replay_portable_predictor_archive_v2(...)` checks the
signed current cohort and independently trusted current controller manifest,
hydrates the five models, predicts and releases their native states. The output
`predictions.json` retains sample IDs, target names, predictions and lifecycle
counts. The training data and an attached training session are unnecessary for
this step.

For the canonical example runner, provide the same real runtime and fresh Node
capture explicitly:

```bash
export NIRS4ALL_CI_U15_DAG_ROOT=/absolute/path/to/dag-ml
export NIRS4ALL_CI_U15_OCTAVE=/absolute/path/to/octave
export NIRS4ALL_CI_U15_NODE_CAPTURE=/absolute/path/to/fresh-node-capture.json
cd examples
bash ./run.sh
```

The runner performs both U15 commands in separate Python interpreters and new
output directories. It fails if the inputs are absent or either operation fails.

## Use the same SDK entry points with another controller

Both training functions pass the public DAG-ML contracts and callbacks through
unchanged, including candidate-local controller factories, progress callbacks
and durable HPO checkpoints. R controllers can use the same SDK entry points.
`execute_training` returns the native `TrainingResult` with typed outcome and
portable export; close its attached lifetime with `detach()` after capture.
The higher-level `run(pipeline, dataset)` retains its normal `RunResult` API.

See {doc}`/reference/public_interfaces` for the portable archive callbacks and
{doc}`multimodal` for the general Python multimodal workflow.
