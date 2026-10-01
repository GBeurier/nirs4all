# Octave models in a Python multimodal workflow

The public Python SDK can schedule native DAG training and host HPO while
Methods models execute in an Octave process. The shipped example uses the
existing twelve-sample numeric NIR, image, series and metadata fixture, three
recorded parameter proposals, and a meta-model trained on out-of-fold branch
predictions. It captures all four source Ridge models and the meta Ridge in one
portable `.n4a` archive. The selected fixture recipe uses `alpha=0.05`.

This development workflow requires a real Octave runtime and compiled Methods
MEX bindings. Its real Octave qualification remains pending until that runtime
is available. The inputs are numeric projections; raw N-D encoders and licensed
MATLAB are outside this example's qualification.

## Prepare the runtime

Install matching `nirs4all`, `dag_ml` and `nirs4all_core` Python wheels. Keep the
DAG-ML checkout containing the shipped Octave adapter and the Methods checkout
containing `bindings/matlab/+n4m/RolePipeline.m`. Use a compiled Methods MEX
linked to the released public ABI 2.14 library. To build it, supply the headers
from that same ABI release and run the Methods build helper:

```bash
export N4M_INCLUDE_DIR=/absolute/path/to/public-abi214/include
export N4M_GENERATED_DIR=/absolute/path/to/public-abi214/include
export N4M_LIB_DIR=/absolute/path/to/public-abi214/lib
cd /absolute/path/to/nirs4all-methods/bindings/matlab
octave --no-gui --no-history --eval "build_mex({'n4m_role_pipeline_mex','n4m_version_mex'})"
```

Select the installed binding locations for the external controller:

```bash
export DAG_ML_OCTAVE_METHODS_PATH=/absolute/path/to/nirs4all-methods/bindings/matlab
export DAG_ML_OCTAVE_MEX_PATH=/absolute/path/to/compiled-mex-directory
export LD_LIBRARY_PATH=/absolute/path/to/public-abi214/lib:${LD_LIBRARY_PATH:-}
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

## Use the same SDK entry points with another controller

Both training functions pass the public DAG-ML contracts and callbacks through
unchanged, including candidate-local controller factories, progress callbacks
and durable HPO checkpoints. R controllers can use the same SDK entry points.
`execute_training` returns the native `TrainingResult` with typed outcome and
portable export; close its attached lifetime with `detach()` after capture.
The higher-level `run(pipeline, dataset)` retains its normal `RunResult` API.

See {doc}`/reference/public_interfaces` for the portable archive callbacks and
{doc}`multimodal` for the general Python multimodal workflow.
