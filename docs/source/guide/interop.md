# Native pipeline and workspace interoperability

This recipe targets the Core 0.4.4 / R 0.7.1 candidate cohort described in
{doc}`interfaces`. Publication is pending. It separates a portable fitted model
from a modern SDK workspace, so each consumer uses the correct artifact.

## Prepare the runtime and data

Use Python 3.11+, Core and IO 0.2.6, the cohort's DAG binding and Methods native
library. Set `NIRS4ALL_CORE_CLI` to the matching Core executable and
`N4M_LIBRARY_PATH` to the matching Methods shared library. Workspace commands
also require the full Python SDK. A Methods library path on CPU is distinct
from the Methods WASM module loaded in a browser.

Download the {download}`dense dataset </_downloads/dense-workflow.dataset.json>`
as `dense-workflow.dataset.json`. Its twelve synthetic samples exercise software
behavior; they are not an instrument validation cohort.

## Fit and reload a finite native recipe

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from nirs4all_core import NativePipeline, run_pipeline

data = json.loads(Path("dense-workflow.dataset.json").read_text())
recipe = {
    "steps": [
        {"method_id": "preprocessing.scaling.standard_scale",
         "role": "transformer", "params": {}},
        {"method_id": "models.regularized.ridge",
         "role": "regressor", "params": {"scale_x": False}},
    ],
    "candidates": [{"alpha": 0.1}, {"alpha": 1.0}],
}
model = run_pipeline(data, recipe)
x = data["dataset"]["sources"][0]["array"]["values"][:2]
with TemporaryDirectory() as directory:
    path = model.export(Path(directory) / "ridge.json")
    loaded = NativePipeline.load(path)
    replay = loaded.predict(x, sample_ids=["demo:new:0", "demo:new:1"])
    assert {item["phase"] for item in replay["lineage"]} == {"PREDICT"}
    print(replay["outputs"][0]["predictions"][0]["values"])
```

The original training invocation owns CV, out-of-fold scoring, candidate
selection and refit. Loading and predicting imports that selected state. Use a
fresh process and independent data for a deployment qualification; the two-row
replay above only demonstrates the API and lifecycle. Raw PLS has a separately
compared profile. Additional catalog methods require their own evidence.

For partial labels, use IO's explicit `to_masked_matrix_regression` (JavaScript
`toMaskedMatrixRegression`) projection. A classification target matrix must
have int64 dtype, distinct target names and a boolean mask of the same shape.
Each native model selects its own target column and observed rows before FIT,
refit and scoring. False cells are normalized to zero, including null or large
placeholders; a true cell must be finite and its class exactly representable in
float32. No implicit ordinal remapping occurs. The complete matrix projection
continues to reject a classification target matrix.

## Relocate a modern SDK workspace

This second recipe requires an existing **native experiment directory** with a
selected portable Archive V2 model. It is not the `ridge.json` record above.
Create such an experiment with `save_experiment(native_results_dir, destination,
run_id=..., winner_variant_id=..., model_archive=...)`, using identifiers from
the actual native result manifest. Set `EXPERIMENT_PATH` to its directory and
`PREDICTION_INPUT` to a JSON file containing `x` (rows in the frozen model's
feature order) and distinct `sample_ids` for a new cohort.

```python
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from nirs4all_core import import_workspace, save_workspace

cohort = json.loads(Path(os.environ["PREDICTION_INPUT"]).read_text())
with TemporaryDirectory() as directory:
    root = Path(directory)
    with save_workspace([os.environ["EXPERIMENT_PATH"]], root / "workspace") as workspace:
        run_id = workspace.runs()[0]["native_run_id"]
        rows = workspace.query_predictions(run_id)
        assert rows
        archive = workspace.export(root / "snapshot.n4w")
    with import_workspace(archive, root / "relocated") as relocated:
        assert len(relocated.query_predictions(run_id)) == len(rows)
        session = relocated.session(run_id)
        try:
            predicted = session.predict(cohort["x"], sample_ids=cohort["sample_ids"])
            assert predicted.y_pred.shape[0] == len(cohort["sample_ids"])
        finally:
            session.close()
```

The snapshot includes the SDK SQLite metadata, Parquet arrays and native
experiments/models. The importer verifies both inventory and semantic links.
It refuses active journals and changed metadata rather than trusting SQLite
alone. A loaded session predicts without fitting. Keep the snapshot after
qualification when it is your durable deliverable; this example uses temporary
directories for repeatable execution.

R opens the relocated directory with `nirs4all_open_workspace(path, python=...)`;
MATLAB/Octave uses `nirs4all.Workspace(path, python)`. The supplied Python
executable must import Core and the full SDK. These handles issue JSON commands
with argument arrays, reopen the snapshot per command and close Python/native
resources before returning. Close the workspace and child session when done;
operations after close fail. They do not retain a database snapshot between
commands. Paths containing spaces or Unicode need no shell quoting inside the
API call.

Browser `openWorkspace` accepts index bytes plus the exact member bytes. It
checks hashes and native experiments and exposes native result queries and
`predictMethods`; it does not open a SQL engine. Preserve its exported bytes
when handing the workspace back to the SDK gateway.

## Transport browser tuning state

A browser SNV/Savitzky–Golay/PLS study exported as
`nirs4all.browser-tuning.v1` is consumed on CPU with
`load_browser_tuning(path).predict(target_free_dataset)`. Keep its package,
search and optimizer snapshot together; the native validation gate rejects
altered study provenance. CPU prediction reuses Methods state without FIT.
Optimizer resumption on CPU is outside this transport profile.

For CPU → browser conformal calibration, use the separately qualified
`methods.pls` Archive V2 profile and explicit calibration/test cohorts. DAG
replays the frozen predictor with the browser controller, then fits the
calibrator from observed calibration targets. See {doc}`deployment` for the
direction-specific contracts and PLS-LDA reference limits, and {doc}`results`
for workspace validation and resource ownership.
