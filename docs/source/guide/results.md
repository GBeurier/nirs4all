# 7. Inspect results and manage the workspace

A prediction value, an experiment, a workspace and a fitted-model export answer different needs. Preserve the smallest artifact that still supports the next operation, and keep the identities linking it to training and evaluation.

## What is stored?

| Object/artifact | Contains | Typical use |
|---|---|---|
| Training outcome | Evaluated plan, OOF, scores, selection and refit references | Audit the completed campaign |
| Prediction result | Values, sample/target identities and provenance | Analyze predictions or compute observed metrics |
| Native result directory | Manifest, score set and prediction Parquet | Reopen checked native results |
| Experiment | Result inventory, view and optional selected model | Compare and reload a complete saved experiment |
| SDK workspace | Runs, chains, metadata, prediction arrays and artifacts | Query many datasets/runs and reuse sessions |
| Model archive | Fitted predictor state and frozen input binding | Predict without the training workspace |

The SDK workspace uses SQLite metadata, Parquet arrays and content-addressed artifacts. Native result directories have a different defined layout. Opening a native result view does not imply the ability to open every historical SDK workspace. Use the workspace/session bridge for the supported portable profiles and an explicit migration for old formats.

## Inspect before comparing

Retain run ID, dataset identity, variant and fold identity, metric/direction, target names and selected-model references. Compare models evaluated on the same protocol; a favorable score from another split or target definition is not a like-for-like improvement. Filter and sort with the metric direction visible.

The native experiment APIs provide result views, comparison and prediction extraction. Python uses `open_experiment` and `save_experiment`; R uses `nirs4all_open_experiment`, `nirs4all_save_experiment` and `nirs4all_result_*`; JavaScript uses `openExperiment`; MATLAB/Octave uses `saveExperiment`, `resultView`, `resultCompare` and `resultPredictions`. MATLAB has no `openExperiment` wrapper.

## Reload and move

Use export/import to move a durable bundle. After relocation, validate member inventories, hashes, model/result links and the new runtime. A saved absolute path to the old training directory is not a portable reference. Refuse inconsistent or substituted archives before prediction. Keep atomic write behavior: failed exports must not leave a destination that looks complete.

A session reuses execution configuration and runtime resources. A loaded prediction session imports saved state and must not train or run HPO. Close native objects and workers when finished, including on exceptions. A prediction cache, checkpoint and selected-model archive have distinct lifecycles and should not be silently substituted.

Read {doc}`/reference/workspace`, {doc}`/reference/storage`, {doc}`/reference/predictions_api`, {doc}`/user_guide/predictions/session_api` and {doc}`/user_guide/predictions/analyzing_results` for exact SDK accessors, query semantics and persistence layouts.

## Modern workspace bridge (candidate cohort)

Core 0.4.4 exposes `save_workspace([experiment_path], destination)`,
`open_workspace(path)` and `import_workspace(archive, destination)`. The returned
`Workspace` queries the actual SDK SQLite store and its Parquet arrays through
`runs()` and `query_predictions(native_run_id)`. `session(native_run_id)` opens
a closeable SDK native prediction session; `export(destination)` publishes an
immutable snapshot. The supported profile contains portable native predictors;
it does not convert foreign host-model weights or import arbitrary old DuckDB
workspaces.

Opening verifies a closed file inventory and hashes, relational links, native
result/score identity and Parquet prediction projection. Active or unlisted
SQLite WAL/SHM files are refused. Altered metrics, model references or orphaned
chains remain invalid even if a file hash is recomputed. Export/import retain
sample, target, fold, variant/refit and provenance identity and reject unsafe
member paths. Writes publish atomically without replacing an existing target.

R uses `nirs4all_workspace_runs`, `nirs4all_workspace_predictions`,
`nirs4all_workspace_session`, `nirs4all_workspace_predict`,
`nirs4all_workspace_export` and `nirs4all_workspace_close`. MATLAB/Octave exposes
`Workspace.runs`, `.predictions`, `.session`, `.export` and `.close`. Both are
stateless JSON-command bridges requiring a Python executable with Core **and
the full SDK** installed. Each command reopens and validates the workspace;
a path handle does not hold a database snapshot between calls. Prediction
loads native state without FIT, and the Python command closes its resources
on completion or error. Closing the facade also invalidates its child sessions.

Browser `openWorkspace(indexBytes, members)` preserves the exact SDK bytes and
validates hashes plus native experiments. Its `predictions`/`compare` read native
experiment results; `predictMethods` uses the qualified native model profile.
It does not decode or independently validate SQLite/Parquet relational content.
The Python gateway performs that validation before exporting the snapshot.

Follow {doc}`interop` for a relocation and session lifecycle recipe.
