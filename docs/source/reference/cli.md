# CLI Reference

The `nirs4all` command currently provides diagnostics, configuration validation, dataset inspection, workspace queries, and artifact maintenance.

:::{important}
This repository does not expose a native `nirs4all run` training command yet. Use `nirs4all.run(...)` from Python, or call that Python API from R/Julia/JavaScript wrappers as shown in {doc}`public_interfaces`.
:::

## Root Commands

```bash
nirs4all --version
nirs4all --test-install
nirs4all --test-integration
```

| Command | Purpose |
| --- | --- |
| `--version` | Print the installed package version. |
| `--test-install` | Print dependency/install diagnostics. |
| `--test-integration` | Run the packaged sample-data integration smoke test. |

## Config Commands

```bash
nirs4all config validate pipeline.yaml --type pipeline --check-imports
nirs4all config validate dataset.yaml --type dataset
nirs4all config schema pipeline
nirs4all config schema dataset
```

| Command | Options | Purpose |
| --- | --- | --- |
| `config validate <config_file>` | `--type pipeline|dataset|auto`, `--check-files`, `--no-check-files`, `--check-imports` | Validate JSON/YAML config files. |
| `config schema pipeline|dataset` | none | Print the JSON schema. |

## Dataset Commands

```bash
nirs4all dataset validate dataset.yaml
nirs4all dataset validate dataset.yaml --format json
nirs4all dataset inspect dataset.yaml --detect
nirs4all dataset export dataset.yaml --format json --output normalized_dataset.json
nirs4all dataset diff dataset_a.yaml dataset_b.yaml
```

| Command | Options | Purpose |
| --- | --- | --- |
| `dataset validate <config_file>` | `--check-files`, `--no-check-files`, `--verbose`, `--format text|json` | Validate dataset configuration or folder path. |
| `dataset inspect <config_file>` | `--detect` | Show sources, task type, aggregation, and optionally detected delimiter/header/signal parameters. |
| `dataset export <config_file>` | `--output`, `--format yaml|json` | Write normalized dataset config. |
| `dataset diff <config1> <config2>` | `--format text|json` | Compare two normalized dataset configs. |

## Workspace Commands

```bash
nirs4all workspace init workspace
nirs4all workspace list-runs --workspace workspace
nirs4all workspace query-best --workspace workspace --metric test_score -n 5
nirs4all workspace filter --workspace workspace --dataset wheat --test-score 0.50
nirs4all workspace stats --workspace workspace --metric test_score
nirs4all workspace list-library --workspace workspace
```

| Command | Options | Purpose |
| --- | --- | --- |
| `workspace init <path>` | none | Create a workspace directory. |
| `workspace list-runs` | `--workspace` | List recorded runs. |
| `workspace query-best` | `--workspace`, `--dataset`, `--metric`, `-n`, `--ascending` | Query top prediction rows by metric. |
| `workspace filter` | `--workspace`, `--dataset`, `--test-score`, `--train-score`, `--val-score` | Filter prediction rows. |
| `workspace stats` | `--workspace`, `--metric` | Show score statistics. |
| `workspace list-library` | `--workspace` | List saved library templates/items. |

## Artifact Commands

```bash
nirs4all artifacts list-orphaned --workspace workspace
nirs4all artifacts cleanup --workspace workspace
nirs4all artifacts cleanup --workspace workspace --force --verbose
nirs4all artifacts stats --workspace workspace
nirs4all artifacts purge --workspace workspace --dataset wheat --force --yes
```

| Command | Options | Purpose |
| --- | --- | --- |
| `artifacts list-orphaned` | `--workspace`, `--dataset` | List binary artifacts not referenced by manifests. |
| `artifacts cleanup` | `--workspace`, `--dataset`, `--force`, `--verbose` | Dry-run by default; with `--force`, delete orphaned artifacts. |
| `artifacts stats` | `--workspace`, `--dataset` | Show storage and deduplication statistics. |
| `artifacts purge` | `--workspace`, `--dataset`, `--force`, `--yes` | Delete all artifacts for one dataset. |

## Conformal and Robustness Audit Commands

The CLI can inspect stored conformal/robustness artifacts and republish
verified robustness reports. These commands are read-only unless an explicit
`--output` path is provided.

```bash
nirs4all workspace conformal list --workspace workspace --json
nirs4all workspace conformal show pls-moisture-conformal --workspace workspace --json
nirs4all workspace conformal show pls-moisture-conformal \
  --workspace workspace \
  --as-predict-result \
  --json
nirs4all workspace tuning export pls-moisture-hpo \
  --workspace workspace \
  --format summary \
  --output artifacts/tuning/pls-moisture/summary.json
nirs4all workspace conformal predict pls-moisture-conformal \
  --workspace workspace \
  --y-pred "13.0,14.0" \
  --sample-ids "sample-003,sample-004" \
  --json
```

`workspace conformal predict` applies an already stored calibrator to already
computed point predictions. It does not train, refit, recalibrate, or mutate the
workspace.

`workspace conformal show --as-predict-result --json` converts the stored
`CalibratedRunResult` through `calibrated.to_predict_result()` and emits the
public prediction payload: point predictions, materialized intervals,
`calibrated_result_fingerprint`, `calibration_replay_source`, and
`tuning_calibration_source`. Use this form for bindings, Studio and notebook
diagnostics that need the same accessors as `nirs4all.predict()`.

`workspace tuning export --format summary` emits the compact
`nirs4all.tuning.summary` contract used by CI, bindings, Studio/Web, and
dashboards. Use `--format json` when the full verified `TuningResult` evidence
tape is required.

```bash
nirs4all workspace robustness list --workspace workspace --json
nirs4all workspace robustness show pls-moisture-robustness --workspace workspace --json
nirs4all workspace robustness evidence --workspace workspace --dataset wheat --json
nirs4all workspace robustness from-prediction \
  --workspace workspace \
  --prediction-id pred-001 \
  --y-true "1.0,2.0,3.0" \
  --scenarios-json '[{"kind":"spectral_offset","severity":0.01}]' \
  --save-to-workspace \
  --workspace-robustness-id pred-001-spectral-audit \
  --format summary \
  --output artifacts/robustness/pred-001/summary.json
nirs4all workspace robustness export pls-moisture-robustness \
  --workspace workspace \
  --format summary \
  --output artifacts/robustness/pls-moisture/summary.json
```

`workspace robustness evidence` inspects stored prediction rows with
`load_arrays=True` and reports the native
`PredictResult.spectral_replay_evidence_status` diagnostic for each prediction.
It is read-only: it never synthesizes `X`, `spectra`, or predictor bundles.
Rows become `ready_for_spectral_replay` only when the prediction carries an
actual row-aligned executable `X`/`spectra` matrix and a
`predictor_bundle`/`model_path`; provenance markers such as
`prediction_arrays.X` remain diagnostic metadata unless the array is also
loaded from the workspace sidecar. When the stored prediction also carries
native conformal provenance, JSON output includes `calibration_replay_source`
and `tuning_calibration_source` from the same `PredictResult` conversion so
CI, bindings and Studio can audit replay/calibration boundaries without parsing
raw workspace rows.

`workspace robustness from-prediction` is the executable CLI bridge from one
stored prediction row to a `RobustnessReport`. It loads the row through
`nirs4all.robustness_from_workspace_prediction()`, so executable `X`/`spectra`
and saved `predictor_bundle`/`model_path` evidence are consumed as spectral/OOD
defaults when present. `--y-true` accepts comma-separated targets;
`--y-true-json` accepts the same values as JSON. `--scenarios-json`,
`--metadata-json`, `--slice-by`, and `--seed` map directly to
`nirs4all.robustness()`. Use `--save-to-workspace` with
`--workspace-robustness-id` to persist the generated report back to the same
workspace linked to the prediction id. Output formats match
`workspace robustness export`: `json`, `summary`, `markdown`, `html`,
`parquet`, and `artifacts`.

`workspace robustness export --format summary` emits the compact
`summary.json` contract used by CI, bindings, Studio/Web, and dashboards. Other
formats are `json`, `markdown`, `html`, `parquet`, and `artifacts`.

Verified report files or artifact directories can also be republished without a
workspace:

```bash
nirs4all robustness-report artifacts/robustness/pls-moisture \
  --format markdown \
  --output artifacts/robustness/pls-moisture/report.md

nirs4all robustness-report artifacts/robustness/pls-moisture/report.json \
  --format summary \
  --output artifacts/robustness/pls-moisture/summary.json

nirs4all robustness-summary-schema --output artifacts/robustness-summary.schema.json
nirs4all tuning-summary-schema --output artifacts/tuning-summary.schema.json
nirs4all keyword-registry --output artifacts/keyword-registry.json
nirs4all tuning-space --input tuning.json --output artifacts/tuning-space.json
nirs4all tuning-space --schema --output artifacts/tuning-space.schema.json
```

The schema command publishes the JSON Schema for
`RobustnessReport.summary_artifact()` and bundle `summary.json` payloads. Static
consumers should validate the schema version and fingerprint before displaying
release cards or Studio result cards.
`tuning-summary-schema` does the same for `TuningResult.summary_artifact()` and
HPO `summary.json` cards.
`keyword-registry` publishes machine-readable keyword effects for forms and
Studio, including HPO controls such as `run.tuning.force_params`, whose
`changes` field records `trial_sequence`, `candidate_fit`, and `selection`.
The exported registry also carries the strict JSON-native schemas for tuning,
conformal, prediction, robustness and workspace metadata, so external forms can
reject non-canonical metadata before invoking Python helpers.
`tuning-space` publishes the same JSON-native
`nirs4all.tuning.ordered_search_space` artifact as
`nirs4all.inspect_tuning_space(...)`: canonical dotted paths, ordered
parameters, decoded `ParameterPatch` force params, `fingerprint`, and
`tuning_fingerprint`. It reads a tuning JSON payload from `--input` or
`--tuning` and does not execute a pipeline. `tuning-space --schema` publishes
the matching JSON Schema for bindings, CLI validators and Studio forms.

## Config-First Workflow

```bash
# 1. Validate the portable files
nirs4all dataset validate dataset.yaml
nirs4all config validate pipeline.yaml --type pipeline --check-imports

# 2. Run through the Python API
python - <<'PY'
import nirs4all

result = nirs4all.run("pipeline.yaml", "dataset.yaml", name="cli_wrapped_run")
print(result.best_score)
result.export("exports/cli_wrapped_run.n4a")
PY

# 3. Inspect the workspace if the run archived predictions
nirs4all workspace query-best --workspace workspace -n 5
```

## Runtime Environment

```bash
N4A_ENGINE=dag-ml python train.py
N4A_NATIVE_RESULTS=./nirs4all_results python train.py
```

See {doc}`public_interfaces` for engine behavior and language wrapper patterns.
