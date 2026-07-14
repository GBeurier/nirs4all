# Workspace CLI Commands

The `nirs4all` CLI provides workspace management commands for organizing experiments, querying results, and managing saved models.

## Installation

The CLI is available after installing nirs4all:

```bash
pip install nirs4all
```

## Usage

```bash
nirs4all workspace <command> [options]
```

---

## Commands

### `init` - Initialize Workspace

Create a new workspace with the standard directory structure.

**Usage:**
```bash
nirs4all workspace init <path>
```

**Example:**
```bash
nirs4all workspace init my_workspace
```

**Output:**
```
✓ Workspace initialized at: my_workspace
  Created:
    - store.sqlite (workspace database)
    - arrays/
    - artifacts/
    - exports/
    - library/
```

---

### `list-runs` - List Runs

List all experimental runs in a workspace.

**Usage:**
```bash
nirs4all workspace list-runs [--workspace <path>]
```

**Options:**
- `--workspace`: Workspace root directory (default: `workspace`)

**Example:**
```bash
nirs4all workspace list-runs --workspace my_workspace
```

**Output:**
```
Found 3 run(s):

  wheat_sample1_baseline
    Dataset: wheat

  corn_sample1
    Dataset: corn
```

---

### `query-best` - Query Best Pipelines

Query the catalog for top-performing pipelines by a specific metric.

**Usage:**
```bash
nirs4all workspace query-best [options]
```

**Options:**
- `--workspace <path>`: Workspace root (default: `workspace`)
- `--dataset <name>`: Filter by dataset name
- `--metric <name>`: Metric to sort by (default: `test_score`)
- `-n <number>`: Number of results (default: 10)
- `--ascending`: Sort ascending (lower is better)

**Examples:**

```bash
# Get top 10 by test_score
nirs4all workspace query-best --workspace my_workspace

# Get top 5 wheat models by validation score
nirs4all workspace query-best --workspace my_workspace --dataset wheat --metric val_score -n 5

# Get worst 3 models (ascending)
nirs4all workspace query-best --workspace my_workspace -n 3 --ascending
```

**Output:**
```
✓ Loaded 142 predictions from catalog

Top 10 pipelines by test_score:
================================================================================

prediction_id                          dataset_name  config_name      test_score
a1b2c3d4-5678-90ab-cdef-1234567890ab  wheat_sample1  advanced_pls     0.5234
e5f6g7h8-9012-34ab-cdef-5678901234cd  wheat_sample1  optimized_rf     0.5198
...
```

---

### `filter` - Filter Predictions

Filter predictions by multiple criteria (dataset, score thresholds).

**Usage:**
```bash
nirs4all workspace filter [options]
```

**Options:**
- `--workspace <path>`: Workspace root (default: `workspace`)
- `--dataset <name>`: Filter by dataset name
- `--test-score <value>`: Minimum test score
- `--train-score <value>`: Minimum train score
- `--val-score <value>`: Minimum validation score

**Examples:**

```bash
# Find all predictions with test_score >= 0.50
nirs4all workspace filter --workspace my_workspace --test-score 0.50

# Find wheat predictions with good train and test scores
nirs4all workspace filter --workspace my_workspace --dataset wheat --test-score 0.45 --train-score 0.40

# Find predictions meeting all criteria
nirs4all workspace filter --workspace my_workspace --test-score 0.50 --val-score 0.48 --train-score 0.45
```

**Output:**
```
Found 23 predictions matching criteria

prediction_id                          dataset_name  test_score  train_score  val_score
a1b2c3d4-5678-90ab-cdef-1234567890ab  wheat_sample1  0.5234     0.4876       0.5012
...
```

---

### `stats` - Catalog Statistics

Show summary statistics for the catalog.

**Usage:**
```bash
nirs4all workspace stats [options]
```

**Options:**
- `--workspace <path>`: Workspace root (default: `workspace`)
- `--metric <name>`: Metric for statistics (default: `test_score`)

**Example:**
```bash
nirs4all workspace stats --workspace my_workspace --metric test_score
```

**Output:**
```
Catalog Statistics
============================================================

Total predictions: 142
Datasets: 3
  - wheat_sample1: 58 predictions
  - corn_sample1: 45 predictions
  - barley_sample1: 39 predictions

test_score statistics:
  Min:    0.3245
  Max:    0.5234
  Mean:   0.4512
  Median: 0.4498
  Std:    0.0456
```

---

### `list-library` - List Library Items

List templates and saved models in the library.

**Usage:**
```bash
nirs4all workspace list-library [--workspace <path>]
```

**Options:**
- `--workspace`: Workspace root directory (default: `workspace`)

**Example:**
```bash
nirs4all workspace list-library --workspace my_workspace
```

**Output:**
```
Templates: 2
  - baseline_pls: Baseline PLS configuration
  - advanced_rf: Random Forest with feature selection

Filtered pipelines: 5
  - wheat_experiment_001: First wheat experiment
  - corn_baseline_v1: Baseline model for corn

Full pipelines: 3
  - production_wheat_v1: Production-ready wheat model
  - deployment_corn_v2: Corn model for deployment

Full runs: 1
  - wheat_baseline_complete: Complete baseline experiment
```

---

## Programmatic Usage

All CLI commands can also be used programmatically:

```python
from nirs4all.pipeline.storage import WorkspaceStore

# Initialize workspace
store = WorkspaceStore("my_workspace")

# Query best predictions
top = store.top_predictions(n=10, metric="test_score")
```

See `examples/user/06_deployment/U03_workspace_management.py` for a complete example.

---

## Workflow Example

```bash
# 1. Initialize workspace
nirs4all workspace init my_project

# 2. Run experiments (using Python API)
# ... your training code ...

# 3. Query results
nirs4all workspace query-best --workspace my_project -n 5

# 4. Filter good models
nirs4all workspace filter --workspace my_project --test-score 0.50

# 5. View statistics
nirs4all workspace stats --workspace my_project

# 6. Check saved models
nirs4all workspace list-library --workspace my_project
```

---

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

## Notes

- All commands default to `workspace/` if `--workspace` is not specified
- The catalog must be populated using `Predictions.archive_to_catalog()` before querying
- Use `--help` with any command for detailed options: `nirs4all workspace <command> --help`
