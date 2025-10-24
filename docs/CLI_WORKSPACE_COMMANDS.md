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
  Created directories:
    - runs/
    - exports/full_pipelines/
    - exports/best_predictions/
    - library/templates/
    - library/trained/filtered/
    - library/trained/pipeline/
    - library/trained/fullrun/
    - catalog/
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

  2024-10-24_wheat_sample1_baseline
    Dataset: wheat
    Date: 2024-10-24
    Custom name: sample1_baseline

  2024-10-23_corn_sample1
    Dataset: corn
    Date: 2024-10-23
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
from nirs4all.workspace import WorkspaceManager
from nirs4all.dataset.predictions import Predictions

# Initialize workspace
workspace = WorkspaceManager("my_workspace")
workspace.initialize_workspace()

# Query catalog
pred = Predictions.load_from_parquet("my_workspace/catalog")
best = pred.query_best(metric="test_score", n=10)
```

See `examples/workspace_integration_example.py` for a complete example.

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

## Notes

- All commands default to `workspace/` if `--workspace` is not specified
- The catalog must be populated using `Predictions.archive_to_catalog()` before querying
- Use `--help` with any command for detailed options: `nirs4all workspace <command> --help`
