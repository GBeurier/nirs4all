# CLI Extension Proposal for nirs4all

**Date**: December 20, 2025
**Status**: Draft Proposal

---

## Overview

This document outlines potential CLI commands that could extend nirs4all's command-line interface to expose the full API functionality. The current CLI supports basic installation tests, workspace management, and artifact operations. This proposal maps API functions to CLI commands using JSON/YAML/text file inputs instead of Python objects.

---

## Current CLI Commands

| Command Group | Commands |
|--------------|----------|
| `--test-install` | Test installation |
| `--test-integration` | Run integration test |
| `workspace` | `init`, `list-runs`, `query-best`, `filter`, `stats`, `list-library` |
| `artifacts` | `list-orphaned`, `cleanup`, `stats`, `purge` |

---

## Proposed CLI Extensions

### 1. Pipeline Execution Commands (`nirs4all run`)

Core pipeline execution functionality from `PipelineRunner.run()`.

#### `nirs4all run`

Run a pipeline on a dataset.

```bash
nirs4all run --pipeline <pipeline.yaml|pipeline.json> \
             --dataset <dataset.yaml|path/to/data> \
             [--name <pipeline_name>] \
             [--workspace <path>] \
             [--verbose <0-3>] \
             [--save-artifacts] \
             [--save-charts] \
             [--random-state <seed>] \
             [--output <predictions.json>]
```

**Input Files:**

*pipeline.yaml example:*
```yaml
steps:
  - class: sklearn.preprocessing.MinMaxScaler
  - y_processing:
      class: sklearn.preprocessing.MinMaxScaler
  - class: sklearn.model_selection.ShuffleSplit
    params:
      n_splits: 3
      test_size: 0.25
  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
```

*dataset.yaml example:*
```yaml
train_x: path/to/train_spectra.csv
train_y: path/to/train_targets.csv
test_x: path/to/test_spectra.csv
test_y: path/to/test_targets.csv
task_type: regression
signal_type: absorbance
aggregate: sample_id
train_x_params:
  header_unit: nm
  delimiter: ","
```

**API Mapping:** `PipelineRunner.run()`

---

### 2. Prediction Commands (`nirs4all predict`)

Apply trained models to new data.

#### `nirs4all predict`

```bash
nirs4all predict --model <model_source> \
                 --data <new_data.csv|dataset.yaml> \
                 [--output <predictions.csv>] \
                 [--all-predictions] \
                 [--verbose <0-3>]
```

**Model source options:**
- Path to pipeline directory
- Path to `.n4a` bundle
- Prediction ID from catalog

**API Mapping:** `PipelineRunner.predict()`

---

### 3. Export Commands (`nirs4all export`)

Export trained pipelines for deployment.

#### `nirs4all export bundle`

Export a trained pipeline to a standalone bundle.

```bash
nirs4all export bundle --source <prediction_id|pipeline_path> \
                       --output <output.n4a> \
                       [--format <n4a|n4a.py>] \
                       [--compress] \
                       [--include-metadata]
```

**API Mapping:** `PipelineRunner.export()`

#### `nirs4all export model`

Export just the model artifact.

```bash
nirs4all export model --source <prediction_id|pipeline_path> \
                      --output <model.joblib|model.pkl|model.h5> \
                      [--fold <fold_index>]
```

**API Mapping:** `PipelineRunner.export_model()`

---

### 4. Retrain Commands (`nirs4all retrain`)

Retrain existing pipelines on new data.

#### `nirs4all retrain`

```bash
nirs4all retrain --source <prediction_id|pipeline_path|bundle.n4a> \
                 --dataset <new_data.yaml> \
                 --mode <full|transfer|finetune> \
                 [--new-model <model.yaml>] \
                 [--epochs <n>] \
                 [--output <predictions.json>] \
                 [--verbose <0-3>]
```

**API Mapping:** `PipelineRunner.retrain()`

---

### 5. Explanation Commands (`nirs4all explain`)

Generate SHAP explanations.

#### `nirs4all explain`

```bash
nirs4all explain --model <prediction_id|pipeline_path> \
                 --data <dataset.yaml> \
                 [--output-dir <shap_results/>] \
                 [--shap-params <shap_config.yaml>] \
                 [--plots] \
                 [--verbose <0-3>]
```

*shap_config.yaml example:*
```yaml
n_samples: 100
feature_names: auto
plot_type: bar
```

**API Mapping:** `PipelineRunner.explain()`

---

### 6. Extract Commands (`nirs4all extract`)

Extract and inspect trained pipelines.

#### `nirs4all extract`

```bash
nirs4all extract --source <prediction_id|pipeline_path> \
                 --output <pipeline_info.json> \
                 [--format <json|yaml|text>]
```

Output includes:
- Pipeline steps
- Preprocessing chain summary
- Model step index
- Execution trace

**API Mapping:** `PipelineRunner.extract()`

---

### 7. Data Commands (`nirs4all data`)

Dataset inspection and manipulation utilities.

#### `nirs4all data info`

Inspect a dataset configuration.

```bash
nirs4all data info <dataset.yaml|path/to/folder>
```

Output:
- Number of samples (train/test)
- Number of features
- Task type (detected or configured)
- Signal type
- Metadata columns

**API Mapping:** `DatasetConfigs`, `SpectroDataset` inspection

#### `nirs4all data validate`

Validate dataset files and configuration.

```bash
nirs4all data validate <dataset.yaml|path/to/folder>
```

#### `nirs4all data convert`

Convert between data formats.

```bash
nirs4all data convert --input <data.csv> \
                      --output <data.parquet> \
                      [--header-unit <nm|cm-1>] \
                      [--signal-type <absorbance|reflectance>]
```

---

### 8. Preprocessing Commands (`nirs4all preprocess`)

Apply preprocessing transformations standalone.

#### `nirs4all preprocess`

Apply a preprocessing pipeline to data.

```bash
nirs4all preprocess --input <spectra.csv> \
                    --output <processed.csv> \
                    --pipeline <preprocessing.yaml> \
                    [--signal-type <absorbance>]
```

*preprocessing.yaml example:*
```yaml
steps:
  - class: nirs4all.operators.StandardNormalVariate
  - class: nirs4all.operators.SavitzkyGolay
    params:
      window_length: 15
      polyorder: 2
      deriv: 1
```

**API Mapping:** Transform operators from `nirs4all.operators`

---

### 9. Analysis Commands (`nirs4all analyze`)

Transfer learning and preprocessing selection.

#### `nirs4all analyze transfer`

Run transfer preprocessing analysis.

```bash
nirs4all analyze transfer --source <source_data.csv> \
                          --target <target_data.csv> \
                          [--preset <balanced|fast|thorough>] \
                          [--output <results.json>] \
                          [--plot]
```

**API Mapping:** `TransferPreprocessingSelector`

---

### 10. Visualization Commands (`nirs4all viz`)

Generate charts and reports from predictions.

#### `nirs4all viz top-k`

Plot top-k model comparison.

```bash
nirs4all viz top-k --predictions <predictions.parquet> \
                   [--k <5>] \
                   [--metric <test_score>] \
                   [--aggregate <sample_id>] \
                   [--output <chart.png>]
```

**API Mapping:** `PredictionAnalyzer.plot_top_k()`

#### `nirs4all viz heatmap`

Generate parameter heatmap.

```bash
nirs4all viz heatmap --predictions <predictions.parquet> \
                     --x-axis <model_name> \
                     --y-axis <preprocessings> \
                     [--output <heatmap.png>]
```

**API Mapping:** `PredictionAnalyzer.plot_heatmap()`

#### `nirs4all viz confusion-matrix`

Generate confusion matrix (classification).

```bash
nirs4all viz confusion-matrix --predictions <predictions.parquet> \
                              [--prediction-id <id>] \
                              [--output <confusion.png>]
```

**API Mapping:** `PredictionAnalyzer.plot_confusion_matrix()`

#### `nirs4all viz score-histogram`

Plot score distribution histogram.

```bash
nirs4all viz score-histogram --predictions <predictions.parquet> \
                             [--metric <test_score>] \
                             [--output <histogram.png>]
```

**API Mapping:** `PredictionAnalyzer.plot_score_histogram()`

#### `nirs4all viz branch-diagram`

Visualize pipeline branching structure.

```bash
nirs4all viz branch-diagram --predictions <predictions.parquet> \
                            [--output <diagram.png>]
```

**API Mapping:** `BranchDiagram`, `plot_branch_diagram()`

---

### 11. Predictions Commands (`nirs4all predictions`)

Query and manipulate prediction catalogs.

#### `nirs4all predictions top`

Query top predictions.

```bash
nirs4all predictions top --input <predictions.parquet|catalog/> \
                         [--n <10>] \
                         [--metric <test_score>] \
                         [--partition <test>] \
                         [--ascending] \
                         [--output <top_predictions.json>]
```

**API Mapping:** `Predictions.top()`

#### `nirs4all predictions filter`

Filter predictions by criteria.

```bash
nirs4all predictions filter --input <predictions.parquet> \
                            [--dataset <name>] \
                            [--model <name>] \
                            [--min-score <0.5>] \
                            [--output <filtered.parquet>]
```

**API Mapping:** `Predictions.filter_by_criteria()`

#### `nirs4all predictions merge`

Merge multiple prediction files.

```bash
nirs4all predictions merge --inputs <pred1.parquet> <pred2.parquet> ... \
                           --output <merged.parquet>
```

**API Mapping:** `Predictions` concatenation

#### `nirs4all predictions archive`

Archive predictions to catalog.

```bash
nirs4all predictions archive --input <predictions.parquet> \
                             --catalog <workspace/catalog/>
```

**API Mapping:** `Predictions.archive_to_catalog()`

---

### 12. Library Commands (`nirs4all library`)

Manage pipeline templates and saved models.

#### `nirs4all library save-template`

Save a pipeline template.

```bash
nirs4all library save-template --pipeline <pipeline.yaml> \
                               --name <template_name> \
                               [--description <text>] \
                               [--workspace <path>]
```

**API Mapping:** `PipelineLibrary.save_template()`

#### `nirs4all library list-templates`

List available templates.

```bash
nirs4all library list-templates [--workspace <path>]
```

**API Mapping:** `LibraryManager.list_templates()`

#### `nirs4all library load-template`

Load and display a template.

```bash
nirs4all library load-template --name <template_name> \
                               [--output <pipeline.yaml>] \
                               [--workspace <path>]
```

**API Mapping:** `PipelineLibrary.load_template()`

---

### 13. Config Commands (`nirs4all config`)

Validate and convert configuration files.

#### `nirs4all config validate`

Validate a pipeline or dataset configuration.

```bash
nirs4all config validate <config.yaml|config.json>
```

#### `nirs4all config convert`

Convert between JSON and YAML formats.

```bash
nirs4all config convert --input <config.json> \
                        --output <config.yaml>
```

#### `nirs4all config expand`

Expand generator syntax to see all combinations.

```bash
nirs4all config expand --input <pipeline_generator.yaml> \
                       [--max <100>] \
                       [--output <expanded_pipelines.yaml>]
```

**API Mapping:** `PipelineConfigs` generator expansion

---

## Implementation Considerations

### Input Format Detection

All commands should auto-detect input format based on file extension:
- `.yaml`, `.yml` → YAML parser
- `.json` → JSON parser
- `.csv` → CSV data file
- `.parquet` → Parquet data file
- Directories → Scan for dataset config

### Output Formats

Support multiple output formats via `--format` flag:
- `json` (default for structured data)
- `yaml`
- `csv` (for tabular data)
- `table` (human-readable ASCII table)

### Error Handling

- Validate all input files before processing
- Provide clear error messages with line numbers for YAML/JSON parse errors
- Support `--dry-run` for destructive operations

### Common Flags

All commands should support:
- `--help` - Detailed help
- `--verbose` / `-v` - Increase verbosity (can be stacked: `-vvv`)
- `--quiet` / `-q` - Suppress non-error output
- `--workspace` / `-w` - Override default workspace path
- `--config` - Global config file for defaults

---

## Priority Ranking

### High Priority (Core Functionality)

1. `nirs4all run` - Execute pipelines
2. `nirs4all predict` - Apply trained models
3. `nirs4all export bundle` - Export for deployment
4. `nirs4all data info` - Dataset inspection

### Medium Priority (Workflow Enhancement)

5. `nirs4all viz top-k` - Quick results visualization
6. `nirs4all predictions top` - Query best models
7. `nirs4all config validate` - Configuration validation
8. `nirs4all preprocess` - Standalone preprocessing
9. `nirs4all retrain` - Model retraining

### Lower Priority (Advanced Features)

10. `nirs4all explain` - SHAP explanations
11. `nirs4all analyze transfer` - Transfer analysis
12. `nirs4all library *` - Template management
13. Visualization commands - Charts generation

---

## Example Workflows

### Workflow 1: Train and Deploy

```bash
# 1. Validate configuration
nirs4all config validate pipeline.yaml
nirs4all data validate dataset.yaml

# 2. Train model
nirs4all run --pipeline pipeline.yaml --dataset dataset.yaml --verbose 1

# 3. Query best model
nirs4all predictions top --input workspace/catalog/ --n 1

# 4. Export for deployment
nirs4all export bundle --source <best_prediction_id> --output model.n4a

# 5. Deploy predictions
nirs4all predict --model model.n4a --data new_samples.csv --output predictions.csv
```

### Workflow 2: Hyperparameter Search

```bash
# 1. Expand generator to preview combinations
nirs4all config expand --input pipeline_search.yaml --max 50

# 2. Run full search
nirs4all run --pipeline pipeline_search.yaml --dataset dataset.yaml

# 3. Visualize results
nirs4all viz heatmap --predictions workspace/catalog/ --x-axis n_components --y-axis preprocessings

# 4. Get top configurations
nirs4all predictions top --input workspace/catalog/ --n 10 --output best_configs.json
```

### Workflow 3: Transfer Learning Analysis

```bash
# 1. Analyze domain shift
nirs4all analyze transfer --source lab_data.csv --target field_data.csv --plot

# 2. Apply recommended preprocessing
nirs4all preprocess --input field_data.csv --output field_preprocessed.csv --pipeline recommended.yaml

# 3. Retrain with transfer
nirs4all retrain --source lab_model.n4a --dataset field_preprocessed.csv --mode transfer
```

---

## Next Steps

1. Implement core `run` and `predict` commands
2. Create JSON/YAML schema definitions for validation
3. Add comprehensive help documentation per command
4. Implement configuration file parsing utilities
5. Add progress indicators for long-running operations
6. Create integration tests for CLI workflows
