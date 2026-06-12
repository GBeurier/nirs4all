# Example Configuration Files for nirs4all

This directory contains reference configuration files for use with the nirs4all CLI and Python API.

## Directory Structure

```
configs/
├── pipelines/          # Pipeline configuration examples
│   ├── simple_pls.yaml        # Basic PLS regression pipeline
│   ├── advanced_preprocessing.yaml  # Pipeline with multiple preprocessing steps
│   ├── multi_model.json       # Pipeline testing multiple models
│   └── generator_search.yaml  # Hyperparameter search with generator syntax
│
└── datasets/           # Dataset configuration examples
    ├── regression_basic.yaml   # Basic regression dataset config
    ├── classification.json     # Classification dataset config
    ├── multi_source.yaml       # Multi-source dataset config
    ├── heterogeneous_repetitions_per_source_aggregate.yaml
    ├── heterogeneous_repetitions_late_fusion.yaml
    ├── heterogeneous_repetitions_cartesian_full.yaml
    └── heterogeneous_repetitions_missing_source.yaml
```

## Usage

### With Python API

```python
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

# Load from config files
pipeline = PipelineConfigs("examples/configs/pipelines/simple_pls.yaml", name="my_pipeline")
dataset = DatasetConfigs("examples/configs/datasets/regression_basic.yaml")

# Run
runner = PipelineRunner(verbose=1)
predictions, _ = runner.run(pipeline, dataset)
```

### With CLI (coming soon)

```bash
nirs4all run --pipeline configs/pipelines/simple_pls.yaml \
             --dataset configs/datasets/regression_basic.yaml
```

## Configuration Validation

You can validate your configuration files before running:

```python
from nirs4all.config import validate_config_file

is_valid, errors, warnings = validate_config_file("my_config.yaml")
if not is_valid:
    print("Errors:", errors)
```

Or via CLI:

```bash
nirs4all config validate my_config.yaml
```

## Key Reference

### Pipeline Config Keys

| Key | Type | Description |
|-----|------|-------------|
| `pipeline` | list | List of pipeline steps (required) |
| `name` | string | Optional pipeline name |
| `description` | string | Optional description |

### Dataset Config Keys

| Key | Type | Description |
|-----|------|-------------|
| `train_x` | string/list | Path(s) to training features |
| `train_y` | string | Path to training targets |
| `test_x` | string/list | Path(s) to test features |
| `test_y` | string | Path to test targets |
| `task_type` | string | "regression", "binary_classification", "multiclass_classification", "auto" |
| `signal_type` | string | "absorbance", "reflectance", "transmittance", etc. |
| `aggregate` | string/bool | Column name or True for y-based aggregation |
| `global_params` | object | Loading parameters (delimiter, header_unit, etc.) |
| `experimental_relation_pipeline` | bool | Opt in to source-aware heterogeneous repetition contracts |
| `repetition_spec` | object | Physical sample key, source repetition columns, cardinalities and missing policies |
| `representations` | list | `rep_fusion` materialization plans such as `per_source_aggregate` or `cartesian_full` |
| `reducers` | list | `ReductionPlan` declarations for score, refit, meta features and final output |
| `fit_influence` | object | Fit influence policy for derived rows, separate from prediction reducers |
| `meta_features` | object | Late-fusion or stacking alignment contract |
| `refit_slots` | list | Scope-aware refit selection slots |
