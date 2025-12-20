# Configuration Handling Analysis for CLI Readiness

**Date**: December 20, 2025
**Status**: Analysis Complete
**Purpose**: Assess nirs4all's JSON/YAML/binary config handling for CLI implementation

---

## Executive Summary

The nirs4all codebase has **robust foundation** for JSON/YAML configuration handling, with most components already supporting file-based configs. However, there are **critical gaps** that must be addressed before implementing the full CLI roadmap:

### ✅ What Works Well
1. **PipelineConfigs** fully supports JSON/YAML file loading
2. **Component serialization** handles sklearn/TF/PyTorch/JAX objects
3. **Bundle format (.n4a)** is production-ready for model deployment
4. Comprehensive test coverage for serialization round-trips

### ⚠️ Critical Gaps
1. **DatasetConfigs** lacks direct JSON/YAML file loading
2. No validation schemas for config files
3. CLI only has basic workspace commands (no run/predict/export)
4. Missing CLI-friendly error messages with line numbers
5. No dataset YAML/JSON examples in documentation

### Estimated Work
- **Phase 1 (Core)**: 2-3 weeks - DatasetConfigs file loading
- **Phase 2 (CLI)**: 4-5 weeks - Run/predict/export commands
- **Phase 3 (Polish)**: 2-3 weeks - Validation, docs, examples

---

## Current State Analysis

### 1. PipelineConfigs - JSON/YAML Handling ✅

**File**: [nirs4all/pipeline/config/pipeline_config.py](../nirs4all/pipeline/config/pipeline_config.py)

**Capabilities**:
```python
# ✅ All these work today:
config = PipelineConfigs("pipeline.json", name="my_pipeline")
config = PipelineConfigs("pipeline.yaml", name="my_pipeline")
config = PipelineConfigs(json_string, name="my_pipeline")
config = PipelineConfigs(yaml_string, name="my_pipeline")
config = PipelineConfigs([list, of, steps], name="my_pipeline")
config = PipelineConfigs({"pipeline": [...]}, name="my_pipeline")
```

**Implementation**: `_load_str_steps()` method handles:
- JSON file paths (`.json`)
- YAML file paths (`.yaml`, `.yml`)
- JSON strings
- YAML strings
- Automatic format detection

**JSON Format**:
```json
{
  "pipeline": [
    {
      "class": "sklearn.preprocessing.MinMaxScaler",
      "params": {
        "feature_range": [0, 1]
      }
    },
    {
      "class": "sklearn.model_selection.ShuffleSplit",
      "params": {
        "n_splits": 5,
        "test_size": 0.25
      }
    },
    {
      "model": {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "params": {
          "n_components": 10
        }
      }
    }
  ]
}
```

**YAML Format**:
```yaml
pipeline:
  - class: sklearn.preprocessing.MinMaxScaler
    params:
      feature_range: [0, 1]

  - class: sklearn.model_selection.ShuffleSplit
    params:
      n_splits: 5
      test_size: 0.25

  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
```

**Test Coverage**: ✅ Comprehensive
- `tests/unit/pipeline/test_serialization.py`: Round-trip tests
- `tests/unit/pipeline/config/test_pipeline_config.py`: Config tests
- `tests/unit/pipeline/test_runner_normalization.py`: Integration tests

**Issues**:
- ❌ No JSON schema validation
- ❌ Generic error messages (no line numbers for parse errors)
- ⚠️ Module path normalization issues (public vs internal paths)

---

### 2. DatasetConfigs - JSON/YAML Handling ⚠️

**File**: [nirs4all/data/config.py](../nirs4all/data/config.py)

**Current Capabilities**:
```python
# ✅ Works today:
config = DatasetConfigs("path/to/folder")  # Auto-browse for files
config = DatasetConfigs({                  # Dict config (in-memory)
    "train_x": "path/to/X.csv",
    "train_y": "path/to/Y.csv",
    "test_x": "path/to/Xtest.csv"
})
config = DatasetConfigs([config1, config2])  # Multi-dataset

# ❌ Does NOT work:
config = DatasetConfigs("dataset.json")    # File path not supported
config = DatasetConfigs("dataset.yaml")    # File path not supported
```

**Implementation**: `parse_config()` in [nirs4all/data/config_parser.py](../nirs4all/data/config_parser.py) only handles:
- String paths → folder browsing
- Dict configs → direct use
- No file loading logic

**Expected Dataset Config Format** (not implemented):
```yaml
# dataset.yaml
train_x: path/to/Xcal.csv
train_y: path/to/Ycal.csv
test_x: path/to/Xval.csv
test_y: path/to/Yval.csv

task_type: regression
signal_type: absorbance
aggregate: sample_id

train_x_params:
  header_unit: nm
  delimiter: ","
  decimal_separator: "."
  has_header: true

test_x_params:
  header_unit: nm
  delimiter: ","
```

**Gap Analysis**:
1. ❌ No JSON/YAML file loading in `DatasetConfigs.__init__()`
2. ❌ No `_load_from_file()` method
3. ❌ No format auto-detection (`.json` vs `.yaml`)
4. ⚠️ Key normalization exists but undocumented
5. ⚠️ Multi-dataset configs only work with lists of dicts/paths

**Test Coverage**: ⚠️ Partial
- Folder browsing tested
- Dict configs tested
- **File loading NOT tested** (doesn't exist)

---

### 3. Component Serialization - Class Path Handling ⚠️

**File**: [nirs4all/pipeline/config/component_serialization.py](../nirs4all/pipeline/config/component_serialization.py)

**Purpose**: Convert Python objects ↔ JSON-serializable dicts

**Capabilities**:
- ✅ Serialize sklearn/TF/PyTorch/JAX objects
- ✅ Deserialize from class paths
- ✅ Handle meta-estimators (stacking/voting)
- ✅ Preserve only non-default parameters
- ✅ Convert tuples to lists for JSON/YAML

**Critical Functions**:
```python
serialize_component(obj)    # Object → JSON-serializable dict
deserialize_component(blob) # Dict → Python object
```

**Example**:
```python
# Serialization
scaler = MinMaxScaler(feature_range=(0, 2))
serialized = serialize_component(scaler)
# → {"class": "sklearn.preprocessing._data.MinMaxScaler",
#    "params": {"feature_range": [0, 2]}}

# Deserialization
obj = deserialize_component(serialized)
# → MinMaxScaler(feature_range=(0, 2))
```

**Issues**:
1. ⚠️ **Module path inconsistency**:
   - User input: `"sklearn.preprocessing.MinMaxScaler"` (public API)
   - Serialized: `"sklearn.preprocessing._data.MinMaxScaler"` (internal)
   - This can cause confusion in CLI YAML files

2. ❌ **Deserialization error handling**:
   - Fails silently on import errors
   - Returns original blob instead of raising
   - No helpful error messages for typos

3. ⚠️ **Type inference limitations**:
   - `_resolve_type()` tries to infer parameter types
   - Not always reliable for complex nested structures

**Recommendations**:
- Keep internal module paths for hash consistency
- Add public API alias mapping for user-facing configs
- Improve error messages with suggestions for typos

---

### 4. Bundle Format (.n4a) - Binary Export/Import ✅

**Files**:
- [nirs4all/pipeline/bundle/generator.py](../nirs4all/pipeline/bundle/generator.py)
- [nirs4all/pipeline/bundle/loader.py](../nirs4all/pipeline/bundle/loader.py)

**Bundle Structure**:
```
model.n4a (ZIP archive)
├── manifest.json          # Metadata (version, source, dates)
├── pipeline.json          # Minimal pipeline config
├── trace.json             # Execution trace
├── artifacts/
│   ├── step_1_MinMaxScaler.joblib
│   ├── step_4_fold0_PLSRegression.joblib
│   ├── step_4_fold1_PLSRegression.joblib
│   └── step_4_fold2_PLSRegression.joblib
└── fold_weights.json      # CV ensemble weights
```

**Capabilities**:
- ✅ Export trained pipelines
- ✅ Load and predict from bundles
- ✅ Support CV ensembles
- ✅ Support branching pipelines
- ✅ Support meta-models (stacking)
- ✅ Portable Python script export (`.n4a.py`)

**CLI Integration**: ✅ Ready
```python
# These work today via BundleGenerator/BundleLoader
generator = BundleGenerator(workspace_path)
generator.export(prediction, "model.n4a")

loader = BundleLoader("model.n4a")
y_pred = loader.predict(X_new)
```

**Test Coverage**: ✅ Comprehensive
- `tests/unit/pipeline/bundle/test_bundle.py`

**Issues**:
- None major - this component is production-ready

---

### 5. Current CLI Implementation

**Files**:
- [nirs4all/cli/main.py](../nirs4all/cli/main.py)
- [nirs4all/cli/commands/workspace.py](../nirs4all/cli/commands/workspace.py)
- [nirs4all/cli/commands/artifacts.py](../nirs4all/cli/commands/artifacts.py) (not shown)

**Current Commands**:
```bash
nirs4all --test-install        # Test installation
nirs4all --test-integration    # Run integration test
nirs4all --version             # Show version

# Workspace commands
nirs4all workspace init <path>
nirs4all workspace list-runs [--workspace <path>]
nirs4all workspace query-best [--dataset <name>] [--metric <metric>] [-n <count>]
nirs4all workspace filter [--dataset <name>] [--test-score <min>]
nirs4all workspace stats [--metric <metric>]
nirs4all workspace list-library

# Artifacts commands (not shown)
nirs4all artifacts list-orphaned
nirs4all artifacts cleanup
nirs4all artifacts stats
nirs4all artifacts purge
```

**Missing Commands** (per CLI_EXTENSION_PROPOSAL.md):
```bash
# High priority
nirs4all run --pipeline <file> --dataset <file>      # ❌ Not implemented
nirs4all predict --model <source> --data <file>      # ❌ Not implemented
nirs4all export bundle --source <id> --output <file> # ❌ Not implemented

# Medium priority
nirs4all viz top-k --predictions <file>               # ❌ Not implemented
nirs4all config validate <file>                       # ❌ Not implemented
nirs4all data info <file>                             # ❌ Not implemented

# All other commands also missing
```

**Gap Analysis**:
1. ❌ No pipeline execution command (`run`)
2. ❌ No prediction command (`predict`)
3. ❌ No export commands (`export bundle`, `export model`)
4. ❌ No config validation command
5. ❌ No data inspection commands
6. ❌ No visualization commands

---

## Work Required for CLI Compliance

### Phase 1: Core Configuration Loading (2-3 weeks)

#### 1.1 DatasetConfigs File Loading

**Implementation**: Add to [nirs4all/data/config_parser.py](../nirs4all/data/config_parser.py)

```python
def parse_config(data_config):
    # Existing: handles string → folder, dict → direct use

    # ADD THIS:
    if isinstance(data_config, str):
        # Check if it's a JSON/YAML file
        if data_config.endswith(('.json', '.yaml', '.yml')):
            return _load_config_from_file(data_config)
        else:
            # Existing folder browsing
            return browse_folder(data_config), folder_to_name(data_config)

    # ... rest of existing logic

def _load_config_from_file(file_path: str):
    """Load dataset config from JSON/YAML file."""
    import json
    import yaml
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix == '.json':
            config = json.load(f)
        else:  # .yaml or .yml
            config = yaml.safe_load(f)

    # Normalize keys
    config = normalize_config_keys(config)

    # Extract dataset name
    dataset_name = config.get('name', path.stem)

    return config, dataset_name
```

**Tests to Add**: `tests/unit/data/test_config_loading.py`
```python
def test_dataset_config_from_json_file(tmp_path):
    """Test loading DatasetConfigs from JSON file."""
    config_data = {
        "train_x": "path/to/X.csv",
        "train_y": "path/to/Y.csv",
        "task_type": "regression"
    }

    json_file = tmp_path / "dataset.json"
    with open(json_file, 'w') as f:
        json.dump(config_data, f)

    configs = DatasetConfigs(str(json_file))
    dataset = configs.get_dataset_at(0)

    assert dataset.name == "dataset"
    # More assertions...

def test_dataset_config_from_yaml_file(tmp_path):
    """Test loading DatasetConfigs from YAML file."""
    # Similar to JSON test
```

**Documentation to Add**: `docs/user_guide/dataset_configuration.md`

---

#### 1.2 Configuration Validation

**Implementation**: New module [nirs4all/config/validator.py](../nirs4all/config/validator.py)

```python
"""Configuration validation with JSON Schema."""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Pipeline schema
PIPELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "pipeline": {
            "type": "array",
            "items": {
                "oneOf": [
                    {"type": "object", "properties": {"class": {"type": "string"}}},
                    {"type": "object", "properties": {"model": {"type": "object"}}},
                    {"type": "object", "properties": {"preprocessing": {"type": "object"}}},
                ]
            }
        }
    },
    "required": ["pipeline"]
}

# Dataset schema
DATASET_SCHEMA = {
    "type": "object",
    "properties": {
        "train_x": {"type": "string"},
        "train_y": {"type": "string"},
        "test_x": {"type": "string"},
        "test_y": {"type": "string"},
        "task_type": {"enum": ["regression", "binary_classification", "multiclass_classification", "auto"]},
        "signal_type": {"type": "string"},
        "aggregate": {"oneOf": [{"type": "string"}, {"type": "boolean"}]},
        "train_x_params": {"type": "object"},
    },
    "anyOf": [
        {"required": ["train_x"]},
        {"required": ["test_x"]}
    ]
}

def validate_pipeline_config(config_path: str) -> Tuple[bool, List[str]]:
    """Validate pipeline configuration file.

    Returns:
        (is_valid, error_messages)
    """
    try:
        import jsonschema

        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                config = json.load(f)
            else:
                import yaml
                config = yaml.safe_load(f)

        jsonschema.validate(config, PIPELINE_SCHEMA)
        return True, []

    except jsonschema.ValidationError as e:
        return False, [f"Validation error: {e.message}"]
    except Exception as e:
        return False, [f"Error loading config: {e}"]

def validate_dataset_config(config_path: str) -> Tuple[bool, List[str]]:
    """Validate dataset configuration file."""
    # Similar to pipeline validation
```

**CLI Command**: Add to [nirs4all/cli/commands/config.py](../nirs4all/cli/commands/config.py)
```python
def config_validate(args):
    """Validate a configuration file."""
    from nirs4all.config.validator import (
        validate_pipeline_config,
        validate_dataset_config
    )

    config_path = args.config_file
    config_type = args.type  # 'pipeline', 'dataset', or 'auto'

    if config_type == 'auto':
        # Infer from file content
        config_type = _infer_config_type(config_path)

    if config_type == 'pipeline':
        is_valid, errors = validate_pipeline_config(config_path)
    else:
        is_valid, errors = validate_dataset_config(config_path)

    if is_valid:
        logger.success(f"✓ Configuration is valid")
        sys.exit(0)
    else:
        logger.error(f"✗ Validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
```

---

#### 1.3 Better Error Messages

**Implementation**: Enhance [nirs4all/pipeline/config/pipeline_config.py](../nirs4all/pipeline/config/pipeline_config.py)

```python
@staticmethod
def _load_str_steps(definition: str) -> List[Any]:
    """Load steps from JSON/YAML file with better error messages."""
    if definition.endswith(('.json', '.yaml', '.yml')):
        if not Path(definition).is_file():
            raise FileNotFoundError(
                f"Configuration file does not exist: {definition}\n"
                f"Please check the file path and try again."
            )

        try:
            with open(definition, 'r', encoding='utf-8') as f:
                if definition.endswith('.json'):
                    pipeline_definition = json.load(f)
                else:
                    pipeline_definition = yaml.safe_load(f)

        except json.JSONDecodeError as exc:
            # Parse error message to extract line number
            line_num = getattr(exc, 'lineno', 'unknown')
            col_num = getattr(exc, 'colno', 'unknown')

            raise ValueError(
                f"Invalid JSON in {definition}\n"
                f"Error at line {line_num}, column {col_num}:\n"
                f"  {exc.msg}\n\n"
                f"Please check your JSON syntax."
            ) from exc

        except yaml.YAMLError as exc:
            # Extract line number from YAML error
            if hasattr(exc, 'problem_mark'):
                mark = exc.problem_mark
                line_num = mark.line + 1
                col_num = mark.column + 1
            else:
                line_num = 'unknown'
                col_num = 'unknown'

            raise ValueError(
                f"Invalid YAML in {definition}\n"
                f"Error at line {line_num}, column {col_num}:\n"
                f"  {exc.problem}\n\n"
                f"Please check your YAML syntax."
            ) from exc
```

---

### Phase 2: CLI Commands (4-5 weeks)

#### 2.1 Run Command

**File**: [nirs4all/cli/commands/run.py](../nirs4all/cli/commands/run.py)

```python
"""Pipeline execution CLI command."""

import argparse
import sys
from pathlib import Path

from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


def run_pipeline(args):
    """Run a pipeline on a dataset."""
    # Load pipeline config
    try:
        pipeline_config = PipelineConfigs(
            args.pipeline,
            name=args.name or "cli_pipeline"
        )
    except Exception as e:
        logger.error(f"Failed to load pipeline config: {e}")
        sys.exit(1)

    # Load dataset config
    try:
        dataset_config = DatasetConfigs(args.dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset config: {e}")
        sys.exit(1)

    # Create runner
    runner = PipelineRunner(
        workspace_path=args.workspace,
        save_artifacts=args.save_artifacts,
        save_charts=args.save_charts,
        verbose=args.verbose,
        random_state=args.random_state
    )

    # Run pipeline
    logger.info("Starting pipeline execution...")
    try:
        predictions, per_dataset = runner.run(
            pipeline_config,
            dataset_config
        )
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

    # Output results
    if args.output:
        output_path = Path(args.output)
        predictions.to_parquet(output_path)
        logger.success(f"Results saved to: {output_path}")
    else:
        # Display top results
        top = predictions.top(n=5)
        logger.info("\nTop 5 results:")
        for i, pred in enumerate(top):
            logger.info(f"  {i+1}. Score: {pred['test_score']:.4f}")

    sys.exit(0)


def add_run_command(subparsers):
    """Add run command to CLI."""
    run_parser = subparsers.add_parser(
        'run',
        help='Run a pipeline on a dataset'
    )

    run_parser.add_argument(
        '--pipeline',
        type=str,
        required=True,
        help='Path to pipeline config (JSON/YAML) or Python list'
    )

    run_parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset config (JSON/YAML) or folder'
    )

    run_parser.add_argument(
        '--name',
        type=str,
        help='Pipeline name (default: cli_pipeline)'
    )

    run_parser.add_argument(
        '--workspace',
        type=str,
        default='workspace',
        help='Workspace path (default: workspace)'
    )

    run_parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help='Verbosity level (default: 1)'
    )

    run_parser.add_argument(
        '--save-artifacts',
        action='store_true',
        help='Save trained artifacts'
    )

    run_parser.add_argument(
        '--save-charts',
        action='store_true',
        help='Save analysis charts'
    )

    run_parser.add_argument(
        '--random-state',
        type=int,
        help='Random seed for reproducibility'
    )

    run_parser.add_argument(
        '--output',
        type=str,
        help='Output file for results (Parquet format)'
    )

    run_parser.set_defaults(func=run_pipeline)
```

**Usage**:
```bash
nirs4all run --pipeline config/pipeline.yaml \
             --dataset config/dataset.yaml \
             --save-artifacts \
             --verbose 1 \
             --output results.parquet
```

---

#### 2.2 Predict Command

**File**: [nirs4all/cli/commands/predict.py](../nirs4all/cli/commands/predict.py)

```python
"""Prediction CLI command."""

def predict_command(args):
    """Run prediction on new data."""
    from nirs4all.pipeline.bundle import BundleLoader
    from nirs4all.data import DatasetConfigs
    import numpy as np

    # Load model (from bundle, folder, or prediction ID)
    if args.model.endswith('.n4a'):
        loader = BundleLoader(args.model)
        # Prediction via bundle
    else:
        # Load from workspace
        from nirs4all.pipeline import Predictor
        predictor = Predictor(
            selection=args.model,
            workspace_path=args.workspace
        )

    # Load data
    if args.data.endswith(('.csv', '.parquet')):
        # Load raw data file
        if args.data.endswith('.csv'):
            X = np.loadtxt(args.data, delimiter=',', skiprows=1)
        else:
            import pandas as pd
            X = pd.read_parquet(args.data).values
    else:
        # Load from dataset config
        dataset_config = DatasetConfigs(args.data)
        dataset = dataset_config.get_dataset_at(0)
        X = dataset.get_X_test() or dataset.get_X_train()

    # Predict
    y_pred = loader.predict(X) if hasattr(loader, 'predict') else predictor.predict(X)

    # Output
    if args.output:
        np.savetxt(args.output, y_pred, delimiter=',')
        logger.success(f"Predictions saved to: {args.output}")
    else:
        print(y_pred)
```

---

#### 2.3 Export Command

**File**: [nirs4all/cli/commands/export.py](../nirs4all/cli/commands/export.py)

```python
"""Export CLI commands."""

def export_bundle(args):
    """Export a trained pipeline to bundle."""
    from nirs4all.pipeline.bundle import BundleGenerator

    generator = BundleGenerator(
        workspace_path=args.workspace,
        verbose=args.verbose
    )

    bundle_path = generator.export(
        source=args.source,
        output_path=args.output,
        format=args.format,
        include_metadata=args.include_metadata,
        compress=not args.no_compress
    )

    logger.success(f"Bundle exported to: {bundle_path}")
```

---

### Phase 3: Polish & Production (2-3 weeks)

#### 3.1 Configuration Examples

**Create**: `examples/configs/`
```
examples/configs/
├── pipelines/
│   ├── simple_pls.yaml
│   ├── advanced_preprocessing.yaml
│   ├── multi_model.json
│   └── generator_search.yaml
└── datasets/
    ├── regression_basic.yaml
    ├── classification.json
    └── multi_source.yaml
```

**Example**: `examples/configs/datasets/regression_basic.yaml`
```yaml
# Basic regression dataset configuration
name: my_regression_dataset

# Training data
train_x: sample_data/regression/Xcal.csv
train_y: sample_data/regression/Ycal.csv

# Test data
test_x: sample_data/regression/Xval.csv
test_y: sample_data/regression/Yval.csv

# Task configuration
task_type: regression
signal_type: absorbance
aggregate: sample_id

# Loading parameters
train_x_params:
  header_unit: nm
  delimiter: ","
  decimal_separator: "."
  has_header: true

test_x_params:
  header_unit: nm
  delimiter: ","
```

---

#### 3.2 Documentation Updates

**Files to Create/Update**:

1. **docs/user_guide/cli_usage.md** - Complete CLI guide
2. **docs/user_guide/config_files.md** - Config file reference
3. **docs/tutorials/cli_workflow.md** - CLI workflow tutorial
4. **README.md** - Add CLI quick start

**Example Section for README.md**:
```markdown
## Quick Start with CLI

### 1. Create configuration files

**Pipeline** (`pipeline.yaml`):
```yaml
pipeline:
  - class: sklearn.preprocessing.MinMaxScaler
  - class: sklearn.model_selection.ShuffleSplit
    params:
      n_splits: 5
  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
```

**Dataset** (`dataset.yaml`):
```yaml
train_x: data/Xcal.csv
train_y: data/Ycal.csv
test_x: data/Xval.csv
test_y: data/Yval.csv
task_type: regression
signal_type: absorbance
```

### 2. Run the pipeline

```bash
nirs4all run --pipeline pipeline.yaml --dataset dataset.yaml
```

### 3. Export the best model

```bash
nirs4all export bundle --source best_prediction_id --output model.n4a
```

### 4. Use for predictions

```bash
nirs4all predict --model model.n4a --data new_samples.csv --output predictions.csv
```
```

---

#### 3.3 Integration Tests

**File**: `tests/cli/test_cli_integration.py`

```python
"""Integration tests for CLI commands."""

import subprocess
import tempfile
from pathlib import Path
import yaml
import json

def test_cli_run_with_yaml_configs(tmp_path):
    """Test complete CLI workflow with YAML configs."""
    # Create pipeline config
    pipeline_config = {
        "pipeline": [
            {"class": "sklearn.preprocessing.MinMaxScaler"},
            {"model": {"class": "sklearn.linear_model.LinearRegression"}}
        ]
    }
    pipeline_file = tmp_path / "pipeline.yaml"
    with open(pipeline_file, 'w') as f:
        yaml.dump(pipeline_config, f)

    # Create dataset config
    # (would need to create actual data files first)

    # Run CLI command
    result = subprocess.run(
        ["nirs4all", "run",
         "--pipeline", str(pipeline_file),
         "--dataset", "sample_data/regression"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "pipeline execution" in result.stdout.lower()
```

---

## Recommendations

### Priority 1 (Immediate - Week 1-2)

1. **Implement DatasetConfigs file loading**
   - Add JSON/YAML file support to `parse_config()`
   - Test with comprehensive test suite
   - Document expected file format

2. **Create config examples**
   - Add `examples/configs/` directory
   - Create 5-10 reference configs
   - Document in user guide

3. **Improve error messages**
   - Add line number reporting for parse errors
   - Add helpful suggestions for common mistakes

### Priority 2 (Core CLI - Week 3-5)

4. **Implement core CLI commands**
   - `nirs4all run`
   - `nirs4all predict`
   - `nirs4all export bundle`

5. **Add config validation**
   - JSON schema definitions
   - `nirs4all config validate` command
   - Helpful validation error messages

### Priority 3 (Enhancement - Week 6-8)

6. **Add remaining CLI commands**
   - Data inspection commands
   - Visualization commands
   - Library management commands

7. **Documentation and polish**
   - Complete CLI user guide
   - Tutorial workflows
   - Integration tests

---

## Breaking Changes / Compatibility Notes

### No Breaking Changes Required ✅

All proposed changes are **additive** - existing code will continue to work:

1. **DatasetConfigs**: File loading is new functionality
   ```python
   # Existing code still works
   config = DatasetConfigs("folder_path")
   config = DatasetConfigs({"train_x": "..."})

   # New functionality
   config = DatasetConfigs("dataset.yaml")  # Added
   ```

2. **PipelineConfigs**: Already supports file loading
   ```python
   # Already works today
   config = PipelineConfigs("pipeline.yaml")
   ```

3. **Bundle format**: No changes to .n4a structure

### Recommendations for Users

When the CLI is released, encourage migration to config files:

**Before** (Python code):
```python
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

pipeline = [MinMaxScaler(), PLSRegression(n_components=10)]
config = PipelineConfigs(pipeline, "my_pipeline")
dataset_config = DatasetConfigs("path/to/data")

runner = PipelineRunner(verbose=1)
predictions = runner.run(config, dataset_config)
```

**After** (CLI with config files):
```bash
# Save pipeline to pipeline.yaml once
# Save dataset config to dataset.yaml once
nirs4all run --pipeline pipeline.yaml --dataset dataset.yaml
```

---

## Testing Strategy

### Unit Tests
- ✅ Component serialization (already exists)
- ✅ PipelineConfigs file loading (already exists)
- ❌ DatasetConfigs file loading (needs to be added)
- ❌ Config validation (needs to be added)

### Integration Tests
- ❌ End-to-end CLI workflows
- ❌ Multi-step pipelines via CLI
- ❌ Export and reload workflows

### Example Files as Tests
- Create `.yaml`/`.json` examples
- Run them in CI/CD
- Ensures examples stay current

---

## Conclusion

The nirs4all codebase is **80% ready** for comprehensive CLI implementation:

✅ **Ready Components**:
- PipelineConfigs (JSON/YAML support)
- Component serialization
- Bundle export/import
- Basic CLI infrastructure

⚠️ **Needs Work**:
- DatasetConfigs file loading (2-3 days)
- Config validation (3-4 days)
- Core CLI commands (2-3 weeks)
- Documentation (1 week)

**Total Estimated Effort**: 6-8 weeks for full CLI roadmap implementation

**Recommended Approach**: Incremental releases
- v0.6.0: DatasetConfigs file loading + config validation
- v0.7.0: Core CLI commands (run, predict, export)
- v0.8.0: Full CLI suite + comprehensive docs

This approach allows early user feedback and reduces risk.
