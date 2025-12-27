# Prediction Reload & Transfer Design Document

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Core Insight: Controller-Agnostic Design](#2-core-insight-controller-agnostic-design)
3. [Proposed Architecture](#3-proposed-architecture)
4. [Multiple Prediction Sources](#4-multiple-prediction-sources)
5. [Controller-Agnostic Prediction Flow](#5-controller-agnostic-prediction-flow)
6. [Standalone Prediction Bundle](#6-standalone-prediction-bundle)
7. [DAG Extraction & Visualization](#7-dag-extraction--visualization)
8. [Retraining & Transfer Support](#8-retraining--transfer-support)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Data Model Changes Summary](#10-data-model-changes-summary)
11. [API Summary](#11-api-summary)
12. [Backwards Compatibility & Migration](#12-backwards-compatibility--migration)
13. [Testing Strategy](#13-testing-strategy)
14. [Summary](#14-summary)
15. [Open Questions](#15-open-questions)

---

## Executive Summary

This document analyzes the current prediction/reload system in nirs4all and proposes a clean, deterministic, controller-agnostic architecture that supports:

- **Prediction**: Replay the exact pipeline path that produced a prediction
- **Multiple Sources**: Predict from prediction, folder, run, artifact, or exported bundle
- **Retraining/Transfer**: Extract and retrain a specific model with new data
- **DAG Extraction**: Isolate and export the minimal subgraph as a standalone package
- **Portability**: Single-file prediction bundles for deployment

### Design Principles

1. **Controller-Agnostic**: The system must work with any controller (existing or custom) without hardcoding operator types
2. **Deterministic**: One identifier → one exact pipeline path, always
3. **Minimal Replay**: Only execute steps that are needed
4. **Composable**: Same infrastructure for predict, retrain, transfer, and export
5. **Portable**: Enable standalone prediction without nirs4all dependency

---

## 1. Current State Analysis

### 1.1 How Training Works Today

```
Pipeline Definition → Expansion (generators) → Step Execution → Predictions + Artifacts
```

**Key Components:**
1. **PipelineConfigs**: User's pipeline definition (may contain `_or_`, generators, etc.)
2. **Generator Expansion**: Expands to multiple concrete pipelines
3. **Step Execution**: Each step (transform, model, etc.) is executed by controllers
4. **Artifact Storage**: Models/transformers saved to `workspace/binaries/<dataset>/`
5. **Predictions Database**: Results stored in `runs/<run_id>/predictions.csv`

**Controllers** (non-exhaustive, user can add custom ones):
- `transformer.py` - X preprocessing (SNV, SG, wavelets, etc.)
- `y_transformer.py` - Target preprocessing
- `feature_augmentation.py` - Multi-branch preprocessing
- `concat_transform.py` - Transform chains
- `sample_partitioner.py` - Train/val/test splitting
- `resampler.py` - Data resampling
- `feature_selection.py` - Feature selection
- `outlier_excluder.py` - Outlier removal branches
- `base_model.py` - Model training
- `meta_model.py` - Stacking/ensemble
- *...and potentially user-defined controllers*

### 1.2 Current Artifact Storage

**Location**: `workspace/binaries/<dataset>/`

**Naming**: `<artifact_type>_<class_name>_<content_hash>.joblib`

**Manifest**: `runs/<run_id>/<pipeline_uid>/manifest.yaml` contains:
```yaml
artifacts:
  items:
    - artifact_id: "0001:3:0"  # pipeline:step:fold
      class_name: "PLSRegression"
      artifact_type: "model"
      content_hash: "abc123..."
      filename: "model_PLSRegression_abc123.joblib"
      step_index: 3
      fold_id: 0
      branch_path: [0]
      depends_on: []
```

### 1.3 Current Prediction Record

```python
prediction = {
    'id': 'abc123def456',
    'dataset_name': 'wheat',
    'config_name': '0001_abc123',
    'pipeline_uid': 'abc123def456',
    'step_idx': 5,
    'op_counter': 42,
    'model_name': 'Q5_PLS_10',      # Custom name OR class name
    'model_classname': 'PLSRegression',
    'fold_id': 0,
    'branch_id': 0,
    'branch_name': 'branch_0',
    'y_pred': [...],
    'y_true': [...],
    # ... scores, etc.
}
```

### 1.4 Current Prediction Flow

```
User calls: predictor.predict(best_prediction, new_dataset)
    │
    ├── 1. Get run_dir from prediction's config_path
    │
    ├── 2. Load manifest.yaml using pipeline_uid
    │
    ├── 3. Load pipeline.json (full pipeline definition)
    │
    ├── 4. Create ArtifactLoader from manifest
    │
    ├── 5. Execute FULL pipeline in predict mode
    │       └── At each step: load artifacts by step_index matching
    │
    └── 6. Filter predictions to find matching model_name/step_idx
```

### 1.5 Problems with Current Approach

| Problem | Description | Impact |
|---------|-------------|--------|
| **Name Ambiguity** | Binary keys use `classname_opnum` but lookup uses `customname_opcounter` | Prediction fails with "model not found" |
| **Full Pipeline Replay** | Entire pipeline is replayed even if only one model is needed | Slow, wasteful for complex pipelines |
| **No DAG Tracking** | No explicit graph of which transforms fed which model | Cannot extract minimal subgraph |
| **Preprocessing Not Linked** | Prediction doesn't store which preprocessing path was used | Ambiguity in multi-branch pipelines |
| **Fold Models Separate** | Each fold has separate artifact but prediction uses avg/w_avg | Complex mapping for CV ensembles |
| **No Dependency Chain** | Artifacts don't track their preprocessing dependencies | Cannot reconstruct pipeline path |
| **Controller Coupling** | Any custom executor would need to know all controller types | Not extensible |

---

## 2. Core Insight: Controller-Agnostic Design

### 2.1 The Problem with Hardcoding Controller Logic

The initial design proposed a `MinimalPipelineExecutor` that knew about each operator type:

```python
# BAD DESIGN - Hardcodes controller knowledge
if step.operator_type == 'transform':
    transformer.transform(X)
elif step.operator_type == 'feature_augmentation':
    # Apply branch transforms...
elif step.operator_type == 'concat_transform':  # Forgot this!
    # ...
elif step.operator_type == 'model':
    model.predict(X)
# What about custom controllers?
```

**Problems:**
- Must enumerate ALL controller types (transform, y_processing, feature_augmentation, concat_transform, resampler, feature_selection, outlier_excluder, sample_filter, etc.)
- Cannot handle user-defined custom controllers
- Duplicates logic that already exists in controllers
- Maintenance nightmare when new controllers are added

### 2.2 The Solution: Reuse Existing Controllers

Instead of a custom minimal executor, we:

1. **Store the exact minimal pipeline** (subset of steps) that produced the prediction
2. **Replay using existing controllers** in predict mode
3. **Inject artifacts by position** (not by name matching)

The controllers already know how to handle predict mode - we just need to:
- Give them the right pipeline subset
- Give them the right artifacts for each step

```python
# GOOD DESIGN - Controller-agnostic
minimal_pipeline = extract_minimal_pipeline(prediction)
artifacts_by_step = load_artifacts_by_step(prediction)

# Use existing executor with minimal pipeline
executor.execute(
    steps=minimal_pipeline,
    mode='predict',
    artifact_provider=artifacts_by_step  # Provides artifacts by step index
)
```

### 2.3 Key Abstractions

```
┌─────────────────────────────────────────────────────────────────┐
│                     Prediction Source                            │
│  (prediction dict, folder path, run, artifact_id, bundle)       │
└─────────────────────────────────────────────┬───────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PredictionResolver                            │
│  Normalizes any source to: (minimal_pipeline, artifact_map)     │
└─────────────────────────────────────────────┬───────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Existing Controllers                          │
│  (transformer, model, feature_augmentation, custom, etc.)       │
│  Each receives artifacts for its step via artifact_provider     │
└─────────────────────────────────────────────┬───────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Prediction                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Proposed Architecture

### 3.1 Core Concept: Execution Trace

During training, we record an **Execution Trace** - the exact sequence of steps and artifacts that produced a prediction. This is controller-agnostic:

```python
@dataclass
class ExecutionTrace:
    """Records the exact path through pipeline that produced a prediction."""

    trace_id: str                    # Unique identifier (hash of content)
    pipeline_uid: str                # Which pipeline version

    # The minimal pipeline definition (subset of original)
    steps: List[dict]                # Serialized step configs

    # Artifacts produced/used at each step
    artifacts: List[StepArtifacts]   # Indexed by step position

    # Branch/fold context
    branch_path: List[int]           # Branch taken at each fork
    fold_id: Optional[Union[int, str]]  # Specific fold or 'w_avg', 'avg'
    fold_weights: Optional[Dict[int, float]]  # For weighted average

    # For CV ensemble
    fold_artifact_ids: Optional[Dict[int, str]]  # Per-fold model artifacts


@dataclass
class StepArtifacts:
    """Artifacts for a single step."""

    step_index: int
    artifact_ids: List[str]          # All artifacts from this step
    primary_artifact_id: Optional[str]  # Main artifact (for model step)
```

### 3.2 New Prediction Record

```python
prediction = {
    # Existing fields (unchanged)
    'id': 'abc123def456',
    'dataset_name': 'wheat',
    'pipeline_uid': 'abc123def456',
    'model_name': 'Q5_PLS_10',
    'model_classname': 'PLSRegression',
    'fold_id': 'w_avg',
    'branch_id': 0,
    'y_pred': [...],
    'y_true': [...],
    # ... scores, metrics, etc.

    # NEW: Execution Trace Reference
    'trace_id': 'trace_xyz789',      # Reference to ExecutionTrace

    # NEW: Direct artifact reference (for quick access)
    'model_artifact_id': '0001:4:all',  # Primary model artifact

    # NEW: For CV ensemble predictions
    'fold_artifact_ids': {            # Per-fold artifacts
        0: '0001:4:0',
        1: '0001:4:1',
    },
    'fold_weights': {0: 0.52, 1: 0.48},
}
```

### 3.3 Artifact Dependency Tracking

Each artifact records its dependencies (controller-agnostic):

```python
@dataclass
class ArtifactRecord:
    artifact_id: str
    class_name: str
    custom_name: Optional[str]       # User-provided name if any
    artifact_type: str               # 'model', 'transformer', 'encoder', etc.
    content_hash: str
    filename: str

    # Position in pipeline
    step_index: int
    branch_path: List[int]
    fold_id: Optional[int]

    # Dependencies (other artifact IDs this depends on)
    depends_on: List[str]            # Direct dependencies

    # Original step config (for reconstruction)
    step_config: dict                # The exact config used
```

### 3.4 Manifest Structure

```yaml
# runs/<run_id>/<pipeline_uid>/manifest.yaml

pipeline_uid: "abc123def456"
created_at: "2024-12-13T10:30:00"

# Artifacts (existing, enhanced)
artifacts:
  items:
    - artifact_id: "0001:0:all"
      class_name: "MinMaxScaler"
      custom_name: null
      artifact_type: "transformer"
      content_hash: "abc123..."
      filename: "transformer_MinMaxScaler_abc123.joblib"
      step_index: 0
      branch_path: []
      fold_id: null
      depends_on: []
      step_config: {"class": "sklearn.preprocessing.MinMaxScaler"}

    - artifact_id: "0001:4:0"
      class_name: "PLSRegression"
      custom_name: "Q5_PLS_10"
      artifact_type: "model"
      content_hash: "xyz789..."
      filename: "model_PLSRegression_xyz789.joblib"
      step_index: 4
      branch_path: [0]
      fold_id: 0
      depends_on: ["0001:0:all", "0001:2:0"]
      step_config: {"model": "...", "name": "Q5_PLS_10"}

# NEW: Execution Traces
execution_traces:
  trace_xyz789:
    pipeline_uid: "abc123def456"
    steps:
      - {"class": "sklearn.preprocessing.MinMaxScaler"}
      - {"y_processing": "sklearn.preprocessing.MinMaxScaler"}
      - {"feature_augmentation": ["SNV", "SavitzkyGolay"]}
      - {"splitter": "RepeatedKFold", "n_splits": 2}
      - {"model": "PLSRegression", "name": "Q5_PLS_10"}
    artifacts:
      - {step_index: 0, artifact_ids: ["0001:0:all"]}
      - {step_index: 1, artifact_ids: ["0001:1:all"]}
      - {step_index: 2, artifact_ids: ["0001:2:0:SNV", "0001:2:0:SG"]}
      - {step_index: 3, artifact_ids: []}
      - {step_index: 4, artifact_ids: ["0001:4:0", "0001:4:1"], primary: "0001:4:all"}
    branch_path: [0]
    fold_id: "w_avg"
    fold_weights: {0: 0.52, 1: 0.48}
```

---

## 4. Multiple Prediction Sources

### 4.1 Supported Sources

Users should be able to predict from multiple sources:

| Source Type | Description | Example |
|-------------|-------------|---------|
| **prediction** | A prediction dict from a previous run | `runner.predict(best_prediction, data)` |
| **folder** | Path to a pipeline folder | `runner.predict("runs/0001_wheat_abc123", data)` |
| **run** | Best prediction from a run | `runner.predict(Run("runs/0001_wheat"), data)` |
| **artifact_id** | Direct artifact reference | `runner.predict("0001:4:all", data)` |
| **bundle** | Exported prediction bundle | `runner.predict("exports/model_bundle.n4a", data)` |
| **trace_id** | Execution trace reference | `runner.predict("trace:xyz789", data)` |

### 4.2 PredictionResolver

A unified resolver that normalizes any source to the required components:

```python
class PredictionResolver:
    """Resolves any prediction source to executable components."""

    def resolve(
        self,
        source: Union[dict, str, Path, 'Run', 'PredictionBundle'],
        workspace: Path
    ) -> ResolvedPrediction:
        """
        Resolve source to:
        - minimal_pipeline: List[dict] - steps to execute
        - artifact_map: Dict[str, Any] - step_index -> loaded artifacts
        - execution_trace: ExecutionTrace - full trace metadata
        - fold_strategy: Optional fold averaging info
        """

        if isinstance(source, dict):
            return self._resolve_from_prediction(source, workspace)

        elif isinstance(source, (str, Path)):
            path = Path(source)

            # Check if it's an artifact ID pattern
            if self._is_artifact_id(str(source)):
                return self._resolve_from_artifact_id(source, workspace)

            # Check if it's a trace ID
            if str(source).startswith("trace:"):
                return self._resolve_from_trace_id(source[6:], workspace)

            # Check if it's a bundle file
            if path.suffix == '.n4a':
                return self._resolve_from_bundle(path)

            # Assume it's a folder path
            return self._resolve_from_folder(path, workspace)

        elif isinstance(source, Run):
            return self._resolve_from_run(source, workspace)

        elif isinstance(source, PredictionBundle):
            return self._resolve_from_bundle_object(source)

        raise ValueError(f"Unknown prediction source type: {type(source)}")

    def _resolve_from_prediction(self, pred: dict, workspace: Path) -> ResolvedPrediction:
        """Resolve from a prediction dictionary."""
        # Load manifest
        run_dir = self._find_run_dir(pred, workspace)
        manifest = self._load_manifest(run_dir, pred['pipeline_uid'])

        # Get execution trace
        trace_id = pred.get('trace_id')
        if trace_id:
            trace = manifest.execution_traces[trace_id]
        else:
            # Legacy: reconstruct trace from prediction metadata
            trace = self._reconstruct_trace(pred, manifest)

        # Load artifacts
        artifact_map = self._load_artifacts(trace, manifest, workspace)

        return ResolvedPrediction(
            minimal_pipeline=trace.steps,
            artifact_map=artifact_map,
            execution_trace=trace,
            fold_strategy=self._get_fold_strategy(pred, trace)
        )

    def _resolve_from_folder(self, folder: Path, workspace: Path) -> ResolvedPrediction:
        """Resolve from a pipeline folder path."""
        # Load manifest from folder
        manifest = self._load_manifest_from_folder(folder)

        # Get the best prediction's trace (or default)
        predictions = self._load_predictions(folder)
        best_pred = predictions.top(n=1)[0]

        return self._resolve_from_prediction(best_pred, workspace)

    def _resolve_from_run(self, run: 'Run', workspace: Path) -> ResolvedPrediction:
        """Resolve from a Run object (best model)."""
        best_prediction = run.best_prediction()
        return self._resolve_from_prediction(best_prediction, workspace)

    def _resolve_from_artifact_id(self, artifact_id: str, workspace: Path) -> ResolvedPrediction:
        """Resolve from direct artifact ID."""
        # Find manifest containing this artifact
        manifest = self._find_manifest_for_artifact(artifact_id, workspace)
        artifact = manifest.artifacts[artifact_id]

        # Reconstruct trace from artifact dependencies
        trace = self._trace_from_artifact(artifact, manifest)
        artifact_map = self._load_artifacts(trace, manifest, workspace)

        return ResolvedPrediction(
            minimal_pipeline=trace.steps,
            artifact_map=artifact_map,
            execution_trace=trace,
            fold_strategy=None
        )

    def _resolve_from_bundle(self, bundle_path: Path) -> ResolvedPrediction:
        """Resolve from exported bundle file."""
        bundle = PredictionBundle.load(bundle_path)
        return ResolvedPrediction(
            minimal_pipeline=bundle.pipeline,
            artifact_map=bundle.artifacts,
            execution_trace=bundle.trace,
            fold_strategy=bundle.fold_strategy
        )


@dataclass
class ResolvedPrediction:
    """Normalized prediction source ready for execution."""

    minimal_pipeline: List[dict]        # Steps to execute
    artifact_map: Dict[int, List[Any]]  # step_index -> loaded artifacts
    execution_trace: ExecutionTrace     # Full trace metadata
    fold_strategy: Optional[FoldStrategy]  # How to combine fold predictions
```

### 4.3 User-Facing API

```python
from nirs4all.pipeline import PipelineRunner
from nirs4all.data import DatasetConfigs

runner = PipelineRunner()
new_data = DatasetConfigs({'X': 'path/to/new_spectra.csv'})

# 1. From prediction dict (most common)
y_pred, _ = runner.predict(best_prediction, new_data)

# 2. From folder path
y_pred, _ = runner.predict("runs/0001_wheat_abc123/", new_data)

# 3. From Run object
from nirs4all.pipeline import Run
run = Run.load("runs/0001_wheat_abc123/")
y_pred, _ = runner.predict(run, new_data)  # Uses best model

# 4. From artifact ID
y_pred, _ = runner.predict("0001:4:all", new_data)

# 5. From exported bundle
y_pred, _ = runner.predict("exports/wheat_model.n4a", new_data)

# 6. From trace ID
y_pred, _ = runner.predict("trace:xyz789", new_data)
```

---

## 5. Controller-Agnostic Prediction Flow

### 5.1 Key Insight: Reuse Existing Infrastructure

Instead of creating a custom executor that knows about all controllers, we:

1. **Extract the minimal pipeline** (only steps that contributed to this prediction)
2. **Provide an artifact injection mechanism** that controllers use
3. **Run through existing executor** in predict mode

### 5.2 Artifact Provider Interface

Controllers request artifacts via a provider interface, not by name matching:

```python
class ArtifactProvider:
    """Provides artifacts to controllers during prediction."""

    def __init__(self, artifact_map: Dict[int, List[Any]]):
        """
        Args:
            artifact_map: step_index -> list of loaded artifacts
        """
        self._artifacts = artifact_map
        self._position = {}  # Track position within each step

    def get_artifacts(self, step_index: int) -> List[Any]:
        """Get all artifacts for a step."""
        return self._artifacts.get(step_index, [])

    def get_artifact(self, step_index: int, index: int = 0) -> Any:
        """Get specific artifact from a step."""
        artifacts = self.get_artifacts(step_index)
        if index < len(artifacts):
            return artifacts[index]
        raise ValueError(f"No artifact at step {step_index}, index {index}")

    def next_artifact(self, step_index: int) -> Any:
        """Get next artifact for a step (for iterating through multiple)."""
        pos = self._position.get(step_index, 0)
        artifact = self.get_artifact(step_index, pos)
        self._position[step_index] = pos + 1
        return artifact


# In RuntimeContext
@dataclass
class RuntimeContext:
    # ... existing fields ...
    artifact_provider: Optional[ArtifactProvider] = None
```

### 5.3 Controller Integration

Controllers check for artifact_provider in predict mode:

```python
# In any controller (transformer, model, etc.)
class BaseController:

    def execute(self, step_info, dataset, context, runtime_context, ...):
        mode = context.state.mode

        if mode == 'predict':
            # Get artifact from provider (not by name matching)
            if runtime_context.artifact_provider:
                artifact = runtime_context.artifact_provider.next_artifact(
                    context.state.step_number
                )
                return self._apply_artifact(artifact, dataset, context)
            else:
                # Legacy: use name-based loading
                return self._legacy_predict(...)

        else:
            # Training mode - create and save artifact
            artifact = self._create_artifact(...)
            self._save_artifact(artifact, runtime_context)
            return result
```

### 5.4 Minimal Pipeline Execution

```python
class MinimalPredictor:
    """Executes minimal pipeline using existing controllers."""

    def __init__(self, resolver: PredictionResolver):
        self.resolver = resolver

    def predict(
        self,
        source: Any,  # prediction, folder, run, artifact_id, bundle
        dataset: DatasetConfigs,
        workspace: Path
    ) -> Tuple[np.ndarray, Predictions]:
        """Execute prediction using any source."""

        # 1. Resolve source to components
        resolved = self.resolver.resolve(source, workspace)

        # 2. Create artifact provider
        artifact_provider = ArtifactProvider(resolved.artifact_map)

        # 3. Create runtime context with provider
        runtime_context = RuntimeContext(
            artifact_provider=artifact_provider,
            # ... other fields
        )

        # 4. Create execution context in predict mode
        context = ExecutionContext(
            state=PipelineState(mode='predict'),
            # ... other fields
        )

        # 5. Execute minimal pipeline using EXISTING executor
        executor = self._get_executor()
        predictions = Predictions()

        executor.execute(
            steps=resolved.minimal_pipeline,  # Only the steps we need
            mode='predict',
            dataset=dataset,
            context=context,
            runtime_context=runtime_context,
            predictions=predictions
        )

        # 6. Handle fold averaging if needed
        if resolved.fold_strategy:
            return self._apply_fold_strategy(predictions, resolved.fold_strategy)

        return predictions.get_y_pred(), predictions
```

### 5.5 Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Request                                    │
│  runner.predict(source, new_data)                                           │
│  source = prediction | folder | run | artifact_id | bundle                  │
└─────────────────────────────────────────────────┬───────────────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PredictionResolver.resolve()                         │
│  - Normalize source to (minimal_pipeline, artifact_map, trace, fold_info)   │
│  - Load artifacts from manifest/bundle                                       │
└─────────────────────────────────────────────────┬───────────────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ArtifactProvider                                    │
│  - Wraps loaded artifacts                                                    │
│  - Provides get_artifact(step_index) interface                              │
└─────────────────────────────────────────────────┬───────────────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Existing Executor (unchanged!)                            │
│  - Executes minimal_pipeline steps                                           │
│  - Controllers get artifacts via artifact_provider                          │
│  - Each controller handles its own predict logic                            │
└─────────────────────────────────────────────────┬───────────────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Predictions                                        │
│  - Apply fold_strategy if needed (avg, w_avg)                               │
│  - Return y_pred                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Standalone Prediction Bundle

### 6.1 Motivation

Users need to deploy predictions without the full nirs4all dependency:
- **Production deployment**: Lightweight container/lambda
- **Sharing**: Send a model to a colleague
- **Archival**: Reproducible predictions years later
- **Edge deployment**: Embedded systems, mobile

### 6.2 Bundle Formats

We propose three bundle formats for different use cases:

#### 6.2.1 Full Bundle (`.n4a`)

Complete nirs4all bundle with all metadata:

```
model_bundle.n4a (ZIP archive)
├── manifest.json           # Bundle metadata
├── pipeline.json           # Minimal pipeline config
├── trace.json              # Execution trace
├── artifacts/
│   ├── step_0_MinMaxScaler.joblib
│   ├── step_1_y_MinMaxScaler.joblib
│   ├── step_2_SNV.joblib
│   ├── step_2_SavitzkyGolay.joblib
│   ├── step_4_fold0_PLSRegression.joblib
│   └── step_4_fold1_PLSRegression.joblib
├── fold_weights.json       # For CV ensemble
└── predict.py              # Standalone script (optional)
```

**Usage with nirs4all:**
```python
from nirs4all.pipeline import PipelineRunner

runner = PipelineRunner()
y_pred, _ = runner.predict("model_bundle.n4a", X_new)
```

#### 6.2.2 Portable Bundle (`.n4a.py`)

Single Python file with embedded artifacts (base64):

```python
#!/usr/bin/env python3
"""
Standalone prediction script generated by nirs4all.
No external dependencies except numpy, scipy, sklearn.

Original prediction ID: abc123def456
Created: 2024-12-13T10:30:00
"""

import base64
import pickle
import numpy as np

# Embedded artifacts (base64 encoded)
ARTIFACTS = {
    "step_0": "gASVxAAA...",  # MinMaxScaler
    "step_2_SNV": "gASVyQAA...",
    "step_4_fold0": "gASV0gAA...",  # PLSRegression
    "step_4_fold1": "gASV0wAA...",
}

FOLD_WEIGHTS = {0: 0.52, 1: 0.48}

def load_artifact(key: str):
    """Load artifact from embedded data."""
    data = base64.b64decode(ARTIFACTS[key])
    return pickle.loads(data)

def predict(X: np.ndarray) -> np.ndarray:
    """
    Make predictions on new data.

    Args:
        X: Input features, shape (n_samples, n_features)

    Returns:
        Predictions, shape (n_samples,)
    """
    # Step 0: X preprocessing
    scaler = load_artifact("step_0")
    X = scaler.transform(X)

    # Step 2: Feature augmentation (SNV + SG)
    snv = load_artifact("step_2_SNV")
    sg = load_artifact("step_2_SG")
    X_snv = snv.transform(X)
    X_sg = sg.transform(X)
    X = np.hstack([X_snv, X_sg])

    # Step 4: Model prediction (weighted average of folds)
    preds = []
    weights = []
    for fold_id, weight in FOLD_WEIGHTS.items():
        model = load_artifact(f"step_4_fold{fold_id}")
        preds.append(model.predict(X))
        weights.append(weight)

    y_pred = np.average(preds, axis=0, weights=weights)

    # Step 1: Inverse y transform
    y_scaler = load_artifact("step_1")
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    return y_pred


if __name__ == "__main__":
    import sys
    import pandas as pd

    if len(sys.argv) < 2:
        print("Usage: python model_bundle.n4a.py input.csv [output.csv]")
        sys.exit(1)

    # Load input
    input_path = sys.argv[1]
    X = pd.read_csv(input_path).values

    # Predict
    y_pred = predict(X)

    # Output
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
        pd.DataFrame({"prediction": y_pred}).to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    else:
        for i, pred in enumerate(y_pred):
            print(f"{i}: {pred:.4f}")
```

**Usage:**
```bash
python model_bundle.n4a.py spectra.csv predictions.csv
```

#### 6.2.3 ONNX Export (Future)

For maximum portability and performance:

```python
# Export to ONNX (preprocessing + model fused)
runner.export_onnx(best_prediction, "model.onnx")

# Use anywhere
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
y_pred = session.run(None, {"input": X})[0]
```

**Note**: ONNX export is more complex because:
- Not all sklearn transformers are ONNX-compatible
- Custom transforms need conversion
- CV ensemble needs special handling

### 6.3 Bundle Generator

```python
class PredictionBundleGenerator:
    """Generates standalone prediction bundles."""

    def __init__(self, resolver: PredictionResolver):
        self.resolver = resolver

    def export(
        self,
        source: Any,
        output_path: Path,
        format: str = 'n4a',  # 'n4a', 'n4a.py', 'onnx'
        include_script: bool = True
    ) -> Path:
        """
        Export prediction source to standalone bundle.

        Args:
            source: Prediction source (dict, folder, run, etc.)
            output_path: Where to save the bundle
            format: Bundle format
            include_script: Include standalone predict.py

        Returns:
            Path to generated bundle
        """
        # Resolve source
        resolved = self.resolver.resolve(source, self.workspace)

        if format == 'n4a':
            return self._export_n4a(resolved, output_path, include_script)
        elif format == 'n4a.py':
            return self._export_portable(resolved, output_path)
        elif format == 'onnx':
            return self._export_onnx(resolved, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_n4a(self, resolved: ResolvedPrediction, output_path: Path, include_script: bool) -> Path:
        """Export as .n4a ZIP archive."""
        import zipfile
        import json

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Manifest
            manifest = {
                'format_version': '1.0',
                'created_at': datetime.now().isoformat(),
                'nirs4all_version': __version__,
                'trace_id': resolved.execution_trace.trace_id,
            }
            zf.writestr('manifest.json', json.dumps(manifest, indent=2))

            # Pipeline
            zf.writestr('pipeline.json', json.dumps(resolved.minimal_pipeline, indent=2))

            # Trace
            zf.writestr('trace.json', json.dumps(resolved.execution_trace.to_dict(), indent=2))

            # Artifacts
            for step_idx, artifacts in resolved.artifact_map.items():
                for i, artifact in enumerate(artifacts):
                    artifact_bytes = self._serialize_artifact(artifact)
                    zf.writestr(f'artifacts/step_{step_idx}_{i}.joblib', artifact_bytes)

            # Fold weights
            if resolved.fold_strategy:
                zf.writestr('fold_weights.json', json.dumps(resolved.fold_strategy.weights))

            # Standalone script
            if include_script:
                script = self._generate_predict_script(resolved)
                zf.writestr('predict.py', script)

        return output_path

    def _export_portable(self, resolved: ResolvedPrediction, output_path: Path) -> Path:
        """Export as single portable Python file."""
        import base64

        # Serialize all artifacts to base64
        artifacts_b64 = {}
        for step_idx, artifacts in resolved.artifact_map.items():
            for i, artifact in enumerate(artifacts):
                key = f"step_{step_idx}_{i}"
                artifact_bytes = self._serialize_artifact(artifact)
                artifacts_b64[key] = base64.b64encode(artifact_bytes).decode('ascii')

        # Generate Python file
        script = self._generate_portable_script(resolved, artifacts_b64)

        output_path.write_text(script)
        return output_path
```

### 6.4 User-Facing API

```python
from nirs4all.pipeline import PipelineRunner

runner = PipelineRunner()

# Export to .n4a bundle
runner.export(best_prediction, "exports/wheat_model.n4a")

# Export to portable Python script
runner.export(best_prediction, "exports/wheat_model.n4a.py", format='n4a.py')

# Export from folder
runner.export("runs/0001_wheat_abc123/", "exports/model.n4a")

# Export with options
runner.export(
    best_prediction,
    "exports/model.n4a",
    format='n4a',
    include_script=True,
    compress=True
)
```

---

## 7. DAG Extraction & Visualization

### 7.1 Pipeline DAG Structure

A pipeline can be represented as a DAG where:
- **Nodes** = Operations (transforms, models, splitters, any controller)
- **Edges** = Data flow (X, y, or both)

```
                    ┌─────────────┐
                    │ Raw Input   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ MinMaxScaler│  ◄── Step 0
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
        │    SNV    │ │   SG    │ │ Gaussian  │  ◄── Step 2 (feature_augmentation)
        └─────┬─────┘ └────┬────┘ └─────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐
                    │ RepeatedKFold│  ◄── Step 3
                    └──────┬──────┘
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
        │   PLS_10  │ │ PLS_20  │ │   Ridge   │  ◄── Step 4 (models)
        └─────┬─────┘ └────┬────┘ └─────┬─────┘
              │            │            │
              ▼            ▼            ▼
         Pred_1       Pred_2       Pred_3
```

### 7.2 Trace-Based Extraction (Controller-Agnostic)

Instead of knowing about controller types, we use the recorded trace:

```python
class TraceBasedExtractor:
    """Extracts minimal pipeline from execution trace."""

    def extract_minimal_pipeline(
        self,
        trace: ExecutionTrace,
        full_pipeline: List[dict]
    ) -> List[dict]:
        """
        Extract minimal pipeline from trace.

        The trace contains step indices and branch choices.
        We simply extract those steps from the full pipeline,
        with branch-specific subsetting.
        """
        minimal_steps = []

        for step_info in trace.steps:
            step_idx = step_info['step_index']
            original = full_pipeline[step_idx]

            # If this step has branch choices, subset to the taken branch
            if 'branch_index' in step_info:
                branch_idx = step_info['branch_index']
                minimal_step = self._extract_branch(original, branch_idx)
            else:
                minimal_step = original

            minimal_steps.append(minimal_step)

        return minimal_steps

    def _extract_branch(self, step: dict, branch_idx: int) -> dict:
        """Extract specific branch from a branching step."""
        # Works for any branching operator (feature_augmentation, outlier_excluder, etc.)
        # The structure is: {"operator_key": [...branches...]}
        for key, value in step.items():
            if isinstance(value, list):
                # This is a branching operator
                if branch_idx < len(value):
                    return {key: [value[branch_idx]]}
        return step  # No branching found, return as-is
```

### 7.3 DAG Visualization

```python
class PipelineDAG:
    """Visualize and analyze pipeline as DAG."""

    def from_trace(self, trace: ExecutionTrace) -> 'PipelineDAG':
        """Build DAG from execution trace."""
        # Each step becomes a node
        # Edges follow artifact dependencies
        pass

    def visualize(self, highlight_path: Optional[List[int]] = None) -> None:
        """Render DAG, optionally highlighting a specific path."""
        # Use matplotlib or graphviz
        pass

    def to_mermaid(self) -> str:
        """Export as Mermaid diagram for documentation."""
        pass

    def get_path_to_prediction(self, prediction: dict) -> List[int]:
        """Get step indices that contributed to a prediction."""
        pass
```

---

## 8. Retraining & Transfer Support

### 8.1 Retraining Use Cases

| Mode | Description | Preprocessing | Model |
|------|-------------|---------------|-------|
| **full** | Train from scratch | Fit new | Fit new |
| **finetune** | Continue training | Use existing | Continue training |
| **transfer** | Use preprocessing, new model | Use existing | Fit new |
| **partial** | Retrain specific step | Mixed | Mixed |

### 8.2 Retrain API Design

```python
from nirs4all.pipeline import PipelineRunner

runner = PipelineRunner()

# Use Case 1: Full retrain with same pipeline structure
predictions, _ = runner.retrain(
    source=best_prediction,  # Any source type
    dataset=new_data,
    mode='full'
)

# Use Case 2: Fine-tune existing model (continues training)
predictions, _ = runner.retrain(
    source=best_prediction,
    dataset=new_data,
    mode='finetune',
    epochs=10
)

# Use Case 3: Transfer preprocessing, new model
predictions, _ = runner.retrain(
    source=best_prediction,
    dataset=new_data,
    mode='transfer',
    new_model=XGBRegressor()
)

# Use Case 4: Extract, modify, and run
minimal_pipeline = runner.extract(best_prediction)
minimal_pipeline.steps[-1] = {"model": RandomForestRegressor(), "name": "RF_new"}
predictions, _ = runner.run(minimal_pipeline, new_data)
```

### 8.3 Transfer Learning Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    Retrain Request                                │
│  runner.retrain(source, new_data, mode='transfer')               │
└─────────────────────────────────────────────┬────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    PredictionResolver                             │
│  → Resolve source to (minimal_pipeline, artifact_map, trace)     │
└─────────────────────────────────────────────┬────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Mode-Specific Setup                            │
│  transfer mode:                                                   │
│    - Preprocessing steps: mode='predict' (use existing artifacts) │
│    - Model step: mode='train' (fit new model)                    │
└─────────────────────────────────────────────┬────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Executor                                       │
│  Execute with per-step mode override:                            │
│    Step 0 (MinMaxScaler): mode='predict', use artifact           │
│    Step 2 (SNV, SG): mode='predict', use artifacts               │
│    Step 4 (Model): mode='train', fit new model                   │
└─────────────────────────────────────────────┬────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    New Artifacts & Predictions                    │
│  - New model artifact saved                                       │
│  - New trace recorded (inherits preprocessing artifacts)         │
│  - New prediction stored                                          │
└──────────────────────────────────────────────────────────────────┘
```

### 8.4 Per-Step Mode Control

For fine-grained control, we can specify mode per step:

```python
@dataclass
class StepMode:
    """Mode override for a specific step."""
    step_index: int
    mode: str  # 'train', 'predict', 'finetune'
    artifact_id: Optional[str] = None  # For 'predict' mode

# Usage
runner.retrain(
    source=best_prediction,
    dataset=new_data,
    step_modes=[
        StepMode(0, 'predict'),  # Use existing scaler
        StepMode(2, 'train'),    # Retrain preprocessing
        StepMode(4, 'train'),    # Retrain model
    ]
)
```

---

## 9. Implementation Roadmap

### Phase 1: Foundation (Critical - Fixes Q5_predict.py)

**Goal**: Make prediction deterministic with artifact IDs.

| Task | File(s) | Description |
|------|---------|-------------|
| 1.1 | `base_model.py`, `prediction_assembler.py` | Store `model_artifact_id` in prediction during training |
| 1.2 | `artifact_registry.py`, `types.py` | Store `custom_name` in ArtifactRecord |
| 1.3 | `predictions.py` | Add `model_artifact_id` field to predictions schema |
| 1.4 | `base_model.py` | In predict mode, load by artifact_id from provider |
| 1.5 | `artifact_loader.py` | Implement `get_step_binaries_by_id()` |

**Deliverable**: Q5_predict.py passes with custom model names.

### Phase 2: Execution Trace Recording

**Goal**: Record the exact path through pipeline.

| Task | File(s) | Description |
|------|---------|-------------|
| 2.1 | `nirs4all/pipeline/trace/` | Create `ExecutionTrace`, `StepArtifacts` dataclasses |
| 2.2 | `execution_context.py` | Add `execution_history` to track steps |
| 2.3 | `step_runner.py` | Record step execution in trace |
| 2.4 | `base_model.py`, controllers | Record artifact_id when creating artifacts |
| 2.5 | `manifest_manager.py` | Save/load execution_traces in manifest |
| 2.6 | `prediction_assembler.py` | Store `trace_id` in predictions |

**Deliverable**: Each prediction has a trace_id referencing its full path.

### Phase 3: Prediction Resolver

**Goal**: Support multiple prediction sources.

| Task | File(s) | Description |
|------|---------|-------------|
| 3.1 | `nirs4all/pipeline/resolver.py` | Create `PredictionResolver` class |
| 3.2 | resolver | `_resolve_from_prediction()` |
| 3.3 | resolver | `_resolve_from_folder()` |
| 3.4 | resolver | `_resolve_from_run()` |
| 3.5 | resolver | `_resolve_from_artifact_id()` |
| 3.6 | resolver | `_resolve_from_bundle()` |
| 3.7 | `context.py` | Create `ArtifactProvider` interface |

**Deliverable**: `runner.predict()` accepts any source type.

### Phase 4: Controller Integration

**Goal**: Controllers use ArtifactProvider in predict mode.

| Task | File(s) | Description |
|------|---------|-------------|
| 4.1 | `RuntimeContext` | Add `artifact_provider` field |
| 4.2 | `base_model.py` | Use `artifact_provider.get_artifact(step_index)` |
| 4.3 | `transformer.py` | Use artifact_provider |
| 4.4 | `y_transformer.py` | Use artifact_provider |
| 4.5 | Other controllers | Update to use artifact_provider (optional, graceful fallback) |

**Deliverable**: Prediction works with artifact injection, controller-agnostic.

### Phase 5: Minimal Pipeline Execution

**Goal**: Execute only needed steps.

| Task | File(s) | Description |
|------|---------|-------------|
| 5.1 | `trace/extractor.py` | Create `TraceBasedExtractor` |
| 5.2 | `predictor.py` | Use minimal pipeline from trace |
| 5.3 | executor | Handle minimal pipeline execution |

**Deliverable**: Prediction runs only required steps.

### Phase 6: Bundle Export

**Goal**: Export standalone prediction packages.

| Task | File(s) | Description |
|------|---------|-------------|
| 6.1 | `nirs4all/pipeline/bundle/` | Create bundle module |
| 6.2 | bundle | `.n4a` format (ZIP with artifacts) |
| 6.3 | bundle | `.n4a.py` format (single portable file) |
| 6.4 | bundle | `runner.export()` API |
| 6.5 | bundle | Load and predict from bundle |

**Deliverable**: `runner.export(prediction, "model.n4a")` creates standalone bundle.

### Phase 7: Retrain & Transfer

**Goal**: Retrain/transfer from existing predictions.

| Task | File(s) | Description |
|------|---------|-------------|
| 7.1 | `runner.py` | Add `retrain()` method |
| 7.2 | executor | Per-step mode control |
| 7.3 | executor | Mode='transfer' implementation |
| 7.4 | executor | Mode='finetune' implementation |

**Deliverable**: `runner.retrain(prediction, new_data, mode='transfer')` works.

### Phase 8: Polish & Documentation

| Task | Description |
|------|-------------|
| 8.1 | Update all examples |
| 8.2 | Add tests for new features |
| 8.3 | Update user documentation |
| 8.4 | Migration guide for old predictions |

---

## 10. Data Model Changes Summary

### 10.1 New Fields in Prediction Record

```python
prediction = {
    # Existing fields (unchanged)
    'id': str,
    'dataset_name': str,
    'pipeline_uid': str,
    'model_name': str,
    'model_classname': str,
    'fold_id': Union[int, str],  # int or 'w_avg', 'avg'
    'branch_id': Optional[int],
    'y_pred': np.ndarray,
    'y_true': np.ndarray,
    # ... scores, metrics, etc.

    # NEW required fields
    'model_artifact_id': str,           # Primary model artifact ID
    'trace_id': str,                    # Reference to ExecutionTrace

    # NEW optional fields (for CV ensemble)
    'fold_artifact_ids': Dict[int, str],  # {fold_id: artifact_id}
    'fold_weights': Dict[int, float],     # {fold_id: weight}
}
```

### 10.2 New Fields in ArtifactRecord

```python
ArtifactRecord = {
    # Existing fields (unchanged)
    'artifact_id': str,
    'class_name': str,
    'artifact_type': str,
    'content_hash': str,
    'filename': str,
    'step_index': int,
    'branch_path': List[int],
    'fold_id': Optional[int],
    'depends_on': List[str],

    # NEW fields
    'custom_name': Optional[str],       # User-provided name
    'step_config': dict,                # Original step config
}
```

### 10.3 New Manifest Section

```yaml
# runs/<run_id>/<pipeline_uid>/manifest.yaml

# Existing sections...
artifacts:
  items: [...]

# NEW section
execution_traces:
  trace_xyz789:
    pipeline_uid: "abc123def456"
    created_at: "2024-12-13T10:30:00"

    # Steps in this trace (controller-agnostic!)
    steps:
      - step_index: 0
        artifact_ids: ["0001:0:all"]
      - step_index: 1
        artifact_ids: ["0001:1:all"]
      - step_index: 2
        branch_index: 0
        artifact_ids: ["0001:2:0:SNV", "0001:2:0:SG"]
      - step_index: 3
        artifact_ids: []  # Splitter, no artifact
      - step_index: 4
        primary_artifact_id: "0001:4:all"
        fold_artifact_ids:
          0: "0001:4:0"
          1: "0001:4:1"

    # Context
    branch_path: [0]
    fold_id: "w_avg"
    fold_weights: {0: 0.52, 1: 0.48}
```

---

## 11. API Summary

### 11.1 User-Facing API

```python
from nirs4all.pipeline import PipelineRunner
from nirs4all.data import DatasetConfigs

runner = PipelineRunner()

# ============================================================
# PREDICT (from any source)
# ============================================================

# From prediction dict (most common)
y_pred, preds = runner.predict(best_prediction, new_data)

# From folder path
y_pred, preds = runner.predict("runs/0001_wheat_abc123/", new_data)

# From Run object (best model)
run = Run.load("runs/0001_wheat_abc123/")
y_pred, preds = runner.predict(run, new_data)

# From artifact ID
y_pred, preds = runner.predict("0001:4:all", new_data)

# From bundle
y_pred, preds = runner.predict("exports/wheat_model.n4a", new_data)

# ============================================================
# EXTRACT (get minimal pipeline for inspection/modification)
# ============================================================

# Extract minimal pipeline
minimal = runner.extract(best_prediction)
print(minimal.steps)
print(minimal.artifacts)
print(minimal.trace)

# Modify and run
minimal.steps[-1] = {"model": RandomForestRegressor()}
y_pred, preds = runner.run(minimal, new_data)

# ============================================================
# EXPORT (create standalone bundle)
# ============================================================

# Export to .n4a bundle
runner.export(best_prediction, "exports/wheat_model.n4a")

# Export to portable Python script
runner.export(best_prediction, "exports/wheat_model.n4a.py", format='n4a.py')

# ============================================================
# RETRAIN (train new model using existing pipeline)
# ============================================================

# Full retrain
y_pred, preds = runner.retrain(best_prediction, new_data, mode='full')

# Transfer (use existing preprocessing, train new model)
y_pred, preds = runner.retrain(best_prediction, new_data, mode='transfer')

# Fine-tune (continue training existing model)
y_pred, preds = runner.retrain(best_prediction, new_data, mode='finetune', epochs=10)
```

### 11.2 Internal API

```python
# PredictionResolver
resolver = PredictionResolver(workspace)
resolved = resolver.resolve(source)  # Returns ResolvedPrediction

# ArtifactProvider
provider = ArtifactProvider(artifact_map)
artifact = provider.get_artifact(step_index)

# ExecutionTrace
trace = ExecutionTrace.from_manifest(manifest, trace_id)
minimal_pipeline = trace.get_steps()
artifact_ids = trace.get_artifact_ids()

# TraceBasedExtractor
extractor = TraceBasedExtractor()
minimal_pipeline = extractor.extract_minimal_pipeline(trace, full_pipeline)

# PredictionBundleGenerator
generator = PredictionBundleGenerator(workspace)
bundle_path = generator.export(source, output_path, format='n4a')
```

---

## 12. Backwards Compatibility & Migration

### 12.1 Graceful Degradation

Old predictions without the new fields will still work:

```python
class PredictionResolver:

    def _resolve_from_prediction(self, pred: dict, workspace: Path) -> ResolvedPrediction:
        # NEW path: use trace_id if available
        if 'trace_id' in pred:
            return self._resolve_from_trace(pred, workspace)

        # LEGACY path: reconstruct from prediction metadata
        warnings.warn(
            f"Prediction {pred['id']} uses legacy format. "
            "Consider re-training for better reproducibility.",
            DeprecationWarning
        )
        return self._legacy_resolve(pred, workspace)

    def _legacy_resolve(self, pred: dict, workspace: Path) -> ResolvedPrediction:
        """Fallback for old predictions without trace_id."""
        # Use existing logic: load full pipeline, match by name
        # This preserves current behavior
        ...
```

### 12.2 Migration Path

Users can migrate old predictions by re-running:

```python
# Re-train to get new format predictions
new_predictions, _ = runner.run(old_pipeline_config, dataset)

# Or upgrade in-place (future utility)
runner.upgrade_predictions("runs/old_run/")
```

### 12.3 Version Markers

New manifests include version information:

```yaml
manifest_version: "2.0"
nirs4all_version: "0.x.y"
features:
  - execution_traces
  - artifact_custom_names
  - prediction_artifact_ids
```

---

## 13. Testing Strategy

### 13.1 Unit Tests

| Test | Description |
|------|-------------|
| `test_execution_trace_recording` | Verify traces are recorded during training |
| `test_artifact_id_in_prediction` | Verify artifact IDs are stored in predictions |
| `test_artifact_provider` | Verify ArtifactProvider interface works |
| `test_prediction_resolver_from_dict` | Resolve from prediction dict |
| `test_prediction_resolver_from_folder` | Resolve from folder path |
| `test_prediction_resolver_from_artifact_id` | Resolve from artifact ID |
| `test_bundle_export_n4a` | Export to .n4a format |
| `test_bundle_export_portable` | Export to .n4a.py format |
| `test_bundle_load_and_predict` | Load bundle and predict |

### 13.2 Integration Tests

| Test | Description |
|------|-------------|
| `test_q5_predict_with_custom_name` | Original failing case (Q5_predict.py) |
| `test_predict_from_folder` | Predict using folder path |
| `test_predict_from_run` | Predict using Run object |
| `test_predict_multibranch` | Branching pipeline prediction |
| `test_predict_stacking` | Meta-model prediction |
| `test_predict_cv_ensemble` | Cross-validation ensemble |
| `test_predict_with_concat_transform` | concat_transform controller |
| `test_predict_with_custom_controller` | User-defined controller |
| `test_retrain_transfer` | Transfer learning |
| `test_retrain_finetune` | Fine-tuning |

### 13.3 Regression Tests

- All existing examples must pass unchanged
- Predictions from old format must work (with deprecation warning)
- Bundle exports must be reproducible

---

## 14. Summary

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Controller-agnostic** | Reuse existing controllers, don't hardcode operator types |
| **Execution Trace** | Record exact path without knowing controller internals |
| **ArtifactProvider** | Clean interface for artifact injection |
| **Multiple sources** | Flexibility for different use cases |
| **Portable bundles** | Deployment without nirs4all dependency |

### Benefits

| Benefit | Description |
|---------|-------------|
| **Deterministic** | artifact_id + trace_id = exact reproducibility |
| **Extensible** | Works with any controller (existing or custom) |
| **Efficient** | Minimal pipeline execution |
| **Portable** | Standalone bundles for deployment |
| **Future-proof** | Same pattern for predict, retrain, transfer |

### Immediate Fix (Phase 1)

The Q5_predict.py issue can be fixed by:
1. Storing `model_artifact_id` in prediction during training
2. Creating `ArtifactProvider` with artifacts loaded by step_index
3. Controllers use `artifact_provider.get_artifact(step_index)` instead of name matching

This preserves all existing controller logic while fixing the name ambiguity.

---

## 15. Open Questions

1. **Bundle size limits**: For portable `.n4a.py`, should we have a size limit (artifacts in base64 can be large)?

2. **ONNX export**: Is this a priority? Requires significant effort for preprocessing conversion.

3. **Custom controller discovery**: How should exported bundles handle user-defined controllers that may not be available at prediction time?

4. **Trace granularity**: Should we track individual transform applications within feature_augmentation, or treat the whole step as atomic?

5. **Parallel execution**: When minimal pipeline has independent branches, should we parallelize?

---

*Document Version: 2.0*
*Last Updated: December 13, 2024*
*Status: Design Review*
