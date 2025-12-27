# Architecture Overview

This document provides a high-level overview of the nirs4all pipeline architecture for developers who want to understand the internals or contribute to the project.

## Architecture Philosophy

The pipeline module is designed around a **layered architecture** with **separation of concerns**:

1. **Orchestration**: Managing multiple datasets and pipeline configurations
2. **Execution**: Running a specific sequence of steps on a specific dataset
3. **Step Logic**: The actual implementation of a pipeline step (model training, preprocessing, etc.)

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        PipelineRunner                           │
│                   (Public API / Facade)                         │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PipelineOrchestrator                        │
│              (Manages datasets × configurations)                 │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       PipelineExecutor                           │
│                (Executes steps sequentially)                     │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         StepRunner                               │
│              (Routes steps to controllers)                       │
└───────────────┬─────────────────────────────────┬───────────────┘
                │                                 │
                ▼                                 ▼
        ┌───────────────┐               ┌─────────────────┐
        │   StepParser  │               │ ControllerRouter│
        │  (Normalize)  │               │   (Dispatch)    │
        └───────────────┘               └────────┬────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │  CONTROLLER_REGISTRY   │
                                    │    (Priority-based)    │
                                    └────────────────────────┘
```

## Key Components

### 1. PipelineRunner

**Location**: `nirs4all/pipeline/runner.py`

**Role**: The public entry point (Facade pattern)

**Responsibilities**:
- Provides simple API for users (`run()`, `predict()`, `export()`, `retrain()`)
- Initializes the environment (workspace, logging)
- Delegates work to the Orchestrator

```python
from nirs4all.pipeline import PipelineRunner

runner = PipelineRunner(save_artifacts=True, verbose=1)
predictions, per_dataset = runner.run(pipeline, dataset)
```

### 2. PipelineOrchestrator

**Location**: `nirs4all/pipeline/execution/orchestrator.py`

**Role**: The high-level manager

**Responsibilities**:
- Iterates over all provided **Datasets**
- Iterates over all provided **Pipeline Configurations**
- Manages global results (aggregating predictions across runs)
- Instantiates a `PipelineExecutor` for each (Dataset, Pipeline) pair

### 3. PipelineExecutor

**Location**: `nirs4all/pipeline/execution/executor.py`

**Role**: The sequence runner

**Responsibilities**:
- Executes a list of steps sequentially
- Manages the **ExecutionContext** (state propagation)
- Handles artifact management (saving models, logs) for a single run
- Catches errors and handles the "continue on error" logic

### 4. StepRunner

**Location**: `nirs4all/pipeline/steps/step_runner.py`

**Role**: The unit executor

**Responsibilities**:
- **Parses** the raw step definition (dict, string, object) using `StepParser`
- **Routes** the step to the appropriate Controller using `ControllerRouter`
- Executes the Controller

### 5. StepParser

**Location**: `nirs4all/pipeline/steps/parser.py`

**Role**: Step configuration parser

**Responsibilities**:
- Normalizes different step syntaxes (dict, string, instance, list) into a canonical `ParsedStep` format
- Extracts keywords from step configurations
- Identifies step types (workflow, serialized, subpipeline, direct)
- Deserializes operators when needed

### 6. ControllerRouter

**Location**: `nirs4all/pipeline/steps/router.py`

**Role**: Controller selection

**Responsibilities**:
- Matches parsed steps to appropriate controllers using priority-based selection
- Queries each controller's `matches()` method
- Returns the highest-priority matching controller

### 7. Controllers

**Location**: `nirs4all/controllers/`

**Role**: The business logic

**Responsibilities**:
- Implements the actual logic for a step (e.g., `ModelController`, `PreprocessingController`)
- Interacts with the `SpectroDataset`
- Updates the `ExecutionContext`
- Returns artifacts (files, objects) to be saved

See {doc}`controllers` for details on the controller system.

## Data Flow

The data flow relies on two main objects passed through the layers:

### SpectroDataset

The data container holding:
- `X`: Feature matrix (spectral data)
- `y`: Target values
- `metadata`: Sample metadata
- `fold_indices`: Cross-validation assignments

It is mutable but typically modified via internal state updates managed by controllers.

### ExecutionContext

A composite object containing:
- **DataSelector**: Immutable configuration for how to read data (e.g., "train" partition)
- **PipelineState**: Mutable state tracking (e.g., current Y-transformation)
- **StepMetadata**: Ephemeral flags for communication between steps
- **Custom**: Controller-specific data (e.g., branch contexts)

## Directory Structure

```
nirs4all/
├── pipeline/                    # Pipeline execution engine
│   ├── runner.py                # PipelineRunner (public API)
│   ├── config/                  # Configuration handling
│   │   ├── config.py            # PipelineConfigs
│   │   └── context.py           # ExecutionContext, RuntimeContext
│   ├── execution/               # Execution infrastructure
│   │   ├── orchestrator.py      # PipelineOrchestrator
│   │   └── executor.py          # PipelineExecutor
│   ├── steps/                   # Step processing
│   │   ├── parser.py            # StepParser
│   │   ├── router.py            # ControllerRouter
│   │   └── step_runner.py       # StepRunner
│   ├── bundle/                  # Export/import bundles
│   │   ├── generator.py         # BundleGenerator
│   │   └── loader.py            # BundleLoader
│   └── storage/                 # Artifact management
│       └── artifacts/           # Artifact registry, loader
├── controllers/                 # Step handlers
│   ├── registry.py              # @register_controller
│   ├── controller.py            # OperatorController base
│   ├── transforms/              # TransformerMixin controllers
│   ├── models/                  # Model training controllers
│   ├── data/                    # Data manipulation (branch, merge)
│   └── splitters/               # Cross-validation controllers
├── data/                        # Data handling
│   ├── config.py                # DatasetConfigs
│   ├── dataset.py               # SpectroDataset
│   └── predictions.py           # Predictions container
└── operators/                   # Pipeline operators
    ├── transforms/              # NIRS-specific transformers
    ├── augmentation/            # Data augmentation
    ├── models/                  # Pre-built models
    └── splitters/               # Splitting algorithms
```

## Common Patterns

### Registry Pattern

Controllers are discovered automatically via the `@register_controller` decorator:

```python
from nirs4all.controllers.registry import register_controller
from nirs4all.controllers.controller import OperatorController

@register_controller
class MyController(OperatorController):
    priority = 50
    # ...
```

### Priority Pattern

Controllers compete for steps based on priority. Lower numbers = higher priority:

| Priority | Use Case |
|----------|----------|
| 1-10 | Special/high-priority controllers |
| 20-50 | Specific operator controllers |
| 80-100 | Generic fallback controllers |
| 1000+ | Catch-all controllers |

### Facade Pattern

`PipelineRunner` hides complexity from users, providing a simple API:

```python
runner = PipelineRunner()
predictions, _ = runner.run(pipeline, dataset)
y_pred, _ = runner.predict(source, new_data)
```

### Context Object Pattern

`ExecutionContext` encapsulates state and is immutably updated through steps. Controllers receive and return context objects rather than modifying global state.

### Strategy Pattern

Controllers implement different strategies for handling different operator types. The router selects the appropriate controller at runtime.

## Extension Points

The system provides multiple extension points:

1. **Custom Controllers**: Add new controllers with `@register_controller`
2. **Custom Keywords**: Use any non-reserved keyword in step dictionaries
3. **Custom Operators**: Any Python callable or class can be an operator
4. **Custom Context Data**: Use `context.custom` dict for controller-specific data
5. **Custom Artifacts**: Controllers can return any serializable artifacts

## See Also

- {doc}`controllers` - Controller registry system
- {doc}`/reference/pipeline_syntax` - Complete pipeline syntax
- {doc}`/user_guide/pipelines/branching` - User-facing branching documentation
- {doc}`/examples/index` - Working examples including advanced usage

**Source files:**
- `nirs4all/pipeline/runner.py` - PipelineRunner implementation
- `nirs4all/pipeline/execution/orchestrator.py` - Orchestrator implementation
- `nirs4all/controllers/` - Controller implementations
