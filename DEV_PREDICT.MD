# Prediction Mode Implementation Design

## Overview

This document outlines the design for adding prediction capabilities to the NIRS4All pipeline system. The solution allows loading a previously trained pipeline and running it in prediction-only mode, where transformers only transform and models only predict, using saved fitted operators.

## Core Design Principles

1. **Controller-Based Filtering**: Each controller decides if it should execute during prediction
2. **Enhanced Pipeline Metadata**: Store step-to-binary mapping in pipeline.json
3. **Binary Loading System**: Pre-load and inject binaries into controllers
4. **Minimal Code Changes**: Leverage existing infrastructure

## Current Architecture Analysis

### Sequential ID Problem
Current system uses `runner.next_op()` for binary IDs, creating sequential counters that break during prediction when steps are omitted.

**Example**:
- Training: Transformer (step 1) → Chart (step 2) → Model (step 3)
- Prediction: Transformer (step 1) → Model (step 3), but Chart skipped

### Solution Approach
Use existing step numbering (`step_number`, `substep_number`) and enhance pipeline.json to include binary mappings.

## Proposed Solution Architecture

### 1. Enhanced Pipeline Configuration Storage

**Location**: `PipelineRunner.run()` after line 68

**Current**:
```python
self.saver.save_json("pipeline.json", config.serializable_steps())
```

**Enhanced**:
```python
# Track step-to-binary mapping during execution
self.step_binaries = {}  # {step_id: [binary_filenames]}

# After pipeline completion, save enhanced configuration
enhanced_config = {
    "steps": config.serializable_steps(),
    "execution_metadata": {
        "step_binaries": self.step_binaries,
        "created_at": datetime.now().isoformat(),
        "pipeline_version": "1.0"
    }
}
self.saver.save_json("pipeline.json", enhanced_config)
```

### 2. Controller Interface Extensions

**Location**: `nirs4all/controllers/controller.py`

**New Abstract Methods**:
```python
@abstractmethod
def supports_prediction_mode(cls) -> bool:
    """Whether this controller should execute during prediction"""

@abstractmethod
def execute(self, step, operator, dataset, context, runner, source=-1,
           mode="train", loaded_binaries=None):
    """Execute with mode awareness and optional pre-loaded binaries"""
```

### 3. Controller-Specific Implementation

#### A. TransformerMixinController
- `supports_prediction_mode() = True`
- In prediction mode: Skip fit(), use loaded binary for transform()

#### B. BaseModelController
- `supports_prediction_mode() = True`
- In prediction mode: Skip training, use loaded binary for predict()

#### C. Chart Controllers
- `supports_prediction_mode() = False`
- Automatically skipped during prediction

#### D. Data Splitters
- `supports_prediction_mode() = False`
- CV/splitting not needed for prediction

### 4. Prediction Runner Implementation

**Location**: New static method in `PipelineRunner`

```python
@staticmethod
def predict(path: Union[str, Path], dataset: SpectroDataset,
           verbose: int = 0) -> Tuple[SpectroDataset, np.ndarray]:
    """
    Load a saved pipeline and run it in prediction mode.

    Args:
        path: Path to saved pipeline directory
        dataset: Dataset to make predictions on
        verbose: Verbosity level

    Returns:
        Updated dataset and final predictions
    """

    # 1. Load pipeline configuration with metadata
    pipeline_data = json.load(open(path / "pipeline.json"))
    step_binaries = pipeline_data["execution_metadata"]["step_binaries"]

    # 2. Pre-load all binaries into memory
    binary_loader = BinaryLoader(path, step_binaries)

    # 3. Create prediction runner
    runner = PipelineRunner(results_path=None, mode="predict")
    runner.binary_loader = binary_loader

    # 4. Run pipeline in prediction mode
    config = PipelineConfig(pipeline_data["steps"])
    return runner.run(config, dataset, mode="predict")
```

### 5. Binary Loading System

**Location**: New file `nirs4all/pipeline/binary_loader.py`

```python
class BinaryLoader:
    """Manages loading and caching of saved pipeline binaries"""

    def __init__(self, simulation_path: Path, step_binaries: Dict[str, List[str]]):
        self.simulation_path = simulation_path
        self.step_binaries = step_binaries
        self._cache = {}

    def get_binaries_for_step(self, step_number: int, substep_number: int) -> List[Any]:
        """Load binaries for specific step"""
        step_id = f"{step_number}_{substep_number}"
        if step_id not in self._cache:
            self._cache[step_id] = self._load_step_binaries(step_id)
        return self._cache[step_id]
```

## Implementation Strategy

### Phase 1: Metadata Tracking

**A. Modify PipelineRunner._execute_controller()**
```python
def _execute_controller(self, ...):
    # Existing execution
    context, binaries = controller.execute(...)

    # Track binaries for this step
    if binaries and hasattr(self, 'step_binaries'):
        step_id = f"{self.step_number}_{self.substep_number}"
        self.step_binaries[step_id] = [binary[0] for binary in binaries]

    self.saver.save_binaries(self.step_number, self.substep_number, binaries)
    return context
```

**B. Enhance Pipeline Configuration Saving**
- Initialize `self.step_binaries = {}` in `__init__`
- Save enhanced configuration in `run()` method

### Phase 2: Controller Interface Updates

**A. Base Controller Class**
```python
class OperatorController(ABC):
    @classmethod
    @abstractmethod
    def supports_prediction_mode(cls) -> bool:
        """Whether controller should execute during prediction"""
        return False

    @abstractmethod
    def execute(self, step, operator, dataset, context, runner, source=-1,
               mode="train", loaded_binaries=None):
        """Execute with mode and binary awareness"""
        pass
```

**B. Controller-Specific Implementations**
- Transform controllers: Load fitted transformers from binaries
- Model controllers: Load trained models from binaries
- Chart controllers: Return early if mode == "predict"

### Phase 3: Prediction Execution Flow

**A. Step Filtering During Execution**
```python
def run_step(self, step, dataset, context, *, mode="train", **kwargs):
    # Select controller
    controller = self._select_controller(step, operator, keyword)

    # Check if controller supports prediction mode
    if mode == "predict" and not controller.supports_prediction_mode():
        print(f"🔄 Skipping step {self.step_number} in prediction mode")
        return context

    # Load binaries if in prediction mode
    loaded_binaries = None
    if mode == "predict" and hasattr(self, 'binary_loader'):
        loaded_binaries = self.binary_loader.get_binaries_for_step(
            self.step_number, self.substep_number
        )

    # Execute with mode awareness
    return self._execute_controller(controller, step, operator, dataset,
                                   context, source, mode, loaded_binaries)
```

## Key Design Benefits

1. **Controller Autonomy**: Each controller decides its prediction behavior
2. **Clean Separation**: Training and prediction modes clearly distinguished
3. **Minimal Changes**: Leverages existing step numbering and binary saving
4. **Maintainable**: Uses established patterns in the codebase
5. **Professional**: Robust error handling and validation
6. **Efficient**: Pre-loads binaries once, caches for reuse

## File Modifications Summary

### New Files
- `nirs4all/pipeline/binary_loader.py` - Binary loading and caching system

### Modified Files
- `nirs4all/pipeline/runner.py` - Add prediction mode and metadata tracking
- `nirs4all/controllers/controller.py` - Enhanced interface with prediction support
- `nirs4all/controllers/sklearn/op_transformermixin.py` - Prediction mode implementation
- `nirs4all/controllers/models/base_model_controller.py` - Prediction mode implementation
- `nirs4all/controllers/chart/*.py` - Skip execution in prediction mode

## Usage Example

```python
# Training
runner = PipelineRunner(results_path="./results")
dataset, history, pipeline = runner.run(config, dataset)

# Prediction
predictions_dataset = PipelineRunner.predict("./results/dataset/pipeline", new_dataset)
```

This design provides a clean, maintainable solution that respects the existing architecture while adding powerful prediction capabilities.