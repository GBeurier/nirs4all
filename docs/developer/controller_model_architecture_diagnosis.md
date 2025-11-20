# Model Controller Architecture Diagnosis

**Date:** 2025-10-30
**Scope:** Analysis of `model_builder.py`, `model_utils.py`, and model controller architecture
**Focus:** Redundancies, misplaced responsibilities, and task type management issues

---

## Quick Reference

### Critical Issues Identified:
1. ⚠️ **120+ lines of duplicate metric calculation** (model_utils.py vs evaluator.py)
2. 🔴 **Task type detected 3-5 times per run** across 5 architectural layers
3. ⚠️ **Model builder in wrong module** (utils/ instead of controllers/)
4. ⚠️ **Mixed responsibilities** in model_utils.py

### Recommended Action Order:
1. **Phase 1** (1-2h): Remove metric calculation redundancy → Use evaluator.py everywhere
2. **Phase 2** (6h): Centralize task type → Single detection, immutable storage
3. **Phase 3** (2-3h): Move model_builder.py → controllers/models/factory.py
4. **Phase 4** (2-3h): Split model_utils.py → controllers/utilities.py + data/ensemble_utils.py
5. **Phase 5** (4-6h, optional): Decompose ModelBuilderFactory into framework-specific classes

### Key Metrics:
- **Code duplication removed:** ~120 lines
- **Performance improvement:** 3-5 task detections → 1 detection
- **Files relocated:** 2 (model_builder.py, model_utils.py split)
- **Estimated total effort:** 15-20 hours
- **Risk level:** Medium (with mitigation strategies in place)

---
## Executive Summary

After analyzing the model controller architecture, I've identified significant **redundancies**, **misplaced responsibilities**, and **organizational issues** with `model_builder.py` and `model_utils.py`. These modules should indeed be relocated and refactored.

### Key Findings:
1. **Redundant metric calculation** between `model_utils.py` and `evaluator.py` (120+ lines of duplicate code)
2. **Model building responsibility** scattered across utils and controllers
3. **Poor cohesion** - utilities that belong with controllers are in separate modules
4. **Inconsistent abstractions** - mixing framework detection, model instantiation, and parameter handling
5. **CRITICAL: Task Type Detection Chaos** - Task type is detected/stored/re-detected **3-5 times per pipeline run** across **5 different architectural layers**

### Impact Summary:

| Issue | Current State | Impact | Priority |
|-------|---------------|--------|----------|
| Metric Redundancy | 120+ duplicate lines in 2 modules | Maintenance burden, divergence risk | HIGH |
| Model Builder Location | In utils/, used only by controllers | Poor cohesion, circular deps risk | HIGH |
| Task Type Detection | 3-5 detections per run, 5 storage locations | Performance loss, inconsistency risk | **CRITICAL** |
| Model Utils Mixing | Controller + prediction + redundant logic | Unclear responsibilities | HIGH |
| Framework Detection | String matching, brittle | Fails on custom models | MEDIUM |

### Task Type Detection Flow (Current - BROKEN):
```
Pipeline Execution
│
├─ 1. Data Load
│   └─ targets.add_targets() → ModelUtils.detect_task_type() [DETECTION #1]
│       ├─ Stores in targets._task_type
│       └─ Threshold: 0.05
│
├─ 2. Data Processing
│   └─ targets.add_processed_targets() → ModelUtils.detect_task_type() [DETECTION #2]
│       ├─ Re-detects and updates _task_type (LEGITIMATE - processing can change task)
│       ├─ BUT: No per-processing storage, loses history
│       └─ Threshold: 0.05
│
├─ 3. Model Training (TensorFlow)
│   └─ tensorflow_model._train_model() → ModelUtils.detect_task_type() [DETECTION #3]
│       ├─ Ignores dataset.task_type
│       └─ Threshold: 0.05
│
├─ 4. Score Calculation (on error)
│   └─ ModelUtils.calculate_scores() → ModelUtils.detect_task_type() [DETECTION #4]
│       ├─ Fallback when classification fails
│       └─ Threshold: 0.01 (DIFFERENT!)
│
└─ 5. Visualization
    └─ TabReportManager._detect_task_type_from_entry() [DETECTION #5]
        ├─ Re-detects from y_true in predictions
        └─ Threshold: 0.05

Result: 5 detections, 2 different thresholds, no single source of truth
```

### Task Type Detection Flow (Proposed - UPDATED):
```
Pipeline Execution
│
├─ 1. Data Load (INITIAL DETECTION)
│   └─ targets.add_targets() → ModelUtils.detect_task_type()
│       ├─ Stores in targets._task_type = REGRESSION
│       ├─ Stores in targets._task_type_by_processing['numeric'] = REGRESSION
│       ├─ Exposed via dataset.task_type
│       └─ Threshold: 0.05
│
├─ 2. Data Processing (CONDITIONAL RE-DETECTION)
│   └─ targets.add_processed_targets('binned', ...) → ModelUtils.detect_task_type()
│       ├─ Detects new task_type = CLASSIFICATION (binning changed it!)
│       ├─ Stores in targets._task_type_by_processing['binned'] = CLASSIFICATION
│       ├─ Updates targets._task_type = CLASSIFICATION
│       └─ Logs: "⚠️ Task type changed: REGRESSION → CLASSIFICATION (processing 'binned')"
│
    ↓ Controllers TRUST dataset.task_type (which reflects current processing)

├─ 3. Controllers → Read dataset.task_type (no detection)
│   └─ task_type = CLASSIFICATION (reflects 'binned' processing)
├─ 4. Training → Use dataset.task_type (no detection)
│   └─ Compiles with classification loss/metrics
├─ 5. Predictions → Store dataset.task_type + target_processing as metadata
│   └─ {'task_type': 'classification', 'target_processing': 'binned'}
└─ 6. Visualization → Read from metadata (no detection)

Result: 1-2 detections (initial + per processing change), tracked per processing,
        controllers trust dataset, future-proof for processing-specific models
```

**Key Improvements:**
- ✅ Detections only happen in Targets layer (centralized)
- ✅ Controllers never detect (trust dataset)
- ✅ Processing changes are tracked and logged
- ✅ Per-processing task_type stored for future use
- ✅ Visualization reads from metadata (no detection)

---

## 1. Redundancy Analysis

### 1.1 Metric Calculation Duplication

**Problem:** Two separate modules calculate the same metrics using identical sklearn functions.

#### `model_utils.py` - `ModelUtils.calculate_scores()` (lines 226-344)
```python
def calculate_scores(y_true, y_pred, task_type, metrics=None):
    # Manual implementation using sklearn metrics
    if task_type == TaskType.REGRESSION:
        scores["mse"] = mean_squared_error(y_true, y_pred)
        scores["mae"] = mean_absolute_error(y_true, y_pred)
        scores["r2"] = r2_score(y_true, y_pred)
        scores["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        scores["accuracy"] = accuracy_score(y_true_class, y_pred_class)
        scores["f1"] = f1_score(y_true_class, y_pred_class, average=average)
        # ... etc
```

#### `evaluator.py` - `eval()` and `eval_multi()` (lines 50-350)
```python
def eval(y_true, y_pred, metric):
    # Same metrics, same sklearn functions
    if metric in ['mse', 'mean_squared_error']:
        return mean_squared_error(y_true, y_pred)
    elif metric in ['rmse', 'root_mean_squared_error']:
        return np.sqrt(mean_squared_error(y_true, y_pred))
    # ... identical calculations

def eval_multi(y_true, y_pred, task_type):
    # Returns dict of all metrics for task type
    # Same functionality as ModelUtils.calculate_scores()
```

**Impact:**
- **120+ lines** of duplicate code
- Two sources of truth for metric calculations
- Maintenance burden (bug fixes need to be applied twice)
- Risk of divergence in behavior

**Current Usage:**
- `ModelUtils.calculate_scores()`: Used in `base_model.py` (`_calculate_and_print_scores()`)
- `evaluator.eval()`: Used in `score_calculator.py`, `predictions.py`, visualization modules

---

### 1.2 Task Type Detection Scattered

**Problem:** Task type detection logic exists in multiple places with slight variations.

- `ModelUtils.detect_task_type()` in `model_utils.py`
- `BaseModelController._detect_task_type()` in `base_model.py` (delegates to ModelUtils)
- Implicit detection in `evaluator.py` based on metric availability

**Inconsistency:** Different thresholds and logic could lead to different classifications.

---

### 1.3 Task Type Management: The Hidden Redundancy Problem

**CRITICAL FINDING:** Task type is detected, stored, and re-detected across **5 different layers** of the architecture, creating a cascading redundancy that affects the entire pipeline.

#### Where Task Type Lives:

1. **Targets Block** (`data/targets.py`):
   - Stores `_task_type` as instance variable
   - Auto-detects on `add_targets()` (line 271)
   - Re-detects on `add_processed_targets()` (line 344)
   - **Problem:** Mutable state that changes based on processing

2. **Dataset** (`data/dataset.py`):
   - Exposes `dataset.task_type` property (line 300)
   - Provides `dataset.set_task_type()` (line 304)
   - Provides `dataset.is_classification` / `is_regression` (lines 328-337)
   - **Problem:** Delegates to targets, but controllers also detect independently

3. **Model Controllers** (`controllers/models/base_model.py`):
   - Detects task type in `_detect_task_type()` (line 674-682)
   - Checks `dataset.task_type` in multiple places (lines 208, 264, 446, 450, 578, 851, 854, 863, 959, 960)
   - Re-detects in `get_xy()` for classification vs regression logic (line 208)
   - **Problem:** Sometimes trusts dataset, sometimes re-detects from y values

4. **TensorFlow Controller** (`tensorflow_model.py`):
   - Re-detects task type in `_train_model()` (line 253): `task_type = ModelUtils.detect_task_type(y_train)`
   - Uses it for loss/metrics configuration
   - **Problem:** Ignores dataset.task_type, always re-detects

5. **Visualization** (`visualization/reports.py`):
   - Re-detects from prediction data (line 75): `ModelUtils.detect_task_type(y_true)`
   - Has custom `_detect_task_type_from_entry()` method (line 70)
   - **Problem:** Doesn't trust stored task_type in prediction metadata

6. **Predictions Storage** (`data/predictions.py`):
   - Stores task_type as string in prediction metadata (lines 125, 189, 221, 259)
   - But uses function parameter default `task_type="regression"` (line 125)
   - **Problem:** Must be passed explicitly, not derived from dataset

#### Concrete Redundancy Examples:

**Example 1: Training Flow**
```python
# 1. Targets auto-detect when data is added
targets.add_targets(y_train)  # → ModelUtils.detect_task_type() called
dataset._task_type = TaskType.REGRESSION

# 2. Controller checks dataset task_type
if dataset.task_type and dataset.task_type.is_classification:  # Line 208

# 3. TensorFlow controller RE-DETECTS anyway
task_type = ModelUtils.detect_task_type(y_train)  # Line 253 tensorflow_model.py

# 4. Base controller RE-DETECTS for validation
actual_task_type = ModelUtils.detect_task_type(y_true, threshold=0.01)  # Line 315 model_utils.py
```

**Example 2: Prediction Storage**
```python
# 1. Dataset has task_type
dataset.task_type  # TaskType.REGRESSION

# 2. But controller passes it manually to assembler
prediction_data = {
    'task_type': dataset.task_type,  # Line 960 base_model.py
    # ... other data
}

# 3. Visualization RE-DETECTS from stored predictions
task_type = TabReportManager._detect_task_type_from_entry(entry)  # Line 40 reports.py
# Calls ModelUtils.detect_task_type(y_true)  # Line 75
```

**Example 3: Predict Mode Recovery**
```python
# In predict mode, dataset.task_type is None
if mode in ("predict", "explain") and dataset.task_type is None:  # Line 264
    # Try to restore from target_model metadata
    target_model = context.get('target_model')
    if target_model and 'task_type' in target_model:
        task_type_str = target_model['task_type']
        dataset.set_task_type(task_type_str)  # Line 267
    # BUT: TensorFlow controller will still re-detect from y_train!
```

#### Impact Analysis:
**Current State Summary:**
```
Detection Count per Pipeline Run:
├─ Targets.add_targets()           → 1x detect (NECESSARY)
├─ Targets.add_processed_targets() → 1x detect (LEGITIMATE - processing changes task)
├─ TensorFlowModel._train_model()  → 1x detect (REDUNDANT - should trust dataset)
├─ ModelUtils.calculate_scores()   → 1x re-detect (REDUNDANT - fallback on error)
└─ Visualization reports           → 1x detect (REDUNDANT - should trust metadata)
   TOTAL: 3-5 detections per run (2 necessary, 3 redundant)
``` Strict: 0.01 (line 315 model_utils.py)
- Processing changes (e.g., discretization) can change detected type
**Responsibility Confusion (RESOLVED):**
- ✅ **Who owns task_type?** → Targets block (data layer)
- ✅ **When is it authoritative?** → After each processing (mutable, tracked per processing)
- ✅ **Should it be mutable?** → YES - processing CAN change task (discretization, binning)
- ✅ **How to handle predict mode?** → Trust stored metadata (task_type + target_processing)
- ✅ **Future: per-processing models?** → Ready with `_task_type_by_processing` dictow classification
  - Dataset stores classification
  - Controller cached regression
  - Metrics are wrong

**Current State Summary:**
```
Detection Count per Pipeline Run:
├─ Targets.add_targets()           → 1x detect
├─ Targets.add_processed_targets() → 1x detect (if processing changes)
├─ TensorFlowModel._train_model()  → 1x detect
├─ ModelUtils.calculate_scores()   → 1x re-detect (on error fallback)
└─ Visualization reports           → 1x detect
   TOTAL: 3-5 detections per run
```

**Responsibility Confusion:**
- ❓ **Who owns task_type?** Dataset? Targets? Controller? Predictions?
- ❓ **When is it authoritative?** At data load? After preprocessing? At model training?
- ❓ **Should it be mutable?** Can preprocessing change task type?
- ❓ **How to handle predict mode?** Re-detect or trust stored metadata?

---

## 2. Responsibility Misplacement Analysis

### 2.1 Model Building (`model_builder.py`)

**Current Location:** `nirs4all/utils/model_builder.py`
**Actual Purpose:** Framework-specific model instantiation for controllers
**Problem:** This is **controller logic**, not a general utility.

#### Why it's misplaced:

1. **Tight coupling with controllers:**
   ```python
   # Only used by model controllers
   - sklearn_model.py: ModelBuilderFactory.build_single_model()
   - tensorflow_model.py: ModelBuilderFactory.build_single_model()
   - optuna.py: controller._get_model_instance() → ModelBuilderFactory
   ```

2. **Not a general utility:**
   - No other modules use it (checked predictions, datasets, operators, visualization)
   - Specific to model controller execution flow
   - Handles controller-specific concerns (dataset context, force_params)

3. **Complex framework-specific logic:**
   ```python
   def _from_tensorflow_callable(model_callable, dataset, force_params):
       # TensorFlow-specific model creation
       # This belongs with TensorFlow controller
   ```

4. **Dataset dependency:**
   ```python
   def build_single_model(model_config, dataset, force_params):
       # Needs dataset.is_classification, dataset.x() for input_dim
       # This is controller-level context, not utility-level
   ```

#### Correct Location:
**`nirs4all/controllers/models/model_factory.py`** (NEW)
- Or integrate into individual controller classes as factory methods
- Co-located with the controllers that need it

---

### 2.2 Model Utils (`model_utils.py`)

**Current Location:** `nirs4all/utils/model_utils.py`
**Mixed Responsibilities:** Split into controller utilities and prediction utilities

#### What belongs in controllers:

```python
# CONTROLLER-SPECIFIC (move to controllers/models/utilities.py or similar)
class TaskType(Enum)                          # Used by controllers
ModelUtils.detect_task_type()                 # Controller decision logic
ModelUtils.get_default_loss()                 # TensorFlow controller config
ModelUtils.get_default_metrics()              # Controller config
ModelUtils.validate_loss_compatibility()      # Controller validation
ModelUtils.get_best_score_metric()            # Controller scoring
ModelUtils.format_scores()                    # Controller output formatting
```

#### What could stay or move elsewhere:

```python
# PREDICTION UTILITIES (move to data/predictions_utils.py or remove)
ModelUtils.compute_weighted_average()         # Used for ensemble predictions
ModelUtils.compute_ensemble_prediction()      # Used by predictions module
ModelUtils._scores_to_weights()               # Prediction averaging support

# REDUNDANT (remove in favor of evaluator.py)
ModelUtils.calculate_scores()                 # Duplicate of evaluator.eval_multi()
```

**Usage Analysis:**
```
Used in:
- base_model.py (8 references) - CONTROLLER
- tensorflow_model.py (3 references) - CONTROLLER
- sklearn_model.py (2 references) - CONTROLLER
- score_calculator.py (2 references) - CONTROLLER COMPONENT
- tensorflow/config.py (1 reference) - CONTROLLER CONFIG
- predictions.py (1 reference) - For ensemble averaging
```

**Conclusion:** 90% of usage is in model controllers. It's controller infrastructure, not general utility.

---

## 3. Architecture Issues

### 3.1 Circular Dependency Risk

```
model_builder.py
  ↓ imports backend.py (TF_AVAILABLE, TORCH_AVAILABLE)
  ↓ imports dataset context
  ↓ used by controllers
  ↓ used by optuna.py
  ↑ which is used by controllers
```

**Problem:** Utils should not depend on domain entities (dataset). Controllers orchestrate domain logic.

---

### 3.2 Violation of Single Responsibility

#### `ModelBuilderFactory` does too much:
1. Parse configuration formats (string, dict, callable, instance)
2. Import classes from module paths
3. Detect frameworks (sklearn, tensorflow, pytorch)
4. Extract dataset dimensions
5. Handle parameter filtering and merging
6. Clone models
7. TensorFlow-specific model creation
8. Prepare and call with signature inspection

**Result:** 500+ line god class that's hard to test and maintain.

---

### 3.3 Poor Separation of Concerns

#### Current structure mixes:
- **Model instantiation** (factory)
- **Framework detection** (introspection)
- **Parameter management** (filtering, merging, forcing)
- **Input dimension extraction** (dataset operations)
- **Model cloning** (framework-specific)

#### Better structure:
```
controllers/models/
  ├── factory/
  │   ├── base_factory.py         # Abstract factory interface
  │   ├── sklearn_factory.py      # Sklearn-specific instantiation
  │   ├── tensorflow_factory.py   # TF-specific instantiation
  │   └── pytorch_factory.py      # PyTorch-specific instantiation
  ├── utilities/
  │   ├── task_utils.py           # TaskType enum, detection
  │   ├── metric_config.py        # Default losses/metrics by task
  │   └── score_utils.py          # Formatting, best metric selection
  └── components/
      └── (existing components)
```

---

## 4. Specific Code Smells

### 4.1 Deep Nesting in `model_builder.py`

```python
# build_single_model() has 4-level delegation
build_single_model()
  → _from_string() / _from_dict() / _from_instance() / _from_callable()
    → import_class() / import_object()
      → prepare_and_call()
        → _filter_params()
```

**Problem:** Hard to follow, hard to debug, hard to extend.

---

### 4.2 Magic Parameter Handling

```python
def prepare_and_call(callable_obj, params_from_caller=None, force_params_from_caller=None):
    # Line 483-494 - Not shown in summary
    # Merges params with signature inspection
    # Force_params override everything
    # But logic is opaque and spread across methods
```

**Problem:** Parameter precedence is unclear. Three sources of params (config, caller, force) with complex merge logic.

---

### 4.3 Framework Detection by String Matching

```python
def detect_framework(model):
    model_desc = str(model.__module__) if hasattr(model, '__module__') else str(type(model))
    if 'tensorflow' in model_desc:
        return 'tensorflow'
    elif 'torch' in model_desc:
        return 'pytorch'
    elif 'sklearn' in model_desc:
        return 'sklearn'
```

**Problem:** Brittle, fails for custom models, doesn't use proper type checking.

---

## 5. Impact on Controllers

### 5.1 External Dependencies Weaken Controller Autonomy

Controllers delegate critical responsibilities to distant modules:

```python
# sklearn_model.py
def _get_model_instance(self, dataset, model_config, force_params):
    # Jumps to utils/model_builder.py
    return ModelBuilderFactory.build_single_model(model_config, dataset, force_params)

# base_model.py
def _calculate_and_print_scores(self, y_true, y_pred, task_type, ...):
    # Jumps to utils/model_utils.py
    scores = ModelUtils.calculate_scores(y_true, y_pred, task_type, metrics)
```

**Problem:**
- Controllers can't own their complete workflow
- Testing requires mocking distant utils
- Changes to utils affect all controllers
- Violates "high cohesion, low coupling" principle

---

### 5.2 Inconsistent Abstraction Levels

```python
# base_model.py line 191-217
def get_xy(self, dataset, context):
    # High-level operation: extract train/test splits

# base_model.py line 674-682
def _detect_task_type(self, y):
    # Low-level operation: delegates to ModelUtils.detect_task_type()

# base_model.py delegates model building to ModelBuilderFactory
# But handles fold management, CV, score tracking internally
```

**Problem:** No clear boundary between what's internal vs external logic.

---

## 6. Task Type Management: Proposed Solution

### 6.1 Design Principles

**Single Source of Truth:** Task type should be:
1. **Detected once** at data ingestion
2. **Stored in Dataset** as immutable after first detection
3. **Trusted by all consumers** (controllers, visualization, predictions)
4. **Recoverable in predict mode** from model metadata

**Clear Lifecycle:**
```
Data Load → Detect → Store (immutable) → Use Everywhere
                ↓
            Serialize in model metadata
                ↓
            Restore in predict mode
```

### 6.2 Responsibility Assignment (UPDATED per user decision)

**Dataset/Targets (Data Layer):**
- ✅ Detect task type on first `add_targets()`
- ✅ Store as `_task_type` (reflects current "active" processing)
- ✅ **KEEP re-detection on `add_processed_targets()`** - Processing CAN change task (quantization, binning, encoding)
- ✅ **NEW: Store per-processing task types** in `_task_type_by_processing` dict for future use
- ✅ Expose via `dataset.task_type` property (returns task_type of current processing)
- ✅ Log warning when task_type changes between processings

**Rationale:**
- Transformations like discretization/binning fundamentally change regression → classification
- Future feature: Models may specify which target processing to use, need per-processing task_type
- Current behavior preserved, enhanced with tracking for extensibility

**Controllers (Execution Layer):**
- ✅ Read `dataset.task_type` (never re-detect)
- ✅ Specify which target processing to use (future: `context['y'] = 'binned'`)
- ✅ Use task_type for configuration (loss, metrics, evaluation)
- ✅ Store task_type AND target_processing in model metadata for predict mode
- ✅ Restore both in predict mode: `context['y'] = metadata['target_processing']`
- ❌ Should NOT call `ModelUtils.detect_task_type()` directly

**Predictions (Storage Layer):**
- ✅ Accept task_type from controller (from `dataset.task_type`)
- ✅ Store task_type AND target_processing as metadata fields
- ❌ Should NOT re-detect from y_true

#### Change 1: Track Task Type Per Processing (UPDATED)
```python
# data/targets.py
class Targets:
    def __init__(self):
        # ... existing init ...
        self._task_type: Optional[TaskType] = None  # Current active task type
        self._task_type_by_processing: Dict[str, TaskType] = {}  # NEW: Per-processing task types

    def add_targets(self, targets):
        if self.num_samples == 0:
            # ... existing logic ...
            # Detect task type for "numeric" processing
            if numeric_data.size > 0:
                self._task_type = ModelUtils.detect_task_type(numeric_data)
                self._task_type_by_processing['numeric'] = self._task_type
                # Also store for 'raw' if it exists
                if 'raw' in self._data:
                    self._task_type_by_processing['raw'] = self._task_type
        else:
            # Append mode: keep existing task_type
            # ... existing append logic ...

    def add_processed_targets(self, processing_name, targets, ...):
        # KEEP re-detection - processing CAN change task type
        # (e.g., discretization: regression → classification)
        if targets.size > 0:
            new_task_type = ModelUtils.detect_task_type(targets)
            self._task_type_by_processing[processing_name] = new_task_type

            # Update global task_type (for current processing)
            if self._task_type != new_task_type:
                print(f"⚠️ Task type changed: {self._task_type} → {new_task_type} "
                      f"(processing '{processing_name}')")
                self._task_type = new_task_type

        # ... rest of existing logic ...

#### Change 2: Remove Detection from TensorFlow Controller (UNCHANGED)
```python
# controllers/models/tensorflow_model.py
def _train_model(self, model, X_train, y_train, X_val, y_val, **kwargs):
    # REMOVE line 253:
    # task_type = ModelUtils.detect_task_type(y_train)

    # USE dataset task_type instead (pass from base controller)
    task_type = kwargs.get('task_type')  # Passed from base controller

    # Or access from self if dataset is stored
    task_type = self.dataset.task_type

    compile_config = TensorFlowCompilationConfig.prepare(train_params, task_type)
    # ... rest of method
```

**Note:** Dataset.task_type now reflects the task type of the current target processing,
which may have changed from 'numeric' if discretization/binning was applied.

#### Change 2: Remove Detection from TensorFlow Controller
```python
# controllers/models/tensorflow_model.py
def _train_model(self, model, X_train, y_train, X_val, y_val, **kwargs):
    # REMOVE line 253:
    # task_type = ModelUtils.detect_task_type(y_train)

    # USE dataset task_type instead (pass from base controller)
    task_type = kwargs.get('task_type')  # Passed from base controller

    # Or access from self if dataset is stored
    task_type = self.dataset.task_type

    compile_config = TensorFlowCompilationConfig.prepare(train_params, task_type)
    # ... rest of method
```

#### Change 3: Remove Re-Detection from ModelUtils
```python
# utils/model_utils.py
class ModelUtils:
    @staticmethod
    def calculate_scores(y_true, y_pred, task_type, metrics=None):
        # REMOVE re-detection fallback (lines 315-333)
        # If classification metrics fail, raise clear error instead

        if task_type == TaskType.REGRESSION:
            # ... regression metrics ...
        else:
            try:
                # ... classification metrics ...
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Classification metrics failed. "
                    f"Verify task_type={task_type} is correct for your data. "
                    f"Original error: {e}"
                )
```

#### Change 4: Trust Task Type in Visualization
#### Change 5: Pass Task Type and Target Processing Explicitly (UPDATED)
```python
# controllers/models/base_model.py
def launch_training(self, dataset, model_config, context, runner, ...):
    # Get task type from dataset (reflects current target processing)
    task_type = dataset.task_type

    # Track which target processing is being used
    target_processing = context.get('y', 'numeric')  # Default to 'numeric'

    # Pass to framework-specific controller
    train_params = model_config.get('train_params', {})
    train_params['task_type'] = task_type  # NEW: Explicit pass

    trained_model = self._train_model(
        model, X_train, y_train, X_val, y_val,
        task_type=task_type,  # NEW: Explicit parameter
        **train_params
### 6.4 Migration Strategy for Task Type (UPDATED)

**Phase 1: Add Per-Processing Task Type Tracking (2 hours)**
1. Add `_task_type_by_processing` dict to Targets
2. KEEP re-detection in `add_processed_targets()` (user requirement)
3. Store task_type for each processing as it's added
4. Add warning log when task_type changes between processings
5. Add `get_task_type_for_processing()` method for future use
6. Update tests to verify per-processing tracking

**Phase 2: Remove Controller Detection (2 hours)**
1. Remove `task_type = ModelUtils.detect_task_type(y_train)` from tensorflow_model.py
2. Pass task_type as parameter from base controller
3. Update sklearn controller similarly
4. Trust dataset.task_type (which now tracks processing changes)

**Phase 3: Remove Fallback Detection (1 hour)**
1. Remove re-detection from `ModelUtils.calculate_scores()`
2. Replace with clear error messages
3. Update error handling tests

**Phase 4: Enhance Metadata Tracking (2 hours) **NEW**
1. Store target_processing alongside task_type in model metadata
2. Update prediction metadata to include target_processing
3. Restore both in predict mode
4. Update visualization to read from enhanced metadata
5. Add tests for discretization → classification workflow

**Total Effort:** 7 hours for complete task_type cleanup + extensibility

**Key Difference from Original Plan:**
- ✅ Targets can re-detect on processing changes (user requirement)
- ✅ Per-processing task_type stored for future flexibility
- ✅ Controllers trust dataset but dataset adapts to processing
- ✅ Future-proof for models specifying target processing
        return ModelUtils.detect_task_type(y_true)
```

#### Change 5: Pass Task Type Explicitly in Base Controller
```python
# controllers/models/base_model.py
def launch_training(self, dataset, model_config, context, runner, ...):
    # Get task type from dataset (single source of truth)
    task_type = dataset.task_type

    # Pass to framework-specific controller
    train_params = model_config.get('train_params', {})
    train_params['task_type'] = task_type  # NEW: Explicit pass

    trained_model = self._train_model(
        model, X_train, y_train, X_val, y_val,
        task_type=task_type,  # NEW: Explicit parameter
        **train_params
    )
```

### 6.4 Migration Strategy for Task Type

**Phase 1: Make Dataset Authoritative (2 hours)**
1. Add `_task_type_locked` flag to Targets
2. Remove re-detection from `add_processed_targets()`
3. Update tests to verify task_type stability

**Phase 2: Remove Controller Detection (2 hours)**
1. Remove `task_type = ModelUtils.detect_task_type(y_train)` from tensorflow_model.py
2. Pass task_type as parameter from base controller
3. Update sklearn controller similarly

**Phase 3: Remove Fallback Detection (1 hour)**
1. Remove re-detection from `ModelUtils.calculate_scores()`
2. Replace with clear error messages
3. Update error handling tests

**Phase 4: Trust Metadata in Visualization (1 hour)**
1. Prioritize metadata task_type over re-detection
2. Keep fallback for backward compatibility
3. Log warning when fallback is used

**Total Effort:** 6 hours for complete task_type cleanup

---

## 7. Recommendations (Updated)

### 7.1 Immediate Actions (High Priority)

#### A. Consolidate Metric Calculation
**Remove:** `ModelUtils.calculate_scores()` from `model_utils.py`
**Replace with:** `evaluator.eval_multi()` everywhere
**Benefit:** Single source of truth, remove 120+ lines of duplication

```python
# Before (in controllers)
scores = ModelUtils.calculate_scores(y_true, y_pred, task_type)

# After
from nirs4all.core import metrics as evaluator
scores = evaluator.eval_multi(y_true, y_pred, task_type.value)
```

#### B. Move Model Building to Controllers Module
**Move:** `model_builder.py` → `controllers/models/factory.py`
**Rename:** `ModelBuilderFactory` → `ModelFactory`
**Benefit:** Co-locate with consumers, clearer ownership

```python
# New structure
nirs4all/controllers/models/
  ├── factory.py                  # ModelFactory (from model_builder.py)
  ├── base_model.py
  ├── sklearn_model.py
  └── tensorflow_model.py
```

#### C. Split Model Utils by Responsibility

**Create:** `controllers/models/utilities.py`
```python
# Task detection and configuration
class TaskType(Enum)
def detect_task_type(y) -> TaskType
def get_default_loss(task_type, framework) -> str
def get_default_metrics(task_type, framework) -> List[str]
def get_best_score_metric(task_type) -> Tuple[str, bool]
def format_scores(scores, precision=4) -> str
def validate_loss_compatibility(loss, task_type, framework) -> bool
```

**Create:** `data/ensemble_utils.py` (or integrate into predictions.py)
```python
# Prediction averaging and ensemble logic
def compute_weighted_average(arrays, scores, higher_is_better)
def compute_ensemble_prediction(predictions_data, ...)
def scores_to_weights(scores, higher_is_better)
```

**Create:** `data/ensemble_utils.py` (or integrate into predictions.py)
```python
# Prediction averaging and ensemble logic
def compute_weighted_average(arrays, scores, higher_is_better)
def compute_ensemble_prediction(predictions_data, ...)
def scores_to_weights(scores, higher_is_better)
```

**Remove:** `model_utils.py` entirely after migration

#### D. Centralize Task Type Management (NEW - HIGH PRIORITY)
**Goal:** Single source of truth for task type, eliminate redundant detection

**Changes:**
1. Make `dataset.task_type` immutable after first detection
2. Remove all re-detection in controllers (tensorflow_model.py line 253)
3. Remove fallback detection in ModelUtils.calculate_scores()
4. Trust task_type metadata in visualization

**Benefits:**
- Eliminate 3-5 redundant detections per pipeline run
- Remove inconsistency risk from different thresholds
- Clear responsibility: Dataset owns, everyone else consumes

---

### 7.2 Refactoring Strategy (Medium Priority)

#### A. Decompose ModelBuilderFactory

**Instead of one god class, create framework-specific factories:**

```python
# controllers/models/factory/base.py
class ModelFactory(ABC):
    @abstractmethod
    def build(self, config, dataset, force_params) -> Any:
        pass

    @abstractmethod
    def clone(self, model) -> Any:
        pass

# controllers/models/factory/sklearn.py
class SklearnModelFactory(ModelFactory):
    def build(self, config, dataset, force_params):
        # Sklearn-specific instantiation

    def clone(self, model):
        from sklearn.base import clone
        return clone(model)

# controllers/models/factory/tensorflow.py
class TensorFlowModelFactory(ModelFactory):
    def build(self, config, dataset, force_params):
        # TensorFlow-specific instantiation

### Phase 2: Centralize Task Type (7 hours) **UPDATED**
1. Add per-processing task_type tracking (`_task_type_by_processing` dict)
2. KEEP re-detection in targets.add_processed_targets() (legitimate - processing changes task)
3. Add warning log when task_type changes between processings
4. Remove detection from tensorflow_model._train_model()
5. Remove fallback detection from ModelUtils.calculate_scores()
6. Update visualization to trust metadata
7. Pass task_type + target_processing explicitly in controller methods
8. Update all tests to verify per-processing behavior
9. Add test for discretization workflow (regression → classification)n code)

---

#### B. Simplify Parameter Handling

**Current:** 3 parameter sources + complex merging + signature inspection
**Proposed:** Explicit precedence with clear rules

```python
class ParameterManager:
    """Handles parameter merging with clear precedence."""

    @staticmethod
    def merge_params(
        config_params: Dict,
        force_params: Dict,
        model_signature: inspect.Signature
    ) -> Dict:
        """
        Merge parameters with clear precedence:
        1. force_params (highest priority)
        2. config_params
        3. Filtered by model signature
        """
        # Clear, testable logic
**Updated Total Effort:** 16-21 hours (with enhanced task type management)
**Priority Order:** Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 (optional)
            k: v for k, v in all_params.items()
            if k in model_signature.parameters
        }
        return valid_params
```

---

### 6.3 Component Responsibilities (Low Priority)

#### Better component architecture in `controllers/models/`:

```python
components/
  ├── identifier_generator.py     # ✅ Good - specific responsibility
  ├── prediction_transformer.py   # ✅ Good - specific responsibility
  ├── prediction_assembler.py     # ✅ Good - specific responsibility
  ├── model_loader.py             # ✅ Good - specific responsibility
  ├── score_calculator.py         # ⚠️ Could use evaluator directly
  └── index_normalizer.py         # ✅ Good - specific responsibility

factory/
  ├── base_factory.py             # NEW - Abstract factory
  ├── sklearn_factory.py          # NEW - Sklearn instantiation
  └── tensorflow_factory.py       # NEW - TensorFlow instantiation

utilities/
  ├── task_utils.py               # NEW - TaskType, detection
  ├── metric_config.py            # NEW - Default losses/metrics
  └── parameter_manager.py        # NEW - Parameter handling
```

6. Task Type Management (NEW - UPDATED):
   - Task type detected on first add_targets()
   - Task type CAN change on add_processed_targets() (discretization, binning)
   - Per-processing task_type tracked in _task_type_by_processing dict
   - Warning logged when task_type changes between processings
   - Controllers trust dataset.task_type (no re-detection)
   - Predict mode correctly restores task_type + target_processing from metadata
   - Error handling when task_type is None
   - Task type + target_processing serialization in model metadata
   - Visualization reads task_type from prediction metadata
   - Future: get_task_type_for_processing() enables per-processing models
4. Run tests to verify no regression

### Phase 2: Centralize Task Type (6 hours) **NEW**
1. Make dataset.task_type immutable (add lock flag)
2. Remove re-detection from targets.add_processed_targets()
3. Remove detection from tensorflow_model._train_model()
4. Remove fallback detection from ModelUtils.calculate_scores()
5. Update visualization to trust metadata
6. Pass task_type explicitly in controller methods
7. Update all tests to verify single-detection behavior
### Medium Risk:
- Phase 2 (task type centralization) - Changes detection location but preserves behavior
  - ✅ KEEPS re-detection on processing changes (user requirement)
  - ✅ Discretization workflows unaffected
  - ⚠️ Need to test per-processing tracking
- Phase 4 (split utils) - Need to ensure all imports are updated
  - Potential for missed references in tests or examples
   - `optuna.py`
3. Rename `ModelBuilderFactory` to `ModelFactory` (optional but cleaner)
4. Run tests

### Phase 4: Split Model Utils (2-3 hours)
1. Create `controllers/models/utilities.py`
### Critical Risk: Task Type Mutability (RESOLVED)
**Original Risk:** Making task_type immutable would break legitimate use cases.

**User Decision:** Keep task_type mutable, track per-processing.

**New Approach:**
- ✅ Task type CAN change when processing changes (discretization, binning)
- ✅ Changes are logged with warnings for visibility
- ✅ Per-processing task_types stored for future model flexibility
- ✅ No breaking changes to discretization workflows

**Remaining Risks:**
- ⚠️ Controllers might cache old task_type before processing changes
  - **Mitigation:** Always read fresh from dataset.task_type
- ⚠️ Per-processing dict could grow large with many processings
  - **Mitigation:** Acceptable overhead, avg 2-5 processings per dataset
- ⚠️ Future feature (per-processing models) needs careful design
  - **Mitigation:** Infrastructure ready, defer implementation detailse 3 → Phase 4 → Phase 5 (optional)

---

## 9. Testing Strategy (Updated)### Critical test coverage needed:
```python
# After refactoring, ensure these scenarios still work:

1. Model building from different formats:
   - String path (class path)
   - Dictionary config ({'class': ..., 'params': ...})
   - Instance (already built model)
   - Callable (function/class)

2. Force params override:
   - Force params override config params
   - Validation for incompatible params

3. Framework detection:
   - Sklearn estimators
   - TensorFlow/Keras models
   - Custom models

4. Metric calculation:
   - Regression metrics match evaluator.py
   - Classification metrics match evaluator.py
   - Edge cases (empty predictions, NaN values)

5. Cross-controller consistency:
   - sklearn and tensorflow use same metric calculation
   - Same task type detection logic
   - Same ensemble averaging logic

6. Task Type Management (NEW):
   - Task type detected once on first add_targets()
   - Task type remains stable across add_processed_targets()
   - Controllers trust dataset.task_type (no re-detection)
   - Predict mode correctly restores task_type from metadata
   - Error handling when task_type is None
   - Task type serialization in model metadata
   - Visualization reads task_type from prediction metadata
```

---

## 10. Risk Assessment (Updated)

data/
  ├── targets.py            ✅ Detects per processing, tracks all task_types
  ├── dataset.py            ✅ Exposes current task_type + per-processing access
  └── ensemble_utils.py     ✅ Prediction ensemble logic
### Medium Risk:
- Phase 2 (task type centralization) - Changes detection behavior
  - Could break workflows that rely on re-detection
  - Need careful testing of discretization pipeline
- Phase 4 (split utils) - Need to ensure all imports are updated
  - Potential for missed references in tests or examples

### High Risk:
- Phase 5 (refactor factories) - Changes internal architecture
  - Could affect Optuna integration
  - Need comprehensive integration tests

### Critical Risk: Task Type Immutability
**New Risk from Phase 2:** Making task_type immutable could break legitimate use cases:
- ⚠️ Discretization changes continuous → discrete (regression → classification)
- ⚠️ Manual override for ambiguous data
- ⚠️ Backward compatibility with pipelines that call set_task_type()

**Mitigation Strategy:**
1. Add `force=True` parameter to `dataset.set_task_type(task_type, force=True)`
**Priority Recommendations:**
1. **Phase 1** (Metric redundancy) - 1-2 hours - **DO FIRST** - Low risk, high impact
2. **Phase 2** (Task type centralization) - 7 hours - **DO SECOND** - Removes redundancy, adds extensibility
3. **Phase 3** (Relocate model_builder) - 2-3 hours - **DO THIRD** - Mechanical refactor
4. **Phase 4** (Split model_utils) - 2-3 hours - **DO FOURTH** - Cleanup after phases 1-2
5. **Phase 5** (Refactor factories) - 4-6 hours - **OPTIONAL** - Nice to have, not critical

**Critical Success Factors (UPDATED):**
- ✅ Task type remains mutable (user requirement satisfied)
- ✅ Per-processing tracking enables future flexibility
- ✅ Discretization workflows unchanged (no breaking changes)
- ✅ Warning logs provide visibility into task_type changes
- ✅ Controllers trust dataset, never re-detect
- ✅ Metadata includes both task_type and target_processing
- ✅ Future-proof for models that specify target processing

**Recommendation:** Start with Phases 1-2 for immediate high-impact improvements with controlled risk. Phase 2 now ENHANCES rather than restricts task type management, making it both cleaner and more extensible.
- Test discretization pipeline specifically
- Test predict mode with legacy model metadata

---

## 11. Conclusion (Updated)

**Current State:**
- `model_builder.py` and `model_utils.py` are **misplaced** in the utils folder
- They contain **controller-specific logic** that should live with controllers
- There is **significant redundancy** with `evaluator.py`
- Responsibilities are **poorly separated**
- **CRITICAL:** Task type is detected **3-5 times per pipeline run** across 5 different layers

**Key Findings:**
1. **120+ lines of duplicate metric calculation** between model_utils and evaluator
2. **Model building logic** belongs in controllers, not utils
3. **Task type detection chaos:** 5 different places detect/store/re-detect independently
4. **No single source of truth** for task type, leading to inconsistency risk

**Recommended State:**
```
Before:
utils/
  ├── model_builder.py      ❌ Controller logic in utils
  ├── model_utils.py        ❌ Mixed responsibilities + redundant metrics + task detection
  └── evaluator.py          ✅ Good

data/
  └── targets.py            ⚠️ Detects task_type + re-detects + mutable

controllers/
  └── models/
      ├── base_model.py     ⚠️ Re-detects task_type
      └── tensorflow_model.py ⚠️ Always re-detects task_type

visualization/
  └── reports.py            ⚠️ Re-detects task_type from data

After:
controllers/models/
  ├── factory.py            ✅ Model building co-located
  ├── utilities.py          ✅ Controller utilities (task detection, config)
  └── components/           ✅ Existing components

data/
  ├── targets.py            ✅ Single detection, immutable task_type
  ├── dataset.py            ✅ Exposes authoritative task_type
  └── ensemble_utils.py     ✅ Prediction ensemble logic

utils/
  └── evaluator.py          ✅ Metric calculation (single source)

All consumers:
  └── Trust dataset.task_type ✅ No re-detection
```

**Benefits of Refactoring:**
1. **Clearer architecture** - Controllers own their complete workflow
2. **Reduced complexity** - Remove 120+ lines of duplicate code
3. **Better testability** - Isolated, cohesive components
4. **Easier maintenance** - Related code lives together
5. **Reduced coupling** - Utils don't depend on domain entities
6. **Performance** - Eliminate redundant task type detection (3-5x per run)
7. **Consistency** - Single source of truth prevents threshold mismatches
8. **Data integrity** - Immutable task_type prevents cascade errors

**Estimated Effort:** 15-20 hours (Phases 1-4 are essential, Phase 5 is optional improvement)

**Priority Recommendations:**
1. **Phase 1** (Metric redundancy) - 1-2 hours - **DO FIRST** - Low risk, high impact
2. **Phase 2** (Task type centralization) - 6 hours - **DO SECOND** - Fixes critical architecture flaw
3. **Phase 3** (Relocate model_builder) - 2-3 hours - **DO THIRD** - Mechanical refactor
4. **Phase 4** (Split model_utils) - 2-3 hours - **DO FOURTH** - Cleanup after phases 1-2
5. **Phase 5** (Refactor factories) - 4-6 hours - **OPTIONAL** - Nice to have, not critical

**Critical Success Factors:**
- Task type immutability must be **soft lock with warnings first** (version 0.5)
- Comprehensive testing of discretization workflows
- Migration guide for users who explicitly set task_type
- Backward compatibility for predict mode with legacy metadata
- Documentation of new task_type lifecycle

**Recommendation:** Start with Phases 1-2 for immediate high-impact improvements with controlled risk. These fix the most critical architectural issues (redundancy and task type chaos) that affect every pipeline run.

