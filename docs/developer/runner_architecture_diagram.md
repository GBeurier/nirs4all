# PipelineRunner Refactoring - Visual Architecture

## Current Architecture (Problematic)

```
┌──────────────────────────────────────────────────────────────────────┐
│                          PipelineRunner                              │
│                         (GOD CLASS - 1050 lines)                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ❌ Responsibilities:                                                │
│     • Multi-pipeline × multi-dataset orchestration                  │
│     • Single pipeline execution                                      │
│     • Single step execution                                          │
│     • Step parsing & deserialization                                 │
│     • Controller selection                                           │
│     • Artifact management                                            │
│     • Workspace setup                                                │
│     • Predictions aggregation                                        │
│     • Mode switching (train/predict/explain)                         │
│     • Data caching                                                   │
│     • Progress tracking & spinner display                            │
│     • Best results reporting                                         │
│     • ... and more                                                   │
│                                                                      │
│  ❌ Problems:                                                        │
│     • 20+ instance variables                                         │
│     • Hardcoded operator lists                                       │
│     • Mixed abstraction levels                                       │
│     • Difficult to test                                              │
│     • Poor extensibility                                             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Proposed Architecture (Clean Separation)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PipelineOrchestrator                             │
│                  (HIGH LEVEL - Multi × Multi)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  Responsibilities:                                                      │
│    ✓ Coordinate N pipelines × M datasets                               │
│    ✓ Workspace initialization                                           │
│    ✓ Global predictions aggregation                                     │
│    ✓ Best results reporting                                             │
│                                                                         │
│  Dependencies:                                                          │
│    • WorkspaceManager                                                   │
│    • PipelineExecutor (creates one per pipeline)                        │
│                                                                         │
│  Size: ~150 lines                                                       │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     │ creates & delegates to
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PipelineExecutor                                │
│                   (MEDIUM LEVEL - 1 × 1)                                │
├─────────────────────────────────────────────────────────────────────────┤
│  Responsibilities:                                                      │
│    ✓ Execute ONE pipeline on ONE dataset                               │
│    ✓ Iterate through steps                                              │
│    ✓ Accumulate predictions for this run                               │
│    ✓ Manage artifacts for this pipeline                                │
│                                                                         │
│  Dependencies:                                                          │
│    • ExecutionContext (immutable)                                       │
│    • ArtifactManager                                                    │
│    • StepRunner (delegates to)                                          │
│                                                                         │
│  Size: ~120 lines                                                       │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     │ delegates to
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            StepRunner                                   │
│                       (LOW LEVEL - Single Step)                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Responsibilities:                                                      │
│    ✓ Execute ONE step                                                   │
│    ✓ Coordinate parsing & routing                                       │
│    ✓ Dispatch to controller                                             │
│    ✓ Handle binary loading                                              │
│                                                                         │
│  Dependencies:                                                          │
│    • StepParser (parses step config)                                    │
│    • ControllerRouter (selects controller)                              │
│    • ExecutionContext                                                   │
│                                                                         │
│  Size: ~90 lines                                                        │
└────────┬──────────────────────────┬─────────────────────────────────────┘
         │                          │
         │ uses                     │ uses
         ▼                          ▼
┌────────────────────┐    ┌──────────────────────────┐
│   StepParser       │    │   ControllerRouter       │
├────────────────────┤    ├──────────────────────────┤
│  Responsibilities: │    │  Responsibilities:       │
│   ✓ Parse step cfg │    │   ✓ Select controller    │
│   ✓ Deserialize    │    │   ✓ Registry lookup      │
│   ✓ Extract keyword│    │   ✓ Priority sorting     │
│                    │    │                          │
│  Size: ~120 lines  │    │  Size: ~70 lines         │
└────────────────────┘    └──────────────────────────┘
```

---

## Data Flow - Training Mode

```
User Code
   │
   │ runner.run(pipeline, dataset)
   ▼
┌─────────────────────────────────────────┐
│ PipelineOrchestrator.execute()          │
│  • Normalize inputs                     │
│  • Create workspace structure           │
└────────┬────────────────────────────────┘
         │
         │ for each (pipeline × dataset):
         ▼
    ┌─────────────────────────────────────┐
    │ PipelineExecutor.execute()          │
    │  • Setup manifest                   │
    │  • Initialize predictions store     │
    └────────┬────────────────────────────┘
             │
             │ for each step:
             ▼
        ┌─────────────────────────────────┐
        │ StepRunner.execute()            │
        │  1. Parse step                  │
        │  2. Route to controller         │
        │  3. Execute controller          │
        │  4. Collect artifacts           │
        └────────┬────────────────────────┘
                 │
                 ├──────────────┬─────────────┐
                 ▼              ▼             ▼
        ┌─────────────┐  ┌──────────┐  ┌──────────────┐
        │ StepParser  │  │ Router   │  │ Controller   │
        │  parse()    │  │  route() │  │  execute()   │
        └─────────────┘  └──────────┘  └──────┬───────┘
                                               │
                                               │ artifacts
                                               ▼
                                    ┌─────────────────────┐
                                    │ ArtifactManager     │
                                    │  persist()          │
                                    └─────────────────────┘
```

---

## Data Flow - Prediction Mode

```
User Code
   │
   │ runner.predict(prediction_obj, dataset)
   ▼
┌─────────────────────────────────────────┐
│ PipelineOrchestrator.execute()          │
│  • Load manifest                        │
│  • Extract pipeline_uid                 │
│  • Create BinaryLoader                  │
└────────┬────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │ PipelineExecutor.execute()          │
    │  • mode = PREDICT                   │
    │  • Skip non-prediction controllers  │
    └────────┬────────────────────────────┘
             │
             │ for each step:
             ▼
        ┌─────────────────────────────────┐
        │ StepRunner.execute()            │
        │  • Load binaries for this step  │
        │  • Execute with loaded model    │
        └────────┬────────────────────────┘
                 │
                 ├──────────────┬─────────────┐
                 ▼              ▼             ▼
        ┌─────────────┐  ┌──────────┐  ┌──────────────┐
        │ StepParser  │  │ Router   │  │ Controller   │
        │  parse()    │  │  route() │  │  execute()   │
        └─────────────┘  └──────────┘  │  (w/ model)  │
                                        └──────┬───────┘
                                               │
                                               │ predictions
                                               ▼
                                    ┌─────────────────────┐
                                    │ Predictions Store   │
                                    │  add_prediction()   │
                                    └─────────────────────┘
```

---

## Context & State Management

### Current (Scattered State)

```
PipelineRunner
├─ self.step_number = 0
├─ self.substep_number = -1
├─ self.operation_count = 0
├─ self.pipeline_uid = None
├─ self.mode = "train"
├─ self.verbose = 0
├─ self.workspace_path = ...
├─ self.saver = ...
├─ self.manifest_manager = ...
├─ self.binary_loader = ...
├─ self.raw_data = {}
├─ self.pp_data = {}
├─ self._figure_refs = []
├─ self._capture_model = False
├─ self._captured_model = None
├─ ... (20+ more)
└─ PROBLEM: Mutable, hard to track, not thread-safe
```

### Proposed (Immutable Context)

```
ExecutionContext (dataclass, frozen=True)
├─ mode: ExecutionMode                    # Enum: TRAIN, PREDICT, EXPLAIN
├─ workspace_path: Path
├─ pipeline_uid: str
├─ manifest_manager: ManifestManager
├─ verbose: int
├─ step_tracker: StepTracker             # Separate mutable tracker
└─ config: Dict[str, Any]

StepTracker (separate, focused)
├─ step_number: int = 0
├─ substep_number: int = 0
├─ operation_count: int = 0
└─ increment(), reset()

✅ Immutable contexts can be safely passed around
✅ Mutable tracking separated into focused class
✅ Thread-safe
✅ Easy to test
```

---

## Controller Selection - Before & After

### Current (Rigid, Hardcoded)

```python
class PipelineRunner:
    WORKFLOW_OPERATORS = [
        "sample_augmentation", "feature_augmentation",
        "branch", "model", "stack", "chart_2d", ...
    ]
    SERIALIZATION_OPERATORS = [
        "class", "function", "module", "object", ...
    ]

    def run_step(self, step, ...):
        # 116 lines of parsing logic mixed with execution
        if isinstance(step, dict):
            if key := next((k for k in step if k in WORKFLOW_OPERATORS), None):
                # ... operator detection
            elif key := next((k for k in step if k in SERIALIZATION_OPERATORS), None):
                # ... serialization detection
            else:
                # ... default handling
        elif isinstance(step, list):
            # ... list handling
        elif isinstance(step, str):
            # ... string handling
        # ... 80+ more lines

❌ Problems:
   • Hardcoded lists need updates for every new operator
   • Parsing mixed with execution
   • Cannot add new syntaxes without modifying runner
   • Difficult to test parsing independently
```

### Proposed (Flexible, Extensible)

```python
class StepParser:
    """Extensible parser - no hardcoded lists."""

    def parse(self, step: Any) -> ParsedStep:
        """Parse any step format into normalized form."""
        if isinstance(step, dict):
            return self._parse_dict(step)
        elif isinstance(step, list):
            return self._parse_list(step)
        elif isinstance(step, str):
            return self._parse_string(step)
        elif hasattr(step, '__call__'):
            return self._parse_callable(step)
        else:
            return self._parse_instance(step)

    def _parse_dict(self, step: Dict) -> ParsedStep:
        """Dict can be: {operator: value}, {class: X}, etc."""
        # Clean parsing logic, ~20 lines
        ...

class ControllerRouter:
    """Route based on controller.matches(), not hardcoded lists."""

    def route(self, parsed_step: ParsedStep) -> OperatorController:
        """Let controllers decide if they match."""
        matches = [
            cls for cls in CONTROLLER_REGISTRY
            if cls.matches(
                parsed_step.step,
                parsed_step.operator,
                parsed_step.keyword
            )
        ]
        if not matches:
            raise NoControllerFoundError(...)

        # Return highest priority match
        matches.sort(key=lambda c: c.priority)
        return matches[0]()

✅ Benefits:
   • No hardcoded lists
   • New operators work automatically (via controller.matches())
   • Parsing testable independently
   • Easy to add new step formats
   • Controllers self-describe their capabilities
```

---

## Example: Adding New Step Type

### Current (Requires Multiple Changes)

```python
# 1. Update WORKFLOW_OPERATORS list
WORKFLOW_OPERATORS = [..., "new_operator"]

# 2. Update run_step() parsing logic
if isinstance(step, dict):
    if key := next((k for k in step if k in WORKFLOW_OPERATORS), None):
        if key == "new_operator":  # NEW CODE
            # Handle new operator
            ...

# 3. Create controller
@register_controller
class NewOperatorController(OperatorController):
    @classmethod
    def matches(cls, step, operator, keyword):
        return keyword == "new_operator"
    ...

❌ 3 locations to change
❌ Easy to forget updating list
❌ Parsing logic grows linearly
```

### Proposed (Single Change)

```python
# 1. Create controller - that's it!
@register_controller
class NewOperatorController(OperatorController):
    priority = 50  # Set appropriate priority

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Define matching logic here."""
        return keyword == "new_operator" or \
               isinstance(operator, NewOperatorType)

    def execute(self, ...):
        ...

✅ Single location to change
✅ Controller self-registers
✅ Parser automatically handles it
✅ Router automatically finds it
```

---

## Testing Comparison

### Current (Difficult)

```python
def test_step_parsing():
    # Need full PipelineRunner with all dependencies
    runner = PipelineRunner(
        workspace_path=tmp_path,
        save_files=False,
        enable_tab_reports=False,
        # ... 15+ parameters
    )
    runner.mode = "train"
    runner.manifest_manager = MagicMock()
    runner.saver = MagicMock()
    runner.binary_loader = MagicMock()
    # ... mock 10+ more dependencies

    # Can't test parsing without execution
    step = {"model": "SVC"}
    context = runner.run_step(step, dataset, context)
    # Assertion on side effects, not parse result

❌ Heavy setup
❌ Can't isolate parsing
❌ Mocking nightmare
```

### Proposed (Clean)

```python
def test_step_parsing():
    """Test parser independently."""
    parser = StepParser()

    # Test dict format
    result = parser.parse({"model": "SVC", "params": {"C": 1.0}})
    assert result.keyword == "model"
    assert result.operator.__class__.__name__ == "SVC"
    assert result.metadata["params"]["C"] == 1.0

def test_controller_routing():
    """Test router independently."""
    router = ControllerRouter()
    parsed = ParsedStep(operator=SVC(), keyword="model", ...)

    controller = router.route(parsed)
    assert isinstance(controller, SklearnModelController)

def test_step_execution():
    """Test execution with mocked parser & router."""
    parser = Mock(return_value=parsed_step)
    router = Mock(return_value=mock_controller)
    runner = StepRunner(parser, router, context)

    result = runner.execute(step, dataset, ctx)
    assert result.artifacts is not None

✅ Lightweight tests
✅ Isolated components
✅ Fast execution
✅ Clear assertions
```

---

## Memory & Performance

### Current Issues

```python
# Full dataset copy
if self.keep_datasets:
    self.raw_data[dataset_name] = dataset.x({}, layout="2d")
    # ⚠️ Copies entire numpy array into memory
    # Problem: 10GB dataset → 10GB extra memory

# Controller instantiation
def _select_controller(self, step, operator, keyword):
    matches = [cls for cls in CONTROLLER_REGISTRY if cls.matches(...)]
    return matches[0]()  # ⚠️ New instance every time
    # Problem: Creates thousands of controller instances

# Predictions accumulation
config_predictions = Predictions()
for step in steps:
    # Keeps growing in memory
    config_predictions.merge(...)
# ⚠️ All predictions in RAM
```

### Proposed Solutions

```python
# 1. Lazy dataset views
class DatasetView:
    def __init__(self, dataset: SpectroDataset):
        self._dataset = dataset
        self._snapshots: Dict[str, weakref.ref] = {}

    def snapshot(self, key: str) -> np.ndarray:
        """Create snapshot only when explicitly requested."""
        if key not in self._snapshots or self._snapshots[key]() is None:
            data = self._dataset.x({}, layout="2d")
            self._snapshots[key] = weakref.ref(data)
        return self._snapshots[key]()
# ✅ No upfront copy
# ✅ Garbage collected when not needed

# 2. Controller caching
@lru_cache(maxsize=128)
def get_controller(controller_class: Type) -> OperatorController:
    """Cache controller instances."""
    return controller_class()

class ControllerRouter:
    def route(self, parsed: ParsedStep) -> OperatorController:
        controller_class = self._find_match(parsed)
        return get_controller(controller_class)
# ✅ Reuse controller instances
# ✅ Significant speedup for long pipelines

# 3. Streaming predictions
class PredictionWriter:
    def __init__(self, output_path: Path):
        self.writer = ParquetWriter(output_path)

    def write(self, prediction: Dict):
        """Write incrementally, don't accumulate."""
        self.writer.append(prediction)
# ✅ Constant memory usage
# ✅ Can handle millions of predictions
```

---

## Backward Compatibility Strategy

```python
# Keep old interface, redirect internally
class PipelineRunner:
    """
    DEPRECATED: Use PipelineOrchestrator instead.

    This class is maintained for backward compatibility and will be
    removed in v0.6.0. Please migrate to the new API.
    """

    def __init__(self, **kwargs):
        warnings.warn(
            "PipelineRunner is deprecated. Use PipelineOrchestrator.",
            DeprecationWarning,
            stacklevel=2
        )
        self._orchestrator = PipelineOrchestrator(**kwargs)

    def run(self, pipeline, dataset, **kwargs):
        """Redirect to new orchestrator."""
        return self._orchestrator.execute(pipeline, dataset, **kwargs)

    def predict(self, *args, **kwargs):
        """Redirect to new orchestrator."""
        return self._orchestrator.execute(*args, mode=ExecutionMode.PREDICT, **kwargs)

# ✅ Users don't break immediately
# ✅ Deprecation warnings guide migration
# ✅ Remove in next major version
```

---

## Summary Comparison

| Aspect | Current | Proposed |
|--------|---------|----------|
| **Lines of code** | 1050 in 1 file | 150+120+90+120+70 = 550 across 5 files |
| **Responsibilities per class** | 13+ | 1-2 per class |
| **Testing difficulty** | High (need 20+ mocks) | Low (isolated units) |
| **Extensibility** | Hard (hardcoded lists) | Easy (plugin pattern) |
| **Memory usage** | High (copies data) | Optimized (lazy loading) |
| **Controller reuse** | None (recreated) | Yes (cached) |
| **Code duplication** | Yes (parsing in multiple places) | No (single parser) |
| **Documentation** | Sparse, outdated | Comprehensive, Google-style |
| **State management** | Mutable, scattered | Immutable, focused |
| **Parallel execution** | Limited support | Full support |

---

**Recommendation: This refactoring should be prioritized before adding new features.**

The current architecture has accumulated too much technical debt. Continuing to build on it will make maintenance exponentially harder.

With this refactoring:
- ✅ New features are easier to add
- ✅ Bugs are easier to fix
- ✅ Code is easier to understand
- ✅ Tests are easier to write
- ✅ Performance is better
- ✅ Team velocity increases

**Estimated ROI: 300% over next 12 months**
- Initial investment: 2-3 weeks
- Time saved: ~6-9 weeks over next year (fewer bugs, faster features)
