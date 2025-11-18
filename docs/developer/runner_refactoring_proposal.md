# PipelineRunner Refactoring Proposal

**Status**: Proposal
**Dependencies**: [Context Refactoring Proposal](CONTEXT_REFACTORING_PROPOSAL.md) (CRITICAL BLOCKER)
**Related Documents**:
- [Architecture Diagrams](RUNNER_ARCHITECTURE_DIAGRAM.md)
- [Code Examples](RUNNER_CODE_EXAMPLES.md)
- [Context Refactoring](CONTEXT_REFACTORING_PROPOSAL.md)

---

## ⚠️ CRITICAL DEPENDENCY

**This refactoring CANNOT proceed without first completing the Context Refactoring.**

The context object (Dict[str, Any]) serves three distinct roles:
1. Data selector for dataset operations
2. Pipeline execution state
3. Controller communication flags

The ExecutionContext proposed here assumes typed context separation. See [CONTEXT_REFACTORING_PROPOSAL.md](CONTEXT_REFACTORING_PROPOSAL.md) for comprehensive analysis.

**Recommended sequence:**
1. Complete Context Refactoring (Phases 1-3)
2. Begin Runner Refactoring
3. Continue Context migration (Phases 4-6) in parallel

---

## Executive Summary

The `PipelineRunner` class has evolved into a **god class** with ~1050 lines and ~20 responsibilities. This proposal outlines a comprehensive refactoring to improve maintainability, readability, extensibility, and performance.

## Current Problems

### 1. **God Class Anti-Pattern**
- Single class handles: execution orchestration, step parsing, controller selection, artifact management, workspace setup, predictions management, mode switching (train/predict/explain), data caching, progress tracking, spinner display, normalization, and more.
- 20+ instance variables
- Methods range from 3 to 120+ lines
- Violates Single Responsibility Principle

### 2. **Mixed Responsibilities**
- **Runner** (orchestrates multiple pipeline × dataset combinations) mixed with **Pipeline Executor** (executes single pipeline on single dataset)
- Cannot execute pipeline without runner infrastructure
- Difficult to test individual components

### 3. **Rigid Controller Selection**
- Hardcoded operator lists: `WORKFLOW_OPERATORS`, `SERIALIZATION_OPERATORS`
- Step parsing logic deeply embedded in `run_step()` (lines 827-943)
- Adding new operator types requires modifying multiple locations
- Poor separation between parsing and execution

### 4. **Poor Naming**
- `run()` vs `_run_single()` vs `run_step()` vs `run_steps()` - confusing hierarchy
- `_normalize_pipeline()` / `_normalize_dataset()` - what's being normalized?
- `_execute_controller()` - too generic
- `next_op()` - unclear purpose

### 5. **State Management**
- 20+ instance variables tracking various states
- State shared between runner and individual executions
- Difficult to reason about state at any given point

### 6. **Performance Issues**
- Full dataset copying when `keep_datasets=True`: `self.raw_data[dataset_name] = dataset.x({}, layout="2d")`
- No lazy loading or streaming capabilities
- Potential memory issues with large datasets

### 7. **Testing Challenges**
- God class makes unit testing difficult
- Need to mock 20+ dependencies
- Cannot test step parsing independently of execution

---

## Proposed Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│ PipelineOrchestrator (high-level: N pipelines × M datasets) │
├─────────────────────────────────────────────────────────────┤
│ - Coordinates multiple runs                                  │
│ - Manages workspace and predictions aggregation             │
│ - Delegates to PipelineExecutor for each (pipeline×dataset) │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴───────────┐
        │                        │
        ▼                        ▼
┌───────────────────┐    ┌──────────────────┐
│ PipelineExecutor  │    │ ExecutionContext │
├───────────────────┤    ├──────────────────┤
│ - Executes single │    │ - Immutable      │
│   pipeline on one │    │ - Step tracking  │
│   dataset         │    │ - Mode (train/   │
│ - Step iteration  │    │   predict)       │
│ - Uses StepRunner │    │ - Partition info │
└────────┬──────────┘    └──────────────────┘
         │
         ▼
┌───────────────────┐
│ StepRunner        │
├───────────────────┤
│ - Executes single │
│   step            │
│ - Controller      │
│   dispatch        │
│ - Uses StepParser │
└────────┬──────────┘
         │
         ├──────────────────┐
         │                  │
         ▼                  ▼
┌────────────────┐   ┌─────────────────┐
│ StepParser     │   │ ControllerRouter│
├────────────────┤   ├─────────────────┤
│ - Parse step   │   │ - Select best   │
│   config       │   │   controller    │
│ - Extract      │   │ - Registry      │
│   operator     │   │   lookup        │
│ - Deserialize  │   │ - Extensible    │
└────────────────┘   └─────────────────┘
```

### Component Breakdown

#### 1. **PipelineOrchestrator** (replaces PipelineRunner.run())
*Responsibility*: Coordinate multiple pipeline × dataset combinations

```python
class PipelineOrchestrator:
    """
    Orchestrates execution of multiple pipelines across multiple datasets.

    High-level coordinator that manages:
    - Workspace initialization
    - Global predictions aggregation
    - Best results reporting
    - Dataset/pipeline normalization

    Attributes:
        workspace_manager: Handles workspace structure and paths
        config: Orchestration configuration (parallelism, verbosity, etc.)
    """

    def __init__(
        self,
        workspace_path: Optional[Path] = None,
        config: OrchestrationConfig = None
    ):
        """
        Initialize orchestrator with workspace and configuration.

        Args:
            workspace_path: Root workspace directory
            config: Configuration for execution behavior
        """

    def execute(
        self,
        pipelines: PipelineConfigs,
        datasets: DatasetConfigs,
        mode: ExecutionMode = ExecutionMode.TRAIN
    ) -> OrchestrationResult:
        """
        Execute all pipeline × dataset combinations.

        Args:
            pipelines: Pipeline configurations to execute
            datasets: Dataset configurations to use
            mode: Execution mode (TRAIN, PREDICT, EXPLAIN)

        Returns:
            OrchestrationResult with aggregated predictions and metadata
        """
```

**Key improvements:**
- Clear single responsibility
- Delegates execution to `PipelineExecutor`
- Manages only high-level concerns
- ~150 lines maximum

---

#### 2. **PipelineExecutor** (replaces PipelineRunner._run_single())
*Responsibility*: Execute a single pipeline on a single dataset

```python
class PipelineExecutor:
    """
    Executes a single pipeline configuration on a single dataset.

    Handles:
    - Step-by-step execution
    - Context propagation
    - Artifact management for one pipeline run
    - Predictions accumulation for this pipeline

    Attributes:
        context: Execution context (immutable)
        artifact_manager: Handles binary persistence
        step_runner: Executes individual steps
    """

    def __init__(
        self,
        context: ExecutionContext,
        artifact_manager: ArtifactManager,
        step_runner: StepRunner
    ):
        """
        Initialize executor with context and dependencies.

        Args:
            context: Execution context (mode, workspace, etc.)
            artifact_manager: Manages artifact persistence
            step_runner: Runs individual pipeline steps
        """

    def execute(
        self,
        steps: List[StepConfig],
        dataset: SpectroDataset
    ) -> ExecutionResult:
        """
        Execute pipeline steps sequentially on dataset.

        Args:
            steps: List of pipeline steps to execute
            dataset: Dataset to operate on

        Returns:
            ExecutionResult with predictions and artifacts
        """
```

**Key improvements:**
- Single responsibility: one pipeline × one dataset
- Stateless except for execution tracking
- Easy to test in isolation
- ~100-150 lines maximum

---

#### 3. **StepRunner** (replaces PipelineRunner.run_step())
*Responsibility*: Execute a single pipeline step

```python
class StepRunner:
    """
    Executes a single pipeline step.

    Handles:
    - Step parsing (delegates to StepParser)
    - Controller selection (delegates to ControllerRouter)
    - Controller execution
    - Binary loading/saving for this step

    Attributes:
        parser: Parses step configuration
        router: Selects appropriate controller
        context: Current execution context
    """

    def __init__(
        self,
        parser: StepParser,
        router: ControllerRouter,
        context: ExecutionContext
    ):
        """
        Initialize step runner with parser and router.

        Args:
            parser: Parses step configurations
            router: Routes to appropriate controller
            context: Execution context
        """

    def execute(
        self,
        step: StepConfig,
        dataset: SpectroDataset,
        pipeline_context: Dict[str, Any],
        loaded_binaries: Optional[List[ArtifactMeta]] = None
    ) -> StepResult:
        """
        Execute a single pipeline step.

        Args:
            step: Step configuration to execute
            dataset: Dataset to operate on
            pipeline_context: Current pipeline context (processing chain, etc.)
            loaded_binaries: Pre-loaded artifacts from manifest

        Returns:
            StepResult with updated context and artifacts
        """
```

**Key improvements:**
- Single step execution only
- Clean separation from orchestration
- ~80-100 lines maximum

---

#### 4. **StepParser** (NEW - extracts parsing logic)
*Responsibility*: Parse step configurations into executable format

```python
class StepParser:
    """
    Parses pipeline step configurations into normalized format.

    Handles multiple step syntaxes:
    - Dictionary: {"model": SVC, "params": {...}}
    - String: "sklearn.preprocessing.StandardScaler"
    - Direct instance: StandardScaler()
    - Nested lists: [[step1, step2], step3]

    Normalizes to canonical format for controller execution.
    """

    def parse(self, step: Any) -> ParsedStep:
        """
        Parse raw step configuration into normalized format.

        Args:
            step: Raw step configuration (dict, str, list, instance)

        Returns:
            ParsedStep with operator, keyword, and metadata

        Raises:
            StepParseError: If step format is invalid
        """
```

**Key improvements:**
- Extensible: easy to add new step formats
- No hardcoded operator lists
- Single responsibility
- ~100-150 lines maximum

---

#### 5. **ControllerRouter** (replaces _select_controller() + hardcoded lists)
*Responsibility*: Select appropriate controller for a step

```python
class ControllerRouter:
    """
    Routes parsed steps to appropriate controllers.

    Uses registry pattern with controller priorities.
    Extensible - new controllers automatically discovered via registry.

    Attributes:
        registry: Controller registry (singleton)
    """

    def route(self, parsed_step: ParsedStep) -> OperatorController:
        """
        Select best controller for parsed step.

        Uses controller.matches() with priority sorting.

        Args:
            parsed_step: Normalized step configuration

        Returns:
            Best matching controller instance

        Raises:
            NoControllerFoundError: If no controller matches
        """
```

**Key improvements:**
- Removes hardcoded operator lists
- Centralized routing logic
- Easy to extend (just register new controllers)
- ~50-80 lines maximum

---

#### 6. **ExecutionContext** (NEW - replaces scattered state)
*Responsibility*: Immutable execution context

```python
@dataclass(frozen=True)
class ExecutionContext:
    """
    Immutable execution context for a pipeline run.

    Contains all execution parameters and state that doesn't change
    during a single pipeline execution.

    Attributes:
        mode: Execution mode (TRAIN, PREDICT, EXPLAIN)
        workspace_path: Workspace root directory
        pipeline_uid: Unique pipeline identifier
        manifest_manager: Artifact manifest manager
        verbose: Verbosity level (0-3)
        step_tracker: Tracks current step/substep numbers
        config: Additional configuration options
    """

    mode: ExecutionMode
    workspace_path: Path
    pipeline_uid: str
    manifest_manager: ManifestManager
    verbose: int = 0
    step_tracker: StepTracker = field(default_factory=StepTracker)
    config: Dict[str, Any] = field(default_factory=dict)

    def with_step(self, step_number: int, substep_number: int = 0) -> 'ExecutionContext':
        """
        Create new context with updated step tracking.

        Args:
            step_number: Current main step number
            substep_number: Current substep number

        Returns:
            New ExecutionContext with updated tracking
        """
```

**Key improvements:**
- Immutable: thread-safe, predictable
- All execution params in one place
- Easy to test
- No hidden state

---

#### 7. **ArtifactManager** (extracts binary management)
*Responsibility*: Manage artifact persistence and loading

```python
class ArtifactManager:
    """
    Manages artifact persistence and retrieval.

    Handles:
    - Binary serialization/deserialization
    - Content-addressed storage
    - Manifest integration
    - Loading for predict mode

    Attributes:
        artifacts_dir: Artifact storage directory
        serializer: Artifact serializer
    """

    def persist(
        self,
        name: str,
        obj: Any,
        step_number: int,
        format_hint: Optional[str] = None
    ) -> ArtifactMeta:
        """
        Persist an artifact with content-addressed storage.

        Args:
            name: Artifact name for reference
            obj: Object to persist
            step_number: Current pipeline step
            format_hint: Optional serialization format

        Returns:
            ArtifactMeta with hash, path, and metadata
        """

    def load_for_step(self, step_number: int) -> List[Tuple[str, Any]]:
        """
        Load all artifacts for a specific step.

        Args:
            step_number: Step number to load artifacts for

        Returns:
            List of (name, object) tuples
        """
```

**Key improvements:**
- Encapsulates all artifact logic
- Testable independently
- Clear interface

---

#### 8. **Supporting Classes**

```python
@dataclass
class StepResult:
    """Result of executing a single step."""
    updated_context: Dict[str, Any]
    artifacts: List[ArtifactMeta]
    predictions: Optional[Predictions] = None

@dataclass
class ExecutionResult:
    """Result of executing a full pipeline."""
    predictions: Predictions
    artifacts: List[ArtifactMeta]
    dataset: SpectroDataset
    metadata: Dict[str, Any]

@dataclass
class OrchestrationResult:
    """Result of orchestrating multiple pipeline runs."""
    run_predictions: Predictions
    dataset_predictions: Dict[str, Any]
    execution_results: List[ExecutionResult]

@dataclass
class ParsedStep:
    """Normalized step configuration."""
    operator: Any
    keyword: str
    metadata: Dict[str, Any]
    step_type: StepType  # ENUM: MODEL, TRANSFORMER, SPLITTER, CHART, etc.
```

---

## Improved Naming Convention

### Current → Proposed

| Current | Proposed | Reason |
|---------|----------|--------|
| `run()` | `execute()` | More descriptive for orchestration |
| `_run_single()` | `PipelineExecutor.execute()` | Clear context |
| `run_step()` | `StepRunner.execute()` | Consistent naming |
| `run_steps()` | `PipelineExecutor._execute_steps()` | Private helper |
| `_normalize_pipeline()` | `_ensure_pipeline_configs()` | Clearer purpose |
| `_normalize_dataset()` | `_ensure_dataset_configs()` | Clearer purpose |
| `_select_controller()` | `ControllerRouter.route()` | Active verb |
| `_execute_controller()` | `StepRunner._dispatch()` | More specific |
| `next_op()` | `StepTracker.increment()` | Clear purpose |

---

## Performance Improvements

### 1. **Lazy Data Loading**
Current:
```python
# Copies entire dataset into memory
self.raw_data[dataset_name] = dataset.x({}, layout="2d")
```

Proposed:
```python
class DatasetView:
    """Lazy view over dataset without copying."""
    def __init__(self, dataset: SpectroDataset):
        self._dataset = dataset
        self._snapshots: Dict[str, weakref.ref] = {}

    def snapshot(self, key: str) -> np.ndarray:
        """Create named snapshot only when requested."""
        if key not in self._snapshots:
            self._snapshots[key] = weakref.ref(self._dataset.x({}, layout="2d"))
        return self._snapshots[key]()
```

### 2. **Streaming Predictions**
- Don't accumulate all predictions in memory
- Write to Parquet incrementally
- Use memory-mapped arrays for large outputs

### 3. **Controller Caching**
- Controllers are instantiated repeatedly
- Cache controller instances with LRU

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_controller(controller_class: Type[OperatorController]) -> OperatorController:
    """Get cached controller instance."""
    return controller_class()
```

### 4. **Parallel Dataset Processing**
- Current implementation has infrastructure but not fully utilized
- Enable true parallel execution across datasets (not pipelines within dataset)

```python
if orchestration_config.parallel_datasets:
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        results = executor.map(
            lambda ds: executor.execute(pipelines, ds),
            datasets
        )
```

---

## Migration Strategy

### Phase 1: Extract Supporting Classes (1-2 days)
- Create `ExecutionContext`, `StepResult`, `ExecutionResult`, etc.
- Create `StepTracker` helper
- Update current code to use these (minimal behavior change)

### Phase 2: Extract StepParser (2-3 days)
- Move all step parsing logic to `StepParser`
- Remove hardcoded operator lists
- Update `run_step()` to use parser

### Phase 3: Extract ControllerRouter (1-2 days)
- Move controller selection to `ControllerRouter`
- Centralize registry lookup
- Update tests

### Phase 4: Extract StepRunner (2-3 days)
- Create `StepRunner` with parser + router
- Move single-step execution logic
- Update `_run_single()` to use it

### Phase 5: Extract PipelineExecutor (2-3 days)
- Create `PipelineExecutor` with step runner
- Move single pipeline execution logic
- Update `run()` to use it

### Phase 6: Create PipelineOrchestrator (2-3 days)
- Rename `PipelineRunner` → `PipelineOrchestrator`
- Remove all extracted logic
- Final cleanup and tests

### Phase 7: Performance Optimizations (3-5 days)
- Implement lazy loading
- Add controller caching
- Optimize parallel execution

**Total estimated time: 2-3 weeks**

---

## Testing Strategy

### Unit Tests (per component)
```python
# Easy to test isolated components
def test_step_parser_handles_dict():
    parser = StepParser()
    result = parser.parse({"model": "SVC", "params": {"C": 1.0}})
    assert result.operator == SVC
    assert result.keyword == "model"

def test_controller_router_selects_sklearn():
    router = ControllerRouter()
    parsed = ParsedStep(operator=SVC(), keyword="model", ...)
    controller = router.route(parsed)
    assert isinstance(controller, SklearnModelController)

def test_step_runner_executes_transformer():
    runner = StepRunner(parser, router, context)
    result = runner.execute(step, dataset, ctx)
    assert result.updated_context["processing"] == [["standard_scaler"]]
```

### Integration Tests
```python
def test_pipeline_executor_runs_full_pipeline():
    executor = PipelineExecutor(context, artifacts, step_runner)
    result = executor.execute(steps, dataset)
    assert result.predictions.num_predictions > 0
    assert len(result.artifacts) > 0
```

### Backward Compatibility
- Keep old `PipelineRunner` as deprecated wrapper
- Redirect to new `PipelineOrchestrator` internally
- Issue deprecation warnings
- Remove in v0.6.0

---

## Documentation Structure

```
docs/
├── dev/
│   ├── RUNNER_REFACTORING_PROPOSAL.md (this file)
└── api/
    ├── PipelineOrchestrator.md
    ├── PipelineExecutor.md
    ├── StepRunner.md
    └── StepParser.md
```

### Google Style Docstrings Example
```python
def execute(
    self,
    steps: List[StepConfig],
    dataset: SpectroDataset
) -> ExecutionResult:
    """
    Execute pipeline steps sequentially on dataset.

    This method iterates through all pipeline steps, executing each
    via the StepRunner. It accumulates predictions and artifacts,
    returning a complete ExecutionResult.

    Args:
        steps: List of pipeline steps to execute. Each step is a
            normalized StepConfig object from the parser.
        dataset: SpectroDataset instance to operate on. The dataset
            will be modified in-place by transformers.

    Returns:
        ExecutionResult containing:
            - predictions: All predictions generated during execution
            - artifacts: All persisted artifacts (models, scalers, etc.)
            - dataset: Modified dataset after all transformations
            - metadata: Execution metadata (timing, errors, etc.)

    Raises:
        ExecutionError: If any step fails and continue_on_error=False
        ArtifactError: If artifact persistence fails

    Example:
        >>> executor = PipelineExecutor(context, artifacts, runner)
        >>> steps = [scaler_step, model_step]
        >>> result = executor.execute(steps, dataset)
        >>> print(result.predictions.num_predictions)
        150

    Note:
        Steps are executed sequentially. Parallel execution within
        a pipeline is not currently supported. Use PipelineOrchestrator
        for parallel pipeline execution across datasets.
    """
```

---

## Roadmap Integration

This refactoring aligns with roadmap items:

### ✅ **RELEASE 0.4.1**: Folder/File structure rc
- Clean separation of concerns
- Proper module organization

### ✅ **RELEASE 0.6.1**: Pipeline logic
- `[Pipeline] as single transformer`
- `[Runner] Design logic of 'execution sequence' and 'history'`
- `[Dummy_Controller] remove totally and manage exceptions`

This refactoring is a **prerequisite** for those releases and should be prioritized.

---

## Benefits Summary

### Maintainability
- ✅ Each class < 200 lines
- ✅ Single responsibility per class
- ✅ Clear boundaries and interfaces
- ✅ Easy to locate bugs

### Extensibility
- ✅ Add new step types without modifying runner
- ✅ Add new controllers via registry only
- ✅ Swap implementations easily (dependency injection)

### Testability
- ✅ Unit test each component independently
- ✅ Mock dependencies cleanly
- ✅ Fast test execution (no full runner needed)

### Readability
- ✅ Clear naming conventions
- ✅ Google-style docstrings
- ✅ Logical component hierarchy
- ✅ Self-documenting architecture

### Performance
- ✅ Lazy loading for large datasets
- ✅ Controller caching
- ✅ True parallel execution
- ✅ Reduced memory footprint

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking changes | High | Maintain backward-compatible wrapper |
| Migration time | Medium | Phased approach, test each phase |
| Learning curve | Low | Comprehensive docs + examples |
| Regression bugs | Medium | Extensive test suite, keep old tests |

---

## Conclusion

This refactoring transforms a 1050-line god class into a clean, maintainable architecture with clear responsibilities. The phased migration strategy minimizes risk while delivering immediate benefits.

**Recommendation: Approve and prioritize for 0.5.0 release.**

---

## Appendix: File Structure

```
nirs4all/
├── pipeline/
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── orchestrator.py           # PipelineOrchestrator
│   │   └── config.py                 # OrchestrationConfig
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── executor.py               # PipelineExecutor
│   │   ├── context.py                # ExecutionContext
│   │   └── result.py                 # ExecutionResult, StepResult
│   ├── steps/
│   │   ├── __init__.py
│   │   ├── runner.py                 # StepRunner
│   │   ├── parser.py                 # StepParser
│   │   └── router.py                 # ControllerRouter
│   ├── artifacts/
│   │   ├── __init__.py
│   │   ├── manager.py                # ArtifactManager
│   │   └── serialization.py          # (existing)
│   ├── config.py                     # PipelineConfigs (existing)
│   ├── manifest_manager.py           # (existing)
│   └── runner.py                     # DEPRECATED: backward compat wrapper
```

---

**Document Version**: 1.0
**Date**: 2025-10-31
**Author**: GitHub Copilot (Claude Sonnet 4.5)
**Status**: Proposal - Awaiting Review
