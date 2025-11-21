# File Saving Architecture Report

## Executive Summary

The file saving architecture in `nirs4all` has been refactored to separate concerns between **Controllers** (logic) and **Executors** (I/O). This "Return, Don't Save" pattern ensures that controllers are pure, testable components that return data objects (`StepOutput`) instead of writing directly to the file system.

## Problem Statement

Previously, controllers were tightly coupled to the file system:
1. **Direct I/O:** Controllers called `runner.saver.save_output()` directly.
2. **Hard to Test:** Unit tests required mocking the file system or the `saver` object.
3. **Inconsistent Returns:** Some controllers returned artifacts, others returned paths, others returned nothing.
4. **Side Effects:** Execution of a controller had the side effect of creating files, making "dry runs" difficult to implement cleanly.

## Solution: "Return, Don't Save" Pattern

The solution implements a clear separation of concerns:

1. **Controllers** generate data and return it in a standardized `StepOutput` container.
2. **PipelineExecutor** iterates through the `StepOutput` and handles the actual persistence using the `SimulationSaver`.

### The `StepOutput` Class

```python
@dataclass
class StepOutput:
    """Standardized output from a controller execution."""
    # Internal binaries (models, transformers)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # User outputs (charts, reports)
    # List of tuples: (data_object, filename_hint, type_hint)
    outputs: List[Tuple[Any, str, str]] = field(default_factory=list)
```

### Controller Implementation

Controllers now focus solely on logic and data generation:

```python
# OLD: Coupled to I/O
def execute(self, ...):
    # ... generate chart ...
    runner.saver.save_output("chart", data, "png")
    return context, []

# NEW: Pure Logic
def execute(self, ...):
    # ... generate chart ...
    return context, StepOutput(
        outputs=[(data, "chart", "png")]
    )
```

### Executor Implementation

The `PipelineExecutor` handles the side effects:

```python
# In PipelineExecutor._execute_steps
step_result = controller.execute(...)

# Handle Outputs
for output_data, name, ext in step_result.outputs:
    self.saver.save_output(name=name, data=output_data, extension=ext)

# Handle Artifacts
for name, artifact in step_result.artifacts.items():
    self.saver.persist_artifact(step_number, name, artifact)
```

## Benefits

1. **Testability:** Controllers can be tested by asserting on the returned `StepOutput` object. No file system mocks needed.
2. **Flexibility:** The executor can easily implement "dry run" modes or redirect outputs to different storage backends (e.g., S3) without changing controller code.
3. **Consistency:** All controllers now have a uniform return signature.
4. **Clean Architecture:** Separation of business logic (controllers) from infrastructure concerns (file I/O).

## Migration Status

All core controllers have been migrated to this pattern:
- ✅ `SpectraChartController`
- ✅ `YChartController`
- ✅ `FoldChartController`
- ✅ `CrossValidatorController` (Splitter)
- ✅ `BaseModelController` (and subclasses)

## Conclusion

The "Return, Don't Save" architecture significantly improves the maintainability and robustness of the `nirs4all` library. It aligns with best practices for software design by decoupling logic from side effects.
