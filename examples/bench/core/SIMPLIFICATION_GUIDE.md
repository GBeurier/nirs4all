# Pipeline Context Simplification Guide

## Overview

This document explains the simplification of the pipeline context system, moving from a complex `PipelineContext` class to a simple dictionary-based approach.

## Problem with Original System

The original system had several issues:

1. **Over-complexity**: `PipelineContext` was 449 lines with complex scope management, push/pop semantics, and nested filtering
2. **Hard to understand**: Complex interactions between `DataSelector`, `ScopeRule`, and `PipelineContext`
3. **Hard to debug**: Multiple layers of abstraction made it difficult to trace data selection
4. **Performance overhead**: Complex context tracking and filtering

## New Simplified Approach

### Core Principle
**Operations handle their own data selection using simple, explicit filters.**

### Key Changes

1. **Context is just a dict**: `{"branch": 0}` instead of complex `PipelineContext`
2. **Operations select data directly**: `dataset.select(partition="train", branch=context["branch"])`
3. **No complex scoping**: Operations specify exactly what data they want
4. **Clear and explicit**: Easy to understand what data each operation uses

### Implementation

#### Before (Complex):
```python
# In PipelineContext (449 lines)
class PipelineContext:
    def __init__(self):
        self.current_filters = {}
        self.scope_stack = []
        self.branch_stack = [0]
        # ... 440 more lines of complexity

# In operation:
def execute(self, dataset, context: PipelineContext):
    fit_view = dataset.select(partition=self.fit_partition, **context.current_filters)
```

#### After (Simple):
```python
# Simple context is just a dict
context = {"branch": 0}

# In operation:
def execute(self, dataset, context: Dict[str, Any]):
    branch = context.get('branch', 0)
    fit_view = dataset.select(partition=self.fit_partition, branch=branch)
```

### Benefits

1. **Clarity**: Operations explicitly show what data they operate on
2. **Simplicity**: Context is just branch information
3. **Maintainability**: No complex scope management to debug
4. **Performance**: No overhead from complex context tracking
5. **Flexibility**: Operations can implement custom selection logic easily
6. **Debuggability**: Easy to see exactly what data is being selected

### Migration Path

To migrate existing operations:

1. Change operation signature:
   ```python
   # Before
   def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:

   # After
   def execute(self, dataset: SpectraDataset, context: Dict[str, Any]) -> None:
   ```

2. Replace complex context usage:
   ```python
   # Before
   view = dataset.select(partition="train", **context.current_filters)

   # After
   branch = context.get('branch', 0)
   view = dataset.select(partition="train", branch=branch)
   ```

3. Remove DataSelector usage:
   ```python
   # Before
   fit_filters = self.data_selector.get_enhanced_scope(operation, context, phase='fit')

   # After
   # Operations handle their own selection logic directly
   ```

### Examples

#### Simple Transformation Operation:
```python
class OperationTransformation(PipelineOperation):
    def execute(self, dataset: SpectraDataset, context: Dict[str, Any]) -> None:
        # Get branch from simple context
        branch = context.get('branch', 0)

        # Direct, explicit data selection
        fit_view = dataset.select(partition=self.fit_partition, branch=branch)

        # ... rest of operation logic
```

#### Simple Pipeline Runner:
```python
class SimplePipelineRunner:
    def __init__(self):
        self.context = {"branch": 0}  # Simple dict

    def _run_branch(self, branch_config, dataset):
        # Save and restore branch context
        old_branch = self.context["branch"]
        self.context["branch"] = new_branch_id
        # ... execute branch steps
        self.context["branch"] = old_branch
```

## File Changes Made

1. **PipelineOperation.py**: Updated base class to use `Dict[str, Any]` instead of `PipelineContext`
2. **OperationTranformation.py**: Simplified to use basic branch-based selection
3. **SimplePipelineRunner.py**: New simplified runner with basic branching support
4. **SimpleDataSelector.py**: Basic selector that just passes through context

## Removed Complexity

- **PipelineContext.py**: 449 lines of complex scope management
- **DataSelector.py**: Complex scoping rules for different operation types
- Complex push/pop scope semantics
- Advanced filtering and data selection rules
- Source management and augmentation tracking
- Centroid and clustering context

## Result

The pipeline system is now:
- **10x simpler**: Context is just `{"branch": 0}`
- **More transparent**: Easy to see what data operations use
- **Easier to debug**: No complex scope stack to trace
- **More maintainable**: Operations handle their own selection logic
- **Better performance**: No overhead from complex context tracking

This change makes the pipeline system much more approachable for new developers and easier to extend with custom operations.
