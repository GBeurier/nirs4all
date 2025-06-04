# Clean Pipeline Architecture Implementation

## Problem Solved

The original pipeline had mixed responsibilities and lengthy operation instantiations. The new design provides:

1. **Clear separation of concerns**
2. **Unified building mechanism**
3. **Simple operation wrappers**
4. **Reusable across parallel branches**

## Architecture Overview

```
PipelineRunner (execution + context)
    ↓ delegates building to
PipelineBuilder (generic building + serialization)
    ↓ creates
PipelineOperation (simple wrappers)
    ↓ wraps
Actual Operators (sklearn, tf, custom)
```

## Key Components

### 1. PipelineRunner - Pure Execution Logic

```python
def _run_step(self, step, dataset, prefix=""):
    """MAIN PARSING LOGIC - Clean execution with builder delegation"""

    # CONTROL OPERATIONS - handled locally
    if isinstance(step, str) and step == "uncluster":
        self._run_uncluster(dataset, prefix)
    elif isinstance(step, list):
        self._run_sub_pipeline(step, dataset, prefix)
    # ... other control operations

    else:
        # DATA OPERATIONS - delegate to builder
        operation = self.builder.build_operation(step)
        self._execute_operation(operation, dataset, prefix)

def _execute_operation(self, operation, dataset, prefix):
    """Simple select and execute"""
    filtered_dataset = dataset.select(**self.context.current_filters)
    operation.execute(filtered_dataset, self.context)
```

**Responsibilities:**
- Parse config structure transparently
- Manage context and filters
- Handle control operations locally
- Delegate data operations to builder
- Simple select → execute pattern

### 2. PipelineBuilder - Generic Building

```python
def build_operation(self, step) -> PipelineOperation:
    """Generic operation builder - handles any step type"""

    if isinstance(step, str):
        return self._build_from_string(step)  # presets
    elif inspect.isclass(step):
        return self._build_from_class(step)   # classes to instantiate
    elif hasattr(step, '__class__'):
        return self._build_from_instance(step)  # instances to clone
    elif isinstance(step, dict):
        return self._build_from_dict(step)    # generic pipeline dict
```

**Responsibilities:**
- Handle all step formats uniformly
- Use serialization for cloning/reuse
- Manage presets and defaults
- Wrap operators in appropriate PipelineOperation

### 3. PipelineOperation - Simple Wrappers

```python
class TransformationOperation(PipelineOperation):
    def __init__(self, transformer, mode="transformation"):
        self.transformer = transformer
        self.mode = mode

    def execute(self, dataset, context):
        """Simple fit and transform"""
        # Fit on appropriate partition
        # Transform specified partitions
        # Handle different modes (transformation, augmentation)
```

**Responsibilities:**
- Simple wrapper around actual operator
- Handle fit/transform patterns
- Support different execution modes
- Use PipelineOperation interface

## Benefits Achieved

### ✅ **Clear Runner Logic**
- Main parsing loop exposes config structure
- Control vs data operations clearly separated
- No lengthy instantiation code
- Context management explicit

### ✅ **Generic Building**
- Handles all step types uniformly
- Serialization ensures reusability across branches
- Presets managed centrally
- Type-safe wrapping

### ✅ **Simple Operations**
- Focused responsibility: just wrap and execute
- Clean API through PipelineOperation interface
- Easy to test and debug
- Minimal code duplication

### ✅ **Maintainable Design**
- Each component has single responsibility
- Easy to extend with new step types
- Clear error boundaries
- Visible execution flow

## Usage Examples

```python
# All these work uniformly:
pipeline = [
    "StandardScaler",                                    # string preset
    sklearn.preprocessing.MinMaxScaler(),                # instance
    sklearn.decomposition.PCA,                          # class
    {"class": "sklearn.cluster.KMeans", "params": {"n_clusters": 3}},  # generic dict
    {"sample_augmentation": ["NoiseAugmenter"]},         # control dict
    ["PCA", "StandardScaler"],                           # sub-pipeline
    "uncluster",                                         # control command
]

runner = PipelineRunner()
runner.run_pipeline(pipeline, dataset)
```

## Implementation Files

- `PipelineRunner_clean.py` - Clean runner implementation
- `PipelineBuilder_clean.py` - Generic builder with serialization
- `TransformationOperation_clean.py` - Simple transformation wrapper
- `demo_clean_architecture.py` - Working demonstration

This architecture provides the exact design you requested: **clean runner with context management, generic builder, and simple operation wrappers**.
