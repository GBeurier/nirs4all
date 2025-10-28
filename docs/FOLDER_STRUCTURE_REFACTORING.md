## Proposed Structure with Operator-Controller Logic

```
nirs4all/
├── nirs4all/
│   │
│   ├── core/                          # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── validation.py
│   │
│   ├── operators/                     # ALL OPERATORS (kept!)
│   │   ├── __init__.py
│   │   ├── base.py                    # NEW: Base Operator ABC
│   │   │
│   │   ├── transforms/                # Transformation operators
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Base transformer
│   │   │   ├── signal/                # Signal processing
│   │   │   │   ├── __init__.py
│   │   │   │   ├── baseline.py        # Detrend, MSC
│   │   │   │   ├── derivatives.py     # Derivate, FirstDerivative
│   │   │   │   ├── filters.py         # Gaussian, SavitzkyGolay
│   │   │   │   └── wavelets.py        # Haar, Wavelet
│   │   │   ├── nirs/                  # NIRS-specific
│   │   │   │   ├── __init__.py
│   │   │   │   ├── normalization.py   # SNV, RSNV, LSNV
│   │   │   │   ├── correction.py      # MSC, EMSC, AreaNormalization
│   │   │   │   └── presets.py
│   │   │   ├── scalers.py
│   │   │   ├── features.py
│   │   │   ├── targets.py
│   │   │   └── resampler.py
│   │   │
│   │   ├── models/                    # Model operators
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Base model operator
│   │   │   ├── sklearn/
│   │   │   │   └── __init__.py
│   │   │   ├── tensorflow/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── nicon.py
│   │   │   │   ├── generic.py
│   │   │   │   └── legacy/
│   │   │   │       ├── __init__.py
│   │   │   │       ├── unet*.py
│   │   │   │       ├── resnet*.py
│   │   │   │       └── ...
│   │   │   └── pytorch/
│   │   │       ├── __init__.py
│   │   │       ├── nicon.py
│   │   │       └── generic.py
│   │   │
│   │   ├── splitters/                 # Split operators
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── splitters.py
│   │   │
│   │   ├── augmentation/              # Augmentation operators
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # ABC augmenter
│   │   │   ├── random.py
│   │   │   └── splines.py
│   │   │
│   │   ├── charts/                    # Chart/Visualization operators
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Base chart operator
│   │   │   ├── fold_charts.py
│   │   │   ├── spectra_charts.py
│   │   │   └── y_chart.py
│   │   │
│   │   ├── data/                      # Data manipulation operators
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── branch.py
│   │   │   ├── merge.py
│   │   │   ├── resampler.py
│   │   │   ├── sample_augmentation.py
│   │   │   └── feature_augmentation.py
│   │   │
│   │   └── flow/                      # Flow control operators
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── condition.py
│   │       ├── scope.py
│   │       ├── sequential.py
│   │       └── dummy.py               # Logging operator
│   │
│   ├── controllers/                   # CONTROLLERS for operators
│   │   ├── __init__.py
│   │   ├── base.py                    # Base Controller ABC
│   │   ├── registry.py                # Controller registry (operator → controller mapping)
│   │   ├── presets.py
│   │   │
│   │   ├── transforms/                # Transform controllers
│   │   │   ├── __init__.py
│   │   │   ├── transformer.py         # TransformerMixin controller
│   │   │   ├── y_transformer.py       # Y-transformer controller
│   │   │   └── cluster.py             # ClusterMixin controller
│   │   │
│   │   ├── models/                    # Model controllers
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py          # Base model controller
│   │   │   ├── helper.py              # Model controller helpers
│   │   │   ├── optuna.py              # Optuna integration
│   │   │   ├── sklearn_model.py       # Sklearn model controller
│   │   │   ├── tensorflow_model.py    # TF model controller
│   │   │   ├── torch_model.py         # PyTorch model controller
│   │   │   └── results/
│   │   │
│   │   ├── splitters/                 # Split controllers
│   │   │   ├── __init__.py
│   │   │   └── split.py               # Split controller
│   │   │
│   │   ├── charts/                    # Chart controllers
│   │   │   ├── __init__.py
│   │   │   └── chart.py               # Chart controller
│   │   │
│   │   ├── data/                      # Data manipulation controllers
│   │   │   ├── __init__.py
│   │   │   └── data.py                # Data controller
│   │   │
│   │   └── flow/                      # Flow control controllers
│   │       ├── __init__.py
│   │       └── flow.py                # Flow controller
│   │
│   ├── data/                          # Dataset management (NOT operators)
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── config.py
│   │   ├── config_parser.py
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   ├── csv_loader.py
│   │   │   └── loader.py
│   │   ├── features.py
│   │   ├── feature_source.py
│   │   ├── targets.py
│   │   ├── indexer.py
│   │   ├── metadata.py
│   │   ├── helpers.py
│   │   ├── io.py
│   │   ├── evaluator.py
│   │   ├── predictions.py
│   │   └── analyzers.py
│   │
│   ├── pipeline/                      # Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── runner.py                  # Uses registry to get controllers
│   │   ├── generator.py
│   │   ├── serialization.py
│   │   ├── io.py
│   │   ├── binary_loader.py
│   │   └── manifest_manager.py
│   │
│   ├── optimization/                  # Hyperparameter tuning
│   │   ├── __init__.py
│   │   ├── optuna_backend.py
│   │   └── strategies.py
│   │
│   ├── workspace/                     # Workspace management
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   ├── library.py
│   │   ├── runs.py
│   │   └── schemas.py
│   │
│   ├── visualization/                 # Analysis & reporting (NOT operators)
│   │   ├── __init__.py
│   │   ├── predictions.py
│   │   ├── heatmaps.py
│   │   ├── pca.py
│   │   ├── shap.py
│   │   └── reports.py
│   │
│   ├── utils/                         # Utilities
│   │   ├── __init__.py
│   │   ├── backend.py
│   │   ├── metrics.py
│   │   ├── emoji.py
│   │   ├── spinner.py
│   │   ├── io.py
│   │   ├── serialization.py
│   │   ├── binning.py
│   │   ├── balancing.py
│   │   ├── model_builder.py
│   │   └── model_utils.py
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── commands/
│   │   │   ├── __init__.py
│   │   │   ├── workspace.py
│   │   │   ├── run.py
│   │   │   └── export.py
│   │   └── test_install.py
│   │
│   └── __init__.py
│
├── tests/
├── examples/
├── docs/
└── ...
```

## Key Architectural Concepts Made Explicit

### 1. **Operator-Controller Relationship**

```python
# operators/base.py
class Operator(ABC):
    """Base class for all operators in the pipeline."""

    @abstractmethod
    def get_controller_type(self) -> str:
        """Return the type of controller that handles this operator."""
        pass

# controllers/base.py
class Controller(ABC):
    """Base class for all controllers."""

    @abstractmethod
    def can_handle(self, operator: Operator) -> bool:
        """Check if this controller can handle the given operator."""
        pass

    @abstractmethod
    def execute(self, operator: Operator, context: dict) -> Any:
        """Execute the operator within the pipeline context."""
        pass

# controllers/registry.py
class ControllerRegistry:
    """Maps operators to their controllers."""

    _registry: Dict[Type[Operator], Type[Controller]] = {}

    @classmethod
    def register(cls, operator_type: Type[Operator],
                 controller_type: Type[Controller]):
        """Register a controller for an operator type."""
        cls._registry[operator_type] = controller_type

    @classmethod
    def get_controller(cls, operator: Operator) -> Controller:
        """Get the appropriate controller for an operator."""
        for op_type, ctrl_type in cls._registry.items():
            if isinstance(operator, op_type):
                return ctrl_type()
        raise ValueError(f"No controller found for {type(operator)}")
```

### 2. **Parallel Structure**

```
operators/
  ├── transforms/          ←→  controllers/transforms/
  ├── models/              ←→  controllers/models/
  ├── splitters/           ←→  controllers/splitters/
  ├── charts/              ←→  controllers/charts/
  ├── data/                ←→  controllers/data/
  └── flow/                ←→  controllers/flow/
```

### 3. **Clear Separation of Concerns**

**Operators** (`operators/`):
- Define **WHAT** to do
- Are the building blocks in pipelines
- Are framework-agnostic definitions
- Examples: `SNV()`, `RandomForestRegressor()`, `FoldChart()`

**Controllers** (`controllers/`):
- Define **HOW** to execute operators
- Handle framework-specific logic
- Manage state, validation, and execution
- Examples: `TransformerController`, `SklearnModelController`, `ChartController`

**Data** (`data/`):
- NOT operators, but data structures
- Dataset loading, predictions storage
- No controller needed

**Visualization** (`visualization/`):
- Analysis tools (post-pipeline)
- NOT pipeline operators
- Used for reports and exploration

### 4. **Documentation Structure**

```
docs/
├── architecture/
│   ├── operator_controller_pattern.md    # NEW: Explain the pattern
│   ├── operator_catalog.md               # All available operators
│   ├── controller_catalog.md             # NEW: All controllers
│   └── creating_operators.md             # NEW: How to create new operators
│
├── api/
│   ├── operators/
│   │   ├── transforms.md
│   │   ├── models.md
│   │   ├── charts.md
│   │   └── flow.md
│   └── controllers/
│       ├── base.md
│       ├── registry.md
│       └── custom_controllers.md
```

## Benefits of This Structure

✅ **Pattern Visibility**: Operator-controller relationship is obvious
✅ **Parallel Structure**: Easy to find corresponding controller for an operator
✅ **Future-proof**: Charts become operators without restructuring
✅ **Clear Semantics**: operators/ contains ONLY pipeline operators
✅ **Extensibility**: Adding new operator type = add to both folders
✅ **Documentation**: Architecture is self-documenting

## Example Usage

```python
# In user code
from nirs4all.operators.transforms.nirs import SNV
from nirs4all.operators.models.sklearn import RandomForest
from nirs4all.operators.charts import FoldChart

pipeline = [
    SNV(),              # Transform operator
    RandomForest(),     # Model operator
    FoldChart()         # Chart operator (will be in future)
]

# In pipeline runner
from nirs4all.controllers.registry import ControllerRegistry

for operator in pipeline:
    controller = ControllerRegistry.get_controller(operator)
    result = controller.execute(operator, context)
```

## Migration Notes

```python
# Current structure compatibility layer
# nirs4all/operators/transformations/__init__.py
"""DEPRECATED: Use nirs4all.operators.transforms instead."""
from nirs4all.operators.transforms import *

# nirs4all/controllers/sklearn/op_transformermixin.py
"""DEPRECATED: Misnamed - this is a controller, not an operator.
Use nirs4all.controllers.transforms.transformer instead."""
from nirs4all.controllers.transforms.transformer import *
```

## Folder Naming Rationale

- `operators/` - Kept as-is, core concept
- `controllers/` - Kept as-is, core concept
- `operators/charts/` - Future operators (charts will become operators)
- `operators/flow/` - Control flow operators (condition, scope, sequential)
- `controllers/` subfolders match `operators/` subfolders
- `data/` and `visualization/` are NOT operators (separate concern)




## Solution Critique: Operator-Controller Pattern

### ✅ Strengths

1. **Architectural Honesty**
   - Structure reflects actual design
   - Operator-controller pattern is explicit
   - Code organization matches mental model

2. **Parallel Clarity**
   ```
   operators/transforms/  ←→  controllers/transforms/
   operators/models/      ←→  controllers/models/
   operators/charts/      ←→  controllers/charts/
   ```
   - Relationship is obvious
   - Easy to find corresponding controller
   - Self-documenting structure

3. **Domain Vocabulary**
   - "Operators" is YOUR term, keep it
   - Established in nirs4all community
   - Consistent with existing documentation

4. **Future-Proof**
   - Charts naturally become `operators/charts/`
   - Flow control already has a home
   - Scales with your actual architecture

5. **Contributor Clarity**
   - New operators → add to `operators/`
   - Need controller → add to `controllers/`
   - Clear separation of concerns

6. **Educational**
   - Forces understanding of pattern
   - Better for research/academic use
   - Showcases unique architecture

### ❌ Weaknesses

1. **Steep Learning Curve**
   - Must understand operator-controller pattern to navigate
   - Not intuitive for sklearn/torch users
   - More concepts to grasp upfront

2. **More Folders**
   - Top-level: `operators/`, `controllers/`, `data/`, `visualization/`
   - Can feel cluttered
   - Harder to scan quickly

3. **Duplication Feel**
   - `operators/transforms/` + `controllers/transforms/` feels redundant
   - Why not just `transforms/`?
   - Requires explanation in docs

4. **Non-Standard**
   - Doesn't match industry conventions
   - Harder to compare with other libraries
   - May feel "weird" to outsiders

5. **Documentation Burden**
   - Must explain operator-controller pattern prominently
   - More architectural docs needed
   - Higher barrier to entry

6. **Three Categories of Things**
   - Operators (pipeline building blocks)
   - Controllers (execution handlers)
   - Others (data, visualization)
   - Adds complexity


### Critical Requirement: Documentation

To make Solution 2 work, you MUST:

1. **Prominent README section**
   ```markdown
   ## Architecture: Operator-Controller Pattern

   nirs4all uses a unique operator-controller architecture:
   - **Operators** (`operators/`) define WHAT to do (e.g., SNV, RandomForest)
   - **Controllers** (`controllers/`) define HOW to execute them
   - This enables flexible, extensible pipelines with consistent interfaces
   ```

2. **Visual diagram in docs**
   ```
   User Pipeline Definition
           ↓
      [Operators]  ← User-facing building blocks
           ↓
      [Registry]   ← Maps operators to controllers
           ↓
     [Controllers] ← Execution logic
           ↓
      [Results]
   ```

3. **Quick start that abstracts it**
   ```python
   # Users don't need to think about controllers
   from nirs4all.operators.transforms.nirs import SNV
   from nirs4all.operators.models.sklearn import PLS

   pipeline = [SNV(), PLS(n_components=10)]
   runner.run(pipeline)  # Controllers handled automatically
   ```
