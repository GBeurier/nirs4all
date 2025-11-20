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

---



# PREVIOUS VERSION

nirs4all/
├── nirs4all/                          # Core library
│   │
│   ├── core/                          # NEW: Core functionality
│   │   ├── __init__.py
│   │   ├── config.py                  # Global configuration
│   │   ├── exceptions.py              # Custom exceptions
│   │   ├── constants.py               # Constants
│   │   └── validation.py              # Input validation
│   │
│   ├── data/                          # RENAMED: dataset/ → data/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # SpectroDataset class
│   │   ├── config.py                  # DatasetConfigs (from dataset_config.py)
│   │   ├── config_parser.py           # Config parsing (from dataset_config_parser.py)
│   │   ├── loaders/                   # NEW: Data loading utilities
│   │   │   ├── __init__.py
│   │   │   ├── csv_loader.py          # MOVED from dataset/
│   │   │   └── loader.py              # MOVED from dataset/
│   │   ├── features.py
│   │   ├── feature_source.py
│   │   ├── targets.py
│   │   ├── indexer.py
│   │   ├── metadata.py
│   │   ├── helpers.py
│   │   ├── io.py
│   │   ├── evaluator.py
│   │   ├── predictions.py             # Predictions storage
│   │   └── analyzers.py               # RENAMED: prediction_analyzer.py → analyzers.py
│   │
│   ├── pipeline/                      # Pipeline management
│   │   ├── __init__.py
│   │   ├── config.py                  # Pipeline configuration
│   │   ├── runner.py                  # Pipeline execution
│   │   ├── generator.py               # Pipeline generation
│   │   ├── serialization.py           # Save/load pipelines
│   │   ├── io.py                      # I/O operations
│   │   ├── binary_loader.py           # Binary artifacts
│   │   └── manifest_manager.py        # Manifest management
│   │
│   ├── transforms/                    # RENAMED: operators/ → transforms/
│   │   ├── __init__.py
│   │   ├── base.py                    # NEW: Base transformer classes
│   │   ├── signal/                    # SPLIT: Signal processing transforms
│   │   │   ├── __init__.py
│   │   │   ├── baseline.py            # NEW: Detrend, baseline correction
│   │   │   ├── derivatives.py         # NEW: Derivate, FirstDerivative, SecondDerivative
│   │   │   ├── filters.py             # NEW: Gaussian, SavitzkyGolay
│   │   │   └── wavelets.py            # NEW: Haar, Wavelet
│   │   ├── nirs/                      # NIRS-specific transforms
│   │   │   ├── __init__.py
│   │   │   ├── normalization.py       # NEW: SNV, RSNV, LSNV
│   │   │   ├── correction.py          # NEW: MSC, EMSC, AreaNormalization
│   │   │   └── presets.py             # MOVED: presets.py
│   │   ├── augmentation/              # Sample/feature augmentation
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # RENAMED: abc_augmenter.py → base.py
│   │   │   ├── random.py
│   │   │   └── splines.py
│   │   ├── splitters/                 # Data splitting
│   │   │   ├── __init__.py
│   │   │   └── splitters.py
│   │   ├── scalers.py                 # MOVED from transformations/
│   │   ├── features.py                # MOVED from transformations/
│   │   ├── targets.py                 # MOVED from transformations/
│   │   └── resampler.py               # MOVED from transformations/
│   │
│   ├── models/                        # Model definitions
│   │   ├── __init__.py
│   │   ├── sklearn/                   # NEW: Sklearn model wrappers
│   │   │   └── __init__.py
│   │   ├── tensorflow/                # RENAMED: cirad_tf.py, generic_tf.py
│   │   │   ├── __init__.py
│   │   │   ├── nicon.py               # RENAMED: cirad_tf.py → nicon.py
│   │   │   ├── generic.py             # RENAMED: generic_tf.py → generic.py
│   │   │   └── legacy/                # RENAMED: legacy_tf/ → legacy/
│   │   │       ├── __init__.py
│   │   │       ├── unet.py
│   │   │       ├── unet_plus.py
│   │   │       ├── unet_plus_plus.py
│   │   │       ├── unet3_plus.py
│   │   │       ├── resnet_1d_cnn.py
│   │   │       ├── resnet_v2_1d_cnn.py
│   │   │       ├── resnext_1d_cnn.py
│   │   │       ├── se_resnet_1d_cnn.py
│   │   │       ├── seresnext_1d_cnn.py
│   │   │       ├── vgg_1d_cnn.py
│   │   │       ├── inception_1d_cnn.py
│   │   │       ├── albunet.py
│   │   │       ├── ternausnet.py
│   │   │       ├── fpn.py
│   │   │       ├── pspnet.py
│   │   │       ├── bcdunet.py
│   │   │       ├── sedunet.py
│   │   │       ├── ibaunet.py
│   │   │       ├── multires_unet.py
│   │   │       ├── ensembled_unet.py
│   │   │       └── dense_inception_unet.py
│   │   └── pytorch/                   # RENAMED: cirad_torch.py, generic_torch.py
│   │       ├── __init__.py
│   │       ├── nicon.py               # RENAMED: cirad_torch.py → nicon.py
│   │       ├── generic.py             # RENAMED: generic_torch.py → generic.py
│   │       └── legacy/                # Future legacy PyTorch models
│   │           └── __init__.py
│   │
│   ├── controllers/                   # Execution controllers
│   │   ├── __init__.py
│   │   ├── registry.py                # Controller registry
│   │   ├── controller.py              # Base controller
│   │   ├── presets.py                 # Preset configurations
│   │   ├── sklearn/                   # Sklearn-specific controllers
│   │   │   ├── __init__.py
│   │   │   ├── model.py               # RENAMED: op_model.py → model.py
│   │   │   ├── split.py               # RENAMED: op_split.py → split.py
│   │   │   ├── transformer.py         # RENAMED: op_transformermixin.py → transformer.py
│   │   │   ├── y_transformer.py       # RENAMED: op_y_transformermixin.py → y_transformer.py
│   │   │   └── cluster.py             # RENAMED: op_clustermixin.py → cluster.py
│   │   ├── tensorflow/                # TensorFlow-specific controllers
│   │   │   ├── __init__.py
│   │   │   └── model.py               # RENAMED: op_model.py → model.py
│   │   ├── torch/                     # PyTorch-specific controllers
│   │   │   ├── __init__.py
│   │   │   └── model.py               # RENAMED: op_model.py → model.py
│   │   ├── models/                    # Model management controllers
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # RENAMED: base_model_controller.py → base.py
│   │   │   ├── helper.py              # RENAMED: model_controller_helper.py → helper.py
│   │   │   ├── optuna.py              # RENAMED: optuna_manager.py → optuna.py
│   │   │   └── results/               # Model results storage
│   │   ├── data/                      # RENAMED: dataset/ → data/
│   │   │   ├── __init__.py
│   │   │   ├── branch.py              # RENAMED: op_branch.py → branch.py
│   │   │   ├── merge.py               # RENAMED: op_merge.py → merge.py
│   │   │   ├── resampler.py           # RENAMED: op_resampler.py → resampler.py
│   │   │   ├── sample_augmentation.py # RENAMED: op_sample_augmentation.py
│   │   │   └── feature_augmentation.py # RENAMED: op_feature_augmentation.py
│   │   ├── path/                      # Path/flow control controllers
│   │   │   ├── __init__.py
│   │   │   ├── condition.py           # RENAMED: op_condition.py → condition.py
│   │   │   ├── scope.py               # RENAMED: op_scope.py → scope.py
│   │   │   └── sequential.py          # RENAMED: op_sequential.py → sequential.py
│   │   ├── visualization/             # RENAMED: chart/ → visualization/
│   │   │   ├── __init__.py
│   │   │   ├── fold_charts.py         # RENAMED: op_fold_charts.py
│   │   │   ├── spectra_charts.py      # RENAMED: op_spectra_charts.py
│   │   │   └── y_chart.py             # RENAMED: op_y_chart.py
│   │   └── logging/                   # RENAMED: log/ → logging/
│   │       ├── __init__.py
│   │       └── dummy.py               # RENAMED: op_dummy.py → dummy.py
│   │
│   ├── optimization/                  # NEW: Hyperparameter tuning
│   │   ├── __init__.py
│   │   ├── optuna_backend.py          # Optuna integration
│   │   └── strategies.py              # CV strategies
│   │
│   ├── workspace/                     # Workspace management
│   │   ├── __init__.py
│   │   ├── manager.py                 # RENAMED: library_manager.py → manager.py
│   │   ├── library.py                 # NEW: Library-specific operations
│   │   ├── runs.py                    # NEW: Run tracking
│   │   └── schemas.py                 # NEW: Pydantic models
│   │
│   ├── visualization/                 # NEW: Plotting utilities
│   │   ├── __init__.py
│   │   ├── predictions.py             # Prediction plots (from PredictionAnalyzer)
│   │   ├── heatmaps.py                # Heatmap utilities
│   │   ├── pca.py                     # MOVED: PCA_analyzer.py → pca.py
│   │   ├── shap.py                    # MOVED: shap_analyzer.py → shap.py
│   │   └── reports.py                 # MOVED: tab_report_manager.py → reports.py
│   │
│   ├── utils/                         # Utilities
│   │   ├── __init__.py
│   │   ├── backend.py                 # RENAMED: backend_utils.py → backend.py
│   │   ├── metrics.py                 # NEW: Metric calculations
│   │   ├── emoji.py                   # Console output
│   │   ├── spinner.py                 # Progress indicators
│   │   ├── io.py                      # NEW: I/O helpers
│   │   ├── serialization.py           # RENAMED: serializer.py → serialization.py
│   │   ├── binning.py                 # Discretization
│   │   ├── balancing.py               # Data balancing
│   │   ├── model_builder.py           # Model building utilities
│   │   └── model_utils.py             # Model utilities
│   │
│   ├── cli/                           # Command-line interface
│   │   ├── __init__.py
│   │   ├── main.py                    # CLI entry point
│   │   ├── commands/                  # NEW: Command modules
│   │   │   ├── __init__.py
│   │   │   ├── workspace.py           # MOVED: workspace_commands.py → workspace.py
│   │   │   ├── run.py                 # NEW: Run commands
│   │   │   └── export.py              # NEW: Export commands
│   │   └── test_install.py            # Installation tests
│   │
│   ├── transformations/               # DEPRECATED: Backward compatibility
│   │   └── __init__.py                # Re-exports from transforms/
│   │
│   └── __init__.py                    # Main package init
│
├── tests/                             # Test suite
│   ├── conftest.py                    # Pytest configuration
│   ├── run_tests.py
│   ├── run_runner_tests.py
│   ├── REFACTORING_CHECKLIST.md
│   ├── unit/                          # Unit tests (ORGANIZED)
│   │   ├── __init__.py
│   │   ├── data/                      # NEW: Data module tests
│   │   │   ├── __init__.py
│   │   │   ├── test_dataset.py
│   │   │   ├── test_config.py
│   │   │   ├── test_predictions.py
│   │   │   ├── test_metadata.py
│   │   │   └── test_loaders.py
│   │   ├── pipeline/                  # NEW: Pipeline tests
│   │   │   ├── __init__.py
│   │   │   ├── test_runner.py
│   │   │   ├── test_config.py
│   │   │   ├── test_generator.py
│   │   │   └── test_serialization.py
│   │   ├── transforms/                # NEW: Transform tests
│   │   │   ├── __init__.py
│   │   │   ├── test_signal.py
│   │   │   ├── test_nirs.py
│   │   │   ├── test_augmentation.py
│   │   │   └── test_splitters.py
│   │   ├── models/                    # NEW: Model tests
│   │   │   ├── __init__.py
│   │   │   ├── test_tensorflow.py
│   │   │   └── test_pytorch.py
│   │   ├── test_binning.py
│   │   ├── test_balancing.py
│   │   ├── test_balancing_value_aware.py
│   │   ├── test_indexer_augmentation.py
│   │   ├── test_transformer_mixin_augmentation.py
│   │   ├── test_split_controller_augmentation.py
│   │   └── test_sample_augmentation_controller.py
│   ├── integration/                   # Integration tests (ORGANIZED)
│   │   ├── __init__.py
│   │   ├── test_phase2_integration.py
│   │   ├── test_dataset_augmentation.py
│   │   └── test_augmentation_end_to_end.py
│   ├── integration_tests/             # Full integration tests (CONSOLIDATED)
│   │   ├── __init__.py
│   │   ├── run_integration_tests.py
│   │   ├── test_basic_pipeline.py
│   │   ├── test_classification_integration.py
│   │   ├── test_multisource_integration.py
│   │   ├── test_groupsplit_integration.py
│   │   ├── test_flexible_inputs_integration.py
│   │   ├── test_resampler.py
│   │   ├── test_shap_integration.py
│   │   ├── test_sample_augmentation_integration.py
│   │   ├── test_prediction_reuse_integration.py
│   │   ├── test_pca_analysis_integration.py
│   │   ├── test_finetune_integration.py
│   │   └── test_comprehensive_integration.py
│   ├── dataset/                       # Dataset-specific tests (MOVE TO unit/data/)
│   │   ├── __init__.py
│   │   ├── test_metadata.py
│   │   ├── test_metadata_loading.py
│   │   ├── test_group_split_validation.py
│   │   ├── test_header_units_step1.py
│   │   ├── test_header_units_step2.py
│   │   ├── test_header_units_step3.py
│   │   ├── test_header_units_step4.py
│   │   ├── test_header_units_step5.py
│   │   ├── test_header_units_step6.py
│   │   ├── test_header_units_step7.py
│   │   └── test_header_units_step8.py
│   ├── pipeline/                      # Pipeline-specific tests (MOVE TO unit/pipeline/)
│   │   ├── __init__.py
│   │   ├── test_binary_loader.py
│   │   └── test_manifest_manager.py
│   ├── pipeline_runner/               # Runner tests (CONSOLIDATE)
│   │   ├── __init__.py
│   │   ├── test_comprehensive.py
│   │   ├── test_normalization.py
│   │   ├── test_state.py
│   │   ├── test_regression_prevention.py
│   │   └── test_predict.py
│   ├── serialization/                 # Serialization tests (MOVE TO unit/pipeline/)
│   │   ├── __init__.py
│   │   ├── test_pipeline_serialization.py
│   │   └── test_serializer.py
│   ├── workspace/                     # Workspace tests
│   │   ├── __init__.py
│   │   ├── test_phase2_catalog_export.py
│   │   ├── test_phase3_library_manager.py
│   │   └── test_phase4_query_reporting.py
│   ├── utils/                         # Utils tests (MOVE TO unit/)
│   │   ├── __init__.py
│   │   └── test_data_generator.py
│   └── fixtures/                      # NEW: Test data & fixtures
│       ├── datasets/
│       │   ├── sample_data/
│       │   └── regression_data/
│       └── pipelines/
│           └── test_configs/
│
├── examples/                          # Example scripts
│   ├── README.md                      # NEW: Examples index
│   ├── basic/                         # NEW: Basic examples
│   │   ├── __init__.py
│   │   ├── Q1_regression.py           # MOVED
│   │   ├── Q1_classif.py              # MOVED
│   │   ├── Q1_groupsplit.py           # MOVED
│   │   ├── Q2_multimodel.py           # MOVED
│   │   └── sample.py                  # MOVED
│   ├── advanced/                      # NEW: Advanced examples
│   │   ├── __init__.py
│   │   ├── Q3_finetune.py             # MOVED
│   │   ├── Q4_multidatasets.py        # MOVED
│   │   ├── Q5_predict.py              # MOVED
│   │   ├── Q6_multisource.py          # MOVED
│   │   ├── Q7_discretization.py       # MOVED
│   │   ├── Q8_shap.py                 # MOVED
│   │   ├── Q9_acp_spread.py           # MOVED
│   │   ├── Q10_resampler.py           # MOVED
│   │   ├── Q11_flexible_inputs.py     # MOVED
│   │   ├── Q12_sample_augmentation.py # MOVED
│   │   ├── Q13_nm_headers.py          # MOVED
│   │   ├── Q14_workspace.py           # MOVED
│   │   ├── metadata_usage.py          # MOVED
│   │   ├── sample_augmentation_examples.py # MOVED
│   │   ├── example_resampler.py       # MOVED
│   │   └── example_sample.py          # MOVED
│   ├── deep_learning/                 # NEW: Deep learning examples
│   │   ├── __init__.py
│   │   ├── Q1_classif_tf.py           # MOVED
│   │   ├── Q5_predict_NN.py           # MOVED
│   │   ├── DL_regression_models.ipynb # MOVED
│   │   ├── custom_NN.py               # MOVED
│   │   └── nicon_custom.py            # MOVED
│   ├── notebooks/                     # NEW: Tutorial notebooks
│   │   ├── Tutorial_1_Beginners_Guide.ipynb # MOVED
│   │   └── Tutorial_2_Advanced_Analysis.ipynb # MOVED
│   ├── data/                          # Example datasets
│   │   ├── sample_data/               # MOVED
│   │   └── regression_data/           # MOVED
│   ├── workspace/                     # Workspace examples
│   ├── generated/                     # Generated outputs
│   ├── out/                           # Output figures
│   ├── DLregression_best_model.csv/
│   ├── log.txt
│   ├── run_Q_examples.ps1
│   ├── test.py                        # DEPRECATE: Test files
│   ├── test2.py                       # DEPRECATE
│   ├── test_hiba.py                   # DEPRECATE
│   └── test_hiba_full.py              # DEPRECATE
│
├── docs/                              # Documentation
│   ├── .readthedocs.yaml              # ReadTheDocs config
│   ├── readthedocs.requirements.txt
│   ├── make.bat
│   ├── user_guide/                    # NEW: User documentation
│   │   ├── installation.md            # NEW
│   │   ├── quickstart.md              # NEW
│   │   ├── tutorials/                 # NEW
│   │   │   ├── basic_regression.md
│   │   │   ├── classification.md
│   │   │   └── deep_learning.md
│   │   ├── preprocessing.md           # MOVED: Preprocessing.md
│   │   ├── preprocessing_cheatsheet.md # MOVED: PREPROCESSING_CHEAT_SHEET.md
│   │   └── sample_augmentation.md     # MOVED: SAMPLE_AUGMENTATION_QUICK_REFERENCE.md
│   ├── api/                           # NEW: API reference
│   │   ├── data.md                    # MOVED: ANALYSIS_DATASET_MODULE.md
│   │   ├── pipeline.md                # MOVED: ANALYSIS_PIPELINE_MODULE.md
│   │   ├── transforms.md              # NEW
│   │   ├── models.md                  # NEW
│   │   └── workspace.md               # MOVED: WORKSPACE_ARCHITECTURE.md
│   ├── developer/                     # NEW: RENAMED from dev/
│   │   ├── architecture.md            # NEW
│   │   ├── contributing.md            # NEW
│   │   ├── testing.md                 # NEW
│   │   └── refactoring_checklist.md   # MOVED from tests/
│   ├── specifications/                # NEW: Technical specs
│   │   ├── pipeline_syntax.md         # MOVED: PIPELINE_SYNTAX_COMPLETE_GUIDE.md
│   │   ├── metrics.md                 # MOVED: METRICS_SPECIFICATIONS.md
│   │   ├── serialization.md           # MOVED: SERIALIZATION_COMPLETE.md
│   │   ├── manifest.md                # MOVED: MANIFEST_ARCHITECTURE.md
│   │   ├── nested_cv.md               # MOVED: NESTED_CROSS_VALIDATION.md
│   │   ├── cross_dataset_metrics.md   # MOVED: CROSS_DATASET_METRICS_EXPLANATION.md
│   │   ├── group_split.md             # MOVED: GROUP_SPLIT_QUICK_REFERENCE.md
│   │   ├── hash_uniqueness.md         # MOVED: HASH_BASED_UNIQUENESS.md
│   │   └── workspace_serialization.md # MOVED: WORKSPACE_SERIALIZATION.md
│   ├── explanations/                  # NEW: Conceptual explanations
│   │   ├── snv.md                     # MOVED: SNV_EXPLANATION.md
│   │   ├── shap.md                    # MOVED: SHAP_EXPLANATION.md
│   │   ├── pls_study.md               # MOVED: PLS_Study.md
│   │   ├── resampler.md               # MOVED: RESAMPLER.md
│   │   └── metadata.md                # MOVED: METADATA_USAGE.md
│   ├── reference/                     # NEW: Reference materials
│   │   ├── operator_catalog.md        # MOVED: Operator_Catalog.md
│   │   ├── combination_generator.md   # MOVED: COMBINATION_GENERATOR.md
│   │   ├── outputs_vs_artifacts.md    # MOVED: OUTPUTS_VS_ARTIFACTS.md
│   │   └── writing_pipelines.md       # MOVED: WRITING_A_PIPELINE.md
│   ├── cli/                           # NEW: CLI documentation
│   │   ├── installation_testing.md    # MOVED: CLI_INSTALLATION_TESTING.md
│   │   └── workspace_commands.md      # MOVED: CLI_WORKSPACE_COMMANDS.md
│   ├── integration/                   # NEW: Integration docs
│   │   └── service_api.md             # MOVED: Service_api.MD
│   ├── reports/                       # NEW: Project reports
│   │   ├── workspace_complete.md      # MOVED: WORKSPACE_COMPLETE_REPORT.md
│   │   ├── workspace_implementation.md # MOVED: WORKSPACE_IMPLEMENTATION_SUMMARY.md
│   │   ├── workspace_integration.md   # MOVED: WORKSPACE_INTEGRATION_COMPLETE.md
│   │   ├── workspace_test_execution.md # MOVED: WORKSPACE_TEST_EXECUTION_SUMMARY.md
│   │   ├── serialization_executive.md # MOVED: SERIALIZATION_EXECUTIVE_SUMMARY.md
│   │   ├── serialization_implementation.md # MOVED: SERIALIZATION_IMPLEMENTATION_SUMMARY.md
│   │   ├── serialization_refactoring.md # MOVED: SERIALIZATION_REFACTORING.md
│   │   ├── predictions_refactoring.md # MOVED: PREDICTIONS_REFACTORING_ANALYSIS.md
│   │   ├── prediction_results_summary.md # MOVED: PREDICTION_RESULTS_LIST_SUMMARY.md
│   │   └── task_type_refactoring.md   # MOVED: TASK_TYPE_REFACTORING_PLAN.md
│   ├── assets/                        # NEW: Images and assets
│   │   ├── nirs4all_logo.png          # MOVED
│   │   ├── pipeline.jpg               # MOVED
│   │   ├── heatmap.png                # MOVED
│   │   ├── candlestick.png            # MOVED
│   │   ├── stacking.png               # MOVED from source/
│   │   └── models_preprocessing.pdf   # MOVED: Models and Preprocessing Strategies.pdf
│   ├── source/                        # Sphinx source
│   │   ├── conf.py
│   │   ├── index.rst
│   │   ├── api.rst
│   │   ├── augmentation.rst
│   │   ├── model_selection.rst
│   │   ├── preprocessing.rst
│   │   ├── simple_pipelines.rst
│   │   ├── sklearn.rst
│   │   └── stacking.rst
│   ├── dev/                           # DEPRECATED: Old dev docs location
│   ├── pp.md                          # DEPRECATE or MOVE
│   ├── MERGE_SERIALIZATION.txt        # DEPRECATE
│   └── WORKSPACE_ARCHITECTURE_ROADMAP.md # MOVE to reports/
│
├── scripts/                           # Utility scripts
│   ├── gc_artifacts.py                # Garbage collection
│   ├── setup_dev.py                   # NEW: Development setup
│   ├── generate_docs.py               # NEW: Documentation generation
│   └── migrate_structure.py           # NEW: Migration helper
│
├── LICENSES/                          # Third-party licenses
│
├── pyproject.toml                     # Modern Python packaging
├── requirements.txt
├── requirements-test.txt
├── requirements-examples.txt
├── pytest.ini
├── Makefile
├── README.md
├── README_LICENSE.md
├── README_LICENSE_FR.md
├── LICENSE
├── LICENSE_FR
├── Roadmap.md
├── INSTALLATION.md
├── CONTRIBUTING.md
├── CONTRIBUTING_FR.md
├── CODE_OF_CONDUCT.md
├── CLA.md
├── CLA_FR.md
├── THIRD_PARTY_NOTICES.md
├── THIRD_PARTY_NOTICES_FR.md
├── CHANGELOG.md                       # NEW: Version history
└── .gitignore