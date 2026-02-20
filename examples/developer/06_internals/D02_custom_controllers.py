"""
D02 - Custom Controllers: Extending Pipeline Execution
=======================================================

Controllers are the execution engine of nirs4all pipelines.
Each step in a pipeline is handled by a matching controller.

This tutorial covers:

* Controller architecture overview
* Creating custom controllers
* Controller registration and priority
* Execution context and runtime context
* Prediction mode support

Prerequisites
-------------
- D01_session_workflow for session management

Duration: ~10 minutes
Difficulty: ★★★★★
"""

# Standard library imports
import argparse
from typing import Any, Optional

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.controllers import OperatorController, register_controller
from nirs4all.controllers.registry import CONTROLLER_REGISTRY

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D02 Custom Controllers Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D02 - Custom Controllers: Extending Pipeline Execution")
print("=" * 60)

print("""
nirs4all uses a controller registry pattern:

    Pipeline Step → Controller Matcher → Execute

Controllers handle:
  - Preprocessing transforms
  - Model training/prediction
  - Data splitting/merging
  - Custom operations

Each controller:
  - Declares what it handles (matches)
  - Has a priority (lower = higher priority)
  - Implements execute() for training
  - Optionally supports prediction mode
""")

# =============================================================================
# Section 1: Controller Architecture
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Controller Architecture")
print("-" * 60)

print("""
📦 Controller Components:

    OperatorController (base class)
    ├── priority: int           # Matching priority
    ├── matches(step, op, kw)   # Whether to handle this step
    ├── execute(...)            # Training execution
    └── supports_prediction_mode() → bool

Built-in controllers:
  - TransformController (sklearn transformers)
  - ModelController (model training)
  - SplitterController (cross-validation)
  - BranchController (branching logic)
  - MergeController (merging logic)
  - GeneratorController (pipeline expansion)
""")

# List available controllers
print("\nRegistered controllers (first 10):")
for controller in CONTROLLER_REGISTRY[:10]:
    print(f"  {controller.__name__}: priority={controller.priority}")

# =============================================================================
# Section 2: Simple Custom Controller
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Simple Custom Controller")
print("-" * 60)

print("""
Create a custom controller by inheriting OperatorController:

    @register_controller
    class MyController(OperatorController):
        priority = 50

        @classmethod
        def matches(cls, step, operator, keyword):
            return keyword == "my_operation"

        def execute(self, step_info, dataset, context, ...):
            # Custom logic here
            return dataset
""")

class PrintDatasetInfo:
    """Custom operator that prints dataset information."""

    def __init__(self, message: str = "Dataset Info"):
        self.message = message

@register_controller
class PrintDatasetInfoController(OperatorController):
    """Controller for PrintDatasetInfo operator.

    Note: This is a simplified example. Production controllers must:
    - Return (context, artifacts_list) tuple
    - Handle multi-source data properly
    - Store fitted state for prediction mode
    """

    priority = 45  # Higher priority than default transformers

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        """Match PrintDatasetInfo operators."""
        return isinstance(operator, PrintDatasetInfo)

    @classmethod
    def use_multi_source(cls) -> bool:
        """Whether this controller handles multi-source datasets."""
        return False

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        """Execute the print info operation.

        Returns:
            Tuple of (context, artifacts_list)
        """
        operator = step_info.operator
        # Access features using dataset.x() method with proper selector
        x_data = dataset.x(context.selector, "2d")
        y_data = dataset.y(context.selector)
        print(f"\n  [{operator.message}]")
        print(f"    X shape: {x_data.shape}")
        print(f"    y shape: {y_data.shape if y_data is not None else 'None'}")

        # Return (updated_context, artifacts_list)
        return context, []

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """This controller runs during prediction too."""
        return True

print("\nCustom controller registered: PrintDatasetInfoController")

# =============================================================================
# Section 3: Using Custom Controllers
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Using Custom Controllers")
print("-" * 60)

print("""
Use custom operators in pipelines:

    pipeline = [
        PrintDatasetInfo("Before scaling"),
        MinMaxScaler(),
        PrintDatasetInfo("After scaling"),
        PLSRegression()
    ]
""")

# Demonstrate a simple pipeline with custom controller
pipeline = [
    PrintDatasetInfo("Data info"),
    MinMaxScaler(),
    ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
    PLSRegression(n_components=5),
]

print("\nRunning pipeline with custom controller:")
result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    verbose=0
)

print(f"\nResult: {result.num_predictions} predictions")

# =============================================================================
# Section 4: Keyword-Based Controllers
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Keyword-Based Controllers")
print("-" * 60)

print("""
Controllers can match pipeline keywords:

    {"my_transform": CustomTransform()}

The controller matches based on the keyword:

    @classmethod
    def matches(cls, step, operator, keyword):
        return keyword == "my_transform"
""")

class LogTransformOperator:
    """Apply log transform to features."""

    def __init__(self, offset: float = 1.0):
        self.offset = offset

@register_controller
class LogTransformController(OperatorController):
    """Controller for log transformation keyword."""

    priority = 40

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        """Match 'log_transform' keyword."""
        return bool(keyword == "log_transform")

    @classmethod
    def use_multi_source(cls) -> bool:
        """Whether this controller handles multi-source datasets."""
        return False

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        """Apply log transformation.

        Returns:
            Tuple of (context, artifacts_list)
        """
        operator = step_info.operator
        offset = getattr(operator, 'offset', 1.0) if hasattr(operator, 'offset') else 1.0

        print(f"  Applying log transform with offset={offset}")
        # Note: In real implementation, use dataset.x(context.selector, "2d")
        # This is a demonstration of keyword-based matching
        return context, []

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True

print("LogTransformController registered for 'log_transform' keyword")

# =============================================================================
# Section 5: Controller Priority
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Controller Priority")
print("-" * 60)

print("""
Priority determines controller matching order:

    priority 10  ← First checked (highest priority)
    priority 20
    priority 50  ← Default
    priority 100 ← Last checked (lowest priority)

Use lower priority to override built-in controllers:

    class MyTransformController(OperatorController):
        priority = 1  # Will be checked before defaults
""")

print("\nController priorities (lower = higher priority):")
print("  0-19:  Reserved for system controllers")
print("  20-39: High-priority custom controllers")
print("  40-59: Standard custom controllers")
print("  60-79: Default built-in controllers")
print("  80-99: Fallback controllers")

# =============================================================================
# Section 6: Execution Context
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Execution Context")
print("-" * 60)

print("""
Controllers receive context objects:

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):

    step_info:
        .step          # Original step definition
        .operator      # The operator instance
        .keyword       # Step keyword ('model', 'preprocessing', etc.)
        .step_index    # Position in pipeline

    context:
        .pipeline_config   # Pipeline configuration
        .execution_mode    # 'train' or 'predict'
        .random_state      # For reproducibility

    runtime_context:
        .artifacts         # Store trained states
        .metrics           # Collect metrics
        .fold_index        # Current fold (during CV)
""")

class ContextAwareOperator:
    """Operator that uses context information."""
    pass

@register_controller
class ContextAwareController(OperatorController):
    """Demonstrates context usage."""

    priority = 45

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return isinstance(operator, ContextAwareOperator)

    @classmethod
    def use_multi_source(cls) -> bool:
        """Whether this controller handles multi-source datasets."""
        return False

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        """Show context information.

        Returns:
            Tuple of (context, artifacts_list)
        """
        print(f"\n  Step index: {step_info.step_index}")
        print(f"  Keyword: {step_info.keyword}")
        print(f"  Mode: {context.execution_mode}")
        print(f"  Fold: {getattr(runtime_context, 'fold_index', 'N/A')}")
        return context, []

print("ContextAwareController registered")

# =============================================================================
# Section 7: Stateful Controllers
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Stateful Controllers (Conceptual)")
print("-" * 60)

print("""
Controllers can save state for prediction by storing
fitted transformers in the runtime_context:

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        # Fit during training
        transformer = MyTransformer()
        transformer.fit(dataset.X)

        # Store for later prediction
        runtime_context.fitted_steps[step_info.step_index] = transformer

        # Transform
        dataset.X = transformer.transform(dataset.X)
        return dataset

During prediction, the fitted transformer is loaded
and used without refitting.
""")

print("\nStateful patterns:")
print("  - Sklearn transformers use fit/transform")
print("  - Custom state via runtime_context")
print("  - Models saved to artifacts for export")

# =============================================================================
# Section 8: Prediction Mode
# =============================================================================
print("\n" + "-" * 60)
print("Example 8: Prediction Mode")
print("-" * 60)

print("""
Controllers can run differently in prediction mode:

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True

    def execute(self, step_info, dataset, context, ...):
        if context.execution_mode == 'predict':
            # Prediction logic (use saved state)
            return self._predict(dataset, context)
        else:
            # Training logic (compute and save state)
            return self._train(dataset, context)

Controllers without prediction support are skipped
during prediction (e.g., CV splitters).
""")

print("\nPrediction mode behaviors:")
print("  TransformController: Uses fitted transformer")
print("  ModelController: Uses trained model")
print("  SplitterController: Skipped (no folds in prediction)")
print("  Custom: Depends on supports_prediction_mode()")

# =============================================================================
# Section 9: Controller Registration
# =============================================================================
print("\n" + "-" * 60)
print("Example 9: Controller Registration")
print("-" * 60)

print("""
Register controllers with the decorator:

    @register_controller
    class MyController(OperatorController):
        ...

The @register_controller decorator:
  1. Validates the class inherits OperatorController
  2. Adds it to CONTROLLER_REGISTRY
  3. Sorts by priority

Controllers are automatically discovered if:
  - Defined in nirs4all.controllers.*
  - Decorated with @register_controller
""")

# Show registry info
print("\nRegistry information:")
print(f"  Total registered: {len(CONTROLLER_REGISTRY)} controllers")
print("  Import: from nirs4all.controllers import register_controller")
print("  Access: from nirs4all.controllers.registry import CONTROLLER_REGISTRY")

# =============================================================================
# Section 10: Complete Custom Controller
# =============================================================================
print("\n" + "-" * 60)
print("Example 10: Complete Custom Controller")
print("-" * 60)

print("""
Complete example: A debug logging controller

    class DebugLog:
        def __init__(self, label: str = "Debug"):
            self.label = label
""")

class DebugLog:
    """Simple debug logging operator."""

    def __init__(self, label: str = "Debug"):
        self.label = label

@register_controller
class DebugLogController(OperatorController):
    """Controller for debug logging."""

    priority = 5  # Very high priority - checked early

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return isinstance(operator, DebugLog)

    @classmethod
    def use_multi_source(cls) -> bool:
        """Whether this controller handles multi-source datasets."""
        return False

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        """Log debug information.

        Returns:
            Tuple of (context, artifacts_list)
        """
        operator = step_info.operator
        # Access data using proper dataset API
        x_data = dataset.x(context.selector, "2d")
        y_data = dataset.y(context.selector)

        print(f"\n  🔍 [{operator.label}] Step {step_info.step_index}")
        print(f"      Mode: {context.execution_mode}")
        print(f"      X: {x_data.shape}, y: {y_data.shape if y_data is not None else 'None'}")
        return context, []

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True

print("DebugLogController registered")
print("\nUsage in pipeline:")
print('  [DebugLog("After preprocessing"), MinMaxScaler(), ...]')

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. Controllers handle pipeline step execution
2. @register_controller decorator for registration
3. matches() determines what steps a controller handles
4. priority controls matching order (lower = higher priority)
5. execute() runs during training
6. supports_prediction_mode() enables prediction use
7. Context objects provide execution information

Controller template:

    @register_controller
    class MyController(OperatorController):
        priority = 50

        @classmethod
        def matches(cls, step, operator, keyword) -> bool:
            return isinstance(operator, MyOperator)

        def execute(self, step_info, dataset, context, runtime_context, **kwargs):
            # Implementation
            return dataset

        @classmethod
        def supports_prediction_mode(cls) -> bool:
            return True  # If needed during prediction

This concludes the Developer track examples!
See the main examples/ directory for more usage patterns.
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
