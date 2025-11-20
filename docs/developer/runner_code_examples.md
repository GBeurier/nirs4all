# PipelineRunner Refactoring - Code Examples

**Related Documents**:
- [Runner Refactoring Proposal](RUNNER_REFACTORING_PROPOSAL.md)
- [Architecture Diagrams](RUNNER_ARCHITECTURE_DIAGRAM.md)
- [Context Refactoring Proposal](CONTEXT_REFACTORING_PROPOSAL.md) ⚠️ **CRITICAL DEPENDENCY**

---

## ⚠️ IMPORTANT: ExecutionContext Superseded

The `ExecutionContext` defined in this document is **INSUFFICIENT** for production use. It was designed for pipeline metadata (mode, workspace, step tracking) but does NOT address the comprehensive context requirements.

**The actual context implementation is defined in [CONTEXT_REFACTORING_PROPOSAL.md](CONTEXT_REFACTORING_PROPOSAL.md)**, which handles:
- Data selection (partition, processing chains, layout)
- Pipeline state (y transformations, evolving processing)
- Controller communication (flags, coordination)

**This document's ExecutionContext** covers only pipeline-level metadata and should be considered a **subset** of the full context architecture.

---

## 1. Pipeline Metadata Context (Subset)

**NOTE**: This is NOT the complete context. See CONTEXT_REFACTORING_PROPOSAL.md for DataSelector, PipelineState, and StepMetadata.

```python
"""
nirs4all/pipeline/execution/context.py

Immutable execution context for pipeline runs.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from nirs4all.pipeline.manifest_manager import ManifestManager


class ExecutionMode(Enum):
    """Execution mode for pipeline runs."""
    TRAIN = "train"
    PREDICT = "predict"
    EXPLAIN = "explain"


@dataclass(frozen=True)
class StepTracker:
    """
    Mutable step tracking (separated from immutable context).

    Tracks current position in pipeline execution.

    Attributes:
        step_number: Current main step (1-indexed)
        substep_number: Current substep within main step (0-indexed)
        operation_count: Total operations executed
    """
    step_number: int = 0
    substep_number: int = 0
    operation_count: int = 0

    def increment_step(self) -> 'StepTracker':
        """
        Increment main step counter.

        Returns:
            New StepTracker with incremented step_number
        """
        return StepTracker(
            step_number=self.step_number + 1,
            substep_number=0,
            operation_count=self.operation_count
        )

    def increment_substep(self) -> 'StepTracker':
        """
        Increment substep counter.

        Returns:
            New StepTracker with incremented substep_number
        """
        return StepTracker(
            step_number=self.step_number,
            substep_number=self.substep_number + 1,
            operation_count=self.operation_count
        )

    def increment_operation(self) -> 'StepTracker':
        """
        Increment operation counter.

        Returns:
            New StepTracker with incremented operation_count
        """
        return StepTracker(
            step_number=self.step_number,
            substep_number=self.substep_number,
            operation_count=self.operation_count + 1
        )

    @property
    def step_id(self) -> str:
        """Get step identifier string (e.g., '1_0' for step 1, substep 0)."""
        return f"{self.step_number}_{self.substep_number}"


@dataclass(frozen=True)
class ExecutionContext:
    """
    Immutable execution context for a pipeline run.

    Contains all execution parameters that don't change during execution.
    Thread-safe due to immutability.

    Attributes:
        mode: Execution mode (TRAIN, PREDICT, EXPLAIN)
        workspace_path: Workspace root directory
        pipeline_uid: Unique pipeline identifier
        manifest_manager: Artifact manifest manager
        verbose: Verbosity level (0=quiet, 1=normal, 2=detailed, 3=debug)
        step_tracker: Mutable step tracking
        continue_on_error: Whether to continue execution if a step fails
        plots_visible: Whether to display plots during execution
        config: Additional configuration options

    Example:
        >>> context = ExecutionContext(
        ...     mode=ExecutionMode.TRAIN,
        ...     workspace_path=Path("workspace"),
        ...     pipeline_uid="0001_abc123",
        ...     manifest_manager=manager,
        ...     verbose=1
        ... )
        >>> new_context = context.with_incremented_step()
    """

    mode: ExecutionMode
    workspace_path: Path
    pipeline_uid: str
    manifest_manager: ManifestManager
    verbose: int = 0
    step_tracker: StepTracker = field(default_factory=StepTracker)
    continue_on_error: bool = False
    plots_visible: bool = False
    config: Dict[str, Any] = field(default_factory=dict)

    def with_incremented_step(self) -> 'ExecutionContext':
        """
        Create new context with incremented step counter.

        Returns:
            New ExecutionContext with updated step tracking
        """
        return ExecutionContext(
            mode=self.mode,
            workspace_path=self.workspace_path,
            pipeline_uid=self.pipeline_uid,
            manifest_manager=self.manifest_manager,
            verbose=self.verbose,
            step_tracker=self.step_tracker.increment_step(),
            continue_on_error=self.continue_on_error,
            plots_visible=self.plots_visible,
            config=self.config
        )

    def with_incremented_substep(self) -> 'ExecutionContext':
        """
        Create new context with incremented substep counter.

        Returns:
            New ExecutionContext with updated substep tracking
        """
        return ExecutionContext(
            mode=self.mode,
            workspace_path=self.workspace_path,
            pipeline_uid=self.pipeline_uid,
            manifest_manager=self.manifest_manager,
            verbose=self.verbose,
            step_tracker=self.step_tracker.increment_substep(),
            continue_on_error=self.continue_on_error,
            plots_visible=self.plots_visible,
            config=self.config
        )

    @property
    def is_training(self) -> bool:
        """Check if context is in training mode."""
        return self.mode == ExecutionMode.TRAIN

    @property
    def is_prediction(self) -> bool:
        """Check if context is in prediction mode."""
        return self.mode == ExecutionMode.PREDICT

    @property
    def is_explanation(self) -> bool:
        """Check if context is in explanation mode."""
        return self.mode == ExecutionMode.EXPLAIN
```

---

## 2. StepParser (Extract Parsing Logic)

```python
"""
nirs4all/pipeline/steps/parser.py

Parse pipeline step configurations into normalized format.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from nirs4all.pipeline.serialization import deserialize_component


class StepType(Enum):
    """Type of pipeline step."""
    MODEL = "model"
    TRANSFORMER = "transformer"
    SPLITTER = "splitter"
    CHART = "chart"
    AUGMENTATION = "augmentation"
    PREPROCESSING = "preprocessing"
    UNKNOWN = "unknown"


@dataclass
class ParsedStep:
    """
    Normalized step configuration.

    Attributes:
        operator: Deserialized operator instance or class
        keyword: Step keyword (e.g., "model", "preprocessing")
        step_type: Detected type of step
        metadata: Additional metadata extracted from step config
        original: Original step configuration (for debugging)
    """
    operator: Any
    keyword: str
    step_type: StepType
    metadata: Dict[str, Any]
    original: Any


class StepParseError(Exception):
    """Raised when step parsing fails."""
    pass


class StepParser:
    """
    Parse pipeline step configurations into normalized format.

    Handles multiple step syntaxes:
    - Dictionary: {"model": SVC, "params": {...}}
    - String: "sklearn.preprocessing.StandardScaler"
    - Direct instance: StandardScaler()
    - Nested lists: [[step1, step2], step3]

    Normalizes to canonical ParsedStep format for controller execution.

    Example:
        >>> parser = StepParser()
        >>> step = {"model": "sklearn.svm.SVC", "params": {"C": 1.0}}
        >>> parsed = parser.parse(step)
        >>> print(parsed.keyword)
        'model'
        >>> print(parsed.operator)
        <sklearn.svm.SVC object>
    """

    # Keywords that indicate specific step types
    STEP_TYPE_KEYWORDS = {
        "model": StepType.MODEL,
        "preprocessing": StepType.PREPROCESSING,
        "split": StepType.SPLITTER,
        "chart_2d": StepType.CHART,
        "chart_3d": StepType.CHART,
        "fold_chart": StepType.CHART,
        "y_chart": StepType.CHART,
        "sample_augmentation": StepType.AUGMENTATION,
        "feature_augmentation": StepType.AUGMENTATION,
    }

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
        if step is None:
            return ParsedStep(
                operator=None,
                keyword="noop",
                step_type=StepType.UNKNOWN,
                metadata={},
                original=step
            )

        if isinstance(step, dict):
            return self._parse_dict(step)
        elif isinstance(step, list):
            return self._parse_list(step)
        elif isinstance(step, str):
            return self._parse_string(step)
        else:
            return self._parse_instance(step)

    def _parse_dict(self, step: Dict[str, Any]) -> ParsedStep:
        """
        Parse dictionary step configuration.

        Handles formats:
        - {"model": "SVC", "params": {...}}
        - {"class": "StandardScaler", "params": {...}}
        - {"preprocessing": StandardScaler()}

        Args:
            step: Dictionary step configuration

        Returns:
            ParsedStep with extracted operator and metadata
        """
        # Check for workflow operator keywords
        for keyword in self.STEP_TYPE_KEYWORDS.keys():
            if keyword in step:
                operator_config = step[keyword]

                # Deserialize if needed
                if isinstance(operator_config, dict):
                    if '_runtime_instance' in operator_config:
                        operator = operator_config['_runtime_instance']
                    else:
                        operator = deserialize_component(operator_config)
                elif isinstance(operator_config, str):
                    operator = deserialize_component(operator_config)
                else:
                    operator = operator_config

                # Extract metadata (params, etc.)
                metadata = {k: v for k, v in step.items() if k != keyword}

                return ParsedStep(
                    operator=operator,
                    keyword=keyword,
                    step_type=self.STEP_TYPE_KEYWORDS.get(keyword, StepType.UNKNOWN),
                    metadata=metadata,
                    original=step
                )

        # Check for serialization format ({"class": ..., "params": ...})
        if "class" in step or "function" in step or "instance" in step:
            operator = deserialize_component(step)
            return ParsedStep(
                operator=operator,
                keyword="",
                step_type=self._infer_type_from_operator(operator),
                metadata={},
                original=step
            )

        # Unknown dict format
        raise StepParseError(f"Cannot parse dict step: {step}")

    def _parse_list(self, step: List[Any]) -> ParsedStep:
        """
        Parse list step configuration (sub-pipeline).

        Args:
            step: List of substeps

        Returns:
            ParsedStep marked as sub-pipeline
        """
        return ParsedStep(
            operator=step,
            keyword="subpipeline",
            step_type=StepType.UNKNOWN,
            metadata={"num_substeps": len(step)},
            original=step
        )

    def _parse_string(self, step: str) -> ParsedStep:
        """
        Parse string step configuration.

        Handles:
        - Module paths: "sklearn.preprocessing.StandardScaler"
        - Keyword references: "model StandardScaler"

        Args:
            step: String step configuration

        Returns:
            ParsedStep with deserialized operator
        """
        # Check if string contains keyword
        for keyword in self.STEP_TYPE_KEYWORDS.keys():
            if keyword in step:
                # Extract class/function name after keyword
                parts = step.split()
                if len(parts) > 1:
                    operator = deserialize_component(parts[1])
                else:
                    operator = deserialize_component(step)

                return ParsedStep(
                    operator=operator,
                    keyword=keyword,
                    step_type=self.STEP_TYPE_KEYWORDS.get(keyword, StepType.UNKNOWN),
                    metadata={},
                    original=step
                )

        # Plain module path
        operator = deserialize_component(step)
        return ParsedStep(
            operator=operator,
            keyword="",
            step_type=self._infer_type_from_operator(operator),
            metadata={},
            original=step
        )

    def _parse_instance(self, step: Any) -> ParsedStep:
        """
        Parse direct instance step configuration.

        Args:
            step: Direct operator instance

        Returns:
            ParsedStep with instance as operator
        """
        return ParsedStep(
            operator=step,
            keyword="",
            step_type=self._infer_type_from_operator(step),
            metadata={},
            original=step
        )

    def _infer_type_from_operator(self, operator: Any) -> StepType:
        """
        Infer step type from operator instance.

        Args:
            operator: Operator instance to inspect

        Returns:
            StepType based on operator characteristics
        """
        if operator is None:
            return StepType.UNKNOWN

        # Check for model characteristics
        if hasattr(operator, 'fit') and hasattr(operator, 'predict'):
            return StepType.MODEL

        # Check for transformer characteristics
        if hasattr(operator, 'fit') and hasattr(operator, 'transform'):
            return StepType.TRANSFORMER

        # Check for splitter characteristics
        if hasattr(operator, 'split'):
            return StepType.SPLITTER

        return StepType.UNKNOWN
```

---

## 3. ControllerRouter (Flexible Routing)

```python
"""
nirs4all/pipeline/steps/router.py

Route parsed steps to appropriate controllers.
"""

from typing import Type, Optional

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import CONTROLLER_REGISTRY
from nirs4all.pipeline.steps.parser import ParsedStep


class NoControllerFoundError(Exception):
    """Raised when no controller matches a step."""
    pass


class ControllerRouter:
    """
    Route parsed steps to appropriate controllers.

    Uses registry pattern with controller priorities.
    Extensible - new controllers automatically discovered via registry.

    Example:
        >>> router = ControllerRouter()
        >>> parsed = ParsedStep(operator=SVC(), keyword="model", ...)
        >>> controller = router.route(parsed)
        >>> isinstance(controller, SklearnModelController)
        True
    """

    def route(self, parsed_step: ParsedStep) -> OperatorController:
        """
        Select best controller for parsed step.

        Uses controller.matches() with priority sorting to find the
        most appropriate controller. Lower priority values are preferred.

        Args:
            parsed_step: Normalized step configuration

        Returns:
            Best matching controller instance

        Raises:
            NoControllerFoundError: If no controller matches the step

        Example:
            >>> parsed = ParsedStep(operator=StandardScaler(), ...)
            >>> controller = router.route(parsed)
            >>> print(controller.__class__.__name__)
            'TransformerMixinController'
        """
        # Find all controllers that match
        matches = [
            cls for cls in CONTROLLER_REGISTRY
            if cls.matches(
                step=parsed_step.original,
                operator=parsed_step.operator,
                keyword=parsed_step.keyword
            )
        ]

        if not matches:
            raise NoControllerFoundError(
                f"No controller found for step: {parsed_step.keyword} "
                f"with operator: {parsed_step.operator.__class__.__name__ if parsed_step.operator else 'None'}\n"
                f"Available controllers: {[cls.__name__ for cls in CONTROLLER_REGISTRY]}\n"
                f"Step details: {parsed_step.metadata}"
            )

        # Sort by priority (lower = higher priority)
        matches.sort(key=lambda c: c.priority)

        # Return highest priority match (instantiated)
        return matches[0]()

    def can_handle(self, parsed_step: ParsedStep) -> bool:
        """
        Check if any controller can handle the step.

        Args:
            parsed_step: Normalized step configuration

        Returns:
            True if at least one controller matches
        """
        try:
            self.route(parsed_step)
            return True
        except NoControllerFoundError:
            return False

    def list_matching_controllers(self, parsed_step: ParsedStep) -> list[Type[OperatorController]]:
        """
        List all controllers that match the step.

        Useful for debugging and understanding controller selection.

        Args:
            parsed_step: Normalized step configuration

        Returns:
            List of matching controller classes, sorted by priority
        """
        matches = [
            cls for cls in CONTROLLER_REGISTRY
            if cls.matches(
                step=parsed_step.original,
                operator=parsed_step.operator,
                keyword=parsed_step.keyword
            )
        ]
        matches.sort(key=lambda c: c.priority)
        return matches
```

---

## 4. StepRunner (Execute Single Step)

```python
"""
nirs4all/pipeline/steps/runner.py

Execute individual pipeline steps.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.execution.context import ExecutionContext
from nirs4all.pipeline.steps.parser import StepParser, ParsedStep
from nirs4all.pipeline.steps.router import ControllerRouter
from nirs4all.pipeline.artifact_serialization import ArtifactMeta


@dataclass
class StepResult:
    """
    Result of executing a single step.

    Attributes:
        updated_context: Updated pipeline context after step execution
        artifacts: List of artifacts generated by this step
        predictions: Optional predictions generated (for model steps)
    """
    updated_context: Dict[str, Any]
    artifacts: List[ArtifactMeta]
    predictions: Optional[Predictions] = None


class StepRunner:
    """
    Execute a single pipeline step.

    Handles:
    - Step parsing (delegates to StepParser)
    - Controller selection (delegates to ControllerRouter)
    - Controller execution
    - Binary loading/saving for this step

    Attributes:
        parser: Parses step configurations
        router: Routes to appropriate controller
        context: Current execution context

    Example:
        >>> parser = StepParser()
        >>> router = ControllerRouter()
        >>> context = ExecutionContext(...)
        >>> runner = StepRunner(parser, router, context)
        >>>
        >>> step = {"preprocessing": "StandardScaler"}
        >>> result = runner.execute(step, dataset, pipeline_context)
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
        self.parser = parser
        self.router = router
        self.context = context

    def execute(
        self,
        step: Any,
        dataset: SpectroDataset,
        pipeline_context: Dict[str, Any],
        prediction_store: Optional[Predictions] = None,
        loaded_binaries: Optional[List[ArtifactMeta]] = None
    ) -> StepResult:
        """
        Execute a single pipeline step.

        Args:
            step: Step configuration to execute
            dataset: Dataset to operate on
            pipeline_context: Current pipeline context (processing chain, etc.)
            prediction_store: External prediction store for model predictions
            loaded_binaries: Pre-loaded artifacts from manifest (for predict mode)

        Returns:
            StepResult with updated context and artifacts

        Raises:
            StepParseError: If step cannot be parsed
            NoControllerFoundError: If no controller matches
            ExecutionError: If step execution fails

        Example:
            >>> step = {"model": "SVC", "params": {"C": 1.0}}
            >>> result = runner.execute(step, dataset, context)
            >>> print(len(result.artifacts))
            1  # Model artifact
        """
        # 1. Parse step configuration
        try:
            parsed_step = self.parser.parse(step)
        except Exception as e:
            if self.context.verbose > 0:
                print(f"❌ Failed to parse step: {step}")
            raise

        # Log parsing result
        if self.context.verbose > 1:
            print(f"📋 Parsed step: {parsed_step.keyword} ({parsed_step.step_type.value})")

        # 2. Handle sub-pipelines recursively
        if parsed_step.keyword == "subpipeline":
            return self._execute_subpipeline(
                parsed_step.operator,  # List of substeps
                dataset,
                pipeline_context,
                prediction_store
            )

        # 3. Route to controller
        try:
            controller = self.router.route(parsed_step)
        except Exception as e:
            if self.context.verbose > 0:
                print(f"❌ Failed to route step: {parsed_step.keyword}")
            raise

        # Log controller selection
        if self.context.verbose > 1:
            print(f"🎯 Selected controller: {controller.__class__.__name__}")

        # 4. Check if controller supports current mode
        if self.context.is_prediction and not controller.supports_prediction_mode():
            if self.context.verbose > 0:
                print(f"⏭️  Skipping {controller.__class__.__name__} in prediction mode")
            return StepResult(
                updated_context=pipeline_context,
                artifacts=[],
                predictions=None
            )

        # 5. Execute controller
        try:
            updated_context, artifacts = controller.execute(
                step=parsed_step.original,
                operator=parsed_step.operator,
                dataset=dataset,
                context=pipeline_context,
                runner=self,  # Pass self for runner interface
                source=-1,
                mode=self.context.mode.value,
                loaded_binaries=loaded_binaries,
                prediction_store=prediction_store
            )
        except Exception as e:
            if self.context.continue_on_error:
                if self.context.verbose > 0:
                    print(f"⚠️  Step failed but continuing: {e}")
                return StepResult(
                    updated_context=pipeline_context,
                    artifacts=[],
                    predictions=None
                )
            else:
                if self.context.verbose > 0:
                    print(f"❌ Step execution failed: {e}")
                raise

        # 6. Log execution result
        if self.context.verbose > 1:
            print(f"✅ Step completed. Artifacts: {len(artifacts)}")

        return StepResult(
            updated_context=updated_context,
            artifacts=artifacts,
            predictions=None  # Predictions are handled by prediction_store
        )

    def _execute_subpipeline(
        self,
        substeps: List[Any],
        dataset: SpectroDataset,
        pipeline_context: Dict[str, Any],
        prediction_store: Optional[Predictions]
    ) -> StepResult:
        """
        Execute a sub-pipeline (list of steps).

        Args:
            substeps: List of substeps to execute
            dataset: Dataset to operate on
            pipeline_context: Current pipeline context
            prediction_store: Prediction store

        Returns:
            StepResult with accumulated artifacts
        """
        if self.context.verbose > 1:
            print(f"🔄 Executing sub-pipeline with {len(substeps)} steps")

        all_artifacts = []

        # Update context for substeps
        substep_context = self.context.with_incremented_substep()

        # Create substep runner
        substep_runner = StepRunner(self.parser, self.router, substep_context)

        # Execute each substep
        for i, substep in enumerate(substeps):
            result = substep_runner.execute(
                substep,
                dataset,
                pipeline_context,
                prediction_store
            )
            pipeline_context = result.updated_context
            all_artifacts.extend(result.artifacts)

            # Update substep context
            substep_context = substep_context.with_incremented_substep()
            substep_runner = StepRunner(self.parser, self.router, substep_context)

        return StepResult(
            updated_context=pipeline_context,
            artifacts=all_artifacts,
            predictions=None
        )

    # Runner interface methods (for compatibility with controllers)

    def next_op(self) -> int:
        """Get next operation ID (for compatibility)."""
        return self.context.step_tracker.operation_count + 1
```

---

## 5. Usage Example

```python
"""
Example of using the refactored components.
"""

from pathlib import Path
from nirs4all.pipeline.execution.context import ExecutionContext, ExecutionMode
from nirs4all.pipeline.steps.parser import StepParser
from nirs4all.pipeline.steps.router import ControllerRouter
from nirs4all.pipeline.steps.runner import StepRunner
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions

# 1. Create execution context
context = ExecutionContext(
    mode=ExecutionMode.TRAIN,
    workspace_path=Path("workspace"),
    pipeline_uid="0001_abc123",
    manifest_manager=manifest_manager,
    verbose=1
)

# 2. Create step runner with parser and router
parser = StepParser()
router = ControllerRouter()
runner = StepRunner(parser, router, context)

# 3. Define pipeline steps
steps = [
    {"preprocessing": "StandardScaler"},
    {"model": "sklearn.svm.SVC", "params": {"C": 1.0, "kernel": "rbf"}},
]

# 4. Execute steps
dataset = SpectroDataset(name="example")
pipeline_context = {"processing": [["raw"]], "y": "numeric"}
prediction_store = Predictions()

for step in steps:
    result = runner.execute(
        step,
        dataset,
        pipeline_context,
        prediction_store
    )
    pipeline_context = result.updated_context

    # Persist artifacts in train mode
    if context.is_training:
        for artifact in result.artifacts:
            manifest_manager.append_artifacts(context.pipeline_uid, [artifact])

    # Update context for next step
    context = context.with_incremented_step()
    runner = StepRunner(parser, router, context)

print(f"Completed pipeline execution with {prediction_store.num_predictions} predictions")
```

---

## 6. Testing Examples

```python
"""
Test examples for refactored components.
"""

import pytest
from nirs4all.pipeline.steps.parser import StepParser, ParsedStep, StepType


class TestStepParser:
    """Tests for StepParser."""

    def test_parse_dict_with_model_keyword(self):
        """Test parsing dict with model keyword."""
        parser = StepParser()
        step = {"model": "sklearn.svm.SVC", "params": {"C": 1.0}}

        result = parser.parse(step)

        assert result.keyword == "model"
        assert result.step_type == StepType.MODEL
        assert result.operator.__class__.__name__ == "SVC"
        assert result.metadata == {"params": {"C": 1.0}}

    def test_parse_string_module_path(self):
        """Test parsing string module path."""
        parser = StepParser()
        step = "sklearn.preprocessing.StandardScaler"

        result = parser.parse(step)

        assert result.operator.__class__.__name__ == "StandardScaler"
        assert result.step_type == StepType.TRANSFORMER

    def test_parse_list_subpipeline(self):
        """Test parsing list as sub-pipeline."""
        parser = StepParser()
        step = [
            {"preprocessing": "StandardScaler"},
            {"preprocessing": "MinMaxScaler"}
        ]

        result = parser.parse(step)

        assert result.keyword == "subpipeline"
        assert result.metadata["num_substeps"] == 2

    def test_parse_none_returns_noop(self):
        """Test that None step returns no-op."""
        parser = StepParser()
        result = parser.parse(None)

        assert result.keyword == "noop"
        assert result.operator is None


class TestControllerRouter:
    """Tests for ControllerRouter."""

    def test_route_sklearn_model(self):
        """Test routing sklearn model to correct controller."""
        from sklearn.svm import SVC
        from nirs4all.controllers.sklearn.op_model import SklearnModelController

        router = ControllerRouter()
        parsed = ParsedStep(
            operator=SVC(),
            keyword="model",
            step_type=StepType.MODEL,
            metadata={},
            original={"model": "SVC"}
        )

        controller = router.route(parsed)

        assert isinstance(controller, SklearnModelController)

    def test_no_controller_raises_error(self):
        """Test that unmatched step raises NoControllerFoundError."""
        from nirs4all.pipeline.steps.router import NoControllerFoundError

        router = ControllerRouter()
        parsed = ParsedStep(
            operator=object(),  # Unknown type
            keyword="unknown",
            step_type=StepType.UNKNOWN,
            metadata={},
            original={}
        )

        with pytest.raises(NoControllerFoundError):
            router.route(parsed)

    def test_can_handle_returns_true_for_valid_step(self):
        """Test can_handle returns True for valid step."""
        from sklearn.preprocessing import StandardScaler

        router = ControllerRouter()
        parsed = ParsedStep(
            operator=StandardScaler(),
            keyword="preprocessing",
            step_type=StepType.TRANSFORMER,
            metadata={},
            original={}
        )

        assert router.can_handle(parsed) is True


class TestStepRunner:
    """Tests for StepRunner."""

    def test_execute_transformer_step(self, mocker):
        """Test executing transformer step."""
        # Mock dependencies
        parser = mocker.Mock()
        router = mocker.Mock()
        context = mocker.Mock()
        controller = mocker.Mock()

        # Setup mocks
        parsed = ParsedStep(
            operator=mocker.Mock(),
            keyword="preprocessing",
            step_type=StepType.TRANSFORMER,
            metadata={},
            original={}
        )
        parser.parse.return_value = parsed
        router.route.return_value = controller
        controller.supports_prediction_mode.return_value = True
        controller.execute.return_value = (
            {"processing": [["standard_scaler"]]},
            []
        )

        # Execute
        runner = StepRunner(parser, router, context)
        dataset = mocker.Mock()
        pipeline_context = {"processing": [["raw"]]}

        result = runner.execute({}, dataset, pipeline_context)

        # Verify
        assert result.updated_context["processing"] == [["standard_scaler"]]
        parser.parse.assert_called_once()
        router.route.assert_called_once()
        controller.execute.assert_called_once()
```

---

These examples demonstrate clean, maintainable, and testable code following the proposed architecture. Each component has a single, clear responsibility and well-defined interfaces.
