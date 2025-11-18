"""Step parser for pipeline step configurations."""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from nirs4all.pipeline.config.component_serialization import deserialize_component


class StepType(Enum):
    """Types of pipeline steps."""
    WORKFLOW = "workflow"  # model, preprocessing, chart, etc.
    SERIALIZED = "serialized"  # class, function, module, etc.
    SUBPIPELINE = "subpipeline"  # nested list of steps
    DIRECT = "direct"  # direct operator instance
    UNKNOWN = "unknown"


@dataclass
class ParsedStep:
    """Normalized step configuration after parsing.

    Attributes:
        operator: Deserialized operator instance (or None for workflow ops)
        keyword: Step keyword (e.g., 'model', 'preprocessing')
        step_type: Type of step (workflow, serialized, etc.)
        original_step: Original step configuration
        metadata: Additional parsing metadata
    """
    operator: Any
    keyword: str
    step_type: StepType
    original_step: Any
    metadata: Dict[str, Any]


class StepParser:
    """Parses pipeline step configurations into normalized format.

    Handles multiple step syntaxes:
    - Dictionary: {"model": SVC, "params": {...}}
    - String: "sklearn.preprocessing.StandardScaler"
    - Direct instance: StandardScaler()
    - Nested lists: [[step1, step2], step3]

    Normalizes to canonical ParsedStep format for controller execution.
    """

    # Known workflow operators
    WORKFLOW_OPERATORS = [
        "sample_augmentation", "feature_augmentation", "branch", "dispatch",
        "model", "stack", "scope", "cluster", "merge", "uncluster", "unscope",
        "chart_2d", "chart_3d", "fold_chart", "y_processing", "y_chart",
        "split", "preprocessing"
    ]

    # Known serialization operators
    SERIALIZATION_OPERATORS = ["class", "function", "module", "object", "pipeline", "instance"]

    def parse(self, step: Any) -> ParsedStep:
        """Parse a pipeline step into normalized format.

        Args:
            step: Raw step configuration

        Returns:
            ParsedStep with normalized operator and metadata

        Raises:
            ValueError: If step format is invalid
        """
        if step is None:
            return ParsedStep(
                operator=None,
                keyword="",
                step_type=StepType.UNKNOWN,
                original_step=step,
                metadata={"skip": True}
            )

        # Handle dictionary steps
        if isinstance(step, dict):
            return self._parse_dict_step(step)

        # Handle list steps (subpipelines)
        if isinstance(step, list):
            return ParsedStep(
                operator=None,
                keyword="",
                step_type=StepType.SUBPIPELINE,
                original_step=step,
                metadata={"steps": step}
            )

        # Handle string steps
        if isinstance(step, str):
            return self._parse_string_step(step)

        # Handle direct operator instances
        return ParsedStep(
            operator=step,
            keyword="",
            step_type=StepType.DIRECT,
            original_step=step,
            metadata={}
        )

    def _parse_dict_step(self, step: Dict[str, Any]) -> ParsedStep:
        """Parse dictionary step configuration."""
        # Check for workflow operators
        for key in self.WORKFLOW_OPERATORS:
            if key in step:
                operator = self._deserialize_operator(step[key])
                return ParsedStep(
                    operator=operator,
                    keyword=key,
                    step_type=StepType.WORKFLOW,
                    original_step=step,
                    metadata={"params": step.get("params", {})}
                )

        # Check for serialization operators
        for key in self.SERIALIZATION_OPERATORS:
            if key in step:
                operator = deserialize_component(step)
                return ParsedStep(
                    operator=operator,
                    keyword=key,
                    step_type=StepType.SERIALIZED,
                    original_step=step,
                    metadata={}
                )

        # No recognized key - try to deserialize the whole dict
        operator = deserialize_component(step)
        return ParsedStep(
            operator=operator,
            keyword="",
            step_type=StepType.SERIALIZED,
            original_step=step,
            metadata={}
        )

    def _parse_string_step(self, step: str) -> ParsedStep:
        """Parse string step configuration."""
        # Check if string contains a workflow operator keyword
        for keyword in self.WORKFLOW_OPERATORS:
            if keyword in step.split():
                return ParsedStep(
                    operator=None,
                    keyword=keyword,
                    step_type=StepType.WORKFLOW,
                    original_step=step,
                    metadata={}
                )

        # Otherwise, deserialize as a class/function reference
        operator = deserialize_component(step)
        return ParsedStep(
            operator=operator,
            keyword=step,
            step_type=StepType.SERIALIZED,
            original_step=step,
            metadata={}
        )

    def _deserialize_operator(self, value: Any) -> Optional[Any]:
        """Deserialize an operator value if needed."""
        if value is None:
            return None

        # Already an instance
        if not isinstance(value, (dict, str)):
            return value

        # Dictionary with class/function
        if isinstance(value, dict):
            if '_runtime_instance' in value:
                return value['_runtime_instance']
            if 'class' in value or 'function' in value:
                return deserialize_component(value)
            # Try to deserialize the whole dict
            return deserialize_component(value)

        # String reference
        if isinstance(value, str):
            return deserialize_component(value)

        return value
