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
        force_layout: Optional forced data layout (overrides controller's preferred layout)
    """
    operator: Any
    keyword: str
    step_type: StepType
    original_step: Any
    metadata: Dict[str, Any]
    force_layout: Optional[str] = None


class StepParser:
    """Parses pipeline step configurations into normalized format.

    Handles multiple step syntaxes:
    - Dictionary: {"model": SVC, "params": {...}}
    - String: "sklearn.preprocessing.StandardScaler"
    - Direct instance: StandardScaler()
    - Nested lists: [[step1, step2], step3]

    Normalizes to canonical ParsedStep format for controller execution.
    """

    # Known serialization operators
    SERIALIZATION_OPERATORS = ["class", "function", "module", "object", "pipeline", "instance"]

    # Reserved keywords that are not operators
    RESERVED_KEYWORDS = ["params", "metadata", "steps", "name", "finetune_params", "train_params", "fit_on_all", "force_layout"]

    # Valid layout values for force_layout
    VALID_LAYOUTS = {"2d", "2d_interleaved", "3d", "3d_transpose"}

    # Priority workflow keywords (ordered by priority, highest first)
    WORKFLOW_KEYWORDS = [
        "model",
        "preprocessing",
        "feature_augmentation",
        "auto_transfer_preproc",
        "concat_transform",
        "y_processing",
        "sample_augmentation",
        "branch",
    ]

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

        # Handle MinimalPipelineStep (from trace extractor)
        from nirs4all.pipeline.trace.extractor import MinimalPipelineStep
        if isinstance(step, MinimalPipelineStep):
            # Extract the step_config and parse it
            return self.parse(step.step_config)

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
        # Extract and validate force_layout if present
        force_layout = step.get("force_layout")
        if force_layout is not None and force_layout not in self.VALID_LAYOUTS:
            raise ValueError(
                f"Invalid force_layout '{force_layout}'. "
                f"Valid options: {self.VALID_LAYOUTS}"
            )

        # Check for serialization operators first
        for key in self.SERIALIZATION_OPERATORS:
            if key in step:
                operator = deserialize_component(step)
                return ParsedStep(
                    operator=operator,
                    keyword=key,
                    step_type=StepType.SERIALIZED,
                    original_step=step,
                    metadata={},
                    force_layout=force_layout
                )

        # Look for potential workflow operators
        # Prioritize known workflow keywords, then fall back to any non-reserved key
        candidates = [
            k for k in step.keys()
            if k not in self.RESERVED_KEYWORDS and k not in self.SERIALIZATION_OPERATORS
        ]

        if candidates:
            # Prioritize known workflow keywords in order
            key = None
            for workflow_key in self.WORKFLOW_KEYWORDS:
                if workflow_key in candidates:
                    key = workflow_key
                    break

            # If no priority keyword found, pick the first candidate
            if key is None:
                key = candidates[0]

            operator = self._deserialize_operator(step[key])
            return ParsedStep(
                operator=operator,
                keyword=key,
                step_type=StepType.WORKFLOW,
                original_step=step,
                metadata={"params": step.get("params", {})},
                force_layout=force_layout
            )

        # No recognized key - try to deserialize the whole dict
        operator = deserialize_component(step)
        return ParsedStep(
            operator=operator,
            keyword="",
            step_type=StepType.SERIALIZED,
            original_step=step,
            metadata={},
            force_layout=force_layout
        )

    def _parse_string_step(self, step: str) -> ParsedStep:
        """Parse string step configuration."""
        # For strings, we can't easily distinguish between keyword and class path
        # unless we check against a list.
        # But we want to avoid hardcoded lists.
        # If it looks like a class path (contains dots), treat as serialized.
        # If it's a single word, treat as keyword/workflow.

        if "." in step:
             # Deserialize as a class/function reference
            operator = deserialize_component(step)
            return ParsedStep(
                operator=operator,
                keyword=step,
                step_type=StepType.SERIALIZED,
                original_step=step,
                metadata={}
            )
        else:
            # Treat as keyword
            return ParsedStep(
                operator=None,
                keyword=step,
                step_type=StepType.WORKFLOW,
                original_step=step,
                metadata={}
            )


    def _deserialize_operator(self, value: Any) -> Optional[Any]:
        """Deserialize an operator value if needed.

        Handles:
        - None: returns None
        - Instances: returns as-is
        - Class types: returns as-is (controller will instantiate)
        - Dict with 'class'/'function': deserializes component
        - String: deserializes as module path
        - List/tuple: recursively deserializes each element
        """
        if value is None:
            return None

        # Handle lists/tuples (for chained operators like y_processing)
        if isinstance(value, (list, tuple)):
            deserialized = [self._deserialize_operator(v) for v in value]
            return deserialized if isinstance(value, list) else tuple(deserialized)

        # Already an instance or class type - return as-is
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
