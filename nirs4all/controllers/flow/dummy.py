"""DummyController.py - A catch-all controller for operators not handled by other controllers in the nirs4all pipeline."""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import json
import inspect

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.steps.parser import ParsedStep
    from nirs4all.pipeline.config.context import ExecutionContext


@register_controller
class DummyController(OperatorController):
    """
    Catch-all controller for operators not handled by other controllers.

    This controller has the lowest priority and will catch any operators that
    don't match other controllers, providing detailed debugging information
    about why they weren't handled elsewhere.
    """

    priority = 1000  # Lowest priority to catch unhandled operators

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """
        Always match as a last resort.

        This controller should only be reached if no other controller
        with higher priority has matched the step/operator/keyword combination.
        """
        return True  # Catch everything that other controllers don't handle

    @classmethod
    def use_multi_source(cls) -> bool:
        """Check if the operator supports multi-source datasets."""
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Dummy controller supports prediction mode."""
        return True

    def _safe_repr(self, obj: Any, max_length: int = 200) -> str:
        """Safely represent an object as a string, truncating if necessary."""
        try:
            if obj is None:
                return "None"

            # Handle common types
            if isinstance(obj, (str, int, float, bool)):
                return repr(obj)
            elif isinstance(obj, (list, tuple, set)):
                if len(obj) > 5:
                    items = [self._safe_repr(item, 50) for item in list(obj)[:5]]
                    return f"{type(obj).__name__}([{', '.join(items)}, ...]) (length: {len(obj)})"
                else:
                    items = [self._safe_repr(item, 50) for item in obj]
                    return f"{type(obj).__name__}([{', '.join(items)}])"
            elif isinstance(obj, dict):
                if len(obj) > 5:
                    items = [f"{self._safe_repr(k, 30)}: {self._safe_repr(v, 30)}"
                            for k, v in list(obj.items())[:5]]
                    return f"dict({{{', '.join(items)}, ...}}) (length: {len(obj)})"
                else:
                    items = [f"{self._safe_repr(k, 30)}: {self._safe_repr(v, 30)}"
                            for k, v in obj.items()]
                    return f"dict({{{', '.join(items)}}})"
            else:
                # For other objects, show type and basic info
                obj_repr = f"{type(obj).__module__}.{type(obj).__name__}"

                # Try to get some useful attributes
                if hasattr(obj, '__dict__'):
                    attrs = []
                    for attr, value in obj.__dict__.items():
                        if not attr.startswith('_') and len(attrs) < 3:
                            attrs.append(f"{attr}={self._safe_repr(value, 30)}")
                    if attrs:
                        obj_repr += f"({', '.join(attrs)})"

                # Truncate if too long
                if len(obj_repr) > max_length:
                    obj_repr = obj_repr[:max_length-3] + "..."

                return obj_repr

        except Exception as e:
            return f"<Error representing object: {type(e).__name__}: {str(e)[:50]}>"

    def _analyze_step_structure(self, step: Any) -> Dict[str, Any]:
        """Analyze the structure of a step to help identify why it wasn't matched."""
        analysis = {
            "type": type(step).__name__,
            "module": getattr(type(step), '__module__', 'unknown'),
            "value": self._safe_repr(step),
        }

        if isinstance(step, dict):
            analysis["keys"] = list(step.keys())
            analysis["key_types"] = {k: type(v).__name__ for k, v in step.items()}

            # Look for common pipeline keywords
            pipeline_keywords = ['model', 'feature_augmentation', 'concat_transform', 'y_processing', 'sample_augmentation']
            found_keywords = [k for k in step.keys() if k in pipeline_keywords]
            if found_keywords:
                analysis["pipeline_keywords"] = found_keywords

        elif hasattr(step, '__class__'):
            # For objects, get class hierarchy and common attributes
            analysis["class_hierarchy"] = [cls.__name__ for cls in step.__class__.__mro__]

            # Check for sklearn/scikit-learn patterns
            if hasattr(step, 'fit') or hasattr(step, 'transform') or hasattr(step, 'predict'):
                analysis["sklearn_methods"] = []
                if hasattr(step, 'fit'):
                    analysis["sklearn_methods"].append('fit')
                if hasattr(step, 'transform'):
                    analysis["sklearn_methods"].append('transform')
                if hasattr(step, 'predict'):
                    analysis["sklearn_methods"].append('predict')

        return analysis

    def _get_context_info(self, context: Any) -> Dict[str, Any]:
        """Extract useful information from the pipeline context."""
        context_info = {}

        # Check if it's an ExecutionContext
        if hasattr(context, 'selector') and hasattr(context, 'state') and hasattr(context, 'metadata'):
            # Extract info directly from ExecutionContext
            context_info['keyword'] = self._safe_repr(context.metadata.keyword)
            context_info['processing'] = self._safe_repr(context.selector.processing)
            context_info['partition'] = self._safe_repr(context.selector.partition)
            context_info['y'] = self._safe_repr(context.state.y_processing)
            context_info['layout'] = self._safe_repr(context.selector.layout)
            context_info['add_feature'] = self._safe_repr(context.metadata.add_feature)

            # Count total context keys (simulated)
            context_info["total_keys"] = "N/A (ExecutionContext)"
            context_info["all_keys"] = ["selector", "state", "metadata", "custom"]
        elif isinstance(context, dict):
            # Legacy dict context
            context_dict = context

            # Key context fields
            important_keys = ['keyword', 'processing', 'partition', 'y', 'layout', 'add_feature']
            for key in important_keys:
                if key in context_dict:
                    context_info[key] = self._safe_repr(context_dict[key])

            # Count total context keys
            context_info["total_keys"] = len(context_dict)
            context_info["all_keys"] = list(context_dict.keys())
        else:
            return {"error": f"Unknown context type: {type(context)}"}

        return context_info

    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple['ExecutionContext', List[Tuple[str, bytes]]]:
        """
        Handle unmatched operators and provide detailed debugging information.
        """
        op = step_info.operator
        config = step_info.original_step

        logger.warning("" + "="*80)
        logger.warning("DUMMY CONTROLLER ACTIVATED - UNHANDLED OPERATOR DETECTED")
        logger.warning("="*80)

        # Basic execution info
        logger.warning("Execution Context:")
        logger.warning(f"   Mode: {mode}")
        logger.warning(f"   Source: {source}")
        logger.warning(f"   Dataset: {dataset.name if hasattr(dataset, 'name') else 'unknown'}")

        # Step analysis
        logger.warning("Step Analysis:")
        step_analysis = self._analyze_step_structure(config)
        for key, value in step_analysis.items():
            logger.warning(f"   {key}: {value}")

        # Operator analysis
        logger.warning("Operator Analysis:")
        if op is not None:
            operator_analysis = self._analyze_step_structure(op)
            for key, value in operator_analysis.items():
                logger.warning(f"   {key}: {value}")
        else:
            logger.warning("   operator: None")

        # Context analysis
        logger.warning("Context Analysis:")
        context_info = self._get_context_info(context)
        for key, value in context_info.items():
            logger.warning(f"   {key}: {value}")

        # Keyword analysis
        if hasattr(context, 'metadata'):
             keyword = context.metadata.keyword
        else:
             keyword = 'unknown'
        logger.warning(f"Keyword: '{keyword}'")

        # Suggestions
        logger.warning("Possible Issues:")
        suggestions = []

        if isinstance(config, dict):
            if not any(k in config for k in ['model', 'feature_augmentation', 'concat_transform', 'y_processing', 'sample_augmentation']):
                suggestions.append("- Step is a dict but doesn't contain recognized pipeline keywords")

            if 'model' in config:
                suggestions.append("- Step contains 'model' - should be handled by a model controller")

            if 'feature_augmentation' in config:
                suggestions.append("- Step contains 'feature_augmentation' - should be handled by FeatureAugmentationController")

            if 'concat_transform' in config:
                suggestions.append("- Step contains 'concat_transform' - should be handled by ConcatAugmentationController")

        elif hasattr(op, 'fit') and hasattr(op, 'transform'):
            suggestions.append("- Step has fit() and transform() methods - should be handled by TransformerMixinController")

        elif hasattr(op, 'fit') and hasattr(op, 'predict'):
            suggestions.append("- Step has fit() and predict() methods - should be handled by a model controller")

        elif hasattr(op, 'split'):
            suggestions.append("- Step has split() method - should be handled by CrossValidatorController")

        if keyword == 'unknown':
            suggestions.append("- Keyword is 'unknown' - check pipeline step configuration")

        if not suggestions:
            suggestions.append("- No obvious issues detected - may need new controller or controller priority adjustment")

        for suggestion in suggestions:
            logger.warning(f"   {suggestion}")

        # Controller registry info
        logger.warning("Debugging Info:")
        logger.warning("   - Check controller priorities and matches() methods")
        logger.warning("   - Verify step format matches expected controller patterns")
        logger.warning("   - Consider adding specific controller for this operator type")

        logger.warning("="*80)
        logger.warning("END DUMMY CONTROLLER REPORT")
        logger.warning("="*80)

        # Return unchanged context - this is just for debugging
        return context, []


