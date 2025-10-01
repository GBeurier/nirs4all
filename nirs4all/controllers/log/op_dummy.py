"""DummyController.py - A catch-all controller for operators not handled by other controllers in the nirs4all pipeline."""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import json
import inspect

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset


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
            pipeline_keywords = ['model', 'feature_augmentation', 'y_processing', 'sample_augmentation']
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

    def _get_context_info(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract useful information from the pipeline context."""
        context_info = {}

        # Key context fields
        important_keys = ['keyword', 'processing', 'partition', 'y', 'layout', 'add_feature']
        for key in important_keys:
            if key in context:
                context_info[key] = self._safe_repr(context[key])

        # Count total context keys
        context_info["total_keys"] = len(context)
        context_info["all_keys"] = list(context.keys())

        return context_info

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """
        Handle unmatched operators and provide detailed debugging information.
        """

        print("\n" + "="*80)
        print("🚨 DUMMY CONTROLLER ACTIVATED - UNHANDLED OPERATOR DETECTED")
        print("="*80)

        # Basic execution info
        print(f"📍 Execution Context:")
        print(f"   Mode: {mode}")
        print(f"   Source: {source}")
        print(f"   Dataset: {dataset.name if hasattr(dataset, 'name') else 'unknown'}")

        # Step analysis
        print(f"\n📋 Step Analysis:")
        step_analysis = self._analyze_step_structure(step)
        for key, value in step_analysis.items():
            print(f"   {key}: {value}")

        # Operator analysis
        print(f"\n🔧 Operator Analysis:")
        if operator is not None:
            operator_analysis = self._analyze_step_structure(operator)
            for key, value in operator_analysis.items():
                print(f"   {key}: {value}")
        else:
            print("   operator: None")

        # Context analysis
        print(f"\n🗂️ Context Analysis:")
        context_info = self._get_context_info(context)
        for key, value in context_info.items():
            print(f"   {key}: {value}")

        # Keyword analysis
        keyword = context.get('keyword', 'unknown')
        print(f"\n� Keyword: '{keyword}'")

        # Suggestions
        print(f"\n💡 Possible Issues:")
        suggestions = []

        if isinstance(step, dict):
            if not any(k in step for k in ['model', 'feature_augmentation', 'y_processing', 'sample_augmentation']):
                suggestions.append("- Step is a dict but doesn't contain recognized pipeline keywords")

            if 'model' in step:
                suggestions.append("- Step contains 'model' - should be handled by a model controller")

            if 'feature_augmentation' in step:
                suggestions.append("- Step contains 'feature_augmentation' - should be handled by FeatureAugmentationController")

        elif hasattr(step, 'fit') and hasattr(step, 'transform'):
            suggestions.append("- Step has fit() and transform() methods - should be handled by TransformerMixinController")

        elif hasattr(step, 'fit') and hasattr(step, 'predict'):
            suggestions.append("- Step has fit() and predict() methods - should be handled by a model controller")

        elif hasattr(step, 'split'):
            suggestions.append("- Step has split() method - should be handled by CrossValidatorController")

        if keyword == 'unknown':
            suggestions.append("- Keyword is 'unknown' - check pipeline step configuration")

        if not suggestions:
            suggestions.append("- No obvious issues detected - may need new controller or controller priority adjustment")

        for suggestion in suggestions:
            print(f"   {suggestion}")

        # Controller registry info
        print(f"\n🎯 Debugging Info:")
        print(f"   - Check controller priorities and matches() methods")
        print(f"   - Verify step format matches expected controller patterns")
        print(f"   - Consider adding specific controller for this operator type")

        print("="*80)
        print("🚨 END DUMMY CONTROLLER REPORT")
        print("="*80 + "\n")

        # Return unchanged context - this is just for debugging
        return context, []

