"""
PipelineBuilder - Generic operation builder with serialization support

Handles all step types:
- strings (presets, commands)
- classes (to instantiate)
- instances (to clone/reuse)
- dicts (generic pipeline operation format)
"""
import inspect
import importlib
from typing import Any, Dict, Union
from sklearn.base import clone, BaseEstimator

from PipelineOperation import PipelineOperation
# from TransformationOperation import TransformationOperation
# from ClusteringOperation import ClusteringOperation
# from ModelOperation import ModelOperation
# from StackOperation import StackOperation


class PipelineBuilder:
    """Generic operation builder using serialization logic"""

    def __init__(self):
        # Import serialization utilities
        try:
            from nirs4all.utils.serialization import _serialize_component, _deserialize_component
            self._serialize = _serialize_component
            self._deserialize = _deserialize_component
        except ImportError:
            # Fallback serialization
            self._serialize = self._simple_serialize
            self._deserialize = self._simple_deserialize

        # Preset mappings
        self.presets = {
            'MinMaxScaler': ('sklearn.preprocessing', 'MinMaxScaler'),
            'StandardScaler': ('sklearn.preprocessing', 'StandardScaler'),
            'PCA': ('sklearn.decomposition', 'PCA'),
            'KMeans': ('sklearn.cluster', 'KMeans'),
            # Add more presets as needed
        }

    def build_operation(self, step: Any) -> PipelineOperation:
        """
        Generic operation builder - handles any step type including normalized configs

        Returns a PipelineOperation that wraps the actual operator
        """

        # Handle normalized dict with runtime instance (priority)
        if isinstance(step, dict) and "_runtime_instance" in step:
            return self._build_from_instance(step["_runtime_instance"])

        # Handle string steps (presets)
        elif isinstance(step, str):
            return self._build_from_string(step)

        # Handle normalized dict with preset
        elif isinstance(step, dict) and "preset" in step:
            return self._build_from_string(step["preset"])

        # Handle class types (to instantiate)
        elif inspect.isclass(step):
            return self._build_from_class(step)

        # Handle instances (to clone/reuse)
        elif hasattr(step, '__class__') and not isinstance(step, dict):
            return self._build_from_instance(step)

        # Handle generic pipeline operation dict
        elif isinstance(step, dict):
            return self._build_from_dict(step)

        else:
            raise ValueError(f"Cannot build operation from {type(step)}: {step}")

    def clone_for_dispatch(self, step: Any) -> Any:
        """
        Clone a step for parallel dispatch using best available method

        Strategy priority:
        1. Use serialization if step was normalized (most reliable)
        2. Use sklearn clone if available
        3. Reinstantiate classes
        4. Fallback to deepcopy
        """

        # If step is normalized dict with no instance, just return it
        # (will be rebuilt fresh each time)
        if isinstance(step, dict) and "_runtime_instance" not in step:
            return step

        # If step has cached instance, we need to clone it
        if isinstance(step, dict) and "_runtime_instance" in step:
            original_instance = step["_runtime_instance"]

            # Try sklearn clone first (fast and reliable for ML objects)
            if hasattr(original_instance, 'get_params'):
                try:
                    cloned_instance = clone(original_instance)
                    cloned_step = step.copy()
                    cloned_step["_runtime_instance"] = cloned_instance
                    return cloned_step
                except Exception:
                    pass  # Fall through to serialization

            # Use serialization-based cloning (most robust)
            try:
                # Remove instance, rebuild from serialized form
                clean_step = {k: v for k, v in step.items() if k != "_runtime_instance"}
                return clean_step  # Builder will recreate instance
            except Exception:
                pass

        # For raw objects (not normalized), try different strategies
        if hasattr(step, 'get_params'):
            try:
                return clone(step)
            except Exception:
                pass

        if inspect.isclass(step):
            return step()  # Fresh instance

        # Last resort - deepcopy (may fail)
        try:
            import copy
            return copy.deepcopy(step)
        except Exception as e:
            raise ValueError(f"Cannot clone step {step}: {e}")

    def _build_from_string(self, step: str) -> PipelineOperation:
        """Build operation from string preset or command"""

        # Handle special commands
        if step in ["PlotData", "PlotClusters", "PlotResults", "PlotModelPerformance",
                   "PlotFeatureImportance", "PlotConfusionMatrix", "uncluster", "unscope"]:
            return MockOperation(name=step)

        # Handle standard presets
        if step in self.presets:
            module_name, class_name = self.presets[step]
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            instance = cls()
            return self._wrap_operator(instance)
        else:
            # Try to create a mock operation for unknown strings
            return MockOperation(name=step)

    def _build_from_class(self, step: type) -> PipelineOperation:
        """Build operation from class (instantiate with defaults)"""
        instance = step()
        return self._wrap_operator(instance)

    def _build_from_instance(self, step: Any) -> PipelineOperation:
        """Build operation from instance (clone if needed)"""
        # Use sklearn clone for reusability across branches
        if hasattr(step, 'get_params'):
            cloned_instance = clone(step)
        else:
            # For non-sklearn objects, use serialization
            serialized = self._serialize(step)
            cloned_instance = self._deserialize(serialized)

        return self._wrap_operator(cloned_instance)

    def _build_from_dict(self, step: Dict[str, Any]) -> PipelineOperation:
        """Build operation from generic pipeline operation dict"""

        # Handle different dict formats
        if 'class' in step:
            # Generic pipeline operation format: {"class": ..., "params": ..."}
            class_spec = step['class']
            params = step.get('params', {})

            # Resolve class
            if isinstance(class_spec, str):
                if '.' in class_spec:
                    # Full module.class path
                    module_name, class_name = class_spec.rsplit('.', 1)
                    module = importlib.import_module(module_name)
                    cls = getattr(module, class_name)
                else:
                    # Preset name
                    return self._build_from_string(class_spec)
            else:
                cls = class_spec            # Instantiate with params
            instance = cls(**params)
            return self._wrap_operator(instance)

        elif 'model' in step:
            # Model configuration - use GenericOperation for now
            model = step['model']
            model_instance = self.build_operation(model)
            return model_instance

        elif 'stack' in step:
            # Stack configuration - use GenericOperation for now
            return GenericOperation(operator=step)

        elif 'cluster' in step:
            # Clustering operation - use GenericOperation for now
            clusterer = step['cluster']
            clusterer_instance = self.build_operation(clusterer)
            return clusterer_instance

        else:
            # Try to deserialize as generic object
            try:
                instance = self._deserialize(step)
                return self._wrap_operator(instance)
            except Exception:
                raise ValueError(f"Cannot build operation from dict: {step}")

    def _wrap_operator(self, operator: Any) -> PipelineOperation:
        """Wrap an operator in appropriate PipelineOperation"""
        # For now, use GenericOperation for all operators
        return GenericOperation(operator=operator)

    def _simple_serialize(self, obj: Any) -> Dict[str, Any]:
        """Simple fallback serialization"""
        if hasattr(obj, 'get_params'):
            # Sklearn-style object
            params = obj.get_params()
            return {
                "class": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                "params": params
            }
        else:
            # Generic object
            return {
                "class": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                "params": {}
            }

    def _simple_deserialize(self, data: Dict[str, Any]) -> Any:
        """Simple fallback deserialization"""
        if "class" not in data:
            raise ValueError("Missing 'class' field in serialized data")

        class_path = data["class"]
        params = data.get("params", {})

        # Import class
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        # Instantiate
        return cls(**params)


class GenericOperation(PipelineOperation):
    """Generic wrapper for operators that don't fit standard categories"""

    def __init__(self, operator: Any):
        self.operator = operator

    def execute(self, dataset, context=None):
        """Generic execution - try common patterns"""
        print(f"  ⚙️ Executing {self.get_name()}")

        # For MVP: don't actually execute, just log what would happen
        if hasattr(self.operator, 'fit_transform'):
            print(f"    📊 Would fit_transform on training data")
        elif hasattr(self.operator, 'transform'):
            print(f"    🔄 Would transform all data")
        elif hasattr(self.operator, 'fit'):
            print(f"    🎯 Would fit on training data")
        else:
            print(f"    💡 Would execute {type(self.operator)}")

    def get_name(self) -> str:
        return f"Generic({self.operator.__class__.__name__})"


class MockOperation(PipelineOperation):
    """Mock operation for MVP testing - doesn't actually execute anything"""

    def __init__(self, name: str):
        self.name = name

    def execute(self, dataset, context=None):
        """Mock execution - just log what would happen"""
        print(f"  🎭 Mock execution: {self.name}")

    def get_name(self) -> str:
        return f"Mock({self.name})"
