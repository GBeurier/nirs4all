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
from TransformationOperation import TransformationOperation
from ClusteringOperation import ClusteringOperation
from ModelOperation import ModelOperation
from StackOperation import StackOperation


class PipelineBuilder:
    """Generic operation builder using serialization logic"""

    def __init__(self):
        # Import serialization utilities
        try:
            from nirs4all.utils.serialization import _serialize_component, _deserialize_component
            self._serialize = _serialize_component
            self._deserialize = _deserialize_component
        except ImportError:
            # Fallback if serialization utils not available
            self._serialize = None
            self._deserialize = None

        # Preset mappings
        self.presets = {
            'MinMaxScaler': ('sklearn.preprocessing', 'MinMaxScaler'),
            'StandardScaler': ('sklearn.preprocessing', 'StandardScaler'),
            'PCA': ('sklearn.decomposition', 'PCA'),
            'KMeans': ('sklearn.cluster', 'KMeans'),
            # Add more presets as needed
        }

        # Track fitted operations for serialization
        self.fitted_operations: Dict[str, Any] = {}

    def get_fitted_operations(self) -> Dict[str, Any]:
        """Return all fitted operations for serialization"""
        return self.fitted_operations.copy()

    def store_fitted_operation(self, operation_id: str, operation: Any):
        """Store a fitted operation for later serialization"""
        self.fitted_operations[operation_id] = operation

    def build_operation(self, step: Any) -> PipelineOperation:
        """
        Generic operation builder - handles any step type

        Returns a PipelineOperation that wraps the actual operator
        """

        # Handle string steps (presets)
        if isinstance(step, str):
            return self._build_from_string(step)

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

    def _build_from_string(self, step: str) -> PipelineOperation:
        """Build operation from string preset"""
        if step in self.presets:
            module_name, class_name = self.presets[step]
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            instance = cls()
            return self._wrap_operator(instance)
        else:
            raise ValueError(f"Unknown preset: {step}")

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
                cls = class_spec

            # Instantiate with params
            instance = cls(**params)
            return self._wrap_operator(instance)

        elif 'model' in step:
            # Model configuration
            model = step['model']
            model_instance = self.build_operation(model)

            # Wrap in ModelOperation with additional config
            return ModelOperation(
                model=model_instance.operator,
                train_params=step.get('train_params', {}),
                finetune_params=step.get('finetune_params', {})
            )

        elif 'stack' in step:
            # Stack configuration
            stack_config = step['stack']
            return StackOperation(
                meta_model=self.build_operation(stack_config['meta']).operator,
                base_models=[self.build_operation(m).operator for m in stack_config['base']]
            )

        elif 'cluster' in step:
            # Clustering operation
            clusterer = step['cluster']
            clusterer_instance = self.build_operation(clusterer)
            return ClusteringOperation(clusterer=clusterer_instance.operator)

        else:
            # Try to deserialize as generic object
            try:
                instance = self._deserialize(step)
                return self._wrap_operator(instance)
            except:
                raise ValueError(f"Cannot build operation from dict: {step}")

    def _wrap_operator(self, operator: Any) -> PipelineOperation:
        """Wrap an operator in appropriate PipelineOperation"""

        # Import sklearn types for checking
        from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator

        # Determine operation type and wrap accordingly
        if hasattr(operator, 'transform') and isinstance(operator, TransformerMixin):
            return TransformationOperation(transformer=operator)
        elif hasattr(operator, 'fit') and isinstance(operator, ClusterMixin):
            return ClusteringOperation(clusterer=operator)
        elif hasattr(operator, 'fit') and isinstance(operator, BaseEstimator):
            return ModelOperation(model=operator)
        else:
            # Generic wrapper for other types
            return GenericOperation(operator=operator)


class GenericOperation(PipelineOperation):
    """Generic wrapper for operators that don't fit standard categories"""

    def __init__(self, operator: Any):
        self.operator = operator

    def execute(self, dataset, context):
        """Generic execution - try common patterns"""
        if hasattr(self.operator, 'fit_transform'):
            X = dataset.get_features()
            X_transformed = self.operator.fit_transform(X)
            dataset.set_features(X_transformed)
        elif hasattr(self.operator, 'transform'):
            X = dataset.get_features()
            X_transformed = self.operator.transform(X)
            dataset.set_features(X_transformed)
        elif hasattr(self.operator, 'fit'):
            X = dataset.get_features()
            try:
                y = dataset.get_targets()
                self.operator.fit(X, y)
            except:
                self.operator.fit(X)
        else:
            raise ValueError(f"Don't know how to execute {type(self.operator)}")

    def get_name(self) -> str:
        return f"Generic({self.operator.__class__.__name__})"
