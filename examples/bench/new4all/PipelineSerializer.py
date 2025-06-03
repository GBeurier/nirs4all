"""
PipelineSerializer - Serialization/deserialization for pipeline configurations
"""
import inspect
import importlib
import json
from typing import Any, Dict, List, Union, Optional, Type, get_type_hints
from pathlib import Path


class PipelineSerializer:
    """Handles serialization and deserialization of pipeline configurations"""

    def __init__(self):
        """Initialize serializer with preset mappings"""
        self.preset_mappings = {}
        self._register_common_presets()

    def _register_common_presets(self):
        """Register common preset mappings"""
        # Common sklearn imports
        self.preset_mappings.update({
            'StandardScaler': 'sklearn.preprocessing.StandardScaler',
            'MinMaxScaler': 'sklearn.preprocessing.MinMaxScaler',
            'RobustScaler': 'sklearn.preprocessing.RobustScaler',
            'RandomForestClassifier': 'sklearn.ensemble.RandomForestClassifier',
            'RandomForestRegressor': 'sklearn.ensemble.RandomForestRegressor',
            'SVC': 'sklearn.svm.SVC',
            'SVR': 'sklearn.svm.SVR',
            'LogisticRegression': 'sklearn.linear_model.LogisticRegression',
            'LinearRegression': 'sklearn.linear_model.LinearRegression',
            'PCA': 'sklearn.decomposition.PCA',
            'KMeans': 'sklearn.cluster.KMeans',
            'StratifiedKFold': 'sklearn.model_selection.StratifiedKFold',
            'RepeatedStratifiedKFold': 'sklearn.model_selection.RepeatedStratifiedKFold',
            'ShuffleSplit': 'sklearn.model_selection.ShuffleSplit',
            'DecisionTreeClassifier': 'sklearn.tree.DecisionTreeClassifier',
            'GradientBoostingClassifier': 'sklearn.ensemble.GradientBoostingClassifier'
        })

    def serialize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize a Python config (like sample.py) to JSON-serializable format"""
        return self._serialize_component(config)

    def deserialize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize a JSON config back to Python objects"""
        return self._deserialize_component(config)

    def _serialize_component(self, obj: Any) -> Any:
        """Serialize a component to JSON-serializable format"""
        # Handle None and basic types
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # Handle collections
        if isinstance(obj, dict):
            return {k: self._serialize_component(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            serialized_list = [self._serialize_component(x) for x in obj]
            if isinstance(obj, tuple):
                return {"__tuple__": serialized_list}
            return serialized_list

        # Handle classes (uninstantiated)
        if inspect.isclass(obj):
            return {"class": f"{obj.__module__}.{obj.__qualname__}"}

        # Handle functions
        if inspect.isfunction(obj) or inspect.isbuiltin(obj):
            func_data = {"function": f"{obj.__module__}.{obj.__name__}"}
            return func_data

        # Handle instances
        if hasattr(obj, '__class__'):
            instance_data = {
                "class": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"
            }

            # Extract parameters that differ from defaults
            params = self._extract_changed_params(obj)
            if params:
                instance_data["params"] = params

            return instance_data

        # String representation as fallback
        return str(obj)

    def _deserialize_component(self, obj: Any) -> Any:
        """Deserialize a component from JSON format"""
        # Handle None and basic types
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # Handle collections
        if isinstance(obj, list):
            return [self._deserialize_component(x) for x in obj]

        if isinstance(obj, dict):
            # Handle special tuple marker
            if "__tuple__" in obj:
                return tuple(self._deserialize_component(x) for x in obj["__tuple__"])

            # Handle class/function/instance objects
            if "class" in obj:
                return self._instantiate_from_config(obj)

            if "function" in obj:
                return self._load_function(obj["function"])

            # Regular dictionary
            return {k: self._deserialize_component(v) for k, v in obj.items()}

        return obj

    def _instantiate_from_config(self, config: Dict[str, Any]) -> Any:
        """Instantiate an object from class configuration"""
        class_path = config["class"]
        params = config.get("params", {})

        # Try to load the class
        try:
            cls = self._load_class(class_path)
        except (ImportError, AttributeError) as e:
            # Try preset mappings
            class_name = class_path.split('.')[-1]
            if class_name in self.preset_mappings:
                cls = self._load_class(self.preset_mappings[class_name])
            else:
                raise ValueError(f"Could not load class: {class_path}") from e

        # Deserialize parameters
        deserialized_params = {}
        for param_name, param_value in params.items():
            deserialized_params[param_name] = self._deserialize_component(param_value)

        # Instantiate with parameters
        try:
            return cls(**deserialized_params)
        except TypeError as e:
            # Try to filter parameters that the constructor accepts
            sig = inspect.signature(cls.__init__)
            valid_params = {k: v for k, v in deserialized_params.items()
                          if k in sig.parameters}
            return cls(**valid_params)

    def _load_class(self, class_path: str) -> Type:
        """Load a class from its module path"""
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _load_function(self, func_path: str) -> Any:
        """Load a function from its module path"""
        module_name, func_name = func_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    def _extract_changed_params(self, obj: Any) -> Dict[str, Any]:
        """Extract parameters that differ from their default values"""
        if not hasattr(obj, '__class__'):
            return {}

        try:
            sig = inspect.signature(obj.__class__.__init__)
        except (ValueError, TypeError):
            return {}

        changed_params = {}

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            # Get default value
            default_value = param.default if param.default != inspect.Parameter.empty else None

            # Get current value
            try:
                current_value = getattr(obj, param_name)
            except AttributeError:
                # Try alternative attribute names or skip
                continue

            # Compare with default
            if current_value != default_value:
                changed_params[param_name] = self._serialize_component(current_value)

        return changed_params

    def register_preset(self, name: str, class_path: str):
        """Register a preset class mapping"""
        self.preset_mappings[name] = class_path

    def serialize_to_file(self, config: Dict[str, Any], filepath: Union[str, Path]):
        """Serialize config and save to file"""
        filepath = Path(filepath)
        serialized_config = self.serialize_config(config)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            if filepath.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                yaml.dump(serialized_config, f, default_flow_style=False, indent=2)
            elif filepath.suffix.lower() == '.json':
                json.dump(serialized_config, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {filepath.suffix}")

    def deserialize_from_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load and deserialize config from file"""
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            if filepath.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                serialized_config = yaml.safe_load(f)
            elif filepath.suffix.lower() == '.json':
                serialized_config = json.load(f)
            else:
                raise ValueError(f"Unsupported format: {filepath.suffix}")

        return self.deserialize_config(serialized_config)
