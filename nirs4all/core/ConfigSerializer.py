"""
ConfigSerializer - Unified config serialization with instance caching

Handles:
- Normalization of any config format to unified dict format
- Instance caching for runtime optimization
- Clean JSON serialization (removes instances)
- Config restoration from JSON
"""
import json
import yaml
import inspect
import copy
from typing import Any, Dict, List, Union
from pathlib import Path

class ConfigSerializer:
    """Handles config serialization with instance management"""

    def __init__(self):
        # Import serialization utilities
        try:
            from nirs4all.utils.serialization import _serialize_component, _deserialize_component
            self._serialize = _serialize_component
            self._deserialize = _deserialize_component
        except ImportError:
            # Fallback simple serialization
            self._serialize = self._simple_serialize
            self._deserialize = self._simple_deserialize

    def normalize_config(self, config: Union[Dict, str, Path]) -> Dict[str, Any]:
        """
        Convert any config format to unified dict format with optional runtime instances

        Supports:
        - JSON/YAML files
        - JSON/YAML strings
        - Dict configs
        - Mixed pipeline definitions (strings, classes, instances, dicts)
        """
        # Handle string inputs (JSON strings, YAML strings, or file paths)
        if isinstance(config, str):
            # Try to parse as JSON first
            try:
                config = json.loads(config)
            except json.JSONDecodeError:
                # Try to parse as YAML
                try:
                    config = yaml.safe_load(config)
                except yaml.YAMLError:
                    # Must be a file path
                    config = self._load_config_file(config)

        # Handle Path objects (definitely files)
        elif isinstance(config, Path):
            config = self._load_config_file(config)

        # Normalize pipeline steps
        if "pipeline" in config:
            config["pipeline"] = self._normalize_pipeline(config["pipeline"])

        return config

    def _load_config_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load config from JSON or YAML file"""
        path = Path(filepath)

        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() == '.json':
                return json.load(f)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")

    def _normalize_pipeline(self, pipeline: List[Any]) -> List[Dict[str, Any]]:
        """Normalize pipeline steps to unified dict format"""
        normalized = []

        for step in pipeline:
            normalized_step = self._normalize_step(step)
            normalized.append(normalized_step)

        return normalized

    def _normalize_step(self, step: Any) -> Dict[str, Any]:
        """Convert any step format to unified dict format"""

        if isinstance(step, str):
            # String preset
            return {"preset": step}

        elif inspect.isclass(step):
            # Class to instantiate
            return {
                "class": f"{step.__module__}.{step.__qualname__}",
                "params": {}
            }

        elif isinstance(step, dict):
            # Already a dict - check for nested structures
            if "pipeline" in step:
                # Nested pipeline
                step["pipeline"] = self._normalize_pipeline(step["pipeline"])
            elif "dispatch" in step:
                # Dispatch branches
                step["dispatch"] = [self._normalize_step(branch) for branch in step["dispatch"]]
            elif "feature_augmentation" in step:
                # Feature augmentation list
                step["feature_augmentation"] = [self._normalize_step(aug) for aug in step["feature_augmentation"]]
            elif "sample_augmentation" in step:
                # Sample augmentation list
                step["sample_augmentation"] = [self._normalize_step(aug) for aug in step["sample_augmentation"]]

            return step

        elif isinstance(step, list):
            # List of steps
            return [self._normalize_step(substep) for substep in step]

        elif hasattr(step, '__class__') and not isinstance(step, (dict, list, str)):
            # Instance object - serialize it but keep instance for runtime
            try:
                serialized = self._serialize(step)
                serialized["_runtime_instance"] = step  # Cache original instance
                return serialized
            except Exception as e:
                # Fallback to class + empty params
                return {
                    "class": f"{step.__class__.__module__}.{step.__class__.__qualname__}",
                    "params": {},
                    "_runtime_instance": step,
                    "_serialization_error": str(e)
                }

        else:
            raise ValueError(f"Cannot normalize step of type {type(step)}: {step}")

    def prepare_for_json(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove runtime instances before JSON serialization"""
        def clean_recursive(obj):
            if isinstance(obj, dict):
                return {k: clean_recursive(v) for k, v in obj.items()
                       if k != "_runtime_instance"}
            elif isinstance(obj, list):
                return [clean_recursive(item) for item in obj]
            else:
                return obj

        return clean_recursive(config)

    def save_config(self, config: Dict[str, Any], filepath: Union[str, Path]):
        """Save normalized config to JSON file"""
        path = Path(filepath)

        # Clean config for JSON
        clean_config = self.prepare_for_json(config)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(clean_config, f, indent=2, default=str)

        print(f"💾 Config saved to {path}")

    def load_config(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load and normalize config from file"""
        config = self._load_config_file(filepath)
        return self.normalize_config(config)

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
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)

        # Instantiate
        return cls(**params)
