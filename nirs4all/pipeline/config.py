"""
PipelineConfigs.py
"""

import json
import logging
from pathlib import Path
from typing import List, Any, Dict, Union
import yaml

from .serialization import serialize_component
from .generator import expand_spec, count_combinations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineConfigs:
    """
    Class to hold the configuration for a pipeline.
    """
    def __init__(self, definition: Union[Dict, List[Any], str], name: str = "", description: str = "No description provided", max_generation_count: int = 10000):
        """
        Initialize the pipeline configuration.
        """
        ## Parse / Format / Validate the configuration
        self.description = description
        self.steps = self._load_steps(definition)
        self.steps = self._preprocess_steps(self.steps)
        self.steps = serialize_component(self.steps, include_runtime=True)

        ## Generation
        self.has_configurations = False
        if self._has_gen_keys(self.steps):
            count = count_combinations(self.steps)
            if count > max_generation_count:
                raise ValueError(f"Configuration expansion would generate {count} configurations, exceeding the limit of {max_generation_count}. Please simplify your configuration.")
            if count > 1:
                self.has_configurations = True
                self.steps = expand_spec(self.steps)

        if not self.has_configurations:
            self.steps = [self.steps]  # Wrap single configuration in a list

        ## Name and hash
        self.names = [
            "config_" + (self.get_hash(steps) if name == "" else name + "_" + self.get_hash(steps)[0:6])
            for steps in self.steps
        ]

        print(f"✅ Loaded pipeline(s) with {len(self.steps)} configuration(s).")

    @staticmethod
    def _has_gen_keys(obj: Any) -> bool:
        """Recursively check if the configuration contains 'or' keys."""
        if isinstance(obj, dict):
            if "_or_" in obj or "_range_" in obj:
                return True
            return any(PipelineConfigs._has_gen_keys(v) for v in obj.values())
        elif isinstance(obj, list):
            return any(PipelineConfigs._has_gen_keys(item) for item in obj)
        return False

    @staticmethod
    def _preprocess_steps(steps: Any) -> Any:
        """
        Preprocess steps to merge *_params into the corresponding component key.
        Recursively handles lists and dicts.
        """
        component_keys = ["model"]  # Add more keys here as needed, e.g., ["model", "y_processing"]

        if isinstance(steps, list):
            return [PipelineConfigs._preprocess_steps(step) for step in steps]
        elif isinstance(steps, dict):
            # Merge *_params into component keys
            for key in component_keys:
                params_key = f"{key}_params"
                if key in steps and params_key in steps:
                    steps[key] = {"class": steps[key], "params": steps[params_key]}
                    del steps[params_key]

            # Recurse on values
            for k, v in steps.items():
                steps[k] = PipelineConfigs._preprocess_steps(v)
            return steps
        else:
            return steps

    @staticmethod
    def _load_steps(definition: Union[Dict, List[Any], str]) -> List[Any]:
        """
        Load steps from a definition which can be a dict, list, or string.
        """
        if isinstance(definition, str):
            return PipelineConfigs._load_str_steps(definition)
        elif isinstance(definition, list):
            return definition
        elif isinstance(definition, dict):
            if "pipeline" in definition:
                return definition["pipeline"]
            else:
                raise ValueError("Invalid pipeline definition format. Expected a list, dict with 'pipeline' key, or string.")
        else:
            raise TypeError("Pipeline definition must be a list, dict, or string.")

    @staticmethod
    def _load_str_steps(definition: str) -> List[Any]:
        """
        Load steps from a string definition which can be a JSON or YAML file path, or a JSON/YAML string.
        """
        if definition.endswith('.json') or definition.endswith('.yaml') or definition.endswith('.yml'):
            if not Path(definition).is_file():
                raise FileNotFoundError(f"Configuration file {definition} does not exist.")

            pipeline_definition = None

            if definition.endswith('.json'):
                try:
                    with open(definition, 'r', encoding='utf-8') as f:
                        pipeline_definition = json.load(f)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON file: {definition}") from exc
            elif definition.endswith('.yaml') or definition.endswith('.yml'):
                try:
                    with open(definition, 'r', encoding='utf-8') as f:
                        pipeline_definition = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    raise ValueError(f"Invalid YAML file: {definition}") from exc
        else:
            try:
                pipeline_definition = json.loads(definition)
            except json.JSONDecodeError as exc:
                try:
                    return yaml.safe_load(definition)
                except yaml.YAMLError as exc2:
                    raise ValueError("Invalid pipeline definition string. Must be a valid JSON or YAML format.") from exc2

        if not pipeline_definition:
            raise ValueError("Pipeline definition is empty or invalid.")

        return PipelineConfigs._load_steps(pipeline_definition)

    @staticmethod
    def serializable_steps(steps) -> Any:
        """Remove runtime instances from the configuration"""
        def clean_recursive(obj):
            if isinstance(obj, dict):
                return {k: clean_recursive(v) for k, v in obj.items() if k != "_runtime_instance"}
            elif isinstance(obj, list):
                return [clean_recursive(item) for item in obj]
            else:
                return obj

        return clean_recursive(steps)

    @staticmethod
    def get_hash(steps) -> str:
        """
        Generate a hash for the pipeline configuration.
        """
        import hashlib
        serializable = json.dumps(PipelineConfigs.serializable_steps(steps), sort_keys=True).encode('utf-8')
        return hashlib.md5(serializable).hexdigest()[0:8]

    @staticmethod
    def _get_step_description(step: Any) -> str:
        """Get a human-readable description of a step"""
        if step is None:
            return "No operation"
        if isinstance(step, dict):
            if len(step) == 1:
                key = next(iter(step.keys()))
                return f"{key}"
            elif "class" in step:
                key = f"{step['class'].split('.')[-1]}"
                if "params" in step:
                    params_str = ", ".join(f"{k}={v}" for k, v in step["params"].items())
                    return f"{key}({params_str})"
                return f"{step['class'].split('.')[-1]}"
            elif "model" in step:
                if "class" in step['model']:
                    key = f"{step['model']['class'].split('.')[-1]}"
                elif "function" in step['model']:
                    key = f"{step['model']['function'].split('.')[-1]}"
                else:
                    key = "unknown_model"
                params_str = ""
                if "params" in step['model']:
                    params_str = ", ".join(f"{k}={v}" for k, v in step['model']["params"].items())
                actions = "train"
                if "finetune_params" in step:
                    actions = "(finetune)"
                return f"{actions} {key}({params_str})"
            else:
                return f"Dict with {len(step)} keys"
        elif isinstance(step, list):
            return f"Sub-pipeline ({len(step)} steps)"
        elif isinstance(step, str):
            return step
        else:
            return str(type(step).__name__)

    @classmethod
    def value_of(cls, obj, key):
        """
        Recursively collect all values of a key in a (possibly nested) serialized object.
        Returns a single string with values joined by commas.
        """

        values = []

        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key:
                    values.append(str(v))
                values.extend(cls.value_of(v, key))
        elif isinstance(obj, list):
            for item in obj:
                values.extend(cls.value_of(item, key))

        return values

    @classmethod
    def value_of_str(cls, obj, key):
        """
        Returns a single string of all values for the given key, joined by commas.
        """
        return ", ".join(cls.value_of(obj, key))