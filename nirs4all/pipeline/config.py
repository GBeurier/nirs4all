"""
PipelineConfig.py
"""

import json
import logging
from pathlib import Path
from typing import List, Any, Dict, Union
import yaml

from .serialization import serialize_component

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineConfig:
    """
    Class to hold the configuration for a pipeline.
    """
    def __init__(self, definition: Union[Dict, List[Any], str], name: str = "", description: str = "No description provided"):
        """
        Initialize the pipeline configuration.
        """
        self.description = description
        self.steps = self._load_steps(definition)
        self.steps = serialize_component(self.steps, include_runtime=True)
        self.name = self.get_hash() if name == "" else name

    def _load_steps(self, definition: Union[Dict, List[Any], str]) -> List[Any]:
        """
        Load steps from a definition which can be a dict, list, or string.
        """
        if isinstance(definition, str):
            return self._load_str_steps(definition)
        elif isinstance(definition, list):
            return definition
        elif isinstance(definition, dict):
            if "pipeline" in definition:
                return definition["pipeline"]
            else:
                raise ValueError("Invalid pipeline definition format. Expected a list, dict with 'pipeline' key, or string.")
        else:
            raise TypeError("Pipeline definition must be a list, dict, or string.")

    def _load_str_steps(self, definition: str) -> List[Any]:
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

        return self._load_steps(pipeline_definition)

    def __repr__(self):
        return f"PipelineConfig(name={self.name}, description={self.description}\nsteps={self.json_steps()})"

    def save(self, filepath: str) -> None:
        """
        Save the pipeline configuration to a file in JSON or YAML format.
        """
        path = Path(filepath)
        serializable_steps = self.serializable_steps()
        if not serializable_steps:
            raise ValueError("No steps to save in the pipeline configuration.")

        if path.suffix.lower() == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(serializable_steps, f, indent=2, default=str)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(serializable_steps, f, default_flow_style=False)
        else:
            raise ValueError("Unsupported format. Use 'json' or 'yaml'.")

        logger.info("Pipeline configuration saved to %s", path)

    def json_steps(self, indent: int = 2, include_runtime: bool = True) -> str:
        """
        Display the pipeline steps as indented JSON while preserving
        the representation of _runtime_instance objects.
        """
        def make_displayable(obj):
            """Convert objects to displayable format, preserving runtime instance info"""
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    if k == "_runtime_instance" and include_runtime:
                        result[k] = f"<{type(v).__name__} object at {hex(id(v))}>"
                    else:
                        result[k] = make_displayable(v)
                return result
            elif isinstance(obj, list):
                return [make_displayable(item) for item in obj]
            else:
                return obj

        displayable_steps = make_displayable(self.steps)
        return json.dumps(displayable_steps, indent=indent, default=str)

    def serializable_steps(self) -> Any:
        """Remove runtime instances from the configuration"""
        def clean_recursive(obj):
            if isinstance(obj, dict):
                return {k: clean_recursive(v) for k, v in obj.items() if k != "_runtime_instance"}
            elif isinstance(obj, list):
                return [clean_recursive(item) for item in obj]
            else:
                return obj

        return clean_recursive(self.steps)

    def get_hash(self) -> str:
        """
        Generate a hash for the pipeline configuration.
        """
        import hashlib
        serializable = json.dumps(self.serializable_steps(), sort_keys=True).encode('utf-8')
        return hashlib.md5(serializable).hexdigest()[0:8]
