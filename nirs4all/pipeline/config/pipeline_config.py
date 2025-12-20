"""
PipelineConfigs.py
"""

import json
import logging
from pathlib import Path
from typing import List, Any, Dict, Union
import yaml

from .component_serialization import serialize_component
from .generator import expand_spec, expand_spec_with_choices, count_combinations


class _ShortNameFormatter(logging.Formatter):
    """Formatter that strips 'nirs4all.' prefix from logger names."""

    def format(self, record: logging.LogRecord) -> str:
        # Strip nirs4all prefix for cleaner output
        if record.name.startswith("nirs4all."):
            record.name = record.name[9:]  # len("nirs4all.") = 9
        return super().format(record)


# Configure logging with simplified module names
_handler = logging.StreamHandler()
_handler.setFormatter(_ShortNameFormatter("%(levelname)s: %(name)s: %(message)s"))
logging.root.addHandler(_handler)
logging.root.setLevel(logging.INFO)

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
        self.steps = serialize_component(self.steps)

        ## Generation
        self.has_configurations = False
        self.generator_choices: List[List[Dict[str, Any]]] = []  # Choices for each pipeline
        was_expanded = False

        if self._has_gen_keys(self.steps):
            count = count_combinations(self.steps)
            if count > max_generation_count:
                raise ValueError(f"Configuration expansion would generate {count} configurations, exceeding the limit of {max_generation_count}. Please simplify your configuration.")
            # Always expand generator syntax, even if count=1
            # The _or_, _range_ etc. must be replaced with actual values
            if count >= 1:
                self.has_configurations = count > 1
                # Use expand_spec_with_choices to track generator choices
                expanded_with_choices = expand_spec_with_choices(self.steps)
                self.steps = [config for config, choices in expanded_with_choices]
                self.generator_choices = [choices for config, choices in expanded_with_choices]
                was_expanded = True

        if not was_expanded:
            self.steps = [self.steps]  # Wrap single configuration in a list
            self.generator_choices = [[]]  # No choices for single config

        ## Name and hash
        if name == "":
            name = "config"
        self.names = [
            name + "_" + self.get_hash(steps)[0:6]
            for steps in self.steps
        ]

        # print(f"✅ {len(self.steps)} pipeline configuration(s).")

    @staticmethod
    def _has_gen_keys(obj: Any, skip_branch: bool = True) -> bool:
        """Recursively check if the configuration contains generator keys.

        Args:
            obj: Configuration object to check
            skip_branch: If True, skip generator detection inside 'branch' keys
                         (these are handled by BranchController at runtime)

        Returns:
            True if generator keys are found at the pipeline level
        """
        if isinstance(obj, dict):
            # Skip content inside 'branch' key - BranchController handles those
            if skip_branch and "branch" in obj:
                # Check other keys but skip branch content
                return any(
                    PipelineConfigs._has_gen_keys(v, skip_branch)
                    for k, v in obj.items()
                    if k != "branch"
                )

            if "_or_" in obj or "_range_" in obj:
                return True
            return any(PipelineConfigs._has_gen_keys(v, skip_branch) for v in obj.values())
        elif isinstance(obj, list):
            return any(PipelineConfigs._has_gen_keys(item, skip_branch) for item in obj)
        return False

    @staticmethod
    def _preprocess_steps(steps: Any) -> Any:
        """
        Preprocess steps to merge *_params into the corresponding component key.
        Recursively handles lists and dicts.
        """

        if isinstance(steps, list):
            return [PipelineConfigs._preprocess_steps(step) for step in steps]
        elif isinstance(steps, dict):
            # Find all XX/XX_params pairs and merge them
            result = steps.copy()

            # Find all keys ending with '_params'
            params_keys = [k for k in result.keys() if k.endswith('_params')]

            for params_key in params_keys:
                # Get the base key (remove '_params' suffix)
                base_key = params_key[:-7]  # Remove '_params'

                if base_key in result:
                    # Merge base_key and params_key into standard format
                    base_value = result[base_key]
                    params_value = result[params_key]

                    # Convert to standard {"class": ..., "params": ...} format
                    result[base_key] = {
                        "class": base_value,
                        "params": params_value
                    }

                    # Remove the params key
                    del result[params_key]

            # Also normalize bare classes in component-like keys to {"class": ...} format
            # This ensures consistent serialization for cases like {"y_processing": MinMaxScaler}
            import inspect
            for key, value in list(result.items()):
                # If the value is a class and the key looks like a component key
                # (not "class" or "params" which have special meaning)
                if (inspect.isclass(value) and
                    key not in ["class", "params"] and
                    not key.endswith("_params")):
                    result[key] = {"class": value}

            # Also handle direct {"class": ClassObject, "params": {...}} format
            # to ensure consistent serialization
            if "class" in result and "params" in result:
                # This is already in the right structure, just ensure class gets serialized properly
                pass

            # Recurse on values
            for k, v in result.items():
                result[k] = PipelineConfigs._preprocess_steps(v)
            return result
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
        """Load steps from a string definition which can be a JSON or YAML file path, or a JSON/YAML string.

        Args:
            definition: Path to a JSON/YAML file, or a JSON/YAML string.

        Returns:
            List of pipeline steps.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If the config file has invalid JSON/YAML syntax.
        """
        if definition.endswith('.json') or definition.endswith('.yaml') or definition.endswith('.yml'):
            if not Path(definition).is_file():
                raise FileNotFoundError(
                    f"Configuration file does not exist: {definition}\n"
                    f"Please check the file path and try again."
                )

            pipeline_definition = None

            if definition.endswith('.json'):
                try:
                    with open(definition, 'r', encoding='utf-8') as f:
                        pipeline_definition = json.load(f)
                except json.JSONDecodeError as exc:
                    # Provide detailed error message with line number
                    raise ValueError(
                        f"Invalid JSON in {definition}\n"
                        f"Error at line {exc.lineno}, column {exc.colno}:\n"
                        f"  {exc.msg}\n\n"
                        f"Common JSON issues:\n"
                        f"  - Missing or extra commas\n"
                        f"  - Unquoted strings\n"
                        f"  - Trailing commas (not allowed in JSON)\n"
                        f"  - Single quotes instead of double quotes"
                    ) from exc
            elif definition.endswith('.yaml') or definition.endswith('.yml'):
                try:
                    with open(definition, 'r', encoding='utf-8') as f:
                        pipeline_definition = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    # Extract line number from YAML error if available
                    if hasattr(exc, 'problem_mark') and exc.problem_mark:
                        mark = exc.problem_mark
                        line_num = mark.line + 1
                        col_num = mark.column + 1
                        problem = getattr(exc, 'problem', 'Unknown error')
                        context = getattr(exc, 'context', '')
                        error_details = f"  {problem}"
                        if context:
                            error_details += f" ({context})"
                        raise ValueError(
                            f"Invalid YAML in {definition}\n"
                            f"Error at line {line_num}, column {col_num}:\n"
                            f"{error_details}\n\n"
                            f"Common YAML issues:\n"
                            f"  - Incorrect indentation (use spaces, not tabs)\n"
                            f"  - Missing colon after key names\n"
                            f"  - Unescaped special characters\n"
                            f"  - Mixing tabs and spaces"
                        ) from exc
                    else:
                        raise ValueError(
                            f"Invalid YAML in {definition}:\n"
                            f"  {exc}\n\n"
                            f"Please check your YAML syntax."
                        ) from exc
        else:
            try:
                pipeline_definition = json.loads(definition)
            except json.JSONDecodeError as exc:
                try:
                    return yaml.safe_load(definition)
                except yaml.YAMLError as exc2:
                    raise ValueError(
                        "Invalid pipeline definition string.\n"
                        "Must be a valid JSON or YAML format.\n\n"
                        f"JSON error: {exc.msg} at line {exc.lineno}\n"
                        f"YAML error: {exc2}"
                    ) from exc2

        if not pipeline_definition:
            raise ValueError(
                f"Pipeline definition is empty or invalid.\n"
                f"The configuration file must contain a 'pipeline' key with a list of steps."
            )

        return PipelineConfigs._load_steps(pipeline_definition)

    @staticmethod
    def get_hash(steps) -> str:
        """
        Generate a hash for the pipeline configuration.

        All objects are fully JSON-serializable (no _runtime_instance).
        No need for default=str hack anymore.
        """
        import hashlib
        serializable = json.dumps(steps, sort_keys=True).encode('utf-8')
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
                # Check for custom model name first
                if "name" in step:
                    custom_name = step["name"]
                    actions = "train"
                    if "finetune_params" in step:
                        actions = "(finetune)"
                    return f"{actions} {custom_name}"

                # Use model class name if no custom name
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