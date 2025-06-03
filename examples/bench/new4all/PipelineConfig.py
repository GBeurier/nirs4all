"""
PipelineConfig - Configuration management for pipelines
"""
import json
import yaml
import copy
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Pipeline configuration storage and management"""

    name: str = "Pipeline"
    experiment: Optional[Dict[str, Any]] = None
    pipeline: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing"""
        if self.experiment is None:
            self.experiment = {}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary"""
        return cls(
            name=config_dict.get('name', 'Pipeline'),
            experiment=config_dict.get('experiment', {}),
            pipeline=config_dict.get('pipeline', []),
            metadata=config_dict.get('metadata', {})
        )

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> 'PipelineConfig':
        """Load config from file (JSON or YAML)"""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r') as f:
            if filepath.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif filepath.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {filepath.suffix}")

        return cls.from_dict(config_dict)
    @classmethod
    def from_python_config(cls, python_config: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from Python dictionary (like sample.py)"""
        from PipelineSerializer import PipelineSerializer

        serializer = PipelineSerializer()
        serialized_config = serializer.serialize_config(python_config)

        return cls.from_dict(serialized_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'name': self.name,
            'experiment': self.experiment,
            'pipeline': self.pipeline,
            'metadata': self.metadata
        }

    def to_python_config(self) -> Dict[str, Any]:
        """Convert config to Python objects (like sample.py)"""
        from PipelineSerializer import PipelineSerializer

        serializer = PipelineSerializer()
        return serializer.deserialize_config(self.to_dict())

    def save(self, filepath: Union[str, Path]) -> None:
        """Save config to file"""
        filepath = Path(filepath)
        config_dict = self.to_dict()

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            if filepath.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif filepath.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {filepath.suffix}")

    def validate(self) -> List[str]:
        """Validate configuration"""
        issues = []

        if not self.name:
            issues.append("Pipeline name is required")

        if not self.pipeline:
            issues.append("Pipeline must have at least one operation")

        # Validate experiment config
        if self.experiment:
            if 'action' not in self.experiment:
                issues.append("Experiment must specify 'action'")
            if 'dataset' not in self.experiment:
                issues.append("Experiment must specify 'dataset'")

        return issues

    def clone(self) -> 'PipelineConfig':
        """Create a deep copy of the config"""
        return PipelineConfig.from_dict(copy.deepcopy(self.to_dict()))
