"""
Pipeline - Main pipeline execution engine with configuration support
"""
import numpy as np
from typing import Dict, List, Optional, Any, Union
import yaml
import json
from pathlib import Path
from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext
from PipelineConfig import PipelineConfig
from PipelineBuilder import PipelineBuilder
from PipelineSerializer import PipelineSerializer


class Pipeline:
    """Main pipeline execution engine with configuration support"""

    def __init__(self, name: str = "Pipeline", config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline

        Parameters:
        -----------
        name : str
            Name of the pipeline
        config : PipelineConfig, optional
            Pipeline configuration
        """
        self.name = name
        self.operations = []
        self.context = PipelineContext()
        self.config = config
        self.builder = PipelineBuilder()
        self.serializer = PipelineSerializer()

        # Execution state
        self.is_fitted = False
        self.execution_history = []
        self.current_step = 0

        # Build operations from config if provided
        if config:
            self.operations = self.builder.build_from_config(config)

    def add_operation(self, operation: PipelineOperation) -> 'Pipeline':
        """Add operation to pipeline"""
        self.operations.append(operation)
        return self

    def add_operations(self, operations: List[PipelineOperation]) -> 'Pipeline':
        """Add multiple operations to pipeline"""
        self.operations.extend(operations)
        return self

    def execute(self, dataset: SpectraDataset) -> SpectraDataset:
        """Execute the complete pipeline"""
        print(f"Executing pipeline '{self.name}' with {len(self.operations)} operations")

        # Reset execution state
        self.current_step = 0
        self.execution_history = []

        try:
            for i, operation in enumerate(self.operations):
                self.current_step = i + 1

                print(f"\nStep {self.current_step}/{len(self.operations)}: {operation.get_name()}")

                # Check if operation can execute (if method exists)
                if hasattr(operation, 'can_execute') and not operation.can_execute(dataset, self.context):
                    error_msg = f"Operation {operation.get_name()} cannot execute"
                    print(f"Skipping: {error_msg}")

                    self.execution_history.append({
                        'step': self.current_step,
                        'operation': operation.get_name(),
                        'status': 'skipped',
                        'reason': error_msg
                    })
                    continue

                # Execute operation
                try:
                    operation.execute(dataset, self.context)

                    self.execution_history.append({
                        'step': self.current_step,
                        'operation': operation.get_name(),
                        'status': 'success'
                    })

                    print(f"✓ {operation.get_name()} completed successfully")

                except Exception as e:
                    error_msg = f"Operation {operation.get_name()} failed: {str(e)}"
                    print(f"✗ {error_msg}")

                    self.execution_history.append({
                        'step': self.current_step,
                        'operation': operation.get_name(),
                        'status': 'failed',
                        'error': str(e)
                    })

                    # Check if we should continue on error
                    continue_on_error = getattr(self.context, 'continue_on_error', False)
                    if not continue_on_error:
                        raise RuntimeError(error_msg) from e

            self.is_fitted = True
            print(f"\nPipeline '{self.name}' completed successfully!")

        except Exception as e:
            print(f"\nPipeline '{self.name}' failed at step {self.current_step}: {str(e)}")
            raise

        return dataset

    def fit(self, dataset: SpectraDataset) -> 'Pipeline':
        """Fit the pipeline (same as execute for compatibility)"""
        self.execute(dataset)
        return self

    def transform(self, dataset: SpectraDataset) -> SpectraDataset:
        """Transform data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        # For now, execute again (in practice, we'd want to distinguish fit/transform)
        return self.execute(dataset)

    def fit_transform(self, dataset: SpectraDataset) -> SpectraDataset:
        """Fit and transform in one step"""
        return self.fit(dataset).transform(dataset)

    def predict(self, dataset: SpectraDataset) -> Dict[str, np.ndarray]:
        """Make predictions using the pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before predict")

        # Execute pipeline in prediction mode
        old_mode = getattr(self.context, 'mode', 'fit')
        setattr(self.context, 'mode', 'predict')

        try:
            self.execute(dataset)
            return getattr(self.context, 'predictions', {})
        finally:
            setattr(self.context, 'mode', old_mode)

    @classmethod
    def from_config(cls, config: Union[str, Path, Dict[str, Any], PipelineConfig], name: Optional[str] = None) -> 'Pipeline':
        """Create pipeline from configuration"""
        if isinstance(config, PipelineConfig):
            pipeline_config = config
        elif isinstance(config, (str, Path)):
            pipeline_config = PipelineConfig.from_file(config)
        else:
            # Dictionary config (like sample.py)
            if 'experiment' in config and 'pipeline' in config:
                # Already in expected format
                pipeline_config = PipelineConfig.from_dict(config)
            else:
                # Python config format (sample.py style), need to serialize first
                serializer = PipelineSerializer()
                serialized_config = serializer.serialize_config(config)
                pipeline_config = PipelineConfig.from_dict(serialized_config)

        # Create pipeline with config
        pipeline_name = name or pipeline_config.name
        pipeline = cls(name=pipeline_name, config=pipeline_config)

        return pipeline

    @classmethod
    def from_python_config(cls, python_config: Dict[str, Any], name: Optional[str] = None) -> 'Pipeline':
        """Create pipeline from Python configuration (like sample.py)"""
        return cls.from_config(python_config, name)

    def to_config(self) -> PipelineConfig:
        """Export pipeline to configuration"""
        if self.config:
            return self.config.clone()

        # Create config from current state
        config_dict = {
            'name': self.name,
            'experiment': {},
            'pipeline': self._operations_to_config(),
            'metadata': {
                'is_fitted': self.is_fitted,
                'execution_history': self.execution_history
            }
        }

        return PipelineConfig.from_dict(config_dict)

    def _operations_to_config(self) -> List[Dict[str, Any]]:
        """Convert operations to configuration format"""
        config_list = []

        for operation in self.operations:
            # Try to serialize the operation
            if hasattr(operation, 'to_config'):
                config_list.append(operation.to_config())
            else:
                # Basic serialization
                config_list.append({
                    'type': operation.__class__.__name__,
                    'name': operation.get_name()
                })

        return config_list

    def save_config(self, filepath: Union[str, Path]) -> None:
        """Save pipeline configuration to file"""
        config = self.to_config()
        config.save(filepath)

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution"""
        if not self.execution_history:
            return {'status': 'not_executed'}

        successful_steps = [h for h in self.execution_history if h['status'] == 'success']
        failed_steps = [h for h in self.execution_history if h['status'] == 'failed']
        skipped_steps = [h for h in self.execution_history if h['status'] == 'skipped']

        return {
            'pipeline_name': self.name,
            'total_operations': len(self.operations),
            'executed_steps': len(self.execution_history),
            'successful_steps': len(successful_steps),
            'failed_steps': len(failed_steps),
            'skipped_steps': len(skipped_steps),
            'is_fitted': self.is_fitted,
            'execution_history': self.execution_history
        }

    def get_dataset_summary(self, dataset: SpectraDataset) -> Dict[str, Any]:
        """Get summary of dataset state after pipeline"""
        summary = {
            'n_sources': len(getattr(dataset, 'X', {})),
            'sources': {}
        }

        # Safely access dataset attributes
        if hasattr(dataset, 'X') and dataset.X:
            for source_name, source_data in dataset.X.items():
                summary['sources'][source_name] = {
                    'shape': source_data.shape,
                    'dtype': str(source_data.dtype)
                }

        # Add target information if available
        if hasattr(dataset, 'target_manager') and dataset.target_manager is not None:
            if hasattr(dataset.target_manager, 'get_summary'):
                target_info = dataset.target_manager.get_summary()
                summary['targets'] = target_info

        # Add context information
        summary['context'] = {
            'active_sources': getattr(self.context, 'active_sources', []),
            'has_predictions': hasattr(self.context, 'predictions'),
            'has_splits': hasattr(self.context, 'data_splits')
        }

        return summary

    def validate(self) -> List[str]:
        """Validate pipeline configuration"""
        issues = []

        if not self.operations:
            issues.append("Pipeline has no operations")

        # Check operation dependencies
        for i, operation in enumerate(self.operations):
            # Basic validation - each operation should have required methods
            if not hasattr(operation, 'execute'):
                issues.append(f"Operation {i} ({operation.__class__.__name__}) missing execute method")

            if not hasattr(operation, 'get_name'):
                issues.append(f"Operation {i} ({operation.__class__.__name__}) missing get_name method")

        # Validate config if present
        if self.config:
            config_issues = self.config.validate()
            issues.extend(config_issues)

        return issues

    def clone(self) -> 'Pipeline':
        """Create a copy of the pipeline"""
        if self.config:
            return Pipeline(name=self.name, config=self.config.clone())
        else:
            # Clone without config
            new_pipeline = Pipeline(name=self.name)
            new_pipeline.operations = self.operations.copy()
            return new_pipeline

    def __repr__(self) -> str:
        """String representation of pipeline"""
        return f"Pipeline(name='{self.name}', operations={len(self.operations)}, fitted={self.is_fitted})"

    def __len__(self) -> int:
        """Number of operations in pipeline"""
        return len(self.operations)
