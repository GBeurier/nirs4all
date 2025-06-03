"""
Pipeline - Main pipeline execution engine
"""
import numpy as np
from typing import Dict, List, Optional, Any, Union
import yaml
import json
from pathlib import Path
from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext
# from OperationFactory import OperationFactory  # Avoid circular import


class Pipeline:
    """Main pipeline execution engine"""

    def __init__(self, name: str = "Pipeline"):
        """
        Initialize pipeline

        Parameters:
        -----------
        name : str
            Name of the pipeline
        """
        self.name = name
        self.operations = []
        self.context = PipelineContext()
        self.factory = None  # Will be created on demand to avoid circular import

        # Execution state
        self.is_fitted = False
        self.execution_history = []
        self.current_step = 0

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

                # Check if operation can execute
                if not operation.can_execute(dataset, self.context):
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
                    if not getattr(self.context, 'continue_on_error', False):
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
        self.context.mode = 'predict'

        try:
            self.execute(dataset)
            return getattr(self.context, 'predictions', {})
        finally:
            self.context.mode = old_mode

    @classmethod
    def from_config(cls, config: Union[str, Path, Dict[str, Any]], name: str = None) -> 'Pipeline':
        """Create pipeline from configuration"""
        if isinstance(config, (str, Path)):
            config_path = Path(config)

            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        else:
            config_dict = config

        # Create pipeline
        pipeline_name = name or config_dict.get('name', 'ConfigPipeline')
        pipeline = cls(pipeline_name)

        # Set pipeline-level configuration
        pipeline_config = config_dict.get('pipeline', {})
        pipeline.context.continue_on_error = pipeline_config.get('continue_on_error', False)
          # Create operations from config
        operations_config = config_dict.get('operations', [])
        from OperationFactory import OperationFactory  # Import here to avoid circular dependency
        factory = OperationFactory()

        for op_config in operations_config:
            operation = factory.create_operation(op_config)
            pipeline.add_operation(operation)

        return pipeline

    def to_config(self) -> Dict[str, Any]:
        """Export pipeline to configuration"""
        config = {
            'name': self.name,
            'pipeline': {
                'continue_on_error': getattr(self.context, 'continue_on_error', False)
            },
            'operations': []
        }

        for operation in self.operations:
            # This would need to be implemented in each operation
            if hasattr(operation, 'to_config'):
                config['operations'].append(operation.to_config())
            else:
                config['operations'].append({
                    'type': operation.__class__.__name__,
                    'name': operation.get_name()
                })

        return config

    def save_config(self, filepath: Union[str, Path]) -> None:
        """Save pipeline configuration to file"""
        filepath = Path(filepath)
        config = self.to_config()

        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {filepath.suffix}")

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
            'n_sources': len(dataset.X),
            'sources': {}
        }

        for source_name, source_data in dataset.X.items():
            summary['sources'][source_name] = {
                'shape': source_data.shape,
                'dtype': str(source_data.dtype)
            }

        # Add target information if available
        if hasattr(dataset, 'target_manager') and dataset.target_manager is not None:
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

            if not hasattr(operation, 'can_execute'):
                issues.append(f"Operation {i} ({operation.__class__.__name__}) missing can_execute method")

        return issues

    def __repr__(self) -> str:
        """String representation of pipeline"""
        return f"Pipeline(name='{self.name}', operations={len(self.operations)}, fitted={self.is_fitted})"

    def __len__(self) -> int:
        """Number of operations in pipeline"""
        return len(self.operations)
