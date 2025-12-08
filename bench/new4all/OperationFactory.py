"""
OperationFactory - Factory for creating pipeline operations from configuration
"""
import numpy as np
from typing import Dict, List, Optional, Any, Type, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC

from PipelineOperation import PipelineOperation
from TransformationOperation import TransformationOperation
from ModelOperation import SklearnModelOperation
from MergeSourcesOperation import MergeSourcesOperation
from ClusteringOperation import ClusteringOperation
from SplitOperation import SplitOperation
from DispatchOperation import DispatchOperation


class OperationFactory:
    """Factory for creating pipeline operations from configuration"""

    def __init__(self):
        """Initialize operation factory"""        # Register operation types
        self.operation_types = {
            'transformation': TransformationOperation,
            'model': SklearnModelOperation,
            'merge_sources': MergeSourcesOperation,
            'clustering': ClusteringOperation,
            'split': SplitOperation,
            'dispatch': DispatchOperation
        }

        # Register sklearn transformers
        self.sklearn_transformers = {
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            'RobustScaler': RobustScaler,
            'PCA': PCA
        }

        # Register sklearn models
        self.sklearn_models = {
            'RandomForestRegressor': RandomForestRegressor,
            'RandomForestClassifier': RandomForestClassifier,
            'LinearRegression': LinearRegression,
            'LogisticRegression': LogisticRegression,
            'SVR': SVR,
            'SVC': SVC
        }

    def create_operation(self, config: Dict[str, Any]) -> PipelineOperation:
        """
        Create operation from configuration

        Parameters:
        -----------
        config : dict
            Operation configuration with 'type' and parameters

        Returns:
        --------
        PipelineOperation
            Created operation instance
        """
        operation_type = config.get('type')
        if not operation_type:
            raise ValueError("Operation config must specify 'type'")

        if operation_type not in self.operation_types:
            raise ValueError(f"Unknown operation type: {operation_type}")

        # Get operation class
        operation_class = self.operation_types[operation_type]

        # Create operation based on type
        if operation_type == 'transformation':
            return self.create_transformation_operation(config)
        elif operation_type == 'model':
            return self.create_model_operation(config)
        elif operation_type == 'merge_sources':
            return self.create_merge_sources_operation(config)
        elif operation_type == 'augmentation':
            return self.create_augmentation_operation(config)
        elif operation_type == 'clustering':
            return self.create_clustering_operation(config)
        elif operation_type == 'split':
            return self.create_split_operation(config)
        elif operation_type == 'dispatch':
            return self.create_dispatch_operation(config)
        else:
            # Generic operation creation
            params = {k: v for k, v in config.items() if k != 'type'}
            return operation_class(**params)

    def create_transformation_operation(self, config: Dict[str, Any]) -> TransformationOperation:
        """Create transformation operation from config"""
        transformer_config = config.get('transformer', {})
        transformer_type = transformer_config.get('type')

        if not transformer_type:
            raise ValueError("Transformation operation requires transformer.type")

        # Create sklearn transformer
        if transformer_type in self.sklearn_transformers:
            transformer_class = self.sklearn_transformers[transformer_type]
            transformer_params = {k: v for k, v in transformer_config.items() if k != 'type'}
            transformer = transformer_class(**transformer_params)
        else:
            raise ValueError(f"Unknown transformer type: {transformer_type}")        # Create operation
        operation_params = {k: v for k, v in config.items() if k not in ['type', 'transformer']}

        return TransformationOperation(
            transformer=transformer,
            **operation_params
        )

    def create_model_operation(self, config: Dict[str, Any]) -> SklearnModelOperation:
        """Create model operation from config"""
        model_config = config.get('model', {})
        model_type = model_config.get('type')

        if not model_type:
            raise ValueError("Model operation requires model.type")

        # Create sklearn model
        if model_type in self.sklearn_models:
            model_class = self.sklearn_models[model_type]
            model_params = {k: v for k, v in model_config.items() if k != 'type'}
            model = model_class(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")        # Map config parameters to ModelOperation parameters
        operation_params = {}
        for k, v in config.items():
            if k not in ['type', 'model']:
                if k == 'fit_partition':
                    operation_params['train_on'] = v
                elif k == 'predict_partitions':
                    operation_params['predict_on'] = v
                else:
                    operation_params[k] = v

        return SklearnModelOperation(
            model=model,
            **operation_params
        )

    def create_merge_sources_operation(self, config: Dict[str, Any]) -> MergeSourcesOperation:
        """Create merge sources operation from config"""
        params = {k: v for k, v in config.items() if k != 'type'}
        return MergeSourcesOperation(**params)
    def create_clustering_operation(self, config: Dict[str, Any]) -> ClusteringOperation:
        """Create clustering operation from config"""
        params = {k: v for k, v in config.items() if k != 'type'}
        return ClusteringOperation(**params)

    def create_split_operation(self, config: Dict[str, Any]) -> SplitOperation:
        """Create split operation from config"""
        params = {k: v for k, v in config.items() if k != 'type'}
        return SplitOperation(**params)

    def create_dispatch_operation(self, config: Dict[str, Any]) -> DispatchOperation:
        """Create dispatch operation from config"""
        # Handle both old and new format
        if 'branches' in config:
            # Old format with branches - convert to new format
            operations = []
            for branch in config['branches']:
                branch_operations = branch.get('operations', [])
                for op_config in branch_operations:
                    operations.append(self.create_operation(op_config))
        else:
            # New format with direct operations list
            operations_config = config.get('operations', [])
            operations = [self.create_operation(op_config) for op_config in operations_config]        # Create dispatch operation with supported parameters only
        supported_params = ['dispatch_strategy', 'max_workers', 'merge_results', 'execution_mode']
        params = {k: v for k, v in config.items() if k in supported_params}

        # Set default strategy for branching
        if 'dispatch_strategy' not in params:
            if 'branches' in config:
                # For branching scenarios, don't merge results to preserve separate branches
                params['dispatch_strategy'] = 'parallel'
                params['merge_results'] = False
            else:
                params['dispatch_strategy'] = 'parallel'

        return DispatchOperation(operations=operations, **params)

    def register_operation_type(self, name: str, operation_class: Type[PipelineOperation]) -> None:
        """Register new operation type"""
        self.operation_types[name] = operation_class

    def register_transformer(self, name: str, transformer_class: Type) -> None:
        """Register new transformer type"""
        self.sklearn_transformers[name] = transformer_class

    def register_model(self, name: str, model_class: Type) -> None:
        """Register new model type"""
        self.sklearn_models[name] = model_class

    def get_available_operations(self) -> List[str]:
        """Get list of available operation types"""
        return list(self.operation_types.keys())

    def get_available_transformers(self) -> List[str]:
        """Get list of available transformer types"""
        return list(self.sklearn_transformers.keys())

    def get_available_models(self) -> List[str]:
        """Get list of available model types"""
        return list(self.sklearn_models.keys())

    def create_operation_from_preset(self, preset_name: str, **kwargs) -> PipelineOperation:
        """Create operation from preset configuration"""
        presets = {
            'standard_scaler': {
                'type': 'transformation',
                'transformer': {'type': 'StandardScaler'},
                'fit_partition': ['train'],
                'transform_partitions': ['train', 'val', 'test']
            },
            'pca': {
                'type': 'transformation',
                'transformer': {'type': 'PCA', 'n_components': 10},
                'fit_partition': ['train'],
                'transform_partitions': ['train', 'val', 'test']
            },
            'random_forest_classifier': {
                'type': 'model',
                'model': {'type': 'RandomForestClassifier', 'n_estimators': 100},
                'target_representation': 'classification',
                'fit_partition': ['train'],
                'predict_partitions': ['val', 'test']
            },
            'random_forest_regressor': {
                'type': 'model',
                'model': {'type': 'RandomForestRegressor', 'n_estimators': 100},
                'target_representation': 'regression',
                'fit_partition': ['train'],
                'predict_partitions': ['val', 'test']
            },
            'train_val_test_split': {
                'type': 'split',
                'split_strategy': 'random',
                'split_ratios': {'train': 0.7, 'val': 0.2, 'test': 0.1},
                'stratified': True
            },
            'noise_augmentation': {
                'type': 'augmentation',
                'augmentation_type': 'noise',
                'noise_level': 0.01,
                'preserve_original': True
            },
            'kmeans_clustering': {
                'type': 'clustering',
                'clustering_method': 'kmeans',
                'n_clusters': 3,
                'store_centroids': True
            }
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}")

        # Merge preset config with provided kwargs
        config = presets[preset_name].copy()

        # Handle nested parameter updates
        for key, value in kwargs.items():
            if key in ['transformer', 'model'] and isinstance(value, dict):
                if key in config:
                    config[key].update(value)
                else:
                    config[key] = value
            else:
                config[key] = value

        return self.create_operation(config)

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate operation configuration"""
        issues = []

        # Check required fields
        if 'type' not in config:
            issues.append("Operation config missing 'type' field")
            return issues

        operation_type = config['type']

        if operation_type not in self.operation_types:
            issues.append(f"Unknown operation type: {operation_type}")
            return issues

        # Type-specific validation
        if operation_type == 'transformation':
            transformer_config = config.get('transformer', {})
            if 'type' not in transformer_config:
                issues.append("Transformation operation missing transformer.type")
            elif transformer_config['type'] not in self.sklearn_transformers:
                issues.append(f"Unknown transformer type: {transformer_config['type']}")

        elif operation_type == 'model':
            model_config = config.get('model', {})
            if 'type' not in model_config:
                issues.append("Model operation missing model.type")
            elif model_config['type'] not in self.sklearn_models:
                issues.append(f"Unknown model type: {model_config['type']}")

        elif operation_type == 'split':
            split_ratios = config.get('split_ratios', {})
            if split_ratios:
                total_ratio = sum(split_ratios.values())
                if abs(total_ratio - 1.0) > 1e-6:
                    issues.append(f"Split ratios sum to {total_ratio}, should sum to 1.0")

        return issues
