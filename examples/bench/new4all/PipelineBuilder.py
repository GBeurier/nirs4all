"""
PipelineBuilder - Build pipeline operations from configuration
"""
from typing import Dict, List, Any, Union, Optional
from PipelineOperation import PipelineOperation
from PipelineConfig import PipelineConfig
from PipelineSerializer import PipelineSerializer
from TransformationOperation import TransformationOperation
from ClusteringOperation import ClusteringOperation, UnclusterOperation
from OptimizationOperation import OptimizationOperation
from DummyOperations import VisualizationOperation  # Keep only visualization as dummy


class PipelineBuilder:
    """Builds pipeline operations from configuration"""

    def __init__(self):
        """Initialize pipeline builder"""
        self.serializer = PipelineSerializer()
        self.operation_builders = {
            'transformation': self._build_transformation,
            'sample_augmentation': self._build_augmentation,
            'feature_augmentation': self._build_augmentation,
            'split': self._build_split,
            'cluster': self._build_cluster,
            'dispatch': self._build_dispatch,
            'stack': self._build_stack,
            'model': self._build_model,
            'finetune_params': self._build_finetune,
            'y_pipeline': self._build_y_pipeline,
            'PlotData': self._build_plot,
            'PlotClusters': self._build_plot,
            'PlotResults': self._build_plot,
            'PlotModelPerformance': self._build_plot,
            'PlotFeatureImportance': self._build_plot,
            'PlotConfusionMatrix': self._build_plot,
            'uncluster': self._build_uncluster
        }

    def build_from_config(self, config: PipelineConfig) -> List[PipelineOperation]:
        """Build pipeline operations from configuration"""
        operations = []

        for step in config.pipeline:
            operation = self._build_operation(step)
            if operation:
                if isinstance(operation, list):
                    operations.extend(operation)
                else:
                    operations.append(operation)

        return operations

    def _build_operation(self, step: Any) -> Union[PipelineOperation, List[PipelineOperation], None]:
        """Build a single operation from configuration step"""
        # Handle string operations (presets or simple operations)
        if isinstance(step, str):
            return self._build_string_operation(step)

        # Handle class instances (already instantiated)
        if hasattr(step, '__class__') and hasattr(step, 'fit'):
            return self._build_transformer_operation(step)

        # Handle dictionaries (complex operations)
        if isinstance(step, dict):
            return self._build_dict_operation(step)

        # Handle serialized class configurations
        if isinstance(step, dict) and 'class' in step:
            # Deserialize and build
            deserialized = self.serializer._deserialize_component(step)
            return self._build_operation(deserialized)

        return None

    def _build_string_operation(self, operation_name: str) -> Optional[PipelineOperation]:
        """Build operation from string name"""
        if operation_name in self.operation_builders:
            return self.operation_builders[operation_name](operation_name)        # Try to import as preset
        try:
            from TransformationOperation import TransformationOperation
            return TransformationOperation(transformer=operation_name)
        except (ImportError, AttributeError):
            return None

    def _build_transformer_operation(self, transformer: Any) -> PipelineOperation:
        """Build transformation operation from transformer instance"""
        from TransformationOperation import TransformationOperation
        return TransformationOperation(transformer=transformer)

    def _build_dict_operation(self, operation_dict: Dict[str, Any]) -> Union[PipelineOperation, List[PipelineOperation], None]:
        """Build operation from dictionary configuration"""
        # Handle different dictionary formats

        # Check for direct class specification
        if 'class' in operation_dict:
            transformer = self.serializer._deserialize_component(operation_dict)
            return self._build_transformer_operation(transformer)

        # Check for known operation types
        for op_type in self.operation_builders:
            if op_type in operation_dict:
                return self.operation_builders[op_type](operation_dict)

        # Check for model specification
        if 'model' in operation_dict:
            return self._build_model(operation_dict)

        return None

    def _build_transformation(self, config: Any) -> PipelineOperation:
        """Build transformation operation"""
        from TransformationOperation import TransformationOperation

        if isinstance(config, str):
            # Simple transformation by name
            return TransformationOperation(transformer=config)
        elif isinstance(config, dict):
            transformer = config.get('transformer', config)
            if isinstance(transformer, dict) and 'class' in transformer:
                transformer = self.serializer._deserialize_component(transformer)
            return TransformationOperation(transformer=transformer)
        else:
            return TransformationOperation(transformer=config)

    def _build_augmentation(self, config: Dict[str, Any]) -> List[PipelineOperation]:
        """Build augmentation operations using TransformationOperation"""
        operations = []

        # Determine augmentation type
        aug_type = 'sample_augmentation' if 'sample_augmentation' in config else 'feature_augmentation'
        augmentations = config[aug_type]

        if not isinstance(augmentations, list):
            augmentations = [augmentations]

        for aug_config in augmentations:
            if aug_config is None:
                continue

            # Handle different augmentation formats
            if isinstance(aug_config, list):
                # Multiple augmentations in parallel
                for sub_aug in aug_config:
                    if sub_aug is not None:
                        transformer = self._resolve_transformer(sub_aug)
                        if transformer:
                            operations.append(TransformationOperation(
                                transformer=transformer,
                                mode=aug_type
                            ))
            else:
                # Single augmentation
                transformer = self._resolve_transformer(aug_config)
                if transformer:
                    operations.append(TransformationOperation(
                        transformer=transformer,
                        mode=aug_type
                    ))

        return operations

    def _build_split(self, config: Any) -> PipelineOperation:
        """Build split operation"""
        from SplitOperation import SplitOperation

        if isinstance(config, dict) and 'class' in config:
            splitter = self.serializer._deserialize_component(config)
        else:
            splitter = config

        return SplitOperation(splitter=splitter)

    def _build_cluster(self, config: Dict[str, Any]) -> PipelineOperation:
        """Build clustering operation"""
        from ClusteringOperation import ClusteringOperation

        cluster_config = config['cluster']
        if isinstance(cluster_config, dict) and 'class' in cluster_config:
            clusterer = self.serializer._deserialize_component(cluster_config)
        else:
            clusterer = cluster_config

        return ClusteringOperation(clusterer=clusterer)

    def _build_dispatch(self, config: Dict[str, Any]) -> PipelineOperation:
        """Build dispatch operation"""
        from DispatchOperation import DispatchOperation

        dispatch_configs = config['dispatch']
        branches = []

        for branch_config in dispatch_configs:
            branch_operations = []

            # Build each operation in the branch
            for op_config in self._flatten_branch_config(branch_config):
                operation = self._build_operation(op_config)
                if operation:
                    if isinstance(operation, list):
                        branch_operations.extend(operation)
                    else:
                        branch_operations.append(operation)

            branches.append(branch_operations)

        return DispatchOperation(branches=branches)

    def _build_stack(self, config: Dict[str, Any]) -> PipelineOperation:
        """Build stacking operation"""
        from StackOperation import StackOperation

        stack_config = config['stack']

        # Main model
        main_model_config = stack_config['model']
        main_model = self._resolve_model(main_model_config)

        # Base learners
        base_learners = []
        for base_config in stack_config.get('base_learners', []):
            base_model = self._resolve_model(base_config.get('model'))
            base_learners.append({
                'model': base_model,
                'y_pipeline': base_config.get('y_pipeline'),
                'finetune_params': base_config.get('finetune_params')
            })

        return StackOperation(
            meta_model=main_model,
            base_learners=base_learners,
            y_pipeline=stack_config.get('y_pipeline')
        )

    def _build_model(self, config: Dict[str, Any]) -> PipelineOperation:
        """Build model operation"""
        from ModelOperation import SklearnModelOperation

        model_config = config.get('model', config)
        model = self._resolve_model(model_config)

        # Extract additional parameters
        y_pipeline = config.get('y_pipeline')
        finetune_params = config.get('finetune_params')

        return SklearnModelOperation(
            model=model,
            y_pipeline=y_pipeline,
            finetune_params=finetune_params
        )

    def _build_finetune(self, config: Any) -> PipelineOperation:
        """Build fine-tuning operation"""
        from OptimizationOperation import OptimizationOperation
        return OptimizationOperation(params=config)

    def _build_y_pipeline(self, config: Any) -> PipelineOperation:
        """Build target preprocessing pipeline"""
        from TransformationOperation import TransformationOperation

        if isinstance(config, list):
            # Multiple transformations
            transformers = []
            for t_config in config:
                transformer = self._resolve_transformer(t_config)
                if transformer:
                    transformers.append(transformer)
            return TransformationOperation(transformer=transformers, target_transformation=True)
        else:
            transformer = self._resolve_transformer(config)
            return TransformationOperation(transformer=transformer, target_transformation=True)

    def _build_plot(self, config: Any) -> PipelineOperation:
        """Build plotting operation"""
        from VisualizationOperation import VisualizationOperation

        plot_type = config if isinstance(config, str) else config.__class__.__name__
        return VisualizationOperation(plot_type=plot_type)

    def _build_uncluster(self, config: Any) -> PipelineOperation:
        """Build uncluster operation"""
        return UnclusterOperation()

    def _resolve_transformer(self, config: Any) -> Any:
        """Resolve transformer from configuration"""
        if isinstance(config, dict) and 'class' in config:
            return self.serializer._deserialize_component(config)
        elif isinstance(config, str):
            # Try to import preset
            try:
                return self.serializer._load_class(f"sklearn.preprocessing.{config}")()
            except:
                return config
        else:
            return config

    def _resolve_model(self, config: Any) -> Any:
        """Resolve model from configuration"""
        if isinstance(config, dict) and 'class' in config:
            return self.serializer._deserialize_component(config)
        elif isinstance(config, str):
            # Try preset mappings
            if config in self.serializer.preset_mappings:
                class_path = self.serializer.preset_mappings[config]
                return self.serializer._load_class(class_path)()
            else:
                return config
        else:
            return config

    def _flatten_branch_config(self, branch_config: Dict[str, Any]) -> List[Any]:
        """Flatten branch configuration into list of operations"""
        operations = []

        # Handle different parts of branch configuration
        if 'model' in branch_config:
            operations.append({'model': branch_config['model']})

        if 'y_pipeline' in branch_config:
            operations.append({'y_pipeline': branch_config['y_pipeline']})

        if 'finetune_params' in branch_config:
            operations.append({'finetune_params': branch_config['finetune_params']})

        if 'stack' in branch_config:
            operations.append({'stack': branch_config['stack']})

        return operations
