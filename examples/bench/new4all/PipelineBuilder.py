"""
PipelineBuilder - Build pipeline operations from configuration
"""
from typing import Dict, List, Any, Union, Optional
import inspect
from PipelineOperation import PipelineOperation
from PipelineConfig import PipelineConfig
from PipelineSerializer import PipelineSerializer
from TransformationOperation import TransformationOperation
from ClusteringOperation import ClusteringOperation, UnclusterOperation
from OptimizationOperation import OptimizationOperation
from DummyOperations import VisualizationOperation


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
            return self._build_string_operation(step)        # Handle class instances (sklearn objects)
        if hasattr(step, '__class__'):
            # Check if it's a splitter (has split method but not fit method for transformation)
            if hasattr(step, 'split') and not (hasattr(step, 'fit') and hasattr(step, 'transform')):
                return self._build_split(step)
            # Check if it's a transformer (has fit method)
            elif hasattr(step, 'fit'):
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
            return self.operation_builders[operation_name](operation_name)

        # Try to import as preset
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
        for key, builder in self.operation_builders.items():
            if key in operation_dict:
                return builder(operation_dict)

        # Handle as a complex model configuration
        if 'model' in operation_dict:
            return self._build_model(operation_dict)

        # If none of the above, try as transformer
        try:
            transformer = self.serializer._deserialize_component(operation_dict)
            return self._build_transformer_operation(transformer)
        except:
            return None

    def _build_transformation(self, config: Any) -> PipelineOperation:
        """Build transformation operation"""
        return TransformationOperation(transformer=config)

    def _build_augmentation(self, config: Dict[str, Any]) -> List[PipelineOperation]:
        """Build augmentation operations"""
        operations = []

        aug_type = 'feature_augmentation' if 'feature_augmentation' in config else 'sample_augmentation'
        aug_list = config[aug_type]

        for aug_config in aug_list:
            if isinstance(aug_config, list):
                # Handle nested augmentations (pipeline or multiple aug paths)
                for sub_aug in aug_config:
                    if sub_aug is not None:
                        transformer = self._resolve_transformer(sub_aug)
                        if transformer:
                            # use correct mode for augmentation
                            operations.append(TransformationOperation(transformer=transformer, mode=aug_type))
            elif aug_config is not None:
                transformer = self._resolve_transformer(aug_config)
                if transformer:
                    # use correct mode for augmentation
                    operations.append(TransformationOperation(transformer=transformer, mode=aug_type))

        return operations

    def _build_split(self, config: Any) -> PipelineOperation:
        """Build splitting operation"""
        from SplitOperation import SplitOperation

        # Handle sklearn splitter objects
        if hasattr(config, 'split'):
            # Extract parameters from sklearn splitter
            if hasattr(config, 'test_size') and config.test_size is not None:
                train_size = 1.0 - config.test_size
                split_ratios = {"train": train_size, "test": config.test_size}
            elif hasattr(config, 'train_size') and config.train_size is not None:
                test_size = 1.0 - config.train_size
                split_ratios = {"train": config.train_size, "test": test_size}
            else:
                split_ratios = {"train": 0.8, "test": 0.2}  # Default

            # Determine if it's stratified
            stratified = 'Stratified' in config.__class__.__name__

            # Get random state if available
            random_state = getattr(config, 'random_state', 42)

            return SplitOperation(
                split_strategy="stratified" if stratified else "random",
                split_ratios=split_ratios,
                stratified=stratified,
                random_state=random_state
            )
        else:
            # Handle other configurations
            return SplitOperation()

    def _build_cluster(self, config: Dict[str, Any]) -> PipelineOperation:
        """Build clustering operation"""
        clusterer = config['cluster']

        # Extract clustering parameters from the clusterer object
        if hasattr(clusterer, '__class__') and clusterer.__class__.__name__ == 'KMeans':
            return ClusteringOperation(
                clustering_method="kmeans",
                n_clusters=getattr(clusterer, 'n_clusters', 3),
                random_state=getattr(clusterer, 'random_state', None)
            )
        else:
            # Default clustering operation
            return ClusteringOperation(clustering_method="kmeans", n_clusters=3)

    def _build_dispatch(self, config: Dict[str, Any]) -> List[PipelineOperation]:
        """Build dispatch operations"""
        from DispatchOperation import DispatchOperation

        branches = config['dispatch']
        branch_operations = []

        for branch in branches:
            if isinstance(branch, list):
                # Handle list-style branch
                branch_ops = []
                for step in branch:
                    op = self._build_operation(step)
                    if op:
                        if isinstance(op, list):
                            branch_ops.extend(op)
                        else:
                            branch_ops.append(op)
                branch_operations.append(branch_ops)
            else:
                # Handle dict-style branch - convert dict to operations
                branch_ops = []
                op = self._build_operation(branch)
                if op:
                    if isinstance(op, list):
                        branch_ops.extend(op)
                    else:
                        branch_ops.append(op)
                branch_operations.append(branch_ops)

        return [DispatchOperation(operations=branch_operations)]

    def _build_stack(self, config: Dict[str, Any]) -> PipelineOperation:
        """Build stacking operation"""
        from StackOperation import StackOperation

        stack_config = config['stack']
        main_model = self._resolve_model(stack_config['model'])

        base_learners = []
        for base_config in stack_config.get('base_learners', []):
            base_model = self._resolve_model(base_config.get('model'))
            base_learners.append(base_model)

        return StackOperation(
            base_learners=base_learners,
            meta_learner=main_model,
        )

    def _build_model(self, config: Dict[str, Any]) -> Union[PipelineOperation, List[PipelineOperation]]:
        """Build model operation and related operations"""
        from ModelOperation import SklearnModelOperation

        model_config = config.get('model', config)
        model = self._resolve_model(model_config)

        # Build the main model operation
        operations = []

        # Add y_pipeline operation if present
        if 'y_pipeline' in config and config['y_pipeline'] is not None:
            y_pipeline_op = self._build_y_pipeline(config['y_pipeline'])
            operations.append(y_pipeline_op)

        # Add the model operation
        model_op = SklearnModelOperation(model=model)
        operations.append(model_op)

        # Add finetune operation if present
        if 'finetune_params' in config and config['finetune_params'] is not None:
            finetune_op = self._build_finetune(config['finetune_params'])
            operations.append(finetune_op)

        # Return single operation or list of operations
        if len(operations) == 1:
            return operations[0]
        else:
            return operations

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
            return TransformationOperation(transformer=transformers)
        else:
            transformer = self._resolve_transformer(config)
            return TransformationOperation(transformer=transformer)

    def _build_plot(self, config: Any) -> PipelineOperation:
        """Build plotting operation"""
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
        elif isinstance(config, type):
            # If it's a class (not an instance), instantiate it
            try:
                return config()
            except Exception as e:
                print(f"Warning: Could not instantiate transformer class {config}: {e}")
                return None
        else:
            return config

    def _resolve_model(self, config: Any) -> Any:
        """Resolve model from configuration with support for TensorFlow models that need input_shape"""

        if isinstance(config, dict):
            if 'class' in config:
                return self.serializer._deserialize_component(config)
            elif 'factory' in config:
                # Handle model factories that need input shape
                factory = config['factory']
                if config.get('requires_input_shape', False):
                    # Return a lambda that will be called with dataset during execution
                    return lambda dataset: factory(self._get_input_shape_from_dataset(dataset))
                else:
                    return factory()
            else:
                return config
        elif isinstance(config, str):
            # Try preset mappings
            if config in self.serializer.preset_mappings:
                class_path = self.serializer.preset_mappings[config]
                return self.serializer._load_class(class_path)()
            else:
                return config
        else:
            # Check if this is a function that needs input_shape
            if callable(config):
                # Check if it has a framework decorator
                if hasattr(config, 'framework'):
                    framework = config.framework
                    if framework in ['tensorflow', 'pytorch']:
                        # Check if the function signature has input_shape parameter
                        try:
                            sig = inspect.signature(config)
                            if 'input_shape' in sig.parameters:
                                # Return a function wrapper that will be called with dataset
                                def model_factory(dataset):
                                    input_shape = self._get_input_shape_from_dataset(dataset)
                                    return config(input_shape)
                                return model_factory
                            else:
                                # Call the function normally
                                return config()
                        except (ValueError, TypeError):
                            # If signature inspection fails, try calling it
                            return config()
                    else:
                        # sklearn or other framework
                        return config()
                else:
                    # No framework decorator, check signature for input_shape
                    try:
                        sig = inspect.signature(config)
                        if 'input_shape' in sig.parameters:
                            # Return a function wrapper that will be called with dataset
                            def model_factory(dataset):
                                input_shape = self._get_input_shape_from_dataset(dataset)
                                return config(input_shape)
                            return model_factory
                        else:
                            # Regular function call
                            return config()
                    except (ValueError, TypeError):
                        # If signature inspection fails, try calling it
                        return config()
            else:
                # Regular model instance
                return config

    def _get_input_shape_from_dataset(self, dataset):
        """Get input shape from dataset for TensorFlow models"""
        # Get a sample from the dataset to determine shape
        sample_view = dataset.select(partition="train")
        if len(sample_view) == 0:
            sample_view = dataset.select(partition="all")

        if len(sample_view) > 0:
            features = sample_view.get_features(concatenate=True)
            if len(features.shape) == 2:
                # Return shape for 2D input (samples, features)
                return (features.shape[1],)
            else:
                # Return shape for 3D input (samples, timesteps, features)
                return features.shape[1:]
        else:
            # Fallback shape
            return (1000,)

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
