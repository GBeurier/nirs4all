import numpy as np
from sklearn.base import BaseEstimator, clone
from typing import Optional, Union, List
from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext


class TransformationOperation(PipelineOperation):
    """
    Advanced transformation operation that works with SpectraDataset and target management.

    Based on specs.yaml TransformationOperation design.
    Supports three modes:
    - transformation: In-place replacement with processing index updates
    - sample_augmentation: Copy train set with new sample/row IDs
    - feature_augmentation: Copy dataset with same sample IDs but different processing
    """

    def __init__(self,
                 transformer: Union[BaseEstimator, List[BaseEstimator]],
                 fit_partition: str = "train",
                 transform_partitions: Optional[List[str]] = None,
                 target_aware: bool = False,
                 mode: str = "transformation"):
        """
        Initialize transformation operation.

        Args:
            transformer: sklearn-compatible transformer or list of transformers
            fit_partition: partition to fit the transformer on
            transform_partitions: partitions to transform (None = all)
            target_aware: whether transformer needs target information
            mode: operation mode - 'transformation', 'sample_augmentation', or 'feature_augmentation'
        """
        if isinstance(transformer, list):
            self.transformers = transformer
            self.transformer = transformer[0]  # For backwards compatibility
        else:
            self.transformers = [transformer]
            self.transformer = transformer

        self.fit_partition = fit_partition
        self.transform_partitions = transform_partitions
        self.target_aware = target_aware
        self.mode = mode
        self.is_fitted = False
        self.fitted_transformers = []

    def execute(self, dataset: SpectraDataset, context: 'PipelineContext'):
        """Execute the transformation operation based on mode."""

        if self.mode == "transformation":
            self._execute_transformation(dataset, context)
        elif self.mode == "sample_augmentation":
            self._execute_sample_augmentation(dataset, context)
        elif self.mode == "feature_augmentation":
            self._execute_feature_augmentation(dataset, context)
        else:
            raise ValueError(f"Unknown transformation mode: {self.mode}")

    def _execute_transformation(self, dataset: SpectraDataset, context: 'PipelineContext'):
        """Standard transformation: in-place replacement with processing index updates."""

        # Get fitting data
        fit_view = dataset.select(partition=self.fit_partition, **context.current_filters)
        if len(fit_view) == 0:
            raise ValueError(f"No {self.fit_partition} data found for fitting transformer")

        # Get features without concatenation - keep sources separate
        X_fit = fit_view.get_features(concatenate=False)
        if isinstance(X_fit, np.ndarray):
            X_fit = [X_fit]

        # Fit transformers per source
        self.fitted_transformers = []
        for source_idx, X_fit_source in enumerate(X_fit):
            source_transformer = clone(self.transformer)

            if self.target_aware and hasattr(source_transformer, 'fit'):
                y_fit = fit_view.get_targets("auto")
                source_transformer.fit(X_fit_source, y_fit)
            else:
                source_transformer.fit(X_fit_source)

            self.fitted_transformers.append(source_transformer)

        # Transform all specified partitions
        partitions_to_transform = self.transform_partitions or dataset.indices["partition"].unique().to_list()

        for partition in partitions_to_transform:
            partition_view = dataset.select(partition=partition, **context.current_filters)
            if len(partition_view) == 0:
                continue

            X_partition = partition_view.get_features(concatenate=False)
            if isinstance(X_partition, np.ndarray):
                X_partition = [X_partition]

            # Transform each source
            for source_idx, (X_source, transformer) in enumerate(zip(X_partition, self.fitted_transformers)):
                X_transformed = transformer.transform(X_source)
                # Update features in-place
                dataset.update_features(partition_view.row_indices, X_transformed, source_idx)
            # Update processing index with hash of transformation
            sample_ids = partition_view.sample_ids
            transformation_hash = self._compute_transformation_hash()
            for sample_id in sample_ids:
                dataset.update_processing([sample_id], f"transformed_{transformation_hash}")

        self.is_fitted = True

    def _execute_sample_augmentation(self, dataset: SpectraDataset, context: 'PipelineContext'):
        """Sample augmentation: copy train set with new sample/row IDs for each transformer."""

        # Get train data
        train_view = dataset.select(partition=self.fit_partition, **context.current_filters)
        if len(train_view) == 0:
            raise ValueError(f"No {self.fit_partition} data found for sample augmentation")        # Get original train features and targets
        X_train = train_view.get_features(concatenate=False)
        if isinstance(X_train, np.ndarray):
            X_train = [X_train]

        # Get targets if they exist
        original_sample_ids = train_view.sample_ids
        try:
            y_train = dataset.get_targets(original_sample_ids)
            has_targets = True
        except (AttributeError, ValueError, IndexError):
            y_train = None
            has_targets = False        # For each transformer, create augmented copies
        for transformer_idx, transformer in enumerate(self.transformers):
            # Fit transformers per source and transform all sources
            X_augmented = []
            for source_idx, X_source in enumerate(X_train):
                # Clone and fit transformer for this source
                fitted_transformer = clone(transformer)

                if self.target_aware and has_targets:
                    fitted_transformer.fit(X_source, y_train)
                else:
                    fitted_transformer.fit(X_source)

                # Transform this source
                X_transformed = fitted_transformer.transform(X_source)
                X_augmented.append(X_transformed)

            # Add augmented data with new sample/row IDs and origin pointing to original samples
            dataset.add_data(                features=X_augmented,
                targets=y_train if has_targets else None,
                partition=self.fit_partition,
                processing=f"augmented_{transformer.__class__.__name__}_{transformer_idx}",
                origin=original_sample_ids  # Origin points to original sample IDs
            )

    def _execute_feature_augmentation(self, dataset: SpectraDataset, context: 'PipelineContext'):
        """Feature augmentation: copy dataset with same sample IDs but different processing."""

        # Get all partitions to augment
        partitions_to_augment = self.transform_partitions or dataset.indices["partition"].unique().to_list()

        # Fit transformers on fit_partition
        fit_view = dataset.select(partition=self.fit_partition, **context.current_filters)
        if len(fit_view) == 0:
            raise ValueError(f"No {self.fit_partition} data found for fitting transformer")

        X_fit = fit_view.get_features(concatenate=False)
        if isinstance(X_fit, np.ndarray):
            X_fit = [X_fit]

        # Process transformers: if a transformer is a list, treat it as a sequential pipeline
        # Otherwise, treat each transformer as a separate augmentation path
        processed_transformers = []
        for transformer in self.transformers:
            if isinstance(transformer, list):
                # Sequential pipeline: treat as one augmentation path
                processed_transformers.append(transformer)
            else:
                # Single transformer: treat as one augmentation path
                processed_transformers.append([transformer])

        # Fit all transformer pipelines per source
        fitted_pipelines_per_source = []
        for transformer_pipeline in processed_transformers:
            source_pipelines = []
            for source_idx, X_fit_source in enumerate(X_fit):
                # Fit the entire pipeline sequentially for this source
                fitted_pipeline = []
                X_current = X_fit_source.copy()

                for step_transformer in transformer_pipeline:
                    fitted_transformer = clone(step_transformer)

                    # Check if transformer requires targets
                    requires_targets = (self.target_aware or
                                        hasattr(fitted_transformer, '__class__') and
                                        any(cls_name in fitted_transformer.__class__.__name__
                                            for cls_name in ['LDA', 'LinearDiscriminantAnalysis',
                                                           'QuadraticDiscriminantAnalysis']))

                    if requires_targets:
                        y_fit = fit_view.get_targets("auto")
                        fitted_transformer.fit(X_current, y_fit)
                    else:
                        fitted_transformer.fit(X_current)

                    # Transform for the next step
                    X_current = fitted_transformer.transform(X_current)
                    fitted_pipeline.append(fitted_transformer)

                source_pipelines.append(fitted_pipeline)
            fitted_pipelines_per_source.append(source_pipelines)

        print(f"Feature augmentation: {len(processed_transformers)} processing paths")
        for i, pipeline in enumerate(processed_transformers):
            pipeline_names = [t.__class__.__name__ for t in pipeline]
            if len(pipeline) > 1:
                print(f"  Path {i+1}: Sequential pipeline {' -> '.join(pipeline_names)}")
            else:
                print(f"  Path {i+1}: Single transformer {pipeline_names[0]}")

        # For each transformer pipeline, create feature-augmented copies of all specified partitions
        for pipeline_idx, source_pipelines in enumerate(fitted_pipelines_per_source):
            current_pipeline = processed_transformers[pipeline_idx]

            # Create a name for this processing path
            if len(current_pipeline) > 1:
                # Sequential pipeline
                pipeline_name = "_".join([t.__class__.__name__ for t in current_pipeline])
            else:
                # Single transformer
                pipeline_name = current_pipeline[0].__class__.__name__

            for partition in partitions_to_augment:
                # For feature augmentation, only process original data (not previously augmented data)
                partition_view = dataset.select(partition=partition, processing="raw", **context.current_filters)
                if len(partition_view) == 0:
                    continue

                # Get features and transform them through the entire pipeline
                X_partition = partition_view.get_features(concatenate=False)
                if isinstance(X_partition, np.ndarray):
                    X_partition = [X_partition]

                X_augmented = []
                for source_idx, X_source in enumerate(X_partition):
                    fitted_pipeline = source_pipelines[source_idx]
                    X_current = X_source.copy()

                    # Apply each step of the pipeline sequentially
                    for fitted_transformer in fitted_pipeline:
                        X_current = fitted_transformer.transform(X_current)

                    X_augmented.append(X_current)

                # Get original sample IDs and targets
                original_sample_ids = partition_view.sample_ids
                try:
                    y_partition = dataset.get_targets(original_sample_ids)
                    has_targets = True
                except (AttributeError, ValueError, IndexError):
                    y_partition = None
                    has_targets = False

                # For feature augmentation, add transformed data with same sample IDs but different processing
                new_sample_ids = dataset.add_data(
                    features=X_augmented,
                    targets=y_partition if has_targets else None,
                    partition=partition,
                    processing=pipeline_name,
                    origin=original_sample_ids
                )

                # Update the sample IDs in the indices to match the original ones
                import polars as pl
                # Find the rows that were just added
                mask = dataset.indices['sample'].is_in(new_sample_ids)
                # Replace the auto-generated sample IDs with the original ones
                for new_id, orig_id in zip(new_sample_ids, original_sample_ids):
                    dataset.indices = dataset.indices.with_columns([
                        pl.when(pl.col('sample') == new_id)
                        .then(orig_id)
                        .otherwise(pl.col('sample'))
                        .alias('sample')
                    ])

        self.is_fitted = True

    def _compute_transformation_hash(self, transformer=None):
        """Compute hash of transformation for processing index."""
        if transformer is None:
            transformer = self.transformer

        # Create a simple hash based on transformer class and parameters
        transformer_info = f"{transformer.__class__.__name__}_{str(transformer.get_params())}"
        return str(hash(transformer_info))[:8]

    def can_execute(self, dataset: SpectraDataset, context: 'PipelineContext') -> bool:
        """Check if transformation can be executed."""
        # Check if fit partition exists
        fit_view = dataset.select(partition=self.fit_partition, **context.current_filters)
        return len(fit_view) > 0

    def get_name(self) -> str:
        """Get operation name."""
        target_suffix = "_TA" if self.target_aware else ""
        mode_suffix = f"_{self.mode.upper()}"
        return f"Transform({self.transformer.__class__.__name__}{target_suffix}{mode_suffix})"


class SklearnTransformer(TransformationOperation):
    """Wrapper for sklearn transformers with enhanced features."""

    def __init__(self, sklearn_transformer: BaseEstimator, **kwargs):
        super().__init__(transformer=sklearn_transformer, **kwargs)

    @classmethod
    def from_config(cls, config: dict) -> 'SklearnTransformer':
        """Create from configuration dictionary."""
        transformer_class = config.get('transformer_class')
        transformer_params = config.get('transformer_params', {})

        # Dynamically import and create transformer
        if transformer_class is not None:
            module_name, class_name = transformer_class.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            transformer_cls = getattr(module, class_name)
            transformer = transformer_cls(**transformer_params)
        else:
            raise ValueError("transformer_class must be specified in config")

        # Extract TransformationOperation parameters
        operation_params = {
            'fit_partition': config.get('fit_partition', 'train'),
            'transform_partitions': config.get('transform_partitions'),
            'target_aware': config.get('target_aware', False),
            'mode': config.get('mode', 'transformation')
        }

        return cls(sklearn_transformer=transformer, **operation_params)