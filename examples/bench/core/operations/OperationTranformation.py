"""
TransformerMixin Operation - Wraps sklearn transformers for pipeline execution

This operation:
1. Fits transformers on train partition data per source
2. Transforms all specified partitions
3. Updates dataset with transformed features
4. Updates processing index with transformation hash
"""
import numpy as np
import hashlib
from typing import Optional, List, Union
from sklearn.base import TransformerMixin, clone

from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext


class OperationTransformation(PipelineOperation):
    """Pipeline operation that wraps sklearn TransformerMixin objects"""

    def __init__(self,
                 transformer: TransformerMixin,
                 fit_partition: str = "train",
                 transform_partitions: Optional[List[str]] = ['train', 'test'],
                 operation_name: Optional[str] = None):
        """
        Initialize transformation operation

        Args:
            transformer: sklearn TransformerMixin object to wrap
            fit_partition: partition to fit the transformer on (default: "train")
            transform_partitions: partitions to transform (None = all partitions)
            operation_name: custom name for this operation
        """
        self.transformer = transformer
        self.fit_partition = fit_partition
        self.transform_partitions = transform_partitions
        self.operation_name = operation_name
        self.fitted_transformers = []  # One per source
        self.is_fitted = False
        self.transformation_hash = None

    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute transformation: fit on train, transform specified partitions"""
        print(f"🔄 Executing {self.get_name()}")

        # Get fitting data from specified partition
        fit_view = dataset.select(partition=self.fit_partition, **context.current_filters)
        if len(fit_view) == 0:
            raise ValueError(f"No data found in partition '{self.fit_partition}' for fitting")        # Get features per source (keep sources separate)
        X_fit = fit_view.get_features(representation="2d_separate")
        if isinstance(X_fit, np.ndarray):
            X_fit = [X_fit]

        print(f"  📊 Fitting on {len(fit_view)} samples from '{self.fit_partition}' partition")
        print(f"  🔧 {len(X_fit)} sources detected, fitting transformer per source")

        # Fit one transformer per source
        self.fitted_transformers = []
        for source_idx, X_source in enumerate(X_fit):
            source_transformer = clone(self.transformer)
            source_transformer.fit(X_source)
            self.fitted_transformers.append(source_transformer)
            print(f"    ✅ Source {source_idx}: fitted {source_transformer.__class__.__name__} on shape {X_source.shape}")

        # Generate transformation hash for processing index
        self.transformation_hash = self._compute_transformation_hash()
        self.is_fitted = True

        # Determine partitions to transform
        partitions_to_transform = self.transform_partitions or dataset.get_partition_names()

        # Transform each partition
        for partition in partitions_to_transform:
            partition_view = dataset.select(partition=partition, **context.current_filters)
            if len(partition_view) == 0:
                print(f"  ⚠️ Skipping partition '{partition}' - no data found")
                continue

            print(f"  🔄 Transforming partition '{partition}': {len(partition_view)} samples")            # Get features per source
            X_partition = partition_view.get_features(representation="2d_separate")
            if isinstance(X_partition, np.ndarray):
                X_partition = [X_partition]

            # Transform each source and update dataset
            for source_idx, (X_source, transformer) in enumerate(zip(X_partition, self.fitted_transformers)):
                X_transformed = transformer.transform(X_source)                # Update features in dataset
                dataset.update_features(partition_view.get_row_indices(), X_transformed, source_idx)
                print(f"    ✅ Source {source_idx}: {X_source.shape} → {X_transformed.shape}")

            # Update processing index for all samples in this partition
            sample_ids = partition_view.get_sample_ids()
            for sample_id in sample_ids:
                dataset.update_processing([sample_id], f"transformed_{self.transformation_hash}")

        print(f"  ✅ Transformation complete. Processing hash: {self.transformation_hash}")

    def _compute_transformation_hash(self) -> str:
        """Compute hash of transformation for processing index"""
        # Create hash based on transformer class and parameters
        try:
            params = self.transformer.get_params() if hasattr(self.transformer, 'get_params') else str(self.transformer)
            transformer_info = f"{self.transformer.__class__.__name__}_{str(params)}"
        except Exception:
            transformer_info = f"{self.transformer.__class__.__name__}_{str(self.transformer)}"
        return hashlib.md5(transformer_info.encode()).hexdigest()[:8]

    def get_name(self) -> str:
        """Get operation name"""
        if self.operation_name:
            return self.operation_name
        return f"Transform({self.transformer.__class__.__name__})"

    def can_execute(self, dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if transformation can be executed"""
        fit_view = dataset.select(partition=self.fit_partition, **context.current_filters)
        return len(fit_view) > 0
