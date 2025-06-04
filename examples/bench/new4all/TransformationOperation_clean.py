"""
TransformationOperation - Simple wrapper around sklearn transformers

Clean design:
- Simple wrapper around actual transformer
- Handles fit/transform patterns
- Supports different modes (transformation, augmentation)
- Uses PipelineOperation interface
"""
import numpy as np
from sklearn.base import clone
from typing import Optional, List

from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext


class TransformationOperation(PipelineOperation):
    """Simple wrapper around sklearn transformers"""

    def __init__(self,
                 transformer,
                 fit_partition: str = "train",
                 transform_partitions: Optional[List[str]] = None,
                 mode: str = "transformation"):
        """
        Initialize transformation operation

        Args:
            transformer: sklearn-compatible transformer
            fit_partition: partition to fit on
            transform_partitions: partitions to transform (None = all)
            mode: 'transformation', 'sample_augmentation', or 'feature_augmentation'
        """
        self.transformer = transformer
        self.fit_partition = fit_partition
        self.transform_partitions = transform_partitions
        self.mode = mode
        self.is_fitted = False

    def execute(self, dataset: SpectraDataset, context: PipelineContext):
        """Execute transformation - simple fit and transform"""

        # Get fitting data
        fit_view = dataset.select(partition=self.fit_partition, **context.current_filters)
        if len(fit_view) == 0:
            raise ValueError(f"No data found for fitting partition: {self.fit_partition}")

        X_fit = fit_view.get_features()

        # Fit transformer
        if not self.is_fitted:
            print(f"  🔧 Fitting {self.transformer.__class__.__name__}")
            self.transformer.fit(X_fit)
            self.is_fitted = True

        # Transform specified partitions
        partitions = self.transform_partitions or dataset.get_partition_names()

        for partition in partitions:
            partition_view = dataset.select(partition=partition, **context.current_filters)
            if len(partition_view) == 0:
                continue

            print(f"  🔄 Transforming partition: {partition}")
            X = partition_view.get_features()
            X_transformed = self.transformer.transform(X)

            if self.mode == "transformation":
                # In-place replacement
                partition_view.set_features(X_transformed)
            elif self.mode == "sample_augmentation":
                # Add as new samples with different IDs
                self._add_augmented_samples(dataset, partition_view, X_transformed)
            elif self.mode == "feature_augmentation":
                # Add as new features with same sample IDs
                self._add_augmented_features(dataset, partition_view, X_transformed)

    def _add_augmented_samples(self, dataset: SpectraDataset, view, X_transformed):
        """Add transformed data as new samples"""
        # Create new sample IDs
        original_ids = view.sample_ids
        new_ids = [f"{sid}_aug_{self.transformer.__class__.__name__}" for sid in original_ids]

        # Add to dataset
        dataset.add_samples(
            sample_ids=new_ids,
            features=X_transformed,
            targets=view.get_targets() if view.has_targets() else None,
            partition=view.partition[0] if hasattr(view, 'partition') else 'train'
        )

    def _add_augmented_features(self, dataset: SpectraDataset, view, X_transformed):
        """Add transformed data as new features"""
        # Add features with processing tag
        processing_tag = f"proc_{self.transformer.__class__.__name__}"
        dataset.add_features(
            sample_ids=view.sample_ids,
            features=X_transformed,
            processing_index=processing_tag
        )

    def get_name(self) -> str:
        return f"Transform({self.transformer.__class__.__name__})"
