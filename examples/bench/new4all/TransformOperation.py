from .PipelineOperation import PipelineOperation
from sklearn.base import TransformerMixin
from typing import Optional, List
import hashlib
from .SpectraDataset import SpectraDataset
from .PipelineContext import PipelineContext

class TransformOperation(PipelineOperation):
    """Handles sklearn transformers efficiently."""

    def __init__(self, transformer: TransformerMixin,
                 fit_on: str = "train",
                 apply_to: Optional[List[str]] = None):
        self.transformer = transformer
        self.fit_on = fit_on
        self.apply_to = apply_to or ["train", "test"]

    def execute(self, dataset: SpectraDataset, context: 'PipelineContext'):
        # Get training data for fitting
        train_view = dataset.select(partition=self.fit_on, **context.current_filters)
        if len(train_view) == 0:
            raise ValueError(f"No {self.fit_on} data found for transformation")

        X_train = train_view.get_features()
        self.transformer.fit(X_train)

        # Apply to specified partitions
        for partition in self.apply_to:
            view = dataset.select(partition=partition, **context.current_filters)
            if len(view) > 0:
                X = view.get_features()
                X_transformed = self.transformer.transform(X)
                dataset.update_features(view.row_indices, X_transformed)

        # Update processing tag
        processing_hash = self._hash_transformer()
        for partition in self.apply_to:
            view = dataset.select(partition=partition, **context.current_filters)
            if len(view) > 0:
                dataset.update_processing(view.sample_ids, processing_hash)

    def _hash_transformer(self) -> str:
        """Create hash for transformer state."""
        transformer_str = f"{self.transformer.__class__.__name__}_{str(self.transformer.get_params())}"
        return hashlib.md5(transformer_str.encode()).hexdigest()[:8]

    def get_name(self) -> str:
        return f"Transform({self.transformer.__class__.__name__})"
