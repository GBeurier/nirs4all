from sklearn.base import BaseEstimator
import polars as pl
import numpy as np
from typing import Optional, List, Dict, Any

try:
    from PipelineOperation import PipelineOperation
    from DatasetView import DatasetView
    from SpectraDataset import SpectraDataset
    from PipelineContext import PipelineContext
except ImportError:
    from PipelineOperation import PipelineOperation
    from DatasetView import DatasetView
    from SpectraDataset import SpectraDataset
    from PipelineContext import PipelineContext

class ModelOperation(PipelineOperation):
    """Handles model training and prediction with advanced target management."""

    def __init__(self, model: BaseEstimator,
                 train_on: str = "train",
                 predict_on: Optional[List[str]] = None,
                 target_representation: str = "auto",
                 target_transformers: Optional[List[BaseEstimator]] = None,
                 transformer_key: str = "model_targets"):
        self.model = model
        self.train_on = train_on
        self.predict_on = predict_on or ["test"]
        self.target_representation = target_representation
        self.target_transformers = target_transformers or []
        self.transformer_key = transformer_key

    def execute(self, dataset: SpectraDataset, context: 'PipelineContext'):        # Get training data
        train_view = dataset.select(partition=self.train_on, **context.current_filters)
        if len(train_view) == 0:
            raise ValueError(f"No {self.train_on} data found for model training")

        X_train = train_view.get_features(concatenate=True)

        # Handle target transformations
        if self.target_transformers:
            y_train = dataset.fit_transform_targets(
                train_view.sample_ids,
                self.target_transformers,
                self.target_representation,
                self.transformer_key
            )
        else:
            y_train = train_view.get_targets(self.target_representation)

        # Train model
        self.model.fit(X_train, y_train)

        # Make predictions
        predictions = {}
        for partition in self.predict_on:
            view = dataset.select(partition=partition, **context.current_filters)
            if len(view) > 0:
                X = view.get_features(concatenate=True)
                y_pred = self.model.predict(X)

                # Store both raw and inverse-transformed predictions
                predictions[partition] = {
                    'sample_ids': view.sample_ids,
                    'raw_predictions': y_pred,
                    'predictions': dataset.inverse_transform_predictions(
                        y_pred,
                        self.target_representation,
                        self.transformer_key if self.target_transformers else "default",
                        to_original=True
                    ) if self.target_transformers else y_pred,
                    'true_targets': view.get_targets("original") if partition in dataset.indices["partition"].unique() else None
                }

        # Store predictions in context for later use
        context.add_predictions(self.get_name(), predictions)

    def can_execute(self, dataset: SpectraDataset, context: 'PipelineContext') -> bool:
        """Check if model operation can be executed"""
        # Check if training partition exists and has data
        train_view = dataset.select(partition=self.train_on, **context.current_filters)
        return len(train_view) > 0

    def get_name(self) -> str:
        transforms = f"_T{len(self.target_transformers)}" if self.target_transformers else ""
        return f"Model({self.model.__class__.__name__}_{self.target_representation}{transforms})"
