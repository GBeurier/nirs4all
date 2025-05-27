from sklearn.base import BaseEstimator
import polars as pl
from .PipelineOperation import PipelineOperation
from .DatasetView import DatasetView
from typing import Optional, List, Dict, Any
from .SpectraDataset import SpectraDataset
from .PipelineContext import PipelineContext

class ModelOperation(PipelineOperation):
    """Handles model training and prediction."""

    def __init__(self, model: BaseEstimator,
                 train_on: str = "train",
                 predict_on: Optional[List[str]] = None):
        self.model = model
        self.train_on = train_on
        self.predict_on = predict_on or ["test"]

    def execute(self, dataset: SpectraDataset, context: 'PipelineContext'):
        # Get training data
        train_view = dataset.select(partition=self.train_on, **context.current_filters)
        if len(train_view) == 0:
            raise ValueError(f"No {self.train_on} data found for model training")

        X_train = train_view.get_features()
        y_train = train_view.get_targets()

        # Train model
        self.model.fit(X_train, y_train)

        # Make predictions
        predictions = {}
        for partition in self.predict_on:
            view = dataset.select(partition=partition, **context.current_filters)
            if len(view) > 0:
                X = view.get_features()
                y_pred = self.model.predict(X)
                predictions[partition] = {
                    'sample_ids': view.sample_ids,
                    'predictions': y_pred
                }

        # Store predictions in context for later use
        context.add_predictions(self.get_name(), predictions)

    def get_name(self) -> str:
        return f"Model({self.model.__class__.__name__})"
