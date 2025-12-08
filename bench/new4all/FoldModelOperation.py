"""
Model operation that works with folds for cross-validation
"""
from sklearn.base import BaseEstimator
import polars as pl
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime

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


class FoldModelOperation(PipelineOperation):
    """Handles model training and prediction with cross-validation folds."""

    def __init__(self,
                 model: BaseEstimator,
                 predict_on: Optional[List[str]] = None,
                 target_representation: str = "auto",
                 target_transformers: Optional[List[BaseEstimator]] = None,
                 transformer_key: str = "model_targets",
                 aggregation: str = "mean",
                 save_individual_models: bool = False):
        """
        Initialize fold model operation

        Parameters:
        -----------
        model : BaseEstimator
            The model to train (will be cloned for each fold)
        predict_on : List[str], optional
            Partitions to predict on, defaults to ["test", "train"]
        target_representation : str
            Target representation to use
        target_transformers : List[BaseEstimator], optional
            Target transformers to apply
        transformer_key : str
            Key for target transformers
        aggregation : str
            How to aggregate fold predictions: "mean", "weighted", "all"
        save_individual_models : bool
            Whether to save individual fold models (memory intensive)
        """
        self.model = model
        self.predict_on = predict_on or ["test", "train"]
        self.target_representation = target_representation
        self.target_transformers = target_transformers or []
        self.transformer_key = transformer_key
        self.aggregation = aggregation
        self.save_individual_models = save_individual_models

        # Storage for fold models and predictions
        self.fold_models = {} if save_individual_models else None
        self.fold_losses = {}  # For weighted aggregation

    def execute(self, dataset: SpectraDataset, context: 'PipelineContext'):
        """Execute fold model training and prediction."""

        # Check if dataset has folds
        if not dataset.folds:
            raise ValueError("Dataset must have folds defined. Use FoldSplitOperation first.")

        model_name = self.get_name()
        fold_predictions = {}

        print(f"Training {model_name} on {len(dataset.folds)} folds...")

        # Train model on each fold
        for fold_def in dataset.folds:
            fold_id = fold_def["fold_id"]
            train_indices = fold_def["train_indices"]
            val_indices = fold_def["val_indices"]

            print(f"  Training fold {fold_id}: {len(train_indices)} train, {len(val_indices)} val")

            # Create model for this fold
            from sklearn.base import clone
            fold_model = clone(self.model)            # Get training data using sample indices
            train_sample_ids = train_indices.tolist()
            train_view = DatasetView(dataset, {"sample": train_sample_ids})

            if len(train_view) == 0:
                print(f"    Skipping fold {fold_id}: no training data")
                continue

            X_train = train_view.get_features(concatenate=True)

            # Handle target transformations
            if self.target_transformers:
                y_train = dataset.fit_transform_targets(
                    train_view.sample_ids,
                    self.target_transformers,
                    self.target_representation,
                    f"{self.transformer_key}_fold_{fold_id}"
                )
            else:
                y_train = train_view.get_targets(self.target_representation)

            # Train model on fold
            fold_model.fit(X_train, y_train)

            # Save model if requested
            if self.save_individual_models:
                self.fold_models[fold_id] = fold_model            # Predict on validation set (out-of-fold)
            val_sample_ids = val_indices.tolist()
            val_view = DatasetView(dataset, {"sample": val_sample_ids})

            if len(val_view) > 0:
                X_val = val_view.get_features(concatenate=True)
                y_val_pred = fold_model.predict(X_val)                # Store validation predictions for this fold
                dataset.add_predictions(
                    sample_ids=[int(sid) for sid in val_view.sample_ids],  # Ensure standard Python ints
                    predictions=y_val_pred,
                    model_name=model_name,
                    partition="train",  # These are out-of-fold train predictions
                    fold=fold_id,
                    prediction_type="oof"  # out-of-fold
                )

                # Calculate validation loss for weighted aggregation
                if "val" in self.predict_on or self.aggregation == "weighted":
                    y_val_true = val_view.get_targets(self.target_representation)
                    val_loss = self._calculate_loss(y_val_true, y_val_pred)
                    self.fold_losses[fold_id] = val_loss

            # Predict on other partitions
            for partition in self.predict_on:
                if partition == "train":
                    continue  # Already handled validation predictions

                partition_view = dataset.select(partition=partition, **context.current_filters)
                if len(partition_view) > 0:
                    X_pred = partition_view.get_features(concatenate=True)
                    y_pred = fold_model.predict(X_pred)                    # Store predictions for this fold and partition
                    dataset.add_predictions(
                        sample_ids=[int(sid) for sid in partition_view.sample_ids],  # Ensure standard Python ints
                        predictions=y_pred,
                        model_name=model_name,
                        partition=partition,
                        fold=fold_id,
                        prediction_type="raw"
                    )

        # Generate aggregated predictions
        self._generate_aggregated_predictions(dataset, model_name)

        # Store aggregated predictions in context for backward compatibility
        context.add_predictions(model_name, self._get_context_predictions(dataset, model_name))

    def _generate_aggregated_predictions(self, dataset: SpectraDataset, model_name: str):
        """Generate aggregated predictions across folds."""

        for partition in self.predict_on:
            # Get aggregated predictions
            if self.aggregation == "mean":
                agg_preds = dataset.get_fold_predictions(
                    model_name, aggregation="mean", partition=partition
                )
            elif self.aggregation == "weighted":
                agg_preds = dataset.get_fold_predictions(
                    model_name, aggregation="weighted", partition=partition
                )
            else:  # "all" - don't aggregate
                continue

            if len(agg_preds["sample_ids"]) > 0:                # Store aggregated predictions
                dataset.add_predictions(
                    sample_ids=[int(sid) for sid in agg_preds["sample_ids"].tolist()],  # Ensure standard Python ints
                    predictions=agg_preds["predictions"],
                    model_name=model_name,
                    partition=partition,
                    fold=-1,  # Aggregated across folds
                    prediction_type=f"aggregated_{self.aggregation}"
                )        # Generate reconstructed training predictions for stacking
        if "train" in self.predict_on:
            oof_preds = dataset.get_reconstructed_train_predictions(model_name)
            if len(oof_preds["sample_ids"]) > 0:
                dataset.add_predictions(
                    sample_ids=[int(sid) for sid in oof_preds["sample_ids"].tolist()],  # Ensure standard Python ints
                    predictions=oof_preds["predictions"],
                    model_name=model_name,
                    partition="train",
                    fold=-1,
                    prediction_type="reconstructed_oof"
                )

    def _calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate loss for weighted aggregation."""
        from sklearn.metrics import mean_squared_error, accuracy_score

        # Determine if this is regression or classification
        if hasattr(self.model, "_estimator_type"):
            if self.model._estimator_type == "classifier":
                return 1.0 - accuracy_score(y_true, y_pred)  # Error rate
            else:
                return mean_squared_error(y_true, y_pred)
        else:
            # Fallback: assume regression
            return mean_squared_error(y_true, y_pred)

    def _get_context_predictions(self, dataset: SpectraDataset, model_name: str) -> Dict[str, Any]:
        """Get predictions in context format for backward compatibility."""

        context_predictions = {}

        for partition in self.predict_on:
            # Get aggregated predictions
            agg_preds = dataset.get_predictions(
                model=model_name,
                partition=partition,
                prediction_type=f"aggregated_{self.aggregation}",
                as_dict=True
            )

            if len(agg_preds["sample_ids"]) > 0:
                context_predictions[partition] = {
                    'sample_ids': agg_preds["sample_ids"],
                    'predictions': agg_preds["predictions"],
                    'raw_predictions': agg_preds["predictions"],  # Same for aggregated
                    'fold_info': {
                        'n_folds': len(dataset.folds),
                        'aggregation': self.aggregation
                    }
                }

        return context_predictions

    def can_execute(self, dataset: SpectraDataset, context: 'PipelineContext') -> bool:
        """Check if fold model operation can be executed."""
        return len(dataset.folds) > 0

    def get_name(self) -> str:
        """Get operation name."""
        transforms = f"_T{len(self.target_transformers)}" if self.target_transformers else ""
        agg = f"_{self.aggregation}" if self.aggregation != "mean" else ""
        return f"FoldModel({self.model.__class__.__name__}_{self.target_representation}{transforms}{agg})"

    def get_fold_model(self, fold_id: int) -> Optional[BaseEstimator]:
        """Get model for specific fold (if saved)."""
        if self.fold_models is None:
            return None
        return self.fold_models.get(fold_id)

    def get_fold_losses(self) -> Dict[int, float]:
        """Get validation losses for each fold."""
        return self.fold_losses.copy()

    def get_fold_predictions_for_partition(self, dataset: SpectraDataset, partition: str) -> Dict[int, Dict[str, np.ndarray]]:
        """Get individual fold predictions for a partition."""
        model_name = self.get_name()
        fold_predictions = {}

        for fold_def in dataset.folds:
            fold_id = fold_def["fold_id"]
            fold_preds = dataset.get_predictions(
                model=model_name,
                partition=partition,
                fold=fold_id,
                as_dict=True
            )

            if len(fold_preds["sample_ids"]) > 0:
                fold_predictions[fold_id] = {
                    "sample_ids": fold_preds["sample_ids"],
                    "predictions": fold_preds["predictions"]
                }

        return fold_predictions
