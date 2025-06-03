import numpy as np
import polars as pl
from typing import Any, Sequence, Union, List, Optional, Dict
from datetime import datetime

try:
    from SpectraFeatures import SpectraFeatures
    from TargetManager import TargetManager
except ImportError:
    # Fallback for direct execution
    from SpectraFeatures import SpectraFeatures
    from TargetManager import TargetManager

# Schema for prediction results
results_schema = {
    "sample": pl.Int64,
    "seed": pl.Int64,
    "branch": pl.Int64,
    "model": pl.Utf8,
    "fold": pl.Int64,
    "stack_index": pl.Int64,
    "prediction": pl.Float64,
    "datetime": pl.Datetime,
    "partition": pl.Utf8,
    "prediction_type": pl.Utf8,
}


class SpectraDataset:
    """Main dataset class with efficient operations and clear interface."""

    def __init__(self, float64: bool = True, task_type: str = "auto"):
        self.float64 = float64

        # Core data
        self.features: Optional[SpectraFeatures] = None
        self.indices = pl.DataFrame({
            "row": pl.Series([], dtype=pl.Int64),
            "sample": pl.Series([], dtype=pl.Int64),
            "origin": pl.Series([], dtype=pl.Int64),
            "partition": pl.Series([], dtype=pl.Utf8),
            "group": pl.Series([], dtype=pl.Int64),
            "branch": pl.Series([], dtype=pl.Int64),
            "processing": pl.Series([], dtype=pl.Utf8),
        })

        # Target management
        self.target_manager = TargetManager(task_type=task_type)

        # Results and folds management
        self.results = pl.DataFrame(schema=results_schema)
        self.folds = []  # List of fold definitions

        # Counters
        self._next_row = 0
        self._next_sample = 0

    def __len__(self) -> int:
        return len(self.indices)

    def add_data(self,
                 features: Union[np.ndarray, List[np.ndarray]],
                 targets: Optional[np.ndarray] = None,
                 partition: str = "train",
                 group: int = 0,
                 branch: int = 0,
                 processing: str = "raw",
                 origin: Optional[Union[int, List[int]]] = None) -> List[int]:
        """Add new data and return sample IDs."""

        if isinstance(features, np.ndarray):
            features = [features]

        n_samples = len(features[0])

        # Generate sample IDs
        sample_ids = list(range(self._next_sample, self._next_sample + n_samples))
        row_ids = list(range(self._next_row, self._next_row + n_samples))

        # Add features
        if self.features is None:
            self.features = SpectraFeatures(features)
        else:
            self.features.append(features)

        # Add indices
        if origin is None:
            origin = sample_ids
        elif isinstance(origin, int):
            origin = [origin] * n_samples

        new_indices = pl.DataFrame({
            "row": row_ids,
            "sample": sample_ids,
            "origin": origin,
            "partition": [partition] * n_samples,
            "group": [group] * n_samples,
            "branch": [branch] * n_samples,
            "processing": [processing] * n_samples,
        })

        self.indices = pl.concat([self.indices, new_indices])

        # Add targets if provided
        if targets is not None:
            self.target_manager.add_targets(sample_ids, targets)

        # Update counters
        self._next_row += n_samples
        self._next_sample += n_samples

        return sample_ids

    def select(self, **filters) -> 'DatasetView':
        """Create an efficient view of the dataset with filters applied."""
        # Import here to avoid circular import
        try:
            from DatasetView import DatasetView
        except ImportError:
            from DatasetView import DatasetView
        return DatasetView(self, filters)

    def get_features(self, row_indices: np.ndarray,
                     source_indices: Optional[Union[int, List[int]]] = None,
                     concatenate: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        """Direct feature access by row indices."""
        if self.features is None:
            return np.array([])
        return self.features.get_by_rows(row_indices, source_indices, concatenate)

    def get_targets(self, sample_ids: List[int],
                   representation: str = "auto",
                   transformer_key: Optional[str] = None) -> np.ndarray:
        """Get targets for specific samples using TargetManager."""
        return self.target_manager.get_targets(sample_ids, representation, transformer_key)

    def update_features(self, row_indices: np.ndarray,
                       new_features: Union[np.ndarray, List[np.ndarray]],
                       source_indices: Optional[Union[int, List[int]]] = None):
        """Update features in-place."""
        if self.features is not None:
            self.features.update_rows(row_indices, new_features, source_indices)

    def update_processing(self, sample_ids: List[int], processing_tag: str):
        """Update processing tags for samples."""
        mask = pl.col("sample").is_in(sample_ids)
        self.indices = self.indices.with_columns(
            pl.when(mask).then(pl.lit(processing_tag)).otherwise(pl.col("processing")).alias("processing")
        )

    # Target management methods
    def fit_transform_targets(self, sample_ids: List[int],
                            transformers: List[Any],
                            representation: str = "auto",
                            transformer_key: str = "default") -> np.ndarray:
        """Fit target transformers and return transformed targets."""
        return self.target_manager.fit_transform_targets(
            sample_ids, transformers, representation, transformer_key)

    def inverse_transform_predictions(self, predictions: np.ndarray,
                                    representation: str = "auto",
                                    transformer_key: str = "default",
                                    to_original: bool = True) -> np.ndarray:
        """Inverse transform predictions back to original format."""
        return self.target_manager.inverse_transform_predictions(
            predictions, representation, transformer_key, to_original)

    def get_target_info(self) -> Dict[str, Any]:
        """Get information about targets."""
        return self.target_manager.get_info()

    @property
    def task_type(self) -> str:
        """Get the task type."""
        return self.target_manager.task_type

    @property
    def n_classes(self) -> int:
        """Get number of classes for classification tasks."""
        return self.target_manager.n_classes_

    @property
    def classes_(self) -> Optional[np.ndarray]:
        """Get class labels for classification tasks."""
        return self.target_manager.classes_

    @property
    def is_binary(self) -> bool:
        """Check if this is a binary classification task."""
        return self.target_manager.is_binary    # Results management methods
    def add_predictions(self,
                       sample_ids: List[int],
                       predictions: np.ndarray,
                       model_name: str,
                       partition: str = "test",
                       fold: int = -1,
                       seed: int = 42,
                       branch: int = 0,
                       stack_index: int = 0,
                       prediction_type: str = "raw") -> None:
        """Add predictions to the results DataFrame."""
        # Create datetime for this prediction batch
        prediction_time = datetime.now()

        # Prepare data for results DataFrame with proper types
        results_data = {
            "sample": [int(sid) for sid in sample_ids],  # Ensure int type, polars will convert to Int64
            "seed": [int(seed)] * len(sample_ids),  # Ensure int type, polars will convert to Int64
            "branch": [int(branch)] * len(sample_ids),  # Ensure int type, polars will convert to Int32
            "model": [str(model_name)] * len(sample_ids),
            "fold": [int(fold)] * len(sample_ids),  # Ensure int type, polars will convert to Int32
            "stack_index": [int(stack_index)] * len(sample_ids),  # Ensure int type, polars will convert to Int32
            "prediction": predictions.flatten().astype(float),  # Ensure float64
            "datetime": [prediction_time] * len(sample_ids),
            "partition": [str(partition)] * len(sample_ids),
            "prediction_type": [str(prediction_type)] * len(sample_ids),
        }

        # Define schema locally to ensure proper typing
        schema = {
            "sample": pl.Int64,
            "seed": pl.Int64,
            "branch": pl.Int32,
            "model": pl.Utf8,
            "fold": pl.Int32,
            "stack_index": pl.Int32,
            "prediction": pl.Float64,
            "datetime": pl.Datetime,
            "partition": pl.Utf8,
            "prediction_type": pl.Utf8,
        }
        new_results = pl.DataFrame(results_data, schema=schema)

        self.results = pl.concat([self.results, new_results])

    def get_predictions(self,
                       sample_ids: Optional[List[int]] = None,
                       model: Optional[str] = None,
                       fold: Optional[int] = None,
                       partition: Optional[str] = None,
                       prediction_type: Optional[str] = None,
                       as_dict: bool = False) -> Union[pl.DataFrame, Dict[str, np.ndarray]]:
        """Get predictions with optional filtering."""

        filtered = self.results        # Apply filters
        if sample_ids is not None:
            filtered = filtered.filter(pl.col("sample").is_in(sample_ids))
        if model is not None:
            filtered = filtered.filter(pl.col("model") == model)
        if fold is not None:
            filtered = filtered.filter(pl.col("fold") == fold)
        if partition is not None:
            filtered = filtered.filter(pl.col("partition") == partition)
        if prediction_type is not None:
            filtered = filtered.filter(pl.col("prediction_type") == prediction_type)

        if as_dict:
            return {
                "sample_ids": filtered["sample"].to_numpy(),
                "predictions": filtered["prediction"].to_numpy(),
                "model": filtered["model"].to_numpy(),
                "fold": filtered["fold"].to_numpy(),
                "partition": filtered["partition"].to_numpy(),
                "prediction_type": filtered["prediction_type"].to_numpy(),
            }

        return filtered

    def get_fold_predictions(self,
                           model_name: str,
                           aggregation: str = "mean",
                           partition: str = "test",
                           prediction_type: str = "raw") -> Dict[str, np.ndarray]:
        """Get aggregated predictions across folds."""

        # Get all predictions for this model (returns DataFrame)
        fold_preds = self.get_predictions(
            model=model_name,
            partition=partition,
            prediction_type=prediction_type,
            as_dict=False  # Ensure we get DataFrame
        )

        if len(fold_preds) == 0:
            return {"sample_ids": np.array([]), "predictions": np.array([])}

        if aggregation == "mean":
            # Simple mean across folds
            assert isinstance(fold_preds, pl.DataFrame), "Expected DataFrame"
            grouped = fold_preds.group_by("sample").agg([
                pl.col("prediction").mean().alias("mean_prediction")
            ])
            return {
                "sample_ids": grouped["sample"].to_numpy(),
                "predictions": grouped["mean_prediction"].to_numpy()
            }

        elif aggregation == "weighted":
            # For now, simple mean (would need loss information for true weighting)
            # TODO: Implement loss-weighted aggregation when loss tracking is added
            return self.get_fold_predictions(model_name, "mean", partition, prediction_type)

        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    def get_reconstructed_train_predictions(self, model_name: str) -> Dict[str, np.ndarray]:
        """Get out-of-fold predictions for training samples (useful for stacking)."""        # Get all fold predictions for training partition (returns DataFrame)
        fold_preds = self.get_predictions(
            model=model_name,
            partition="train",
            prediction_type="raw",
            as_dict=False  # Ensure we get DataFrame
        )

        if len(fold_preds) == 0:
            return {"sample_ids": np.array([]), "predictions": np.array([])}        # For each sample, we want the prediction from the fold where it was NOT in training
        # This requires fold information to be properly stored
        # For now, return the available predictions
        assert isinstance(fold_preds, pl.DataFrame), "Expected DataFrame"
        unique_samples = fold_preds.group_by("sample").agg([
            pl.col("prediction").first().alias("oof_prediction")
        ])

        return {
            "sample_ids": unique_samples["sample"].to_numpy(),
            "predictions": unique_samples["oof_prediction"].to_numpy()
        }

    # Fold management methods
    def add_folds(self, fold_definitions: List[Dict[str, Any]]) -> None:
        """Add fold definitions to the dataset."""
        self.folds = fold_definitions

    def get_fold(self, fold_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific fold definition."""
        for fold in self.folds:
            if fold.get("fold_id") == fold_id:
                return fold
        return None

    def iter_folds(self):
        """Iterate over all folds."""
        for fold in self.folds:
            yield fold

    def clear_results(self, model: Optional[str] = None) -> None:
        """Clear results, optionally for a specific model."""
        if model is not None:
            self.results = self.results.filter(pl.col("model") != model)
        else:
            self.results = pl.DataFrame(schema=results_schema)

    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of stored results."""
        if len(self.results) == 0:
            return {"n_predictions": 0, "models": [], "partitions": [], "folds": []}

        return {
            "n_predictions": len(self.results),
            "models": self.results["model"].unique().to_list(),
            "partitions": self.results["partition"].unique().to_list(),
            "folds": sorted(self.results["fold"].unique().to_list()),
            "prediction_types": self.results["prediction_type"].unique().to_list(),
        }
