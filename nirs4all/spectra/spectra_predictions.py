"""
SpectraPredictions.py
This module defines the SpectraPredictions class, which manages prediction results and related operations.
"""

from datetime import datetime
from typing import Any, Union, List, Optional, Dict
import numpy as np
import polars as pl


class SpectraPredictions:
    """Class to manage prediction results and related operations."""

    RESULTS_SCHEMA = {
        "sample": pl.Series([], dtype=pl.Int64),       # Sample identifier
        "seed": pl.Series([], dtype=pl.Int64),         # Random seed used for this prediction
        "branch": pl.Series([], dtype=pl.Int64),       # Branch identifier
        "model": pl.Series([], dtype=pl.Utf8),         # Model name
        "fold": pl.Series([], dtype=pl.Int64),         # Fold index (for cross-validation)
        "stack_index": pl.Series([], dtype=pl.Int64),  # Stack index (if applicable)
        "prediction": pl.Series([], dtype=pl.Float64),  # Prediction values
        "datetime": pl.Series([], dtype=pl.Datetime),  # Timestamp of prediction
        "partition": pl.Series([], dtype=pl.Utf8),     # Partition this prediction belongs to
        "prediction_type": pl.Series([], dtype=pl.Utf8)  # Type of prediction (e.g., raw, transformed)
    }

    def __init__(self):
        """Initialize the SpectraPredictions with empty results DataFrame."""
        # Initialize with empty DataFrame but with proper schema and one dummy row to ensure correct types
        dummy_data = {
            "sample": [0],
            "seed": [0],
            "branch": [0],
            "model": [""],
            "fold": [0],
            "stack_index": [0],
            "prediction": [0.0],
            "datetime": [datetime.now()],
            "partition": [""],
            "prediction_type": [""],
        }
        self.results = pl.DataFrame(dummy_data)
        # Remove the dummy row to get an empty DataFrame with proper schema
        self.results = self.results.filter(pl.col("sample") == -1)  # This will create empty DataFrame

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
            "branch": [int(branch)] * len(sample_ids),  # Ensure int type, polars will convert to Int64
            "model": [str(model_name)] * len(sample_ids),
            "fold": [int(fold)] * len(sample_ids),  # Ensure int type, polars will convert to Int64
            "stack_index": [int(stack_index)] * len(sample_ids),  # Ensure int type, polars will convert to Int64
            "prediction": predictions.flatten().astype(float),  # Ensure float64
            "datetime": [prediction_time] * len(sample_ids),
            "partition": [str(partition)] * len(sample_ids),
            "prediction_type": [str(prediction_type)] * len(sample_ids),
        }

        # Define schema locally to ensure proper typing
        schema = {
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

        filtered = self.results

        # Apply filters
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
        """Get out-of-fold predictions for training samples (useful for stacking)."""

        # Get all fold predictions for training partition (returns DataFrame)
        fold_preds = self.get_predictions(
            model=model_name,
            partition="train",
            prediction_type="raw",
            as_dict=False  # Ensure we get DataFrame
        )

        if len(fold_preds) == 0:
            return {"sample_ids": np.array([]), "predictions": np.array([])}

        # For each sample, we want the prediction from the fold where it was NOT in training
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

    def clear_results(self, model: Optional[str] = None) -> None:
        """Clear results, optionally for a specific model."""
        if model is not None:
            self.results = self.results.filter(pl.col("model") != model)
        else:
            # Define schema to match the initialization schema
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

    def __len__(self) -> int:
        """Return the number of prediction records."""
        return len(self.results)

    def __repr__(self) -> str:
        """Return string representation of the predictions."""
        summary = self.get_results_summary()
        text = f"SpectraPredictions: {summary['n_predictions']} predictions\n"
        text += f"Models: {summary['models']}\n"
        text += f"Partitions: {summary['partitions']}\n"
        text += f"Folds: {summary['folds']}\n"
        text += f"Prediction types: {summary['prediction_types']}"
        return text
