"""
Predictions management for SpectroDataset.

This module contains Predictions class for storing and managing model predictions
with metadata about dataset, pipeline, models, and partitions.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import json

from .prediction_helpers import PredictionHelpers


class Predictions:
    """
    Storage for model predictions with metadata.

    Stores predictions in a simple dictionary structure with keys:
    - dataset: dataset name
    - pipeline: pipeline name
    - model: model name/type
    - partition: 'train_fold_X', 'val_fold_X', 'test'

    Each entry contains:
    - y_true: true values (inverse transformed)
    - y_pred: predictions (inverse transformed)
    - sample_indices: corresponding sample indices
    - fold_idx: fold index (if applicable)
    - metadata: additional metadata
    """

    def __init__(self, filepath: Optional[str] = None):
        """Initialize Predictions storage, optionally loading from file."""
        self._predictions: Dict[str, Dict[str, Any]] = {}
        if filepath and Path(filepath).exists():
            self.load_from_file(filepath)
        self.run_path = ""

    @classmethod
    def load_from_file(cls, filepath: str) -> 'Predictions':
        """
        Load predictions from JSON file.

        Args:
            filepath: Path to JSON file containing saved predictions

        Returns:
            Predictions instance with loaded data
        """
        instance = cls()
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Convert numpy arrays back from lists
            for key, pred_data in data.items():
                if 'y_true' in pred_data:
                    pred_data['y_true'] = np.array(pred_data['y_true'])
                if 'y_pred' in pred_data:
                    pred_data['y_pred'] = np.array(pred_data['y_pred'])

            instance._predictions = data
            print(f"📥 Loaded {len(data)} predictions from {filepath}")

        except Exception as e:
            print(f"⚠️ Error loading predictions from {filepath}: {e}")

        return instance

    def save_to_file(self, filepath: str) -> None:
        """
        Save predictions to JSON file.

        Args:
            filepath: Path where to save predictions JSON file
        """
        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {}
            for key, pred_data in self._predictions.items():
                serializable_pred = pred_data.copy()
                if 'y_true' in serializable_pred:
                    serializable_pred['y_true'] = serializable_pred['y_true'].tolist()
                if 'y_pred' in serializable_pred:
                    serializable_pred['y_pred'] = serializable_pred['y_pred'].tolist()
                serializable_data[key] = serializable_pred

            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)

            # print(f"💾 Saved {len(self._predictions)} predictions to {filepath}")

        except Exception as e:
            print(f"⚠️ Error saving predictions to {filepath}: {e}")

    def merge_predictions(self, other: 'Predictions') -> None:
        """
        Merge predictions from another Predictions instance.
        New predictions are added, existing ones are kept (no replacement).

        Args:
            other: Another Predictions instance to merge from
        """
        for key, pred_data in other._predictions.items():
            if key not in self._predictions:
                self._predictions[key] = pred_data.copy()
            # If key already exists, keep the existing one (no replacement)

    def add_prediction(
        self,
        dataset: str,
        pipeline: str,
        model: str,
        partition: str = None,
        y_true: np.ndarray = None,
        y_pred: np.ndarray = None,
        sample_indices: Optional[List[int]] = None,
        fold_idx: Optional[Union[int, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        # New schema parameters
        pipeline_path: str = "",
        real_model: str = None,
        custom_model_name: Optional[str] = None
    ) -> None:
        """
        Add prediction results with backward compatibility.

        This method supports both old and new schemas during transition.
        """
        # Handle backward compatibility - detect if old or new schema
        if real_model is None:
            # Old schema - convert to new schema
            real_model = model  # Use model as real_model for now
            base_model = model.split('_')[0] if '_' in model else model
        else:
            # New schema
            base_model = model

        # Handle partition format conversion
        if partition is not None:
            # Convert old partition format to new if needed
            if 'fold_' in partition:
                # Old format like "val_fold_0" -> new format "val" with fold_idx
                parts = partition.split('_fold_')
                if len(parts) == 2:
                    partition = parts[0]
                    if fold_idx is None:
                        try:
                            fold_idx = int(parts[1])
                        except ValueError:
                            pass

        # Generate a unique key based on the new schema
        key = f"{dataset}_{pipeline}_{real_model}_{partition}"
        if fold_idx is not None:
            key += f"_fold_{fold_idx}"

        # Check for duplicate predictions and warn
        if key in self._predictions:
            print(f"⚠️  WARNING: Overwriting existing prediction: {key}")

        # Ensure arrays are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Validate shapes match
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        # Create sample indices if not provided
        if sample_indices is None:
            sample_indices = list(range(len(y_true)))

        # Store prediction with new schema
        self._predictions[key] = {
            'dataset': dataset,
            'pipeline': pipeline,
            'pipeline_path': pipeline_path,
            'model': base_model,
            'real_model': real_model,
            'custom_model_name': custom_model_name,
            'partition': partition,
            'y_true': y_true.copy(),
            'y_pred': y_pred.copy(),
            'sample_indices': list(sample_indices),
            'fold_idx': fold_idx,
            'metadata': metadata or {},
            'path': self.run_path,
        }

    def get_predictions(
        self,
        dataset: Optional[str] = None,
        pipeline: Optional[str] = None,
        model: Optional[str] = None,
        real_model: Optional[str] = None,
        partition: Optional[str] = None,
        fold_idx: Optional[Union[int, str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get predictions matching the filter criteria.

        Args:
            dataset: Filter by dataset name
            pipeline: Filter by pipeline name
            model: Filter by base model class
            real_model: Filter by real model identifier
            partition: Filter by partition name
            fold_idx: Filter by fold index

        Returns:
            Dictionary of matching predictions
        """
        filtered_predictions = {}

        for key, pred_data in self._predictions.items():
            # Apply filters
            if dataset is not None and pred_data['dataset'] != dataset:
                continue
            if pipeline is not None and pred_data['pipeline'] != pipeline:
                continue
            if model is not None and pred_data['model'] != model:
                continue
            if real_model is not None and pred_data.get('real_model') != real_model:
                continue
            if partition is not None and pred_data['partition'] != partition:
                continue
            if fold_idx is not None and pred_data.get('fold_idx') != fold_idx:
                continue

            filtered_predictions[key] = pred_data

        return filtered_predictions

    def get_prediction_data(
        self,
        dataset: str,
        pipeline: str,
        model: str,
        partition: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get specific prediction data.

        Args:
            dataset: Dataset name
            pipeline: Pipeline name
            model: Model name
            partition: Partition name

        Returns:
            Prediction data dictionary or None if not found
        """
        key = f"{dataset}_{pipeline}_{model}_{partition}"
        return self._predictions.get(key)

    def combine_folds(
        self,
        dataset: str,
        pipeline: str,
        model: str,
        partition_pattern: str = "val_fold"
    ) -> Optional[Dict[str, Any]]:
        """
        Combine predictions from multiple folds.

        Args:
            dataset: Dataset name
            pipeline: Pipeline name
            model: Model name
            partition_pattern: Pattern to match (e.g., "val_fold" for validation folds)

        Returns:
            Combined prediction data or None if no matching folds found
        """
        return PredictionHelpers.combine_folds(
            self._predictions, dataset, pipeline, model, partition_pattern, self.run_path
        )

    def list_keys(self) -> List[str]:
        """List all available prediction keys."""
        return list(self._predictions.keys())

    def list_datasets(self) -> List[str]:
        """List all unique dataset names."""
        return list(set(pred['dataset'] for pred in self._predictions.values()))

    def list_pipelines(self) -> List[str]:
        """List all unique pipeline names."""
        return list(set(pred['pipeline'] for pred in self._predictions.values()))

    def list_models(self) -> List[str]:
        """List all unique model names."""
        return list(set(pred['model'] for pred in self._predictions.values()))

    def list_partitions(self) -> List[str]:
        """List all unique partition names."""
        return list(set(pred['partition'] for pred in self._predictions.values()))

    def calculate_average_predictions(
        self,
        dataset: str,
        pipeline: str,
        model: str,
        partition: str,
        store_result: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate average predictions across folds for the same partition with new schema.

        Args:
            dataset: Dataset name
            pipeline: Pipeline name
            model: Base model class name
            partition: Partition name ('train', 'val', 'test')
            store_result: Whether to store the result in predictions

        Returns:
            Average prediction data or None if no matching folds found
        """
        result, key = PredictionHelpers.calculate_average_predictions(
            self._predictions, dataset, pipeline, model, partition, self.run_path
        )
        if result and store_result and key:
            self._predictions[key] = result
        return result

    def calculate_weighted_average_predictions(
        self,
        dataset: str,
        pipeline: str,
        model: str,
        test_partition: str = "test",
        val_partition: str = "val",
        metric: str = 'rmse',
        store_result: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate weighted average predictions based on validation performance with new schema.

        Args:
            dataset: Dataset name
            pipeline: Pipeline name
            model: Base model class name
            test_partition: Partition name for test predictions to average
            val_partition: Partition name for validation predictions to use for weighting
            metric: Metric to use for weighting ('rmse', 'mae', 'r2')
            store_result: Whether to store the result in predictions

        Returns:
            Weighted average prediction data or None if insufficient data
        """
        result, key = PredictionHelpers.calculate_weighted_average_predictions(
            self._predictions, dataset, pipeline, model, test_partition, val_partition,
            metric, self.run_path
        )
        if result and store_result and key:
            self._predictions[key] = result
        return result

    def calculate_scores_for_predictions(
        self,
        dataset: Optional[str] = None,
        pipeline: Optional[str] = None,
        model: Optional[str] = None,
        partition: Optional[str] = None,
        task_type: str = "auto"
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate scores for predictions with automatic task type detection.

        Args:
            dataset: Filter by dataset name
            pipeline: Filter by pipeline name
            model: Filter by model name
            partition: Filter by partition name
            task_type: Task type ("regression", "binary_classification", "multiclass_classification", or "auto")

        Returns:
            Dict mapping prediction keys to their calculated scores
        """
        predictions = self.get_predictions(dataset, pipeline, model, partition)
        return PredictionHelpers.calculate_scores_for_predictions(predictions, task_type)

    def get_scores_ranking(
        self,
        metric: str,
        dataset: Optional[str] = None,
        pipeline: Optional[str] = None,
        model: Optional[str] = None,
        partition: Optional[str] = None,
        ascending: bool = True,
        task_type: str = "auto"
    ) -> List[Tuple[str, float]]:
        """
        Get predictions ranked by a specific metric score.

        Args:
            metric: Metric name to rank by (e.g., 'mse', 'accuracy', 'f1')
            dataset: Filter by dataset name
            pipeline: Filter by pipeline name
            model: Filter by model name
            partition: Filter by partition name
            ascending: If True, lower scores rank higher (for error metrics)
            task_type: Task type for appropriate score calculation

        Returns:
            List of (prediction_key, score) tuples sorted by score
        """
        predictions = self.get_predictions(dataset, pipeline, model, partition)
        scores_dict = PredictionHelpers.calculate_scores_for_predictions(predictions, task_type)
        return PredictionHelpers.get_scores_ranking(predictions, scores_dict, metric, ascending)

    def get_best_score(
        self,
        metric: str,
        dataset: Optional[str] = None,
        pipeline: Optional[str] = None,
        model: Optional[str] = None,
        partition: Optional[str] = None,
        task_type: str = "auto"
    ) -> Optional[Tuple[str, float]]:
        """
        Get the best (lowest or highest depending on metric) score for a metric.

        Args:
            metric: Metric name
            dataset: Filter by dataset name
            pipeline: Filter by pipeline name
            model: Filter by model name
            partition: Filter by partition name
            task_type: Task type for appropriate score calculation

        Returns:
            Tuple of (prediction_key, best_score) or None if no predictions found
        """
        predictions = self.get_predictions(dataset, pipeline, model, partition)
        scores_dict = PredictionHelpers.calculate_scores_for_predictions(predictions, task_type)
        return PredictionHelpers.get_best_score(predictions, scores_dict, metric, task_type)

    def get_all_scores_summary(
        self,
        dataset: Optional[str] = None,
        pipeline: Optional[str] = None,
        model: Optional[str] = None,
        task_type: str = "auto"
    ) -> pd.DataFrame:
        """
        Get a comprehensive summary of all scores across all partitions.

        Args:
            dataset: Filter by dataset name
            pipeline: Filter by pipeline name
            model: Filter by model name
            task_type: Task type for appropriate score calculation

        Returns:
            DataFrame with scores for all predictions
        """
        predictions = self.get_predictions(dataset, pipeline, model)
        scores_dict = PredictionHelpers.calculate_scores_for_predictions(predictions, task_type)
        return PredictionHelpers.get_all_scores_summary(predictions, scores_dict, self.run_path)

    def save_predictions_to_csv(
        self,
        filepath: str,
        dataset: Optional[str] = None,
        pipeline: Optional[str] = None,
        model: Optional[str] = None,
        partition: Optional[str] = None,
        include_scores: bool = True,
        path: Optional[str] = None,
        task_type: str = "auto"
    ) -> None:
        """
        Save predictions to CSV file.

        Args:
            filepath: Output CSV file path
            dataset: Filter by dataset name
            pipeline: Filter by pipeline name
            model: Filter by model name
            partition: Filter by partition name
            include_scores: Whether to include calculated scores
            task_type: Task type for score calculation
        """
        predictions = self.get_predictions(dataset, pipeline, model, partition)
        scores_dict = None
        if include_scores:
            scores_dict = PredictionHelpers.calculate_scores_for_predictions(predictions, task_type)
        PredictionHelpers.save_predictions_to_csv(
            predictions, filepath, scores_dict, self.run_path, task_type
        )

    def print_best_scores_summary(
        self,
        dataset: Optional[str] = None,
        pipeline: Optional[str] = None,
        task_type: str = "auto",
        show_current_pipeline_best: bool = True
    ) -> None:
        """
        Print a concise summary of best scores with both overall and current pipeline bests.

        Args:
            dataset: Filter by dataset name
            pipeline: Filter by pipeline name (for current pipeline best)
            task_type: Task type for appropriate score calculation
            show_current_pipeline_best: Whether to show current pipeline best score
        """
        predictions = self.get_predictions(dataset)
        current_pipeline_predictions = {
            k: v for k, v in predictions.items()
            if pipeline is None or v['pipeline'] == pipeline
        } if pipeline else {}
        PredictionHelpers.print_best_scores_summary(
            predictions, pipeline, task_type, show_current_pipeline_best
        )

    def clear(self) -> None:
        """Clear all predictions."""
        self._predictions.clear()

    def __len__(self) -> int:
        """Return number of stored predictions."""
        return len(self._predictions)

    def __repr__(self) -> str:
        if not self._predictions:
            return "Predictions(empty)"
        return f"Predictions({len(self._predictions)} entries)"

    def __str__(self) -> str:
        if not self._predictions:
            return "📈 Predictions: No predictions stored"

        datasets = self.list_datasets()
        pipelines = self.list_pipelines()
        models = self.list_models()

        return (f"📈 Predictions: {len(self._predictions)} entries\n"
                f"   Datasets: {datasets}\n"
                f"   Pipelines: {pipelines}\n"
                f"   Models: {models}")

    @classmethod
    def load_dataset_predictions(cls, dataset, saver):
        """Load existing predictions for a dataset and return count before loading."""
        try:
            if hasattr(saver, 'base_path'):
                from pathlib import Path
                base_path = Path(saver.base_path)
                dataset_name = dataset.name
                dataset_folder = base_path / dataset_name
                predictions_file = dataset_folder / f"{dataset_name}_predictions.json"

                if predictions_file.exists():
                    existing_predictions = cls.load_from_file(str(predictions_file))
                    return existing_predictions
        except Exception as e:
            print(f"⚠️ Could not load existing predictions: {e}")

        return Predictions()

    def display_best_scores_summary(self, dataset_name: str, predictions_before_count: int = 0):
        """Display best scores summary for the dataset after all pipelines are complete."""
        all_keys = list(self._predictions.keys())
        PredictionHelpers.display_best_scores_summary(
            self._predictions, dataset_name, predictions_before_count, all_keys
        )
