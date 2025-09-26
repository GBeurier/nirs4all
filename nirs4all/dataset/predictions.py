"""
Predictions management for SpectroDataset.

This module contains Predictions class for storing and managing model predictions
with metadata about dataset, pipeline, models, and partitions.
"""

from typing import Dict, Any, List, Optional
import numpy as np


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

    def __init__(self):
        """Initialize empty Predictions storage."""
        self._predictions: Dict[str, Dict[str, Any]] = {}

    def add_prediction(
        self,
        dataset: str,
        pipeline: str,
        model: str,
        partition: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_indices: Optional[List[int]] = None,
        fold_idx: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add prediction results.

        Args:
            dataset: Dataset name
            pipeline: Pipeline name
            model: Model name/type
            partition: Partition type ('train_fold_X', 'val_fold_X', 'test')
            y_true: True values (should be inverse transformed)
            y_pred: Predicted values (should be inverse transformed)
            sample_indices: Corresponding sample indices
            fold_idx: Fold index if applicable
            metadata: Additional metadata dictionary
        """
        key = f"{dataset}_{pipeline}_{model}_{partition}"

        # Ensure arrays are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Validate shapes match
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        # Create sample indices if not provided
        if sample_indices is None:
            sample_indices = list(range(len(y_true)))

        # Store prediction
        self._predictions[key] = {
            'dataset': dataset,
            'pipeline': pipeline,
            'model': model,
            'partition': partition,
            'y_true': y_true.copy(),
            'y_pred': y_pred.copy(),
            'sample_indices': list(sample_indices),
            'fold_idx': fold_idx,
            'metadata': metadata or {}
        }

    def get_predictions(
        self,
        dataset: Optional[str] = None,
        pipeline: Optional[str] = None,
        model: Optional[str] = None,
        partition: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get predictions matching the filter criteria.

        Args:
            dataset: Filter by dataset name
            pipeline: Filter by pipeline name
            model: Filter by model name
            partition: Filter by partition name

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
            if partition is not None and pred_data['partition'] != partition:
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
        matching_predictions = []

        for key, pred_data in self._predictions.items():
            if (pred_data['dataset'] == dataset and
                    pred_data['pipeline'] == pipeline and
                    pred_data['model'] == model and
                    partition_pattern in pred_data['partition']):
                matching_predictions.append(pred_data)

        if not matching_predictions:
            return None

        # Combine all predictions
        all_y_true = []
        all_y_pred = []
        all_sample_indices = []
        all_fold_indices = []

        for pred_data in matching_predictions:
            all_y_true.append(pred_data['y_true'])
            all_y_pred.append(pred_data['y_pred'])
            all_sample_indices.extend(pred_data['sample_indices'])
            fold_idx = pred_data.get('fold_idx', 0)
            all_fold_indices.extend([fold_idx] * len(pred_data['y_true']))

        return {
            'dataset': dataset,
            'pipeline': pipeline,
            'model': model,
            'partition': f"combined_{partition_pattern}",
            'y_true': np.concatenate(all_y_true),
            'y_pred': np.concatenate(all_y_pred),
            'sample_indices': all_sample_indices,
            'fold_indices': all_fold_indices,
            'metadata': {'num_folds': len(matching_predictions)}
        }

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
        partition_pattern: str = "test_fold",
        store_result: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate average predictions across folds for the same samples.

        Args:
            dataset: Dataset name
            pipeline: Pipeline name
            model: Model name
            partition_pattern: Pattern to match (e.g., "test_fold" for test predictions across folds)
            store_result: Whether to store the result in predictions

        Returns:
            Average prediction data or None if no matching folds found
        """
        # Find all matching fold predictions
        fold_predictions = []
        for key, pred_data in self._predictions.items():
            # Check if model name matches exactly or as a base name
            model_match = (pred_data['model'] == model or
                          pred_data['model'].startswith(f"{model}_"))

            if (pred_data['dataset'] == dataset
                and pred_data['pipeline'] == pipeline
                and model_match
                and partition_pattern in pred_data['partition']):
                fold_predictions.append(pred_data)

        if len(fold_predictions) < 2:
            return None

        # Sort by fold index if available
        fold_predictions.sort(key=lambda x: x.get('fold_idx', 0))

        # Calculate average predictions
        # Assume all folds predict on the same samples in the same order
        y_preds = [np.array(fp['y_pred']).flatten() for fp in fold_predictions]
        avg_y_pred = np.mean(y_preds, axis=0)

        # Use y_true from first fold (should be same across folds)
        y_true = fold_predictions[0]['y_true']
        sample_indices = fold_predictions[0]['sample_indices']

        # Determine the representative model name
        representative_model = model if not model.endswith('_') else fold_predictions[0]['model'].split('_')[0]

        result = {
            'dataset': dataset,
            'pipeline': pipeline,
            'model': representative_model,
            'partition': f"avg_{partition_pattern}",
            'y_true': y_true,
            'y_pred': avg_y_pred,
            'sample_indices': sample_indices,
            'fold_idx': None,
            'metadata': {
                'num_folds': len(fold_predictions),
                'calculation_type': 'average',
                'source_partitions': [fp['partition'] for fp in fold_predictions],
                'source_models': [fp['model'] for fp in fold_predictions]
            }
        }

        if store_result:
            key = f"{dataset}_{pipeline}_{representative_model}_avg_{partition_pattern}"
            self._predictions[key] = result

        return result

    def calculate_weighted_average_predictions(
        self,
        dataset: str,
        pipeline: str,
        model: str,
        test_partition_pattern: str = "test_fold",
        val_partition_pattern: str = "val_fold",
        metric: str = 'rmse',
        store_result: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate weighted average predictions based on validation performance.

        Args:
            dataset: Dataset name
            pipeline: Pipeline name
            model: Model name
            test_partition_pattern: Pattern for test predictions to average
            val_partition_pattern: Pattern for validation predictions to use for weighting
            metric: Metric to use for weighting ('rmse', 'mae', 'r2')
            store_result: Whether to store the result in predictions

        Returns:
            Weighted average prediction data or None if insufficient data
        """
        # Find test and validation predictions for each fold
        test_predictions = []
        val_predictions = []

        for key, pred_data in self._predictions.items():
            if (pred_data['dataset'] == dataset
                and pred_data['pipeline'] == pipeline
                and pred_data['model'] == model):

                if test_partition_pattern in pred_data['partition']:
                    test_predictions.append(pred_data)
                elif val_partition_pattern in pred_data['partition']:
                    val_predictions.append(pred_data)

        if len(test_predictions) < 2 or len(val_predictions) < 2:
            return None

        # Sort by fold index
        test_predictions.sort(key=lambda x: x.get('fold_idx', 0))
        val_predictions.sort(key=lambda x: x.get('fold_idx', 0))

        # Calculate validation scores for each fold
        weights = []
        for val_pred in val_predictions:
            y_true = np.array(val_pred['y_true']).flatten()
            y_pred = np.array(val_pred['y_pred']).flatten()

            # Remove NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            if len(y_true) == 0:
                weights.append(0.0)
                continue

            # Calculate metric score
            if metric == 'rmse':
                score = np.sqrt(np.mean((y_true - y_pred) ** 2))
                # For RMSE, lower is better, so use 1/score as weight
                weight = 1.0 / (score + 1e-8)  # Add small epsilon to avoid division by zero
            elif metric == 'mae':
                score = np.mean(np.abs(y_true - y_pred))
                weight = 1.0 / (score + 1e-8)
            elif metric == 'r2':
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                # For R², higher is better, so use score directly
                weight = max(0, score)  # Ensure non-negative weight
            else:
                weight = 1.0  # Equal weighting if metric not recognized

            weights.append(weight)

        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight <= 0:
            # Fallback to equal weighting
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / total_weight for w in weights]

        # Calculate weighted average predictions
        y_preds = [np.array(tp['y_pred']).flatten() for tp in test_predictions]
        weighted_avg_pred = np.average(y_preds, axis=0, weights=weights)

        # Use y_true from first test fold
        y_true = test_predictions[0]['y_true']
        sample_indices = test_predictions[0]['sample_indices']

        result = {
            'dataset': dataset,
            'pipeline': pipeline,
            'model': model,
            'partition': f"weighted_avg_{test_partition_pattern}",
            'y_true': y_true,
            'y_pred': weighted_avg_pred,
            'sample_indices': sample_indices,
            'fold_idx': None,
            'metadata': {
                'num_folds': len(test_predictions),
                'calculation_type': 'weighted_average',
                'weighting_metric': metric,
                'weights': weights,
                'source_partitions': [tp['partition'] for tp in test_predictions]
            }
        }

        if store_result:
            key = f"{dataset}_{pipeline}_{model}_weighted_avg_{test_partition_pattern}"
            self._predictions[key] = result

        return result

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