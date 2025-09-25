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