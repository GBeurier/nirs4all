"""
Prediction Storage - Simple external prediction storage logic

This module handles all prediction storage operations that were
previously scattered throughout the model controller. It provides
a clean interface for storing training predictions, creating averages,
and generating prediction CSVs.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset


class PredictionStore:
    """
    Simple prediction storage handler.

    This class externalizes all prediction storage logic from the controller,
    making it easy to understand and maintain.
    """

    def __init__(self):
        pass

    def store_training_predictions(
        self,
        trained_model: Any,
        model_id: str,
        model_uuid: str,
        y_train: np.ndarray,
        y_train_pred: np.ndarray,
        y_val: np.ndarray,
        y_val_pred: np.ndarray,
        y_test: np.ndarray,
        y_test_pred: np.ndarray,
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        fold_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Store predictions from training with all metadata.

        Returns a dictionary with prediction data that can be merged
        into the main predictions dict.
        """

        # Get dataset and pipeline names
        dataset_name = getattr(runner.saver, 'dataset_name', 'unknown') or 'unknown'
        pipeline_name = getattr(runner.saver, 'pipeline_name', 'unknown') or 'unknown'
        pipeline_path = str(runner.saver.current_path) if hasattr(runner.saver, 'current_path') else ""

        # Get model class name
        model_class_name = trained_model.__class__.__name__

        predictions = {}

        # Store training predictions
        if fold_idx is not None:
            # Fold-based training - store as train_fold_X
            train_partition = f"train_fold_{fold_idx}"
            val_partition = f"val_fold_{fold_idx}"
        else:
            # Single training
            train_partition = "train"
            val_partition = "val"

        # Store train predictions
        train_key = f"{dataset_name}_{pipeline_name}_{model_uuid}_{train_partition}"
        predictions[train_key] = {
            'dataset': dataset_name,
            'pipeline': pipeline_name,
            'pipeline_path': pipeline_path,
            'model': model_class_name,
            'real_model': model_uuid,
            'partition': train_partition,
            'y_true': self._ensure_2d(y_train),
            'y_pred': self._ensure_2d(y_train_pred),
            'fold_idx': fold_idx,
            'model_id': model_id,
            'trained_model': trained_model,  # Store for model saving
            'metadata': {
                'y_processing': context.get('y', 'numeric'),
                'model_type': model_class_name,
                'real_model': model_uuid,
                'partition': train_partition,
                'n_samples': len(y_train),
                'n_features': getattr(dataset, 'num_features', 0)
            }
        }

        # Store validation predictions
        val_key = f"{dataset_name}_{pipeline_name}_{model_uuid}_{val_partition}"
        predictions[val_key] = {
            'dataset': dataset_name,
            'pipeline': pipeline_name,
            'pipeline_path': pipeline_path,
            'model': model_class_name,
            'real_model': model_uuid,
            'partition': val_partition,
            'y_true': self._ensure_2d(y_val),
            'y_pred': self._ensure_2d(y_val_pred),
            'fold_idx': fold_idx,
            'model_id': model_id,
            'trained_model': trained_model,
            'metadata': {
                'y_processing': context.get('y', 'numeric'),
                'model_type': model_class_name,
                'real_model': model_uuid,
                'partition': val_partition,
                'n_samples': len(y_val),
                'n_features': getattr(dataset, 'num_features', 0)
            }
        }

        # Store test predictions (always test regardless of fold)
        test_partition = "test" if fold_idx is None else f"test_fold_{fold_idx}"
        test_key = f"{dataset_name}_{pipeline_name}_{model_uuid}_{test_partition}"
        predictions[test_key] = {
            'dataset': dataset_name,
            'pipeline': pipeline_name,
            'pipeline_path': pipeline_path,
            'model': model_class_name,
            'real_model': model_uuid,
            'partition': test_partition,
            'y_true': self._ensure_2d(y_test),
            'y_pred': self._ensure_2d(y_test_pred),
            'fold_idx': fold_idx,
            'model_id': model_id,
            'trained_model': trained_model,
            'metadata': {
                'y_processing': context.get('y', 'numeric'),
                'model_type': model_class_name,
                'real_model': model_uuid,
                'partition': test_partition,
                'n_samples': len(y_test),
                'n_features': getattr(dataset, 'num_features', 0)
            }
        }

        # Store in dataset predictions object
        if hasattr(dataset, '_predictions'):
            for key, pred_data in predictions.items():
                dataset._predictions.add_prediction(
                    dataset=pred_data['dataset'],
                    pipeline=pred_data['pipeline'],
                    pipeline_path=pred_data['pipeline_path'],
                    model=pred_data['model'],
                    real_model=pred_data['real_model'],
                    partition=pred_data['partition'],
                    y_true=pred_data['y_true'],
                    y_pred=pred_data['y_pred'],
                    fold_idx=pred_data['fold_idx'],
                    metadata=pred_data['metadata']
                )

        return predictions

    def create_fold_averages(
        self,
        predictions: Dict,
        model_config: Dict[str, Any],
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset',
        num_folds: int
    ) -> Dict[str, Any]:
        """
        Create average and weighted average predictions across folds.

        This implements the user's pseudo-code:
        "create avg and w-avg (based on pred and metadata of prediction)"
        """

        avg_predictions = {}

        # Get dataset and pipeline names
        dataset_name = getattr(runner.saver, 'dataset_name', 'unknown') or 'unknown'
        pipeline_name = getattr(runner.saver, 'pipeline_name', 'unknown') or 'unknown'
        pipeline_path = str(runner.saver.current_path) if hasattr(runner.saver, 'current_path') else ""

        # Group fold predictions by base model name (without fold suffix)
        model_groups = {}
        print("DEBUG: Processing predictions for averaging")

        for key, pred_data in predictions.items():
            partition = pred_data.get('partition', '')
            fold_idx = pred_data.get('fold_idx')
            model_id = pred_data.get('model_id', '')

            print(f"DEBUG: {key} - partition: {partition}, fold_idx: {fold_idx}")

            if pred_data.get('fold_idx') is not None and 'test' in pred_data.get('partition', ''):
                # Extract base model name by removing fold suffix
                model_id = pred_data.get('model_id', '')

                # Extract base name (remove _foldX suffix)
                if '_fold' in model_id:
                    base_model_id = model_id.split('_fold')[0]
                else:
                    base_model_id = model_id

                print(f"DEBUG: Adding to group {base_model_id}")

                if base_model_id not in model_groups:
                    model_groups[base_model_id] = []
                model_groups[base_model_id].append(pred_data)

        print(f"DEBUG: Found {len(model_groups)} groups: {list(model_groups.keys())}")

        # Create averages for each model group
        for base_model_id, fold_preds in model_groups.items():
            if len(fold_preds) < 2:
                continue  # Need at least 2 folds for averaging

            # Extract model info from first prediction
            first_pred = fold_preds[0]
            model_class_name = first_pred['model']

            # Collect all y_true and y_pred
            y_true_folds = [pred['y_true'] for pred in fold_preds]
            y_pred_folds = [pred['y_pred'] for pred in fold_preds]

            # Simple average
            y_true_avg = np.mean(y_true_folds, axis=0)
            y_pred_avg = np.mean(y_pred_folds, axis=0)

            # Create average prediction
            avg_model_uuid = f"{base_model_id}_avg"
            avg_key = f"{dataset_name}_{pipeline_name}_{avg_model_uuid}_test"

            avg_predictions[avg_key] = {
                'dataset': dataset_name,
                'pipeline': pipeline_name,
                'pipeline_path': pipeline_path,
                'model': model_class_name,
                'real_model': avg_model_uuid,
                'partition': 'test',
                'y_true': self._ensure_2d(y_true_avg),
                'y_pred': self._ensure_2d(y_pred_avg),
                'fold_idx': 'avg',
                'model_id': f"{base_model_id}_avg",
                'metadata': {
                    'y_processing': context.get('y', 'numeric'),
                    'model_type': model_class_name,
                    'real_model': avg_model_uuid,
                    'partition': 'test',
                    'n_samples': len(y_true_avg),
                    'n_features': getattr(dataset, 'num_features', 0),
                    'is_average': True,
                    'num_folds': len(fold_preds)
                }
            }

            # Weighted average (simple implementation - weight by inverse error)
            try:
                weights = []
                for pred in fold_preds:
                    mse = np.mean((pred['y_true'] - pred['y_pred']) ** 2)
                    weight = 1.0 / (mse + 1e-8)  # Avoid division by zero
                    weights.append(weight)

                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalize

                # Weighted average
                y_pred_weighted = np.average(y_pred_folds, axis=0, weights=weights)

                # Create weighted average prediction
                wavg_model_uuid = f"{base_model_id}_w-avg"
                wavg_key = f"{dataset_name}_{pipeline_name}_{wavg_model_uuid}_test"

                avg_predictions[wavg_key] = {
                    'dataset': dataset_name,
                    'pipeline': pipeline_name,
                    'pipeline_path': pipeline_path,
                    'model': model_class_name,
                    'real_model': wavg_model_uuid,
                    'partition': 'test',
                    'y_true': self._ensure_2d(y_true_avg),  # Same y_true as average
                    'y_pred': self._ensure_2d(y_pred_weighted),
                    'fold_idx': 'w-avg',
                    'model_id': f"{base_model_id}_w-avg",
                    'metadata': {
                        'y_processing': context.get('y', 'numeric'),
                        'model_type': model_class_name,
                        'real_model': wavg_model_uuid,
                        'partition': 'test',
                        'n_samples': len(y_true_avg),
                        'n_features': getattr(dataset, 'num_features', 0),
                        'is_weighted_average': True,
                        'num_folds': len(fold_preds),
                        'weights': weights.tolist()
                    }
                }

            except Exception as e:
                print(f"⚠️ Could not create weighted average: {e}")

        # Store averages in dataset predictions
        if hasattr(dataset, '_predictions'):
            for key, pred_data in avg_predictions.items():
                dataset._predictions.add_prediction(
                    dataset=pred_data['dataset'],
                    pipeline=pred_data['pipeline'],
                    pipeline_path=pred_data['pipeline_path'],
                    model=pred_data['model'],
                    real_model=pred_data['real_model'],
                    partition=pred_data['partition'],
                    y_true=pred_data['y_true'],
                    y_pred=pred_data['y_pred'],
                    fold_idx=pred_data['fold_idx'],
                    metadata=pred_data['metadata']
                )

        return avg_predictions

    def create_prediction_csv(
        self,
        y_true: Optional[np.ndarray],
        y_pred: np.ndarray
    ) -> str:
        """
        Create CSV string for predictions as requested in user's pseudo-code:
        "return and save as csv the predictions (train, test)"
        """

        # Ensure arrays are 2D and flatten for CSV
        if y_true is not None:
            y_true_flat = self._ensure_2d(y_true).flatten()
        else:
            y_true_flat = None

        y_pred_flat = self._ensure_2d(y_pred).flatten()

        # Create DataFrame
        if y_true_flat is not None:
            data = {
                'y_true': y_true_flat,
                'y_pred': y_pred_flat,
                'sample_index': range(len(y_pred_flat))
            }
        else:
            data = {
                'y_pred': y_pred_flat,
                'sample_index': range(len(y_pred_flat))
            }

        df = pd.DataFrame(data)

        # Return CSV string
        return df.to_csv(index=False)

    def _ensure_2d(self, arr: np.ndarray) -> np.ndarray:
        """Ensure array is 2D for consistent storage."""
        if arr is None:
            return arr

        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        elif arr.ndim == 0:
            return arr.reshape(1, 1)
        else:
            return arr

    def _transform_predictions_to_original_space(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset: 'SpectroDataset',
        current_y_processing: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform predictions back to original space if needed.

        This is a simplified version - the real implementation would
        handle inverse transforms properly.
        """

        # For now, just return as-is
        # In a real implementation, this would:
        # 1. Check if y was transformed (scaled, normalized, etc.)
        # 2. Apply inverse transform to get back to original space
        # 3. Handle different transformation types

        return y_true, y_pred