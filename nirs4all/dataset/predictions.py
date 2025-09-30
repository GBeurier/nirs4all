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
            'metadata': {'num_folds': len(matching_predictions)},
            'path': self.run_path
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
        # Find all fold predictions for this model and partition
        fold_predictions = []
        for key, pred_data in self._predictions.items():
            if (pred_data['dataset'] == dataset and
                pred_data['pipeline'] == pipeline and
                pred_data['model'] == model and
                pred_data['partition'] == partition and
                isinstance(pred_data.get('fold_idx'), int)):  # Only numeric fold indices
                fold_predictions.append(pred_data)

        if len(fold_predictions) < 2:
            return None

        # Sort by fold index
        fold_predictions.sort(key=lambda x: x.get('fold_idx', 0))

        # Calculate average predictions
        y_preds = [np.array(fp['y_pred']).flatten() for fp in fold_predictions]
        avg_y_pred = np.mean(y_preds, axis=0)

        # Use y_true from first fold (should be same across folds)
        y_true = fold_predictions[0]['y_true']
        sample_indices = fold_predictions[0]['sample_indices']
        pipeline_path = fold_predictions[0].get('pipeline_path', '')

        # Create real_model name for the average
        base_real_model = fold_predictions[0]['real_model']
        # Remove fold identifier and add avg
        base_parts = base_real_model.split('_fold')[0] if '_fold' in base_real_model else base_real_model
        avg_real_model = f"{base_parts}_avg"

        # Extract custom model name if available from any fold
        custom_model_name = None
        for fp in fold_predictions:
            if fp.get('custom_model_name'):
                custom_model_name = fp['custom_model_name']
                break

        result = {
            'dataset': dataset,
            'pipeline': pipeline,
            'pipeline_path': pipeline_path,
            'model': model,
            'real_model': avg_real_model,
            'custom_model_name': custom_model_name,
            'partition': partition,
            'y_true': y_true,
            'y_pred': avg_y_pred,
            'sample_indices': sample_indices,
            'fold_idx': 'avg',
            'metadata': {
                'num_folds': len(fold_predictions),
                'calculation_type': 'average',
                'source_folds': [fp['fold_idx'] for fp in fold_predictions],
                'source_real_models': [fp['real_model'] for fp in fold_predictions]
            },
            'path': self.run_path
        }

        if store_result:
            key = f"{dataset}_{pipeline}_{avg_real_model}_{partition}_fold_avg"
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
        # Find test and validation predictions for each fold
        test_predictions = []
        val_predictions = []

        for key, pred_data in self._predictions.items():
            if (pred_data['dataset'] == dataset and
                pred_data['pipeline'] == pipeline and
                pred_data['model'] == model and
                isinstance(pred_data.get('fold_idx'), int)):  # Only numeric fold indices

                if pred_data['partition'] == test_partition:
                    test_predictions.append(pred_data)
                elif pred_data['partition'] == val_partition:
                    val_predictions.append(pred_data)

        if len(test_predictions) < 2 or len(val_predictions) < 2:
            return None

        # Sort by fold index
        test_predictions.sort(key=lambda x: x.get('fold_idx') if x.get('fold_idx') is not None else 0)
        val_predictions.sort(key=lambda x: x.get('fold_idx') if x.get('fold_idx') is not None else 0)

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
        pipeline_path = test_predictions[0].get('pipeline_path', '')

        # Create real_model name for the weighted average
        base_real_model = test_predictions[0]['real_model']
        # Remove fold identifier and add weighted_avg
        base_parts = base_real_model.split('_fold')[0] if '_fold' in base_real_model else base_real_model
        weighted_avg_real_model = f"{base_parts}_weighted_avg"

        # Extract custom model name if available from any test fold
        custom_model_name = None
        for tp in test_predictions:
            if tp.get('custom_model_name'):
                custom_model_name = tp['custom_model_name']
                break

        result = {
            'dataset': dataset,
            'pipeline': pipeline,
            'pipeline_path': pipeline_path,
            'model': model,
            'real_model': weighted_avg_real_model,
            'custom_model_name': custom_model_name,
            'partition': test_partition,
            'y_true': y_true,
            'y_pred': weighted_avg_pred,
            'sample_indices': sample_indices,
            'fold_idx': 'weighted_avg',
            'metadata': {
                'num_folds': len(test_predictions),
                'calculation_type': 'weighted_average',
                'weighting_metric': metric,
                'weights': weights,
                'source_folds': [tp['fold_idx'] for tp in test_predictions],
                'source_real_models': [tp['real_model'] for tp in test_predictions]
            },
            'path': self.run_path
        }

        if store_result:
            key = f"{dataset}_{pipeline}_{weighted_avg_real_model}_{test_partition}_fold_weighted_avg"
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
        from nirs4all.utils.model_utils import ModelUtils, TaskType

        predictions = self.get_predictions(dataset, pipeline, model, partition)
        scores_dict = {}

        for key, pred_data in predictions.items():
            y_true = pred_data['y_true']
            y_pred = pred_data['y_pred']

            # Auto-detect task type if not specified
            if task_type == "auto":
                detected_task_type = ModelUtils.detect_task_type(y_true)
            else:
                task_type_mapping = {
                    "regression": TaskType.REGRESSION,
                    "binary_classification": TaskType.BINARY_CLASSIFICATION,
                    "multiclass_classification": TaskType.MULTICLASS_CLASSIFICATION
                }
                detected_task_type = task_type_mapping.get(task_type, TaskType.REGRESSION)

            scores = ModelUtils.calculate_scores(y_true, y_pred, detected_task_type)
            scores_dict[key] = scores

        return scores_dict

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
        scores_dict = self.calculate_scores_for_predictions(
            dataset, pipeline, model, partition, task_type
        )

        rankings = []
        for key, scores in scores_dict.items():
            if metric in scores:
                rankings.append((key, scores[metric]))

        # Sort by score
        rankings.sort(key=lambda x: x[1], reverse=not ascending)

        return rankings

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
        from nirs4all.utils.model_utils import ModelUtils, TaskType

        # Determine if higher or lower is better for this metric
        if task_type == "auto":
            # Use common knowledge about metrics
            lower_is_better_metrics = {'mse', 'mae', 'rmse', 'log_loss', 'loss'}
            ascending = metric.lower() in lower_is_better_metrics
        else:
            task_type_mapping = {
                "regression": TaskType.REGRESSION,
                "binary_classification": TaskType.BINARY_CLASSIFICATION,
                "multiclass_classification": TaskType.MULTICLASS_CLASSIFICATION
            }
            detected_task_type = task_type_mapping.get(task_type, TaskType.REGRESSION)
            best_metric, higher_is_better = ModelUtils.get_best_score_metric(detected_task_type)

            if metric == best_metric:
                ascending = not higher_is_better
            else:
                # Default heuristic
                lower_is_better_metrics = {'mse', 'mae', 'rmse', 'log_loss', 'loss'}
                ascending = metric.lower() in lower_is_better_metrics

        rankings = self.get_scores_ranking(
            metric, dataset, pipeline, model, partition, ascending, task_type
        )

        return rankings[0] if rankings else None

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
        scores_dict = self.calculate_scores_for_predictions(
            dataset, pipeline, model, task_type=task_type
        )

        rows = []
        for key, scores in scores_dict.items():
            pred_data = self._predictions[key]

            row = {
                'prediction_key': key,
                'dataset': pred_data['dataset'],
                'pipeline': pred_data['pipeline'],
                'model': pred_data['model'],
                'partition': pred_data['partition'],
                'fold_idx': pred_data.get('fold_idx'),
                'n_samples': len(pred_data['y_true']),
                'path': self.run_path
            }

            # Add all scores
            row.update(scores)
            rows.append(row)

        return pd.DataFrame(rows)

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

        if not predictions:
            print(f"No predictions found matching the criteria")
            return

        all_rows = []

        # Calculate scores if requested
        scores_dict = {}
        if include_scores:
            scores_dict = self.calculate_scores_for_predictions(
                dataset, pipeline, model, partition, task_type
            )

        for key, pred_data in predictions.items():
            y_true = pred_data['y_true'].flatten()
            y_pred = pred_data['y_pred'].flatten()
            sample_indices = pred_data['sample_indices']

            # Create rows for each sample
            for i, (true_val, pred_val, sample_idx) in enumerate(zip(y_true, y_pred, sample_indices)):
                row = {
                    'prediction_key': key,
                    'dataset': pred_data['dataset'],
                    'pipeline': pred_data['pipeline'],
                    'model': pred_data['model'],
                    'partition': pred_data['partition'],
                    'fold_idx': pred_data.get('fold_idx'),
                    'sample_index': sample_idx,
                    'y_true': true_val,
                    'y_pred': pred_val,
                    'residual': true_val - pred_val,
                    'absolute_error': abs(true_val - pred_val),
                    'path': self.run_path
                }

                # Add scores (same for all samples from same prediction)
                if include_scores and key in scores_dict:
                    row.update(scores_dict[key])

                all_rows.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(all_rows)
        df.to_csv(filepath, index=False)
        # print(f"💾 Saved {len(all_rows)} prediction records to {filepath}")

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
        from nirs4all.utils.model_utils import ModelUtils, TaskType

        # Get all predictions
        all_predictions = self.get_predictions(dataset)
        current_pipeline_predictions = self.get_predictions(dataset, pipeline) if pipeline else {}

        if not all_predictions:
            print("No predictions found")
            return

        # Determine primary metric based on task type
        if task_type == "auto":
            # Use first prediction to detect task type
            first_pred = next(iter(all_predictions.values()))
            detected_task_type = ModelUtils.detect_task_type(first_pred['y_true'])
        else:
            task_type_mapping = {
                "regression": TaskType.REGRESSION,
                "binary_classification": TaskType.BINARY_CLASSIFICATION,
                "multiclass_classification": TaskType.MULTICLASS_CLASSIFICATION
            }
            detected_task_type = task_type_mapping.get(task_type, TaskType.REGRESSION)

        best_metric, higher_is_better = ModelUtils.get_best_score_metric(detected_task_type)
        direction = "↑" if higher_is_better else "↓"

        print(f"📊 Task: {detected_task_type.value} | Best metric: {best_metric} {direction}")

        # Get rankings for the best metric
        all_rankings = self.get_scores_ranking(
            best_metric, dataset, ascending=not higher_is_better, task_type=task_type
        )

        if not all_rankings:
            print("No scores calculated")
            return

        # Find current pipeline rankings
        current_rankings = []
        if show_current_pipeline_best and pipeline and current_pipeline_predictions:
            current_rankings = self.get_scores_ranking(
                best_metric, dataset, pipeline, ascending=not higher_is_better, task_type=task_type
            )

        # Print best overall
        if all_rankings:
            best_key, best_score = all_rankings[0]
            pred_data = self._predictions[best_key]
            # Use custom name if available, otherwise model name
            model_name = pred_data.get('custom_model_name') or pred_data['model']
            partition = pred_data['partition']
            pipeline_name = pred_data['pipeline']
            print(f"🏆 Best Overall: {model_name} ({pipeline_name}/{partition}) = {best_score:.4f} {direction}")

        # Print best from current pipeline
        if current_rankings and show_current_pipeline_best:
            curr_key, curr_score = current_rankings[0]
            curr_pred_data = self._predictions[curr_key]
            # Use custom name if available, otherwise model name
            curr_model_name = curr_pred_data.get('custom_model_name') or curr_pred_data['model']
            curr_partition = curr_pred_data['partition']
            print(f"🥇 Best This Run: {curr_model_name} ({curr_partition}) = {curr_score:.4f} {direction}")
        elif show_current_pipeline_best:
            print("🥇 Best This Run: No predictions found for current pipeline")

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
        try:
            from nirs4all.utils.model_utils import ModelUtils

            # Get all predictions for analysis
            all_keys = self.list_keys()
            if len(all_keys) == 0:
                print("No predictions found")
                return

            print("-" * 140)

            # Find best from this run (new predictions)
            # If predictions_before_count > 0, we have existing predictions, so new ones are after that index
            if predictions_before_count > 0 and len(all_keys) > predictions_before_count:
                new_predictions = all_keys[predictions_before_count:]
            else:
                # Either no previous predictions or we want to analyze all predictions
                new_predictions = all_keys

            best_this_run = None
            best_this_run_score = None
            best_this_run_model = None

            # Find best overall
            best_overall = None
            best_overall_score = None
            best_overall_model = None

            # Track if higher scores are better (for direction arrow)
            higher_is_better = False            # Analyze all predictions to find best scores
            for key in all_keys:
                try:
                    # Parse key to extract components
                    key_parts = key.split('_')
                    if len(key_parts) >= 4:
                        pred_dataset_name = key_parts[0]
                        pipeline_name = '_'.join(key_parts[1:-2])
                        model_name = key_parts[-2]
                        partition_name = key_parts[-1]

                        # Only process predictions for this dataset
                        if pred_dataset_name != dataset_name:
                            continue

                        pred_data = self.get_prediction_data(
                            pred_dataset_name, pipeline_name, model_name, partition_name
                        )

                        # Only consider test partition predictions to avoid train overfitting
                        if (pred_data and 'y_true' in pred_data and 'y_pred' in pred_data and
                                pred_data.get('partition') == 'test'):
                            # Use real_model name for display (includes operation counter)
                            display_model_name = pred_data.get('real_model', model_name)

                            task_type = ModelUtils.detect_task_type(pred_data['y_true'])
                            scores = ModelUtils.calculate_scores(pred_data['y_true'], pred_data['y_pred'], task_type)
                            best_metric, metric_higher_is_better = ModelUtils.get_best_score_metric(task_type)
                            score = scores.get(best_metric)

                            # Update our global higher_is_better (should be consistent across all predictions)
                            higher_is_better = metric_higher_is_better

                            if score is not None:
                                # Check if this is best overall
                                if (best_overall_score is None or
                                        (higher_is_better and score > best_overall_score) or
                                        (not higher_is_better and score < best_overall_score)):
                                    best_overall = pipeline_name
                                    best_overall_score = score
                                    best_overall_model = display_model_name

                                # Check if this is best from this run
                                if (key in new_predictions and
                                        (best_this_run_score is None or
                                         (higher_is_better and score > best_this_run_score) or
                                         (not higher_is_better and score < best_this_run_score))):
                                    best_this_run = pipeline_name
                                    best_this_run_score = score
                                    best_this_run_model = display_model_name
                except Exception:
                    continue

            # Display results if we have meaningful data
            # Determine direction based on whether higher is better for this metric
            direction = "↑" if higher_is_better else "↓"

            if best_this_run and best_this_run_score is not None:
                # Use the real model name directly (already includes operation counter)
                display_name = best_this_run_model

                # Extract clean config name (remove hash and test parts)
                config_part = best_this_run.split('_')
                clean_config = '_'.join(config_part[1:4]) if len(config_part) >= 4 else 'unknown'

                print(f"🏆 Best from this run: {display_name} ({clean_config}) - {best_metric}={best_this_run_score:.4f}{direction}")

            if best_overall and best_overall_score is not None:
                # Use the real model name directly (already includes operation counter)
                display_name = best_overall_model

                # Extract clean config name (remove hash and test parts)
                config_part = best_overall.split('_')
                clean_config = '_'.join(config_part[1:4]) if len(config_part) >= 4 else 'unknown'

                if predictions_before_count > 0 and best_overall != best_this_run:
                    print(f"🥇 Best overall: {display_name} ({clean_config}) - {best_metric}={best_overall_score:.4f}{direction}")
                elif predictions_before_count == 0:
                    # Only show overall if it's different from this run, or if there were no previous predictions
                    print(f"🥇 Best overall: {display_name} ({clean_config}) - {best_metric}={best_overall_score:.4f}{direction}")

        except Exception as e:
            print(f"⚠️ Could not display best scores summary: {e}")
            import traceback
            traceback.print_exc()
