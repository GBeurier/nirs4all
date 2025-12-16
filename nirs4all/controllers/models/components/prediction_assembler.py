"""
Prediction Data Assembler - Assemble prediction data for storage

This component creates structured prediction records from model outputs.
Extracted from launch_training() lines 462-494 and _create_fold_averages()
to eliminate duplicate assembly logic.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import numpy as np


@dataclass
class PartitionPrediction:
    """Single partition prediction data."""

    partition: str  # 'train', 'val', or 'test'
    indices: List[int]  # Sample indices
    y_true: np.ndarray  # True values (unscaled)
    y_pred: np.ndarray  # Predicted values (unscaled)
    score: float  # Evaluation score for this partition


@dataclass
class PredictionRecord:
    """Complete prediction record for storage."""

    metadata: dict  # Model and pipeline metadata
    partitions: List[Tuple[str, List[int], np.ndarray, np.ndarray]]  # [(partition, indices, y_true, y_pred)]


class PredictionDataAssembler:
    """Assembles prediction data for storage.

    Creates structured prediction records with all metadata required
    for storage in the prediction database.

    Example:
        >>> assembler = PredictionDataAssembler()
        >>> record = assembler.assemble(
        ...     dataset=dataset,
        ...     identifiers=identifiers,
        ...     scores={'train': 0.95, 'val': 0.90, 'test': 0.88},
        ...     predictions={'train': y_train_pred, 'val': y_val_pred, 'test': y_test_pred},
        ...     true_values={'train': y_train, 'val': y_val, 'test': y_test},
        ...     indices={'train': train_idx, 'val': val_idx, 'test': test_idx},
        ...     runner=runner,
        ...     X_shape=X_train.shape,
        ...     best_params=params
        ... )
    """

    def assemble(
        self,
        dataset: Any,
        identifiers: Any,  # ModelIdentifiers
        scores: dict,  # {'train': float, 'val': float, 'test': float}
        predictions: dict,  # {'train': ndarray, 'val': ndarray, 'test': ndarray}
        true_values: dict,  # {'train': ndarray, 'val': ndarray, 'test': ndarray}
        indices: dict,  # {'train': list, 'val': list, 'test': list}
        runner: Any,
        X_shape: Tuple[int, ...],
        best_params: Optional[dict] = None,
        context: Any = None  # ExecutionContext for branch info
    ) -> dict:
        """Assemble complete prediction record.

        Args:
            dataset: SpectroDataset instance
            identifiers: ModelIdentifiers with name, id, etc.
            scores: Dictionary of scores per partition
            predictions: Dictionary of prediction arrays per partition (unscaled)
            true_values: Dictionary of true value arrays per partition (unscaled)
            indices: Dictionary of sample indices per partition
            runner: PipelineRunner instance
            X_shape: Shape of input data (for n_features)
            best_params: Optional hyperparameters from optimization
            context: Optional ExecutionContext for branch information

        Returns:
            Dictionary ready for storage in prediction database
        """
        pipeline_uid = getattr(runner, 'pipeline_uid', None)
        pipeline_name = runner.saver.pipeline_id
        dataset_name = dataset.name

        # Get trace_id from runtime context (Phase 2)
        trace_id = runner.get_trace_id() if hasattr(runner, 'get_trace_id') else None

        # Ensure task_type is a string (convert from enum if needed)
        task_type_str = str(dataset.task_type.value) if hasattr(dataset.task_type, 'value') else str(dataset.task_type)

        # Extract branch info from context
        branch_id = None
        branch_name = None
        if context is not None:
            selector = getattr(context, 'selector', None)
            if selector is not None:
                branch_id = getattr(selector, 'branch_id', None)
                branch_name = getattr(selector, 'branch_name', None)

        prediction_data = {
            'dataset_name': dataset_name,
            'dataset_path': dataset_name,
            'config_name': pipeline_name,
            'config_path': f"{dataset_name}/{pipeline_name}",
            'pipeline_uid': pipeline_uid if pipeline_uid else "",
            'trace_id': trace_id,  # Phase 2: Link to execution trace
            'step_idx': int(identifiers.step_id) if identifiers.step_id else 0,
            'op_counter': int(identifiers.operation_counter),
            'model_name': identifiers.name,
            'model_classname': identifiers.classname,
            'model_path': f"{dataset_name}/{pipeline_name}/{identifiers.step_id}_{identifiers.name}_{identifiers.operation_counter}.pkl",
            'fold_id': identifiers.fold_idx,
            'val_score': scores.get('val', 0.0),
            'test_score': scores.get('test', 0.0),
            'train_score': scores.get('train', 0.0),
            'metric': scores.get('metric', 'unknown'),
            'task_type': task_type_str,
            'n_features': X_shape[1] if len(X_shape) > 1 else 1,
            'preprocessings': dataset.short_preprocessings_str(),
            'best_params': best_params if best_params is not None else {},
            'partitions': [],
            'partition_metadata': {},  # Store metadata per partition for aggregation
            'branch_id': branch_id,
            'branch_name': branch_name or "",
        }

        # Cache train metadata since val samples come from train partition
        train_meta_df = None
        try:
            train_meta_df = dataset.metadata({"partition": "train"})
        except (KeyError, AttributeError, ValueError, TypeError):
            pass

        # Add partition data with metadata
        for partition in ['train', 'val', 'test']:
            if partition in predictions:
                prediction_data['partitions'].append((
                    partition,
                    indices[partition],
                    true_values[partition],
                    predictions[partition]
                ))
                # Extract metadata for this partition if available
                # Note: For 'val' partition in CV, samples come from train partition
                # so we use train metadata with val indices to select the right rows
                try:
                    if partition == 'val':
                        # Validation samples are from train partition
                        partition_metadata = train_meta_df
                    elif partition == 'train':
                        partition_metadata = train_meta_df
                    else:  # test
                        partition_metadata = dataset.metadata({"partition": partition})

                    if partition_metadata is not None and len(partition_metadata) > 0:
                        # Get the sample indices for this partition's predictions
                        partition_indices = indices.get(partition, [])

                        # Convert to dictionary with column names as keys and arrays as values
                        metadata_dict = {}
                        for col in partition_metadata.columns:
                            col_data = partition_metadata[col].to_numpy()

                            # If we have indices and they're within bounds, use them to select metadata
                            if len(partition_indices) > 0 and max(partition_indices) < len(col_data):
                                # Select metadata rows corresponding to prediction indices
                                selected_data = col_data[partition_indices]
                                metadata_dict[col] = selected_data.tolist()
                            elif len(col_data) == len(predictions[partition]):
                                # Direct match - use as-is
                                metadata_dict[col] = col_data.tolist()

                        if metadata_dict:
                            prediction_data['partition_metadata'][partition] = metadata_dict
                except (KeyError, AttributeError, ValueError, TypeError, IndexError):
                    # If metadata extraction fails, continue without it
                    pass

        return prediction_data

    def assemble_fold_average(
        self,
        base_prediction: dict,
        averaged_predictions: dict,  # {'train': ndarray, 'val': ndarray, 'test': ndarray}
        averaged_scores: dict,  # {'train': float, 'val': float, 'test': float}
        is_weighted: bool = False
    ) -> dict:
        """Assemble prediction record for fold-averaged model.

        Args:
            base_prediction: Base prediction record from a single fold (for metadata)
            averaged_predictions: Dictionary of averaged prediction arrays
            averaged_scores: Dictionary of averaged scores
            is_weighted: Whether averaging was weighted by scores

        Returns:
            Dictionary ready for storage as fold-averaged prediction
        """
        avg_prediction = base_prediction.copy()

        # Update name to indicate averaging
        suffix = "_weighted_avg" if is_weighted else "_avg"
        avg_prediction['model_name'] = base_prediction['model_name'] + suffix
        avg_prediction['fold_id'] = None  # No specific fold for average

        # Ensure step_idx is an integer (copy from base may have string)
        if 'step_idx' in avg_prediction:
            step_idx = avg_prediction['step_idx']
            avg_prediction['step_idx'] = int(step_idx) if step_idx and str(step_idx).strip() else 0

        # Update scores
        avg_prediction['train_score'] = averaged_scores.get('train', 0.0)
        avg_prediction['val_score'] = averaged_scores.get('val', 0.0)
        avg_prediction['test_score'] = averaged_scores.get('test', 0.0)

        # Update partition predictions with averaged values
        updated_partitions = []
        for partition, indices, y_true, _ in base_prediction['partitions']:
            if partition in averaged_predictions:
                updated_partitions.append((
                    partition,
                    indices,
                    y_true,
                    averaged_predictions[partition]
                ))
        avg_prediction['partitions'] = updated_partitions

        return avg_prediction
