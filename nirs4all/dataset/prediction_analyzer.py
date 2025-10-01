"""
PredictionAnalyzer - Advanced analysis and visualization of pipeline prediction results

This module provides comprehensive analysis capabilities for prediction data stored in dataset._predictions,
allowing users to filter, aggregate, and visualize model performance across different configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import re
import warnings

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning

from nirs4all.dataset.predictions import Predictions


class PredictionAnalyzer:
    """
    Advanced analyzer for prediction results with filtering, aggregation, and visualization capabilities.

    Features:
    - Smart filtering by partition type (train/test/val), canonical model, dataset
    - Automatic fold averaging for aggregated predictions
    - Multiple visualization types with metric-based ranking
    - Comprehensive performance analysis across configurations
    """

    def __init__(self, predictions_obj: Predictions, dataset_name_override: str = None):
        """
        Initialize with a predictions object from dataset._predictions.

        Args:
            predictions_obj: The predictions object containing prediction data
            dataset_name_override: Override for dataset name display
        """
        self.predictions = predictions_obj
        self.dataset_name_override = dataset_name_override
        self.raw_data = self._extract_prediction_data()
        self.processed_data = self._process_and_organize_data()

    def _detect_task_type(self, y_true: np.ndarray) -> str:
        """Detect if the task is classification or regression based on y_true values."""
        y_true = np.array(y_true).flatten()
        y_true = y_true[~np.isnan(y_true)]  # Remove NaN values

        if len(y_true) == 0:
            return 'regression'  # Default

        unique_values = np.unique(y_true)
        n_unique = len(unique_values)

        # If few unique values and they are integers (or close to integers), it's classification
        if n_unique <= 20:  # Arbitrary threshold
            # Check if all values are integers or very close to integers
            is_integer_like = np.allclose(y_true, np.round(y_true))
            if is_integer_like:
                return 'classification'

        return 'regression'

    def _extract_prediction_data(self) -> List[Dict]:
        """Extract prediction data from the predictions object."""
        try:
            pred_data = self.predictions.get_predictions()
            if isinstance(pred_data, dict):
                return list(pred_data.values())
            elif isinstance(pred_data, list):
                return pred_data
        except AttributeError:
            pass
        return []

    def _process_and_organize_data(self) -> Dict[str, Any]:
        """Process and organize prediction data for efficient querying."""
        processed = {
            'by_partition_type': defaultdict(list),
            'by_canonical_model': defaultdict(list),
            'by_dataset': defaultdict(list),
            'fold_groups': defaultdict(list),
            'canonical_models': set(),
            'datasets': set(),
            'partition_types': set()
        }

        for pred_record in self.raw_data:
            dataset = self.dataset_name_override or pred_record.get('dataset', 'unknown')
            model = pred_record.get('model', 'unknown')
            real_model = pred_record.get('real_model', model)  # New schema field
            partition = pred_record.get('partition', 'unknown')
            fold_idx = pred_record.get('fold_idx')

            # Extract canonical model name using both model and real_model
            canonical_model = self._extract_canonical_model_name(model, real_model)

            # Determine partition type using new schema
            partition_type = self._classify_partition_type(partition, fold_idx)

            # Calculate metrics
            y_true = np.array(pred_record.get('y_true', []))
            y_pred = np.array(pred_record.get('y_pred', []))
            metrics = self._calculate_metrics(y_true, y_pred)

            # Extract pipeline information for enhanced display
            pipeline_info = self._extract_pipeline_info(pred_record)
            custom_model_name = pred_record.get('custom_model_name', None)
            enhanced_model_name = self._create_enhanced_model_name(canonical_model, real_model, pipeline_info, custom_model_name)

            # Create processed record
            processed_record = {
                'dataset': dataset,
                'model': model,
                'real_model': real_model,
                'canonical_model': canonical_model,
                'enhanced_model_name': enhanced_model_name,
                'pipeline_info': pipeline_info,
                'partition': partition,
                'partition_type': partition_type,
                'y_true': y_true,
                'y_pred': y_pred,
                'metrics': metrics,
                'sample_count': len(y_true),
                'fold_idx': fold_idx,
                'metadata': pred_record.get('metadata', {}),
                'path': pred_record.get('path', '')
            }

            # Organize by different dimensions
            processed['by_partition_type'][partition_type].append(processed_record)
            processed['by_canonical_model'][canonical_model].append(processed_record)
            processed['by_dataset'][dataset].append(processed_record)

            # Group folds for averaging
            if partition_type in ['train_fold', 'val_fold', 'test_fold']:
                fold_key = f"{dataset}_{canonical_model}_{partition_type.replace('_fold', '')}"
                processed['fold_groups'][fold_key].append(processed_record)

            # Track unique values
            processed['canonical_models'].add(canonical_model)
            processed['datasets'].add(dataset)
            processed['partition_types'].add(partition_type)

        return processed

    def _extract_canonical_model_name(self, model_name: str, real_model: str = None) -> str:
        """Extract canonical model name by removing parameters and suffixes.

        With new schema, use the base model field preferentially.
        """
        # If we have a real_model, extract the base model from it
        if real_model:
            # New naming scheme: real_model format examples:
            # - "PLS-10_cp_1_fold0" (individual fold)
            # - "PLS-10_cp_step1_avg" (average model)
            # - "PLS-10_cp_step1_w_avg" (weighted average model)
            # - "PLSRegression_2" (instance name)

            # Remove fold and aggregation suffixes
            canonical = real_model

            # Remove fold suffix
            if '_fold' in canonical:
                canonical = canonical.split('_fold')[0]

            # Remove aggregation suffixes
            if canonical.endswith('_avg'):
                canonical = canonical[:-4]  # Remove '_avg'
            elif canonical.endswith('_w_avg'):
                canonical = canonical[:-6]  # Remove '_w_avg'

            # Remove step information
            if '_step' in canonical:
                canonical = canonical.split('_step')[0]

            # Remove the operation counter (last number) if present
            parts = canonical.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                canonical = '_'.join(parts[:-1])

            return canonical

        # Fallback to the original model field
        canonical = re.sub(r'\([^)]*\)', '', model_name)  # Remove parentheses with params
        canonical = re.sub(r'_\d+$', '', canonical)  # Remove trailing numbers
        canonical = canonical.strip()
        return canonical if canonical else model_name

    def _classify_partition_type(self, partition: str, fold_idx: Any = None) -> str:
        """Classify partition into high-level types.

        With new schema, partition is clean ('test', 'val', 'train') and fold_idx indicates type.
        All test predictions are on the same test set - just from different fold models.
        """
        # Handle aggregated predictions based on fold_idx
        if fold_idx == 'avg':
            return f'averaged_{partition}'
        elif fold_idx == 'w_avg':
            return f'weighted_averaged_{partition}'

        # Handle old format for backward compatibility
        if 'global_train' in partition:
            return 'train'  # No distinction for train
        elif 'train_fold' in partition:
            return 'train'
        elif 'val_fold' in partition:
            return 'val'
        elif 'test_fold' in partition and 'avg' not in partition:
            return 'test'  # All test predictions are on the same test set
        elif 'avg_test_fold' in partition:
            return 'averaged_test'
        elif 'weighted_avg_test_fold' in partition:
            return 'weighted_averaged_test'
        elif 'test' in partition and 'fold' not in partition and 'avg' not in partition:
            return 'test'  # All test predictions are on the same test set

        # New schema format - all test predictions are on the same test set
        elif partition == 'test' and isinstance(fold_idx, int):
            return 'test'  # Individual fold prediction on test set
        elif partition == 'val' and isinstance(fold_idx, int):
            return 'val'  # Individual fold prediction on val set
        elif partition == 'train' and isinstance(fold_idx, int):
            return 'train'  # Individual fold prediction on train set
        elif partition == 'test' and fold_idx is None:
            return 'test'  # Still the same test set
        elif partition == 'train' and fold_idx is None:
            return 'train'

        return partition

    def _extract_pipeline_info(self, pred_record: Dict) -> Dict[str, str]:
        """Extract pipeline configuration information from prediction record."""
        pipeline_info = {}

        # Extract pipeline name from the key or metadata
        # Keys follow format: dataset_pipeline_model_partition_fold_X
        if 'pipeline' in pred_record:
            pipeline_info['pipeline_name'] = pred_record['pipeline']
        else:
            # Try to extract from dataset/path info
            dataset_name = pred_record.get('dataset', '')
            real_model = pred_record.get('real_model', '')

            # Extract pipeline from real_model if it contains step info
            if '_step' in real_model:
                step_part = real_model.split('_step')[1]
                if step_part:
                    step_num = ''.join(filter(str.isdigit, step_part.split('_')[0]))
                    pipeline_info['step'] = step_num

        return pipeline_info

    def _create_enhanced_model_name(self, canonical_model: str, real_model: str, pipeline_info: Dict, custom_model_name: str = None) -> str:
        """Create an enhanced model name with pipeline details, prioritizing custom names."""

        # Priority 1: Custom Name (if provided)
        if custom_model_name:
            enhanced_name = custom_model_name

            # Extract and add operation counter from real_model
            if real_model and '_' in real_model:
                parts = real_model.split('_')
                # The last part is usually the operation counter (e.g., "10" in "PLS-20_cp_10")
                if len(parts) > 1 and parts[-1].isdigit():
                    counter = parts[-1]
                    # Check if the custom name already has the counter
                    if not enhanced_name.endswith(f'_{counter}'):
                        enhanced_name = f"{enhanced_name}_{counter}"

            # Add step and fold information to custom name
            if real_model and '_step' in real_model:
                parts = real_model.split('_')
                step_part = None
                for part in parts:
                    if part.startswith('step'):
                        step_part = part
                        break

                if step_part:
                    enhanced_name += f" (step {step_part.replace('step', '')})"

            # Add fold information if it's not an aggregation
            if real_model and '_fold' in real_model and not any(x in real_model for x in ['avg', 'weighted']):
                fold_part = real_model.split('_fold')[-1]
                if fold_part.isdigit():
                    enhanced_name += f" fold{fold_part}"
            elif real_model and 'avg' in real_model:
                if 'weighted' in real_model:
                    enhanced_name += " [w-avg]"
                else:
                    enhanced_name += " [avg]"

            return enhanced_name

        # Priority 2: Enhanced Name based on canonical model
        enhanced_name = canonical_model

        # Extract and add operation counter from real_model
        if real_model and '_' in real_model:
            parts = real_model.split('_')
            # The last part is usually the operation counter (e.g., "10" in "PLS-20_cp_10")
            if len(parts) > 1 and parts[-1].isdigit():
                counter = parts[-1]
                # Add counter to canonical model name
                enhanced_name = f"{canonical_model}_{counter}"

        if real_model and real_model != canonical_model:
            # Extract meaningful parts from real_model
            if '_step' in real_model:
                parts = real_model.split('_')
                step_part = None
                for part in parts:
                    if part.startswith('step'):
                        step_part = part
                        break

                if step_part:
                    enhanced_name = f"{enhanced_name} (step {step_part.replace('step', '')})"

            # Add fold information if it's not an aggregation
            if '_fold' in real_model and not any(x in real_model for x in ['avg', 'weighted']):
                fold_part = real_model.split('_fold')[-1]
                if fold_part.isdigit():
                    enhanced_name += f" fold{fold_part}"
            elif 'avg' in real_model:
                if 'weighted' in real_model:
                    enhanced_name += " [WEIGHTED AVG]"
                else:
                    enhanced_name += " [AVG]"

        return enhanced_name

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate common metrics based on task type (regression or classification)."""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
                   'accuracy': np.nan, 'f1': np.nan, 'precision': np.nan, 'recall': np.nan}

        task_type = self._detect_task_type(y_true)

        if task_type == 'regression':
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))

            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
                   'accuracy': np.nan, 'f1': np.nan, 'precision': np.nan, 'recall': np.nan}
        else:  # classification
            # For classification, convert predictions to class labels if they are probabilities
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                # Multi-class with probabilities
                y_pred_labels = np.argmax(y_pred, axis=1)
            else:
                # Binary or already class labels
                y_pred_labels = np.round(y_pred).astype(int)

            y_true_labels = y_true.astype(int)

            accuracy = accuracy_score(y_true_labels, y_pred_labels)
            f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

            # Suppress sklearn warnings for precision and recall
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                precision = precision_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
                recall = recall_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)

            return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
                   'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

    def get_aggregated_prediction(self, dataset: str, canonical_model: str,
                                  partition_type: str) -> Optional[Dict[str, Any]]:
        """
        Get or create aggregated prediction from folds.

        Args:
            dataset: Dataset name
            canonical_model: Canonical model name
            partition_type: 'train', 'val', or 'test'

        Returns:
            Aggregated prediction record or None if no folds found
        """
        fold_key = f"{dataset}_{canonical_model}_{partition_type}"
        folds = self.processed_data['fold_groups'].get(fold_key, [])

        if not folds:
            return None

        # Aggregate predictions
        all_y_true = []
        all_y_pred = []
        all_metrics = []

        for fold in folds:
            all_y_true.extend(fold['y_true'])
            all_y_pred.extend(fold['y_pred'])
            all_metrics.append(fold['metrics'])

        # Calculate aggregated metrics
        agg_y_true = np.array(all_y_true)
        agg_y_pred = np.array(all_y_pred)
        agg_metrics = self._calculate_metrics(agg_y_true, agg_y_pred)

        return {
            'dataset': dataset,
            'canonical_model': canonical_model,
            'partition_type': f'aggregated_{partition_type}',
            'y_true': agg_y_true,
            'y_pred': agg_y_pred,
            'metrics': agg_metrics,
            'sample_count': len(agg_y_true),
            'fold_count': len(folds),
            'individual_fold_metrics': all_metrics
        }

    def filter_predictions(self, dataset: Optional[str] = None,
                           canonical_model: Optional[str] = None,
                           partition_type: Optional[str] = None,
                           include_aggregated: bool = True) -> List[Dict[str, Any]]:
        """
        Filter predictions by various criteria.

        Args:
            dataset: Filter by dataset name
            canonical_model: Filter by canonical model name
            partition_type: Filter by partition type ('train', 'val', 'test', 'aggregated_train', etc.)
            include_aggregated: Whether to include aggregated predictions from folds

        Returns:
            List of filtered prediction records
        """
        results = []

        # Get base predictions
        candidates = self.raw_data
        if dataset:
            candidates = [p for p in candidates if p.get('dataset') == dataset]
        if canonical_model:
            candidates = [p for p in candidates if self._extract_canonical_model_name(
                p.get('model', ''), p.get('real_model', '')) == canonical_model]
        if partition_type and not partition_type.startswith('aggregated_'):
            candidates = [p for p in candidates if self._classify_partition_type(
                p.get('partition', ''), p.get('fold_idx')) == partition_type]

        # Convert to processed format
        for pred in candidates:
            dataset_name = self.dataset_name_override or pred.get('dataset', 'unknown')
            model_name = pred.get('model', 'unknown')
            real_model_name = pred.get('real_model', model_name)
            canonical = self._extract_canonical_model_name(model_name, real_model_name)
            part_type = self._classify_partition_type(pred.get('partition', ''), pred.get('fold_idx'))

            y_true = np.array(pred.get('y_true', []))
            y_pred = np.array(pred.get('y_pred', []))
            metrics = self._calculate_metrics(y_true, y_pred)

            results.append({
                'dataset': dataset_name,
                'model': model_name,
                'canonical_model': canonical,
                'partition': pred.get('partition', ''),
                'partition_type': part_type,
                'y_true': y_true,
                'y_pred': y_pred,
                'metrics': metrics,
                'sample_count': len(y_true),
                'fold_idx': pred.get('fold_idx'),
                'metadata': pred.get('metadata', {}),
                'path': pred.get('path', '')
            })

        # Add aggregated predictions if requested
        if include_aggregated and partition_type and partition_type.startswith('aggregated_'):
            base_type = partition_type.replace('aggregated_', '')
            if dataset and canonical_model:
                agg_pred = self.get_aggregated_prediction(dataset, canonical_model, base_type)
                if agg_pred:
                    results.append(agg_pred)

        return results

    def get_top_k(self, k: int = 5, metric: str = 'rmse',
                  partition_type: str = 'test', dataset: Optional[str] = None,
                  canonical_model: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get top K performing predictions.

        Args:
            k: Number of top predictions to return
            metric: Metric to rank by
            partition_type: Partition type to consider ('test', 'val', 'train')
            dataset: Dataset filter
            canonical_model: Canonical model filter

        Returns:
            List of top K predictions
        """
        # Determine which partition to use - all test predictions are on the same test set
        if partition_type == 'test':
            # Get all test predictions (individual fold predictions and aggregated) from processed data
            candidates = self.processed_data['by_partition_type'].get('test', [])
            candidates.extend(self.processed_data['by_partition_type'].get('averaged_test', []))
            candidates.extend(self.processed_data['by_partition_type'].get('weighted_averaged_test', []))
        elif partition_type == 'val':
            # Get all val predictions (individual fold predictions and aggregated) from processed data
            candidates = self.processed_data['by_partition_type'].get('val', [])
            candidates.extend(self.processed_data['by_partition_type'].get('averaged_val', []))
            candidates.extend(self.processed_data['by_partition_type'].get('weighted_averaged_val', []))
        else:  # train
            candidates = self.processed_data['by_partition_type'].get('train', [])
            candidates.extend(self.processed_data['by_partition_type'].get('averaged_train', []))
            candidates.extend(self.processed_data['by_partition_type'].get('weighted_averaged_train', []))

        # Apply additional filters
        if dataset:
            candidates = [c for c in candidates if c.get('dataset') == dataset]
        if canonical_model:
            candidates = [c for c in candidates if c.get('canonical_model') == canonical_model]

        if not candidates:
            return []

        # Sort by metric (lower is better for rmse, mse, mae; higher for r2)
        higher_better = metric in ['r2', 'accuracy']
        candidates.sort(key=lambda x: x['metrics'].get(metric, float('inf') if higher_better else float('-inf')),
                       reverse=higher_better)

        # Remove duplicates by canonical model and real model, keeping the best performing one
        seen_models = {}
        unique_candidates = []

        for candidate in candidates:
            # Create a unique key based on canonical model and step/fold info (but ignore avg/weighted_avg suffix)
            canonical = candidate.get('canonical_model', 'unknown')
            real_model = candidate.get('real_model', 'unknown')

            # Extract base model info without avg/weighted_avg for deduplication
            base_key = real_model
            if '_avg' in real_model or '_weighted_avg' in real_model:
                # For aggregated models, create a base key that groups similar aggregations
                parts = real_model.split('_')
                base_parts = []
                for part in parts:
                    if part not in ['avg', 'weighted']:
                        base_parts.append(part)
                base_key = '_'.join(base_parts)

            model_key = f"{canonical}_{base_key}"

            if model_key not in seen_models:
                seen_models[model_key] = candidate
                unique_candidates.append(candidate)
            else:
                # Keep the better performing one
                current_score = candidate['metrics'].get(metric, float('inf') if higher_better else float('-inf'))
                existing_score = seen_models[model_key]['metrics'].get(metric, float('inf') if higher_better else float('-inf'))

                if (higher_better and current_score > existing_score) or (not higher_better and current_score < existing_score):
                    # Replace with better performing model
                    idx = unique_candidates.index(seen_models[model_key])
                    unique_candidates[idx] = candidate
                    seen_models[model_key] = candidate

        return unique_candidates[:k]

    def plot_top_k_comparison(self, k: int = 5, metric: str = 'rmse',
                            partition_type: str = 'test', dataset: Optional[str] = None,
                            figsize: Tuple[int, int] = (16, 10)) -> Figure:
        """
        Plot top K models with predicted vs true and residuals.

        Args:
            k: Number of top models to show
            metric: Metric for ranking
            partition_type: Partition type ('test', 'val', 'train')
            dataset: Dataset filter
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        top_predictions = self.get_top_k(k, metric, partition_type, dataset)

        if not top_predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No {partition_type} predictions found', ha='center', va='center', fontsize=16)
            return fig

        n_plots = len(top_predictions)
        cols = 2
        rows = n_plots

        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        if n_plots == 1:
            axes = [axes]

        for i, pred in enumerate(top_predictions):
            # Predicted vs True plot
            ax_scatter = axes[i][0] if n_plots > 1 else axes[0]

            # Ensure arrays are properly shaped and same size
            y_true = np.asarray(pred['y_true']).flatten()
            y_pred = np.asarray(pred['y_pred']).flatten()

            # Check if arrays have the same size for scatter plot
            if len(y_true) != len(y_pred):
                print(f"⚠️ Warning: Array size mismatch for {pred['canonical_model']}: y_true({len(y_true)}) vs y_pred({len(y_pred)})")
                # Use the minimum length to avoid scatter plot error
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]

            ax_scatter.scatter(y_true, y_pred, alpha=0.6, s=20)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            ax_scatter.set_xlabel('True Values')
            ax_scatter.set_ylabel('Predicted Values')
            # Use enhanced_model_name (with custom names) in title
            display_name = pred.get('enhanced_model_name', pred.get('real_model', pred['canonical_model']))
            ax_scatter.set_title(f'{display_name} ({pred["partition_type"]})')
            ax_scatter.grid(True, alpha=0.3)

            # Residuals plot
            ax_resid = axes[i][1] if n_plots > 1 else axes[1]

            # Ensure arrays are properly shaped for residuals calculation
            y_true_flat = np.asarray(y_true).flatten()
            y_pred_flat = np.asarray(y_pred).flatten()

            # Check if arrays have the same size
            if len(y_true_flat) != len(y_pred_flat):
                print(f"⚠️ Warning: Array size mismatch for {pred['canonical_model']}: y_true({len(y_true_flat)}) vs y_pred({len(y_pred_flat)})")
                # Use the minimum length to avoid scatter plot error
                min_len = min(len(y_true_flat), len(y_pred_flat))
                y_true_flat = y_true_flat[:min_len]
                y_pred_flat = y_pred_flat[:min_len]

            residuals = y_true_flat - y_pred_flat
            ax_resid.scatter(y_pred_flat, residuals, alpha=0.6, s=20)
            ax_resid.axhline(y=0, color='r', linestyle='--', alpha=0.8)
            ax_resid.set_xlabel('Predicted Values')
            ax_resid.set_ylabel('Residuals')
            # Use unique ID in residuals title as well
            ax_resid.set_title(f'Residuals - {metric.upper()}: {pred["metrics"][metric]:.4f}')
            ax_resid.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_top_k_confusionMatrix(self, k: int = 5, metric: str = 'accuracy',
                                 partition_type: str = 'test', dataset: Optional[str] = None,
                                 figsize: Tuple[int, int] = (16, 10)) -> Figure:
        """
        Plot confusion matrices for top K classification models.

        Args:
            k: Number of top models to show
            metric: Metric for ranking
            partition_type: Partition type ('test', 'val', 'train')
            dataset: Dataset filter
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        top_predictions = self.get_top_k(k, metric, partition_type, dataset)

        if not top_predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No {partition_type} predictions found', ha='center', va='center', fontsize=16)
            return fig

        n_plots = len(top_predictions)
        cols = int(np.ceil(np.sqrt(n_plots)))
        rows = int(np.ceil(n_plots / cols))

        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        if n_plots == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()

        for i, pred in enumerate(top_predictions):
            if rows > 1 and cols > 1:
                ax = axes[i // cols, i % cols]
            else:
                ax = axes[i]

            y_true = np.asarray(pred['y_true']).flatten()
            y_pred = np.asarray(pred['y_pred']).flatten()

            # Convert predictions to class labels if needed
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred_labels = np.argmax(y_pred, axis=1)
            else:
                y_pred_labels = np.round(y_pred).astype(int)

            y_true_labels = y_true.astype(int)

            # Ensure both arrays are 1-dimensional and same length
            y_true_labels = y_true_labels.flatten()
            y_pred_labels = y_pred_labels.flatten()

            # Check array compatibility
            if len(y_true_labels) != len(y_pred_labels):
                print(f"⚠️ Warning: Array length mismatch for confusion matrix in {pred['canonical_model']}: y_true({len(y_true_labels)}) vs y_pred({len(y_pred_labels)})")
                min_len = min(len(y_true_labels), len(y_pred_labels))
                y_true_labels = y_true_labels[:min_len]
                y_pred_labels = y_pred_labels[:min_len]

            # Compute confusion matrix
            cm = confusion_matrix(y_true_labels, y_pred_labels)

            # Plot confusion matrix
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            # Use enhanced_model_name (with custom names) in title
            display_name = pred.get('enhanced_model_name', pred.get('real_model', pred['canonical_model']))
            ax.set_title(f'{display_name}\n{metric.upper()}: {pred["metrics"][metric]:.4f}')

            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Add labels - ensure arrays are 1D before concatenating
            classes = np.unique(np.concatenate([y_true_labels.ravel(), y_pred_labels.ravel()]))
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

            # Add text annotations
            thresh = cm.max() / 2.
            for ii in range(cm.shape[0]):
                for jj in range(cm.shape[1]):
                    ax.text(jj, ii, format(cm[ii, jj], 'd'),
                           ha="center", va="center",
                           color="white" if cm[ii, jj] > thresh else "black")

        # Hide empty subplots
        for i in range(n_plots, rows * cols):
            if rows > 1 and cols > 1:
                axes[i // cols, i % cols].set_visible(False)
            else:
                axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_score_histogram(self, metric: str = 'rmse', dataset: Optional[str] = None,
                           partition_type: Optional[str] = None, bins: int = 20,
                           figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot histogram of scores for specified metric.

        Args:
            metric: Metric to plot
            dataset: Dataset filter
            partition_type: Partition type filter
            bins: Number of histogram bins
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        predictions = self.filter_predictions(dataset=dataset, partition_type=partition_type)

        if not predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No predictions found', ha='center', va='center', fontsize=16)
            return fig

        scores = [p['metrics'].get(metric, np.nan) for p in predictions]
        scores = [s for s in scores if not np.isnan(s)]

        if not scores:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No valid {metric} scores found', ha='center', va='center', fontsize=16)
            return fig

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(scores, bins=bins, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'{metric.upper()} Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {metric.upper()} Scores')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = np.mean(scores)
        median_val = np.median(scores)
        ax.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='g', linestyle='--', label=f'Median: {median_val:.4f}')
        ax.legend()

        return fig

    def plot_performance_heatmap(self, x_axis: str = 'canonical_model', y_axis: str = 'dataset',
                               metric: str = 'rmse', dataset: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Plot heatmap of performance by canonical model and dataset or pipeline config.

        Args:
            x_axis: X-axis dimension ('canonical_model' or 'dataset')
            y_axis: Y-axis dimension ('dataset' or 'pipeline_config')
            metric: Metric to display
            dataset: Dataset filter (if x_axis is 'canonical_model')
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        # Get all relevant predictions
        predictions = self.filter_predictions(dataset=dataset)

        if not predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No predictions found', ha='center', va='center', fontsize=16)
            return fig

        # Group by x and y dimensions
        grouped_data = defaultdict(lambda: defaultdict(list))

        for pred in predictions:
            x_val = pred.get(x_axis, 'unknown')
            if y_axis == 'dataset':
                y_val = pred.get('dataset', 'unknown')
            else:  # pipeline_config - use partition_type as proxy
                y_val = pred.get('partition_type', 'unknown')

            grouped_data[y_val][x_val].append(pred['metrics'].get(metric, np.nan))

        # Extract unique values
        y_labels = sorted(grouped_data.keys())
        x_labels = sorted(set(x for y_data in grouped_data.values() for x in y_data.keys()))

        # Create matrix
        matrix = np.full((len(y_labels), len(x_labels)), np.nan)

        for i, y_val in enumerate(y_labels):
            for j, x_val in enumerate(x_labels):
                scores = grouped_data[y_val].get(x_val, [])
                if scores:
                    # Take best score (lowest for rmse, highest for r2)
                    higher_better = metric in ['r2', 'accuracy', 'f1', 'precision', 'recall']
                    matrix[i, j] = max(scores) if higher_better else min(scores)

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        if np.any(~np.isnan(matrix)):
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax, label=metric.upper())

            # Add values
            for i in range(len(y_labels)):
                for j in range(len(x_labels)):
                    if not np.isnan(matrix[i, j]):
                        ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
                               color='white' if matrix[i, j] > np.nanmean(matrix) else 'black', fontsize=8)

        ax.set_xticks(range(len(x_labels)))
        ax.set_yticks(range(len(y_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(x_axis.replace('_', ' ').title())
        ax.set_ylabel(y_axis.replace('_', ' ').title())
        ax.set_title(f'{metric.upper()} Performance Heatmap')

        return fig

    def plot_candlestick_models(self, metric: str = 'rmse', dataset: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Plot candlestick chart showing avg/variance per canonical model.

        Args:
            metric: Metric to analyze
            dataset: Dataset filter
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        predictions = self.filter_predictions(dataset=dataset)

        if not predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No predictions found', ha='center', va='center', fontsize=16)
            return fig

        # Group by canonical model
        model_stats = defaultdict(list)

        for pred in predictions:
            canonical = pred['canonical_model']
            score = pred['metrics'].get(metric, np.nan)
            if not np.isnan(score):
                model_stats[canonical].append(score)

        if not model_stats:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No valid {metric} scores found', ha='center', va='center', fontsize=16)
            return fig

        # Calculate stats for each model
        models = []
        means = []
        mins = []
        maxs = []
        q25s = []
        q75s = []

        for model, scores in model_stats.items():
            models.append(model)
            means.append(np.mean(scores))
            mins.append(np.min(scores))
            maxs.append(np.max(scores))
            q25s.append(np.percentile(scores, 25))
            q75s.append(np.percentile(scores, 75))

        # Sort by mean performance
        higher_better = metric in ['r2', 'accuracy', 'f1', 'precision', 'recall']
        sort_indices = np.argsort(means)
        if higher_better:
            sort_indices = sort_indices[::-1]

        models = [models[i] for i in sort_indices]
        means = [means[i] for i in sort_indices]
        mins = [mins[i] for i in sort_indices]
        maxs = [maxs[i] for i in sort_indices]
        q25s = [q25s[i] for i in sort_indices]
        q75s = [q75s[i] for i in sort_indices]

        # Create candlestick plot
        fig, ax = plt.subplots(figsize=figsize)

        for i, model in enumerate(models):
            # High-low line
            ax.plot([i, i], [mins[i], maxs[i]], color='black', linewidth=1)
            # Rectangle for Q25-Q75
            ax.add_patch(plt.Rectangle((i-0.3, q25s[i]), 0.6, q75s[i]-q25s[i],
                                     fill=True, color='lightblue', alpha=0.7))
            # Mean line
            ax.plot([i-0.3, i+0.3], [means[i], means[i]], color='red', linewidth=2)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(f'{metric.upper()} Score')
        ax.set_title(f'{metric.upper()} Distribution by Model (Candlestick)')
        ax.grid(True, alpha=0.3)

        return fig

    def plot_all_models_barplot(self, metric: str = 'rmse', dataset: Optional[str] = None,
                              partition_type: Optional[str] = None,
                              figsize: Tuple[int, int] = (14, 8)) -> Figure:
        """
        Plot barplot showing all models with specified metric.

        Args:
            metric: Metric to display
            dataset: Dataset filter
            partition_type: Partition type filter
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        predictions = self.filter_predictions(dataset=dataset, partition_type=partition_type)

        if not predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No predictions found', ha='center', va='center', fontsize=16)
            return fig

        # Group by canonical model and take best score
        model_scores = defaultdict(lambda: {'best_score': float('inf') if metric not in ['r2'] else float('-inf'),
                                          'count': 0})

        higher_better = metric in ['r2', 'accuracy', 'f1', 'precision', 'recall']

        for pred in predictions:
            canonical = pred['canonical_model']
            score = pred['metrics'].get(metric, np.nan)

            if not np.isnan(score):
                current_best = model_scores[canonical]['best_score']
                if (higher_better and score > current_best) or (not higher_better and score < current_best):
                    model_scores[canonical]['best_score'] = score
                model_scores[canonical]['count'] += 1

        if not model_scores:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No valid {metric} scores found', ha='center', va='center', fontsize=16)
            return fig

        # Prepare data for plotting
        models = []
        scores = []
        counts = []

        for model, data in model_scores.items():
            models.append(model)
            scores.append(data['best_score'])
            counts.append(data['count'])

        # Sort by score
        sort_indices = np.argsort(scores)
        if higher_better:
            sort_indices = sort_indices[::-1]

        models = [models[i] for i in sort_indices]
        scores = [scores[i] for i in sort_indices]
        counts = [counts[i] for i in sort_indices]

        # Create bar plot
        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(range(len(models)), scores, alpha=0.7, edgecolor='black')

        # Color bars based on performance
        norm_scores = np.array(scores) / np.max(np.abs(scores)) if scores else np.array([])
        colors = ['red' if s < 0.5 else 'green' for s in norm_scores]

        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # Add value labels
        for i, (bar, score, count) in enumerate(zip(bars, scores, counts)):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{score:.3f}\n({count} runs)', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(f'Best {metric.upper()} Score')
        ax.set_title(f'Best {metric.upper()} by Model')
        ax.grid(True, alpha=0.3, axis='y')

        return fig

    def plot_performance_matrix(self, metric: str = 'rmse', partition_type: str = 'test',
                               normalize: bool = True, figsize: Tuple[int, int] = (14, 10)) -> Figure:
        """
        Plot matrix showing best performance by model type for each dataset.

        Args:
            metric: Metric to display (default: 'rmse')
            partition_type: Partition type to consider ('test', 'val', 'train')
            normalize: Whether to normalize scores for better color comparison
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        # Get all predictions for the specified partition type
        predictions = self.filter_predictions(partition_type=partition_type)

        if not predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No {partition_type} predictions found', ha='center', va='center', fontsize=16)
            return fig

        # Group by dataset and canonical model to find best performance
        dataset_model_scores = defaultdict(lambda: defaultdict(list))

        for pred in predictions:
            dataset = pred['dataset']
            canonical_model = pred['canonical_model']
            score = pred['metrics'].get(metric, np.nan)

            if not np.isnan(score):
                dataset_model_scores[dataset][canonical_model].append(score)

        if not dataset_model_scores:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No valid {metric} scores found', ha='center', va='center', fontsize=16)
            return fig

        # Extract unique datasets and models
        datasets = sorted(dataset_model_scores.keys())
        all_models = set()
        for dataset_data in dataset_model_scores.values():
            all_models.update(dataset_data.keys())
        models = sorted(all_models)

        # Create matrix with best scores
        matrix = np.full((len(datasets), len(models)), np.nan)
        best_scores = {}  # Store best scores for each dataset-model combination

        higher_better = metric in ['r2', 'accuracy', 'f1', 'precision', 'recall']

        for i, dataset in enumerate(datasets):
            for j, model in enumerate(models):
                scores = dataset_model_scores[dataset].get(model, [])
                if scores:
                    # Get best score (lowest for rmse/mse/mae, highest for r2/accuracy)
                    best_score = max(scores) if higher_better else min(scores)
                    matrix[i, j] = best_score
                    best_scores[(dataset, model)] = best_score

        # Normalize scores if requested
        if normalize and not np.all(np.isnan(matrix)):
            # For RMSE and similar metrics (lower is better), we want to invert for color mapping
            if not higher_better:
                # Normalize inversely for "lower is better" metrics
                valid_scores = matrix[~np.isnan(matrix)]
                if len(valid_scores) > 0:
                    min_score = np.min(valid_scores)
                    max_score = np.max(valid_scores)
                    if max_score != min_score:
                        # Invert normalization: best (lowest) scores become 1, worst (highest) become 0
                        matrix_norm = np.full_like(matrix, np.nan)
                        valid_mask = ~np.isnan(matrix)
                        matrix_norm[valid_mask] = 1 - (matrix[valid_mask] - min_score) / (max_score - min_score)
                        matrix = matrix_norm
            else:
                # Standard normalization for "higher is better" metrics
                valid_scores = matrix[~np.isnan(matrix)]
                if len(valid_scores) > 0:
                    min_score = np.min(valid_scores)
                    max_score = np.max(valid_scores)
                    if max_score != min_score:
                        matrix_norm = np.full_like(matrix, np.nan)
                        valid_mask = ~np.isnan(matrix)
                        matrix_norm[valid_mask] = (matrix[valid_mask] - min_score) / (max_score - min_score)
                        matrix = matrix_norm

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Use a color map where better performance is greener
        cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap

        # Create masked array to handle NaN values
        masked_matrix = np.ma.masked_invalid(matrix)

        im = ax.imshow(masked_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1 if normalize else None)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        if normalize:
            if higher_better:
                cbar.set_label(f'Normalized {metric.upper()} (1=best, 0=worst)')
            else:
                cbar.set_label(f'Normalized {metric.upper()} (1=best, 0=worst)')
        else:
            cbar.set_label(f'{metric.upper()} Score')

        # Set ticks and labels
        ax.set_xticks(range(len(models)))
        ax.set_yticks(range(len(datasets)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_yticklabels(datasets)
        ax.set_xlabel('Model Type')
        ax.set_ylabel('Dataset')

        title = f'Best {metric.upper()} Performance Matrix'
        if normalize:
            title += ' (Normalized)'
        ax.set_title(title)

        # Add text annotations with actual scores
        for i in range(len(datasets)):
            for j in range(len(models)):
                if not np.isnan(matrix[i, j]):
                    # Get original score for annotation
                    original_score = best_scores.get((datasets[i], models[j]), matrix[i, j])

                    # Choose text color based on background
                    if normalize:
                        text_color = 'white' if matrix[i, j] < 0.5 else 'black'
                    else:
                        text_color = 'white' if matrix[i, j] > np.nanmean(matrix) else 'black'

                    ax.text(j, i, f'{original_score:.3f}',
                           ha='center', va='center', color=text_color, fontsize=9, weight='bold')

        plt.tight_layout()
        return fig

    def plot_score_boxplots_by_dataset(self, metric: str = 'rmse', partition_type: str = 'test',
                                      figsize: Tuple[int, int] = (14, 8)) -> Figure:
        """
        Plot box plots showing score distributions for each dataset.

        Args:
            metric: Metric to display (default: 'rmse')
            partition_type: Partition type to consider ('test', 'val', 'train')
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        # Get all predictions for the specified partition type
        predictions = self.filter_predictions(partition_type=partition_type)

        if not predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No {partition_type} predictions found', ha='center', va='center', fontsize=16)
            return fig

        # Group scores by dataset
        dataset_scores = defaultdict(list)

        for pred in predictions:
            dataset = pred['dataset']
            score = pred['metrics'].get(metric, np.nan)

            if not np.isnan(score):
                dataset_scores[dataset].append(score)

        if not dataset_scores:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No valid {metric} scores found', ha='center', va='center', fontsize=16)
            return fig

        # Prepare data for box plots
        datasets = sorted(dataset_scores.keys())
        scores_list = [dataset_scores[dataset] for dataset in datasets]

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create box plots
        bp = ax.boxplot(scores_list, labels=datasets, patch_artist=True)

        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Customize the plot
        ax.set_xlabel('Dataset')
        ax.set_ylabel(f'{metric.upper()} Score')
        ax.set_title(f'{metric.upper()} Score Distribution by Dataset ({partition_type} partition)')
        ax.grid(True, alpha=0.3, axis='y')

        # Rotate x-axis labels if needed
        if len(datasets) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Add statistics as text
        for i, (dataset, scores) in enumerate(zip(datasets, scores_list)):
            mean_score = np.mean(scores)
            median_score = np.median(scores)
            std_score = np.std(scores)
            n_scores = len(scores)

            # Add text above each box plot
            y_pos = max(scores) + (max(max(s) for s in scores_list) - min(min(s) for s in scores_list)) * 0.05
            ax.text(i + 1, y_pos, f'n={n_scores}\μ={mean_score:.3f}\nσ={std_score:.3f}',
                   ha='center', va='bottom', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig
