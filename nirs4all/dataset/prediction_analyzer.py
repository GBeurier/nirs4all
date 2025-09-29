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
            partition = pred_record.get('partition', 'unknown')

            # Extract canonical model name (remove parameters and suffixes)
            canonical_model = self._extract_canonical_model_name(model)

            # Determine partition type
            partition_type = self._classify_partition_type(partition)

            # Calculate metrics
            y_true = np.array(pred_record.get('y_true', []))
            y_pred = np.array(pred_record.get('y_pred', []))
            metrics = self._calculate_metrics(y_true, y_pred)

            # Create processed record
            processed_record = {
                'dataset': dataset,
                'model': model,
                'canonical_model': canonical_model,
                'partition': partition,
                'partition_type': partition_type,
                'y_true': y_true,
                'y_pred': y_pred,
                'metrics': metrics,
                'sample_count': len(y_true),
                'fold_idx': pred_record.get('fold_idx'),
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

    def _extract_canonical_model_name(self, model_name: str) -> str:
        """Extract canonical model name by removing parameters and suffixes."""
        # Remove common parameter patterns
        canonical = re.sub(r'\([^)]*\)', '', model_name)  # Remove parentheses with params
        canonical = re.sub(r'_\d+$', '', canonical)  # Remove trailing numbers
        canonical = canonical.strip()
        return canonical if canonical else model_name

    def _classify_partition_type(self, partition: str) -> str:
        """Classify partition into high-level types."""
        if 'global_train' in partition:
            return 'global_train'
        elif 'test' in partition and 'fold' not in partition and 'avg' not in partition:
            return 'global_test'
        elif 'train_fold' in partition:
            return 'train_fold'
        elif 'val_fold' in partition:
            return 'val_fold'
        elif 'test_fold' in partition and 'avg' not in partition:
            return 'test_fold'
        elif 'avg_test_fold' in partition:
            return 'averaged_test'
        elif 'weighted_avg_test_fold' in partition:
            return 'weighted_averaged_test'
        else:
            return partition

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
            candidates = [p for p in candidates if self._extract_canonical_model_name(p.get('model', '')) == canonical_model]
        if partition_type and not partition_type.startswith('aggregated_'):
            candidates = [p for p in candidates if self._classify_partition_type(p.get('partition', '')) == partition_type]

        # Convert to processed format
        for pred in candidates:
            dataset_name = self.dataset_name_override or pred.get('dataset', 'unknown')
            model_name = pred.get('model', 'unknown')
            canonical = self._extract_canonical_model_name(model_name)
            part_type = self._classify_partition_type(pred.get('partition', ''))

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
        # Determine which partition to use
        if partition_type == 'test':
            # Prefer global_test, then aggregated_test
            candidates = self.filter_predictions(dataset=dataset, canonical_model=canonical_model,
                                                   partition_type='global_test')
            if not candidates:
                candidates = self.filter_predictions(dataset=dataset, canonical_model=canonical_model,
                                                     partition_type='aggregated_test', include_aggregated=True)
        elif partition_type == 'val':
            # Use aggregated_val if available, otherwise best val_fold
            candidates = self.filter_predictions(dataset=dataset, canonical_model=canonical_model,
                                               partition_type='aggregated_val', include_aggregated=True)
            if not candidates:
                candidates = self.filter_predictions(dataset=dataset, canonical_model=canonical_model,
                                                   partition_type='val_fold')
        else:  # train
            candidates = self.filter_predictions(dataset=dataset, canonical_model=canonical_model,
                                               partition_type='aggregated_train', include_aggregated=True)
            if not candidates:
                candidates = self.filter_predictions(dataset=dataset, canonical_model=canonical_model,
                                                   partition_type='train_fold')

        if not candidates:
            return []

        # Sort by metric (lower is better for rmse, mse, mae; higher for r2)
        higher_better = metric in ['r2', 'accuracy']
        candidates.sort(key=lambda x: x['metrics'].get(metric, float('inf') if higher_better else float('-inf')),
                       reverse=higher_better)

        return candidates[:k]

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
            y_true = pred['y_true']
            y_pred = pred['y_pred']

            ax_scatter.scatter(y_true, y_pred, alpha=0.6, s=20)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            ax_scatter.set_xlabel('True Values')
            ax_scatter.set_ylabel('Predicted Values')
            ax_scatter.set_title(f'{pred["canonical_model"]} ({pred["partition_type"]})')
            ax_scatter.grid(True, alpha=0.3)

            # Residuals plot
            ax_resid = axes[i][1] if n_plots > 1 else axes[1]
            residuals = y_true - y_pred
            ax_resid.scatter(y_pred, residuals, alpha=0.6, s=20)
            ax_resid.axhline(y=0, color='r', linestyle='--', alpha=0.8)
            ax_resid.set_xlabel('Predicted Values')
            ax_resid.set_ylabel('Residuals')
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

            y_true = pred['y_true']
            y_pred = pred['y_pred']

            # Convert predictions to class labels if needed
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                y_pred_labels = np.argmax(y_pred, axis=1)
            else:
                y_pred_labels = np.round(y_pred).astype(int)

            y_true_labels = y_true.astype(int)

            # Compute confusion matrix
            cm = confusion_matrix(y_true_labels, y_pred_labels)

            # Plot confusion matrix
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.set_title(f'{pred["canonical_model"]}\n{metric.upper()}: {pred["metrics"][metric]:.4f}')

            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Add labels
            classes = np.unique(np.concatenate([y_true_labels, y_pred_labels]))
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
