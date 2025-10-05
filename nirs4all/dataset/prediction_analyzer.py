"""
PredictionAnalyzer - Advanced analysis and visualization of pipeline prediction results

This module provides comprehensive analysis capabilities for prediction data,
allowing users to filter, aggregate, and visualize model performance across different configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import re

from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from nirs4all.dataset.predictions import Predictions
from nirs4all.utils.model_utils import ModelUtils, TaskType


class PredictionAnalyzer:
    """
    Advanced analyzer for prediction results with filtering, aggregation, and visualization capabilities.

    Features:
    - Smart filtering by partition type (train/test/val), canonical model, dataset
    - Multiple visualization types with metric-based ranking
    - Comprehensive performance analysis across configurations
    """

    def __init__(self, predictions_obj: Predictions, dataset_name_override: str = None):
        """
        Initialize with a predictions object.

        Args:
            predictions_obj: The predictions object containing prediction data
            dataset_name_override: Override for dataset name display
        """
        self.predictions = predictions_obj
        self.dataset_name_override = dataset_name_override
        self.model_utils = ModelUtils()

    def _natural_sort_key(self, text: str):
        """Generate a sorting key that handles numeric components naturally.

        E.g., 'PLSRegression_10_cp' will sort after 'PLSRegression_2_cp'
        """
        def convert(part):
            if part.isdigit():
                return int(part)
            return part.lower()

        return [convert(c) for c in re.split(r'(\d+)', str(text))]

    def _get_enhanced_predictions(self, **filters) -> List[Dict[str, Any]]:
        """Get predictions with enhanced metrics calculated on-the-fly."""
        predictions = self.predictions.filter_predictions(**filters)
        enhanced_predictions = []

        for pred in predictions:
            # Calculate metrics using ModelUtils if not already present
            y_true = np.array(pred.get('y_true', []))
            y_pred = np.array(pred.get('y_pred', []))

            if len(y_true) > 0 and len(y_pred) > 0:
                try:
                    task_type = self.model_utils.detect_task_type(y_true)
                    metrics = self.model_utils.calculate_scores(y_true, y_pred, task_type)
                except Exception as e:
                    print(f"⚠️ Error calculating metrics: {e}")
                    metrics = {}
            else:
                metrics = {}

            # Enhanced prediction record
            enhanced_pred = {
                'dataset_name': self.dataset_name_override or pred.get('dataset_name', 'unknown'),
                'model_name': pred.get('model_name', 'unknown'),
                'partition': pred.get('partition', 'unknown'),
                'fold_id': pred.get('fold_id'),
                'y_true': y_true,
                'y_pred': y_pred,
                'metrics': metrics,
                'sample_count': len(y_true),
                'metadata': pred.get('metadata', {})
            }
            enhanced_predictions.append(enhanced_pred)

        return enhanced_predictions

    def get_top_k(self, k: int = 5, metric: str = 'rmse',
                  partition: str = '', dataset_name: Optional[str] = None,
                  model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get top K performing predictions using the Predictions API.

        Args:
            k: Number of top predictions to return
            metric: Metric to rank by
            partition: Partition to consider ('test', 'val', 'train')
            dataset_name: Dataset filter
            model_name: Model filter

        Returns:
            List of top K predictions
        """
        filters = {'partition': partition}
        if dataset_name:
            filters['dataset_name'] = dataset_name
        if model_name:
            filters['model_name'] = model_name

        # Use the Predictions API directly
        try:
            top_predictions = self.predictions.top_k(
                k=k,
                metric=metric,
                ascending=(metric not in ['r2', 'accuracy', 'f1']),
                **filters
            )
            return top_predictions
        except Exception as e:
            print(f"⚠️ Error getting top k predictions: {e}")
            # Fallback to manual calculation
            enhanced_preds = self._get_enhanced_predictions(**filters)
            if not enhanced_preds:
                return []

            # Sort by metric
            higher_better = metric in ['r2', 'accuracy', 'f1', 'precision', 'recall']
            enhanced_preds.sort(
                key=lambda x: x['metrics'].get(metric, float('inf') if higher_better else float('-inf')),
                reverse=higher_better
            )
            return enhanced_preds[:k]

    def plot_top_k_comparison(self, k: int = 5, metric: str = 'rmse',
                              partition: str = '', dataset_name: Optional[str] = None,
                              figsize: Tuple[int, int] = (16, 10)) -> Figure:
        """
        Plot top K models with predicted vs true and residuals.

        Args:
            k: Number of top models to show
            metric: Metric for ranking
            partition: Partition type ('test', 'val', 'train')
            dataset_name: Dataset filter
            figsize: Figure size

        Returns:
            matplotlib Figure
        """

        if partition == '':
            partition = 'val'
        top_predictions = self.get_top_k(k, metric, partition, dataset_name)
        if not top_predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No {partition} predictions found',
                   ha='center', va='center', fontsize=16)
            return fig

        n_plots = len(top_predictions)
        cols = 2
        rows = n_plots

        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # Handle different subplot configurations
        if n_plots == 1:
            # For single row, axes is 1D array [ax_left, ax_right]
            axes = [axes]  # Wrap in list to make it [[ax_left, ax_right]]
        elif rows == 1:
            # This shouldn't happen with our logic, but just in case
            axes = [axes]

        for i, pred in enumerate(top_predictions):
            # Predicted vs True plot
            ax_scatter = axes[i][0]

            y_true = np.asarray(pred['y_true']).flatten()
            y_pred = np.asarray(pred['y_pred']).flatten()

            # Check if arrays have the same size for scatter plot
            if len(y_true) != len(y_pred):
                print(f"⚠️ Warning: Array size mismatch for {pred['model_name']}: "
                      f"y_true({len(y_true)}) vs y_pred({len(y_pred)})")
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]

            ax_scatter.scatter(y_true, y_pred, alpha=0.6, s=20)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            ax_scatter.set_xlabel('True Values')
            ax_scatter.set_ylabel('Predicted Values')

            model_display = pred['model_name'] + " [" + pred["id"] + "]"
            # if pred.get('fold_id') is not None:
            #     model_display += f" (Fold {pred['fold_id']})"

            title = f'{model_display}'
            # if partition not in ['all', 'ALL', 'All', '_all_', '']:
                # title += f' - {partition}'
            ax_scatter.set_title(title)
            ax_scatter.grid(True, alpha=0.3)

            # Residuals plot
            ax_resid = axes[i][1]

            residuals = y_true - y_pred
            ax_resid.scatter(y_pred, residuals, alpha=0.6, s=20)
            ax_resid.axhline(y=0, color='r', linestyle='--', alpha=0.8)
            ax_resid.set_xlabel('Predicted Values')
            ax_resid.set_ylabel('Residuals')

            # Calculate metric on-the-fly if not in prediction
            score_value = pred.get(metric, 'N/A')
            ax_resid.set_title(f'Residuals - {metric.upper()}: {score_value:.4f}'
                               if isinstance(score_value, (int, float)) else f'Residuals - {metric.upper()}: {score_value}')
            ax_resid.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_top_k_confusionMatrix(self, k: int = 5, metric: str = 'accuracy',
                                   partition: str = '', dataset_name: Optional[str] = None,
                                   figsize: Tuple[int, int] = (16, 10)) -> Figure:
        """
        Plot confusion matrices for top K classification models.

        Args:
            k: Number of top models to show
            metric: Metric for ranking
            partition: Partition type ('test', 'val', 'train')
            dataset_name: Dataset filter
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        print(self.predictions)
        top_predictions = self.predictions.top_k(k, partition=partition, ascending=False)
        print("length top_predictions", len(top_predictions))

        if not top_predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No {partition} predictions found',
                   ha='center', va='center', fontsize=16)
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

            if len(y_true_labels) != len(y_pred_labels):
                print(f"⚠️ Warning: Array length mismatch for confusion matrix in {pred['model_name']}: "
                      f"y_true({len(y_true_labels)}) vs y_pred({len(y_pred_labels)})")
                min_len = min(len(y_true_labels), len(y_pred_labels))
                y_true_labels = y_true_labels[:min_len]
                y_pred_labels = y_pred_labels[:min_len]

            # Compute confusion matrix
            confusion_mat = sk_confusion_matrix(y_true_labels, y_pred_labels)

            # Plot confusion matrix
            im = ax.imshow(confusion_mat, interpolation='nearest', cmap='Blues')

            model_display = pred.get('model_name', 'Unknown')
            # Get metric value
            score_value = pred.get('val_score') if metric == 'loss' else pred.get('test_score')
            if score_value is None:
                # Try to get from computed metrics if available
                if 'metrics' in pred and metric in pred['metrics']:
                    score_value = pred['metrics'][metric]
                else:
                    score_value = 'N/A'
            score_str = f'{score_value:.4f}' if isinstance(score_value, (int, float)) else str(score_value)

            ax.set_title(f'{model_display}\n{metric.upper()}: {score_str}')

            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Add labels
            classes = np.unique(np.concatenate([y_true_labels.ravel(), y_pred_labels.ravel()]))
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

            # Add text annotations
            thresh = confusion_mat.max() / 2.
            for ii in range(confusion_mat.shape[0]):
                for jj in range(confusion_mat.shape[1]):
                    ax.text(jj, ii, format(confusion_mat[ii, jj], 'd'),
                            ha="center", va="center",
                            color="white" if confusion_mat[ii, jj] > thresh else "black")        # Hide empty subplots
        for i in range(n_plots, rows * cols):
            if rows > 1 and cols > 1:
                axes[i // cols, i % cols].set_visible(False)
            else:
                axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_score_histogram(self, metric: str = 'rmse', dataset_name: Optional[str] = None,
                             partition: Optional[str] = None, bins: int = 20,
                             figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot histogram of scores for specified metric.

        Args:
            metric: Metric to plot
            dataset_name: Dataset filter
            partition: Partition filter
            bins: Number of histogram bins
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        filters = {}
        if dataset_name:
            filters['dataset_name'] = dataset_name
        if partition:
            filters['partition'] = partition

        predictions = self._get_enhanced_predictions(**filters)

        if not predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No predictions found', ha='center', va='center', fontsize=16)
            return fig

        scores = [p['metrics'].get(metric, np.nan) for p in predictions]
        scores = [s for s in scores if not np.isnan(s)]

        if not scores:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No valid {metric} scores found',
                   ha='center', va='center', fontsize=16)
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

    def plot_performance_heatmap(self, x_axis: str = 'model_name', y_axis: str = 'dataset_name',
                                 metric: str = 'rmse', partition: str = '',
                                 figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Plot heatmap of performance by model and dataset.

        Args:
            x_axis: X-axis dimension ('model_name' or 'dataset_name')
            y_axis: Y-axis dimension ('dataset_name' or 'model_name')
            metric: Metric to display
            partition: Partition filter
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        predictions = self.get_top_k(-1, metric, partition)

        if not predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No predictions found', ha='center', va='center', fontsize=16)
            return fig

        # Group by x and y dimensions
        grouped_data = defaultdict(lambda: defaultdict(list))

        for pred in predictions:
            x_val = pred.get(x_axis, 'unknown')
            y_val = pred.get(y_axis, 'unknown')
            score = pred.get(metric, np.nan)

            if not np.isnan(score):
                grouped_data[y_val][x_val].append(score)

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
                               color='white' if matrix[i, j] > np.nanmean(matrix) else 'black',
                               fontsize=8)

        ax.set_xticks(range(len(x_labels)))
        ax.set_yticks(range(len(y_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(x_axis.replace('_', ' ').title())
        ax.set_ylabel(y_axis.replace('_', ' ').title())
        ax.set_title(f'{metric.upper()} Performance Heatmap')

        return fig

    def plot_candlestick_models(self, metric: str = 'rmse', partition: str = '',
                                figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Plot candlestick chart showing avg/variance per model.

        Args:
            metric: Metric to analyze
            partition: Partition filter
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        predictions = self.predictions.top_k(-1, metric, partition=partition)

        if not predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No predictions found', ha='center', va='center', fontsize=16)
            return fig

        # Group by model
        model_stats = defaultdict(list)

        for pred in predictions:
            model = pred['model_classname']
            score = pred.get(metric, np.nan)
            if not np.isnan(score):
                model_stats[model].append(score)

        if not model_stats:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No valid {metric} scores found',
                   ha='center', va='center', fontsize=16)
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

    def plot_performance_matrix(self, metric: str = 'rmse', partition: str = '', separate_avg: bool = False,
                               normalize: bool = True, figsize: Tuple[int, int] = (14, 10)) -> Figure:
        """
        Plot matrix showing best performance by model type for each dataset.

        Args:
            metric: Metric to display (default: 'rmse')
            partition: Partition type to consider ('test', 'val', 'train')
            normalize: Whether to normalize scores for better color comparison
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        # Get all predictions for the specified partition type
        predictions = self.predictions.top_k(-1, metric, partition=partition)

        if not predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No {partition} predictions found', ha='center', va='center', fontsize=16)
            return fig

        # Group by dataset and model to find best performance
        dataset_model_scores = defaultdict(lambda: defaultdict(list))

        for pred in predictions:
            dataset = pred['dataset_name']
            if separate_avg:
                model = pred['model_name']
            else:
                model = pred['model_classname']
            score = pred.get(metric, np.nan)

            if not np.isnan(score):
                dataset_model_scores[dataset][model].append(score)

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
        cmap = 'RdYlGn'  # Red-Yellow-Green colormap

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

    def plot_score_boxplots_by_dataset(self, metric: str = 'rmse', partition: str = 'val',
                                      figsize: Tuple[int, int] = (14, 8)) -> Figure:
        """
        Plot box plots showing score distributions for each dataset.

        Args:
            metric: Metric to display (default: 'rmse')
            partition: Partition type to consider ('test', 'val', 'train')
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        # Get all predictions for the specified partition type
        predictions = self._get_enhanced_predictions(partition=partition)

        if not predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No {partition} predictions found', ha='center', va='center', fontsize=16)
            return fig

        # Group scores by dataset
        dataset_scores = defaultdict(list)

        for pred in predictions:
            dataset = pred['dataset_name']
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

        # Create box plots with custom styling
        bp = ax.boxplot(scores_list, patch_artist=True,
                       widths=0.2,  # Make boxes narrower
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2, color='white'))

        # Use more vibrant colors
        n_datasets = len(datasets)
        if n_datasets <= 3:
            # Use distinct, vibrant colors for few datasets
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:n_datasets]  # Blue, Orange, Green
        else:
            # Use generated colors
            colors = [f'C{i}' for i in range(n_datasets)]

        # Style the boxes with vibrant colors and better transparency
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)  # More opaque
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)

        # Customize the plot
        ax.set_xlabel('Dataset')
        ax.set_ylabel(f'{metric.upper()} Score')
        ax.set_title(f'{metric.upper()} Score Distribution by Dataset ({partition} partition)')
        ax.grid(True, alpha=0.3, axis='y')

        # Set dataset labels
        ax.set_xticks(range(1, len(datasets) + 1))
        ax.set_xticklabels(datasets)

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
            ax.text(i + 1, y_pos, f'n={n_scores}\nμ={mean_score:.3f}\nσ={std_score:.3f}',
                   ha='center', va='bottom', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_all_models_barplot(self, metric: str = 'rmse', partition: str = 'val',
                                figsize: Tuple[int, int] = (14, 8)) -> Figure:
        """
        Plot barplot showing all models with specified metric.

        Args:
            metric: Metric to display
            partition: Partition filter
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        predictions = self._get_enhanced_predictions(partition=partition)

        if not predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No predictions found', ha='center', va='center', fontsize=16)
            return fig

        # Group by model and take best score
        model_scores = defaultdict(lambda: {'best_score': float('inf') if metric not in ['r2'] else float('-inf'),
                                          'count': 0})

        higher_better = metric in ['r2', 'accuracy', 'f1', 'precision', 'recall']

        for pred in predictions:
            model = pred['model_name']
            score = pred['metrics'].get(metric, np.nan)

            if not np.isnan(score):
                current_best = model_scores[model]['best_score']
                if (higher_better and score > current_best) or (not higher_better and score < current_best):
                    model_scores[model]['best_score'] = score
                model_scores[model]['count'] += 1

        if not model_scores:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No valid {metric} scores found',
                   ha='center', va='center', fontsize=16)
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
        if scores:
            norm_scores = np.array(scores) / np.max(np.abs(scores))
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

    def plot_variable_heatmap(self, x_var: str, y_var: str, filters: Dict[str, Any] = {},
                              partition: str = 'val', metric: str = 'rmse', figsize: Tuple[int, int] = (12, 8),
                              normalize: bool = True, best_only: bool = True, display_n: bool = True,
                              score_partition: str = 'test', score_metric: str = '') -> Figure:
        """
        Plot heatmap showing performance by two variables from predictions.

        Args:
            filters: Dictionary of filters to apply to predictions (e.g., {"dataset": "regression", "partition": "test"})
            x_var: Variable for x-axis (e.g., "model_name", "preprocessings", "config_name")
            y_var: Variable for y-axis (e.g., "model_name", "preprocessings", "config_name")
            metric: Metric to display (e.g., 'rmse', 'r2', 'accuracy')
            figsize: Figure size
            normalize: Whether to normalize scores for better color comparison
            best_only: Whether to take best score per x_var/y_var combination (True) or average (False)

        Returns:
            matplotlib Figure

        Example:
            plot_variable_heatmap(predictions,
                                 {"dataset": "regression", "partition": "test"},
                                 "model_name", "preprocessings", 'rmse')
        """
        if score_metric == '':
            score_metric = metric
        filters['partition'] = partition  # Decide if filters or particition param takes precedence
        return self._create_variable_heatmap(filters, x_var, y_var, metric, figsize, normalize, best_only, display_n, score_partition, score_metric)

    def _create_variable_heatmap(self, filters: Dict[str, Any], x_var: str, y_var: str,
                                 metric: str, figsize: Tuple[int, int],
                                 normalize: bool, best_only: bool, display_n: bool = True,
                                 score_partition: str = 'test', score_metric: str = '') -> Figure:
        """Helper method to create variable heatmap (split for complexity)."""
        # Get predictions using the existing top_k method with filters
        try:
            # Use top_k with k=-1 to get all predictions matching filters
            if x_var == "partition" or y_var == "partition" or filters.get('partition') in ['all', 'ALL', 'All', '_all_']:
                filters['partition'] = '_all_'

            predictions = self.predictions.top_k(k=-1, metric=metric, aggregate_partitions=[score_partition], **filters)  # True only if score_partition and partition are different
        except Exception as e:
            print(f"⚠️ Error getting predictions: {e}")
            # Fallback to filter_predictions
            predictions = self.predictions.filter_predictions(**filters)

        if not predictions:
            fig, ax = plt.subplots(figsize=figsize)
            filter_str = ", ".join([f"{k}={v}" for k, v in filters.items()])
            ax.text(0.5, 0.5, f'No predictions found for filters: {filter_str}',
                    ha='center', va='center', fontsize=16)
            return fig

        # Group by x_var and y_var to aggregate scores
        var_scores = self._extract_scores_by_variables(predictions, x_var, y_var, metric)
        var_display_scores = None
        if (score_metric != metric or score_partition != filters['partition']) and filters['partition'] != '_all_':
            var_display_scores = self._extract_scores_by_variables(predictions, x_var, y_var, score_metric, score_partition)

        if not var_scores:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No valid {metric} scores found',
                    ha='center', va='center', fontsize=16)
            return fig

        # Create matrix and plot
        return self._plot_heatmap_matrix(var_scores, x_var, y_var, metric, filters,
                                         figsize, normalize, best_only, display_n, var_display_scores,
                                         score_partition, score_metric)

    def _extract_scores_by_variables(self, predictions: List[Dict], x_var: str, y_var: str,
                                     metric: str, score_partition: str = '') -> Dict:
        """Extract and group scores by x and y variables."""
        var_scores = defaultdict(lambda: defaultdict(list))

        for pred in predictions:
            x_val = pred.get(x_var, 'unknown')
            y_val = pred.get(y_var, 'unknown')

            # Get score - try different possible locations
            score = self._extract_metric_score(pred, metric, score_partition)

            if score is not None and not (isinstance(score, float) and np.isnan(score)):
                var_scores[y_val][x_val].append(score)

        return var_scores

    def _extract_metric_score(self, pred: Dict, metric: str, score_partition: str = '') -> Optional[float]:
        """Extract metric score from prediction dictionary."""
        # # Try different possible locations for the score
        # if isinstance(pred.get(metric), (int, float)):
        #     return float(pred[metric])
        # elif 'metrics' in pred and isinstance(pred['metrics'], dict):
        #     score = pred['metrics'].get(metric, np.nan)
        #     if isinstance(score, (int, float)) and not isinstance(score, dict):
        #         return float(score)
        # else:
        #     # Calculate on-the-fly using evaluator
        y_true = pred.get('y_true', [])
        y_pred = pred.get('y_pred', [])
        if score_partition:
            y_true = pred[score_partition]['y_true']
            y_pred = pred[score_partition]['y_pred']

        if len(y_true) > 0 and len(y_pred) > 0:
            try:
                from .evaluator import eval
                score = eval(np.array(y_true), np.array(y_pred), metric)
                return float(score)
            except Exception as e:
                print(f"⚠️ Error calculating {metric}: {e}")
        return None

    def _plot_heatmap_matrix(self, var_scores: Dict, x_var: str, y_var: str, metric: str,
                             filters: Dict, figsize: Tuple[int, int], normalize: bool,
                             best_only: bool, display_n: bool, var_display_scores: Optional[Dict] = None,
                             score_partition: str = '', score_metric: str = '') -> Figure:
        """Create the actual heatmap plot from scores matrix."""
        # Extract unique values with natural sorting
        y_labels = sorted(var_scores.keys(), key=self._natural_sort_key)
        x_labels = sorted(set(x for y_data in var_scores.values() for x in y_data.keys()), key=self._natural_sort_key)

        if x_var == 'partition':
            x_labels = ['train', 'val', 'test']
        if y_var == 'partition':
            y_labels = ['train', 'val', 'test']

        # Create matrix
        matrix = np.full((len(y_labels), len(x_labels)), np.nan)
        score_counts = np.zeros((len(y_labels), len(x_labels)))

        higher_better = metric in ['r2', 'accuracy', 'f1', 'precision', 'recall']

        for i, y_val in enumerate(y_labels):
            for j, x_val in enumerate(x_labels):
                scores = var_scores[y_val].get(x_val, [])
                display_scores = None if var_display_scores is None else var_display_scores[y_val].get(x_val, [])
                # print(y_val, x_val, "\n", scores, "\n", display_scores, "\n---")
                if scores:
                    score_counts[i, j] = len(scores)
                    # get index or min or max score
                    max_score_index = np.argmax(scores) if higher_better else np.argmin(scores)
                    # print(f"Scores for ({y_val}, {x_val}): {scores}, best index: {max_score_index}")
                    if best_only:
                        # print(f"Best score for ({y_val}, {x_val}): {scores[max_score_index]}")
                        # print(f"Display score for ({y_val}, {x_val}): {display_scores[max_score_index] if display_scores else 'N/A'}")
                        if display_scores is not None:
                            matrix[i, j] = display_scores[max_score_index]
                        else:
                            matrix[i, j] = scores[max_score_index]
                    else:
                        if display_scores is not None:
                            matrix[i, j] = np.mean(display_scores)
                        else:
                            matrix[i, j] = np.mean(scores)

        # Normalize scores if requested
        if normalize and not np.all(np.isnan(matrix)):
            matrix = self._normalize_matrix(matrix, higher_better)

        # Create the plot
        return self._render_heatmap_plot(matrix, score_counts, var_scores, x_labels, y_labels,
                                         x_var, y_var, metric, filters, figsize, normalize,
                                         best_only, higher_better, display_n, var_display_scores,
                                         score_partition, score_metric)

    def _normalize_matrix(self, matrix: np.ndarray, higher_better: bool) -> np.ndarray:
        """Normalize matrix values for better color comparison."""
        valid_scores = matrix[~np.isnan(matrix)]
        if len(valid_scores) > 0:
            min_score = np.min(valid_scores)
            max_score = np.max(valid_scores)
            if max_score != min_score:
                if not higher_better:
                    # For "lower is better" metrics, invert for color mapping
                    matrix = 1 - (matrix - min_score) / (max_score - min_score)
                else:
                    # Standard normalization for "higher is better" metrics
                    matrix = (matrix - min_score) / (max_score - min_score)
        return matrix

    def _render_heatmap_plot(self, matrix: np.ndarray, score_counts: np.ndarray,
                             var_scores: Dict, x_labels: List, y_labels: List,
                             x_var: str, y_var: str, metric: str, filters: Dict,
                             figsize: Tuple[int, int], normalize: bool, best_only: bool,
                             higher_better: bool, display_n: bool, var_display_scores: Optional[Dict] = None,
                             score_partition: str = '', score_metric: str = '') -> Figure:
        """Render the final heatmap plot."""
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Use a color map where better performance is greener
        cmap = 'RdYlGn'  # Red-Yellow-Green colormap

        # Create masked array to handle NaN values
        masked_matrix = np.ma.masked_invalid(matrix)

        im = ax.imshow(masked_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1 if normalize else None)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        if normalize:
            cbar.set_label(f'Normalized {metric.upper()} (1=best, 0=worst)')
        else:
            cbar.set_label(f'{metric.upper()} Score')

        # Set ticks and labels
        ax.set_xticks(range(len(x_labels)))
        ax.set_yticks(range(len(y_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(x_var.replace('_', ' ').title())
        ax.set_ylabel(y_var.replace('_', ' ').title())

        # Create title
        title = "Best " if best_only else "Average "
        partition = filters.get('partition', '')
        if partition in ['all', 'ALL', 'All', '_all_', '']:
            title += f"{score_metric.upper()}"
        else:
            if best_only:
                title += f"Model [{metric.upper()} in {partition}] - Score [{score_metric.upper()} in {score_partition}]"
            else:
                title += f"Score [{score_metric.upper()} in {score_partition}]"


        # if best_only:
        #     if partition in ['all', 'ALL', 'All', '_all_', '']:
        #     else:
        #         title += f" (best in {partition} partition)"

        # title = f'{metric.upper()} Performance Heatmap'
        # if 'partition' in filters and filters['partition'] not in ['all', 'ALL', 'All', '_all_', '']:
        #     title += f' - ({filters["partition"]})'
        # if normalize:
            # title += ' [Norm]'
        ax.set_title(title)

        # Add text annotations
        self._add_heatmap_annotations(ax, matrix, score_counts, var_scores, x_labels, y_labels,
                                      masked_matrix, normalize, best_only, higher_better, display_n, var_display_scores)

        plt.tight_layout()
        return fig

    def _add_heatmap_annotations(self, ax, matrix: np.ndarray, score_counts: np.ndarray,
                                 var_scores: Dict, x_labels: List, y_labels: List,
                                 masked_matrix, normalize: bool, best_only: bool,
                                 higher_better: bool, display_n: bool, var_display_scores: Optional[Dict] = None) -> None:
        """Add text annotations to heatmap cells."""
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                if not np.isnan(matrix[i, j]) and score_counts[i, j] > 0:
                    count_text = f'n={int(score_counts[i, j])}'
                    # Show original score if normalized
                    if normalize:
                        # Reconstruct original score for display
                        orig_scores = var_scores[y_labels[i]].get(x_labels[j], [])
                        disp_orig_scores = None if var_display_scores is None else var_display_scores[y_labels[i]].get(x_labels[j], [])
                        if orig_scores:
                            if best_only:
                                best_index = np.argmax(orig_scores) if higher_better else np.argmin(orig_scores)
                                if disp_orig_scores is not None:
                                    orig_score = disp_orig_scores[best_index]
                                else:
                                    orig_score = orig_scores[best_index]
                            else:
                                if disp_orig_scores is not None:
                                    orig_score = np.mean(disp_orig_scores)
                                else:
                                    orig_score = np.mean(orig_scores)
                            score_text = f'{orig_score:.3f}'
                        else:
                            score_text = 'N/A'
                    else:
                        score_text = f'{matrix[i, j]:.3f}'

                    # Combine score and count
                    if display_n:
                        text = f'{score_text}\n{count_text}'
                    else:
                        text = score_text
                    ax.text(j, i, text, ha='center', va='center', fontsize=8,
                            color='white' if masked_matrix[i, j] < 0.0 else 'black')

    def plot_variable_candlestick(self, filters: Dict[str, Any], variable: str,
                                  metric: str = 'rmse', figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Plot candlestick chart showing metric distribution by a single variable.

        Args:
            filters: Dictionary of filters to apply to predictions (e.g., {"dataset": "regression", "partition": "test"})
            variable: Variable to group by (e.g., "model_name", "preprocessings", "config_name")
            metric: Metric to analyze (e.g., 'rmse', 'r2', 'accuracy')
            figsize: Figure size

        Returns:
            matplotlib Figure

        Example:
            plot_variable_candlestick({"dataset": "regression", "partition": "test"},
                                    "model_name", 'rmse')
        """
        # Get predictions using the existing top_k method with filters
        try:
            # Use top_k with k=-1 to get all predictions matching filters
            predictions = self.predictions.top_k(k=-1, metric=metric, **filters)
        except Exception as e:
            print(f"⚠️ Error getting predictions: {e}")
            # Fallback to filter_predictions
            predictions = self.predictions.filter_predictions(**filters)

        if not predictions:
            fig, ax = plt.subplots(figsize=figsize)
            filter_str = ", ".join([f"{k}={v}" for k, v in filters.items()])
            ax.text(0.5, 0.5, f'No predictions found for filters: {filter_str}',
                    ha='center', va='center', fontsize=16)
            return fig

        # Group scores by variable
        variable_scores = defaultdict(list)

        for pred in predictions:
            var_val = pred.get(variable, 'unknown')

            # Get score - try different possible locations
            score = self._extract_metric_score(pred, metric)

            if score is not None and not (isinstance(score, float) and np.isnan(score)):
                variable_scores[var_val].append(score)

        if not variable_scores:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No valid {metric} scores found',
                    ha='center', va='center', fontsize=16)
            return fig

        # Calculate statistics for each variable value
        var_labels = []
        means = []
        mins = []
        maxs = []
        q25s = []
        q75s = []
        counts = []

        # First collect data with natural sorting of variable values
        sorted_var_vals = sorted(variable_scores.keys(), key=self._natural_sort_key)
        higher_better = metric in ['r2', 'accuracy', 'f1', 'precision', 'recall']
        for var_val in sorted_var_vals:
            scores = variable_scores[var_val]
            if scores:  # Only include if we have scores
                var_labels.append(str(var_val))
                means.append(np.mean(scores))
                mins.append(np.min(scores))
                maxs.append(np.max(scores))
                q25s.append(np.percentile(scores, 25))
                q75s.append(np.percentile(scores, 75))
                counts.append(len(scores))

        if not var_labels:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No valid data for variable {variable}',
                    ha='center', va='center', fontsize=16)
            return fig

        # Create candlestick plot
        fig, ax = plt.subplots(figsize=figsize)

        # Use the same color map as heatmap (RdYlGn - Red-Yellow-Green)
        # Normalize mean values for color mapping
        if len(means) > 1:
            mean_array = np.array(means)
            if higher_better:
                # For higher-is-better metrics, higher values get greener
                norm_means = (mean_array - np.min(mean_array)) / (np.max(mean_array) - np.min(mean_array))
            else:
                # For lower-is-better metrics, lower values get greener
                norm_means = 1 - (mean_array - np.min(mean_array)) / (np.max(mean_array) - np.min(mean_array))
        else:
            norm_means = np.array([0.5])  # Default to middle color for single value

        # Get colors from RdYlGn colormap
        import matplotlib.cm as cm
        from matplotlib.patches import Rectangle
        cmap = cm.get_cmap('RdYlGn')
        colors = [cmap(norm_val) for norm_val in norm_means]

        for i, _ in enumerate(var_labels):
            # High-low line (whiskers)
            ax.plot([i, i], [mins[i], maxs[i]], color='black', linewidth=1.0, alpha=0.8)

            # Rectangle for Q25-Q75 (box) - made narrower
            box_height = q75s[i] - q25s[i]
            box = Rectangle((i - 0.15, q25s[i]), 0.3, box_height,
                            fill=True, color=colors[i], alpha=0.8,
                            edgecolor='black', linewidth=1.0)
            ax.add_patch(box)

            # Mean line (darker line in the middle) - made narrower
            ax.plot([i - 0.15, i + 0.15], [means[i], means[i]],
                    color='darkred', linewidth=2.0, alpha=0.9)

            # Add count as text below
            ax.text(i, mins[i] - (maxs[i] - mins[i]) * 0.05, f'n={counts[i]}',
                    ha='center', va='top', fontsize=9, alpha=0.8)        # Customize the plot
        ax.set_xticks(range(len(var_labels)))
        ax.set_xticklabels(var_labels, rotation=45, ha='right')
        ax.set_ylabel(f'{metric.upper()} Score')

        # Create title
        filter_str = ", ".join([f"{k}={v}" for k, v in filters.items()])
        title = f'{metric.upper()} Distribution by {variable.replace("_", " ").title()}'
        if filter_str:
            title += f' ({filter_str})'
        ax.set_title(title)

        ax.grid(True, alpha=0.3, axis='y')

        # Add legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor=cmap(0.7), edgecolor='black', alpha=0.8, label='Q25-Q75 (IQR)'),
            Line2D([0], [0], color='darkred', linewidth=2.0, alpha=0.9, label='Mean'),
            Line2D([0], [0], color='black', linewidth=1.0, alpha=0.8, label='Min-Max Range')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        return fig
