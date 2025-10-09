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
        if partition == 'all':
            filters.pop('partition')  # Remove to include all partitions

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

        When partition is '' or 'all', displays predictions from all partitions (train, val, test)
        with different colors for each partition in the same scatter plot.

        Args:
            k: Number of top models to show
            metric: Metric for ranking
            partition: Partition type ('test', 'val', 'train', '' or 'all' for all partitions)
            dataset_name: Dataset filter
            figsize: Figure size

        Returns:
            matplotlib Figure
        """

        if partition == '':
            partition = 'all'

        # Define partition colors
        partition_colors = {
            'train': '#1f77b4',  # Blue
            'val': '#ff7f0e',    # Orange
            'test': '#2ca02c'    # Green
        }

        # Check if we need to aggregate multiple partitions
        show_all_partitions = partition in ['all', 'ALL', 'All', '_all_']

        if show_all_partitions:
            # Get top k models based on aggregated performance across all partitions
            top_predictions = self.get_top_k(k, metric, partition, dataset_name)
            if not top_predictions:
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, f'No predictions found',
                       ha='center', va='center', fontsize=16)
                return fig

            # For each top model, collect predictions from all partitions
            model_predictions_by_partition = []
            for pred in top_predictions:
                model_id = pred.get('id') or pred.get('model_name')

                # Get predictions for this model from each partition
                partition_data = {}
                for part in ['train', 'val', 'test']:
                    part_preds = self.predictions.filter_predictions(
                        partition=part,
                        id=model_id
                    )
                    if part_preds:
                        # Take the first matching prediction (should be unique per partition)
                        partition_data[part] = part_preds[0]

                if partition_data:
                    model_predictions_by_partition.append({
                        'model_name': pred['model_name'],
                        'model_id': model_id,
                        'partitions': partition_data,
                        'aggregate_score': pred.get(metric, 'N/A')
                    })

            if not model_predictions_by_partition:
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, f'No multi-partition predictions found',
                       ha='center', va='center', fontsize=16)
                return fig

            n_plots = len(model_predictions_by_partition)
        else:
            # Single partition mode - original behavior
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
            axes = [axes]
        elif rows == 1:
            axes = [axes]

        if show_all_partitions:
            # Plot with multiple partitions per model
            for i, model_data in enumerate(model_predictions_by_partition):
                ax_scatter = axes[i][0]
                ax_resid = axes[i][1]

                # Collect data from all partitions
                all_y_true = []
                all_y_pred = []
                all_colors = []
                partition_labels = []

                for part_name in ['train', 'val', 'test']:
                    if part_name in model_data['partitions']:
                        pred = model_data['partitions'][part_name]
                        y_true = np.asarray(pred['y_true']).flatten()
                        y_pred = np.asarray(pred['y_pred']).flatten()

                        # Check size mismatch
                        if len(y_true) != len(y_pred):
                            min_len = min(len(y_true), len(y_pred))
                            y_true = y_true[:min_len]
                            y_pred = y_pred[:min_len]

                        # Scatter plot with partition-specific color
                        ax_scatter.scatter(y_true, y_pred, alpha=0.6, s=20,
                                         color=partition_colors[part_name],
                                         label=part_name.capitalize())

                        # Collect for residuals
                        all_y_true.extend(y_true)
                        all_y_pred.extend(y_pred)
                        all_colors.extend([partition_colors[part_name]] * len(y_true))
                        partition_labels.append(part_name)

                # Add diagonal line
                if all_y_true:
                    min_val = min(all_y_true + all_y_pred)
                    max_val = max(all_y_true + all_y_pred)
                    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                    ax_scatter.set_xlabel('True Values')
                    ax_scatter.set_ylabel('Predicted Values')
                    ax_scatter.legend(loc='best')

                    model_display = model_data['model_name'] + " [" + model_data['model_id'] + "]"
                    ax_scatter.set_title(f'{model_display}')
                    ax_scatter.grid(True, alpha=0.3)

                    # Residuals plot with colors
                    all_y_true = np.array(all_y_true)
                    all_y_pred = np.array(all_y_pred)
                    residuals = all_y_true - all_y_pred

                    ax_resid.scatter(all_y_pred, residuals, alpha=0.6, s=20, c=all_colors)
                    ax_resid.axhline(y=0, color='r', linestyle='--', alpha=0.8)
                    ax_resid.set_xlabel('Predicted Values')
                    ax_resid.set_ylabel('Residuals')

                    score_value = model_data['aggregate_score']
                    ax_resid.set_title(f'Residuals - {metric.upper()}: {score_value:.4f}'
                                     if isinstance(score_value, (int, float)) else f'Residuals - {metric.upper()}: {score_value}')
                    ax_resid.grid(True, alpha=0.3)
        else:
            # Original single-partition plotting
            for i, pred in enumerate(top_predictions):
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
                title = f'{model_display} - {partition}'
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
                                   rank_partition: str = 'val', display_partition: str = 'test',
                                   dataset_name: Optional[str] = None,
                                   figsize: Tuple[int, int] = (16, 10), **filters) -> Figure:
        """
        Plot confusion matrices for top K classification models.

        Models are ranked by the metric on rank_partition, then confusion matrices
        are displayed using predictions from display_partition.

        Args:
            k: Number of top models to show
            metric: Metric for ranking (default: 'accuracy')
            rank_partition: Partition used for ranking models (default: 'val')
            display_partition: Partition to display confusion matrix from (default: 'test')
            dataset_name: Dataset filter (optional)
            figsize: Figure size
            **filters: Additional filters (e.g., config_name="config1")

        Returns:
            matplotlib Figure
        """
        # Build filters
        if dataset_name:
            filters['dataset_name'] = dataset_name

        # Use top() method: rank on rank_partition, display from display_partition
        top_predictions = self.predictions.top(
            n=k,
            rank_metric=metric,
            rank_partition=rank_partition,
            display_metrics=[metric],
            display_partition=display_partition,
            **filters
        )

        if not top_predictions:
            fig, ax = plt.subplots(figsize=figsize)
            filter_desc = ', '.join(f"{k}={v}" for k, v in filters.items()) if filters else 'none'
            ax.text(0.5, 0.5, f'No predictions found\nFilters: {filter_desc}',
                   ha='center', va='center', fontsize=14)
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
            # Get metric value - the metric should be available from the display_partition
            score_value = pred.get(metric)
            if score_value is None:
                score_value = 'N/A'
            score_str = f'{score_value:.4f}' if isinstance(score_value, (int, float)) else str(score_value)

            # Create title showing both partitions if different
            title = f'{model_display}\n{metric.upper()}: {score_str}'
            if rank_partition != display_partition:
                title += f' [{display_partition}]'
            ax.set_title(title)

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
            metric: Metric to plot (default: 'rmse')
            dataset_name: Dataset filter (optional)
            partition: Partition to display scores from (default: 'test')
            bins: Number of histogram bins
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        from nirs4all.dataset.predictions import PredictionResult

        filters = {}
        if dataset_name:
            filters['dataset_name'] = dataset_name
        if partition:
            filters['partition'] = partition
        else:
            filters['partition'] = 'test'  # Default to test partition

        # Use top_k to get all predictions for the specified partition
        predictions = self.predictions.top_k(
            k=-1,  # Get all predictions
            metric=metric,
            **filters
        )

        if not predictions:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No predictions found', ha='center', va='center', fontsize=16)
            return fig

        # Extract scores - use eval_score() to get the metric value
        scores = []
        for pred in predictions:
            # Ensure pred is a PredictionResult object
            if not isinstance(pred, PredictionResult):
                pred = PredictionResult(pred)

            try:
                # Use eval_score to compute the metric
                pred_scores = pred.eval_score(metrics=[metric])
                score = pred_scores.get(metric)
                if score is not None and not np.isnan(score):
                    scores.append(float(score))
            except Exception as e:
                # Fallback: try to get stored score
                if metric in pred:
                    score = pred[metric]
                    if score is not None and not np.isnan(score):
                        scores.append(float(score))

        if not scores:
            fig, ax = plt.subplots(figsize=figsize)
            partition_str = partition if partition else 'test'
            ax.text(0.5, 0.5, f'No valid {metric} scores found for partition "{partition_str}"',
                    ha='center', va='center', fontsize=16)
            return fig


        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(scores, bins=bins, alpha=0.7, edgecolor='black', color='#35B779')
        ax.set_xlabel(f'{metric.upper()} Score')
        ax.set_ylabel('Frequency')

        partition_label = partition if partition else 'test'
        ax.set_title(f'Distribution of {metric.upper()} Scores\n({len(scores)} predictions, partition: {partition_label})')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = float(np.mean(scores))
        median_val = float(np.median(scores))
        std_val = float(np.std(scores))
        min_val = float(min(scores))
        max_val = float(max(scores))

        ax.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='g', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')

        # Add text box with statistics
        stats_text = f'n={len(scores)}\nμ={mean_val:.4f}\nσ={std_val:.4f}\nmin={min_val:.4f}\nmax={max_val:.4f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

        ax.legend()

        return fig

    # def plot_performance_heatmap(self, x_axis: str = 'model_name', y_axis: str = 'dataset_name',
    #                              metric: str = 'rmse', partition: str = '',
    #                              figsize: Tuple[int, int] = (12, 8)) -> Figure:
    #     """
    #     Plot heatmap of performance by model and dataset.

    #     Args:
    #         x_axis: X-axis dimension ('model_name' or 'dataset_name')
    #         y_axis: Y-axis dimension ('dataset_name' or 'model_name')
    #         metric: Metric to display
    #         partition: Partition filter
    #         figsize: Figure size

    #     Returns:
    #         matplotlib Figure
    #     """
    #     predictions = self.get_top_k(-1, metric, partition)

    #     if not predictions:
    #         fig, ax = plt.subplots(figsize=figsize)
    #         ax.text(0.5, 0.5, 'No predictions found', ha='center', va='center', fontsize=16)
    #         return fig

    #     # Group by x and y dimensions
    #     grouped_data = defaultdict(lambda: defaultdict(list))

    #     for pred in predictions:
    #         x_val = pred.get(x_axis, 'unknown')
    #         y_val = pred.get(y_axis, 'unknown')
    #         score = pred.get(metric, np.nan)

    #         if not np.isnan(score):
    #             grouped_data[y_val][x_val].append(score)

    #     # Extract unique values
    #     y_labels = sorted(grouped_data.keys())
    #     x_labels = sorted(set(x for y_data in grouped_data.values() for x in y_data.keys()))

    #     # Create matrix
    #     matrix = np.full((len(y_labels), len(x_labels)), np.nan)

    #     for i, y_val in enumerate(y_labels):
    #         for j, x_val in enumerate(x_labels):
    #             scores = grouped_data[y_val].get(x_val, [])
    #             if scores:
    #                 # Take best score (lowest for rmse, highest for r2)
    #                 higher_better = metric in ['r2', 'accuracy', 'f1', 'precision', 'recall']
    #                 matrix[i, j] = max(scores) if higher_better else min(scores)

    #     # Create heatmap
    #     fig, ax = plt.subplots(figsize=figsize)

    #     if np.any(~np.isnan(matrix)):
    #         im = ax.imshow(matrix, cmap='viridis', aspect='auto')
    #         plt.colorbar(im, ax=ax, label=metric.upper())

    #         # Add values
    #         for i in range(len(y_labels)):
    #             for j in range(len(x_labels)):
    #                 if not np.isnan(matrix[i, j]):
    #                     ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
    #                            color='white' if matrix[i, j] > np.nanmean(matrix) else 'black',
    #                            fontsize=8)

    #     ax.set_xticks(range(len(x_labels)))
    #     ax.set_yticks(range(len(y_labels)))
    #     ax.set_xticklabels(x_labels, rotation=45, ha='right')
    #     ax.set_yticklabels(y_labels)
    #     ax.set_xlabel(x_axis.replace('_', ' ').title())
    #     ax.set_ylabel(y_axis.replace('_', ' ').title())
    #     ax.set_title(f'{metric.upper()} Performance Heatmap')

    #     return fig

    # def plot_candlestick_models(self, metric: str = 'rmse', partition: str = '',
    #                             figsize: Tuple[int, int] = (12, 8)) -> Figure:
    #     """
    #     Plot candlestick chart showing avg/variance per model.

    #     Args:
    #         metric: Metric to analyze
    #         partition: Partition filter
    #         figsize: Figure size

    #     Returns:
    #         matplotlib Figure
    #     """
    #     predictions = self.predictions.top_k(-1, metric, partition=partition)

    #     if not predictions:
    #         fig, ax = plt.subplots(figsize=figsize)
    #         ax.text(0.5, 0.5, 'No predictions found', ha='center', va='center', fontsize=16)
    #         return fig

    #     # Group by model
    #     model_stats = defaultdict(list)

    #     for pred in predictions:
    #         model = pred['model_classname']
    #         score = pred.get(metric, np.nan)
    #         if not np.isnan(score):
    #             model_stats[model].append(score)

    #     if not model_stats:
    #         fig, ax = plt.subplots(figsize=figsize)
    #         ax.text(0.5, 0.5, f'No valid {metric} scores found',
    #                ha='center', va='center', fontsize=16)
    #         return fig

    #     # Calculate stats for each model
    #     models = []
    #     means = []
    #     mins = []
    #     maxs = []
    #     q25s = []
    #     q75s = []

    #     for model, scores in model_stats.items():
    #         models.append(model)
    #         means.append(np.mean(scores))
    #         mins.append(np.min(scores))
    #         maxs.append(np.max(scores))
    #         q25s.append(np.percentile(scores, 25))
    #         q75s.append(np.percentile(scores, 75))

    #     # Sort by mean performance
    #     higher_better = metric in ['r2', 'accuracy', 'f1', 'precision', 'recall']
    #     sort_indices = np.argsort(means)
    #     if higher_better:
    #         sort_indices = sort_indices[::-1]

    #     models = [models[i] for i in sort_indices]
    #     means = [means[i] for i in sort_indices]
    #     mins = [mins[i] for i in sort_indices]
    #     maxs = [maxs[i] for i in sort_indices]
    #     q25s = [q25s[i] for i in sort_indices]
    #     q75s = [q75s[i] for i in sort_indices]

    #     # Create candlestick plot
    #     fig, ax = plt.subplots(figsize=figsize)

    #     for i, model in enumerate(models):
    #         # High-low line
    #         ax.plot([i, i], [mins[i], maxs[i]], color='black', linewidth=1)
    #         # Rectangle for Q25-Q75
    #         ax.add_patch(plt.Rectangle((i-0.3, q25s[i]), 0.6, q75s[i]-q25s[i],
    #                                  fill=True, color='lightblue', alpha=0.7))
    #         # Mean line
    #         ax.plot([i-0.3, i+0.3], [means[i], means[i]], color='red', linewidth=2)

    #     ax.set_xticks(range(len(models)))
    #     ax.set_xticklabels(models, rotation=45, ha='right')
    #     ax.set_ylabel(f'{metric.upper()} Score')
    #     ax.set_title(f'{metric.upper()} Distribution by Model (Candlestick)')
    #     ax.grid(True, alpha=0.3)

    #     return fig

    # def plot_performance_matrix(self, metric: str = 'rmse', partition: str = '', separate_avg: bool = False,
    #                            normalize: bool = True, figsize: Tuple[int, int] = (14, 10)) -> Figure:
    #     """
    #     Plot matrix showing best performance by model type for each dataset.

    #     Args:
    #         metric: Metric to display (default: 'rmse')
    #         partition: Partition type to consider ('test', 'val', 'train')
    #         normalize: Whether to normalize scores for better color comparison
    #         figsize: Figure size

    #     Returns:
    #         matplotlib Figure
    #     """
    #     # Get all predictions for the specified partition type
    #     predictions = self.predictions.top_k(-1, metric, partition=partition)

    #     if not predictions:
    #         fig, ax = plt.subplots(figsize=figsize)
    #         ax.text(0.5, 0.5, f'No {partition} predictions found', ha='center', va='center', fontsize=16)
    #         return fig

    #     # Group by dataset and model to find best performance
    #     dataset_model_scores = defaultdict(lambda: defaultdict(list))

    #     for pred in predictions:
    #         dataset = pred['dataset_name']
    #         if separate_avg:
    #             model = pred['model_name']
    #         else:
    #             model = pred['model_classname']
    #         score = pred.get(metric, np.nan)

    #         if not np.isnan(score):
    #             dataset_model_scores[dataset][model].append(score)

    #     if not dataset_model_scores:
    #         fig, ax = plt.subplots(figsize=figsize)
    #         ax.text(0.5, 0.5, f'No valid {metric} scores found', ha='center', va='center', fontsize=16)
    #         return fig

    #     # Extract unique datasets and models
    #     datasets = sorted(dataset_model_scores.keys())
    #     all_models = set()
    #     for dataset_data in dataset_model_scores.values():
    #         all_models.update(dataset_data.keys())
    #     models = sorted(all_models)

    #     # Create matrix with best scores
    #     matrix = np.full((len(datasets), len(models)), np.nan)
    #     best_scores = {}  # Store best scores for each dataset-model combination

    #     higher_better = metric in ['r2', 'accuracy', 'f1', 'precision', 'recall']

    #     for i, dataset in enumerate(datasets):
    #         for j, model in enumerate(models):
    #             scores = dataset_model_scores[dataset].get(model, [])
    #             if scores:
    #                 # Get best score (lowest for rmse/mse/mae, highest for r2/accuracy)
    #                 best_score = max(scores) if higher_better else min(scores)
    #                 matrix[i, j] = best_score
    #                 best_scores[(dataset, model)] = best_score

    #     # Normalize scores if requested
    #     if normalize and not np.all(np.isnan(matrix)):
    #         # For RMSE and similar metrics (lower is better), we want to invert for color mapping
    #         if not higher_better:
    #             # Normalize inversely for "lower is better" metrics
    #             valid_scores = matrix[~np.isnan(matrix)]
    #             if len(valid_scores) > 0:
    #                 min_score = np.min(valid_scores)
    #                 max_score = np.max(valid_scores)
    #                 if max_score != min_score:
    #                     # Invert normalization: best (lowest) scores become 1, worst (highest) become 0
    #                     matrix_norm = np.full_like(matrix, np.nan)
    #                     valid_mask = ~np.isnan(matrix)
    #                     matrix_norm[valid_mask] = 1 - (matrix[valid_mask] - min_score) / (max_score - min_score)
    #                     matrix = matrix_norm
    #         else:
    #             # Standard normalization for "higher is better" metrics
    #             valid_scores = matrix[~np.isnan(matrix)]
    #             if len(valid_scores) > 0:
    #                 min_score = np.min(valid_scores)
    #                 max_score = np.max(valid_scores)
    #                 if max_score != min_score:
    #                     matrix_norm = np.full_like(matrix, np.nan)
    #                     valid_mask = ~np.isnan(matrix)
    #                     matrix_norm[valid_mask] = (matrix[valid_mask] - min_score) / (max_score - min_score)
    #                     matrix = matrix_norm

    #     # Create the plot
    #     fig, ax = plt.subplots(figsize=figsize)

    #     # Use a color map where better performance is greener
    #     cmap = 'RdYlGn'  # Red-Yellow-Green colormap

    #     # Create masked array to handle NaN values
    #     masked_matrix = np.ma.masked_invalid(matrix)

    #     im = ax.imshow(masked_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1 if normalize else None)

    #     # Add colorbar
    #     cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    #     if normalize:
    #         if higher_better:
    #             cbar.set_label(f'Normalized {metric.upper()} (1=best, 0=worst)')
    #         else:
    #             cbar.set_label(f'Normalized {metric.upper()} (1=best, 0=worst)')
    #     else:
    #         cbar.set_label(f'{metric.upper()} Score')

    #     # Set ticks and labels
    #     ax.set_xticks(range(len(models)))
    #     ax.set_yticks(range(len(datasets)))
    #     ax.set_xticklabels(models, rotation=45, ha='right')
    #     ax.set_yticklabels(datasets)
    #     ax.set_xlabel('Model Type')
    #     ax.set_ylabel('Dataset')

    #     title = f'Best {metric.upper()} Performance Matrix'
    #     if normalize:
    #         title += ' (Normalized)'
    #     ax.set_title(title)

    #     # Add text annotations with actual scores
    #     for i in range(len(datasets)):
    #         for j in range(len(models)):
    #             if not np.isnan(matrix[i, j]):
    #                 # Get original score for annotation
    #                 original_score = best_scores.get((datasets[i], models[j]), matrix[i, j])

    #                 # Choose text color based on background
    #                 if normalize:
    #                     text_color = 'white' if matrix[i, j] < 0.5 else 'black'
    #                 else:
    #                     text_color = 'white' if matrix[i, j] > np.nanmean(matrix) else 'black'

    #                 ax.text(j, i, f'{original_score:.3f}',
    #                        ha='center', va='center', color=text_color, fontsize=9, weight='bold')

    #     plt.tight_layout()
    #     return fig

    # def plot_score_boxplots_by_dataset(self, metric: str = 'rmse', partition: str = 'val',
    #                                   figsize: Tuple[int, int] = (14, 8)) -> Figure:
    #     """
    #     Plot box plots showing score distributions for each dataset.

    #     Args:
    #         metric: Metric to display (default: 'rmse')
    #         partition: Partition type to consider ('test', 'val', 'train')
    #         figsize: Figure size

    #     Returns:
    #         matplotlib Figure
    #     """
    #     # Get all predictions for the specified partition type
    #     predictions = self._get_enhanced_predictions(partition=partition)

    #     if not predictions:
    #         fig, ax = plt.subplots(figsize=figsize)
    #         ax.text(0.5, 0.5, f'No {partition} predictions found', ha='center', va='center', fontsize=16)
    #         return fig

    #     # Group scores by dataset
    #     dataset_scores = defaultdict(list)

    #     for pred in predictions:
    #         dataset = pred['dataset_name']
    #         score = pred['metrics'].get(metric, np.nan)

    #         if not np.isnan(score):
    #             dataset_scores[dataset].append(score)

    #     if not dataset_scores:
    #         fig, ax = plt.subplots(figsize=figsize)
    #         ax.text(0.5, 0.5, f'No valid {metric} scores found', ha='center', va='center', fontsize=16)
    #         return fig

    #     # Prepare data for box plots
    #     datasets = sorted(dataset_scores.keys())
    #     scores_list = [dataset_scores[dataset] for dataset in datasets]

    #     # Create the plot
    #     fig, ax = plt.subplots(figsize=figsize)

    #     # Create box plots with custom styling
    #     bp = ax.boxplot(scores_list, patch_artist=True,
    #                    widths=0.2,  # Make boxes narrower
    #                    boxprops=dict(linewidth=1.5),
    #                    whiskerprops=dict(linewidth=1.5),
    #                    capprops=dict(linewidth=1.5),
    #                    medianprops=dict(linewidth=2, color='white'))

    #     # Use more vibrant colors
    #     n_datasets = len(datasets)
    #     if n_datasets <= 3:
    #         # Use distinct, vibrant colors for few datasets
    #         colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:n_datasets]  # Blue, Orange, Green
    #     else:
    #         # Use generated colors
    #         colors = [f'C{i}' for i in range(n_datasets)]

    #     # Style the boxes with vibrant colors and better transparency
    #     for patch, color in zip(bp['boxes'], colors):
    #         patch.set_facecolor(color)
    #         patch.set_alpha(0.8)  # More opaque
    #         patch.set_edgecolor('black')
    #         patch.set_linewidth(1.2)

    #     # Customize the plot
    #     ax.set_xlabel('Dataset')
    #     ax.set_ylabel(f'{metric.upper()} Score')
    #     ax.set_title(f'{metric.upper()} Score Distribution by Dataset ({partition} partition)')
    #     ax.grid(True, alpha=0.3, axis='y')

    #     # Set dataset labels
    #     ax.set_xticks(range(1, len(datasets) + 1))
    #     ax.set_xticklabels(datasets)

    #     # Rotate x-axis labels if needed
    #     if len(datasets) > 5:
    #         plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    #     # Add statistics as text
    #     for i, (dataset, scores) in enumerate(zip(datasets, scores_list)):
    #         mean_score = np.mean(scores)
    #         median_score = np.median(scores)
    #         std_score = np.std(scores)
    #         n_scores = len(scores)

    #         # Add text above each box plot
    #         y_pos = max(scores) + (max(max(s) for s in scores_list) - min(min(s) for s in scores_list)) * 0.05
    #         ax.text(i + 1, y_pos, f'n={n_scores}\nμ={mean_score:.3f}\nσ={std_score:.3f}',
    #                ha='center', va='bottom', fontsize=9,
    #                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    #     plt.tight_layout()
    #     return fig

    # def plot_all_models_barplot(self, metric: str = 'rmse', partition: str = 'val',
    #                             figsize: Tuple[int, int] = (14, 8)) -> Figure:
    #     """
    #     Plot barplot showing all models with specified metric.

    #     Args:
    #         metric: Metric to display
    #         partition: Partition filter
    #         figsize: Figure size

    #     Returns:
    #         matplotlib Figure
    #     """
    #     predictions = self._get_enhanced_predictions(partition=partition)

    #     if not predictions:
    #         fig, ax = plt.subplots(figsize=figsize)
    #         ax.text(0.5, 0.5, 'No predictions found', ha='center', va='center', fontsize=16)
    #         return fig

    #     # Group by model and take best score
    #     model_scores = defaultdict(lambda: {'best_score': float('inf') if metric not in ['r2'] else float('-inf'),
    #                                       'count': 0})

    #     higher_better = metric in ['r2', 'accuracy', 'f1', 'precision', 'recall']

    #     for pred in predictions:
    #         model = pred['model_name']
    #         score = pred['metrics'].get(metric, np.nan)

    #         if not np.isnan(score):
    #             current_best = model_scores[model]['best_score']
    #             if (higher_better and score > current_best) or (not higher_better and score < current_best):
    #                 model_scores[model]['best_score'] = score
    #             model_scores[model]['count'] += 1

    #     if not model_scores:
    #         fig, ax = plt.subplots(figsize=figsize)
    #         ax.text(0.5, 0.5, f'No valid {metric} scores found',
    #                ha='center', va='center', fontsize=16)
    #         return fig

    #     # Prepare data for plotting
    #     models = []
    #     scores = []
    #     counts = []

    #     for model, data in model_scores.items():
    #         models.append(model)
    #         scores.append(data['best_score'])
    #         counts.append(data['count'])

    #     # Sort by score
    #     sort_indices = np.argsort(scores)
    #     if higher_better:
    #         sort_indices = sort_indices[::-1]

    #     models = [models[i] for i in sort_indices]
    #     scores = [scores[i] for i in sort_indices]
    #     counts = [counts[i] for i in sort_indices]

    #     # Create bar plot
    #     fig, ax = plt.subplots(figsize=figsize)

    #     bars = ax.bar(range(len(models)), scores, alpha=0.7, edgecolor='black')

    #     # Color bars based on performance
    #     if scores:
    #         norm_scores = np.array(scores) / np.max(np.abs(scores))
    #         colors = ['red' if s < 0.5 else 'green' for s in norm_scores]

    #         for bar, color in zip(bars, colors):
    #             bar.set_color(color)

    #     # Add value labels
    #     for i, (bar, score, count) in enumerate(zip(bars, scores, counts)):
    #         ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
    #                 f'{score:.3f}\n({count} runs)', ha='center', va='bottom', fontsize=8)

    #     ax.set_xticks(range(len(models)))
    #     ax.set_xticklabels(models, rotation=45, ha='right')
    #     ax.set_ylabel(f'Best {metric.upper()} Score')
    #     ax.set_title(f'Best {metric.upper()} by Model')
    #     ax.grid(True, alpha=0.3, axis='y')

    #     return fig

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
        x_labels = [x if len(x) <= 25 else x[:22] + '...' for x in x_labels]
        y_labels = [y if len(y) <= 25 else y[:22] + '...' for y in y_labels]
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

    def plot_heatmap_v2(self, x_var: str, y_var: str, rank_metric: str = 'rmse', rank_partition: str = 'val', display_metric: str = '',
                        display_partition: str = 'test', figsize: Tuple[int, int] = (12, 8), normalize: bool = True,
                        aggregation: str = 'best', show_counts: bool = True, **filters) -> Figure:
        """
        Plot heatmap showing performance across two variables with flexible ranking and display.

        This is a cleaner, faster version that uses optimized data retrieval.
        Models are ranked by rank_metric on rank_partition, then display_metric scores from
        display_partition are shown in the heatmap.

        Args:
            x_var: Variable for x-axis (e.g., "model_name", "preprocessings", "config_name")
            y_var: Variable for y-axis (e.g., "model_name", "preprocessings", "config_name")
            rank_metric: Metric used to rank/select best models (default: 'rmse')
            rank_partition: Partition used for ranking models (default: 'val')
            display_metric: Metric to display in heatmap (default: same as rank_metric)
            display_partition: Partition to display scores from (default: 'test')
            figsize: Figure size tuple
            normalize: Whether to normalize scores to [0,1] for better color comparison
            aggregation: How to aggregate multiple scores per cell: 'best', 'mean', 'median'
            show_counts: Whether to show sample counts in cells
            **filters: Additional filters (e.g., dataset_name="mydata", config_name="config1")

        Returns:
            matplotlib Figure

        Example:
            # Rank on val RMSE, display test RMSE
            fig = analyzer.plot_heatmap_v2("model_name", "preprocessings")

            # Rank on val accuracy, display test F1
            fig = analyzer.plot_heatmap_v2("model_name", "dataset_name",
                                          rank_metric='accuracy',
                                          display_metric='f1')
        """
        # Default display_metric to rank_metric if not specified
        if not display_metric:
            display_metric = rank_metric

        # Determine if metrics are "higher is better"
        rank_higher_better = rank_metric.lower() in ['r2', 'accuracy', 'f1', 'precision', 'recall', 'auc']
        display_higher_better = display_metric.lower() in ['r2', 'accuracy', 'f1', 'precision', 'recall', 'auc']

        # Strategy: Use fast top_k() calls and merge by model identity when needed
        # 1. Get ranking from rank_partition
        rank_filters = {k: v for k, v in filters.items() if k != 'partition'}
        rank_filters['partition'] = rank_partition

        # Get all predictions from rank partition (fast - uses stored scores)
        rank_predictions = self.predictions.top_k(k=-1, metric=rank_metric, ascending=(not rank_higher_better), **rank_filters)

        if not rank_predictions:
            return self._create_empty_heatmap_figure(figsize, filters, "No predictions found")

        # 2. If rank_partition == display_partition and metrics match, we already have the data
        if rank_partition == display_partition and rank_metric == display_metric:
            all_predictions = rank_predictions
        else:
            # Need to get scores from display partition
            # Optimization: Build lookup of what model identities we need
            unique_models = {}  # identity_key -> rank_pred
            for pred in rank_predictions:
                identity_key = (
                    pred.get('config_name', ''),
                    pred.get('step_idx', 0),
                    pred.get('model_name', ''),
                    pred.get('fold_id', ''),
                    pred.get('op_counter', 0)
                )
                unique_models[identity_key] = pred

            # Get display predictions using top_k with display partition
            display_filters = {k: v for k, v in filters.items() if k != 'partition'}
            display_filters['partition'] = display_partition

            # Use top_k for display partition (it's faster than filter_predictions)
            display_predictions = self.predictions.top_k(
                k=-1,
                metric=display_metric if display_metric == rank_metric else "",  # Use same metric if possible for speed
                ascending=(not display_higher_better),
                **display_filters
            )

            # Build lookup for display scores indexed by model identity
            display_lookup = {}
            for pred in display_predictions:
                identity_key = (
                    pred.get('config_name', ''),
                    pred.get('step_idx', 0),
                    pred.get('model_name', ''),
                    pred.get('fold_id', ''),
                    pred.get('op_counter', 0)
                )

                # Only keep predictions for models we're interested in
                if identity_key not in unique_models:
                    continue

                # Get the display score
                display_score_field = f'{display_partition}_score'
                score = pred.get(display_score_field)

                # If score is None or we need a different metric, compute it
                if score is None or (display_metric != rank_metric and display_metric != pred.get('metric', '')):
                    # Compute the metric from y_true and y_pred
                    try:
                        from nirs4all.utils.evaluator import Evaluator
                        y_true = pred.get('y_true', [])
                        y_pred = pred.get('y_pred', [])
                        if y_true and y_pred:
                            score = Evaluator.eval(y_true, y_pred, display_metric)
                    except Exception:
                        score = None

                if score is not None:
                    display_lookup[identity_key] = {
                        'score': score,
                        'x_var': pred.get(x_var),
                        'y_var': pred.get(y_var),
                    }

            # Merge: For each ranked prediction, attach its display score AND rank score
            all_predictions = []
            for identity_key, rank_pred in unique_models.items():
                if identity_key in display_lookup:
                    # Create merged prediction with display score
                    merged_pred = rank_pred.copy()
                    merged_pred[f'{display_partition}_score'] = display_lookup[identity_key]['score']
                    # Store the rank score for proper aggregation
                    merged_pred['_rank_score'] = rank_pred.get(f'{rank_partition}_score')
                    # Also copy x_var and y_var from display if available
                    merged_pred[x_var] = display_lookup[identity_key].get('x_var', rank_pred.get(x_var))
                    merged_pred[y_var] = display_lookup[identity_key].get('y_var', rank_pred.get(y_var))
                    all_predictions.append(merged_pred)

        # Build score matrix - use {display_partition}_score field for display
        # Pass rank score field for proper aggregation when rank != display partition
        display_score_field = f'{display_partition}_score'
        rank_score_field = '_rank_score' if rank_partition != display_partition else display_score_field
        score_dict = self._build_score_dict_with_ranking(all_predictions, x_var, y_var, display_score_field, rank_score_field)

        if not score_dict:
            return self._create_empty_heatmap_figure(figsize, filters, f'No valid {display_metric} scores found')

        # Extract sorted labels and build matrices (use display metric's directionality)
        y_labels, x_labels, matrix, count_matrix = self._build_heatmap_matrices(score_dict, aggregation, display_higher_better)

        # Normalize if requested
        normalized_matrix = self._normalize_heatmap_matrix(matrix, normalize, display_higher_better)

        # Create and render the plot
        return self._render_heatmap_v2(matrix, normalized_matrix, count_matrix, x_labels, y_labels, x_var, y_var, rank_metric, rank_partition,
                                       display_metric, display_partition, figsize, normalize, aggregation, show_counts)

    def _create_empty_heatmap_figure(self, figsize: Tuple[int, int], filters: Dict, message: str) -> Figure:
        """Create empty figure with message for heatmap."""
        fig, ax = plt.subplots(figsize=figsize)
        filter_desc = ', '.join(f"{k}={v}" for k, v in filters.items()) if filters else 'none'
        full_message = f'{message}\nFilters: {filter_desc}'
        ax.text(0.5, 0.5, full_message, ha='center', va='center', fontsize=14)
        ax.set_title('No Data Available')
        return fig

    def _build_score_dict(self, all_predictions: List[Dict], x_var: str, y_var: str, display_metric: str) -> Dict:
        """Build dictionary of scores grouped by x_var and y_var."""
        score_dict = defaultdict(lambda: defaultdict(list))
        for pred in all_predictions:
            x_val = str(pred.get(x_var, 'unknown'))
            y_val = str(pred.get(y_var, 'unknown'))
            score = pred.get(display_metric)
            if score is not None and not np.isnan(score):
                score_dict[y_val][x_val].append(score)
        return score_dict

    def _build_score_dict_with_ranking(self, all_predictions: List[Dict], x_var: str, y_var: str,
                                       display_score_field: str, rank_score_field: str) -> Dict:
        """
        Build dictionary of scores grouped by x_var and y_var, keeping ranking information.

        Returns dict structure: {y_val: {x_val: [(display_score, rank_score), ...]}}
        """
        score_dict = defaultdict(lambda: defaultdict(list))
        for pred in all_predictions:
            x_val = str(pred.get(x_var, 'unknown'))
            y_val = str(pred.get(y_var, 'unknown'))
            display_score = pred.get(display_score_field)
            rank_score = pred.get(rank_score_field)

            if display_score is not None and not np.isnan(display_score):
                # Store tuple of (display_score, rank_score) for proper aggregation
                score_dict[y_val][x_val].append((display_score, rank_score if rank_score is not None else display_score))
        return score_dict

    def _build_heatmap_matrices(self, score_dict: Dict, aggregation: str, higher_better: bool) -> Tuple[List, List, np.ndarray, np.ndarray]:
        """
        Build matrices for heatmap from score dictionary.

        score_dict can contain either:
        - Simple scores: {y_val: {x_val: [score1, score2, ...]}}
        - Tuples with ranking: {y_val: {x_val: [(display_score, rank_score), ...]}}
        """
        y_labels = sorted(score_dict.keys(), key=self._natural_sort_key)
        x_labels = sorted(set(x for y_data in score_dict.values() for x in y_data.keys()), key=self._natural_sort_key)

        matrix = np.full((len(y_labels), len(x_labels)), np.nan)
        count_matrix = np.zeros((len(y_labels), len(x_labels)), dtype=int)

        for i, y_val in enumerate(y_labels):
            for j, x_val in enumerate(x_labels):
                scores = score_dict[y_val].get(x_val, [])
                if scores:
                    count_matrix[i, j] = len(scores)

                    # Check if scores are tuples (display_score, rank_score) or simple values
                    if scores and isinstance(scores[0], tuple):
                        # Scores with ranking information
                        if aggregation == 'best':
                            # Select based on rank_score, display the corresponding display_score
                            best_idx = np.argmin([rank for _, rank in scores]) if not higher_better else np.argmax([rank for _, rank in scores])
                            matrix[i, j] = scores[best_idx][0]  # Display score of best ranked model
                        elif aggregation == 'mean':
                            matrix[i, j] = np.mean([disp for disp, _ in scores])
                        elif aggregation == 'median':
                            matrix[i, j] = np.median([disp for disp, _ in scores])
                    else:
                        # Simple scores (backward compatibility)
                        if aggregation == 'best':
                            matrix[i, j] = max(scores) if higher_better else min(scores)
                        elif aggregation == 'mean':
                            matrix[i, j] = np.mean(scores)
                        elif aggregation == 'median':
                            matrix[i, j] = np.median(scores)

        return y_labels, x_labels, matrix, count_matrix

    def _normalize_heatmap_matrix(self, matrix: np.ndarray, normalize: bool, higher_better: bool) -> np.ndarray:
        """Normalize matrix for heatmap display."""
        normalized_matrix = matrix.copy()
        if normalize:
            valid_mask = ~np.isnan(matrix)
            if valid_mask.any():
                min_val = np.nanmin(matrix)
                max_val = np.nanmax(matrix)
                if max_val > min_val:
                    # For "higher is better": (score - min) / (max - min), For "lower is better": 1 - (score - min) / (max - min)
                    normalized_matrix[valid_mask] = (matrix[valid_mask] - min_val) / (max_val - min_val)
                    if not higher_better:
                        normalized_matrix[valid_mask] = 1 - normalized_matrix[valid_mask]
        return normalized_matrix

    def _render_heatmap_v2(self, matrix: np.ndarray, normalized_matrix: np.ndarray, count_matrix: np.ndarray, x_labels: List, y_labels: List,
                           x_var: str, y_var: str, rank_metric: str, rank_partition: str, display_metric: str, display_partition: str,
                           figsize: Tuple[int, int], normalize: bool, aggregation: str, show_counts: bool) -> Figure:
        """Render the final heatmap plot."""
        fig, ax = plt.subplots(figsize=figsize)

        # Use RdYlGn colormap (Red-Yellow-Green)
        masked_matrix = np.ma.masked_invalid(normalized_matrix)
        im = ax.imshow(masked_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar_label = f'Normalized {display_metric.upper()}\n(1=best, 0=worst)' if normalize else f'{display_metric.upper()} Score'
        cbar.set_label(cbar_label, fontsize=10)

        # Set axis labels and ticks - Truncate long labels
        x_labels_display = [lbl[:25] + '...' if len(lbl) > 25 else lbl for lbl in x_labels]
        y_labels_display = [lbl[:25] + '...' if len(lbl) > 25 else lbl for lbl in y_labels]

        ax.set_xticks(range(len(x_labels)))
        ax.set_yticks(range(len(y_labels)))
        ax.set_xticklabels(x_labels_display, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(y_labels_display, fontsize=9)
        ax.set_xlabel(x_var.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel(y_var.replace('_', ' ').title(), fontsize=11)

        # Create title
        title_parts = [f'{aggregation.title()} {display_metric.upper()}']
        if rank_partition != display_partition or rank_metric != display_metric:
            title_parts.append(f'(ranked by {rank_metric.upper()} on {rank_partition})')
        title_parts.append(f'[{display_partition}]')
        ax.set_title(' '.join(title_parts), fontsize=12, pad=10)

        # Add text annotations to cells
        self._add_heatmap_v2_annotations(ax, matrix, normalized_matrix, count_matrix, show_counts, x_labels, y_labels)

        plt.tight_layout()
        return fig

    def _add_heatmap_v2_annotations(self, ax, matrix: np.ndarray, normalized_matrix: np.ndarray, count_matrix: np.ndarray,
                                    show_counts: bool, x_labels: List, y_labels: List) -> None:
        """Add text annotations to heatmap cells."""
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                if not np.isnan(matrix[i, j]):
                    score_text = f'{matrix[i, j]:.3f}'
                    text = f'{score_text}\n(n={count_matrix[i, j]})' if show_counts and count_matrix[i, j] > 1 else score_text
                    # Always use black text for better readability
                    ax.text(j, i, text, ha='center', va='center', fontsize=8, color='black')

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
