"""
Prediction Visualizer - Graphical analysis of pipeline prediction results

This module provides visualization capabilities for prediction data stored in dataset._predictions,
allowing users to compare model performance across different configurations and datasets.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import pandas as pd


class PredictionVisualizer:
    """
    Visualizes prediction results from dataset._predictions with configurable metrics and layouts.

    Features:
    - Matrix visualization with configs vs models
    - Multiple metric support
    - Sorting by best model/config
    - Color-coded heatmaps
    - Flexible subplot arrangements
    """

    def __init__(self, predictions_obj: Any, dataset_name_override: str = None):
        """
        Initialize with a predictions object from dataset._predictions.

        Args:
            predictions_obj: The predictions object containing prediction data
            dataset_name_override: Override for dataset name display (fixes "unknown" issue)
        """
        self.predictions = predictions_obj
        self.dataset_name_override = dataset_name_override
        self.data = self._extract_prediction_data()

    def _extract_prediction_data(self) -> List[Dict]:
        """Extract prediction data from the predictions object."""
        if hasattr(self.predictions, '_predictions'):
            pred_dict = self.predictions._predictions
            if isinstance(pred_dict, dict):
                # Convert dictionary values to list
                return list(pred_dict.values())
            elif isinstance(pred_dict, list):
                return pred_dict
        elif hasattr(self.predictions, 'get_predictions'):
            try:
                pred_data = self.predictions.get_predictions()
                if isinstance(pred_data, dict):
                    return list(pred_data.values())
                elif isinstance(pred_data, list):
                    return pred_data
            except Exception:
                pass
        elif hasattr(self.predictions, 'data'):
            return self.predictions.data if self.predictions.data else []

        return []

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate common regression metrics."""
        # Ensure arrays are flat
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

        # Calculate metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))

        # R² calculation
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def _organize_data_by_dataset_config_model(self) -> Dict[str, Dict[str, Dict[str, Dict]]]:
        """
        Organize prediction data by dataset -> config -> model -> metrics.

        Returns:
            Nested dictionary structure for easy access
        """
        organized = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for pred_record in self.data:
            try:
                dataset = self.dataset_name_override or pred_record.get('dataset', 'unknown')
                pipeline = pred_record.get('pipeline', 'unknown')
                model = pred_record.get('model', 'unknown')
                y_true = pred_record.get('y_true', [])
                y_pred = pred_record.get('y_pred', [])

                # Calculate metrics for this prediction
                metrics = self._calculate_metrics(y_true, y_pred)

                # Store in organized structure
                organized[dataset][pipeline][model] = {
                    'metrics': metrics,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'metadata': pred_record.get('metadata', {})
                }

            except Exception as e:
                print(f"⚠️ Error processing prediction record: {e}")
                continue

        return dict(organized)

    def _filter_data_by_prediction_types(self, prediction_filter: str = 'all') -> List[Dict]:
        """
        Filter prediction data based on prediction type.

        Args:
            prediction_filter: Filter type - 'all', 'best_only', 'folds_only', 'averaged_only', or 'global_only'

        Returns:
            Filtered list of prediction records
        """
        if prediction_filter == 'all':
            return self.data

        filtered_data = []

        if prediction_filter == 'best_only':
            # Group by base model and find best performance
            import re
            from collections import defaultdict

            model_groups = defaultdict(list)
            for pred_record in self.data:
                model_name = pred_record.get('model', 'unknown')
                # Extract base model name
                base_model_match = re.match(r'(.+?)_\d+$', model_name)
                base_model = base_model_match.group(1) if base_model_match else model_name
                model_groups[base_model].append(pred_record)

            # For each base model, find the best performing prediction
            for base_model, predictions in model_groups.items():
                best_pred = None
                best_r2 = -float('inf')

                for pred in predictions:
                    y_true = pred.get('y_true', [])
                    y_pred = pred.get('y_pred', [])
                    metrics = self._calculate_metrics(y_true, y_pred)

                    if metrics['r2'] > best_r2:
                        best_r2 = metrics['r2']
                        best_pred = pred

                if best_pred:
                    filtered_data.append(best_pred)

        elif prediction_filter == 'folds_only':
            # Only fold predictions (train_fold, val_fold, test_fold)
            for pred_record in self.data:
                partition = pred_record.get('partition', '')
                if 'fold' in partition and 'avg' not in partition:
                    filtered_data.append(pred_record)

        elif prediction_filter == 'averaged_only':
            # Only averaged predictions (avg_, weighted_avg_)
            for pred_record in self.data:
                partition = pred_record.get('partition', '')
                if 'avg' in partition:
                    filtered_data.append(pred_record)

        elif prediction_filter == 'global_only':
            # Only global predictions (global_train, test without fold)
            for pred_record in self.data:
                partition = pred_record.get('partition', '')
                if ('global' in partition or
                        ('test' in partition and 'fold' not in partition and 'avg' not in partition)):
                    filtered_data.append(pred_record)

        return filtered_data

    def plot_performance_matrix(self,
                               dataset: Optional[str] = None,
                               metric: str = 'rmse',
                               sort_by: str = 'model',
                               ascending: bool = True,
                               figsize: Tuple[int, int] = (12, 8),
                               cmap: str = 'RdYlGn_r',
                               show_values: bool = True,
                               prediction_filter: str = 'all') -> plt.Figure:
        """
        Plot a performance matrix with configs on y-axis and models on x-axis.

        Args:
            dataset: Dataset to visualize (if None, uses first available)
            metric: Metric to display ('mse', 'rmse', 'mae', 'r2')
            sort_by: Sort by 'model', 'config', or 'none'
            ascending: Sort order
            figsize: Figure size
            cmap: Colormap for heatmap
            show_values: Whether to show metric values in cells
            prediction_filter: Filter predictions ('all', 'best_only', 'folds_only', 'averaged_only', 'global_only')

        Returns:
            matplotlib Figure object
        """
        # Apply filtering
        original_data = self.data
        self.data = self._filter_data_by_prediction_types(prediction_filter)
        organized_data = self._organize_data_by_dataset_config_model()
        self.data = original_data  # Restore original data

        if not organized_data:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No prediction data available',
                   ha='center', va='center', fontsize=16)
            ax.set_title('Prediction Performance Matrix - No Data')
            return fig

        # Select dataset
        if dataset is None:
            dataset = list(organized_data.keys())[0]

        if dataset not in organized_data:
            available = list(organized_data.keys())
            raise ValueError(f"Dataset '{dataset}' not found. Available: {available}")

        dataset_data = organized_data[dataset]

        # Get all configs and models
        all_configs = sorted(dataset_data.keys())
        all_models = sorted(set(
            model for config_data in dataset_data.values()
            for model in config_data.keys()
        ))

        # Create matrix
        matrix = np.full((len(all_configs), len(all_models)), np.nan)

        for i, config in enumerate(all_configs):
            for j, model in enumerate(all_models):
                if model in dataset_data[config]:
                    metrics = dataset_data[config][model]['metrics']
                    matrix[i, j] = metrics.get(metric, np.nan)

        # Sort if requested
        if sort_by == 'model':
            # Sort by average performance across configs for each model
            model_avgs = np.nanmean(matrix, axis=0)
            sort_indices = np.argsort(model_avgs)
            if not ascending:
                sort_indices = sort_indices[::-1]
            matrix = matrix[:, sort_indices]
            all_models = [all_models[i] for i in sort_indices]

        elif sort_by == 'config':
            # Sort by average performance across models for each config
            config_avgs = np.nanmean(matrix, axis=1)
            sort_indices = np.argsort(config_avgs)
            if not ascending:
                sort_indices = sort_indices[::-1]
            matrix = matrix[sort_indices, :]
            all_configs = [all_configs[i] for i in sort_indices]

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Handle colormap scaling
        valid_values = matrix[~np.isnan(matrix)]
        if len(valid_values) > 0:
            vmin, vmax = valid_values.min(), valid_values.max()
        else:
            vmin, vmax = 0, 1

        # Create heatmap
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

        # Set ticks and labels
        ax.set_xticks(range(len(all_models)))
        ax.set_yticks(range(len(all_configs)))
        ax.set_xticklabels(all_models, rotation=45, ha='right')
        ax.set_yticklabels(all_configs)

        # Add values to cells if requested
        if show_values:
            for i in range(len(all_configs)):
                for j in range(len(all_models)):
                    value = matrix[i, j]
                    if not np.isnan(value):
                        text_color = 'white' if value > (vmin + vmax) / 2 else 'black'
                        ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                               color=text_color, fontsize=8)

        # Labels and title
        ax.set_xlabel('Models')
        ax.set_ylabel('Pipeline Configurations')
        ax.set_title(f'{metric.upper()} Performance Matrix - Dataset: {dataset}')

        # Add colorbar
        plt.colorbar(im, ax=ax, label=metric.upper())

        plt.tight_layout()
        return fig

    def plot_multi_metric_comparison(self,
                                    dataset: Optional[str] = None,
                                    metrics: List[str] = None,
                                    sort_by: str = 'r2',
                                    ascending: bool = False,
                                    figsize: Tuple[int, int] = (16, 10),
                                    prediction_filter: str = 'all') -> plt.Figure:
        """
        Plot multiple metrics in subplots for comprehensive comparison.

        Args:
            dataset: Dataset to visualize
            metrics: List of metrics to show
            sort_by: Metric to sort by
            ascending: Sort order
            figsize: Figure size
            prediction_filter: Filter predictions ('all', 'best_only', 'folds_only', 'averaged_only', 'global_only')

        Returns:
            matplotlib Figure object
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2', 'mse']

        # Apply filtering
        original_data = self.data
        self.data = self._filter_data_by_prediction_types(prediction_filter)

        n_metrics = len(metrics)
        cols = 2
        rows = (n_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue

            try:
                # Get organized data
                organized_data = self._organize_data_by_dataset_config_model()

                if not organized_data:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(f'{metric.upper()}')
                    continue

                # Select dataset
                if dataset is None:
                    dataset = list(organized_data.keys())[0]

                dataset_data = organized_data[dataset]
                all_configs = sorted(dataset_data.keys())
                all_models = sorted(set(
                    model for config_data in dataset_data.values()
                    for model in config_data.keys()
                ))

                # Create matrix for this metric
                matrix = np.full((len(all_configs), len(all_models)), np.nan)

                for ci, config in enumerate(all_configs):
                    for mi, model in enumerate(all_models):
                        if model in dataset_data[config]:
                            metrics_dict = dataset_data[config][model]['metrics']
                            matrix[ci, mi] = metrics_dict.get(metric, np.nan)

                # Sort by the specified metric if it's the current metric
                if metric == sort_by:
                    if sort_by in ['r2']:  # Higher is better
                        config_avgs = np.nanmean(matrix, axis=1)
                        sort_indices = np.argsort(config_avgs)
                        if not ascending:
                            sort_indices = sort_indices[::-1]
                    else:  # Lower is better (rmse, mae, mse)
                        config_avgs = np.nanmean(matrix, axis=1)
                        sort_indices = np.argsort(config_avgs)
                        if ascending:
                            sort_indices = sort_indices[::-1]

                    matrix = matrix[sort_indices, :]
                    all_configs = [all_configs[idx] for idx in sort_indices]

                # Plot heatmap
                valid_values = matrix[~np.isnan(matrix)]
                if len(valid_values) > 0:
                    vmin, vmax = valid_values.min(), valid_values.max()
                    cmap = 'RdYlGn' if metric == 'r2' else 'RdYlGn_r'

                    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

                    # Add values
                    for ci in range(len(all_configs)):
                        for mi in range(len(all_models)):
                            value = matrix[ci, mi]
                            if not np.isnan(value):
                                text_color = 'white' if value > (vmin + vmax) / 2 else 'black'
                                ax.text(mi, ci, f'{value:.3f}', ha='center', va='center',
                                       color=text_color, fontsize=6)

                    ax.set_xticks(range(len(all_models)))
                    ax.set_yticks(range(len(all_configs)))
                    ax.set_xticklabels(all_models, rotation=45, ha='right', fontsize=8)
                    ax.set_yticklabels(all_configs, fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')

                ax.set_title(f'{metric.upper()}')

            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                ax.set_title(f'{metric.upper()} (Error)')

        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')

        # Restore original data
        self.data = original_data
        plt.tight_layout()
        return fig

    def plot_prediction_scatter(self,
                               dataset: Optional[str] = None,
                               model: Optional[str] = None,
                               config: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 8),
                               prediction_filter: str = 'all') -> plt.Figure:
        """
        Create scatter plots of true vs predicted values.

        Args:
            dataset: Dataset to visualize
            model: Specific model to show
            config: Specific config to show
            figsize: Figure size
            prediction_filter: Filter predictions ('all', 'best_only', 'folds_only', 'averaged_only', 'global_only')

        Returns:
            matplotlib Figure object
        """
        # Apply filtering
        original_data = self.data
        self.data = self._filter_data_by_prediction_types(prediction_filter)
        organized_data = self._organize_data_by_dataset_config_model()
        self.data = original_data  # Restore original data

        if not organized_data:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No prediction data available',
                   ha='center', va='center', fontsize=16)
            ax.set_title('Prediction Scatter Plot - No Data')
            return fig

        # Select dataset
        if dataset is None:
            dataset = list(organized_data.keys())[0]

        dataset_data = organized_data[dataset]

        # Collect all prediction pairs
        scatter_data = []

        for cfg, config_data in dataset_data.items():
            if config and cfg != config:
                continue

            for mdl, model_data in config_data.items():
                if model and mdl != model:
                    continue

                y_true = np.array(model_data['y_true']).flatten()
                y_pred = np.array(model_data['y_pred']).flatten()

                # Remove NaN values
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                y_true = y_true[mask]
                y_pred = y_pred[mask]

                if len(y_true) > 0:
                    scatter_data.append({
                        'config': cfg,
                        'model': mdl,
                        'y_true': y_true,
                        'y_pred': y_pred,
                        'metrics': model_data['metrics']
                    })

        if not scatter_data:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No matching prediction data',
                   ha='center', va='center', fontsize=16)
            ax.set_title('Prediction Scatter Plot - No Matching Data')
            return fig

        # Create plots
        n_plots = len(scatter_data)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, data in enumerate(scatter_data):
            if i >= len(axes):
                break

            ax = axes[i]
            y_true = data['y_true']
            y_pred = data['y_pred']

            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.6, s=20)

            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect')

            # Labels and metrics
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')

            metrics = data['metrics']
            title = f"{data['config']} - {data['model']}"
            subtitle = f"R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}"
            ax.set_title(f"{title}\n{subtitle}", fontsize=10)

            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        return fig

    def summary_report(self) -> str:
        """Generate a text summary of the prediction data."""
        organized_data = self._organize_data_by_dataset_config_model()

        if not organized_data:
            return "No prediction data available."

        report = []
        report.append("🔍 PREDICTION SUMMARY REPORT")
        report.append("=" * 50)

        for dataset, dataset_data in organized_data.items():
            report.append(f"\n📊 Dataset: {dataset}")
            report.append("-" * 30)

            n_configs = len(dataset_data)
            all_models = set()
            all_metrics = []

            for config, config_data in dataset_data.items():
                all_models.update(config_data.keys())
                for model, model_data in config_data.items():
                    metrics = model_data['metrics']
                    all_metrics.append({
                        'config': config,
                        'model': model,
                        **metrics
                    })

            report.append(f"• Configurations: {n_configs}")
            report.append(f"• Models: {len(all_models)} ({', '.join(sorted(all_models))})")
            report.append(f"• Total predictions: {len(all_metrics)}")

            if all_metrics:
                df = pd.DataFrame(all_metrics)

                report.append("\n🏆 Best Performance (by R²):")
                if 'r2' in df.columns:
                    best_r2 = df.loc[df['r2'].idxmax()]
                    report.append(f"   {best_r2['config']} + {best_r2['model']}: R²={best_r2['r2']:.4f}")

                report.append("\n🎯 Best Performance (by RMSE):")
                if 'rmse' in df.columns:
                    best_rmse = df.loc[df['rmse'].idxmin()]
                    report.append(f"   {best_rmse['config']} + {best_rmse['model']}: RMSE={best_rmse['rmse']:.4f}")

        return "\n".join(report)

    def comprehensive_prediction_summary(self) -> str:
        """
        Generate a comprehensive summary showing all prediction types for each model.
        Uses smart grouping to recognize cross-validation fold instances.

        Returns:
            Detailed text report of all prediction types
        """
        if not self.data:
            return "No prediction data available."

        import re
        from collections import defaultdict

        # Group by dataset, pipeline, then by BASE model name (smart grouping)
        organized = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

        for pred_record in self.data:
            dataset = self.dataset_name_override or pred_record.get('dataset', 'unknown')
            pipeline = pred_record.get('pipeline', 'unknown')
            model_name = pred_record.get('model', 'unknown')
            partition = pred_record.get('partition', 'unknown')

            # Extract base model name (remove _X suffix)
            base_model_match = re.match(r'(.+?)_\d+$', model_name)
            base_model = base_model_match.group(1) if base_model_match else model_name

            # Calculate metrics
            y_true = pred_record.get('y_true', [])
            y_pred = pred_record.get('y_pred', [])
            metrics = self._calculate_metrics(y_true, y_pred)

            # Categorize partition types for better organization
            if 'global_train' in partition:
                partition_type = 'Global Train'
            elif 'test' in partition and 'fold' not in partition and 'avg' not in partition:
                partition_type = 'Global Test'
            elif 'test_fold' in partition:
                partition_type = 'Fold Test'
            elif 'train_fold' in partition:
                partition_type = 'Fold Train'
            elif 'val_fold' in partition:
                partition_type = 'Fold Val'
            elif 'avg_test_fold' in partition:
                partition_type = 'Average'
            elif 'weighted_avg_test_fold' in partition:
                partition_type = 'Weighted Average'
            else:
                partition_type = partition.replace('_', ' ').title()

            organized[dataset][pipeline][base_model][partition_type].append({
                'partition': partition,
                'model_instance': model_name,
                'metrics': metrics,
                'sample_count': len(y_true) if hasattr(y_true, '__len__') else 0,
                'fold_idx': pred_record.get('fold_idx', 0),
                'metadata': pred_record.get('metadata', {})
            })

        report = []
        report.append("🔍 COMPREHENSIVE PREDICTION SUMMARY")
        report.append("=" * 60)

        for dataset, dataset_data in organized.items():
            report.append(f"\n📊 Dataset: {dataset}")
            report.append("─" * 40)

            for pipeline, pipeline_data in dataset_data.items():
                report.append(f"\n  🔧 Pipeline: {pipeline}")
                report.append("  " + "─" * 35)

                for base_model, prediction_types in pipeline_data.items():
                    report.append(f"\n    🤖 Base Model: {base_model}")
                    report.append("    " + "─" * 30)

                    # Track available and missing prediction types
                    all_prediction_types = {'Global Train', 'Global Test', 'Fold Train', 'Fold Val', 'Fold Test', 'Average', 'Weighted Average'}
                    available_types = set(prediction_types.keys())
                    missing_types = all_prediction_types - available_types

                    # Display each available prediction type
                    for pred_type in sorted(available_types):
                        preds = prediction_types[pred_type]
                        report.append(f"\n      📈 {pred_type}:")

                        if len(preds) == 1:
                            # Single prediction
                            pred = preds[0]
                            metrics = pred['metrics']
                            sample_count = pred['sample_count']
                            report.append(f"        ✅ {sample_count:,} samples")
                            report.append(f"        📊 RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
                            report.append(f"        📈 R²: {metrics['r2']:.4f}, MSE: {metrics['mse']:.4f}")

                            if pred_type in ['Average', 'Weighted Average']:
                                metadata = pred['metadata']
                                if 'weights' in metadata:
                                    weights = metadata['weights']
                                    report.append(f"        ⚖️  Weights: [{', '.join(f'{w:.3f}' for w in weights)}]")

                        else:
                            # Multiple predictions (e.g., fold predictions)
                            report.append(f"        ✅ {len(preds)} instances:")

                            total_samples = 0
                            all_rmse = []
                            all_r2 = []

                            for pred in sorted(preds, key=lambda x: x['fold_idx'] if x['fold_idx'] is not None else 0):
                                metrics = pred['metrics']
                                sample_count = pred['sample_count']
                                total_samples += sample_count
                                all_rmse.append(metrics['rmse'])
                                all_r2.append(metrics['r2'])

                                fold_info = pred['partition'].split('_')[-1] if '_' in pred['partition'] else pred['fold_idx']
                                report.append(f"          Instance {fold_info}: {sample_count:,} samples, "
                                            f"RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
                                report.append(f"            Model: {pred['model_instance']}")

                            # Show aggregate statistics
                            if len(all_rmse) > 1:
                                report.append(f"        📊 Aggregate: {total_samples:,} total samples")
                                report.append(f"        📈 Average Performance: RMSE={np.mean(all_rmse):.4f}±{np.std(all_rmse):.4f}")
                                report.append(f"                               R²={np.mean(all_r2):.4f}±{np.std(all_r2):.4f}")

                    # Show availability summary
                    if available_types:
                        report.append(f"\n      ✅ Available: {', '.join(sorted(available_types))}")
                    if missing_types:
                        report.append(f"      ⚠️  Missing: {', '.join(sorted(missing_types))}")

        return "\n".join(report)

    def get_all_predictions_for_model(
        self,
        dataset: str,
        pipeline: str,
        model: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get all prediction results for a specific model in organized format.

        Args:
            dataset: Dataset name
            pipeline: Pipeline name
            model: Model name

        Returns:
            Dictionary organized by prediction type containing metrics and data
        """
        results = {
            'global_train': None,
            'test': None,
            'fold_predictions': {'train': [], 'val': [], 'test': []},
            'average_predictions': None,
            'weighted_average_predictions': None
        }

        for pred_record in self.data:
            if (pred_record.get('dataset') == dataset and
                pred_record.get('pipeline') == pipeline and
                pred_record.get('model') == model):

                partition = pred_record.get('partition', '')
                y_true = pred_record.get('y_true', [])
                y_pred = pred_record.get('y_pred', [])
                metrics = self._calculate_metrics(y_true, y_pred)

                pred_data = {
                    'partition': partition,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'metrics': metrics,
                    'sample_count': len(y_true) if hasattr(y_true, '__len__') else 0,
                    'fold_idx': pred_record.get('fold_idx'),
                    'metadata': pred_record.get('metadata', {})
                }

                # Categorize prediction type
                if 'global_train' in partition:
                    results['global_train'] = pred_data
                elif 'test' in partition and 'fold' not in partition and 'avg' not in partition:
                    results['test'] = pred_data
                elif 'train_fold' in partition:
                    results['fold_predictions']['train'].append(pred_data)
                elif 'val_fold' in partition:
                    results['fold_predictions']['val'].append(pred_data)
                elif 'test_fold' in partition and 'avg' not in partition:
                    results['fold_predictions']['test'].append(pred_data)
                elif 'avg_test_fold' in partition:
                    results['average_predictions'] = pred_data
                elif 'weighted_avg_test_fold' in partition:
                    results['weighted_average_predictions'] = pred_data

        # Sort fold predictions by fold index
        for fold_type in results['fold_predictions']:
            results['fold_predictions'][fold_type].sort(
                key=lambda x: x['fold_idx'] if x['fold_idx'] is not None else 0
            )

        return results

    def plot_filtered_predictions(self,
                                  dataset: Optional[str] = None,
                                  prediction_filter: str = 'all',
                                  metric: str = 'rmse',
                                  chart_type: str = 'bar',
                                  figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot filtered predictions with different chart types.

        Args:
            dataset: Dataset to visualize
            prediction_filter: Filter type ('all', 'best_only', 'folds_only', 'averaged_only', 'global_only')
            metric: Metric to display
            chart_type: 'bar', 'scatter', or 'matrix'
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        # Apply filtering
        original_data = self.data
        self.data = self._filter_data_by_prediction_types(prediction_filter)

        try:
            if chart_type == 'bar':
                return self._plot_bar_chart(dataset, metric, figsize, prediction_filter)
            elif chart_type == 'scatter':
                return self.plot_prediction_scatter(dataset=dataset, figsize=figsize, prediction_filter='all')
            elif chart_type == 'matrix':
                return self.plot_performance_matrix(dataset=dataset, metric=metric, figsize=figsize, prediction_filter='all')
            else:
                raise ValueError(f"Unknown chart_type: {chart_type}. Use 'bar', 'scatter', or 'matrix'")
        finally:
            # Always restore original data
            self.data = original_data

    def _plot_bar_chart(self, dataset: Optional[str], metric: str, figsize: Tuple[int, int], prediction_filter: str) -> plt.Figure:
        """Create a bar chart showing prediction performance by type."""
        import re
        from collections import defaultdict

        # Group predictions by type and model
        prediction_groups = defaultdict(list)

        for pred_record in self.data:
            if dataset and pred_record.get('dataset', 'unknown') != dataset:
                continue

            model_name = pred_record.get('model', 'unknown')
            partition = pred_record.get('partition', 'unknown')

            # Extract base model name
            base_model_match = re.match(r'(.+?)_\d+$', model_name)
            base_model = base_model_match.group(1) if base_model_match else model_name

            # Categorize partition
            if 'global_train' in partition:
                pred_type = 'Global Train'
            elif 'test' in partition and 'fold' not in partition and 'avg' not in partition:
                pred_type = 'Global Test'
            elif 'test_fold' in partition and 'avg' not in partition:
                pred_type = 'CV Test Folds'
            elif 'val_fold' in partition:
                pred_type = 'CV Val Folds'
            elif 'train_fold' in partition:
                pred_type = 'CV Train Folds'
            elif 'avg_test_fold' in partition:
                pred_type = 'Average Test'
            elif 'weighted_avg_test_fold' in partition:
                pred_type = 'Weighted Avg Test'
            else:
                pred_type = partition.replace('_', ' ').title()

            y_true = pred_record.get('y_true', [])
            y_pred = pred_record.get('y_pred', [])
            metrics = self._calculate_metrics(y_true, y_pred)

            prediction_groups[f"{base_model} - {pred_type}"].append({
                'metric_value': metrics.get(metric, np.nan),
                'sample_count': len(y_true) if hasattr(y_true, '__len__') else 0,
                'model': base_model,
                'type': pred_type
            })

        if not prediction_groups:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No prediction data for filter: {prediction_filter}',
                   ha='center', va='center', fontsize=16)
            ax.set_title(f'Filtered Predictions ({prediction_filter}) - No Data')
            return fig

        # Aggregate multiple predictions of same type
        plot_data = []
        for group_name, predictions in prediction_groups.items():
            if len(predictions) == 1:
                pred = predictions[0]
                plot_data.append({
                    'name': group_name,
                    'value': pred['metric_value'],
                    'samples': pred['sample_count'],
                    'model': pred['model'],
                    'type': pred['type']
                })
            else:
                # Multiple predictions (e.g., multiple folds) - combine them
                valid_metrics = [p['metric_value'] for p in predictions if not np.isnan(p['metric_value'])]
                total_samples = sum(p['sample_count'] for p in predictions)

                if valid_metrics:
                    avg_metric = np.mean(valid_metrics)
                    plot_data.append({
                        'name': group_name,
                        'value': avg_metric,
                        'samples': total_samples,
                        'model': predictions[0]['model'],
                        'type': predictions[0]['type'],
                        'fold_count': len(predictions)
                    })

        # Sort by metric value (best first)
        if metric in ['r2']:
            plot_data.sort(key=lambda x: x['value'], reverse=True)
        else:
            plot_data.sort(key=lambda x: x['value'])

        # Create bar chart
        fig, ax = plt.subplots(figsize=figsize)

        names = [d['name'] for d in plot_data]
        values = [d['value'] for d in plot_data]
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))

        bars = ax.bar(range(len(names)), values, color=colors)

        # Add value labels on bars
        for i, (bar, data) in enumerate(zip(bars, plot_data)):
            height = bar.get_height()
            sample_info = f"{data['samples']} samples"
            if 'fold_count' in data:
                sample_info += f" ({data['fold_count']} folds)"

            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}\n{sample_info}',
                   ha='center', va='bottom', fontsize=8)

        # Customize plot
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel(f'{metric.upper()} Score')

        # Get dataset name for title
        display_dataset = self.dataset_name_override or dataset or 'Unknown Dataset'
        ax.set_title(f'{metric.upper()} Performance by Prediction Type\nDataset: {display_dataset} | Filter: {prediction_filter}')

        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        return fig