"""
Tab Report Generator - Generate standardized tab-based CSV reports for best performing models

This module provides functionality to:
- Automatically detect the best performing model for each dataset
- Generate task-specific tab templates (regression vs classification)
- Reconstruct cross-validation metrics from fold predictions
- Save formatted CSV reports with proper naming conventions
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path
import csv
import os

from nirs4all.dataset.predictions import Predictions
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer
from nirs4all.utils.model_utils import ModelUtils, TaskType


class TabReportGenerator:
    """Generate standardized tab-based CSV reports for best performing models."""

    def __init__(self):
        """Initialize the tab report generator."""
        pass

    def generate_best_score_report(
        self,
        predictions: Predictions,
        dataset_name: str,
        save_path: str,
        enable_tab_reports: bool = True,
        dataset=None
    ) -> Optional[str]:
        """
        Generate a tab-based CSV report for the best performing model.

        Args:
            predictions: Predictions object containing all model results
            dataset_name: Name of the dataset
            save_path: Base path where to save the report
            enable_tab_reports: Whether to generate tab reports

        Returns:
            str: Path to the generated report file, or None if not generated
        """
        if not enable_tab_reports:
            return None

        try:
            # Find the best model using PredictionAnalyzer
            analyzer = PredictionAnalyzer(predictions, dataset_name)

            # Detect task type from the first available prediction
            task_type = self._detect_task_type(predictions)
            if task_type is None:
                print("⚠️ Could not detect task type for tab report generation")
                return None

            # Get best metric for this task type
            best_metric, higher_is_better = ModelUtils.get_best_score_metric(task_type)

            # Get top 1 model based on test partition performance
            top_models = analyzer.get_top_k(1, best_metric, partition_type='test')
            if not top_models:
                print("⚠️ No test predictions found for tab report generation")
                return None

            best_model = top_models[0]

            # Generate the tab report
            report_data = self._generate_tab_data(
                best_model, predictions, task_type, dataset_name, dataset
            )

            if report_data is None:
                print("⚠️ Could not generate tab data for best model")
                return None

            # Create filename
            filename = self._create_report_filename(best_model, task_type)
            report_path = os.path.join(save_path, filename)

            # Save the report
            self._save_tab_report(report_data, report_path, task_type)

            # print(f"📊 Generated tab report: {filename}")
            return report_path

        except Exception as e:
            print(f"⚠️ Error generating tab report: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _detect_task_type(self, predictions: Predictions) -> Optional[TaskType]:
        """Detect task type from available predictions."""
        try:
            pred_data = predictions.get_predictions()
            if not pred_data:
                return None

            # Get first prediction to detect task type
            first_pred = next(iter(pred_data.values()))
            y_true = first_pred.get('y_true', [])
            if len(y_true) == 0:
                return None

            return ModelUtils.detect_task_type(np.array(y_true))
        except Exception as e:
            print(f"⚠️ Error detecting task type: {e}")
            return None

    def _generate_tab_data(
        self,
        best_model: Dict[str, Any],
        predictions: Predictions,
        task_type: TaskType,
        dataset_name: str,
        dataset=None
    ) -> Optional[Dict[str, Any]]:
        """Generate tab data for the best model."""
        try:
            # Extract model information
            canonical_model = best_model.get('canonical_model', 'unknown')
            enhanced_model_name = best_model.get('enhanced_model_name', canonical_model)

            # Get all predictions for this model to reconstruct metrics
            model_predictions = self._get_model_predictions(
                best_model, predictions, canonical_model
            )

            if not model_predictions or not any(model_predictions.values()):
                print("⚠️ No model predictions found for tab report")
                return None

            # Extract nfeatures directly from dataset
            nfeatures = 0
            if dataset:
                nfeatures = self._extract_nfeatures_from_dataset(dataset)
            if nfeatures == 0:
                nfeatures = best_model.get("metadata", {}).get("n_features", 0)

            # Calculate metrics for different partitions
            metrics_data = self._calculate_partition_metrics(
                model_predictions, task_type, dataset_name, dataset
            )

            # Create tab data structure
            tab_data = {
                'model_name': enhanced_model_name,
                'canonical_model': canonical_model,
                'task_type': task_type,
                'dataset_name': dataset_name,
                'nfeatures': nfeatures,
                'metrics': metrics_data
            }

            return tab_data

        except Exception as e:
            print(f"⚠️ Error generating tab data: {e}")
            return None

    def _get_model_predictions(
        self,
        best_model: Dict[str, Any],
        predictions: Predictions,
        canonical_model: str
    ) -> Dict[str, List[Dict]]:
        """Get all predictions for the best model across different partitions."""
        try:
            all_predictions = predictions.get_predictions()
            model_predictions = {
                'train': [],
                'val': [],
                'test': []
            }

            # Find all predictions related to this model
            for key, pred_data in all_predictions.items():
                # Check if this prediction belongs to our best model
                pred_canonical = pred_data.get('model', '')
                pred_real_model = pred_data.get('real_model', '')

                # Extract canonical model name from prediction
                if pred_real_model:
                    pred_canonical_extracted = self._extract_canonical_model_name(pred_real_model)
                else:
                    pred_canonical_extracted = self._extract_canonical_model_name(pred_canonical)

                if pred_canonical_extracted == canonical_model:
                    partition = pred_data.get('partition', '')
                    if partition in ['train', 'val', 'test']:
                        model_predictions[partition].append(pred_data)

            return model_predictions

        except Exception as e:
            print(f"⚠️ Error getting model predictions: {e}")
            return {}

    def _extract_nfeatures_from_dataset(self, dataset) -> int:
        """Extract number of features directly from dataset."""
        try:
            if dataset is None:
                return 0

            # For SpectroDataset, use methods with selectors
            if hasattr(dataset, 'x') and callable(dataset.x):
                try:
                    # Try with numeric selectors (most reliable)
                    for selector in [0, 1, 'train', 'test', 'all']:
                        try:
                            x_data = dataset.x(selector)
                            if hasattr(x_data, 'shape') and len(x_data.shape) > 1:
                                return x_data.shape[1]
                            elif hasattr(x_data, 'shape'):
                                return x_data.shape[0]
                        except Exception:
                            continue
                except Exception:
                    pass

            # If dataset method fails, try to reload a fresh dataset with same name
            if hasattr(dataset, 'name'):
                try:
                    from nirs4all.dataset import DatasetConfigs
                    fresh_config = DatasetConfigs(f"../../sample_data/{dataset.name}")
                    for config, name in fresh_config.configs:
                        if name == dataset.name:
                            fresh_dataset = fresh_config.get_dataset(config, name)
                            if hasattr(fresh_dataset, 'x') and callable(fresh_dataset.x):
                                for selector in [0, 1]:
                                    try:
                                        x_data = fresh_dataset.x(selector)
                                        if hasattr(x_data, 'shape') and len(x_data.shape) > 1:
                                            return x_data.shape[1]
                                    except Exception:
                                        continue
                            break
                except Exception:
                    pass

            # Try to get features from the dataset (non-callable attributes)
            if hasattr(dataset, 'features') and dataset.features is not None:
                if hasattr(dataset.features, 'shape'):
                    return dataset.features.shape[1] if len(dataset.features.shape) > 1 else dataset.features.shape[0]
                elif hasattr(dataset.features, '__len__'):
                    return len(dataset.features)

            # Try alternative methods
            if hasattr(dataset, 'X') and dataset.X is not None:
                if hasattr(dataset.X, 'shape'):
                    return dataset.X.shape[1] if len(dataset.X.shape) > 1 else dataset.X.shape[0]

            # Fallback to any available data shape
            for attr in ['data', 'training_data']:
                if hasattr(dataset, attr):
                    data = getattr(dataset, attr)
                    if data is not None and hasattr(data, 'shape') and len(data.shape) > 1:
                        return data.shape[1]

            return 0
        except Exception as e:
            print(f"⚠️ Error extracting nfeatures from dataset: {e}")
            return 0

    def _extract_canonical_model_name(self, model_name: str) -> str:
        """Extract canonical model name from full model name."""
        if not model_name:
            return 'unknown'

        # Remove step and fold information first
        canonical = model_name
        if '_step' in canonical:
            canonical = canonical.split('_step')[0]

        # Remove fold information
        if '_fold' in canonical:
            canonical = canonical.split('_fold')[0]

        # Remove avg/weighted_avg suffixes
        canonical = canonical.replace('_avg', '').replace('_weighted', '')

        # Handle different model naming patterns:
        # For our naming like "PLS-20_cp_9" -> "PLS-20_cp" (remove only the final numeric counter)
        # For class names like "PLSRegression_10" -> "PLSRegression" (remove parameter suffix)

        parts = canonical.split('_')

        # If it contains a dash (custom name pattern like "PLS-20_cp_9")
        if '-' in canonical:
            # Look for the pattern where the last part is a numeric counter
            # but NOT a parameter (like the "20" in "PLS-20")
            if len(parts) >= 3 and parts[-1].isdigit():
                # Only remove if it's clearly a counter (not a model parameter)
                # Check if it's a pattern like "PLS-20_cp_9" where "9" is the counter
                if len(parts) >= 3 and parts[-2] in ['cp', 'component', 'comp']:
                    canonical = '_'.join(parts[:-1])
                # Otherwise keep the full name for patterns like "PLS-20"
        # If it's a simple class name pattern like "PLSRegression_10"
        elif '-' not in canonical and len(parts) >= 2 and parts[-1].isdigit():
            # For sklearn class names, remove the final numeric parameter
            canonical = parts[0]

        return canonical

    def _calculate_partition_metrics(
        self,
        model_predictions: Dict[str, List[Dict]],
        task_type: TaskType,
        dataset_name: str,
        dataset=None
    ) -> Dict[str, Dict]:
        """Calculate metrics for each partition (train, val, test)."""
        partition_metrics = {}

        for partition, pred_list in model_predictions.items():
            if not pred_list:
                partition_metrics[partition] = {}
                continue

            if partition == 'val':
                # For validation, we need to handle cross-validation reconstruction
                partition_metrics[partition] = self._reconstruct_cv_metrics(
                    pred_list, task_type
                )
            else:
                # For train and test, aggregate if multiple predictions exist
                partition_metrics[partition] = self._aggregate_partition_metrics(
                    pred_list, task_type
                )

        return partition_metrics

    def _reconstruct_cv_metrics(
        self,
        val_predictions: List[Dict],
        task_type: TaskType
    ) -> Dict[str, float]:
        """Reconstruct cross-validation metrics from validation predictions."""
        try:
            if not val_predictions:
                return {}

            # Aggregate all validation predictions
            all_y_true = []
            all_y_pred = []

            for pred_data in val_predictions:
                y_true = np.array(pred_data.get('y_true', []), dtype=np.float32).flatten()
                y_pred = np.array(pred_data.get('y_pred', []), dtype=np.float32).flatten()

                if len(y_true) > 0 and len(y_pred) > 0:
                    all_y_true.extend(y_true)
                    all_y_pred.extend(y_pred)

            if not all_y_true or not all_y_pred:
                return {}

            # Calculate aggregated metrics with all required metrics
            y_true_array = np.array(all_y_true)
            y_pred_array = np.array(all_y_pred)

            if task_type == TaskType.REGRESSION:
                metrics = ModelUtils.calculate_scores(
                    y_true_array,
                    y_pred_array,
                    task_type,
                    metrics=['mse', 'mae', 'r2', 'rmse']  # Request specific metrics
                )
            else:
                metrics = ModelUtils.calculate_scores(
                    y_true_array,
                    y_pred_array,
                    task_type,
                    metrics=['accuracy', 'precision', 'recall', 'f1', 'auc']  # Request specific metrics
                )

            # Add additional statistics
            metrics.update(self._calculate_additional_stats(y_true_array, y_pred_array, task_type))

            return metrics

        except Exception as e:
            print(f"⚠️ Error reconstructing CV metrics: {e}")
            return {}

    def _aggregate_partition_metrics(
        self,
        partition_predictions: List[Dict],
        task_type: TaskType
    ) -> Dict[str, float]:
        """Aggregate metrics for a single partition (train or test)."""
        try:
            if not partition_predictions:
                return {}

            # If multiple predictions, we need to handle them correctly based on partition
            if len(partition_predictions) == 1:
                pred_data = partition_predictions[0]
                y_true = np.array(pred_data.get('y_true', []), dtype=np.float32).flatten()
                y_pred = np.array(pred_data.get('y_pred', []), dtype=np.float32).flatten()
            else:
                # For multiple predictions, determine aggregation strategy
                first_pred = partition_predictions[0]
                first_y_true = np.array(first_pred.get('y_true', []), dtype=np.float32).flatten()

                # Check if all predictions have the same y_true (same test set)
                same_test_set = True
                for pred_data in partition_predictions[1:]:
                    pred_y_true = np.array(pred_data.get('y_true', []), dtype=np.float32).flatten()
                    if not np.array_equal(first_y_true, pred_y_true):
                        same_test_set = False
                        break

                if same_test_set:
                    # Same test set: exclude avg/w_avg predictions to avoid double counting
                    # Only use fold predictions (not virtual avg models)
                    fold_predictions = []
                    for pred_data in partition_predictions:
                        fold_idx = pred_data.get('fold_idx', '')
                        # Skip avg and w_avg virtual models, only include actual fold predictions
                        if fold_idx not in ['avg', 'w_avg', 'w-avg']:
                            fold_predictions.append(pred_data)

                    if fold_predictions:
                        y_true = first_y_true
                        all_y_pred = []
                        for pred_data in fold_predictions:
                            pred_y_pred = np.array(pred_data.get('y_pred', []), dtype=np.float32).flatten()
                            if len(pred_y_pred) > 0:
                                all_y_pred.append(pred_y_pred)

                        if all_y_pred:
                            y_pred = np.mean(all_y_pred, axis=0)  # Average predictions
                        else:
                            y_pred = np.array([])
                    else:
                        # Fallback to first prediction if no fold predictions found
                        y_true = first_y_true
                        y_pred = np.array(first_pred.get('y_pred', []), dtype=np.float32).flatten()
                else:
                    # Different test sets: concatenate (for train/val from different folds)
                    all_y_true = []
                    all_y_pred = []
                    for pred_data in partition_predictions:
                        pred_y_true = np.array(pred_data.get('y_true', []), dtype=np.float32).flatten()
                        pred_y_pred = np.array(pred_data.get('y_pred', []), dtype=np.float32).flatten()
                        if len(pred_y_true) > 0 and len(pred_y_pred) > 0:
                            all_y_true.extend(pred_y_true)
                            all_y_pred.extend(pred_y_pred)

                    y_true = np.array(all_y_true)
                    y_pred = np.array(all_y_pred)

            if len(y_true) == 0 or len(y_pred) == 0:
                return {}

            # Calculate metrics with all required metrics
            if task_type == TaskType.REGRESSION:
                metrics = ModelUtils.calculate_scores(
                    y_true, y_pred, task_type,
                    metrics=['mse', 'mae', 'r2', 'rmse']  # Request specific metrics
                )
            else:
                metrics = ModelUtils.calculate_scores(
                    y_true, y_pred, task_type,
                    metrics=['accuracy', 'precision', 'recall', 'f1', 'auc']  # Request specific metrics
                )

            # Add additional statistics
            metrics.update(self._calculate_additional_stats(y_true, y_pred, task_type))

            return metrics

        except Exception as e:
            print(f"⚠️ Error aggregating partition metrics: {e}")
            return {}

    def _calculate_additional_stats(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: TaskType
    ) -> Dict[str, float]:
        """Calculate additional statistics for the tab report."""
        try:
            stats = {}

            if task_type == TaskType.REGRESSION:
                # Calculate regression-specific statistics
                residuals = y_true - y_pred

                # Basic statistics
                stats['nsample'] = len(y_true)
                stats['mean'] = float(np.mean(y_true))
                stats['median'] = float(np.median(y_true))
                stats['min'] = float(np.min(y_true)) if len(y_true) > 0 else 0.0
                stats['max'] = float(np.max(y_true))
                stats['sd'] = float(np.std(y_true))
                stats['cv'] = float(np.std(y_true) / np.mean(y_true)) if np.mean(y_true) != 0 else 0.0

                # Prediction-specific statistics
                stats['bias'] = float(np.mean(residuals))
                stats['sep'] = float(np.std(residuals))  # Standard Error of Prediction

                # RPD (Ratio of Performance to Deviation)
                if stats['sep'] != 0:
                    stats['rpd'] = float(stats['sd'] / stats['sep'])
                else:
                    stats['rpd'] = float('inf')

                # Q-Value (a measure of prediction quality)
                # Q = sqrt(1 - R²) / R²
                # Get R² from the calculated metrics (assuming it exists)
                r2 = 0.0  # Default value
                if len(y_true) > 1:  # Need at least 2 points to calculate R²
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    if ss_tot != 0:
                        r2 = 1 - (ss_res / ss_tot)

                if r2 > 0 and r2 < 1:
                    stats['q_value'] = float(np.sqrt(1 - r2) / r2)
                else:
                    stats['q_value'] = float('inf')

                # Consistency (percentage of predictions within acceptable range)
                # Define acceptable range as ±1 standard deviation
                acceptable_range = stats['sd']
                within_range = np.abs(residuals) <= acceptable_range
                stats['consistency'] = float(np.sum(within_range) / len(residuals) * 100)

            else:  # Classification
                # For classification, add sample count (actual count, not accumulated)
                stats['nsample'] = len(y_true)

                # Calculate class-specific metrics if needed
                unique_classes = np.unique(y_true)
                stats['n_classes'] = len(unique_classes)

                # Calculate specificity and AUC
                try:
                    from sklearn.metrics import confusion_matrix, roc_auc_score

                    # Convert predictions to class labels if needed
                    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                        y_pred_class = np.argmax(y_pred, axis=1)
                        y_pred_proba = y_pred[:, 1] if y_pred.shape[1] == 2 else y_pred.max(axis=1)
                    else:
                        y_pred_class = np.round(y_pred).astype(int)
                        y_pred_proba = y_pred

                    y_true_class = np.round(y_true).astype(int)

                    # Calculate specificity
                    cm = confusion_matrix(y_true_class, y_pred_class)
                    if len(unique_classes) == 2 and cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        if (tn + fp) > 0:
                            stats['specificity'] = float(tn / (tn + fp))
                        else:
                            stats['specificity'] = 0.0

                        # Calculate AUC for binary classification
                        try:
                            if len(np.unique(y_true_class)) == 2:
                                stats['auc'] = float(roc_auc_score(y_true_class, y_pred_proba))
                            else:
                                stats['auc'] = 0.0
                        except Exception:
                            stats['auc'] = 0.0
                    else:
                        # Multi-class: calculate macro-averaged specificity
                        specificities = []
                        for i, class_label in enumerate(unique_classes):
                            # One-vs-rest specificity
                            y_true_binary = (y_true_class == class_label).astype(int)
                            y_pred_binary = (y_pred_class == class_label).astype(int)
                            cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
                            if cm_binary.shape == (2, 2):
                                tn, fp, fn, tp = cm_binary.ravel()
                                if (tn + fp) > 0:
                                    specificities.append(tn / (tn + fp))

                        if specificities:
                            stats['specificity'] = float(np.mean(specificities))
                        else:
                            stats['specificity'] = 0.0

                        # Multi-class AUC (macro-averaged)
                        try:
                            if len(unique_classes) > 2:
                                stats['auc'] = float(roc_auc_score(y_true_class, y_pred_proba if len(y_pred.shape) == 1 else y_pred, multi_class='ovr', average='macro'))
                            else:
                                stats['auc'] = 0.0
                        except Exception:
                            stats['auc'] = 0.0

                except Exception as e:
                    print(f"⚠️ Error calculating classification metrics: {e}")
                    stats['specificity'] = 0.0
                    stats['auc'] = 0.0

            return stats

        except Exception as e:
            print(f"⚠️ Error calculating additional stats: {e}")
            return {}

    def _create_report_filename(
        self,
        best_model: Dict[str, Any],
        task_type: TaskType
    ) -> str:
        """Create a filename for the tab report."""
        try:
            # Get model name with counter
            enhanced_name = best_model.get('enhanced_model_name', 'unknown')
            real_model = best_model.get('real_model', '')

            # Extract counter from real_model if available
            model_name = enhanced_name
            if real_model and '_' in real_model:
                parts = real_model.split('_')
                # Look for numeric counter at the end
                if parts[-1].isdigit():
                    counter = parts[-1]
                    # Add counter if not already in enhanced name
                    if not enhanced_name.endswith(f'_{counter}'):
                        model_name = f"{enhanced_name}_{counter}"

            # Clean the model name for filename
            clean_name = model_name.replace(' ', '_').replace('[', '').replace(']', '')
            clean_name = clean_name.replace('(', '').replace(')', '').replace('/', '_')

            # Create filename
            filename = f"best_score_report_{clean_name}.csv"

            return filename

        except Exception as e:
            print(f"⚠️ Error creating filename: {e}")
            return "best_score_report_unknown.csv"

    def _save_tab_report(
        self,
        tab_data: Dict[str, Any],
        report_path: str,
        task_type: TaskType
    ) -> None:
        """Save the tab report as a CSV file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(report_path), exist_ok=True)

            metrics = tab_data['metrics']

            if task_type == TaskType.REGRESSION:
                self._save_regression_tab(metrics, report_path, tab_data)
            else:  # Classification
                self._save_classification_tab(metrics, report_path, tab_data)

        except Exception as e:
            print(f"⚠️ Error saving tab report: {e}")

    def _save_regression_tab(self, metrics: Dict[str, Dict], report_path: str, tab_data: Dict[str, Any]) -> None:
        """Save regression tab report."""
        # Define column headers for regression (removed Q-Value)
        headers = [
            '', 'Nsample', 'Nfeature', 'Mean', 'Median', 'Min', 'Max', 'SD', 'CV',
            'R²', 'RMSE', 'MSE', 'SEP', 'MAE', 'RPD', 'Bias', 'Consistency (%)'
        ]

        # Create rows
        rows = []

        # Cross-validation row
        cv_metrics = metrics.get('val', {})
        nfeatures = tab_data.get('nfeatures', '')
        cv_row = ['Cros Val']
        cv_row.extend([
            cv_metrics.get('nsample', ''),
            nfeatures if nfeatures > 0 else '',
            f"{cv_metrics.get('mean', ''):.3f}" if cv_metrics.get('mean') is not None else '',
            f"{cv_metrics.get('median', ''):.3f}" if cv_metrics.get('median') is not None else '',
            f"{cv_metrics.get('min', ''):.3f}" if cv_metrics.get('min') is not None else '',
            f"{cv_metrics.get('max', ''):.3f}" if cv_metrics.get('max') is not None else '',
            f"{cv_metrics.get('sd', ''):.3f}" if cv_metrics.get('sd') else '',
            f"{cv_metrics.get('cv', ''):.3f}" if cv_metrics.get('cv') else '',
            f"{cv_metrics.get('r2', ''):.3f}" if cv_metrics.get('r2') else '',
            f"{cv_metrics.get('rmse', ''):.3f}" if cv_metrics.get('rmse') else '',
            f"{cv_metrics.get('mse', ''):.3f}" if cv_metrics.get('mse') else '',  # Added MSE
            f"{cv_metrics.get('sep', ''):.3f}" if cv_metrics.get('sep') else '',
            f"{cv_metrics.get('mae', ''):.3f}" if cv_metrics.get('mae') else '',
            f"{cv_metrics.get('rpd', ''):.2f}" if cv_metrics.get('rpd') and cv_metrics.get('rpd') != float('inf') else '',
            f"{cv_metrics.get('bias', ''):.3f}" if cv_metrics.get('bias') else '',
            f"{cv_metrics.get('consistency', ''):.1f}" if cv_metrics.get('consistency') else ''
        ])
        rows.append(cv_row)

        # Train row
        train_metrics = metrics.get('train', {})
        train_row = ['Train']
        train_row.extend([
            train_metrics.get('nsample', ''),
            nfeatures if nfeatures > 0 else '',  # Add NFeatures for train
            f"{train_metrics.get('mean', ''):.3f}" if train_metrics.get('mean') is not None else '',
            f"{train_metrics.get('median', ''):.3f}" if train_metrics.get('median') is not None else '',
            f"{train_metrics.get('min', ''):.3f}" if train_metrics.get('min') is not None else '',
            f"{train_metrics.get('max', ''):.3f}" if train_metrics.get('max') is not None else '',
            f"{train_metrics.get('sd', ''):.3f}" if train_metrics.get('sd') else '',
            f"{train_metrics.get('cv', ''):.3f}" if train_metrics.get('cv') else '',
            f"{train_metrics.get('r2', ''):.3f}" if train_metrics.get('r2') else '',
            f"{train_metrics.get('rmse', ''):.3f}" if train_metrics.get('rmse') else '',
            f"{train_metrics.get('mse', ''):.3f}" if train_metrics.get('mse') else '',  # Added MSE
            f"{train_metrics.get('sep', ''):.3f}" if train_metrics.get('sep') else '',
            f"{train_metrics.get('mae', ''):.3f}" if train_metrics.get('mae') else '',
            f"{train_metrics.get('rpd', ''):.2f}" if train_metrics.get('rpd') and train_metrics.get('rpd') != float('inf') else '',
            f"{train_metrics.get('bias', ''):.3f}" if train_metrics.get('bias') else '',
            f"{train_metrics.get('consistency', ''):.1f}" if train_metrics.get('consistency') else ''
        ])
        rows.append(train_row)

        # Test row
        test_metrics = metrics.get('test', {})
        test_row = ['Test']
        test_row.extend([
            test_metrics.get('nsample', ''),
            nfeatures if nfeatures > 0 else '',  # Add NFeatures for test
            f"{test_metrics.get('mean', ''):.3f}" if test_metrics.get('mean') is not None else '',
            f"{test_metrics.get('median', ''):.3f}" if test_metrics.get('median') is not None else '',
            f"{test_metrics.get('min', ''):.3f}" if test_metrics.get('min') is not None else '',
            f"{test_metrics.get('max', ''):.3f}" if test_metrics.get('max') is not None else '',
            f"{test_metrics.get('sd', ''):.3f}" if test_metrics.get('sd') else '',
            f"{test_metrics.get('cv', ''):.3f}" if test_metrics.get('cv') else '',
            f"{test_metrics.get('r2', ''):.3f}" if test_metrics.get('r2') else '',
            f"{test_metrics.get('rmse', ''):.3f}" if test_metrics.get('rmse') else '',
            f"{test_metrics.get('mse', ''):.3f}" if test_metrics.get('mse') else '',  # Added MSE
            f"{test_metrics.get('sep', ''):.3f}" if test_metrics.get('sep') else '',
            f"{test_metrics.get('mae', ''):.3f}" if test_metrics.get('mae') else '',
            f"{test_metrics.get('rpd', ''):.2f}" if test_metrics.get('rpd') and test_metrics.get('rpd') != float('inf') else '',
            f"{test_metrics.get('bias', ''):.3f}" if test_metrics.get('bias') else '',
            f"{test_metrics.get('consistency', ''):.1f}" if test_metrics.get('consistency') else ''
        ])
        rows.append(test_row)

        # Write CSV
        with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(rows)

    def _save_classification_tab(self, metrics: Dict[str, Dict], report_path: str, tab_data: Dict[str, Any]) -> None:
        """Save classification tab report."""
        # Check if this is binary classification to determine if AUC should be included
        is_binary = self._is_binary_classification(metrics)

        # Define column headers for classification (conditionally include AUC)
        if is_binary:
            headers = ['', 'Nsample', 'Nfeatures', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Specificity', 'AUC']
        else:
            headers = ['', 'Nsample', 'Nfeatures', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Specificity']

        # Create rows
        rows = []

        nfeatures = tab_data.get('nfeatures', '')

        # Cross-validation row
        cv_metrics = metrics.get('val', {})
        cv_row = ['Cros Val']
        cv_row.extend([
            cv_metrics.get('nsample', ''),
            nfeatures if nfeatures > 0 else '',
            f"{cv_metrics.get('accuracy', ''):.3f}" if cv_metrics.get('accuracy') else '',
            f"{cv_metrics.get('precision', ''):.3f}" if cv_metrics.get('precision') else '',
            f"{cv_metrics.get('recall', ''):.3f}" if cv_metrics.get('recall') else '',
            f"{cv_metrics.get('f1', ''):.3f}" if cv_metrics.get('f1') else '',
            f"{cv_metrics.get('specificity', ''):.3f}" if cv_metrics.get('specificity') else ''
        ])
        if is_binary:
            cv_row.append(f"{cv_metrics.get('auc', ''):.3f}" if cv_metrics.get('auc') else '')
        rows.append(cv_row)

        # Train row
        train_metrics = metrics.get('train', {})
        train_row = ['Train']
        train_row.extend([
            train_metrics.get('nsample', ''),
            nfeatures if nfeatures > 0 else '',
            f"{train_metrics.get('accuracy', ''):.3f}" if train_metrics.get('accuracy') else '',
            f"{train_metrics.get('precision', ''):.3f}" if train_metrics.get('precision') else '',
            f"{train_metrics.get('recall', ''):.3f}" if train_metrics.get('recall') else '',
            f"{train_metrics.get('f1', ''):.3f}" if train_metrics.get('f1') else '',
            f"{train_metrics.get('specificity', ''):.3f}" if train_metrics.get('specificity') else ''
        ])
        if is_binary:
            train_row.append(f"{train_metrics.get('auc', ''):.3f}" if train_metrics.get('auc') else '')
        rows.append(train_row)

        # Test row
        test_metrics = metrics.get('test', {})
        test_row = ['Test']
        test_row.extend([
            test_metrics.get('nsample', ''),
            nfeatures if nfeatures > 0 else '',
            f"{test_metrics.get('accuracy', ''):.3f}" if test_metrics.get('accuracy') else '',
            f"{test_metrics.get('precision', ''):.3f}" if test_metrics.get('precision') else '',
            f"{test_metrics.get('recall', ''):.3f}" if test_metrics.get('recall') else '',
            f"{test_metrics.get('f1', ''):.3f}" if test_metrics.get('f1') else '',
            f"{test_metrics.get('specificity', ''):.3f}" if test_metrics.get('specificity') else ''
        ])
        if is_binary:
            test_row.append(f"{test_metrics.get('auc', ''):.3f}" if test_metrics.get('auc') else '')
        rows.append(test_row)

        # Write CSV
        with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(rows)

    def _is_binary_classification(self, metrics: Dict[str, Dict]) -> bool:
        """Check if this is binary classification based on available metrics."""
        # Check if AUC is available in any partition (AUC is typically only calculated for binary)
        for partition_metrics in metrics.values():
            if partition_metrics.get('auc') is not None:
                return True
        return False
