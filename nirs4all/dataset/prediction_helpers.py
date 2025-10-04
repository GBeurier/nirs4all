"""
Prediction helper utilities for SpectroDataset.

This module contains utility functions for working with predictions,
including score calculations, ranking, combining predictions, and display functions.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


class PredictionHelpers:
    """Helper utilities for prediction analysis and manipulation."""

    @staticmethod
    def calculate_scores_for_predictions(
        predictions: Dict[str, Dict[str, Any]],
        task_type: str = "auto"
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate scores for predictions with automatic task type detection.

        Args:
            predictions: Dictionary of predictions to score
            task_type: Task type ("regression", "binary_classification", "multiclass_classification", or "auto")

        Returns:
            Dict mapping prediction keys to their calculated scores
        """
        from nirs4all.utils.model_utils import ModelUtils, TaskType

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

    @staticmethod
    def get_scores_ranking(
        predictions: Dict[str, Dict[str, Any]],
        scores_dict: Dict[str, Dict[str, float]],
        metric: str,
        ascending: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Get predictions ranked by a specific metric score.

        Args:
            predictions: Dictionary of predictions
            scores_dict: Pre-calculated scores dictionary
            metric: Metric name to rank by (e.g., 'mse', 'accuracy', 'f1')
            ascending: If True, lower scores rank higher (for error metrics)

        Returns:
            List of (prediction_key, score) tuples sorted by score
        """
        rankings = []
        for key, scores in scores_dict.items():
            if metric in scores:
                rankings.append((key, scores[metric]))

        # Sort by score
        rankings.sort(key=lambda x: x[1], reverse=not ascending)

        return rankings

    @staticmethod
    def get_best_score(
        predictions: Dict[str, Dict[str, Any]],
        scores_dict: Dict[str, Dict[str, float]],
        metric: str,
        task_type: str = "auto"
    ) -> Optional[Tuple[str, float]]:
        """
        Get the best (lowest or highest depending on metric) score for a metric.

        Args:
            predictions: Dictionary of predictions
            scores_dict: Pre-calculated scores dictionary
            metric: Metric name
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

        rankings = PredictionHelpers.get_scores_ranking(
            predictions, scores_dict, metric, ascending
        )

        return rankings[0] if rankings else None

    @staticmethod
    def get_all_scores_summary(
        predictions: Dict[str, Dict[str, Any]],
        scores_dict: Dict[str, Dict[str, float]],
        run_path: str = ""
    ) -> pd.DataFrame:
        """
        Get a comprehensive summary of all scores across all partitions.

        Args:
            predictions: Dictionary of predictions
            scores_dict: Pre-calculated scores dictionary
            run_path: Path information for the run

        Returns:
            DataFrame with scores for all predictions
        """
        rows = []
        for key, scores in scores_dict.items():
            pred_data = predictions[key]

            row = {
                'prediction_key': key,
                'dataset': pred_data['dataset'],
                'pipeline': pred_data['pipeline'],
                'model': pred_data['model'],
                'partition': pred_data['partition'],
                'fold_idx': pred_data.get('fold_idx'),
                'n_samples': len(pred_data['y_true']),
                'path': run_path
            }

            # Add all scores
            row.update(scores)
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def combine_folds(
        predictions: Dict[str, Dict[str, Any]],
        dataset: str,
        pipeline: str,
        model: str,
        partition_pattern: str = "val_fold",
        run_path: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Combine predictions from multiple folds.

        Args:
            predictions: Dictionary of all predictions
            dataset: Dataset name
            pipeline: Pipeline name
            model: Model name
            partition_pattern: Pattern to match (e.g., "val_fold" for validation folds)
            run_path: Path information for the run

        Returns:
            Combined prediction data or None if no matching folds found
        """
        matching_predictions = []

        for key, pred_data in predictions.items():
            matches = pred_data['dataset'] == dataset and pred_data['pipeline'] == pipeline and pred_data['model'] == model and partition_pattern in pred_data['partition']
            if matches:
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
            'path': run_path
        }

    @staticmethod
    def calculate_average_predictions(
        predictions: Dict[str, Dict[str, Any]],
        dataset: str,
        pipeline: str,
        model: str,
        partition: str,
        run_path: str = ""
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Calculate average predictions across folds for the same partition with new schema.

        Args:
            predictions: Dictionary of all predictions
            dataset: Dataset name
            pipeline: Pipeline name
            model: Base model class name
            partition: Partition name ('train', 'val', 'test')
            run_path: Path information for the run
            store_result: Whether to store the result in predictions

        Returns:
            Tuple of (result_dict, key) or (None, None) if no matching folds found
        """
        # Find all fold predictions for this model and partition
        fold_predictions = []
        for key, pred_data in predictions.items():
            condition = (pred_data['dataset'] == dataset and
                        pred_data['pipeline'] == pipeline and
                        pred_data['model'] == model and
                        pred_data['partition'] == partition and
                        isinstance(pred_data.get('fold_idx'), int))
            if condition:
                fold_predictions.append(pred_data)

        if len(fold_predictions) < 2:
            return None, None

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
            'path': run_path
        }

        key = f"{dataset}_{pipeline}_{avg_real_model}_{partition}_fold_avg"
        return result, key

    @staticmethod
    def calculate_weighted_average_predictions(
        predictions: Dict[str, Dict[str, Any]],
        dataset: str,
        pipeline: str,
        model: str,
        test_partition: str = "test",
        val_partition: str = "val",
        metric: str = 'rmse',
        run_path: str = ""
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Calculate weighted average predictions based on validation performance with new schema.

        Args:
            predictions: Dictionary of all predictions
            dataset: Dataset name
            pipeline: Pipeline name
            model: Base model class name
            test_partition: Partition name for test predictions to average
            val_partition: Partition name for validation predictions to use for weighting
            metric: Metric to use for weighting ('rmse', 'mae', 'r2')
            run_path: Path information for the run
            store_result: Whether to store the result in predictions

        Returns:
            Tuple of (result_dict, key) or (None, None) if insufficient data
        """
        # Find test and validation predictions for each fold
        test_predictions = []
        val_predictions = []

        for key, pred_data in predictions.items():
            if (pred_data['dataset'] == dataset and
                pred_data['pipeline'] == pipeline and
                pred_data['model'] == model and
                isinstance(pred_data.get('fold_idx'), int)):  # Only numeric fold indices

                if pred_data['partition'] == test_partition:
                    test_predictions.append(pred_data)
                elif pred_data['partition'] == val_partition:
                    val_predictions.append(pred_data)

        if len(test_predictions) < 2 or len(val_predictions) < 2:
            return None, None

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
            'path': run_path
        }

        key = f"{dataset}_{pipeline}_{weighted_avg_real_model}_{test_partition}_fold_weighted_avg"
        return result, key

    @staticmethod
    def save_predictions_to_csv(
        predictions: Dict[str, Dict[str, Any]],
        filepath: str,
        scores_dict: Optional[Dict[str, Dict[str, float]]] = None,
        run_path: str = "",
        task_type: str = "auto"
    ) -> None:
        """
        Save predictions to CSV file.

        Args:
            predictions: Dictionary of predictions to save
            filepath: Output CSV file path
            scores_dict: Pre-calculated scores dictionary (optional)
            run_path: Path information for the run
            task_type: Task type for score calculation if scores_dict not provided
        """
        if not predictions:
            print("No predictions found matching the criteria")
            return

        all_rows = []

        # Calculate scores if requested and not provided
        if scores_dict is None:
            scores_dict = PredictionHelpers.calculate_scores_for_predictions(
                predictions, task_type
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
                    'path': run_path
                }

                # Add scores (same for all samples from same prediction)
                if scores_dict and key in scores_dict:
                    row.update(scores_dict[key])

                all_rows.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(all_rows)
        df.to_csv(filepath, index=False)
        # print(f"💾 Saved {len(all_rows)} prediction records to {filepath}")

    @staticmethod
    def print_best_scores_summary(
        predictions: Dict[str, Dict[str, Any]],
        pipeline: Optional[str] = None,
        task_type: str = "auto",
        show_current_pipeline_best: bool = True
    ) -> None:
        """
        Print a concise summary of best scores with both overall and current pipeline bests.

        Args:
            predictions: Dictionary of all predictions
            pipeline: Filter by pipeline name (for current pipeline best)
            task_type: Task type for appropriate score calculation
            show_current_pipeline_best: Whether to show current pipeline best score
        """
        from nirs4all.utils.model_utils import ModelUtils, TaskType

        # Get all predictions
        all_predictions = predictions
        current_pipeline_predictions = {
            k: v for k, v in predictions.items()
            if pipeline is None or v['pipeline'] == pipeline
        } if pipeline else {}

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
        scores_dict = PredictionHelpers.calculate_scores_for_predictions(all_predictions, task_type)
        all_rankings = PredictionHelpers.get_scores_ranking(
            all_predictions, scores_dict, best_metric, ascending=not higher_is_better
        )

        if not all_rankings:
            print("No scores calculated")
            return

        # Find current pipeline rankings
        current_rankings = []
        if show_current_pipeline_best and pipeline and current_pipeline_predictions:
            current_scores_dict = PredictionHelpers.calculate_scores_for_predictions(
                current_pipeline_predictions, task_type
            )
            current_rankings = PredictionHelpers.get_scores_ranking(
                current_pipeline_predictions, current_scores_dict,
                best_metric, ascending=not higher_is_better
            )

        # Print best overall
        if all_rankings:
            best_key, best_score = all_rankings[0]
            pred_data = predictions[best_key]
            # Use custom name if available, otherwise model name
            model_name = pred_data.get('custom_model_name') or pred_data['model']
            partition = pred_data['partition']
            pipeline_name = pred_data['pipeline']
            print(f"🏆 Best Overall: {model_name} ({pipeline_name}/{partition}) = {best_score:.4f} {direction}")

        # Print best from current pipeline
        if current_rankings and show_current_pipeline_best:
            curr_key, curr_score = current_rankings[0]
            curr_pred_data = predictions[curr_key]
            # Use custom name if available, otherwise model name
            curr_model_name = curr_pred_data.get('custom_model_name') or curr_pred_data['model']
            curr_partition = curr_pred_data['partition']
            print(f"🥇 Best This Run: {curr_model_name} ({curr_partition}) = {curr_score:.4f} {direction}")
        elif show_current_pipeline_best:
            print("🥇 Best This Run: No predictions found for current pipeline")

    @staticmethod
    def display_best_scores_summary(
        predictions: Dict[str, Dict[str, Any]],
        dataset_name: str,
        predictions_before_count: int = 0,
        all_keys: Optional[List[str]] = None
    ) -> None:
        """
        Display best scores summary for the dataset after all pipelines are complete.

        Args:
            predictions: Dictionary of all predictions
            dataset_name: Name of the dataset
            predictions_before_count: Number of predictions that existed before this run
            all_keys: List of all prediction keys (optional, will be computed if not provided)
        """
        try:
            from nirs4all.utils.model_utils import ModelUtils

            # Get all predictions for analysis
            if all_keys is None:
                all_keys = list(predictions.keys())

            if len(all_keys) == 0:
                print("No predictions found")
                return

            print("-" * 120)

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
            higher_is_better = False

            # Analyze all predictions to find best scores
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

                        pred_data = predictions.get(key)
                        if not pred_data:
                            continue

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

                # Calculate unscaled score for better display
                unscaled_score = best_this_run_score * 1000 if best_metric in ['mse', 'rmse', 'mae'] else best_this_run_score
                print(f"🏆 Best from config ({clean_config}): {display_name} ({clean_config}) - loss({best_metric})={best_this_run_score:.4f}{direction} - score({best_metric}): {unscaled_score:.4f}")

            if best_overall and best_overall_score is not None:
                # Use the real model name directly (already includes operation counter)
                display_name = best_overall_model

                # Extract clean config name (remove hash and test parts)
                config_part = best_overall.split('_')
                clean_config = '_'.join(config_part[1:4]) if len(config_part) >= 4 else 'unknown'

                # Calculate unscaled score for better display
                unscaled_overall_score = best_overall_score * 1000 if best_metric in ['mse', 'rmse', 'mae'] else best_overall_score

                if predictions_before_count > 0 and best_overall != best_this_run:
                    print(f"🥇 Best overall: {display_name} ({clean_config}) - loss({best_metric})={best_overall_score:.4f}{direction} - score({best_metric}): {unscaled_overall_score:.4f}")
                elif predictions_before_count == 0:
                    # Only show overall if it's different from this run, or if there were no previous predictions
                    print(f"🥇 Best overall: {display_name} ({clean_config}) - loss({best_metric})={best_overall_score:.4f}{direction} - score({best_metric}): {unscaled_overall_score:.4f}")

        except Exception as e:
            print(f"⚠️ Could not display best scores summary: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def generate_best_score_tab_report(
        predictions,  # Type: Predictions - avoid circular import
        dataset_name: str,
        save_path: str,
        enable_tab_reports: bool = True,
        dataset=None
    ) -> Optional[str]:
        """Generate best score tab report using TabReportGenerator."""
        if not enable_tab_reports:
            return None

        try:
            # from nirs4all.utils.tab_report_generator import TabReportGenerator
            report_path = TabReportGenerator.generate_best_score_report(
                predictions, dataset_name, save_path, enable_tab_reports, dataset
            )

            # Print formatted table if report was generated
            if report_path:
                PredictionHelpers._print_formatted_tab_report(report_path)

            return report_path
        except Exception as e:
            print(f"⚠️ Could not generate tab report: {e}")
            return None

    @staticmethod
    def _print_formatted_tab_report(report_path: str) -> None:
        """Print tab report as a formatted table."""
        try:
            import csv
            from pathlib import Path

            # Read the CSV file
            with open(report_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                rows = list(reader)

            if not rows:
                return

            print(f"📊 Tab report saved: {Path(report_path).name}")

            # Calculate column widths (minimum 10 characters per column)
            col_widths = []
            for col_idx in range(len(rows[0])):
                max_width = max(len(str(rows[row_idx][col_idx])) for row_idx in range(len(rows)))
                col_widths.append(max(max_width, 10))

            # Create separator line
            separator = '|' + '|'.join('-' * (width + 2) for width in col_widths) + '|'

            # Print the formatted table
            print(separator)

            # Print header
            header_row = '|' + '|'.join(f" {str(rows[0][j]):<{col_widths[j]}} " for j in range(len(rows[0]))) + '|'
            print(header_row)
            print(separator)

            # Print data rows
            for i in range(1, len(rows)):
                data_row = '|' + '|'.join(f" {str(rows[i][j]):<{col_widths[j]}} " for j in range(len(rows[i]))) + '|'
                print(data_row)

            print(separator)

        except Exception as e:
            print(f"⚠️ Could not format table: {e}")