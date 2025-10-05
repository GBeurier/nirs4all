"""
Tab Report Manager - Simplified tab report generation with formatting and saving

This module provides a clean interface for generating standardized tab-based CSV reports
using pre-calculated metrics and statistics from the evaluator module.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import csv
import os
import io

# Import evaluator functions
import nirs4all.dataset.evaluator as evaluator


class TabReportManager:
    """Generate standardized tab-based CSV reports with pre-calculated data."""

    @staticmethod
    def generate_best_score_tab_report(
        best_by_partition: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, Optional[str]]:
        """
        Generate best score tab report from partition data.

        Args:
            best_by_partition: Dict mapping partition names ('train', 'val', 'test') to prediction entries

        Returns:
            Tuple of (formatted_string, csv_string_content)
        """
        if not best_by_partition:
            return "No prediction data available", None

        # Detect task type from first available prediction
        first_entry = next(iter(best_by_partition.values()))
        task_type = TabReportManager._detect_task_type_from_entry(first_entry)

        # Extract n_features from metadata if available
        n_features = first_entry.get('n_features', 0)

        # Calculate metrics and stats for each partition
        partitions_data = {}

        for partition_name, entry in best_by_partition.items():
            if partition_name in ['train', 'val', 'test']:
                y_true = np.array(entry['y_true'])
                y_pred = np.array(entry['y_pred'])

                partitions_data[partition_name] = TabReportManager._calculate_partition_data(
                    y_true, y_pred, task_type
                )

        # Generate formatted string (matching PredictionHelpers format)
        formatted_string = TabReportManager._format_as_table_string(
            partitions_data, n_features, task_type
        )

        # Generate CSV string content
        csv_string = TabReportManager._generate_csv_string(
            partitions_data, n_features, task_type
        )

        return formatted_string, csv_string

    @staticmethod
    def _detect_task_type_from_entry(entry: Dict[str, Any]) -> str:
        """Detect task type from a prediction entry."""
        y_true = np.array(entry.get('y_true', []))
        if len(y_true) == 0:
            return "regression"

        # Simple heuristic: if all values are integers and < 50 unique values, it's classification
        unique_vals = np.unique(y_true)
        if len(unique_vals) <= 2 and np.allclose(y_true, np.round(y_true)):
            return "binary_classification"
        elif len(unique_vals) <= 50 and np.allclose(y_true, np.round(y_true)):
            return "multiclass_classification"
        else:
            return "regression"

    @staticmethod
    def _format_as_table_string(
        partitions_data: Dict[str, Dict[str, Any]],
        n_features: int,
        task_type: str
    ) -> str:
        """Format the report data as a table string (matching PredictionHelpers format)."""
        if not partitions_data:
            return "No partition data available"

        # Prepare headers based on task type
        if task_type == 'regression':
            headers = ['', 'Nsample', 'Nfeature', 'Mean', 'Median', 'Min', 'Max', 'SD', 'CV',
                       'R²', 'RMSE', 'MSE', 'SEP', 'MAE', 'RPD', 'Bias', 'Consistency']
        else:  # Classification
            is_binary = 'roc_auc' in partitions_data.get('val', {}) or 'roc_auc' in partitions_data.get('test', {})
            if is_binary:
                headers = ['', 'Nsample', 'Nfeatures', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Specificity', 'AUC']
            else:
                headers = ['', 'Nsample', 'Nfeatures', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Specificity']

        # Prepare rows
        rows = []

        # Add partition rows in order: val (Cross Val), train, test
        for partition_name in ['val', 'train', 'test']:
            if partition_name not in partitions_data:
                continue

            data = partitions_data[partition_name]
            display_name = "Cros Val" if partition_name == 'val' else partition_name.capitalize()

            if task_type == 'regression':
                row = [
                    display_name,
                    str(data.get('nsample', '')),
                    str(n_features) if n_features > 0 else '',
                    f"{data.get('mean', ''):.3f}" if data.get('mean') is not None else '',
                    f"{data.get('median', ''):.3f}" if data.get('median') is not None else '',
                    f"{data.get('min', ''):.3f}" if data.get('min') is not None else '',
                    f"{data.get('max', ''):.3f}" if data.get('max') is not None else '',
                    f"{data.get('sd', ''):.3f}" if data.get('sd') else '',
                    f"{data.get('cv', ''):.3f}" if data.get('cv') else '',
                    f"{data.get('r2', ''):.3f}" if data.get('r2') else '',
                    f"{data.get('rmse', ''):.3f}" if data.get('rmse') else '',
                    f"{data.get('mse', ''):.3f}" if data.get('mse') else '',
                    f"{data.get('sep', ''):.3f}" if data.get('sep') else '',
                    f"{data.get('mae', ''):.3f}" if data.get('mae') else '',
                    f"{data.get('rpd', ''):.2f}" if data.get('rpd') and data.get('rpd') != float('inf') else '',
                    f"{data.get('bias', ''):.3f}" if data.get('bias') else '',
                    f"{data.get('consistency', ''):.1f}" if data.get('consistency') else ''
                ]
            else:  # Classification
                row = [
                    display_name,
                    str(data.get('nsample', '')),
                    str(n_features) if n_features > 0 else '',
                    f"{data.get('accuracy', ''):.3f}" if data.get('accuracy') else '',
                    f"{data.get('precision', ''):.3f}" if data.get('precision') else '',
                    f"{data.get('recall', ''):.3f}" if data.get('recall') else '',
                    f"{data.get('f1', ''):.3f}" if data.get('f1') else '',
                    f"{data.get('specificity', ''):.3f}" if data.get('specificity') else ''
                ]
                if is_binary:
                    row.append(f"{data.get('roc_auc', ''):.3f}" if data.get('roc_auc') else '')

            rows.append(row)

        # Calculate column widths (minimum 10 characters per column)
        all_rows = [headers] + rows
        col_widths = []
        for col_idx in range(len(headers)):
            max_width = max(len(str(all_rows[row_idx][col_idx])) for row_idx in range(len(all_rows)))
            col_widths.append(max(max_width, 6))

        # Generate formatted table string
        lines = []

        # Create separator line
        separator = '|' + '|'.join('-' * (width + 2) for width in col_widths) + '|'
        lines.append(separator)

        # Add header
        header_row = '|' + '|'.join(f" {str(headers[j]):<{col_widths[j]}} " for j in range(len(headers))) + '|'
        lines.append(header_row)
        lines.append(separator)

        # Add data rows
        for row in rows:
            data_row = '|' + '|'.join(f" {str(row[j]):<{col_widths[j]}} " for j in range(len(row))) + '|'
            lines.append(data_row)

        lines.append(separator)

        return '\n'.join(lines)

    @staticmethod
    def _generate_csv_string(
        partitions_data: Dict[str, Dict[str, Any]],
        n_features: int,
        task_type: str
    ) -> str:
        """Generate CSV string content."""
        # Prepare headers based on task type
        if task_type == 'regression':
            headers = ['', 'Nsample', 'Nfeature', 'Mean', 'Median', 'Min', 'Max', 'SD', 'CV',
                       'R²', 'RMSE', 'MSE', 'SEP', 'MAE', 'RPD', 'Bias', 'Consistency (%)']
        else:  # Classification
            is_binary = 'roc_auc' in partitions_data.get('val', {}) or 'roc_auc' in partitions_data.get('test', {})
            if is_binary:
                headers = ['', 'Nsample', 'Nfeatures', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Specificity', 'AUC']
            else:
                headers = ['', 'Nsample', 'Nfeatures', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Specificity']

        # Prepare rows
        rows = [headers]

        # Add partition rows in order: val (Cross Val), train, test
        for partition_name in ['val', 'train', 'test']:
            if partition_name not in partitions_data:
                continue

            data = partitions_data[partition_name]
            display_name = "Cros Val" if partition_name == 'val' else partition_name.capitalize()

            if task_type == 'regression':
                row = [
                    display_name,
                    data.get('nsample', ''),
                    n_features if n_features > 0 else '',
                    f"{data.get('mean', ''):.3f}" if data.get('mean') is not None else '',
                    f"{data.get('median', ''):.3f}" if data.get('median') is not None else '',
                    f"{data.get('min', ''):.3f}" if data.get('min') is not None else '',
                    f"{data.get('max', ''):.3f}" if data.get('max') is not None else '',
                    f"{data.get('sd', ''):.3f}" if data.get('sd') else '',
                    f"{data.get('cv', ''):.3f}" if data.get('cv') else '',
                    f"{data.get('r2', ''):.3f}" if data.get('r2') else '',
                    f"{data.get('rmse', ''):.3f}" if data.get('rmse') else '',
                    f"{data.get('mse', ''):.3f}" if data.get('mse') else '',
                    f"{data.get('sep', ''):.3f}" if data.get('sep') else '',
                    f"{data.get('mae', ''):.3f}" if data.get('mae') else '',
                    f"{data.get('rpd', ''):.2f}" if data.get('rpd') and data.get('rpd') != float('inf') else '',
                    f"{data.get('bias', ''):.3f}" if data.get('bias') else '',
                    f"{data.get('consistency', ''):.1f}" if data.get('consistency') else ''
                ]
            else:  # Classification
                row = [
                    display_name,
                    data.get('nsample', ''),
                    n_features if n_features > 0 else '',
                    f"{data.get('accuracy', ''):.3f}" if data.get('accuracy') else '',
                    f"{data.get('precision', ''):.3f}" if data.get('precision') else '',
                    f"{data.get('recall', ''):.3f}" if data.get('recall') else '',
                    f"{data.get('f1', ''):.3f}" if data.get('f1') else '',
                    f"{data.get('specificity', ''):.3f}" if data.get('specificity') else ''
                ]
                if is_binary:
                    row.append(f"{data.get('roc_auc', ''):.3f}" if data.get('roc_auc') else '')

            rows.append(row)

        # Generate CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(rows)

        # Return as string
        csv_content = output.getvalue()
        output.close()

        return csv_content

    @staticmethod
    def _calculate_partition_data(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str
    ) -> Dict[str, Any]:
        """Calculate metrics and statistics for a single partition."""
        # Get descriptive statistics for y_true
        stats = evaluator.get_stats(y_true)

        # Get metrics based on task type
        if task_type.lower() == 'regression':
            metric_names = ['mse', 'rmse', 'mae', 'r2', 'bias', 'sep', 'rpd']
        elif task_type.lower() == 'binary_classification':
            metric_names = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'roc_auc']
        else:  # multiclass_classification
            metric_names = ['accuracy', 'precision', 'recall', 'f1', 'specificity']

        metrics_list = evaluator.eval_list(y_true, y_pred, metric_names)

        # Combine stats and metrics into a single dict
        partition_data = {}
        if stats:
            partition_data.update(stats)

        # Convert metrics list to dictionary
        if metrics_list and len(metrics_list) == len(metric_names):
            metrics_dict = dict(zip(metric_names, metrics_list))
            partition_data.update(metrics_dict)

        # Add additional regression-specific calculations
        if task_type.lower() == 'regression':
            # Calculate consistency (percentage within 1 SD)
            residuals = y_pred - y_true
            acceptable_range = stats.get('sd', 1.0) if stats else 1.0
            within_range = np.abs(residuals) <= acceptable_range
            partition_data['consistency'] = float(np.sum(within_range) / len(residuals) * 100) if len(residuals) > 0 else 0.0

        return partition_data
