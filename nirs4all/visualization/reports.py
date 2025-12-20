"""
Tab Report Manager - Simplified tab report generation with formatting and saving

This module provides a clean interface for generating standardized tab-based CSV reports
using pre-calculated metrics and statistics from the evaluator module.
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import csv
import os
import io

# Import evaluator functions
import nirs4all.core.metrics as evaluator
from nirs4all.core.task_type import TaskType
from nirs4all.core.task_detection import detect_task_type
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


class TabReportManager:
    """Generate standardized tab-based CSV reports with pre-calculated data."""

    @staticmethod
    def generate_best_score_tab_report(
        best_by_partition: Dict[str, Dict[str, Any]],
        aggregate: Optional[Union[str, bool]] = None,
        aggregate_method: Optional[str] = None,
        aggregate_exclude_outliers: bool = False
    ) -> Tuple[str, Optional[str]]:
        """
        Generate best score tab report from partition data.

        Args:
            best_by_partition: Dict mapping partition names ('train', 'val', 'test') to prediction entries
            aggregate: Sample aggregation setting for computing additional aggregated metrics.
                - None (default): No aggregation, only raw scores displayed
                - True: Aggregate by y_true values (group by target)
                - str: Aggregate by specified metadata column (e.g., 'sample_id', 'ID')
                When set, both raw and aggregated scores are included in the output.
                Aggregated rows are marked with an asterisk (*).
            aggregate_method: Aggregation method for combining predictions.
                - None (default): Use 'mean' for regression, 'vote' for classification
                - 'mean': Average predictions within each group
                - 'median': Median prediction within each group
                - 'vote': Majority voting (for classification)
            aggregate_exclude_outliers: If True, exclude outliers using T² statistic
                before aggregation (default: False).

        Returns:
            Tuple of (formatted_string, csv_string_content)
            If aggregate is set, both raw and aggregated scores are included.
        """
        if not best_by_partition:
            return "No prediction data available", None

        # Get task type from first available non-None prediction's metadata
        first_entry = next((v for v in best_by_partition.values() if v is not None), None)
        if first_entry is None:
            return "No prediction data available", None

        task_type = TabReportManager._get_task_type_from_entry(first_entry)

        # Extract n_features from metadata if available
        n_features = first_entry.get('n_features', 0)

        # Normalize aggregate parameter: True -> 'y', str -> str, None/False -> None
        effective_aggregate: Optional[str] = None
        if aggregate is True:
            effective_aggregate = 'y'
        elif isinstance(aggregate, str):
            effective_aggregate = aggregate

        # Calculate metrics and stats for each partition
        partitions_data = {}
        aggregated_partitions_data = {}

        for partition_name, entry in best_by_partition.items():
            if partition_name in ['train', 'val', 'test'] and entry is not None:
                y_true = np.array(entry['y_true'])
                y_pred = np.array(entry['y_pred'])

                # Calculate raw (non-aggregated) metrics
                partitions_data[partition_name] = TabReportManager._calculate_partition_data(
                    y_true, y_pred, task_type
                )

                # Calculate aggregated metrics if requested
                if effective_aggregate:
                    agg_result = TabReportManager._aggregate_predictions(
                        y_true=y_true,
                        y_pred=y_pred,
                        aggregate=effective_aggregate,
                        metadata=entry.get('metadata', {}),
                        partition_name=partition_name,
                        method=aggregate_method,
                        exclude_outliers=aggregate_exclude_outliers
                    )
                    if agg_result is not None:
                        agg_y_true, agg_y_pred = agg_result
                        aggregated_partitions_data[partition_name] = TabReportManager._calculate_partition_data(
                            agg_y_true, agg_y_pred, task_type
                        )

        # Generate formatted string (matching PredictionHelpers format)
        formatted_string = TabReportManager._format_as_table_string(
            partitions_data, n_features, task_type,
            aggregated_partitions_data=aggregated_partitions_data,
            aggregate_column=effective_aggregate
        )

        # Generate CSV string content
        csv_string = TabReportManager._format_as_csv_string(
            partitions_data, n_features, task_type,
            aggregated_partitions_data=aggregated_partitions_data,
            aggregate_column=effective_aggregate
        )

        return formatted_string, csv_string

    @staticmethod
    def _aggregate_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        aggregate: str,
        metadata: Dict[str, Any],
        partition_name: str = "",
        method: Optional[str] = None,
        exclude_outliers: bool = False
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Aggregate predictions by a group column.

        Args:
            y_true: True values array
            y_pred: Predicted values array
            aggregate: Group column name or 'y' to group by y_true
            metadata: Metadata dictionary containing group column
            partition_name: Partition name for error messages
            method: Aggregation method ('mean', 'median', 'vote'). Default is 'mean'.
            exclude_outliers: If True, exclude outliers using T² statistic before aggregation.

        Returns:
            Tuple of (aggregated_y_true, aggregated_y_pred) or None if aggregation fails
        """
        from nirs4all.data.predictions import Predictions

        # Determine group IDs
        if aggregate == 'y':
            group_ids = y_true
        else:
            if aggregate not in metadata:
                logger.debug(
                    f"Aggregation column '{aggregate}' not found in metadata for partition '{partition_name}'. "
                    f"Available columns: {list(metadata.keys())}. Skipping aggregation."
                )
                return None
            group_ids = np.asarray(metadata[aggregate])

        if len(group_ids) != len(y_pred):
            logger.debug(
                f"Aggregation column '{aggregate}' length ({len(group_ids)}) doesn't match "
                f"predictions length ({len(y_pred)}) for partition '{partition_name}'. Skipping aggregation."
            )
            return None

        try:
            result = Predictions.aggregate(
                y_pred=y_pred,
                group_ids=group_ids,
                y_true=y_true,
                method=method,
                exclude_outliers=exclude_outliers
            )
            agg_y_true = result.get('y_true')
            agg_y_pred = result.get('y_pred')
            if agg_y_true is None or agg_y_pred is None:
                return None
            return agg_y_true, agg_y_pred
        except Exception as e:
            logger.debug(f"Aggregation failed for partition '{partition_name}': {e}")
            return None

    @staticmethod
    def _get_task_type_from_entry(entry: Dict[str, Any]) -> TaskType:
        """
        Get task type from a prediction entry's metadata.

        Prioritizes stored task_type from metadata, falls back to detection only
        if metadata is missing (for backward compatibility with old predictions).

        Args:
            entry: Prediction entry dictionary

        Returns:
            TaskType: The task type for this prediction
        """
        # First, try to get from metadata
        task_type_str = entry.get('task_type')
        if task_type_str:
            # Convert string to TaskType enum
            try:
                if isinstance(task_type_str, str):
                    return TaskType(task_type_str)
                elif isinstance(task_type_str, TaskType):
                    return task_type_str
            except (ValueError, KeyError):
                pass  # Fall through to detection

        # Fallback: detect from y_true (for backward compatibility)
        logger.warning("task_type not found in prediction metadata, detecting from data")
        y_true = np.array(entry.get('y_true', []))
        if len(y_true) == 0:
            return TaskType.REGRESSION
        return detect_task_type(y_true)

    @staticmethod
    def _format_as_table_string(
        partitions_data: Dict[str, Dict[str, Any]],
        n_features: int,
        task_type: TaskType,
        aggregated_partitions_data: Optional[Dict[str, Dict[str, Any]]] = None,
        aggregate_column: Optional[str] = None
    ) -> str:
        """Format the report data as a table string (matching PredictionHelpers format).

        Args:
            partitions_data: Dict of partition name to metrics data
            n_features: Number of features
            task_type: Task type (regression or classification)
            aggregated_partitions_data: Optional dict of partition name to aggregated metrics
            aggregate_column: Name of column used for aggregation (for footer note)

        Returns:
            Formatted table string
        """
        if not partitions_data:
            return "No partition data available"

        # Prepare headers based on task type
        if task_type == TaskType.REGRESSION:
            headers = ['', 'Nsample', 'Nfeature', 'Mean', 'Median', 'Min', 'Max', 'SD', 'CV',
                       'R2', 'RMSE', 'MSE', 'SEP', 'MAE', 'RPD', 'Bias', 'Consistency']
        else:  # Classification
            is_binary = 'roc_auc' in partitions_data.get('val', {}) or 'roc_auc' in partitions_data.get('test', {})
            if is_binary:
                headers = ['', 'Nsample', 'Nfeatures', 'Accuracy', 'Bal. Acc', 'Precision', 'Bal. Prec', 'Recall', 'Bal. Rec', 'F1-score', 'Specificity', 'AUC']
            else:
                headers = ['', 'Nsample', 'Nfeatures', 'Accuracy', 'Bal. Acc', 'Precision', 'Bal. Prec', 'Recall', 'Bal. Rec', 'F1-score', 'Specificity']

        # Check if we have aggregated data
        has_aggregated = aggregated_partitions_data is not None and len(aggregated_partitions_data) > 0

        # Prepare rows
        rows = []

        # Add partition rows in order: val (Cross Val), train, test
        for partition_name in ['val', 'train', 'test']:
            if partition_name not in partitions_data:
                continue

            data = partitions_data[partition_name]
            display_name = "Cros Val" if partition_name == 'val' else partition_name.capitalize()

            # Add raw (non-aggregated) row
            row = TabReportManager._build_table_row(
                display_name, data, n_features, task_type,
                'roc_auc' in partitions_data.get('val', {}) or 'roc_auc' in partitions_data.get('test', {})
            )
            rows.append(row)

            # Add aggregated row if available
            if has_aggregated and aggregated_partitions_data is not None and partition_name in aggregated_partitions_data:
                agg_data = aggregated_partitions_data[partition_name]
                agg_row = TabReportManager._build_table_row(
                    f"{display_name}*", agg_data, n_features, task_type,
                    'roc_auc' in partitions_data.get('val', {}) or 'roc_auc' in partitions_data.get('test', {}),
                    is_aggregated=True
                )
                rows.append(agg_row)

        # Calculate column widths (minimum 6 characters per column)
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

        # Add footer note if aggregated
        if has_aggregated and aggregate_column:
            agg_label = "y (target values)" if aggregate_column == 'y' else aggregate_column
            lines.append(f"* Aggregated by {agg_label}")

        return '\n'.join(lines)

    @staticmethod
    def _build_table_row(
        display_name: str,
        data: Dict[str, Any],
        n_features: int,
        task_type: TaskType,
        is_binary: bool = False,
        is_aggregated: bool = False
    ) -> list:
        """Build a single table row for either raw or aggregated data.

        Args:
            display_name: Row label (e.g., 'Train', 'Test', 'Train*')
            data: Metrics dictionary for this partition
            n_features: Number of features
            task_type: Task type (regression or classification)
            is_binary: Whether this is binary classification
            is_aggregated: Whether this is an aggregated row (stats columns blank)

        Returns:
            List of formatted cell values
        """
        if task_type == TaskType.REGRESSION:
            # For aggregated rows, skip descriptive stats (Mean, Median, Min, Max, SD, CV)
            # since they don't make sense after averaging predictions
            if is_aggregated:
                row = [
                    display_name,
                    str(data.get('nsample', '')),
                    str(n_features) if n_features > 0 else '',
                    '',  # Mean - blank for aggregated
                    '',  # Median - blank for aggregated
                    '',  # Min - blank for aggregated
                    '',  # Max - blank for aggregated
                    '',  # SD - blank for aggregated
                    '',  # CV - blank for aggregated
                    f"{data.get('r2', ''):.3f}" if data.get('r2') else '',
                    f"{data.get('rmse', ''):.3f}" if data.get('rmse') else '',
                    f"{data.get('mse', ''):.3f}" if data.get('mse') else '',
                    f"{data.get('sep', ''):.3f}" if data.get('sep') else '',
                    f"{data.get('mae', ''):.3f}" if data.get('mae') else '',
                    f"{data.get('rpd', ''):.2f}" if data.get('rpd') and data.get('rpd') != float('inf') else '',
                    f"{data.get('bias', ''):.3f}" if data.get('bias') else '',
                    f"{data.get('consistency', ''):.1f}" if data.get('consistency') else ''
                ]
            else:
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
                f"{data.get('balanced_accuracy', ''):.3f}" if data.get('balanced_accuracy') else '',
                f"{data.get('precision', ''):.3f}" if data.get('precision') else '',
                f"{data.get('balanced_precision', ''):.3f}" if data.get('balanced_precision') else '',
                f"{data.get('recall', ''):.3f}" if data.get('recall') else '',
                f"{data.get('balanced_recall', ''):.3f}" if data.get('balanced_recall') else '',
                f"{data.get('f1', ''):.3f}" if data.get('f1') else '',
                f"{data.get('specificity', ''):.3f}" if data.get('specificity') else ''
            ]
            if is_binary:
                row.append(f"{data.get('roc_auc', ''):.3f}" if data.get('roc_auc') else '')

        return row

    @staticmethod
    def _format_as_csv_string(
        partitions_data: Dict[str, Dict[str, Any]],
        n_features: int,
        task_type: TaskType,
        aggregated_partitions_data: Optional[Dict[str, Dict[str, Any]]] = None,
        aggregate_column: Optional[str] = None
    ) -> str:
        """Generate CSV string content.

        Args:
            partitions_data: Dict of partition name to metrics data
            n_features: Number of features
            task_type: Task type (regression or classification)
            aggregated_partitions_data: Optional dict of partition name to aggregated metrics
            aggregate_column: Name of column used for aggregation (for data annotation)

        Returns:
            CSV formatted string
        """
        # Prepare headers based on task type
        if task_type == TaskType.REGRESSION:
            headers = ['', 'Nsample', 'Nfeature', 'Mean', 'Median', 'Min', 'Max', 'SD', 'CV',
                       'R2', 'RMSE', 'MSE', 'SEP', 'MAE', 'RPD', 'Bias', 'Consistency (%)', 'Aggregated']
        else:  # Classification
            is_binary = 'roc_auc' in partitions_data.get('val', {}) or 'roc_auc' in partitions_data.get('test', {})
            if is_binary:
                headers = ['', 'Nsample', 'Nfeatures', 'Accuracy', 'Bal. Acc', 'Precision', 'Bal. Prec', 'Recall', 'Bal. Rec', 'F1-score', 'Specificity', 'AUC', 'Aggregated']
            else:
                headers = ['', 'Nsample', 'Nfeatures', 'Accuracy', 'Bal. Acc', 'Precision', 'Bal. Prec', 'Recall', 'Bal. Rec', 'F1-score', 'Specificity', 'Aggregated']

        # Check if we have aggregated data
        has_aggregated = aggregated_partitions_data is not None and len(aggregated_partitions_data) > 0

        # Prepare rows
        rows = [headers]

        # Add partition rows in order: val (Cross Val), train, test
        for partition_name in ['val', 'train', 'test']:
            if partition_name not in partitions_data:
                continue

            data = partitions_data[partition_name]
            display_name = "Cros Val" if partition_name == 'val' else partition_name.capitalize()

            # Add raw (non-aggregated) row
            row = TabReportManager._build_csv_row(
                display_name, data, n_features, task_type,
                'roc_auc' in partitions_data.get('val', {}) or 'roc_auc' in partitions_data.get('test', {}),
                is_aggregated=False
            )
            rows.append(row)

            # Add aggregated row if available
            if has_aggregated and aggregated_partitions_data is not None and partition_name in aggregated_partitions_data:
                agg_data = aggregated_partitions_data[partition_name]
                agg_row = TabReportManager._build_csv_row(
                    f"{display_name}*", agg_data, n_features, task_type,
                    'roc_auc' in partitions_data.get('val', {}) or 'roc_auc' in partitions_data.get('test', {}),
                    is_aggregated=True,
                    aggregate_column=aggregate_column
                )
                rows.append(agg_row)

        # Generate CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(rows)

        # Return as string
        csv_content = output.getvalue()
        output.close()

        return csv_content

    @staticmethod
    def _build_csv_row(
        display_name: str,
        data: Dict[str, Any],
        n_features: int,
        task_type: TaskType,
        is_binary: bool = False,
        is_aggregated: bool = False,
        aggregate_column: Optional[str] = None
    ) -> list:
        """Build a single CSV row for either raw or aggregated data.

        Args:
            display_name: Row label (e.g., 'Train', 'Test', 'Train*')
            data: Metrics dictionary for this partition
            n_features: Number of features
            task_type: Task type (regression or classification)
            is_binary: Whether this is binary classification
            is_aggregated: Whether this is an aggregated row
            aggregate_column: Name of column used for aggregation

        Returns:
            List of cell values for CSV row
        """
        aggregated_label = aggregate_column if is_aggregated and aggregate_column else ''

        if task_type == TaskType.REGRESSION:
            if is_aggregated:
                row = [
                    display_name,
                    data.get('nsample', ''),
                    n_features if n_features > 0 else '',
                    '',  # Mean - blank for aggregated
                    '',  # Median - blank for aggregated
                    '',  # Min - blank for aggregated
                    '',  # Max - blank for aggregated
                    '',  # SD - blank for aggregated
                    '',  # CV - blank for aggregated
                    f"{data.get('r2', ''):.3f}" if data.get('r2') else '',
                    f"{data.get('rmse', ''):.3f}" if data.get('rmse') else '',
                    f"{data.get('mse', ''):.3f}" if data.get('mse') else '',
                    f"{data.get('sep', ''):.3f}" if data.get('sep') else '',
                    f"{data.get('mae', ''):.3f}" if data.get('mae') else '',
                    f"{data.get('rpd', ''):.2f}" if data.get('rpd') and data.get('rpd') != float('inf') else '',
                    f"{data.get('bias', ''):.3f}" if data.get('bias') else '',
                    f"{data.get('consistency', ''):.1f}" if data.get('consistency') else '',
                    aggregated_label
                ]
            else:
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
                    f"{data.get('consistency', ''):.1f}" if data.get('consistency') else '',
                    ''  # Not aggregated
                ]
        else:  # Classification
            row = [
                display_name,
                data.get('nsample', ''),
                n_features if n_features > 0 else '',
                f"{data.get('accuracy', ''):.3f}" if data.get('accuracy') else '',
                f"{data.get('balanced_accuracy', ''):.3f}" if data.get('balanced_accuracy') else '',
                f"{data.get('precision', ''):.3f}" if data.get('precision') else '',
                f"{data.get('balanced_precision', ''):.3f}" if data.get('balanced_precision') else '',
                f"{data.get('recall', ''):.3f}" if data.get('recall') else '',
                f"{data.get('balanced_recall', ''):.3f}" if data.get('balanced_recall') else '',
                f"{data.get('f1', ''):.3f}" if data.get('f1') else '',
                f"{data.get('specificity', ''):.3f}" if data.get('specificity') else ''
            ]
            if is_binary:
                row.append(f"{data.get('roc_auc', ''):.3f}" if data.get('roc_auc') else '')
            row.append(aggregated_label)

        return row

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
            metric_names = ['accuracy', 'balanced_accuracy', 'precision', 'balanced_precision', 'recall', 'balanced_recall', 'f1', 'specificity', 'roc_auc']
        else:  # multiclass_classification
            metric_names = ['accuracy', 'balanced_accuracy', 'precision', 'balanced_precision', 'recall', 'balanced_recall', 'f1', 'specificity']

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
