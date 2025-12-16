"""
Branch Analysis - Statistical analysis and comparison for pipeline branches.

This module provides tools for analyzing and comparing performance
across different pipeline branches.

Features:
- Branch summary statistics (mean, std, min, max)
- Statistical significance testing between branches
- DataFrame and LaTeX export for publications
- Nested branch analysis support

Example:
    >>> from nirs4all.visualization.analysis.branch import BranchAnalyzer
    >>> analyzer = BranchAnalyzer(predictions)
    >>> summary = analyzer.summary(metrics=['rmse', 'r2'])
    >>> print(summary.to_markdown())
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class BranchSummary:
    """Branch summary statistics container with export capabilities.

    Provides DataFrame-like access and export to markdown, LaTeX, and CSV.

    Attributes:
        data: List of dictionaries with branch statistics.
        metrics: List of metrics computed.
        columns: Column names in order.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        metrics: List[str]
    ):
        """Initialize BranchSummary.

        Args:
            data: List of dictionaries, one per branch.
            metrics: List of metrics that were computed.
        """
        self.data = data
        self.metrics = metrics
        self._df: Optional['pd.DataFrame'] = None

        # Define column order
        base_cols = ['branch_name', 'branch_id', 'count']
        metric_cols = []
        for metric in metrics:
            metric_cols.extend([
                f'{metric}_mean', f'{metric}_std',
                f'{metric}_min', f'{metric}_max'
            ])
        self.columns = base_cols + metric_cols

    def to_dataframe(self) -> 'pd.DataFrame':
        """Convert to pandas DataFrame.

        Returns:
            pandas DataFrame with branch statistics.

        Raises:
            ImportError: If pandas is not installed.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for DataFrame export")

        if self._df is None:
            self._df = pd.DataFrame(self.data)
            # Reorder columns
            cols = [c for c in self.columns if c in self._df.columns]
            self._df = self._df[cols]

        return self._df

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert to dictionary keyed by branch name.

        Returns:
            Dictionary mapping branch_name to statistics.
        """
        return {row['branch_name']: row for row in self.data}

    def to_markdown(
        self,
        precision: int = 3,
        include_std: bool = True
    ) -> str:
        """Export as markdown table.

        Args:
            precision: Decimal places for floating point values.
            include_std: If True, include std columns.

        Returns:
            Markdown-formatted table string.
        """
        if not self.data:
            return "No branch data available"

        # Determine columns to include
        cols = ['branch_name', 'branch_id', 'count']
        for metric in self.metrics:
            cols.append(f'{metric}_mean')
            if include_std:
                cols.append(f'{metric}_std')

        # Filter columns that exist
        cols = [c for c in cols if c in self.data[0]]

        # Build header
        header = '| ' + ' | '.join(cols) + ' |'
        separator = '|' + '|'.join(['---'] * len(cols)) + '|'

        # Build rows
        rows = []
        for row in self.data:
            values = []
            for col in cols:
                val = row.get(col, '')
                if isinstance(val, float):
                    if np.isnan(val):
                        values.append('N/A')
                    else:
                        values.append(f'{val:.{precision}f}')
                else:
                    values.append(str(val) if val is not None else '')
            rows.append('| ' + ' | '.join(values) + ' |')

        return '\n'.join([header, separator] + rows)

    def to_latex(
        self,
        caption: str = "Branch Performance Comparison",
        label: str = "tab:branch_comparison",
        precision: int = 3,
        include_std: bool = True,
        mean_std_combined: bool = True
    ) -> str:
        """Export as LaTeX table for publications.

        Args:
            caption: Table caption.
            label: LaTeX label for referencing.
            precision: Decimal places for floating point values.
            include_std: If True, include std columns.
            mean_std_combined: If True, format as "mean ± std".

        Returns:
            LaTeX-formatted table string.
        """
        if not self.data:
            return "% No branch data available"

        # Determine columns
        if mean_std_combined and include_std:
            display_cols = ['Branch', 'ID', 'N']
            for metric in self.metrics:
                display_cols.append(metric.upper())
        else:
            display_cols = ['Branch', 'ID', 'N']
            for metric in self.metrics:
                display_cols.append(f'{metric}_mean')
                if include_std:
                    display_cols.append(f'{metric}_std')

        # Build LaTeX table
        n_cols = len(display_cols)
        col_spec = 'l' + 'c' * (n_cols - 1)

        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            f'\\caption{{{caption}}}',
            f'\\label{{{label}}}',
            f'\\begin{{tabular}}{{{col_spec}}}',
            r'\toprule',
        ]

        # Header
        header = ' & '.join(display_cols) + r' \\'
        lines.append(header)
        lines.append(r'\midrule')

        # Data rows
        for row in self.data:
            values = [
                self._escape_latex(str(row.get('branch_name', ''))),
                str(row.get('branch_id', '')),
                str(row.get('count', '')),
            ]

            for metric in self.metrics:
                mean_val = row.get(f'{metric}_mean')
                std_val = row.get(f'{metric}_std')

                if mean_val is None or np.isnan(mean_val):
                    values.append('--')
                elif mean_std_combined and include_std:
                    if std_val is None or np.isnan(std_val):
                        values.append(f'{mean_val:.{precision}f}')
                    else:
                        values.append(f'${mean_val:.{precision}f} \\pm {std_val:.{precision}f}$')
                else:
                    values.append(f'{mean_val:.{precision}f}')
                    if include_std:
                        if std_val is None or np.isnan(std_val):
                            values.append('--')
                        else:
                            values.append(f'{std_val:.{precision}f}')

            lines.append(' & '.join(values) + r' \\')

        lines.extend([
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ])

        return '\n'.join(lines)

    def to_csv(
        self,
        path: str,
        precision: int = 6
    ) -> None:
        """Export to CSV file.

        Args:
            path: Output file path.
            precision: Decimal places for floating point values.
        """
        df = self.to_dataframe()
        df.to_csv(path, index=False, float_format=f'%.{precision}f')

    @staticmethod
    def _escape_latex(text: str) -> str:
        """Escape special LaTeX characters.

        Args:
            text: Input string.

        Returns:
            LaTeX-safe string.
        """
        replacements = [
            ('_', r'\_'),
            ('&', r'\&'),
            ('%', r'\%'),
            ('#', r'\#'),
            ('{', r'\{'),
            ('}', r'\}'),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    def __repr__(self) -> str:
        """String representation."""
        return self.to_markdown()

    def __len__(self) -> int:
        """Number of branches."""
        return len(self.data)

    def __getitem__(self, key: Union[int, str]) -> Dict[str, Any]:
        """Get branch by index or name.

        Args:
            key: Integer index or branch name string.

        Returns:
            Dictionary with branch statistics.
        """
        if isinstance(key, int):
            return self.data[key]
        else:
            for row in self.data:
                if row.get('branch_name') == key:
                    return row
            raise KeyError(f"Branch '{key}' not found")


class BranchAnalyzer:
    """Analyze and compare performance across pipeline branches.

    Provides statistical analysis, hypothesis testing, and comparison
    tools for branched pipeline results.

    Attributes:
        predictions: Predictions object containing prediction data.
    """

    def __init__(self, predictions):
        """Initialize BranchAnalyzer.

        Args:
            predictions: Predictions object with branch metadata.
        """
        self.predictions = predictions

    def summary(
        self,
        metrics: Optional[List[str]] = None,
        partition: str = 'test',
        aggregate: Optional[str] = None
    ) -> BranchSummary:
        """Generate summary statistics for each branch.

        Computes mean, std, min, max for each metric across branches.

        Args:
            metrics: List of metrics to compute (default: ['rmse', 'r2']).
            partition: Partition to compute metrics from (default: 'test').
            aggregate: If provided, aggregate predictions by this column
                      before computing statistics.

        Returns:
            BranchSummary object with statistics.
        """
        if metrics is None:
            metrics = ['rmse', 'r2']

        # Get all predictions, optionally aggregated
        all_preds = self._get_predictions(aggregate=aggregate)

        if not all_preds:
            return BranchSummary([], metrics)

        # Group by branch
        branch_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for pred in all_preds:
            branch_name = pred.get('branch_name', 'no_branch')
            branch_groups[branch_name].append(pred)

        # Compute statistics for each branch
        summary_data = []

        for branch_name, preds in sorted(branch_groups.items()):
            branch_id = preds[0].get('branch_id') if preds else None

            row = {
                'branch_name': branch_name,
                'branch_id': branch_id,
                'count': len(preds),
            }

            # Collect scores for each metric
            for metric in metrics:
                scores = self._collect_scores(preds, metric, partition)

                if scores:
                    row[f'{metric}_mean'] = float(np.mean(scores))
                    row[f'{metric}_std'] = float(np.std(scores))
                    row[f'{metric}_min'] = float(np.min(scores))
                    row[f'{metric}_max'] = float(np.max(scores))
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
                    row[f'{metric}_min'] = np.nan
                    row[f'{metric}_max'] = np.nan

            summary_data.append(row)

        return BranchSummary(summary_data, metrics)

    def compare(
        self,
        branch1: Union[str, int],
        branch2: Union[str, int],
        metric: str = 'rmse',
        partition: str = 'test',
        test: str = 'ttest'
    ) -> Dict[str, Any]:
        """Statistical comparison between two branches.

        Performs hypothesis testing to determine if there's a significant
        difference between two branches.

        Args:
            branch1: First branch name or ID.
            branch2: Second branch name or ID.
            metric: Metric to compare (default: 'rmse').
            partition: Partition for scores (default: 'test').
            test: Statistical test ('ttest', 'wilcoxon', 'mannwhitney').

        Returns:
            Dictionary with:
                - statistic: Test statistic
                - p_value: P-value
                - significant: Boolean at alpha=0.05
                - branch1_mean: Mean of branch1
                - branch2_mean: Mean of branch2
                - effect_size: Cohen's d effect size

        Raises:
            ImportError: If scipy is not available.
            ValueError: If branches not found or insufficient data.
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required for statistical testing")

        # Get predictions for each branch
        preds1 = self._get_branch_predictions(branch1)
        preds2 = self._get_branch_predictions(branch2)

        if not preds1 or not preds2:
            raise ValueError("One or both branches have no predictions")

        # Collect scores
        scores1 = self._collect_scores(preds1, metric, partition)
        scores2 = self._collect_scores(preds2, metric, partition)

        if len(scores1) < 2 or len(scores2) < 2:
            raise ValueError("Insufficient data for statistical testing")

        # Perform test
        if test == 'ttest':
            result = stats.ttest_ind(scores1, scores2)
        elif test == 'wilcoxon':
            # Wilcoxon requires paired samples
            min_len = min(len(scores1), len(scores2))
            result = stats.wilcoxon(scores1[:min_len], scores2[:min_len])
        elif test == 'mannwhitney':
            result = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
        else:
            raise ValueError(f"Unknown test: {test}")

        statistic = result.statistic  # type: ignore[union-attr]
        p_value = result.pvalue  # type: ignore[union-attr]

        # Compute effect size (Cohen's d)
        var1 = (len(scores1) - 1) * np.var(scores1)
        var2 = (len(scores2) - 1) * np.var(scores2)
        n_total = len(scores1) + len(scores2) - 2
        pooled_std = np.sqrt((var1 + var2) / n_total)
        effect_size = (
            (np.mean(scores1) - np.mean(scores2)) / pooled_std
            if pooled_std > 0 else 0
        )

        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': float(p_value) < 0.05,
            'branch1_mean': float(np.mean(scores1)),
            'branch2_mean': float(np.mean(scores2)),
            'branch1_std': float(np.std(scores1)),
            'branch2_std': float(np.std(scores2)),
            'effect_size': float(effect_size),
            'test': test,
            'n1': len(scores1),
            'n2': len(scores2),
        }

    def rank_branches(
        self,
        metric: str = 'rmse',
        partition: str = 'test',
        ascending: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Rank branches by mean performance.

        Args:
            metric: Metric to rank by (default: 'rmse').
            partition: Partition for scores (default: 'test').
            ascending: Sort order. If None, auto-detect based on metric.

        Returns:
            List of dicts with branch_name, mean, std, rank.
        """
        summary = self.summary(metrics=[metric], partition=partition)

        # Determine sort order
        if ascending is None:
            higher_better_metrics = [
                'accuracy', 'balanced_accuracy', 'r2', 'f1',
                'precision', 'recall', 'specificity', 'roc_auc'
            ]
            ascending = metric.lower() not in higher_better_metrics

        # Sort by mean
        ranked = sorted(
            summary.data,
            key=lambda x: x.get(f'{metric}_mean', float('inf') if ascending else float('-inf')),
            reverse=not ascending
        )

        # Add rank
        for i, row in enumerate(ranked):
            row['rank'] = i + 1

        return ranked

    def pairwise_comparison(
        self,
        metric: str = 'rmse',
        partition: str = 'test',
        test: str = 'ttest'
    ) -> 'pd.DataFrame':
        """Compute pairwise statistical comparisons between all branches.

        Args:
            metric: Metric to compare (default: 'rmse').
            partition: Partition for scores (default: 'test').
            test: Statistical test to use.

        Returns:
            DataFrame with p-values for all branch pairs.

        Raises:
            ImportError: If pandas or scipy not available.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for pairwise comparison")

        branches = self.get_branch_names()
        n = len(branches)

        # Initialize matrix
        p_values = np.ones((n, n))

        for i, b1 in enumerate(branches):
            for j, b2 in enumerate(branches):
                if i < j:
                    try:
                        result = self.compare(b1, b2, metric, partition, test)
                        p_values[i, j] = result['p_value']
                        p_values[j, i] = result['p_value']
                    except ValueError:
                        pass

        return pd.DataFrame(p_values, index=branches, columns=branches)

    def get_branch_names(self) -> List[str]:
        """Get list of unique branch names.

        Returns:
            List of branch names.
        """
        try:
            names = self.predictions.get_unique_values('branch_name')
            return [n for n in names if n is not None]
        except (ValueError, KeyError):
            return []

    def get_branch_ids(self) -> List[int]:
        """Get list of unique branch IDs.

        Returns:
            List of branch IDs.
        """
        try:
            ids = self.predictions.get_unique_values('branch_id')
            return sorted([int(x) for x in ids if x is not None])
        except (ValueError, KeyError):
            return []

    def _get_predictions(
        self,
        aggregate: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all predictions, optionally aggregated.

        Args:
            aggregate: If provided, aggregate by this column.

        Returns:
            List of prediction dictionaries.
        """
        n = self.predictions.num_predictions
        if n == 0:
            return []

        if aggregate:
            # Use top for aggregation
            return list(self.predictions.top(
                n=n,
                rank_metric='rmse',  # Arbitrary, just need all preds
                aggregate=aggregate,
                aggregate_partitions=True
            ))
        else:
            # Use filter_predictions for regular access
            return self.predictions.filter_predictions()

    def _get_branch_predictions(
        self,
        branch: Union[str, int]
    ) -> List[Dict[str, Any]]:
        """Get predictions for a specific branch.

        Args:
            branch: Branch name (str) or ID (int).

        Returns:
            List of prediction dictionaries.
        """
        if isinstance(branch, int):
            return self.predictions.filter_predictions(branch_id=branch)
        else:
            return self.predictions.filter_predictions(branch_name=branch)

    def _collect_scores(
        self,
        predictions: List[Dict[str, Any]],
        metric: str,
        partition: str
    ) -> List[float]:
        """Collect scores from predictions.

        Args:
            predictions: List of prediction dictionaries.
            metric: Metric to extract.
            partition: Partition to get scores from.

        Returns:
            List of score values.
        """
        from nirs4all.core import metrics as evaluator

        scores = []

        for pred in predictions:
            score = None

            # Try 1: partitions dict (from top method)
            partitions = pred.get('partitions', {})
            if partitions and isinstance(partitions, dict):
                partition_data = partitions.get(partition, {})
                if isinstance(partition_data, dict):
                    score = partition_data.get(metric)

            # Try 2: {partition}_score fields (from filter_predictions)
            if score is None:
                # For MSE/RMSE, the field might be named val_score, test_score, train_score
                partition_score_key = f'{partition}_score'
                score_field = pred.get(partition_score_key)
                if score_field is not None:
                    # Check if the metric matches what's stored
                    pred_metric = pred.get('metric', '').lower()
                    if pred_metric == metric.lower():
                        score = score_field

            # Try 3: scores dict
            if score is None:
                scores_dict = pred.get('scores', {})
                if isinstance(scores_dict, dict):
                    partition_scores = scores_dict.get(partition, {})
                    if isinstance(partition_scores, dict):
                        score = partition_scores.get(metric)

            # Try 4: Compute from y_true/y_pred
            if score is None:
                y_true = pred.get('y_true')
                y_pred = pred.get('y_pred')
                # Only use if partition matches
                pred_partition = pred.get('partition', '')
                has_data = y_true is not None and y_pred is not None
                if has_data and pred_partition == partition:
                    try:
                        score = evaluator.eval(y_true, y_pred, metric)
                    except Exception:
                        continue

            # Validate score is a numeric type
            if score is not None and isinstance(score, (int, float, np.floating, np.integer)):
                score_float = float(score)
                if not np.isnan(score_float):
                    scores.append(score_float)

        return scores
