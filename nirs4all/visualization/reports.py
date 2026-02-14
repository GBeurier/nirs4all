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
from nirs4all.visualization.naming import get_metric_names

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
    def generate_per_model_summary(
        refit_entries: list,
        ascending: bool = True,
        metric: str = "rmse",
        aggregate: Optional[Union[str, bool]] = None,
        aggregate_method: Optional[str] = None,
        aggregate_exclude_outliers: bool = False,
        predictions: Optional[Any] = None,
        pred_index: Optional[dict] = None,
        report_naming: str = "nirs",
        verbose: int = 0,
    ) -> str:
        """Generate a per-model summary table for refit entries.

        Uses configurable naming convention (NIRS or ML):
        - NIRS mode (default): RMSEP, RMSECV, Mean_Fold_RMSEP, Selection_Score
        - ML mode: Test_Score, CV_Score, Mean_Fold_Test, Selection_Score
        - Classification: Adapts to task type with proper metric names

        Args:
            refit_entries: Refit prediction entries (fold_id="final")
                with test_score already populated.
            ascending: Whether lower scores are better.
            metric: Metric name (e.g. ``"rmse"``, ``"mse"``, ``"balanced_accuracy"``).
            aggregate: Aggregation column (``str``), ``True`` for 'y', or
                ``None``/``False`` to disable.
            aggregate_method: Aggregation method (``"mean"``, ``"median"``,
                ``"vote"``).
            aggregate_exclude_outliers: Exclude outliers before aggregation.
            predictions: ``Predictions`` instance for partition lookup.
                Required when *aggregate* is set.
            pred_index: Pre-built prediction index from ``_build_prediction_index``.
                If None, will build index internally (slower).
            report_naming: Naming convention (``"nirs"`` or ``"ml"``).
            verbose: Verbosity level. When >= 2, consistency checks are
                logged as warnings.

        Returns:
            Formatted table string with chemometrics-standard column names.
        """
        valid_entries = [e for e in refit_entries if e.get("test_score") is not None]
        if not valid_entries:
            return ""

        # Sort by RMSECV (pre-computed by enrich_refit_entries), falling back
        # to selection_score then test_score when RMSECV is unavailable.
        def _sort_key(e: dict) -> float:
            rmsecv = e.get("rmsecv")
            if rmsecv is not None:
                return rmsecv
            sel = e.get("selection_score")
            if sel is not None:
                return sel
            return e["test_score"]

        entries = sorted(valid_entries, key=_sort_key, reverse=not ascending)

        # Infer task type from first entry (all entries in a report should have same task type)
        task_type = entries[0].get("task_type", "regression") if entries else "regression"
        is_regression = task_type == "regression"

        # Get metric names using the configurable naming system
        task_type_str = "regression" if is_regression else "classification"
        metric_names = get_metric_names(report_naming, task_type_str, metric)
        test_metric_name = metric_names["test_score"]
        cv_metric_name = metric_names["cv_score"]
        mean_fold_test_name = metric_names["mean_fold_test"]
        wmean_fold_test_name = metric_names["wmean_fold_test"]

        # Normalize aggregate parameter
        effective_aggregate: Optional[str] = None
        if aggregate is True:
            effective_aggregate = 'y'
        elif isinstance(aggregate, str):
            effective_aggregate = aggregate

        # Build prediction index once to avoid O(N*M) complexity (if not provided)
        if pred_index is None and predictions is not None:
            pred_index = TabReportManager._build_prediction_index(predictions)

        # Compute aggregated test scores when aggregation is configured
        agg_scores: dict[int, float | None] = {}
        if effective_aggregate and predictions is not None and pred_index is not None:
            eval_metric = metric
            for idx, entry in enumerate(entries):
                agg_score = TabReportManager._compute_aggregated_test_score_indexed(
                    entry, predictions, pred_index, effective_aggregate,
                    aggregate_method, aggregate_exclude_outliers, eval_metric,
                )
                agg_scores[idx] = agg_score

        show_agg = bool(agg_scores) and any(v is not None for v in agg_scores.values())

        def _fmt(value: float | None) -> str:
            """Format a numeric value for display (no transformations)."""
            if value is None:
                return "N/A"
            return f"{value:.4f}"

        def _truncate(text: str, max_len: int = 30) -> str:
            """Truncate text if too long."""
            if len(text) <= max_len:
                return text
            return text[:max_len-3] + "..."

        # Check if we have multiple datasets (global summary)
        datasets = set(e.get("dataset_name") for e in entries)
        show_dataset = len(datasets) > 1

        # Check if we have multi-criteria refit (extract from config_name)
        criteria_labels = {}
        has_multi_criteria = False
        for idx, entry in enumerate(entries):
            config_name = entry.get("config_name", "")
            label = TabReportManager._extract_criterion_label(config_name)
            if label:
                criteria_labels[idx] = label
                has_multi_criteria = True

        # Extract selection criterion from first entry's metadata
        selection_criterion_name = ""
        if entries:
            first_metadata = entries[0].get("metadata", {})
            if isinstance(first_metadata, dict):
                criterion = first_metadata.get("selection_criterion", "")
                if criterion == "rmsecv":
                    selection_criterion_name = " (RMSECV)"
                elif criterion == "mean_val":
                    selection_criterion_name = " (Mean_Fold)"
                elif criterion:  # Any other criterion
                    selection_criterion_name = f" ({criterion})"

        headers = ["#", "Model"]
        if show_dataset:
            headers.append("Dataset")
        headers.append(test_metric_name)
        if show_agg:
            headers.append(f"{test_metric_name}*")
        headers.extend([
            mean_fold_test_name,
            wmean_fold_test_name,
            f"Selection_Score{selection_criterion_name}",
            cv_metric_name,
        ])
        if has_multi_criteria:
            headers.append("Selected_By")
        headers.append("Preprocessing")

        rows = []
        for i, entry in enumerate(entries):
            model_name = entry.get("model_name", "unknown")
            dataset_name = entry.get("dataset_name", "")
            preprocessing = entry.get("preprocessings", "")

            row = [
                str(i + 1),
                model_name,
            ]
            if show_dataset:
                row.append(_truncate(dataset_name, 25))

            row.append(_fmt(entry.get("test_score")))

            if show_agg:
                row.append(_fmt(agg_scores.get(i)))

            # Read pre-computed values from enriched entries
            row.extend([
                _fmt(entry.get("mean_fold_test")),
                _fmt(entry.get("wmean_fold_test")),
                _fmt(entry.get("selection_score")),
                _fmt(entry.get("rmsecv")),
            ])

            if has_multi_criteria:
                row.append(criteria_labels.get(i, ""))

            row.append(_truncate(preprocessing, 40))
            rows.append(row)

        all_rows = [headers] + rows
        col_widths = [
            max(max(len(str(row[j])) for row in all_rows), 6)
            for j in range(len(headers))
        ]

        # Add sorting indicator header
        sort_direction = "ascending (lower is better)" if ascending else "descending (higher is better)"
        sorting_info = f"Sorted by: {cv_metric_name} ({sort_direction})"

        lines = [sorting_info, ""]  # Start with sorting info and blank line
        separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
        lines.append(separator)
        lines.append("|" + "|".join(f" {h:<{col_widths[j]}} " for j, h in enumerate(headers)) + "|")
        lines.append(separator)
        for row in rows:
            lines.append("|" + "|".join(f" {row[j]:<{col_widths[j]}} " for j in range(len(row))) + "|")
        lines.append(separator)

        # Run consistency checks when verbose >= 2
        if verbose >= 2 and pred_index is not None:
            TabReportManager._check_consistency(entries, pred_index, metric)

        return "\n".join(lines)

    @staticmethod
    def _check_consistency(
        entries: list,
        pred_index: dict,
        metric: str,
    ) -> None:
        """Run lightweight consistency checks and log warnings.

        Checks for:
        1. OOF index completeness: whether the union of fold validation
           predictions covers all expected samples (based on the refit
           entry's n_samples).
        2. Score sign consistency: whether scores for the same metric are
           all positive, all negative, or mixed (which suggests
           inconsistent sign conventions).
        3. Fold count mismatch: whether different configs report different
           numbers of CV folds.

        Args:
            entries: Sorted refit entries with test_score populated.
            pred_index: Pre-built prediction index from
                ``_build_prediction_index``.
            metric: Metric name (for sign-consistency check context).
        """
        oof_index = pred_index.get("oof_preds", {})

        # -- Check 1: OOF index completeness --
        for entry in entries:
            dataset_name = entry.get("dataset_name")
            config_name = entry.get("config_name", "")
            model_name = entry.get("model_name")
            step_idx = entry.get("step_idx", 0)
            chain_id = entry.get("chain_id")
            branch_id = entry.get("branch_id") if chain_id is None else None
            expected_n = entry.get("n_samples", 0)

            oof_key = (dataset_name, config_name, model_name, step_idx, chain_id or branch_id)
            oof_preds = oof_index.get(oof_key, [])

            # Resolve CV config for refit entries
            if not oof_preds and entry.get("fold_id") == "final":
                cv_config = TabReportManager._resolve_cv_config_name(config_name)
                oof_preds = oof_index.get((dataset_name, cv_config, model_name, step_idx, chain_id or branch_id), [])

            if oof_preds and expected_n > 0:
                total_oof = sum(len(np.asarray(p.get("y_true", [])).ravel()) for p in oof_preds)
                if total_oof < expected_n:
                    logger.warning(
                        f"[Consistency] OOF predictions for '{model_name}' (config='{config_name}'): "
                        f"union covers {total_oof}/{expected_n} samples. "
                        f"Some folds may have missing predictions."
                    )

        # -- Check 2: Score sign consistency --
        scores_for_metric: list[float] = []
        for entry in entries:
            score = entry.get("test_score")
            if score is not None:
                scores_for_metric.append(score)

        if len(scores_for_metric) >= 2:
            has_positive = any(s > 0 for s in scores_for_metric)
            has_negative = any(s < 0 for s in scores_for_metric)
            if has_positive and has_negative:
                logger.warning(
                    f"[Consistency] Mixed positive/negative test scores detected for metric '{metric}': "
                    f"min={min(scores_for_metric):.4f}, max={max(scores_for_metric):.4f}. "
                    f"This may indicate inconsistent sign conventions."
                )

        # -- Check 3: Fold count mismatch --
        fold_counts: dict[str, int] = {}
        for entry in entries:
            config_name = entry.get("config_name", "")
            dataset_name = entry.get("dataset_name", "")
            model_name = entry.get("model_name")
            step_idx = entry.get("step_idx", 0)
            chain_id = entry.get("chain_id")
            branch_id = entry.get("branch_id") if chain_id is None else None

            oof_key = (dataset_name, config_name, model_name, step_idx, chain_id or branch_id)
            oof_preds = oof_index.get(oof_key, [])

            if not oof_preds and entry.get("fold_id") == "final":
                cv_config = TabReportManager._resolve_cv_config_name(config_name)
                oof_preds = oof_index.get((dataset_name, cv_config, model_name, step_idx, chain_id or branch_id), [])

            if oof_preds:
                label = f"{config_name}/{model_name}"
                fold_counts[label] = len(oof_preds)

        if fold_counts:
            unique_counts = set(fold_counts.values())
            if len(unique_counts) > 1:
                details = ", ".join(f"'{k}'={v}" for k, v in fold_counts.items())
                logger.warning(
                    f"[Consistency] Different configs report different fold counts: {details}. "
                    f"This may indicate heterogeneous CV strategies."
                )

    @staticmethod
    def enrich_refit_entries(
        refit_entries: list[dict],
        pred_index: dict,
        metric: str = "rmse",
    ) -> None:
        """Compute CV metrics and store them on refit entries.

        Enriches each refit entry in-place with:

        - ``rmsecv``: Pooled out-of-fold CV metric (RMSECV for regression).
        - ``mean_fold_test``: Arithmetic mean of per-fold test scores.
        - ``wmean_fold_test``: Sample-weighted mean of per-fold test scores.

        These values are computed once from the CV fold predictions
        referenced by ``pred_index`` and stored on the entry dicts so
        that downstream consumers (reports, sorting) can read them
        directly without recomputing.

        Args:
            refit_entries: List of refit prediction entry dicts
                (``fold_id="final"``).  Modified in-place.
            pred_index: Pre-built prediction index from
                ``_build_prediction_index``.
            metric: Metric name (e.g. ``"rmse"``).
        """
        for entry in refit_entries:
            task_type = entry.get("task_type", "regression")

            rmsecv = TabReportManager._compute_oof_cv_metric_indexed(
                entry, pred_index, metric, task_type
            )
            entry["rmsecv"] = rmsecv

            avg_test, wmean_test = TabReportManager._compute_cv_test_averages_indexed(
                entry, pred_index, metric
            )
            entry["mean_fold_test"] = avg_test
            entry["wmean_fold_test"] = wmean_test

    @staticmethod
    def _build_prediction_index(predictions: Any) -> dict[str, Any]:
        """Build an index of predictions for O(1) lookups.

        Performs a single linear scan of the prediction buffer and builds
        multiple hash-map indices keyed by composite tuples.  This replaces
        repeated O(N) scans with O(1) dict lookups, which is critical when
        generating per-model summary reports with many refit entries.

        Branch disambiguation uses ``chain_id`` (preferred) with a fallback
        to ``branch_id`` so that stacking pipelines with multiple branches
        are correctly separated in the index.

        Args:
            predictions: ``Predictions`` instance whose ``_buffer`` will be
                scanned.

        Returns:
            Dictionary containing the following index maps:

            - ``partitions``: maps ``(dataset, config, model, fold, step)``
              to ``{partition_name: entry}``.  Used for quick partition
              lookups (train/val/test) of a specific fold.
            - ``oof_preds``: maps ``(dataset, config, model, step, chain_id)``
              to a list of val-partition entries for individual CV folds
              (excludes avg/w_avg/final).  Used for pooled OOF metric
              computation (RMSECV).
            - ``test_preds``: maps ``(dataset, config, model, step, chain_id)``
              to a list of test-partition entries for individual CV folds
              (excludes final/avg/w_avg).  Used for mean-fold test score
              computation.
            - ``w_avg``: maps ``(dataset, config, model, step, chain_id)``
              to ``{partition_name: entry}`` for the ``w_avg`` virtual fold.
            - ``predictions``: the original ``Predictions`` object (kept for
              downstream callers that need filter methods).
        """
        partitions_index: dict[tuple, dict[str, dict]] = {}
        oof_index: dict[tuple, list] = {}
        test_index: dict[tuple, list] = {}
        w_avg_index: dict[tuple, dict[str, dict]] = {}

        for row in predictions._buffer:
            dataset_name = row.get("dataset_name")
            config_name = row.get("config_name", "")
            model_name = row.get("model_name")
            fold_id = row.get("fold_id", "")
            step_idx = row.get("step_idx", 0)
            partition = row.get("partition")

            # Use chain_id for branch disambiguation (preferred), fallback to branch_id
            chain_id = row.get("chain_id")
            branch_id = row.get("branch_id") if chain_id is None else None

            # Index for get_entry_partitions
            key = (dataset_name, config_name, model_name, fold_id, step_idx)
            if key not in partitions_index:
                partitions_index[key] = {}
            if partition in ("train", "val", "test") and partition not in partitions_index[key]:
                partitions_index[key][partition] = row

            # Index for get_oof_predictions (val partition, exclude avg/w_avg)
            # Include chain_id for branch disambiguation
            if partition == "val" and fold_id not in ("avg", "w_avg", "final", None, ""):
                oof_key = (dataset_name, config_name, model_name, step_idx, chain_id or branch_id)
                if oof_key not in oof_index:
                    oof_index[oof_key] = []
                oof_index[oof_key].append(row)

            # Index for test predictions (test partition, exclude final/avg/w_avg)
            # Include chain_id for branch disambiguation
            if partition == "test" and fold_id not in ("final", "avg", "w_avg", None, ""):
                test_key = (dataset_name, config_name, model_name, step_idx, chain_id or branch_id)
                if test_key not in test_index:
                    test_index[test_key] = []
                test_index[test_key].append(row)

            # Index for w_avg fold entries (for w_avg enrichment)
            # Include chain_id for branch disambiguation
            if fold_id == "w_avg" and partition in ("train", "val", "test"):
                w_avg_key = (dataset_name, config_name, model_name, step_idx, chain_id or branch_id)
                if w_avg_key not in w_avg_index:
                    w_avg_index[w_avg_key] = {}
                if partition not in w_avg_index[w_avg_key]:
                    w_avg_index[w_avg_key][partition] = row

        return {
            "partitions": partitions_index,
            "oof_preds": oof_index,
            "test_preds": test_index,
            "w_avg": w_avg_index,
            "predictions": predictions,
        }

    @staticmethod
    def _compute_aggregated_test_score_indexed(
        entry: dict,
        predictions: Any,
        pred_index: dict,
        aggregate: str,
        aggregate_method: Optional[str],
        aggregate_exclude_outliers: bool,
        metric: str,
    ) -> Optional[float]:
        """Compute aggregated test score using pre-built index.

        Args:
            entry: Refit prediction entry.
            predictions: ``Predictions`` instance.
            pred_index: Pre-built prediction index from ``_build_prediction_index``.
            aggregate: Aggregation column name.
            aggregate_method: Aggregation method.
            aggregate_exclude_outliers: Exclude outliers flag.
            metric: Metric name for evaluation.

        Returns:
            Aggregated score or ``None`` if aggregation is not possible.
        """
        try:
            dataset_name = entry.get("dataset_name")
            config_name = entry.get("config_name", "")
            model_name = entry.get("model_name")
            fold_id = entry.get("fold_id", "")
            step_idx = entry.get("step_idx", 0)

            key = (dataset_name, config_name, model_name, fold_id, step_idx)
            partitions = pred_index["partitions"].get(key, {})
            test_entry = partitions.get("test")

            if test_entry is None:
                return None

            y_true = np.array(test_entry["y_true"])
            y_pred = np.array(test_entry["y_pred"])
            metadata = test_entry.get("metadata", {})

            agg_result = TabReportManager._aggregate_predictions(
                y_true=y_true,
                y_pred=y_pred,
                aggregate=aggregate,
                metadata=metadata,
                partition_name="test",
                method=aggregate_method,
                exclude_outliers=aggregate_exclude_outliers,
            )
            if agg_result is None:
                return None

            agg_y_true, agg_y_pred = agg_result
            return evaluator.eval(agg_y_true, agg_y_pred, metric)
        except Exception:
            return None

    @staticmethod
    def _resolve_cv_config_name(config_name: str) -> str:
        """Derive the original CV config_name from a refit config_name.

        Refit entries get a suffix appended during the refit phase.  This
        method strips known suffixes to recover the original CV
        config_name so that report code can look up CV fold predictions
        that correspond to a given refit entry.

        Supported suffix patterns (checked in order of specificity):

        - ``_stacking_refit`` -- stacking refit pipelines.
        - ``_refit_<criteria>`` -- multi-criteria refit with encoded
          criterion labels (e.g. ``_refit_rmsecvt3_mean_valt3``).
        - ``_refit`` -- simple single-criterion refit.

        Args:
            config_name: Refit pipeline config name, e.g.
                ``"PLS_10_refit"`` or ``"PLS_10_refit_rmsecvt3"``.

        Returns:
            The original CV config name with refit suffixes removed,
            e.g. ``"PLS_10"``.  If no known suffix is found, returns
            *config_name* unchanged.
        """
        import re

        # Handle stacking refit first (more specific)
        if config_name.endswith("_stacking_refit"):
            return config_name[: -len("_stacking_refit")]

        # Handle multi-criteria refit: _refit_<criteria> (e.g., _refit_rmsecvt3_mean_valt3)
        # Pattern: _refit_ followed by criterion labels (rmsecvt3, mean_valt3, etc.)
        match = re.search(r"_refit_[a-z0-9_]+$", config_name)
        if match:
            return config_name[: match.start()]

        # Handle simple refit
        if config_name.endswith("_refit"):
            return config_name[: -len("_refit")]

        return config_name

    @staticmethod
    def _extract_criterion_label(config_name: str) -> str:
        """Extract criterion label from refit config_name.

        Args:
            config_name: Config name like ``model_refit_rmsecvt3_mean_valt3``.

        Returns:
            Human-readable criterion label like ``"rmsecv(top3), mean_val(top3)"``
            or empty string if not a multi-criteria refit.
        """
        import re

        # Extract multi-criteria suffix: _refit_<criteria>
        match = re.search(r"_refit_([a-z0-9_]+)$", config_name)
        if not match:
            return ""

        suffix = match.group(1)
        criteria = []

        # Parse rmsecvt<N> patterns
        for m in re.finditer(r"rmsecvt(\d+)", suffix):
            k = m.group(1)
            criteria.append(f"rmsecv(top{k})")

        # Parse mean_valt<N> patterns
        for m in re.finditer(r"mean_valt(\d+)", suffix):
            k = m.group(1)
            criteria.append(f"mean_val(top{k})")

        return ", ".join(criteria) if criteria else ""

    @staticmethod
    def _compute_oof_cv_metric_indexed(
        entry: dict,
        pred_index: dict,
        metric: str = "rmse",
        task_type: str = "regression",
    ) -> Optional[float]:
        """Compute the pooled out-of-fold cross-validation metric.

        Collects all individual-fold validation predictions for the pipeline
        configuration that produced *entry*, concatenates them into a single
        pool, and computes the metric over the pooled set.

        For regression this yields RMSECV (Predicted Residual Error Sum of
        Squares):  ``RMSECV = sqrt( sum((y_true - y_pred)^2) / N )``,
        where the sum runs over all OOF samples across all folds.

        For classification the pooled OOF labels are passed to the standard
        metric evaluator (e.g. balanced accuracy computed on all OOF
        predictions at once).

        For refit entries (``fold_id="final"``), the method resolves the
        original CV config name via ``_resolve_cv_config_name`` to find
        the matching CV fold predictions.  Branch disambiguation uses
        ``chain_id`` (preferred) or ``branch_id``.

        Args:
            entry: Refit prediction entry (must contain ``dataset_name``,
                ``config_name``, ``model_name``, ``step_idx``).
            pred_index: Pre-built prediction index from
                ``_build_prediction_index``.
            metric: Metric name (e.g., ``"rmse"``, ``"balanced_accuracy"``).
            task_type: ``"regression"`` or ``"classification"``.

        Returns:
            Pooled OOF CV metric value, or ``None`` if no OOF predictions
            are available for this entry.
        """
        try:
            oof_index = pred_index.get("oof_preds", {})

            dataset_name = entry.get("dataset_name")
            config_name = entry.get("config_name", "")
            model_name = entry.get("model_name")
            fold_id = entry.get("fold_id", "")
            step_idx = entry.get("step_idx", 0)
            is_refit = fold_id == "final"

            # Get chain_id for branch disambiguation (fallback to branch_id)
            chain_id = entry.get("chain_id")
            branch_id = entry.get("branch_id") if chain_id is None else None

            # Use the pre-built OOF index (including step_idx and chain_id for branch disambiguation)
            oof_key = (dataset_name, config_name, model_name, step_idx, chain_id or branch_id)
            oof_preds = oof_index.get(oof_key, [])

            # For refit entries, resolve original CV config_name and fall back
            # without chain_id (refit gets a new chain_id per execution pass).
            if is_refit and not oof_preds:
                cv_config = TabReportManager._resolve_cv_config_name(config_name)
                # Try exact match first (with chain_id)
                oof_preds = oof_index.get((dataset_name, cv_config, model_name, step_idx, chain_id or branch_id), [])
                # Fall back: match without chain_id (refit always gets a fresh chain_id)
                if not oof_preds:
                    for k, v in oof_index.items():
                        if k[0] == dataset_name and k[1] == cv_config and k[2] == model_name and k[3] == step_idx:
                            oof_preds = v
                            break

            if not oof_preds:
                return None

            # Collect all y_true and y_pred from all folds
            y_true_arrays = []
            y_pred_arrays = []
            y_proba_arrays = []

            for fold_pred in oof_preds:
                y_true = fold_pred.get("y_true")
                y_pred = fold_pred.get("y_pred")
                y_proba = fold_pred.get("y_proba")

                if y_true is not None and y_pred is not None:
                    y_true_flat = y_true.ravel() if hasattr(y_true, 'ravel') else np.asarray(y_true).ravel()
                    y_pred_flat = y_pred.ravel() if hasattr(y_pred, 'ravel') else np.asarray(y_pred).ravel()
                    y_true_arrays.append(y_true_flat)
                    y_pred_arrays.append(y_pred_flat)

                    if y_proba is not None:
                        y_proba_arrays.append(y_proba)

            if not y_true_arrays:
                return None

            all_y_true = np.concatenate(y_true_arrays)
            all_y_pred = np.concatenate(y_pred_arrays)
            all_y_proba = np.concatenate(y_proba_arrays) if y_proba_arrays else None

            # Compute metric based on task type
            if task_type == "regression":
                # RMSECV for regression
                squared_errors = (all_y_true - all_y_pred) ** 2
                press = np.sum(squared_errors)
                mse = press / len(all_y_true)
                return np.sqrt(mse)
            else:
                # Classification: use evaluator to compute metric
                result = evaluator.eval(
                    y_true=all_y_true,
                    y_pred=all_y_pred,
                    y_proba=all_y_proba,
                    metrics=[metric],
                )
                return result.get(metric)

        except Exception:
            return None

    @staticmethod
    def _compute_cv_test_averages_indexed(
        entry: dict,
        pred_index: dict,
        metric: str,
    ) -> tuple[Optional[float], Optional[float]]:
        """Compute mean and weighted-mean test scores across CV folds.

        Looks up individual-fold test predictions for the pipeline that
        produced *entry* and computes two averages of their ``test_score``
        values:

        - **Mean**: arithmetic mean of per-fold test scores.
        - **Weighted mean**: weighted by per-fold sample count
          (``sum(score_i * n_i) / sum(n_i)``), which accounts for
          unequal fold sizes.

        When the primary metric is ``"mse"``, the stored per-fold scores
        are in MSE units.  Both averages are converted to RMSE
        (``sqrt(max(value, 0))``) for display consistency with RMSEP.

        For refit entries (``fold_id="final"``), the original CV config
        name is resolved via ``_resolve_cv_config_name`` to locate the
        matching CV fold test predictions.

        Args:
            entry: Refit prediction entry (must contain ``dataset_name``,
                ``config_name``, ``model_name``, ``step_idx``).
            pred_index: Pre-built prediction index from
                ``_build_prediction_index``.
            metric: Primary metric name.  If ``"mse"``, the returned
                scores are converted from MSE to RMSE for display.

        Returns:
            Tuple of ``(avg_test_score, weighted_avg_test_score)`` in
            display-ready units.  Returns ``(None, None)`` if no
            per-fold test predictions are available.
        """
        try:
            test_index = pred_index.get("test_preds", {})

            dataset_name = entry.get("dataset_name")
            config_name = entry.get("config_name", "")
            model_name = entry.get("model_name")
            fold_id = entry.get("fold_id", "")
            step_idx = entry.get("step_idx", 0)
            is_refit = fold_id == "final"

            # Get chain_id for branch disambiguation (fallback to branch_id)
            chain_id = entry.get("chain_id")
            branch_id = entry.get("branch_id") if chain_id is None else None

            # Use the pre-built test index (including step_idx and chain_id for branch disambiguation)
            test_key = (dataset_name, config_name, model_name, step_idx, chain_id or branch_id)
            test_preds = test_index.get(test_key, [])

            # For refit entries, resolve original CV config_name and fall back
            # without chain_id (refit gets a new chain_id per execution pass).
            if is_refit and not test_preds:
                cv_config = TabReportManager._resolve_cv_config_name(config_name)
                test_preds = test_index.get((dataset_name, cv_config, model_name, step_idx, chain_id or branch_id), [])
                # Fall back: match without chain_id (refit always gets a fresh chain_id)
                if not test_preds:
                    for k, v in test_index.items():
                        if k[0] == dataset_name and k[1] == cv_config and k[2] == model_name and k[3] == step_idx:
                            test_preds = v
                            break

            if not test_preds:
                return None, None

            test_scores = []
            sample_counts = []

            for fold_pred in test_preds:
                test_score = fold_pred.get("test_score")
                n_samples = fold_pred.get("n_samples", 0)

                if test_score is not None and n_samples > 0:
                    test_scores.append(test_score)
                    sample_counts.append(n_samples)

            if not test_scores:
                return None, None

            avg_test = np.mean(test_scores)

            total_samples = sum(sample_counts)
            if total_samples > 0:
                weighted_avg_test = sum(score * count for score, count in zip(test_scores, sample_counts)) / total_samples
            else:
                weighted_avg_test = avg_test

            # Convert MSE to RMSE for display if metric is "mse"
            if metric.lower() == "mse":
                avg_test = np.sqrt(max(avg_test, 0.0))
                weighted_avg_test = np.sqrt(max(weighted_avg_test, 0.0))

            return avg_test, weighted_avg_test

        except Exception:
            return None, None

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
