"""
Controller for sample filtering operations.

This controller handles sample filtering operators, identifying and marking
samples for exclusion from training datasets based on various criteria
(outliers, quality issues, etc.).
"""

from typing import Any, List, Tuple, Optional, Dict, TYPE_CHECKING
import numpy as np

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.operators.filters.base import SampleFilter, CompositeFilter
from nirs4all.pipeline.config.component_serialization import deserialize_component

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext
    from nirs4all.pipeline.steps.parser import ParsedStep
    from nirs4all.pipeline.steps.runtime import RuntimeContext


@register_controller
class SampleFilterController(OperatorController):
    """
    Controller for sample filtering operations.

    This controller orchestrates sample filtering by:
    1. Retrieving train samples (base only, no augmented) and their X/y values
    2. Applying each filter's get_mask() method to identify outliers
    3. Combining masks according to the specified mode (any/all)
    4. Marking excluded samples in the dataset's indexer
    5. Generating filtering report (optional)

    Sample filters are non-destructive - they mark samples as excluded in the
    indexer rather than removing data. Excluded samples can be re-included
    using dataset._indexer.mark_included().

    Pipeline syntax:
        {
            "sample_filter": {
                "filters": [
                    YOutlierFilter(method="iqr", threshold=1.5),
                    XOutlierFilter(method="mahalanobis"),
                ],
                "mode": "any",  # "any" = exclude if ANY filter flags
                "report": True,  # Generate filtering report
                "cascade_to_augmented": True,  # Also exclude augmented samples
            }
        }

    Note:
        Filtering only runs during training mode - in prediction mode,
        this controller does nothing to avoid excluding prediction samples.
    """

    priority = 5  # Execute early, before augmentation (which has priority=10)

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match sample_filter keyword in pipeline."""
        return keyword == "sample_filter"

    @classmethod
    def use_multi_source(cls) -> bool:
        """Sample filtering operates on the dataset level, not per-source."""
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """
        Sample filtering only runs during training.

        Prediction samples should never be filtered/excluded - we want to
        predict on all provided samples. Filters were fitted during training
        and their thresholds don't apply to new data.
        """
        return False

    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple['ExecutionContext', List]:
        """
        Execute sample filtering operation.

        This method:
        1. Retrieves training data (base samples only)
        2. Fits and applies each filter to identify outliers
        3. Combines filter masks using the specified mode
        4. Marks excluded samples in the dataset's indexer
        5. Optionally prints a filtering report

        Args:
            step_info: Parsed step containing operator and configuration
            dataset: Dataset to operate on
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index (unused, filtering is dataset-level)
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binaries (filters may be persisted)
            prediction_store: External prediction store (unused)

        Returns:
            Tuple of (updated_context, persisted_artifacts)

        Raises:
            ValueError: If no filters are specified
            ValueError: If invalid mode is specified
        """
        # Skip during prediction mode
        if mode != "train":
            return context, []

        # Extract configuration from step
        step = step_info.original_step
        config = step.get("sample_filter", {})

        filters_raw = config.get("filters", [])
        filter_mode = config.get("mode", "any")
        report = config.get("report", False)
        cascade_to_augmented = config.get("cascade_to_augmented", True)

        if not filters_raw:
            raise ValueError("sample_filter requires at least one filter in 'filters' list")

        if filter_mode not in ("any", "all"):
            raise ValueError(f"mode must be 'any' or 'all', got '{filter_mode}'")

        # Deserialize filters (they may be stored as serialized class paths)
        filters: List[SampleFilter] = []
        for f in filters_raw:
            deserialized = deserialize_component(f)
            if not isinstance(deserialized, SampleFilter):
                raise TypeError(
                    f"Filter must be a SampleFilter instance, got {type(deserialized).__name__}"
                )
            filters.append(deserialized)

        # Get train samples (base only, no augmented) for filtering
        train_context = context.with_partition("train")
        base_sample_indices = dataset._indexer.x_indices(  # noqa: SLF001
            train_context.selector, include_augmented=False, include_excluded=False
        )

        if len(base_sample_indices) == 0:
            if runtime_context.step_runner.verbose > 0:
                logger.info("   SampleFilter: No training samples to filter")
            return context, []

        # Get X and y for base train samples
        # Use a selector that returns only base samples
        base_selector = train_context.selector.with_augmented(False)
        X_train = dataset.x(base_selector, layout="2d", concat_source=True)
        y_train = dataset.y(base_selector)

        # Handle empty or None y_train
        if y_train is None or len(y_train) == 0:
            if runtime_context.step_runner.verbose > 0:
                logger.info("   SampleFilter: No target values available for filtering")
            return context, []

        # Flatten y if needed
        if y_train.ndim > 1:
            y_train = y_train.flatten()

        # Fit all filters and collect masks
        masks: List[np.ndarray] = []
        filter_reports: List[Dict[str, Any]] = []

        for filter_obj in filters:
            try:
                # Fit the filter
                filter_obj.fit(X_train, y_train)

                # Get the mask (True = keep, False = exclude)
                mask = filter_obj.get_mask(X_train, y_train)
                masks.append(mask)

                # Collect stats for report
                if report:
                    stats = filter_obj.get_filter_stats(X_train, y_train)
                    filter_reports.append(stats)
            except ValueError as e:
                # Handle edge cases like insufficient data
                if runtime_context.step_runner.verbose > 0:
                    logger.warning(f"   SampleFilter: {filter_obj.__class__.__name__} "
                          f"could not be applied: {e}")
                # Create a neutral mask (keep all)
                masks.append(np.ones(len(X_train), dtype=bool))
                if report:
                    filter_reports.append({
                        "n_samples": len(X_train),
                        "n_excluded": 0,
                        "n_kept": len(X_train),
                        "exclusion_rate": 0.0,
                        "reason": filter_obj.exclusion_reason,
                        "warning": str(e),
                    })

        # Combine masks according to mode
        if len(masks) == 1:
            combined_mask = masks[0]
        else:
            stacked = np.stack(masks, axis=0)
            if filter_mode == "any":
                # Exclude if ANY filter flags -> keep only if ALL filters say keep
                combined_mask = np.all(stacked, axis=0)
            else:  # "all"
                # Exclude only if ALL filters flag -> keep if ANY filter says keep
                combined_mask = np.any(stacked, axis=0)

        # Get indices of samples to exclude
        exclude_mask = ~combined_mask
        samples_to_exclude = base_sample_indices[exclude_mask].tolist()

        # Warn if all samples would be excluded
        n_remaining = int(np.sum(combined_mask))
        if n_remaining == 0 and len(base_sample_indices) > 0:
            import warnings
            warnings.warn(
                f"Sample filtering would exclude ALL {len(base_sample_indices)} samples. "
                "Consider adjusting filter thresholds. Keeping at least one sample.",
                UserWarning
            )
            # Keep at least one sample (the first one)
            combined_mask[0] = True
            samples_to_exclude = base_sample_indices[~combined_mask].tolist()

        # Mark samples as excluded in the indexer
        n_excluded = 0
        if samples_to_exclude:
            # Create a combined reason string
            if len(filters) == 1:
                reason = filters[0].exclusion_reason
            else:
                filter_names = [f.exclusion_reason for f in filters]
                reason = f"sample_filter({filter_mode}:{','.join(filter_names)})"

            n_excluded = dataset._indexer.mark_excluded(  # noqa: SLF001
                samples_to_exclude,
                reason=reason,
                cascade_to_augmented=cascade_to_augmented
            )

        # Print report if requested
        if report or runtime_context.step_runner.verbose > 0:
            self._print_report(
                n_total=len(base_sample_indices),
                n_excluded=len(samples_to_exclude),
                n_cascaded=n_excluded - len(samples_to_exclude) if cascade_to_augmented else 0,
                filter_reports=filter_reports if report else None,
                mode=filter_mode,
                verbose=runtime_context.step_runner.verbose
            )

        # Persist filters for reference (not used in prediction, but for audit)
        artifacts = []
        if filters:
            for i, filter_obj in enumerate(filters):
                operator_name = f"sample_filter_{filter_obj.exclusion_reason}_{runtime_context.next_op()}"
                artifact = runtime_context.saver.persist_artifact(
                    step_number=runtime_context.step_number,
                    name=operator_name,
                    obj=filter_obj,
                    format_hint='sklearn',
                    branch_id=context.selector.branch_id,
                    branch_name=context.selector.branch_name
                )
                artifacts.append(artifact)

        return context, artifacts

    def _print_report(
        self,
        n_total: int,
        n_excluded: int,
        n_cascaded: int,
        filter_reports: Optional[List[Dict[str, Any]]],
        mode: str,
        verbose: int
    ) -> None:
        """Print filtering report to console."""
        logger.debug("--- Sample Filtering Report ---")
        logger.debug(f"Total base samples: {n_total}")
        logger.debug(f"Samples excluded: {n_excluded} ({100 * n_excluded / n_total:.1f}%)" if n_total > 0 else "Samples excluded: 0")
        if n_cascaded > 0:
            logger.debug(f"Augmented samples also excluded: {n_cascaded}")
        logger.debug(f"Combination mode: {mode}")

        if filter_reports:
            logger.debug("Per-filter breakdown:")
            for i, stats in enumerate(filter_reports):
                logger.debug(f"  Filter {i + 1}: {stats.get('reason', 'unknown')}")
                logger.debug(f"    - Excluded: {stats.get('n_excluded', 0)} samples")
                if 'method' in stats:
                    logger.debug(f"    - Method: {stats['method']}")
                if 'lower_bound' in stats and stats['lower_bound'] is not None:
                    logger.debug(f"    - Bounds: [{stats['lower_bound']:.4f}, {stats['upper_bound']:.4f}]")

        logger.debug("-------------------------------")
