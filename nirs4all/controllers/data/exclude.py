"""
Controller for sample exclusion operations.

This controller handles the `exclude` keyword, marking samples for exclusion
from training based on filter criteria. Unlike `tag`, `exclude` both stores
tag values AND marks samples as excluded from training.
"""

from typing import Any, List, Tuple, Optional, Dict, Union, TYPE_CHECKING
import numpy as np

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.operators.filters.base import SampleFilter
from nirs4all.pipeline.config.component_serialization import deserialize_component

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext
    from nirs4all.pipeline.steps.parser import ParsedStep
    from nirs4all.pipeline.steps.runtime import RuntimeContext


@register_controller
class ExcludeController(OperatorController):
    """
    Controller for sample exclusion operations.

    This controller computes exclusion masks using SampleFilter instances,
    stores the results as tag columns for analysis, and marks matching
    samples as excluded from training.

    The key difference from `tag`:
    - `tag` = compute and store tag (never removes samples)
    - `exclude` = compute tag AND remove from training (always removes)

    For tag-only behavior without exclusion, use the `tag` keyword instead.

    Pipeline syntax:
        # Single filter
        {"exclude": YOutlierFilter(method="iqr")}

        # Multiple filters with mode
        {"exclude": [YOutlierFilter(), XOutlierFilter()], "mode": "any"}
        {"exclude": [Filter1(), Filter2()], "mode": "all"}

    Args:
        mode: How to combine multiple filter masks:
            - "any" (default): Exclude if ANY filter flags the sample
            - "all": Exclude only if ALL filters flag the sample

    Note:
        Exclusion only runs during training mode - prediction samples
        are never excluded to ensure all predictions are generated.
    """

    priority = 5  # Execute early, same as TagController

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match exclude keyword in pipeline."""
        return keyword == "exclude"

    @classmethod
    def use_multi_source(cls) -> bool:
        """Exclusion operates at dataset level, not per-source."""
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """
        Exclusion only runs during training.

        Prediction samples should never be filtered/excluded - we want to
        predict on all provided samples.
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
        Execute sample exclusion operation.

        This method:
        1. Parses exclusion configuration (single filter or list with mode)
        2. Gets training samples (base only, no augmented)
        3. Fits and applies each filter to identify outliers
        4. Combines filter masks using the specified mode
        5. Stores exclusion tags in dataset's indexer (for analysis)
        6. Marks excluded samples in the dataset's indexer
        7. Returns persisted artifacts for reproducibility

        Args:
            step_info: Parsed step containing operator and configuration
            dataset: Dataset to operate on
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index (unused, exclusion is dataset-level)
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binaries (unused, exclusion skips prediction)
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
        filters, filter_mode, cascade_to_augmented = self._parse_config(step)

        if not filters:
            raise ValueError("exclude keyword requires at least one filter")

        # Get train samples (base only, no augmented) for filtering
        train_context = context.with_partition("train")
        base_sample_indices = dataset._indexer.x_indices(  # noqa: SLF001
            train_context.selector, include_augmented=False, include_excluded=False
        )

        if len(base_sample_indices) == 0:
            if runtime_context.step_runner.verbose > 0:
                logger.info("   ExcludeController: No training samples to filter")
            return context, []

        # Get X and y for base train samples (explicitly exclude augmented)
        base_selector = train_context.selector
        X_train = dataset.x(base_selector, layout="2d", concat_source=True, include_augmented=False)
        y_train = dataset.y(base_selector, include_augmented=False)

        # Handle empty or None y_train
        if y_train is None or len(y_train) == 0:
            if runtime_context.step_runner.verbose > 0:
                logger.info("   ExcludeController: No target values available for filtering")
            return context, []

        # Flatten y if needed
        if y_train.ndim > 1:
            y_train = y_train.flatten()

        # Fit all filters and collect masks
        masks: List[np.ndarray] = []
        filter_names: List[str] = []

        for filter_obj in filters:
            try:
                # Fit the filter
                filter_obj.fit(X_train, y_train)

                # Get the mask (True = keep, False = exclude)
                mask = filter_obj.get_mask(X_train, y_train)
                masks.append(mask)
                filter_names.append(self._get_filter_name(filter_obj))

            except ValueError as e:
                # Handle edge cases like insufficient data
                if runtime_context.step_runner.verbose > 0:
                    logger.warning(
                        f"   ExcludeController: {filter_obj.__class__.__name__} "
                        f"could not be applied: {e}"
                    )
                # Create a neutral mask (keep all)
                masks.append(np.ones(len(X_train), dtype=bool))
                filter_names.append(self._get_filter_name(filter_obj))

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

        # Store exclusion tags for each filter (for analysis)
        self._store_exclusion_tags(
            dataset=dataset,
            sample_indices=base_sample_indices,
            filters=filters,
            masks=masks,
            filter_names=filter_names
        )

        # Get indices of samples to exclude
        exclude_mask = ~combined_mask
        samples_to_exclude = base_sample_indices[exclude_mask].tolist()

        # Warn if all samples would be excluded
        n_remaining = int(np.sum(combined_mask))
        if n_remaining == 0 and len(base_sample_indices) > 0:
            import warnings
            warnings.warn(
                f"Exclusion would exclude ALL {len(base_sample_indices)} samples. "
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
                reason = f"exclude({filter_mode}:{','.join(filter_names)})"

            n_excluded = dataset._indexer.mark_excluded(  # noqa: SLF001
                samples_to_exclude,
                reason=reason,
                cascade_to_augmented=cascade_to_augmented
            )

        # Log exclusion summary
        if runtime_context.step_runner.verbose > 0:
            n_cascaded = n_excluded - len(samples_to_exclude) if cascade_to_augmented else 0
            logger.info(
                f"   ExcludeController: Excluded {len(samples_to_exclude)}/{len(base_sample_indices)} "
                f"samples ({100 * len(samples_to_exclude) / len(base_sample_indices):.1f}%)"
                + (f" + {n_cascaded} augmented" if n_cascaded > 0 else "")
            )

        # Update _may_contain_nan flag: if exclusion removed all NaN-containing
        # samples, downstream controllers can skip the NaN guard entirely.
        if dataset._may_contain_nan and n_excluded > 0:
            dataset._may_contain_nan = dataset.has_nan

        # Persist filters for reference (not used in prediction, but for audit)
        artifacts = []
        for filter_obj in filters:
            operator_name = f"exclude_{filter_obj.exclusion_reason}_{runtime_context.next_op()}"
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

    def _parse_config(
        self,
        step: Dict[str, Any]
    ) -> Tuple[List[SampleFilter], str, bool]:
        """
        Parse exclusion configuration from step.

        Args:
            step: Step dictionary containing exclude configuration

        Returns:
            Tuple of (filters, mode, cascade_to_augmented)

        Raises:
            ValueError: If invalid mode is specified
            TypeError: If filter is not a SampleFilter instance
        """
        config = step.get("exclude", {})
        filter_mode = step.get("mode", "any")
        cascade_to_augmented = step.get("cascade_to_augmented", True)

        if filter_mode not in ("any", "all"):
            raise ValueError(f"mode must be 'any' or 'all', got '{filter_mode}'")

        filters = self._parse_filters(config)
        return filters, filter_mode, cascade_to_augmented

    def _parse_filters(
        self,
        config: Union[Any, List[Any]]
    ) -> List[SampleFilter]:
        """
        Parse filter configuration into list of SampleFilter instances.

        Args:
            config: Configuration in one of these formats:
                   - Single filter: YOutlierFilter()
                   - List of filters: [YOutlierFilter(), XOutlierFilter()]

        Returns:
            List of SampleFilter instances

        Raises:
            TypeError: If filter is not a SampleFilter instance
        """
        filters = []

        # If config is already a SampleFilter instance, use it directly
        if isinstance(config, SampleFilter):
            filters.append(config)
            return filters

        # Check if config is a serialized component dict
        if isinstance(config, dict) and any(key in config for key in ("class", "function", "instance")):
            filter_obj = deserialize_component(config)
            if not isinstance(filter_obj, SampleFilter):
                raise TypeError(
                    f"Exclude filter must be a SampleFilter instance, "
                    f"got {type(filter_obj).__name__}"
                )
            filters.append(filter_obj)
            return filters

        if isinstance(config, list):
            # List format: [Filter1(), Filter2()]
            for filter_def in config:
                if isinstance(filter_def, SampleFilter):
                    filter_obj = filter_def
                else:
                    filter_obj = deserialize_component(filter_def)
                if not isinstance(filter_obj, SampleFilter):
                    raise TypeError(
                        f"Exclude filter must be a SampleFilter instance, "
                        f"got {type(filter_obj).__name__}"
                    )
                filters.append(filter_obj)

        else:
            # Single filter (might be serialized)
            filter_obj = deserialize_component(config)
            if not isinstance(filter_obj, SampleFilter):
                raise TypeError(
                    f"Exclude filter must be a SampleFilter instance, "
                    f"got {type(filter_obj).__name__}"
                )
            filters.append(filter_obj)

        return filters

    def _get_filter_name(self, filter_obj: SampleFilter) -> str:
        """
        Get name from a filter for tagging and reporting.

        Uses the filter's tag_name if set, otherwise falls back
        to the filter's exclusion_reason (which defaults to class name).

        Args:
            filter_obj: SampleFilter instance

        Returns:
            Filter name string
        """
        if hasattr(filter_obj, 'tag_name') and filter_obj.tag_name is not None:
            return filter_obj.tag_name
        return filter_obj.exclusion_reason

    def _store_exclusion_tags(
        self,
        dataset: 'SpectroDataset',
        sample_indices: np.ndarray,
        filters: List[SampleFilter],
        masks: List[np.ndarray],
        filter_names: List[str]
    ) -> None:
        """
        Store exclusion tags for each filter for analysis.

        Each filter's exclusion mask is stored as a boolean tag column,
        allowing users to analyze which samples were flagged by which filters.

        Args:
            dataset: Dataset to store tags in
            sample_indices: Sample indices being filtered
            filters: List of filter instances
            masks: List of boolean masks (True = keep, False = exclude)
            filter_names: List of names for each filter
        """
        for filter_obj, mask, name in zip(filters, masks, filter_names):
            # Create tag name with 'excluded_' prefix to indicate these are exclusion tags
            tag_name = f"excluded_{name}"

            # Invert mask: mask=True (keep) -> tag=False, mask=False (exclude) -> tag=True
            tag_values = ~mask

            # Ensure tag column exists
            if not dataset._indexer._store.has_tag_column(tag_name):  # noqa: SLF001
                dataset.add_tag(tag_name, dtype="bool")

            # Set tag values for samples
            dataset.set_tag(tag_name, sample_indices.tolist(), tag_values.tolist())
