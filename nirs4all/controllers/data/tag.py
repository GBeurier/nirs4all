"""
Controller for sample tagging operations.

This controller handles the `tag` keyword, computing and storing tags
on samples without removing them. Tags can later be used for branching,
analysis, or conditional processing.
"""

from typing import TYPE_CHECKING, Any, Optional, Union

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
class TagController(OperatorController):
    """
    Controller for sample tagging operations.

    This controller computes tags on samples using SampleFilter instances
    and stores the results as tag columns in the dataset's indexer.
    Unlike `exclude`, the `tag` keyword never removes samples - it only
    stores computed values for later use.

    Tags can be used for:
    - Analysis and reporting (e.g., identifying outliers)
    - Conditional branching (e.g., `branch: {by_tag: "is_outlier"}`)
    - Grouping samples for specialized processing

    Pipeline syntax:
        # Single filter (tag name from filter's tag_name or class name)
        {"tag": YOutlierFilter(method="iqr")}

        # Multiple filters (each stores its own tag)
        {"tag": [YOutlierFilter(), XOutlierFilter()]}

        # Named tags (explicit tag names)
        {"tag": {"outliers": YOutlierFilter(), "leverage": HighLeverageFilter()}}

    Note:
        Tags are computed fresh during both training and prediction modes.
        This allows analyzing prediction samples with the same criteria.
    """

    priority = 5  # Execute early, same as ExcludeController

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match tag keyword in pipeline."""
        return keyword == "tag"

    @classmethod
    def use_multi_source(cls) -> bool:
        """Tag operations are dataset-level, not per-source."""
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """
        Tags are computed fresh on prediction data.

        This allows identifying outliers or special cases in new data
        for analysis and conditional processing.
        """
        return True

    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: list[tuple[str, Any]] | None = None,
        prediction_store: Any | None = None
    ) -> tuple['ExecutionContext', list]:
        """
        Execute tagging operation.

        This method:
        1. Parses tag configuration (single, list, or dict of filters)
        2. Gets samples for the current partition
        3. Fits filters on training data (or uses loaded binaries in prediction)
        4. Computes tag values (boolean mask from get_mask)
        5. Stores tags in dataset's indexer
        6. Returns persisted artifacts for reproducibility

        Args:
            step_info: Parsed step containing operator and configuration
            dataset: Dataset to operate on
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index (unused, tagging is dataset-level)
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded filter binaries for prediction mode
            prediction_store: External prediction store (unused)

        Returns:
            Tuple of (updated_context, persisted_artifacts)

        Raises:
            ValueError: If no filters are specified
            TypeError: If filter is not a SampleFilter instance
        """
        # Extract configuration from step
        step = step_info.original_step
        config = step.get("tag", {})

        # Parse taggers from configuration
        taggers = self._parse_taggers(config)

        if not taggers:
            raise ValueError("tag keyword requires at least one filter")

        # Get samples based on current partition
        partition = context.selector.partition
        include_augmented = context.selector.include_augmented

        # Get sample indices for the current context
        sample_indices = dataset._indexer.x_indices(  # noqa: SLF001
            context.selector, include_augmented=include_augmented, include_excluded=False
        )

        if len(sample_indices) == 0:
            if runtime_context.step_runner.verbose > 0:
                logger.info("   TagController: No samples to tag")
            return context, []

        # Get X and y data for samples
        selector = context.selector.with_augmented(include_augmented)
        X_data = dataset.x(selector, layout="2d", concat_source=True)
        assert isinstance(X_data, np.ndarray), "concat_source=True should return a single ndarray"
        X = X_data
        y = dataset.y(selector)

        # Flatten y if needed
        if y is not None and y.ndim > 1:
            y = y.flatten()

        # Process each tagger
        artifacts = []
        binary_index = 0  # Index into loaded_binaries

        for tag_name, filter_obj in taggers:
            try:
                # In prediction mode, try to use loaded binary
                if mode == "predict" and loaded_binaries and binary_index < len(loaded_binaries):
                    _, loaded_filter = loaded_binaries[binary_index]
                    binary_index += 1
                    if isinstance(loaded_filter, SampleFilter):
                        filter_obj = loaded_filter
                else:
                    # Fit the filter in training mode
                    filter_obj.fit(X, y)

                # Get the mask (True = keep = False for outlier tag)
                mask = filter_obj.get_mask(X, y)

                # For boolean tags, invert mask to get "is_outlier" semantics
                # mask=True means "keep" (not outlier), we want tag=True to mean "is outlier"
                tag_values = ~mask

                # Ensure tag column exists
                if not dataset._indexer._store.has_tag_column(tag_name):  # noqa: SLF001
                    dataset.add_tag(tag_name, dtype="bool")

                # Set tag values for samples
                dataset.set_tag(tag_name, sample_indices.tolist(), tag_values.tolist())

                if runtime_context.step_runner.verbose > 0:
                    n_tagged = int(np.sum(tag_values))
                    logger.info(
                        f"   TagController: Set tag '{tag_name}' - "
                        f"{n_tagged}/{len(sample_indices)} samples tagged True"
                    )

                # Persist filter for reproducibility (only in train mode)
                if mode == "train":
                    operator_name = f"tag_{tag_name}_{runtime_context.next_op()}"
                    artifact = (filter_obj, operator_name, "sklearn")
                    artifacts.append(artifact)

            except ValueError as e:
                # Handle edge cases like insufficient data
                if runtime_context.step_runner.verbose > 0:
                    logger.warning(
                        f"   TagController: {filter_obj.__class__.__name__} "
                        f"could not be applied: {e}"
                    )
                # Don't create tag if filter fails

        return context, artifacts

    def _parse_taggers(
        self,
        config: Any | list[Any] | dict[str, Any]
    ) -> list[tuple[str, SampleFilter]]:
        """
        Parse tagger configuration into list of (tag_name, filter) tuples.

        Args:
            config: Configuration in one of three formats:
                   - Single filter: YOutlierFilter()
                   - List of filters: [YOutlierFilter(), XOutlierFilter()]
                   - Named dict: {"outliers": YOutlierFilter()}

        Returns:
            List of (tag_name, filter) tuples

        Raises:
            TypeError: If filter is not a SampleFilter instance
        """
        taggers = []

        # If config is already a SampleFilter instance, use it directly
        if isinstance(config, SampleFilter):
            tag_name = self._get_tag_name(config)
            taggers.append((tag_name, config))
            return taggers

        # Check if config is a serialized component dict (has "class", "function", or "instance" key)
        if isinstance(config, dict) and any(key in config for key in ("class", "function", "instance")):
            # This is a serialized component, deserialize it
            filter_obj = deserialize_component(config)
            if not isinstance(filter_obj, SampleFilter):
                raise TypeError(
                    f"Tag filter must be a SampleFilter instance, "
                    f"got {type(filter_obj).__name__}"
                )
            tag_name = self._get_tag_name(filter_obj)
            taggers.append((tag_name, filter_obj))
            return taggers

        if isinstance(config, dict):
            # Named dict format: {"tag_name": Filter()}
            for tag_name, filter_def in config.items():
                # Handle both live instances and serialized components
                filter_obj = filter_def if isinstance(filter_def, SampleFilter) else deserialize_component(filter_def)
                if not isinstance(filter_obj, SampleFilter):
                    raise TypeError(
                        f"Tag filter must be a SampleFilter instance, "
                        f"got {type(filter_obj).__name__}"
                    )
                taggers.append((tag_name, filter_obj))

        elif isinstance(config, list):
            # List format: [Filter1(), Filter2()]
            for filter_def in config:
                # Handle both live instances and serialized components
                filter_obj = filter_def if isinstance(filter_def, SampleFilter) else deserialize_component(filter_def)
                if not isinstance(filter_obj, SampleFilter):
                    raise TypeError(
                        f"Tag filter must be a SampleFilter instance, "
                        f"got {type(filter_obj).__name__}"
                    )
                tag_name = self._get_tag_name(filter_obj)
                taggers.append((tag_name, filter_obj))

        else:
            # Single filter (might be serialized)
            filter_obj = deserialize_component(config)
            if not isinstance(filter_obj, SampleFilter):
                raise TypeError(
                    f"Tag filter must be a SampleFilter instance, "
                    f"got {type(filter_obj).__name__}"
                )
            tag_name = self._get_tag_name(filter_obj)
            taggers.append((tag_name, filter_obj))

        return taggers

    def _get_tag_name(self, filter_obj: SampleFilter) -> str:
        """
        Get tag name from a filter.

        Uses the filter's tag_name attribute if set, otherwise falls back
        to the filter's exclusion_reason (which defaults to class name).

        Args:
            filter_obj: SampleFilter instance

        Returns:
            Tag name string
        """
        # Check for tag_name attribute (added in Task 2.1.2)
        if hasattr(filter_obj, 'tag_name') and filter_obj.tag_name is not None:
            return filter_obj.tag_name

        # Fall back to exclusion_reason (which defaults to class name)
        return filter_obj.exclusion_reason
