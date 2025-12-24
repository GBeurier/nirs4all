"""
Repetition Transformation Controllers.

This module provides controllers for transforming spectral repetitions
(multiple spectra per sample) into either separate sources or additional
preprocessings.

These transformations physically reshape the dataset structure:

- `rep_to_sources`: Each repetition becomes a separate data source
  Input: 1 source × (120 samples, 1 pp, 500 features)
  Output: 4 sources × (30 samples, 1 pp, 500 features)

- `rep_to_pp`: Repetitions become additional preprocessing slots
  Input: 1 source × (120 samples, 1 pp, 500 features)
  Output: 1 source × (30 samples, 4 pp, 500 features)

These are typically used early in the pipeline, before cross-validation,
as they change the fundamental dataset structure.

Example:
    >>> # Transform 4 repetitions per sample into 4 sources
    >>> pipeline = [
    ...     {"rep_to_sources": "Sample_ID"},
    ...     ShuffleSplit(n_splits=3),
    ...     PLSRegression(n_components=10)
    ... ]
    >>>
    >>> # Use dataset's aggregate column (from DatasetConfigs)
    >>> pipeline = [
    ...     {"rep_to_sources": True},  # Uses aggregate column
    ...     {"source_branch": {...}},   # Per-source preprocessing
    ...     {"merge_sources": "concat"},
    ...     PLSRegression()
    ... ]
    >>>
    >>> # Transform to preprocessings for multi-PP models
    >>> pipeline = [
    ...     {"rep_to_pp": "Sample_ID"},
    ...     ShuffleSplit(n_splits=3),
    ...     {"model": NiConNet()}  # Handles multi-PP input
    ... ]

Keywords: "rep_to_sources", "rep_to_pp"
Priority: 3 (early in pipeline, before CV)
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.operators.data.repetition import RepetitionConfig
from nirs4all.pipeline.execution.result import StepOutput

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.pipeline.steps.parser import ParsedStep

logger = get_logger(__name__)


@register_controller
class RepToSourcesController(OperatorController):
    """Controller for transforming repetitions into separate data sources.

    This controller handles the `rep_to_sources` pipeline keyword, which
    groups samples by a metadata column (typically sample ID) and reshapes
    each repetition index into a separate data source.

    Before: 1 source × (n_samples, n_pp, n_features)
    After:  n_reps sources × (n_unique_samples, n_pp, n_features)

    This enables:
        - Per-repetition preprocessing via source_branch
        - Multi-source modeling strategies
        - Repetition-aware feature fusion

    Attributes:
        priority: Controller priority (3 = early, before CV).
    """

    priority = 3

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if the step matches the rep_to_sources controller.

        Args:
            step: Original step configuration
            operator: Deserialized operator
            keyword: Step keyword

        Returns:
            True if keyword is "rep_to_sources"
        """
        return keyword == "rep_to_sources"

    @classmethod
    def use_multi_source(cls) -> bool:
        """This controller operates on the whole dataset, not per-source."""
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Repetition transformation should NOT run in prediction mode.

        The transformation happens once during training. During prediction,
        the model expects the same structure that was used during training.
        The controller should be skipped in prediction mode - the user must
        ensure prediction data has the same structure as training data
        after transformation.
        """
        return False

    def execute(
        self,
        step_info: "ParsedStep",
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Execute rep_to_sources transformation.

        Reshapes the dataset by grouping samples by the specified column
        and creating one source per repetition index.

        Args:
            step_info: Parsed step containing rep_to_sources configuration
            dataset: Dataset to transform
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index (not used, operates on all sources)
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binary objects (not used)
            prediction_store: External prediction store (not used)

        Returns:
            Tuple of (context, StepOutput with transformation info)

        Raises:
            ValueError: If column not found or groups have unequal sizes
                and on_unequal="error".
        """
        # Parse configuration
        raw_config = step_info.original_step.get("rep_to_sources")
        config = RepetitionConfig.from_step_value(raw_config)

        # Log what we're doing
        column_desc = config.column if config.column else f"dataset.aggregate ({dataset.aggregate})"
        logger.info(f"Transforming repetitions to sources using column: {column_desc}")

        # Perform the transformation
        original_samples = dataset.num_samples
        original_sources = dataset.n_sources

        dataset.reshape_reps_to_sources(config)

        new_samples = dataset.num_samples
        new_sources = dataset.n_sources

        # Build output
        output = StepOutput()
        output.set_transform_metadata({
            "transformation": "rep_to_sources",
            "column": config.resolve_column(dataset.aggregate) if not config.uses_dataset_aggregate else config.column,
            "original_samples": original_samples,
            "new_samples": new_samples,
            "original_sources": original_sources,
            "new_sources": new_sources,
            "n_reps": new_sources // original_sources,
        })

        return context, output


@register_controller
class RepToPPController(OperatorController):
    """Controller for transforming repetitions into additional preprocessings.

    This controller handles the `rep_to_pp` pipeline keyword, which
    groups samples by a metadata column and reshapes each repetition
    into a preprocessing dimension.

    Before: n_sources × (n_samples, n_pp, n_features)
    After:  n_sources × (n_unique_samples, n_pp × n_reps, n_features)

    This enables:
        - Multi-preprocessing input for models like NiConNet
        - Repetition-as-preprocessing fusion strategies
        - Consistent sample count for cross-validation

    Attributes:
        priority: Controller priority (3 = early, before CV).
    """

    priority = 3

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if the step matches the rep_to_pp controller.

        Args:
            step: Original step configuration
            operator: Deserialized operator
            keyword: Step keyword

        Returns:
            True if keyword is "rep_to_pp"
        """
        return keyword == "rep_to_pp"

    @classmethod
    def use_multi_source(cls) -> bool:
        """This controller operates on the whole dataset, not per-source."""
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Repetition transformation should NOT run in prediction mode.

        The transformation happens once during training. During prediction,
        the model expects the same structure that was used during training.
        """
        return False

    def execute(
        self,
        step_info: "ParsedStep",
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Execute rep_to_pp transformation.

        Reshapes the dataset by grouping samples by the specified column
        and stacking repetitions into the preprocessing dimension.

        Args:
            step_info: Parsed step containing rep_to_pp configuration
            dataset: Dataset to transform
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index (not used, operates on all sources)
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binary objects (not used)
            prediction_store: External prediction store (not used)

        Returns:
            Tuple of (context, StepOutput with transformation info)

        Raises:
            ValueError: If column not found or groups have unequal sizes
                and on_unequal="error".
        """
        # Parse configuration
        raw_config = step_info.original_step.get("rep_to_pp")
        config = RepetitionConfig.from_step_value(raw_config)

        # Log what we're doing
        column_desc = config.column if config.column else f"dataset.aggregate ({dataset.aggregate})"
        logger.info(f"Transforming repetitions to preprocessings using column: {column_desc}")

        # Perform the transformation
        original_samples = dataset.num_samples
        original_pp = len(dataset.features_processings(0))

        dataset.reshape_reps_to_preprocessings(config)

        new_samples = dataset.num_samples
        new_pp = len(dataset.features_processings(0))

        # Build output
        output = StepOutput()
        output.set_transform_metadata({
            "transformation": "rep_to_pp",
            "column": config.resolve_column(dataset.aggregate) if not config.uses_dataset_aggregate else config.column,
            "original_samples": original_samples,
            "new_samples": new_samples,
            "original_pp": original_pp,
            "new_pp": new_pp,
            "n_reps": new_pp // original_pp,
        })

        return context, output


# Expose for imports
__all__ = [
    "RepToSourcesController",
    "RepToPPController",
]
