from typing import Any, Dict, List, Tuple, Optional, Set, TYPE_CHECKING

from sklearn.base import TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.spectra.spectra_dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.pipeline.steps.parser import ParsedStep
import copy


# Valid action modes for feature_augmentation
VALID_ACTIONS = ("extend", "add", "replace")


@register_controller
class FeatureAugmentationController(OperatorController):
    """Controller for feature augmentation with multiple action modes.

    The feature_augmentation controller supports three action modes that control
    how preprocessing operations interact with existing processings:

    - **extend** (default): Add new processings to the set. Each operation runs
      independently on the base processing. If a processing already exists, it
      is not duplicated. Growth pattern is linear.

    - **add**: Chain each operation on top of ALL existing processings. Keep
      original processings alongside new chained versions. Growth pattern is
      multiplicative with originals (n + n×m).

    - **replace**: Chain each operation on top of ALL existing processings.
      Discard original processings, keeping only the chained versions. Growth
      pattern is multiplicative without originals (n×m).

    Example:
        >>> # Extend mode (default) - linear growth
        >>> {"feature_augmentation": [SNV, Gaussian], "action": "extend"}
        >>> # With raw_A already present: raw_A, raw_SNV, raw_Gaussian

        >>> # Add mode - multiplicative with originals
        >>> {"feature_augmentation": [SNV, Gaussian], "action": "add"}
        >>> # With raw_A present: raw_A, raw_A_SNV, raw_A_Gaussian

        >>> # Replace mode - multiplicative, discards originals
        >>> {"feature_augmentation": [SNV, Gaussian], "action": "replace"}
        >>> # With raw_A present: raw_A_SNV, raw_A_Gaussian (raw_A discarded)
    """

    priority = 10

    @staticmethod
    def normalize_generator_spec(spec: Any) -> Any:
        """Normalize generator spec for feature_augmentation context.

        In feature_augmentation context, multi-selection should use combinations
        by default since the order of parallel feature channels doesn't matter.
        Translates legacy 'size' to 'pick' for explicit semantics.

        Args:
            spec: Generator specification (may contain _or_, size, pick, arrange).

        Returns:
            Normalized spec with 'size' converted to 'pick' if needed.
        """
        if not isinstance(spec, dict):
            return spec

        # If explicit pick/arrange specified, honor it
        if "pick" in spec or "arrange" in spec:
            return spec

        # Convert legacy size to pick (combinations) for feature_augmentation
        if "size" in spec and "_or_" in spec:
            result = dict(spec)
            result["pick"] = result.pop("size")
            return result

        return spec

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword == "feature_augmentation"

    @classmethod
    def use_multi_source(cls) -> bool:
        """Check if the operator supports multi-source datasets."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Feature augmentation should NOT execute during prediction mode - transformations are already applied and saved."""
        return True

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
    ) -> Tuple['ExecutionContext', List[Tuple[str, bytes]]]:
        """Execute feature augmentation with specified action mode.

        Args:
            step_info: Parsed step information containing the operation list and action mode.
            dataset: The spectroscopic dataset to process.
            context: Current execution context with processing state.
            runtime_context: Runtime infrastructure for step execution.
            source: Source index (-1 for all sources).
            mode: Execution mode ("train", "predict", etc.).
            loaded_binaries: Pre-loaded binary artifacts for prediction mode.
            prediction_store: Store for prediction-time state.

        Returns:
            Tuple of (updated_context, artifacts_list).

        Raises:
            ValueError: If action mode is invalid.
        """
        op = step_info.operator

        try:
            initial_context = context.copy()
            original_source_processings = copy.deepcopy(initial_context.selector.processing)
            all_artifacts = []

            # Parse action mode (default: "add" for backward compatibility)
            action = step_info.original_step.get("action", "add")
            if action not in VALID_ACTIONS:
                raise ValueError(
                    f"Invalid action: '{action}'. Must be one of {VALID_ACTIONS}."
                )

            operations = step_info.original_step["feature_augmentation"]

            # Skip empty operations
            if not operations:
                return context, all_artifacts

            if action == "extend":
                context, all_artifacts = self._execute_extend_mode(
                    operations, dataset, initial_context, runtime_context,
                    original_source_processings, loaded_binaries, prediction_store
                )
            elif action == "add":
                context, all_artifacts = self._execute_add_mode(
                    operations, dataset, initial_context, runtime_context,
                    original_source_processings, loaded_binaries, prediction_store
                )
            elif action == "replace":
                context, all_artifacts = self._execute_replace_mode(
                    operations, dataset, initial_context, runtime_context,
                    original_source_processings, loaded_binaries, prediction_store
                )

            return context, all_artifacts

        except Exception as e:
            logger.error(f"Error applying feature augmentation: {e}")
            raise

    def _execute_extend_mode(
        self,
        operations: List[Any],
        dataset: 'SpectroDataset',
        initial_context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        original_source_processings: List[List[str]],
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple['ExecutionContext', List[Tuple[str, bytes]]]:
        """Execute extend mode: add new processings to set (no chaining).

        Each operation runs independently on the base processing (typically "raw").
        If a processing already exists in the set, it is not duplicated.
        This mode produces linear growth.

        Args:
            operations: List of preprocessing operations to apply.
            dataset: The dataset to process.
            initial_context: Starting execution context.
            runtime_context: Runtime infrastructure.
            original_source_processings: Original processing chains per source.
            loaded_binaries: Pre-loaded artifacts for prediction mode.
            prediction_store: Prediction-time state store.

        Returns:
            Tuple of (updated_context, artifacts_list).
        """
        all_artifacts = []

        # Track existing processings per source (use set for deduplication)
        existing_processings_per_source: List[Set[str]] = [
            set(procs) for procs in original_source_processings
        ]

        # Get base processing (first/root processing) for each source
        base_processings = []
        for src_procs in original_source_processings:
            # Use "raw" as base, or first processing if available
            base = src_procs[0] if src_procs else "raw"
            base_processings.append([base])

        for i, operation in enumerate(operations):
            if operation is None:
                continue

            # Each operation starts from the base processing (not chained)
            local_context = initial_context.copy()
            local_context = local_context.with_metadata(add_feature=True)

            # Use base processing for this operation
            local_context = local_context.with_processing(copy.deepcopy(base_processings))

            # Run substep
            if runtime_context.step_runner:
                runtime_context.substep_number += 1
                result = runtime_context.step_runner.execute(
                    operation, dataset, local_context, runtime_context,
                    loaded_binaries=loaded_binaries, prediction_store=prediction_store
                )
                all_artifacts.extend(result.artifacts)

                # Track new processings (for deduplication awareness)
                for sdx in range(dataset.n_sources):
                    new_procs = dataset.features_processings(sdx)
                    existing_processings_per_source[sdx].update(new_procs)

        # Collect all processings from the dataset
        new_processing = []
        for sdx in range(dataset.n_sources):
            processing_ids = dataset.features_processings(sdx)
            new_processing.append(processing_ids)

        context = initial_context.with_processing(new_processing)
        return context, all_artifacts

    def _execute_add_mode(
        self,
        operations: List[Any],
        dataset: 'SpectroDataset',
        initial_context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        original_source_processings: List[List[str]],
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple['ExecutionContext', List[Tuple[str, bytes]]]:
        """Execute add mode: chain operations on all existing, keep originals.

        Each operation is chained on top of ALL existing processings. Original
        processings are preserved alongside the new chained versions.
        This mode produces multiplicative growth with originals (n + n×m).

        This is the legacy/backward-compatible behavior.

        Args:
            operations: List of preprocessing operations to apply.
            dataset: The dataset to process.
            initial_context: Starting execution context.
            runtime_context: Runtime infrastructure.
            original_source_processings: Original processing chains per source.
            loaded_binaries: Pre-loaded artifacts for prediction mode.
            prediction_store: Prediction-time state store.

        Returns:
            Tuple of (updated_context, artifacts_list).
        """
        all_artifacts = []

        for i, operation in enumerate(operations):
            if operation is None:
                continue

            # Each operation starts from the original processings (parallel chaining)
            source_processings = copy.deepcopy(original_source_processings)
            local_context = initial_context.copy()
            local_context = local_context.with_metadata(add_feature=True)
            local_context = local_context.with_processing(copy.deepcopy(source_processings))

            # Run substep
            if runtime_context.step_runner:
                runtime_context.substep_number += 1
                result = runtime_context.step_runner.execute(
                    operation, dataset, local_context, runtime_context,
                    loaded_binaries=loaded_binaries, prediction_store=prediction_store
                )
                all_artifacts.extend(result.artifacts)

        # Collect all processings from the dataset (includes originals + new)
        new_processing = []
        for sdx in range(dataset.n_sources):
            processing_ids = dataset.features_processings(sdx)
            new_processing.append(processing_ids)

        context = initial_context.with_processing(new_processing)
        return context, all_artifacts

    def _execute_replace_mode(
        self,
        operations: List[Any],
        dataset: 'SpectroDataset',
        initial_context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        original_source_processings: List[List[str]],
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple['ExecutionContext', List[Tuple[str, bytes]]]:
        """Execute replace mode: chain operations on all existing, discard originals.

        Each operation is chained on top of ALL existing processings. Original
        processings are discarded from the context (only chained versions remain).
        This mode produces multiplicative growth without originals (n×m).

        Note: Original processings remain in the dataset but are excluded from
        the context's processing list. This allows them to be used if needed later.

        Args:
            operations: List of preprocessing operations to apply.
            dataset: The dataset to process.
            initial_context: Starting execution context.
            runtime_context: Runtime infrastructure.
            original_source_processings: Original processing chains per source.
            loaded_binaries: Pre-loaded artifacts for prediction mode.
            prediction_store: Prediction-time state store.

        Returns:
            Tuple of (updated_context, artifacts_list).
        """
        all_artifacts = []

        # Track which processings existed before this step (to exclude them)
        original_processing_sets: List[Set[str]] = [
            set(procs) for procs in original_source_processings
        ]

        for i, operation in enumerate(operations):
            if operation is None:
                continue

            # Each operation starts from the original processings (parallel chaining)
            source_processings = copy.deepcopy(original_source_processings)
            local_context = initial_context.copy()
            local_context = local_context.with_metadata(add_feature=True)
            local_context = local_context.with_processing(copy.deepcopy(source_processings))

            # Run substep
            if runtime_context.step_runner:
                runtime_context.substep_number += 1
                result = runtime_context.step_runner.execute(
                    operation, dataset, local_context, runtime_context,
                    loaded_binaries=loaded_binaries, prediction_store=prediction_store
                )
                all_artifacts.extend(result.artifacts)

        # Collect processings, EXCLUDING the original ones
        new_processing = []
        for sdx in range(dataset.n_sources):
            all_procs = dataset.features_processings(sdx)
            # Filter out original processings (keep only newly chained ones)
            filtered_procs = [
                proc for proc in all_procs
                if proc not in original_processing_sets[sdx]
            ]
            # If no new processings were created, keep originals (safety fallback)
            if not filtered_procs:
                filtered_procs = list(original_source_processings[sdx])
            new_processing.append(filtered_procs)

        context = initial_context.with_processing(new_processing)
        return context, all_artifacts
