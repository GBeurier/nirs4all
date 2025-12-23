"""
Source Branch Controller for per-source pipeline execution.

This controller enables per-source pipeline execution for multi-source datasets.
Each data source (e.g., NIR, markers, Raman) can have its own independent
preprocessing pipeline.

Unlike regular branching (`branch`), which creates parallel paths that all
process the same data, source branching assigns each source to a specific
processing pipeline based on its name or index.

Phase 10 Implementation:
- Parse source_branch configurations
- Create per-source execution contexts
- Execute source-specific pipelines
- Support prediction mode
- Integration with merge_sources

Example:
    >>> # Different preprocessing per source
    >>> pipeline = [
    ...     {"source_branch": {
    ...         "NIR": [SNV(), SavitzkyGolay()],
    ...         "markers": [VarianceThreshold(), MinMaxScaler()],
    ...     }},
    ...     {"merge_sources": "concat"},  # Combine sources after
    ...     PLSRegression(n_components=10)
    ... ]
    >>>
    >>> # Automatic source branching (same empty pipeline per source - isolation only)
    >>> pipeline = [
    ...     {"source_branch": "auto"},
    ...     {"merge_sources": "concat"},
    ...     PLSRegression(n_components=10)
    ... ]

Keywords: "source_branch"
Priority: 5 (same as BranchController)
"""

import copy
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.operators.data.merge import SourceBranchConfig
from nirs4all.pipeline.execution.result import StepOutput

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.pipeline.steps.parser import ParsedStep

logger = get_logger(__name__)


class SourceBranchConfigParser:
    """Parser for source_branch step configurations.

    Handles multiple syntax formats for source branching and normalizes
    them to SourceBranchConfig.

    Supported syntaxes:
        - Simple string: "auto" (isolate each source)
        - Dict with source names: {"NIR": [steps], "markers": [steps]}
        - Dict with indices: {0: [steps], 1: [steps]}
        - Dict with special keys: {"_default_": [steps], "_merge_after_": False}
    """

    @classmethod
    def parse(cls, raw_config: Any) -> SourceBranchConfig:
        """Parse raw source_branch configuration into SourceBranchConfig.

        Args:
            raw_config: The value from {"source_branch": raw_config}

        Returns:
            Normalized SourceBranchConfig instance.

        Raises:
            ValueError: If configuration format is invalid.
        """
        if isinstance(raw_config, str):
            return cls._parse_string(raw_config)
        elif isinstance(raw_config, list):
            return cls._parse_list(raw_config)
        elif isinstance(raw_config, dict):
            return cls._parse_dict(raw_config)
        elif isinstance(raw_config, SourceBranchConfig):
            return raw_config
        else:
            raise ValueError(
                f"Invalid source_branch config type: {type(raw_config).__name__}. "
                f"Expected string, list, dict, or SourceBranchConfig."
            )

    @classmethod
    def _parse_string(cls, config_str: str) -> SourceBranchConfig:
        """Parse simple string configuration.

        Args:
            config_str: "auto" or other string mode

        Returns:
            SourceBranchConfig instance.

        Raises:
            ValueError: If string is not recognized.
        """
        if config_str == "auto":
            return SourceBranchConfig(source_pipelines="auto")
        else:
            raise ValueError(
                f"Unknown source_branch mode: '{config_str}'. "
                f"Expected 'auto' or dict configuration."
            )

    @classmethod
    def _parse_list(cls, config_list: List[Any]) -> SourceBranchConfig:
        """Parse list-indexed configuration.

        Converts a list of pipelines to a dict with string indices as keys.
        Each list index maps to the corresponding source by position.

        Example:
            >>> [
            ...     [MinMaxScaler()],           # becomes "0": [MinMaxScaler()]
            ...     [MinMaxScaler()],           # becomes "1": [MinMaxScaler()]
            ...     [PCA(20), MinMaxScaler()]   # becomes "2": [PCA(20), MinMaxScaler()]
            ... ]

        Args:
            config_list: List of pipeline steps, indexed by source position.

        Returns:
            SourceBranchConfig instance with string indices as source keys.
        """
        source_pipelines = {}
        for idx, value in enumerate(config_list):
            # Use string indices as keys (matching source_0, source_1, etc.)
            key = str(idx)
            # Normalize steps to list
            if value is None:
                steps = []
            elif isinstance(value, list):
                steps = value
            else:
                steps = [value]
            source_pipelines[key] = steps

        return SourceBranchConfig(
            source_pipelines=source_pipelines,
            default_pipeline=None,
            merge_after=False,  # Don't auto-merge; user controls with explicit merge step
            merge_strategy="concat",
        )

    @classmethod
    def _parse_dict(cls, config_dict: Dict[str, Any]) -> SourceBranchConfig:
        """Parse dictionary configuration.

        Args:
            config_dict: Dict with source names/indices as keys, steps as values.
                May contain special keys like "_default_", "_merge_after_".

        Returns:
            SourceBranchConfig instance.
        """
        # Extract special configuration keys
        merge_after = config_dict.pop("_merge_after_", True)
        merge_strategy = config_dict.pop("_merge_strategy_", "concat")
        default_pipeline = config_dict.pop("_default_", None)

        # Remaining keys are source -> pipeline mappings
        source_pipelines = {}
        for key, value in config_dict.items():
            # Normalize steps to list
            if value is None:
                steps = []
            elif isinstance(value, list):
                steps = value
            else:
                steps = [value]

            source_pipelines[key] = steps

        return SourceBranchConfig(
            source_pipelines=source_pipelines,
            default_pipeline=default_pipeline,
            merge_after=merge_after,
            merge_strategy=merge_strategy,
        )


@register_controller
class SourceBranchController(OperatorController):
    """Controller for per-source pipeline execution.

    This controller enables per-source pipeline execution for multi-source
    datasets. Each data source gets its own independent processing pipeline.

    Key behaviors:
        - Creates per-source execution contexts
        - Executes source-specific pipelines
        - Stores source contexts for subsequent steps or auto-merge
        - Optionally auto-merges sources after processing

    Unlike regular BranchController:
        - Operates on the data provenance dimension (sources), not execution paths
        - Each source's data is isolated during its pipeline execution
        - Sources can have completely different preprocessing chains
        - Designed for multi-modal data (NIR, markers, Raman, etc.)

    Attributes:
        priority: Controller priority (5 = same as BranchController).
    """

    priority = 5

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if the step matches the source_branch controller.

        Args:
            step: Original step configuration
            operator: Deserialized operator
            keyword: Step keyword

        Returns:
            True if keyword is "source_branch"
        """
        return keyword == "source_branch"

    @classmethod
    def use_multi_source(cls) -> bool:
        """Source branch controller supports multi-source datasets."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Source branch controller should execute in prediction mode."""
        return True

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
        """Execute source branch step.

        For each source, runs a specific sub-pipeline (if defined) and
        updates the processing context. Uses existing infrastructure:

        1. Get source names and current processing chains
        2. For each source with a defined pipeline:
           - Create a context with processing limited to that source
           - Run the sub-pipeline steps
           - Collect artifacts
        3. Update context with new processing chains
        4. Optionally auto-merge sources

        The TransformerController will naturally apply transforms only
        to the source whose processing is in the context.

        Args:
            step_info: Parsed step containing source_branch configuration
            dataset: Dataset to operate on (must have multiple sources)
            context: Pipeline execution context
            runtime_context: Runtime infrastructure context
            source: Data source index
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binary objects for prediction mode
            prediction_store: External prediction store

        Returns:
            Tuple of (updated_context, StepOutput with artifacts)

        Raises:
            ValueError: If dataset has only one source.
        """
        # Parse configuration
        raw_config = step_info.original_step.get("source_branch")
        config = SourceBranchConfigParser.parse(raw_config)

        # Validate multi-source dataset
        n_sources = dataset.n_sources

        if n_sources == 0:
            raise ValueError(
                "source_branch requires a dataset with feature sources. "
                "No sources found in dataset. "
                "[Error: SOURCEBRANCH-E001]"
            )

        if n_sources == 1:
            logger.warning(
                "source_branch called on single-source dataset. "
                "This is effectively a no-op. Consider removing this step. "
                "[Warning: SOURCEBRANCH-E002]"
            )
            # Continue anyway - just passes through the single source

        # Get source names
        source_names = self._get_source_names(dataset, n_sources)
        logger.info(f"Source branching: {n_sources} sources, mode={mode}")

        # Get pipeline mappings for all sources
        source_mappings = config.get_all_source_mappings(source_names)

        # Log configuration
        for src_name, steps in source_mappings.items():
            step_names = self._get_step_names(steps)
            logger.info(f"  Source '{src_name}': {step_names or '[passthrough]'}")

        # V3: Start source branch step recording in trace
        recorder = runtime_context.trace_recorder
        if recorder is not None:
            recorder.start_branch_step(
                step_index=runtime_context.step_number,
                branch_count=n_sources,
                operator_config={"source_branch": True, "n_sources": n_sources},
            )

        # Get current processing chains for all sources
        current_processing = list(context.selector.processing) if context.selector.processing else []
        # Ensure we have processing for all sources
        while len(current_processing) < n_sources:
            # Get default processing for this source from dataset
            src_idx = len(current_processing)
            current_processing.append(dataset.features_processings(src_idx))

        # Store initial context and feature state
        all_artifacts = []

        # Track new processing chains per source after transformations
        new_processing_per_source: List[List[str]] = [list(p) for p in current_processing]

        # Execute per-source pipelines
        source_contexts: List[Dict[str, Any]] = []

        for src_idx, src_name in enumerate(source_names):
            steps = source_mappings.get(src_name, [])

            if not steps:
                # No pipeline for this source - passthrough
                logger.info(f"  Source '{src_name}' (index {src_idx}): [passthrough]")
                source_contexts.append({
                    "source_id": src_idx,
                    "source_name": src_name,
                    "context": context.copy(),
                    "features_snapshot": None,
                    "pipeline_steps": [],
                })
                continue

            logger.info(f"  Processing source '{src_name}' (index {src_idx})")

            # V3: Enter source context in trace recorder
            if recorder is not None:
                recorder.enter_branch(src_idx)

            # Create context with processing for only this source
            # We create a processing list where only the current source has entries
            # Other sources get empty lists so transforms skip them
            source_specific_processing = []
            for i in range(n_sources):
                if i == src_idx:
                    source_specific_processing.append(list(current_processing[i]))
                else:
                    source_specific_processing.append([])  # Empty = skip this source

            source_context = context.copy()
            source_context = source_context.with_processing(source_specific_processing)

            # Store the current source index in custom for reference
            source_context.custom["_current_source_idx"] = src_idx
            source_context.custom["_current_source_name"] = src_name

            # Get source-specific binaries for prediction mode
            source_binaries = loaded_binaries
            if mode in ("predict", "explain"):
                if hasattr(runtime_context, 'artifact_provider') and runtime_context.artifact_provider is not None:
                    # Artifacts for source-specific steps will be loaded by substeps
                    pass

            # Execute source pipeline steps
            for substep_idx, substep in enumerate(steps):
                if hasattr(runtime_context, 'step_runner') and runtime_context.step_runner:
                    runtime_context.substep_number = substep_idx
                    result = runtime_context.step_runner.execute(
                        step=substep,
                        dataset=dataset,
                        context=source_context,
                        runtime_context=runtime_context,
                        loaded_binaries=source_binaries,
                        prediction_store=prediction_store
                    )
                    source_context = result.updated_context
                    all_artifacts.extend(result.artifacts)

            # V3: Exit source context in trace recorder
            if recorder is not None:
                recorder.exit_branch()

            # Update new processing for this source from the context
            if source_context.selector.processing and len(source_context.selector.processing) > src_idx:
                new_processing_per_source[src_idx] = list(source_context.selector.processing[src_idx])

            # Store the source context
            source_contexts.append({
                "source_id": src_idx,
                "source_name": src_name,
                "context": source_context,
                "features_snapshot": None,
                "pipeline_steps": steps,
            })

            logger.success(f"  Source '{src_name}' processing completed")

        # V3: End source branch step in trace
        if recorder is not None:
            recorder.end_step()

        # Build updated context with combined processing from all sources
        result_context = context.copy()
        result_context = result_context.with_processing(new_processing_per_source)

        # Store source contexts for later merge operations
        result_context.custom["source_branch_contexts"] = source_contexts
        result_context.custom["in_source_branch_mode"] = True

        # NOTE: We do NOT set in_branch_mode=True here because source_branch
        # operates on separate sources, not parallel copies of the same data.
        # The merge step will detect in_source_branch_mode and handle it appropriately.
        # Setting in_branch_mode would cause the executor to incorrectly try to
        # replace dataset sources with branch snapshots.

        # Auto-merge if configured
        if config.merge_after:
            logger.info(f"  Auto-merging sources with strategy: {config.merge_strategy}")
            result_context, merge_output = self._auto_merge_sources(
                dataset=dataset,
                context=result_context,
                source_contexts=source_contexts,
                strategy=config.merge_strategy,
            )
            all_artifacts.extend(merge_output.artifacts)

        # Build metadata
        metadata = {
            "source_branch": True,
            "n_sources": n_sources,
            "source_names": source_names,
            "merge_after": config.merge_after,
            "merge_strategy": config.merge_strategy if config.merge_after else None,
            "source_branch_config": config.to_dict(),
        }

        logger.success(
            f"Source branch step completed: {n_sources} sources processed"
            f"{' (auto-merged)' if config.merge_after else ''}"
        )

        return result_context, StepOutput(
            artifacts=all_artifacts,
            metadata=metadata
        )

    def _get_source_names(
        self,
        dataset: "SpectroDataset",
        n_sources: int
    ) -> List[str]:
        """Get source names from dataset.

        Args:
            dataset: The dataset
            n_sources: Number of sources

        Returns:
            List of source names (generates default names if not available)
        """
        # Try to get source names from dataset
        try:
            source_names = []
            for i in range(n_sources):
                # Check if dataset has source_name method
                if hasattr(dataset, 'source_name'):
                    name = dataset.source_name(i)
                    if name:
                        source_names.append(name)
                        continue
                # Fall back to default naming
                source_names.append(f"source_{i}")
            return source_names
        except Exception:
            return [f"source_{i}" for i in range(n_sources)]

    def _get_step_names(self, steps: List[Any]) -> str:
        """Get human-readable names for a list of steps.

        Args:
            steps: List of pipeline steps

        Returns:
            Comma-separated string of step names.
        """
        if not steps:
            return ""

        names = []
        for step in steps:
            if hasattr(step, "__class__"):
                names.append(step.__class__.__name__)
            elif isinstance(step, dict):
                keys = [k for k in step.keys() if not k.startswith("_")]
                if keys:
                    names.append(keys[0])
            else:
                names.append(str(step)[:20])

        return ", ".join(names)

    def _snapshot_source_features(
        self,
        dataset: "SpectroDataset",
        source_idx: int
    ) -> Any:
        """Snapshot features for a specific source.

        Args:
            dataset: The dataset
            source_idx: Source index to snapshot

        Returns:
            List containing deep copy of the source feature data.
            Returns a list (not a single FeatureSource) for compatibility
            with merge controller's _collect_features which expects a list.
        """
        try:
            if source_idx < len(dataset._features.sources):
                # Return as a list for compatibility with merge feature collection
                return [copy.deepcopy(dataset._features.sources[source_idx])]
            return None
        except Exception as e:
            logger.warning(f"Failed to snapshot source {source_idx}: {e}")
            return None

    def _auto_merge_sources(
        self,
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        source_contexts: List[Dict[str, Any]],
        strategy: str,
    ) -> Tuple["ExecutionContext", StepOutput]:
        """Automatically merge sources after source branching.

        Calls the MergeController's execute_source_merge method to combine
        all sources back into a unified feature matrix.

        Args:
            dataset: The dataset
            context: Execution context
            source_contexts: List of source context dicts (for metadata)
            strategy: Merge strategy ("concat", "stack", "dict")

        Returns:
            Tuple of (updated_context, StepOutput)
        """
        import numpy as np
        from nirs4all.operators.data.merge import SourceMergeConfig

        # Collect features from all sources using the updated context's processing
        source_features = []
        source_names = []
        n_sources = dataset.n_sources

        for src_idx in range(n_sources):
            try:
                # Get features for this source
                X = dataset.x(
                    selector=context.selector,
                    layout="2d",
                    concat_source=False,
                    include_augmented=True,
                    include_excluded=False
                )

                # X is a list of per-source arrays
                if isinstance(X, list) and src_idx < len(X):
                    features = X[src_idx]
                elif not isinstance(X, list) and src_idx == 0:
                    features = X
                else:
                    logger.warning(f"Source {src_idx} not found in dataset output")
                    continue

                source_features.append(features)
                source_names.append(source_contexts[src_idx]["source_name"] if src_idx < len(source_contexts) else f"source_{src_idx}")

            except Exception as e:
                logger.warning(f"Failed to collect features from source {src_idx}: {e}")
                continue

        if not source_features:
            logger.warning("No source features to merge")
            return context, StepOutput()

        result_context = context.copy()

        # Apply merge strategy
        if strategy == "concat":
            merged = np.concatenate(source_features, axis=1)
        elif strategy == "stack":
            # Check if shapes are compatible
            shapes = [f.shape[1] for f in source_features]
            if len(set(shapes)) > 1:
                logger.warning(
                    f"Source feature dimensions differ: {shapes}. "
                    "Falling back to concat."
                )
                merged = np.concatenate(source_features, axis=1)
            else:
                merged = np.stack(source_features, axis=1)
        elif strategy == "dict":
            # Dict strategy - store in context for downstream use
            merged_dict = {
                name: features
                for name, features in zip(source_names, source_features)
            }
            result_context.custom["merged_sources_dict"] = merged_dict
            result_context.custom["source_merge_applied"] = True
            result_context.custom["in_source_branch_mode"] = False
            return result_context, StepOutput(metadata={"merge_strategy": "dict"})
        else:
            merged = np.concatenate(source_features, axis=1)

        # Store merged features in dataset
        dataset.add_merged_features(
            features=merged,
            processing_name="source_merged",
            source=0
        )

        # Update processing to use the merged features
        result_context = result_context.with_processing([["source_merged"]])

        # Clear source branch mode
        result_context.custom["source_branch_contexts"] = []
        result_context.custom["in_source_branch_mode"] = False
        result_context.custom["source_merge_applied"] = True

        logger.info(f"  Auto-merged {len(source_features)} sources → shape {merged.shape}")

        return result_context, StepOutput(
            metadata={"auto_merge": True, "merge_strategy": strategy}
        )


# Expose for imports
__all__ = [
    "SourceBranchController",
    "SourceBranchConfigParser",
]
