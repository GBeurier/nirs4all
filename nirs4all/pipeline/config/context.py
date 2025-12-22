"""
Context classes for pipeline execution.

This module provides typed context components that replace the Dict[str, Any] context
pattern used throughout the pipeline system. It separates three distinct concerns:

1. DataSelector: Immutable data selection parameters for dataset.x() and dataset.y()
2. PipelineState: Mutable pipeline state that evolves through transformations
3. StepMetadata: Metadata for controller coordination and step tracking
4. ExecutionContext: Composite context with custom data extensibility
5. ArtifactProvider: Interface for providing artifacts during prediction replay

The separation enables:
- Type safety throughout the codebase
- Clear interfaces between components
- Better testability
- Explicit controller communication
- Future extensibility via custom dict

Example:
    >>> selector = DataSelector(partition="train", processing=[["raw"]])
    >>> state = PipelineState(y_processing="numeric")
    >>> metadata = StepMetadata(keyword="transform")
    >>> context = ExecutionContext(selector=selector, state=state, metadata=metadata)
    >>> new_context = context.with_partition("test")
"""

from dataclasses import dataclass, field, replace as dataclass_replace, fields
from typing import Any, Dict, List, Optional, Iterator, Protocol, Tuple, Union
from copy import deepcopy
from collections.abc import MutableMapping
from abc import ABC, abstractmethod


@dataclass
class DataSelector(MutableMapping):
    """
    Mutable data selection parameters for dataset operations.

    This class replaces the dict-based Selector pattern used by dataset.x() and dataset.y().
    It implements the MutableMapping protocol, so it can be used as a dictionary.
    It supports arbitrary keys via an internal dict to allow flexibility.

    Attributes:
        partition: Data partition to select ("train", "test", "all", "val")
        processing: List of processing chains (one per data source)
        layout: Data layout for X retrieval ("2d", "3d", "4d")
        concat_source: Whether to concatenate multiple sources
        fold_id: Optional fold identifier for cross-validation
        include_augmented: Whether to include augmented samples
        y: Optional target processing version (e.g. "numeric", "scaled")
        branch_id: Optional branch identifier for pipeline branching (0-indexed)
            DEPRECATED: Use branch_path instead for nested branch support.
        branch_path: List of branch indices for nested branching (e.g., [0, 2] for
            branch 2 inside branch 0). Empty list means pre-branch/shared artifacts.
        branch_name: Optional human-readable branch name for tracking

    Example:
        >>> selector = DataSelector(partition="train", processing=[["raw"]])
        >>> selector["y"] = "scaled"  # Direct modification
        >>> selector["custom_key"] = "value"  # Arbitrary keys supported
        >>> print(selector["partition"])
    """

    partition: str = "all"
    processing: List[List[str]] = field(default_factory=lambda: [["raw"]])
    layout: str = "2d"
    concat_source: bool = True
    fold_id: Optional[int] = None
    include_augmented: bool = False
    y: Optional[str] = None
    branch_id: Optional[int] = None
    branch_path: List[int] = field(default_factory=list)
    branch_name: Optional[str] = None
    _extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __iter__(self) -> Iterator[str]:
        """Iterate over non-None fields and extra keys."""
        # Yield defined fields if they are not None
        for f in fields(self):
            if f.name == "_extra":
                continue
            if getattr(self, f.name) is not None:
                yield f.name
        # Yield extra keys
        yield from self._extra

    def __getitem__(self, key: str) -> Any:
        """Get field value or extra key."""
        # Check if it's a defined field
        if hasattr(self, key) and key != "_extra":
            val = getattr(self, key)
            if val is not None:
                return val

        # Check extra keys
        if key in self._extra:
            return self._extra[key]

        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set field value or extra key."""
        if hasattr(self, key) and key != "_extra":
            setattr(self, key, value)
        else:
            self._extra[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete extra key or set field to None."""
        if hasattr(self, key) and key != "_extra":
            setattr(self, key, None)
        elif key in self._extra:
            del self._extra[key]
        else:
            raise KeyError(key)

    def __len__(self) -> int:
        """Count of non-None fields and extra keys."""
        return sum(1 for _ in self)

    def copy(self) -> "DataSelector":
        """Create a deep copy of the selector."""
        new_selector = DataSelector(
            partition=self.partition,
            processing=deepcopy(self.processing),
            layout=self.layout,
            concat_source=self.concat_source,
            fold_id=self.fold_id,
            include_augmented=self.include_augmented,
            y=self.y,
            branch_id=self.branch_id,
            branch_path=list(self.branch_path),
            branch_name=self.branch_name
        )
        new_selector._extra = deepcopy(self._extra)
        return new_selector

    def with_partition(self, partition: str) -> "DataSelector":
        """
        Create new selector with updated partition.

        Args:
            partition: New partition value

        Returns:
            New DataSelector with updated partition
        """
        new_selector = self.copy()
        new_selector.partition = partition
        return new_selector

    def with_processing(self, processing: List[List[str]]) -> "DataSelector":
        """
        Create new selector with updated processing chains.

        Args:
            processing: New processing chains

        Returns:
            New DataSelector with updated processing
        """
        new_selector = self.copy()
        new_selector.processing = processing
        return new_selector

    def with_layout(self, layout: str) -> "DataSelector":
        """
        Create new selector with updated layout.

        Args:
            layout: New layout value

        Returns:
            New DataSelector with updated layout
        """
        new_selector = self.copy()
        new_selector.layout = layout
        return new_selector

    def with_fold(self, fold_id: Optional[int]) -> "DataSelector":
        """
        Create new selector with updated fold_id.

        Args:
            fold_id: New fold identifier

        Returns:
            New DataSelector with updated fold_id
        """
        new_selector = self.copy()
        new_selector.fold_id = fold_id
        return new_selector

    def with_augmented(self, include_augmented: bool) -> "DataSelector":
        """
        Create new selector with updated include_augmented flag.

        Args:
            include_augmented: Whether to include augmented samples

        Returns:
            New DataSelector with updated include_augmented
        """
        new_selector = self.copy()
        new_selector.include_augmented = include_augmented
        return new_selector

    def with_branch(
        self,
        branch_id: Optional[int] = None,
        branch_name: Optional[str] = None,
        branch_path: Optional[List[int]] = None
    ) -> "DataSelector":
        """
        Create new selector with updated branch information.

        Args:
            branch_id: Branch identifier (0-indexed). DEPRECATED: Use branch_path.
            branch_name: Human-readable branch name
            branch_path: List of branch indices for nested branching

        Returns:
            New DataSelector with updated branch info
        """
        new_selector = self.copy()
        new_selector.branch_id = branch_id
        new_selector.branch_name = branch_name

        # Support both old branch_id and new branch_path
        if branch_path is not None:
            new_selector.branch_path = list(branch_path)
        elif branch_id is not None:
            # Convert single branch_id to branch_path for compatibility
            new_selector.branch_path = [branch_id]
        else:
            new_selector.branch_path = []

        return new_selector


@dataclass
class PipelineState:
    """
    Mutable pipeline state that evolves through execution.

    This class tracks state that changes as the pipeline executes:
    - Y transformation state (e.g., "encoded_LabelEncoder_001")
    - Current step number in execution

    Unlike DataSelector, this is mutable because state must evolve.

    Attributes:
        y_processing: Current y transformation identifier
        step_number: Current step number (1-indexed)
        mode: Execution mode ("train", "predict", "explain")

    Example:
        >>> state = PipelineState(y_processing="numeric")
        >>> state.step_number = 2  # Mutable update
        >>> state.y_processing = "encoded_LabelEncoder_001"
    """

    y_processing: str = "numeric"
    step_number: int = 0
    mode: str = "train"

    def copy(self) -> "PipelineState":
        """
        Create a deep copy of this state.

        Returns:
            Deep copy of PipelineState
        """
        return PipelineState(
            y_processing=self.y_processing,
            step_number=self.step_number,
            mode=self.mode
        )


@dataclass
class StepMetadata:
    """
    Metadata for controller coordination and step tracking.

    This class handles:
    - Controller coordination flags (augment_sample, add_feature, replace_processing)
    - Step identification (step_id, keyword)
    - Target specification for augmentation operations

    These are ephemeral flags set/cleared between steps for controller communication.

    Attributes:
        keyword: Step keyword (e.g., "model", "transform")
        step_id: Step identifier (e.g., "001", "002")
        augment_sample: Flag for sample augmentation mode
        add_feature: Flag for feature augmentation mode
        replace_processing: Flag to replace processing chains
        target_samples: Target sample IDs for augmentation
        target_features: Target feature indices for augmentation

    Example:
        >>> metadata = StepMetadata(keyword="transform", step_id="001")
        >>> metadata.augment_sample = True
        >>> metadata.target_samples = [42]
    """

    keyword: str = ""
    step_id: str = ""
    augment_sample: bool = False
    add_feature: bool = False
    replace_processing: bool = False
    target_samples: List[int] = field(default_factory=list)
    target_features: List[int] = field(default_factory=list)

    def copy(self) -> "StepMetadata":
        """
        Create a deep copy of this metadata.

        Returns:
            Deep copy of StepMetadata
        """
        return StepMetadata(
            keyword=self.keyword,
            step_id=self.step_id,
            augment_sample=self.augment_sample,
            add_feature=self.add_feature,
            replace_processing=self.replace_processing,
            target_samples=self.target_samples.copy(),
            target_features=self.target_features.copy()
        )

    def reset_ephemeral_flags(self) -> None:
        """Reset ephemeral flags after step execution.

        Clears augment_sample, add_feature, replace_processing flags
        and target lists to prevent leakage between steps.
        """
        self.augment_sample = False
        self.add_feature = False
        self.replace_processing = False
        self.target_samples.clear()
        self.target_features.clear()


class ArtifactProvider(ABC):
    """Abstract interface for providing artifacts during prediction replay.

    The ArtifactProvider enables controller-agnostic artifact injection:
    controllers request artifacts by step index rather than by name matching,
    which is deterministic and works with any controller type.

    This interface is used during prediction mode to provide pre-loaded
    artifacts (transformers, models, etc.) to controllers without requiring
    them to know about the artifact storage system.

    Implementations:
        - MapArtifactProvider: In-memory dictionary-based provider
        - LoaderArtifactProvider: Wraps ArtifactLoader for lazy loading

    Example:
        >>> provider = MapArtifactProvider(artifact_map)
        >>> artifacts = provider.get_artifacts_for_step(step_index=2)
        >>> for artifact_id, obj in artifacts:
        ...     process(obj)
    """

    @abstractmethod
    def get_artifact(self, step_index: int, fold_id: Optional[int] = None) -> Optional[Any]:
        """Get a single artifact for a step.

        Args:
            step_index: 1-based step index
            fold_id: Optional fold ID for fold-specific artifacts

        Returns:
            Artifact object or None if not found
        """
        pass

    @abstractmethod
    def get_artifacts_for_step(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None,
        branch_id: Optional[int] = None,
        source_index: Optional[int] = None,
        substep_index: Optional[int] = None
    ) -> List[Tuple[str, Any]]:
        """Get all artifacts for a step.

        Args:
            step_index: 1-based step index
            branch_path: Optional branch path filter
            branch_id: Optional branch ID filter
            source_index: Optional source/dataset index filter for multi-source
            substep_index: Optional substep index filter for branch substeps

        Returns:
            List of (artifact_id, artifact_object) tuples
        """
        pass

    @abstractmethod
    def get_fold_artifacts(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None
    ) -> List[Tuple[int, Any]]:
        """Get all fold-specific artifacts for a step.

        Args:
            step_index: 1-based step index
            branch_path: Optional branch path filter

        Returns:
            List of (fold_id, artifact_object) tuples, sorted by fold_id
        """
        pass

    @abstractmethod
    def has_artifacts_for_step(self, step_index: int) -> bool:
        """Check if artifacts exist for a step.

        Args:
            step_index: 1-based step index

        Returns:
            True if artifacts are available for this step
        """
        pass

    def get_primary_artifact(self, step_index: int) -> Optional[Any]:
        """Get the primary artifact for a step.

        The primary artifact is typically the main model or transformer
        for the step. Default implementation returns the first artifact.

        Args:
            step_index: 1-based step index

        Returns:
            Primary artifact object or None if not found
        """
        artifacts = self.get_artifacts_for_step(step_index)
        if artifacts:
            return artifacts[0][1]
        return None

    def get_artifact_by_chain(self, chain_path: str) -> Optional[Any]:
        """Get artifact by V3 chain path (optional V3 method).

        Args:
            chain_path: Full operator chain path (e.g., "s1.MinMaxScaler>s3.PLS")

        Returns:
            Artifact object or None if not found
        """
        return None  # Default implementation returns None

    def get_artifacts_for_chain_prefix(
        self,
        chain_prefix: str
    ) -> List[Tuple[str, Any]]:
        """Get all artifacts matching a chain path prefix (optional V3 method).

        Args:
            chain_prefix: Chain path prefix to match

        Returns:
            List of (chain_path, artifact_object) tuples
        """
        return []  # Default implementation returns empty list


class MapArtifactProvider(ArtifactProvider):
    """In-memory artifact provider backed by a dictionary.

    Provides artifacts from a pre-loaded dictionary mapping step indices
    to artifacts. Used when artifacts are resolved from an ExecutionTrace
    or when loading from a bundle.

    Attributes:
        artifact_map: Dictionary mapping step_index to list of (artifact_id, object) tuples
        fold_weights: Optional fold weights for CV ensemble averaging

    Example:
        >>> artifact_map = {
        ...     1: [("0001:1:all", snv_transformer)],
        ...     2: [("0001:2:0", model_fold0), ("0001:2:1", model_fold1)]
        ... }
        >>> provider = MapArtifactProvider(artifact_map)
        >>> transformer = provider.get_artifact(step_index=1)
    """

    def __init__(
        self,
        artifact_map: Dict[int, List[Tuple[str, Any]]],
        fold_weights: Optional[Dict[int, float]] = None,
        primary_artifacts: Optional[Dict[int, str]] = None
    ):
        """Initialize map-based artifact provider.

        Args:
            artifact_map: Mapping of step_index -> list of (artifact_id, object)
            fold_weights: Optional fold weights for CV models
            primary_artifacts: Optional mapping of step_index -> primary artifact_id
        """
        self.artifact_map = artifact_map
        self.fold_weights = fold_weights or {}
        self.primary_artifacts = primary_artifacts or {}

    def get_artifact(self, step_index: int, fold_id: Optional[int] = None) -> Optional[Any]:
        """Get a single artifact for a step.

        If fold_id is specified, returns the fold-specific artifact.
        Otherwise, returns the primary or first artifact.

        Args:
            step_index: 1-based step index
            fold_id: Optional fold ID for fold-specific artifacts

        Returns:
            Artifact object or None if not found
        """
        artifacts = self.artifact_map.get(step_index, [])
        if not artifacts:
            return None

        if fold_id is not None:
            # Look for fold-specific artifact
            for artifact_id, obj in artifacts:
                # Check if artifact_id contains fold info (e.g., "0001:2:0")
                parts = artifact_id.split(":")
                if len(parts) >= 3:
                    try:
                        artifact_fold = int(parts[-1])
                        if artifact_fold == fold_id:
                            return obj
                    except ValueError:
                        pass
            return None

        # Return primary artifact if specified
        if step_index in self.primary_artifacts:
            primary_id = self.primary_artifacts[step_index]
            for artifact_id, obj in artifacts:
                if artifact_id == primary_id:
                    return obj

        # Return first artifact
        return artifacts[0][1] if artifacts else None

    def get_artifacts_for_step(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None,
        branch_id: Optional[int] = None,
        source_index: Optional[int] = None,
        substep_index: Optional[int] = None
    ) -> List[Tuple[str, Any]]:
        """Get all artifacts for a step.

        Args:
            step_index: 1-based step index
            branch_path: Optional branch path filter (not used in map provider)
            branch_id: Optional branch ID filter (not used in map provider)
            source_index: Optional source/dataset index filter (not used in map provider)
            substep_index: Optional substep index filter (not used in map provider)

        Returns:
            List of (artifact_id, artifact_object) tuples
        """
        return self.artifact_map.get(step_index, [])

    def get_fold_artifacts(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None
    ) -> List[Tuple[int, Any]]:
        """Get all fold-specific artifacts for a step.

        Args:
            step_index: 1-based step index
            branch_path: Optional branch path filter (not used in map provider)

        Returns:
            List of (fold_id, artifact_object) tuples, sorted by fold_id
        """
        artifacts = self.artifact_map.get(step_index, [])
        fold_artifacts = []

        for artifact_id, obj in artifacts:
            parts = artifact_id.split(":")
            if len(parts) >= 3:
                fold_part = parts[-1]
                if fold_part != "all":
                    try:
                        fold_id = int(fold_part)
                        fold_artifacts.append((fold_id, obj))
                    except ValueError:
                        pass

        return sorted(fold_artifacts, key=lambda x: x[0])

    def has_artifacts_for_step(self, step_index: int) -> bool:
        """Check if artifacts exist for a step.

        Args:
            step_index: 1-based step index

        Returns:
            True if artifacts are available for this step
        """
        return step_index in self.artifact_map and len(self.artifact_map[step_index]) > 0

    def get_fold_weights(self) -> Dict[int, float]:
        """Get fold weights for CV ensemble averaging.

        Returns:
            Dictionary mapping fold_id to weight
        """
        return self.fold_weights.copy()


class LoaderArtifactProvider(ArtifactProvider):
    """Artifact provider backed by an ArtifactLoader.

    Wraps an ArtifactLoader to provide artifacts on-demand with lazy loading
    and caching. Used when loading from a manifest for prediction.

    Attributes:
        loader: The underlying ArtifactLoader
        trace: Optional ExecutionTrace for step-to-artifact mapping
    """

    def __init__(
        self,
        loader: Any,  # ArtifactLoader
        trace: Optional[Any] = None  # ExecutionTrace
    ):
        """Initialize loader-based artifact provider.

        Args:
            loader: ArtifactLoader instance for loading artifacts
            trace: Optional ExecutionTrace for step mapping
        """
        self.loader = loader
        self.trace = trace

    def get_artifact(self, step_index: int, fold_id: Optional[int] = None) -> Optional[Any]:
        """Get a single artifact for a step.

        If trace is available, uses trace to find artifact IDs.
        Otherwise, uses loader's step-based lookup.

        Args:
            step_index: 1-based step index
            fold_id: Optional fold ID for fold-specific artifacts

        Returns:
            Artifact object or None if not found
        """
        if self.trace is not None:
            step = self.trace.get_step(step_index)
            if step and step.artifacts:
                if fold_id is not None and step.artifacts.fold_artifact_ids:
                    artifact_id = step.artifacts.fold_artifact_ids.get(fold_id)
                    if artifact_id:
                        return self.loader.load_by_id(artifact_id)
                elif step.artifacts.primary_artifact_id:
                    return self.loader.load_by_id(step.artifacts.primary_artifact_id)
                elif step.artifacts.artifact_ids:
                    return self.loader.load_by_id(step.artifacts.artifact_ids[0])
            return None

        # Fallback: use loader's step-based lookup
        artifacts = self.loader.load_for_step(step_index=step_index, fold_id=fold_id)
        if artifacts:
            return artifacts[0][1]
        return None

    def get_artifacts_for_step(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None,
        branch_id: Optional[int] = None,
        source_index: Optional[int] = None,
        substep_index: Optional[int] = None
    ) -> List[Tuple[str, Any]]:
        """Get all artifacts for a step.

        Args:
            step_index: 1-based step index
            branch_path: Optional branch path filter
            branch_id: Optional branch ID filter
            source_index: Optional source index filter for multi-source pipelines
            substep_index: Optional substep index filter (not used in loader provider)

        Returns:
            List of (artifact_id, artifact_object) tuples
        """
        # Determine target branch for filtering
        target_branch: Optional[int] = None
        if branch_path is not None and len(branch_path) > 0:
            target_branch = branch_path[0]
        elif branch_id is not None:
            target_branch = branch_id

        if self.trace is not None:
            step = self.trace.get_step(step_index)
            if step and step.artifacts:
                # Get artifact IDs, optionally filtered by source_index
                if source_index is not None and step.artifacts.by_source:
                    # Use by_source index for filtering
                    artifact_ids = step.artifacts.by_source.get(source_index, [])
                else:
                    # Use all artifact IDs
                    artifact_ids = step.artifacts.artifact_ids

                results = []
                for artifact_id in artifact_ids:
                    try:
                        # Filter by branch_path using artifact record metadata
                        if target_branch is not None:
                            record = self.loader.get_record(artifact_id)
                            if record is not None:
                                artifact_branch = None
                                if record.branch_path and len(record.branch_path) > 0:
                                    artifact_branch = record.branch_path[0]
                                # Include if: artifact has no branch (shared) or matches target
                                if artifact_branch is not None and artifact_branch != target_branch:
                                    continue  # Skip - wrong branch

                        obj = self.loader.load_by_id(artifact_id)
                        results.append((artifact_id, obj))
                    except (KeyError, FileNotFoundError):
                        pass
                return results
            return []

        # Fallback: use loader's step-based lookup
        return self.loader.load_for_step(step_index=step_index, branch_path=branch_path)

    def get_fold_artifacts(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None
    ) -> List[Tuple[int, Any]]:
        """Get all fold-specific artifacts for a step.

        Args:
            step_index: 1-based step index
            branch_path: Optional branch path filter

        Returns:
            List of (fold_id, artifact_object) tuples, sorted by fold_id
        """
        if self.trace is not None:
            step = self.trace.get_step(step_index)
            if step and step.artifacts and step.artifacts.fold_artifact_ids:
                results = []
                for fold_id, artifact_id in step.artifacts.fold_artifact_ids.items():
                    try:
                        obj = self.loader.load_by_id(artifact_id)
                        results.append((fold_id, obj))
                    except (KeyError, FileNotFoundError):
                        pass
                return sorted(results, key=lambda x: x[0])
            return []

        # Fallback: use loader's fold model lookup
        return self.loader.load_fold_models(step_index=step_index, branch_path=branch_path)

    def has_artifacts_for_step(self, step_index: int) -> bool:
        """Check if artifacts exist for a step.

        Args:
            step_index: 1-based step index

        Returns:
            True if artifacts are available for this step
        """
        if self.trace is not None:
            step = self.trace.get_step(step_index)
            return step is not None and step.has_artifacts()

        # Fallback: use loader's step check
        return self.loader.has_binaries_for_step(step_index)

    def get_artifact_by_chain(self, chain_path: str) -> Optional[Any]:
        """Get artifact by V3 chain path.

        Args:
            chain_path: Full operator chain path (e.g., "s1.MinMaxScaler>s3.PLS")

        Returns:
            Artifact object or None if not found
        """
        if hasattr(self.loader, 'load_by_chain'):
            return self.loader.load_by_chain(chain_path)
        return None

    def get_artifacts_for_chain_prefix(
        self,
        chain_prefix: str
    ) -> List[Tuple[str, Any]]:
        """Get all artifacts matching a chain path prefix.

        Args:
            chain_prefix: Chain path prefix to match

        Returns:
            List of (chain_path, artifact_object) tuples
        """
        if hasattr(self.loader, 'load_by_chain_prefix'):
            return self.loader.load_by_chain_prefix(chain_prefix)
        return []


class ExecutionContext:
    """
    Composite execution context with extensibility.

    This class combines the three context components and provides:
    - Immutable data selection via DataSelector
    - Mutable state tracking via PipelineState
    - Controller coordination via StepMetadata
    - Custom data storage for controller-specific needs

    The context supports deep copying for controller isolation while sharing
    processing chains between selector and operations.

    Attributes:
        selector: Immutable data selector
        state: Mutable pipeline state
        metadata: Mutable step metadata
        custom: Dict for controller-specific custom data
        aggregate_column: Sample aggregation column for prediction aggregation.
            - None: No aggregation (default)
            - 'y': Aggregate by y_true values
            - str: Aggregate by specified metadata column

    Example:
        >>> context = ExecutionContext(
        ...     selector=DataSelector(partition="train"),
        ...     state=PipelineState(y_processing="numeric"),
        ...     metadata=StepMetadata(keyword="transform")
        ... )
        >>> context.custom["my_controller"] = {"threshold": 0.5}
        >>> train_ctx = context.with_partition("train")
    """

    def __init__(
        self,
        selector: Optional[DataSelector] = None,
        state: Optional[PipelineState] = None,
        metadata: Optional[StepMetadata] = None,
        custom: Optional[Dict[str, Any]] = None,
        aggregate_column: Optional[str] = None
    ):
        """
        Initialize execution context.

        Args:
            selector: Data selector (default: DataSelector())
            state: Pipeline state (default: PipelineState())
            metadata: Step metadata (default: StepMetadata())
            custom: Custom data dict (default: {})
            aggregate_column: Sample aggregation column for prediction aggregation.
                - None: No aggregation (default)
                - 'y': Aggregate by y_true values
                - str: Aggregate by specified metadata column
        """
        self.selector = selector if selector is not None else DataSelector()
        self.state = state if state is not None else PipelineState()
        self.metadata = metadata if metadata is not None else StepMetadata()
        self.custom = custom if custom is not None else {}
        self.aggregate_column = aggregate_column

    def copy(self) -> "ExecutionContext":
        """
        Create a deep copy of this context.

        This preserves the copy semantics expected by controllers.

        Returns:
            Deep copy of ExecutionContext
        """
        return ExecutionContext(
            selector=self.selector.copy(),
            state=self.state.copy(),
            metadata=self.metadata.copy(),
            custom=deepcopy(self.custom),
            aggregate_column=self.aggregate_column
        )

    def with_partition(self, partition: str) -> "ExecutionContext":
        """
        Create new context with updated partition.

        Args:
            partition: New partition value

        Returns:
            New ExecutionContext with updated partition
        """
        new_ctx = self.copy()
        new_ctx.selector = new_ctx.selector.with_partition(partition)
        return new_ctx

    def with_processing(self, processing: List[List[str]]) -> "ExecutionContext":
        """
        Create new context with updated processing chains.

        Args:
            processing: New processing chains

        Returns:
            New ExecutionContext with updated processing
        """
        new_ctx = self.copy()
        new_ctx.selector = new_ctx.selector.with_processing(processing)
        return new_ctx

    def with_layout(self, layout: str) -> "ExecutionContext":
        """
        Create new context with updated layout.

        Args:
            layout: New layout value

        Returns:
            New ExecutionContext with updated layout
        """
        new_ctx = self.copy()
        new_ctx.selector = new_ctx.selector.with_layout(layout)
        return new_ctx

    def with_step_number(self, step_number: int) -> "ExecutionContext":
        """
        Create new context with updated step number.

        Args:
            step_number: New step number

        Returns:
            New ExecutionContext with updated step number
        """
        new_ctx = self.copy()
        new_ctx.state = dataclass_replace(new_ctx.state, step_number=step_number)
        return new_ctx

    def with_y(self, y_processing: str) -> "ExecutionContext":
        """
        Create new context with updated y processing.

        Args:
            y_processing: New y processing value

        Returns:
            New ExecutionContext with updated y processing
        """
        new_ctx = self.copy()
        new_ctx.state = dataclass_replace(new_ctx.state, y_processing=y_processing)
        return new_ctx

    def with_metadata(self, **kwargs) -> "ExecutionContext":
        """
        Create new context with updated metadata fields.

        Args:
            **kwargs: Metadata fields to update

        Returns:
            New ExecutionContext with updated metadata
        """
        new_ctx = self.copy()
        new_ctx.metadata = dataclass_replace(new_ctx.metadata, **kwargs)
        return new_ctx

    def with_branch(
        self,
        branch_id: Optional[int] = None,
        branch_name: Optional[str] = None
    ) -> "ExecutionContext":
        """
        Create new context with updated branch information.

        Args:
            branch_id: Branch identifier (0-indexed)
            branch_name: Human-readable branch name

        Returns:
            New ExecutionContext with updated branch info
        """
        new_ctx = self.copy()
        new_ctx.selector = new_ctx.selector.with_branch(branch_id, branch_name)
        return new_ctx

    def get_selector(self) -> DataSelector:
        """
        Get the data selector.

        Returns:
            DataSelector instance
        """
        return self.selector


@dataclass
class RuntimeContext:
    """
    Runtime infrastructure components for pipeline execution.

    This class holds references to infrastructure components that are needed
    during execution but are not part of the data flow or pipeline state.
    It replaces the "God Object" pattern of passing the runner everywhere.

    Attributes:
        saver: SimulationSaver for file operations
        manifest_manager: ManifestManager for pipeline tracking
        artifact_loader: ArtifactLoader for predict/explain modes
        artifact_provider: ArtifactProvider for controller-agnostic artifact injection (Phase 3)
        artifact_registry: ArtifactRegistry for artifact management (v2 system)
        pipeline_uid: Current pipeline unique identifier
        step_runner: StepRunner for executing sub-steps
        operation_count: Counter for operation IDs
        substep_number: Current substep number
        trace_recorder: TraceRecorder for recording execution traces (Phase 2)
        retrain_config: RetrainConfig for retrain mode control (Phase 7)
    """
    saver: Any = None
    manifest_manager: Any = None
    artifact_loader: Any = None
    artifact_provider: Optional["ArtifactProvider"] = None  # Phase 3: controller-agnostic artifact injection
    artifact_registry: Any = None
    pipeline_uid: Optional[str] = None
    step_runner: Any = None
    step_number: int = 0
    operation_count: int = 0
    substep_number: int = -1
    processing_counter: int = 0  # Global counter for unique processing indices within a step
    artifact_load_counter: Dict[int, int] = field(default_factory=dict)  # Per-source artifact load counter
    target_model: Optional[Dict[str, Any]] = None
    explainer: Any = None
    trace_recorder: Any = None  # TraceRecorder instance for execution trace recording
    retrain_config: Any = None  # Phase 7: RetrainConfig for retrain mode control

    def next_op(self) -> int:
        """Get the next operation ID."""
        self.operation_count += 1
        return self.operation_count

    def next_processing_index(self) -> int:
        """Get the next unique processing index for artifact identification.

        This counter persists across all sub-operations within a step (e.g.,
        feature_augmentation), ensuring each transformer gets a unique substep_index.
        The counter is reset at the start of each step.

        Returns:
            int: A unique processing index within the current step.
        """
        idx = self.processing_counter
        self.processing_counter += 1
        return idx

    def reset_processing_counter(self) -> None:
        """Reset the processing counter at the start of each step."""
        self.processing_counter = 0
        self.artifact_load_counter = {}  # Also reset artifact load counter

    def next_artifact_load_index(self, source_index: int) -> int:
        """Get the next artifact load index for a source during prediction.

        This counter tracks how many artifacts have been loaded for each source
        across all sub-operations within a step (e.g., feature_augmentation).

        Args:
            source_index: The source index to track.

        Returns:
            int: The next artifact index to load for this source.
        """
        if source_index not in self.artifact_load_counter:
            self.artifact_load_counter[source_index] = 0
        idx = self.artifact_load_counter[source_index]
        self.artifact_load_counter[source_index] += 1
        return idx

    def should_train_step(self, step_index: int, is_model: bool = False) -> bool:
        """Determine if a step should train based on retrain configuration.

        Phase 7 Feature:
            When a retrain_config is present, this method delegates to it
            to determine whether a step should train or use existing artifacts.

        Args:
            step_index: 1-based step index
            is_model: Whether this is the model step

        Returns:
            True if the step should train, False if it should use existing artifacts
        """
        if self.retrain_config is None:
            # No retrain config: default behavior based on artifact_provider
            if self.artifact_provider is not None:
                # If we have artifacts for this step, use them (don't train)
                return not self.artifact_provider.has_artifacts_for_step(step_index)
            # No artifacts available: train
            return True

        # Delegate to retrain_config
        return self.retrain_config.should_train_step(step_index, is_model)

    def record_step_start(
        self,
        step_index: int,
        operator_type: str = "",
        operator_class: str = "",
        operator_config: Optional[Dict[str, Any]] = None,
        branch_path: Optional[List[int]] = None,
        branch_name: str = "",
        mode: str = "train"
    ) -> None:
        """Record the start of a step execution in the trace.

        Args:
            step_index: 1-based step index
            operator_type: Type of operator (e.g., "transform", "model")
            operator_class: Class name of operator
            operator_config: Serialized operator configuration
            branch_path: Branch indices if in branch context
            branch_name: Human-readable branch name
            mode: Execution mode ("train", "predict", "explain")
        """
        if self.trace_recorder is not None:
            from nirs4all.pipeline.trace import StepExecutionMode

            exec_mode = (
                StepExecutionMode.PREDICT if mode in ("predict", "explain")
                else StepExecutionMode.TRAIN
            )
            self.trace_recorder.start_step(
                step_index=step_index,
                operator_type=operator_type,
                operator_class=operator_class,
                operator_config=operator_config,
                execution_mode=exec_mode,
                branch_path=branch_path,
                branch_name=branch_name
            )

    def record_step_artifact(
        self,
        artifact_id: str,
        is_primary: bool = False,
        fold_id: Optional[int] = None,
        chain_path: Optional[str] = None,
        branch_path: Optional[List[int]] = None,
        source_index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an artifact created during the current step (V3).

        Args:
            artifact_id: The V3 artifact ID
            is_primary: Whether this is the primary artifact (e.g., model)
            fold_id: CV fold ID if fold-specific artifact
            chain_path: V3 operator chain path for this artifact
            branch_path: Branch path for indexing
            source_index: Source index for multi-source artifacts
            metadata: Additional artifact metadata
        """
        if self.trace_recorder is not None:
            self.trace_recorder.record_artifact(
                artifact_id=artifact_id,
                is_primary=is_primary,
                fold_id=fold_id,
                chain_path=chain_path,
                branch_path=branch_path,
                source_index=source_index,
                metadata=metadata
            )

    def record_step_end(
        self,
        is_model: bool = False,
        fold_weights: Optional[Dict[int, float]] = None,
        skip_trace: bool = False
    ) -> None:
        """Record the end of a step execution.

        Args:
            is_model: Whether this is the model step
            fold_weights: Per-fold weights for CV models
            skip_trace: If True, don't add this step to the trace
        """
        if self.trace_recorder is not None:
            self.trace_recorder.end_step(
                is_model=is_model,
                fold_weights=fold_weights,
                skip_trace=skip_trace
            )

    def record_input_shapes(
        self,
        input_shape: Optional[tuple] = None,
        features_shape: Optional[List[tuple]] = None
    ) -> None:
        """Record input shapes for the current step.

        Args:
            input_shape: 2D layout shape (samples, features)
            features_shape: List of 3D shapes per source (samples, processings, features)
        """
        if self.trace_recorder is not None:
            self.trace_recorder.record_input_shapes(
                input_shape=input_shape,
                features_shape=features_shape
            )

    def record_output_shapes(
        self,
        output_shape: Optional[tuple] = None,
        features_shape: Optional[List[tuple]] = None
    ) -> None:
        """Record output shapes for the current step.

        Args:
            output_shape: 2D layout shape (samples, features)
            features_shape: List of 3D shapes per source (samples, processings, features)
        """
        if self.trace_recorder is not None:
            self.trace_recorder.record_output_shapes(
                output_shape=output_shape,
                features_shape=features_shape
            )

    def get_trace_id(self) -> Optional[str]:
        """Get the current trace ID.

        Returns:
            Trace ID or None if no trace recorder
        """
        if self.trace_recorder is not None:
            return self.trace_recorder.trace_id
        return None

    def get_execution_trace(self) -> Optional[Any]:
        """Get the current execution trace.

        Returns the trace object that has been built during execution.
        This can be used to generate post-execution diagrams with actual shapes.

        Returns:
            ExecutionTrace object or None if no trace recorder
        """
        if self.trace_recorder is not None:
            return self.trace_recorder.trace
        return None
