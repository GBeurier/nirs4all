"""
Context classes for pipeline execution.

This module provides typed context components that replace the Dict[str, Any] context
pattern used throughout the pipeline system. It separates three distinct concerns:

1. DataSelector: Immutable data selection parameters for dataset.x() and dataset.y()
2. PipelineState: Mutable pipeline state that evolves through transformations
3. StepMetadata: Metadata for controller coordination and step tracking
4. ExecutionContext: Composite context with custom data extensibility

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
from typing import Any, Dict, List, Optional
from copy import deepcopy
from collections.abc import Mapping


@dataclass(frozen=True)
class DataSelector(Mapping):
    """
    Immutable data selection parameters for dataset operations.

    This class replaces the dict-based Selector pattern used by dataset.x() and dataset.y().
    All fields are immutable to ensure data selection consistency.
    It implements the Mapping protocol, so it can be used as a dictionary.

    Processing chains are stored here (not in PipelineState) because:
    - Future flow controllers need processing paths for cache selection
    - Feature caching requires processing chains to identify data variants

    Attributes:
        partition: Data partition to select ("train", "test", "all", "val")
        processing: List of processing chains (one per data source)
        layout: Data layout for X retrieval ("2d", "3d", "4d")
        concat_source: Whether to concatenate multiple sources
        fold_id: Optional fold identifier for cross-validation
        include_augmented: Whether to include augmented samples

    Example:
        >>> selector = DataSelector(partition="train", processing=[["raw"]])
        >>> train_selector = selector.with_partition("train")
        >>> # Use as dict
        >>> print(selector["partition"])
        >>> print(dict(selector))
    """

    partition: str = "all"
    processing: List[List[str]] = field(default_factory=lambda: [["raw"]])
    layout: str = "2d"
    concat_source: bool = True
    fold_id: Optional[int] = None
    include_augmented: bool = False

    def __iter__(self):
        """Iterate over non-None fields."""
        for f in fields(self):
            if getattr(self, f.name) is not None:
                yield f.name

    def __getitem__(self, key):
        """Get field value if not None."""
        try:
            val = getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

        if val is None:
            raise KeyError(key)
        return val

    def __len__(self):
        """Count of non-None fields."""
        return sum(1 for _ in self)

    def with_partition(self, partition: str) -> "DataSelector":
        """
        Create new selector with updated partition.

        Args:
            partition: New partition value

        Returns:
            New DataSelector with updated partition
        """
        return DataSelector(
            partition=partition,
            processing=self.processing,
            layout=self.layout,
            concat_source=self.concat_source,
            fold_id=self.fold_id,
            include_augmented=self.include_augmented
        )

    def with_processing(self, processing: List[List[str]]) -> "DataSelector":
        """
        Create new selector with updated processing chains.

        Args:
            processing: New processing chains

        Returns:
            New DataSelector with updated processing
        """
        return DataSelector(
            partition=self.partition,
            processing=processing,
            layout=self.layout,
            concat_source=self.concat_source,
            fold_id=self.fold_id,
            include_augmented=self.include_augmented
        )

    def with_layout(self, layout: str) -> "DataSelector":
        """
        Create new selector with updated layout.

        Args:
            layout: New layout value

        Returns:
            New DataSelector with updated layout
        """
        return DataSelector(
            partition=self.partition,
            processing=self.processing,
            layout=layout,
            concat_source=self.concat_source,
            fold_id=self.fold_id,
            include_augmented=self.include_augmented
        )

    def with_fold(self, fold_id: Optional[int]) -> "DataSelector":
        """
        Create new selector with updated fold_id.

        Args:
            fold_id: New fold identifier

        Returns:
            New DataSelector with updated fold_id
        """
        return DataSelector(
            partition=self.partition,
            processing=self.processing,
            layout=self.layout,
            concat_source=self.concat_source,
            fold_id=fold_id,
            include_augmented=self.include_augmented
        )

    def with_augmented(self, include_augmented: bool) -> "DataSelector":
        """
        Create new selector with updated include_augmented flag.

        Args:
            include_augmented: Whether to include augmented samples

        Returns:
            New DataSelector with updated include_augmented
        """
        return DataSelector(
            partition=self.partition,
            processing=self.processing,
            layout=self.layout,
            concat_source=self.concat_source,
            fold_id=self.fold_id,
            include_augmented=include_augmented
        )


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
        custom: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize execution context.

        Args:
            selector: Data selector (default: DataSelector())
            state: Pipeline state (default: PipelineState())
            metadata: Step metadata (default: StepMetadata())
            custom: Custom data dict (default: {})
        """
        self.selector = selector if selector is not None else DataSelector()
        self.state = state if state is not None else PipelineState()
        self.metadata = metadata if metadata is not None else StepMetadata()
        self.custom = custom if custom is not None else {}

    def copy(self) -> "ExecutionContext":
        """
        Create a deep copy of this context.

        This preserves the copy semantics expected by controllers.

        Returns:
            Deep copy of ExecutionContext
        """
        return ExecutionContext(
            selector=DataSelector(
                partition=self.selector.partition,
                processing=deepcopy(self.selector.processing),
                layout=self.selector.layout,
                concat_source=self.selector.concat_source,
                fold_id=self.selector.fold_id,
                include_augmented=self.selector.include_augmented
            ),
            state=self.state.copy(),
            metadata=self.metadata.copy(),
            custom=deepcopy(self.custom)
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

    def get_selector(self) -> DataSelector:
        """
        Get the data selector.

        Returns:
            DataSelector instance
        """
        return self.selector
