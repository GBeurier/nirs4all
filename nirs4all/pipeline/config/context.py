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
from typing import Any, Dict, List, Optional, Iterator
from copy import deepcopy
from collections.abc import MutableMapping


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
            y=self.y
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
            selector=self.selector.copy(),
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
