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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from copy import deepcopy


@dataclass(frozen=True)
class DataSelector:
    """
    Immutable data selection parameters for dataset operations.

    This class replaces the dict-based Selector pattern used by dataset.x() and dataset.y().
    All fields are immutable to ensure data selection consistency.

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
        >>> test_selector = selector.with_partition("test")
    """

    partition: str = "all"
    processing: List[List[str]] = field(default_factory=lambda: [["raw"]])
    layout: str = "2d"
    concat_source: bool = True
    fold_id: Optional[int] = None
    include_augmented: bool = False

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dict format for backward compatibility.

        Omits None values to match original dict-based context behavior.

        Returns:
            Dict representation of selector (excluding None values)
        """
        result = {}

        # Only include non-None values
        if self.partition is not None:
            result["partition"] = self.partition
        if self.processing is not None:
            result["processing"] = self.processing
        if self.layout is not None:
            result["layout"] = self.layout
        if self.concat_source is not None:
            result["concat_source"] = self.concat_source
        if self.include_augmented is not None:
            result["include_augmented"] = self.include_augmented
        if self.fold_id is not None:
            result["fold_id"] = self.fold_id

        return result


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

    def get_selector(self) -> DataSelector:
        """
        Get the data selector.

        Returns:
            DataSelector instance
        """
        return self.selector

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dict format for backward compatibility.

        This allows gradual migration from dict-based context.

        Returns:
            Dict representation combining all components
        """
        result = self.selector.to_dict()
        result["y"] = self.state.y_processing
        result["step_number"] = self.state.step_number
        result["mode"] = self.state.mode
        result["keyword"] = self.metadata.keyword
        result["step_id"] = self.metadata.step_id

        if self.metadata.augment_sample:
            result["augment_sample"] = True
        if self.metadata.add_feature:
            result["add_feature"] = True
        if self.metadata.replace_processing:
            result["replace_processing"] = True
        if self.metadata.target_samples:
            result["target_samples"] = self.metadata.target_samples
        if self.metadata.target_features:
            result["target_features"] = self.metadata.target_features

        # Include custom data
        result.update(self.custom)

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionContext":
        """
        Create ExecutionContext from dict format.

        This supports migration from old dict-based context.

        Args:
            data: Dict representation of context

        Returns:
            ExecutionContext instance
        """
        # Extract known fields
        selector = DataSelector(
            partition=data.get("partition", "all"),
            processing=data.get("processing", [["raw"]]),
            layout=data.get("layout", "2d"),
            concat_source=data.get("concat_source", True),
            fold_id=data.get("fold_id"),
            include_augmented=data.get("include_augmented", False)
        )

        state = PipelineState(
            y_processing=data.get("y", "numeric"),
            step_number=data.get("step_number", 0),
            mode=data.get("mode", "train")
        )

        metadata = StepMetadata(
            keyword=data.get("keyword", ""),
            step_id=data.get("step_id", ""),
            augment_sample=data.get("augment_sample", False),
            add_feature=data.get("add_feature", False),
            replace_processing=data.get("replace_processing", False),
            target_samples=data.get("target_samples", []),
            target_features=data.get("target_features", [])
        )

        # Extract custom data (everything not in known fields)
        known_fields = {
            "partition", "processing", "layout", "concat_source", "fold_id",
            "include_augmented", "y", "step_number", "mode", "keyword", "step_id",
            "augment_sample", "add_feature", "replace_processing",
            "target_samples", "target_features"
        }
        custom = {k: v for k, v in data.items() if k not in known_fields}

        return cls(selector=selector, state=state, metadata=metadata, custom=custom)

    def __getitem__(self, key: str) -> Any:
        """
        Dict-like access for backward compatibility.

        Args:
            key: Key to access

        Returns:
            Value for key

        Raises:
            KeyError: If key not found
        """
        return self.to_dict()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Dict-like assignment (updates custom data).

        Args:
            key: Key to set
            value: Value to set
        """
        # Try to update known fields
        if key == "partition":
            self.selector = self.selector.with_partition(value)
        elif key == "processing":
            self.selector = self.selector.with_processing(value)
        elif key == "layout":
            self.selector = self.selector.with_layout(value)
        elif key == "fold_id":
            self.selector = self.selector.with_fold(value)
        elif key == "include_augmented":
            self.selector = self.selector.with_augmented(value)
        elif key == "y":
            self.state.y_processing = value
        elif key == "step_number":
            self.state.step_number = value
        elif key == "mode":
            self.state.mode = value
        elif key == "keyword":
            self.metadata.keyword = value
        elif key == "step_id":
            self.metadata.step_id = value
        elif key == "augment_sample":
            self.metadata.augment_sample = value
        elif key == "add_feature":
            self.metadata.add_feature = value
        elif key == "replace_processing":
            self.metadata.replace_processing = value
        elif key == "target_samples":
            self.metadata.target_samples = value
        elif key == "target_features":
            self.metadata.target_features = value
        else:
            # Unknown field goes to custom data
            self.custom[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Dict-like get with default.

        Args:
            key: Key to get
            default: Default value if key not found

        Returns:
            Value for key or default
        """
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        """
        Dict-like membership test.

        Args:
            key: Key to check

        Returns:
            True if key exists
        """
        return key in self.to_dict()

    def keys(self):
        """Get all keys (dict-like interface)."""
        return self.to_dict().keys()

    def values(self):
        """Get all values (dict-like interface)."""
        return self.to_dict().values()

    def items(self):
        """Get all items (dict-like interface)."""
        return self.to_dict().items()

    def pop(self, key: str, default: Any = None) -> Any:
        """
        Pop a key from the context (dict-like interface).

        For known fields, this resets them to default values.
        For custom fields, removes them from custom dict.

        Args:
            key: Key to pop
            default: Default value if key not found

        Returns:
            Value that was popped
        """
        current_value = self.get(key, default)

        # Remove from custom data if it's there
        if key in self.custom:
            del self.custom[key]
        # For known fields, reset to None/default
        elif key in ["partition", "processing", "layout", "fold_id", "include_augmented",
                     "y", "step_number", "mode", "keyword", "step_id",
                     "augment_sample", "add_feature", "replace_processing",
                     "target_samples", "target_features"]:
            # Reset known fields by setting to None/default
            if key == "partition":
                self.selector = self.selector.with_partition(None)
            elif key == "processing":
                self.selector = self.selector.with_processing(None)
            elif key == "augment_sample":
                self.metadata.augment_sample = False
            elif key == "add_feature":
                self.metadata.add_feature = False
            elif key == "replace_processing":
                self.metadata.replace_processing = False
            elif key == "target_samples":
                self.metadata.target_samples = []
            elif key == "target_features":
                self.metadata.target_features = []
            # Other fields are less commonly popped

        return current_value

    def __len__(self) -> int:
        """Return number of items in context (dict-like interface)."""
        return len(self.to_dict())
