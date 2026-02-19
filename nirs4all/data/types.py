from collections.abc import Sequence

# Import DataSelector for type annotations
from typing import TYPE_CHECKING, Any, Literal, Union, overload

import numpy as np

if TYPE_CHECKING:
    from nirs4all.pipeline.config.context import DataSelector, ExecutionContext

IndexDict = dict[str, Any]
# Selector can be either dict (legacy) or DataSelector (new typed) or ExecutionContext
Selector = Union[IndexDict, 'DataSelector', 'ExecutionContext', None]  # noqa: UP007
SourceSelector = int | list[int] | None
OutputData = np.ndarray | list[np.ndarray]
InputData = np.ndarray | list[np.ndarray]
InputFeatures = list[np.ndarray] | list[list[np.ndarray]]
Layout = Literal["2d", "3d", "2d_t", "3d_i"]
InputTarget = np.ndarray | Sequence

# Indexer-specific types
SampleIndices = list[int] | np.ndarray
PartitionType = Literal["train", "test", "val", "validation"]
ProcessingList = list[str]
SampleConfig = dict[str, Any]

def get_num_samples(data: InputData | OutputData) -> int:
    if isinstance(data, np.ndarray):
        return data.shape[0]
    if isinstance(data, list) and data and isinstance(data[0], np.ndarray):
        return data[0].shape[0]
    raise TypeError("Expected ndarray or list of ndarray")
