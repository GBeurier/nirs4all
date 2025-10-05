from typing import Dict, Any, Optional, Literal, Sequence, overload, Union
import numpy as np

IndexDict = Dict[str, Any]
Selector = Optional[IndexDict]
SourceSelector = Optional[Union[int, list[int]]]
OutputData = Union[np.ndarray, list[np.ndarray]]
InputData = Union[np.ndarray, list[np.ndarray]]
InputFeatures = Union[list[np.ndarray], list[list[np.ndarray]]]
Layout = Literal["2d", "3d", "2d_t", "3d_i"]
InputTarget = Union[np.ndarray, Sequence]

# Indexer-specific types
SampleIndices = Union[list[int], np.ndarray]
PartitionType = Literal["train", "test", "val", "validation"]
ProcessingList = list[str]
SampleConfig = Dict[str, Any]

def get_num_samples(data: Union[InputData, OutputData]) -> int:
    if isinstance(data, np.ndarray):
        return data.shape[0]
    if isinstance(data, list) and data and isinstance(data[0], np.ndarray):
        return data[0].shape[0]
    raise TypeError("Expected ndarray or list of ndarray")