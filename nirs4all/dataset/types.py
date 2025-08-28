from typing import List, Dict, Any, Optional, Literal
import numpy as np

IndexDict = Dict[str, Any]
Selector = Optional[IndexDict]
SourceSelector = Optional[int | List[int]]
OutputData = np.ndarray | List[np.ndarray]
InputData = np.ndarray | List[np.ndarray]
Layout = Literal["2d", "3d", "2d_t", "3d_i"]

# Indexer-specific types
SampleIndices = int | List[int] | np.ndarray
PartitionType = Literal["train", "test", "val", "validation"]
ProcessingList = List[str]
SampleConfig = Dict[str, Any]
