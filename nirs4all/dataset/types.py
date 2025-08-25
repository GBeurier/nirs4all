from typing import List, Dict, Any, Optional, Literal
import numpy as np

Selector = Optional[str | Dict[str, Any]]
SourceSelector = Optional[int | List[int]]
OutputData = np.ndarray | List[np.ndarray]
InputData = np.ndarray | List[np.ndarray]
Layout = Literal["2d", "3d", "2d_t", "3d_i"]