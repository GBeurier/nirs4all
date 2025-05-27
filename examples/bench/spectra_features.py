from typing import Any, Sequence, Mapping, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl


class SpectraFeatures:
    """Class to handle spectral features, allowing for multiple sources."""
    def __init__(self, sources: Union[np.ndarray, List[np.ndarray]]) -> None:
        if isinstance(sources, np.ndarray):
            sources = [sources]
        self.sources = sources
        self.total_width = sum(source.shape[1] for source in sources)
        self.ranges = []
        start = 0
        for source in sources:
            end = start + source.shape[1]
            self.ranges.append((start, end))
            start = end

    def __len__(self) -> int:
        return len(self.sources[0]) if self.sources else 0

    def __repr__(self) -> str:
        text = "num_sources: " + str(self.num_sources) + "\n"
        for i, spectra in enumerate(self.sources):
            text += f"Source {i}: shape={spectra.shape}, dtype={spectra.dtype}\n"
        return text

    @property
    def num_sources(self) -> int:
        """Return the number of sources."""
        return len(self.sources)

    @property
    def max_width(self) -> int:
        """Return the maximum width of the sources."""
        return max(source.shape[1] for source in self.sources) if self.sources else 0

    def concatenate(self, sources: Union[np.ndarray, List[np.ndarray]]) -> None:
        """Concatenate new sources to the existing ones."""
        if isinstance(sources, np.ndarray):
            sources = [sources]

        for i, source in enumerate(sources):
            if i < len(self.sources):
                self.sources[i] = np.concatenate((self.sources[i], source), axis=0)
            else:
                self.sources.append(source)

    def to_numpy(self,
                 indices: Optional[np.ndarray] = None,
                 sources: Optional[Union[int, Sequence[int]]] = None,
                 merge: bool = True,
                 pad: bool = False,
                 pad_value: float = np.nan
                 ) -> Union[np.ndarray, List[np.ndarray]]:
        """Convert sources to a NumPy array or list of arrays based on indices."""
        if indices is None:
            indices = np.arange(len(self))

        if self.num_sources == 0:
            return np.array([])

        from_sources = np.arange(self.num_sources)
        if isinstance(sources, int):
            from_sources = [sources]
        elif isinstance(sources, Sequence):
            from_sources = list(sources)

        if len(from_sources) == 0:
            return np.array([])

        if len(from_sources) == 1:
            return self.sources[from_sources[0]][indices]

        if merge:
            return np.concatenate([self.sources[i][indices] for i in from_sources], axis=1)

        return [self.sources[i][indices] for i in from_sources]
