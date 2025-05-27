import numpy as np
from typing import Optional, Union, List

class SpectraFeatures:
    """Efficient multi-source spectral features with lazy operations."""

    def __init__(self, sources: Union[np.ndarray, List[np.ndarray]]):
        if isinstance(sources, np.ndarray):
            sources = [sources]
        self.sources = sources
        self._validate_sources()

        # Precompute ranges for efficient slicing
        self.ranges = []
        start = 0
        for source in sources:
            end = start + source.shape[1]
            self.ranges.append((start, end))
            start = end

        self.total_width = start

    def _validate_sources(self):
        """Ensure all sources have the same number of samples."""
        if not self.sources:
            return
        n_samples = len(self.sources[0])
        for i, source in enumerate(self.sources[1:], 1):
            if len(source) != n_samples:
                raise ValueError(f"Source {i} has {len(source)} samples, expected {n_samples}")

    def __len__(self) -> int:
        return len(self.sources[0]) if self.sources else 0

    def get_by_rows(self, row_indices: np.ndarray,
                    source_indices: Optional[Union[int, List[int]]] = None,
                    concatenate: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        """Efficient row-based selection with optional source filtering."""
        if not self.sources:
            return np.array([])

        if source_indices is None:
            selected_sources = self.sources
        elif isinstance(source_indices, int):
            return self.sources[source_indices][row_indices]
        else:
            selected_sources = [self.sources[i] for i in source_indices]

        if len(selected_sources) == 1:
            return selected_sources[0][row_indices]

        if concatenate:
            return np.concatenate([src[row_indices] for src in selected_sources], axis=1)
        else:
            return [src[row_indices] for src in selected_sources]

    def update_rows(self, row_indices: np.ndarray, new_data: Union[np.ndarray, List[np.ndarray]],
                    source_indices: Optional[Union[int, List[int]]] = None):
        """Update specific rows with new data."""
        if isinstance(new_data, np.ndarray):
            if source_indices is None:
                if len(self.sources) == 1:
                    self.sources[0][row_indices] = new_data
                else:
                    # Split concatenated data back to sources
                    start = 0
                    for i, source in enumerate(self.sources):
                        end = start + source.shape[1]
                        self.sources[i][row_indices] = new_data[:, start:end]
                        start = end
            else:
                self.sources[source_indices][row_indices] = new_data
        else:
            if source_indices is None:
                source_indices = list(range(len(new_data)))
            for i, data in enumerate(new_data):
                self.sources[source_indices[i]][row_indices] = data

    def append(self, new_sources: Union[np.ndarray, List[np.ndarray]]):
        """Efficiently append new samples."""
        if isinstance(new_sources, np.ndarray):
            new_sources = [new_sources]

        if not self.sources:
            self.sources = new_sources
            return

        for i, new_source in enumerate(new_sources):
            if i < len(self.sources):
                self.sources[i] = np.vstack([self.sources[i], new_source])
            else:
                self.sources.append(new_source)
