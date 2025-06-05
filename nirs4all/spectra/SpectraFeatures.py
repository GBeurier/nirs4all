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
        self.merge_index = -1

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
                    concatenate: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
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

    def add_source(self, name: str, features: np.ndarray, row_indices: Optional[np.ndarray] = None):
        """Add a new feature source."""
        if row_indices is None:
            # Add as new samples
            if self.sources:
                if features.shape[0] != len(self.sources[0]):
                    raise ValueError(f"New source must have {len(self.sources[0])} samples")
            self.sources.append(features)
            self._update_ranges()
        else:
            # Create a new source with the same number of samples as existing sources
            # and fill in only the specified rows
            if not self.sources:
                raise ValueError("Cannot add partial source when no sources exist")

            n_samples = self.sources[0].shape[0]
            n_features = features.shape[1]

            # Create new source filled with zeros
            new_source = np.zeros((n_samples, n_features))
            # Fill in the specified rows
            new_source[row_indices] = features

            self.sources.append(new_source)
            self._update_ranges()

    def get_source(self, source_idx: int, row_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Get features from a specific source."""
        if source_idx >= len(self.sources):
            raise ValueError(f"Source {source_idx} does not exist")

        source = self.sources[source_idx]
        if row_indices is not None:
            return source[row_indices]
        return source

    def update_source(self, source_idx: int, row_indices: np.ndarray, new_features: np.ndarray):
        """Update features for a specific source."""
        if source_idx >= len(self.sources):
            raise ValueError(f"Source {source_idx} does not exist")

        self.sources[source_idx][row_indices] = new_features
        self._update_ranges()

    def _update_ranges(self):
        """Update feature ranges after source modifications."""
        self.ranges = []
        start = 0
        for source in self.sources:
            end = start + source.shape[1]
            self.ranges.append((start, end))
            start = end
        self.total_width = start

    @property
    def shape(self) -> tuple:
        """Get the shape of the concatenated features."""
        if not self.sources:
            return (0, 0)

        return (len(self.sources[0]), self.sources[0].shape[0], self.total_width)

    def replace_with_merged(self, merged_features: np.ndarray):
        """Replace all sources with a single merged source."""
        if merged_features.ndim != 2:
            raise ValueError("Merged features must be 2D array")

        # Verify the number of samples matches
        if self.sources and len(merged_features) != len(self.sources[0]):
            raise ValueError(f"Merged features has {len(merged_features)} samples, expected {len(self.sources[0])}")

        # Replace all sources with the merged result
        self.sources = [merged_features]
        self._update_ranges()