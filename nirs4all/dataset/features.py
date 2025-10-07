from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import polars as pl

from nirs4all.dataset.feature_source import FeatureSource
from nirs4all.dataset.helpers import InputData, InputFeatures, ProcessingList, SampleIndices

class Features:
    """Manages N aligned NumPy sources + a Polars index."""

    def __init__(self, cache: bool = False):
        """Initialize empty feature block."""
        self.sources: List[FeatureSource] = []
        self.cache = cache

    def add_samples(self, data: InputData, headers: Optional[Union[List[str], List[List[str]]]] = None) -> None:
        if isinstance(data, np.ndarray):
            data = [data]

        n_sources = len(data)
        if not self.sources:
            self.sources = [FeatureSource() for _ in range(n_sources)]
        elif len(self.sources) != n_sources:
            raise ValueError(f"Expected {len(self.sources)} sources, got {n_sources}")

        # verify headers
        if headers is not None:
            if isinstance(headers[0], str):
                headers = [headers] * n_sources
            if len(headers) != n_sources:
                raise ValueError(f"Expected {n_sources} headers lists, got {len(headers)}")
        else:
            headers = [None] * n_sources

        for src, arr, hdr in zip(self.sources, data, headers):
            src.add_samples(arr, hdr)

    def update_features(self, source_processings: ProcessingList, features: InputFeatures, processings: ProcessingList, source: int = -1) -> None:
        # Handle empty features list
        if not features:
            return
        self.sources[source if source >= 0 else 0].update_features(source_processings, features, processings)

    @property
    def num_samples(self) -> int:
        """Get the number of samples (rows) across all sources."""
        if not self.sources:
            return 0
        return self.sources[0].num_samples

    @property
    def num_processings(self) -> Union[List[int], int]:
        """Get the number of unique processing IDs per source."""
        if not self.sources:
            return 0
        res = []
        for src in self.sources:
            res.append(src.num_processings)
        if len(res) == 1:
            return res[0]
        return res

    @property
    def preprocessing_str(self) -> Union[List[List[str]], List[str]]:
        """Get the list of processing IDs per source."""
        if not self.sources:
            return []
        res = []
        for src in self.sources:
            res.append(src.processing_ids)
        return res

    @property
    def headers_list(self) -> Union[List[List[str]], List[str]]:
        """Get the list of feature headers per source."""
        if not self.sources:
            return []
        res = []
        for src in self.sources:
            res.append(src.headers)
        return res

    def headers(self, src: int) -> List[str]:
        """Get the list of feature headers for a specific source."""
        if not self.sources:
            return []
        return self.sources[src].headers

    @property
    def num_features(self) -> Union[List[int], int]:
        """Get the number of features per source."""
        if not self.sources:
            return 0
        res = []
        for src in self.sources:
            res.append(src.num_features)
        if len(res) == 1:
            return res[0]
        return res

    def augment_samples(self,
                        sample_indices: List[int],
                        data: InputData,
                        processings: ProcessingList,
                        count: Union[int, List[int]]) -> None:
        """
        Create augmented samples from existing ones.

        Args:
            sample_indices: List of sample indices to augment
            data: Augmented feature data (single array or list of arrays for multi-source)
            processings: Processing names for the augmented data
            count: Number of augmentations per sample (int) or per sample list
        """
        if isinstance(data, np.ndarray):
            data = [data]

        if len(self.sources) != len(data):
            raise ValueError(f"Expected {len(self.sources)} sources, got {len(data)}")

        # Normalize count to list
        if isinstance(count, int):
            count_list = [count] * len(sample_indices)
        else:
            count_list = list(count)
            if len(count_list) != len(sample_indices):
                raise ValueError("count must be an int or a list with the same length as sample_indices")

        # Add augmented data to each source
        for src, arr in zip(self.sources, data):
            src.augment_samples(sample_indices, arr, processings, count_list)

    def x(self, indices: SampleIndices, layout: str = "2d", concat_source: bool = True) -> Union[np.ndarray, list[np.ndarray]]:
        if not self.sources:
            raise ValueError("No features available")

        res = []
        for src in self.sources:
            res.append(src.x(indices, layout))

        if concat_source and len(res) > 1:
            return np.concatenate(res, axis=res[0].ndim - 1)

        if len(res) == 1:
            return res[0]

        return res

    def __repr__(self):
        n_sources = len(self.sources)
        n_samples = self.num_samples
        return f"FeatureBlock(sources={n_sources}, samples={n_samples})"

    def __str__(self):
        n_sources = len(self.sources)
        n_samples = self.num_samples
        summary = f"Features (samples={n_samples}, sources={n_sources}):"
        for i, source in enumerate(self.sources):
            summary += f"\n- Source {i}: {source}"
        if n_sources == 0:
            summary += "\n- No sources available"
        # unique augmentations
        # summary += f"\nUnique augmentations: {self.index.uniques('augmentation')}"
        # summary += f"\nIndex:\n{self.index.df}"
        return summary
