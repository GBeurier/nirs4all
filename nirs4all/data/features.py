from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import polars as pl

from nirs4all.data._features import FeatureSource
from nirs4all.data.types import InputData, InputFeatures, ProcessingList, SampleIndices

class Features:
    """Manages N aligned NumPy sources + a Polars index.

    This class coordinates multiple FeatureSource objects, ensuring they remain
    aligned in terms of sample count while allowing different feature dimensions
    and processing pipelines per source.

    Attributes:
        sources: List of FeatureSource objects managing individual feature arrays.
        cache: Whether to enable caching for operations.
    """

    def __init__(self, cache: bool = False):
        """Initialize empty feature block.

        Args:
            cache: If True, enables caching for operations (not yet implemented).
        """
        self.sources: List[FeatureSource] = []
        self.cache = cache

    def add_samples(self, data: InputData, headers: Optional[Union[List[str], List[List[str]]]] = None,
                    header_unit: Optional[Union[str, List[str]]] = None) -> None:
        """Add samples to all sources, ensuring alignment.

        Args:
            data: Single 2D array or list of 2D arrays, one per source.
            headers: Optional feature headers. Single list applies to all sources,
                or list of lists for per-source headers.
            header_unit: Optional unit type for headers ("cm-1", "nm", "none", "text", "index").
                Single string applies to all sources, or list for per-source units.

        Raises:
            ValueError: If number of data arrays doesn't match existing sources,
                or if headers/units lists don't match number of sources.
        """
        if isinstance(data, np.ndarray):
            data = [data]

        n_sources = len(data)
        if not self.sources:
            self.sources = [FeatureSource() for _ in range(n_sources)]
        elif len(self.sources) != n_sources:
            raise ValueError(f"Expected {len(self.sources)} sources, got {n_sources}")

        # Prepare headers list
        if headers is not None:
            if isinstance(headers[0], str):
                headers_list = [headers] * n_sources
            else:
                headers_list = headers
            if len(headers_list) != n_sources:
                raise ValueError(f"Expected {n_sources} headers lists, got {len(headers_list)}")
        else:
            headers_list = [None] * n_sources

        # Prepare header_unit list
        if header_unit is not None:
            if isinstance(header_unit, str):
                units_list = [header_unit] * n_sources
            else:
                units_list = header_unit
            if len(units_list) != n_sources:
                raise ValueError(f"Expected {n_sources} header units, got {len(units_list)}")
        else:
            units_list = [None] * n_sources

        # Add samples and set headers with units
        for src, arr, hdr, unit in zip(self.sources, data, headers_list, units_list):
            src.add_samples(arr, hdr)
            if hdr is not None and unit is not None:
                src.set_headers(hdr, unit=unit)

    def add_samples_batch_3d(self, data: Union[np.ndarray, List[np.ndarray]]) -> None:
        """Add multiple samples with 3D data in a single operation - O(N) instead of O(N²).

        This method is optimized for bulk insertion of augmented samples where
        each sample may have multiple processings. Much faster than calling
        add_samples() in a loop.

        Args:
            data: Single 3D array of shape (n_samples, n_processings, n_features)
                  or list of 3D arrays for multi-source datasets.

        Raises:
            ValueError: If number of data arrays doesn't match existing sources,
                or if data dimensions don't match.
        """
        if isinstance(data, np.ndarray):
            data = [data]

        n_sources = len(data)
        if not self.sources:
            raise ValueError("Cannot add samples to empty feature block - add initial samples first")
        if len(self.sources) != n_sources:
            raise ValueError(f"Expected {len(self.sources)} sources, got {n_sources}")

        # Add samples to each source using batch method
        for src, arr in zip(self.sources, data):
            src.add_samples_batch_3d(arr)

    def update_features(self, source_processings: ProcessingList, features: InputFeatures, processings: ProcessingList, source: int = -1) -> None:
        """Update or add new feature processings to a specific source.

        Args:
            source_processings: List of existing processing names to replace. Empty string "" means add new.
            features: Feature arrays to add or replace (single array or list of arrays).
            processings: Target processing names for the features.
            source: Source index to update (default: 0 if negative).
        """
        # Handle empty features list
        if not features:
            return
        self.sources[source if source >= 0 else 0].update_features(source_processings, features, processings)

    @property
    def num_samples(self) -> int:
        """Get the number of samples (rows) across all sources.

        Returns:
            Number of samples in the first source (all sources have the same count).
        """
        if not self.sources:
            return 0
        return self.sources[0].num_samples

    @property
    def num_processings(self) -> Union[List[int], int]:
        """Get the number of unique processing IDs per source.

        Returns:
            Single int if only one source, otherwise list of ints (one per source).
        """
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
        """Get the list of processing IDs per source.

        Returns:
            List of processing ID lists, one per source.
        """
        if not self.sources:
            return []
        res = []
        for src in self.sources:
            res.append(src.processing_ids)
        return res

    @property
    def headers_list(self) -> Union[List[List[str]], List[str]]:
        """Get the list of feature headers per source.

        Returns:
            List of header lists, one per source.
        """
        if not self.sources:
            return []
        res = []
        for src in self.sources:
            res.append(src.headers)
        return res

    def headers(self, src: int) -> List[str]:
        """Get the list of feature headers for a specific source.

        Args:
            src: Source index.

        Returns:
            List of header strings for the specified source.
        """
        if not self.sources:
            return []
        return self.sources[src].headers

    @property
    def num_features(self) -> Union[List[int], int]:
        """Get the number of features per source.

        Returns:
            Single int if only one source, otherwise list of ints (one per source).
        """
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

    def keep_sources(self, source_indices: Union[int, List[int]]) -> None:
        """Keep only specified sources, removing all others.

        Used after merge operations with output_as="features" to consolidate
        to a single source.

        Args:
            source_indices: Single source index or list of source indices to keep.

        Raises:
            ValueError: If no sources exist or source indices are invalid.
        """
        if not self.sources:
            raise ValueError("No sources available to filter")

        # Normalize to list
        if isinstance(source_indices, int):
            source_indices = [source_indices]

        # Validate indices
        n_sources = len(self.sources)
        for idx in source_indices:
            if idx < 0 or idx >= n_sources:
                raise ValueError(f"Invalid source index {idx}, have {n_sources} sources")

        # Keep only specified sources
        self.sources = [self.sources[i] for i in source_indices]

    def x(self, indices: SampleIndices, layout: str = "2d", concat_source: bool = True) -> Union[np.ndarray, list[np.ndarray]]:
        """Retrieve feature data for specified samples.

        Args:
            indices: Sample indices to retrieve.
            layout: Data layout format ("2d", "2d_interleaved", "3d", "3d_transpose").
            concat_source: If True and multiple sources exist, concatenate along feature dimension.

        Returns:
            Feature array(s) in the requested layout. Single array if concat_source=True or
            only one source, otherwise list of arrays.

        Raises:
            ValueError: If no features are available.
        """
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
