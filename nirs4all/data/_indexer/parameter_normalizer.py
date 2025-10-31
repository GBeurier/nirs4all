"""
Validate and normalize input parameters for the indexer.

This module provides the ParameterNormalizer class for handling various
input formats and converting them to consistent internal representations.
"""

from typing import Union, List, Any, Optional, Dict
import numpy as np

from nirs4all.data.types import SampleIndices, ProcessingList, PartitionType, IndexDict


class ParameterNormalizer:
    """
    Validate and normalize indexer input parameters.

    This class centralizes all parameter validation and normalization logic,
    ensuring consistent handling of various input formats across the indexer.

    Responsibilities:
    - Normalize sample indices (int, list, array) to lists
    - Normalize single values or lists to fixed-length lists
    - Prepare processing lists for storage
    - Convert IndexDict to method parameters
    - Validate parameter combinations
    """

    def __init__(self, default_processings: List[str]):
        """
        Initialize the parameter normalizer.

        Args:
            default_processings: Default processing list to use when None is provided.
        """
        self._default_processings = default_processings

    def normalize_indices(
        self,
        indices: SampleIndices,
        count: int,
        param_name: str
    ) -> List[int]:
        """
        Normalize various index formats to a list of integers.

        Handles single values, lists, and numpy arrays, ensuring the result
        matches the expected count.

        Args:
            indices: Index specification - can be:
                    - int: Single value, repeated count times
                    - List[int]: Must have length == count
                    - np.ndarray: Must have length == count
            count: Expected length of the result list.
            param_name: Parameter name for error messages.

        Returns:
            List[int]: Normalized list of indices with length == count.

        Raises:
            ValueError: If list/array length doesn't match count.

        Examples:
            >>> normalizer = ParameterNormalizer(["raw"])
            >>>
            >>> # Single value
            >>> normalizer.normalize_indices(5, 3, "sample_indices")
            [5, 5, 5]
            >>>
            >>> # List
            >>> normalizer.normalize_indices([1, 2, 3], 3, "sample_indices")
            [1, 2, 3]
            >>>
            >>> # Numpy array
            >>> normalizer.normalize_indices(np.array([10, 20]), 2, "origin_indices")
            [10, 20]
        """
        if isinstance(indices, (int, np.integer)):
            return [int(indices)] * count
        elif isinstance(indices, np.ndarray):
            result = indices.tolist()
        else:
            result = list(indices)

        if len(result) != count:
            raise ValueError(
                f"{param_name} length ({len(result)}) must match count ({count})"
            )

        return [int(x) for x in result]

    def normalize_single_or_list(
        self,
        value: Union[Any, List[Any]],
        count: int,
        param_name: str,
        allow_none: bool = False
    ) -> List[Any]:
        """
        Normalize single value or list to a list of specified length.

        Args:
            value: Value specification - can be:
                  - Single value: Repeated count times
                  - List: Must have length == count
                  - None: Repeated count times (if allow_none=True)
            count: Expected length of the result list.
            param_name: Parameter name for error messages.
            allow_none: Whether None is allowed as a value.

        Returns:
            List[Any]: Normalized list with length == count.

        Raises:
            ValueError: If list length doesn't match count.

        Examples:
            >>> normalizer = ParameterNormalizer(["raw"])
            >>>
            >>> # Single value
            >>> normalizer.normalize_single_or_list(1, 3, "group")
            [1, 1, 1]
            >>>
            >>> # List
            >>> normalizer.normalize_single_or_list([1, 2, 3], 3, "group")
            [1, 2, 3]
            >>>
            >>> # None (with allow_none=True)
            >>> normalizer.normalize_single_or_list(None, 2, "augmentation", allow_none=True)
            [None, None]
        """
        if value is None and allow_none:
            return [None] * count
        elif isinstance(value, (int, np.integer, str)) or value is None:
            return [value] * count
        else:
            result = list(value)
            if len(result) != count:
                raise ValueError(
                    f"{param_name} length ({len(result)}) must match count ({count})"
                )
            return result

    def prepare_processings(
        self,
        processings: Union[ProcessingList, List[ProcessingList], None],
        count: int
    ) -> List[List[str]]:
        """
        Prepare processing lists for storage (native Polars List format).

        Handles various processing specifications and normalizes to a list of
        processing lists (one per sample).

        Args:
            processings: Processing specification:
                        - None: Use default for all samples
                        - List[str]: Single list for all samples (e.g., ["raw", "msc"])
                        - List[List[str]]: One list per sample, must match count
            count: Number of samples (expected length).

        Returns:
            List[List[str]]: List of processing lists, one per sample.

        Raises:
            ValueError: If list of lists doesn't match count.

        Examples:
            >>> normalizer = ParameterNormalizer(["raw"])
            >>>
            >>> # None → default
            >>> normalizer.prepare_processings(None, 2)
            [['raw'], ['raw']]
            >>>
            >>> # Single list → replicate
            >>> normalizer.prepare_processings(["raw", "msc"], 2)
            [['raw', 'msc'], ['raw', 'msc']]
            >>>
            >>> # List of lists → validate count
            >>> normalizer.prepare_processings([["raw"], ["raw", "msc"]], 2)
            [['raw'], ['raw', 'msc']]
        """
        if processings is None:
            # Use default for all samples
            return [self._default_processings] * count

        if not isinstance(processings, list):
            raise TypeError(f"processings must be list or None, got {type(processings)}")

        if len(processings) == 0:
            # Empty list → use default
            return [self._default_processings] * count

        # Check if it's a list of strings or list of lists
        if isinstance(processings[0], str):
            # Single list for all samples: ["raw", "msc"]
            return [processings] * count
        elif isinstance(processings[0], list):
            # List of lists: [["raw"], ["raw", "msc"]]
            if len(processings) != count:
                raise ValueError(
                    f"processings length ({len(processings)}) must match count ({count})"
                )
            return processings
        else:
            raise TypeError(
                f"processings must contain strings or lists, got {type(processings[0])}"
            )

    def convert_indexdict_to_params(
        self,
        index_dict: IndexDict,
        count: int
    ) -> Dict[str, Any]:
        """
        Convert IndexDict to method parameters.

        Maps dictionary keys to method parameter names, handling special cases
        like "sample" → "sample_indices" and "origin" → "origin_indices".

        Args:
            index_dict: Dictionary of column specifications.
            count: Number of samples (for validation).

        Returns:
            Dict[str, Any]: Method parameters ready for use.

        Examples:
            >>> normalizer = ParameterNormalizer(["raw"])
            >>>
            >>> index_dict = {
            ...     "sample": [10, 11],
            ...     "partition": "train",
            ...     "group": 1
            ... }
            >>> params = normalizer.convert_indexdict_to_params(index_dict, 2)
            >>> # params: {
            >>> #     "sample_indices": [10, 11],
            >>> #     "partition": "train",
            >>> #     "group": 1
            >>> # }
        """
        params = {}

        # Handle special mappings
        if "sample" in index_dict:
            params["sample_indices"] = index_dict["sample"]
        if "origin" in index_dict:
            params["origin_indices"] = index_dict["origin"]

        # Handle direct mappings
        direct_mappings = ["partition", "group", "branch", "processings", "augmentation"]
        for key in direct_mappings:
            if key in index_dict:
                params[key] = index_dict[key]

        # Handle any other columns as overrides
        for key, value in index_dict.items():
            if key not in ["sample", "origin"] + direct_mappings:
                params[key] = value

        return params

    def validate_count(self, count: int, param_name: str = "count") -> None:
        """
        Validate that count is positive.

        Args:
            count: Count value to validate.
            param_name: Parameter name for error messages.

        Raises:
            ValueError: If count <= 0.
        """
        if count <= 0:
            raise ValueError(f"{param_name} must be positive, got {count}")

    def validate_partition(self, partition: PartitionType) -> None:
        """
        Validate partition value.

        Args:
            partition: Partition value to validate.

        Raises:
            ValueError: If partition is not a valid value.

        Note:
            Currently accepts any string for flexibility, but could be
            enhanced to enforce specific values.
        """
        valid_partitions = {"train", "test", "val", "validation"}
        if partition not in valid_partitions:
            # For now, allow any partition but could warn or raise
            pass
