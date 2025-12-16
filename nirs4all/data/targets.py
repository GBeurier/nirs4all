"""Target data management with processing chains."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.base import TransformerMixin

from nirs4all.data.types import SampleIndices
from nirs4all.data._targets.converters import NumericConverter
from nirs4all.data._targets.processing_chain import ProcessingChain
from nirs4all.data._targets.transformers import TargetTransformer
from nirs4all.core.task_type import TaskType
from nirs4all.core.task_detection import detect_task_type

# Re-export for backward compatibility
from nirs4all.data._targets.encoders import FlexibleLabelEncoder  # noqa: F401


class Targets:
    """
    Target manager that stores target arrays with processing chains.

    Manages multiple versions of target data (raw, numeric, scaled, etc.) with
    processing ancestry tracking and transformation capabilities. Delegates
    specialized operations to helper components for better maintainability.

    Attributes:
        num_samples (int): Number of samples in target data
        num_targets (int): Number of target variables
        num_classes (int): Number of unique classes (for classification tasks)
        num_processings (int): Number of processing versions
        processing_ids (list of str): Names of available processings

    Examples:
        >>> targets = Targets()
        >>> targets.add_targets(np.array([1, 2, 3, 1, 2]))
        >>> targets.num_samples
        5
        >>> targets.num_classes
        3

        >>> # Add scaled version
        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = StandardScaler()
        >>> scaled_data = scaler.fit_transform(targets.get_targets('numeric'))
        >>> targets.add_processed_targets('scaled', scaled_data, 'numeric', scaler)

        >>> # Transform predictions back to numeric space
        >>> predictions = model.predict(X_test)
        >>> numeric_preds = targets.transform_predictions(
        ...     predictions, 'scaled', 'numeric'
        ... )

    See Also:
        ProcessingChain: Manages processing ancestry
        NumericConverter: Converts raw data to numeric
        TargetTransformer: Transforms predictions between states
    """

    def __init__(self):
        """Initialize empty target manager."""
        # Core data storage
        self._data: Dict[str, np.ndarray] = {}

        # Delegate to specialized components
        self._processing_chain = ProcessingChain()
        self._converter = NumericConverter()
        self._transformer = TargetTransformer(self._processing_chain)

        # Performance caching
        self._stats_cache: Dict[str, Any] = {}

        # Task type detection
        self._task_type: Optional[TaskType] = None
        self._task_type_forced: bool = False  # If True, task type was explicitly set and should not be re-detected
        self._task_type_by_processing: Dict[str, TaskType] = {}  # Track task_type per processing

    def __repr__(self) -> str:
        """
        Return unambiguous string representation.

        Returns:
            str: String showing samples, targets, and processings
        """
        return (
            f"Targets(samples={self.num_samples}, "
            f"targets={self.num_targets}, "
            f"processings={self._processing_chain.processing_ids})"
        )

    def __str__(self) -> str:
        """
        Return readable string representation with statistics.

        Returns:
            str: Multi-line string with processing statistics

        Notes:
        - Skips 'raw' processing in display
        - Shows min/max/mean for numeric processings
        - Computed statistics are not cached
        """
        if self.num_samples == 0:
            return "Targets:\n(empty)"

        # Show statistics for each processing (excluding "raw")
        processing_stats = []
        for proc_name in self._processing_chain.processing_ids:
            if proc_name == "raw":
                continue  # Skip raw processing in display

            data = self._data[proc_name]
            if np.issubdtype(data.dtype, np.number) and data.size > 0:
                try:
                    min_val = round(float(np.min(data)), 3)
                    max_val = round(float(np.max(data)), 3)
                    mean_val = round(float(np.mean(data)), 3)
                    processing_stats.append((proc_name, min_val, max_val, mean_val))
                except (TypeError, ValueError):
                    # Skip non-numeric data
                    processing_stats.append((proc_name, "N/A", "N/A", "N/A"))
            else:
                processing_stats.append((proc_name, "N/A", "N/A", "N/A"))

        # Format output
        visible_processings = [p for p in self._processing_chain.processing_ids if p != "raw"]
        result = f"Targets: (samples={self.num_samples}, targets={self.num_targets}, processings={visible_processings})"

        for proc_name, min_val, max_val, mean_val in processing_stats:
            result += f"\n- {proc_name}: min={min_val}, max={max_val}, mean={mean_val}"

        return result

    @property
    def num_samples(self) -> int:
        """
        Get the number of samples.

        Returns:
            int: Number of samples (0 if no data)
        """
        if not self._data:
            return 0
        # Use first available processing to get sample count
        first_data = next(iter(self._data.values()))
        return first_data.shape[0]

    @property
    def num_targets(self) -> int:
        """
        Get the number of target variables.

        Returns:
            int: Number of targets (0 if no data)
        """
        if not self._data:
            return 0
        # Use first available processing to get target count
        first_data = next(iter(self._data.values()))
        return first_data.shape[1]

    @property
    def num_processings(self) -> int:
        """
        Get the number of unique processings.

        Returns:
            int: Number of processing versions
        """
        return self._processing_chain.num_processings

    @property
    def processing_ids(self) -> List[str]:
        """
        Get the list of processing IDs.

        Returns:
            list of str: Copy of processing names
        """
        return self._processing_chain.processing_ids

    @property
    def num_classes(self) -> int:
        """
        Get the number of unique classes from numeric targets.

        Returns:
            int: Number of unique classes

        Raises:
            ValueError: If no target data available
            ValueError: If numeric targets not available

        Notes:
        - Uses numeric targets (not raw)
        - For multi-target, uses first column
        - Result is cached until data changes
        - NaN values are excluded from count
        """
        # Check cache first
        if 'num_classes' in self._stats_cache:
            return self._stats_cache['num_classes']

        if self.num_samples == 0:
            raise ValueError("Cannot compute num_classes: no target data available")

        # Get numeric targets (all samples)
        y_numeric = self._data.get("numeric")
        if y_numeric is None:
            raise ValueError("Cannot compute num_classes: numeric targets not available")

        # For multi-target, use first column (typical for classification)
        if y_numeric.ndim > 1:
            y_numeric = y_numeric[:, 0]

        # Count unique classes
        unique_classes = np.unique(y_numeric[~np.isnan(y_numeric)])
        num_classes = len(unique_classes)

        # Cache result
        self._stats_cache['num_classes'] = num_classes
        return num_classes

    @property
    def task_type(self) -> Optional[TaskType]:
        """
        Get the detected task type.

        Returns:
            TaskType enum or None if no targets added
        """
        return self._task_type

    @property
    def task_type_forced(self) -> bool:
        """Check if task type was explicitly forced (disabling auto-detection)."""
        return self._task_type_forced

    def set_task_type(self, task_type: TaskType, forced: bool = True) -> None:
        """
        Set the task type explicitly.

        Args:
            task_type: TaskType enum value
            forced: If True, prevents auto-detection from overriding this value
                   in subsequent processing (e.g., after MinMaxScaler). Default True.
        """
        self._task_type = task_type
        self._task_type_forced = forced

    def get_task_type_for_processing(self, processing: str) -> Optional[TaskType]:
        """
        Get the task type for a specific processing.

        This method allows retrieving the task type that was detected when a specific
        processing was added. Useful for understanding how different transformations
        (e.g., discretization, binning) affect the task type.

        Args:
            processing (str): Processing name to query

        Returns:
            Optional[TaskType]: Task type for the processing, or None if not available

        Examples:
            >>> targets.add_targets([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> targets.get_task_type_for_processing('numeric')
            TaskType.REGRESSION

            >>> # After discretization
            >>> targets.add_processed_targets('binned', [0, 0, 1, 1, 2], 'numeric')
            >>> targets.get_task_type_for_processing('binned')
            TaskType.MULTICLASS_CLASSIFICATION
        """
        return self._task_type_by_processing.get(processing)

    def add_targets(self, targets: Union[np.ndarray, List, tuple]) -> None:
        """
        Add target samples. Can be called multiple times to append.

        Automatically creates 'raw' and 'numeric' processings on first call.
        Subsequent calls append to existing data.

        Args:
            targets (array-like): Target data as 1D (single target) or 2D (multiple targets)

        Raises:
            ValueError: If processings beyond 'raw' and 'numeric' exist
            ValueError: If target dimensions don't match existing data

        Notes:
        - First call: creates 'raw' and 'numeric' processings
        - Subsequent calls: appends to existing arrays
        - Invalidates statistics cache

        Examples:
        >>> targets = Targets()
        >>> targets.add_targets([1, 2, 3])
        >>> targets.num_samples
        3
        >>> targets.add_targets([4, 5])
        >>> targets.num_samples
        5
        """
        if self.num_processings > 2:  # Allow if only "raw" and "numeric" exist
            raise ValueError("Cannot add new samples after additional processings have been created.")

        targets = np.asarray(targets)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        elif targets.ndim != 2:
            raise ValueError(f"Targets must be 1D or 2D array, got {targets.ndim}D")

        # First time: initialize structure
        if self.num_processings == 0:
            # Add "raw" processing (preserves original data types)
            self._data["raw"] = targets.copy()
            self._processing_chain.add_processing("raw", ancestor=None, transformer=None)

            # Automatically create "numeric" processing (converts to numeric format)
            numeric_data, transformer = self._converter.convert(targets)
            self._data["numeric"] = numeric_data
            self._processing_chain.add_processing("numeric", ancestor="raw", transformer=transformer)

            # Detect task type when targets are first added (use numeric data for detection)
            if numeric_data.size > 0:
                self._task_type = detect_task_type(numeric_data)
                self._task_type_by_processing['numeric'] = self._task_type
                # Also store for 'raw' if it exists
                if 'raw' in self._data:
                    self._task_type_by_processing['raw'] = self._task_type
        else:
            # Subsequent times: append to existing data
            if targets.shape[1] != self.num_targets:
                raise ValueError(f"Target data has {targets.shape[1]} targets, expected {self.num_targets}")

            # Append to raw data
            self._data["raw"] = np.vstack([self._data["raw"], targets])

            # Update numeric data using existing transformer
            numeric_data, _ = self._converter.convert(
                targets,
                self._processing_chain.get_transformer("numeric")
            )
            self._data["numeric"] = np.vstack([self._data["numeric"], numeric_data])

        # Invalidate cache
        self._stats_cache.clear()

    def add_processed_targets(self,
                              processing_name: str,
                              targets: Union[np.ndarray, List, tuple],
                              ancestor: str = "numeric",
                              transformer: Optional[TransformerMixin] = None,
                              mode: str = "train",
                              labelizer: bool = True) -> None:
        """
        Add processed version of target data.

        Args:
            processing_name (str): Unique name for this processing
            targets (array-like): Processed target data (same number of samples)
            ancestor (str, optional): Source processing name. Defaults to 'numeric'.
            transformer (TransformerMixin, optional): Transformer used to create this processing
            mode (str, optional): Mode for validation ('train' enforces shape checks). Defaults to 'train'.
            labelizer (bool, optional): Legacy parameter (currently unused). Defaults to True.

        Raises:
            ValueError: If processing_name already exists
            ValueError: If ancestor doesn't exist
            ValueError: If shape doesn't match existing data (in train mode)

        Examples:
        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = StandardScaler()
        >>> scaled = scaler.fit_transform(targets.get_targets('numeric'))
        >>> targets.add_processed_targets('scaled', scaled, 'numeric', scaler)
        """
        if self._processing_chain.has_processing(processing_name):
            raise ValueError(f"Processing '{processing_name}' already exists")

        if not self._processing_chain.has_processing(ancestor):
            raise ValueError(f"Ancestor processing '{ancestor}' does not exist")

        targets = np.asarray(targets)
        if mode == "train":
            if targets.ndim == 1:
                targets = targets.reshape(-1, 1)
            elif targets.ndim != 2:
                raise ValueError(f"Targets must be 1D or 2D array, got {targets.ndim}D")

            if targets.shape[0] != self.num_samples:
                raise ValueError(f"Target data has {targets.shape[0]} samples, expected {self.num_samples}")

            if targets.shape[1] != self.num_targets:
                raise ValueError(f"Target data has {targets.shape[1]} targets, expected {self.num_targets}")

        self._data[processing_name] = targets.copy()
        self._processing_chain.add_processing(processing_name, ancestor, transformer)
        self._stats_cache.clear()

        # Re-detect task type after adding processed targets (e.g., discretization may change regression to classification)
        # But only if task type was not explicitly forced
        if targets.size > 0:
            new_task_type = detect_task_type(targets)
            self._task_type_by_processing[processing_name] = new_task_type

            # Only update global task_type if not forced
            if not self._task_type_forced and self._task_type != new_task_type:
                print(f"⚠️  Task type changed: {self._task_type.value if self._task_type else 'None'} → {new_task_type.value} "
                      f"(processing '{processing_name}')")
                self._task_type = new_task_type

    def get_targets(self,
                    processing: str = "numeric",
                    indices: Optional[Union[List[int], np.ndarray]] = None) -> np.ndarray:
        """
        Get target data for a specific processing.

        Args:
            processing (str, optional): Processing name to retrieve. Defaults to 'numeric'.
            indices (array-like of int, optional): Sample indices to retrieve (None for all)

        Returns:
            np.ndarray: Target array of shape (n_samples, n_targets) or
            (selected_samples, n_targets)

        Raises:
            ValueError: If processing doesn't exist

        Examples:
        >>> targets.get_targets('numeric')
        array([[1.], [2.], [3.]])

        >>> targets.get_targets('numeric', indices=[0, 2])
        array([[1.], [3.]])
        """
        if not self._processing_chain.has_processing(processing):
            available = self._processing_chain.processing_ids
            raise ValueError(f"Processing '{processing}' not found. Available: {available}")

        data = self._data[processing]

        if indices is None or len(indices) == 0 or data.shape[0] == 0:
            return data.copy()

        indices = np.asarray(indices, dtype=int)
        return data[indices]

    def y(self,
          indices: SampleIndices,
          processing: str) -> np.ndarray:
        """
        Convenience method to get targets with indices.

        Alias for get_targets with different parameter order.

        Args:
            indices (array-like of int): Sample indices to retrieve
            processing (str): Processing name

        Returns:
            np.ndarray: Target array for specified indices

        Examples:
        >>> targets.y([0, 1, 2], 'numeric')
        array([[1.], [2.], [3.]])
        """
        if len(self._data) == 0:
            return np.array([])

        return self.get_targets(processing, indices)

    def get_processing_ancestry(self, processing: str) -> List[str]:
        """
        Get the full ancestry chain for a processing.

        Args:
            processing (str): Processing name

        Returns:
            list of str: Processing names from root to specified processing

        Raises:
            ValueError: If processing doesn't exist

        Examples:
        >>> targets.get_processing_ancestry('scaled')
        ['raw', 'numeric', 'scaled']
        """
        return self._processing_chain.get_ancestry(processing)

    def invert_transform(self,
                         y_pred: np.ndarray,
                         from_processing: str,
                         to_processing: str = "raw") -> np.ndarray:
        """
        Inverse transform predictions from one processing back to another.

        Args:
            y_pred (np.ndarray): Predictions to transform
            from_processing (str): Source processing name
            to_processing (str, optional): Target processing name. Defaults to 'raw'.

        Returns:
            np.ndarray: Inverse transformed predictions

        Notes:
        This method delegates to transform_predictions for the actual transformation.

        See Also:
        transform_predictions: Main transformation method
        """
        return self.transform_predictions(y_pred, from_processing, to_processing)

    def transform_predictions(self,
                              y_pred: np.ndarray,
                              from_processing: str,
                              to_processing: str) -> np.ndarray:
        """
        Transform predictions from one processing state to another.

        Applies appropriate forward or inverse transformations based on
        the ancestry relationship between processings.

        Args:
            y_pred (np.ndarray): Prediction array to transform
            from_processing (str): Current processing state of predictions
            to_processing (str): Target processing state

        Returns:
            np.ndarray: Transformed predictions in target processing state

        Raises:
            ValueError: If either processing doesn't exist
            ValueError: If no transformation path exists
            ValueError: If transformation fails

        Examples:
        >>> # Model trained on scaled targets
        >>> predictions = model.predict(X_test)
        >>> # Transform back to numeric space
        >>> numeric_preds = targets.transform_predictions(
        ...     predictions, 'scaled', 'numeric'
        ... )

        Notes:
        - Empty predictions return empty array
        - Uses cached ancestry for efficiency
        - Handles both forward and inverse transformations

        See Also:
        TargetTransformer: Handles transformation logic
        """
        return self._transformer.transform(
            y_pred, from_processing, to_processing, self._data
        )
