from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer, LabelEncoder

from nirs4all.dataset.helpers import SampleIndices


class Targets:
    """
    Target manager that stores target arrays with processing chains.

    Manages multiple versions of target data (raw, numeric, scaled, etc.) with
    processing ancestry tracking and transformation capabilities.
    """

    def __init__(self):
        """Initialize empty target manager."""
        # Core data storage - all target data is stored here by processing name
        self._data: Dict[str, np.ndarray] = {}  # Maps processing name to target array

        # Processing chain management
        self._processing_ids: List[str] = []  # Processing names in order of creation
        self._processing_id_to_index: Dict[str, int] = {}  # Maps processing name to index in _processing_ids
        self._ancestors: Dict[str, str] = {}  # Maps processing to its source processing
        self._transformers: Dict[str, TransformerMixin] = {}  # Maps processing to its transformer

    def __repr__(self):
        return f"Targets(samples={self.num_samples}, targets={self.num_targets}, processings={self._processing_ids})"

    def __str__(self) -> str:
        if self.num_samples == 0:
            return "Targets:\n(empty)"

        # Show statistics for each processing (excluding "raw")
        processing_stats = []
        for proc_name in self._processing_ids:
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
        visible_processings = [p for p in self._processing_ids if p != "raw"]
        result = f"Targets: (samples={self.num_samples}, targets={self.num_targets}, processings={visible_processings})"

        for proc_name, min_val, max_val, mean_val in processing_stats:
            result += f"\n- {proc_name}: min={min_val}, max={max_val}, mean={mean_val}"

        return result

        # lines = [f"Targets with {self.num_samples} samples and {self.num_targets} targets"]
        # for proc_id in self._processing_ids:
        #     data = self._data[proc_id]
        #     ancestor = self._ancestors.get(proc_id, "none")
        #     transformer = type(self._transformers.get(proc_id, type(None))).__name__
        #     lines.append(f"  • {proc_id}: {data.shape}, ancestor={ancestor}, transformer={transformer}")
        # return "\n".join(lines)

    @property
    def num_samples(self) -> int:
        """Get the number of samples."""
        if not self._data:
            return 0
        # Use first available processing to get sample count
        first_data = next(iter(self._data.values()))
        return first_data.shape[0]

    @property
    def num_targets(self) -> int:
        """Get the number of targets."""
        if not self._data:
            return 0
        # Use first available processing to get target count
        first_data = next(iter(self._data.values()))
        return first_data.shape[1]

    @property
    def num_processings(self) -> int:
        """Get the number of unique processings."""
        return len(self._processing_ids)

    @property
    def processing_ids(self) -> List[str]:
        """Get the list of processing IDs."""
        return self._processing_ids.copy()

    def add_targets(self, targets: Union[np.ndarray, List, tuple]) -> None:
        """
        Add target samples. Can be called multiple times to append new targets.

        Args:
            targets: Target data as 1D array (single target) or 2D array (multiple targets)
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
            self._add_processing("raw", targets, ancestor=None, transformer=None)

            # Automatically create "numeric" processing (converts to numeric format)
            numeric_data, transformer = self._make_numeric(targets)
            self._add_processing("numeric", numeric_data, ancestor="raw", transformer=transformer)
        else:
            # Subsequent times: append to existing data
            if targets.shape[1] != self.num_targets:
                raise ValueError(f"Target data has {targets.shape[1]} targets, expected {self.num_targets}")

            # Append to raw data
            self._data["raw"] = np.vstack([self._data["raw"], targets])

            # Update numeric data using existing transformer
            numeric_data, _ = self._make_numeric(targets)
            self._data["numeric"] = np.vstack([self._data["numeric"], numeric_data])

    def add_processed_targets(self,
                              processing_name: str,
                              targets: Union[np.ndarray, List, tuple],
                              ancestor: str = "numeric",
                              transformer: Optional[TransformerMixin] = None,
                              mode: str = "train",
                              labelizer: bool = True) -> None:
        """
        Add processed target data.

        Args:
            processing_name: Name for this processing
            targets: Processed target data (must have same number of samples as existing data)
            ancestor: Source processing name (default: "numeric")
            transformer: Transformer used to create this processing
        """
        if processing_name in self._processing_id_to_index:
            raise ValueError(f"Processing '{processing_name}' already exists")

        if ancestor not in self._processing_id_to_index:
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

        self._add_processing(processing_name, targets, ancestor, transformer)
        # if labelizer and self._detect_task_type(targets) == "classification":
            # classif_targets, classif_transformer = self._make_numeric(targets)
            # self._add_processing(processing_name, classif_targets, ancestor, classif_transformer)

    def get_targets(self,
                    processing: str = "numeric",
                    indices: Optional[Union[List[int], np.ndarray]] = None) -> np.ndarray:
        """
        Get target data for a specific processing.

        Args:
            processing: Processing name (default: "numeric")
            indices: Sample indices to retrieve (None for all samples)

        Returns:
            Target array of shape (n_samples, n_targets) or (selected_samples, n_targets)
        """
        if processing not in self._processing_id_to_index:
            available = list(self._processing_id_to_index.keys())
            raise ValueError(f"Processing '{processing}' not found. Available: {available}")

        data = self._data[processing]

        if indices is None or len(indices) == 0 or data.shape[0] == 0:
            return data

        indices = np.asarray(indices, dtype=int)
        return data[indices]

    def y(self, indices: SampleIndices, processing: str) -> np.ndarray:
        return self.get_targets(processing, indices)

    def get_processing_ancestry(self, processing: str) -> List[str]:
        """
        Get the full ancestry chain for a processing.

        Args:
            processing: Processing name

        Returns:
            List of processing names from root to the specified processing
        """
        if processing not in self._processing_id_to_index:
            raise ValueError(f"Processing '{processing}' not found")

        ancestry = []
        current = processing

        while current is not None:
            ancestry.append(current)
            current = self._ancestors.get(current)

        return list(reversed(ancestry))

    def invert_transform(self,
                         y_pred: np.ndarray,
                         from_processing: str,
                         to_processing: str = "raw") -> np.ndarray:
        """
        Inverse transform predictions from one processing back to another.

        Args:
            y_pred: Predictions to transform
            from_processing: Source processing name
            to_processing: Target processing name (default: "raw")

        Returns:
            Inverse transformed predictions
        """
        if from_processing == to_processing:
            return y_pred

        # Get ancestry chains
        from_ancestry = self.get_processing_ancestry(from_processing)
        to_ancestry = self.get_processing_ancestry(to_processing)

        # Find common ancestor
        common_ancestor = None
        for ancestor in reversed(from_ancestry):
            if ancestor in to_ancestry:
                common_ancestor = ancestor
                break

        if common_ancestor is None:
            raise ValueError(f"No common ancestor found between '{from_processing}' and '{to_processing}'")

        # Inverse transform from from_processing to common_ancestor
        current = y_pred
        current_proc = from_processing

        while current_proc != common_ancestor:
            ancestor = self._ancestors[current_proc]
            transformer = self._transformers.get(current_proc)

            if transformer is not None and hasattr(transformer, 'inverse_transform'):
                current = transformer.inverse_transform(current)

            current_proc = ancestor

        # Forward transform from common_ancestor to to_processing (if needed)
        if common_ancestor != to_processing:
            # This would require forward transformation capability
            # For now, we'll only support inverse transformation up the ancestry chain
            raise ValueError(f"Forward transformation from '{common_ancestor}' to '{to_processing}' not supported")

        return current

    def transform_predictions(self,
                              y_pred: np.ndarray,
                              from_processing: str,
                              to_processing: str) -> np.ndarray:
        """
        Transform predictions from one processing state to another.

        This is specifically designed for transforming model predictions back through
        the processing chain. For example, transforming predictions from "minmax-detrend-standard"
        back to "numeric" by applying inverse transforms in the correct order.

        Args:
            y_pred: Prediction array to transform
            from_processing: Current processing state of the predictions
            to_processing: Target processing state

        Returns:
            Transformed predictions in the target processing state

        Example:
            # Model trained on "scaled" targets produces predictions
            predictions = model.predict(X_test)
            # Transform predictions back to "numeric" space
            numeric_preds = targets.transform_predictions(predictions, "scaled", "numeric")
        """
        if from_processing == to_processing:
            return y_pred.copy()

        if from_processing not in self._processing_id_to_index:
            available = list(self._processing_id_to_index.keys())
            raise ValueError(f"From processing '{from_processing}' not found. Available: {available}")

        if to_processing not in self._processing_id_to_index:
            available = list(self._processing_id_to_index.keys())
            raise ValueError(f"To processing '{to_processing}' not found. Available: {available}")

        # Get ancestry chains to understand the transformation path
        from_ancestry = self.get_processing_ancestry(from_processing)
        to_ancestry = self.get_processing_ancestry(to_processing)

        # Find common ancestor
        common_ancestor = None
        for ancestor in reversed(from_ancestry):
            if ancestor in to_ancestry:
                common_ancestor = ancestor
                break

        if common_ancestor is None:
            raise ValueError(f"No common ancestor found between '{from_processing}' and '{to_processing}'")

        current_predictions = y_pred.copy()

        if(current_predictions.shape[0] == 0):
            return current_predictions

        # Step 1: Inverse transform from from_processing back to common_ancestor
        current_proc = from_processing
        while current_proc != common_ancestor:
            ancestor = self._ancestors[current_proc]
            transformer = self._transformers.get(current_proc)

            if transformer is not None and hasattr(transformer, 'inverse_transform'):
                try:
                    current_predictions = transformer.inverse_transform(current_predictions)  # type: ignore
                except Exception as e:
                    raise ValueError(f"Failed to inverse transform from '{current_proc}' to '{ancestor}': {e}") from e
            else:
                raise ValueError(f"No inverse transformer available for processing '{current_proc}'")

            current_proc = ancestor

        # Step 2: Forward transform from common_ancestor to to_processing (if needed)
        if common_ancestor != to_processing:
            # Build path from common_ancestor to to_processing
            target_ancestry = self.get_processing_ancestry(to_processing)
            common_idx = target_ancestry.index(common_ancestor)
            forward_path = target_ancestry[common_idx + 1:]

            # Apply forward transformations
            for next_proc in forward_path:
                transformer = self._transformers.get(next_proc)

                if transformer is not None and hasattr(transformer, 'transform'):
                    try:
                        current_predictions = transformer.transform(current_predictions)  # type: ignore
                    except Exception as e:
                        raise ValueError(f"Failed to forward transform to '{next_proc}': {e}") from e
                else:
                    raise ValueError(f"No forward transformer available for processing '{next_proc}'")

        return current_predictions

    def _add_processing(self,
                        processing_name: str,
                        data: np.ndarray,
                        ancestor: Optional[str],
                        transformer: Optional[TransformerMixin]) -> None:
        """Internal method to add a processing."""


        if processing_name not in self._processing_id_to_index:
            # New processing: add to lists and mappings
            idx = len(self._processing_ids)
            self._processing_ids.append(processing_name)
            self._processing_id_to_index[processing_name] = idx
            self._data[processing_name] = data.copy()

            if ancestor is not None:
                self._ancestors[processing_name] = ancestor

            if transformer is not None:
                self._transformers[processing_name] = transformer
        else:
            # Existing processing: just update the data
            self._data[processing_name] = data.copy()

    def _make_numeric(self, y_raw: np.ndarray) -> tuple[np.ndarray, TransformerMixin]:
        """Convert raw targets to purely numeric data and return (numeric, transformer)."""
        # If we already have a numeric transformer, reuse it
        if "numeric" in self._transformers:
            existing_transformer = self._transformers["numeric"]
            if hasattr(existing_transformer, 'transform'):
                y_numeric = existing_transformer.transform(y_raw)  # type: ignore
                return y_numeric.astype(np.float32), existing_transformer

        # Check if data is already numeric
        if np.issubdtype(y_raw.dtype, np.number):
            # Data is already numeric, just use identity transformer
            transformer = FunctionTransformer(validate=False)
            transformer.fit(y_raw)
            return y_raw.astype(np.float32), transformer

        # Handle non-numeric data column by column
        y_numeric = np.empty_like(y_raw, dtype=np.float32)
        column_transformers = {}

        for col in range(y_raw.shape[1]):
            col_data = y_raw[:, col]

            if col_data.dtype.kind in {"U", "S", "O"}:  # strings / objects
                le = LabelEncoder()
                y_numeric[:, col] = le.fit_transform(col_data)
                column_transformers[col] = le
            else:
                # Try to convert to numeric
                try:
                    y_numeric[:, col] = col_data.astype(np.float32)
                    column_transformers[col] = None  # No transformation needed
                except (ValueError, TypeError):
                    # Fallback to LabelEncoder
                    le = LabelEncoder()
                    y_numeric[:, col] = le.fit_transform(col_data.astype(str))
                    column_transformers[col] = le

        # Create a simple wrapper transformer that remembers the column-wise transformers
        class SimpleColumnTransformer(TransformerMixin):
            def __init__(self, column_transformers_dict):
                self.column_transformers = column_transformers_dict

            def fit(self, _X, _y=None):
                return self

            def transform(self, X):
                X = np.asarray(X)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)

                result = np.empty_like(X, dtype=np.float32)
                for col, transformer in self.column_transformers.items():
                    if transformer is None:
                        # No transformation, just convert to numeric
                        result[:, col] = X[:, col].astype(np.float32)
                    else:
                        # Apply the transformer
                        result[:, col] = transformer.transform(X[:, col])
                return result

            def inverse_transform(self, X):
                X = np.asarray(X)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)

                result = np.empty(X.shape, dtype=object)
                for col, transformer in self.column_transformers.items():
                    if transformer is None:
                        # No transformation was applied
                        result[:, col] = X[:, col]
                    elif hasattr(transformer, 'inverse_transform'):
                        # Apply inverse transformation
                        result[:, col] = transformer.inverse_transform(X[:, col].astype(int))
                    else:
                        result[:, col] = X[:, col]
                return result

        transformer = SimpleColumnTransformer(column_transformers)
        return y_numeric, transformer
