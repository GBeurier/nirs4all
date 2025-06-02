import numpy as np
import polars as pl
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import warnings


class TargetManager:
    """
    Comprehensive target management for different ML tasks.

    Handles:
    - Original targets (strings, numbers, etc.)
    - Label encoding for classification
    - Target transformations via pipelines
    - Conversion between classification and regression
    - Inverse transformations for evaluation
    """

    def __init__(self, task_type: str = "auto"):
        """
        Initialize target manager.

        Args:
            task_type: "classification", "regression", "binary", or "auto"
        """
        self.task_type = task_type
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
          # Storage for different target representations
        self.targets = pl.DataFrame({
            "sample": pl.Series([], dtype=pl.Int64),
            "original": pl.Series([], dtype=pl.Utf8),  # String representation of original
            "regression": pl.Series([], dtype=pl.Float64),  # Numeric representation
            "classification": pl.Series([], dtype=pl.Int64),  # Label encoded
            "n_classes": pl.Series([], dtype=pl.Int64),  # Number of classes for this sample
        })

        # Pipeline transformers for each representation
        self.regression_transformers: Dict[str, List[BaseEstimator]] = {}
        self.classification_transformers: Dict[str, List[BaseEstimator]] = {}

        # Metadata
        self.classes_ = None
        self.n_classes_ = 0
        self.is_binary = False

    def add_targets(self, sample_ids: List[int], targets: np.ndarray) -> None:
        """Add new targets for given samples."""
        targets = np.asarray(targets)

        if not self.is_fitted:
            self._fit_encoders(targets)
          # Convert to different representations
        original_targets = [str(t) for t in targets.tolist()]  # Convert to strings
        regression_targets = self._to_regression(targets)
        classification_targets = self._to_classification(targets)

        new_targets = pl.DataFrame({
            "sample": sample_ids,
            "original": original_targets,
            "regression": regression_targets,
            "classification": classification_targets,
            "n_classes": [self.n_classes_] * len(sample_ids),
        })

        self.targets = pl.concat([self.targets, new_targets])

    def _fit_encoders(self, targets: np.ndarray) -> None:
        """Fit label encoders and determine task type."""
        targets = np.asarray(targets)

        # Auto-detect task type if needed
        if self.task_type == "auto":
            self.task_type = self._detect_task_type(targets)

        # Fit label encoder for classification tasks
        if self.task_type in ["classification", "binary"]:
            self.label_encoder.fit(targets)
            self.classes_ = self.label_encoder.classes_
            self.n_classes_ = len(self.classes_)
            self.is_binary = self.n_classes_ == 2
        else:
            # For regression, create dummy classes based on binning
            self._create_regression_classes(targets)

        self.is_fitted = True

    def _detect_task_type(self, targets: np.ndarray) -> str:
        """Auto-detect task type from targets."""
        targets = np.asarray(targets)

        # Check if targets are strings or have few unique values
        unique_values = np.unique(targets)

        if targets.dtype.kind in ['U', 'S', 'O']:  # String types
            return "binary" if len(unique_values) == 2 else "classification"

        # Numeric targets
        if len(unique_values) <= 10 and np.all(targets == targets.astype(int)):
            return "binary" if len(unique_values) == 2 else "classification"

        return "regression"

    def _create_regression_classes(self, targets: np.ndarray) -> None:
        """Create artificial classes for regression via binning."""
        targets = np.asarray(targets, dtype=float)

        # Use quantile-based binning
        n_bins = min(10, len(np.unique(targets)))
        _, bin_edges = np.histogram(targets, bins=n_bins)

        # Create class labels based on bins
        binned = np.digitize(targets, bin_edges) - 1
        binned = np.clip(binned, 0, n_bins - 1)

        self.label_encoder.fit(binned)
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        self.regression_bin_edges = bin_edges

    def _to_regression(self, targets: np.ndarray) -> List[float]:
        """Convert targets to regression format."""
        targets = np.asarray(targets)

        if self.task_type == "regression":
            return targets.astype(float).tolist()
        elif self.task_type in ["classification", "binary"]:
            # Convert class labels to numeric values
            if targets.dtype.kind in ['U', 'S', 'O']:
                encoded = self.label_encoder.transform(targets)
            else:
                encoded = targets.astype(int)
            return encoded.astype(float).tolist()
        else:
            return targets.astype(float).tolist()

    def _to_classification(self, targets: np.ndarray) -> List[int]:
        """Convert targets to classification format."""
        targets = np.asarray(targets)

        if self.task_type in ["classification", "binary"]:
            if targets.dtype.kind in ['U', 'S', 'O']:
                return self.label_encoder.transform(targets).tolist()
            else:
                return targets.astype(int).tolist()
        else:
            # For regression, use binning
            targets_float = targets.astype(float)
            binned = np.digitize(targets_float, self.regression_bin_edges) - 1
            binned = np.clip(binned, 0, self.n_classes_ - 1)
            return binned.tolist()

    def get_targets(self, sample_ids: List[int],
                   representation: str = "auto",
                   transformer_key: Optional[str] = None) -> np.ndarray:
        """
        Get targets for specific samples.

        Args:
            sample_ids: List of sample IDs
            representation: "original", "regression", "classification", or "auto"
            transformer_key: Key for transformed targets (if any)
        """
        if representation == "auto":
            representation = "classification" if self.task_type in ["classification", "binary"] else "regression"

        # Get base targets
        filtered = self.targets.filter(pl.col("sample").is_in(sample_ids))

        if transformer_key and transformer_key in self._get_transformers(representation):
            # Return transformed targets
            base_targets = filtered.select(representation).to_series().to_numpy()
            transformers = self._get_transformers(representation)[transformer_key]

            transformed = base_targets.reshape(-1, 1)
            for transformer in transformers:
                transformed = transformer.transform(transformed)

            return transformed.flatten()
        else:
            # Return base representation
            return filtered.select(representation).to_series().to_numpy()

    def fit_transform_targets(self, sample_ids: List[int],
                            transformers: List[BaseEstimator],
                            representation: str = "auto",
                            transformer_key: str = "default") -> np.ndarray:
        """
        Fit transformers on targets and return transformed values.

        Args:
            sample_ids: Sample IDs to transform
            transformers: List of sklearn transformers
            representation: Target representation to use
            transformer_key: Key to store transformers for later inverse transform
        """
        if representation == "auto":
            representation = "classification" if self.task_type in ["classification", "binary"] else "regression"

        targets = self.get_targets(sample_ids, representation)

        # Fit and transform
        transformed = targets.reshape(-1, 1)
        fitted_transformers = []

        for transformer in transformers:
            transformer.fit(transformed)
            transformed = transformer.transform(transformed)
            fitted_transformers.append(transformer)

        # Store fitted transformers
        transformer_dict = self._get_transformers(representation)
        transformer_dict[transformer_key] = fitted_transformers

        return transformed.flatten()

    def inverse_transform_predictions(self, predictions: np.ndarray,
                                    representation: str = "auto",
                                    transformer_key: str = "default",
                                    to_original: bool = True) -> np.ndarray:
        """
        Inverse transform predictions back to original format.

        Args:
            predictions: Predictions to inverse transform
            representation: Representation used for training
            transformer_key: Key of transformers used
            to_original: Whether to convert to original target format
        """
        if representation == "auto":
            representation = "classification" if self.task_type in ["classification", "binary"] else "regression"

        result = predictions.copy()

        # Inverse transform through pipelines
        if transformer_key in self._get_transformers(representation):
            transformers = self._get_transformers(representation)[transformer_key]

            result = result.reshape(-1, 1)
            for transformer in reversed(transformers):
                if hasattr(transformer, 'inverse_transform'):
                    result = transformer.inverse_transform(result)
            result = result.flatten()

        # Convert to original format if requested
        if to_original:
            if self.task_type in ["classification", "binary"]:
                # Convert back to original class labels
                result = result.astype(int)
                result = np.clip(result, 0, self.n_classes_ - 1)
                if hasattr(self.label_encoder, 'classes_'):
                    result = self.label_encoder.inverse_transform(result)

        return result

    def _get_transformers(self, representation: str) -> Dict[str, List[BaseEstimator]]:
        """Get transformer dictionary for given representation."""
        if representation == "regression":
            return self.regression_transformers
        else:
            return self.classification_transformers

    def get_info(self) -> Dict[str, Any]:
        """Get information about the target manager."""
        return {
            "task_type": self.task_type,
            "n_classes": self.n_classes_,
            "is_binary": self.is_binary,
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
            "n_samples": len(self.targets),
            "regression_transformers": list(self.regression_transformers.keys()),
            "classification_transformers": list(self.classification_transformers.keys()),
        }
