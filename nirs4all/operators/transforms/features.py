import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.interpolate import interp1d
from typing import List, Optional, Union


class CropTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, start: int = 0, end: int = None):
        self.start = start
        self.end = end

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if self.end is None or self.end > X.shape[1]:
            self.end = X.shape[1]
        return X[:, self.start:self.end]


class ResampleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if X.ndim != 2:
            raise ValueError("Input must be a 2D numpy array")

        resampled = []
        for x in X:
            if len(x) == self.num_samples:
                resampled.append(x)
            else:
                f = interp1d(np.linspace(0, 1, len(x)), x, kind='linear')
                resampled.append(f(np.linspace(0, 1, self.num_samples)))

        return np.array(resampled)


class FlattenPreprocessing(BaseEstimator, TransformerMixin):
    """Flatten the preprocessing dimension of a 3D feature array.

    Transforms a 3D array of shape (samples, preprocessings, features) into
    a 2D array of shape (samples, preprocessings * features) by horizontally
    concatenating all preprocessing views.

    This is useful after feature_augmentation when you want to flatten multiple
    preprocessing views into a single feature vector for models that expect 2D input.

    Args:
        sources: Which sources to apply the flattening to.
            - "all" (default): Apply to all sources
            - List of indices: [0, 2] to apply only to sources 0 and 2
            - Single int: Apply to only that source
            If a source is not in the list, it is passed through unchanged.

    Example:
        >>> # Input: (100, 4, 2151) - 4 preprocessing views of 2151 features each
        >>> flattener = FlattenPreprocessing()
        >>> output = flattener.transform(X)
        >>> # Output: (100, 8604) - 4 * 2151 = 8604 features

        >>> # Apply only to specific sources
        >>> flattener = FlattenPreprocessing(sources=[0, 2])
        >>> # Only sources 0 and 2 will be flattened

    Note:
        - If input is already 2D, it is returned unchanged.
        - The transformer is stateless (fit does nothing).
    """

    def __init__(
        self,
        sources: Union[str, int, List[int]] = "all",
    ):
        self.sources = sources

    def fit(self, X, y=None):
        """Fit is a no-op for this transformer."""
        return self

    def transform(self, X):
        """Flatten the preprocessing dimension.

        Args:
            X: Input array. Can be:
                - 2D array (samples, features): returned unchanged
                - 3D array (samples, preprocessings, features): flattened to 2D

        Returns:
            2D numpy array of shape (samples, preprocessings * features).
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        # Already 2D - nothing to flatten
        if X.ndim == 2:
            return X

        # Must be 3D to flatten
        if X.ndim != 3:
            raise ValueError(
                f"FlattenPreprocessing expects 2D or 3D input, got {X.ndim}D array "
                f"with shape {X.shape}"
            )

        # 3D array: (samples, preprocessings, features)
        n_samples, n_preprocessings, n_features = X.shape

        # Reshape to (samples, preprocessings * features)
        flattened = X.reshape(n_samples, n_preprocessings * n_features)

        return flattened

    def _should_apply_to_source(self, source_idx: int) -> bool:
        """Check if flattening should be applied to the given source.

        Args:
            source_idx: Index of the source.

        Returns:
            True if flattening should be applied.
        """
        if self.sources == "all":
            return True
        if isinstance(self.sources, int):
            return source_idx == self.sources
        if isinstance(self.sources, list):
            return source_idx in self.sources
        return True


# # Example usage:
# if __name__ == "__main__":
#     X = np.array([
#         [1.0, 2.0, 3.0, 4.0, 5.0],
#         [6.0, 7.0, 8.0, 9.0, 10.0]
#     ])

#     crop_transformer = CropTransformer(start=1, end=4)
#     resample_transformer = ResampleTransformer(num_samples=3)

#     X_cropped = crop_transformer.transform(X)
#     X_resampled = resample_transformer.transform(X)

#     print("Original X:")
#     print(X)
#     print("Cropped X:")
#     print(X_cropped)
#     print("Resampled X:")
#     print(X_resampled)
