import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.interpolate import interp1d


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
