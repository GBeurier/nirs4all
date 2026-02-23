import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression

class ZeroShotRouterPLSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X, y):
        var_x = np.var(X, axis=1)
        var_dx = np.var(np.diff(X, axis=1), axis=1)
        ratio = np.mean(var_x / (var_dx + 1e-6))
        
        if ratio > 1000:
            self.preprocessor_ = "SNV"
            self.X_mean_ = np.mean(X, axis=1, keepdims=True)
            self.X_std_ = np.std(X, axis=1, keepdims=True)
            self.X_std_[self.X_std_ == 0] = 1.0
            X_pre = (X - self.X_mean_) / self.X_std_
        else:
            self.preprocessor_ = "SG"
            from scipy.signal import savgol_filter
            X_pre = savgol_filter(X, window_length=15, polyorder=2, deriv=1, axis=1)
            
        self.pls_ = PLSRegression(n_components=self.n_components)
        self.pls_.fit(X_pre, y)
        return self

    def predict(self, X):
        if self.preprocessor_ == "SNV":
            m = np.mean(X, axis=1, keepdims=True)
            s = np.std(X, axis=1, keepdims=True)
            s[s == 0] = 1.0
            X_pre = (X - m) / s
        else:
            from scipy.signal import savgol_filter
            X_pre = savgol_filter(X, window_length=15, polyorder=2, deriv=1, axis=1)
        return self.pls_.predict(X_pre).flatten()
