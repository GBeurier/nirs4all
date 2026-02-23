import numpy as np
from nirs4all.operators.models.sklearn.aom_pls import LinearOperator, AOMPLSRegressor, default_operator_bank

class PseudoLinearSNVOperator(LinearOperator):
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._p = None

    @property
    def name(self) -> str: return "PseudoLinearSNV"

    @property
    def family(self) -> str: return "Scatter"

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=1, keepdims=True)
        self.std_ = np.std(X, axis=1, keepdims=True)
        self.std_[self.std_ == 0] = 1.0
        return self

    def initialize(self, p: int) -> None:
        self._p = p

    def apply(self, X: np.ndarray) -> np.ndarray:
        m = np.mean(X, axis=1, keepdims=True)
        s = np.std(X, axis=1, keepdims=True)
        s[s == 0] = 1.0
        return (X - m) / s

    def apply_adjoint(self, c: np.ndarray) -> np.ndarray:
        return c

    def frobenius_norm_sq(self) -> float:
        return float(self._p)

class PseudoLinearAOMPLSRegressor(AOMPLSRegressor):
    def fit(self, X, y, X_val=None, y_val=None):
        snv = PseudoLinearSNVOperator()
        snv.fit(X)
        self.operators_ = default_operator_bank() + [snv]
        original_bank = default_operator_bank
        import nirs4all.operators.models.sklearn.aom_pls as aom_module
        aom_module.default_operator_bank = lambda: self.operators_
        try:
            super().fit(X, y, X_val, y_val)
        finally:
            aom_module.default_operator_bank = original_bank
        return self
