import numpy as np
from nirs4all.operators.models.sklearn.aom_pls import LinearOperator, AOMPLSRegressor, default_operator_bank

class PseudoLinearSNVOperator(LinearOperator):
    """
    Pseudo-Linear Standard Normal Variate (SNV) Operator.

    SNV is inherently non-linear because it divides by the per-sample standard
    deviation. This operator implements the exact analytical adjoint of the SNV
    transform, allowing it to be used as a "pseudo-linear" operator within the
    AOM-PLS NIPALS loop.
    """
    def __init__(self):
        self.mean_spectrum_ = None
        self.mean_s_ = None
        self.std_s_ = None
        self.S_x_ = None
        self._p = None

    @property
    def name(self) -> str: return "PseudoLinearSNV"

    @property
    def family(self) -> str: return "Scatter"

    def fit(self, X: np.ndarray):
        self.mean_spectrum_ = np.mean(X, axis=0)
        self.mean_s_ = np.mean(self.mean_spectrum_)
        self.std_s_ = np.std(self.mean_spectrum_)
        if self.std_s_ == 0: self.std_s_ = 1.0
        self.S_x_ = (self.mean_spectrum_ - self.mean_s_) / self.std_s_
        return self

    def initialize(self, p: int) -> None:
        self._p = p

    def apply(self, X: np.ndarray) -> np.ndarray:
        m = np.mean(X, axis=1, keepdims=True)
        s = np.std(X, axis=1, keepdims=True)
        s[s == 0] = 1.0
        return (X - m) / s

    def apply_adjoint(self, c: np.ndarray) -> np.ndarray:
        c_mean = np.mean(c)
        dot_prod = np.dot(self.S_x_, c) / self._p
        return (c - c_mean - dot_prod * self.S_x_) / self.std_s_

    def frobenius_norm_sq(self) -> float:
        return float(self._p)

class PseudoLinearAOMPLSRegressor(AOMPLSRegressor):
    """
    AOM-PLS Regressor with Pseudo-Linear SNV.

    This regressor injects the PseudoLinearSNVOperator into the default
    operator bank before fitting.
    """
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
