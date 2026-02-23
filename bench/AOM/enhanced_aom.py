import numpy as np
from nirs4all.operators.models.sklearn.aom_pls import AOMPLSRegressor, default_operator_bank
from pseudo_linear_aom import PseudoLinearSNVOperator

class EnhancedAOMPLSRegressor(AOMPLSRegressor):
    """
    Enhanced AOM-PLS Regressor.

    This model clones the baseline AOM-PLS but injects additional, powerful
    operators into the default operator bank. Specifically, it adds the
    PseudoLinearSNVOperator, which provides an exact analytical adjoint for
    the Standard Normal Variate (SNV) transform, allowing it to be used
    natively within the NIPALS extraction loop.
    """
    def __init__(self, n_components=15, **kwargs):
        super().__init__(n_components=n_components, **kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        # Create a rich operator bank
        snv_op = PseudoLinearSNVOperator()
        snv_op.fit(X)

        self.operators_ = default_operator_bank() + [snv_op]

        original_bank = default_operator_bank
        import nirs4all.operators.models.sklearn.aom_pls as aom_module
        aom_module.default_operator_bank = lambda: self.operators_
        try:
            super().fit(X, y, X_val, y_val)
        finally:
            aom_module.default_operator_bank = original_bank
        return self
