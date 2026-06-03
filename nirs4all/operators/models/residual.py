"""Residual model-combination operator.

Fits a BASE model, computes leakage-safe out-of-fold (OOF) residuals ``r = y - yhat_base_oof``, trains a
LEARNER on those residuals, and predicts ``base(X) + lambda * gate * learner(X)``. Both stages are ordinary
nirs4all models routed through their own framework controllers, so the learner can be ANY nirs4all model
(sklearn, the PyTorch ``nicon`` CNN, TabPFN, XGBoost, MLP) with no wrapper.

The combination is a first-class pipeline construct and composes under ``_cartesian_`` / ``_or_``::

    {"residual": {"base": {"_or_": [PLSRegression(10), Ridge()]},
                  "learner": {"_or_": [RandomForestRegressor(), nicon, TabPFNRegressor()]}}}

expands to every base x learner combination.

Example:
    >>> from nirs4all.operators.models import ResidualModel
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> pipeline = [
    ...     KFold(n_splits=5),
    ...     {"model": ResidualModel(base=PLSRegression(10), learner=RandomForestRegressor())},
    ... ]
"""

from typing import Any

from .base import BaseModelOperator


class ResidualModel(BaseModelOperator):
    """Residual combination of a base model and a residual learner.

    Attributes:
        base: nirs4all model for the (linear) first stage.
        learner: nirs4all model trained on the OOF residuals (any framework).
        lam: scalar weight on the learner term (default 1.0).
        gate: "auto" (closed-form OOF least-squares scalar in [0, 1]), a fixed float, or False (=1.0).
        rli_threshold: if gate=="auto", abstain (g=0) when the OOF residual-learnability is below this.
        train_params: optional dict forwarded to the learner's model step (e.g. epochs for nicon).
        finetune_space: optional hyperparameter search space forwarded to the learner.
    """

    _webapp_meta = {"category": "meta", "tier": "advanced",
                    "tags": ["residual", "combination", "hybrid", "ensemble"]}

    def __init__(
        self,
        base: Any = None,
        learner: Any = None,
        lam: float = 1.0,
        gate: Any = "auto",
        rli_threshold: float = 0.0,
        train_params: dict[str, Any] | None = None,
        finetune_space: dict[str, Any] | None = None,
        name: str | None = None,
    ):
        if base is None or learner is None:
            raise ValueError("ResidualModel requires both `base` and `learner` models.")
        self.base = base
        self.learner = learner
        self.lam = float(lam)
        self.gate = gate
        self.rli_threshold = float(rli_threshold)
        self.train_params = train_params
        self.finetune_space = finetune_space
        self._name = name

    def get_controller_type(self) -> str:
        return "residual"

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = {
            "base": self.base, "learner": self.learner, "lam": self.lam, "gate": self.gate,
            "rli_threshold": self.rli_threshold, "train_params": self.train_params,
            "finetune_space": self.finetune_space, "name": self._name,
        }
        if deep:
            for stage in ("base", "learner"):
                obj = getattr(self, stage)
                if hasattr(obj, "get_params"):
                    for k, v in obj.get_params(deep=True).items():
                        params[f"{stage}__{k}"] = v
        return params

    def set_params(self, **params) -> "ResidualModel":
        nested = {"base": {}, "learner": {}}
        for key, value in params.items():
            stage, _, sub = key.partition("__")
            if sub and stage in nested:
                nested[stage][sub] = value
            elif key == "name":
                self._name = value
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        for stage, sub in nested.items():
            obj = getattr(self, stage)
            if sub and hasattr(obj, "set_params"):
                obj.set_params(**sub)
        return self

    @property
    def name(self) -> str:
        if self._name:
            return self._name
        return f"Residual_{type(self.base).__name__}+{type(self.learner).__name__}"

    def __repr__(self) -> str:
        return (f"ResidualModel(base={type(self.base).__name__}, "
                f"learner={type(self.learner).__name__}, lam={self.lam}, gate={self.gate})")
