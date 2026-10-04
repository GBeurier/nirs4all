"""Joint Torch regression with separately named dense input tensors.

The DAG owns row selection, folds and refit. This host adapter retains each
input name through the shared Torch trainer and the user's forward method.
"""
from __future__ import annotations

import base64
import importlib
from collections.abc import Mapping
from typing import Any, Self

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .torch_estimator import DagMLTorchEstimator


class DagMLNamedTorchEstimator(DagMLTorchEstimator):
    """Cloneable adapter for an importable factory or a trusted module template.

    A factory receives ``input_shapes={source: non_sample_shape}``. Its module implements
    ``forward(**inputs)`` and returns one regression column. The inherited
    constructor exposes the existing Torch training controls; this first named
    profile requires CPU, Adam/MSE and two to four complete dense sources.
    """

    def _named_features(self, X: Any, *, fitting: bool) -> dict[str, np.ndarray]:
        if not isinstance(X, Mapping) or not 2 <= len(X) <= 4:
            raise ValueError("Named Torch inputs require a mapping of two to four sources.")
        names = tuple(X)
        if any(not isinstance(name, str) or not name.isascii() or not name.isidentifier() or name == "y" for name in names):
            raise ValueError("Named Torch sources must be distinct ASCII Python identifiers other than y.")
        if not fitting and set(names) != set(self.input_shapes_):
            raise ValueError("Named Torch prediction source names differ from fitted inputs.")
        blocks = {}
        count = None
        for name in names:
            raw = np.asarray(X[name])
            if not 2 <= raw.ndim <= 4 or not all(raw.shape) or raw.dtype not in (np.dtype("float32"), np.dtype("float64")):
                raise ValueError(f"Named Torch source {name!r} requires a nonempty rank-2 to rank-4 float32/float64 tensor.")
            if count is not None and len(raw) != count:
                raise ValueError("Named Torch sources must have the same sample count.")
            count = len(raw)
            if raw.size > 16_777_216 or not np.isfinite(raw).all():
                raise ValueError(f"Named Torch source {name!r} must be bounded and finite.")
            with np.errstate(over="ignore", invalid="ignore"):
                values = raw.astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"Named Torch source {name!r} is not finite after float32 conversion.")
            if not fitting and (raw.shape[1:] != self.input_shapes_[name] or raw.dtype.str != self.input_dtypes_[name]):
                raise ValueError(f"Named Torch source {name!r} changed fitted shape or dtype.")
            blocks[name] = values
        return blocks

    def _new_named_model(self, input_shapes: dict[str, tuple[int, ...]]) -> Any:
        if (self.factory_path is None) == (self.template_blob is None):
            raise ValueError("Named Torch requires exactly one importable factory or trusted template.")
        if self.template_blob is not None:
            import cloudpickle

            return cloudpickle.loads(base64.b64decode(self.template_blob))
        module_name, _, name = str(self.factory_path).rpartition(".")
        factory = getattr(importlib.import_module(module_name), name)
        params = dict(self.factory_params or {})
        if "input_shapes" in params:
            raise ValueError("Named Torch input_shapes are supplied by the admitted source contract.")
        return factory(input_shapes=input_shapes, **params)

    def validate_configuration(self) -> None:
        """Refuse unsupported controls before a native run dispatches callbacks."""
        if (self.factory_path is None) == (self.template_blob is None):
            raise ValueError("Named Torch requires exactly one importable factory or trusted template.")
        if (self.device != "cpu" or self.task_type != "regression" or self.num_classes is not None
                or self.force_layout not in (None, "2d") or self.optimizer != "Adam" or self.loss != "MSELoss"
                or self.learning_rate is not None):
            raise ValueError("Named Torch requires CPU mono-y regression with Adam/MSELoss and explicit lr.")
        for name, maximum in (("epochs", 100), ("batch_size", 1024), ("patience", 100)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"Named Torch {name} must be an integer in [1,{maximum}].")
        if type(self.lr) not in (float, int) or not np.isfinite(self.lr) or not 1e-6 <= self.lr <= 0.1:
            raise ValueError("Named Torch lr must be finite in [1e-6,0.1].")

    def fit(self, X: Any, y: Any) -> Self:
        import torch

        from nirs4all.controllers.models.torch_model import PyTorchModelController
        from nirs4all.core.task_type import TaskType

        self.validate_configuration()
        features = self._named_features(X, fitting=True)
        count = len(next(iter(features.values())))
        target = np.asarray(y)
        if target.shape not in ((count,), (count, 1)) or target.dtype.kind not in "fiu":
            raise ValueError("Named Torch requires one numeric target per sample.")
        with np.errstate(over="ignore", invalid="ignore"):
            target = target.astype(np.float32).reshape(count, 1)
        if not np.isfinite(target).all():
            raise ValueError("Named Torch targets must be finite after float32 conversion.")
        shapes = {name: values.shape[1:] for name, values in features.items()}
        model = self._new_named_model(shapes)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Named Torch factory must return torch.nn.Module.")
        floating_state = [tensor for tensor in (*model.parameters(), *model.buffers()) if tensor.is_floating_point()]
        if any(tensor.dtype != torch.float32 for tensor in floating_state):
            raise ValueError("Named Torch currently requires float32 model parameters and buffers.")
        parameters = sum(parameter.numel() for parameter in model.parameters())
        if not 0 < parameters <= 1_000_000 or count * self.epochs * parameters > 100_000_000:
            raise ValueError("Named Torch model exceeds the CPU parameter or training-work budget.")
        trained = PyTorchModelController()._train_model(
            model, {name: torch.from_numpy(values) for name, values in features.items()}, torch.from_numpy(target),
            device="cpu", task_type=TaskType.REGRESSION, epochs=self.epochs, batch_size=self.batch_size,
            patience=self.patience, optimizer=self.optimizer, lr=self.lr, loss=self.loss,
        )
        self.model_ = trained
        self.input_shapes_ = shapes
        self.input_dtypes_ = {name: np.asarray(X[name]).dtype.str for name in features}
        self.source_names_ = tuple(features)
        self.n_features_in_ = sum(int(np.prod(shape)) for shape in shapes.values())
        return self

    def predict(self, X: Any) -> np.ndarray:
        import torch

        check_is_fitted(self, ["model_", "input_shapes_", "input_dtypes_"])
        features = self._named_features(X, fitting=False)
        self.model_.eval()
        with torch.no_grad():
            result = self.model_(**{name: torch.from_numpy(features[name]) for name in self.source_names_})
        if not isinstance(result, torch.Tensor):
            raise ValueError("Named Torch regression must return one numeric prediction tensor.")
        values = result.detach().cpu().numpy()
        count = len(next(iter(features.values())))
        if values.shape != (count, 1) or not np.isfinite(values).all():
            raise ValueError("Named Torch regression must return one finite prediction per sample.")
        return np.asarray(values)

    def __getstate__(self) -> dict[str, Any]:
        import cloudpickle

        state = dict(self.__dict__)
        if "model_" in state:
            state["_fitted_module_bytes"] = cloudpickle.dumps(state.pop("model_"))
        return state
