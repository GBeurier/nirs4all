"""Cloneable DAG host for the PyTorch models accepted by the legacy controller.

Factories cross the DAG JSON boundary by import path. Caller-supplied module
templates cross it as trusted cloudpickle bytes, so initial weights and custom
constructor state are preserved. Each fit creates a fresh module.
"""

from __future__ import annotations

import base64
import copy
import importlib
import inspect
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


def torch_model_params(model: Any) -> dict[str, Any] | None:
    """Encode a PyTorch factory or module template for the DAG model node."""
    if isinstance(model, dict) and model.get("framework") == "pytorch" and model.get("type") == "function":
        function = model.get("func")
        if isinstance(function, dict):
            function = function.get("func")
        spec = torch_model_params(function)
        if spec is None:
            raise TypeError("PyTorch function configuration has no callable factory")
        spec["factory_params"] = dict(model.get("params") or {})
        return spec

    if getattr(model, "framework", None) == "pytorch" and (inspect.isfunction(model) or inspect.isclass(model)):
        path = f"{model.__module__}.{model.__qualname__}"
        module_name, _, name = path.rpartition(".")
        try:
            resolved: Any = importlib.import_module(module_name)
            for part in name.split("."):
                resolved = getattr(resolved, part)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"PyTorch factory {path!r} must be importable for DAG execution") from exc
        if resolved is not model:
            raise ValueError(f"PyTorch factory {path!r} must be importable for DAG execution")
        return {"factory_path": path}

    # A supplied module may carry non-default constructor arguments or initial
    # weights. Preserve that exact template, then copy it afresh for each fold.
    if not any(base.__module__.startswith("torch.") for base in type(model).__mro__):
        return None
    try:
        import torch
    except ImportError:
        return None
    if isinstance(model, torch.nn.Module):
        import cloudpickle

        return {"template_blob": base64.b64encode(cloudpickle.dumps(model)).decode("ascii")}
    return None


class DagMLTorchEstimator(BaseEstimator):
    """sklearn-shaped adapter using the existing PyTorch controller's fit loop."""

    framework = "pytorch"

    def __init__(
        self,
        factory_path: str | None = None,
        template_blob: str | None = None,
        factory_params: dict[str, Any] | None = None,
        task_type: str | None = None,
        num_classes: int | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        optimizer: Any = "Adam",
        lr: float = 0.001,
        learning_rate: float | None = None,
        loss: Any = "MSELoss",
    ) -> None:
        self.factory_path = factory_path
        self.template_blob = template_blob
        self.factory_params = factory_params
        self.task_type = task_type
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.optimizer = optimizer
        self.lr = lr
        self.learning_rate = learning_rate
        self.loss = loss

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = dict(super().get_params(deep=deep))
        if deep:
            params.update(self.factory_params or {})
        return params

    def set_params(self, **params: Any) -> DagMLTorchEstimator:
        own = super().get_params(deep=False)
        for key, value in params.items():
            if key in own:
                setattr(self, key, value)
            else:
                factory_params = dict(self.factory_params or {})
                factory_params[key] = value
                self.factory_params = factory_params
        return self

    def _new_model(self, input_shape: tuple[int, ...]) -> Any:
        if self.template_blob is not None:
            import cloudpickle

            model = cloudpickle.loads(base64.b64decode(self.template_blob))
            # Legacy accepts decorated, deferred-build module instances such as
            # SimpleRegressor(input_shape=None). Rebuild these once the fold's
            # feature shape is known; keep already-initialized weights intact.
            if (
                getattr(model, "framework", None) == "pytorch"
                and not any(True for _ in model.parameters())
                and "input_shape" in inspect.signature(type(model)).parameters
            ):
                from nirs4all.controllers.models.factory import ModelFactory

                return ModelFactory.reconstruct_object(model, force_params={**(self.factory_params or {}), "input_shape": input_shape})
            return model
        if self.factory_path is None:
            raise ValueError("PyTorch DAG model requires a factory or a module template")
        module_name, _, name = self.factory_path.rpartition(".")
        factory: Any = importlib.import_module(module_name)
        for part in name.split("."):
            factory = getattr(factory, part)
        from nirs4all.controllers.models.factory import ModelFactory

        params = dict(self.factory_params or {})
        params["input_shape"] = input_shape
        if self.num_classes is not None:
            params["num_classes"] = self.num_classes
        return ModelFactory.prepare_and_call(factory, params)

    def _features(self, X: Any) -> np.ndarray:
        data = np.asarray(X, dtype=np.float32)
        if data.ndim == 2 and self.input_layout_ == "channels_first":
            return data[:, np.newaxis, :]
        if data.ndim not in (2, 3):
            raise ValueError(f"PyTorch model requires 2D or 3D features, got {data.shape}")
        return data

    def fit(self, X: Any, y: Any) -> DagMLTorchEstimator:
        import torch

        from nirs4all.controllers.models.torch.data_prep import PyTorchDataPreparation
        from nirs4all.controllers.models.torch_model import PyTorchModelController
        from nirs4all.core.task_type import TaskType

        raw = np.asarray(X, dtype=np.float32)
        if raw.ndim not in (2, 3):
            raise ValueError(f"PyTorch model requires 2D or 3D features, got {raw.shape}")
        factory_shape = tuple(raw.shape[1:]) if raw.ndim == 3 else (1, raw.shape[1])
        model = self._new_model(factory_shape)
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"PyTorch factory produced {type(model).__name__}, expected torch.nn.Module")
        self.input_layout_ = "channels_first" if self.factory_path is not None or any(isinstance(layer, torch.nn.Conv1d) for layer in model.modules()) else "flat"
        features = self._features(raw)
        x_tensor, y_tensor = PyTorchDataPreparation.prepare_data(features, np.asarray(y))
        task_type = TaskType(self.task_type or "regression")
        if (
            task_type == TaskType.MULTICLASS_CLASSIFICATION
            and isinstance(self.loss, str)
            and self.loss.lower() in {"crossentropy", "crossentropyloss"}
            and y_tensor is not None
            and y_tensor.ndim == 2
            and y_tensor.shape[1] == 1
        ):
            # The shared data preparer produces float column targets for
            # regression. CrossEntropyLoss needs integer class indices.
            y_tensor = y_tensor.reshape(-1).long()
        controller = PyTorchModelController()
        training = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "optimizer": copy.deepcopy(self.optimizer),
            "lr": self.learning_rate if self.learning_rate is not None else self.lr,
            "loss": self.loss,
            "task_type": task_type,
        }
        self.model_ = controller._train_model(model, x_tensor, y_tensor, **training)
        self.n_features_in_ = raw.shape[1] if raw.ndim == 2 else int(np.prod(raw.shape[1:]))
        return self

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        if self.template_blob is not None and "model_" in state:
            import cloudpickle

            state["_fitted_module_bytes"] = cloudpickle.dumps(state.pop("model_"))
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        model_bytes = state.pop("_fitted_module_bytes", None)
        self.__dict__.update(state)
        if model_bytes is not None:
            import cloudpickle

            self.model_ = cloudpickle.loads(model_bytes)

    def predict(self, X: Any) -> np.ndarray:
        from nirs4all.controllers.models.torch_model import PyTorchModelController

        if not hasattr(self, "model_"):
            raise ValueError("PyTorch DAG estimator is not fitted")
        return np.asarray(PyTorchModelController()._predict_model(self.model_, self._features(X)))
