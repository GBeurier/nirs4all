"""DAG host adapter for legacy TensorFlow and JAX model factories."""

from __future__ import annotations

import base64
import importlib
import inspect
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


def framework_model_params(model: Any) -> dict[str, Any] | None:
    """Encode a TensorFlow/JAX factory or template for DAG reconstruction."""
    if isinstance(model, dict) and model.get("framework") in {"tensorflow", "jax"} and model.get("type") == "function":
        function = model.get("func")
        if isinstance(function, dict):
            function = function.get("func")
        spec = framework_model_params(function)
        if spec is None:
            raise TypeError("Framework function configuration has no callable factory")
        spec["factory_params"] = dict(model.get("params") or {})
        return spec

    framework = getattr(model, "framework", None)
    if framework in {"tensorflow", "jax"} and (inspect.isfunction(model) or inspect.isclass(model)):
        path = f"{model.__module__}.{model.__qualname__}"
        module_name, _, name = path.rpartition(".")
        try:
            resolved: Any = importlib.import_module(module_name)
            for part in name.split("."):
                resolved = getattr(resolved, part)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"{framework} factory {path!r} must be importable for DAG execution") from exc
        if resolved is not model:
            raise ValueError(f"{framework} factory {path!r} must be importable for DAG execution")
        return {"framework": framework, "factory_path": path}

    bases = type(model).__mro__
    if any(base.__module__.startswith(("keras.", "tensorflow.")) for base in bases):
        framework = "tensorflow"
    elif any(base.__module__.startswith("flax.") for base in bases):
        framework = "jax"
    else:
        return None
    import cloudpickle

    return {"framework": framework, "template_blob": base64.b64encode(cloudpickle.dumps(model)).decode("ascii")}


class DagMLFrameworkEstimator(BaseEstimator):
    """Cloneable host estimator that delegates training to the legacy controller."""

    def __init__(
        self,
        framework: str = "tensorflow",
        factory_path: str | None = None,
        template_blob: str | None = None,
        factory_params: dict[str, Any] | None = None,
        input_layout: str = "channels_first",
        task_type: str | None = None,
        num_classes: int | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        optimizer: Any = "Adam",
        lr: float = 0.001,
        learning_rate: float | None = None,
        loss: Any = None,
        metrics: Any = None,
        validation_split: float = 0.2,
    ) -> None:
        self.framework = framework
        self.factory_path = factory_path
        self.template_blob = template_blob
        self.factory_params = factory_params
        self.input_layout = input_layout
        self.task_type = task_type
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.optimizer = optimizer
        self.lr = lr
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.validation_split = validation_split

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = dict(super().get_params(deep=deep))
        if deep:
            params.update(self.factory_params or {})
        return params

    def set_params(self, **params: Any) -> DagMLFrameworkEstimator:
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

            return cloudpickle.loads(base64.b64decode(self.template_blob))
        if self.factory_path is None:
            raise ValueError("DAG framework model requires a factory or a model template")
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
        raw = np.asarray(X, dtype=np.float32)
        if raw.ndim == 2:
            return raw[:, :, np.newaxis]
        if raw.ndim != 3:
            raise ValueError(f"{self.framework} model requires 2D or 3D features, got {raw.shape}")
        if self.input_layout == "channels_first":
            return np.transpose(raw, (0, 2, 1))
        if self.input_layout != "channels_last":
            raise ValueError("input_layout must be 'channels_first' or 'channels_last'")
        return raw

    def fit(self, X: Any, y: Any) -> DagMLFrameworkEstimator:
        from nirs4all.core.task_type import TaskType

        features = self._features(X)
        model = self._new_model(tuple(features.shape[1:]))
        task_type = TaskType(self.task_type or "regression")
        controls: dict[str, Any] = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "optimizer": self.optimizer,
            "lr": self.learning_rate if self.learning_rate is not None else self.lr,
            "task_type": task_type,
            "verbose": 0,
        }
        if self.loss is not None:
            controls["loss"] = self.loss
        controller: Any
        if self.framework == "tensorflow":
            from nirs4all.controllers.models.tensorflow_model import TensorFlowModelController

            controller = TensorFlowModelController()
            x_train, y_train = controller._prepare_data(features, np.asarray(y), {})
            if self.metrics is not None:
                controls["metrics"] = self.metrics
            controls["validation_split"] = self.validation_split
        elif self.framework == "jax":
            from nirs4all.controllers.models.jax_model import JaxModelController

            controller = JaxModelController()
            x_train, y_train = controller._prepare_data(features, np.asarray(y), {})
        else:
            raise ValueError(f"unsupported framework {self.framework!r}")
        assert y_train is not None
        self.model_ = controller._train_model(model, x_train, y_train, **controls)
        self.n_features_in_ = int(np.prod(np.asarray(X).shape[1:]))
        return self

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        if self.framework == "jax" and "model_" in state:
            import cloudpickle

            state["_jax_model_bytes"] = cloudpickle.dumps(state.pop("model_"))
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        model_bytes = state.pop("_jax_model_bytes", None)
        self.__dict__.update(state)
        if model_bytes is not None:
            import cloudpickle

            self.model_ = cloudpickle.loads(model_bytes)

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("DAG framework estimator is not fitted")
        features = self._features(X)
        if self.framework == "tensorflow":
            predictions = np.asarray(self.model_.predict(features, verbose=0))
        else:
            from nirs4all.controllers.models.jax.data_prep import JaxDataPreparation

            predictions = np.asarray(self.model_.predict(JaxDataPreparation.prepare_features(features)))
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if self.task_type and "classification" in self.task_type and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1).reshape(-1, 1)
        return predictions
