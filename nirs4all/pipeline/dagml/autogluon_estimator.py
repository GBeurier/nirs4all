"""Cloneable DAG host adapter for AutoGluon's directory-backed predictor."""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import type_of_target

from nirs4all.utils.backend import require_backend


def autogluon_step_estimator(step: dict[str, Any]) -> DagMLAutoGluonEstimator | None:
    """Adapt the explicit configurations routed to the legacy AutoGluon controller.

    Legacy reads constructor/fit options from the step's top-level ``params``
    and applies ``model_params`` as overrides. The nested ``model`` value is a
    framework selector; its own params are not used by that controller.
    """
    model = step.get("model")
    selected = (
        step.get("framework") == "autogluon"
        or isinstance(model, dict) and model.get("framework") == "autogluon"
        or getattr(type(model), "__module__", "").startswith("autogluon.")
        or isinstance(model, type) and model.__module__.startswith("autogluon.")
    )
    if not selected:
        return None
    params = dict(step.get("params") or {})
    params.update(step.get("model_params") or {})
    return DagMLAutoGluonEstimator(params=params)


class DagMLAutoGluonEstimator(BaseEstimator):
    """Fit one AutoGluon predictor per DAG fold and keep its artifact replayable."""

    def __init__(self, params: dict[str, Any] | None = None, fit_params: dict[str, Any] | None = None) -> None:
        self.params = params
        self.fit_params = fit_params

    def fit(self, X: Any, y: Any) -> DagMLAutoGluonEstimator:
        require_backend("autogluon", feature="AutoGluon AutoML")
        from autogluon.tabular import TabularPredictor

        values = np.asarray(y)
        if values.ndim > 1 and values.shape[1] != 1:
            raise ValueError("AutoGluon TabularPredictor requires one target column")
        labels = values.reshape(-1)
        target_kind = type_of_target(labels)
        problem_type = {"binary": "binary", "multiclass": "multiclass"}.get(target_kind, "regression")
        options = dict(self.params or {})
        random_state = options.pop("random_state", None)
        predictor_options = {
            "label": "__target__",
            "path": tempfile.mkdtemp(prefix="nirs4all_autogluon_"),
            "problem_type": problem_type,
            "verbosity": options.pop("verbosity", 0),
        }
        if "eval_metric" in options:
            predictor_options["eval_metric"] = options.pop("eval_metric")
        fit_options = {"presets": "medium_quality", **options, **(self.fit_params or {})}
        if random_state is not None:
            fit_options["ag_args_fit"] = {**fit_options.get("ag_args_fit", {}), "random_seed": random_state}
        train = pd.DataFrame(np.asarray(X))
        train["__target__"] = labels
        self.predictor_ = TabularPredictor(**predictor_options)
        self.predictor_.fit(train, **fit_options)
        self.n_features_in_ = train.shape[1] - 1
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "predictor_"):
            raise ValueError("AutoGluon predictor is not fitted")
        output = np.asarray(self.predictor_.predict(pd.DataFrame(np.asarray(X))))
        return output.reshape(-1, 1) if output.ndim == 1 else output

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "predictor_"):
            raise ValueError("AutoGluon predictor is not fitted")
        return np.asarray(self.predictor_.predict_proba(pd.DataFrame(np.asarray(X))))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        predictor = state.pop("predictor_", None)
        state.pop("_external_directory_owner", None)
        if state.get("_external_artifact_id") is not None:
            return state
        if predictor is not None:
            directory = Path(predictor.path)
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file in sorted(directory.rglob("*")):
                    if file.is_file():
                        archive.write(file, file.relative_to(directory))
            state["_predictor_directory"] = payload.getvalue()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        directory_bytes = state.pop("_predictor_directory", None)
        self.__dict__.update(state)
        if directory_bytes is not None:
            require_backend("autogluon", feature="AutoGluon archive replay")
            from autogluon.tabular import TabularPredictor

            directory = Path(tempfile.mkdtemp(prefix="nirs4all_autogluon_replay_"))
            with zipfile.ZipFile(io.BytesIO(directory_bytes)) as archive:
                for member in archive.infolist():
                    destination = (directory / member.filename).resolve()
                    if not destination.is_relative_to(directory.resolve()):
                        raise ValueError("AutoGluon artifact contains a path outside its model directory")
                    archive.extract(member, directory)
            self.predictor_ = TabularPredictor.load(str(directory))

    def load_external_directory(self, directory: Path) -> None:
        """Load a verified sidecar after the enclosing archive has been checked."""
        require_backend("autogluon", feature="AutoGluon archive replay")
        from autogluon.tabular import TabularPredictor

        self.predictor_ = TabularPredictor.load(str(directory))
        self.__dict__.pop("_external_artifact_id", None)
