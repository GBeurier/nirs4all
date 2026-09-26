"""Cross-language trained pipelines of n4m role steps (envelope version 7).

The envelope carries a portable recipe whose steps are generic n4m role
tokens (``"n4m:<catalog method id>"``, see
:mod:`nirs4all.pipeline.config.component_serialization`) and, for every
fitted step, its native N4ME state. Any n4m binding (Python, R, JS/WASM,
Rust) rebuilds the same estimators from those bytes and predicts
identically; numerics stay in Methods.

Recipe steps, in order: sample filters (training only: they drop training
rows and carry no state), transformers and selectors, then one regressor or
classifier.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "nirs4all.n4m.trained_pipeline.v7"


def _step(token: Any) -> Any:
    from nirs4all.pipeline.config.component_serialization import N4M_ROLE_PREFIX, deserialize_component

    name = token["class"] if isinstance(token, dict) else token
    if not isinstance(name, str) or not name.startswith(N4M_ROLE_PREFIX):
        raise ValueError(f"portable n4m role recipes contain n4m:<method id> steps only, got {token!r}")
    return deserialize_component(token)


class PortableN4MRolePipeline:
    """A fitted recipe of n4m role steps, portable as N4ME states.

    Args:
        recipe: ``{"pipeline": [step tokens]}``.
        estimators: The fitted transformers / selectors and the final model,
            in recipe order (filters excluded).
        n_features: Input width.
    """

    def __init__(self, recipe: dict[str, Any], estimators: list[Any], n_features: int) -> None:
        self.recipe = recipe
        self.estimators = estimators
        self.n_features = n_features

    @classmethod
    def fit_recipe(cls, recipe: dict[str, Any], X: Any, y: Any) -> PortableN4MRolePipeline:
        """Fit every step of ``recipe`` natively on ``X``, ``y``."""
        from n4m.roles import NativeClassifier, NativeRegressor, NativeSampleFilter, NativeSelector, NativeTransformer

        steps = [_step(token) for token in recipe["pipeline"]]
        if not steps or not isinstance(steps[-1], (NativeRegressor, NativeClassifier)):
            raise ValueError("a portable n4m role recipe ends with one regressor or classifier")
        values = np.asarray(X, dtype=np.float64)
        targets = np.asarray(y)
        n_features = values.shape[1]
        fitted: list[Any] = []
        for step in steps[:-1]:
            # Intermediate steps see y only when their method needs it
            # (a classification target is labels, not numbers).
            step_y = targets if step.input_requirements()["y"] == "required" else None
            if isinstance(step, NativeSampleFilter):
                keep = step.fit(values, step_y).get_mask(values, step_y)
                values, targets = values[keep], targets[keep]
            elif isinstance(step, (NativeTransformer, NativeSelector)):
                values = step.fit(values, step_y).transform(values)
                fitted.append(step)
            else:
                raise ValueError(f"{type(step).__name__} is not a portable pipeline step")
        fitted.append(steps[-1].fit(values, targets))
        return cls(recipe, fitted, n_features)

    @classmethod
    def from_json(cls, source: str | Path) -> PortableN4MRolePipeline:
        """Read a version 7 envelope (JSON text or path) and rebuild its estimators."""
        from n4m.roles import NativeEstimator, NativeSampleFilter

        text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
        if not text.lstrip().startswith("{"):
            text = Path(text).read_text(encoding="utf-8")
        document = json.loads(text)
        if not isinstance(document, dict) or document.get("schema") != SCHEMA:
            raise ValueError("unsupported trained n4m pipeline envelope")
        recipe, states = document["recipe"], document["states"]
        stateful = [token for token in recipe["pipeline"] if not isinstance(_step(token), NativeSampleFilter)]
        if len(states) != len(stateful):
            raise ValueError("envelope states do not match the recipe steps")
        estimators = []
        for token, state in zip(stateful, states, strict=True):
            payload = base64.b64decode(state["n4me_base64"], validate=True)
            if hashlib.sha256(payload).hexdigest() != state["sha256"]:
                raise ValueError(f"N4ME state of {state['method_id']} fails its checksum")
            estimator = NativeEstimator.from_n4me(payload)
            if estimator._method_id != state["method_id"] or type(_step(token)) is not type(estimator):
                raise ValueError(f"N4ME state {state['method_id']} does not match its recipe step")
            if "class_names" in state:
                estimator._label_names_ = np.asarray(state["class_names"])
            estimators.append(estimator)
        return cls(recipe, estimators, int(document["n_features"]))

    def to_json(self, file: str | Path | None = None) -> str:
        """The version 7 envelope; also written to ``file`` when given."""
        states = []
        for estimator in self.estimators:
            payload = estimator.to_n4me(allow_training_rows=True)
            state = {
                "method_id": estimator._method_id,
                "n4me_base64": base64.b64encode(payload).decode("ascii"),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            # Classifier label names (N4ME holds integer class ids only).
            names = getattr(estimator, "_label_names_", None)
            if names is not None:
                state["class_names"] = names.tolist()
            states.append(state)
        text = json.dumps({"schema": SCHEMA, "recipe": self.recipe, "n_features": self.n_features, "states": states}, indent=1)
        if file is not None:
            Path(file).write_text(text + "\n", encoding="utf-8")
        return text

    def _features(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} input columns")
        for step in self.estimators[:-1]:
            values = step.transform(values)
        return values

    def predict(self, X: Any) -> np.ndarray:
        """Predictions (regressor) or class labels (classifier) of the final model."""
        prediction: np.ndarray = self.estimators[-1].predict(self._features(X))
        return prediction

    def retrain(self, X: Any, y: Any) -> PortableN4MRolePipeline:
        """Fit the same recipe afresh on new training rows."""
        return type(self).fit_recipe(self.recipe, X, y)
