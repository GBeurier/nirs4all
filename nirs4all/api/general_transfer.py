"""Frozen preprocessing primitives for captured-host transfer training.

These are scientific host estimators, not an execution engine. The DAG still
owns training tasks. Frozen state is copied privately before use, and only the
new final estimator receives a fit on the new training partition.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from copy import deepcopy
from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class FrozenTransferTransform(TransformerMixin, BaseEstimator):
    """Apply a captured transformation without fitting it on transfer data."""

    def __init__(self, fitted: Any, state_fingerprint: str | None = None) -> None:
        self.fitted = fitted
        self.state_fingerprint = state_fingerprint if state_fingerprint is not None else _state_fingerprint(fitted)

    def __sklearn_clone__(self) -> FrozenTransferTransform:
        # sklearn's ordinary clone would reset the captured transformation.
        # Each task instead owns a private copy of its deliberately frozen state.
        return type(self)(deepcopy(self.fitted), self.state_fingerprint)

    def fit(self, X: Any, y: Any = None) -> FrozenTransferTransform:
        if _state_fingerprint(self.fitted) != self.state_fingerprint:
            raise ValueError("captured transfer state was not hydrated or changed before execution")
        return self

    def transform(self, X: Any) -> Any:
        return self.fitted.transform(X)

    def inverse_transform(self, X: Any) -> Any:
        return self.fitted.inverse_transform(X)

    def __sklearn_is_fitted__(self) -> bool:
        return True


def fresh_training_estimator(estimator: Any) -> Any:
    """Remove our deliberate transfer freezes before a subsequent full retrain."""
    from sklearn.base import clone

    if estimator is None or isinstance(estimator, str) and estimator == "passthrough":
        return estimator
    if isinstance(estimator, FrozenTransferTransform):
        return fresh_training_estimator(estimator.fitted)
    if isinstance(estimator, Pipeline):
        fresh = clone(estimator)
        if fresh is estimator:
            raise ValueError("cloning retained a captured estimator")
        fresh.steps = [(name, fresh_training_estimator(step)) for name, step in estimator.steps]
        return fresh
    fresh = clone(estimator)
    if fresh is estimator:
        raise ValueError("cloning retained a captured estimator")
    return fresh


def transfer_training_steps(estimator: Any, target: Any, new_model: Any = None) -> list[dict[str, Any]]:
    """Freeze the captured X/y transforms and initialize a fresh final model.

    Only the recorded sklearn pipeline boundary is used to separate a final
    estimator from preprocessing. Prediction-only fusion wrappers are not
    guessed into a training graph.
    """
    preprocessing = None
    model = estimator
    if isinstance(estimator, Pipeline):
        if not estimator.steps:
            raise ValueError("transfer requires a nonempty captured pipeline")
        model = estimator.steps[-1][1]
        if len(estimator.steps) > 1:
            preprocessing = estimator[:-1]
    model = model if new_model is None else new_model
    if not callable(getattr(model, "fit", None)) or not callable(getattr(model, "predict", None)):
        raise ValueError("transfer requires a trainable final estimator")
    fresh_model = fresh_training_estimator(model)
    if fresh_model is model:
        raise ValueError("transfer model clone retained the source estimator")
    if preprocessing is not None:
        fresh_model = Pipeline([
            ("captured_preprocessing", FrozenTransferTransform(deepcopy(preprocessing))),
            ("model", fresh_model),
        ])
    steps = []
    if target is not None:
        if not callable(getattr(target, "transform", None)) or not callable(getattr(target, "inverse_transform", None)):
            raise ValueError("transfer target preprocessing must transform and inverse-transform")
        steps.append({"y_processing": FrozenTransferTransform(deepcopy(target))})
    steps.append({"model": fresh_model})
    return steps


def _state_fingerprint(fitted: Any) -> str:
    # This hashes an already verified trusted Python object; no pickle is read
    # from DSL/JSON. Only the fingerprint enters the native graph contract.
    return "sha256:" + hashlib.sha256(pickle.dumps(fitted, protocol=5)).hexdigest()


def _frozen_transforms(operator: Any) -> list[FrozenTransferTransform]:
    values = [operator]
    if callable(getattr(operator, "get_params", None)):
        values.extend(operator.get_params(deep=True).values())
    return [value for value in values if isinstance(value, FrozenTransferTransform)]


class BoundTransferOperator:
    """Typed process-local binding, never a JSON-reconstructible fitted object."""

    def __init__(self, operator: Any) -> None:
        self.operator = operator

    def instantiate(self, variant_overrides: dict[str, Any] | None) -> Any:
        from sklearn.base import clone

        if variant_overrides:
            raise ValueError("captured transfer does not accept a new parameter search")
        for frozen in _frozen_transforms(self.operator):
            frozen.fit(None)
        return clone(self.operator)


def bind_transfer_operators(nodes: dict[str, dict[str, Any]], steps: list[Any]) -> None:
    """Attach frozen host objects after compilation, preserving native identity.

    The in-process full-train executor alone owns this binding. Its compiled
    JSON graph contains constructor specs and fingerprints, never pickle bytes
    or object handles usable by another process. No normal operator is replaced.
    """
    for step in steps:
        if not isinstance(step, dict):
            continue
        for keyword, kind in (("model", "model"), ("y_processing", "y_transform")):
            operator = step.get(keyword)
            frozen = _frozen_transforms(operator)
            if not frozen:
                continue
            candidates = [node for node in nodes.values() if node["kind"] == kind]
            if len(candidates) != 1:
                raise ValueError("captured transfer requires one unambiguous compiled operator")
            node = candidates[0]
            contract = json.dumps(node, sort_keys=True)
            for transform in frozen:
                if transform.state_fingerprint not in contract:
                    raise ValueError("compiled transfer operator lost its captured-state fingerprint")
                transform.fit(None)
            node["_nirs4all_transfer_binding"] = BoundTransferOperator(operator)
