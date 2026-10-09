"""Editable declarations for verified captured REFIT chains."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def workspace_chain_snapshot(workspace_path: str | Path, chain_id: str) -> list[Any] | None:
    """Return fresh constructor declarations for a selected chain, never fitted state.

    The persisted host chain marker describes one captured predictor, not an
    authoring operator. Read its verified artifact through the replay owner and
    retain the selected model parameters and concrete preprocessing recipe.
    Recorded CV splitters are cloned without their old dataset row identities.
    Unsupported joined/multi-source captures refuse rather than guessing.
    """
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
    from nirs4all.pipeline.dagml.general_workspace import load_general_workspace_chain
    from nirs4all.pipeline.dagml.node_runner import _FittedXChain, _FrozenTransform
    from nirs4all.pipeline.dagml.steps import FrozenDagMlSplitStep, _is_split_step

    loaded = load_general_workspace_chain(workspace_path, chain_id)
    if loaded is None:
        return None

    def transforms(operator: Any) -> list[Any]:
        if isinstance(operator, Pipeline):
            return [step for _, child in operator.steps for step in transforms(child)]
        if isinstance(operator, _FrozenTransform):
            return transforms(operator.transformer)
        if isinstance(operator, _FittedXChain):
            if operator.source_steps is not None:
                raise ValueError("chain snapshot requires explicit source topology for a multi-source predictor")
            return [step for child in operator.steps for step in transforms(child)]
        if not callable(getattr(operator, "get_params", None)) or not callable(getattr(operator, "transform", None)):
            raise ValueError("chain snapshot preprocessing cannot be reconstructed from constructor parameters")
        return [clone(operator)]

    estimator = loaded["artifact"]["estimator"]
    preprocessing: list[Any] = []
    if isinstance(estimator, Pipeline):
        if not estimator.steps:
            raise ValueError("chain snapshot predictor has no model")
        preprocessing = [step for _, child in estimator.steps[:-1] for step in transforms(child)]
        estimator = estimator.steps[-1][1]
    if not callable(getattr(estimator, "get_params", None)) or not callable(getattr(estimator, "fit", None)):
        raise ValueError("chain snapshot model cannot be reconstructed from constructor parameters")

    steps: list[Any] = []
    training = loaded.get("training_pipeline") or []
    if isinstance(training, dict):
        training = training.get("pipeline", training.get("steps", []))
    for recorded in training:
        operator = deserialize_component(recorded)
        if isinstance(operator, FrozenDagMlSplitStep):
            steps.append(deepcopy(operator.splitter))
        elif _is_split_step(operator):
            steps.append(deepcopy(operator))
    target = loaded["artifact"].get("y_transform")
    if target is not None:
        from nirs4all.api.general_transfer import fresh_training_estimator

        fresh_target = fresh_training_estimator(target)
        if fresh_target is not None:
            steps.append({"y_processing": fresh_target})
    steps.extend(preprocessing)
    steps.append({"model": clone(estimator)})
    return serialize_component(steps)
