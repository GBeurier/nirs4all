"""Expand full-training candidates without manufacturing cross-validation evidence."""

from __future__ import annotations

from typing import Any

from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs

from .steps import _flatten_steps, _wrap_param_keyed_operator


def expand_full_train_variants(pipeline: list[Any], name: str = "") -> list[tuple[list[Any], str]]:
    """Return concrete step lists and their standard ``PipelineConfigs`` names.

    Reuse the public generator grammar, expansion limit and naming contract.
    Deserialization precedes caller-side splitter detection: canonical Studio
    definitions may contain splitters as class paths, including inside choices.
    Each candidate must execute as its own real DAG run, retaining its fitted
    artifacts. This helper neither fits nor selects a candidate and produces no
    scores or synthetic folds.
    """
    serialized = serialize_component(PipelineConfigs._preprocess_steps(pipeline))
    normalized = [_wrap_param_keyed_operator(step) for step in serialized]
    configs = PipelineConfigs(normalized, name=name)
    return [
        (_flatten_steps(deserialize_component(steps)), variant_name)
        for steps, variant_name in zip(configs.steps, configs.names, strict=True)
    ]
