"""Expand successive model checkpoints without fitting or resampling their folds."""

from __future__ import annotations

from typing import Any

from .public_normalization import normalize_model_steps
from .steps import DagMlSplitStep, FrozenDagMlSplitStep, _is_split_step, _split_pipeline


def sequential_model_pipelines(pipeline: Any) -> list[list[Any]] | None:
    """Retain the cumulative non-model prefix at each top-level model checkpoint.

    Models do not transform the input of the following checkpoint. In particular,
    a transform after a model must not be retroactively applied to that model.
    Branches and merges have their own execution semantics and are not expanded.
    """
    if not isinstance(pipeline, list):
        return None
    steps = normalize_model_steps(pipeline)
    if any(isinstance(step, dict) and any(key in step for key in ("branch", "merge", "exclude", "sample_augmentation")) for step in steps):
        return None
    if sum(isinstance(step, dict) and "model" in step for step in steps) < 2:
        return None
    prefix: list[Any] = []
    children = []
    for step in steps:
        if isinstance(step, dict) and "model" in step:
            children.append([*prefix, step])
        else:
            prefix.append(step)
    # A final chart still describes the final checkpoint, not an earlier model.
    last_model = max(index for index, step in enumerate(steps) if isinstance(step, dict) and "model" in step)
    children[-1].extend(steps[last_model + 1:])
    return children


def share_model_folds(pipelines: list[list[Any]], spectro: Any) -> list[list[Any]]:
    """Resolve a shared splitter once, retaining grouping metadata and row IDs."""
    from .folds import _build_folds, _build_group_folds, _is_repetition_dataset

    captured: dict[int, FrozenDagMlSplitStep] = {}
    result = []
    pool = [int(sample) for sample in spectro.index_column("sample", {"partition": "train"})]
    for pipeline in pipelines:
        split_steps = [step for step in pipeline if _is_split_step(step)]
        if not split_steps:
            result.append(pipeline)
            continue
        # More than one splitter remains the normal backend's validation concern.
        if len(split_steps) != 1:
            result.append(pipeline)
            continue
        original = split_steps[0]
        key = id(original)
        if key not in captured:
            _, splitter = _split_pipeline(pipeline)
            folds = _build_group_folds(splitter, spectro, pool) if _is_repetition_dataset(spectro) else _build_folds(splitter, spectro, pool, set())
            wrapper = splitter if isinstance(splitter, DagMlSplitStep) else DagMlSplitStep(splitter=splitter)
            # Preserve the original operator object, not dataclasses.asdict's
            # recursive deep copy of mutable/user-provided splitter state.
            fields = {name: getattr(wrapper, name) for name in DagMlSplitStep.__dataclass_fields__}
            captured[key] = FrozenDagMlSplitStep(
                **fields, sample_pool=tuple(pool),
                folds=tuple((tuple(train), tuple(validation)) for train, validation in folds),
            )
        result.append([captured[key] if step is original else step for step in pipeline])
    return result
