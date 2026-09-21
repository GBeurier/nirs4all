"""Lower documented per-source prediction stacking to native nested branches.

This phase only declares source slices and preserves scientific operators.
It does not read matrices, fit transformations, choose folds or execute models.
The historical source-layout concatenation masquerading as prediction stacking
is deliberately not reproduced by this lowering.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from sklearn.compose import ColumnTransformer

from .host_finetune import attach_host_finetune_splitter
from .steps import _is_split_step


def lower_source_stacking(
    pipeline: list[Any], branch_body: list[Any] | dict[str, list[Any]], *, source_widths: list[int], source_names: list[str],
    source_descriptors: list[dict[str, Any]] | None = None,
) -> tuple[list[Any], list[list[Any]], dict[str, Any]]:
    """Declare source-bound branches; raw tensors stay intact until encoding."""
    if len(source_widths) < 2 or len(source_names) != len(source_widths):
        raise ValueError("source stacking requires at least two aligned source widths and names")
    if any(type(width) is not int or width <= 0 for width in source_widths):
        raise ValueError("source stacking widths must be positive integers")
    if any(not isinstance(name, str) or not name for name in source_names):
        raise ValueError("source stacking names must be non-empty strings")
    source_steps = [index for index, step in enumerate(pipeline)
                    if isinstance(step, dict) and isinstance(step.get("branch"), dict) and step["branch"].get("by_source") in (True, "auto")]
    if len(source_steps) != 1:
        raise ValueError("source stacking requires exactly one by_source branch")
    splitters = [step for step in pipeline if _is_split_step(step)]
    if len(splitters) != 1:
        raise ValueError("source stacking requires one explicit outer splitter")
    if isinstance(branch_body, dict) and (len(set(source_names)) != len(source_names) or set(branch_body) != set(source_names)):
        raise ValueError("source stacking branch names must exactly match the dataset source names")
    bodies = [branch_body[name] for name in source_names] if isinstance(branch_body, dict) else [branch_body] * len(source_names)
    if any(not isinstance(body, list) or not body or not isinstance(body[-1], dict) or "model" not in body[-1] for body in bodies):
        raise ValueError("each source stacking branch must end in a model step")
    if source_descriptors is not None and [descriptor.get("source_id") for descriptor in source_descriptors] != source_names:
        raise ValueError("raw source descriptors must match source names in dataset order")

    branches = []
    sources = []
    start = 0
    for index, (width, name) in enumerate(zip(source_widths, source_names, strict=True)):
        branch = copy.deepcopy(bodies[index])
        if source_descriptors is None:
            columns = list(range(start, start + width))
            selector = ColumnTransformer([("source", "passthrough", columns)], remainder="drop", sparse_threshold=0)
            branch = [selector, *branch]
        # A grouped HPO study must clone the real outer split policy inside its
        # own training universe, including group constraints, not use a default.
        branch = attach_host_finetune_splitter([splitters[0], *branch])[1:]
        branches.append(branch)
        if source_descriptors is None:
            sources.append({"source_index": index, "source_name": name, "column_start": start, "column_count": width})
        else:
            sources.append({"source_index": index, "source_name": name, "input_schema": copy.deepcopy(source_descriptors[index])})
        start += width
    layout: dict[str, Any] = {"schema": "nirs4all.source-stacking-layout.v1", "sources": sources, "total_columns": start}
    if source_descriptors is not None:
        layout = {"schema": "nirs4all.source-stacking-layout.v2", "kind": "typed_source_blocks", "sources": sources}
    layout["fingerprint"] = hashlib.sha256(json.dumps(layout, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    lowered = copy.deepcopy(pipeline)
    # Labels include the physical index even when two inputs share a name.
    labels = source_names if source_descriptors is not None else [f"source_{index}" for index in range(len(branches))]
    lowered[source_steps[0]] = {"branch": dict(zip(labels, branches, strict=True))}
    return lowered, branches, layout
