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
    pipeline: list[Any], branch_body: list[Any], *, source_widths: list[int], source_names: list[str],
) -> tuple[list[Any], list[list[Any]], dict[str, Any]]:
    """Build named, source-sliced branches with a fingerprinted feature layout."""
    if len(source_widths) < 2 or len(source_names) != len(source_widths):
        raise ValueError("source stacking requires at least two aligned source widths and names")
    if any(type(width) is not int or width <= 0 for width in source_widths):
        raise ValueError("source stacking widths must be positive integers")
    if any(not isinstance(name, str) or not name for name in source_names):
        raise ValueError("source stacking names must be non-empty strings")
    source_steps = [index for index, step in enumerate(pipeline)
                    if isinstance(step, dict) and isinstance(step.get("branch"), dict) and step["branch"].get("by_source") is True]
    if len(source_steps) != 1:
        raise ValueError("source stacking requires exactly one by_source branch")
    splitters = [step for step in pipeline if _is_split_step(step)]
    if len(splitters) != 1:
        raise ValueError("source stacking requires one explicit outer splitter")
    if not branch_body or not isinstance(branch_body[-1], dict) or "model" not in branch_body[-1]:
        raise ValueError("each source stacking branch must end in a model step")

    branches = []
    sources = []
    start = 0
    for index, (width, name) in enumerate(zip(source_widths, source_names, strict=True)):
        columns = list(range(start, start + width))
        selector = ColumnTransformer([("source", "passthrough", columns)], remainder="drop", sparse_threshold=0)
        branch = [selector, *copy.deepcopy(branch_body)]
        # A grouped HPO study must clone the real outer split policy inside its
        # own training universe, including group constraints, not use a default.
        branch = attach_host_finetune_splitter([splitters[0], *branch])[1:]
        branches.append(branch)
        sources.append({"source_index": index, "source_name": name, "column_start": start, "column_count": width})
        start += width
    layout: dict[str, Any] = {"schema": "nirs4all.source-stacking-layout.v1", "sources": sources, "total_columns": start}
    layout["fingerprint"] = hashlib.sha256(json.dumps(layout, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    lowered = copy.deepcopy(pipeline)
    # Labels include the physical index even when two inputs share a name.
    lowered[source_steps[0]] = {"branch": {f"source_{index}": branch for index, branch in enumerate(branches)}}
    return lowered, branches, layout
