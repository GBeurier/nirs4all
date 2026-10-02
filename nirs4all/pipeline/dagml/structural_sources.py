"""Validate declared dense source subsets and lower an existing sklearn selector."""

from __future__ import annotations

import re
from typing import Any

from sklearn.compose import ColumnTransformer

from nirs4all.pipeline.dagml_bridge import _qualname, _strict_json_safe

DENSE_CONCAT_MARKER = "nirs4all_structural_dense_concat"
SOURCE_SELECTION_METADATA = "nirs4all_structural_source_selection"


def validate_source_selection_alternatives(alternatives: Any, *, source_widths: list[int] | None = None) -> list[list[int]]:
    """Resolve the existing source-index aliases without enumerating recipes."""
    if not isinstance(alternatives, list) or not alternatives:
        raise ValueError("structural source alternatives must be a nonempty list of source-concat merge declarations")
    if source_widths is not None and (len(source_widths) < 2 or any(type(width) is not int or width < 1 for width in source_widths)):
        raise ValueError("structural source selection requires at least two positive dense source widths")
    selections: list[list[int]] = []
    for alternative in alternatives:
        if not isinstance(alternative, dict) or set(alternative) != {"merge"}:
            raise ValueError("structural source alternatives require merge={'sources': {'strategy': 'concat', 'sources': [...]}}")
        merge = alternative["merge"]
        if not isinstance(merge, dict) or set(merge) != {"sources"}:
            raise ValueError("structural source merge accepts only a sources declaration")
        config = merge["sources"]
        if not isinstance(config, dict) or set(config) != {"strategy", "sources"} or config["strategy"] != "concat":
            raise ValueError("structural source selection requires strategy='concat' and an explicit source list")
        sources = config["sources"]
        if not isinstance(sources, list) or not sources:
            raise ValueError("structural source selections must be nonempty lists")
        indices: list[int] = []
        for source in sources:
            if type(source) is int:
                index = source
            elif isinstance(source, str) and re.fullmatch(r"source_[0-9]+", source):
                index = int(source.removeprefix("source_"))
            else:
                raise ValueError("structural source selections support integer indices or source_<index> aliases")
            if index < 0 or (source_widths is not None and index >= len(source_widths)):
                raise ValueError("structural source index is outside the declared input layout")
            if index in indices:
                raise ValueError("structural source selections cannot repeat a source")
            indices.append(index)
        selections.append(indices)
    return selections


def build_source_selection_step(alternative: Any, branch_id: str, source_layout: dict[str, Any]) -> dict[str, Any]:
    """Lower one declared selection with concrete columns and signed input layout."""
    if not isinstance(source_layout, dict) or source_layout.get("kind") != "by_source_concat":
        raise ValueError("structural source selection requires the complete dense source-concat layout")
    blocks = source_layout.get("blocks")
    source_ids, source_order = source_layout.get("source_ids"), source_layout.get("source_order")
    if (not isinstance(blocks, list) or not isinstance(source_ids, list) or not isinstance(source_order, list)
            or len(blocks) != len(source_ids) or len(blocks) != len(source_order)
            or any(not isinstance(name, str) or not name for name in [*source_ids, *source_order])
            or len(set(source_ids)) != len(source_ids) or len(set(source_order)) != len(source_order)):
        raise ValueError("structural input source IDs, names and blocks must form one ordered complete layout")
    widths: list[int] = []
    input_width = 0
    for index, block in enumerate(blocks):
        if (not isinstance(block, dict) or type(block.get("source_index")) is not int or block["source_index"] != index
                or block.get("source_id") != source_ids[index] or block.get("source_name") != source_order[index]
                or type(block.get("column_start")) is not int or block["column_start"] != input_width
                or type(block.get("column_count")) is not int or block["column_count"] < 1):
            raise ValueError("structural input source columns must be positive, contiguous and match the ordered source IDs")
        widths.append(block["column_count"])
        input_width += block["column_count"]
    indices = validate_source_selection_alternatives([alternative], source_widths=widths)[0]
    columns = [column for index in indices for column in range(blocks[index]["column_start"], blocks[index]["column_start"] + widths[index])]
    selector = ColumnTransformer([("selected", "passthrough", columns)], remainder="drop", sparse_threshold=0)
    # Deep params add an unprefixed ``selected`` key that is not a constructor
    # argument. The actual constructor contains only passthrough strings and
    # explicit integer columns; strict JSON coercion preserves their order.
    params = _strict_json_safe(selector.get_params(deep=False), "structural source selector")
    selection = {
        "schema": "nirs4all.structural-source-selection.v1",
        "source_indices": indices,
        "source_ids": [source_ids[index] for index in indices],
        "source_order": [source_order[index] for index in indices],
        "input_source_widths": widths,
        "input_width": input_width,
        "selected_width": len(columns),
        "columns": columns,
        "input_source_layout": source_layout,
    }
    return {
        "kind": "transform", "id": f"t:{branch_id}", "operator": {"class": _qualname(selector)}, "params": params,
        "metadata": {DENSE_CONCAT_MARKER: True, SOURCE_SELECTION_METADATA: _strict_json_safe(selection, "structural source layout")},
    }
