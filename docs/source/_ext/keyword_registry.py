"""Sphinx directive rendering the nirs4all lifecycle keyword registry."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective

from nirs4all import (
    get_keyword_registry,
    keyword_registry_json,
    keyword_registry_schema_json,
    robustness_summary_schema_json,
    tuning_summary_schema_json,
)
from nirs4all.pipeline.keyword_registry import AliasSpec


def _text_cell(text: str, *, literal: bool = False) -> nodes.entry:
    entry = nodes.entry()
    paragraph = nodes.paragraph()
    paragraph += nodes.literal(text=text) if literal else nodes.Text(text)
    entry += paragraph
    return entry


def _join(values: list[str]) -> str:
    return ", ".join(value.replace("_", " ") for value in values) if values else "—"


def _format_aliases(aliases: Sequence[AliasSpec]) -> str:
    if not aliases:
        return "—"
    return "; ".join(f"{alias['name']} → {alias['canonical']} ({alias['mode'].replace('_', ' ')})" for alias in aliases)


def _format_engines(engine_support: Mapping[str, str]) -> str:
    return "; ".join(f"{engine}: {support.replace('_', ' ')}" for engine, support in engine_support.items())


def _format_schema(value_schema: Mapping[str, Any]) -> str:
    return json.dumps(value_schema, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class KeywordEffectsDirective(SphinxDirective):
    """Render all lifecycle-v1 entries as a documentation table."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        registry = get_keyword_registry()
        entries = registry["entries"]

        table = nodes.table(classes=["keyword-effects", "colwidths-auto"])
        table["ids"].append("keyword-effects-registry-v1")
        title = nodes.title(text=f"Lifecycle keyword registry v{registry['registry_version']}")
        table += title

        tgroup = nodes.tgroup(cols=10)
        table += tgroup
        for width in (15, 7, 9, 13, 14, 8, 10, 10, 15, 14):
            tgroup += nodes.colspec(colwidth=width)

        header = nodes.row()
        for label in (
            "Path",
            "Status",
            "Scope",
            "Value schema",
            "Meaning",
            "Reads",
            "Changes",
            "Calibration",
            "Engine support",
            "Read-only aliases",
        ):
            header += _text_cell(label)
        thead = nodes.thead()
        thead += header
        tgroup += thead

        tbody = nodes.tbody()
        for item in entries:
            row = nodes.row()
            row += _text_cell(item["path"], literal=True)
            row += _text_cell(item["status"])
            row += _text_cell(item["scope"].replace("_", " "))
            row += _text_cell(_format_schema(item["value_schema"]), literal=True)
            row += _text_cell(item["summary"])
            row += _text_cell(_join(item["reads"]))
            row += _text_cell(_join(item["changes"]))
            row += _text_cell(item["invalidates_calibration"].replace("_", " "))
            row += _text_cell(_format_engines(item["engine_support"]))
            row += _text_cell(_format_aliases(item["aliases"]))
            tbody += row
        tgroup += tbody

        return [table]


def write_keyword_registry_static_artifact(app: Sphinx, exception: Exception | None) -> None:
    """Publish the keyword/effect registry as a static docs JSON artifact."""

    if exception is not None:
        return
    static_dir = Path(app.outdir) / "_static"
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "keyword-registry.json").write_text(keyword_registry_json(indent=2), encoding="utf-8")
    (static_dir / "keyword-registry.schema.json").write_text(keyword_registry_schema_json(indent=2), encoding="utf-8")
    (static_dir / "robustness-summary.schema.json").write_text(robustness_summary_schema_json(indent=2), encoding="utf-8")
    (static_dir / "tuning-summary.schema.json").write_text(tuning_summary_schema_json(indent=2), encoding="utf-8")


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the directive with Sphinx."""

    app.add_directive("keyword-effects", KeywordEffectsDirective)
    app.connect("build-finished", write_keyword_registry_static_artifact)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
