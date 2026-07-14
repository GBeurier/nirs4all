"""CLI export for the machine-readable keyword/effect registry."""

from __future__ import annotations

from pathlib import Path

from nirs4all.pipeline.keyword_registry import keyword_registry_json, keyword_registry_schema_json


def keyword_registry_export(args) -> None:
    """Emit the keyword/effect registry as deterministic JSON."""

    indent = None if args.compact else args.indent
    payload = keyword_registry_schema_json(indent=indent) if args.schema else keyword_registry_json(indent=indent)
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        return
    print(payload, end="")


def add_keyword_registry_command(subparsers) -> None:
    """Register the top-level keyword registry export command."""

    parser = subparsers.add_parser(
        "keyword-registry",
        help="Export the machine-readable nirs4all keyword/effect registry",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Write the registry JSON to this file instead of stdout",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation to use when not compact (default: 2)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact single-line JSON",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Emit the registry JSON Schema instead of the registry document",
    )
    parser.set_defaults(func=keyword_registry_export)
