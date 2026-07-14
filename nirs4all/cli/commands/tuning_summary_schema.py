"""CLI export for the tuning summary JSON Schema."""

from __future__ import annotations

from pathlib import Path

from nirs4all import tuning_summary_schema_json


def tuning_summary_schema_export(args) -> None:
    """Emit the tuning summary JSON Schema as deterministic JSON."""

    payload = tuning_summary_schema_json(indent=None if args.compact else args.indent)
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        return
    print(payload, end="")


def add_tuning_summary_schema_command(subparsers) -> None:
    """Register the top-level tuning summary schema export command."""

    parser = subparsers.add_parser(
        "tuning-summary-schema",
        help="Export the JSON Schema for tuning summary artifacts",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Write the JSON Schema to this file instead of stdout",
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
    parser.set_defaults(func=tuning_summary_schema_export)
