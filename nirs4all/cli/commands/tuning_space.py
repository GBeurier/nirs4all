"""CLI inspection for native tuning search-space contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nirs4all import inspect_tuning_space, tuning_space_schema_json


def tuning_space_inspect(args: Any) -> None:
    """Emit the canonical ordered tuning-space artifact as deterministic JSON."""

    if args.schema:
        if args.tuning is not None or args.input is not None:
            raise ValueError("--schema cannot be combined with --tuning or --input")
        output = tuning_space_schema_json(indent=None if args.compact else args.indent)
        if args.output is not None:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(output, encoding="utf-8")
            return
        print(output, end="")
        return
    if args.tuning is not None and args.input is not None:
        raise ValueError("provide either --tuning JSON or --input, not both")
    if args.tuning is None and args.input is None:
        raise ValueError("provide --tuning JSON, --input, or --schema")
    raw = Path(args.input).read_text(encoding="utf-8") if args.input is not None else args.tuning
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid tuning JSON: {exc.msg}") from exc
    artifact = inspect_tuning_space(payload)
    output = json.dumps(
        artifact,
        indent=None if args.compact else args.indent,
        sort_keys=True,
        separators=(",", ":") if args.compact else None,
    )
    if args.output is not None:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output + "\n", encoding="utf-8")
        return
    print(output)


def add_tuning_space_command(subparsers: Any) -> None:
    """Register the top-level tuning-space inspection command."""

    parser = subparsers.add_parser(
        "tuning-space",
        help="Inspect canonical ordered tuning.space patches",
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--tuning",
        help="Tuning JSON payload to inspect",
    )
    source.add_argument(
        "--input",
        "-i",
        help="Read the tuning JSON payload from this file",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Write the ordered search-space JSON to this file instead of stdout",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Emit the ordered search-space JSON Schema instead of inspecting a tuning payload",
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
    parser.set_defaults(func=tuning_space_inspect)
