"""Public native workflows without caller-authored DAG manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _execute_value(args: argparse.Namespace) -> Any:
    import nirs4all_core as core

    operation = args.workflow_command
    if operation in ("run", "retrain"):
        options = {"archive": args.archive, "methods_library_path": args.methods_library,
                   "run_id": args.run_id, "results_directory": args.results_directory, "cli": args.native_cli}
        if operation == "run":
            options.update(source_id=args.source, components=args.components)
            result = core.run(core.dataset(Path(args.dataset)), **options)
        else:
            result = core.retrain(core.load(args.model, cli=args.native_cli), core.dataset(Path(args.dataset)), **options)
        value = result.outcome
    elif operation == "predict":
        with Path(args.input).open(encoding="utf-8") as stream:
            record = json.load(stream)
        if not isinstance(record, dict) or set(record) != {"x", "sample_ids"}:
            raise ValueError("prediction input requires exactly x and sample_ids")
        value = core.predict(Path(args.archive), record["x"], sample_ids=record["sample_ids"],
                             methods_library_path=args.methods_library, cli=args.native_cli)
    else:
        result = core.load(args.model, cli=args.native_cli)
        value = ({"path": str(core.export(result, args.destination))} if operation == "export" else result.outcome)
    return value


def _execute(args: argparse.Namespace) -> None:
    if not args.output:
        print(json.dumps(_execute_value(args), ensure_ascii=False, allow_nan=False))
        return
    output = Path(args.output).resolve()
    destinations: list[Path] = []
    if args.workflow_command in ("run", "retrain"):
        archive = Path(args.archive).resolve()
        destinations.extend((archive, Path(args.results_directory).resolve()
                             if args.results_directory else Path(str(archive) + ".results")))
    elif args.workflow_command == "export":
        destinations.append(Path(args.destination).resolve())
    if any(output == path or output in path.parents or path in output.parents for path in destinations):
        raise ValueError("JSON output overlaps a workflow publication destination")
    # Reserve before loading data or fitting: failed or racing destinations must
    # not leave a costly trained archive behind. Exclusive open preserves any
    # existing output, including a dangling symlink at the caller's path.
    requested = Path(args.output)
    stream = requested.open("x", encoding="utf-8")
    try:
        with stream:
            stream.write(json.dumps(_execute_value(args), ensure_ascii=False, allow_nan=False))
    except BaseException:
        requested.unlink(missing_ok=True)
        raise


def add_workflow_commands(subparsers: Any) -> None:
    """Register the bounded native workflow cycle."""
    command = subparsers.add_parser("workflow", help="Run, replay and persist a native regression workflow")
    commands = command.add_subparsers(dest="workflow_command", required=True)
    for operation in ("run", "predict", "retrain", "export", "load"):
        parser = commands.add_parser(operation)
        parser.add_argument("--native-cli", help="Core native CLI path (or NIRS4ALL_CORE_CLI)")
        parser.add_argument("--output", help="New JSON output file; defaults to stdout")
        if operation in ("run", "retrain"):
            parser.add_argument("dataset", help="IO JSON/YAML dataset declaration")
            parser.add_argument("--archive", required=True, help="New Archive V2 destination")
            parser.add_argument("--results-directory")
            parser.add_argument("--run-id")
        if operation in ("run", "predict", "retrain"):
            parser.add_argument("--methods-library")
        if operation == "run":
            parser.add_argument("--source", default="spectra")
            parser.add_argument("--components", nargs="+", type=int, default=[1, 2])
        if operation == "predict":
            parser.add_argument("--archive", required=True)
            parser.add_argument("--input", required=True, help="Target-free JSON with x and sample_ids")
        if operation in ("retrain", "export", "load"):
            parser.add_argument("model", help="Exported workflow directory")
        if operation == "export":
            parser.add_argument("destination")
        parser.set_defaults(func=_execute)
