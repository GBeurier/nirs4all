"""Query and export a checked native experiment without running training."""

from __future__ import annotations

import argparse
import json
from typing import Any


def _query(args: argparse.Namespace) -> None:
    from nirs4all.api.public_results import open_experiment

    experiment = open_experiment(args.directory)
    if args.results_command == "inspect":
        value = experiment.summary()
    elif args.results_command == "compare":
        value = experiment.compare(variant_id=args.variant, partition=args.partition)
    elif args.results_command == "predictions":
        value = experiment.predictions(variant_id=args.variant, partition=args.partition, fold_id=args.fold)
    else:
        value = {"path": str(experiment.export(args.output, kind=args.kind, variant_id=args.variant,
                                             partition=args.partition, fold_id=args.fold))}
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def add_results_commands(subparsers: Any) -> None:
    """Register `results inspect|compare|predictions|export`."""
    command = subparsers.add_parser("results", help="Inspect saved native experiment results")
    queries = command.add_subparsers(dest="results_command", required=True)
    for name in ("inspect", "compare", "predictions", "export"):
        parser = queries.add_parser(name)
        parser.add_argument("directory")
        if name != "inspect":
            parser.add_argument("--variant")
            parser.add_argument("--partition")
        if name in ("predictions", "export"):
            parser.add_argument("--fold")
        if name == "export":
            parser.add_argument("--kind", choices=("scores", "predictions"), default="predictions")
            parser.add_argument("--output", required=True)
        parser.set_defaults(func=_query)
