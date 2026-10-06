"""Common native HPO trial budgets and checkpoint resume CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _run(args: argparse.Namespace) -> None:
    from nirs4all.api.public_tuning import generate_variants, tune_native
    record = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.tuning_command == "generate":
        constraints = json.loads(Path(args.constraints).read_text()) if args.constraints else None
        result: dict[str, Any] = {"schema": "nirs4all.generation.v1", "variants": generate_variants(
            record, strategy=args.strategy, constraints=constraints, seed=args.seed, count=args.count,
            max_variants=args.max_variants, cli=args.cli)}
        result["seed_decimals"] = [str(variant["seed"]) for variant in result["variants"]]
    else:
        workflow = tune_native(record, trials=args.trials, seed=args.seed, sampler=args.sampler, metric=args.metric,
                               source_id=args.source_id, checkpoint=args.checkpoint, archive=args.archive,
                               methods_library_path=args.methods_library, cli=args.cli, run_id=args.run_id)
        result = workflow.outcome
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))


def add_tuning_commands(subparsers: Any) -> None:
    """Register native tuning run/resume and constrained variant generation."""
    parser = subparsers.add_parser("tuning", help="Native PLS HPO and variant generation")
    commands = parser.add_subparsers(dest="tuning_command", required=True)
    for name in ("run", "resume", "generate"):
        command = commands.add_parser(name)
        command.add_argument("--input", required=True)
        command.add_argument("--cli")
        command.add_argument("--seed", type=int, default=0 if name == "generate" else 91)
        if name == "generate":
            command.add_argument("--strategy", choices=("cartesian", "zip", "random"), default="cartesian")
            command.add_argument("--constraints")
            command.add_argument("--count", type=int)
            command.add_argument("--max-variants", type=int, default=10000)
        else:
            command.add_argument("--archive", required=True)
            command.add_argument("--methods-library", required=True)
            command.add_argument("--source-id", default="spectra")
            command.add_argument("--trials", type=int, required=True)
            command.add_argument("--sampler", choices=("random", "tpe", "sobol", "lhs"), default="random")
            command.add_argument("--metric", choices=("rmse", "mae", "r2"), default="rmse")
            command.add_argument("--checkpoint", required=name == "resume")
            command.add_argument("--run-id")
        command.set_defaults(func=_run)
