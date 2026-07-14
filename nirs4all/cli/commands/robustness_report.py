"""CLI publication helpers for robustness reports."""

from __future__ import annotations

from pathlib import Path

from nirs4all.api.robustness import RobustnessReport


def robustness_report_export(args) -> None:
    """Load a verified robustness report artifact and publish it."""

    source = Path(args.input)
    report = RobustnessReport.load_artifacts(source) if source.is_dir() else RobustnessReport.load_json(source)
    output_format = str(args.format)

    if output_format == "json":
        payload = report.to_json(indent=None if args.compact else args.indent)
        if args.output is None:
            print(payload, end="")
            return
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        return

    if output_format == "summary":
        payload = report.to_summary_json(indent=None if args.compact else args.indent)
        if args.output is None:
            print(payload, end="")
            return
        report.save_summary(args.output)
        return

    if output_format == "markdown":
        payload = report.to_markdown()
        if args.output is None:
            print(payload, end="")
            return
        report.save_markdown(args.output)
        return

    if output_format == "html":
        payload = report.to_html()
        if args.output is None:
            print(payload, end="")
            return
        report.save_html(args.output)
        return

    if output_format == "artifacts":
        if args.output is None:
            raise ValueError("--output is required when --format=artifacts")
        report.save_artifacts(args.output)
        return

    if args.output is None:
        raise ValueError("--output is required when --format=parquet")
    report.save_parquet(args.output)


def add_robustness_report_command(subparsers) -> None:
    """Register the top-level robustness report publication command."""

    parser = subparsers.add_parser(
        "robustness-report",
        help="Publish a verified robustness report JSON artifact",
    )
    parser.add_argument(
        "input",
        help="Input robustness report JSON file or artifact directory",
    )
    parser.add_argument(
        "--format",
        choices=("json", "summary", "markdown", "html", "parquet", "artifacts"),
        default="markdown",
        help="Output artifact format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Write the artifact to this file or directory instead of stdout",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation when --format=json/summary and not compact (default: 2)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact JSON when --format=json",
    )
    parser.set_defaults(func=robustness_report_export)
