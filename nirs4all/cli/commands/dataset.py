"""
Dataset CLI commands for nirs4all.

Provides commands for dataset validation, inspection, and diagnostics.

Phase 8 Implementation - Dataset Configuration Roadmap
Section 8.4: Error Handling & Diagnostics - Validation CLI
"""

import json
import sys
from pathlib import Path
from typing import Optional

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


def dataset_validate(args):
    """Validate a dataset configuration file."""
    from nirs4all.data.schema.validation import (
        ConfigValidator,
        DiagnosticReport,
        DiagnosticBuilder,
        ErrorRegistry,
    )
    from nirs4all.data.parsers.normalizer import ConfigNormalizer

    config_path = args.config_file
    verbose = args.verbose
    check_files = args.check_files
    output_format = args.format

    # Check file exists first
    if not Path(config_path).exists():
        if output_format == "json":
            print(json.dumps({"error": f"File not found: {config_path}", "is_valid": False}))
        else:
            print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    # Create diagnostic report
    report = DiagnosticReport(config_path=config_path)
    builder = DiagnosticBuilder()

    try:
        # Load and normalize config
        normalizer = ConfigNormalizer()
        config, dataset_name = normalizer.normalize(config_path)

        if config is None:
            report.add(builder.create(
                ErrorRegistry.E100,
                details="Could not parse configuration file"
            ))
        else:
            # Validate configuration
            validator = ConfigValidator(check_file_existence=check_files)
            result = validator.validate(config)

            # Convert validation errors/warnings to diagnostic messages
            for error in result.errors:
                report.add(DiagnosticBuilder().create(
                    ErrorRegistry.E100,
                    details=str(error),
                    location=config_path
                ))

            for warning in result.warnings:
                report.add(DiagnosticBuilder().create(
                    ErrorRegistry.E204,
                    path=config_path,
                    encoding="",
                    location=config_path
                ))

            if verbose and config:
                # Add info about detected configuration
                print(f"\nDataset name: {dataset_name}")
                if "train_x" in config:
                    print(f"Training features: {config['train_x']}")
                if "test_x" in config:
                    print(f"Test features: {config['test_x']}")
                if "task_type" in config:
                    print(f"Task type: {config['task_type']}")

    except FileNotFoundError as e:
        report.add(builder.file_not_found(str(e)))
    except ValueError as e:
        report.add(builder.create(
            ErrorRegistry.E100,
            details=str(e),
            location=config_path
        ))
    except Exception as e:
        report.add(builder.create(
            ErrorRegistry.E900,
            error=str(e),
            location=config_path
        ))
        if verbose:
            import traceback
            traceback.print_exc()

    # Output results
    if output_format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report)

    # Exit with appropriate code
    sys.exit(0 if report.is_valid else 1)


def dataset_inspect(args):
    """Inspect a dataset configuration and show details."""
    from nirs4all.data.parsers.normalizer import ConfigNormalizer
    from nirs4all.data.detection import detect_file_parameters

    config_path = args.config_file
    detect_params = args.detect

    # Check file exists
    if not Path(config_path).exists():
        print(f"Error: File not found: {config_path}")
        sys.exit(1)

    # Load configuration
    normalizer = ConfigNormalizer()
    config, dataset_name = normalizer.normalize(config_path)

    if config is None:
        print(f"Error: Could not load configuration from {config_path}")
        sys.exit(1)

    print(f"Dataset Configuration: {config_path}")
    print("=" * 60)
    print(f"\nDataset Name: {dataset_name}")

    # Show data sources
    print("\nData Sources:")
    for key in ["train_x", "train_y", "train_group", "test_x", "test_y", "test_group"]:
        if key in config and config[key]:
            value = config[key]
            if isinstance(value, list):
                print(f"  {key}:")
                for i, v in enumerate(value):
                    print(f"    [{i}] {v}")
            else:
                print(f"  {key}: {value}")

    # Show parameters
    if "global_params" in config and config["global_params"]:
        print("\nGlobal Parameters:")
        for key, value in config["global_params"].items():
            print(f"  {key}: {value}")

    # Task and aggregation
    if "task_type" in config:
        print(f"\nTask Type: {config['task_type']}")
    if "aggregate" in config:
        print(f"Aggregation: {config['aggregate']}")
        if "aggregate_method" in config:
            print(f"Aggregation Method: {config['aggregate_method']}")

    # Auto-detect file parameters if requested
    if detect_params:
        print("\nAuto-detected Parameters:")
        train_x = config.get("train_x")
        if train_x:
            if isinstance(train_x, list):
                train_x = train_x[0]
            if isinstance(train_x, str) and Path(train_x).exists():
                result = detect_file_parameters(train_x)
                print(f"  Delimiter: '{result.delimiter}' (confidence: {result.confidence.get('delimiter', 0):.2f})")
                print(f"  Decimal separator: '{result.decimal_separator}' (confidence: {result.confidence.get('decimal_separator', 0):.2f})")
                print(f"  Has header: {result.has_header} (confidence: {result.confidence.get('has_header', 0):.2f})")
                print(f"  Header unit: {result.header_unit} (confidence: {result.confidence.get('header_unit', 0):.2f})")
                if result.signal_type:
                    print(f"  Signal type: {result.signal_type} (confidence: {result.confidence.get('signal_type', 0):.2f})")
                if result.warnings:
                    print("  Warnings:")
                    for warning in result.warnings:
                        print(f"    - {warning}")


def dataset_export(args):
    """Export dataset configuration to normalized format."""
    from nirs4all.data.parsers.normalizer import ConfigNormalizer
    from nirs4all.data.serialization import ConfigSerializer, SerializationFormat

    config_path = args.config_file
    output_path = args.output
    output_format = args.format

    # Load configuration
    normalizer = ConfigNormalizer()
    config, dataset_name = normalizer.normalize(config_path)

    if config is None:
        print(f"Error: Could not load configuration from {config_path}")
        sys.exit(1)

    # Serialize
    serializer = ConfigSerializer(normalize=True)
    fmt = SerializationFormat.YAML if output_format == "yaml" else SerializationFormat.JSON

    if output_path:
        serializer.save(config, output_path, format=fmt)
        print(f"Exported to: {output_path}")
    else:
        if fmt == SerializationFormat.YAML:
            print(serializer.to_yaml(config))
        else:
            print(serializer.to_json(config))


def dataset_diff(args):
    """Compare two dataset configurations."""
    from nirs4all.data.parsers.normalizer import ConfigNormalizer
    from nirs4all.data.serialization import diff_configs

    config1_path = args.config1
    config2_path = args.config2
    output_format = args.format

    # Load configurations
    normalizer = ConfigNormalizer()
    config1, name1 = normalizer.normalize(config1_path)
    config2, name2 = normalizer.normalize(config2_path)

    if config1 is None:
        print(f"Error: Could not load configuration from {config1_path}")
        sys.exit(1)
    if config2 is None:
        print(f"Error: Could not load configuration from {config2_path}")
        sys.exit(1)

    # Diff
    diff = diff_configs(config1, config2)

    if output_format == "json":
        result = {
            "config1": config1_path,
            "config2": config2_path,
            "is_identical": diff.is_identical(),
            "added": list(diff.added.keys()),
            "removed": list(diff.removed.keys()),
            "changed": list(diff.changed.keys()),
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"Comparing: {config1_path} vs {config2_path}")
        print("=" * 60)
        print(diff)


def add_dataset_commands(subparsers):
    """Add dataset commands to CLI."""

    # Dataset command group
    dataset_parser = subparsers.add_parser(
        'dataset',
        help='Dataset configuration commands'
    )
    dataset_subparsers = dataset_parser.add_subparsers(dest='dataset_command')

    # dataset validate
    validate_parser = dataset_subparsers.add_parser(
        'validate',
        help='Validate a dataset configuration file'
    )
    validate_parser.add_argument(
        'config_file',
        type=str,
        help='Path to the dataset configuration file (JSON, YAML, or folder path)'
    )
    validate_parser.add_argument(
        '--check-files',
        action='store_true',
        default=True,
        help='Check that referenced data files exist (default: true)'
    )
    validate_parser.add_argument(
        '--no-check-files',
        action='store_false',
        dest='check_files',
        help='Do not check if referenced data files exist'
    )
    validate_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed validation information'
    )
    validate_parser.add_argument(
        '-f', '--format',
        type=str,
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    validate_parser.set_defaults(func=dataset_validate)

    # dataset inspect
    inspect_parser = dataset_subparsers.add_parser(
        'inspect',
        help='Inspect a dataset configuration'
    )
    inspect_parser.add_argument(
        'config_file',
        type=str,
        help='Path to the dataset configuration file'
    )
    inspect_parser.add_argument(
        '--detect',
        action='store_true',
        help='Auto-detect file parameters (delimiter, encoding, etc.)'
    )
    inspect_parser.set_defaults(func=dataset_inspect)

    # dataset export
    export_parser = dataset_subparsers.add_parser(
        'export',
        help='Export dataset configuration to normalized format'
    )
    export_parser.add_argument(
        'config_file',
        type=str,
        help='Path to the dataset configuration file'
    )
    export_parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path (prints to stdout if not specified)'
    )
    export_parser.add_argument(
        '-f', '--format',
        type=str,
        choices=['yaml', 'json'],
        default='yaml',
        help='Output format (default: yaml)'
    )
    export_parser.set_defaults(func=dataset_export)

    # dataset diff
    diff_parser = dataset_subparsers.add_parser(
        'diff',
        help='Compare two dataset configurations'
    )
    diff_parser.add_argument(
        'config1',
        type=str,
        help='Path to first configuration file'
    )
    diff_parser.add_argument(
        'config2',
        type=str,
        help='Path to second configuration file'
    )
    diff_parser.add_argument(
        '-f', '--format',
        type=str,
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    diff_parser.set_defaults(func=dataset_diff)
