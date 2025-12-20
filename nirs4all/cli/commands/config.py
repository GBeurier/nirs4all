"""
Configuration validation CLI commands for nirs4all.

Provides commands for validating pipeline and dataset configuration files.
"""

import sys
from pathlib import Path

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


def config_validate(args):
    """Validate a configuration file."""
    from nirs4all.config.validator import (
        validate_config_file,
        get_validation_summary
    )

    config_path = args.config_file

    # Check file exists first
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # Determine config type
    config_type = None
    if args.type != 'auto':
        config_type = args.type

    # Validate
    is_valid, errors, warnings = validate_config_file(
        config_path,
        config_type=config_type,
        check_files=args.check_files,
        check_class_paths=args.check_imports
    )

    # Display results
    summary = get_validation_summary(is_valid, errors, warnings, config_path)
    print(summary)

    # Exit with appropriate code
    if is_valid:
        sys.exit(0)
    else:
        sys.exit(1)


def config_show_schema(args):
    """Display the JSON schema for a configuration type."""
    import json
    from nirs4all.config.validator import PIPELINE_SCHEMA, DATASET_SCHEMA

    if args.schema_type == 'pipeline':
        schema = PIPELINE_SCHEMA
    else:
        schema = DATASET_SCHEMA

    print(json.dumps(schema, indent=2))


def add_config_commands(subparsers):
    """Add config commands to CLI."""

    # Config command group
    config_parser = subparsers.add_parser(
        'config',
        help='Configuration file management commands'
    )
    config_subparsers = config_parser.add_subparsers(dest='config_command')

    # config validate
    validate_parser = config_subparsers.add_parser(
        'validate',
        help='Validate a configuration file'
    )
    validate_parser.add_argument(
        'config_file',
        type=str,
        help='Path to the configuration file (JSON or YAML)'
    )
    validate_parser.add_argument(
        '--type',
        type=str,
        choices=['pipeline', 'dataset', 'auto'],
        default='auto',
        help='Configuration type (default: auto-detect)'
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
        '--check-imports',
        action='store_true',
        default=False,
        help='Verify that class paths can be imported'
    )
    validate_parser.set_defaults(func=config_validate)

    # config schema
    schema_parser = config_subparsers.add_parser(
        'schema',
        help='Display JSON schema for configuration type'
    )
    schema_parser.add_argument(
        'schema_type',
        type=str,
        choices=['pipeline', 'dataset'],
        help='Type of schema to display'
    )
    schema_parser.set_defaults(func=config_show_schema)
