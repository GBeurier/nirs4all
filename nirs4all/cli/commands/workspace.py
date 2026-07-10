"""
Workspace management CLI commands for nirs4all.

Provides commands for workspace initialization, run management, and
library operations.
"""

import sys
from pathlib import Path

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

def _validate_workspace_exists(workspace_path: Path) -> None:
    """Validate that a workspace path exists, exit with code 1 if not."""
    if not workspace_path.exists():
        logger.error(f"Workspace path does not exist: {workspace_path}")
        sys.exit(1)


def workspace_init(args):
    """Initialize a new workspace."""
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.path)

    # Validate parent directory exists and path is not a file
    if not workspace_path.parent.exists():
        logger.error(f"Parent directory does not exist: {workspace_path.parent}")
        sys.exit(1)
    if workspace_path.exists() and workspace_path.is_file():
        logger.error(f"Path exists and is a file, not a directory: {workspace_path}")
        sys.exit(1)

    # WorkspaceStore creates the SQLite database and workspace directories
    store = WorkspaceStore(workspace_path)
    store.close()

    # Also create standard directories
    (workspace_path / "exports").mkdir(parents=True, exist_ok=True)
    (workspace_path / "library").mkdir(parents=True, exist_ok=True)

    logger.success(f"Workspace initialized at: {workspace_path}")
    logger.info("  Created:")
    logger.info("    - store.sqlite (workspace database)")
    logger.info("    - exports/")
    logger.info("    - library/")

def workspace_list_runs(args):
    """List all runs in workspace."""
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)
    with WorkspaceStore(workspace_path) as store:
        runs = store.list_runs()

    if runs.height == 0:
        logger.info("No runs found in workspace.")
        return

    logger.info(f"Found {runs.height} run(s):\n")
    for row in runs.to_dicts():
        logger.info(f"  {row.get('name', 'unknown')}")
        logger.info(f"    Status: {row.get('status', 'unknown')}")
        logger.info(f"    Created: {row.get('created_at', 'unknown')}")
        logger.info("")

def workspace_query_best(args):
    """Query best predictions from workspace store."""
    from nirs4all.core.metrics import infer_ascending
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)

    try:
        ascending = True if args.ascending else infer_ascending(args.metric)
        with WorkspaceStore(workspace_path) as store:
            top_df = store.top_predictions(
                n=args.n,
                dataset_name=args.dataset,
                metric=args.metric,
                ascending=ascending,
            )
        if top_df.height == 0:
            logger.info("No predictions found matching criteria.")
            return

        logger.info(f"Top {args.n} predictions by {args.metric}:")
        logger.info(f"{'=' * 80}\n")

        df = top_df.to_pandas()
        logger.info(df.to_string(index=False))
    except Exception as e:
        logger.error(f"Error querying predictions: {e}")
        sys.exit(1)

def workspace_query_filter(args):
    """Filter predictions by criteria."""
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)

    try:
        with WorkspaceStore(workspace_path) as store:
            filtered = store.query_predictions(
                dataset_name=args.dataset,
            )

        logger.info(f"Found {filtered.height} predictions matching criteria\n")

        if filtered.height > 0:
            df = filtered.to_pandas()
            logger.info(df.to_string(index=False))
    except Exception as e:
        logger.error(f"Error filtering predictions: {e}")
        sys.exit(1)

def workspace_stats(args):
    """Show workspace statistics."""
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)

    logger.info("Workspace Statistics")
    logger.info(f"{'=' * 60}\n")

    try:
        with WorkspaceStore(workspace_path) as store:
            all_preds = store.query_predictions()
            runs = store.list_runs()

        logger.info(f"Total predictions: {all_preds.height}")

        if all_preds.height > 0 and "dataset_name" in all_preds.columns:
            datasets = all_preds["dataset_name"].unique().to_list()
            logger.info(f"Datasets: {len(datasets)}")
            for ds in datasets:
                count = all_preds.filter(all_preds["dataset_name"] == ds).height
                logger.info(f"  - {ds}: {count} predictions")

        logger.info(f"\nRuns: {runs.height}")
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        sys.exit(1)

def workspace_list_library(args):
    """List items in library."""
    from nirs4all.pipeline.storage.library import PipelineLibrary

    workspace_path = Path(args.workspace)
    _validate_workspace_exists(workspace_path)
    library = PipelineLibrary(workspace_path)

    templates = library.list_templates()
    logger.info(f"Templates: {len(templates)}")
    for t in templates:
        logger.info(f"  - {t['name']}: {t.get('description', 'No description')}")

def workspace_inspect(args):
    """Inspect workspace format without opening it writable."""
    from nirs4all.workspace.compat import inspect_workspace_format

    info = inspect_workspace_format(Path(args.path))
    logger.info(f"Path: {info.path}")
    logger.info(f"Format: {info.format}")
    logger.info(f"Conversion required: {info.conversion_required}")
    logger.info(info.message)
    if info.conversion_command:
        logger.info(f"Conversion command: {info.conversion_command}")

def workspace_convert(args):
    """Convert a legacy workspace by delegating to nirs4all-tools."""
    try:
        from nirs4all_tools.cli import main as tools_main
    except Exception as exc:
        logger.error(
            "nirs4all-tools is required for conversion. Install it with: "
            "pip install nirs4all[transition] or pip install nirs4all-tools"
        )
        raise SystemExit(1) from exc

    argv = [
        "legacy",
        "migrate",
        str(args.input),
        "--output",
        str(args.output),
        "--target",
        "nirs4all-workspace-v2",
    ]
    if args.verify:
        argv.append("--verify")
    if args.dry_run:
        argv.append("--dry-run")
    if args.strict:
        argv.append("--strict")
    raise SystemExit(tools_main(argv))

def add_workspace_commands(subparsers):
    """Add workspace commands to CLI."""

    # Workspace command group
    workspace = subparsers.add_parser(
        'workspace',
        help='Workspace management commands'
    )
    workspace_subparsers = workspace.add_subparsers(dest='workspace_command')

    # workspace init
    init_parser = workspace_subparsers.add_parser(
        'init',
        help='Initialize a new workspace'
    )
    init_parser.add_argument(
        'path',
        type=str,
        help='Path to workspace directory'
    )
    init_parser.set_defaults(func=workspace_init)

    # workspace list-runs
    list_runs_parser = workspace_subparsers.add_parser(
        'list-runs',
        help='List all runs in workspace'
    )
    list_runs_parser.add_argument(
        '--workspace',
        type=str,
        default='workspace',
        help='Workspace root directory (default: workspace)'
    )
    list_runs_parser.set_defaults(func=workspace_list_runs)

    # workspace query-best
    query_best_parser = workspace_subparsers.add_parser(
        'query-best',
        help='Query best predictions from workspace'
    )
    query_best_parser.add_argument(
        '--workspace',
        type=str,
        default='workspace',
        help='Workspace root directory (default: workspace)'
    )
    query_best_parser.add_argument(
        '--dataset',
        type=str,
        help='Filter by dataset name'
    )
    query_best_parser.add_argument(
        '--metric',
        type=str,
        default='test_score',
        help='Metric to sort by (default: test_score)'
    )
    query_best_parser.add_argument(
        '-n',
        type=int,
        default=10,
        help='Number of results (default: 10)'
    )
    query_best_parser.add_argument(
        '--ascending',
        action='store_true',
        help='Sort ascending (lower is better)'
    )
    query_best_parser.set_defaults(func=workspace_query_best)

    # workspace filter
    filter_parser = workspace_subparsers.add_parser(
        'filter',
        help='Filter predictions by criteria'
    )
    filter_parser.add_argument(
        '--workspace',
        type=str,
        default='workspace',
        help='Workspace root directory (default: workspace)'
    )
    filter_parser.add_argument(
        '--dataset',
        type=str,
        help='Filter by dataset name'
    )
    filter_parser.add_argument(
        '--test-score',
        type=float,
        help='Minimum test score'
    )
    filter_parser.add_argument(
        '--train-score',
        type=float,
        help='Minimum train score'
    )
    filter_parser.add_argument(
        '--val-score',
        type=float,
        help='Minimum validation score'
    )
    filter_parser.set_defaults(func=workspace_query_filter)

    # workspace stats
    stats_parser = workspace_subparsers.add_parser(
        'stats',
        help='Show workspace statistics'
    )
    stats_parser.add_argument(
        '--workspace',
        type=str,
        default='workspace',
        help='Workspace root directory (default: workspace)'
    )
    stats_parser.add_argument(
        '--metric',
        type=str,
        default='test_score',
        help='Metric for statistics (default: test_score)'
    )
    stats_parser.set_defaults(func=workspace_stats)

    # workspace list-library
    list_library_parser = workspace_subparsers.add_parser(
        'list-library',
        help='List items in library'
    )
    list_library_parser.add_argument(
        '--workspace',
        type=str,
        default='workspace',
        help='Workspace root directory (default: workspace)'
    )
    list_library_parser.set_defaults(func=workspace_list_library)

    # workspace inspect
    inspect_parser = workspace_subparsers.add_parser(
        'inspect',
        help='Inspect workspace format without opening it writable'
    )
    inspect_parser.add_argument(
        'path',
        type=str,
        help='Workspace directory or legacy artifact to inspect'
    )
    inspect_parser.set_defaults(func=workspace_inspect)

    # workspace convert
    convert_parser = workspace_subparsers.add_parser(
        'convert',
        help='Convert a legacy workspace/artifact into a fresh V1 workspace'
    )
    convert_parser.add_argument(
        'input',
        type=str,
        help='Legacy workspace directory or artifact (read-only input)'
    )
    convert_parser.add_argument(
        '--output',
        required=True,
        type=str,
        help='Fresh output directory for the converted workspace'
    )
    convert_mode = convert_parser.add_mutually_exclusive_group()
    convert_mode.add_argument(
        '--verify',
        action='store_true',
        help='Verify the converted output after migration'
    )
    convert_mode.add_argument(
        '--dry-run',
        action='store_true',
        help='Inspect and simulate conversion without writing an output store'
    )
    convert_parser.add_argument(
        '--strict',
        action='store_true',
        help='Abort on the first unsupported legacy item'
    )
    convert_parser.set_defaults(func=workspace_convert)
