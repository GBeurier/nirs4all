"""
Workspace management CLI commands for nirs4all.

Provides commands for workspace initialization, run management, catalog queries,
and library operations.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


def workspace_init(args):
    """Initialize a new workspace."""
    from nirs4all.workspace import WorkspaceManager

    workspace_path = Path(args.path)
    ws = WorkspaceManager(workspace_path)
    ws.initialize_workspace()

    logger.success(f"Workspace initialized at: {workspace_path}")
    logger.info("  Created directories:")
    logger.info("    - runs/")
    logger.info("    - exports/full_pipelines/")
    logger.info("    - exports/best_predictions/")
    logger.info("    - library/templates/")
    logger.info("    - library/trained/filtered/")
    logger.info("    - library/trained/pipeline/")
    logger.info("    - library/trained/fullrun/")
    logger.info("    - catalog/")


def workspace_list_runs(args):
    """List all runs in workspace."""
    from nirs4all.workspace import WorkspaceManager

    workspace_path = Path(args.workspace)
    ws = WorkspaceManager(workspace_path)

    runs = ws.list_runs()

    if not runs:
        logger.info("No runs found in workspace.")
        return

    logger.info(f"Found {len(runs)} run(s):\n")
    for run_info in runs:
        logger.info(f"  {run_info['name']}")
        logger.info(f"    Dataset: {run_info['dataset']}")
        logger.info(f"    Date: {run_info['date']}")
        if run_info.get('custom_name'):
            logger.info(f"    Custom name: {run_info['custom_name']}")
        logger.info("")


def workspace_query_best(args):
    """Query best pipelines from catalog."""
    from nirs4all.data.predictions import Predictions

    workspace_path = Path(args.workspace)
    catalog_dir = workspace_path / "catalog"

    if not catalog_dir.exists():
        logger.error(f"Catalog not found at {catalog_dir}")
        logger.info("Run pipelines and archive predictions first.")
        sys.exit(1)

    meta_file = catalog_dir / "predictions_meta.parquet"
    if not meta_file.exists():
        logger.error("No predictions in catalog.")
        logger.info("Archive pipeline predictions using Predictions.archive_to_catalog()")
        sys.exit(1)

    # Load predictions from catalog
    try:
        preds = Predictions.load_from_parquet(catalog_dir)
        logger.success(f"Loaded {preds._df.height} predictions from catalog\n")
    except Exception as e:
        logger.error(f"Error loading catalog: {e}")
        sys.exit(1)

    # Query best
    best = preds.query_best(
        dataset_name=args.dataset,
        metric=args.metric,
        n=args.n,
        ascending=args.ascending
    )

    if best.height == 0:
        logger.info("No predictions found matching criteria.")
        return

    # Display results
    logger.info(f"Top {args.n} pipelines by {args.metric}:")
    logger.info(f"{'='*80}\n")

    # Convert to pandas for nice display
    df = best.to_pandas()
    logger.info(df.to_string(index=False))


def workspace_query_filter(args):
    """Filter predictions by criteria."""
    from nirs4all.data.predictions import Predictions

    workspace_path = Path(args.workspace)
    catalog_dir = workspace_path / "catalog"

    if not catalog_dir.exists():
        logger.error(f"Catalog not found at {catalog_dir}")
        sys.exit(1)

    # Load predictions
    preds = Predictions.load_from_parquet(catalog_dir)

    # Build metric thresholds
    thresholds = {}
    if args.test_score:
        thresholds['test_score'] = args.test_score
    if args.train_score:
        thresholds['train_score'] = args.train_score
    if args.val_score:
        thresholds['val_score'] = args.val_score

    # Apply filters
    filtered = preds.filter_by_criteria(
        dataset_name=args.dataset,
        metric_thresholds=thresholds if thresholds else None
    )

    logger.info(f"Found {filtered.height} predictions matching criteria\n")

    if filtered.height > 0:
        df = filtered.to_pandas()
        logger.info(df.to_string(index=False))


def workspace_stats(args):
    """Show catalog statistics."""
    from nirs4all.data.predictions import Predictions

    workspace_path = Path(args.workspace)
    catalog_dir = workspace_path / "catalog"

    if not catalog_dir.exists():
        logger.error(f"Catalog not found at {catalog_dir}")
        sys.exit(1)

    # Load predictions
    preds = Predictions.load_from_parquet(catalog_dir)

    logger.info("Catalog Statistics")
    logger.info(f"{'='*60}\n")
    logger.info(f"Total predictions: {preds._df.height}")

    # Datasets
    if 'dataset_name' in preds._df.columns:
        datasets = preds._df['dataset_name'].unique().to_list()
        logger.info(f"Datasets: {len(datasets)}")
        for ds in datasets:
            count = preds._df.filter(preds._df['dataset_name'] == ds).height
            logger.info(f"  - {ds}: {count} predictions")

    logger.info("")

    # Metric statistics
    metric = args.metric
    if metric in preds._df.columns:
        stats = preds.get_summary_stats(metric=metric)
        logger.info(f"{metric} statistics:")
        logger.info(f"  Min:    {stats['min']:.4f}")
        logger.info(f"  Max:    {stats['max']:.4f}")
        logger.info(f"  Mean:   {stats['mean']:.4f}")
        logger.info(f"  Median: {stats['median']:.4f}")
        logger.info(f"  Std:    {stats['std']:.4f}")


def workspace_list_library(args):
    """List items in library."""
    from nirs4all.workspace import LibraryManager

    workspace_path = Path(args.workspace)
    library_dir = workspace_path / "library"

    if not library_dir.exists():
        logger.error(f"Library not found at {library_dir}")
        sys.exit(1)

    library = LibraryManager(library_dir)

    # List templates
    templates = library.list_templates()
    logger.info(f"Templates: {len(templates)}")
    for t in templates:
        logger.info(f"  - {t['name']}: {t.get('description', 'No description')}")
    logger.info("")

    # List filtered
    filtered = library.list_filtered()
    logger.info(f"Filtered pipelines: {len(filtered)}")
    for f in filtered:
        logger.info(f"  - {f['name']}: {f.get('description', 'No description')}")
    logger.info("")

    # List full pipelines
    pipelines = library.list_pipelines()
    logger.info(f"Full pipelines: {len(pipelines)}")
    for p in pipelines:
        logger.info(f"  - {p['name']}: {p.get('description', 'No description')}")
    logger.info("")

    # List full runs
    fullruns = library.list_fullruns()
    logger.info(f"Full runs: {len(fullruns)}")
    for r in fullruns:
        logger.info(f"  - {r['name']}: {r.get('description', 'No description')}")


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
        help='Query best pipelines from catalog'
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
        help='Show catalog statistics'
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
