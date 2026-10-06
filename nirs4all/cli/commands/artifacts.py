"""
Artifact management CLI commands for nirs4all.

Provides commands for managing binary artifacts stored in workspace/artifacts/:
- list-orphaned: Show artifacts not referenced by any manifest
- cleanup: Delete orphaned artifacts
- stats: Show storage statistics and deduplication info
- purge: Delete known-owned unpublished artifacts for a dataset
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

def _format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.2f} MB"
    else:
        return f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"

def _registry(args):
    """Inspect the shared artifact tree without creating workspace state."""
    from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry

    workspace = Path(args.workspace).resolve()
    if not (workspace / "artifacts").is_dir():
        logger.info("No artifacts found (artifacts/ directory does not exist)")
        return None
    return ArtifactRegistry(workspace=workspace, dataset=args.dataset or "")


def artifacts_list_orphaned(args):
    """List unreferenced shared artifacts across the entire workspace."""
    registry = _registry(args)
    if registry is None:
        return
    if args.dataset:
        logger.info("Shared orphan blobs have no dataset identity; listing workspace-wide orphans")
    orphans = registry.find_orphaned_artifacts(scan_all_manifests=True)
    for relative in orphans:
        logger.info(f"  * {relative} ({_format_bytes((registry.binaries_dir / relative).stat().st_size)})")
    logger.info(f"Total orphaned: {len(orphans)} files")


def artifacts_cleanup(args):
    """Delete only workspace-wide blobs unreferenced by any dataset or store row."""
    registry = _registry(args)
    if registry is None:
        return
    if args.dataset:
        logger.info("Cannot attribute shared orphan blobs to a dataset; omit --dataset for workspace-wide cleanup")
        return
    deleted, freed = registry.delete_orphaned_artifacts(dry_run=not args.force, scan_all_manifests=True)
    action = "Deleted" if args.force else "Would delete"
    logger.info(f"{action} {len(deleted)} orphaned artifacts ({_format_bytes(freed)})")
    if args.verbose:
        for relative in deleted:
            logger.info(f"   * {relative}")


def artifacts_stats(args):
    """Show shared workspace storage statistics once, without double-counting datasets."""
    registry = _registry(args)
    if registry is None:
        return
    stats = registry.get_stats(scan_all_manifests=True)
    logger.info("Artifact Storage Statistics (shared workspace)")
    logger.info(f"   Artifacts path: {stats['binaries_path']}")
    logger.info(f"   Files on disk: {stats['disk_file_count']}")
    logger.info(f"   Disk usage: {_format_bytes(stats['disk_usage_bytes'])}")
    logger.info(f"   Orphaned: {stats['orphaned_count']} files ({_format_bytes(stats['orphaned_size_bytes'])})")


def artifacts_purge(args):
    """Purge known-owned unpublished artifacts; preserve every durable reference."""
    if not args.dataset:
        logger.error("--dataset is required for purge command")
        sys.exit(1)
    registry = _registry(args)
    if registry is None:
        return
    candidates = registry.get_purge_candidates()
    if not candidates:
        logger.info("No unpublished artifacts with known dataset ownership; live references and shared blobs are preserved")
        return
    if not args.force:
        logger.info(f"Would purge {len(candidates)} unpublished artifacts; use --force to confirm")
        return
    if not args.yes:
        response = input(f"Delete {len(candidates)} unpublished artifacts for '{args.dataset}'? [y/N]: ")
        if response.lower() not in ('y', 'yes'):
            return
    deleted, freed = registry.purge_dataset_artifacts(confirm=True)
    logger.success(f"Purged {deleted} unpublished artifacts for dataset '{args.dataset}' ({_format_bytes(freed)})")

def add_artifacts_commands(subparsers):
    """Add artifact management commands to CLI."""

    # Artifacts command group
    artifacts = subparsers.add_parser(
        'artifacts',
        help='Artifact management commands'
    )
    artifacts_subparsers = artifacts.add_subparsers(dest='artifacts_command')

    # Common arguments
    def add_common_args(parser):
        parser.add_argument(
            '--workspace', '-w',
            type=str,
            default='workspace',
            help='Workspace root directory (default: workspace)'
        )
        parser.add_argument(
            '--dataset', '-d',
            type=str,
            help='Dataset name (default: all datasets)'
        )

    # artifacts list-orphaned
    list_orphaned_parser = artifacts_subparsers.add_parser(
        'list-orphaned',
        help='List artifacts not referenced by any manifest'
    )
    add_common_args(list_orphaned_parser)
    list_orphaned_parser.set_defaults(func=artifacts_list_orphaned)

    # artifacts cleanup
    cleanup_parser = artifacts_subparsers.add_parser(
        'cleanup',
        help='Delete orphaned artifacts'
    )
    add_common_args(cleanup_parser)
    cleanup_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Actually delete files (default is dry run)'
    )
    cleanup_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='List each deleted file'
    )
    cleanup_parser.set_defaults(func=artifacts_cleanup)

    # artifacts stats
    stats_parser = artifacts_subparsers.add_parser(
        'stats',
        help='Show artifact storage statistics'
    )
    add_common_args(stats_parser)
    stats_parser.set_defaults(func=artifacts_stats)

    # artifacts purge
    purge_parser = artifacts_subparsers.add_parser(
        'purge',
        help='Purge unpublished dataset artifacts while preserving live references'
    )
    purge_parser.add_argument(
        '--workspace', '-w',
        type=str,
        default='workspace',
        help='Workspace root directory (default: workspace)'
    )
    purge_parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset name (required)'
    )
    purge_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Confirm destructive operation'
    )
    purge_parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )
    purge_parser.set_defaults(func=artifacts_purge)
