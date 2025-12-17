"""
Artifact management CLI commands for nirs4all.

Provides commands for managing binary artifacts stored in workspace/binaries/:
- list-orphaned: Show artifacts not referenced by any manifest
- cleanup: Delete orphaned artifacts
- stats: Show storage statistics and deduplication info
- purge: Delete all artifacts for a dataset
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
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


def _get_all_datasets(workspace: Path) -> List[str]:
    """Get all datasets with artifacts in the workspace."""
    binaries_dir = workspace / "binaries"
    if not binaries_dir.exists():
        return []

    return [
        d.name for d in binaries_dir.iterdir()
        if d.is_dir()
    ]


def artifacts_list_orphaned(args):
    """List orphaned artifacts not referenced by any manifest."""
    from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry

    workspace_path = Path(args.workspace).resolve()
    binaries_dir = workspace_path / "binaries"

    if not binaries_dir.exists():
        logger.info("No artifacts found (binaries/ directory does not exist)")
        return

    # Get datasets to check
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = _get_all_datasets(workspace_path)

    if not datasets:
        logger.info("No datasets with artifacts found")
        return

    total_orphans = 0
    total_size = 0

    for dataset in datasets:
        registry = ArtifactRegistry(
            workspace=workspace_path,
            dataset=dataset
        )

        orphans = registry.find_orphaned_artifacts(scan_all_manifests=True)

        if orphans:
            logger.info(f"\nDataset: {dataset}")
            logger.info("-" * 60)

            for filename in orphans:
                filepath = registry.binaries_dir / filename
                if filepath.exists():
                    size = filepath.stat().st_size
                    total_size += size
                    logger.info(f"  * {filename} ({_format_bytes(size)})")
                else:
                    logger.info(f"  * {filename} (file missing)")

            total_orphans += len(orphans)

    if total_orphans == 0:
        logger.success("No orphaned artifacts found")
    else:
        logger.info(f"\n{'='*60}")
        logger.info(f"Total orphaned: {total_orphans} files ({_format_bytes(total_size)})")
        logger.info(f"\nRun 'nirs4all artifacts cleanup' to remove orphaned artifacts")


def artifacts_cleanup(args):
    """Delete orphaned artifacts."""
    from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry

    workspace_path = Path(args.workspace).resolve()
    binaries_dir = workspace_path / "binaries"

    if not binaries_dir.exists():
        logger.info("No artifacts found (binaries/ directory does not exist)")
        return

    # Get datasets to clean
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = _get_all_datasets(workspace_path)

    if not datasets:
        logger.info("No datasets with artifacts found")
        return

    dry_run = not args.force
    total_deleted = 0
    total_freed = 0

    if dry_run:
        logger.info("DRY RUN - No files will be deleted")
        logger.info("   Use --force to actually delete files\n")

    for dataset in datasets:
        registry = ArtifactRegistry(
            workspace=workspace_path,
            dataset=dataset
        )

        deleted, bytes_freed = registry.delete_orphaned_artifacts(
            dry_run=dry_run,
            scan_all_manifests=True
        )

        if deleted:
            action = "Would delete" if dry_run else "Deleted"
            logger.info(f"\nDataset: {dataset}")
            logger.info(f"   {action} {len(deleted)} orphaned artifacts ({_format_bytes(bytes_freed)})")

            if args.verbose:
                for filename in deleted:
                    logger.info(f"   * {filename}")

            total_deleted += len(deleted)
            total_freed += bytes_freed

    if total_deleted == 0:
        logger.success("No orphaned artifacts to clean up")
    else:
        action = "Would free" if dry_run else "Freed"
        logger.info(f"\n{'='*60}")
        logger.info(f"Total: {total_deleted} files, {action} {_format_bytes(total_freed)}")


def artifacts_stats(args):
    """Show artifact storage statistics."""
    from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry

    workspace_path = Path(args.workspace).resolve()
    binaries_dir = workspace_path / "binaries"

    if not binaries_dir.exists():
        logger.info("No artifacts found (binaries/ directory does not exist)")
        return

    # Get datasets to report on
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = _get_all_datasets(workspace_path)

    if not datasets:
        logger.info("No datasets with artifacts found")
        return

    logger.info("Artifact Storage Statistics")
    logger.info("=" * 70)

    grand_total_files = 0
    grand_total_size = 0
    grand_total_orphans = 0
    grand_orphan_size = 0

    for dataset in datasets:
        registry = ArtifactRegistry(
            workspace=workspace_path,
            dataset=dataset
        )

        stats = registry.get_stats(scan_all_manifests=True)

        logger.info(f"\nDataset: {dataset}")
        logger.info("-" * 60)
        logger.info(f"   Binaries path:     {stats['binaries_path']}")
        logger.info(f"   Files on disk:     {stats['disk_file_count']}")
        logger.info(f"   Disk usage:        {_format_bytes(stats['disk_usage_bytes'])}")

        if stats['total_artifacts'] > 0:
            logger.info(f"   Registered refs:   {stats['total_artifacts']}")
            logger.info(f"   Unique files:      {stats['unique_files']}")
            dedup_pct = stats['deduplication_ratio'] * 100
            logger.info(f"   Deduplication:     {dedup_pct:.1f}%")

        if stats['by_type']:
            logger.info(f"   By type:")
            for type_name, count in sorted(stats['by_type'].items()):
                logger.info(f"      {type_name}: {count}")

        if stats['orphaned_count'] > 0:
            logger.warning(f"   Orphaned:       {stats['orphaned_count']} files ({_format_bytes(stats['orphaned_size_bytes'])})")
            grand_total_orphans += stats['orphaned_count']
            grand_orphan_size += stats['orphaned_size_bytes']

        grand_total_files += stats['disk_file_count']
        grand_total_size += stats['disk_usage_bytes']

    # Print grand totals if multiple datasets
    if len(datasets) > 1:
        logger.info(f"\n{'='*70}")
        logger.info("Grand Total")
        logger.info(f"   Datasets:          {len(datasets)}")
        logger.info(f"   Total files:       {grand_total_files}")
        logger.info(f"   Total disk usage:  {_format_bytes(grand_total_size)}")
        if grand_total_orphans > 0:
            logger.warning(f"   Total orphaned:  {grand_total_orphans} files ({_format_bytes(grand_orphan_size)})")


def artifacts_purge(args):
    """Delete ALL artifacts for a dataset."""
    from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry

    workspace_path = Path(args.workspace).resolve()
    dataset = args.dataset

    if not dataset:
        logger.error("--dataset is required for purge command")
        sys.exit(1)

    registry = ArtifactRegistry(
        workspace=workspace_path,
        dataset=dataset
    )

    if not registry.binaries_dir.exists():
        logger.info(f"No artifacts found for dataset '{dataset}'")
        return

    # Count files that would be deleted
    file_count = sum(1 for f in registry.binaries_dir.iterdir() if f.is_file())
    total_size = sum(
        f.stat().st_size for f in registry.binaries_dir.iterdir() if f.is_file()
    )

    if file_count == 0:
        logger.info(f"No artifacts to purge for dataset '{dataset}'")
        return

    if not args.force:
        logger.warning(f"This will delete ALL {file_count} artifacts for dataset '{dataset}'")
        logger.info(f"   Total size: {_format_bytes(total_size)}")
        logger.info(f"\n   This action cannot be undone!")
        logger.info(f"\n   Use --force to confirm deletion")
        return

    # Confirm with user if interactive
    if not args.yes:
        response = input(f"\nDelete all {file_count} artifacts for '{dataset}'? [y/N]: ")
        if response.lower() not in ('y', 'yes'):
            logger.info("Aborted")
            return

    files_deleted, bytes_freed = registry.purge_dataset_artifacts(confirm=True)

    logger.success(f"Purged {files_deleted} artifacts for dataset '{dataset}'")
    logger.info(f"   Freed: {_format_bytes(bytes_freed)}")


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
        help='Delete ALL artifacts for a dataset (destructive!)'
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
