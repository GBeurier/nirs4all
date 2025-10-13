#!/usr/bin/env python3
"""
Garbage Collection Script for Orphaned Artifacts

This script finds and optionally removes artifact files that are no longer
referenced by any pipeline manifest.

Usage:
    python scripts/gc_artifacts.py                     # Dry run
    python scripts/gc_artifacts.py --results-dir ./results --dry-run
    python scripts/gc_artifacts.py --force             # Actually delete
"""

import argparse
import yaml
from pathlib import Path
from typing import Set, Dict, List
from datetime import datetime


def find_orphaned_artifacts(results_dir: Path) -> Dict[str, List[Path]]:
    """
    Find artifacts not referenced by any pipeline manifest.

    Args:
        results_dir: Path to results directory

    Returns:
        Dictionary mapping hash values to list of orphaned artifact file paths
    """
    artifacts_dir = results_dir / "artifacts" / "objects"
    pipelines_dir = results_dir / "pipelines"

    if not artifacts_dir.exists():
        print(f"⚠️  Artifacts directory not found: {artifacts_dir}")
        return {}

    if not pipelines_dir.exists():
        print(f"⚠️  Pipelines directory not found: {pipelines_dir}")
        return {}

    # Collect all artifact hashes from filesystem
    all_artifacts: Dict[str, List[Path]] = {}
    print(f"📁 Scanning artifacts in: {artifacts_dir}")

    for hash_prefix_dir in artifacts_dir.iterdir():
        if not hash_prefix_dir.is_dir():
            continue

        for artifact_file in hash_prefix_dir.iterdir():
            if artifact_file.is_file():
                # Extract hash from filename (remove extension)
                hash_value = artifact_file.stem
                if hash_value not in all_artifacts:
                    all_artifacts[hash_value] = []
                all_artifacts[hash_value].append(artifact_file)

    print(f"   Found {len(all_artifacts)} unique artifacts on disk")

    # Collect referenced hashes from all manifests
    referenced_hashes: Set[str] = set()
    manifest_count = 0

    print(f"📄 Scanning pipeline manifests in: {pipelines_dir}")

    for pipeline_dir in pipelines_dir.iterdir():
        if not pipeline_dir.is_dir():
            continue

        manifest_path = pipeline_dir / "manifest.yaml"
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    manifest = yaml.safe_load(f)
                    manifest_count += 1

                    for artifact in manifest.get("artifacts", []):
                        # Extract hash from "sha256:abc123..." format
                        hash_with_prefix = artifact.get("hash", "")
                        if hash_with_prefix:
                            hash_value = hash_with_prefix.split(":")[-1]
                            referenced_hashes.add(hash_value)
            except Exception as e:
                print(f"⚠️  Error reading manifest {manifest_path}: {e}")

    print(f"   Found {manifest_count} pipeline manifests")
    print(f"   Found {len(referenced_hashes)} unique referenced artifacts")

    # Find orphans
    orphaned_hashes = set(all_artifacts.keys()) - referenced_hashes
    orphaned_artifacts = {h: all_artifacts[h] for h in orphaned_hashes}

    return orphaned_artifacts


def cleanup_orphaned_artifacts(
    results_dir: Path,
    dry_run: bool = True,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Remove orphaned artifacts.

    Args:
        results_dir: Path to results directory
        dry_run: If True, only show what would be deleted
        verbose: If True, print details for each file

    Returns:
        Dictionary with statistics about the cleanup
    """
    orphans = find_orphaned_artifacts(results_dir)

    if not orphans:
        print("\n✅ No orphaned artifacts found!")
        return {"count": 0, "size": 0, "files": []}

    print(f"\n🗑️  Found {len(orphans)} orphaned artifacts")

    total_size = 0
    deleted_files = []

    for hash_value, artifact_files in orphans.items():
        for artifact_file in artifact_files:
            try:
                size = artifact_file.stat().st_size
                total_size += size

                if dry_run:
                    if verbose:
                        print(f"   Would delete: {artifact_file.relative_to(results_dir)} ({_format_size(size)})")
                else:
                    artifact_file.unlink()
                    deleted_files.append(str(artifact_file.relative_to(results_dir)))
                    if verbose:
                        print(f"   ✅ Deleted: {artifact_file.relative_to(results_dir)} ({_format_size(size)})")
            except Exception as e:
                print(f"   ❌ Error processing {artifact_file}: {e}")

    print(f"\n📊 Total space: {_format_size(total_size)}")

    if dry_run:
        print("\n💡 This was a dry run. Run with --force to actually delete files.")
    else:
        print(f"\n✅ Deleted {len(deleted_files)} artifact files")

    return {
        "count": len(deleted_files) if not dry_run else len(orphans),
        "size": total_size,
        "files": deleted_files
    }


def _format_size(size_bytes: int) -> str:
    """Format byte size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def list_artifacts_stats(results_dir: Path):
    """Display statistics about artifacts."""
    artifacts_dir = results_dir / "artifacts" / "objects"
    pipelines_dir = results_dir / "pipelines"

    if not artifacts_dir.exists():
        print(f"⚠️  Artifacts directory not found: {artifacts_dir}")
        return

    # Count all artifacts
    total_artifacts = 0
    total_size = 0

    for hash_prefix_dir in artifacts_dir.iterdir():
        if not hash_prefix_dir.is_dir():
            continue
        for artifact_file in hash_prefix_dir.iterdir():
            if artifact_file.is_file():
                total_artifacts += 1
                total_size += artifact_file.stat().st_size

    # Count pipelines
    pipeline_count = 0
    if pipelines_dir.exists():
        pipeline_count = sum(1 for p in pipelines_dir.iterdir() if p.is_dir() and (p / "manifest.yaml").exists())

    print("\n📊 Artifact Statistics")
    print("=" * 60)
    print(f"Total artifacts:     {total_artifacts}")
    print(f"Total size:          {_format_size(total_size)}")
    print(f"Pipeline manifests:  {pipeline_count}")
    print(f"Artifacts directory: {artifacts_dir}")
    print(f"Pipelines directory: {pipelines_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Clean up orphaned artifacts not referenced by any pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show what would be deleted (dry run)
  python scripts/gc_artifacts.py

  # Actually delete orphaned artifacts
  python scripts/gc_artifacts.py --force

  # Use custom results directory
  python scripts/gc_artifacts.py --results-dir /path/to/results --force

  # Show artifact statistics
  python scripts/gc_artifacts.py --stats
"""
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Path to results directory (default: ./results)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Actually delete files (default is dry run)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry run mode (show what would be deleted)"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show artifact statistics only"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show summary (less verbose output)"
    )

    args = parser.parse_args()

    # Resolve results directory
    results_dir = args.results_dir.resolve()

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return 1

    print(f"🔍 Using results directory: {results_dir}\n")

    # Show stats if requested
    if args.stats:
        list_artifacts_stats(results_dir)
        return 0

    # Cleanup artifacts
    dry_run = not args.force
    verbose = not args.quiet

    try:
        stats = cleanup_orphaned_artifacts(results_dir, dry_run=dry_run, verbose=verbose)
        return 0
    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
