"""
Workspace Manager - Top-level workspace coordination.

Manages the workspace directory structure for user-friendly organization
of experimental runs, exports, library, and catalog.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional


class WorkspaceManager:
    """Manages workspace-level operations."""

    def __init__(self, workspace_root: Path):
        """
        Initialize workspace manager.

        Args:
            workspace_root: Root directory for workspace
        """
        self.root = Path(workspace_root)
        self.runs_dir = self.root / "runs"
        self.exports_dir = self.root / "exports"
        self.library_dir = self.root / "library"
        self.catalog_dir = self.root / "catalog"

    def initialize_workspace(self) -> None:
        """Create workspace directory structure."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        # Library with 3 types
        (self.library_dir / "templates").mkdir(parents=True, exist_ok=True)
        (self.library_dir / "trained" / "filtered").mkdir(parents=True, exist_ok=True)
        (self.library_dir / "trained" / "pipeline").mkdir(parents=True, exist_ok=True)
        (self.library_dir / "trained" / "fullrun").mkdir(parents=True, exist_ok=True)

        # Catalog with archives
        (self.catalog_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.catalog_dir / "archives" / "filtered").mkdir(parents=True, exist_ok=True)
        (self.catalog_dir / "archives" / "pipeline").mkdir(parents=True, exist_ok=True)
        (self.catalog_dir / "archives" / "best_predictions").mkdir(parents=True, exist_ok=True)

        # Exports with best_predictions
        (self.exports_dir / "best_predictions").mkdir(parents=True, exist_ok=True)
        (self.exports_dir / "session_reports").mkdir(parents=True, exist_ok=True)

    def create_run(self, dataset_name: str, run_name: str = None) -> 'RunManager':
        """Create a new run for a dataset with optional custom name.

        Creates run directory:
        - Without custom name: YYYY-MM-DD_dataset/
        - With custom name: YYYY-MM-DD_dataset_runname/
        """
        from .run_manager import RunManager

        date_str = datetime.now().strftime("%Y-%m-%d")
        if run_name:
            run_dir = self.runs_dir / f"{date_str}_{dataset_name}_{run_name}"
        else:
            run_dir = self.runs_dir / f"{date_str}_{dataset_name}"
        return RunManager(run_dir, dataset_name)

    def list_runs(self) -> list[dict]:
        """List all runs with metadata."""
        runs = []
        if self.runs_dir.exists():
            for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
                if run_dir.is_dir():
                    # Parse run directory name
                    parts = run_dir.name.split("_", 2)
                    date_str = parts[0] if len(parts) > 0 else ""
                    dataset = parts[1] if len(parts) > 1 else ""
                    custom_name = parts[2] if len(parts) > 2 else None

                    runs.append({
                        "path": run_dir,
                        "date": date_str,
                        "dataset": dataset,
                        "custom_name": custom_name,
                        "name": run_dir.name
                    })
        return runs
