"""
Run Manager - Coordinates run-level operations.

Manages individual experimental runs including pipeline registration,
sequential numbering, and run-level metadata.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class RunManager:
    """Manages individual run operations."""

    def __init__(self, run_dir: Path, dataset_name: str):
        """
        Initialize run manager.

        Args:
            run_dir: Directory for this run
            dataset_name: Name of the dataset
        """
        self.run_dir = Path(run_dir)
        self.dataset_name = dataset_name
        self.binaries_dir = self.run_dir / "_binaries"
        self.run_config_file = self.run_dir / "run_config.json"
        self.run_summary_file = self.run_dir / "run_summary.json"
        self.run_log_file = self.run_dir / "run.log"

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize run structure with configuration.

        Args:
            config: Run configuration dictionary
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.binaries_dir.mkdir(exist_ok=True)

        # Save run configuration
        run_config = {
            "dataset_name": self.dataset_name,
            "created_at": datetime.now().isoformat(),
            **config
        }

        with open(self.run_config_file, 'w') as f:
            json.dump(run_config, f, indent=2)

    def get_next_pipeline_number(self) -> int:
        """
        Get next sequential pipeline number in run directory.

        Counts existing pipeline directories (excludes _binaries).
        Returns: Next number (e.g., 1, 2, 3...)
        """
        if not self.run_dir.exists():
            return 1

        existing = [d for d in self.run_dir.iterdir()
                    if d.is_dir() and not d.name.startswith("_")]
        return len(existing) + 1

    def create_pipeline_dir(self, pipeline_hash: str, pipeline_name: str = None) -> Path:
        """
        Create pipeline directory with sequential numbering.

        Args:
            pipeline_hash: Hash of pipeline configuration
            pipeline_name: Optional custom name for pipeline

        Returns:
            Path to created pipeline directory
        """
        pipeline_num = self.get_next_pipeline_number()

        # Build pipeline_id with optional custom name
        if pipeline_name:
            pipeline_id = f"{pipeline_num:04d}_{pipeline_name}_{pipeline_hash}"
        else:
            pipeline_id = f"{pipeline_num:04d}_{pipeline_hash}"

        pipeline_dir = self.run_dir / pipeline_id
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        return pipeline_dir

    def list_pipelines(self) -> list[Path]:
        """List all pipeline directories in this run."""
        if not self.run_dir.exists():
            return []

        return sorted([d for d in self.run_dir.iterdir()
                      if d.is_dir() and not d.name.startswith("_")])

    def update_summary(self, summary_data: Dict[str, Any]) -> None:
        """
        Update run summary with aggregated results.

        Args:
            summary_data: Summary data to save
        """
        summary = {
            "dataset_name": self.dataset_name,
            "run_date": datetime.now().strftime("%Y-%m-%d"),
            "updated_at": datetime.now().isoformat(),
            **summary_data
        }

        with open(self.run_summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
