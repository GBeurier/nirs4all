"""
Manifest Manager - Pipeline manifest and dataset index management

Manages pipeline manifests with sequential numbering and content-addressed artifacts.
Provides centralized pipeline registration, lookup, and lifecycle management.

Architecture:
    workspace/runs/YYYY-MM-DD_dataset/
    ├── artifacts/objects/           # Content-addressed binaries
    ├── 0001_abc123/                 # Sequential pipelines
    │   ├── manifest.yaml
    │   ├── metrics.json
    │   └── predictions.csv
    ├── 0002_def456/
    └── predictions.json             # Global predictions
"""

import uuid
import yaml
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _sanitize_for_yaml(obj: Any) -> Any:
    """
    Recursively sanitize data structures for safe YAML serialization.
    Converts tuples to lists to avoid Python-specific YAML tags.

    Args:
        obj: Object to sanitize

    Returns:
        Sanitized object safe for YAML safe_load
    """
    if isinstance(obj, tuple):
        return [_sanitize_for_yaml(item) for item in obj]
    elif isinstance(obj, list):
        return [_sanitize_for_yaml(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _sanitize_for_yaml(value) for key, value in obj.items()}
    else:
        return obj


class ManifestManager:
    """
    Manage pipeline manifests with sequential numbering.

    This class handles:
    - Creating new pipelines with sequential numbering (0001_hash, 0002_hash)
    - Saving/loading pipeline manifests
    - Content-addressed artifact storage
    """

    def __init__(self, results_dir: Union[str, Path]):
        """
        Initialize manifest manager.

        Args:
            results_dir: Path to run directory (workspace/runs/YYYY-MM-DD_dataset/)
        """
        self.results_dir = Path(results_dir)
        self.artifacts_dir = self.results_dir / "artifacts" / "objects"

        # Ensure directories exist
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def create_pipeline(
        self,
        name: str,
        dataset: str,
        pipeline_config: dict,
        pipeline_hash: str,
        metadata: Optional[dict] = None
    ) -> tuple[str, Path]:
        """
        Create new pipeline with sequential numbering.

        Args:
            name: Pipeline name (for human reference)
            dataset: Dataset name
            pipeline_config: Pipeline configuration dict
            pipeline_hash: Hash of pipeline config (first 6 chars)
            metadata: Optional initial metadata

        Returns:
            Tuple of (pipeline_id, pipeline_dir)
            pipeline_id format: "0001_abc123" or "0001_name_abc123"
        """
        # Get sequential number
        pipeline_num = self.get_next_pipeline_number()

        # Build pipeline_id with optional custom name
        if name and name != "pipeline":  # Don't include generic "pipeline" name
            pipeline_id = f"{pipeline_num:04d}_{name}_{pipeline_hash}"
        else:
            pipeline_id = f"{pipeline_num:04d}_{pipeline_hash}"

        # Create directory
        pipeline_dir = self.results_dir / pipeline_id
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        # Create manifest
        uid = str(uuid.uuid4())
        manifest = {
            "uid": uid,
            "pipeline_id": pipeline_id,
            "name": name,
            "dataset": dataset,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "pipeline": pipeline_config,
            "metadata": metadata or {},
            "artifacts": [],
            "predictions": []
        }

        self.save_manifest(pipeline_id, manifest)

        return pipeline_id, pipeline_dir

    def save_manifest(self, pipeline_id: str, manifest: dict) -> None:
        """
        Save manifest YAML file.

        Args:
            pipeline_id: Pipeline ID (e.g., "0001_abc123")
            manifest: Complete manifest dictionary
        """
        manifest_path = self.results_dir / pipeline_id / "manifest.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Sanitize manifest to convert tuples to lists for YAML compatibility
        sanitized_manifest = _sanitize_for_yaml(manifest)

        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.dump(sanitized_manifest, f, default_flow_style=False, sort_keys=False)

    def load_manifest(self, pipeline_id: str) -> dict:
        """
        Load manifest YAML file.

        Args:
            pipeline_id: Pipeline ID (e.g., "0001_abc123")

        Returns:
            Manifest dictionary

        Raises:
            FileNotFoundError: If manifest doesn't exist
        """
        manifest_path = self.results_dir / pipeline_id / "manifest.yaml"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def update_manifest(self, pipeline_id: str, updates: dict) -> None:
        """
        Update specific fields in a manifest.

        Args:
            pipeline_id: Pipeline ID
            updates: Dictionary of fields to update
        """
        manifest = self.load_manifest(pipeline_id)
        manifest.update(updates)
        self.save_manifest(pipeline_id, manifest)

    def append_artifacts(self, pipeline_id: str, artifacts: List[dict]) -> None:
        """
        Append artifacts to a pipeline manifest.

        Args:
            pipeline_id: Pipeline ID
            artifacts: List of artifact metadata dictionaries
        """
        manifest = self.load_manifest(pipeline_id)
        manifest["artifacts"].extend(artifacts)
        self.save_manifest(pipeline_id, manifest)

    def append_prediction(self, pipeline_id: str, prediction: dict) -> None:
        """
        Append a prediction record to pipeline manifest.

        Args:
            pipeline_id: Pipeline ID
            prediction: Prediction metadata dictionary
        """
        manifest = self.load_manifest(pipeline_id)
        manifest["predictions"].append(prediction)
        self.save_manifest(pipeline_id, manifest)

    def list_pipelines(self) -> List[str]:
        """
        List all pipeline IDs in this run.

        Returns:
            List of pipeline IDs (e.g., ["0001_abc123", "0002_def456"])
        """
        if not self.results_dir.exists():
            return []

        return sorted([d.name for d in self.results_dir.iterdir()
                      if d.is_dir() and not d.name.startswith("artifacts")
                      and d.name[0].isdigit()])

    def delete_pipeline(self, pipeline_id: str) -> None:
        """
        Delete pipeline directory and manifest.

        Args:
            pipeline_id: Pipeline ID to delete
        """
        pipeline_dir = self.results_dir / pipeline_id

        if pipeline_dir.exists():
            shutil.rmtree(pipeline_dir)

    def get_artifact_path(self, content_hash: str) -> Path:
        """
        Get path for content-addressed artifact.

        Args:
            content_hash: Content hash of artifact

        Returns:
            Path to artifact in artifacts/objects/<hash[:2]>/<hash>
        """
        return self.artifacts_dir / content_hash[:2] / content_hash

    def artifact_exists(self, content_hash: str) -> bool:
        """
        Check if artifact exists in storage.

        Args:
            content_hash: Content hash to check

        Returns:
            True if artifact exists
        """
        # Check all possible extensions
        artifact_dir = self.artifacts_dir / content_hash[:2] / content_hash
        if artifact_dir.exists():
            return True

        # Check for files with this hash as base name
        parent = self.artifacts_dir / content_hash[:2]
        if parent.exists():
            return any(f.stem == content_hash for f in parent.iterdir())

        return False

    def pipeline_exists(self, pipeline_id: str) -> bool:
        """
        Check if a pipeline exists.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            True if manifest exists
        """
        manifest_path = self.results_dir / pipeline_id / "manifest.yaml"
        return manifest_path.exists()

    def get_pipeline_path(self, pipeline_id: str) -> Path:
        """
        Get the directory path for a pipeline.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Path to pipeline directory
        """
        return self.results_dir / pipeline_id

    def list_all_pipelines(self) -> List[Dict[str, Any]]:
        """
        List all pipelines in this run.

        Returns:
            List of pipeline info dictionaries
        """
        pipelines = []

        for pipeline_id in self.list_pipelines():
            manifest_path = self.results_dir / pipeline_id / "manifest.yaml"
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = yaml.safe_load(f)

                    pipelines.append({
                        "pipeline_id": pipeline_id,
                        "uid": manifest.get("uid"),
                        "name": manifest.get("name"),
                        "dataset": manifest.get("dataset"),
                        "created_at": manifest.get("created_at"),
                        "num_artifacts": len(manifest.get("artifacts", [])),
                        "num_predictions": len(manifest.get("predictions", []))
                    })
                except (yaml.YAMLError, OSError, KeyError):
                    continue

        return pipelines

    def get_next_pipeline_number(self, run_dir: Optional[Path] = None) -> int:
        """
        Get next sequential pipeline number for workspace runs.

        Counts existing pipeline directories (excludes _binaries).

        Args:
            run_dir: Run directory to count pipelines in. If None, uses results_dir.

        Returns:
            Next number (e.g., 1, 2, 3...)
        """
        target_dir = Path(run_dir) if run_dir else self.results_dir

        if not target_dir.exists():
            return 1

        # Count only numbered pipeline directories (exclude artifacts, etc.)
        existing = [d for d in target_dir.iterdir()
                    if d.is_dir() and d.name[0:4].isdigit()]
        return len(existing) + 1
