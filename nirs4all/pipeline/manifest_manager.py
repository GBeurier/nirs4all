"""
Manifest Manager - Pipeline manifest and dataset index management

Manages UID-based pipeline manifests and dataset indexes using YAML files.
Provides centralized pipeline registration, lookup, and lifecycle management.

Architecture:
    results/
    ├── pipelines/<uid>/manifest.yaml    # Single file per pipeline
    └── datasets/<name>/index.yaml       # Name → UID mapping
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
    Manage pipeline manifests and dataset indexes using YAML.

    This class handles:
    - Creating new pipelines with unique UIDs
    - Saving/loading pipeline manifests
    - Registering pipelines in dataset indexes
    - Looking up pipeline UIDs by name
    - Deleting pipelines and cleaning up references
    """

    def __init__(self, results_dir: Union[str, Path]):
        """
        Initialize manifest manager.

        Args:
            results_dir: Path to results directory (contains pipelines/, datasets/, artifacts/)
        """
        self.results_dir = Path(results_dir)
        self.artifacts_dir = self.results_dir / "artifacts" / "objects"
        self.pipelines_dir = self.results_dir / "pipelines"
        self.datasets_dir = self.results_dir / "datasets"

        # Ensure directories exist
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.pipelines_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def create_pipeline(
        self,
        name: str,
        dataset: str,
        pipeline_config: dict,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Create new pipeline with UID and initial manifest.

        Args:
            name: Pipeline name (for human reference)
            dataset: Dataset name
            pipeline_config: Pipeline configuration dict
            metadata: Optional initial metadata

        Returns:
            Pipeline UID (UUID string)
        """
        uid = str(uuid.uuid4())
        pipeline_dir = self.pipelines_dir / uid
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "uid": uid,
            "name": name,
            "dataset": dataset,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "pipeline": pipeline_config,
            "metadata": metadata or {},
            "artifacts": [],
            "predictions": []
        }

        self.save_manifest(uid, manifest)
        self.register_in_dataset(dataset, name, uid)

        return uid

    def save_manifest(self, uid: str, manifest: dict) -> None:
        """
        Save manifest YAML file.

        Args:
            uid: Pipeline UID
            manifest: Complete manifest dictionary
        """
        manifest_path = self.pipelines_dir / uid / "manifest.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    def load_manifest(self, uid: str) -> dict:
        """
        Load manifest YAML file.

        Args:
            uid: Pipeline UID

        Returns:
            Manifest dictionary

        Raises:
            FileNotFoundError: If manifest doesn't exist
        """
        manifest_path = self.pipelines_dir / uid / "manifest.yaml"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def update_manifest(self, uid: str, updates: dict) -> None:
        """
        Update specific fields in a manifest.

        Args:
            uid: Pipeline UID
            updates: Dictionary of fields to update (shallow merge)
        """
        manifest = self.load_manifest(uid)
        manifest.update(updates)
        self.save_manifest(uid, manifest)

    def append_artifacts(self, uid: str, artifacts: List[dict]) -> None:
        """
        Append artifacts to a pipeline manifest.

        Args:
            uid: Pipeline UID
            artifacts: List of artifact metadata dictionaries
        """
        manifest = self.load_manifest(uid)
        manifest["artifacts"].extend(artifacts)
        self.save_manifest(uid, manifest)

    def append_prediction(self, uid: str, prediction: dict) -> None:
        """
        Append a prediction record to pipeline manifest.

        Args:
            uid: Pipeline UID
            prediction: Prediction metadata dictionary
        """
        manifest = self.load_manifest(uid)
        manifest["predictions"].append(prediction)
        self.save_manifest(uid, manifest)

    def get_pipeline_uid(self, dataset: str, pipeline_name: str) -> Optional[str]:
        """
        Get pipeline UID from dataset index.

        Args:
            dataset: Dataset name
            pipeline_name: Pipeline name

        Returns:
            Pipeline UID or None if not found
        """
        index_path = self.datasets_dir / dataset / "index.yaml"

        if not index_path.exists():
            return None

        with open(index_path, "r", encoding="utf-8") as f:
            index = yaml.safe_load(f)

        return index.get("pipelines", {}).get(pipeline_name)

    def list_pipelines(self, dataset: str) -> Dict[str, str]:
        """
        List all pipelines for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            Dictionary mapping pipeline names to UIDs
        """
        index_path = self.datasets_dir / dataset / "index.yaml"

        if not index_path.exists():
            return {}

        with open(index_path, "r", encoding="utf-8") as f:
            index = yaml.safe_load(f)

        return index.get("pipelines", {})

    def register_in_dataset(self, dataset: str, pipeline_name: str, uid: str) -> None:
        """
        Register pipeline in dataset index.

        Args:
            dataset: Dataset name
            pipeline_name: Pipeline name
            uid: Pipeline UID
        """
        dataset_dir = self.datasets_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        index_path = dataset_dir / "index.yaml"

        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = yaml.safe_load(f) or {}
        else:
            index = {
                "dataset": dataset,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "pipelines": {}
            }

        if "pipelines" not in index:
            index["pipelines"] = {}

        index["pipelines"][pipeline_name] = uid

        with open(index_path, "w", encoding="utf-8") as f:
            yaml.dump(index, f, default_flow_style=False)

    def unregister_from_dataset(self, dataset: str, pipeline_name: str) -> None:
        """
        Remove pipeline from dataset index.

        Args:
            dataset: Dataset name
            pipeline_name: Pipeline name
        """
        index_path = self.datasets_dir / dataset / "index.yaml"

        if not index_path.exists():
            return

        with open(index_path, "r", encoding="utf-8") as f:
            index = yaml.safe_load(f)

        if "pipelines" in index and pipeline_name in index["pipelines"]:
            del index["pipelines"][pipeline_name]

        with open(index_path, "w", encoding="utf-8") as f:
            yaml.dump(index, f, default_flow_style=False)

    def delete_pipeline(self, uid: str) -> None:
        """
        Delete pipeline manifest (artifacts remain for garbage collection).

        Args:
            uid: Pipeline UID
        """
        # Load manifest to get dataset and name
        try:
            manifest = self.load_manifest(uid)
            dataset = manifest.get("dataset")
            name = manifest.get("name")

            # Remove from dataset index
            if dataset and name:
                self.unregister_from_dataset(dataset, name)
        except FileNotFoundError:
            pass  # Manifest already gone

        # Delete pipeline directory
        pipeline_dir = self.pipelines_dir / uid
        if pipeline_dir.exists():
            shutil.rmtree(pipeline_dir)

    def pipeline_exists(self, uid: str) -> bool:
        """
        Check if a pipeline exists.

        Args:
            uid: Pipeline UID

        Returns:
            True if manifest exists
        """
        manifest_path = self.pipelines_dir / uid / "manifest.yaml"
        return manifest_path.exists()

    def get_pipeline_path(self, uid: str) -> Path:
        """
        Get the directory path for a pipeline.

        Args:
            uid: Pipeline UID

        Returns:
            Path to pipeline directory
        """
        return self.pipelines_dir / uid

    def list_all_pipelines(self) -> List[Dict[str, Any]]:
        """
        List all pipelines across all datasets.

        Returns:
            List of pipeline info dictionaries (uid, name, dataset, created_at)
        """
        pipelines = []

        if not self.pipelines_dir.exists():
            return pipelines

        for pipeline_dir in self.pipelines_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue

            manifest_path = pipeline_dir / "manifest.yaml"
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = yaml.safe_load(f)

                    pipelines.append({
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
