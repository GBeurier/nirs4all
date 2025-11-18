"""Artifact manager for handling pipeline artifact persistence."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nirs4all.pipeline.storage.artifacts import artifact_persistence
from nirs4all.pipeline.storage.artifacts.artifact_persistence import ArtifactMeta
from nirs4all.pipeline.storage.manifest_manager import ManifestManager


class ArtifactManager:
    """Manages artifact persistence and retrieval.

    Handles:
    - Binary serialization/deserialization
    - Content-addressed storage
    - Manifest integration
    - Loading for predict mode

    Attributes:
        artifacts_dir: Artifact storage directory
        manifest_manager: Manifest manager for tracking artifacts
    """

    def __init__(
        self,
        artifacts_dir: Path,
        manifest_manager: Optional[ManifestManager] = None
    ):
        """Initialize artifact manager.

        Args:
            artifacts_dir: Directory for storing artifacts
            manifest_manager: Optional manifest manager for tracking
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_manager = manifest_manager

    def persist(
        self,
        artifact: Any,
        name: str,
        step: int,
        format_hint: Optional[str] = None
    ) -> ArtifactMeta:
        """Persist an artifact to storage.

        Args:
            artifact: Artifact to persist (model, transformer, etc.)
            name: Artifact identifier (e.g., 'model', 'scaler')
            step: Pipeline step number
            format_hint: Optional format hint for serialization

        Returns:
            ArtifactMeta with persistence information

        Raises:
            ValueError: If artifact cannot be serialized
        """
        # Save artifact using artifact_persistence module
        meta = artifact_persistence.persist(
            obj=artifact,
            artifacts_dir=self.artifacts_dir,
            name=name,
            format_hint=format_hint
        )
        # Add step number
        meta['step'] = step

        return meta

    def load(self, meta: ArtifactMeta) -> Any:
        """Load an artifact from storage.

        Args:
            meta: Artifact metadata

        Returns:
            Loaded artifact

        Raises:
            FileNotFoundError: If artifact file doesn't exist
            ValueError: If artifact cannot be deserialized
        """
        artifact_path = self.artifacts_dir / meta['path']

        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        # Load using artifact_persistence module
        with open(artifact_path, 'rb') as f:
            data = f.read()

        artifact = artifact_persistence.from_bytes(data, format=meta['format'])

        return artifact

    def load_for_step(self, step_number: int, pipeline_uid: str) -> List[Tuple[str, Any]]:
        """Load all artifacts for a specific pipeline step.

        Args:
            step_number: Step number to load artifacts for
            pipeline_uid: Pipeline unique identifier

        Returns:
            List of (name, artifact) tuples

        Raises:
            ValueError: If manifest not found or step doesn't exist
        """
        if not self.manifest_manager:
            raise ValueError("Manifest manager not configured")

        # Load manifest
        manifest = self.manifest_manager.load_manifest(pipeline_uid)

        # Get artifacts for this step
        step_artifacts = [
            artifact for artifact in manifest.get('artifacts', [])
            if artifact.get('step') == step_number
        ]

        # Load each artifact
        loaded = []
        for artifact_meta in step_artifacts:
            try:
                artifact = self.load(artifact_meta)
                loaded.append((artifact_meta['name'], artifact))
            except Exception as e:
                # Log warning but continue - some artifacts may be optional
                import warnings
                warnings.warn(
                    f"Failed to load artifact {artifact_meta['name']} "
                    f"for step {step_number}: {e}"
                )

        return loaded

    def get_artifact_path(self, meta: ArtifactMeta) -> Path:
        """Get full path to an artifact.

        Args:
            meta: Artifact metadata

        Returns:
            Full path to artifact file
        """
        return self.artifacts_dir / meta['path']

    def artifact_exists(self, meta: ArtifactMeta) -> bool:
        """Check if an artifact exists in storage.

        Args:
            meta: Artifact metadata

        Returns:
            True if artifact file exists
        """
        return self.get_artifact_path(meta).exists()
