"""
Binary Loader - Manages loading and caching of saved pipeline binaries

REFACTORED: Now loads from manifest-based artifact metadata using the new serializer.

This module provides functionality to load and cache binaries saved during
pipeline execution for use in prediction mode. It handles efficient loading
and memory management of fitted transformers and trained models.
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import warnings


class BinaryLoader:
    """
    Manages loading and caching of saved pipeline binaries from manifest artifacts.

    This class handles the loading of saved fitted transformers and trained models
    from content-addressed storage, providing efficient caching to avoid redundant
    file I/O operations during prediction mode execution.

    REFACTORED: Now works with manifest artifact metadata instead of direct file paths.
    """

    def __init__(self, artifacts: List[Dict[str, Any]], results_dir: Path):
        """
        Initialize the binary loader with artifacts from manifest.

        Args:
            artifacts: List of artifact metadata dictionaries from manifest
            results_dir: Path to results directory (contains artifacts/ subdirectory)
        """
        self.results_dir = Path(results_dir)
        self.artifacts_by_step: Dict[int, List[Dict[str, Any]]] = {}
        self._cache: Dict[str, Any] = {}

        # Group artifacts by step number
        for artifact in artifacts:
            step = artifact.get("step", -1)
            if step not in self.artifacts_by_step:
                self.artifacts_by_step[step] = []
            self.artifacts_by_step[step].append(artifact)

    def get_step_binaries(self, step_id: str) -> List[Tuple[str, Any]]:
        """
        Load binary artifacts for a specific step ID using the new serializer.

        Args:
            step_id: Step identifier in format "step_substep" (e.g., "0_0", "1_0")
                     Also supports just step number as int or str (e.g., 0, "0")

        Returns:
            List of (name, loaded_object) tuples
        """
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import load

        # Parse step_id - handle both "step_substep" format and simple step number
        if isinstance(step_id, int):
            step_num = step_id
        elif "_" in str(step_id):
            step_num = int(str(step_id).split("_")[0])
        else:
            step_num = int(step_id)

        # Check cache first
        cache_key = f"step_{step_num}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # No artifacts for this step
        if step_num not in self.artifacts_by_step:
            return []

        artifacts = self.artifacts_by_step[step_num]
        loaded_binaries = []

        for artifact in artifacts:
            try:
                # Load using new serializer
                obj = load(artifact, self.results_dir)
                name = artifact.get("name", "unknown")
                loaded_binaries.append((name, obj))

            except FileNotFoundError:
                warnings.warn(f"Artifact file not found: {artifact.get('path', 'unknown')}. Skipping.")
                continue
            except (ValueError, IOError, OSError) as e:
                warnings.warn(f"Failed to load artifact {artifact.get('name', 'unknown')}: {e}. Skipping.")
                continue

        # Cache the loaded binaries
        self._cache[cache_key] = loaded_binaries

        return loaded_binaries

    def has_binaries_for_step(self, step_number: int, substep_number: Optional[int] = None) -> bool:
        """
        Check if binaries exist for a specific step.

        Args:
            step_number: The main step number
            substep_number: The substep number (ignored in new architecture)

        Returns:
            True if binaries exist for this step
        """
        return step_number in self.artifacts_by_step and len(self.artifacts_by_step[step_number]) > 0

    def clear_cache(self) -> None:
        """Clear the binary cache to free memory."""
        self._cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_steps": list(self._cache.keys()),
            "cache_size": len(self._cache),
            "available_steps": list(self.artifacts_by_step.keys()),
            "total_available_artifacts": sum(len(artifacts) for artifacts in self.artifacts_by_step.values())
        }

    @classmethod
    def from_manifest(cls, manifest: Dict[str, Any], results_dir: Path) -> 'BinaryLoader':
        """
        Create a BinaryLoader from a pipeline manifest.

        Args:
            manifest: Pipeline manifest dictionary
            results_dir: Path to results directory

        Returns:
            Initialized BinaryLoader instance
        """
        artifacts = manifest.get("artifacts", [])
        return cls(artifacts, results_dir)
