"""
Binary Loader - Manages loading and caching of saved pipeline binaries

REFACTORED: Now loads from manifest-based artifact metadata using the new serializer.

This module provides functionality to load and cache binaries saved during
pipeline execution for use in prediction mode. It handles efficient loading
and memory management of fitted transformers and trained models.

Supports branch-aware loading for pipeline branching (Phase 2).
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

    Supports branch-aware artifact loading for pipeline branching:
    - Artifacts without branch_id are shared (pre-branch)
    - Artifacts with branch_id are specific to that branch

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
        self.artifacts_by_step_branch: Dict[Tuple[int, Optional[int]], List[Dict[str, Any]]] = {}
        self._cache: Dict[str, Any] = {}

        # Group artifacts by step number and by (step, branch_id)
        for artifact in artifacts:
            step = artifact.get("step", -1)
            branch_id = artifact.get("branch_id")  # May be None for pre-branch artifacts

            # Group by step (backward compatible)
            if step not in self.artifacts_by_step:
                self.artifacts_by_step[step] = []
            self.artifacts_by_step[step].append(artifact)

            # Group by (step, branch_id) for branch-aware loading
            key = (step, branch_id)
            if key not in self.artifacts_by_step_branch:
                self.artifacts_by_step_branch[key] = []
            self.artifacts_by_step_branch[key].append(artifact)

    def get_step_binaries(
        self,
        step_id: str,
        branch_id: Optional[int] = None
    ) -> List[Tuple[str, Any]]:
        """
        Load binary artifacts for a specific step ID and optional branch.

        For branched pipelines:
        - If branch_id is None: returns shared (pre-branch) artifacts for this step
        - If branch_id is provided: returns artifacts specific to that branch,
          falling back to shared artifacts if no branch-specific ones exist

        Args:
            step_id: Step identifier in format "step_substep" (e.g., "0_0", "1_0")
                     Also supports just step number as int or str (e.g., 0, "0")
            branch_id: Optional branch ID (None for pre-branch steps or to get all)

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

        # Build cache key including branch_id
        cache_key = f"step_{step_num}_branch_{branch_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try to find artifacts with priority:
        # 1. Branch-specific artifacts (step, branch_id)
        # 2. Shared artifacts (step, None)
        # 3. All artifacts for step (backward compatibility)

        artifacts = []

        if branch_id is not None:
            # First try branch-specific
            key = (step_num, branch_id)
            if key in self.artifacts_by_step_branch:
                artifacts = self.artifacts_by_step_branch[key]
            else:
                # Fall back to shared (pre-branch) artifacts
                key = (step_num, None)
                artifacts = self.artifacts_by_step_branch.get(key, [])
        else:
            # No branch specified - get shared artifacts or all for backward compat
            key = (step_num, None)
            if key in self.artifacts_by_step_branch:
                artifacts = self.artifacts_by_step_branch[key]
            elif step_num in self.artifacts_by_step:
                # Backward compatibility: get all artifacts for this step
                artifacts = self.artifacts_by_step[step_num]

        if not artifacts:
            return []

        loaded_binaries = []

        for artifact in artifacts:
            try:
                # Load using new serializer
                obj = load(artifact, self.results_dir)
                name = artifact.get("name", "unknown")
                loaded_binaries.append((name, obj))

            except FileNotFoundError:
                artifact_branch = artifact.get("branch_id")
                artifact_name = artifact.get("name", "unknown")
                warnings.warn(
                    f"Artifact file not found: {artifact.get('path', 'unknown')} "
                    f"(name={artifact_name}, branch_id={artifact_branch}). Skipping."
                )
                continue
            except (ValueError, IOError, OSError) as e:
                warnings.warn(f"Failed to load artifact {artifact.get('name', 'unknown')}: {e}. Skipping.")
                continue

        # Cache the loaded binaries
        self._cache[cache_key] = loaded_binaries

        return loaded_binaries

    def get_branch_binaries(
        self,
        step_num: int,
        branch_id: int
    ) -> List[Tuple[str, Any]]:
        """
        Load binary artifacts for a specific step and branch.

        Convenience method that combines branch-specific and shared artifacts
        for a given branch at a specific step.

        Args:
            step_num: Pipeline step number
            branch_id: Branch ID to load artifacts for

        Returns:
            List of (name, loaded_object) tuples
        """
        return self.get_step_binaries(step_num, branch_id=branch_id)

    def has_binaries_for_step(
        self,
        step_number: int,
        substep_number: Optional[int] = None,
        branch_id: Optional[int] = None
    ) -> bool:
        """
        Check if binaries exist for a specific step.

        Args:
            step_number: The main step number
            substep_number: The substep number (ignored in new architecture)
            branch_id: Optional branch ID to check for branch-specific artifacts

        Returns:
            True if binaries exist for this step (and optionally branch)
        """
        if branch_id is not None:
            # Check for branch-specific artifacts
            key = (step_number, branch_id)
            if key in self.artifacts_by_step_branch:
                return len(self.artifacts_by_step_branch[key]) > 0
            # Fall back to checking shared artifacts
            key = (step_number, None)
            return key in self.artifacts_by_step_branch and len(self.artifacts_by_step_branch[key]) > 0

        # No branch specified - check step-level
        return step_number in self.artifacts_by_step and len(self.artifacts_by_step[step_number]) > 0

    def get_available_branches(self, step_number: int) -> List[Optional[int]]:
        """
        Get list of branch IDs that have artifacts at a specific step.

        Args:
            step_number: Pipeline step number

        Returns:
            List of branch IDs (may include None for shared artifacts)
        """
        branches = set()
        for (step, branch_id) in self.artifacts_by_step_branch.keys():
            if step == step_number:
                branches.add(branch_id)
        return sorted(branches, key=lambda x: (x is None, x))

    def get_artifacts_for_branch(self, branch_id: Optional[int]) -> List[Dict[str, Any]]:
        """
        Get all artifact metadata for a specific branch.

        Args:
            branch_id: Branch ID to get artifacts for (None for shared/pre-branch artifacts)

        Returns:
            List of artifact metadata dictionaries for the specified branch
        """
        result = []
        for (step, bid), artifacts in self.artifacts_by_step_branch.items():
            if bid == branch_id:
                result.extend(artifacts)
        return result

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
            "available_step_branches": list(self.artifacts_by_step_branch.keys()),
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
