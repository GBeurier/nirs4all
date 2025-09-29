"""
Binary Loader - Manages loading and caching of saved pipeline binaries

This module provides functionality to load and cache binaries saved during
pipeline execution for use in prediction mode. It handles efficient loading
and memory management of fitted transformers and trained models.
"""

import pickle
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import warnings


class BinaryLoader:
    """
    Manages loading and caching of saved pipeline binaries.

    This class handles the loading of saved fitted transformers and trained models
    from disk, providing efficient caching to avoid redundant file I/O operations
    during prediction mode execution.
    """

    def __init__(self, simulation_path: Path, step_binaries: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the binary loader.

        Args:
            simulation_path: Path to the saved pipeline directory
            step_binaries: Mapping of step IDs to binary filenames
        """
        self.simulation_path = Path(simulation_path)
        self.step_binaries = step_binaries or {}
        self._cache: Dict[str, List[Any]] = {}
        self._validate_simulation_path()

    def _validate_simulation_path(self) -> None:
        """Validate that the simulation path exists and contains required files."""
        if not self.simulation_path.exists():
            raise FileNotFoundError(f"Simulation path does not exist: {self.simulation_path}")

        pipeline_json_path = self.simulation_path / "pipeline.json"
        if not pipeline_json_path.exists():
            raise FileNotFoundError(f"Pipeline configuration not found: {pipeline_json_path}")

        # Load step binaries from pipeline.json if not provided
        if not self.step_binaries:
            try:
                with open(pipeline_json_path, 'r') as f:
                    pipeline_data = json.load(f)

                # Handle both old and new format
                if "execution_metadata" in pipeline_data:
                    self.step_binaries = pipeline_data["execution_metadata"].get("step_binaries", {})
                else:
                    # Old format - no metadata, warn user
                    warnings.warn(
                        f"Pipeline at {self.simulation_path} was saved without binary metadata. "
                        "This pipeline needs to be re-run in training mode to support prediction. "
                        "Use save_files=True when training to enable prediction mode.",
                        UserWarning
                    )
                    self.step_binaries = {}

            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Invalid pipeline configuration: {e}")

    def get_binaries_for_step(self, step_number: int, substep_number: int) -> List[Any]:
        """
        Load binaries for a specific step.

        Args:
            step_number: The main step number
            substep_number: The substep number

        Returns:
            List of loaded binary objects (fitted transformers, trained models, etc.)

        Raises:
            FileNotFoundError: If binary files cannot be found
            RuntimeError: If binary files cannot be loaded
        """
        step_id = f"{step_number}_{substep_number}"

        # Check cache first
        if step_id in self._cache:
            return self._cache[step_id]

        # Load binaries if not cached
        self._cache[step_id] = self._load_step_binaries(step_id)
        return self._cache[step_id]

    def _load_step_binaries(self, step_id: str) -> List[Any]:
        """
        Load binary files for a specific step ID.

        Args:
            step_id: Step identifier in format "step_substep"

        Returns:
            List of loaded binary objects
        """
        if step_id not in self.step_binaries:
            # No binaries for this step - this is normal for some steps
            return []

        binary_filenames = self.step_binaries[step_id]
        loaded_binaries = []

        for filename in binary_filenames:
            binary_path = self.simulation_path / filename

            if not binary_path.exists():
                warnings.warn(f"Binary file not found: {binary_path}. Skipping.")
                continue

            try:
                # Handle different file types appropriately
                if filename.endswith('.csv'):
                    # Load CSV files as text
                    with open(binary_path, 'r', encoding='utf-8') as f:
                        csv_content = f.read()
                    loaded_binaries.append((filename, csv_content))
                elif filename.endswith(('.json', '.txt')):
                    # Load text files as strings
                    with open(binary_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    loaded_binaries.append((filename, text_content))
                elif filename.endswith(('.pkl', '.pickle')):
                    # Load pickle files as objects
                    with open(binary_path, 'rb') as f:
                        binary_obj = pickle.load(f)
                    loaded_binaries.append((filename, binary_obj))
                else:
                    # Default to pickle for other files (backward compatibility)
                    with open(binary_path, 'rb') as f:
                        binary_obj = pickle.load(f)
                    loaded_binaries.append((filename, binary_obj))

            except Exception as e:
                # Skip problematic files with a warning instead of failing completely
                warnings.warn(f"Failed to load binary file {binary_path}: {e}. Skipping.")
                continue

        return loaded_binaries

    def has_binaries_for_step(self, step_number: int, substep_number: int) -> bool:
        """
        Check if binaries exist for a specific step.

        Args:
            step_number: The main step number
            substep_number: The substep number

        Returns:
            True if binaries exist for this step
        """
        step_id = f"{step_number}_{substep_number}"
        return step_id in self.step_binaries and len(self.step_binaries[step_id]) > 0

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
            "available_steps": list(self.step_binaries.keys()),
            "total_available_binaries": sum(len(binaries) for binaries in self.step_binaries.values())
        }

    @classmethod
    def from_pipeline_path(cls, pipeline_path: Path) -> 'BinaryLoader':
        """
        Create a BinaryLoader from a pipeline directory path.

        Args:
            pipeline_path: Path to the saved pipeline directory

        Returns:
            Initialized BinaryLoader instance
        """
        return cls(pipeline_path)