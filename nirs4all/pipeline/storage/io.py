"""
Simulation IO Manager - Save and manage simulation outputs

Provides organized storage for pipeline simulation results with
dataset and pipeline-based folder structure management.

REFACTORED: Now uses content-addressed artifact storage via serializer.
Delegates to focused classes: PipelineWriter, WorkspaceExporter, PredictionResolver.
"""
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid
import shutil

from nirs4all.pipeline.storage.io_writer import PipelineWriter
from nirs4all.pipeline.storage.io_exporter import WorkspaceExporter
from nirs4all.pipeline.storage.io_resolver import PredictionResolver


def _get_default_workspace_path() -> Path:
    """Get the default workspace path.

    Checks NIRS4ALL_WORKSPACE environment variable first, then falls back
    to ./workspace in the current working directory.

    Returns:
        Default workspace path.
    """
    env_workspace = os.environ.get("NIRS4ALL_WORKSPACE")
    if env_workspace:
        return Path(env_workspace)
    return Path.cwd() / "workspace"


class SimulationSaver:
    """
    Manages saving simulation results with flat pipeline structure.

    Acts as a facade coordinating:
    - PipelineWriter: File I/O within pipeline directories
    - WorkspaceExporter: Exporting best results
    - PredictionResolver: Resolving prediction targets

    Works with ManifestManager to create: base_path/NNNN_hash/files
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None, save_artifacts: bool = True, save_charts: bool = True):
        """
        Initialize the simulation saver.

        Args:
            base_path: Base directory (run directory: workspace/runs/YYYY-MM-DD_dataset/)
            save_artifacts: Whether to save binary artifacts (models, transformers)
            save_charts: Whether to save charts and visual outputs
        """
        self.base_path = Path(base_path) if base_path is not None else None
        self.pipeline_id: Optional[str] = None  # e.g., "0001_abc123"
        self.pipeline_dir: Optional[Path] = None
        self._metadata: Dict[str, Any] = {}
        self.save_artifacts = save_artifacts
        self.save_charts = save_charts

        # Delegate components (created when needed)
        self._writer: Optional[PipelineWriter] = None
        self._exporter: Optional[WorkspaceExporter] = None
        self._resolver: Optional[PredictionResolver] = None

    @property
    def writer(self) -> PipelineWriter:
        """Get or create PipelineWriter instance."""
        if self._writer is None:
            if self.pipeline_dir is None:
                raise RuntimeError("Must call register() before accessing writer")
            self._writer = PipelineWriter(self.pipeline_dir, self.save_charts)
        return self._writer

    @property
    def exporter(self) -> WorkspaceExporter:
        """Get or create WorkspaceExporter instance."""
        if self._exporter is None:
            workspace_path = self.base_path.parent.parent if self.base_path else _get_default_workspace_path()
            self._exporter = WorkspaceExporter(workspace_path)
        return self._exporter

    @property
    def resolver(self) -> PredictionResolver:
        """Get or create PredictionResolver instance."""
        if self._resolver is None:
            workspace_path = self.base_path.parent.parent if self.base_path else _get_default_workspace_path()
            self._resolver = PredictionResolver(workspace_path)
        return self._resolver

    def register(self, pipeline_id: str) -> Path:
        """
        Register a pipeline ID and set current directory.

        Args:
            pipeline_id: Pipeline ID from ManifestManager (e.g., "0001_abc123")

        Returns:
            Path to the pipeline directory
        """
        self.pipeline_id = pipeline_id
        self.pipeline_dir = self.base_path / pipeline_id

        # Directory should already exist from ManifestManager.create_pipeline()
        if not self.pipeline_dir.exists():
            self.pipeline_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        self._metadata = {
            "pipeline_id": pipeline_id,
            "created_at": datetime.now().isoformat(),
            "session_id": str(uuid.uuid4()),
            "files": {},
            "binaries": {}
        }

        # Reset writer to ensure it uses the new pipeline directory
        self._writer = None

        return self.pipeline_dir

    def get_predict_targets(self, prediction_obj: Union[Dict[str, Any], str]):
        """Get target variable names for prediction from a prediction object.

        Delegates to PredictionResolver.
        """
        return self.resolver.resolve_target(prediction_obj)

    def _find_prediction_by_id(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Search for a prediction by ID in global predictions databases.

        Delegates to PredictionResolver.
        """
        return self.resolver.find_prediction_by_id(prediction_id)


    def save_file(self,
                  filename: str,
                  content: str,
                  overwrite: bool = True,
                  encoding: str = 'utf-8',
                  warn_on_overwrite: bool = True) -> Path:
        """Save a text file to the pipeline directory.

        Delegates to PipelineWriter.
        """
        return self.writer.save_file(filename, content, overwrite, encoding, warn_on_overwrite)

    def save_json(self,
                  filename: str,
                  data: Any,
                  overwrite: bool = True,
                  indent: Optional[int] = 2) -> Path:
        """Save data as JSON file.

        Delegates to PipelineWriter.
        """
        return self.writer.save_json(filename, data, overwrite, indent)

    def persist_artifact(
        self,
        step_number: int,
        name: str,
        obj: Any,
        format_hint: Optional[str] = None,
        branch_id: Optional[int] = None,
        branch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Persist artifact using the serializer with content-addressed storage.

        NOTE: This is for internal binary artifacts (models, transformers, etc.)
        For human-readable outputs (charts, reports), use save_output() instead.

        Args:
            step_number: Pipeline step number
            name: Artifact name (for reference)
            obj: Object to persist
            format_hint: Optional format hint for serializer
            branch_id: Optional branch ID for pipeline branching
            branch_name: Optional human-readable branch name

        Returns:
            Artifact metadata dictionary (empty if save_artifacts=False)
        """
        # Skip if save_artifacts is disabled
        if not self.save_artifacts:
            return {
                "name": name,
                "step": step_number,
                "branch_id": branch_id,
                "branch_name": branch_name,
                "skipped": True,
                "reason": "save_artifacts=False"
            }

        from nirs4all.pipeline.storage.artifacts.artifact_persistence import persist

        self._check_registered()

        # Use _binaries directory (managed by ManifestManager)
        artifacts_dir = self.base_path / "_binaries"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Persist using new serializer with branch info
        artifact = persist(
            obj,
            artifacts_dir,
            name,
            format_hint,
            branch_id=branch_id,
            branch_name=branch_name
        )
        artifact["step"] = step_number

        # Note: metadata tracking removed - using manifest system now

        return artifact

    def save_output(
        self,
        step_number: int,
        name: str,
        data: Union[bytes, str],
        extension: str = ".png"
    ) -> Optional[Path]:
        """
        Save a human-readable output file (chart, report, etc.).

        Delegates to PipelineWriter.

        Args:
            step_number: Pipeline step number (unused, kept for compatibility)
            name: Output name (e.g., "2D_Chart")
            data: Binary or text data to save
            extension: File extension (e.g., ".png", ".csv", ".txt")

        Returns:
            Path to saved file, or None if save_charts=False
        """
        return self.writer.save_output(name, data, extension)

    def get_path(self) -> Path:
        """Get the current pipeline path.

        Delegates to PipelineWriter.
        """
        return self.writer.get_path()

    def list_files(self) -> Dict[str, List[str]]:
        """
        List all saved files in the current pipeline.

        Returns:
            Dictionary with file lists
        """
        self._check_registered()

        return {
            "files": list(self._metadata["files"].keys()),
            "binaries": list(self._metadata["binaries"].keys()),
            "all_files": self.writer.list_files()
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get the current metadata."""
        return self._metadata.copy()

    def cleanup(self, confirm: bool = False) -> None:
        """
        Remove the current simulation directory and all its contents.

        Args:
            confirm: Must be True to actually delete files

        Raises:
            RuntimeError: If not registered or confirm is False
        """
        self._check_registered()

        if not confirm:
            raise RuntimeError("cleanup() requires confirm=True to prevent accidental deletion")

        if self.pipeline_dir.exists():
            shutil.rmtree(self.pipeline_dir)
            warnings.warn(f"Deleted simulation directory: {self.pipeline_dir}")

    def _check_registered(self) -> None:
        """Check if pipeline is registered."""
        if self.pipeline_dir is None:
            raise RuntimeError("Must call register() before saving files")

    def _is_valid_name(self, name: str) -> bool:
        """Check if name is valid for filesystem use."""
        if not name or not isinstance(name, str):
            return False

        # Check for invalid characters
        invalid_chars = set('<>:"/\\|?*')
        if any(char in invalid_chars for char in name):
            return False

        # Check for reserved names (Windows)
        reserved_names = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3',
                         'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
                         'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6',
                         'LPT7', 'LPT8', 'LPT9'}
        if name.upper() in reserved_names:
            return False

        return True

    def register_workspace(self, workspace_root: Path, dataset_name: str, pipeline_hash: str,
                          run_name: str = None, pipeline_name: str = None) -> Path:
        """
        Register pipeline in workspace structure with optional custom names.

        Creates:
        - Without custom names: workspace_root/runs/{dataset}/NNNN_{hash}/
        - With run_name: workspace_root/runs/{dataset}_{runname}/NNNN_{hash}/
        - With pipeline_name: workspace_root/runs/{dataset}/NNNN_{pipelinename}_{hash}/
        - With both: workspace_root/runs/{dataset}_{runname}/NNNN_{pipelinename}_{hash}/

        All pipelines for a dataset are stored in the same folder regardless of date.

        Returns:
            Full path to pipeline directory
        """
        from datetime import datetime

        # Build run_id with optional custom name (no date prefix)
        if run_name:
            run_id = f"{dataset_name}_{run_name}"
        else:
            run_id = dataset_name
        run_dir = workspace_root / "runs" / run_id

        # Count existing pipelines for sequential numbering
        # Exclude directories starting with underscore (like _binaries)
        if run_dir.exists():
            existing = [d for d in run_dir.iterdir()
                       if d.is_dir() and not d.name.startswith("_")]
            pipeline_num = len(existing) + 1
        else:
            pipeline_num = 1

        # Build pipeline_id with optional custom name
        if pipeline_name:
            pipeline_id = f"{pipeline_num:04d}_{pipeline_name}_{pipeline_hash}"
        else:
            pipeline_id = f"{pipeline_num:04d}_{pipeline_hash}"
        pipeline_dir = run_dir / pipeline_id

        # Create pipeline directory only (not _binaries - created lazily when artifacts are saved)
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        # Update internal state to use this directory
        self.dataset_name = dataset_name
        self.pipeline_name = pipeline_id
        self.current_path = pipeline_dir
        self.dataset_path = run_dir

        # Initialize metadata
        self._metadata = {
            "dataset_name": dataset_name,
            "pipeline_name": pipeline_id,
            "created_at": datetime.now().isoformat(),
            "session_id": run_id,
            "files": {},
            "binaries": {}
        }

        return pipeline_dir

    def export_pipeline_full(self, pipeline_dir: Path, exports_dir: Path,
                            dataset_name: str, run_date: str, custom_name: str = None) -> Path:
        """Export full pipeline results to flat structure with optional custom name.

        Delegates to WorkspaceExporter.

        Args:
            pipeline_dir: Path to pipeline (NNNN_hash/ or NNNN_pipelinename_hash/)
            exports_dir: Workspace exports directory (unused, maintained for compatibility)
            dataset_name: Dataset name
            run_date: Run date (YYYYMMDD)
            custom_name: Optional custom name for export

        Returns: Path to exported directory
        """
        return self.exporter.export_pipeline_full(pipeline_dir, dataset_name, run_date, custom_name)

    def export_best_prediction(self, predictions_file: Path, exports_dir: Path,
                              dataset_name: str, run_date: str, pipeline_id: str,
                              custom_name: str = None) -> Path:
        """Export predictions CSV to best_predictions/ folder with optional custom name.

        Delegates to WorkspaceExporter.

        Args:
            predictions_file: Path to predictions.csv
            exports_dir: Workspace exports directory (unused, maintained for compatibility)
            dataset_name, run_date, pipeline_id: Metadata for naming
            custom_name: Optional custom name for export

        Returns: Path to exported CSV
        """
        return self.exporter.export_best_prediction(
            predictions_file, dataset_name, run_date, pipeline_id, custom_name
        )

    def export_best_for_dataset(self, dataset_name: str, workspace_path: Path,
                                runs_dir: Path, mode: str = "predictions") -> Optional[Path]:
        """Export best results for a dataset to exports/ folder.

        Delegates to WorkspaceExporter.

        Creates exports/{dataset_name}/ with best predictions, pipeline config, and charts.
        Files are renamed to include run date for tracking.

        Args:
            dataset_name: Dataset name (matches global prediction JSON filename)
            workspace_path: Workspace root path (unused, maintained for compatibility)
            runs_dir: Runs directory path
            mode: Export mode - "predictions", "template", "trained", or "full"

        Returns:
            Path to export directory, or None if no predictions found
        """
        return self.exporter.export_best_for_dataset(dataset_name, runs_dir, mode)



