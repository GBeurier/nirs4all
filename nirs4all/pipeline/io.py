"""
Simulation IO Manager - Save and manage simulation outputs

Provides organized storage for pipeline simulation results with
dataset and pipeline-based folder structure management.

REFACTORED: Now uses content-addressed artifact storage via serializer.
"""
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO, Tuple
import uuid
import shutil
from nirs4all.dataset.predictions import Predictions


class SimulationSaver:
    """
    Manages saving simulation results with organized folder structure.

    Creates and manages directory structure: base_path/dataset_name/pipeline_name/
    Provides methods to save files, binaries, and metadata with overwrite protection.
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = "simulations", save_files: bool = True):
        """
        Initialize the simulation saver.

        Args:
            base_path: Base directory for all simulation outputs
            save_files: Whether to actually save files (can disable for dry runs)
        """
        self.base_path = Path(base_path) if base_path is not None else Path("results")
        self.dataset_name: Optional[str] = None
        self.pipeline_name: Optional[str] = None
        self.current_path: Optional[Path] = None
        self._metadata: Dict[str, Any] = {}
        self.dataset_path: Optional[Path] = None
        self.save_files = save_files

    def register(self, dataset_name: str, pipeline_name: str, mode: str) -> Path:
        """
        Register a dataset and pipeline name, creating the directory structure.

        Args:
            dataset_name: Name of the dataset
            pipeline_name: Name of the pipeline

        Returns:
            Path to the created simulation directory

        Raises:
            ValueError: If names contain invalid characters
        """
        # Validate names
        if not self._is_valid_name(dataset_name):
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        if not self._is_valid_name(pipeline_name):
            raise ValueError(f"Invalid pipeline name: {pipeline_name}")

        self.dataset_name = dataset_name
        self.pipeline_name = pipeline_name

        # Create directory structure
        self.dataset_path = self.base_path / dataset_name
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.current_path = self.base_path / dataset_name / pipeline_name
        if mode != "predict" and mode != "explain":
            self.current_path.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        self._metadata = {
            "dataset_name": dataset_name,
            "pipeline_name": pipeline_name,
            "created_at": datetime.now().isoformat(),
            "session_id": str(uuid.uuid4()),
            "files": {},
            "binaries": {}
        }

        return self.current_path

    def _find_prediction_by_id(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Search for a prediction by ID in all predictions.json files (recursively)."""
        results_dir = Path(self.base_path)
        if not results_dir.exists():
            return None

        for predictions_file in results_dir.rglob("predictions.json"):
            if not predictions_file.is_file():
                continue

            try:
                predictions = Predictions.load_from_file_cls(str(predictions_file))
                for pred in predictions.filter_predictions():
                    if pred.get('id') == prediction_id:
                        return pred
            except Exception:
                continue

        return None


    def get_predict_targets(self, prediction_obj: Union[Dict[str, Any], str]) :
        """Get target variable names for prediction from a prediction object."""
        targets = []
        # 1. Resolve input to get config path and model info
        if isinstance(prediction_obj, dict):
            config_path = prediction_obj['config_path']
            target_model = prediction_obj if 'model_name' in prediction_obj else None
            return config_path, target_model
        elif isinstance(prediction_obj, str):
            if prediction_obj.startswith(str(self.base_path)) or Path(prediction_obj).exists():
                # Config path
                config_path = prediction_obj.replace(str(self.base_path), '')
                target_model = None  # TODO get the best model from this config path (retrieve from predictions)
                return config_path, target_model
            else:
                # Prediction ID - find it
                target_model = self._find_prediction_by_id(prediction_obj)
                if not target_model:
                    raise ValueError(f"Prediction ID not found: {prediction_obj}")
                config_path = target_model['config_path']
                return config_path, target_model
        else:
            raise ValueError(f"Invalid prediction_obj type: {type(prediction_obj)}")


    def save_file(self,
                  filename: str,
                  content: str,
                  overwrite: bool = True,
                  encoding: str = 'utf-8',
                  warn_on_overwrite: bool = True,
                  into_dataset: bool = False) -> Path:

        self._check_registered()

        filepath = self.current_path / filename
        if into_dataset and self.dataset_path is not None:
            filepath = self.dataset_path / filename

        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File {filename} already exists. Use overwrite=True to replace.")

        if filepath.exists() and warn_on_overwrite:
            warnings.warn(f"Overwriting existing file: {filename}")

        # Save content
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)

        # Note: metadata tracking removed - using manifest system now

        return filepath

    def save_json(self,
                  filename: str,
                  data: Any,
                  overwrite: bool = True,
                  indent: Optional[int] = 2) -> Path:
        if not filename.endswith('.json'):
            filename += '.json'

        json_content = json.dumps(data, indent=indent, default=str)
        return self.save_file(filename, json_content, overwrite, warn_on_overwrite=False)

    def persist_artifact(
        self,
        step_number: int,
        name: str,
        obj: Any,
        format_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Persist artifact using the new serializer with content-addressed storage.

        NOTE: This is for internal binary artifacts (models, transformers, etc.)
        For human-readable outputs (charts, reports), use save_output() instead.

        Args:
            step_number: Pipeline step number
            name: Artifact name (for reference)
            obj: Object to persist
            format_hint: Optional format hint for serializer

        Returns:
            Artifact metadata dictionary (empty if save_files=False)
        """
        # Skip if save_files is disabled
        if not self.save_files:
            return {
                "name": name,
                "step": step_number,
                "skipped": True,
                "reason": "save_files=False"
            }

        from nirs4all.utils.serializer import persist

        self._check_registered()

        # Create artifacts directory structure
        artifacts_dir = self.base_path / "artifacts" / "objects"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Persist using new serializer
        artifact = persist(obj, artifacts_dir, name, format_hint)
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
        Save a human-readable output file (chart, report, etc.) to the outputs directory.

        Outputs are organized as: base_path/outputs/<dataset>_<pipeline>/<name>

        Args:
            step_number: Pipeline step number
            name: Output name (e.g., "2D_Chart")
            data: Binary or text data to save
            extension: File extension (e.g., ".png", ".csv", ".txt")

        Returns:
            Path to saved file, or None if save_files=False
        """
        # Skip if save_files is disabled
        if not self.save_files:
            return None

        self._check_registered()

        # Create outputs directory structure
        output_dir = self.base_path / "outputs" / f"{self.dataset_name}_{self.pipeline_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename
        if not name.endswith(extension):
            filename = f"{name}{extension}"
        else:
            filename = name

        filepath = output_dir / filename

        # Save the file
        if isinstance(data, bytes):
            filepath.write_bytes(data)
        elif isinstance(data, str):
            filepath.write_text(data, encoding="utf-8")
        else:
            raise TypeError(f"Data must be bytes or str, got {type(data)}")

        # Update metadata (optional tracking)
        # Note: metadata tracking removed - using manifest system now

        return filepath

    def get_path(self) -> Path:
        """Get the current simulation path."""
        self._check_registered()
        return self.current_path

    def list_files(self) -> Dict[str, List[str]]:
        """
        List all saved files in the current simulation.

        Returns:
            Dictionary with 'files' and 'binaries' keys containing file lists
        """
        self._check_registered()

        return {
            "files": list(self._metadata["files"].keys()),
            "binaries": list(self._metadata["binaries"].keys()),
            "all_files": [f.name for f in self.current_path.glob("*") if f.is_file()]
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

        if self.current_path.exists():
            shutil.rmtree(self.current_path)
            warnings.warn(f"Deleted simulation directory: {self.current_path}")

    def _check_registered(self) -> None:
        """Check if dataset and pipeline are registered."""
        if self.current_path is None:
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
        - Without custom names: workspace_root/runs/{date}_{dataset}/NNNN_{hash}/
        - With run_name: workspace_root/runs/{date}_{dataset}_{runname}/NNNN_{hash}/
        - With pipeline_name: workspace_root/runs/{date}_{dataset}/NNNN_{pipelinename}_{hash}/
        - With both: workspace_root/runs/{date}_{dataset}_{runname}/NNNN_{pipelinename}_{hash}/

        Returns:
            Full path to pipeline directory
        """
        from datetime import datetime

        run_date = datetime.now().strftime("%Y-%m-%d")

        # Build run_id with optional custom name
        if run_name:
            run_id = f"{run_date}_{dataset_name}_{run_name}"
        else:
            run_id = f"{run_date}_{dataset_name}"
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

        # Create structure
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "_binaries").mkdir(exist_ok=True)

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

        Args:
            pipeline_dir: Path to pipeline (NNNN_hash/ or NNNN_pipelinename_hash/)
            exports_dir: Workspace exports directory
            dataset_name: Dataset name
            run_date: Run date (YYYYMMDD)
            custom_name: Optional custom name for export

        Creates export directory:
        - Without custom_name: dataset_run_pipelineid/
        - With custom_name: customname_pipelineid/

        Returns: Path to exported directory
        """
        pipeline_dir = Path(pipeline_dir)
        exports_dir = Path(exports_dir)
        exports_dir.mkdir(parents=True, exist_ok=True)

        pipeline_id = pipeline_dir.name  # e.g., "0001_a1b2c3" or "0001_baseline_a1b2c3"

        if custom_name:
            export_name = f"{custom_name}_{pipeline_id}"
        else:
            export_name = f"{dataset_name}_{run_date}_{pipeline_id}"
        export_path = exports_dir / export_name

        # Copy entire pipeline folder
        shutil.copytree(pipeline_dir, export_path, dirs_exist_ok=True)

        return export_path

    def export_best_prediction(self, predictions_file: Path, exports_dir: Path,
                              dataset_name: str, run_date: str, pipeline_id: str,
                              custom_name: str = None) -> Path:
        """Export predictions CSV to best_predictions/ folder with optional custom name.

        Args:
            predictions_file: Path to predictions.csv
            exports_dir: Workspace exports directory
            dataset_name, run_date, pipeline_id: Metadata for naming
            custom_name: Optional custom name for export

        Creates CSV filename:
        - Without custom_name: dataset_run_pipelineid.csv
        - With custom_name: customname_pipelineid.csv

        Returns: Path to exported CSV
        """
        predictions_file = Path(predictions_file)
        exports_dir = Path(exports_dir)

        best_dir = exports_dir / "best_predictions"
        best_dir.mkdir(parents=True, exist_ok=True)

        if custom_name:
            csv_name = f"{custom_name}_{pipeline_id}.csv"
        else:
            csv_name = f"{dataset_name}_{run_date}_{pipeline_id}.csv"
        dest_path = best_dir / csv_name

        shutil.copy2(predictions_file, dest_path)

        return dest_path

