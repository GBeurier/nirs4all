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

from nirs4all.utils.emoji import CHECK, WARNING
from nirs4all.data.predictions import Predictions


class SimulationSaver:
    """
    Manages saving simulation results with flat pipeline structure.

    Works with ManifestManager to create: base_path/NNNN_hash/files
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None, save_files: bool = True):
        """
        Initialize the simulation saver.

        Args:
            base_path: Base directory (run directory: workspace/runs/YYYY-MM-DD_dataset/)
            save_files: Whether to actually save files (can disable for dry runs)
        """
        self.base_path = Path(base_path) if base_path is not None else None
        self.pipeline_id: Optional[str] = None  # e.g., "0001_abc123"
        self.pipeline_dir: Optional[Path] = None
        self._metadata: Dict[str, Any] = {}
        self.save_files = save_files

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

        return self.pipeline_dir

    def _find_prediction_by_id(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Search for a prediction by ID in global predictions databases at workspace root."""
        # Navigate to workspace root (base_path is runs/YYYY-MM-DD_dataset/)
        workspace_root = Path(self.base_path).parent.parent
        if not workspace_root.exists():
            return None

        # Search in global prediction databases (dataset_name.meta.parquet and dataset_name.json files at workspace root)
        # Try Parquet files first (new format)
        for predictions_file in workspace_root.glob("*.meta.parquet"):
            if not predictions_file.is_file():
                continue

            try:
                predictions = Predictions.load_from_file_cls(str(predictions_file))
                for pred in predictions.filter_predictions(load_arrays=True):
                    if pred.get('id') == prediction_id:
                        return pred
            except Exception:
                continue

        # Fall back to JSON files (legacy format)
        for predictions_file in workspace_root.glob("*.json"):
            if not predictions_file.is_file():
                continue

            try:
                predictions = Predictions.load_from_file_cls(str(predictions_file))
                for pred in predictions.filter_predictions(load_arrays=True):
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
                  warn_on_overwrite: bool = True) -> Path:

        self._check_registered()

        filepath = self.pipeline_dir / filename

        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File {filename} already exists. Use overwrite=True to replace.")

        if filepath.exists() and warn_on_overwrite:
            warnings.warn(f"Overwriting existing file: {filename}")

        # Save content
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)

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
        Persist artifact using the serializer with content-addressed storage.

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

        from nirs4all.pipeline.artifact_serialization import persist

        self._check_registered()

        # Use _binaries directory (managed by ManifestManager)
        artifacts_dir = self.base_path / "_binaries"
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
        Save a human-readable output file (chart, report, etc.) directly to the pipeline directory.

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

        # Save directly in pipeline folder (no outputs/ subdirectory)
        output_dir = self.pipeline_dir

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

        return filepath

    def get_path(self) -> Path:
        """Get the current pipeline path."""
        self._check_registered()
        return self.pipeline_dir

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
            "all_files": [f.name for f in self.pipeline_dir.glob("*") if f.is_file()]
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

    def export_best_for_dataset(self, dataset_name: str, workspace_path: Path,
                                runs_dir: Path, mode: str = "predictions") -> Optional[Path]:
        """Export best results for a dataset to exports/ folder.

        Creates exports/{dataset_name}/ with best predictions, pipeline config, and charts.
        Files are renamed to include run date for tracking.

        Args:
            dataset_name: Dataset name (matches global prediction JSON filename)
            workspace_path: Workspace root path
            runs_dir: Runs directory path
            mode: Export mode - "predictions", "template", "trained", or "full"
                - predictions: Only predictions CSV and summary
                - template: Pipeline config only (no binaries)
                - trained: Pipeline + binaries (for deployment)
                - full: Pipeline + binaries + source dataset

        Returns:
            Path to export directory, or None if no predictions found
        """
        workspace_path = Path(workspace_path)
        runs_dir = Path(runs_dir)

        # Load global predictions for this dataset
        predictions_file = workspace_path / f"{dataset_name}.json"
        if not predictions_file.exists():
            print(f"{WARNING} No predictions found for dataset '{dataset_name}'")
            return None

        from nirs4all.data.predictions import Predictions
        predictions = Predictions.load_from_file_cls(str(predictions_file))
        if predictions.num_predictions == 0:
            print(f"{WARNING} No predictions in database for '{dataset_name}'")
            return None

        # Get best prediction
        best = predictions.get_best(ascending=True)  # Adjust based on task type if needed

        # Create export directory structure
        exports_dir = workspace_path / "exports" / dataset_name
        exports_dir.mkdir(parents=True, exist_ok=True)

        # Get run date from run directory name
        run_dirs = list(runs_dir.glob(f"*_{dataset_name}"))
        if not run_dirs:
            print(f"{WARNING} No run directory found for dataset '{dataset_name}'")
            return None

        run_dir = run_dirs[-1]  # Get most recent run
        run_date = run_dir.name.split('_')[0]  # Extract date from directory name

        # Find the pipeline directory
        config_name = best['config_name']
        pipeline_dir = None
        for pd in run_dir.iterdir():
            if pd.is_dir() and config_name in pd.name and not pd.name.startswith('_'):
                pipeline_dir = pd
                break

        if not pipeline_dir:
            print(f"{WARNING} Pipeline directory not found for config '{config_name}'")
            return None

        # Export predictions
        pred_filename = f"{run_date}_{best['model_name']}_predictions.csv"
        pred_path = exports_dir / pred_filename
        Predictions.save_predictions_to_csv(best["y_true"], best["y_pred"], pred_path)
        print(f"{CHECK} Exported predictions: {pred_path}")

        # Export pipeline config
        pipeline_json = pipeline_dir / "pipeline.json"
        if pipeline_json.exists():
            config_filename = f"{run_date}_{best['model_name']}_pipeline.json"
            config_path = exports_dir / config_filename
            shutil.copy(pipeline_json, config_path)
            print(f"{CHECK} Exported pipeline config: {config_path}")

        # Export charts if they exist
        for chart_file in pipeline_dir.glob("*.png"):
            chart_filename = f"{run_date}_{best['model_name']}_{chart_file.name}"
            chart_path = exports_dir / chart_filename
            shutil.copy(chart_file, chart_path)
            print(f"{CHECK} Exported chart: {chart_path}")

        # Handle different export modes for binaries
        if mode in ["trained", "full"]:
            binaries_dir = run_dir / "_binaries"
            if binaries_dir.exists():
                export_binaries_dir = exports_dir / "_binaries"
                export_binaries_dir.mkdir(exist_ok=True)

                # Copy referenced binaries from manifest
                manifest_file = pipeline_dir / "manifest.yaml"
                if manifest_file.exists():
                    import yaml
                    with open(manifest_file, 'r') as f:
                        manifest = yaml.safe_load(f)

                    for artifact in manifest.get('artifacts', []):
                        binary_name = Path(artifact['path']).name
                        src = binaries_dir / binary_name
                        if src.exists():
                            shutil.copy(src, export_binaries_dir / binary_name)

                    print(f"{CHECK} Exported {len(list(export_binaries_dir.iterdir()))} binaries")

        # Create summary metadata
        from datetime import datetime
        summary = {
            "dataset": dataset_name,
            "model_name": best['model_name'],
            "pipeline_id": config_name,
            "prediction_id": best['id'],
            "test_score": best.get('test_score'),
            "val_score": best.get('val_score'),
            "export_date": datetime.now().isoformat(),
            "export_mode": mode,
            "run_date": run_date
        }
        summary_path = exports_dir / f"{run_date}_{best['model_name']}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"{CHECK} Exported summary: {summary_path}")

        return exports_dir



