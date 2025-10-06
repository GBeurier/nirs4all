"""
Simulation IO Manager - Save and manage simulation outputs

Provides organized storage for pipeline simulation results with
dataset and pipeline-based folder structure management.
"""
import json
import pickle
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

    def __init__(self, base_path: Optional[Union[str, Path]] = "simulations"):
        """
        Initialize the simulation saver.

        Args:
            base_path: Base directory for all simulation outputs
        """
        self.base_path = Path(base_path) if base_path is not None else Path("results")
        self.dataset_name: Optional[str] = None
        self.pipeline_name: Optional[str] = None
        self.current_path: Optional[Path] = None
        self._metadata: Dict[str, Any] = {}
        self.dataset_path: Optional[Path] = None

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

        # Save initial metadata
        if mode != "predict" and mode != "explain":
            self._save_metadata()

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

        # Update metadata
        self._metadata["files"][filename] = {
            "path": str(filepath.relative_to(self.base_path)),
            "size": filepath.stat().st_size,
            "encoding": encoding,
            "saved_at": datetime.now().isoformat(),
            "overwritten": filepath.existed_before if hasattr(filepath, 'existed_before') else False
        }
        self._save_metadata()

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


    def save_binary(self,
                    filename: str,
                    data: Union[bytes, BinaryIO, Any],
                    overwrite: bool = False,
                    pickle_if_object: bool = True,
                    into_dataset: bool = False) -> Path:
        """
        Deprecated: Use save_files or save_file instead.

        Save binary data or objects to a file.

        Args:
            filename: Name of the file
            data: Binary data, file-like object, or Python object
            overwrite: Whether to overwrite existing files
            pickle_if_object: Whether to pickle non-bytes objects

        Returns:
            Path to the saved file

        Raises:
            RuntimeError: If not registered
            FileExistsError: If file exists and overwrite=False
            TypeError: If data type is not supported
        """
        warnings.warn(
            "save_binary is deprecated and will be removed in a future version. "
            "Use save_files or save_file instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._check_registered()

        filepath = self.current_path / Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)  # Create subdirectories if needed

        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File {filename} already exists. Use overwrite=True to replace.")

        if filepath.exists():
            warnings.warn(f"Overwriting existing binary file: {filename}")

        # Handle different data types using modern Path methods
        data_type = "unknown"

        if hasattr(data, "read"):
            # File-like object - read the data
            file_data = data.read()
            if isinstance(file_data, str):
                file_data = file_data.encode("utf-8")
            filepath.write_bytes(bytes(file_data))
            data_type = "file_like"

        elif isinstance(data, (bytes, bytearray, memoryview)):
            # Direct binary data
            filepath.write_bytes(bytes(data))
            data_type = "bytes"

        elif isinstance(data, str):
            # Text data
            filepath.write_text(data, encoding="utf-8")
            data_type = "text"

        elif filepath.suffix.lower() in {".pkl", ".pickle", ".p"}:
            # Pickle with explicit extension
            with filepath.open("wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            data_type = "pickled_object"

        elif pickle_if_object:
            # Default to pickle for other objects
            filepath = filepath.with_suffix(filepath.suffix + '.pkl')
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with filepath.open("wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            data_type = "pickled_object"

        else:
            raise TypeError(f"Unsupported type for {filename}: {type(data).__name__}")

        # Update metadata (keeping this advantage from original method)
        final_filename = str(filepath.relative_to(self.current_path))
        self._metadata["binaries"][final_filename] = {
            "path": str(filepath.relative_to(self.base_path)),
            "size": filepath.stat().st_size,
            "data_type": data_type,
            "saved_at": datetime.now().isoformat()
        }
        self._save_metadata()

        return filepath

    def save_files(self, step_number: int, substep_number: int, files: List[Tuple[Union[str, Path], Any]], into_dataset: bool = False) -> List[Path]:
        """
        Save multiple files in a single operation.

        Args:
            files: List of (filename, data) tuples where data may be:
                  - bytes/bytearray/memoryview -> written as binary
                  - file-like (has .read)     -> read then written
                  - str                       -> written as UTF-8 text
                  - any object with .pkl/.pickle/.p extension -> pickled
                  - any other object -> raises TypeError

        Returns:
            List of Path objects for saved files
        """
        self._check_registered()

        saved_paths: List[Path] = []
        saved_names = []
        for fname, obj in files:
            name = str(step_number)
            name += "_" + str(fname)
            filepath = self.current_path / Path(name)


            # if not full_save and filepath.suffix.lower() in {".pkl", ".pickle", ".p"}:
            #     continue

            filepath.parent.mkdir(parents=True, exist_ok=True)

            data_type = "unknown"

            if hasattr(obj, "read"):
                # File-like object
                file_data = obj.read()
                if isinstance(file_data, str):
                    file_data = file_data.encode("utf-8")
                filepath.write_bytes(bytes(file_data))
                data_type = "file_like"
                saved_paths.append(filepath)

            elif isinstance(obj, (bytes, bytearray, memoryview)):
                # Binary data
                filepath.write_bytes(bytes(obj))
                data_type = "bytes"
                saved_paths.append(filepath)

            elif isinstance(obj, str):
                # Text data
                filepath.write_text(obj, encoding="utf-8")
                data_type = "text"
                saved_paths.append(filepath)

            elif filepath.suffix.lower() in {".pkl", ".pickle", ".p"}:
                # Pickle with explicit extension
                with filepath.open("wb") as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                data_type = "pickled_object"
                saved_paths.append(filepath)

            else:
                raise TypeError(f"Unsupported type for {name}: {type(obj).__name__}")

            # print(f"💾 Saved {name} to {filepath}")  # Debug print
            saved_names.append(name)

            # Update metadata for each file
            relative_filename = str(filepath.relative_to(self.current_path))
            if str(step_number) not in self._metadata["binaries"]:
                self._metadata["binaries"][str(step_number)] = []
            metadata_entry = {
                "path": str(filepath.relative_to(self.base_path)),
                "op_filename": str(fname),
                "op_name": str(fname).split('.')[0],
                "relative_path": relative_filename,
                "step": step_number,
                "size": filepath.stat().st_size,
                "data_type": data_type,
                "saved_at": datetime.now().isoformat()
            }
            self._metadata["binaries"][str(step_number)].append(metadata_entry)
        # Save metadata once after all files
        self._save_metadata()
        # if len(saved_names) > 1:
        #     print(f"💾 Saved {len(saved_names)} files.")
        # elif len(saved_names) == 1:
        #     print(f"💾 Saved file: {saved_names[0]}")

        return saved_paths


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

    def _save_metadata(self) -> None:
        """Save metadata to file."""
        if self.current_path is None:
            return

        metadata_path = self.current_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self._metadata, f, indent=2, default=str)

