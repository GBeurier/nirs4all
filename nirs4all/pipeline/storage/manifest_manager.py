"""
Manifest Manager - Pipeline manifest and dataset index management

Manages pipeline manifests with sequential numbering and content-addressed artifacts.
Provides centralized pipeline registration, lookup, and lifecycle management.

Architecture (v2):
    workspace/
    ├── binaries/<dataset>/          # Centralized artifact storage
    │   ├── model_PLSRegression_abc123.joblib
    │   └── transformer_StandardScaler_def456.pkl
    └── runs/<dataset>/
        ├── 0001_pls_abc123/
        │   └── manifest.yaml        # References artifacts by path
        ├── 0002_rf_def456/
        │   └── manifest.yaml
        └── predictions.json         # Global predictions

Manifest Schema Versions:
    v1.0: Legacy format with flat artifacts list
    v2.0: New format with structured artifacts section and schema_version field
"""

import enum
import logging
import uuid
import yaml
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from nirs4all.pipeline.trace import ExecutionTrace

from nirs4all.pipeline.config.component_serialization import deserialize_component


logger = logging.getLogger(__name__)


# Schema version constants
MANIFEST_SCHEMA_V1 = "1.0"
MANIFEST_SCHEMA_V2 = "2.0"
CURRENT_MANIFEST_SCHEMA = MANIFEST_SCHEMA_V2


def _sanitize_for_yaml(obj: Any) -> Any:
    """
    Recursively sanitize data structures for safe YAML serialization.
    Converts tuples to lists, enums to their values, and removes runtime-only keys
    to avoid Python-specific YAML tags.

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
        # Remove runtime-only keys that cannot be safely serialized to YAML
        sanitized = {}
        for key, value in obj.items():
            if key == '_runtime_instance':
                # Skip runtime instances - they're Python objects that can't be serialized safely
                continue
            sanitized[key] = _sanitize_for_yaml(value)
        return sanitized
    elif isinstance(obj, enum.Enum):
        # Convert enum to its value (string, int, etc.)
        return obj.value
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # Basic YAML-safe types
        return obj
    elif hasattr(obj, '__class__') and obj.__class__.__module__ not in ('builtins', '__builtin__'):
        # Non-builtin objects that might not be YAML-serializable
        # Try to get a reasonable string representation
        if hasattr(obj, 'name'):
            return obj.name
        elif hasattr(obj, 'value'):
            return obj.value
        else:
            return str(obj)
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
            results_dir: Path to run directory (workspace/runs/<dataset>/)
        """
        self.results_dir = Path(results_dir)
        self.artifacts_dir = self.results_dir / "_binaries"
        # Note: _binaries directory is created lazily when artifacts are actually saved
        # to avoid empty _binaries folders

    def create_pipeline(
        self,
        name: str,
        dataset: str,
        pipeline_config: dict,
        pipeline_hash: str,
        metadata: Optional[dict] = None,
        generator_choices: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[str, Path]:
        """
        Create new pipeline with sequential numbering.

        Args:
            name: Pipeline name (for human reference)
            dataset: Dataset name
            pipeline_config: Pipeline configuration dict
            pipeline_hash: Hash of pipeline config (first 6 chars)
            metadata: Optional initial metadata
            generator_choices: List of generator choices that produced this pipeline.
                Each choice is a dict like {"_or_": selected_value} or {"_range_": 18}.

        Returns:
            Tuple of (pipeline_id, pipeline_dir)
            pipeline_id format: "0001_abc123" or "0001_name_abc123"
        """
        # Get sequential number
        pipeline_num = self.get_next_pipeline_number()

        # Build pipeline_id with optional custom name
        if name and name != "pipeline":  # Don't include generic "pipeline" name
            # Check if name already ends with the hash (avoid duplication like "0004_pls_9b4be0_9b4be0")
            if name.endswith(f"_{pipeline_hash}"):
                pipeline_id = f"{pipeline_num:04d}_{name}"
            else:
                pipeline_id = f"{pipeline_num:04d}_{name}_{pipeline_hash}"
        else:
            pipeline_id = f"{pipeline_num:04d}_{pipeline_hash}"

        # Create directory
        pipeline_dir = self.results_dir / pipeline_id
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        # Create manifest with v2 schema
        uid = str(uuid.uuid4())
        manifest = {
            "schema_version": CURRENT_MANIFEST_SCHEMA,
            "uid": uid,
            "pipeline_id": pipeline_id,
            "name": name,
            "dataset": dataset,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": pipeline_config,
            "generator_choices": generator_choices or [],
            "metadata": metadata or {},
            "artifacts": {
                "schema_version": CURRENT_MANIFEST_SCHEMA,
                "items": []
            },
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

        Supports both v1 (list) and v2 (dict with items) artifact formats.

        Args:
            pipeline_id: Pipeline ID
            artifacts: List of artifact metadata dictionaries
        """
        manifest = self.load_manifest(pipeline_id)
        artifacts_section = manifest.get("artifacts", [])

        if self._is_v2_artifacts(artifacts_section):
            # v2 format: dict with "items" list
            artifacts_section["items"].extend(artifacts)
        else:
            # v1 format: flat list
            artifacts_section.extend(artifacts)

        manifest["artifacts"] = artifacts_section
        self.save_manifest(pipeline_id, manifest)

    def append_artifacts_v2(
        self,
        pipeline_id: str,
        records: List[Dict[str, Any]]
    ) -> None:
        """
        Append v2 ArtifactRecords to a pipeline manifest.

        Args:
            pipeline_id: Pipeline ID
            records: List of ArtifactRecord instances or dicts
        """
        from nirs4all.pipeline.storage.artifacts.types import ArtifactRecord

        manifest = self.load_manifest(pipeline_id)
        artifacts_section = manifest.get("artifacts", {})

        # Ensure v2 format
        if not self._is_v2_artifacts(artifacts_section):
            # Convert v1 to v2
            artifacts_section = self._convert_artifacts_to_v2(artifacts_section)

        # Append records as dicts
        for record in records:
            if isinstance(record, ArtifactRecord):
                artifacts_section["items"].append(record.to_dict())
            else:
                artifacts_section["items"].append(record)

        manifest["artifacts"] = artifacts_section
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

    def save_execution_trace(self, pipeline_id: str, trace: 'ExecutionTrace') -> None:
        """
        Save an execution trace to the pipeline manifest.

        Execution traces record the exact path through the pipeline that produced
        a prediction, enabling deterministic replay for prediction, transfer, and export.

        Args:
            pipeline_id: Pipeline ID
            trace: ExecutionTrace instance to save

        Note:
            The trace is stored in the manifest under "execution_traces" keyed by trace_id.
        """
        from nirs4all.pipeline.trace import ExecutionTrace

        manifest = self.load_manifest(pipeline_id)

        # Initialize execution_traces section if not present
        if "execution_traces" not in manifest:
            manifest["execution_traces"] = {}

        # Convert trace to dict and store keyed by trace_id
        trace_dict = trace.to_dict() if hasattr(trace, 'to_dict') else trace
        trace_id = trace_dict.get("trace_id", "unknown")
        manifest["execution_traces"][trace_id] = trace_dict

        self.save_manifest(pipeline_id, manifest)

    def load_execution_trace(self, pipeline_id: str, trace_id: str) -> Optional['ExecutionTrace']:
        """
        Load a specific execution trace from the pipeline manifest.

        Args:
            pipeline_id: Pipeline ID
            trace_id: Trace ID to load

        Returns:
            ExecutionTrace instance or None if not found
        """
        from nirs4all.pipeline.trace import ExecutionTrace

        try:
            manifest = self.load_manifest(pipeline_id)
        except FileNotFoundError:
            return None

        traces = manifest.get("execution_traces", {})
        trace_dict = traces.get(trace_id)

        if trace_dict is None:
            return None

        return ExecutionTrace.from_dict(trace_dict)

    def list_execution_traces(self, pipeline_id: str) -> List[str]:
        """
        List all execution trace IDs for a pipeline.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            List of trace IDs
        """
        try:
            manifest = self.load_manifest(pipeline_id)
        except FileNotFoundError:
            return []

        traces = manifest.get("execution_traces", {})
        return list(traces.keys())

    def get_latest_execution_trace(self, pipeline_id: str) -> Optional['ExecutionTrace']:
        """
        Get the most recent execution trace for a pipeline.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Most recent ExecutionTrace or None if none exist
        """
        from nirs4all.pipeline.trace import ExecutionTrace

        try:
            manifest = self.load_manifest(pipeline_id)
        except FileNotFoundError:
            return None

        traces = manifest.get("execution_traces", {})
        if not traces:
            return None

        # Sort by created_at to get the latest
        sorted_traces = sorted(
            traces.items(),
            key=lambda x: x[1].get("created_at", ""),
            reverse=True
        )

        if sorted_traces:
            trace_dict = sorted_traces[0][1]
            return ExecutionTrace.from_dict(trace_dict)

        return None

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

    def extract_top_preprocessings(
        self,
        predictions: List[Dict[str, Any]],
        top_k: int = 3,
        step_name: str = "feature_augmentation",
        exclude_scalers: bool = True,
        verbose: bool = False
    ) -> List[List[Any]]:
        """Extract top K unique preprocessing pipelines from ranked predictions.

        Given a list of predictions (typically from `predictions.top()`), extracts
        the preprocessing pipeline that was actually used for each prediction by
        parsing the display string and deserializing the transformers.

        Iterates through ALL predictions until top_k unique preprocessings are found.
        This ensures we get the best-performing unique preprocessings even if the
        top predictions share the same preprocessing (e.g., different folds).

        This method is designed for pipeline chaining: run pipeline 1, get top
        predictions, extract their preprocessings, use in pipeline 2.

        Args:
            predictions: List of prediction dictionaries, typically from
                `predictions.top(n=..., rank_metric="rmse")`. Should be sorted
                by score (best first). Each prediction must have:
                - 'preprocessings': display string (e.g., "ExtendedMSC>Detr>MinMax")
            top_k: Number of unique preprocessings to extract. Will iterate through
                all predictions until this many unique preprocessings are found.
            step_name: Unused, kept for backward compatibility.
            exclude_scalers: If True, remove scaler transformers from each pipeline.
            verbose: If True, print tracing information.

        Returns:
            List of up to top_k unique preprocessing pipelines. Each pipeline
            is a list of transformer instances ready for use in pipeline config.

        Example:
            >>> manager = ManifestManager(runs_dir)
            >>> top_preds = predictions.top(n=50, rank_metric="rmse")  # Get many predictions
            >>> top_pp = manager.extract_top_preprocessings(top_preds, top_k=3)
            >>> # top_pp = [[ExtendedMSC(), Detrend()], [SNV()], [MSC(), FirstDer()]]
            >>> # Use in next pipeline:
            >>> pipeline = [{"feature_augmentation": {"_or_": top_pp}}, ...]
        """
        result = []
        seen_display_strings = set()

        # Iterate through ALL predictions to find top_k unique preprocessings
        for pred in predictions:
            # Stop once we have enough unique preprocessings
            if len(result) >= top_k:
                break

            # Get the preprocessing display string from the prediction
            pp_display = pred.get("preprocessings", "")
            if not pp_display:
                continue

            # Skip if we already have this preprocessing (deduplication)
            if pp_display in seen_display_strings:
                if verbose:
                    logger.debug(f"[extract] Skipping duplicate: {pp_display}")
                continue

            # Parse the display string to extract preprocessings
            deserialized = self._parse_display_string(pp_display, verbose=verbose)

            if exclude_scalers and deserialized:
                deserialized = self._strip_trailing_scalers(deserialized, verbose=verbose)

            if not deserialized:
                if verbose:
                    logger.debug(f"[extract] Empty after parsing: {pp_display}")
                continue

            # Success - add to results and mark as seen
            seen_display_strings.add(pp_display)
            result.append(deserialized)

            if verbose:
                names = [type(t).__name__ for t in deserialized]
                logger.debug(f"[extract] #{len(result)}: {names} (from {pp_display})")

        if verbose:
            logger.debug(f"[extract] Extracted {len(result)} unique preprocessing(s)")

        return result

    def extract_generator_choice(
        self,
        prediction: Dict[str, Any],
        choice_index: int,
        instantiate: bool = False,
        verbose: bool = False
    ) -> Optional[Any]:
        """Extract a specific generator choice from a prediction's pipeline manifest.

        Given a prediction (from `predictions.top()` or similar), loads the
        corresponding pipeline manifest and returns the generator choice at
        the specified index.

        Generator choices are stored in the manifest's `generator_choices` field,
        which is a list of dicts like:
            [{"_or_": "StandardScaler"}, {"_range_": 18}, {"_or_": {...}}]

        This method allows extracting the value of a specific choice, either as
        the raw JSON node (for re-use in pipeline specs) or as an instantiated
        Python object.

        Args:
            prediction: Prediction dictionary with 'pipeline_uid' field.
            choice_index: Index of the choice in the generator_choices list (0-based).
            instantiate: If True, deserialize the choice value into a Python object.
                        If False, return the raw JSON value.
            verbose: If True, print debug information.

        Returns:
            The choice value (JSON or instantiated object), or None if:
            - The prediction has no pipeline_uid
            - The manifest doesn't exist or has no generator_choices
            - The choice_index is out of range

        Example:
            >>> manager = ManifestManager(runs_dir)
            >>> top_pred = predictions.top(n=1)[0]
            >>> # Get raw JSON of first choice
            >>> scaler_spec = manager.extract_generator_choice(top_pred, 0)
            >>> # scaler_spec = "sklearn.preprocessing._data.StandardScaler"
            >>>
            >>> # Get instantiated object
            >>> scaler = manager.extract_generator_choice(top_pred, 0, instantiate=True)
            >>> # scaler = StandardScaler()
            >>>
            >>> # Get second choice (e.g., model spec)
            >>> model_spec = manager.extract_generator_choice(top_pred, 1)
            >>> # model_spec = {'class': '...PLSRegression', 'params': {'n_components': 3}}
        """
        # Get pipeline_uid from prediction
        pipeline_uid = prediction.get("pipeline_uid")
        if not pipeline_uid:
            if verbose:
                logger.debug("[extract_choice] No pipeline_uid in prediction")
            return None

        # Load manifest
        try:
            manifest = self.load_manifest(pipeline_uid)
        except FileNotFoundError:
            if verbose:
                logger.debug(f"[extract_choice] Manifest not found for {pipeline_uid}")
            return None

        # Get generator_choices
        choices = manifest.get("generator_choices", [])
        if not choices:
            if verbose:
                logger.debug(f"[extract_choice] No generator_choices in manifest for {pipeline_uid}")
            return None

        # Check index bounds
        if choice_index < 0 or choice_index >= len(choices):
            if verbose:
                logger.debug(f"[extract_choice] Index {choice_index} out of range (0-{len(choices)-1})")
            return None

        # Get the choice entry
        choice_entry = choices[choice_index]

        # Extract the value from the choice dict (e.g., {"_or_": value} -> value)
        # The choice dict has exactly one key which is the generator keyword
        if isinstance(choice_entry, dict) and len(choice_entry) == 1:
            keyword = next(iter(choice_entry.keys()))
            value = choice_entry[keyword]
        else:
            # Unexpected format, return as-is
            value = choice_entry

        if verbose:
            logger.debug(f"[extract_choice] Choice {choice_index}: {value}")

        # Optionally instantiate
        if instantiate:
            return deserialize_component(value)
        else:
            return value

    def extract_all_generator_choices(
        self,
        prediction: Dict[str, Any],
        instantiate: bool = False,
        verbose: bool = False
    ) -> List[Any]:
        """Extract all generator choices from a prediction's pipeline manifest.

        Similar to extract_generator_choice but returns all choices at once.

        Args:
            prediction: Prediction dictionary with 'pipeline_uid' field.
            instantiate: If True, deserialize all choice values into Python objects.
                        If False, return raw JSON values.
            verbose: If True, print debug information.

        Returns:
            List of choice values (JSON or instantiated objects).
            Empty list if no choices are available.

        Example:
            >>> manager = ManifestManager(runs_dir)
            >>> top_pred = predictions.top(n=1)[0]
            >>> all_choices = manager.extract_all_generator_choices(top_pred)
            >>> # all_choices = ["StandardScaler", {'class': '...', 'params': {...}}]
        """
        pipeline_uid = prediction.get("pipeline_uid")
        if not pipeline_uid:
            if verbose:
                logger.debug("[extract_choices] No pipeline_uid in prediction")
            return []

        try:
            manifest = self.load_manifest(pipeline_uid)
        except FileNotFoundError:
            if verbose:
                logger.debug(f"[extract_choices] Manifest not found for {pipeline_uid}")
            return []

        choices = manifest.get("generator_choices", [])
        if not choices:
            return []

        results = []
        for choice_entry in choices:
            # Extract value from choice dict
            if isinstance(choice_entry, dict) and len(choice_entry) == 1:
                keyword = next(iter(choice_entry.keys()))
                value = choice_entry[keyword]
            else:
                value = choice_entry

            if instantiate:
                try:
                    results.append(deserialize_component(value))
                except Exception as e:
                    if verbose:
                        logger.debug(f"[extract_choices] Failed to deserialize: {value} - {e}")
                    results.append(value)  # Return raw value on failure
            else:
                results.append(value)

        if verbose:
            logger.debug(f"[extract_choices] Extracted {len(results)} choice(s)")

        return results

    def _parse_display_string(self, display: str, verbose: bool = False) -> List[Any]:
        """Parse a preprocessing display string back to transformer instances.

        The display string format is like:
        - "ExtendedMSC>Detr>ExtendedMSC>MinMax"
        - "SNV>1stDer"
        - "raw" (no preprocessing)

        Args:
            display: Display string from prediction.
            verbose: Print debug info.

        Returns:
            List of transformer instances.
        """
        # Mapping from abbreviated display names to full class paths
        # These paths must match the actual module locations
        abbrev_to_class = {
            # NIRS-specific transforms
            "SNV": "nirs4all.operators.transforms.scalers.StandardNormalVariate",
            "MSC": "nirs4all.operators.transforms.nirs.MultiplicativeScatterCorrection",
            "ExtendedMSC": "nirs4all.operators.transforms.nirs.ExtendedMultiplicativeScatterCorrection",
            "EMSC": "nirs4all.operators.transforms.nirs.ExtendedMultiplicativeScatterCorrection",
            "RSNV": "nirs4all.operators.transforms.scalers.RobustStandardNormalVariate",
            "AreaNorm": "nirs4all.operators.transforms.nirs.AreaNormalization",
            # Signal processing transforms
            "SG": "nirs4all.operators.transforms.nirs.SavitzkyGolay",
            "1stDer": "nirs4all.operators.transforms.nirs.FirstDerivative",
            "2ndDer": "nirs4all.operators.transforms.nirs.SecondDerivative",
            "Detr": "nirs4all.operators.transforms.signal.Detrend",
            "Gauss": "nirs4all.operators.transforms.signal.Gaussian",
            # Wavelet transforms
            "Haar": "nirs4all.operators.transforms.wavelets.Haar",
            # Sklearn scalers
            "MinMax": "sklearn.preprocessing.MinMaxScaler",
            "Std": "sklearn.preprocessing.StandardScaler",
            "Rbt": "sklearn.preprocessing.RobustScaler",
        }

        if not display or display == "raw" or display == "None":
            return []

        # Handle multi-source pipelines (split by |)
        # Take the main part (usually the second one contains actual preprocessing)
        parts = display.split("|")
        if len(parts) > 1:
            # Find the part with actual preprocessing (not just scalers)
            for part in parts:
                if any(abbrev in part for abbrev in ["SNV", "MSC", "1stDer", "2ndDer", "Detr", "SG"]):
                    display = part
                    break
            else:
                display = parts[-1]  # Use last part if no preprocessing found

        # Split by > to get individual transformer names
        names = display.split(">")
        transformers = []

        for name in names:
            name = name.strip()
            if name in abbrev_to_class:
                class_path = abbrev_to_class[name]
                try:
                    instance = deserialize_component(class_path)
                    transformers.append(instance)
                except Exception as e:
                    if verbose:
                        logger.debug(f"[parse] Failed to deserialize {name}: {e}")
            elif verbose and name:
                logger.debug(f"[parse] Unknown transformer: {name}")

        return transformers

    def _flatten_preprocessing_options(self, content: Any) -> List[Any]:
        """Flatten nested preprocessing options into individual pipelines.

        The manifest can have various structures:
        - Single string: "ClassName"
        - List of strings: ["Class1", "Class2"] = single pipeline with 2 steps
        - Nested lists: [[["A", "B"], ["C"]]] = multiple pipeline options

        Returns:
            List of individual preprocessing pipelines.
        """
        result = []

        if not content:
            return result

        if isinstance(content, str):
            result.append([content])
        elif isinstance(content, list):
            # Check if first element is also a list (nested options)
            if content and isinstance(content[0], list):
                for option in content:
                    result.extend(self._flatten_preprocessing_options(option))
            else:
                # Check if all elements are strings (single pipeline)
                if all(isinstance(x, str) for x in content):
                    result.append(content)
                else:
                    # Mixed - recurse
                    for item in content:
                        result.extend(self._flatten_preprocessing_options(item))

        return result

    def _generate_display_string(self, pipeline: Any) -> str:
        """Generate a display string from serialized preprocessing.

        Converts class names to abbreviated form matching short_preprocessings_str().

        Args:
            pipeline: Serialized preprocessing (string or list of strings/dicts).

        Returns:
            Display string like "ExtendedMSC>Detr>MinMax".
        """
        # Class name replacements (matches dataset.short_preprocessings_str)
        replacements = [
            ("ExtendedMultiplicativeScatterCorrection", "ExtendedMSC"),
            ("MultiplicativeScatterCorrection", "MSC"),
            ("StandardNormalVariate", "SNV"),
            ("RobustStandardNormalVariate", "RSNV"),
            ("SavitzkyGolay", "SG"),
            ("FirstDerivative", "1stDer"),
            ("SecondDerivative", "2ndDer"),
            ("Detrend", "Detr"),
            ("Gaussian", "Gauss"),
            ("Haar", "Haar"),
            ("LogTransform", "Log"),
            ("MinMaxScaler", "MinMax"),
            ("RobustScaler", "Rbt"),
            ("StandardScaler", "Std"),
            ("QuantileTransformer", "Quant"),
            ("PowerTransformer", "Pow"),
            ("AreaNormalization", "AreaNorm"),
        ]

        def get_class_name(item: Any) -> str:
            """Extract class name from various formats."""
            if isinstance(item, str):
                # Full path like "nirs4all.operators.transforms.nirs.SNV"
                return item.rpartition(".")[2]
            elif isinstance(item, dict):
                class_path = item.get("class", "")
                return class_path.rpartition(".")[2]
            return ""

        # Extract names
        names = []
        if isinstance(pipeline, list):
            for item in pipeline:
                name = get_class_name(item)
                if name:
                    names.append(name)
        elif isinstance(pipeline, str):
            name = get_class_name(pipeline)
            if name:
                names.append(name)

        # Apply replacements
        abbreviated = []
        for name in names:
            abbrev = name
            for long_name, short in replacements:
                if name == long_name:
                    abbrev = short
                    break
            abbreviated.append(abbrev)

        return ">".join(abbreviated)

    def _display_strings_match(self, generated: str, target: str) -> bool:
        """Check if generated display string is a component of the target.

        The target display string represents the full preprocessing chain applied
        during training, which may combine multiple options from _or_ picks plus
        scalers. The generated string is from a single option in the manifest.

        For example:
        - target: "ExtendedMSC>Detr>ExtendedMSC>MinMax" (full chain)
        - generated: "ExtendedMSC>Detr" (one option)
        - Should match because "ExtendedMSC>Detr" is a prefix of the target

        Args:
            generated: Generated display string from a single manifest option.
            target: Target display string from prediction (full chain).

        Returns:
            True if generated is a component of target's preprocessing chain.
        """
        # Exact match
        if generated == target:
            return True

        # Strip scalers from target for comparison
        scalers = {"MinMax", "Std", "Rbt", "Quant", "Pow"}

        def strip_scalers(s: str) -> str:
            parts = s.split(">")
            # Strip from start
            while parts and parts[0] in scalers:
                parts.pop(0)
            # Strip from end
            while parts and parts[-1] in scalers:
                parts.pop()
            return ">".join(parts)

        target_core = strip_scalers(target)
        gen_core = strip_scalers(generated)

        # Check if generated is exactly the target (after stripping scalers)
        if gen_core == target_core:
            return True

        # Check if generated is a prefix of target
        # This handles pick:(1,2) case where multiple options are combined
        if target_core.startswith(gen_core + ">") or target_core.startswith(gen_core):
            return True

        return False

    def _strip_trailing_scalers(
        self,
        pipeline: List[Any],
        verbose: bool = False
    ) -> List[Any]:
        """Remove trailing scaler transformers from a pipeline.

        Scalers at the end of a pipeline are typically added for normalization
        before the model, not as part of the preprocessing. This method removes
        them to get only the actual preprocessing transformers.

        Args:
            pipeline: List of transformer instances.
            verbose: If True, print when scalers are removed.

        Returns:
            Pipeline with trailing scalers removed.
        """
        # Common scaler class names to strip
        scaler_names = {
            'MinMaxScaler', 'StandardScaler', 'RobustScaler', 'MaxAbsScaler',
            'Normalizer', 'QuantileTransformer', 'PowerTransformer'
        }

        result = list(pipeline)

        # Remove scalers from the end
        while result and type(result[-1]).__name__ in scaler_names:
            removed = result.pop()
            if verbose:
                logger.debug(f"[extract_preprocessings] Stripped trailing scaler: {type(removed).__name__}")

        # Also remove scalers from the beginning (often added before preprocessing)
        while result and type(result[0]).__name__ in scaler_names:
            removed = result.pop(0)
            if verbose:
                logger.debug(f"[extract_preprocessings] Stripped leading scaler: {type(removed).__name__}")

        return result

    def _deserialize_preprocessing(self, pp_pipeline: Any) -> List[Any]:
        """Deserialize a preprocessing pipeline from manifest format.

        Args:
            pp_pipeline: Preprocessing pipeline in manifest format.
                Can be a string (class name), a list of class names,
                or a nested structure.

        Returns:
            List of transformer instances.
        """
        deserialized = []

        if isinstance(pp_pipeline, list):
            for item in pp_pipeline:
                if isinstance(item, str):
                    instance = deserialize_component(item)
                    deserialized.append(instance)
                elif isinstance(item, list):
                    # Nested list - this is a sub-pipeline
                    sub_pipeline = [
                        deserialize_component(cn)
                        for cn in item
                        if isinstance(cn, str)
                    ]
                    if sub_pipeline:
                        deserialized.extend(sub_pipeline)
                elif isinstance(item, dict):
                    # Dict format with class and params
                    instance = deserialize_component(item)
                    deserialized.append(instance)
        elif isinstance(pp_pipeline, str):
            instance = deserialize_component(pp_pipeline)
            deserialized.append(instance)
        elif isinstance(pp_pipeline, dict):
            instance = deserialize_component(pp_pipeline)
            deserialized.append(instance)

        return deserialized

    # =========================================================================
    # Schema Version Detection and Conversion (v2)
    # =========================================================================

    @staticmethod
    def get_schema_version(manifest: Dict[str, Any]) -> str:
        """Detect manifest schema version.

        Args:
            manifest: Manifest dictionary

        Returns:
            Schema version string ("1.0" or "2.0")
        """
        # Check explicit schema_version field
        if "schema_version" in manifest:
            return manifest["schema_version"]

        # Check artifacts section format
        artifacts = manifest.get("artifacts", [])
        if isinstance(artifacts, dict) and "schema_version" in artifacts:
            return artifacts["schema_version"]

        # Legacy detection: if artifacts is a list, it's v1
        if isinstance(artifacts, list):
            return MANIFEST_SCHEMA_V1

        # Default to v1 for unknown formats
        return MANIFEST_SCHEMA_V1

    @staticmethod
    def _is_v2_artifacts(artifacts_section: Any) -> bool:
        """Check if artifacts section is v2 format.

        v2 format is a dict with "items" key.
        v1 format is a flat list.

        Args:
            artifacts_section: The artifacts section from manifest

        Returns:
            True if v2 format
        """
        return (isinstance(artifacts_section, dict)
                and "items" in artifacts_section)

    @staticmethod
    def _convert_artifacts_to_v2(artifacts_list: List[Dict]) -> Dict[str, Any]:
        """Convert v1 artifacts list to v2 format.

        Args:
            artifacts_list: v1 format flat list of artifacts

        Returns:
            v2 format dict with schema_version and items
        """
        return {
            "schema_version": MANIFEST_SCHEMA_V2,
            "items": artifacts_list
        }

    def get_artifacts_list(self, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get artifacts as a flat list regardless of schema version.

        Args:
            manifest: Manifest dictionary

        Returns:
            List of artifact metadata dictionaries
        """
        artifacts = manifest.get("artifacts", [])

        if self._is_v2_artifacts(artifacts):
            return artifacts.get("items", [])
        elif isinstance(artifacts, list):
            return artifacts
        else:
            return []

    def upgrade_manifest_to_v2(self, pipeline_id: str) -> None:
        """Upgrade a v1 manifest to v2 format in place.

        Args:
            pipeline_id: Pipeline ID to upgrade
        """
        manifest = self.load_manifest(pipeline_id)

        # Check if already v2
        if self.get_schema_version(manifest) == MANIFEST_SCHEMA_V2:
            logger.debug(f"Manifest {pipeline_id} already at v2")
            return

        # Add schema version
        manifest["schema_version"] = MANIFEST_SCHEMA_V2

        # Convert artifacts section
        artifacts = manifest.get("artifacts", [])
        if isinstance(artifacts, list):
            manifest["artifacts"] = self._convert_artifacts_to_v2(artifacts)

        # Remove old version field if present
        manifest.pop("version", None)

        self.save_manifest(pipeline_id, manifest)
        logger.info(f"Upgraded manifest {pipeline_id} to v2")
