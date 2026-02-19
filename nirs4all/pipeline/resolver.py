"""
Prediction Resolver - Unified resolution of prediction sources.

This module provides the PredictionResolver class which normalizes any prediction
source (dict, folder, Run, artifact_id, bundle) to the components needed for
prediction replay:
    - Minimal pipeline (subset of steps)
    - Artifact map (artifacts keyed by step index)
    - Execution trace (for deterministic replay)
    - Fold strategy (for CV ensemble averaging)

The resolver is controller-agnostic: it doesn't know about specific controller
types, but provides the artifacts and trace needed by any controller to replay
a prediction.

Supported Source Types:
    - prediction dict: Most common, from a previous run's Predictions object
    - folder path: Path to a pipeline directory containing manifest.yaml
    - Run object: Best prediction from a Run
    - artifact_id: Direct artifact reference (e.g., "0001:4:all")
    - bundle: Exported prediction bundle (.n4a file)
    - trace_id: Execution trace reference (e.g., "trace:xyz789")

Example:
    >>> resolver = PredictionResolver(workspace_path)
    >>> resolved = resolver.resolve(best_prediction)
    >>> # Use resolved.minimal_pipeline, resolved.artifact_provider
    >>> y_pred = execute_prediction(resolved, new_data)
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from nirs4all.pipeline.config.context import (
    ArtifactProvider,
    LoaderArtifactProvider,
    MapArtifactProvider,
)
from nirs4all.pipeline.storage.artifacts.artifact_loader import ArtifactLoader
from nirs4all.pipeline.trace import ExecutionTrace

logger = logging.getLogger(__name__)

RESOLUTION_MODE_AUTO = "auto"
RESOLUTION_MODE_STORE = "store"
RESOLUTION_MODE_FILESYSTEM = "filesystem"
VALID_RESOLUTION_MODES = {
    RESOLUTION_MODE_AUTO,
    RESOLUTION_MODE_STORE,
    RESOLUTION_MODE_FILESYSTEM,
}

# ---------------------------------------------------------------------------
# Lightweight manifest helpers (replace ManifestManager dependency)
# ---------------------------------------------------------------------------

def _load_manifest(run_dir: Path, pipeline_uid: str) -> dict[str, Any]:
    """Load a manifest YAML file from disk.

    Args:
        run_dir: Parent run directory.
        pipeline_uid: Pipeline identifier (directory name under *run_dir*).

    Returns:
        Parsed manifest dictionary.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
    """
    manifest_path = run_dir / pipeline_uid / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_execution_trace(run_dir: Path, pipeline_uid: str, trace_id: str) -> ExecutionTrace | None:
    """Load a specific execution trace from a pipeline manifest.

    Args:
        run_dir: Parent run directory.
        pipeline_uid: Pipeline identifier.
        trace_id: Trace identifier to load.

    Returns:
        ExecutionTrace instance or ``None`` if not found.
    """
    try:
        manifest = _load_manifest(run_dir, pipeline_uid)
    except FileNotFoundError:
        return None
    traces = manifest.get("execution_traces", {})
    trace_dict = traces.get(trace_id)
    if trace_dict is None:
        return None
    return ExecutionTrace.from_dict(trace_dict)

def _get_latest_execution_trace(run_dir: Path, pipeline_uid: str) -> ExecutionTrace | None:
    """Return the most recent execution trace for a pipeline.

    Args:
        run_dir: Parent run directory.
        pipeline_uid: Pipeline identifier.

    Returns:
        Most recent ExecutionTrace or ``None``.
    """
    try:
        manifest = _load_manifest(run_dir, pipeline_uid)
    except FileNotFoundError:
        return None
    traces = manifest.get("execution_traces", {})
    if not traces:
        return None
    sorted_traces = sorted(
        traces.items(),
        key=lambda x: x[1].get("created_at", ""),
        reverse=True,
    )
    if sorted_traces:
        return ExecutionTrace.from_dict(sorted_traces[0][1])
    return None

def _list_pipelines(run_dir: Path) -> list[str]:
    """List numbered pipeline directories under *run_dir*.

    Args:
        run_dir: Directory to scan.

    Returns:
        Sorted list of pipeline directory names.
    """
    if not run_dir.exists():
        return []
    return sorted(
        d.name for d in run_dir.iterdir()
        if d.is_dir() and d.name[:4].isdigit()
    )

def _list_execution_traces(run_dir: Path, pipeline_uid: str) -> list[str]:
    """List execution trace IDs in a pipeline manifest.

    Args:
        run_dir: Parent run directory.
        pipeline_uid: Pipeline identifier.

    Returns:
        List of trace ID strings.
    """
    try:
        manifest = _load_manifest(run_dir, pipeline_uid)
    except FileNotFoundError:
        return []
    return list(manifest.get("execution_traces", {}).keys())

class SourceType(StrEnum):
    """Type of prediction source.

    Attributes:
        PREDICTION: Dictionary from Predictions object
        FOLDER: Path to pipeline folder
        RUN: Run object (best prediction from run)
        ARTIFACT_ID: Direct artifact reference string
        BUNDLE: Exported .n4a bundle file
        TRACE_ID: Execution trace reference
        MODEL_FILE: Direct model file (.joblib, .pkl, .h5, .pt, etc.)
        UNKNOWN: Unrecognized source type
    """

    PREDICTION = "prediction"
    PREDICTION_ID = "prediction_id"
    FOLDER = "folder"
    RUN = "run"
    ARTIFACT_ID = "artifact_id"
    BUNDLE = "bundle"
    TRACE_ID = "trace_id"
    MODEL_FILE = "model_file"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value

class FoldStrategy(StrEnum):
    """Strategy for combining fold predictions in CV ensembles.

    Attributes:
        AVERAGE: Simple average of fold predictions
        WEIGHTED_AVERAGE: Weighted average using fold weights
        SINGLE: Use a single fold's prediction
    """

    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    SINGLE = "single"

    def __str__(self) -> str:
        return self.value

@dataclass
class ResolvedPrediction:
    """Normalized prediction source ready for execution.

    Contains all components needed to replay a prediction:
    - minimal_pipeline: Subset of steps needed for this prediction
    - artifact_provider: Provider for artifacts by step index
    - trace: Execution trace for deterministic replay
    - fold_strategy: How to combine fold predictions (for CV)
    - fold_weights: Per-fold weights (for weighted average)

    Attributes:
        source_type: Type of the original source
        minimal_pipeline: List of pipeline steps needed for replay
        artifact_provider: Provider for step artifacts
        trace: ExecutionTrace if available
        fold_strategy: Strategy for combining folds
        fold_weights: Per-fold weights for weighted averaging
        model_step_index: Index of the model step
        target_model: Target model metadata for filtering
        pipeline_uid: Pipeline unique identifier
        run_dir: Path to run directory
        manifest: Full manifest dictionary (for metadata access)
    """

    source_type: SourceType = SourceType.UNKNOWN
    minimal_pipeline: list[Any] = field(default_factory=list)
    artifact_provider: ArtifactProvider | None = None
    trace: ExecutionTrace | None = None
    fold_strategy: FoldStrategy = FoldStrategy.WEIGHTED_AVERAGE
    fold_weights: dict[int, float] = field(default_factory=dict)
    model_step_index: int | None = None
    target_model: dict[str, Any] = field(default_factory=dict)
    pipeline_uid: str = ""
    run_dir: Path | None = None
    manifest: dict[str, Any] = field(default_factory=dict)

    def has_trace(self) -> bool:
        """Check if execution trace is available.

        Returns:
            True if trace is available for deterministic replay
        """
        return self.trace is not None

    def has_fold_artifacts(self) -> bool:
        """Check if fold-specific artifacts are available.

        Returns:
            True if this is a CV ensemble with multiple folds
        """
        return len(self.fold_weights) > 1

    def get_preprocessing_chain(self) -> str:
        """Get the preprocessing chain summary.

        Returns:
            Preprocessing chain string (e.g., "SNV>SG>MinMax") or empty
        """
        if self.trace:
            return self.trace.preprocessing_chain
        return ""

class PredictionResolver:
    """Resolves any prediction source to executable components.

    This class provides a unified interface for resolving prediction sources
    to the components needed for prediction replay, regardless of the source
    type (dict, folder, run, artifact_id, bundle).

    The resolver is designed to be controller-agnostic: it doesn't know about
    specific controller types, but provides artifacts and trace that any
    controller can use.

    Attributes:
        workspace_path: Root workspace directory
        runs_dir: Directory containing run outputs

    Example:
        >>> resolver = PredictionResolver(workspace_path)
        >>> resolved = resolver.resolve(best_prediction)
        >>> # Execute using resolved components
        >>> provider = resolved.artifact_provider
        >>> artifacts = provider.get_artifacts_for_step(step_index=1)
    """

    def __init__(
        self,
        workspace_path: str | Path,
        runs_dir: str | Path | None = None,
        store: Any | None = None,
        resolution_mode: str = RESOLUTION_MODE_AUTO,
    ):
        """Initialize prediction resolver.

        Args:
            workspace_path: Root workspace directory
            runs_dir: Optional runs directory override (default: workspace/runs)
            store: Optional WorkspaceStore for DuckDB-backed resolution
            resolution_mode: Resolver mode policy.
                - ``"auto"``: Try store first, fallback to filesystem.
                - ``"store"``: Store-only resolution (no filesystem fallback).
                - ``"filesystem"``: Filesystem-only resolution (skip store).
        """
        self.workspace_path = Path(workspace_path)
        self.runs_dir = Path(runs_dir) if runs_dir else self.workspace_path / "runs"
        self.store = store
        self.resolution_mode = self._validate_resolution_mode(resolution_mode)

    @staticmethod
    def _validate_resolution_mode(mode: str) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized not in VALID_RESOLUTION_MODES:
            valid = ", ".join(sorted(VALID_RESOLUTION_MODES))
            raise ValueError(f"Invalid resolution_mode: {mode!r}. Expected one of: {valid}")
        return normalized

    def resolve(
        self,
        source: dict[str, Any] | str | Path | Any,
        verbose: int = 0,
        resolution_mode: str | None = None,
    ) -> ResolvedPrediction:
        """Resolve any prediction source to executable components.

        Detects the source type and delegates to the appropriate resolver.

        Args:
            source: Prediction source (dict, folder path, Run, artifact_id, bundle)
            verbose: Verbosity level for logging
            resolution_mode: Optional per-call override for resolver mode.

        Returns:
            ResolvedPrediction with all components for replay

        Raises:
            ValueError: If source type cannot be determined or resolved
            FileNotFoundError: If referenced files/directories don't exist
        """
        mode = self.resolution_mode if resolution_mode is None else self._validate_resolution_mode(resolution_mode)
        source_type = self._detect_source_type(source)

        if verbose > 0:
            logger.info(f"Resolving prediction source of type: {source_type}")

        if source_type == SourceType.PREDICTION:
            # Type narrowing: we know source is dict at this point
            assert isinstance(source, dict)
            return self._resolve_from_prediction(source, verbose, mode)
        elif source_type == SourceType.PREDICTION_ID:
            # String prediction ID — look up the full prediction dict from the store
            assert isinstance(source, str) and self.store is not None
            prediction_dict = self.store.get_prediction(source)
            # Map store column names to expected prediction dict keys
            if "pipeline_id" in prediction_dict and "pipeline_uid" not in prediction_dict:
                prediction_dict["pipeline_uid"] = prediction_dict["pipeline_id"]
            return self._resolve_from_prediction(prediction_dict, verbose, mode)
        elif source_type == SourceType.FOLDER:
            # Type narrowing: we know source is str or Path at this point
            assert isinstance(source, (str, Path))
            return self._resolve_from_folder(source, verbose)
        elif source_type == SourceType.RUN:
            return self._resolve_from_run(source, verbose)
        elif source_type == SourceType.ARTIFACT_ID:
            # Type narrowing: artifact_id is always a string
            assert isinstance(source, str)
            return self._resolve_from_artifact_id(source, verbose)
        elif source_type == SourceType.BUNDLE:
            # Type narrowing: bundle path is str or Path
            assert isinstance(source, (str, Path))
            return self._resolve_from_bundle(source, verbose)
        elif source_type == SourceType.TRACE_ID:
            # Type narrowing: trace_id is always a string
            assert isinstance(source, str)
            return self._resolve_from_trace_id(source, verbose)
        elif source_type == SourceType.MODEL_FILE:
            # Type narrowing: model file path is str or Path
            assert isinstance(source, (str, Path))
            return self._resolve_from_model_file(source, verbose)
        else:
            raise ValueError(
                f"Cannot resolve prediction source: {source}\n"
                f"Detected type: {source_type}\n"
                f"Supported types: prediction dict, folder path, Run object, "
                f"artifact_id string, bundle path, trace_id string, model file"
            )

    def _detect_source_type(self, source: Any) -> SourceType:
        """Detect the type of prediction source.

        Args:
            source: Any prediction source

        Returns:
            SourceType enum value
        """
        # Dictionary: prediction dict
        if isinstance(source, dict):
            # Check for prediction-specific keys
            if any(k in source for k in ["id", "pipeline_uid", "config_path", "model_name"]):
                return SourceType.PREDICTION
            # Could still be a prediction dict with minimal fields
            return SourceType.PREDICTION

        # String or Path
        if isinstance(source, (str, Path)):
            source_str = str(source)

            # Trace ID reference (e.g., "trace:xyz789")
            if source_str.startswith("trace:"):
                return SourceType.TRACE_ID

            # Artifact ID pattern (e.g., "0001:4:all" or "abc123:2:0")
            if self._looks_like_artifact_id(source_str):
                return SourceType.ARTIFACT_ID

            # Bundle file (.n4a extension)
            if source_str.endswith(".n4a") or source_str.endswith(".n4a.py"):
                return SourceType.BUNDLE

            # Model file (direct model binary)
            path = Path(source_str)
            model_extensions = {'.joblib', '.pkl', '.h5', '.hdf5', '.keras', '.pt', '.pth', '.ckpt'}
            if path.suffix.lower() in model_extensions and path.exists():
                return SourceType.MODEL_FILE

            # Path to folder or file
            if path.is_dir():
                # Check if it's a model folder (AutoGluon, TF SavedModel) vs pipeline folder
                if self._is_model_folder(path):
                    return SourceType.MODEL_FILE
                return SourceType.FOLDER
            if path.exists() and path.suffix in (".yaml", ".json"):
                return SourceType.FOLDER  # Treat as folder containing manifest

            # Try relative path under runs_dir
            run_path = self.runs_dir / source_str
            if run_path.is_dir():
                return SourceType.FOLDER

            # Could be a pipeline_uid (filesystem only)
            if self.store is None and self._find_pipeline_dir(source_str):
                return SourceType.FOLDER

            # When store is available, check if the string is a prediction ID
            if self.store is not None:
                pred = self.store.get_prediction(source_str)
                if pred is not None:
                    return SourceType.PREDICTION_ID

                # Or a pipeline_id -- treat as prediction-style source
                if self.store.get_pipeline(source_str) is not None:
                    return SourceType.PREDICTION

        # Run object (duck typing)
        if hasattr(source, "predictions") and hasattr(source, "best"):
            return SourceType.RUN

        return SourceType.UNKNOWN

    def _is_model_folder(self, path: Path) -> bool:
        """Check if a folder is a model folder (not a pipeline folder).

        Model folders contain model binaries (AutoGluon, TF SavedModel).
        Pipeline folders contain manifest.yaml.

        Args:
            path: Path to folder

        Returns:
            True if this is a model folder
        """
        # AutoGluon TabularPredictor
        if (path / "predictor.pkl").exists():
            return True
        # TensorFlow SavedModel
        if (path / "saved_model.pb").exists():
            return True
        # Keras SavedModel
        if (path / "keras_metadata.pb").exists():
            return True
        # JAX/Orbax checkpoint
        return bool((path / "checkpoint").exists())

    def _looks_like_artifact_id(self, s: str) -> bool:
        """Check if string looks like an artifact ID.

        Artifact IDs have format: "pipeline_id:step_index:fold_id"
        or "pipeline_id:branch:step_index:fold_id"

        Args:
            s: String to check

        Returns:
            True if it looks like an artifact ID
        """
        parts = s.split(":")
        if len(parts) < 3:
            return False

        # Last part should be "all" or a number (fold_id)
        last = parts[-1]
        if last != "all" and not last.isdigit():
            return False

        # Second-to-last should be a number (step_index)
        try:
            int(parts[-2])
            return True
        except ValueError:
            return False

    def _find_pipeline_dir(
        self,
        pipeline_uid: str,
        trace_id: str | None = None
    ) -> Path | None:
        """Find pipeline directory by UID.

        Searches run directories for a matching pipeline folder.
        If trace_id is provided, verifies the trace exists in the manifest.

        Args:
            pipeline_uid: Pipeline unique identifier
            trace_id: Optional trace ID to match (ensures correct run is found)

        Returns:
            Path to pipeline directory or None
        """
        if not self.runs_dir.exists():
            return None

        candidates = []

        # Search in all run directories
        for run_dir in sorted(self.runs_dir.iterdir(), key=lambda p: p.name):
            if not run_dir.is_dir():
                continue

            # Check for exact match
            pipeline_dir = run_dir / pipeline_uid
            if pipeline_dir.is_dir() and (pipeline_dir / "manifest.yaml").exists():
                candidates.append(pipeline_dir)
                continue

            # Check for partial match (pipeline_uid might be just the hash part)
            for subdir in sorted(run_dir.iterdir(), key=lambda p: p.name):
                if subdir.is_dir() and pipeline_uid in subdir.name and (subdir / "manifest.yaml").exists():
                    candidates.append(subdir)

        if not candidates:
            return None

        candidates = sorted(candidates, key=lambda p: (p.parent.name, p.name))

        # If no trace_id specified, return first candidate
        if not trace_id:
            return candidates[0]

        # If trace_id specified, find the candidate with matching trace
        for pipeline_dir in candidates:
            try:
                manifest = _load_manifest(pipeline_dir.parent, pipeline_uid)
                traces = manifest.get("execution_traces", {})
                if trace_id in traces:
                    return pipeline_dir
            except Exception:
                continue

        # Fallback to first candidate if trace not found in any
        return candidates[0]

    # =========================================================================
    # Source-specific resolvers
    # =========================================================================

    def _resolve_from_prediction(
        self,
        prediction: dict[str, Any],
        verbose: int = 0,
        resolution_mode: str = RESOLUTION_MODE_AUTO,
    ) -> ResolvedPrediction:
        """Resolve from a prediction dictionary.

        The prediction dict comes from a Predictions object and contains
        metadata about the model that produced it.

        New format (with trace_id):
            Uses trace_id to load execution trace for deterministic replay.

        Legacy format (without trace_id):
            Falls back to pipeline_uid/config_path for reconstruction.

        Args:
            prediction: Prediction dictionary with metadata
            verbose: Verbosity level
            resolution_mode: Resolution policy (auto/store/filesystem)

        Returns:
            ResolvedPrediction with components for replay
        """
        # Store-first or store-only modes
        if resolution_mode in (RESOLUTION_MODE_AUTO, RESOLUTION_MODE_STORE) and self.store is not None:
            resolved = self._resolve_from_store(prediction, verbose)
            if resolved is not None:
                return resolved
            if resolution_mode == RESOLUTION_MODE_STORE:
                pipeline_uid = prediction.get("pipeline_uid", "")
                raise FileNotFoundError(
                    f"Store-only resolution failed for pipeline_uid={pipeline_uid!r}. "
                    "No matching store-backed pipeline/chain was found."
                )
            logger.info(
                "Resolver auto mode fallback: store miss; using filesystem path "
                f"(pipeline_uid={prediction.get('pipeline_uid', '')!r})"
            )
        elif resolution_mode == RESOLUTION_MODE_STORE:
            raise ValueError(
                "Store-only resolution requested but resolver has no store instance."
            )

        result = ResolvedPrediction(source_type=SourceType.PREDICTION)

        # Extract target model metadata
        result.target_model = {
            k: v for k, v in prediction.items()
            if k not in ("y_pred", "y_true", "X")
        }

        # Get pipeline_uid
        pipeline_uid = prediction.get("pipeline_uid")
        if not pipeline_uid:
            # Try config_path as fallback
            config_path = prediction.get("config_path", "")
            pipeline_uid = Path(config_path).name if "/" in config_path or "\\" in config_path else config_path

        if not pipeline_uid:
            raise ValueError(
                "Prediction has no pipeline_uid or config_path. "
                "Cannot resolve to a pipeline for replay."
            )

        result.pipeline_uid = pipeline_uid

        # Find run directory
        run_dir = self._find_run_dir_for_prediction(prediction)
        if run_dir is None:
            raise FileNotFoundError(
                f"Cannot find run directory for pipeline: {pipeline_uid}"
            )
        result.run_dir = run_dir

        # Load manifest
        try:
            manifest = _load_manifest(run_dir, pipeline_uid)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Manifest not found for pipeline: {pipeline_uid}\n"
                f"Expected at: {run_dir / pipeline_uid / 'manifest.yaml'}"
            ) from e

        result.manifest = manifest

        # Try to load execution trace (Phase 2+)
        trace_id = prediction.get("trace_id")
        if trace_id:
            trace = _load_execution_trace(run_dir, pipeline_uid, trace_id)
            if trace:
                result.trace = trace
                result.model_step_index = trace.model_step_index
                result.fold_weights = trace.fold_weights or {}
                if result.fold_weights:
                    result.fold_strategy = FoldStrategy.WEIGHTED_AVERAGE

        # Load artifact loader
        artifact_loader = ArtifactLoader.from_manifest(manifest, run_dir)

        # Create artifact provider
        if result.trace:
            # Use trace-based provider for deterministic replay
            result.artifact_provider = LoaderArtifactProvider(
                loader=artifact_loader,
                trace=result.trace
            )
        else:
            # Fallback: create map-based provider from manifest
            artifact_map = self._build_artifact_map_from_manifest(manifest, artifact_loader)
            result.artifact_provider = MapArtifactProvider(
                artifact_map=artifact_map,
                fold_weights=result.fold_weights
            )

        # Load minimal pipeline
        result.minimal_pipeline = self._load_pipeline_steps(run_dir, pipeline_uid)

        # Extract fold weights from prediction if not in trace
        if not result.fold_weights:
            fold_weights = prediction.get("fold_weights", {})
            if fold_weights:
                result.fold_weights = {int(k): v for k, v in fold_weights.items()}
                result.fold_strategy = FoldStrategy.WEIGHTED_AVERAGE

        return result

    def _resolve_from_folder(
        self,
        folder: str | Path,
        verbose: int = 0
    ) -> ResolvedPrediction:
        """Resolve from a folder path.

        The folder can be:
        - A pipeline directory containing manifest.yaml
        - A run directory containing multiple pipelines

        Args:
            folder: Path to folder
            verbose: Verbosity level

        Returns:
            ResolvedPrediction with components for replay
        """
        result = ResolvedPrediction(source_type=SourceType.FOLDER)

        folder_path = Path(folder)

        # Handle relative paths
        if not folder_path.is_absolute():
            # Try as relative to runs_dir first
            candidate = self.runs_dir / folder
            if candidate.is_dir():
                folder_path = candidate
            else:
                # Try as relative to cwd
                folder_path = Path.cwd() / folder

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        # Check for manifest in folder
        manifest_path = folder_path / "manifest.yaml"
        if manifest_path.exists():
            # This is a pipeline directory
            pipeline_uid = folder_path.name
            run_dir = folder_path.parent
        else:
            # Look for newest pipeline in folder (this is a run dir)
            run_dir = folder_path
            pipelines = _list_pipelines(run_dir)
            if not pipelines:
                raise ValueError(f"No pipelines found in folder: {folder}")
            pipeline_uid = pipelines[-1]  # Most recent

        result.run_dir = run_dir
        result.pipeline_uid = pipeline_uid

        # Load manifest
        manifest = _load_manifest(run_dir, pipeline_uid)
        result.manifest = manifest

        # Try to load latest execution trace
        trace = _get_latest_execution_trace(run_dir, pipeline_uid)
        if trace:
            result.trace = trace
            result.model_step_index = trace.model_step_index
            result.fold_weights = trace.fold_weights or {}
            if result.fold_weights:
                result.fold_strategy = FoldStrategy.WEIGHTED_AVERAGE

        # Load artifact loader
        artifact_loader = ArtifactLoader.from_manifest(manifest, run_dir)

        # Create artifact provider
        if result.trace:
            result.artifact_provider = LoaderArtifactProvider(
                loader=artifact_loader,
                trace=result.trace
            )
        else:
            artifact_map = self._build_artifact_map_from_manifest(manifest, artifact_loader)
            result.artifact_provider = MapArtifactProvider(
                artifact_map=artifact_map,
                fold_weights=result.fold_weights
            )

        # Load minimal pipeline
        result.minimal_pipeline = self._load_pipeline_steps(run_dir, pipeline_uid)

        return result

    def _resolve_from_run(
        self,
        run: Any,
        verbose: int = 0
    ) -> ResolvedPrediction:
        """Resolve from a Run object.

        Uses the best prediction from the Run.

        Args:
            run: Run object with predictions attribute
            verbose: Verbosity level

        Returns:
            ResolvedPrediction with components for replay
        """
        # Get best prediction from Run
        if hasattr(run, "best"):
            best_pred = run.best()
        elif hasattr(run, "predictions") and hasattr(run.predictions, "top"):
            top = run.predictions.top(n=1)
            if top:
                best_pred = top[0]
            else:
                raise ValueError("Run has no predictions")
        else:
            raise ValueError("Run object has no 'best' method or predictions")

        # Delegate to prediction resolver
        result = self._resolve_from_prediction(best_pred, verbose)
        result.source_type = SourceType.RUN
        return result

    def _resolve_from_artifact_id(
        self,
        artifact_id: str,
        verbose: int = 0
    ) -> ResolvedPrediction:
        """Resolve from an artifact ID string.

        Artifact ID format: "pipeline_id:step_index:fold_id"
        or "pipeline_id:branch:step_index:fold_id"

        Args:
            artifact_id: Artifact ID string
            verbose: Verbosity level

        Returns:
            ResolvedPrediction with components for replay
        """
        result = ResolvedPrediction(source_type=SourceType.ARTIFACT_ID)

        # Parse artifact ID
        parts = artifact_id.split(":")
        if len(parts) < 3:
            raise ValueError(
                f"Invalid artifact ID format: {artifact_id}\n"
                f"Expected: 'pipeline_id:step_index:fold_id' or similar"
            )

        # Extract pipeline_id (first part or parts before step_index)
        # Format: pipeline_id:step:fold or pipeline_id:branch:step:fold
        fold_str = parts[-1]
        step_index = int(parts[-2])

        if len(parts) == 3:
            pipeline_id = parts[0]
            branch_path = []
        else:
            # More complex format with branch path
            pipeline_id = parts[0]
            branch_path = [int(p) for p in parts[1:-2]]

        # Find pipeline
        pipeline_dir = self._find_pipeline_dir(pipeline_id)
        if pipeline_dir is None:
            raise FileNotFoundError(f"Cannot find pipeline: {pipeline_id}")

        run_dir = pipeline_dir.parent
        pipeline_uid = pipeline_dir.name

        result.run_dir = run_dir
        result.pipeline_uid = pipeline_uid
        result.model_step_index = step_index

        # Load manifest
        manifest = _load_manifest(run_dir, pipeline_uid)
        result.manifest = manifest

        # Load artifact loader
        artifact_loader = ArtifactLoader.from_manifest(manifest, run_dir)

        # Build artifact map focused on the specific artifact
        artifact_map = self._build_artifact_map_from_manifest(
            manifest, artifact_loader, up_to_step=step_index
        )
        result.artifact_provider = MapArtifactProvider(artifact_map=artifact_map)

        # Load minimal pipeline
        result.minimal_pipeline = self._load_pipeline_steps(run_dir, pipeline_uid)

        # Set target model metadata
        result.target_model = {
            "step_idx": step_index,
            "pipeline_uid": pipeline_uid,
            "artifact_id": artifact_id
        }

        if fold_str != "all":
            result.target_model["fold_id"] = int(fold_str)

        return result

    def _resolve_from_bundle(
        self,
        bundle_path: str | Path,
        verbose: int = 0
    ) -> ResolvedPrediction:
        """Resolve from an exported prediction bundle (.n4a file).

        Bundle format is a ZIP archive containing:
        - manifest.json: Bundle metadata
        - pipeline.json: Minimal pipeline config
        - trace.json: Execution trace
        - artifacts/: Directory with artifact binaries

        Phase 6 Implementation:
            This method loads a .n4a bundle file and creates a ResolvedPrediction
            that can be used with the existing prediction infrastructure.

        Args:
            bundle_path: Path to .n4a bundle file
            verbose: Verbosity level

        Returns:
            ResolvedPrediction with components for replay
        """
        from nirs4all.pipeline.bundle import BundleLoader

        bundle_path = Path(bundle_path)

        # Handle relative paths
        if not bundle_path.is_absolute():
            # Try as relative to workspace
            candidate = self.workspace_path / bundle_path
            if candidate.exists():
                bundle_path = candidate
            else:
                # Try as relative to cwd
                bundle_path = Path.cwd() / bundle_path

        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found: {bundle_path}")

        if verbose > 0:
            logger.info(f"Loading bundle: {bundle_path}")

        # Load bundle and convert to ResolvedPrediction
        loader = BundleLoader(bundle_path)
        return loader.to_resolved_prediction()

    def _resolve_from_trace_id(
        self,
        trace_ref: str,
        verbose: int = 0
    ) -> ResolvedPrediction:
        """Resolve from a trace ID reference.

        Trace ID format: "trace:xyz789" or "pipeline_uid:trace:xyz789"

        Args:
            trace_ref: Trace ID reference string
            verbose: Verbosity level

        Returns:
            ResolvedPrediction with components for replay
        """
        result = ResolvedPrediction(source_type=SourceType.TRACE_ID)

        # Parse trace reference
        if not trace_ref.startswith("trace:"):
            raise ValueError(f"Invalid trace reference: {trace_ref}")

        parts = trace_ref.split(":")
        if len(parts) == 2:
            # Format: "trace:xyz789" - need to search for trace
            trace_id = parts[1]
            pipeline_uid, run_dir = self._find_trace_location(trace_id)
        else:
            # Format: "pipeline_uid:trace:xyz789"
            pipeline_uid = parts[0]
            trace_id = parts[2]
            pipeline_dir = self._find_pipeline_dir(pipeline_uid)
            if pipeline_dir is None:
                raise FileNotFoundError(f"Cannot find pipeline: {pipeline_uid}")
            run_dir = pipeline_dir.parent

        result.run_dir = run_dir
        result.pipeline_uid = pipeline_uid

        # Load manifest and trace
        manifest = _load_manifest(run_dir, pipeline_uid)
        result.manifest = manifest

        trace = _load_execution_trace(run_dir, pipeline_uid, trace_id)
        if trace is None:
            raise ValueError(f"Trace not found: {trace_id} in pipeline {pipeline_uid}")

        result.trace = trace
        result.model_step_index = trace.model_step_index
        result.fold_weights = trace.fold_weights or {}
        if result.fold_weights:
            result.fold_strategy = FoldStrategy.WEIGHTED_AVERAGE

        # Load artifact loader and provider
        artifact_loader = ArtifactLoader.from_manifest(manifest, run_dir)
        result.artifact_provider = LoaderArtifactProvider(
            loader=artifact_loader,
            trace=trace
        )

        # Load minimal pipeline
        result.minimal_pipeline = self._load_pipeline_steps(run_dir, pipeline_uid)

        return result

    def _resolve_from_model_file(
        self,
        model_path: str | Path,
        verbose: int = 0
    ) -> ResolvedPrediction:
        """Resolve from a direct model file.

        Creates a minimal ResolvedPrediction with just the model loaded.
        No preprocessing artifacts are available - the model is used as-is.

        Supports:
        - .joblib, .pkl: sklearn/general models
        - .h5, .hdf5, .keras: TensorFlow/Keras models
        - .pt, .pth, .ckpt: PyTorch models
        - Folders: AutoGluon (predictor.pkl), TensorFlow SavedModel (saved_model.pb)

        Args:
            model_path: Path to model file or folder
            verbose: Verbosity level

        Returns:
            ResolvedPrediction with model loaded

        Example:
            >>> resolver = PredictionResolver(workspace_path)
            >>> resolved = resolver.resolve("models/pls_wheat.joblib")
            >>> # Use resolved.artifact_provider.get_artifacts_for_step(0)
        """
        from nirs4all.controllers.models.factory import ModelFactory

        result = ResolvedPrediction(source_type=SourceType.MODEL_FILE)
        path = Path(model_path)

        if verbose > 0:
            logger.info(f"Loading model from file: {path}")

        # Load the model using ModelFactory
        model = ModelFactory._load_model_from_file(str(path))

        # Create a simple artifact provider with just the model
        # The model is placed at step 0 (single-step pipeline)
        # artifact_id format: "model_file:0:0" (source:step:fold)
        artifact_id = f"model_file:{path.stem}:0:0"
        artifact_map: dict[int, list[tuple[str, Any]]] = {
            0: [(artifact_id, model)]  # step 0, fold 0
        }
        result.artifact_provider = MapArtifactProvider(artifact_map)
        result.model_step_index = 0
        result.fold_strategy = FoldStrategy.SINGLE
        result.fold_weights = {0: 1.0}

        # Minimal pipeline is just the model
        result.minimal_pipeline = [model]

        # Set metadata from path
        result.pipeline_uid = f"model_file_{path.stem}"

        if verbose > 0:
            framework = self._detect_model_framework(model)
            logger.info(f"Loaded {framework} model: {type(model).__name__}")

        return result

    def _detect_model_framework(self, model: Any) -> str:
        """Detect the framework of a loaded model.

        Args:
            model: Loaded model instance

        Returns:
            Framework name: 'sklearn', 'tensorflow', 'pytorch', 'jax', 'autogluon', 'unknown'
        """
        module = type(model).__module__

        if 'sklearn' in module or 'sklearn' in str(type(model)):
            return 'sklearn'
        if 'tensorflow' in module or 'keras' in module:
            return 'tensorflow'
        if 'torch' in module:
            return 'pytorch'
        if 'jax' in module or 'flax' in module:
            return 'jax'
        if 'autogluon' in module:
            return 'autogluon'
        if 'xgboost' in module:
            return 'xgboost'
        if 'lightgbm' in module:
            return 'lightgbm'
        if 'catboost' in module:
            return 'catboost'

        return 'unknown'

    # =========================================================================
    # Store-based resolution (DuckDB)
    # =========================================================================

    def _resolve_from_store(
        self,
        prediction: dict[str, Any],
        verbose: int = 0,
    ) -> ResolvedPrediction | None:
        """Try to resolve a prediction using the WorkspaceStore.

        Looks up the pipeline and chain data in the DuckDB store and
        builds a :class:`ResolvedPrediction` with a :class:`MapArtifactProvider`
        backed by store-loaded artifacts.

        Args:
            prediction: Prediction dictionary with pipeline_uid.
            verbose: Verbosity level.

        Returns:
            A :class:`ResolvedPrediction` if the pipeline/chain exist in the
            store, or ``None`` if resolution should fall through to the
            filesystem path.
        """
        pipeline_uid = prediction.get("pipeline_uid")
        if not pipeline_uid or self.store is None:
            return None

        # Look up pipeline in store
        pipeline = self.store.get_pipeline(pipeline_uid)
        if pipeline is None:
            return None

        # Get chains for this pipeline
        chains_df = self.store.get_chains_for_pipeline(pipeline_uid)
        if chains_df.is_empty():
            return None

        # Pick the matching chain
        chain_id = self._pick_chain_for_prediction(chains_df, prediction)
        chain = self.store.get_chain(chain_id)
        if chain is None:
            return None

        if verbose > 0:
            logger.info(f"Resolved pipeline {pipeline_uid} via store (chain {chain_id})")

        result = ResolvedPrediction(source_type=SourceType.PREDICTION)
        result.pipeline_uid = pipeline_uid
        result.model_step_index = chain["model_step_idx"]

        # Build artifact map from the target chain, plus source model chains
        # (chains with a lower model_step_idx) needed for meta-model stacking.
        # Chains at the same model_step_idx but different branch_path are NOT
        # loaded to avoid mixing branch-specific preprocessing artifacts.
        target_step_idx = chain["model_step_idx"]
        artifact_map, source_artifact_map = self._build_artifact_map_from_chain(chain)
        for cid in chains_df["chain_id"].to_list():
            if cid == chain_id:
                continue
            c = self.store.get_chain(cid)
            if c is not None and c["model_step_idx"] < target_step_idx:
                partial_map, partial_source_map = self._build_artifact_map_from_chain(c)
                for step_idx, entries in partial_map.items():
                    if step_idx not in artifact_map:
                        artifact_map[step_idx] = []
                    artifact_map[step_idx].extend(entries)
                for key, entries in partial_source_map.items():
                    if key not in source_artifact_map:
                        source_artifact_map[key] = []
                    source_artifact_map[key].extend(entries)

        # Derive fold weights from the target chain
        fold_artifacts = chain.get("fold_artifacts") or {}
        fold_weights = dict.fromkeys(range(len(fold_artifacts)), 1.0)
        result.fold_weights = fold_weights
        if len(fold_weights) > 1:
            result.fold_strategy = FoldStrategy.WEIGHTED_AVERAGE

        result.artifact_provider = MapArtifactProvider(
            artifact_map=artifact_map,
            fold_weights=fold_weights,
            source_artifact_map=source_artifact_map,
        )

        # Minimal pipeline from expanded_config, trimmed to the target model step.
        # model_step_idx is 1-based; expanded_config is a 0-based list.
        # Include all steps up to and including the model step so that
        # step numbering matches artifact_map keys.
        expanded_config = pipeline.get("expanded_config", [])
        if not isinstance(expanded_config, list):
            expanded_config = [expanded_config]
        model_step_idx = chain["model_step_idx"]
        if model_step_idx and model_step_idx <= len(expanded_config):
            result.minimal_pipeline = expanded_config[:model_step_idx]
        else:
            result.minimal_pipeline = expanded_config

        # Target model metadata
        result.target_model = {
            k: v for k, v in prediction.items()
            if k not in ("y_pred", "y_true", "X")
        }

        # Extract fold weights from prediction if available
        pred_fold_weights = prediction.get("fold_weights")
        if pred_fold_weights:
            result.fold_weights = {int(k): v for k, v in pred_fold_weights.items()}
            result.fold_strategy = FoldStrategy.WEIGHTED_AVERAGE

        return result

    def _build_artifact_map_from_chain(
        self,
        chain: dict[str, Any],
    ) -> tuple[dict[int, list[tuple[str, Any]]], dict[tuple[int, int], list[tuple[str, Any]]]]:
        """Build an artifact map from a store chain.

        Loads shared (preprocessing) and fold (model) artifacts from the
        store and organises them into the format expected by
        :class:`MapArtifactProvider`.

        Args:
            chain: Chain dictionary from ``store.get_chain()``.

        Returns:
            Tuple of (artifact_map, source_artifact_map) where:
            - artifact_map maps step_index → list of (artifact_id, object)
            - source_artifact_map maps (step_index, source_index) → list of
              (artifact_id, object) for multi-source steps
        """
        artifact_map: dict[int, list[tuple[str, Any]]] = {}
        source_artifact_map: dict[tuple[int, int], list[tuple[str, Any]]] = {}
        shared_artifacts = chain.get("shared_artifacts") or {}
        fold_artifacts = chain.get("fold_artifacts") or {}
        model_step_idx = chain["model_step_idx"]

        # Extract per-source metadata (embedded by ChainBuilder)
        source_map_meta = shared_artifacts.get("_source_map") if isinstance(shared_artifacts, dict) else None

        # Load shared artifacts (preprocessing transforms)
        for step_idx_str, artifact_ids_val in shared_artifacts.items():
            if step_idx_str.startswith("_"):
                continue  # Skip metadata keys like _source_map
            step_idx = int(step_idx_str)
            # shared_artifacts values are lists of artifact IDs
            aid_list = artifact_ids_val if isinstance(artifact_ids_val, list) else [artifact_ids_val]
            loaded: list[tuple[str, Any]] = []
            for artifact_id in aid_list:
                try:
                    obj = self.store.load_artifact(artifact_id)
                    loaded.append((artifact_id, obj))
                except (KeyError, FileNotFoundError) as e:
                    logger.warning(f"Failed to load shared artifact {artifact_id}: {e}")
            if loaded:
                artifact_map[step_idx] = loaded

        # Build per-source artifact map from _source_map metadata
        if source_map_meta:
            for step_idx_str, sources in source_map_meta.items():
                step_idx = int(step_idx_str)
                loaded_at_step = dict(artifact_map.get(step_idx, []))
                for source_idx_str, aid_list in sources.items():
                    source_idx = int(source_idx_str)
                    entries = [(aid, loaded_at_step[aid]) for aid in aid_list if aid in loaded_at_step]
                    if entries:
                        source_artifact_map[(step_idx, source_idx)] = entries

        # Load fold artifacts (model) — append to existing entries at this step
        if fold_artifacts:
            fold_list: list[tuple[str, Any]] = []
            for _fold_id, artifact_id in fold_artifacts.items():
                try:
                    obj = self.store.load_artifact(artifact_id)
                    fold_list.append((artifact_id, obj))
                except (KeyError, FileNotFoundError) as e:
                    logger.warning(f"Failed to load fold artifact {artifact_id}: {e}")
            if fold_list:
                if model_step_idx in artifact_map:
                    artifact_map[model_step_idx].extend(fold_list)
                else:
                    artifact_map[model_step_idx] = fold_list

        return artifact_map, source_artifact_map

    def _pick_chain_for_prediction(
        self,
        chains_df: Any,
        prediction: dict[str, Any],
    ) -> str:
        """Select the chain that matches a prediction.

        Chain selection is deterministic and fail-fast: if multiple chains
        remain after applying all available discriminators, a
        :class:`ResolverAmbiguityError` is raised.

        Args:
            chains_df: Polars DataFrame of chains (from
                ``store.get_chains_for_pipeline``).
            prediction: Prediction dictionary.

        Returns:
            The ``chain_id`` of the best-matching chain.
        """
        if len(chains_df) == 1:
            return chains_df["chain_id"][0]

        candidates: list[dict[str, Any]] = list(chains_df.iter_rows(named=True))

        # Direct chain_id from prediction/store row is authoritative.
        chain_id = prediction.get("chain_id")
        if chain_id:
            exact = [row for row in candidates if str(row.get("chain_id")) == str(chain_id)]
            if len(exact) == 1:
                return str(exact[0]["chain_id"])
            if len(exact) > 1:
                ids = [str(row.get("chain_id")) for row in self._sort_chain_candidates(exact)]
                raise ValueError(
                    f"Multiple chains matched explicit chain_id={chain_id!r}: {ids}"
                )

        # Structural branch-path matching.
        expected_branch_path = self._normalize_branch_path(prediction.get("branch_path"))
        if expected_branch_path is None and prediction.get("branch_id") is not None:
            expected_branch_path = self._normalize_branch_path([prediction.get("branch_id")])

        if expected_branch_path is not None and "branch_path" in chains_df.columns:
            branch_matched = [
                row for row in candidates
                if self._normalize_branch_path(row.get("branch_path")) == expected_branch_path
            ]
            if branch_matched:
                candidates = branch_matched

        # Match by model step index when present.
        step_idx = prediction.get("step_idx")
        if step_idx is not None:
            try:
                expected_step_idx = int(step_idx)
            except (TypeError, ValueError):
                expected_step_idx = None
            if expected_step_idx is not None:
                step_matched = [
                    row for row in candidates
                    if int(row.get("model_step_idx", -1) or -1) == expected_step_idx
                ]
                if step_matched:
                    candidates = step_matched

        # Match by model class.
        model_classname = prediction.get("model_classname") or prediction.get("model_class", "")
        if model_classname:
            class_matched = [
                row for row in candidates
                if str(row.get("model_class") or "") == str(model_classname)
            ]
            if class_matched:
                candidates = class_matched

        # Match by preprocessing chain.
        preprocessings = prediction.get("preprocessings", "")
        if preprocessings:
            preproc_matched = [
                row for row in candidates
                if str(row.get("preprocessings") or "") == str(preprocessings)
            ]
            if preproc_matched:
                candidates = preproc_matched

        if not candidates:
            raise FileNotFoundError("No chain candidates available for prediction resolution.")

        if len(candidates) == 1:
            return str(candidates[0]["chain_id"])

        candidate_ids = [str(row.get("chain_id")) for row in self._sort_chain_candidates(candidates)]
        logger.warning(
            "Ambiguous chain resolution for prediction — %d candidates. "
            "Falling back to filesystem resolution. "
            "Include chain_id or branch_path metadata to disambiguate.",
            len(candidate_ids),
        )
        return None

    @staticmethod
    def _sort_chain_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a deterministic ordering for candidate chains."""
        return sorted(
            candidates,
            key=lambda row: (
                str(row.get("pipeline_id") or ""),
                str(row.get("chain_id") or ""),
            ),
        )

    @staticmethod
    def _normalize_branch_path(value: Any) -> tuple[int, ...] | None:
        """Normalize branch-path encodings to a canonical tuple[int, ...]."""
        if value is None:
            return None

        parsed = value
        if isinstance(parsed, str):
            text = parsed.strip()
            if text == "":
                return None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return None

        if isinstance(parsed, int):
            return (int(parsed),)

        if isinstance(parsed, (list, tuple)):
            normalized: list[int] = []
            for item in parsed:
                try:
                    normalized.append(int(item))
                except (TypeError, ValueError):
                    return None
            return tuple(normalized)

        return None

    def _find_trace_location(self, trace_id: str) -> tuple[str, Path]:
        """Find the pipeline and run directory containing a trace.

        Searches all runs for a matching trace ID.

        Args:
            trace_id: Trace ID to find

        Returns:
            Tuple of (pipeline_uid, run_dir)

        Raises:
            FileNotFoundError: If trace not found
        """
        if not self.runs_dir.exists():
            raise FileNotFoundError(f"Runs directory not found: {self.runs_dir}")

        for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue

            for pipeline_uid in _list_pipelines(run_dir):
                trace_ids = _list_execution_traces(run_dir, pipeline_uid)
                if trace_id in trace_ids:
                    return pipeline_uid, run_dir

        raise FileNotFoundError(f"Trace not found: {trace_id}")

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _find_run_dir_for_prediction(
        self,
        prediction: dict[str, Any]
    ) -> Path | None:
        """Find run directory for a prediction.

        Args:
            prediction: Prediction dictionary

        Returns:
            Path to run directory or None
        """
        # Check for explicit run_dir
        if "run_dir" in prediction:
            return Path(prediction["run_dir"])

        # Try to find by pipeline_uid (and trace_id if available)
        pipeline_uid = prediction.get("pipeline_uid")
        trace_id = prediction.get("trace_id")
        if pipeline_uid:
            pipeline_dir = self._find_pipeline_dir(pipeline_uid, trace_id)
            if pipeline_dir:
                return pipeline_dir.parent

        # Try config_path
        config_path = prediction.get("config_path", "")
        if config_path:
            # Extract dataset name from config_path
            parts = Path(config_path).parts
            if parts:
                dataset_name = parts[0]
                # Find matching run directory
                matching_dirs = sorted(
                    self.runs_dir.glob(f"*_{dataset_name}"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                if matching_dirs:
                    return matching_dirs[0]

        # Fallback: use most recent run
        run_dirs = sorted(
            self.runs_dir.glob("*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        for run_dir in run_dirs:
            if run_dir.is_dir() and not run_dir.name.startswith("."):
                return run_dir

        return None

    def _load_pipeline_steps(
        self,
        run_dir: Path,
        pipeline_uid: str
    ) -> list[Any]:
        """Load pipeline steps from pipeline.json.

        Args:
            run_dir: Run directory path
            pipeline_uid: Pipeline unique identifier

        Returns:
            List of pipeline steps

        Raises:
            FileNotFoundError: If pipeline.json not found
        """
        pipeline_dir = run_dir / pipeline_uid
        pipeline_json = pipeline_dir / "pipeline.json"

        if not pipeline_json.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_json}")

        with open(pipeline_json, encoding="utf-8") as f:
            pipeline_data = json.load(f)

        if isinstance(pipeline_data, dict) and "steps" in pipeline_data:
            return list(pipeline_data["steps"])
        if isinstance(pipeline_data, list):
            return pipeline_data
        # Fallback: wrap in list
        return [pipeline_data]

    def _build_artifact_map_from_manifest(
        self,
        manifest: dict[str, Any],
        artifact_loader: ArtifactLoader,
        up_to_step: int | None = None
    ) -> dict[int, list[tuple[str, Any]]]:
        """Build artifact map from manifest.

        Creates a mapping of step_index -> list of (artifact_id, object) tuples.

        Args:
            manifest: Manifest dictionary
            artifact_loader: ArtifactLoader for loading artifacts
            up_to_step: Optional maximum step index to include

        Returns:
            Dictionary mapping step indices to artifact lists
        """
        artifact_map: dict[int, list[tuple[str, Any]]] = {}

        # Get artifacts from manifest
        artifacts_section = manifest.get("artifacts", {})
        if isinstance(artifacts_section, dict) and "items" in artifacts_section:
            items = artifacts_section["items"]
        elif isinstance(artifacts_section, list):
            items = artifacts_section
        else:
            items = []

        for item in items:
            artifact_id = item.get("artifact_id", "")
            step_index = item.get("step_index", 0)

            if up_to_step is not None and step_index > up_to_step:
                continue

            try:
                obj = artifact_loader.load_by_id(artifact_id)
                if step_index not in artifact_map:
                    artifact_map[step_index] = []
                artifact_map[step_index].append((artifact_id, obj))
            except (KeyError, FileNotFoundError, IsADirectoryError, OSError) as e:
                logger.warning(f"Failed to load artifact {artifact_id}: {e}")

        return artifact_map
