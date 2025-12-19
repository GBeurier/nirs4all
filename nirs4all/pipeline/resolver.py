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
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from nirs4all.pipeline.config.context import (
    ArtifactProvider,
    MapArtifactProvider,
    LoaderArtifactProvider,
)
from nirs4all.pipeline.trace import ExecutionTrace
from nirs4all.pipeline.storage.artifacts.artifact_loader import ArtifactLoader
from nirs4all.pipeline.storage.manifest_manager import ManifestManager


logger = logging.getLogger(__name__)


class SourceType(str, Enum):
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
    FOLDER = "folder"
    RUN = "run"
    ARTIFACT_ID = "artifact_id"
    BUNDLE = "bundle"
    TRACE_ID = "trace_id"
    MODEL_FILE = "model_file"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


class FoldStrategy(str, Enum):
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
    minimal_pipeline: List[Any] = field(default_factory=list)
    artifact_provider: Optional[ArtifactProvider] = None
    trace: Optional[ExecutionTrace] = None
    fold_strategy: FoldStrategy = FoldStrategy.WEIGHTED_AVERAGE
    fold_weights: Dict[int, float] = field(default_factory=dict)
    model_step_index: Optional[int] = None
    target_model: Dict[str, Any] = field(default_factory=dict)
    pipeline_uid: str = ""
    run_dir: Optional[Path] = None
    manifest: Dict[str, Any] = field(default_factory=dict)

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
        workspace_path: Union[str, Path],
        runs_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize prediction resolver.

        Args:
            workspace_path: Root workspace directory
            runs_dir: Optional runs directory override (default: workspace/runs)
        """
        self.workspace_path = Path(workspace_path)
        self.runs_dir = Path(runs_dir) if runs_dir else self.workspace_path / "runs"

    def resolve(
        self,
        source: Union[Dict[str, Any], str, Path, Any],
        verbose: int = 0
    ) -> ResolvedPrediction:
        """Resolve any prediction source to executable components.

        Detects the source type and delegates to the appropriate resolver.

        Args:
            source: Prediction source (dict, folder path, Run, artifact_id, bundle)
            verbose: Verbosity level for logging

        Returns:
            ResolvedPrediction with all components for replay

        Raises:
            ValueError: If source type cannot be determined or resolved
            FileNotFoundError: If referenced files/directories don't exist
        """
        source_type = self._detect_source_type(source)

        if verbose > 0:
            logger.info(f"Resolving prediction source of type: {source_type}")

        if source_type == SourceType.PREDICTION:
            # Type narrowing: we know source is dict at this point
            assert isinstance(source, dict)
            return self._resolve_from_prediction(source, verbose)
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

            # Could be a pipeline_uid
            if self._find_pipeline_dir(source_str):
                return SourceType.FOLDER

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
        if (path / "checkpoint").exists():
            return True
        return False

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
        trace_id: Optional[str] = None
    ) -> Optional[Path]:
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
        for run_dir in self.runs_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Check for exact match
            pipeline_dir = run_dir / pipeline_uid
            if pipeline_dir.is_dir() and (pipeline_dir / "manifest.yaml").exists():
                candidates.append(pipeline_dir)
                continue

            # Check for partial match (pipeline_uid might be just the hash part)
            for subdir in run_dir.iterdir():
                if subdir.is_dir() and pipeline_uid in subdir.name:
                    if (subdir / "manifest.yaml").exists():
                        candidates.append(subdir)

        if not candidates:
            return None

        # If no trace_id specified, return first candidate
        if not trace_id:
            return candidates[0]

        # If trace_id specified, find the candidate with matching trace
        for pipeline_dir in candidates:
            try:
                manifest_manager = ManifestManager(pipeline_dir.parent)
                manifest = manifest_manager.load_manifest(pipeline_uid)
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
        prediction: Dict[str, Any],
        verbose: int = 0
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

        Returns:
            ResolvedPrediction with components for replay
        """
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
            if "/" in config_path or "\\" in config_path:
                pipeline_uid = Path(config_path).name
            else:
                pipeline_uid = config_path

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
        manifest_manager = ManifestManager(run_dir)
        try:
            manifest = manifest_manager.load_manifest(pipeline_uid)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Manifest not found for pipeline: {pipeline_uid}\n"
                f"Expected at: {run_dir / pipeline_uid / 'manifest.yaml'}"
            )

        result.manifest = manifest

        # Try to load execution trace (Phase 2+)
        trace_id = prediction.get("trace_id")
        if trace_id:
            trace = manifest_manager.load_execution_trace(pipeline_uid, trace_id)
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
        folder: Union[str, Path],
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
            pipelines = ManifestManager(run_dir).list_pipelines()
            if not pipelines:
                raise ValueError(f"No pipelines found in folder: {folder}")
            pipeline_uid = pipelines[-1]  # Most recent

        result.run_dir = run_dir
        result.pipeline_uid = pipeline_uid

        # Load manifest
        manifest_manager = ManifestManager(run_dir)
        manifest = manifest_manager.load_manifest(pipeline_uid)
        result.manifest = manifest

        # Try to load latest execution trace
        trace = manifest_manager.get_latest_execution_trace(pipeline_uid)
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
        manifest_manager = ManifestManager(run_dir)
        manifest = manifest_manager.load_manifest(pipeline_uid)
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
        bundle_path: Union[str, Path],
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
        manifest_manager = ManifestManager(run_dir)
        manifest = manifest_manager.load_manifest(pipeline_uid)
        result.manifest = manifest

        trace = manifest_manager.load_execution_trace(pipeline_uid, trace_id)
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
        model_path: Union[str, Path],
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
        artifact_map: Dict[int, List[Tuple[str, Any]]] = {
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

    def _find_trace_location(self, trace_id: str) -> Tuple[str, Path]:
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

            manifest_manager = ManifestManager(run_dir)
            for pipeline_uid in manifest_manager.list_pipelines():
                trace_ids = manifest_manager.list_execution_traces(pipeline_uid)
                if trace_id in trace_ids:
                    return pipeline_uid, run_dir

        raise FileNotFoundError(f"Trace not found: {trace_id}")

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _find_run_dir_for_prediction(
        self,
        prediction: Dict[str, Any]
    ) -> Optional[Path]:
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
    ) -> List[Any]:
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

        with open(pipeline_json, "r", encoding="utf-8") as f:
            pipeline_data = json.load(f)

        if isinstance(pipeline_data, dict) and "steps" in pipeline_data:
            return list(pipeline_data["steps"])
        if isinstance(pipeline_data, list):
            return pipeline_data
        # Fallback: wrap in list
        return [pipeline_data]

    def _build_artifact_map_from_manifest(
        self,
        manifest: Dict[str, Any],
        artifact_loader: ArtifactLoader,
        up_to_step: Optional[int] = None
    ) -> Dict[int, List[Tuple[str, Any]]]:
        """Build artifact map from manifest.

        Creates a mapping of step_index -> list of (artifact_id, object) tuples.

        Args:
            manifest: Manifest dictionary
            artifact_loader: ArtifactLoader for loading artifacts
            up_to_step: Optional maximum step index to include

        Returns:
            Dictionary mapping step indices to artifact lists
        """
        artifact_map: Dict[int, List[Tuple[str, Any]]] = {}

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
