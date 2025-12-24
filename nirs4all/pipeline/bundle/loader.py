"""
Bundle Loader - Load and predict from exported bundles.

This module provides the BundleLoader class for loading prediction bundles
(.n4a format) and running predictions without needing the original workspace.

The loader supports:
    - Loading bundle metadata and structure
    - Extracting artifacts on-demand
    - Building artifact provider for prediction
    - Resolving bundles for PredictionResolver
    - Full support for branching pipelines
    - Full support for meta-models (stacking)
    - CV ensemble prediction with fold weights

Example:
    >>> from nirs4all.pipeline.bundle import BundleLoader
    >>>
    >>> loader = BundleLoader("model.n4a")
    >>> print(loader.metadata.pipeline_uid)
    >>> y_pred = loader.predict(X_new)
    >>>
    >>> # Or via PredictionResolver
    >>> resolver = PredictionResolver(workspace_path)
    >>> resolved = resolver.resolve("model.n4a")  # Automatically uses BundleLoader
"""

import base64
import json
import logging
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry

from nirs4all.pipeline.config.context import (
    ArtifactProvider,
    MapArtifactProvider,
)
from nirs4all.pipeline.trace import ExecutionTrace, StepArtifacts
from nirs4all.pipeline.trace.execution_trace import StepExecutionMode
from nirs4all.pipeline.storage.artifacts.operator_chain import OperatorChain, OperatorNode


logger = logging.getLogger(__name__)


@dataclass
class BundleMetadata:
    """Metadata for a prediction bundle.

    Contains information about the bundle format, source, and contents.

    Attributes:
        bundle_format_version: Version of the bundle format
        nirs4all_version: Version of nirs4all that created the bundle
        created_at: ISO timestamp of bundle creation
        pipeline_uid: UID of the source pipeline
        source_type: Type of source that was exported
        model_step_index: Index of the model step
        fold_strategy: Strategy for combining fold predictions
        preprocessing_chain: Summary of preprocessing steps
        trace_id: ID of the execution trace (if available)
        original_manifest: Subset of original manifest metadata
        partitioner_routing: Routing info for metadata partitioner branches
    """
    bundle_format_version: str = "1.0"
    nirs4all_version: str = ""
    created_at: str = ""
    pipeline_uid: str = ""
    source_type: str = ""
    model_step_index: Optional[int] = None
    fold_strategy: str = "weighted_average"
    preprocessing_chain: str = ""
    trace_id: Optional[str] = None
    original_manifest: Dict[str, Any] = field(default_factory=dict)
    partitioner_routing: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BundleMetadata":
        """Create BundleMetadata from dictionary.

        Args:
            data: Dictionary from bundle manifest.json

        Returns:
            BundleMetadata instance
        """
        return cls(
            bundle_format_version=data.get("bundle_format_version", "1.0"),
            nirs4all_version=data.get("nirs4all_version", ""),
            created_at=data.get("created_at", ""),
            pipeline_uid=data.get("pipeline_uid", ""),
            source_type=data.get("source_type", ""),
            model_step_index=data.get("model_step_index"),
            fold_strategy=data.get("fold_strategy", "weighted_average"),
            preprocessing_chain=data.get("preprocessing_chain", ""),
            trace_id=data.get("trace_id"),
            original_manifest=data.get("original_manifest", {}),
            partitioner_routing=data.get("partitioner_routing", {}),
        )


class BundleArtifactProvider(ArtifactProvider):
    """Artifact provider for bundles.

    Provides artifacts from a loaded bundle, with lazy loading and caching.
    Supports branching pipelines and meta-models.

    Attributes:
        bundle_path: Path to the bundle file
        artifact_cache: Cache of loaded artifacts
        artifact_index: Index mapping step/fold to artifact filenames
        fold_weights: Fold weights for CV ensemble
        step_info: Information about each step (operator_type, branch_path, etc.)
    """

    def __init__(
        self,
        bundle_path: Union[str, Path],
        artifact_index: Dict[str, str],
        fold_weights: Optional[Dict[int, float]] = None,
        step_info: Optional[Dict[int, Dict[str, Any]]] = None,
        trace: Optional[ExecutionTrace] = None
    ):
        """Initialize bundle artifact provider.

        Args:
            bundle_path: Path to the .n4a bundle file
            artifact_index: Mapping of artifact keys to filenames in bundle
            fold_weights: Optional fold weights for CV models
            step_info: Optional step metadata (operator_type, branch_path, etc.)
            trace: Optional execution trace for detailed step info
        """
        self.bundle_path = Path(bundle_path)
        self.artifact_index = artifact_index
        self.fold_weights = fold_weights or {}
        self.step_info = step_info or {}
        self.trace = trace
        self._cache: Dict[str, Any] = {}

    def get_artifact(
        self,
        step_index: int,
        fold_id: Optional[int] = None
    ) -> Optional[Any]:
        """Get a single artifact for a step.

        Args:
            step_index: 1-based step index
            fold_id: Optional fold ID for fold-specific artifacts

        Returns:
            Artifact object or None if not found
        """
        if fold_id is not None:
            key = f"step_{step_index}_fold{fold_id}"
        else:
            key = f"step_{step_index}"

        return self._load_artifact(key)

    def get_artifacts_for_step(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None
    ) -> List[Tuple[str, Any]]:
        """Get all artifacts for a step.

        Args:
            step_index: 1-based step index
            branch_path: Optional branch path filter

        Returns:
            List of (artifact_id, artifact_object) tuples
        """
        results = []
        prefix = f"step_{step_index}"

        for key in self.artifact_index:
            if key == prefix or key.startswith(f"{prefix}_"):
                # Filter by branch_path if specified and we have trace
                if branch_path is not None and self.trace:
                    step = self.trace.get_step(step_index)
                    if step and step.branch_path:
                        # Check if step is on the target branch or is a shared step
                        if step.branch_path != branch_path:
                            # Check if it's a parent branch (allowed)
                            if not self._is_parent_branch(branch_path, step.branch_path):
                                continue

                artifact = self._load_artifact(key)
                if artifact is not None:
                    results.append((key, artifact))

        return results

    def _is_parent_branch(
        self,
        target_path: List[int],
        check_path: List[int]
    ) -> bool:
        """Check if check_path is a parent of target_path."""
        if len(check_path) >= len(target_path):
            return False
        return target_path[:len(check_path)] == check_path

    def get_fold_artifacts(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None
    ) -> List[Tuple[int, Any]]:
        """Get all fold-specific artifacts for a step.

        Args:
            step_index: 1-based step index
            branch_path: Optional branch path filter

        Returns:
            List of (fold_id, artifact_object) tuples, sorted by fold_id
        """
        results = []

        for key in self.artifact_index:
            if key.startswith(f"step_{step_index}_fold"):
                try:
                    fold_id = int(key.split("_fold")[1])

                    # Filter by branch_path if specified and we have trace
                    if branch_path is not None and self.trace:
                        step = self.trace.get_step(step_index)
                        if step and step.branch_path:
                            if step.branch_path != branch_path:
                                if not self._is_parent_branch(branch_path, step.branch_path):
                                    continue

                    artifact = self._load_artifact(key)
                    if artifact is not None:
                        results.append((fold_id, artifact))
                except (ValueError, IndexError):
                    pass

        return sorted(results, key=lambda x: x[0])

    def get_step_operator_type(self, step_index: int) -> Optional[str]:
        """Get the operator type for a step.

        Args:
            step_index: 1-based step index

        Returns:
            Operator type string or None
        """
        # Try trace first
        if self.trace:
            step = self.trace.get_step(step_index)
            if step:
                return step.operator_type

        # Fallback to step_info
        if step_index in self.step_info:
            return self.step_info[step_index].get("operator_type")

        return None

    def get_meta_model_sources(
        self,
        step_index: int
    ) -> List[Tuple[int, str]]:
        """Get source model info for a meta-model step.

        Args:
            step_index: Step index of the meta-model

        Returns:
            List of (source_step_index, artifact_id) tuples
        """
        sources = []

        # Check trace for dependency info
        if self.trace:
            step = self.trace.get_step(step_index)
            if step and step.metadata:
                source_models = step.metadata.get("source_models", [])
                for source_info in source_models:
                    if isinstance(source_info, dict):
                        src_step = source_info.get("step_index", 0)
                        src_id = source_info.get("artifact_id", "")
                        sources.append((src_step, src_id))

        return sources

    def has_artifacts_for_step(self, step_index: int) -> bool:
        """Check if artifacts exist for a step.

        Args:
            step_index: 1-based step index

        Returns:
            True if artifacts are available for this step
        """
        prefix = f"step_{step_index}"
        return any(
            k == prefix or k.startswith(f"{prefix}_")
            for k in self.artifact_index
        )

    def _load_artifact(self, key: str) -> Optional[Any]:
        """Load artifact from bundle.

        Args:
            key: Artifact key

        Returns:
            Loaded artifact or None
        """
        # Check cache
        if key in self._cache:
            return self._cache[key]

        # Check index
        if key not in self.artifact_index:
            return None

        filename = self.artifact_index[key]

        try:
            import io
            import joblib

            with zipfile.ZipFile(self.bundle_path, 'r') as zf:
                with zf.open(f"artifacts/{filename}") as f:
                    buffer = io.BytesIO(f.read())
                    artifact = joblib.load(buffer)

            # Cache for future use
            self._cache[key] = artifact
            return artifact

        except Exception as e:
            logger.warning(f"Failed to load artifact {key}: {e}")
            return None

    def get_fold_weights(self) -> Dict[int, float]:
        """Get fold weights for CV ensemble.

        Returns:
            Fold weights dictionary
        """
        return self.fold_weights.copy()


class BundleLoader:
    """Load and use prediction bundles.

    Provides functionality for loading .n4a bundles, extracting metadata,
    and running predictions.

    Attributes:
        bundle_path: Path to the bundle file
        metadata: Bundle metadata
        trace: Execution trace (if available)
        pipeline_config: Pipeline configuration
        fold_weights: Fold weights for CV ensemble
        artifact_provider: Provider for artifacts

    Example:
        >>> loader = BundleLoader("model.n4a")
        >>> print(f"Pipeline: {loader.metadata.pipeline_uid}")
        >>> print(f"Preprocessing: {loader.metadata.preprocessing_chain}")
        >>> y_pred = loader.predict(X_new)
    """

    def __init__(self, bundle_path: Union[str, Path]):
        """Initialize bundle loader.

        Args:
            bundle_path: Path to the .n4a bundle file

        Raises:
            FileNotFoundError: If bundle file doesn't exist
            ValueError: If bundle format is invalid
        """
        self.bundle_path = Path(bundle_path)

        if not self.bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found: {bundle_path}")

        if not zipfile.is_zipfile(self.bundle_path):
            raise ValueError(f"Invalid bundle format: {bundle_path}")

        # Load bundle contents
        self.metadata: Optional[BundleMetadata] = None
        self.trace: Optional[ExecutionTrace] = None
        self.pipeline_config: Dict[str, Any] = {}
        self.fold_weights: Dict[int, float] = {}
        self._artifact_index: Dict[str, str] = {}
        self.artifact_provider: Optional[BundleArtifactProvider] = None

        self._load_bundle()

    def _load_bundle(self) -> None:
        """Load bundle metadata and build artifact index."""
        with zipfile.ZipFile(self.bundle_path, 'r') as zf:
            # Load manifest.json
            if 'manifest.json' in zf.namelist():
                with zf.open('manifest.json') as f:
                    manifest_data = json.load(f)
                    self.metadata = BundleMetadata.from_dict(manifest_data)
            else:
                raise ValueError("Bundle missing manifest.json")

            # Load pipeline.json
            if 'pipeline.json' in zf.namelist():
                with zf.open('pipeline.json') as f:
                    self.pipeline_config = json.load(f)

            # Load trace.json (if present)
            if 'trace.json' in zf.namelist():
                with zf.open('trace.json') as f:
                    trace_data = json.load(f)
                    self.trace = ExecutionTrace.from_dict(trace_data)

            # Load fold_weights.json (if present)
            if 'fold_weights.json' in zf.namelist():
                with zf.open('fold_weights.json') as f:
                    weights_data = json.load(f)
                    # Convert string keys back to int
                    self.fold_weights = {int(k): v for k, v in weights_data.items()}

            # Build artifact index
            for name in zf.namelist():
                if name.startswith('artifacts/') and name != 'artifacts/':
                    filename = name.split('/')[-1]
                    # Parse key from filename (e.g., "step_1_SNV.joblib" -> "step_1")
                    key = self._filename_to_key(filename)
                    self._artifact_index[key] = filename

        # Create artifact provider with trace for full functionality
        self.artifact_provider = BundleArtifactProvider(
            bundle_path=self.bundle_path,
            artifact_index=self._artifact_index,
            fold_weights=self.fold_weights,
            step_info={},  # Step info is now inferred from trace
            trace=self.trace
        )

    def _filename_to_key(self, filename: str) -> str:
        """Convert artifact filename to key.

        Args:
            filename: Artifact filename (e.g., "step_1_SNV.joblib")

        Returns:
            Key (e.g., "step_1", "step_4_fold0", or "step_3_SNV" for substeps)
        """
        # Remove extension
        name = filename.rsplit('.', 1)[0]

        # Parse step and fold info
        parts = name.split('_')

        if len(parts) >= 2 and parts[0] == 'step':
            step_idx = parts[1]

            # Check for fold info
            for i, part in enumerate(parts):
                if part.startswith('fold'):
                    return f"step_{step_idx}_{part}"

            # If there are more than 2 parts (step_3_SNV), it's a substep artifact
            # Return full name (without extension) as the key to handle multiple
            # artifacts per step (e.g., feature_augmentation with SNV and SG)
            if len(parts) > 2:
                return name  # e.g., "step_3_StandardNormalVariate"

            return f"step_{step_idx}"

        return name

    def predict(
        self,
        X: np.ndarray,
        branch_path: Optional[List[int]] = None
    ) -> np.ndarray:
        """Run prediction on input data.

        Applies all preprocessing steps and model(s) from the bundle.
        Supports branching pipelines, meta-models (stacking), and CV ensembles.

        Args:
            X: Input features as numpy array
            branch_path: Optional branch path filter for multi-branch pipelines

        Returns:
            Predictions as numpy array
        """
        if self.artifact_provider is None:
            raise RuntimeError("Bundle not loaded properly: no artifact provider")

        X_current = X.copy()

        # Get step execution order from trace or artifact index
        if self.trace:
            # Use trace for accurate step order and info
            return self._predict_with_trace(X_current, branch_path)
        else:
            # Fallback: infer from artifact index
            return self._predict_from_index(X_current)

    def _predict_with_trace(
        self,
        X: np.ndarray,
        branch_path: Optional[List[int]] = None
    ) -> np.ndarray:
        """Predict using execution trace for step info.

        Args:
            X: Input features
            branch_path: Optional branch path filter

        Returns:
            Predictions
        """
        X_current = X.copy()
        model_step = self.metadata.model_step_index if self.metadata else None
        y_processing_step_idx = None  # Track y_processing step for inverse_transform

        # Get steps up to model from trace
        if model_step is not None:
            steps = self.trace.get_steps_up_to_model()
        else:
            steps = self.trace.steps

        for step in steps:
            step_idx = step.step_index
            op_type = step.operator_type

            # Filter by branch if specified
            if branch_path is not None:
                if step.branch_path:
                    # Skip if not on target branch and not a parent branch
                    if step.branch_path != branch_path:
                        if not self._is_parent_of(branch_path, step.branch_path):
                            continue

            # Skip steps that were skipped during training
            if step.execution_mode == StepExecutionMode.SKIP:
                continue

            # Handle different operator types
            if op_type == "meta_model":
                # Stacking: get predictions from source models first
                y_pred = self._predict_meta_model(X_current, step, branch_path)
                return self._apply_y_inverse_transform(y_pred, y_processing_step_idx, branch_path)

            elif op_type in ("model",):
                # Regular model step - may have CV folds
                y_pred = self._predict_model_step(X_current, step_idx, branch_path)
                return self._apply_y_inverse_transform(y_pred, y_processing_step_idx, branch_path)

            elif op_type == "y_processing":
                # y_processing transforms targets, not features - skip but track for inverse
                y_processing_step_idx = step_idx
                continue

            elif op_type == "feature_augmentation":
                # Feature augmentation: apply each transformer and concatenate with original
                X_current = self._transform_feature_augmentation(X_current, step_idx, branch_path)

            else:
                # Preprocessing step - transform X
                X_current = self._transform_step(X_current, step_idx, branch_path)

        raise RuntimeError("No model step found in bundle")

    def _predict_meta_model(
        self,
        X: np.ndarray,
        meta_step,
        branch_path: Optional[List[int]] = None
    ) -> np.ndarray:
        """Predict using a meta-model (stacking).

        Args:
            X: Input features (already preprocessed)
            meta_step: ExecutionStep for the meta-model
            branch_path: Optional branch path filter

        Returns:
            Predictions from meta-model
        """
        # Get source model info from trace metadata
        source_models = []
        if meta_step.metadata:
            source_info = meta_step.metadata.get("source_models", [])
            for src in source_info:
                if isinstance(src, dict):
                    src_step_idx = src.get("step_index", 0)
                    source_models.append(src_step_idx)

        # If no source info in metadata, find model steps before this one
        if not source_models and self.trace:
            for prev_step in self.trace.steps:
                if prev_step.step_index < meta_step.step_index:
                    if prev_step.operator_type == "model":
                        source_models.append(prev_step.step_index)

        # Get predictions from each source model
        source_predictions = []
        for src_step_idx in source_models:
            # Run prediction on source model
            y_src = self._predict_model_step(X, src_step_idx, branch_path)
            source_predictions.append(y_src)

        if not source_predictions:
            raise RuntimeError(
                f"Meta-model at step {meta_step.step_index} has no source models"
            )

        # Stack source predictions as features for meta-model
        X_meta = np.column_stack(source_predictions)

        # Get and apply meta-model
        meta_step_idx = meta_step.step_index
        fold_artifacts = self.artifact_provider.get_fold_artifacts(meta_step_idx, branch_path)

        if fold_artifacts:
            # CV ensemble on meta-model
            fold_preds = []
            for fold_id, model in fold_artifacts:
                weight = self.fold_weights.get(fold_id, 1.0)
                y_fold = model.predict(X_meta)
                fold_preds.append((weight, y_fold))

            if self.fold_weights:
                total_weight = sum(w for w, _ in fold_preds)
                return np.asarray(sum(w * y for w, y in fold_preds) / total_weight)
            else:
                return np.mean([y for _, y in fold_preds], axis=0)
        else:
            # Single meta-model
            artifacts = self.artifact_provider.get_artifacts_for_step(meta_step_idx, branch_path)
            if not artifacts:
                raise RuntimeError(f"No artifacts for meta-model step {meta_step_idx}")
            _, model = artifacts[0]
            return model.predict(X_meta)

    def _predict_model_step(
        self,
        X: np.ndarray,
        step_idx: int,
        branch_path: Optional[List[int]] = None
    ) -> np.ndarray:
        """Predict from a model step (handles CV ensemble).

        Args:
            X: Input features
            step_idx: Model step index
            branch_path: Optional branch path filter

        Returns:
            Predictions
        """
        fold_artifacts = self.artifact_provider.get_fold_artifacts(step_idx, branch_path)

        if fold_artifacts:
            # CV ensemble - average predictions across folds
            fold_preds = []
            for fold_id, model in fold_artifacts:
                weight = self.fold_weights.get(fold_id, 1.0)
                y_fold = model.predict(X)
                fold_preds.append((weight, y_fold))

            if self.fold_weights:
                total_weight = sum(w for w, _ in fold_preds)
                return np.asarray(sum(w * y for w, y in fold_preds) / total_weight)
            else:
                return np.mean([y for _, y in fold_preds], axis=0)
        else:
            # Single model
            artifacts = self.artifact_provider.get_artifacts_for_step(step_idx, branch_path)
            if not artifacts:
                raise RuntimeError(f"No artifacts for model step {step_idx}")
            _, model = artifacts[0]
            return model.predict(X)

    def _transform_step(
        self,
        X: np.ndarray,
        step_idx: int,
        branch_path: Optional[List[int]] = None
    ) -> np.ndarray:
        """Apply transformation step to X.

        Args:
            X: Input features
            step_idx: Step index
            branch_path: Optional branch path filter

        Returns:
            Transformed X
        """
        artifacts = self.artifact_provider.get_artifacts_for_step(step_idx, branch_path)

        for _, transformer in artifacts:
            if hasattr(transformer, 'transform'):
                X = transformer.transform(X)

        return X

    def _apply_y_inverse_transform(
        self,
        y_pred: np.ndarray,
        y_processing_step_idx: Optional[int],
        branch_path: Optional[List[int]] = None
    ) -> np.ndarray:
        """Apply inverse transform to predictions if y_processing was used.

        Args:
            y_pred: Model predictions (possibly in scaled space)
            y_processing_step_idx: Step index of y_processing transformer, or None
            branch_path: Optional branch path filter

        Returns:
            Predictions in original scale (inverse transformed if applicable)
        """
        if y_processing_step_idx is None:
            return y_pred

        # Get the y_processing transformer
        artifacts = self.artifact_provider.get_artifacts_for_step(
            y_processing_step_idx, branch_path
        )

        if not artifacts:
            return y_pred

        # Apply inverse_transform from each y_processing artifact
        y_current = y_pred
        for _, transformer in artifacts:
            if hasattr(transformer, 'inverse_transform'):
                # Ensure proper shape for inverse_transform
                if y_current.ndim == 1:
                    y_current = y_current.reshape(-1, 1)
                    y_current = transformer.inverse_transform(y_current)
                    y_current = y_current.ravel()
                else:
                    y_current = transformer.inverse_transform(y_current)

        return y_current

    def _transform_feature_augmentation(
        self,
        X: np.ndarray,
        step_idx: int,
        branch_path: Optional[List[int]] = None
    ) -> np.ndarray:
        """Apply feature augmentation transformation.

        Feature augmentation creates multiple feature channels by applying
        each transformer and concatenating the results with the original features.

        For 'add' mode (default): [original, transformed1, transformed2, ...]
        Each transformer is applied to the original X, not chained.

        Args:
            X: Input features
            step_idx: Step index
            branch_path: Optional branch path filter

        Returns:
            Augmented features (concatenated original + transformed channels)
        """
        artifacts = self.artifact_provider.get_artifacts_for_step(step_idx, branch_path)

        if not artifacts:
            return X

        # Start with original features
        feature_channels = [X]

        # Apply each transformer to original X and collect results
        for _, transformer in artifacts:
            if hasattr(transformer, 'transform'):
                X_transformed = transformer.transform(X)
                feature_channels.append(X_transformed)

        # Concatenate all feature channels horizontally
        return np.hstack(feature_channels)

    def _predict_from_index(self, X: np.ndarray) -> np.ndarray:
        """Fallback prediction when no trace is available.

        Uses artifact index to infer step order. Limited functionality
        (no branching, no meta-model support).

        Args:
            X: Input features

        Returns:
            Predictions
        """
        X_current = X.copy()
        model_step = self.metadata.model_step_index if self.metadata else None
        y_processing_step_idx = None  # Track y_processing step for inverse_transform

        # Get sorted step indices
        step_indices = set()
        for key in self._artifact_index:
            if key.startswith('step_'):
                parts = key.split('_')
                if len(parts) >= 2:
                    try:
                        step_indices.add(int(parts[1]))
                    except ValueError:
                        pass

        # Process each step
        for step_idx in sorted(step_indices):
            is_model = (step_idx == model_step)

            # Check operator type from step_info
            op_type = None
            if step_idx in self.step_info:
                op_type = self.step_info[step_idx].get("operator_type")

            # Handle y_processing - skip but track for inverse_transform
            if op_type == "y_processing":
                y_processing_step_idx = step_idx
                continue

            artifacts = self.artifact_provider.get_artifacts_for_step(step_idx)

            if is_model:
                y_pred = self._predict_model_step(X_current, step_idx)
                return self._apply_y_inverse_transform(y_pred, y_processing_step_idx)
            elif op_type == "feature_augmentation":
                # Feature augmentation: apply each transformer and concatenate with original
                X_current = self._transform_feature_augmentation(X_current, step_idx)
            else:
                # Preprocessing step
                X_current = self._transform_step(X_current, step_idx)

        raise RuntimeError("No model step found in bundle")

    def _is_parent_of(
        self,
        target_path: List[int],
        check_path: List[int]
    ) -> bool:
        """Check if check_path is a parent of target_path."""
        if len(check_path) >= len(target_path):
            return False
        return target_path[:len(check_path)] == check_path

    def get_step_info(self) -> List[Dict[str, Any]]:
        """Get information about steps in the bundle.

        Returns:
            List of step info dictionaries
        """
        steps = []

        if self.trace:
            for step in self.trace.steps:
                steps.append({
                    "step_index": step.step_index,
                    "operator_type": step.operator_type,
                    "operator_class": step.operator_class,
                    "has_artifacts": step.has_artifacts(),
                    "artifact_count": len(step.artifacts.artifact_ids),
                })
        else:
            # Build from artifact index
            step_indices = set()
            for key in self._artifact_index:
                if key.startswith('step_'):
                    parts = key.split('_')
                    if len(parts) >= 2:
                        try:
                            step_indices.add(int(parts[1]))
                        except ValueError:
                            pass

            for idx in sorted(step_indices):
                if self.artifact_provider:
                    artifacts = self.artifact_provider.get_artifacts_for_step(idx)
                    steps.append({
                        "step_index": idx,
                        "artifact_count": len(artifacts),
                    })
                else:
                    steps.append({
                        "step_index": idx,
                        "artifact_count": 0,
                    })

        return steps

    def to_resolved_prediction(self) -> Any:
        """Convert bundle to ResolvedPrediction for use with Predictor.

        Returns:
            ResolvedPrediction instance
        """
        from nirs4all.pipeline.resolver import (
            ResolvedPrediction,
            SourceType,
            FoldStrategy,
        )

        if self.metadata is None:
            raise RuntimeError("Bundle not loaded properly: no metadata")

        # Parse fold strategy
        try:
            fold_strategy = FoldStrategy(self.metadata.fold_strategy)
        except ValueError:
            fold_strategy = FoldStrategy.WEIGHTED_AVERAGE

        return ResolvedPrediction(
            source_type=SourceType.BUNDLE,
            minimal_pipeline=self.pipeline_config.get("steps", []),
            artifact_provider=self.artifact_provider,
            trace=self.trace,
            fold_strategy=fold_strategy,
            fold_weights=self.fold_weights,
            model_step_index=self.metadata.model_step_index,
            target_model={
                "pipeline_uid": self.metadata.pipeline_uid,
                "preprocessing_chain": self.metadata.preprocessing_chain,
            },
            pipeline_uid=self.metadata.pipeline_uid,
            run_dir=None,
            manifest={},
        )

    def __repr__(self) -> str:
        pipeline_uid = self.metadata.pipeline_uid if self.metadata else "unknown"
        return (
            f"BundleLoader(path={self.bundle_path.name!r}, "
            f"pipeline_uid={pipeline_uid!r}, "
            f"artifacts={len(self._artifact_index)})"
        )

    def get_chain_for_artifact(self, artifact_key: str) -> Optional[OperatorChain]:
        """Get the operator chain for an artifact from the bundle.

        Args:
            artifact_key: Artifact key (e.g., "step_1", "step_4_fold0")

        Returns:
            OperatorChain for the artifact or None if not found
        """
        if not self.trace:
            return None

        # Parse step and fold from key
        parts = artifact_key.split("_")
        if len(parts) < 2 or parts[0] != "step":
            return None

        try:
            step_index = int(parts[1])
        except ValueError:
            return None

        # Get step from trace
        step = self.trace.get_step(step_index)
        if not step:
            return None

        # Build chain from step's chain path or construct from step info
        if step.input_chain_path:
            chain = OperatorChain.from_path(
                step.input_chain_path,
                pipeline_id=self.metadata.pipeline_uid if self.metadata else ""
            )
            # Add this step's node
            node = OperatorNode(
                step_index=step.step_index,
                operator_class=step.operator_class,
                branch_path=step.branch_path,
                substep_index=step.substep_index,
            )
            return chain.append(node)
        else:
            # Construct a simple chain from step info
            node = OperatorNode(
                step_index=step.step_index,
                operator_class=step.operator_class,
                branch_path=step.branch_path,
                substep_index=step.substep_index,
            )
            return OperatorChain(
                nodes=[node],
                pipeline_id=self.metadata.pipeline_uid if self.metadata else ""
            )

    def get_merged_chains(
        self,
        import_context_chain: OperatorChain,
        step_offset: int = 0
    ) -> Dict[str, OperatorChain]:
        """Get all artifact chains merged with an import context chain.

        Used when importing a bundle into another pipeline. Each artifact's
        chain is prefixed with the import context chain.

        Args:
            import_context_chain: Chain from the importing pipeline context
            step_offset: Step offset to apply to bundle steps

        Returns:
            Dict mapping artifact keys to merged chains
        """
        merged_chains: Dict[str, OperatorChain] = {}

        for artifact_key in self._artifact_index:
            artifact_chain = self.get_chain_for_artifact(artifact_key)
            if artifact_chain:
                merged = artifact_chain.merge_with_prefix(
                    import_context_chain,
                    step_offset=step_offset
                )
                merged_chains[artifact_key] = merged

        return merged_chains

    def import_artifacts_to_registry(
        self,
        registry: 'ArtifactRegistry',
        import_context_chain: Optional[OperatorChain] = None,
        step_offset: int = 0,
        new_pipeline_id: Optional[str] = None
    ) -> Dict[str, str]:
        """Import bundle artifacts into an artifact registry.

        Registers all artifacts from this bundle into the target registry,
        optionally merging with an import context chain for proper V3 tracking.

        Args:
            registry: Target ArtifactRegistry to import into
            import_context_chain: Optional chain from import context to prefix
            step_offset: Step offset for imported artifacts
            new_pipeline_id: New pipeline ID for imported artifacts

        Returns:
            Dict mapping original artifact keys to new artifact IDs
        """
        from nirs4all.pipeline.storage.artifacts.types import ArtifactType

        id_mapping: Dict[str, str] = {}

        # Determine pipeline_id
        pipeline_id = new_pipeline_id or (
            self.metadata.pipeline_uid if self.metadata else "imported"
        )

        # Get merged chains if import context provided
        if import_context_chain:
            merged_chains = self.get_merged_chains(import_context_chain, step_offset)
        else:
            merged_chains = {}

        # Import each artifact
        for artifact_key in self._artifact_index:
            artifact = self._load_artifact(artifact_key)
            if artifact is None:
                continue

            # Get chain for this artifact
            if artifact_key in merged_chains:
                chain = merged_chains[artifact_key]
            else:
                chain = self.get_chain_for_artifact(artifact_key)
                if chain:
                    chain = chain.with_pipeline_id(pipeline_id)

            if chain is None:
                # Fallback: create minimal chain
                parts = artifact_key.split("_")
                step_idx = int(parts[1]) if len(parts) >= 2 else 0
                chain = OperatorChain(
                    nodes=[OperatorNode(
                        step_index=step_idx + step_offset,
                        operator_class=artifact.__class__.__name__,
                    )],
                    pipeline_id=pipeline_id
                )

            # Determine fold_id from key
            fold_id = None
            if "_fold" in artifact_key:
                try:
                    fold_id = int(artifact_key.split("_fold")[1])
                except (ValueError, IndexError):
                    pass

            # Determine artifact type
            if hasattr(artifact, 'predict'):
                artifact_type = ArtifactType.MODEL
            elif hasattr(artifact, 'transform'):
                artifact_type = ArtifactType.TRANSFORMER
            else:
                artifact_type = ArtifactType.MODEL

            # Generate chain path
            chain_path = chain.to_path()

            # Register with registry
            artifact_id = registry.generate_id(chain_path, fold_id, pipeline_id)
            record = registry.register(
                obj=artifact,
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                chain_path=chain_path,
            )

            id_mapping[artifact_key] = artifact_id
            logger.debug(f"Imported bundle artifact {artifact_key} as {artifact_id}")

        return id_mapping

    # =========================================================================
    # Phase 4: Metadata-Based Prediction Routing
    # =========================================================================

    def has_partitioner_routing(self) -> bool:
        """Check if the bundle has metadata partitioner routing info.

        Returns:
            True if the bundle contains partitioner routing configuration.
        """
        if self.metadata is None:
            return False
        return bool(self.metadata.partitioner_routing)

    def get_partitioner_routing(self, step_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get partitioner routing info for a specific step or all steps.

        Args:
            step_index: Specific step index, or None for all

        Returns:
            Routing info dict or None
        """
        if not self.has_partitioner_routing():
            return None

        if step_index is not None:
            return self.metadata.partitioner_routing.get(str(step_index))

        return self.metadata.partitioner_routing

    def predict_with_metadata(
        self,
        X: np.ndarray,
        metadata: Dict[str, np.ndarray],
        fallback_branch: Optional[int] = None
    ) -> np.ndarray:
        """Run prediction with metadata-based sample routing.

        For bundles with metadata partitioner branches, this method routes
        each sample to the appropriate branch based on its metadata value.
        Each sample is processed by the transformers and models from its
        matching branch.

        Args:
            X: Input features as numpy array (n_samples, n_features)
            metadata: Dict mapping column names to value arrays. Must include
                the column used for partitioning during training.
            fallback_branch: Optional branch ID to use for samples with
                unknown metadata values. If None, raises error for unknowns.

        Returns:
            Predictions as numpy array

        Raises:
            ValueError: If required metadata column is missing or
                samples have unknown values without fallback.

        Example:
            >>> loader = BundleLoader("model.n4a")
            >>> X_new = np.random.randn(100, 500)
            >>> metadata = {"site": np.array(["A"]*50 + ["B"]*50)}
            >>> y_pred = loader.predict_with_metadata(X_new, metadata)
        """
        if not self.has_partitioner_routing():
            # No partitioner - use standard prediction
            logger.debug("No partitioner routing info, using standard prediction")
            return self.predict(X)

        if self.artifact_provider is None:
            raise RuntimeError("Bundle not loaded properly: no artifact provider")

        n_samples = X.shape[0]
        y_pred = np.full(n_samples, np.nan)

        # Process each partitioner step
        for step_idx_str, routing_info in self.metadata.partitioner_routing.items():
            step_idx = int(step_idx_str)
            column = routing_info.get("column")

            if not column:
                logger.warning(f"Partitioner at step {step_idx} has no column info")
                continue

            if column not in metadata:
                available_cols = list(metadata.keys())
                raise ValueError(
                    f"Metadata column '{column}' required for prediction routing "
                    f"but not found. Available columns: {available_cols}"
                )

            column_values = metadata[column]
            if len(column_values) != n_samples:
                raise ValueError(
                    f"Metadata column '{column}' has {len(column_values)} values "
                    f"but X has {n_samples} samples"
                )

            # Get partition configuration
            partitions = routing_info.get("partitions", [])
            group_values = routing_info.get("group_values", {})

            if not partitions:
                logger.warning(f"Partitioner at step {step_idx} has no partition info")
                continue

            # Build partition-to-branch mapping
            partition_to_branch = {name: idx for idx, name in enumerate(partitions)}

            # Build value-to-partition mapping
            value_to_partition = {}
            if group_values:
                # Grouped values
                for group_name, values in group_values.items():
                    for v in values:
                        value_to_partition[v] = group_name
            # Add direct partition names as values
            for partition_name in partitions:
                if partition_name not in value_to_partition:
                    value_to_partition[partition_name] = partition_name

            # Route samples to branches
            sample_branches: Dict[int, List[int]] = {}  # branch_id -> sample_indices
            unknown_samples = []

            for sample_idx, value in enumerate(column_values):
                # Find partition for this value
                partition_name = value_to_partition.get(value)

                if partition_name is None:
                    # Try string conversion
                    partition_name = value_to_partition.get(str(value))

                if partition_name is None:
                    # Check if value itself is a partition name
                    if value in partition_to_branch:
                        partition_name = value
                    elif str(value) in partition_to_branch:
                        partition_name = str(value)

                if partition_name is not None and partition_name in partition_to_branch:
                    branch_id = partition_to_branch[partition_name]
                    if branch_id not in sample_branches:
                        sample_branches[branch_id] = []
                    sample_branches[branch_id].append(sample_idx)
                else:
                    unknown_samples.append(sample_idx)

            # Handle unknown samples
            if unknown_samples:
                if fallback_branch is not None:
                    if fallback_branch not in sample_branches:
                        sample_branches[fallback_branch] = []
                    sample_branches[fallback_branch].extend(unknown_samples)
                    logger.warning(
                        f"{len(unknown_samples)} samples with unknown metadata values "
                        f"routed to fallback branch {fallback_branch}"
                    )
                else:
                    unknown_values = set(column_values[i] for i in unknown_samples[:5])
                    raise ValueError(
                        f"{len(unknown_samples)} samples have metadata values "
                        f"not seen during training: {unknown_values}. "
                        f"Use fallback_branch parameter to handle unknown values."
                    )

            # Process each branch
            for branch_id, sample_indices in sample_branches.items():
                if not sample_indices:
                    continue

                # Extract subset of data for this branch
                X_branch = X[sample_indices]

                logger.debug(
                    f"Branch {branch_id} ({partitions[branch_id] if branch_id < len(partitions) else 'fallback'}): "
                    f"{len(sample_indices)} samples"
                )

                # Run prediction with branch path
                branch_path = [branch_id]
                y_branch = self.predict(X_branch, branch_path=branch_path)

                # Store predictions in correct positions
                for local_idx, global_idx in enumerate(sample_indices):
                    y_pred[global_idx] = y_branch[local_idx] if len(y_branch.shape) == 1 else y_branch[local_idx, 0]

        # Check for any unprocessed samples
        unprocessed = np.isnan(y_pred)
        if np.any(unprocessed):
            n_unprocessed = np.sum(unprocessed)
            logger.warning(f"{n_unprocessed} samples were not processed (no matching branch)")

        return y_pred

    def get_required_metadata_columns(self) -> List[str]:
        """Get the metadata columns required for prediction routing.

        Returns:
            List of column names needed for routing, empty if no routing needed.
        """
        if not self.has_partitioner_routing():
            return []

        columns = []
        for routing_info in self.metadata.partitioner_routing.values():
            column = routing_info.get("column")
            if column and column not in columns:
                columns.append(column)

        return columns


def load_bundle(bundle_path: Union[str, Path]) -> BundleLoader:
    """Convenience function to load a bundle.

    Args:
        bundle_path: Path to the .n4a bundle file

    Returns:
        BundleLoader instance
    """
    return BundleLoader(bundle_path)
