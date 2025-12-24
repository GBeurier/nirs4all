"""
Bundle Generator - Export trained pipelines as standalone bundles.

This module provides the BundleGenerator class for exporting trained pipelines
to various bundle formats (.n4a, .n4a.py) for deployment, sharing, or archival.

Bundle Formats:
    .n4a: Full nirs4all bundle (ZIP archive) containing:
        - manifest.json: Bundle metadata and version info
        - pipeline.json: Minimal pipeline configuration
        - trace.json: Execution trace for deterministic replay
        - artifacts/: Directory with artifact binaries
        - fold_weights.json: CV fold weights (if applicable)

    .n4a.py: Portable Python script with:
        - Embedded artifacts (base64 encoded)
        - Standalone predict() function
        - No nirs4all dependency (only numpy, joblib required)

Example:
    >>> from nirs4all.pipeline.bundle import BundleGenerator
    >>>
    >>> generator = BundleGenerator(workspace_path)
    >>> bundle_path = generator.export(best_prediction, "exports/model.n4a")
    >>> print(f"Bundle saved to: {bundle_path}")
"""

import base64
import hashlib
import json
import logging
import zipfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from nirs4all.pipeline.resolver import PredictionResolver, ResolvedPrediction
from nirs4all.pipeline.trace import ExecutionTrace


logger = logging.getLogger(__name__)


# Bundle format version for compatibility checking
BUNDLE_FORMAT_VERSION = "1.0"


class BundleFormat(str, Enum):
    """Supported bundle export formats.

    Attributes:
        N4A: Full ZIP bundle with all artifacts and metadata
        N4A_PY: Portable Python script with embedded artifacts
    """
    N4A = "n4a"
    N4A_PY = "n4a.py"

    def __str__(self) -> str:
        return self.value


class BundleGenerator:
    """Generate standalone prediction bundles from trained pipelines.

    This class exports trained pipelines to bundle formats that can be
    used for deployment, sharing, or archival without requiring the
    original workspace or full nirs4all installation.

    Attributes:
        workspace_path: Path to the workspace root
        resolver: PredictionResolver for resolving prediction sources
        verbose: Verbosity level for logging

    Example:
        >>> generator = BundleGenerator(workspace_path)
        >>> generator.export(best_prediction, "model.n4a")
        >>>
        >>> # Export to portable script
        >>> generator.export(best_prediction, "model.n4a.py", format="n4a.py")
    """

    def __init__(
        self,
        workspace_path: Union[str, Path],
        verbose: int = 0
    ):
        """Initialize bundle generator.

        Args:
            workspace_path: Path to workspace root
            verbose: Verbosity level (0=quiet, 1=info, 2=debug)
        """
        self.workspace_path = Path(workspace_path)
        self.resolver = PredictionResolver(workspace_path)
        self.verbose = verbose

    def export(
        self,
        source: Union[Dict[str, Any], str, Path],
        output_path: Union[str, Path],
        format: Union[str, BundleFormat] = BundleFormat.N4A,
        include_metadata: bool = True,
        compress: bool = True
    ) -> Path:
        """Export a prediction source to a bundle.

        Args:
            source: Prediction source (prediction dict, folder path, etc.)
            output_path: Path for the output bundle
            format: Bundle format ('n4a' or 'n4a.py')
            include_metadata: Whether to include full metadata in bundle
            compress: Whether to compress artifacts (for .n4a format)

        Returns:
            Path to the created bundle

        Raises:
            ValueError: If format is not supported
            FileNotFoundError: If source cannot be resolved
        """
        # Normalize format
        if isinstance(format, str):
            format_lower = format.lower()
            if format_lower == "n4a" or format_lower == ".n4a":
                format = BundleFormat.N4A
            elif format_lower in ("n4a.py", ".n4a.py"):
                format = BundleFormat.N4A_PY
            else:
                raise ValueError(f"Unsupported bundle format: {format}")

        # Resolve the prediction source
        resolved = self.resolver.resolve(source, verbose=self.verbose)

        if self.verbose > 0:
            logger.info(f"Exporting bundle from {resolved.source_type} source")

        # Dispatch to appropriate exporter
        if format == BundleFormat.N4A:
            return self._export_n4a(resolved, output_path, include_metadata, compress)
        elif format == BundleFormat.N4A_PY:
            return self._export_n4a_py(resolved, output_path, include_metadata)
        else:
            raise ValueError(f"Unsupported bundle format: {format}")

    def _export_n4a(
        self,
        resolved: ResolvedPrediction,
        output_path: Union[str, Path],
        include_metadata: bool,
        compress: bool
    ) -> Path:
        """Export to .n4a ZIP bundle format.

        Bundle structure:
            model_bundle.n4a (ZIP archive)
            ├── manifest.json           # Bundle metadata
            ├── pipeline.json           # Minimal pipeline config
            ├── trace.json              # Execution trace
            ├── artifacts/
            │   ├── step_0_MinMaxScaler.joblib
            │   └── step_4_fold0_PLSRegression.joblib
            └── fold_weights.json       # For CV ensemble

        Args:
            resolved: Resolved prediction source
            output_path: Output path for bundle
            include_metadata: Whether to include full metadata
            compress: Whether to compress (ZIP_DEFLATED vs ZIP_STORED)

        Returns:
            Path to created bundle
        """
        output_path = Path(output_path)

        # Ensure .n4a extension
        if not output_path.suffix.lower().endswith('.n4a'):
            output_path = output_path.with_suffix('.n4a')

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED

        with zipfile.ZipFile(output_path, 'w', compression=compression) as zf:
            # 1. Write manifest.json
            manifest = self._create_bundle_manifest(resolved, include_metadata)
            zf.writestr('manifest.json', json.dumps(manifest, indent=2))

            # 2. Write pipeline.json
            pipeline_config = self._extract_pipeline_config(resolved)
            zf.writestr('pipeline.json', json.dumps(pipeline_config, indent=2))

            # 3. Write trace.json (if available)
            if resolved.trace:
                trace_dict = resolved.trace.to_dict()
                zf.writestr('trace.json', json.dumps(trace_dict, indent=2))

            # 4. Write fold_weights.json (if applicable)
            if resolved.fold_weights:
                # Convert int keys to strings for JSON
                weights_json = {str(k): v for k, v in resolved.fold_weights.items()}
                zf.writestr('fold_weights.json', json.dumps(weights_json, indent=2))

            # 5. Write artifacts
            artifacts_written = self._write_artifacts_to_zip(zf, resolved)

            if self.verbose > 0:
                logger.info(f"Bundle created: {output_path}")
                logger.info(f"  Artifacts: {artifacts_written}")
                logger.info(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")

        return output_path

    def _export_n4a_py(
        self,
        resolved: ResolvedPrediction,
        output_path: Union[str, Path],
        include_metadata: bool
    ) -> Path:
        """Export to .n4a.py portable Python script format.

        Creates a standalone Python script with embedded artifacts that can
        run predictions without nirs4all installed.

        Args:
            resolved: Resolved prediction source
            output_path: Output path for script
            include_metadata: Whether to include metadata comments

        Returns:
            Path to created script
        """
        output_path = Path(output_path)

        # Ensure .n4a.py extension
        if not str(output_path).endswith('.n4a.py'):
            if output_path.suffix == '.py':
                output_path = output_path.with_suffix('.n4a.py')
            else:
                output_path = Path(str(output_path) + '.n4a.py')

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate script content
        script_content = self._generate_portable_script(resolved, include_metadata)

        # Write script
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        # Make executable on Unix
        try:
            import os
            os.chmod(output_path, 0o755)
        except (OSError, AttributeError):
            pass

        if self.verbose > 0:
            logger.info(f"Portable script created: {output_path}")
            logger.info(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")

        return output_path

    def _create_bundle_manifest(
        self,
        resolved: ResolvedPrediction,
        include_metadata: bool
    ) -> Dict[str, Any]:
        """Create bundle manifest with metadata.

        Args:
            resolved: Resolved prediction source
            include_metadata: Whether to include full metadata

        Returns:
            Manifest dictionary
        """
        import nirs4all

        manifest = {
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "nirs4all_version": getattr(nirs4all, '__version__', 'unknown'),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_uid": resolved.pipeline_uid,
            "source_type": str(resolved.source_type),
            "model_step_index": resolved.model_step_index,
            "fold_strategy": str(resolved.fold_strategy),
            "preprocessing_chain": resolved.get_preprocessing_chain(),
        }

        if include_metadata and resolved.manifest:
            manifest["original_manifest"] = {
                "dataset": resolved.manifest.get("dataset", ""),
                "name": resolved.manifest.get("name", ""),
                "created_at": resolved.manifest.get("created_at", ""),
            }

        if resolved.trace:
            manifest["trace_id"] = resolved.trace.trace_id

            # Extract metadata partitioner routing info from trace
            routing_info = self._extract_partitioner_routing_info(resolved.trace)
            if routing_info:
                manifest["partitioner_routing"] = routing_info

        return manifest

    def _extract_partitioner_routing_info(
        self,
        trace: ExecutionTrace
    ) -> Optional[Dict[str, Any]]:
        """Extract metadata partitioner routing information from trace.

        This information is needed for prediction mode to route samples
        to the correct branch based on metadata values.

        Args:
            trace: Execution trace

        Returns:
            Routing info dict or None if no partitioner was used
        """
        routing_info = {}

        for step in trace.steps:
            if step.operator_class == "MetadataPartitionerController":
                step_routing = {
                    "step_index": step.step_index,
                    "branch_count": 0,
                    "partitions": [],
                }

                # Extract partition info from step metadata
                if step.metadata:
                    step_routing["column"] = step.metadata.get("column")
                    step_routing["partitions"] = step.metadata.get("partitions", [])
                    step_routing["branch_count"] = step.metadata.get("branch_count", 0)
                    step_routing["group_values"] = step.metadata.get("group_values")
                    step_routing["min_samples"] = step.metadata.get("min_samples", 1)

                    # Store partition-to-branch mapping
                    partitioner_config = step.metadata.get("metadata_partitioner_config", {})
                    if partitioner_config:
                        step_routing.update(partitioner_config)

                routing_info[str(step.step_index)] = step_routing

        return routing_info if routing_info else None

    def _extract_pipeline_config(
        self,
        resolved: ResolvedPrediction
    ) -> Dict[str, Any]:
        """Extract pipeline configuration from resolved prediction.

        Args:
            resolved: Resolved prediction source

        Returns:
            Pipeline configuration dictionary
        """
        # Start with minimal pipeline steps
        steps = []
        for step in resolved.minimal_pipeline:
            if isinstance(step, dict):
                steps.append(step)
            else:
                # Convert to dict if needed
                steps.append({"step": step})

        config = {
            "steps": steps,
            "model_step_index": resolved.model_step_index,
        }

        # Add trace step info if available
        if resolved.trace:
            config["trace_steps"] = [
                {
                    "step_index": s.step_index,
                    "operator_type": s.operator_type,
                    "operator_class": s.operator_class,
                }
                for s in resolved.trace.steps
            ]

        return config

    def _write_artifacts_to_zip(
        self,
        zf: zipfile.ZipFile,
        resolved: ResolvedPrediction
    ) -> int:
        """Write artifacts to ZIP file.

        Args:
            zf: ZipFile object to write to
            resolved: Resolved prediction source

        Returns:
            Number of artifacts written
        """
        count = 0

        if not resolved.artifact_provider:
            return count

        # Get all artifact IDs from trace or minimal pipeline
        artifact_ids = []
        if resolved.trace:
            artifact_ids = resolved.trace.get_artifact_ids()

        # For each step, write artifacts
        step_indices = set()
        if resolved.trace:
            for step in resolved.trace.steps:
                step_indices.add(step.step_index)

        for step_index in sorted(step_indices):
            artifacts = resolved.artifact_provider.get_artifacts_for_step(step_index)

            for artifact_id, artifact_obj in artifacts:
                try:
                    # Serialize artifact
                    artifact_bytes = self._serialize_artifact(artifact_obj)

                    # Get artifact record for accurate step/fold info
                    record = None
                    if hasattr(resolved.artifact_provider, 'artifact_loader'):
                        record = resolved.artifact_provider.artifact_loader.get_record(artifact_id)

                    # Create filename from artifact info (pass step_index from loop)
                    artifact_name = self._artifact_filename(
                        artifact_id, artifact_obj, record, step_index=step_index
                    )
                    archive_path = f"artifacts/{artifact_name}"

                    # Write to ZIP
                    zf.writestr(archive_path, artifact_bytes)
                    count += 1

                    if self.verbose > 1:
                        logger.debug(f"  Added artifact: {archive_path}")

                except Exception as e:
                    logger.warning(f"Failed to serialize artifact {artifact_id}: {e}")

        return count

    def _serialize_artifact(self, artifact: Any) -> bytes:
        """Serialize an artifact to bytes.

        Args:
            artifact: Artifact object to serialize

        Returns:
            Serialized bytes
        """
        import io
        import joblib

        buffer = io.BytesIO()
        joblib.dump(artifact, buffer)
        return buffer.getvalue()

    def _artifact_filename(
        self,
        artifact_id: str,
        artifact_obj: Any,
        record: Optional[Any] = None,
        step_index: Optional[int] = None
    ) -> str:
        """Generate filename for artifact.

        Args:
            artifact_id: Artifact ID (V2 or V3 format)
            artifact_obj: Artifact object
            record: Optional ArtifactRecord for step/fold info
            step_index: Optional step index (from loop context)

        Returns:
            Filename for artifact
        """
        # Get step_idx and fold_part from artifact record if available (V3)
        step_idx = step_index if step_index is not None else 0
        fold_part = "all"

        if record is not None:
            step_idx = getattr(record, 'step_index', step_idx) or step_idx
            fold_id = getattr(record, 'fold_id', None)
            fold_part = str(fold_id) if fold_id is not None else "all"
        else:
            # Fallback: parse from artifact_id
            # Check V3 format (contains $)
            if "$" in artifact_id:
                # V3: pipeline$hash:fold
                fold_part = artifact_id.split(":")[-1] if ":" in artifact_id else "all"
                # step_idx already set from parameter
            else:
                # V2: pipeline:branch:step:fold or pipeline:step:fold
                parts = artifact_id.split(":")
                if len(parts) >= 2:
                    try:
                        step_idx = int(parts[-2]) if step_index is None else step_idx
                        fold_part = parts[-1]
                    except ValueError:
                        pass

        # Get class name
        class_name = artifact_obj.__class__.__name__

        # Generate filename
        if fold_part != "all":
            return f"step_{step_idx}_fold{fold_part}_{class_name}.joblib"
        else:
            return f"step_{step_idx}_{class_name}.joblib"

    def _generate_portable_script(
        self,
        resolved: ResolvedPrediction,
        include_metadata: bool
    ) -> str:
        """Generate portable Python prediction script.

        Args:
            resolved: Resolved prediction source
            include_metadata: Whether to include metadata comments

        Returns:
            Python script content
        """
        import nirs4all

        # Collect artifacts
        artifacts_data = {}
        step_info = {}
        # Track substep counters for multiple artifacts per step
        step_substep_counter = {}

        if resolved.trace and resolved.artifact_provider:
            for step in resolved.trace.steps:
                step_index = step.step_index
                artifacts = resolved.artifact_provider.get_artifacts_for_step(step_index)

                for artifact_id, artifact_obj in artifacts:
                    artifact_bytes = self._serialize_artifact(artifact_obj)
                    encoded = base64.b64encode(artifact_bytes).decode('ascii')

                    # Parse fold info from artifact_id
                    parts = artifact_id.split(":")
                    fold_part = parts[-1] if len(parts) >= 3 else "all"

                    if fold_part != "all":
                        key = f"step_{step_index}_fold{fold_part}"
                    else:
                        # Handle multiple artifacts per step (e.g., feature_augmentation)
                        base_key = f"step_{step_index}"
                        if base_key in artifacts_data:
                            # Multiple artifacts for same step - use substep counter
                            if step_index not in step_substep_counter:
                                step_substep_counter[step_index] = 1
                            else:
                                step_substep_counter[step_index] += 1
                            key = f"step_{step_index}_sub{step_substep_counter[step_index]}"
                        else:
                            key = base_key

                    artifacts_data[key] = encoded

                    if step_index not in step_info:
                        step_info[step_index] = {
                            "operator_type": step.operator_type,
                            "operator_class": step.operator_class,
                        }

        # Generate fold weights
        fold_weights_code = ""
        if resolved.fold_weights:
            fold_weights_code = repr(resolved.fold_weights)
        else:
            fold_weights_code = "{}"

        # Build script
        script = self._build_portable_script_template(
            artifacts_data=artifacts_data,
            step_info=step_info,
            fold_weights=fold_weights_code,
            model_step_index=resolved.model_step_index,
            preprocessing_chain=resolved.get_preprocessing_chain(),
            pipeline_uid=resolved.pipeline_uid,
            nirs4all_version=getattr(nirs4all, '__version__', 'unknown'),
            created_at=datetime.now(timezone.utc).isoformat(),
            include_metadata=include_metadata
        )

        return script

    def _build_portable_script_template(
        self,
        artifacts_data: Dict[str, str],
        step_info: Dict[int, Dict[str, str]],
        fold_weights: str,
        model_step_index: Optional[int],
        preprocessing_chain: str,
        pipeline_uid: str,
        nirs4all_version: str,
        created_at: str,
        include_metadata: bool
    ) -> str:
        """Build the portable script from template.

        Args:
            artifacts_data: Mapping of artifact keys to base64 data
            step_info: Mapping of step index to operator info
            fold_weights: Fold weights as Python code
            model_step_index: Index of model step
            preprocessing_chain: Summary of preprocessing
            pipeline_uid: Pipeline UID
            nirs4all_version: nirs4all version used
            created_at: Creation timestamp
            include_metadata: Whether to include metadata

        Returns:
            Complete script content
        """
        # Format artifacts as Python dict literal
        artifacts_lines = []
        for key, data in artifacts_data.items():
            # Split long base64 strings for readability
            if len(data) > 100:
                artifacts_lines.append(f'    "{key}": (')
                # Split into lines of ~80 chars
                for i in range(0, len(data), 80):
                    chunk = data[i:i + 80]
                    artifacts_lines.append(f'        "{chunk}"')
                artifacts_lines.append('    ),')
            else:
                artifacts_lines.append(f'    "{key}": "{data}",')

        artifacts_dict_str = "{\n" + "\n".join(artifacts_lines) + "\n}"

        # Format step info
        step_info_lines = []
        for idx, info in sorted(step_info.items()):
            step_info_lines.append(
                f'    {idx}: {{"operator_type": "{info["operator_type"]}", '
                f'"operator_class": "{info["operator_class"]}"}}'
            )
        step_info_str = "{\n" + ",\n".join(step_info_lines) + "\n}"

        script = f'''#!/usr/bin/env python3
"""
Standalone prediction script generated by nirs4all.
Requires only numpy and joblib for execution.

Original pipeline UID: {pipeline_uid}
Preprocessing chain: {preprocessing_chain}
Generated from nirs4all version: {nirs4all_version}
Created: {created_at}
"""

import base64
import io
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import joblib
except ImportError:
    raise ImportError(
        "joblib is required for loading artifacts. "
        "Install with: pip install joblib"
    )


# =============================================================================
# Embedded Artifacts (base64 encoded)
# =============================================================================

ARTIFACTS = {artifacts_dict_str}

STEP_INFO = {step_info_str}

FOLD_WEIGHTS: Dict[int, float] = {fold_weights}

MODEL_STEP_INDEX: Optional[int] = {model_step_index}


# =============================================================================
# Artifact Loading
# =============================================================================

def load_artifact(key: str) -> Any:
    """Load an artifact from embedded base64 data.

    Args:
        key: Artifact key (e.g., "step_1" or "step_4_fold0")

    Returns:
        Deserialized artifact object
    """
    if key not in ARTIFACTS:
        raise KeyError(f"Artifact not found: {{key}}")

    data = ARTIFACTS[key]

    # Handle multi-line strings
    if isinstance(data, tuple):
        data = "".join(data)

    decoded = base64.b64decode(data)
    buffer = io.BytesIO(decoded)
    return joblib.load(buffer)


def get_step_artifacts(step_index: int) -> List[Tuple[str, Any]]:
    """Get all artifacts for a step.

    Args:
        step_index: Step index to get artifacts for

    Returns:
        List of (key, artifact) tuples
    """
    prefix = f"step_{{step_index}}"
    results = []

    for key in ARTIFACTS:
        # Match step_N, step_N_foldM, or step_N_subM
        if key == prefix or key.startswith(f"{{prefix}}_fold") or key.startswith(f"{{prefix}}_sub"):
            results.append((key, load_artifact(key)))

    return results


def get_fold_artifacts(step_index: int) -> List[Tuple[int, Any]]:
    """Get fold-specific artifacts for a step.

    Args:
        step_index: Step index

    Returns:
        List of (fold_id, artifact) tuples sorted by fold_id
    """
    results = []

    for key in ARTIFACTS:
        if key.startswith(f"step_{{step_index}}_fold"):
            fold_id = int(key.split("_fold")[1])
            results.append((fold_id, load_artifact(key)))

    return sorted(results, key=lambda x: x[0])


# =============================================================================
# Prediction Logic
# =============================================================================

def _apply_y_inverse_transform(y_pred: np.ndarray, y_processing_step_idx: Optional[int]) -> np.ndarray:
    """Apply inverse transform to predictions if y_processing was used.

    Args:
        y_pred: Model predictions (possibly in scaled space)
        y_processing_step_idx: Step index of y_processing transformer, or None

    Returns:
        Predictions in original scale (inverse transformed if applicable)
    """
    if y_processing_step_idx is None:
        return y_pred

    # Get the y_processing transformer
    step_artifacts = get_step_artifacts(y_processing_step_idx)

    if not step_artifacts:
        return y_pred

    # Apply inverse_transform from each y_processing artifact
    y_current = y_pred
    for _, transformer in step_artifacts:
        if hasattr(transformer, 'inverse_transform'):
            # Ensure proper shape for inverse_transform
            if y_current.ndim == 1:
                y_current = y_current.reshape(-1, 1)
                y_current = transformer.inverse_transform(y_current)
                y_current = y_current.ravel()
            else:
                y_current = transformer.inverse_transform(y_current)

    return y_current


def predict(X: np.ndarray) -> np.ndarray:
    """
    Run prediction on input data.

    This function applies the preprocessing transformers and model(s)
    embedded in this bundle to produce predictions.

    Args:
        X: Input features as numpy array, shape (n_samples, n_features)

    Returns:
        Predictions as numpy array, shape (n_samples,) or (n_samples, n_outputs)
    """
    # Process through each step
    X_current = X.copy()
    y_processing_step_idx = None  # Track y_processing step for inverse_transform

    # Get sorted step indices
    step_indices = sorted(set(
        int(k.split("_")[1].replace("fold", "").split("_")[0])
        for k in ARTIFACTS.keys()
    ))

    for step_idx in step_indices:
        info = STEP_INFO.get(step_idx, {{}})
        op_type = info.get("operator_type", "")
        op_class = info.get("operator_class", "")

        # Handle y_processing - skip but track for inverse_transform
        if op_type == "y_processing":
            y_processing_step_idx = step_idx
            continue

        # Check if this is the model step
        is_model_step = (
            step_idx == MODEL_STEP_INDEX or
            op_type in ("model", "meta_model") or
            any(
                k.startswith(f"step_{{step_idx}}_fold") or k == f"step_{{step_idx}}"
                for k in ARTIFACTS
                if "model" in ARTIFACTS.get(k, "").lower() or op_type == "model"
            )
        )

        # Get artifacts for this step
        step_artifacts = get_step_artifacts(step_idx)

        if is_model_step:
            # Model step - may have multiple folds
            fold_artifacts = get_fold_artifacts(step_idx)

            if fold_artifacts:
                # CV ensemble - average predictions across folds
                fold_preds = []

                for fold_id, model in fold_artifacts:
                    weight = FOLD_WEIGHTS.get(fold_id, 1.0)
                    y_fold = model.predict(X_current)
                    fold_preds.append((weight, y_fold))

                if FOLD_WEIGHTS:
                    # Weighted average
                    total_weight = sum(w for w, _ in fold_preds)
                    y_pred = sum(w * y for w, y in fold_preds) / total_weight
                else:
                    # Simple average
                    y_pred = np.mean([y for _, y in fold_preds], axis=0)

                return _apply_y_inverse_transform(y_pred, y_processing_step_idx)
            else:
                # Single model
                _, model = step_artifacts[0]
                y_pred = model.predict(X_current)
                return _apply_y_inverse_transform(y_pred, y_processing_step_idx)

        elif op_type == "feature_augmentation":
            # Feature augmentation: apply each transformer and concatenate with original
            feature_channels = [X_current]
            for key, transformer in step_artifacts:
                if hasattr(transformer, 'transform'):
                    X_transformed = transformer.transform(X_current)
                    feature_channels.append(X_transformed)
            X_current = np.hstack(feature_channels)

        else:
            # Preprocessing step - transform X
            for key, transformer in step_artifacts:
                if hasattr(transformer, 'transform'):
                    X_current = transformer.transform(X_current)
                elif hasattr(transformer, 'fit_transform'):
                    # Some transformers only have fit_transform
                    X_current = transformer.fit_transform(X_current)

    raise RuntimeError("No model step found in pipeline")


def predict_from_file(
    input_path: str,
    output_path: Optional[str] = None,
    delimiter: str = ",",
    skip_header: bool = True
) -> np.ndarray:
    """
    Load data from CSV file, run prediction, and optionally save results.

    Args:
        input_path: Path to input CSV file
        output_path: Optional path to save predictions
        delimiter: CSV delimiter
        skip_header: Whether to skip header row

    Returns:
        Predictions array
    """
    # Load input data
    skip = 1 if skip_header else 0
    X = np.loadtxt(input_path, delimiter=delimiter, skiprows=skip)

    # Run prediction
    y_pred = predict(X)

    # Save if output path provided
    if output_path:
        np.savetxt(output_path, y_pred, delimiter=delimiter)
        print(f"Predictions saved to: {{output_path}}")

    return y_pred


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print()
        print("Usage: python {{}} input.csv [output.csv]".format(sys.argv[0]))
        print()
        print("Arguments:")
        print("  input.csv   Input data file (CSV format)")
        print("  output.csv  Optional output file for predictions")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        y_pred = predict_from_file(input_file, output_file)
        if output_file is None:
            print("Predictions:")
            print(y_pred)
    except Exception as e:
        print(f"Error: {{e}}", file=sys.stderr)
        sys.exit(1)
'''

        return script
