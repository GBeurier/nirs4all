"""
Central Artifact Serializer - Framework-aware object persistence

Provides content-addressed storage for ML artifacts with automatic framework detection.
Supports sklearn, TensorFlow, PyTorch, XGBoost, CatBoost, and generic objects.

Architecture:
    - Content-addressed storage using SHA256 hashing
    - Framework-specific serialization formats
    - Deduplication via hash-based storage
    - Git-style sharded directories (hash[:2]/hash.ext)
"""

import hashlib
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypedDict, Union
import importlib.util


class ArtifactMeta(TypedDict):
    """Metadata for a persisted artifact."""
    hash: str           # SHA256 hash with "sha256:" prefix
    name: str           # Original name for reference
    path: str           # Relative path from results root
    format: str         # Serialization format used
    format_version: str  # Version of the library used (e.g., 'sklearn==1.3.0')
    nirs4all_version: str  # Version of nirs4all (e.g., '0.4.1')
    size: int           # Size in bytes
    saved_at: str       # ISO timestamp
    step: int           # Pipeline step number (set by caller)
    branch_id: Optional[int]      # Branch ID for pipeline branching (None if not in branch)
    branch_name: Optional[str]    # Human-readable branch name


# Framework detection cache
_FRAMEWORK_CACHE = {
    'sklearn': None,
    'tensorflow': None,
    'keras': None,
    'torch': None,
    'xgboost': None,
    'catboost': None,
    'lightgbm': None,
    'cloudpickle': None,
    'joblib': None
}


def _check_framework(name: str) -> bool:
    """Check if a framework is available (cached)."""
    if _FRAMEWORK_CACHE[name] is None:
        _FRAMEWORK_CACHE[name] = importlib.util.find_spec(name) is not None
    return _FRAMEWORK_CACHE[name]


def _detect_framework(obj: Any) -> str:
    """
    Detect the framework/type of an object for optimal serialization.

    Returns:
        Format string: 'sklearn_pickle', 'tensorflow_keras', 'pytorch_state',
                       'xgboost_json', 'catboost_cbm', 'lightgbm_txt', 'pickle'
    """
    obj_type = type(obj).__name__
    obj_module = type(obj).__module__

    # Sklearn objects
    if 'sklearn' in obj_module:
        return 'sklearn_pickle'

    # TensorFlow/Keras models
    if 'tensorflow' in obj_module or 'keras' in obj_module:
        # Check if it's a Keras model
        if _check_framework('tensorflow'):
            import tensorflow as tf
            if isinstance(obj, tf.keras.Model):
                return 'tensorflow_keras'
        return 'tensorflow_saved_model'

    # PyTorch models
    if 'torch' in obj_module:
        if _check_framework('torch'):
            import torch
            if isinstance(obj, torch.nn.Module):
                return 'pytorch_state_dict'
        return 'pytorch_pickle'

    # XGBoost models
    if 'xgboost' in obj_module:
        return 'xgboost_json'

    # CatBoost models
    if 'catboost' in obj_module:
        return 'catboost_cbm'

    # LightGBM models
    if 'lightgbm' in obj_module:
        return 'lightgbm_txt'

    # Numpy arrays
    if obj_module == 'numpy' or obj_type == 'ndarray':
        return 'numpy_npy'

    # Generic fallback
    return 'pickle'


def _format_to_extension(format: str) -> str:
    """Map format string to file extension."""
    ext_map = {
        'sklearn_pickle': 'pkl',
        'tensorflow_keras': 'keras',
        'tensorflow_saved_model': 'pb',  # Will be a directory
        'pytorch_state_dict': 'pt',
        'pytorch_pickle': 'pkl',
        'xgboost_json': 'json',
        'xgboost_ubj': 'ubj',
        'catboost_cbm': 'cbm',
        'lightgbm_txt': 'txt',
        'numpy_npy': 'npy',
        'pickle': 'pkl',
        'cloudpickle': 'pkl',
        'joblib': 'joblib'
    }
    return ext_map.get(format, 'pkl')


def compute_hash(data: bytes) -> str:
    """Compute SHA256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def _get_library_version(obj: Any) -> str:
    """Get version of the library that created the object.

    Args:
        obj: Object to inspect

    Returns:
        Version string like 'sklearn==1.3.0' or empty string if unknown
    """
    obj_module = type(obj).__module__

    try:
        if 'sklearn' in obj_module:
            import sklearn
            return f"sklearn=={sklearn.__version__}"
        elif 'tensorflow' in obj_module or 'keras' in obj_module:
            import tensorflow as tf
            return f"tensorflow=={tf.__version__}"
        elif 'torch' in obj_module:
            import torch
            return f"torch=={torch.__version__}"
        elif 'xgboost' in obj_module:
            import xgboost
            return f"xgboost=={xgboost.__version__}"
        elif 'catboost' in obj_module:
            import catboost
            return f"catboost=={catboost.__version__}"
        elif 'lightgbm' in obj_module:
            import lightgbm
            return f"lightgbm=={lightgbm.__version__}"
        elif 'numpy' in obj_module:
            import numpy
            return f"numpy=={numpy.__version__}"
    except (ImportError, AttributeError):
        pass

    return ""


def _get_nirs4all_version() -> str:
    """Get current nirs4all version.

    Returns:
        Version string like '0.4.1' or empty string if not available
    """
    try:
        from nirs4all import __version__
        return __version__
    except (ImportError, AttributeError):
        return ""


def to_bytes(obj: Any, format_hint: Optional[str] = None) -> Tuple[bytes, str]:
    """
    Serialize object to bytes using appropriate format.

    Args:
        obj: Object to serialize
        format_hint: Optional format override ('sklearn', 'tensorflow', etc.)

    Returns:
        (bytes, format_string) tuple
    """
    # Determine format
    if format_hint:
        format = f"{format_hint}_pickle"  # Simple mapping for hints
    else:
        format = _detect_framework(obj)

    try:
        # Sklearn objects - use joblib if available, else pickle
        if format == 'sklearn_pickle':
            if _check_framework('joblib'):
                import joblib
                import io
                buffer = io.BytesIO()
                joblib.dump(obj, buffer, compress=3)
                return buffer.getvalue(), 'joblib'
            else:
                return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), 'sklearn_pickle'

        # TensorFlow Keras models - save to .keras format
        elif format == 'tensorflow_keras':
            import tensorflow as tf
            import io
            import tempfile
            import zipfile

            # Save to temporary file then read as bytes
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                obj.save(tmp_path)
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                return data, 'tensorflow_keras'
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # PyTorch state dict
        elif format == 'pytorch_state_dict':
            import torch
            import io

            buffer = io.BytesIO()
            torch.save(obj.state_dict(), buffer)
            return buffer.getvalue(), 'pytorch_state_dict'

        # XGBoost - use JSON format for cross-platform compatibility
        elif format == 'xgboost_json':
            import json
            import tempfile

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                obj.save_model(tmp_path)
                with open(tmp_path, 'r') as f:
                    json_str = f.read()
                return json_str.encode('utf-8'), 'xgboost_json'
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # CatBoost - use native format
        elif format == 'catboost_cbm':
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.cbm', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                obj.save_model(tmp_path, format='cbm')
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                return data, 'catboost_cbm'
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # LightGBM - use text format
        elif format == 'lightgbm_txt':
            import tempfile

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                obj.save_model(tmp_path)
                with open(tmp_path, 'r') as f:
                    text = f.read()
                return text.encode('utf-8'), 'lightgbm_txt'
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # Numpy arrays
        elif format == 'numpy_npy':
            import numpy as np
            import io

            buffer = io.BytesIO()
            np.save(buffer, obj)
            return buffer.getvalue(), 'numpy_npy'

        # Generic pickle fallback - try cloudpickle first
        else:
            if _check_framework('cloudpickle'):
                import cloudpickle
                return cloudpickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), 'cloudpickle'
            else:
                return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), 'pickle'

    except Exception as e:
        # Fallback to pickle if specialized serialization fails
        warnings.warn(f"Failed to serialize with format {format}, falling back to pickle: {e}")
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), 'pickle'


def from_bytes(data: bytes, format: str) -> Any:
    """
    Deserialize object from bytes based on format.

    Args:
        data: Serialized bytes
        format: Format string from artifact metadata

    Returns:
        Deserialized object
    """
    try:
        # Joblib format
        if format == 'joblib':
            import joblib
            import io
            buffer = io.BytesIO(data)
            return joblib.load(buffer)

        # Sklearn pickle
        elif format == 'sklearn_pickle':
            return pickle.loads(data)

        # TensorFlow Keras
        elif format == 'tensorflow_keras':
            import tensorflow as tf
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            try:
                model = tf.keras.models.load_model(tmp_path)
                return model
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # PyTorch state dict
        elif format == 'pytorch_state_dict':
            import torch
            import io

            buffer = io.BytesIO(data)
            state_dict = torch.load(buffer)
            # Note: Caller needs to know model architecture to load state_dict
            # This is a limitation - we return the state_dict and caller must handle it
            return state_dict

        # XGBoost JSON
        elif format == 'xgboost_json':
            import xgboost as xgb
            import tempfile

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                tmp.write(data.decode('utf-8'))
                tmp_path = tmp.name

            try:
                model = xgb.Booster()
                model.load_model(tmp_path)
                return model
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # CatBoost
        elif format == 'catboost_cbm':
            from catboost import CatBoost
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.cbm', delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            try:
                model = CatBoost()
                model.load_model(tmp_path, format='cbm')
                return model
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # LightGBM
        elif format == 'lightgbm_txt':
            import lightgbm as lgb
            import tempfile

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
                tmp.write(data.decode('utf-8'))
                tmp_path = tmp.name

            try:
                model = lgb.Booster(model_file=tmp_path)
                return model
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # Numpy arrays
        elif format == 'numpy_npy':
            import numpy as np
            import io

            buffer = io.BytesIO(data)
            return np.load(buffer, allow_pickle=False)

        # Cloudpickle
        elif format == 'cloudpickle':
            import cloudpickle
            return cloudpickle.loads(data)

        # Generic pickle
        else:
            return pickle.loads(data)

    except Exception as e:
        # Try generic pickle as last resort
        warnings.warn(f"Failed to deserialize with format {format}, trying pickle: {e}")
        return pickle.loads(data)


def is_serializable(obj: Any) -> bool:
    """
    Check if an object can be serialized.

    Args:
        obj: Object to check

    Returns:
        True if serializable, False otherwise
    """
    try:
        data, _ = to_bytes(obj)
        return len(data) > 0
    except Exception:
        return False


def persist(
    obj: Any,
    artifacts_dir: Union[str, Path],
    name: str,
    format_hint: Optional[str] = None,
    branch_id: Optional[int] = None,
    branch_name: Optional[str] = None
) -> ArtifactMeta:
    """
    Persist object to _binaries storage with meaningful names.

    Args:
        obj: Object to persist
        artifacts_dir: Path to run _binaries/ directory
        name: Artifact name (e.g., "scaler", "model")
        format_hint: Optional format hint ('sklearn', 'tensorflow', etc.)
        branch_id: Optional branch ID for pipeline branching
        branch_name: Optional human-readable branch name

    Returns:
        ArtifactMeta with hash, path, format, size, and branch info

    Raises:
        ValueError: If object cannot be serialized
    """
    artifacts_dir = Path(artifacts_dir)

    # 1. Serialize to bytes
    data, format = to_bytes(obj, format_hint)

    # 2. Compute SHA256 hash (short version for filename)
    hash_value = compute_hash(data)
    short_hash = hash_value[:6]  # Use first 6 chars for filename

    # 3. Determine extension
    ext = _format_to_extension(format)

    # 4. Get class name for meaningful filename
    class_name = obj.__class__.__name__

    # 5. Handle special cases for better naming
    if class_name == "bytes":
        # For raw bytes objects, use the custom name or fallback to "data"
        if name and name != "artifact":
            # Use custom name without extension (e.g., "folds_ShuffleSplit_seed42" -> "folds_ShuffleSplit_seed42")
            class_name = name.replace('.csv', '').replace('.', '_')
        else:
            class_name = "data"

    # 6. Create filename for deduplication: <ClassName>_<short_hash>.<ext>
    # Always use class name for deduplication (ignore custom names)
    dedup_filename = f"{class_name}_{short_hash}.{ext}"
    artifact_path = artifacts_dir / dedup_filename

    # 7. Write file (if not exists - deduplication works)
    if not artifact_path.exists():
        artifact_path.write_bytes(data)

    # 8. Return metadata (relative path for portability)
    relative_path = dedup_filename  # Just the filename, no subdirectories

    return {
        "hash": f"sha256:{hash_value}",
        "name": name,
        "path": relative_path,
        "format": format,
        "format_version": _get_library_version(obj),
        "nirs4all_version": _get_nirs4all_version(),
        "size": len(data),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "step": -1,  # Caller must set this
        "branch_id": branch_id,
        "branch_name": branch_name
    }


def load(
    artifact_meta: ArtifactMeta,
    results_dir: Union[str, Path],
    binaries_dir: Optional[Union[str, Path]] = None
) -> Any:
    """
    Load object from artifact metadata.

    Args:
        artifact_meta: Artifact metadata dictionary
        results_dir: Path to run directory
        binaries_dir: Optional path to centralized binaries directory

    Returns:
        Deserialized object

    Raises:
        FileNotFoundError: If artifact file doesn't exist
        ValueError: If artifact cannot be deserialized
    """
    results_dir = Path(results_dir)

    # Try centralized binaries first (v2 architecture)
    if binaries_dir is not None:
        binaries_path = Path(binaries_dir)
        artifact_path = binaries_path / artifact_meta["path"]
        if artifact_path.exists():
            data = artifact_path.read_bytes()
            return from_bytes(data, artifact_meta["format"])

    # Fall back to _binaries in results_dir
    artifact_path = results_dir / "_binaries" / artifact_meta["path"]
    if artifact_path.exists():
        data = artifact_path.read_bytes()
        return from_bytes(data, artifact_meta["format"])

    raise FileNotFoundError(f"Artifact not found: {artifact_path}")


def get_artifact_size(artifact_meta: ArtifactMeta, results_dir: Union[str, Path]) -> int:
    """Get the actual size of an artifact file on disk."""
    results_dir = Path(results_dir)
    artifact_path = results_dir / "artifacts" / artifact_meta["path"]

    if artifact_path.exists():
        return artifact_path.stat().st_size
    return 0
