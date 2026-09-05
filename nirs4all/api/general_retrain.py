"""Resolve a captured winner into fresh estimators for a full DAG retrain.

Sources follow the existing trusted-host artifact integrity policy. Cloning
their constructor parameters must not retain estimator objects, rerun a
generator search, or make these Python artifacts portable Core archives.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from nirs4all.pipeline.dagml.rt import RtError


def captured_training_spec(source: Any) -> tuple[list[Any], dict[str, Any]] | None:
    """Select a recorded host winner before execution, without a retry fallback.

    Existing bundles with a training specification retain their published
    replay behavior. Prediction dictionaries resolve only recorded workspace
    chains; caller-provided model objects or parameters are not authoritative.
    """
    from nirs4all.pipeline.dagml.general_archive import general_archive_manifest, load_general_archive
    from nirs4all.pipeline.dagml.general_workspace import load_general_workspace_chain

    if isinstance(source, dict):
        workspace, chain_id = source.get("workspace_path"), source.get("chain_id")
        if not isinstance(workspace, (str, Path)) or not isinstance(chain_id, str) or not chain_id:
            return None
        loaded = load_general_workspace_chain(workspace, chain_id)
        if loaded is None:
            return None
        estimator = loaded["artifact"]["estimator"]
        target_transform = loaded["artifact"].get("y_transform")
        identity = {
            "source_kind": "workspace_chain", "source_chain_id": chain_id,
            "source_artifact_fingerprint": loaded["metadata"]["artifact_fingerprint"],
            "source_integrity_verified": loaded["metadata"]["artifact_integrity_verified"],
        }
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_symlink() or general_archive_manifest(path) is None:
            return None
        with zipfile.ZipFile(path) as archive:
            if "train_pipeline.json" in archive.namelist():
                return None
        loaded = load_general_archive(path)
        from .result import _DagmlExportedModel

        wrapper = loaded["artifact"]["estimator"]
        if not isinstance(wrapper, _DagmlExportedModel):
            raise _unsupported("full retrain requires a captured trainable estimator, not a prediction-only fusion wrapper")
        estimator, target_transform = wrapper.estimator, wrapper.y_transform
        identity = {
            "source_kind": "n4a_bundle", "source_bundle": path.name,
            "source_bundle_sha256": loaded["archive_fingerprint"].removeprefix("sha256:"),
            "source_artifact_fingerprint": loaded["artifact"]["content_fingerprint"],
            "source_integrity_verified": loaded["artifact_integrity_verified"],
        }
    else:
        return None

    from sklearn.base import clone

    if not callable(getattr(estimator, "fit", None)):
        raise _unsupported("the captured predictor has no training interface")
    try:
        fresh_model = clone(estimator)
        fresh_target = clone(target_transform) if target_transform is not None else None
    except (TypeError, RuntimeError) as error:
        raise _unsupported("the captured estimator parameters cannot be cloned for a fresh full retrain") from error
    original_objects = _estimator_ids(estimator) | _estimator_ids(target_transform)
    if original_objects & (_estimator_ids(fresh_model) | _estimator_ids(fresh_target)):
        raise _unsupported("cloning retained a captured estimator; full retrain requires fresh trainable objects")
    steps = []
    if fresh_target is not None:
        steps.append({"y_processing": fresh_target})
    steps.append({"model": fresh_model})
    return steps, {
        "schema_version": 1, "operation": "retrain", "mode": "full", "engine": "dag-ml",
        **identity, "training_contract": "captured_estimator_parameters.v1",
        "learned_state_reused": False, "parameter_search_repeated": False,
    }


def _estimator_ids(estimator: Any) -> set[int]:
    """Reject frozen/custom clones retaining the source or nested estimators."""
    if estimator is None:
        return set()
    parameters = estimator.get_params(deep=True)
    return {id(estimator), *(id(value) for value in parameters.values() if callable(getattr(value, "fit", None)))}


def _unsupported(message: str) -> RtError:
    return RtError.invalid_request(message, verb="run", unsupported_capability="dagml_full_retrain_captured_predictor")
