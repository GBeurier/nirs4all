"""Verified captured REFIT models in the general library workspace.

Only the recorded full-training artifact is replayed. A CV score on the same
chain is not evidence that individual CV models were retained.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any


def load_general_workspace_chain(workspace_path: str | Path, chain_id: str) -> dict[str, Any] | None:
    """Inspect provenance, verify the exact artifact bytes, then load trusted Python."""
    import joblib

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    root = Path(workspace_path)
    if not (root / "store.sqlite").is_file():
        return None
    with WorkspaceStore(root) as store:
        chain = store.get_chain(chain_id)
        if chain is None:
            return None
        steps = chain.get("steps") or []
        marker = steps[0].get("dagml_host_replay") if len(steps) == 1 and isinstance(steps[0], dict) else None
        if not isinstance(marker, dict) or marker.get("schema") != "nirs4all.dagml-workspace-refit.v1":
            return None
        artifact_id = (chain.get("fold_artifacts") or {}).get("final")
        if not isinstance(artifact_id, str) or steps[0].get("artifact_id") != artifact_id:
            raise ValueError("DAG workspace chain does not identify one captured REFIT artifact")
        path = store.get_artifact_path(artifact_id)
        if not path.resolve().is_relative_to((root / "artifacts").resolve()):
            raise ValueError("DAG workspace artifact escapes its workspace")
        payload = path.read_bytes()
        fingerprint = "sha256:" + hashlib.sha256(payload).hexdigest()
        if fingerprint != marker.get("artifact_fingerprint"):
            raise ValueError("DAG workspace artifact fingerprint mismatch; refusing to deserialize")
        artifact = joblib.load(io.BytesIO(payload))
        pipeline_record = store.get_pipeline(chain["pipeline_id"])
    if not isinstance(artifact, dict) or not callable(getattr(artifact.get("estimator"), "predict", None)):
        raise ValueError("DAG workspace artifact is not a captured predictor")
    return {
        "artifact": artifact, "pipeline": [{"model": artifact["estimator"]}],
        "manifest": {key: marker[key] for key in ("relation_replay_manifest", "relation_materialization_manifest") if key in marker},
        "target_names": marker.get("target_names", ["y"]), "chain": chain,
        "training_pipeline": pipeline_record.get("expanded_config") if pipeline_record else None,
        "metadata": {
            "chain_id": chain_id, "workspace_path": str(root), "artifact_fingerprint": fingerprint,
            "artifact_integrity_verified": True, "artifact_scope": "full_training_refit",
            "cv_artifacts_available": False, "portable": False,
        },
    }


def predict_general_workspace_chain(loaded: dict[str, Any], data: Any) -> Any:
    """Execute a verified workspace predictor through native DAG PREDICT only."""
    from nirs4all.api.result import PredictResult

    from .dataset import _materialize_dataset
    from .general_replay import predict_captured_artifact

    values, evidence = predict_captured_artifact(
        loaded["artifact"], _materialize_dataset(data), pipeline=loaded["pipeline"], target_names=loaded["target_names"],
    )
    evidence.update(loaded["metadata"])
    return PredictResult(y_pred=values, metadata=evidence, model_name=loaded["chain"].get("model_name") or "")
