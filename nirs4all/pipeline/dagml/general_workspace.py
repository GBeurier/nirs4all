"""Verified captured REFIT models in the general library workspace.

Only the recorded full-training artifact is replayed. A CV score on the same
chain is not evidence that individual CV models were retained.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from pathlib import Path
from typing import Any


def load_general_workspace_chain(workspace_path: str | Path, chain_id: str) -> dict[str, Any] | None:
    """Inspect provenance, verify the exact artifact bytes, then load trusted Python."""
    import joblib

    from nirs4all.pipeline.storage.store_queries import GET_ARTIFACT, GET_CHAIN, GET_PIPELINE
    from nirs4all.pipeline.storage.store_schema import SCHEMA_VERSION

    root = Path(workspace_path)
    database = root / "store.sqlite"
    if not database.is_file():
        return None
    sidecars = [Path(f"{database}{suffix}") for suffix in ("-wal", "-shm", "-journal")]

    def signature() -> tuple[int, int, int, int]:
        if any(path.exists() for path in sidecars):
            raise RuntimeError("DAG workspace replay refuses an active SQLite journal")
        stat = database.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    before = signature()
    # Replay is a reader, not a store owner: never migrate, reconcile arrays,
    # enable WAL, or create files while inspecting a captured predictor.
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise RuntimeError(f"DAG workspace replay requires schema {SCHEMA_VERSION}, got {version}")
        row = connection.execute(GET_CHAIN, [chain_id]).fetchone()
        if row is None:
            return None
        chain = dict(row)
        for field in ("steps", "fold_artifacts", "shared_artifacts", "branch_path", "relation_replay_manifest"):
            value = chain.get(field)
            chain[field] = json.loads(value) if value is not None else None
        steps = chain.get("steps") or []
        marker = steps[0].get("dagml_host_replay") if len(steps) == 1 and isinstance(steps[0], dict) else None
        if not isinstance(marker, dict) or marker.get("schema") != "nirs4all.dagml-workspace-refit.v1":
            return None
        artifact_id = (chain.get("fold_artifacts") or {}).get("final")
        if not isinstance(artifact_id, str) or steps[0].get("artifact_id") != artifact_id:
            raise ValueError("DAG workspace chain does not identify one captured REFIT artifact")
        artifact_record = connection.execute(GET_ARTIFACT, [artifact_id]).fetchone()
        if artifact_record is None:
            raise KeyError(f"Unknown artifact: {artifact_id}")
        path = root / "artifacts" / str(artifact_record["artifact_path"])
        if not path.resolve().is_relative_to((root / "artifacts").resolve()):
            raise ValueError("DAG workspace artifact escapes its workspace")
        payload = path.read_bytes()
        fingerprint = "sha256:" + hashlib.sha256(payload).hexdigest()
        if fingerprint != marker.get("artifact_fingerprint"):
            raise ValueError("DAG workspace artifact fingerprint mismatch; refusing to deserialize")
        pipeline_row = connection.execute(GET_PIPELINE, [chain["pipeline_id"]]).fetchone()
        pipeline_record = dict(pipeline_row) if pipeline_row is not None else None
        if pipeline_record is not None:
            for field in ("expanded_config", "original_template", "generator_choices"):
                value = pipeline_record.get(field)
                pipeline_record[field] = json.loads(value) if value is not None else None
    finally:
        connection.close()
        if signature() != before:
            raise RuntimeError("DAG workspace replay detected a database change during immutable read")
    # The immutable metadata snapshot and exact payload hash are verified before
    # any trusted Python object is reconstructed, even if its path is replaced.
    artifact = joblib.load(io.BytesIO(payload))
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
