"""Trusted Python model archive replay through an explicit DAG PREDICT phase.

These host artifacts are not portable Core archives. Their digest protects
against corruption, not a malicious producer: joblib can execute Python code.
"""

from __future__ import annotations

import hashlib
import io
import json
import warnings
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nirs4all.api.result import PredictResult


def general_archive_manifest(path: str | Path) -> dict[str, Any] | None:
    """Inspect the provenance marker without importing/deserializing operators."""
    source = Path(path)
    if not source.is_file() or not zipfile.is_zipfile(source):
        return None
    with zipfile.ZipFile(source) as archive:
        manifests = [member for member in archive.infolist() if member.filename == "manifest.json"]
        if len(manifests) != 1:
            return None
        if manifests[0].file_size > 1024 * 1024:
            raise ValueError("general archive manifest exceeds 1 MiB")
        manifest = json.loads(archive.read(manifests[0]))
    return manifest if isinstance(manifest, dict) and manifest.get("source_type") == "dagml_native" else None


def load_general_archive(path: str | Path, *, expected_archive_fingerprint: str | None = None) -> dict[str, Any]:
    """Verify the exact archive member bytes before loading one trusted model."""
    import joblib

    source = Path(path)
    # One immutable byte snapshot prevents a manifest/payload replacement race.
    data = source.read_bytes()
    archive_fingerprint = "sha256:" + hashlib.sha256(data).hexdigest()
    if expected_archive_fingerprint is not None and archive_fingerprint != expected_archive_fingerprint:
        raise ValueError("general Session source archive changed after loading")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("general archive contains duplicate ZIP member names")
        manifest = json.loads(archive.read("manifest.json"))
        if not isinstance(manifest, dict) or manifest.get("source_type") != "dagml_native":
            raise ValueError("archive is not a captured DAG host-model archive")
        members = [name for name in names if name.startswith("artifacts/") and not name.endswith("/")]
        if len(members) != 1 or not members[0].endswith(".joblib"):
            raise ValueError("general archive requires exactly one captured host-model payload")
        member = members[0]
        if archive.getinfo(member).file_size > 512 * 1024 * 1024:
            raise ValueError("general archive model payload exceeds 512 MiB")
        payload = archive.read(member)
        fingerprint = "sha256:" + hashlib.sha256(payload).hexdigest()
        expected = manifest.get("artifact_integrity", {}).get(member)
        if expected is not None and expected != fingerprint:
            raise ValueError("general archive artifact content fingerprint mismatch; refusing to deserialize")
        if "artifact_integrity" in manifest and expected is None:
            raise ValueError("general archive integrity manifest omits its model artifact")
        if expected is None:
            warnings.warn("This older DAG host archive has no recorded artifact digest. Load only from a trusted producer; integrity provenance is unavailable.", UserWarning, stacklevel=2)
        model = joblib.load(io.BytesIO(payload))
    if not callable(getattr(model, "predict", None)):
        raise ValueError("general archive model is not predict-capable")
    return {
        "artifact": {"artifact_id": member, "estimator": model, "y_transform": None, "content_fingerprint": fingerprint},
        "manifest": manifest, "archive_fingerprint": archive_fingerprint,
        "artifact_integrity_verified": expected is not None,
        "pipeline": [{"model": model}],
        "model_name": source.stem,
    }


def predict_general_archive(path: str | Path, data: Any, *, expected_archive_fingerprint: str | None = None) -> PredictResult:
    """Replay a captured aggregate model; no old executor or retraining is used."""
    from nirs4all.api.result import PredictResult

    from .dataset import _materialize_dataset
    from .general_replay import predict_captured_artifact

    loaded = load_general_archive(path, expected_archive_fingerprint=expected_archive_fingerprint)
    values, metadata = predict_captured_artifact(
        loaded["artifact"], _materialize_dataset(data), pipeline=loaded["pipeline"],
        target_names=loaded["manifest"].get("target_names", ["y"]),
    )
    metadata.update({
        "archive_fingerprint": loaded["archive_fingerprint"],
        "artifact_integrity_verified": loaded["artifact_integrity_verified"],
        "training_provenance": loaded["manifest"], "portable": False,
    })
    return PredictResult(y_pred=values, metadata=metadata, model_name=loaded["model_name"])


def predict_general_result(result: Any, data: Any) -> PredictResult:
    """Use the live fitted artifact of a general Session without requiring disk."""
    from nirs4all.api.result import PredictResult

    from .dataset import _materialize_dataset
    from .general_replay import predict_captured_artifact

    children = getattr(result, "runs", None)
    if children is not None:
        result = result._source_run(None)
    artifacts = result._dagml_refit_artifacts
    if len(artifacts) != 1:
        raise NotImplementedError("live Session replay for a multi-artifact composition requires its captured aggregate model")
    artifact = artifacts[0]
    values, metadata = predict_captured_artifact(
        artifact, _materialize_dataset(data), pipeline=[{"model": artifact["estimator"]}], target_names=result._dagml_target_names,
    )
    metadata["portable"] = False
    metadata["training_evaluation"] = {name: item["evaluation"] for name, item in result.per_dataset.items() if "evaluation" in item}
    return PredictResult(y_pred=values, metadata=metadata, model_name=result.best.get("model_name", ""))


def copy_general_archive(source: Path, target: str | Path, *, expected_fingerprint: str | None) -> Path:
    """Save a loaded Session without fitting or changing its original model bytes."""
    import os
    import tempfile

    payload = source.read_bytes()
    if "sha256:" + hashlib.sha256(payload).hexdigest() != expected_fingerprint:
        raise ValueError("general Session source archive changed after loading")
    destination = Path(target).with_suffix(".n4a")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".n4a-save-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
