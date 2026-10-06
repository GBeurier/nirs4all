"""Native results persistence for the dag-ml backend (P3 Slice 2b-i, B-C HYBRID).

ADDITIVE, OFF-by-default, native-only on-disk results for ``run(engine="dag-ml")``. The dag-ml path
is in-memory and touches NO legacy workspace (no SQLite store, no ArrayStore); this writer PRESERVES
that: it persists ONLY the things the dag-ml run already produced in memory (the native ScoreSet, the
projected prediction rows, captured refit artifacts, and a run header) and NEVER imports/instantiates
:class:`~nirs4all.pipeline.storage.workspace_store.WorkspaceStore` /
:class:`~nirs4all.pipeline.storage.array_store.ArrayStore`.

Layout (one directory per run, default ``./nirs4all_results/<run_id>/``):

* ``score_set.json`` — the dag-ml ScoreSet stored VERBATIM (the raw ``outcome["scores"]`` dict the
  projection consumed; for an operator sweep, the synthesized multi-variant ScoreSet
  :func:`~nirs4all.pipeline.dagml.result._project_operator_sweep` builds and feeds the projection).
  The canonical native object is written AS-IS — never re-authored Python-side; the manifest records
  its content hash.
* ``predictions.parquet`` — a columnar PROJECTION of the in-memory ``RunResult.predictions`` rows. A
  convenience view, NOT the source of truth. Arrays (``y_true`` / ``y_pred`` / ``y_proba``) are carried
  ONLY where the row has them (the direct-block rows); score-only rows store empty arrays + an
  ``arrays_present`` flag. SHAPE is carried (``y_true_shape`` / ``y_pred_shape`` / ``y_proba_shape``) so
  a multi-target row round-trips (the legacy portable parquet flattens without shape — incomplete for
  multi-target).
* ``artifacts/`` — the fitted REFIT model binaries (P3 Slice 2c-i). Each captured ``{estimator,
  y_transform}`` is joblib-serialized to ``artifacts/<node>/<variant>.joblib`` and recorded as a manifest
  ``artifacts[]`` ArtifactRef entry. ONLY the leakage-safe REFIT estimators are persisted (FIT_CV/OOF
  models never are). Both the in-process mechanism and the subprocess adapter capture refit models.
* ``manifest.json`` — the run header (run_id, engine, versions, datasets, configs/variants, models,
  metric, task_type) + CAPABILITY FLAGS (``has_model_artifacts`` true when any model artifact was
  captured + persisted; ``has_aggregate_predictions`` false for this slice) + the ScoreSet content hash +
  producer-node summaries + the ArtifactRef ``artifacts[]`` list + optional replay manifests for native
  composite exports + the manifest's own ``schema_version``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import joblib
import numpy as np
import polars as pl

from nirs4all.core.metrics import is_higher_better

if TYPE_CHECKING:
    from nirs4all.api.result import RunResult
    from nirs4all.data.predictions import Predictions

# The manifest's own schema version — bumped when the manifest layout changes (independent of the
# native dag-ml ScoreSet schema, which is owned by dag-ml and stored verbatim). v2 adds the model
# ArtifactRef ``artifacts[]`` list + the live ``has_model_artifacts`` capability flag (P3 Slice 2c-i).
# v3 adds ``stacking_replay`` metadata for native .n4a replay of branch stacking artifacts.
# v4 records generated views without a model; v5 adds one qualified,
# prediction-only multimodal refit artifact bound to those views.
MANIFEST_SCHEMA_VERSION = 3
_GENERATED_VIEW_SCHEMA_VERSION = 4
_GENERATED_PREDICT_SCHEMA_VERSION = 5

_DEFAULT_RESULTS_ROOT = "nirs4all_results"
_ENV_GATE = "N4A_NATIVE_RESULTS"
_GENERATED_VIEW_MANIFEST_FILE = "generated_view_manifest.json"
_MAX_GENERATED_VIEW_MANIFEST_BYTES = 64 * 1024 * 1024
_GENERATED_VIEW_FORBIDDEN_REPLAY_KEYS = (
    "stacking_replay", "host_hpo", "separation_replay", "residual_replay",
    "relation_replay_manifest", "source_stacking",
)

# The artifacts subtree holding the joblib-serialized fitted REFIT models, relative to the run dir.
_ARTIFACTS_DIR = "artifacts"
_STACKING_PRODUCER_NODE = "merge:stack"
_SECOND_STACKING_PRODUCER_NODE = "merge:stack.level2"
_META_MODEL_CONTROLLER_ID = "controller:nirs4all.meta_model"
_ARTIFACT_REFIT_MARKER = ":nirs4all:refit:"

# The per-row columns the parquet projection carries. The three array columns + their shape columns are
# appended per row; everything else is a scalar/JSON-encoded column.
_ARRAY_FIELDS = ("y_true", "y_pred", "y_proba")


def native_results_enabled(results_path: str | Path | None) -> bool:
    """Whether the native results writer should fire for this run (OFF by default).

    Enabled when EITHER an explicit ``results_path`` is given (the clean ``run()`` parameter, threaded
    past the empty ``_HONORED_RUNNER_KWARGS`` allowlist as a named arg) OR the ``N4A_NATIVE_RESULTS``
    env var is set to a truthy value (the per-process override). With NEITHER, the dag-ml run stays
    100% in-memory and writes nothing — behaviorally identical to today.
    """
    if results_path is not None:
        return True
    value = os.environ.get(_ENV_GATE, "")
    return value.strip().lower() not in {"0", "false", "no", ""}


def _resolve_run_dir(results_path: str | Path | None, run_id: str) -> Path:
    """Resolve the run output directory.

    An explicit ``results_path`` is the RUN-SET root (``<results_path>/<run_id>/``) — a stable parent
    the caller chose; the env-only gate defaults to ``./nirs4all_results/<run_id>/``. ``run_id`` is a
    sortable timestamp + short id so successive runs sort chronologically and never collide.
    """
    root = Path(results_path) if results_path is not None else Path.cwd() / _DEFAULT_RESULTS_ROOT
    return root / run_id


def _mint_run_id() -> str:
    """A sortable, collision-resistant run id: ``YYYYMMDDTHHMMSSffffffZ-<8hex>``."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    return f"{stamp}Z-{uuid.uuid4().hex[:8]}"


def _canonical_json(obj: Any) -> str:
    """Deterministic JSON text for hashing/writing (sorted keys, compact separators)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _score_set_hash(score_set: dict[str, Any] | None) -> str:
    """SHA-256 of the canonical-JSON ScoreSet — recorded in the manifest + verified by the reader."""
    return hashlib.sha256(_canonical_json(score_set).encode("utf-8")).hexdigest()


def _validated_generated_view_manifest(value: Any) -> tuple[bytes, str]:
    """Validate a generated-view manifest with its DAG-ML owner before persistence."""
    if not isinstance(value, dict):
        raise ValueError("generated view manifest must be a JSON object")
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("generated view manifest is not finite JSON") from exc
    return payload, _validate_generated_view_manifest_bytes(payload)


def _validate_generated_view_manifest_bytes(payload: bytes) -> str:
    """Validate the original bytes, including duplicate-key rejection, before load."""
    if len(payload) > _MAX_GENERATED_VIEW_MANIFEST_BYTES:
        raise ValueError("generated view manifest exceeds 64 MiB")
    import importlib

    native = importlib.import_module("dag_ml._dag_ml")
    validator = getattr(native, "validate_generated_view_manifest_in_process", None)
    if not callable(validator):
        raise RuntimeError("installed DAG-ML lacks generated view manifest validation")
    try:
        fingerprint = validator(payload.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - translate the native validation boundary.
        raise ValueError(f"generated view manifest is invalid: {exc}") from exc
    if not isinstance(fingerprint, str):
        raise ValueError("generated view manifest validator returned no fingerprint")
    return fingerprint


def _bytes_fingerprint(data: bytes) -> str:
    """SHA-256 of a model artifact's bytes — the ArtifactRef ``content_fingerprint`` (verified before load)."""
    return hashlib.sha256(data).hexdigest()


def _artifact_uri(artifact_id: str, index: int) -> str:
    """A filesystem-safe relative URI under ``artifacts/`` for one captured model artifact.

    The dag-ml ``artifact_id`` (e.g. ``artifact:model:compat.1:nirs4all:refit:variant:base``) is sanitized
    to a single path-safe filename — non-``[A-Za-z0-9._-]`` runs collapse to ``_`` — under ``artifacts/``,
    with the capture ``index`` prefixed so two artifacts that sanitize to the same name never collide.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", artifact_id).strip("_") or "artifact"
    return f"{_ARTIFACTS_DIR}/{index:03d}_{safe}.joblib"


def _branch_index_from_artifact_id(artifact_id: Any) -> int | None:
    """Return the canonical branch index encoded in ``artifact:branch:<n>.node:...`` ids."""
    if not isinstance(artifact_id, str):
        return None
    parts = artifact_id.split(":", 3)
    if len(parts) < 3 or parts[0] != "artifact" or parts[1] != "branch":
        return None
    raw_index = parts[2].split(".", 1)[0]
    try:
        return int(raw_index)
    except ValueError:
        return None


def _producer_node_from_artifact_id(artifact_id: Any) -> str | None:
    """Return the dag-ml producer node encoded in ``artifact:<producer>:nirs4all:refit:...`` ids."""
    if not isinstance(artifact_id, str) or not artifact_id.startswith("artifact:"):
        return None
    body = artifact_id[len("artifact:") :]
    producer, marker, _rest = body.partition(_ARTIFACT_REFIT_MARKER)
    if marker != _ARTIFACT_REFIT_MARKER or not producer:
        return None
    return producer


# The serialization backend the writer persists with + the reader can load. dag-ml's ArtifactRef.backend
# is the SERIALIZATION backend (ADR-16 / dag-ml-core ArtifactBackend enum: joblib/torch/tensorflow/onnx/
# safetensors/json/raw), NOT the ML framework. These refit estimators are joblib-dumped, so the backend is
# "joblib" — and the reader loads ONLY this backend (it joblib.loads), refusing any other before any load.
_JOBLIB_BACKEND = "joblib"


def _write_model_artifacts(run_dir: Path, refit_artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Joblib-serialize each captured REFIT model + build its manifest ArtifactRef entry (P3 Slice 2c-i).

    Each ``refit_artifacts`` entry is ``{artifact_id, estimator, y_transform, kind, controller_id,
    backend}`` (captured from the in-process store or the subprocess adapter; the node runner emits ``backend="joblib"``).
    We joblib-dump ``{estimator, y_transform}`` to ``artifacts/<uri>`` and return one ArtifactRef per
    artifact whose fields are dag-ml ArtifactRef-IDENTICAL: ``backend`` (the SERIALIZATION backend — the
    node runner's captured ``"joblib"``, NOT the ML framework, per ADR-16 / dag-ml ``ArtifactBackend``),
    ``uri`` (relative to the run dir), ``content_fingerprint`` (sha256 of the written bytes — NOT
    ``content_hash``), ``size_bytes``, ``kind``, plus ``controller_id`` when available and the source
    ``artifact_id``. Named Torch preserves the original native REFIT fingerprint
    and stores the carrier SHA separately as ``serialization_fingerprint``.
    An EMPTY input writes nothing and returns ``[]``.
    """
    if not refit_artifacts:
        return []
    (run_dir / _ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    refs: list[dict[str, Any]] = []
    from .host_artifacts import file_fingerprint, stage_host_artifacts

    for index, artifact in enumerate(refit_artifacts):
        uri = _artifact_uri(str(artifact.get("artifact_id") or f"artifact_{index}"), index)
        payload = {"estimator": artifact["estimator"], "y_transform": artifact["y_transform"]}
        if "multimodal_tuning_evidence" in artifact:
            payload["multimodal_tuning_evidence"] = artifact["multimodal_tuning_evidence"]
        from .named_torch_estimator import DagMLNamedTorchEstimator

        named = isinstance(payload["estimator"], DagMLNamedTorchEstimator)
        late = hasattr(payload["estimator"], "_nirs4all_late_partial_refit_origin")
        if late or "late_partial_refit_origin" in artifact or "late_partial_refit_fingerprint" in artifact:
            from .multimodal_contracts import validate_late_partial_refit_origin

            validate_late_partial_refit_origin(artifact, artifact)
            payload["late_partial_refit_origin"] = json.loads(json.dumps(artifact["late_partial_refit_origin"]))
            payload["late_partial_refit_fingerprint"] = artifact["late_partial_refit_fingerprint"]
        if named or "named_refit_origin" in artifact or "named_refit_fingerprint" in artifact:
            from .node_runner import validate_named_refit_origin

            if not named:
                raise ValueError("named Torch REFIT provenance requires its named estimator")
            validate_named_refit_origin(artifact, artifact)
            payload["named_refit_origin"] = json.loads(json.dumps(artifact["named_refit_origin"]))
            payload["named_refit_fingerprint"] = artifact["named_refit_fingerprint"]
        if "fold_estimators" in artifact:
            payload["fold_estimators"] = artifact["fold_estimators"]
        if "fold_selection" in artifact:
            payload["fold_selection"] = artifact["fold_selection"]
        with stage_host_artifacts(payload, run_dir, f"host_artifacts/artifact_{index}") as host_artifacts:
            joblib.dump(payload, run_dir / uri)
        fingerprint, size = file_fingerprint(run_dir / uri)
        # ArtifactRef ``backend`` = the SERIALIZATION backend the node runner recorded for these artifacts
        # ("joblib"); fall back to "joblib" only if the capture somehow lacked it (we always joblib-dump).
        backend = artifact.get("backend") or _JOBLIB_BACKEND
        ref = {
            "artifact_id": artifact.get("artifact_id"),
            "backend": backend,
            "uri": uri,
            "content_fingerprint": fingerprint.removeprefix("sha256:"),
            "size_bytes": size,
            "kind": artifact.get("kind"),
            "controller_id": artifact.get("controller_id"),
        }
        if named:
            # The native fingerprint binds the genuine REFIT, not the joblib
            # carrier. Never replace that origin with a freshly exported hash.
            ref["serialization_fingerprint"] = ref["content_fingerprint"]
            ref["content_fingerprint"] = artifact["content_fingerprint"]
            ref["named_refit_origin"] = json.loads(json.dumps(payload["named_refit_origin"]))
            ref["named_refit_fingerprint"] = payload["named_refit_fingerprint"]
        if late:
            ref["serialization_fingerprint"] = fingerprint.removeprefix("sha256:")
            ref["content_fingerprint"] = artifact["content_fingerprint"]
            ref["late_partial_refit_origin"] = json.loads(json.dumps(payload["late_partial_refit_origin"]))
            ref["late_partial_refit_fingerprint"] = payload["late_partial_refit_fingerprint"]
        if host_artifacts:
            ref["host_artifacts"] = host_artifacts
        branch_index = _branch_index_from_artifact_id(ref["artifact_id"])
        if branch_index is not None:
            # Neutral branch metadata. Export interprets this as source_index only when the native run
            # manifest proves the shape is by_source fusion; for duplication fusion it remains branch order.
            ref["branch_index"] = branch_index
        producer_node = _producer_node_from_artifact_id(ref["artifact_id"])
        if producer_node is not None:
            ref["producer_node"] = producer_node
        refs.append(ref)
    return refs


def _as_list(value: Any) -> list[Any]:
    """Normalize an array-like row field to a flat python list (empty list for an absent array)."""
    if value is None:
        return []
    arr = np.asarray(value)
    if arr.size == 0:
        return []
    return cast(list[Any], arr.ravel().tolist())


def _shape_of(value: Any) -> list[int]:
    """Row-array shape as a list (``[]`` for an absent/empty array). Carries multi-target width."""
    if value is None:
        return []
    arr = np.asarray(value)
    if arr.size == 0:
        return []
    return list(arr.shape)


def _physical_sample_ids(entry: dict[str, Any], expected_size: int) -> list[str]:
    """Return native identities only when the row proves exact coverage.

    Older native result directories did not persist these identities. They stay
    readable, but must never acquire identities from positional indices.
    """
    metadata = entry.get("metadata")
    candidate = metadata.get("physical_sample_id") if isinstance(metadata, dict) else None
    if not isinstance(candidate, (list, tuple)):
        return []
    values = list(candidate)
    if len(values) != expected_size or not all(isinstance(value, str) and value for value in values):
        return []
    if len(set(values)) != len(values):
        return []
    return values


def _projection_rows(predictions: Predictions) -> list[dict[str, Any]]:
    """Project the in-memory prediction rows into flat, parquet-writable dicts (carrying shape).

    Reads the rows WITH arrays from the in-memory buffer (no store). Each row carries the identity +
    role columns (dataset / config_name / variant_id / model_name / partition / fold_id), the
    per-sample arrays FLATTENED with a paired ``*_shape`` column so multi-target round-trips, the
    per-row score scalars + the nested ``scores`` dict (JSON-encoded), and the metric / task_type /
    target metadata. ``arrays_present`` flags whether this row had real y arrays (a direct-block row)
    vs. a score-only row (empty arrays).
    """
    rows: list[dict[str, Any]] = []
    for entry in predictions.filter_predictions(load_arrays=True):
        y_true = entry.get("y_true")
        y_pred = entry.get("y_pred")
        y_proba = entry.get("y_proba")
        sample_indices = entry.get("sample_indices") or []
        sample_ids = _physical_sample_ids(entry, len(sample_indices))
        # Target width from the 2D array shape (1 for single-target / score-only rows); target_names are
        # not carried on the projected rows, so derive them from the width so the names are CONSISTENT with
        # the persisted shape (``["y"]`` single / ``["y0", ...]`` multi) — the SHAPE columns are the
        # round-trip source of truth, names are descriptive metadata.
        true_shape = _shape_of(y_true)
        target_width = int(true_shape[1]) if len(true_shape) > 1 else 1
        target_names = [f"y{i}" for i in range(target_width)] if target_width > 1 else ["y"]
        row: dict[str, Any] = {
            "dataset": entry.get("dataset_name", ""),
            "config_name": entry.get("config_name", ""),
            # No dedicated variant_id column on the projected rows — a sweep's per-variant identity is
            # carried by (config_name, model_name); the column mirrors config_name so a downstream
            # reader can group by variant without re-deriving it.
            "variant_id": str(entry.get("config_name") or ""),
            "model_name": entry.get("model_name", ""),
            "partition": entry.get("partition", ""),
            "fold_id": str(entry.get("fold_id") or ""),
            "refit_context": str(entry.get("refit_context") or ""),
            "sample_indices": [int(i) for i in (sample_indices.tolist() if isinstance(sample_indices, np.ndarray) else sample_indices)],
            "sample_ids": sample_ids,
            "y_true": [float(v) for v in _as_list(y_true)],
            "y_pred": [float(v) for v in _as_list(y_pred)],
            "y_proba": [float(v) for v in _as_list(y_proba)],
            "y_true_shape": true_shape,
            "y_pred_shape": _shape_of(y_pred),
            "y_proba_shape": _shape_of(y_proba),
            "weights": [float(v) for v in _as_list(entry.get("weights"))],
            "arrays_present": bool(_as_list(y_pred)),
            "val_score": _opt_float(entry.get("val_score")),
            "test_score": _opt_float(entry.get("test_score")),
            "train_score": _opt_float(entry.get("train_score")),
            "scores": _canonical_json(entry.get("scores") or {}),
            "result_metadata": _canonical_json(entry.get("result_metadata") or {}),
            "metric": entry.get("metric", ""),
            "task_type": entry.get("task_type", ""),
            "target_width": target_width,
            "target_names": _canonical_json(target_names),
        }
        rows.append(row)
    return rows


def _opt_float(value: Any) -> float | None:
    """Coerce a score scalar to float, preserving ``None`` (a null score in the parquet)."""
    return float(value) if value is not None else None


def _score_set_producer_nodes(score_set: dict[str, Any] | None, *, final_only: bool = False) -> list[str]:
    """Producer nodes present in the native ScoreSet, optionally limited to final/off-fold reports."""
    reports = (score_set or {}).get("reports")
    if not isinstance(reports, list):
        return []
    nodes: set[str] = set()
    for report in reports:
        if not isinstance(report, dict):
            continue
        if final_only and not (report.get("fold_id") is None and report.get("partition") in {"final", "test"}):
            continue
        producer = report.get("producer_node")
        if producer is not None:
            nodes.add(str(producer))
    return sorted(nodes)


def _stacking_replay_manifest(
    score_set: dict[str, Any] | None, artifact_refs: list[dict[str, Any]],
    selectors: list[dict[str, Any]] | None = None,
    *, probability_producers: set[str] | None = None,
    source_orders: dict[str, list[str]] | None = None,
    source_ports: dict[str, dict[str, str]] | None = None, _allow_multi: bool = True,
    outer_fold_ids: list[str] | None = None,
    producer_classes: dict[str, str] | None = None,
    target_node: str = _STACKING_PRODUCER_NODE,
) -> dict[str, Any] | None:
    """Build the native stacking replay manifest when base + meta artifacts are unambiguous.

    dag-ml's meta-node builds meta-features by sorting base prediction-input keys and concatenating each
    producer's prediction-value block in that order. Persisting the matching producer order lets the .n4a
    exporter reconstruct the same meta-feature matrix from raw X using the captured base REFIT models,
    then replay the captured meta REFIT model without invoking the legacy bridge.
    """
    # A no-test run can capture the meta REFIT estimator without emitting a
    # final meta score (its training input is OOF, not raw-feature inference).
    # The real REFIT artifact/controller identity below attests replayability;
    # do not require a fabricated final score merely to export that estimator.
    scored_producers = _score_set_producer_nodes(score_set)
    if target_node not in scored_producers:
        return None
    if _allow_multi and target_node == _STACKING_PRODUCER_NODE and _SECOND_STACKING_PRODUCER_NODE in scored_producers:
        by_producer: dict[str, list[dict[str, Any]]] = {}
        for ref in artifact_refs:
            producer = str(ref.get("producer_node") or _producer_node_from_artifact_id(ref.get("artifact_id")) or "")
            by_producer.setdefault(producer, []).append(ref)
        stage_nodes = [_STACKING_PRODUCER_NODE]
        for level in range(2, len(artifact_refs) + 1):
            node = f"{_STACKING_PRODUCER_NODE}.level{level}"
            if node not in scored_producers:
                break
            stage_nodes.append(node)
        if len(stage_nodes) < 2 or any(len(by_producer.get(node, [])) != 1 for node in stage_nodes):
            return None
        meta_refs = [by_producer[node][0] for node in stage_nodes]
        if any(ref.get("controller_id") == _META_MODEL_CONTROLLER_ID and all(ref is not meta for meta in meta_refs) for ref in artifact_refs):
            return None
        first_refs = [ref for ref in artifact_refs if all(ref is not meta for meta in meta_refs[1:])]
        first_stage = _stacking_replay_manifest(
            score_set, first_refs, selectors,
                probability_producers=probability_producers, source_orders=source_orders, source_ports=source_ports,
                _allow_multi=False, outer_fold_ids=outer_fold_ids, producer_classes=producer_classes,
        )
        if first_stage is None:
            return None
        stages = [first_stage]
        for previous_node, node, ref, _previous_ref in zip(stage_nodes, stage_nodes[1:], meta_refs[1:], meta_refs, strict=False):
            source_nodes = (source_orders or {}).get(node, [previous_node])
            if len(set(source_nodes)) != len(source_nodes) or any(len(by_producer.get(source, [])) != 1 for source in source_nodes):
                return None
            stages.append({
                "schema_version": 1,
                "producer_node": node,
                "meta_artifact_id": ref.get("artifact_id"),
                "base_producers": [{
                    "artifact_id": by_producer[source][0].get("artifact_id"),
                    "producer_node": source,
                    "meta_feature_key": f"{source}.{(source_ports or {}).get(node, {}).get(source, 'oof')}",
                    "column_block": "probability_values" if (source_ports or {}).get(node, {}).get(source) == "proba" or source in (probability_producers or set()) else "prediction_values",
                    **({"column_projection": "selected_class"} if (source_ports or {}).get(node, {}).get(source) == "proba"
                       and by_producer[source][0].get("late_partial_refit_origin", {}).get("class_labels") is None else {}),
                } for source in source_nodes],
                "meta_feature_construction": {
                    "kind": "base_prediction_column_stack",
                    "producer_order": "declared_source_order" if node in (source_orders or {}) else "sorted_prediction_input_base_key",
                    "prediction_space": "selected_class_probability" if all((source_ports or {}).get(node, {}).get(source) == "proba" or source in (probability_producers or set()) for source in source_nodes) else "original_target",
                    "column_blocks": "one block per base producer, preserving target column order",
                },
            })
        return {
            "schema_version": 2,
            "producer_node": stage_nodes[-1],
            "stages": stages,
        }

    meta_refs = [
        ref
        for ref in artifact_refs
        if _producer_node_from_artifact_id(ref.get("artifact_id")) == target_node
        and ref.get("controller_id") == _META_MODEL_CONTROLLER_ID
    ]
    if len(meta_refs) != 1:
        return None
    meta_ref = meta_refs[0]

    base_refs = [
        ref
        for ref in artifact_refs
        if ref is not meta_ref and _producer_node_from_artifact_id(ref.get("artifact_id")) is not None
    ]
    if not base_refs:
        return None

    ordered_base_refs = sorted(base_refs, key=lambda ref: str(ref.get("producer_node") or _producer_node_from_artifact_id(ref.get("artifact_id")) or ""))
    base_producers: list[dict[str, Any]] = []
    for ref in ordered_base_refs:
        producer_node = str(ref.get("producer_node") or _producer_node_from_artifact_id(ref.get("artifact_id")) or "")
        if not producer_node:
            return None
        entry = {
            "artifact_id": ref.get("artifact_id"),
            "producer_node": producer_node,
            # This is the base key order used by node_runner._ordered_oof_specs after suffix stripping.
            "meta_feature_key": f"{producer_node}.{(source_ports or {}).get(target_node, {}).get(producer_node, 'oof')}",
            "column_block": "probability_values" if (source_ports or {}).get(target_node, {}).get(producer_node) == "proba" else "prediction_values",
            **({"column_projection": "selected_class"} if (source_ports or {}).get(target_node, {}).get(producer_node) == "proba"
               and ref.get("late_partial_refit_origin", {}).get("class_labels") is None else {}),
        }
        if ref.get("branch_index") is not None:
            entry["branch_index"] = int(ref["branch_index"])
        base_producers.append(entry)

    replay_groups: list[dict[str, Any]] = []
    if selectors:
        reports = (score_set or {}).get("reports", [])
        for selector in selectors:
            branch = selector.get("branch")
            model = selector.get("model")
            selected = [
                index for index, entry in enumerate(base_producers)
                if (model is None or entry["producer_node"] == model)
                and (branch is None or entry["producer_node"].startswith(f"branch:{str(branch).removeprefix('branch_')}" + "."))
            ]
            if not selected:
                return None
            if selector.get("select", "all") != "all":
                from dag_ml import select_stacking_producers_json  # type: ignore[attr-defined]  # PyO3 export has no Python stub

                selected_nodes = json.loads(select_stacking_producers_json(json.dumps({
                    "producer_nodes": [base_producers[index]["producer_node"] for index in selected],
                        "select": selector["select"], "metric": selector.get("metric") or "rmse",
                        "reports": reports,
                        "fold_ids": outer_fold_ids or [],
                        "producer_classes": producer_classes or {},
                })))
                selected_by_node = {base_producers[index]["producer_node"]: index for index in selected}
                selected = [selected_by_node[node] for node in selected_nodes]
                if not selected:
                    return None
            aggregate = selector.get("aggregate")
            if aggregate is None:
                replay_groups.extend({"key": base_producers[index]["meta_feature_key"], "members": [index]}
                                     for index in selected)
                continue
            if aggregate not in {"mean", "weighted_mean", "proba_mean"}:
                return None
            group: dict[str, Any] = {
                "key": f"merge:stack.branch.{branch}.oof", "members": selected,
                "aggregate": aggregate,
            }
            if aggregate == "weighted_mean":
                metric = selector.get("metric") or "rmse"
                higher_better = is_higher_better(metric)
                weights = []
                for index in selected:
                    producer = base_producers[index]["producer_node"]
                    # Native stacking averages eligible fold scores; inner
                    # validation evidence must not set full-refit replay weights.
                    scores = [float(report["metrics"][metric]) for report in reports
                              if report.get("producer_node") == producer
                              and report.get("partition") == "validation"
                              and report.get("level") == "sample"
                              and report.get("fold_id") is not None
                              and (not outer_fold_ids or report["fold_id"] in outer_fold_ids)
                              and metric in report.get("metrics", {})
                              and np.isfinite(report["metrics"][metric])]
                    score = float(np.mean(scores)) if scores else None
                    if score is None:
                        weights.append(0.0)
                    elif higher_better:
                        weights.append(max(float(score), 0.0))
                    elif score >= 0:
                        weights.append(1.0 / (float(score) + 1e-10))
                    else:
                        weights.append(0.0)
                group["weights"] = weights if any(weight > 0.0 for weight in weights) else None
            if aggregate == "proba_mean" or selector.get("metadata", {}).get("prediction_output") == "proba":
                group["proba"] = True
            replay_groups.append(group)
        if target_node not in (source_orders or {}):
            replay_groups.sort(key=lambda group: group["key"])

    return {
        "schema_version": 1,
        "producer_node": target_node,
        "meta_artifact_id": meta_ref.get("artifact_id"),
        "meta_producer_node": str(meta_ref.get("producer_node") or _producer_node_from_artifact_id(meta_ref.get("artifact_id")) or target_node),
        "base_producers": base_producers,
        **({"reduction_groups": replay_groups} if selectors else {}),
        "meta_feature_construction": {
            "kind": "base_prediction_column_stack",
            "producer_order": "sorted_prediction_input_base_key",
            "prediction_space": "original_target",
            "column_blocks": "one block per base producer, preserving target column order",
        },
    }


def _manifest_header(result: RunResult, predictions: Predictions, score_set: dict[str, Any] | None, run_id: str, run_dir: Path, artifact_refs: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the run-header manifest (versions, datasets, configs/variants, models, capability flags).

    ``artifact_refs`` is the list of dag-ml-identical model ArtifactRef entries
    (:func:`_write_model_artifacts`). ``has_model_artifacts`` is TRUE iff any was captured + persisted (the
    in-process or subprocess mechanism with at least one REFIT model); FALSE only when no model was captured.
    """
    from nirs4all import __version__ as nirs4all_version

    try:
        from dag_ml import __version__ as dag_ml_version
    except Exception:  # noqa: BLE001 - version is best-effort metadata, never fail the run for it.
        dag_ml_version = "unknown"

    rows = predictions.filter_predictions(load_arrays=False)
    datasets = sorted({str(row.get("dataset_name", "")) for row in rows if row.get("dataset_name")})
    config_names = sorted({str(row.get("config_name", "")) for row in rows if row.get("config_name")})
    model_names = sorted({str(row.get("model_name", "")) for row in rows if row.get("model_name")})
    metrics = sorted({str(row.get("metric", "")) for row in rows if row.get("metric")})
    task_types = sorted({str(row.get("task_type", "")) for row in rows if row.get("task_type")})

    best = result.best
    selected_variant = str(best.get("config_name") or "") if isinstance(best, dict) else ""

    score_meta = score_set or {}
    host_searches = [
        evidence for artifact in result._dagml_refit_artifacts
        for evidence in getattr(artifact.get("estimator"), "_nirs4all_host_hpo_history", [])
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "engine": "dag-ml",
        "nirs4all_version": nirs4all_version,
        "dag_ml_version": dag_ml_version,
        "datasets": datasets,
        "config_names": config_names,
        "variant_names": config_names,
        "model_names": model_names,
        "target_names": getattr(result, "_dagml_target_names", ["y"]),
        "metric": metrics[0] if len(metrics) == 1 else metrics,
        "task_type": task_types[0] if len(task_types) == 1 else task_types,
        "selected_variant": selected_variant,
        "plan_id": score_meta.get("plan_id"),
        "bundle_id": score_meta.get("bundle_id"),
        "producer_nodes": _score_set_producer_nodes(score_set),
        "final_producer_nodes": _score_set_producer_nodes(score_set, final_only=True),
        "num_predictions": predictions.num_predictions,
        "score_set_hash": _score_set_hash(score_set),
        "capabilities": {
            "has_model_artifacts": bool(artifact_refs),
            "has_aggregate_predictions": False,
        },
        "artifacts": artifact_refs,
        "evaluation": {dataset: metadata["evaluation"] for dataset, metadata in getattr(result, "per_dataset", {}).items() if "evaluation" in metadata},
        "files": {
            "score_set": "score_set.json",
            "predictions": "predictions.parquet",
        },
    }
    stacking_replay = _stacking_replay_manifest(
        score_set, artifact_refs, getattr(result, "_dagml_stacking_selectors", None),
        probability_producers=getattr(result, "_dagml_stacking_probability_producers", None),
        source_orders=getattr(result, "_dagml_stacking_source_orders", None),
        source_ports=getattr(result, "_dagml_stacking_source_ports", None),
        outer_fold_ids=getattr(result, "_dagml_stacking_outer_fold_ids", None),
        producer_classes=getattr(result, "_dagml_stacking_producer_classes", None),
        _allow_multi=not getattr(result, "_dagml_stacking_independent_terminal", False),
        target_node=getattr(result, "_dagml_stacking_replay_producer", _STACKING_PRODUCER_NODE),
    )
    if host_searches and getattr(result, "_dagml_generated_view_manifest", None) is None:
        manifest["host_hpo"] = {"profile": "host_optimizer_search_v1", "portable": False, "searches": host_searches}
    if stacking_replay is not None:
        manifest["stacking_replay"] = stacking_replay
    for key in ("relation_replay_manifest", "relation_materialization_manifest", "source_stacking", "stacking_evaluation", "separation_replay", "residual_replay"):
        recorded = [metadata[key] for metadata in getattr(result, "per_dataset", {}).values() if isinstance(metadata.get(key), dict)]
        if recorded and all(value == recorded[0] for value in recorded):
            manifest[key] = recorded[0]
    return manifest


def separation_replay_manifest(graph: dict[str, Any], metadata_key: str, artifacts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Record the selected native artifact for every metadata-fanned model.

    Operator SELECT leaves inactive choice nodes in the compiled union graph but
    refits only the winning choice of each metadata branch. Match graph nodes
    from the captured artifacts, never require artifacts from inactive choices.
    """
    if not artifacts:
        return None
    model_nodes = [node for node in graph.get("nodes", []) if node.get("kind") == "model"]
    if len(model_nodes) < 2 or len(artifacts) < 2:
        return None
    members: list[dict[str, str]] = []
    seen_values: set[str] = set()
    for artifact in artifacts:
        artifact_id = str(artifact.get("artifact_id", ""))
        matches = [node for node in model_nodes if artifact_id.startswith(f"artifact:{node['id']}:")]
        if len(matches) != 1:
            return None
        node = matches[0]
        selector = ((node.get("metadata") or {}).get("dsl_branch_selector") or {}).get("metadata") or {}
        value = selector.get(metadata_key)
        if value is None:
            return None
        value = str(value)
        if value in seen_values:
            raise ValueError("metadata separation replay has duplicate selected values")
        seen_values.add(value)
        members.append({"value": value, "artifact_id": artifact_id})
    return {
        "kind": "by_metadata_concat", "producer_node": "merge:concat",
        "metadata_key": metadata_key, "members": members,
    }


def write_native_results(
    result: RunResult,
    score_set: dict[str, Any] | None,
    results_path: str | Path | None,
) -> Path:
    """Write the native results directory for a dag-ml run; return the run directory.

    Writes ``manifest.json`` + ``score_set.json`` (VERBATIM) + ``predictions.parquet`` + the
    ``artifacts/`` model tree (the captured fitted REFIT estimators, P3 Slice 2c-i) under
    ``<root>/<run_id>/``. Called ONLY when :func:`native_results_enabled` (OFF by default). NEVER
    touches the legacy workspace store. The fitted models are read from ``result._dagml_refit_artifacts``
    (captured from the in-process store or the subprocess adapter); an empty list writes no
    ``artifacts/`` payload and records ``has_model_artifacts:false``.
    """
    if score_set is None:
        raise ValueError("write_native_results requires a dag-ml ScoreSet (got None); the native writer is only called for a real dag-ml run.")
    generated_manifest = getattr(result, "_dagml_generated_view_manifest", None)
    generated_payload = _validated_generated_view_manifest(generated_manifest) if generated_manifest is not None else None
    generated_predict_contract = getattr(result, "_dagml_generated_prediction_contract", None)
    if generated_predict_contract is not None:
        if generated_payload is None or len(result._dagml_refit_artifacts) != 1:  # noqa: SLF001
            raise ValueError("generated prediction requires one captured refit artifact and a view manifest")
        from .multimodal_contracts import generated_prediction_contract

        if generated_prediction_contract(result._dagml_refit_artifacts[0]["estimator"]) != generated_predict_contract:  # noqa: SLF001
            raise ValueError("generated prediction contract disagrees with the captured refit model")
    if generated_payload is not None and any(
        isinstance(metadata, dict)
        and any(key in metadata for key in _GENERATED_VIEW_FORBIDDEN_REPLAY_KEYS)
        for metadata in getattr(result, "per_dataset", {}).values()
    ):
        raise ValueError("native results generated views cannot carry model replay metadata")
    run_id = _mint_run_id()
    run_dir = _resolve_run_dir(results_path, run_id)
    # exist_ok=False: a minted run_id must be unique; a collision means a real bug, not a silent reuse.
    run_dir.mkdir(parents=True, exist_ok=False)

    predictions = result.predictions

    # score_set.json — the canonical native object, VERBATIM (no Python re-authoring).
    (run_dir / "score_set.json").write_text(_canonical_json(score_set), encoding="utf-8")

    # predictions.parquet — the columnar convenience projection (arrays carry shape).
    rows = _projection_rows(predictions)
    schema: dict[str, Any] = {
        "dataset": pl.Utf8, "config_name": pl.Utf8, "variant_id": pl.Utf8, "model_name": pl.Utf8,
        "partition": pl.Utf8, "fold_id": pl.Utf8, "refit_context": pl.Utf8,
        "sample_indices": pl.List(pl.Int64),
        "sample_ids": pl.List(pl.Utf8),
        "y_true": pl.List(pl.Float64), "y_pred": pl.List(pl.Float64), "y_proba": pl.List(pl.Float64),
        "y_true_shape": pl.List(pl.Int64), "y_pred_shape": pl.List(pl.Int64), "y_proba_shape": pl.List(pl.Int64),
        "weights": pl.List(pl.Float64), "arrays_present": pl.Boolean,
        "val_score": pl.Float64, "test_score": pl.Float64, "train_score": pl.Float64,
        "scores": pl.Utf8, "metric": pl.Utf8, "task_type": pl.Utf8, "result_metadata": pl.Utf8,
        "target_width": pl.Int64, "target_names": pl.Utf8,
    }
    pl.DataFrame(rows, schema=schema).write_parquet(run_dir / "predictions.parquet")

    # artifacts/ — joblib-serialize the captured fitted REFIT models (P3 Slice 2c-i) + their ArtifactRefs.
    # Only fitted REFIT models produce payloads; empty captures leave the capability flag false.
    # v4 keeps the historical no-model state. v5 persists exactly one fitted
    # multimodal model for prediction on an explicit, schema-matched cohort.
    artifact_refs = (
        _write_model_artifacts(run_dir, result._dagml_refit_artifacts)  # noqa: SLF001
        if generated_payload is None or generated_predict_contract is not None else []
    )

    initial_package = getattr(result, "_dagml_initial_full_refit_package", None)
    if generated_payload is not None:
        initial_package = None
    if initial_package is not None:
        from dag_ml import InitialFullRefitPackage

        InitialFullRefitPackage(initial_package)
        (run_dir / "initial_full_refit_package.json").write_text(_canonical_json(initial_package), encoding="utf-8")

    source_captures = getattr(result, "_dagml_source_training_captures", [])
    source_payload = None
    if source_captures:
        from .attested_by_source import validate_source_package_bindings

        validate_source_package_bindings([item["package"] for item in source_captures], artifact_refs)
        source_payload = _canonical_json(source_captures).encode("utf-8")
        (run_dir / "source_training_captures.json").write_bytes(source_payload)

    # manifest.json — the run header + capability flags + the ScoreSet hash + the model ArtifactRefs.
    manifest = _manifest_header(result, predictions, score_set, run_id, run_dir, artifact_refs)
    if source_payload is not None:
        manifest["source_training_captures_ref"] = {"path": "source_training_captures.json", "sha256": hashlib.sha256(source_payload).hexdigest()}
    if initial_package is not None:
        manifest["files"]["initial_full_refit_package"] = "initial_full_refit_package.json"
        manifest["initial_full_refit_package_fingerprint"] = initial_package["package_fingerprint"]
    if generated_payload is not None:
        payload_bytes, fingerprint = generated_payload
        manifest["schema_version"] = (_GENERATED_PREDICT_SCHEMA_VERSION if generated_predict_contract is not None
                                      else _GENERATED_VIEW_SCHEMA_VERSION)
        (run_dir / _GENERATED_VIEW_MANIFEST_FILE).write_bytes(payload_bytes)
        manifest["generated_view_manifest_ref"] = {
            "path": _GENERATED_VIEW_MANIFEST_FILE,
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "fingerprint": fingerprint,
        }
        if generated_predict_contract is not None:
            manifest["generated_model_replay"] = {
                **generated_predict_contract,
                "view_manifest_fingerprint": fingerprint,
                "artifact_id": artifact_refs[0]["artifact_id"],
            }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return run_dir


def read_native_results(run_dir: str | Path) -> dict[str, Any]:
    """Read a native results directory back into a :class:`Predictions`-consumable form.

    Returns ``{"manifest", "score_set", "predictions", "artifacts"}`` where ``predictions`` is a populated
    in-memory :class:`~nirs4all.data.predictions.Predictions` whose rows round-trip the writer's projection
    (the per-sample arrays reshaped from their ``*_shape`` columns, so a multi-target row recovers its 2D
    shape), and ``artifacts`` is the list of rehydrated ``{artifact_id, estimator, y_transform, ...}`` model
    payloads (P3 Slice 2c-i; empty when the run has no model artifacts). VALIDATES the ScoreSet against the
    manifest's recorded hash, and — for EACH model artifact — VERIFIES the on-disk bytes against the
    recorded ``content_fingerprint`` BEFORE :func:`joblib.load` (joblib.load executes code: verify-then-load
    on trusted input). A corrupt/edited ``score_set.json`` OR a tampered model artifact raises
    :class:`ValueError` before any load.
    """
    from nirs4all.data.predictions import Predictions

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    score_set_text = (run_dir / "score_set.json").read_text(encoding="utf-8")
    score_set = json.loads(score_set_text)

    expected_hash = manifest.get("score_set_hash")
    actual_hash = _score_set_hash(score_set)
    if expected_hash != actual_hash:
        raise ValueError(
            f"native results score_set.json hash mismatch in {run_dir}: manifest recorded {expected_hash!r} "
            f"but score_set.json hashes to {actual_hash!r} (the ScoreSet was edited or corrupted)."
        )

    generated_manifest = None
    generated_ref = manifest.get("generated_view_manifest_ref")
    generated_path = run_dir / _GENERATED_VIEW_MANIFEST_FILE
    generated_version = manifest.get("schema_version")
    if generated_version in (_GENERATED_VIEW_SCHEMA_VERSION, _GENERATED_PREDICT_SCHEMA_VERSION) and generated_ref is None:
        raise ValueError("native generated results require a generated view manifest reference")
    if generated_ref is not None and generated_version not in (_GENERATED_VIEW_SCHEMA_VERSION, _GENERATED_PREDICT_SCHEMA_VERSION):
        raise ValueError("native results generated view manifest requires schema v4 or v5")
    if generated_version != _GENERATED_PREDICT_SCHEMA_VERSION and "generated_model_replay" in manifest:
        raise ValueError("native generated-model replay requires schema v5")
    if manifest.get("schema_version") == _GENERATED_VIEW_SCHEMA_VERSION:
        capabilities = manifest.get("capabilities")
        if (manifest.get("artifacts") != [] or not isinstance(capabilities, dict)
                or capabilities.get("has_model_artifacts") is not False):
            raise ValueError("native results v4 cannot contain replayable model artifacts")
        if ("initial_full_refit_package_fingerprint" in manifest
                or "initial_full_refit_package" in manifest.get("files", {})
                or any(key in manifest for key in _GENERATED_VIEW_FORBIDDEN_REPLAY_KEYS)):
            raise ValueError("native results v4 cannot contain model replay metadata")
    if generated_version == _GENERATED_PREDICT_SCHEMA_VERSION:
        from .multimodal_contracts import generated_input_schema_sha256

        capabilities = manifest.get("capabilities")
        refs = manifest.get("artifacts")
        contract = manifest.get("generated_model_replay")
        if (not isinstance(capabilities, dict) or capabilities.get("has_model_artifacts") is not True
                or not isinstance(refs, list) or len(refs) != 1 or not isinstance(refs[0], dict)
                or not isinstance(contract, dict)
                or set(contract) != {"schema_version", "mode", "source_order", "input_schema", "input_schema_sha256", "view_manifest_fingerprint", "artifact_id"}
                or contract.get("schema_version") != 1 or contract.get("mode") != "explicit_cohort_predict_only"
                or not isinstance(contract.get("input_schema"), dict) or not contract["input_schema"]
                or not isinstance(contract.get("source_order"), list)
                or not all(isinstance(name, str) and name for name in contract["source_order"])
                or len(contract["source_order"]) != len(set(contract["source_order"]))
                or set(contract["source_order"]) != set(contract["input_schema"])
                or contract.get("artifact_id") != refs[0].get("artifact_id")
                or not isinstance(generated_ref, dict)
                or contract.get("view_manifest_fingerprint") != generated_ref.get("fingerprint")
                or "initial_full_refit_package_fingerprint" in manifest
                or "initial_full_refit_package" in manifest.get("files", {})):
            raise ValueError("native results v5 has an invalid generated-model replay contract")
        try:
            if contract["input_schema_sha256"] != generated_input_schema_sha256(contract["input_schema"]):
                raise ValueError("native results v5 generated input schema fingerprint mismatch")
        except (TypeError, ValueError) as exc:
            raise ValueError("native results v5 generated input schema fingerprint mismatch") from exc
    if generated_ref is None:
        if os.path.lexists(generated_path):
            raise ValueError("native results contain an undeclared generated view manifest")
    else:
        if (not isinstance(generated_ref, dict)
                or set(generated_ref) != {"path", "sha256", "fingerprint"}
                or generated_ref.get("path") != _GENERATED_VIEW_MANIFEST_FILE):
            raise ValueError("native results generated view manifest reference is invalid")
        if generated_path.is_symlink():
            raise ValueError("native results generated view manifest must be a regular file")
        try:
            if not stat.S_ISREG(os.lstat(generated_path).st_mode):
                raise ValueError("native results generated view manifest must be a regular file")
            flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
            descriptor = os.open(generated_path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("native results generated view manifest must be a regular file")
                payload_bytes = stream.read(_MAX_GENERATED_VIEW_MANIFEST_BYTES + 1)
        except OSError as exc:
            raise ValueError("native results generated view manifest is missing or unreadable") from exc
        if len(payload_bytes) > _MAX_GENERATED_VIEW_MANIFEST_BYTES:
            raise ValueError("native results generated view manifest exceeds 64 MiB")
        if hashlib.sha256(payload_bytes).hexdigest() != generated_ref["sha256"]:
            raise ValueError("native results generated view manifest byte fingerprint mismatch")
        fingerprint = _validate_generated_view_manifest_bytes(payload_bytes)
        if fingerprint != generated_ref["fingerprint"]:
            raise ValueError("native results generated view manifest TCV1 fingerprint mismatch")
        generated_manifest = json.loads(payload_bytes)

    predictions = Predictions()
    df = pl.read_parquet(run_dir / "predictions.parquet")
    # V1 native result directories did not carry stable sample IDs. Preserve
    # backward readability but do not fabricate them from row positions.
    has_sample_ids = "sample_ids" in df.columns
    for row in df.iter_rows(named=True):
        result_metadata = json.loads(row.get("result_metadata") or "{}")
        if not isinstance(result_metadata, dict):
            raise ValueError("native prediction result_metadata must be a JSON object")
        predictions.add_prediction(
            dataset_name=row["dataset"],
            config_name=row["config_name"],
            model_name=row["model_name"],
            partition=row["partition"],
            fold_id=row["fold_id"] or None,
            refit_context=row["refit_context"] or None,
            sample_indices=[int(i) for i in row["sample_indices"]] if row["sample_indices"] else None,
            metadata=(
                {"physical_sample_id": list(row["sample_ids"])}
                if has_sample_ids
                and isinstance(row["sample_ids"], list)
                and row["sample_ids"]
                and len(row["sample_ids"]) == len(row["sample_indices"] or [])
                and all(isinstance(sample_id, str) and sample_id for sample_id in row["sample_ids"])
                and len(set(row["sample_ids"])) == len(row["sample_ids"])
                else None
            ),
            weights=[float(w) for w in row["weights"]] if row["weights"] else None,
            y_true=_restore_array(row["y_true"], row["y_true_shape"]),
            y_pred=_restore_array(row["y_pred"], row["y_pred_shape"]),
            y_proba=_restore_array(row["y_proba"], row["y_proba_shape"]),
            val_score=row["val_score"],
            test_score=row["test_score"],
            train_score=row["train_score"],
            scores=json.loads(row["scores"]) if row["scores"] else None,
            result_metadata=result_metadata,
            metric=row["metric"],
            task_type=row["task_type"],
        )
    predictions.flush()

    source_captures = []
    source_ref = manifest.get("source_training_captures_ref")
    if source_ref is not None:
        if (not isinstance(source_ref, dict) or set(source_ref) != {"path", "sha256"}
                or source_ref["path"] != "source_training_captures.json"):
            raise ValueError("native source training capture reference is invalid")
        source_path = run_dir / source_ref["path"]
        if source_path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("native source training capture is oversized")
        source_bytes = source_path.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != source_ref["sha256"]:
            raise ValueError("native source training capture hash differs")
        source_captures = json.loads(source_bytes)
        if not isinstance(source_captures, list) or not source_captures:
            raise ValueError("native source training captures must be a non-empty list")
        from dag_ml import TrainingOutcome

        from .attested_by_source import validate_source_package_bindings

        for item in source_captures:
            TrainingOutcome(item["outcome"])
            if item["package"]["training_outcome"]["outcome_fingerprint"] != item["outcome"]["outcome_fingerprint"]:
                raise ValueError("native source package does not bind its original training outcome")
        validate_source_package_bindings([item["package"] for item in source_captures], manifest.get("artifacts", []))
    elif (run_dir / "source_training_captures.json").exists():
        raise ValueError("native results contain an undeclared source training capture")
    artifacts = _rehydrate_artifacts(run_dir, manifest.get("artifacts", []))
    if generated_version == _GENERATED_PREDICT_SCHEMA_VERSION:
        from .multimodal_contracts import generated_prediction_contract

        expected = dict(manifest["generated_model_replay"])
        expected.pop("view_manifest_fingerprint")
        expected.pop("artifact_id")
        if generated_prediction_contract(artifacts[0]["estimator"]) != expected:
            raise ValueError("native results v5 model disagrees with its generated prediction contract")

    initial_package = None
    initial_path = manifest.get("files", {}).get("initial_full_refit_package")
    if initial_path is not None:
        if initial_path != "initial_full_refit_package.json":
            raise ValueError("native results initial full-refit package path is invalid")
        initial_package = json.loads((run_dir / initial_path).read_text(encoding="utf-8"))
        from dag_ml import InitialFullRefitPackage

        InitialFullRefitPackage(initial_package)
        if initial_package["package_fingerprint"] != manifest.get("initial_full_refit_package_fingerprint"):
            raise ValueError("native results initial full-refit package fingerprint mismatch")
    return {"manifest": manifest, "score_set": score_set, "predictions": predictions, "artifacts": artifacts, "initial_full_refit_package": initial_package, "generated_view_manifest": generated_manifest, "source_training_captures": source_captures}


def _validate_portable_uri(uri: Any) -> str:
    """Validate a manifest artifact ``uri`` is a PORTABLE relative path, returning it (else raise).

    Mirrors dag-ml's ``validate_relative_artifact_uri`` (dag-ml-core
    ``runtime/prediction_store.rs``) so an EDITED manifest cannot point the reader at an arbitrary file:
    a ``joblib.load`` of an absolute path / ``..`` traversal / URI scheme would read+execute pickle
    opcodes from outside the run dir. Refused (BEFORE any ``read_bytes`` / ``joblib.load``):

    * a non-string / empty uri;
    * a control character;
    * an absolute path (leading ``/`` or ``\\``) or a Windows drive prefix (``C:``);
    * a scheme / colon in the FIRST path segment (``http://``, ``s3://``, ``file://``, ...);
    * any ``..`` component (parent-directory traversal).
    """
    if not isinstance(uri, str) or not uri:
        raise ValueError(f"native results model artifact has an empty / non-string uri ({uri!r})")
    # Reject ALL Unicode control chars (category Cc = C0 + DEL + C1), mirroring Rust's char::is_control
    # in dag-ml's validate_relative_artifact_uri — not just C0/DEL (e.g. NEL U+0085 must be refused too).
    if any(unicodedata.category(ch) == "Cc" for ch in uri):
        raise ValueError(f"native results model artifact uri {uri!r} has control characters")
    if uri.startswith(("/", "\\")):
        raise ValueError(f"native results model artifact uri {uri!r} must be a relative path (not absolute)")
    if len(uri) >= 2 and uri[0].isascii() and uri[0].isalpha() and uri[1] == ":":
        raise ValueError(f"native results model artifact uri {uri!r} must be a relative path (no drive prefix)")
    segments = re.split(r"[/\\]", uri)
    if ":" in segments[0]:
        raise ValueError(f"native results model artifact uri {uri!r} must not include a scheme or colon in its first path segment")
    if ".." in segments:
        raise ValueError(f"native results model artifact uri {uri!r} must not contain `..` components (path traversal)")
    return uri


def _rehydrate_artifacts(run_dir: Path, artifact_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate the URI + backend, verify the bytes against ``content_fingerprint``, THEN joblib-load (2c-i).

    :func:`joblib.load` executes pickle opcodes, so each artifact is treated as TRUSTED INPUT and three
    guards run BEFORE any read/load (a tampered manifest never reaches the filesystem or the unpickler):

    1. **Portable URI** — the ``uri`` must be a safe relative path within the run dir
       (:func:`_validate_portable_uri`): an absolute path / ``..`` traversal / URI scheme is refused
       BEFORE ``read_bytes`` (so the reader cannot be steered at an arbitrary file to load).
    2. **Backend** — only the joblib serialization backend is loadable here (we ``joblib.load``); an
       unknown / unexpected ``backend`` is refused before the load.
    3. **Content fingerprint** — a mismatch between the on-disk bytes' sha256 and the recorded
       ``content_fingerprint`` raises before the load (a corrupted/edited payload never unpickles).
       Named Torch checks ``serialization_fingerprint`` for the carrier, then verifies
       the retained native fingerprint against its genuine REFIT origin and state.

    Each loaded payload is ``{estimator, y_transform}``; the returned entry merges in the ArtifactRef's
    identity/metadata (``artifact_id`` / ``kind`` / ``controller_id`` / ``backend`` / ``uri``).
    """
    rehydrated: list[dict[str, Any]] = []
    from .host_artifacts import _safe_uri, file_fingerprint, hydrate_host_artifacts, verify_host_artifacts

    for ref in artifact_refs:
        uri = _validate_portable_uri(ref.get("uri"))
        backend = ref.get("backend")
        if backend != _JOBLIB_BACKEND:
            raise ValueError(
                f"native results model artifact {uri!r} has unsupported backend {backend!r}: only "
                f"{_JOBLIB_BACKEND!r} artifacts are loadable here — refusing to joblib.load it."
            )
        path = run_dir / uri
        sidecar_refs = ref.get("host_artifacts")
        named = any(key in ref for key in ("named_refit_origin", "named_refit_fingerprint"))
        late = any(key in ref for key in ("late_partial_refit_origin", "late_partial_refit_fingerprint"))
        anchored = named or late
        expected = ref.get("serialization_fingerprint") if anchored else ref.get("content_fingerprint")
        if "serialization_fingerprint" in ref and not anchored:
            raise ValueError("native results serialization fingerprint lacks its original REFIT provenance")
        if late and (not isinstance(ref.get("late_partial_refit_origin"), dict)
                     or ref.get("content_fingerprint") != ref.get("late_partial_refit_fingerprint")
                     or not isinstance(expected, str) or len(expected) != 64):
            raise ValueError("native results late partial artifact lacks its original REFIT or serialization fingerprint")
        if named and (not isinstance(ref.get("named_refit_origin"), dict)
                      or ref.get("content_fingerprint") != ref.get("named_refit_fingerprint")
                      or not isinstance(expected, str) or len(expected) != 64):
            raise ValueError("native results named Torch artifact lacks its original REFIT or serialization fingerprint")
        sidecar_owner = tempfile.TemporaryDirectory(prefix="nirs4all_native_sidecars_") if sidecar_refs else None
        try:
            if sidecar_owner is not None:
                for directory_ref in cast(list[dict[str, Any]], sidecar_refs):
                    for file_ref in directory_ref["files"]:
                        safe_uri = _safe_uri(file_ref["uri"])
                        target = Path(sidecar_owner.name) / safe_uri
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with (run_dir / safe_uri).open("rb") as source_stream, target.open("wb") as target_stream:
                            shutil.copyfileobj(source_stream, target_stream, 1024 * 1024)
            directories = verify_host_artifacts(Path(sidecar_owner.name) if sidecar_owner else run_dir, sidecar_refs)
            with tempfile.TemporaryDirectory(prefix="nirs4all_native_model_") as snapshot_dir:
                snapshot = Path(snapshot_dir) / "model.joblib"
                with path.open("rb") as source_stream, snapshot.open("wb") as target_stream:
                    shutil.copyfileobj(source_stream, target_stream, 1024 * 1024)
                actual, size = file_fingerprint(snapshot)
                actual = actual.removeprefix("sha256:")
                if expected != actual or size != ref.get("size_bytes"):
                    raise ValueError(
                        f"native results model artifact {uri!r} content_fingerprint mismatch in {run_dir}: manifest "
                        f"recorded {expected!r} but the bytes hash to {actual!r} (the artifact was edited or "
                        "corrupted) — refusing to joblib.load it."
                    )
                payload = joblib.load(snapshot)
            hydrate_host_artifacts(payload, directories, owner=sidecar_owner)
            from .named_torch_estimator import DagMLNamedTorchEstimator

            if named or isinstance(payload["estimator"], DagMLNamedTorchEstimator):
                from .node_runner import validate_named_refit_origin

                if (not named or payload.get("named_refit_origin") != ref.get("named_refit_origin")
                        or payload.get("named_refit_fingerprint") != ref.get("named_refit_fingerprint")):
                    raise ValueError("native results named Torch REFIT provenance disagrees with its manifest")
                validate_named_refit_origin(payload, ref)
            if late or hasattr(payload["estimator"], "_nirs4all_late_partial_refit_origin"):
                from .multimodal_contracts import validate_late_partial_refit_origin

                if (not late or payload.get("late_partial_refit_origin") != ref.get("late_partial_refit_origin")
                        or payload.get("late_partial_refit_fingerprint") != ref.get("late_partial_refit_fingerprint")):
                    raise ValueError("native results late partial REFIT provenance disagrees with its manifest")
                validate_late_partial_refit_origin(payload, ref)
        except Exception:
            if sidecar_owner is not None:
                sidecar_owner.cleanup()
            raise
        entry = {
            "artifact_id": ref.get("artifact_id"),
            "estimator": payload["estimator"],
            "y_transform": payload["y_transform"],
            "kind": ref.get("kind"),
            "controller_id": ref.get("controller_id"),
            "backend": ref.get("backend"),
            "uri": uri,
            "content_fingerprint": ref["content_fingerprint"] if anchored else actual,
        }
        if named:
            entry["serialization_fingerprint"] = actual
            entry["named_refit_origin"] = json.loads(json.dumps(payload["named_refit_origin"]))
            entry["named_refit_fingerprint"] = payload["named_refit_fingerprint"]
        if late:
            entry["serialization_fingerprint"] = actual
            entry["late_partial_refit_origin"] = json.loads(json.dumps(payload["late_partial_refit_origin"]))
            entry["late_partial_refit_fingerprint"] = payload["late_partial_refit_fingerprint"]
        for key in ("fold_estimators", "fold_selection"):
            if key in payload:
                entry[key] = payload[key]
        if ref.get("branch_index") is not None:
            entry["branch_index"] = int(ref["branch_index"])
        if ref.get("producer_node") is not None:
            entry["producer_node"] = str(ref["producer_node"])
        rehydrated.append(entry)
    return rehydrated


def _restore_array(values: list[float] | None, shape: list[int] | None) -> np.ndarray | None:
    """Rebuild a per-sample array from its flattened values + recorded shape (``None`` if empty)."""
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if shape and len(shape) > 1:
        arr = arr.reshape(tuple(int(d) for d in shape))
    return arr
