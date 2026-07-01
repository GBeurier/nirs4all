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
  models never are). Present only for the in-process mechanism (the subprocess mechanism fits in a child
  process this one cannot reach → ``has_model_artifacts:false`` + NO ``artifacts[]`` entries).
* ``manifest.json`` — the run header (run_id, engine, versions, datasets, configs/variants, models,
  metric, task_type) + CAPABILITY FLAGS (``has_model_artifacts`` true when any model artifact was
  captured + persisted; ``has_aggregate_predictions`` false for this slice) + the ScoreSet content hash +
  producer-node summaries + the ArtifactRef ``artifacts[]`` list + the manifest's own ``schema_version``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import polars as pl

if TYPE_CHECKING:
    from nirs4all.api.result import RunResult
    from nirs4all.data.predictions import Predictions

# The manifest's own schema version — bumped when the manifest layout changes (independent of the
# native dag-ml ScoreSet schema, which is owned by dag-ml and stored verbatim). v2 adds the model
# ArtifactRef ``artifacts[]`` list + the live ``has_model_artifacts`` capability flag (P3 Slice 2c-i).
MANIFEST_SCHEMA_VERSION = 2

_DEFAULT_RESULTS_ROOT = "nirs4all_results"
_ENV_GATE = "N4A_NATIVE_RESULTS"

# The artifacts subtree holding the joblib-serialized fitted REFIT models, relative to the run dir.
_ARTIFACTS_DIR = "artifacts"

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


# The serialization backend the writer persists with + the reader can load. dag-ml's ArtifactRef.backend
# is the SERIALIZATION backend (ADR-16 / dag-ml-core ArtifactBackend enum: joblib/torch/tensorflow/onnx/
# safetensors/json/raw), NOT the ML framework. These refit estimators are joblib-dumped, so the backend is
# "joblib" — and the reader loads ONLY this backend (it joblib.loads), refusing any other before any load.
_JOBLIB_BACKEND = "joblib"


def _write_model_artifacts(run_dir: Path, refit_artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Joblib-serialize each captured REFIT model + build its manifest ArtifactRef entry (P3 Slice 2c-i).

    Each ``refit_artifacts`` entry is ``{artifact_id, estimator, y_transform, kind, controller_id,
    backend}`` (captured host-side from the in-process store; the node runner emits ``backend="joblib"``).
    We joblib-dump ``{estimator, y_transform}`` to ``artifacts/<uri>`` and return one ArtifactRef per
    artifact whose fields are dag-ml ArtifactRef-IDENTICAL: ``backend`` (the SERIALIZATION backend — the
    node runner's captured ``"joblib"``, NOT the ML framework, per ADR-16 / dag-ml ``ArtifactBackend``),
    ``uri`` (relative to the run dir), ``content_fingerprint`` (sha256 of the written bytes — NOT
    ``content_hash``), ``size_bytes``, ``kind``, plus ``controller_id`` when available and the source
    ``artifact_id``. An EMPTY input writes nothing and returns ``[]`` (the subprocess mechanism → no
    loadable artifacts, never a faked payload).
    """
    if not refit_artifacts:
        return []
    (run_dir / _ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    refs: list[dict[str, Any]] = []
    for index, artifact in enumerate(refit_artifacts):
        uri = _artifact_uri(str(artifact.get("artifact_id") or f"artifact_{index}"), index)
        payload = {"estimator": artifact["estimator"], "y_transform": artifact["y_transform"]}
        joblib.dump(payload, run_dir / uri)
        data = (run_dir / uri).read_bytes()
        # ArtifactRef ``backend`` = the SERIALIZATION backend the node runner recorded for these artifacts
        # ("joblib"); fall back to "joblib" only if the capture somehow lacked it (we always joblib-dump).
        backend = artifact.get("backend") or _JOBLIB_BACKEND
        refs.append(
            {
                "artifact_id": artifact.get("artifact_id"),
                "backend": backend,
                "uri": uri,
                "content_fingerprint": _bytes_fingerprint(data),
                "size_bytes": len(data),
                "kind": artifact.get("kind"),
                "controller_id": artifact.get("controller_id"),
            }
        )
    return refs


def _as_list(value: Any) -> list[Any]:
    """Normalize an array-like row field to a flat python list (empty list for an absent array)."""
    if value is None:
        return []
    arr = np.asarray(value)
    if arr.size == 0:
        return []
    return arr.ravel().tolist()


def _shape_of(value: Any) -> list[int]:
    """Row-array shape as a list (``[]`` for an absent/empty array). Carries multi-target width."""
    if value is None:
        return []
    arr = np.asarray(value)
    if arr.size == 0:
        return []
    return list(arr.shape)


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


def _manifest_header(result: RunResult, predictions: Predictions, score_set: dict[str, Any] | None, run_id: str, run_dir: Path, artifact_refs: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the run-header manifest (versions, datasets, configs/variants, models, capability flags).

    ``artifact_refs`` is the list of dag-ml-identical model ArtifactRef entries
    (:func:`_write_model_artifacts`). ``has_model_artifacts`` is TRUE iff any was captured + persisted (the
    in-process mechanism with at least one REFIT model); FALSE (with an empty ``artifacts`` list) for the
    subprocess mechanism, which cannot reach the child-process models.
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
    return {
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
        "files": {
            "score_set": "score_set.json",
            "predictions": "predictions.parquet",
        },
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
    (captured host-side from the in-process store); an empty list (subprocess mechanism) writes no
    ``artifacts/`` payload and records ``has_model_artifacts:false``.
    """
    if score_set is None:
        raise ValueError("write_native_results requires a dag-ml ScoreSet (got None); the native writer is only called for a real dag-ml run.")
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
        "y_true": pl.List(pl.Float64), "y_pred": pl.List(pl.Float64), "y_proba": pl.List(pl.Float64),
        "y_true_shape": pl.List(pl.Int64), "y_pred_shape": pl.List(pl.Int64), "y_proba_shape": pl.List(pl.Int64),
        "weights": pl.List(pl.Float64), "arrays_present": pl.Boolean,
        "val_score": pl.Float64, "test_score": pl.Float64, "train_score": pl.Float64,
        "scores": pl.Utf8, "metric": pl.Utf8, "task_type": pl.Utf8,
        "target_width": pl.Int64, "target_names": pl.Utf8,
    }
    pl.DataFrame(rows, schema=schema).write_parquet(run_dir / "predictions.parquet")

    # artifacts/ — joblib-serialize the captured fitted REFIT models (P3 Slice 2c-i) + their ArtifactRefs.
    # Empty for the subprocess mechanism (no capturable child-process models) → no payload, flag false.
    artifact_refs = _write_model_artifacts(run_dir, result._dagml_refit_artifacts)  # noqa: SLF001

    # manifest.json — the run header + capability flags + the ScoreSet hash + the model ArtifactRefs.
    manifest = _manifest_header(result, predictions, score_set, run_id, run_dir, artifact_refs)
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

    predictions = Predictions()
    df = pl.read_parquet(run_dir / "predictions.parquet")
    for row in df.iter_rows(named=True):
        predictions.add_prediction(
            dataset_name=row["dataset"],
            config_name=row["config_name"],
            model_name=row["model_name"],
            partition=row["partition"],
            fold_id=row["fold_id"] or None,
            refit_context=row["refit_context"] or None,
            sample_indices=[int(i) for i in row["sample_indices"]] if row["sample_indices"] else None,
            weights=[float(w) for w in row["weights"]] if row["weights"] else None,
            y_true=_restore_array(row["y_true"], row["y_true_shape"]),
            y_pred=_restore_array(row["y_pred"], row["y_pred_shape"]),
            y_proba=_restore_array(row["y_proba"], row["y_proba_shape"]),
            val_score=row["val_score"],
            test_score=row["test_score"],
            train_score=row["train_score"],
            scores=json.loads(row["scores"]) if row["scores"] else None,
            metric=row["metric"],
            task_type=row["task_type"],
        )
    predictions.flush()

    artifacts = _rehydrate_artifacts(run_dir, manifest.get("artifacts", []))

    return {"manifest": manifest, "score_set": score_set, "predictions": predictions, "artifacts": artifacts}


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

    Each loaded payload is ``{estimator, y_transform}``; the returned entry merges in the ArtifactRef's
    identity/metadata (``artifact_id`` / ``kind`` / ``controller_id`` / ``backend`` / ``uri``).
    """
    rehydrated: list[dict[str, Any]] = []
    for ref in artifact_refs:
        uri = _validate_portable_uri(ref.get("uri"))
        backend = ref.get("backend")
        if backend != _JOBLIB_BACKEND:
            raise ValueError(
                f"native results model artifact {uri!r} has unsupported backend {backend!r}: only "
                f"{_JOBLIB_BACKEND!r} artifacts are loadable here — refusing to joblib.load it."
            )
        path = run_dir / uri
        data = path.read_bytes()
        expected = ref.get("content_fingerprint")
        actual = _bytes_fingerprint(data)
        if expected != actual:
            raise ValueError(
                f"native results model artifact {uri!r} content_fingerprint mismatch in {run_dir}: manifest "
                f"recorded {expected!r} but the bytes hash to {actual!r} (the artifact was edited or "
                "corrupted) — refusing to joblib.load it."
            )
        payload = joblib.load(path)  # uri + backend + fingerprint all verified above — trusted bytes
        rehydrated.append(
            {
                "artifact_id": ref.get("artifact_id"),
                "estimator": payload["estimator"],
                "y_transform": payload["y_transform"],
                "kind": ref.get("kind"),
                "controller_id": ref.get("controller_id"),
                "backend": ref.get("backend"),
                "uri": uri,
            }
        )
    return rehydrated


def _restore_array(values: list[float] | None, shape: list[int] | None) -> np.ndarray | None:
    """Rebuild a per-sample array from its flattened values + recorded shape (``None`` if empty)."""
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if shape and len(shape) > 1:
        arr = arr.reshape(tuple(int(d) for d in shape))
    return arr
