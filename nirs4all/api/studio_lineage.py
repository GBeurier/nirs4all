"""Exact Studio job and dataset identities on library-owned workspace runs."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .studio_scientific import StudioScientificJobError


def validate_studio_provenance(value: Any, job_id: str) -> dict[str, Any]:
    """Validate the bounded host-provided identity map before scientific work."""
    if not isinstance(value, dict) or set(value) != {"job_id", "dataset_ids_by_hash"} or value.get("job_id") != job_id:
        raise StudioScientificJobError("invalid_provenance", "Studio provenance must bind this job and an exact dataset hash map")
    mapping = value["dataset_ids_by_hash"]
    if not isinstance(mapping, dict) or not 1 <= len(mapping) <= 256:
        raise StudioScientificJobError("invalid_provenance", "Studio dataset hash map must contain 1 to 256 identities")
    if any(not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
           or not isinstance(identifier, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", identifier) is None
           for digest, identifier in mapping.items()):
        raise StudioScientificJobError("invalid_provenance", "Studio provenance requires SHA-256 content hashes and bounded dataset identifiers")
    return {"job_id": job_id, "dataset_ids_by_hash": dict(mapping)}


def record_studio_run_provenance(workspace_path: str | Path, run_ids: list[str], provenance: dict[str, Any]) -> dict[str, str]:
    """Attach exact declared identities to completed native runs, with no name inference."""
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    resolved: dict[str, str] = {}
    with WorkspaceStore(Path(workspace_path)) as store, store.transaction():
        for run_id in run_ids:
            record = store.get_run(run_id)
            if record is None:
                raise StudioScientificJobError("missing_persistence", "Cannot attach provenance to an absent workspace run")
            datasets = record.get("datasets") or []
            ids = []
            for dataset in datasets:
                dataset_id = provenance["dataset_ids_by_hash"].get(dataset.get("hash"))
                if dataset_id is None:
                    raise StudioScientificJobError("provenance_mismatch", "Recorded dataset content does not match the authorized Studio hash map")
                dataset["linked_dataset_id"] = dataset_id
                ids.append(dataset_id)
            config = dict(record.get("config") or {})
            existing = config.get("studio_provenance")
            attached = {"job_id": provenance["job_id"], "dataset_ids_by_hash": dict(provenance["dataset_ids_by_hash"])}
            if existing is not None and existing != attached:
                raise StudioScientificJobError("provenance_conflict", "A workspace run already belongs to different Studio provenance")
            config["studio_provenance"] = attached
            store._execute_with_retry("UPDATE runs SET config = ?, datasets = ? WHERE run_id = ?",  # noqa: SLF001 -- owner metadata writer
                                      [json.dumps(config), json.dumps(datasets), run_id])
            if len(set(ids)) == 1:
                resolved[run_id] = ids[0]
    return resolved


def recover_studio_job_lineage(workspace_path: str | Path, *, pipeline: list[Any], datasets: list[dict[str, Any]],
                              run_name: str, started_at: str, completed_at: str) -> dict[str, Any]:
    """Read old job lineage only when recipe, cohort, exact name and time agree.

    This never fits, mutates a workspace or repairs ambiguous evidence. Dataset
    descriptors contain an authorized ``dataset_id`` and canonical ``config``.
    A caller may persist returned identities using its existing job authority.
    """
    from nirs4all.api.run import _is_single_pipeline
    from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
    from nirs4all.pipeline.dagml.dataset import _materialize_dataset
    from nirs4all.pipeline.dagml.sequential_models import sequential_model_pipelines
    from nirs4all.pipeline.storage.store_schema import SCHEMA_VERSION

    def timestamp(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)

    start, end = timestamp(started_at), timestamp(completed_at)
    if end < start or (end - start).total_seconds() > 86_400:
        raise ValueError("Lineage recovery requires a bounded valid job interval")
    if not 1 <= len(datasets) <= 256 or not pipeline or len(pipeline) > 256:
        raise ValueError("Lineage recovery requires bounded pipeline/dataset declarations")
    # Workspace timestamps have one-second precision. Compare their exact
    # recorded interval rather than requiring an unavailable fractional part.
    start, end = start.replace(microsecond=0), end.replace(microsecond=0)

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            if str(value.get("class", "")).endswith(".FrozenDagMlSplitStep"):
                return normalize(value.get("params", {}).get("splitter"))
            return {key: normalize(child) for key, child in value.items()}
        if isinstance(value, list):
            return [normalize(child) for child in value]
        return value

    def recipe(value: Any) -> Any:
        # Canonicalize alias module paths and default parameters through the
        # same serializer used by publication, after unwrapping saved fold IDs.
        return normalize(serialize_component(deserialize_component(normalize(value))))

    pipelines = [pipeline] if _is_single_pipeline(pipeline) else pipeline
    expected = []
    for dataset in datasets:
        if set(dataset) != {"dataset_id", "config"} or not isinstance(dataset["dataset_id"], str):
            raise ValueError("Lineage dataset requires its exact Studio ID and authorized config")
        spectro = _materialize_dataset(dataset["config"])
        for index, original in enumerate(pipelines):
            base = f"{run_name}_p{index}" if len(pipelines) > 1 or len(datasets) > 1 else run_name
            steps = deserialize_component(original)
            checkpoints = sequential_model_pipelines(steps)
            for model_index, selected in enumerate(checkpoints or [steps]):
                name = f"{base}_p{model_index}" if checkpoints is not None else base
                expected.append((spectro.content_hash(), dataset["dataset_id"], name, recipe(selected)))

    root = Path(workspace_path)
    database = root / "store.sqlite"
    if not database.is_file():
        return {"run_ids": [], "dataset_run_ids": {}}
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise ValueError("Lineage recovery requires the current workspace schema")
        rows = connection.execute("SELECT run_id, name, config, datasets, created_at FROM runs "
                                  "WHERE status = 'completed' AND created_at >= ? AND created_at <= ? LIMIT 2049",
                                  [start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")]).fetchall()
        if len(rows) > 2048:
            raise ValueError("Lineage recovery candidate set exceeds its bound")
    finally:
        connection.close()
    matches: dict[str, str] = {}
    ambiguous: set[str] = set()
    for digest, dataset_id, name, expected_recipe in expected:
        candidates = []
        for row in rows:
            if row["name"] != name or not start <= timestamp(row["created_at"]) <= end:
                continue
            recorded_datasets = json.loads(row["datasets"])
            if len(recorded_datasets) != 1 or recorded_datasets[0].get("hash") != digest:
                continue
            config = json.loads(row["config"])
            if recipe(config.get("pipeline")) == expected_recipe:
                candidates.append(row["run_id"])
        if len(candidates) == 1:
            run_id = candidates[0]
            if run_id in matches and matches[run_id] != dataset_id:
                ambiguous.add(run_id)
            matches[run_id] = dataset_id
    for run_id in ambiguous:
        matches.pop(run_id, None)
    return {"run_ids": list(matches), "dataset_run_ids": matches}


def reconcile_studio_job_lineage(workspace_path: str | Path, job_id: str, **recovery: Any) -> dict[str, Any]:
    """Persist only the exact identities returned by the bounded recovery reader."""
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    recovered = recover_studio_job_lineage(workspace_path, **recovery)
    if not recovered["run_ids"]:
        return recovered
    mapping: dict[str, str] = {}
    with WorkspaceStore(Path(workspace_path)) as store:
        for run_id, dataset_id in recovered["dataset_run_ids"].items():
            run = store.get_run(run_id)
            datasets = run.get("datasets") if run else None
            if not isinstance(datasets, list) or len(datasets) != 1:
                raise ValueError("Recovered lineage no longer identifies one dataset")
            digest = datasets[0]["hash"]
            if digest in mapping and mapping[digest] != dataset_id:
                raise ValueError("Recovered content hash has ambiguous Studio dataset identities")
            mapping[digest] = dataset_id
    provenance = validate_studio_provenance({"job_id": job_id, "dataset_ids_by_hash": mapping}, job_id)
    recovered["dataset_run_ids"] = record_studio_run_provenance(workspace_path, recovered["run_ids"], provenance)
    return recovered
