"""Owner-published history SQL stays executable, exact and read-only in Store v5."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from nirs4all.pipeline.storage import WorkspaceStore
from nirs4all.pipeline.storage.store_schema import SCHEMA_VERSION


@pytest.fixture
def history_store(tmp_path: Path) -> Path:
    workspace = tmp_path / "history"
    WorkspaceStore(workspace).close()
    database = workspace / "store.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executemany(
            "INSERT INTO runs (run_id, name, config, datasets, status, created_at, completed_at, project_id) VALUES (?, ?, '{}', '[]', 'completed', ?, ?, ?)",
            [
                ("run-a", "User_refit", "2026-09-01T10:00:00+02:00", "2026-09-01T10:01:15+02:00", "project-a"),
                ("run-b", "Second", "2026-09-02T10:00:00+02:00", None, "project-b"),
                ("run-c", "Third", "2026-09-03T10:00:00+02:00", None, "project-a"),
            ],
        )
        connection.execute("INSERT INTO pipelines (pipeline_id, run_id, name, dataset_name) VALUES ('p', 'run-a', 'P_refit', 'corn')")
        connection.execute(
            "INSERT INTO chains (chain_id, pipeline_id, steps, model_step_idx, model_class, fold_artifacts, shared_artifacts) VALUES ('c', 'p', '[]', 1, 'Ridge', ?, ?)",
            ('{"0":{"model":"artifact-a"},"1":["artifact-a","artifact-b"]}', '{"shared":"artifact-b"}'),
        )
        for artifact_id, size in [("artifact-a", 5), ("artifact-b", 7), ("unused", 999)]:
            connection.execute(
                "INSERT INTO artifacts (artifact_id, artifact_path, content_hash, size_bytes) VALUES (?, ?, 'hash', ?)",
                (artifact_id, f"{artifact_id}.joblib", size),
            )
        for index, (fold, partition, refit) in enumerate([
            ("0", "val", None), ("0", "train", None), ("1", "val", None),
            ("avg", "val", None), ("w_avg", "val", None), ("0_agg", "val", None),
            ("final", "train", "full_train"), ("final", "test", "full_train"),
        ]):
            connection.execute(
                """INSERT INTO predictions (prediction_id, pipeline_id, chain_id, dataset_name, model_name, model_class, fold_id, partition, refit_context, metric, task_type)
                   VALUES (?, 'p', 'c', 'corn', 'Ridge', 'Ridge', ?, ?, ?, 'rmse', 'regression')""",
                (f"prediction-{index}", fold, partition, refit),
            )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")
    return database


def _contract() -> dict[str, Any]:
    resource = files("nirs4all.pipeline.storage").joinpath("contracts/workspace_store_run_history_v1.json")
    contract: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    assert contract["schema_id"] == "nirs4all.workspace-store-run-history.v1"
    assert contract["workspace_store_schema_version"] == SCHEMA_VERSION == 5
    return contract


def test_history_filters_before_pagination_and_counts_the_complete_selection(history_store: Path) -> None:
    queries = _contract()["queries"]
    before = hashlib.sha256(history_store.read_bytes()).hexdigest(), history_store.stat().st_mtime_ns
    with sqlite3.connect(history_store.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        first = dict(connection.execute(queries["runs_page"], ("project-a", 1, 0)).fetchone())
        second = dict(connection.execute(queries["runs_page"], ("project-a", 1, 1)).fetchone())
        assert first["run_id"] == "run-c"
        assert first["duration_seconds"] is None
        assert second["run_id"] == "run-a"
        assert second["name"] == "User_refit"
        assert second["duration_seconds"] == 75
        assert connection.execute(queries["runs_total"], ("project-a",)).fetchone()[0] == 2
        assert connection.execute(queries["runs_total"], (None,)).fetchone()[0] == 3
        assert connection.execute(queries["runs_total"], ("' OR 1=1 --",)).fetchone()[0] == 0
    assert (hashlib.sha256(history_store.read_bytes()).hexdigest(), history_store.stat().st_mtime_ns) == before
    assert not list(history_store.parent.glob("store.sqlite-*"))


def test_history_counts_actual_prediction_roles_without_double_counting(history_store: Path) -> None:
    query = _contract()["queries"]["run_counts"]
    with sqlite3.connect(history_store.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        assert connection.execute(query, ("run-a",)).fetchone() == (1, 2, 2, 1)
        assert connection.execute(query, ("run-b",)).fetchone() == (0, 0, 0, 0)


def test_history_sums_distinct_registered_artifacts_without_opening_files(history_store: Path) -> None:
    queries = _contract()["queries"]
    with sqlite3.connect(history_store.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        assert connection.execute(queries["run_artifact_size"], ("run-a",)).fetchone() == (12,)
        assert connection.execute(queries["run_artifact_size"], ("run-b",)).fetchone() == (0,)
        assert connection.execute(queries["run_model_classes"], ("run-a",)).fetchall() == [("Ridge", 1)]
    assert not list(history_store.parent.rglob("*.joblib"))
