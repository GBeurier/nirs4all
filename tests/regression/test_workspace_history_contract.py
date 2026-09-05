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


def test_history_stats_are_not_limited_by_the_page_size(history_store: Path) -> None:
    with sqlite3.connect(history_store) as connection:
        connection.executemany("INSERT INTO runs (run_id, name, status) VALUES (?, 'Historical', ?)", [(f"bulk-{i}", "failed" if i % 2 else "running") for i in range(600)])
        connection.executemany("INSERT INTO runs (run_id, name, status) VALUES (?, 'Interrupted', ?)", [("cancelled", "cancelled"), ("partial", "partial")])
        connection.commit()
    with sqlite3.connect(history_store.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        assert connection.execute(_contract()["queries"]["run_stats"]).fetchone() == (300, 0, 3, 300, 1, 1, 1, 605)


def test_history_dataset_metadata_uses_first_stored_prediction(history_store: Path) -> None:
    with sqlite3.connect(history_store) as connection:
        connection.execute("UPDATE predictions SET n_samples=99, n_features=300")
        connection.execute("UPDATE predictions SET n_samples=37 WHERE prediction_id='prediction-0'")
        connection.commit()
    with sqlite3.connect(history_store.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        rows = connection.execute(_contract()["queries"]["run_dataset_metadata"], ("run-a",)).fetchall()
        assert rows[0] == ("corn", 37, 300, "regression", "rmse")
        assert rows[1][1] == 99


def test_status_filter_precedes_pagination_and_preserves_total_on_empty_page(history_store: Path) -> None:
    with sqlite3.connect(history_store) as connection:
        connection.executemany(
            "INSERT INTO runs(run_id,name,status,created_at,project_id) VALUES (?, 'New', 'completed', '2026-10-01', 'project-a')",
            [(f"new-{i}",) for i in range(600)],
        )
        connection.executemany(
            "INSERT INTO runs(run_id,name,status,created_at,project_id) VALUES (?, 'Old failure', 'failed', '2026-01-01', 'project-a')",
            [("failed-a",), ("failed-b",)],
        )
        connection.commit()
    queries = _contract()["queries"]
    with sqlite3.connect(history_store.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        assert connection.execute(queries["runs_filtered_page"], ("project-a", 1, 1, '["failed"]')).fetchone()[0] == "failed-b"
        assert connection.execute(queries["runs_filtered_total"], ("project-a", '["failed"]')).fetchone() == (2,)
        assert connection.execute(queries["runs_filtered_page"], ("project-a", 1, 2, '["failed"]')).fetchall() == []
        assert connection.execute(queries["runs_filtered_total"], ("project-a", '[]')).fetchone() == (604,)
        assert connection.execute(queries["runs_filtered_total"], (None, json.dumps(["' OR 1=1 --"]))).fetchone() == (0,)


def test_history_materializes_only_page_chains_and_reduces_exact_finite_historical_cohort(history_store: Path) -> None:
    with sqlite3.connect(history_store) as connection:
        connection.execute("UPDATE chains SET dataset_name='corn',metric='rmse',cv_val_score=0.5 WHERE chain_id='c'")
        connection.execute("INSERT INTO pipelines(pipeline_id,run_id,name,dataset_name) VALUES ('other','run-b','Historical','corn')")
        for chain, dataset, metric, score, prediction in [
            ("old-a", "corn", "rmse", 0.2, True), ("old-b", "corn", "rmse", 0.8, True),
            ("positive-inf", "corn", "rmse", float("inf"), True), ("negative-inf", "corn", "rmse", float("-inf"), True),
            ("different-metric", "corn", "mae", -9.0, True), ("different-data", "other", "rmse", -8.0, True),
            ("unpublished", "corn", "rmse", -7.0, False),
        ]:
            connection.execute(
                "INSERT INTO chains(chain_id,pipeline_id,steps,model_step_idx,model_class,dataset_name,metric,cv_val_score) VALUES (?, 'other','[]',1,'Ridge',?,?,?)",
                (chain, dataset, metric, score),
            )
            if prediction:
                connection.execute(
                    "INSERT INTO predictions(prediction_id,pipeline_id,chain_id,dataset_name,model_name,model_class,fold_id,partition,metric,task_type) VALUES (?, 'other', ?, ?, 'Ridge','Ridge','0','val',?,'regression')",
                    (chain, chain, dataset, metric),
                )
        connection.commit()
    queries = _contract()["queries"]
    with sqlite3.connect(history_store.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        page = connection.execute(queries["page_chains"], ('["run-a"]', 500, 0)).fetchall()
        assert [row[0] for row in page] == ["c"]
        assert connection.execute(queries["page_chains"], ('["run-a"]', 500, 500)).fetchall() == []
        assert connection.execute(queries["historical_score_extrema"], ("corn", "rmse", "run-a")).fetchone() == (0.2, 0.8)
        assert connection.execute(queries["historical_score_extrema"], ("corn", "rmse", "run-b")).fetchone() == (0.5, 0.5)
        assert connection.execute(queries["historical_score_extrema"], ("corn", "RMSE", "run-a")).fetchone() == (None, None)
