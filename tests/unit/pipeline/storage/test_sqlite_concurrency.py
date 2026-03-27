"""SQLite WAL concurrency tests.

Verify that SQLite WAL mode resolves the locking issues that motivated
the migration from DuckDB.  Tests exercise concurrent readers, writer-
does-not-block-reader behaviour, successive runs on the same workspace,
and reader-after-writer-closes patterns.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


def _make_store(tmp_path: Path) -> WorkspaceStore:
    return WorkspaceStore(tmp_path / "workspace")


def _seed_run(store: WorkspaceStore, suffix: str = "") -> str:
    """Insert a run and return its run_id."""
    run_id = store.begin_run(f"run{suffix}", config={}, datasets=[{"name": "ds"}])
    store.complete_run(run_id, summary={"score": 0.5})
    return run_id


class TestSQLiteConcurrency:
    """Verify that SQLite WAL resolves the locking issues that motivated the migration."""

    def test_concurrent_readers(self, tmp_path: Path) -> None:
        """Multiple WorkspaceStore instances can read simultaneously."""
        workspace = tmp_path / "workspace"
        store = WorkspaceStore(workspace)
        run_id = _seed_run(store)
        store.close()

        errors: list[str] = []

        def reader(idx: int) -> None:
            try:
                s = WorkspaceStore(workspace)
                run = s.get_run(run_id)
                if run is None:
                    errors.append(f"reader-{idx}: run not found")
                s.close()
            except Exception as exc:
                errors.append(f"reader-{idx}: {exc}")

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Concurrent reader errors: {errors}"

    def test_writer_does_not_block_readers(self, tmp_path: Path) -> None:
        """A write transaction does not block concurrent reads."""
        workspace = tmp_path / "workspace"
        store = WorkspaceStore(workspace)
        _seed_run(store)
        store.close()

        read_ok = threading.Event()
        write_done = threading.Event()
        errors: list[str] = []

        def writer() -> None:
            try:
                s = WorkspaceStore(workspace)
                for i in range(10):
                    _seed_run(s, suffix=f"_w{i}")
                s.close()
                write_done.set()
            except Exception as exc:
                errors.append(f"writer: {exc}")
                write_done.set()

        def reader() -> None:
            try:
                s = WorkspaceStore(workspace)
                df = s.list_runs()
                assert len(df) >= 1
                s.close()
                read_ok.set()
            except Exception as exc:
                errors.append(f"reader: {exc}")
                read_ok.set()

        wt = threading.Thread(target=writer)
        rt = threading.Thread(target=reader)
        wt.start()
        rt.start()
        wt.join(timeout=15)
        rt.join(timeout=15)

        assert errors == [], f"Concurrent read/write errors: {errors}"
        assert read_ok.is_set(), "Reader should have completed"

    def test_successive_runs_same_workspace(self, tmp_path: Path) -> None:
        """Two sequential WorkspaceStore open/close cycles on the same workspace succeed."""
        workspace = tmp_path / "workspace"

        # First open/close cycle
        store1 = WorkspaceStore(workspace)
        run_id1 = _seed_run(store1, suffix="_1")
        store1.close()

        # Second open/close cycle
        store2 = WorkspaceStore(workspace)
        run_id2 = _seed_run(store2, suffix="_2")

        # Both runs visible
        assert store2.get_run(run_id1) is not None
        assert store2.get_run(run_id2) is not None
        df = store2.list_runs()
        assert len(df) == 2
        store2.close()

    def test_reader_after_writer_closes(self, tmp_path: Path) -> None:
        """A reader can open the store immediately after a writer closes."""
        workspace = tmp_path / "workspace"

        store_w = WorkspaceStore(workspace)
        run_id = _seed_run(store_w)
        store_w.close()

        # Immediately open a new store and read
        store_r = WorkspaceStore(workspace)
        run = store_r.get_run(run_id)
        assert run is not None
        assert run["name"] == "run"
        store_r.close()

    def test_multiple_writers_sequential(self, tmp_path: Path) -> None:
        """Sequential writers on the same workspace do not conflict."""
        workspace = tmp_path / "workspace"
        run_ids = []

        for i in range(5):
            store = WorkspaceStore(workspace)
            rid = _seed_run(store, suffix=f"_{i}")
            run_ids.append(rid)
            store.close()

        # Final read
        store = WorkspaceStore(workspace)
        df = store.list_runs()
        assert len(df) == 5
        for rid in run_ids:
            assert store.get_run(rid) is not None
        store.close()
