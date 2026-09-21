"""Standalone installed-wheel regression for atomic workspace schema initialization.

Only stdlib and pytest fixtures are required; CI can extract this file outside
its source checkout to exercise the installed nirs4all wheel.
"""
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest


def _seed_database(path):
    from nirs4all.pipeline.storage.store_schema import create_schema
    with closing(sqlite3.connect(path, isolation_level=None)) as connection:
        create_schema(connection)


def test_current_schema_open_neither_writes_nor_waits_for_an_active_writer(tmp_path):
    from nirs4all.pipeline.storage.store_schema import create_schema
    path = tmp_path / "store.sqlite"
    _seed_database(path)
    writer = sqlite3.connect(path, isolation_level=None)
    reader = sqlite3.connect(path, isolation_level=None, timeout=0)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO runs(run_id, name) VALUES ('active', 'active writer')")
        schema_before = reader.execute("PRAGMA schema_version").fetchone()[0]
        statements = []
        reader.set_trace_callback(statements.append)
        create_schema(reader)
        assert not reader.in_transaction
        assert reader.execute("PRAGMA schema_version").fetchone()[0] == schema_before
        assert all(sql.lstrip().upper().startswith(("SELECT", "PRAGMA")) for sql in statements)
        assert not any("USER_VERSION =" in sql.upper() for sql in statements)
        assert reader.execute("SELECT COUNT(*) FROM v_chain_summary").fetchone()[0] == 0
    finally:
        writer.rollback()
        writer.close()
        reader.close()


def test_view_replacement_is_invisible_to_concurrent_reader(tmp_path):
    from nirs4all.pipeline.storage.store_schema import create_schema
    path = tmp_path / "store.sqlite"
    _seed_database(path)
    with closing(sqlite3.connect(path, isolation_level=None)) as setup:
        setup.execute("DROP VIEW v_chain_summary")
        setup.execute("CREATE VIEW v_chain_summary AS SELECT 7 AS previous_definition")
    dropped = threading.Event()
    proceed = threading.Event()

    class PausedConnection(sqlite3.Connection):
        def execute(self, sql, parameters=(), /):
            result = super().execute(sql, parameters)
            if sql == "DROP VIEW IF EXISTS v_chain_summary":
                dropped.set()
                if not proceed.wait(10):
                    raise TimeoutError("Reader did not release migration")
            return result

    def upgrade():
        with closing(sqlite3.connect(path, isolation_level=None, factory=PausedConnection)) as connection:
            create_schema(connection)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(upgrade)
        try:
            assert dropped.wait(10), "Migration did not reach view replacement"
            with closing(sqlite3.connect(path, isolation_level=None, timeout=0)) as reader:
                # The writer really executed DROP, but its change is uncommitted.
                assert reader.execute("SELECT previous_definition FROM v_chain_summary").fetchone()[0] == 7
        finally:
            proceed.set()
        future.result(timeout=10)
    with closing(sqlite3.connect(path)) as reader:
        assert reader.execute("SELECT COUNT(*) FROM v_chain_summary").fetchone()[0] == 0


def test_failed_view_upgrade_rolls_back_the_previous_definition(tmp_path):
    from nirs4all.pipeline.storage.store_schema import create_schema
    path = tmp_path / "store.sqlite"
    _seed_database(path)
    with closing(sqlite3.connect(path, isolation_level=None)) as setup:
        setup.execute("DROP VIEW v_chain_summary")
        setup.execute("CREATE VIEW v_chain_summary AS SELECT 7 AS previous_definition")

    class FailingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=(), /):
            if sql.startswith("CREATE VIEW IF NOT EXISTS v_chain_summary"):
                raise sqlite3.OperationalError("injected migration failure")
            return super().execute(sql, parameters)

    with closing(sqlite3.connect(path, isolation_level=None, factory=FailingConnection)) as connection:
        with pytest.raises(sqlite3.OperationalError, match="injected migration failure"):
            create_schema(connection)
        assert not connection.in_transaction
    with closing(sqlite3.connect(path)) as reader:
        assert reader.execute("SELECT previous_definition FROM v_chain_summary").fetchone()[0] == 7


@pytest.mark.parametrize("journal_mode", ["wal", "delete"])
def test_schema_upgrade_preserves_callers_uncommitted_transaction(tmp_path, journal_mode):
    from nirs4all.pipeline.storage.store_schema import SCHEMA_VERSION, create_schema
    path = tmp_path / "store.sqlite"
    _seed_database(path)
    with closing(sqlite3.connect(path, isolation_level=None)) as caller:
        caller.execute(f"PRAGMA journal_mode={journal_mode}")
        caller.execute("BEGIN")
        caller.execute("INSERT INTO runs(run_id, name) VALUES ('uncommitted', 'caller')")
        caller.execute(f"PRAGMA user_version = {SCHEMA_VERSION - 1}")
        create_schema(caller)
        assert caller.in_transaction
        with closing(sqlite3.connect(path)) as observer:
            assert observer.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
        caller.rollback()
        assert caller.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
        assert caller.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_parallel_workspace_initializers_keep_chain_summary_readable(tmp_path):
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
    with WorkspaceStore(tmp_path):
        pass
    barrier = threading.Barrier(4)

    def open_and_query():
        barrier.wait(timeout=10)
        for _ in range(8):
            with WorkspaceStore(tmp_path) as store:
                assert store.query_chain_summaries().is_empty()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(open_and_query) for _ in range(4)]
        for future in futures:
            future.result(timeout=20)


def test_current_version_with_missing_view_is_repaired(tmp_path):
    from nirs4all.pipeline.storage.store_schema import create_schema
    path = tmp_path / "store.sqlite"
    _seed_database(path)
    with closing(sqlite3.connect(path, isolation_level=None)) as connection:
        connection.execute("DROP VIEW v_chain_summary")
        create_schema(connection)
        assert connection.execute("SELECT COUNT(*) FROM v_chain_summary").fetchone()[0] == 0
