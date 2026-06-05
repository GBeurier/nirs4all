"""ArrayStore inter-process advisory lock tests (debt handoff item #1, step 3b).

Every ArrayStore mutation is a whole-file read-modify-write; without a lock, two
processes mutating the same workspace silently lose one side's writes
(last-writer-wins). Mutations now serialize on an exclusive ``arrays/.lock``
(``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows, no-op elsewhere). Each
acquisition opens its own file descriptor, so the blocking behaviour is observable
in-process with two concurrent holders — which is what these tests exercise (true
multi-process coverage would test the same flock semantics through more machinery).
"""

from __future__ import annotations

import threading
import time

import pytest

from nirs4all.pipeline.storage import array_store as array_store_module
from nirs4all.pipeline.storage.array_store import ArrayStore
from tests.unit.pipeline.storage.test_array_store import _make_record

needs_real_lock = pytest.mark.skipif(
    array_store_module.fcntl is None and array_store_module.msvcrt is None,
    reason="no locking primitive on this platform (lock is a documented no-op)",
)

class TestProcessLock:
    @needs_real_lock
    def test_mutation_blocks_while_lock_held(self, tmp_path):
        """A mutation started while another holder owns the lock waits for release."""
        store = ArrayStore(tmp_path)
        done = threading.Event()
        started = threading.Event()

        def blocked_mutation():
            started.set()
            store.delete_batch({"someone"})  # acquires the lock internally
            done.set()

        with store._process_lock():
            t = threading.Thread(target=blocked_mutation, daemon=True)
            t.start()
            assert started.wait(timeout=5)
            # The mutation must NOT complete while we hold the lock.
            assert not done.wait(timeout=0.4)
        # Released: the mutation proceeds promptly.
        assert done.wait(timeout=5)
        t.join(timeout=5)
        assert "someone" in store._read_tombstones()

    def test_sequential_mutations_do_not_deadlock(self, tmp_path):
        """save_batch -> delete_batch -> compact in sequence (non-reentrant lock,
        mutations never nest)."""
        store = ArrayStore(tmp_path)
        store.save_batch([_make_record("p1")])
        store.delete_batch({"p1"})
        stats = store.compact()
        assert stats["wheat"]["rows_removed"] == 1
        assert store.load_batch(["p1"]) == {}
