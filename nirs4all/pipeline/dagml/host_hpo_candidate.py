"""Candidate-local Python process for host HPO callbacks and RNG isolation."""

from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, cast

import cloudpickle

from .resources import current_execution_resources

_PROVIDER_SNAPSHOT_LOCK = threading.RLock()


class HostHpoCandidate:
    """Keep one candidate's provider, view handles, models and RNG in one child."""

    def __init__(
        self, index: int, *, provider: Any, dataset: Any, identity: Any,
        graph: dict[str, Any], operator_seed: int,
    ) -> None:
        self.index = index
        self._io_lock = threading.Lock()
        self._closed = False
        self._private_dir = tempfile.TemporaryDirectory(prefix="host-hpo-candidate-")
        root = Path(self._private_dir.name)
        request_path = root / "request.pkl"
        self._log_path = root / "worker.log"
        try:
            # Multiple Rust callbacks can create candidates at the same time.
            # The PLAN cohort is shared, so detaching its live store and taking
            # each cloudpickle snapshot must be one indivisible operation.
            with _PROVIDER_SNAPSHOT_LOCK:
                cohort = provider.cohort if provider is not None else None
                parent_store = getattr(cohort, "_generated_view_store", None)
                if parent_store is not None and cohort is not None:
                    del cohort._generated_view_store
                try:
                    with request_path.open("wb") as stream:
                        cloudpickle.dump({
                            "schema": "nirs4all.host-hpo-candidate.v1",
                            "provider": provider, "dataset": dataset, "identity": identity,
                            "graph": graph, "operator_seed": operator_seed,
                            "resources": current_execution_resources(),
                        }, stream)
                finally:
                    if parent_store is not None and cohort is not None:
                        cohort._generated_view_store = parent_store
        except BaseException:
            self._private_dir.cleanup()
            raise
        self._log_stream = self._log_path.open("w+b")
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-m", "nirs4all.pipeline.dagml.host_hpo_candidate_worker", str(request_path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log_stream,
            )
        except BaseException:
            self._log_stream.close()
            self._private_dir.cleanup()
            raise

    def call(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Exchange one view or operator callback with this candidate's child."""
        with self._io_lock:
            if self._closed:
                raise RuntimeError(f"host HPO candidate {self.index} is closed")
            if self._process.poll() is not None:
                raise RuntimeError(self._failure("exited before callback"))
            assert self._process.stdin is not None and self._process.stdout is not None
            try:
                pickle.dump({"kind": kind, "payload": payload}, self._process.stdin)
                self._process.stdin.flush()
                response = pickle.load(self._process.stdout)  # noqa: S301 - private child pipe
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RuntimeError(self._failure("lost its callback pipe")) from exc
            if not isinstance(response, dict) or type(response.get("ok")) is not bool:
                raise ValueError("host HPO candidate worker returned an invalid callback response")
            if not response["ok"]:
                raise RuntimeError(
                    f"host HPO candidate {self.index} {kind} failed: "
                    f"{response.get('error_type')}: {response.get('error')}"
                )
            if not isinstance(response.get("result"), dict):
                raise ValueError("host HPO candidate worker returned a non-mapping result")
            return cast(dict[str, Any], response["result"])

    def _failure(self, reason: str) -> str:
        log = self._log_path.read_text(encoding="utf-8", errors="replace") if self._log_path.exists() else ""
        return f"host HPO candidate {self.index} {reason}: {log[-2000:].strip()}"

    def close(self) -> None:
        """Reap the candidate process and remove its private request and log."""
        with self._io_lock:
            if self._closed:
                return
            self._closed = True
            if self._process.poll() is None:
                assert self._process.stdin is not None
                try:
                    pickle.dump({"kind": "stop"}, self._process.stdin)
                    self._process.stdin.flush()
                    self._process.wait(timeout=5)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait()
            if self._process.stdin is not None:
                self._process.stdin.close()
            if self._process.stdout is not None:
                self._process.stdout.close()
            self._log_stream.close()
            self._private_dir.cleanup()
