"""Run generated-view DAG-ML campaigns in an isolated Python interpreter."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cloudpickle

from .resources import current_execution_resources
from .worker_environment import scientific_worker_environment


def _relay_worker_progress(directory: Path, callback: Callable[[dict[str, Any]], Any]) -> None:
    """Answer one checkpoint callback from a child through its private directory."""
    event_path = directory / "progress.event"
    answer_path = directory / "progress.answer"
    if not event_path.is_file() or answer_path.exists():
        return
    try:
        with event_path.open("rb") as stream:
            event = cloudpickle.load(stream)  # noqa: S301 - private event from our child
    except FileNotFoundError:
        # The child can retire the preceding event after its answer is read.
        return
    if not isinstance(event, dict):
        raise ValueError("generated HPO worker sent an invalid progress event")
    response = callback(event)
    decision = True if response is None else bool(response)
    temporary = directory / "progress.answer.tmp"
    with temporary.open("wb") as stream:
        cloudpickle.dump(decision, stream)
    os.replace(temporary, answer_path)


def _run_cancellable_worker(
    command: list[str], should_stop: Any = None, *,
    progress_dir: Path | None = None, progress_callback: Callable[[dict[str, Any]], Any] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Observe the parent cancellation token while the isolated child executes."""
    from .cancellation import DagRunCancelled, check_cancellation

    check_cancellation()
    if should_stop is not None and should_stop():
        raise DagRunCancelled("DAG run cancelled by caller")
    # Candidate-local HPO workers inherit this process group on POSIX. Keep
    # them in the cancellation scope of the outer generated-view worker.
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
        env=scientific_worker_environment(command[0]),
    )
    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.1)
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                if progress_dir is not None and progress_callback is not None:
                    _relay_worker_progress(progress_dir, progress_callback)
                check_cancellation()
                if should_stop is not None and should_stop():
                    raise DagRunCancelled("DAG run cancelled by caller") from None
    except BaseException:
        if process.poll() is None:
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            elif os.name == "nt":
                # taskkill /T includes candidate grandchildren; terminate()
                # alone only reaches the outer worker on Windows.
                with contextlib.suppress(OSError):
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                else:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                process.communicate()
            finally:
                # communicate() only waits for the outer worker. Candidate
                # children have separate pipes and may ignore SIGTERM, so reap
                # the remaining group even when the worker exited promptly.
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
        raise


def run_generated_subprocess(
    *,
    dsl: dict[str, Any],
    envelope: dict[str, Any],
    graph: dict[str, Any],
    dataset: Any,
    dataset_path: str,
    dataset_pickle: str | None,
    workdir: Any,
    venv_python: str | None,
    selection_metric: str,
    sample_metadata: dict[str, dict[str, Any]] | None,
    random_state: int | None,
    refit: bool,
    refit_top_k: int,
) -> dict[str, Any]:
    """Transfer the PLAN provider and return the child's native CV/refit outcome.

    The live receipt store is never serialized. Only a trusted run-local request
    and response cross the process boundary; the child creates its own store.
    """
    store = getattr(dataset, "_generated_view_store", None)
    if store is None or dataset_pickle is None:
        raise ValueError("generated subprocess requires a live provider and a PLAN dataset pickle")
    provider = store.provider_for_worker()
    cohort = provider.cohort
    if getattr(cohort, "_generated_view_store", None) is not store:
        raise ValueError("generated subprocess provider lost its PLAN view-store link")
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="generated-worker-", dir=root) as private_dir:
        request_path = Path(private_dir) / "request.pkl"
        response_path = Path(private_dir) / "response.pkl"
        request = {
            "schema": "nirs4all.generated-worker.v1",
            "provider": provider,
            "dataset_path": dataset_path,
            "dataset_pickle": dataset_pickle,
            "dsl": dsl,
            "envelope": envelope,
            "graph": graph,
            "selection_metric": selection_metric,
            "sample_metadata": sample_metadata,
            "random_state": random_state,
            "refit": refit,
            "refit_top_k": refit_top_k,
            "resources": current_execution_resources(),
        }
        # The cohort also belongs to the public result. Its temporary parent
        # link must not drag the unserializable live store into the request.
        del cohort._generated_view_store
        try:
            with request_path.open("wb") as stream:
                cloudpickle.dump(request, stream)
        finally:
            cohort._generated_view_store = store
        command = [
            str(venv_python or sys.executable), "-P", "-s", "-B", "-m", "nirs4all.pipeline.dagml.generated_worker",
            str(request_path), str(response_path),
        ]
        completed = _run_cancellable_worker(command)
        if completed.returncode != 0:
            raise RuntimeError(
                f"generated DAG-ML worker failed with exit code {completed.returncode}:\n"
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        if not response_path.is_file():
            raise RuntimeError("generated DAG-ML worker did not return an outcome")
        with response_path.open("rb") as stream:
            outcome = cloudpickle.load(stream)  # noqa: S301 - private response from our child
    if not isinstance(outcome, dict) or outcome.get("returncode") != 0:
        raise ValueError("generated DAG-ML worker returned an invalid outcome")
    manifest = outcome.get("generated_view_manifest")
    if (not isinstance(manifest, dict) or not isinstance(outcome.get("scores"), dict)
            or not isinstance(outcome.get("results"), list) or not isinstance(outcome.get("refit_artifacts"), list)):
        raise ValueError("generated DAG-ML worker omitted native view or refit evidence")
    dataset._dagml_generated_view_manifest = manifest
    return outcome
