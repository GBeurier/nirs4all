"""Isolated Python worker for generated-view HPO and its selected refit."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import cloudpickle


def _relay_progress(event: dict[str, Any], directory: Path) -> bool:
    """Wait for the caller's decision before advancing the native search."""
    event_path = directory / "progress.event"
    answer_path = directory / "progress.answer"
    temporary = directory / "progress.event.tmp"
    with temporary.open("wb") as stream:
        cloudpickle.dump(event, stream)
    os.replace(temporary, event_path)
    while not answer_path.is_file():
        time.sleep(0.05)
    with answer_path.open("rb") as stream:
        decision = cloudpickle.load(stream)  # noqa: S301 - private answer from our parent
    event_path.unlink()
    answer_path.unlink()
    if type(decision) is not bool:
        raise ValueError("generated HPO parent sent an invalid progress decision")
    return decision


def execute(request: dict[str, Any]) -> Any:
    """Build a worker-local view store and run HPO without a nested subprocess."""
    from .generated_views import GeneratedViewStore
    from .multimodal_tuning import run_multimodal_tuning
    from .resources import bind_execution_resources, reset_execution_resources

    provider = request["provider"]
    cohort = provider.cohort
    if getattr(cohort, "_generated_view_store", None) is not None:
        raise ValueError("generated HPO worker PLAN cohort unexpectedly contains a live view store")
    cohort._generated_view_store = GeneratedViewStore(provider)
    previous_mode = os.environ.get("N4A_DAGML_INPROCESS")
    os.environ["N4A_DAGML_INPROCESS"] = "1"
    token = bind_execution_resources(request["resources"])
    try:
        tuning = request["tuning"]
        if request.get("progress_dir") is not None:
            directory = Path(request["progress_dir"])
            tuning = {**tuning, "progress_callback": lambda event: _relay_progress(event, directory)}
        return run_multimodal_tuning(
            request["pipeline"], cohort, tuning, run_options=request["run_options"],
        )
    finally:
        reset_execution_resources(token)
        if previous_mode is None:
            os.environ.pop("N4A_DAGML_INPROCESS", None)
        else:
            os.environ["N4A_DAGML_INPROCESS"] = previous_mode


def main() -> int:
    """Read one private request and publish one complete result atomically."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m nirs4all.pipeline.dagml.generated_hpo_worker REQUEST RESPONSE")
    request_path, response_path = Path(sys.argv[1]), Path(sys.argv[2])
    with request_path.open("rb") as stream:
        request = cloudpickle.load(stream)  # noqa: S301 - trusted parent-owned private request
    if (not isinstance(request, dict)
            or request.get("schema") not in {"nirs4all.generated-hpo-worker.v1", "nirs4all.generated-hpo-worker.v2"}
            or (request["schema"] == "nirs4all.generated-hpo-worker.v2" and not request.get("progress_dir"))):
        raise ValueError("generated HPO worker received an invalid request")
    from .multimodal_tuning import MultimodalTuningStopped

    try:
        result = execute(request)
    except MultimodalTuningStopped as exc:
        result = {"status": "stopped", "evidence": exc.evidence}
    temporary = response_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        cloudpickle.dump(result, stream)
    os.replace(temporary, response_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
