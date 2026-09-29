"""Isolated Python worker for generated-view HPO and its selected refit."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cloudpickle


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
        return run_multimodal_tuning(
            request["pipeline"], cohort, request["tuning"], run_options=request["run_options"],
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
    if not isinstance(request, dict) or request.get("schema") != "nirs4all.generated-hpo-worker.v1":
        raise ValueError("generated HPO worker received an invalid request")
    result = execute(request)
    temporary = response_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        cloudpickle.dump(result, stream)
    os.replace(temporary, response_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
