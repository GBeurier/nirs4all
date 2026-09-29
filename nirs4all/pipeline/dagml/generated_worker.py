"""Isolated Python worker for native generated-view CV/refit campaigns.

The request is a trusted, run-local cloudpickle payload. The PLAN dataset is
loaded separately from the existing host pickle; only the PLAN-complete IO
provider crosses into this process. A fresh GeneratedViewStore owns all view
handles and buffers here, alongside the native scheduler and node callback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cloudpickle


def execute(request: dict[str, Any]) -> dict[str, Any]:
    """Run one campaign with a worker-local view store and resource context."""
    from nirs4all.pipeline.runner import init_global_random_state

    from .generated_views import GeneratedViewStore
    from .in_process_runner import _load_dataset, run_cv_refit_bundle
    from .resources import bind_execution_resources, reset_execution_resources

    dataset, fold_children, fold_feature_views = _load_dataset(
        request["dataset_path"], request["dataset_pickle"],
    )
    if getattr(dataset, "_generated_view_store", None) is not None:
        raise ValueError("generated worker PLAN dataset unexpectedly contains a live view store")
    if request["random_state"] is not None:
        init_global_random_state(request["random_state"])
    dataset._generated_view_store = GeneratedViewStore(request["provider"])
    token = bind_execution_resources(request["resources"])
    try:
        return run_cv_refit_bundle(
            dsl=request["dsl"], envelope=request["envelope"], graph=request["graph"],
            dataset_path=request["dataset_path"], dataset_pickle=request["dataset_pickle"],
            dataset=dataset, fold_children=fold_children, fold_feature_views=fold_feature_views,
            selection_metric=request["selection_metric"], sample_metadata=request["sample_metadata"],
            random_state=request["random_state"], refit=request["refit"],
            refit_top_k=request["refit_top_k"],
        )
    finally:
        reset_execution_resources(token)


def main() -> int:
    """Read one private request and publish one complete response atomically."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m nirs4all.pipeline.dagml.generated_worker REQUEST RESPONSE")
    request_path, response_path = Path(sys.argv[1]), Path(sys.argv[2])
    with request_path.open("rb") as stream:
        request = cloudpickle.load(stream)  # noqa: S301 - trusted parent-owned private request
    if not isinstance(request, dict) or request.get("schema") != "nirs4all.generated-worker.v1":
        raise ValueError("generated worker received an invalid request")
    outcome = execute(request)
    temporary = response_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        cloudpickle.dump(outcome, stream)
    os.replace(temporary, response_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
