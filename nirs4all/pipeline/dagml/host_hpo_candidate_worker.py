"""Private callback loop for one parallel host HPO candidate."""

from __future__ import annotations

import pickle
import sys
import traceback
from pathlib import Path
from typing import Any

import cloudpickle


def execute(request: dict[str, Any], input_stream: Any, output_stream: Any) -> None:
    """Own one provider copy and all fitted candidate state until a stop frame."""
    from nirs4all.pipeline.runner import init_global_random_state

    from .generated_views import GeneratedViewStore
    from .multimodal_tuning import _evaluate_host_task
    from .resolver import MaterializationResolver
    from .resources import bind_execution_resources, reset_execution_resources

    provider = request["provider"]
    if provider is not None and getattr(provider.cohort, "_generated_view_store", None) is not None:
        raise ValueError("host HPO candidate provider contains a parent view store")
    store = GeneratedViewStore(provider) if provider is not None else None
    graph = request["graph"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    resolver = MaterializationResolver(request["dataset"], request["identity"])
    model_store: dict[Any, Any] = {}
    init_global_random_state(request["operator_seed"])
    token = bind_execution_resources(request["resources"])
    try:
        while True:
            try:
                message = pickle.load(input_stream)  # noqa: S301 - private parent pipe
            except EOFError:
                return
            if not isinstance(message, dict):
                raise ValueError("generated HPO candidate received an invalid callback frame")
            kind = message.get("kind")
            if kind == "stop":
                return
            try:
                if kind == "view":
                    if store is None:
                        raise ValueError("static HPO candidate cannot materialize a generated view")
                    result = store(message["payload"])
                elif kind == "operator":
                    task = message["payload"]
                    if store is not None and not task.get("data_view_receipts"):
                        raise ValueError("generated HPO candidate has no native data-view receipts")
                    result = _evaluate_host_task(
                        task, resolver=resolver, nodes=nodes, graph=graph,
                        model_store=model_store, view_store=store,
                        operator_seed=request["operator_seed"],
                    )
                else:
                    raise ValueError("generated HPO candidate received an unknown callback kind")
                response = {"ok": True, "result": result}
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
            pickle.dump(response, output_stream)
            output_stream.flush()
    finally:
        reset_execution_resources(token)


def main() -> int:
    """Read a trusted run-local request and reserve stdout for callback frames."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m nirs4all.pipeline.dagml.host_hpo_candidate_worker REQUEST")
    protocol_output = sys.stdout.buffer
    sys.stdout = sys.stderr
    with Path(sys.argv[1]).open("rb") as stream:
        request = cloudpickle.load(stream)  # noqa: S301 - private parent-owned request
    if not isinstance(request, dict) or request.get("schema") != "nirs4all.host-hpo-candidate.v1":
        raise ValueError("host HPO candidate received an invalid request")
    execute(request, sys.stdin.buffer, protocol_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
