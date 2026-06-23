"""nirs4all process adapter — speaks the dag-ml JSONL frame protocol.

Wraps :func:`run_node` in the stdin/stdout control-frame loop that ``dag-ml-cli`` drives
(``--describe`` handshake, then ``init``/``task``/``close`` frames, or bare one-shot
tasks). Mirrors ``dag-ml/examples/adapters/sklearn_process_controller.py`` exactly, but the
node execution fetches real ``SpectroDataset`` rows + real operators instead of synthesizing.

Launched per worker by the coordinator. ``main`` loads the dataset and the compiled graph
from env (``N4A_DAGML_DATASET_PATH`` / ``N4A_DAGML_GRAPH_PATH``) — the coordinator passes
only ``sample_ids`` per task, so the adapter owns the ``sample_id → X/y`` materialization
(the resolver) and the ``node_id → operator`` lookup (the compiled graph). :func:`run_jsonl_loop`
is dependency-injected so it is testable in-process with no subprocess.

Note: the launcher must re-exec this under the project venv — ``dag-ml-cli`` runs a ``.py``
adapter under the *system* ``python3`` (which lacks nirs4all), so a non-``.py`` shim is used.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import IO, Any

from .node_runner import run_node

_FRAME_SCHEMA_VERSION = 1
_DESCRIPTION_SCHEMA_VERSION = 1
_PROTOCOL = "dag-ml-process-adapter"
_CAPABILITIES = ["control_frames_v1", "node_task_json_v1", "node_result_json_v1", "persistent_workers", "worker_env", "stateful_refit_artifacts"]

NodeHandler = Callable[[dict[str, Any]], dict[str, Any]]


def describe() -> dict[str, Any]:
    """The coordinator handshake description (modes + capabilities advertised)."""
    return {
        "schema_version": _DESCRIPTION_SCHEMA_VERSION,
        "protocol": _PROTOCOL,
        "adapter_id": "nirs4all-dagml-process-adapter",
        "supported_modes": ["one_shot", "jsonl"],
        "capabilities": sorted(_CAPABILITIES),
    }


def _emit(out: IO[str], payload: dict[str, Any]) -> None:
    json.dump(payload, out, sort_keys=True)
    out.write("\n")
    out.flush()


def _error(code: str, message: str) -> dict[str, Any]:
    return {"type": "error", "schema_version": _FRAME_SCHEMA_VERSION, "error": {"code": code, "message": message, "retryable": False}}


def run_jsonl_loop(infile: IO[str], outfile: IO[str], handle: NodeHandler) -> None:
    """Drive the control-frame loop: ``init``/``close`` acks, ``task`` results, bare one-shot tasks."""
    for raw in infile:
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit(outfile, _error("invalid_json", str(exc)))
            continue
        frame_type = payload.get("type") if isinstance(payload, dict) else None
        if frame_type is None:  # bare one-shot task
            _emit(outfile, handle(payload))
        elif frame_type == "init":
            _emit(outfile, {"type": "ack", "schema_version": _FRAME_SCHEMA_VERSION, "status": "initialized"})
        elif frame_type == "task":
            task = payload.get("task")
            if not isinstance(task, dict):
                _emit(outfile, _error("invalid_task_frame", "task frame is missing object field `task`"))
            else:
                _emit(outfile, {"type": "result", "schema_version": _FRAME_SCHEMA_VERSION, "result": handle(task)})
        elif frame_type == "close":
            _emit(outfile, {"type": "ack", "schema_version": _FRAME_SCHEMA_VERSION, "status": "closed"})
            return
        else:
            _emit(outfile, _error("unsupported_frame", f"unsupported frame type `{frame_type}`"))


def _build_handler() -> NodeHandler:
    """Construct the node handler from the dataset + compiled graph named in the env."""
    from nirs4all.data.config import DatasetConfigs

    from .identity import mint_identity
    from .resolver import MaterializationResolver

    dataset = DatasetConfigs(os.environ["N4A_DAGML_DATASET_PATH"]).get_dataset_at(0)
    resolver = MaterializationResolver(dataset, mint_identity(dataset))
    with open(os.environ["N4A_DAGML_GRAPH_PATH"], encoding="utf-8") as handle:
        nodes = {node["id"]: node for node in json.load(handle)["nodes"]}
    store: dict[str, Any] = {}
    return lambda task: run_node(task, resolver, nodes.__getitem__, store)


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "--describe":
        _emit(sys.stdout, describe())
        return 0
    run_jsonl_loop(sys.stdin, sys.stdout, _build_handler())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
