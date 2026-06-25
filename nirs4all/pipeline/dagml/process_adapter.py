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
from typing import IO, Any, Protocol

from .node_runner import run_node


class _Writer(Protocol):
    def write(self, text: str, /) -> int: ...
    def flush(self) -> None: ...

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


def _emit(out: _Writer, payload: dict[str, Any]) -> None:
    json.dump(payload, out, sort_keys=True)
    out.write("\n")
    out.flush()


def _error(code: str, message: str) -> dict[str, Any]:
    return {"type": "error", "schema_version": _FRAME_SCHEMA_VERSION, "error": {"code": code, "message": message, "retryable": False}}


def run_jsonl_loop(infile: IO[str], outfile: _Writer, handle: NodeHandler) -> None:
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
    """Construct the node handler from the dataset + compiled graph named in the env.

    A ``sample_augmentation`` run pickles the AUGMENTED ``SpectroDataset`` (the synthetic rows are
    stochastic and not reproducible across processes, so re-running augmentation here would diverge);
    ``N4A_DAGML_DATASET_PICKLE`` then points the adapter at that exact materialized dataset so its
    ``mint_identity`` + resolver match the wire ids the DSL/envelope/fold-set were built from. Without
    augmentation the plain path-load is used.
    """
    import pickle

    from nirs4all.data.config import DatasetConfigs

    from .identity import mint_identity
    from .resolver import MaterializationResolver

    pickle_path = os.environ.get("N4A_DAGML_DATASET_PICKLE")
    if pickle_path:
        with open(pickle_path, "rb") as handle:
            dataset = pickle.load(handle)  # noqa: S301 - host-written augmented dataset for this run
    else:
        dataset = DatasetConfigs(os.environ["N4A_DAGML_DATASET_PATH"]).get_dataset_at(0)
    resolver = MaterializationResolver(dataset, mint_identity(dataset))
    with open(os.environ["N4A_DAGML_GRAPH_PATH"], encoding="utf-8") as handle:
        graph = json.load(handle)
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = graph.get("edges", [])
    # Linear-pipeline y_processing: a single floating y_transform node applies to the model.
    y_transform_node = next((node for node in graph["nodes"] if node["kind"] == "y_transform"), None)
    # Optional sample_id -> metadata map: lets a separation-branch model honor its branch_view
    # selector (the runtime delivers the full fold + the selector; the adapter applies it). Absent
    # for non-branch pipelines, where every NodeTask data view carries no branch_view (a no-op).
    sample_metadata = None
    meta_path = os.environ.get("N4A_DAGML_SAMPLE_META_PATH")
    if meta_path:
        with open(meta_path, encoding="utf-8") as handle:
            sample_metadata = json.load(handle)
    store: dict[int, Any] = {}
    return lambda task: run_node(task, resolver, nodes.__getitem__, store, edges, y_transform_node, sample_metadata)


class _Tee:
    """Write to two text streams (stdout for the coordinator + a robust capture file)."""

    def __init__(self, primary: IO[str], secondary: IO[str]) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, text: str) -> int:
        self._secondary.write(text)
        return self._primary.write(text)

    def flush(self) -> None:
        self._secondary.flush()
        self._primary.flush()


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "--describe":
        _emit(sys.stdout, describe())
        return 0
    capture = os.environ.get("N4A_DAGML_RESULT_CAPTURE")
    if capture:
        # Capture frames from inside the (single persistent) adapter — robust vs `tee` pipe
        # buffering that can split a large result frame across reads.
        with open(capture, "a", encoding="utf-8") as capture_file:
            run_jsonl_loop(sys.stdin, _Tee(sys.stdout, capture_file), _build_handler())
    else:
        run_jsonl_loop(sys.stdin, sys.stdout, _build_handler())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
