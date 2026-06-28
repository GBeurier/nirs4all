"""In-process dag-ml execution (Mechanism B) — the perf-parity twin of :mod:`.cli_runner`.

``cli_runner.run_cv_refit_bundle`` drives ``dag-ml-cli`` as a SUBPROCESS: it writes the DSL /
envelope / graph to disk, launches the process adapter, and reads back ``bundle.json``. This module
runs the SAME campaign IN-PROCESS through the PyO3 bridge (``dag_ml._dag_ml.run_cv_refit_in_process``)
— no subprocess, no per-call import tax. The bridge owns the data path (the SAME envelope-based Rust
``InMemoryDataProvider`` the CLI uses, so the per-fold ``data_views``/``sample_ids`` are produced
IDENTICALLY); only operator execution crosses back to Python, through an ``op_callback`` that closes
over the SAME :func:`node_runner.run_node` the subprocess adapter calls.

The returned ``{node_results, scores}`` carries the native ``ScoreSet`` (cross-fold OOF average +
the REFIT final/test reports) — byte-identical to the subprocess ``bundle.scores`` — so the caller
maps it into a ``RunResult`` with the SAME score-extraction (:func:`result._scores_to_run_result`).

Parity contract: in-process scores == subprocess scores == legacy, for any pipeline the subprocess
path supports. The dataset / fold-children / sample-metadata are loaded with the EXACT same logic the
subprocess adapter uses (:func:`process_adapter._build_handler`), so identity (and therefore the wire
ids) match — the only divergence is that the materialization happens in THIS process, not a child.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

from nirs4all.pipeline.dagml_bridge import controller_manifests

from .identity import mint_identity
from .node_runner import run_node
from .resolver import MaterializationResolver


def in_process_enabled() -> bool:
    """True when the in-process mechanism (Mechanism B) is opted in via ``N4A_DAGML_INPROCESS=1``.

    Default OFF: the router keeps the subprocess :mod:`.cli_runner` path (Mechanism A) until the
    in-process path is proven at parity across the full shape matrix, so nothing changes by default.
    """
    return os.environ.get("N4A_DAGML_INPROCESS") == "1"


def _load_dataset(dataset_path: str, dataset_pickle: str | None) -> tuple[Any, dict[str, dict[int, list[int]]] | None]:
    """Load the dataset + optional fold-local children, mirroring :func:`process_adapter._build_handler`.

    A ``dataset_pickle`` (augmentation runs) is preferred over the reloadable path: a dict payload
    carries ``{"dataset": ..., "fold_children": ...}`` (fold-local augmentation), a bare pickle is the
    dataset alone; without a pickle the dataset is reloaded from the path. Identical to the subprocess
    child's load, so ``mint_identity`` yields the same wire ids the DSL / envelope / fold-set use.
    """
    if dataset_pickle:
        with open(dataset_pickle, "rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 - host-written dataset for this run
        if isinstance(payload, dict):
            return payload["dataset"], payload.get("fold_children")
        return payload, None
    from nirs4all.data.config import DatasetConfigs

    return DatasetConfigs(dataset_path).get_dataset_at(0), None


def run_cv_refit_bundle(
    *,
    dsl: dict[str, Any],
    envelope: dict[str, Any],
    graph: dict[str, Any],
    dataset_path: str,
    selection_metric: str = "rmse",
    sample_metadata: dict[str, dict[str, Any]] | None = None,
    dataset_pickle: str | None = None,
    dataset: Any | None = None,
    fold_children: dict[str, dict[int, list[int]]] | None = None,
) -> dict[str, Any]:
    """Run a CV+refit bundle IN-PROCESS; return ``{returncode, stdout, results, scores}``.

    Drop-in for :func:`cli_runner.run_cv_refit_bundle` (same keyword signature minus the subprocess-only
    ``workdir`` / ``dagml_cli`` / ``venv_python``): the router can dispatch to either by the env flag.

    * ``scores`` — the native ``ScoreSet`` (``{reports: [...]}``), byte-identical to the subprocess
      ``bundle.scores`` the CLI emits; mapped by :func:`result._scores_to_run_result`.
    * ``results`` — the per-node ``NodeResult`` frames (FIT_CV validation/OOF + REFIT predictions),
      matching the subprocess ``results.jsonl`` capture.
    * ``returncode`` / ``stdout`` — always ``0`` / ``""`` on success; a bridge refusal raises
      (``DagMlError``) instead of returning a non-zero code, so the caller's ``returncode != 0`` guard
      is a no-op here (success-path parity).

    ``dataset`` is the host's ALREADY-MATERIALIZED ``SpectroDataset`` (with ``fold_children`` for a
    fold-local augmentation run): when given, the resolver is built from it directly — no disk reload.
    ``run_via_dagml`` already materialized this exact dataset (its identity fingerprint equals the
    reloadable path's, verified in :func:`dataset._dataset_inputs`) and, for augmentation / rep-fusion,
    it IS the object that was pickled for the subprocess — so the resolver is byte-identical to the
    reload/pickle path. When ``dataset`` is ``None`` the dataset is loaded per :func:`_load_dataset`.
    """
    import importlib

    # The compiled PyO3 extension `dag_ml._dag_ml` is not re-exported by the package facade and
    # ships no stub, so import it dynamically (the facade only wraps the control-plane JSON fns).
    dag_ml_ext = importlib.import_module("dag_ml._dag_ml")

    if dataset is None:
        dataset, fold_children = _load_dataset(dataset_path, dataset_pickle)
    if sample_metadata is None:
        meta_path = os.environ.get("N4A_DAGML_SAMPLE_META_PATH")
        if meta_path and Path(meta_path).exists():
            sample_metadata = json.loads(Path(meta_path).read_text(encoding="utf-8"))

    # The op_callback IS process_adapter._build_handler's lambda — the SAME run_node over the SAME
    # resolver/nodes/edges/y_transform/store, so operators execute identically to the subprocess.
    resolver = MaterializationResolver(dataset, mint_identity(dataset), fold_children)
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = graph.get("edges", [])
    y_transform_node = next((node for node in graph["nodes"] if node["kind"] == "y_transform"), None)
    store: dict[int, Any] = {}
    op_callback = lambda task: run_node(task, resolver, nodes.__getitem__, store, edges, y_transform_node, sample_metadata)  # noqa: E731

    payload = json.loads(
        dag_ml_ext.run_cv_refit_in_process(
            json.dumps(dsl),
            json.dumps(envelope),
            json.dumps(controller_manifests()),
            op_callback,
            selection_metric,
        )
    )
    return {
        "returncode": 0,
        "stdout": "",
        "results": payload.get("node_results", []),
        "scores": payload.get("scores"),
    }


def run_cv_refit_bundle_router(
    *,
    dsl: dict[str, Any],
    envelope: dict[str, Any],
    graph: dict[str, Any],
    dataset_path: str,
    workdir: Any,
    dagml_cli: str,
    venv_python: str,
    selection_metric: str = "rmse",
    sample_metadata: dict[str, dict[str, Any]] | None = None,
    dataset_pickle: str | None = None,
    dataset: Any | None = None,
    fold_children: dict[str, dict[int, list[int]]] | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Route a CV+refit bundle run to the in-process (Mechanism B) or subprocess (Mechanism A) runner.

    Both branches return the SAME outcome shape — ``{returncode, stdout, results, scores}`` — so the
    ``_run_*`` call sites read ``outcome["scores"]`` uniformly (no ``bundle.json`` re-read). The subprocess
    branch (the default, Mechanism A) is untouched: it drives :func:`cli_runner.run_cv_refit_bundle` and
    lifts ``scores`` from the ``bundle.json`` the CLI wrote. The in-process branch (``N4A_DAGML_INPROCESS=1``)
    drives :func:`run_cv_refit_bundle` here and gets ``scores`` directly from the bridge.

    Keeping ``workdir`` / ``dagml_cli`` / ``venv_python`` in the signature lets the call sites pass the
    SAME kwargs to either path; the in-process branch ignores those subprocess-only inputs.

    ``dataset`` (the host's already-materialized ``SpectroDataset``, with ``fold_children`` for a
    fold-local augmentation run) is consumed ONLY by the in-process branch — it builds the resolver from
    that in-memory dataset, skipping the duplicate disk reload. The subprocess branch ignores it: the
    adapter re-materializes from ``dataset_path`` / ``dataset_pickle`` (env channel) as before.

    ``random_state`` is forwarded ONLY to the subprocess branch (it sets ``N4A_RANDOM_STATE`` in the
    PER-CALL child env so the fresh-python adapter seeds its global RNG before fitting). The in-process
    branch ignores it: it fits operators in THIS process, whose global RNG ``run_via_dagml`` already
    seeded — so re-seeding here would be redundant.
    """
    if in_process_enabled():
        return run_cv_refit_bundle(
            dsl=dsl,
            envelope=envelope,
            graph=graph,
            dataset_path=dataset_path,
            selection_metric=selection_metric,
            sample_metadata=sample_metadata,
            dataset_pickle=dataset_pickle,
            dataset=dataset,
            fold_children=fold_children,
        )

    from .cli_runner import run_cv_refit_bundle as _subprocess_run_cv_refit_bundle

    outcome = _subprocess_run_cv_refit_bundle(
        dsl=dsl,
        envelope=envelope,
        graph=graph,
        dataset_path=dataset_path,
        workdir=workdir,
        dagml_cli=dagml_cli,
        venv_python=venv_python,
        selection_metric=selection_metric,
        sample_metadata=sample_metadata,
        dataset_pickle=dataset_pickle,
        random_state=random_state,
    )
    # Lift the native ScoreSet from the bundle the CLI wrote, so the outcome shape matches the
    # in-process branch (the call sites read outcome["scores"], not bundle.json). Only on success;
    # a non-zero returncode is handled by the caller's guard before scores are ever consumed.
    bundle_path = Path(workdir) / "bundle.json"
    outcome["scores"] = json.loads(bundle_path.read_text()).get("scores") if outcome["returncode"] == 0 and bundle_path.exists() else None
    return outcome
