"""Assemble dag-ml-cli inputs for a CV+refit run and drive the binary (Phase 2b-ii.3).

This is the integration seam that makes dag-ml execute a nirs4all pipeline end-to-end. Two
things the plain compat DSL does NOT provide had to be added as top-level DSL fields (the
compat lowerer preserves them verbatim):

* ``data_bindings`` — binds the **model** node's input ``x`` to the data source, carrying the
  envelope's fingerprint triple (the runtime keys its data provider by that triple, not by
  source id). The compat ``sources`` step is a no-op, and there is no auto-edge from the graph
  interface input to the first node, so without a binding the node gets empty ``data_views``.
  The binding targets the model node specifically because our node runner re-fetches features
  by the task's own sample_ids (it does not consume upstream edge data).
* ``split_invocation.fold_set`` — a materialized ``FoldSet``. KFold *params* are inert: dag-ml
  has no runtime/host splitter that materializes the outer fold set, so the host must compute
  the folds (``build_fold_set``) and embed them.

With both, the runtime emits per-fold ``data_views{sample_ids}`` and the adapter resolves real
X/y. Verified end-to-end against ``dag-ml-cli run-process-dsl-cv-refit-bundle``: the FIT_CV OOF
matches direct sklearn KFold. The envelope must be scoped to the CV universe (``build_envelope(
…, sample_ints=train_pool)``) so every relation lives inside the fold set.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nirs4all.pipeline.dagml_bridge import build_dagml_plan, controller_manifests, pipeline_to_dsl

from .envelope import build_fold_set

if TYPE_CHECKING:
    from .identity import IdentityMap

_SOURCE_ID = "src0"


def model_node_id(pipeline: list[Any], *, dsl_id: str = "nirs4all-pipeline") -> str:
    """The compiler's id for the model node (compile-first; do not re-derive the ordinal)."""
    plan = build_dagml_plan(pipeline, plan_id="plan:probe", dsl_id=dsl_id).to_dict()
    return str(next(node["id"] for node in plan["graph_plan"]["graph"]["nodes"] if node["kind"] == "model"))


def data_bindings_for(model_id: str, envelope: dict[str, Any], *, source_id: str = _SOURCE_ID) -> list[dict[str, Any]]:
    """One DataBinding on the model node's ``x`` input, carrying the envelope's fingerprints."""
    return [
        {
            "node_id": model_id,
            "input_name": "x",
            "request_id": envelope["plan"]["id"],
            "schema_fingerprint": envelope["schema_fingerprint"],
            "plan_fingerprint": envelope["plan_fingerprint"],
            "relation_fingerprint": envelope["relation_fingerprint"],
            "output_representation": "tabular_numeric",
            "feature_set_id": "x",
            "source_ids": [source_id],
            "require_relations": True,
        }
    ]


def split_invocation_for(identity: IdentityMap, folds: list[tuple[list[int], list[int]]], *, n_splits: int, shuffle: bool = True) -> dict[str, Any]:
    """A split_invocation with an embedded, materialized FoldSet (params alone are inert)."""
    return {
        "id": "split:outer",
        "controller_id": None,
        "params": {"kind": "kfold", "n_splits": n_splits, "shuffle": shuffle},
        "fold_set": build_fold_set(identity, folds, set_id="folds.outer"),
    }


def assemble_cv_refit_dsl(pipeline: list[Any], identity: IdentityMap, envelope: dict[str, Any], folds: list[tuple[list[int], list[int]]], *, dsl_id: str = "nirs4all-pipeline", n_splits: int, source_id: str = _SOURCE_ID) -> dict[str, Any]:
    """The executable compat DSL: lowered pipeline + embedded fold_set + model data binding."""
    dsl = pipeline_to_dsl(pipeline, dsl_id)
    dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=n_splits)
    dsl["data_bindings"] = data_bindings_for(model_node_id(pipeline, dsl_id=dsl_id), envelope, source_id=source_id)
    return dsl


def write_launcher_shim(path: Path, venv_python: str, *, capture: Path | None = None) -> Path:
    """Write a non-``.py`` executable shim that re-execs the adapter under the project venv.

    dag-ml-cli runs a ``.py`` adapter under the *system* python3 (which lacks nirs4all), so the
    adapter must be a non-``.py`` executable. ``capture`` tees the adapter's result frames for
    inspection (the OOF predictions are not persisted to the cache without a requires_oof edge).
    """
    run = f'{venv_python} -m nirs4all.pipeline.dagml.process_adapter "$@"'
    body = f"#!/bin/sh\n{run} | tee {capture}\n" if capture is not None else f"#!/bin/sh\nexec {run}\n"
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IRUSR)
    return path


def run_cv_refit_bundle(
    *,
    dsl: dict[str, Any],
    envelope: dict[str, Any],
    graph: dict[str, Any],
    dataset_path: str,
    workdir: Path,
    dagml_cli: str,
    venv_python: str,
) -> dict[str, Any]:
    """Write inputs + shim, run ``dag-ml-cli run-process-dsl-cv-refit-bundle``, return outputs.

    Returns ``{returncode, stdout, results}`` where ``results`` are the adapter's captured
    NodeResult frames (carrying the FIT_CV validation/OOF predictions).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "dsl.json").write_text(json.dumps(dsl))
    (workdir / "controllers.json").write_text(json.dumps(controller_manifests()))
    (workdir / "envelope.json").write_text(json.dumps(envelope))
    (workdir / "graph.json").write_text(json.dumps(graph))
    capture = workdir / "results.jsonl"
    shim = write_launcher_shim(workdir / "n4a_adapter", venv_python, capture=capture)

    env = {**os.environ, "N4A_DAGML_DATASET_PATH": dataset_path, "N4A_DAGML_GRAPH_PATH": str(workdir / "graph.json")}
    proc = subprocess.run(
        [
            dagml_cli, "run-process-dsl-cv-refit-bundle",
            "--dsl", str(workdir / "dsl.json"), "--controllers", str(workdir / "controllers.json"),
            "--envelope", str(workdir / "envelope.json"), "--adapter", str(shim), "--persistent",
            "--bundle-id", "bundle:n4a", "--plan-id", "plan:n4a",
            "--output", str(workdir / "bundle.json"), "--prediction-cache-output", str(workdir / "cache.json"),
        ],
        capture_output=True, text=True, env=env, check=False,
    )
    results = [json.loads(line) for line in capture.read_text().splitlines() if line.strip()] if capture.exists() else []
    return {"returncode": proc.returncode, "stdout": proc.stdout + proc.stderr, "results": results}
