"""Run generated-view DAG-ML campaigns in an isolated Python interpreter."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cloudpickle

from .resources import current_execution_resources


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
            str(venv_python or sys.executable), "-m", "nirs4all.pipeline.dagml.generated_worker",
            str(request_path), str(response_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
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
