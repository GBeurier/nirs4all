"""Run generated-view HPO in a child with its own Python RNG and view store."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cloudpickle

from .generated_subprocess import _run_cancellable_worker
from .resources import current_execution_resources


def run_generated_hpo_subprocess(
    pipeline: Any, cohort: Any, tuning: dict[str, Any], *, run_options: dict[str, Any],
) -> Any:
    """Transfer PLAN and return a complete HPO result from an isolated worker."""
    from nirs4all.api.result import RunResult

    store = getattr(cohort, "_generated_view_store", None)
    if store is None:
        raise ValueError("generated HPO subprocess requires the PLAN provider and its view store")
    progress = tuning.get("progress_callback")
    if progress is not None and not callable(progress):
        raise TypeError("tuning.progress_callback must be callable")
    should_stop = run_options.get("should_stop")
    if should_stop is not None and not callable(should_stop):
        raise TypeError("should_stop must be a zero-argument cancellation callback")
    provider = store.provider_for_worker()
    if cohort is not provider.cohort:
        raise ValueError("generated HPO subprocess provider lost its PLAN view-store link")
    with tempfile.TemporaryDirectory(prefix="generated-hpo-worker-") as private_dir:
        request_path = Path(private_dir) / "request.pkl"
        response_path = Path(private_dir) / "response.pkl"
        request = {
            "schema": "nirs4all.generated-hpo-worker.v2" if progress is not None else "nirs4all.generated-hpo-worker.v1",
            "provider": provider,
            "pipeline": pipeline,
            "tuning": {key: value for key, value in tuning.items() if key != "progress_callback"},
            "progress_dir": private_dir if progress is not None else None,
            "run_options": {key: value for key, value in run_options.items() if key != "should_stop"},
            "resources": current_execution_resources(),
        }
        del cohort._generated_view_store
        try:
            with request_path.open("wb") as stream:
                cloudpickle.dump(request, stream)
        finally:
            cohort._generated_view_store = store
        command = [
            str(run_options.get("venv_python") or sys.executable),
            "-P", "-s", "-B", "-m", "nirs4all.pipeline.dagml.generated_hpo_worker",
            str(request_path), str(response_path),
        ]
        completed: subprocess.CompletedProcess[str] = _run_cancellable_worker(
            command, should_stop, progress_dir=Path(private_dir), progress_callback=progress,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"generated HPO worker failed with exit code {completed.returncode}:\n"
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        if not response_path.is_file():
            raise RuntimeError("generated HPO worker did not return a result")
        with response_path.open("rb") as stream:
            result = cloudpickle.load(stream)  # noqa: S301 - private response from our child
    if isinstance(result, dict) and result.get("status") == "stopped":
        from .multimodal_tuning import MultimodalTuningStopped

        if not isinstance(result.get("evidence"), dict):
            raise ValueError("generated HPO worker returned invalid cancellation evidence")
        raise MultimodalTuningStopped(result["evidence"])
    if (not isinstance(result, RunResult) or result.tuning_result is None
            or not isinstance(result._dagml_generated_view_manifest, dict)):
        raise ValueError("generated HPO worker returned an invalid native result")
    return result
