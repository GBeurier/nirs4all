"""Public native training and host HPO for explicitly supplied controllers."""

from __future__ import annotations

import importlib
from typing import Any, cast


def execute_training(
    request: Any,
    data_envelopes: Any,
    relations: Any,
    training_influence: Any,
    op_callback: Any,
    *,
    outcome_id: str,
    run_id: str,
    bundle_id: str,
    warnings: Any = (),
    diagnostics: Any = None,
    artifact_callback: Any | None = None,
) -> Any:
    """Execute a signed native DAG request with the caller's controller.

    Named data envelopes, relations and training influence are native DAG-ML
    contracts. ``op_callback`` receives a native NodeTask and returns its
    NodeResult. ``artifact_callback`` exports RAW fitted model bytes and can
    hydrate/release them for replay. Controllers may execute in Python or
    forward these frames to a persistent R or Octave process.

    DAG-ML validates the contracts and owns CV, nested OOF, selection, REFIT
    and lifecycle. Arguments, native errors and the result pass through
    unchanged. This entry point neither builds contracts nor runs numerical
    kernels or another execution engine.

    Returns:
        The native DAG-ML ``TrainingResult`` DTO, with its typed outcome,
        portable package export and explicit ``detach()`` lifecycle. It is
        distinct from the high-level ``RunResult`` returned by ``run()``.
    """

    dag_ml = importlib.import_module("dag_ml")
    return dag_ml.execute_training(
        request, data_envelopes, relations, training_influence, op_callback,
        outcome_id=outcome_id, run_id=run_id, bundle_id=bundle_id,
        warnings=warnings, diagnostics=diagnostics, artifact_callback=artifact_callback,
    )


def run_host_hpo_search(
    dsl: Any,
    envelope: Any,
    controller_manifests: Any,
    request: Any,
    op_callback: Any,
    optimizer_callback: Any,
    *,
    resume_checkpoint: Any = None,
    progress_callback: Any = None,
    candidate_callback_factory: Any = None,
    view_callback_factory: Any = None,
    resume_view_validator: Any = None,
) -> dict[str, Any]:
    """Run native FIT_CV trials with host optimizer and controller callbacks.

    Supply the pipeline DSL, signed envelope, trusted controller manifests
    and native HPO request. DAG-ML owns folds, nested OOF, objective scores,
    selection and checkpoint validation. Host callbacks propose parameters
    and execute the scheduled tasks, including through external processes.

    Resume checkpoints, progress and candidate-local controller/generated
    view factories retain the public DAG-ML semantics. The trial budget is
    the target total across resumed calls; this search performs no REFIT.
    All inputs and native errors pass through unchanged.

    Returns:
        The native search dictionary, including trial evidence and selected
        proposal; durable searches also include status and checkpoint.
    """

    dag_ml = importlib.import_module("dag_ml")
    return cast(dict[str, Any], dag_ml.run_host_hpo_search_in_process(
        dsl, envelope, controller_manifests, request, op_callback, optimizer_callback,
        resume_checkpoint=resume_checkpoint, progress_callback=progress_callback,
        candidate_callback_factory=candidate_callback_factory,
        view_callback_factory=view_callback_factory, resume_view_validator=resume_view_validator,
    ))


__all__ = ["execute_training", "run_host_hpo_search"]
