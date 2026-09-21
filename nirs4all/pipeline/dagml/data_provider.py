"""Execute an IO data source through the native preparation scheduler."""

from __future__ import annotations

import copy
from typing import Any


def prepare_data_provider(dataset: Any, *, engine: str, should_stop: Any = None) -> Any:
    """Materialize a finite provider once, before folds or scientific fitting.

    The provider owns its seed independently of estimator ``random_state``.
    Native PLAN derives the effective task seed and attests execution; IO owns
    assembly and validation. Only the realized cohort enters training/replay.
    """
    import nirs4all_io

    provider_type = getattr(nirs4all_io, "DataProvider", ())
    if not isinstance(dataset, provider_type):
        return dataset
    if engine != "dag-ml":
        raise ValueError("DataProvider requires the general DAG-ML profile; use engine='dag-ml' or omit engine")
    if should_stop is not None and not callable(should_stop):
        raise TypeError("should_stop must be a zero-argument cancellation callback")

    from .cancellation import DagRunCancelled

    cancelled: DagRunCancelled | None = None

    def check_stopped() -> None:
        nonlocal cancelled
        if should_stop is not None and should_stop():
            cancelled = DagRunCancelled("DAG data provider cancelled by caller")
            raise cancelled

    check_stopped()
    import dag_ml

    execute = getattr(dag_ml, "execute_data_provider", None)
    if execute is None:
        raise RuntimeError("The installed dag-ml binding lacks execute_data_provider; install a binding with data-provider PLAN support")
    recipe = dataset.recipe()
    cohorts: list[Any] = []

    def provide(task: dict[str, Any]) -> dict[str, Any]:
        check_stopped()
        cohort = dataset.materialize(seed=task["seed"], context=recipe["context"])
        check_stopped()
        cohorts.append(cohort)
        return {
            "handle": {"handle": 1, "kind": "data", "owner_controller": task["node_plan"]["controller_id"]},
            "metadata": {"content_fingerprint": dataset.fingerprint, "sample_count": len(cohort.sample_ids)},
        }

    try:
        evidence = execute(recipe, provide)
    except Exception:
        # The native callback boundary reports host exceptions as runtime
        # errors. Preserve the public cooperative cancellation type.
        if cancelled is not None:
            raise cancelled from None
        raise
    if len(cohorts) != 1:
        raise RuntimeError("native data-provider preparation did not execute exactly one source task")
    cohort = cohorts[0]
    # Host provenance contains no callback, mutable provider, or training data.
    # MultimodalSpectroDataset copies it before any subprocess serialization.
    cohort._data_provider_evidence = {"recipe": copy.deepcopy(recipe), "execution": evidence}
    return cohort
