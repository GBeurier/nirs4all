"""Execute an IO data source through the native preparation scheduler."""

from __future__ import annotations

import copy
from typing import Any


def prepare_data_provider(
    dataset: Any, *, engine: str, should_stop: Any = None,
    pipeline: Any = None, refit: Any = None, tuning: Any = None,
    calibration: Any = None, terminal_predict: Any = None,
    save_artifacts: Any = None, project: Any = None,
    runner_kwargs: dict[str, Any] | None = None, session: Any = None,
    random_state: int | None = None,
) -> Any:
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
    if recipe["params"]["_io_assembly"].get("view_generation"):
        from .generated_views import qualified_generated_by_source_pipeline, qualified_generated_model_pipeline

        model_shape = qualified_generated_model_pipeline(pipeline)
        by_source_shape = qualified_generated_by_source_pipeline(pipeline)

        tuning_ok = tuning is None
        if isinstance(tuning, dict) and model_shape and len(pipeline) == 2:
            from nirs4all.operators.models.multimodal import MultimodalClassifier, MultimodalRegressor

            from .tuning_contracts import SUPPORTED_TUNING_KEYS, parse_tuning_spec

            controls = {key: value for key, value in tuning.items() if key in SUPPORTED_TUNING_KEYS}
            spec = parse_tuning_spec(controls)
            tuning_ok = (
                isinstance(pipeline[1]["model"], (MultimodalRegressor, MultimodalClassifier))
                and not (tuning.keys() - SUPPORTED_TUNING_KEYS - {"progress_callback"})
                and spec.engine == "n4m" and spec.sampler in {None, "random"}
                and spec.pruner in {None, "none"} and spec.force_params is None
                and (spec.seed is None or 0 <= spec.seed < 2**32)
                and (tuning.get("progress_callback") is None or callable(tuning["progress_callback"]))
            )

        # Preflight before PLAN: no unsupported pipeline may materialize the
        # eager cohort and then silently train against it instead of views.
        qualified = (
            (model_shape or by_source_shape)
            and refit is True and tuning_ok and calibration is None
            and terminal_predict is None
            and save_artifacts is False and project is None and session is None
            and "workspace_path" not in (runner_kwargs or {})
        )
        if not qualified:
            raise NotImplementedError(
                "generate_view fold-view and training-content attestation is qualified only for "
                "[KFold, GroupKFold, StratifiedKFold or StratifiedGroupKFold, "
                "optional concrete X transformer, {'model': a concrete estimator}], or the existing "
                "distinct by_source preprocessing + source concat + concrete model shape, "
                "with refit=True, save_artifacts=False, and no calibration, project, session or workspace; "
                "tuning requires one multimodal estimator with the supported n4m search profile and no separate transformer"
            )
        if random_state is not None and (type(random_state) is not int or not 0 <= random_state < 2**32):
            raise ValueError("generate_view requires a non-negative 32-bit integer random_state or None")
        from .in_process_runner import in_process_enabled

        if not in_process_enabled():
            if tuning is not None:
                raise NotImplementedError("generate_view tuning requires in-process execution; subprocess HPO is not qualified")
            if should_stop is not None:
                raise NotImplementedError("generate_view subprocess execution does not support a cancellation callback")
        import importlib
        import inspect

        try:
            native = importlib.import_module("dag_ml._dag_ml")
        except ImportError as exc:
            raise NotImplementedError("generate_view requires the DAG-ML in-process binding") from exc
        run_cv_refit = getattr(native, "run_cv_refit_in_process", None)
        if not callable(run_cv_refit) or "root_seed" not in inspect.signature(run_cv_refit).parameters:
            raise NotImplementedError("generate_view requires a DAG-ML in-process binding with root_seed support")
        if tuning is not None:
            hpo = getattr(native, "run_host_hpo_search_in_process", None)
            if not callable(hpo) or not {"view_callback_factory", "resume_view_validator"}.issubset(inspect.signature(hpo).parameters):
                raise NotImplementedError("generate_view tuning requires a DAG-ML binding with checkpointed HPO view support")
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
    if recipe["params"]["_io_assembly"].get("view_generation"):
        from .generated_views import GeneratedViewStore

        # This run-local callback is detached before any dataset pickle is
        # produced; it is never part of archive provenance or replay state.
        cohort._generated_view_store = GeneratedViewStore(dataset)
    return cohort
