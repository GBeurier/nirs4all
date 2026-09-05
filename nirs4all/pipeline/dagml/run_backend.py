"""Run a nirs4all pipeline on the **dag-ml** engine and return a ``RunResult`` (ADR-17 backend).

This is the operational seam for ``engine="dag-ml"``: it assembles the executable compat DSL,
drives ``dag-ml-cli`` through the nirs4all process adapter, and maps dag-ml's **native**
``bundle.scores`` — per-fold validation RMSE/R², the cross-fold OOF average (``cv_best_score``)
and the final-test score (``best_rmse``), all computed in Rust — into an in-memory
:class:`~nirs4all.data.predictions.Predictions`, wrapped in a :class:`~nirs4all.api.result.RunResult`.

The public API persists a workspace projection of the executed results. Supports the
vertical-slice shape (feature transforms + one model + an OOF/KFold-style splitter). Non-partition
CV (e.g. ``ShuffleSplit``) is not yet supported by the dag-ml ``FoldSet`` (see migration notes).

The implementation is split across cohesive sibling modules — this module owns the entry point
(:func:`run_via_dagml`) and the path dispatch (:func:`_dispatch_run`); the detectors, dataset
materialization, exclude/tag resolution, fold construction, score mapping, and the per-shape
``_run_*`` paths live in :mod:`.detect`, :mod:`.dataset`, :mod:`.exclude`, :mod:`.folds`,
:mod:`.result`, and :mod:`.run_paths` respectively. The names re-exported below keep the historical
``nirs4all.pipeline.dagml.run_backend`` import surface stable.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np

from nirs4all.api.result import RunResult
from nirs4all.core.metrics import is_higher_better

from .dataset import _dataset_inputs, _materialize_dataset
from .detect import (
    _detect_by_source_branch,
    _detect_by_source_concat_shared_preproc,
    _detect_by_source_distinct_preproc_concat,
    _detect_by_source_stacking_branch,
    _detect_duplication_branch,
    _detect_named_metamodel_feature_stack,
    _detect_rep_fusion,
    _detect_separation_branch,
    _detect_separation_preproc_concat,
    _detect_source_concat_merge,
    _detect_stacking_branch,
    _fusion_merge_aggregate,
    _generation_kind,
    _is_augmentation_step,
    _is_constrained_operator_generator,
    _is_duplication_branch_step,
    _is_exclude_step,
    _is_flat_single_operator_generator,
    _is_stacking_merge_step,
    _is_unconstrained_operator_generator,
)
from .errors import DagMlUnavailable, DagMlUnsupported, _OperatorLoweringUnsupported
from .exclude import _excluded_from_pool, _resolve_exclude, _resolve_tags
from .finetune_lowering import (
    PUBLIC_DAGML_SELECTION_METRICS,
    lower_deterministic_finetune_params_to_generators,
    reject_native_training_param_overrides,
)
from .folds import _build_folds, _build_group_folds, _is_repetition_dataset, _repetition_groups_for_pool
from .native_results import native_results_enabled, write_native_results
from .result import _project_operator_sweep, _scores_to_run_result
from .run_paths import (
    _FUSION_MERGE_NODE_ID,
    _augmentation_is_leakage_free,
    _canonical_source_branch,
    _operator_is_stateless,
    _reshape_for_rep_fusion,
    _run_augmentation,
    _run_by_source_branch,
    _run_by_source_concat_shared_preproc,
    _run_by_source_distinct_preproc_concat,
    _run_by_source_stacking_branch,
    _run_concrete_scores,
    _run_duplication_branch,
    _run_named_metamodel_feature_stack,
    _run_native_generation,
    _run_native_operator_generation,
    _run_rep_fusion,
    _run_repetition,
    _run_separation_branch,
    _run_separation_preproc_concat,
    _run_source_concat_merge,
    _run_stacking_branch,
)
from .steps import _expand_operator_generators, _is_split_step


def _default_dagml_cli() -> Path:
    """Return the preferred dag-ml-cli candidate for the current workspace layout."""
    explicit = os.environ.get("N4A_DAGML_CLI")
    if explicit:
        return Path(explicit).expanduser()

    workspace = Path(__file__).resolve().parents[4]
    candidates = [
        workspace / "RC-v1-dagml" / "target" / "release" / "dag-ml-cli",
        workspace / "RC-v1-dagml" / "target" / "debug" / "dag-ml-cli",
        workspace / "dag-ml" / "target" / "release" / "dag-ml-cli",
        workspace / "dag-ml" / "target" / "debug" / "dag-ml-cli",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


_DEFAULT_CLI = _default_dagml_cli()

# Names re-exported for the stable ``nirs4all.pipeline.dagml.run_backend`` import surface (the parity
# suite and any caller import these private helpers directly from this module).
__all__ = [
    "DagMlUnavailable",
    "DagMlUnsupported",
    "_FUSION_MERGE_NODE_ID",
    "_augmentation_is_leakage_free",
    "_build_folds",
    "_build_group_folds",
    "_canonical_source_branch",
    "_detect_by_source_branch",
    "_detect_by_source_concat_shared_preproc",
    "_detect_by_source_distinct_preproc_concat",
    "_detect_by_source_stacking_branch",
    "_detect_duplication_branch",
    "_detect_named_metamodel_feature_stack",
    "_detect_rep_fusion",
    "_detect_separation_branch",
    "_detect_separation_preproc_concat",
    "_detect_source_concat_merge",
    "_detect_stacking_branch",
    "_excluded_from_pool",
    "_fusion_merge_aggregate",
    "_generation_kind",
    "_is_augmentation_step",
    "_is_flat_single_operator_generator",
    "_is_stacking_merge_step",
    "_is_unconstrained_operator_generator",
    "_operator_is_stateless",
    "_repetition_groups_for_pool",
    "_reshape_for_rep_fusion",
    "_run_separation_preproc_concat",
    "_resolve_exclude",
    "_run_native_operator_generation",
    "_run_by_source_concat_shared_preproc",
    "_run_by_source_distinct_preproc_concat",
    "_run_by_source_stacking_branch",
    "_run_rep_fusion",
    "_run_source_concat_merge",
    "preflight_dagml_backend",
    "run_via_dagml",
]


def _has_finetune_params(pipeline: list[Any]) -> bool:
    """Whether any pipeline step asks legacy Optuna finetuning to mutate model params."""
    return any(isinstance(step, dict) and "finetune_params" in step for step in pipeline)


def _metric_objective(metric: str) -> str:
    """Return the native selection direction implied by a metric name."""

    return "maximize" if is_higher_better(metric) else "minimize"


def _lower_public_finetune_params(pipeline: Any) -> tuple[list[Any], dict[str, str]]:
    """Lower public deterministic ``finetune_params`` before dag-ml routing."""

    steps = list(pipeline)
    if not _has_finetune_params(steps):
        return steps, {}
    try:
        return lower_deterministic_finetune_params_to_generators(
            steps,
            context="public engine='dag-ml'",
            supported_selection_metrics=PUBLIC_DAGML_SELECTION_METRICS,
        )
    except (TypeError, ValueError) as exc:
        raise NotImplementedError(
            f"engine='dag-ml' supports only deterministic finetune_params.model_params grids/ranges natively; adaptive n4m/Optuna finetune_params still require the Python optimizer path. Details: {exc}"
        ) from exc


# Residual options supported by this executor. Unknown options are rejected
# before work; there is no implicit legacy execution.
_HONORED_RUNNER_KWARGS: frozenset[str] = frozenset({"workspace_path", "store_run_id", "should_stop"})

_PERSISTENCE_REJECT_MESSAGES: dict[str, str] = {
    "store_run_id": "engine='dag-ml' cannot yet attach execution to an existing store_run_id.",
    "keep_datasets": "engine='dag-ml' does not yet persist dataset snapshots.",
}


def _reject_unsupported_run_options(*, refit: Any, project: str | None, session: Any, cache: Any, runner_kwargs: dict[str, Any]) -> None:
    """Validate execution options before any operator or durable write.

    General sessions own DAG results without a PipelineRunner. Workspace and
    project options are handled by the post-execution storage projection.
    Unsupported refit/cache/runner options remain explicit parity gaps, never
    invitations to run a different engine after failure.
    """
    if refit is not True:
        raise DagMlUnsupported(f"engine='dag-ml' always runs native CV+refit on the single CV winner and cannot honor refit={refit!r} (disable / custom top-k / ranking selection).")
    if session is not None:
        session._prepare_dagml_run()
    if cache is not None:
        raise DagMlUnsupported("engine='dag-ml' runs no nirs4all StepCache, so it cannot honor a CacheConfig.")
    # Unknown execution options are refused before work, never silently dropped.
    for key in runner_kwargs:
        if key in _HONORED_RUNNER_KWARGS:
            continue
        if key in _PERSISTENCE_REJECT_MESSAGES:
            raise DagMlUnsupported(_PERSISTENCE_REJECT_MESSAGES[key])
        raise DagMlUnsupported(f"engine='dag-ml' does not yet honor the run() option {key!r}.")


def preflight_dagml_backend(cli: str) -> None:
    """Require the in-process extension or an explicitly available DAG CLI.

    Availability failure is reported to the caller before execution. It never
    changes the selected engine or retries with PipelineRunner.
    """
    from .in_process_runner import _dagml_extension_loads, in_process_enabled

    if in_process_enabled() and _dagml_extension_loads():
        return
    if Path(cli).exists():
        return
    raise DagMlUnavailable(
        "the dag-ml backend is not available: the in-process extension 'dag_ml._dag_ml' did not load "
        f"and the dag-ml-cli binary was not found at {cli} (build it: cargo build -p dag-ml-cli --release). "
        "Install a working DAG-ML runtime before retrying."
    )


def run_via_dagml(
    pipeline: Any,
    dataset: Any,
    *,
    name: str = "",
    random_state: int | None = None,
    refit: bool | dict[str, Any] | list[dict[str, Any]] | None = True,
    project: str | None = None,
    session: Any | None = None,
    cache: Any | None = None,
    runner_kwargs: dict[str, Any] | None = None,
    dagml_cli: str | Path | None = None,
    venv_python: str | None = None,
    workdir: str | Path | None = None,
    save_charts: bool = True,
    plots_visible: bool = False,
    results_path: str | Path | None = None,
    verbose: int = 0,
    save_artifacts: bool | None = None,
    report_naming: str = "nirs",
) -> RunResult:
    """Execute a general pipeline and project its scored results.

    The Rust in-process/CLI runtime is chosen by the execution router. The
    public API supplies save_artifacts=True by default: persist captured models
    under workspace/native_results and exact predictions in WorkspaceStore.
    False omits fitted model persistence; an explicitly requested workspace or
    project still receives metadata/arrays. None is the internal direct-call
    memory-only mode. An explicit results_path
    independently requests the verified native results format.

    project tags the stored run; session retains the scored result and workspace
    without constructing PipelineRunner. verbose configures logging, and
    report_naming selects the existing NIRS/ML metric display labels. Remaining
    unsupported execution options are rejected before operators execute.
    """
    # Validate execution and presentation options before any operator runs.
    if isinstance(verbose, bool) or not isinstance(verbose, int) or verbose not in range(4):
        raise ValueError("verbose must be an integer from 0 through 3")
    if save_artifacts is not None and not isinstance(save_artifacts, bool):
        raise TypeError("save_artifacts must be a bool")
    if report_naming not in {"nirs", "ml", "auto"}:
        raise ValueError("report_naming must be 'nirs', 'ml', or 'auto'")
    effective_runner_kwargs = dict(runner_kwargs or {})
    should_stop = effective_runner_kwargs.get("should_stop")
    if should_stop is not None and not callable(should_stop):
        raise TypeError("should_stop must be a zero-argument cancellation callback")
    if should_stop is not None and should_stop():
        from .cancellation import DagRunCancelled

        raise DagRunCancelled("DAG run cancelled by caller")
    if session is not None and session.workspace_path is not None:
        effective_runner_kwargs.setdefault("workspace_path", session.workspace_path)
    parent_run_id = effective_runner_kwargs.get("store_run_id")
    if parent_run_id is not None:
        from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

        if not isinstance(parent_run_id, str) or not parent_run_id or not effective_runner_kwargs.get("workspace_path"):
            raise ValueError("store_run_id requires an existing run ID and its explicit workspace_path")
        with WorkspaceStore(effective_runner_kwargs["workspace_path"]) as parent_store:
            parent_run = parent_store.get_run(parent_run_id)
            if parent_run is None or parent_run["status"] != "running":
                raise ValueError("store_run_id must identify an existing running run in the supplied workspace")
    _reject_unsupported_run_options(refit=refit, project=project, session=session, cache=cache, runner_kwargs=effective_runner_kwargs)
    from nirs4all.core.logging import configure_logging, get_logger

    configure_logging(verbose=verbose)
    logger = get_logger(__name__)
    logger.info("Training pipeline with engine='dag-ml'")

    # Apply the honored options. random_state seeds the global RNG exactly as legacy run() does, so the
    # dag-ml engine's stochastic paths (augmentation / randomized splitters) and any unseeded stochastic
    # operator (e.g. RandomForestRegressor()) are reproducible too. This seeds the PARENT process, which
    # covers the IN-PROCESS path (Mechanism B fits operators in this process via op_callback). For the
    # SUBPROCESS path (Mechanism A) the adapter re-execs a FRESH python whose global RNG is unseeded, so
    # `random_state` is THREADED down to cli_runner, which puts it in the PER-CALL child env dict only
    # (no os.environ mutation → no leak / concurrency hazard) so the adapter seeds the same way at startup.
    if random_state is not None:
        from nirs4all.pipeline.runner import init_global_random_state

        init_global_random_state(random_state)

    # Fail before execution if neither DAG mechanism is installed. Ordinary
    # operator errors later propagate untouched; no execution is retried.
    cli = str(dagml_cli or _default_dagml_cli())
    preflight_dagml_backend(cli)

    from nirs4all.pipeline.config.component_serialization import deserialize_component

    # Public PipelineConfigs/Studio declarations use the library's existing
    # canonical serialization. Resolve operators before splitter/shape checks;
    # a serialized KFold must not accidentally enter the no-splitter route.
    pipeline = deserialize_component(pipeline)

    # Materialize the host dataset from ANY input legacy `run()` accepts (path / config /
    # DatasetConfigs / live SpectroDataset / (X, y) tuple / array) — DatasetConfigs alone silently
    # skips the in-memory ones, so `_materialize_dataset` wraps them with the legacy normalization.
    spectro = _materialize_dataset(dataset)
    requested_charts = isinstance(pipeline, list) and any(_is_chart_step(step) for step in pipeline)
    if requested_charts and (save_charts or plots_visible):
        from .chart_projection import validate_chart_projection

        validate_chart_projection(pipeline, spectro)
    base_dir = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="n4a_dagml_"))
    # `dataset_arg` is the reloadable path (clean file-path datasets, no pickle — fast); `host_pickle`
    # is set only when the adapter cannot faithfully reload from a path (in-memory inputs, or a path
    # whose re-load diverges from the host identity), and ships the byte-identical host dataset.
    dataset_arg, host_pickle = _dataset_inputs(dataset, spectro, base_dir / "host")

    # When WE allocated `base_dir` (no caller `workdir`), it holds only run scratch — the host pickle,
    # the per-path shim/JSON artifacts, and dag-ml's bundle.json (read into memory before we return).
    # Nothing in the returned RunResult points into it (scores are parsed in-memory by
    # `_scores_to_run_result`), so it is safe to delete on every dispatch return/raise path. A
    # caller-provided `workdir` is theirs — never delete it.
    from .cancellation import SHOULD_STOP, check_cancellation

    cancellation_token = SHOULD_STOP.set(should_stop)
    try:
        result = _dispatch_run(
            pipeline,
            spectro,
            base_dir,
            dataset_arg,
            host_pickle,
            cli,
            venv_python,
            name,
            random_state,
            save_charts=save_charts,
            plots_visible=plots_visible,
        )
        from .envelope import target_names

        result._dagml_target_names = target_names(spectro)
        check_cancellation()
        _attach_export_spec(result, pipeline, dataset, name, random_state)
        workspace_path = None
        if save_artifacts or project is not None or "workspace_path" in effective_runner_kwargs or (requested_charts and save_charts):
            from nirs4all.pipeline.runner import _get_default_workspace_path

            workspace_path = Path(effective_runner_kwargs.get("workspace_path") or _get_default_workspace_path())
        if requested_charts:
            from .chart_projection import render_run_charts

            chart_paths = render_run_charts(result, pipeline, spectro, workspace_path=workspace_path, save_charts=save_charts, plots_visible=plots_visible, verbose=verbose)
            for metadata in result.per_dataset.values():
                metadata["chart_reports"] = chart_paths
        if save_artifacts and result._dagml_score_set is not None and not native_results_enabled(results_path):
            assert workspace_path is not None
            result._dagml_results_dir = write_native_results(result, result._dagml_score_set, workspace_path / "native_results")
        # Explicit native output remains independent of the workspace option.
        # The writer verifies real ScoreSet provenance; host-only paths must
        # not fabricate one to make an unsupported export look portable.
        if native_results_enabled(results_path):
            result._dagml_results_dir = write_native_results(result, result._dagml_score_set, results_path)  # noqa: SLF001
        if workspace_path is not None:
            from .workspace_projection import publish_workspace_result

            publish_workspace_result(result, pipeline, spectro, workspace_path, name=name, project=project, report_naming=report_naming, store_run_id=parent_run_id)
        if session is not None:
            session._adopt_dagml_result(result, dataset)
        from nirs4all.visualization.naming import get_metric_names

        full_train = any(metadata.get("execution_profile") == "full_train" for metadata in result.per_dataset.values())
        selected = result.best if full_train else result.cv_best
        metric_names = get_metric_names(cast(Any, report_naming), str(selected.get("task_type", "regression")), str(selected.get("metric", "rmse")))
        if full_train:
            logger.info("DAG-ML completed without cross-validation: training_score=%s, %s=%s", selected.get("train_score"), metric_names["test_score"], selected.get("test_score"))
        else:
            logger.info("DAG-ML completed: %s=%s", metric_names["cv_score"], result.cv_best_score)
        return result
    finally:
        SHOULD_STOP.reset(cancellation_token)
        if workdir is None:
            shutil.rmtree(base_dir, ignore_errors=True)


def _attach_export_spec(result: RunResult, pipeline: Any, dataset: Any, name: str, random_state: int | None) -> None:
    """Freeze the authoring inputs for export metadata and explicit legacy-refit compatibility.

    Normal export uses captured fitted artifacts and never trains again. Only
    ``compatibility='legacy-refit'`` can explicitly request a new legacy training
    run. The authoring inputs are FROZEN here, at run time, so a later
    mutation of the live ``pipeline`` / in-memory ``dataset`` (arrays, SpectroDataset) cannot make the
    export represent a different run than the one scored:

    * pipeline — ALWAYS deepcopied (cheap; the pipeline list holds mutable estimator instances).
    * dataset — a RELOADABLE on-disk PATH / file-based ``DatasetConfigs`` (``_reloadable_path`` resolves a
      path) is kept verbatim, since it replays from disk (on-disk files must be unchanged at export time,
      documented on ``_dagml_export_spec``). Any NON-reloadable / IN-MEMORY form — a ``SpectroDataset`` /
      ``ndarray`` / ``tuple``, OR a ``DatasetConfigs`` wrapping a preloaded ``SpectroDataset`` / in-memory
      arrays (a mutable descriptor that ``_reloadable_path`` returns ``None`` for) — is DEEPCOPIED to
      snapshot it, so a post-run mutation cannot corrupt the export.

    ``_dagml_export_stochastic`` flags a run whose explicitly requested legacy refit MAY differ from the dag-ml-scored
    model, so export can WARN. Two signals flag (any → flagged): (a) CERTAIN — a ``sample_augmentation``
    step (the dag-ml run only kept its augmented snapshot in the now-deleted temp dir, re-augmentation is not
    reproducible across processes, and the augmenter's own RNG is not covered by ``run(random_state)``);
    (b) CONSERVATIVE — the outer ``run(random_state) is None`` (nothing globally seeded, so global-RNG-dependent
    components are not reproducibly seeded across the engine boundary; this MAY over-warn a fully-deterministic
    pipeline — the safe direction for a "may differ" caveat).

    A per-estimator "is this op unseeded-stochastic" probe is deliberately NOT attempted: ``random_state``
    use is solver/config-conditional (``Ridge()`` / ``PCA(svd_solver="full")`` carry a DORMANT
    ``random_state=None`` yet are deterministic — a false alarm — while ``MLPRegressor(shuffle=False)`` is
    stochastic via weight init, and wrapped estimators like ``MetaModel(model=RandomForestRegressor())``
    hide theirs), so any static heuristic both over- and under-warns. The export()/export_model() WARNING
    and docstrings instead document the uncertain middle (a seeded run whose individual component left
    ``random_state=None``) as a general caveat. The per-run warning is CONSERVATIVE (the ``run-None`` signal
    may over-warn a fully-deterministic pipeline — the safe direction); the docstring removes the silent surprise.
    """
    import copy

    from .dataset import _reloadable_path

    frozen_pipeline = copy.deepcopy(pipeline)
    # Keep a reloadable on-disk path/config by reference; snapshot anything in-memory (incl. a
    # DatasetConfigs wrapping a preloaded SpectroDataset / arrays, which is path-less and mutable).
    frozen_dataset = dataset if _reloadable_path(dataset) is not None else copy.deepcopy(dataset)

    steps = pipeline if isinstance(pipeline, list) else []
    has_augmentation = any(_is_augmentation_step(step) for step in steps)
    stochastic = has_augmentation or random_state is None

    result._dagml_export_spec = {"pipeline": frozen_pipeline, "dataset": frozen_dataset, "name": name, "random_state": random_state}  # noqa: SLF001
    result._dagml_export_stochastic = stochastic  # noqa: SLF001


def _derive_config_name(pipeline: Any, name: str) -> str:
    """Derive the canonical legacy ``config_name`` for a genuinely-CONCRETE pipeline (else ``""``).

    Legacy does NOT use the raw ``name`` as ``config_name`` — :class:`PipelineConfigs` derives it as
    ``"{label}_{display_hash}"`` per expanded variant, where ``label`` is the per-pipeline name
    ``run()`` builds (``"{name}_p0"`` for a named single pipeline, the literal ``"config"`` when the
    name is empty) and ``display_hash`` is the 8-char hash of the SERIALIZED expanded steps. We REUSE
    that exact mechanism (no hand-rolled hash) by constructing a ``PipelineConfigs`` from the pipeline +
    the same ``"{name}_p0"`` label and reading its canonical ``.names[0]``. The refit ``_refit`` suffix
    is applied downstream in :func:`~nirs4all.pipeline.dagml.result._scores_to_run_result` (the standalone
    refit rows), matching legacy ``"{cv_config_name}_refit"``.

    SCOPE — a genuinely-single CONCRETE pipeline only (NO generator of ANY kind). A GENERATOR / SWEEP
    pipeline yields a WINNER-ONLY projection on the dag-ml backend and is NOT cleanly variant-mappable
    (dag-ml's native generation / the Python ``_expand_*`` path normalize + order variants differently
    from ``PipelineConfigs.expand_spec_with_choices``, and the per-variant display hash diverges), so it
    returns ``""`` rather than ship a wrong ``config_name`` (per-variant generator config_names are the
    sweep work, #55).

    Generator detection is ROUTING-ALIGNED, not based on ``PipelineConfigs.expansion_count``:
    ``expansion_count`` MISSES a native sibling-param sweep — ``{"model": M(), "p": {"_range_"/"_grid_":
    …}}`` routes to ``_run_native_generation`` / the ``_expand_operator_generators`` path, yet
    ``PipelineConfigs`` does not always expand the sibling-param into >1 variant (e.g. ``_grid_`` keeps
    ``expansion_count == 1``), so gating on it would emit the WRONG single ``"{name}_p0_{template_hash}"``
    for a sweep. Instead we blank when EITHER the dag-ml router classifies a generator
    (:func:`~nirs4all.pipeline.dagml.detect._generation_kind` != ``"none"`` — covers the native
    ``param_model`` sweep) OR any generator keyword is present anywhere
    (:func:`~nirs4all.pipeline.config._generator.keywords.has_nested_generator_keywords` — covers
    ``_or_`` / ``_range_`` / ``_grid_`` / ``_cartesian_`` / param-keyed and the operator-expand path).
    Only a pipeline both routers call generator-free derives a real name.

    Any derivation failure (an exotic pipeline form ``PipelineConfigs`` rejects) also returns ``""`` so a
    label is dropped rather than wrong.
    """
    try:
        from nirs4all.pipeline.config._generator.keywords import has_nested_generator_keywords
        from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
        from nirs4all.pipeline.dagml.detect import _generation_kind

        # Normalize ANY accepted pipeline form (list / dict-wrapped / path / PipelineConfigs) to its raw
        # step list with the SAME loader PipelineConfigs uses, so the generator detection sees the real
        # steps (a dict wrapper's keys would otherwise hide a sweep, e.g. {"pipeline": [...sweep...]}).
        if isinstance(pipeline, PipelineConfigs):
            return ""  # A pre-built PipelineConfigs carries its own (possibly multi-variant) names; do not relabel.
        steps = PipelineConfigs._load_steps(pipeline)  # noqa: SLF001 - reuse the exact legacy step-loading.
        if _generation_kind(steps) != "none" or has_nested_generator_keywords(steps):
            return ""  # Any generator / sweep path — winner-only projection is not cleanly mappable (#55).

        if name == "":
            label = ""  # PipelineConfigs maps "" → the literal "config" prefix (the unnamed legacy default).
        else:
            # run() builds the per-pipeline label as f"{name}_p{idx}"; a single dag-ml run is always idx 0.
            label = f"{name}_p0"
        configs = PipelineConfigs(steps, name=label)
        if configs.expansion_count != 1:
            return ""  # Defensive: any residual multi-variant expansion stays unlabeled.
        return configs.names[0]
    except Exception:  # noqa: BLE001 - an underivable pipeline drops the label rather than emit a wrong one.
        return ""


def _derive_variant_config_names(pipeline: Any, name: str) -> list[str]:
    """Derive the ORDERED legacy per-variant ``config_name``s for a SWEEP pipeline (#55), else ``[]``.

    For a generator / sweep, legacy assigns each EXPANDED variant its own ``"{label}_{display_hash}"``
    (``label`` = ``"config"`` unnamed / ``"{name}_p0"`` named), in :class:`PipelineConfigs` expand order.
    We REUSE that exact mechanism (no hand-rolled hash) — ``PipelineConfigs(steps).names`` IS that ordered
    list. The per-variant projection (:func:`~nirs4all.pipeline.dagml.result._scores_to_run_result`) maps
    these onto the CV variants positionally (the winner takes index 0 + the ``"_refit"`` suffix on its
    refit rows), so a sweep carries the legacy SET + count of config names with a legacy-correct winner
    label. Returns ``[]`` for a non-generator pipeline (the single path uses :func:`_derive_config_name`)
    or any underivable form, so the projection falls back to the blank ``config_name`` rather than guess.
    """
    try:
        from nirs4all.pipeline.config.pipeline_config import PipelineConfigs

        if isinstance(pipeline, PipelineConfigs):
            return list(pipeline.names)  # a pre-built PipelineConfigs already carries its per-variant names.
        steps = PipelineConfigs._load_steps(pipeline)  # noqa: SLF001 - reuse the exact legacy step-loading.
        label = "" if name == "" else f"{name}_p0"  # "" → the literal "config" prefix (unnamed legacy default).
        configs = PipelineConfigs(steps, name=label)
        if configs.expansion_count <= 1:
            return []  # not a multi-variant sweep — the single derived config_name path applies.
        return list(configs.names)
    except Exception:  # noqa: BLE001 - an underivable pipeline drops the labels rather than emit wrong ones.
        return []


def _native_param_variant_model_params(pipeline: Any, name: str) -> list[dict[str, Any]]:
    """The per-variant MODEL ``params`` dicts aligned 1:1 with :func:`_derive_variant_config_names`.

    For a NON-degenerate model param sweep (``_grid_``), legacy expands EACH grid variant into its own
    concrete model with that variant's specific params and selects the TRUE CV-best — so the winner's
    ``config_name`` is the WINNING variant's name, NOT ``names[0]``. The native dag-ml run likewise refits
    the true CV-best but emits an OPAQUE variant hash with no params in the reports, so the host recovers
    the winning variant by CONTENT: it matches the winner's refit model params against THIS ordered list
    of per-variant model params (the same :class:`PipelineConfigs` expansion ``_derive_variant_config_names``
    reads ``.names`` from, so element ``i`` is variant ``names[i]``'s model params).

    Each entry is the variant's model-step serialized ``params`` dict (``{"class", "params"}`` form —
    :func:`~nirs4all.pipeline.config.component_serialization.serialize_component` is what ``PipelineConfigs``
    stores), so a content match is an exact swept-param-value comparison. Returns ``[]`` for a non-sweep
    pipeline or any underivable form (the caller then falls back to the positional ``names[0]`` pairing).
    """
    try:
        from nirs4all.pipeline.config.pipeline_config import PipelineConfigs

        if isinstance(pipeline, PipelineConfigs):
            configs = pipeline
        else:
            steps = PipelineConfigs._load_steps(pipeline)  # noqa: SLF001 - reuse the exact legacy step-loading.
            label = "" if name == "" else f"{name}_p0"
            configs = PipelineConfigs(steps, name=label)
        if configs.expansion_count <= 1:
            return []
        params_per_variant: list[dict[str, Any]] = []
        for variant_steps in configs.steps:
            model_params: dict[str, Any] = {}
            for step in variant_steps:
                if isinstance(step, dict) and "model" in step:
                    model = step["model"]
                    # PipelineConfigs stores the expanded model as the serialized {"class", "params"} dict.
                    model_params = dict(model["params"]) if isinstance(model, dict) and isinstance(model.get("params"), dict) else (model.get_params() if hasattr(model, "get_params") else {})
                    break
            params_per_variant.append(model_params)
        return params_per_variant
    except Exception:  # noqa: BLE001 - an underivable pipeline drops the param recovery (positional fallback).
        return []


def _can_unwrap_preprocessing_step(step: Any) -> bool:
    """Whether a ``{"preprocessing": ...}`` wrapper is proven equivalent to a bare transform.

    Legacy ``StepParser`` accepts the explicit ``{"preprocessing": op}`` keyword wrapper as a pure
    synonym for a bare transform step (``RESERVED_KEYWORDS`` never includes ``preprocessing``). The dag-ml
    bridge's :func:`~nirs4all.pipeline.dagml_bridge._step_to_dsl` lowers only bare operator instances and
    the structural keywords (``model``/``y_processing``/``concat_transform``/``feature_augmentation``/
    generators), so a ``{"preprocessing": op}`` dict hits its fail-loud "does not serialize step keyword(s)"
    path → legacy fallback. A wrapper carrying ONLY the ``preprocessing`` key is a pure synonym — the
    operator fits/transforms identically to the bare form on the native X-chain.

    Two modifier-bearing wrappers are also equivalent for the CURRENT native concrete path:

    * ``force_layout='2d'`` on a preprocessing step is not consumed by legacy preprocessing controllers
      (only model controllers read ``ParsedStep.force_layout``), and the native sklearn path already
      materializes the model input as 2D.
    * ``fit_on_all=True`` is equivalent only for stateless transforms: fitting on all rows vs fold-train
      rows cannot change learned state when the existing leakage gate proves the operator is stateless.

    Anything else remains wrapped, then fails loud in :func:`_unsupported_fallback_reason`; this prevents
    silent native runs for stateful fit-scope changes, non-2D layouts, NA modifiers, names, or any unproven
    composition.
    """
    if not isinstance(step, dict) or "preprocessing" not in step:
        return False
    keys = set(step)
    if keys == {"preprocessing"}:
        return True
    modifiers = keys - {"preprocessing"}
    if modifiers == {"force_layout"}:
        return step.get("force_layout") == "2d"
    if modifiers == {"fit_on_all"}:
        return step.get("fit_on_all") is True and _operator_is_stateless(step["preprocessing"])
    return False


_CHART_STEP_KEYWORDS = frozenset(
    {
        "chart_2d",
        "chart_3d",
        "2d_chart",
        "3d_chart",
        "y_chart",
        "chart_y",
        "fold_chart",
        "chart_fold",
        "augment_chart",
        "augmentation_chart",
        "augment_details_chart",
        "augmentation_details_chart",
        "exclusion_chart",
        "chart_exclusion",
        "spectra_dist",
        "spectral_distribution",
        "spectra_envelope",
    }
)


def _is_chart_step(step: Any) -> bool:
    """Whether ``step`` is one of the legacy chart-only side-effect commands."""
    if isinstance(step, str):
        return step in _CHART_STEP_KEYWORDS or step.startswith("fold_")
    if isinstance(step, dict):
        return any(str(key) in _CHART_STEP_KEYWORDS or str(key).startswith("fold_") for key in step)
    return False


def _strip_chart_steps(pipeline: list[Any]) -> list[Any]:
    """Separate presentation commands from the numerical DAG; render after scoring."""
    return [step for step in pipeline if not _is_chart_step(step)]


def _unwrap_preprocessing_steps(pipeline: list[Any]) -> list[Any]:
    """Unwrap preprocessing wrappers whose legacy semantics match a bare native transform.

    The caller derives ``config_name`` from the ORIGINAL (wrapped) pipeline before unwrapping, so the
    dag-ml ``RunResult`` keeps the legacy-matching config name.
    """
    return [step["preprocessing"] if _can_unwrap_preprocessing_step(step) else step for step in pipeline]


def _unsupported_fallback_reason(pipeline: list[Any]) -> str | None:
    """Why an unhandled raw DSL shape must fall back before the generic concrete path.

    Called only AFTER the native composition detectors have had first refusal. At that point any remaining
    ``branch`` / ``merge`` / modifier-bearing ``preprocessing`` step is not a plain concrete transform/model
    pipeline: letting it fall through to ``_run_concrete_scores`` either drops semantics or fails later as an
    unrelated runtime error. Return a catchable, explicit coverage-boundary reason instead.
    """
    for step in pipeline:
        if isinstance(step, dict) and "preprocessing" in step and set(step) != {"preprocessing"}:
            modifiers = sorted(set(step) - {"preprocessing"})
            return (
                "engine='dag-ml' cannot yet honor this modifier-bearing {'preprocessing': ...} step "
                f"(modifier key(s): {modifiers}); only modifier-free wrappers plus proven stateless/2D "
                "modifier wrappers run natively."
            )

    for step in pipeline:
        if isinstance(step, dict) and "branch" in step:
            return (
                "engine='dag-ml' does not yet support this raw branch/merge composition: it did not match "
                "the native separation, by_source-fusion, duplication-fusion, or stacking detectors, so "
                "running the generic concrete path would drop branch semantics."
            )

    for step in pipeline:
        if not (isinstance(step, dict) and "merge" in step):
            continue
        spec = step["merge"]
        if isinstance(spec, dict) and spec.get("sources") == "concat":
            return (
                "engine='dag-ml' does not yet support {'merge': {'sources': 'concat'}}: the native "
                "multi-source path materializes early-fusion blocks directly and cannot reproduce legacy's "
                "source-concat merge boundary."
            )
        return (
            f"engine='dag-ml' does not yet support raw merge step {spec!r} outside a handled native branch "
            "composition."
        )

    return None


def _dispatch_run(
    pipeline: Any,
    spectro: Any,
    base_dir: Path,
    dataset_arg: str,
    host_pickle: str | None,
    cli: str,
    venv_python: str | None,
    name: str = "",
    random_state: int | None = None,
    save_charts: bool = True,
    plots_visible: bool = False,
) -> RunResult:
    """Route the materialized run to the matching native dag-ml path and map its scores.

    Extracted from :func:`run_via_dagml` so the many ``return _run_*(...)`` dispatch points all run
    under the caller's ``try/finally`` temp-dir cleanup (Python runs ``finally`` on every return
    path). All sub-paths write only under ``base_dir``; the returned RunResult is built in-memory.

    ``name`` (the run() pipeline label) is DERIVED into the canonical legacy ``config_name`` via
    :func:`_derive_config_name` (``PipelineConfigs``: ``config_{hash}`` unnamed / ``{name}_p0_{hash}``
    named for a genuinely-CONCRETE pipeline; ``""`` for ANY generator / sweep path — native
    ``param_model`` sweeps, the ``_expand_operator_generators`` multi-variant path, and ``_or_`` /
    ``_range_`` / ``_grid_`` / param-keyed — whose winner-only projection is not cleanly variant-mappable,
    #55) and forwarded to every ``_run_*`` path, so the native RunResult's predictions carry the SAME
    ``config_name`` legacy would set — including the downstream ``_refit`` suffix on the refit rows.

    ``random_state`` is forwarded to every ``_run_*`` path so the SUBPROCESS adapter's fresh-python global
    RNG is seeded per call via cli_runner's child env (no process-global mutation); the in-process path
    ignores it (the parent is already seeded in :func:`run_via_dagml`).
    """
    from nirs4all.core import detect_task_type

    pipeline, finetune_overrides = _lower_public_finetune_params(pipeline)
    reject_native_training_param_overrides(list(pipeline), context="engine='dag-ml'")
    config_name = _derive_config_name(pipeline, name)
    # The ordered legacy per-variant config names for a SWEEP (empty for a single concrete pipeline). The
    # native-generation and operator-expand paths below project EVERY variant's CV rows (legacy
    # num_predictions parity), labeling them with these (the winner takes index 0 + the "_refit" suffix).
    variant_config_names = _derive_variant_config_names(pipeline, name)
    # The per-variant MODEL params aligned 1:1 with `variant_config_names` — the native param-sweep path
    # matches the winner's refit model params against these to recover the WINNING variant's config_name
    # (a non-degenerate `_grid_` selects the true CV-best, not index 0). Empty for a non-sweep pipeline.
    variant_model_params = _native_param_variant_model_params(pipeline, name)

    is_classification = "classif" in str(detect_task_type(np.asarray(spectro.y({"partition": "train"}))))
    # CV-selection metric MUST mirror legacy Predictions._resolve_effective_metric: its DEFAULT for a
    # classification candidate is `balanced_accuracy` (NOT plain `accuracy`), so a classification run on
    # dag-ml ranks/reports the SAME metric legacy does (#60). dag-ml-core exposes a native
    # `BalancedAccuracy` kind reachable via `--selection-metric balanced_accuracy` (CLI) and the in-process
    # bridge's `parse_selection_metric`. Regression stays `rmse`.
    metric = "balanced_accuracy" if is_classification else "rmse"
    if "selection_metric" in finetune_overrides:
        metric = finetune_overrides["selection_metric"]
    if "selection_objective" in finetune_overrides:
        expected_objective = _metric_objective(metric)
        if finetune_overrides["selection_objective"] != expected_objective:
            raise NotImplementedError(
                f"engine='dag-ml' does not yet support overriding the native selection direction for metric {metric!r}; use direction={expected_objective!r} or choose a metric with the desired objective."
            )
    task_type = "classification" if is_classification else "regression"

    # Unwrap proven-equivalent `{"preprocessing": op}` wrappers to bare operators BEFORE detection/dispatch
    # (the bridge only lowers bare operators). Stateful `fit_on_all`, non-2D `force_layout`, NA modifiers, and
    # other unproven wrappers stay as dicts and still fall back loudly. `config_name` / `variant_config_names`
    # / `variant_model_params` were derived from the ORIGINAL pipeline above, so the dag-ml RunResult keeps
    # the legacy-matching name; `_attach_export_spec` likewise sees the original.
    pipeline = _strip_chart_steps(list(pipeline))
    from .public_normalization import normalize_model_steps

    pipeline = normalize_model_steps(pipeline)
    pipeline = _unwrap_preprocessing_steps(list(pipeline))
    if not any(_is_split_step(step) for step in pipeline):
        from .full_train import run_full_train

        return run_full_train(pipeline, spectro, metric=metric, task_type=task_type, config_name=config_name)

    # Detect the special-composition steps UP FRONT so the repetition guard below can reject an
    # unsupported combination BEFORE any non-group dispatch path (branch/augmentation/exclude) runs.
    detected = _detect_separation_branch(list(pipeline))
    detected_separation_preproc_concat = _detect_separation_preproc_concat(list(pipeline))
    detected_duplication = _detect_duplication_branch(list(pipeline))
    detected_stacking = _detect_stacking_branch(list(pipeline))
    detected_named_metamodel_stack = _detect_named_metamodel_feature_stack(list(pipeline))
    detected_by_source = _detect_by_source_branch(list(pipeline), spectro.features_sources())
    detected_by_source_concat = _detect_by_source_concat_shared_preproc(list(pipeline), spectro.features_sources())
    detected_by_source_distinct_concat = _detect_by_source_distinct_preproc_concat(list(pipeline), spectro.features_sources())
    detected_by_source_stacking = _detect_by_source_stacking_branch(list(pipeline), spectro.features_sources())
    detected_rep_fusion = _detect_rep_fusion(list(pipeline))
    detected_source_concat = _detect_source_concat_merge(list(pipeline), spectro.features_sources())
    augmentation_steps = [step for step in pipeline if _is_augmentation_step(step)]

    if _has_finetune_params(list(pipeline)):
        raise NotImplementedError("engine='dag-ml' did not lower finetune_params before native dispatch; this is an internal routing bug, not a supported execution path.")
    # REP FUSION (`rep_to_sources` / `rep_to_pp`, #31): a one-time HOST RESHAPE that turns each replicate
    # of a physical sample into a feature SOURCE (→ MULTI-SOURCE early fusion S3 / MB-PLS S5) or a
    # PROCESSING layer (→ the feature-axis concat S6). After the reshape the unit of analysis is the
    # physical SAMPLE (folds/OOF sample-grain — distinct from the plain repetition rep-grain path #21,
    # below). Detected BEFORE the repetition guard because the reshape CONSUMES the rep grouping (the
    # reshaped dataset is no longer a repetition dataset); the reshape feeds the already-native
    # multi-source / feature-concat materialization, pickled for the adapter (the on-disk dataset has no
    # such structure). A reshape combined with branch/exclude/augmentation is rejected by `_detect_rep_fusion`
    # (returns None) and falls through to the bridge's fail-loud path naming #31.
    if detected_rep_fusion is not None:
        return _run_rep_fusion(
            list(pipeline),
            detected_rep_fusion,
            spectro,
            dataset_arg,
            cli,
            venv_python or sys.executable,
            base_dir / "rep_fusion",
            metric,
            task_type,
            config_name=config_name,
            variant_config_names=variant_config_names,
            is_classification=is_classification,
            random_state=random_state,
        )

    # REPETITIONS (sample-grain grouping): when the dataset declares a repetition column, several stored
    # rows share one physical sample. The split must be GROUP-aware — all replicates of a sample land on
    # the SAME fold side — and each rep row is scored individually (the repetition grain), which is what
    # nirs4all's `cv_best_score`/`best_rmse` report (the sample-level `_agg` aggregation is a separate twin
    # entry, NOT those scores). Folds are over the rep ROWS, group-grouped (a clean OOF partition), and the
    # envelope emits `group_id` so dag-ml-data refuses any fold that splits a group. The first slice handles
    # the supported transform+model+splitter shape only.
    #
    # This guard runs BEFORE the branch/augmentation/exclude dispatch below: those paths build folds
    # WITHOUT the group constraint, so a repetition dataset reaching them could split a sample's reps
    # across train/val (silent leakage). An unhandled composition therefore fails LOUD here (naming #21)
    # rather than taking a non-group path and running wrong.
    if _is_repetition_dataset(spectro):
        if is_classification and getattr(spectro, "aggregate_method", None) == "vote":
            raise NotImplementedError(
                "engine='dag-ml' does not yet support classification repetition datasets with "
                "sample-level vote aggregation; the final-test surface would be scored at the "
                "repetition row grain instead of the legacy sample-vote grain (backlog #21)."
            )
        if (
            augmentation_steps
            or detected is not None
            or detected_separation_preproc_concat is not None
            or detected_duplication is not None
            or detected_stacking is not None
            or detected_named_metamodel_stack is not None
            or detected_by_source is not None
            or detected_by_source_concat is not None
            or detected_by_source_distinct_concat is not None
            or detected_by_source_stacking is not None
            or detected_source_concat is not None
            or any(_is_exclude_step(step) for step in pipeline)
        ):
            raise NotImplementedError(
                "engine='dag-ml' does not yet support a repetition dataset combined with "
                "exclude/branch/sample_augmentation (the group constraint would be lost); backlog #21."
            )
        return _run_repetition(
            list(pipeline), spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "repetition", metric, task_type, dataset_pickle=host_pickle, config_name=config_name, random_state=random_state
        )

    # by_metadata stateless preprocessing + concat feature reassembly + downstream model.
    # This path projects the branch feature boundary, then scores one downstream model on the
    # reassembled feature matrix; model-in-branch fan-out remains owned by `_run_separation_branch`.
    if detected_separation_preproc_concat is not None:
        branch_step, preproc_body, downstream_body = detected_separation_preproc_concat
        return _run_separation_preproc_concat(list(pipeline), branch_step, preproc_body, downstream_body, spectro, metric, task_type, config_name=config_name)

    # Separation branch (by_metadata/by_tag) + concat merge → ONE native fan-out run: dag-ml fans the
    # branch into one model node per partition value (discovered from the envelope metadata/tags),
    # runs per-partition FIT_CV, and the native concat-merge handler reassembles a full-universe OOF.
    # Detected on the ORIGINAL pipeline (before exclude consumption) so an exclude step beside the
    # branch is still visible — exclude+branch is rejected (out of scope) rather than silently dropped.
    if detected is not None:
        branch_step, branch_body = detected
        return _run_separation_branch(
            list(pipeline),
            branch_step,
            branch_body,
            spectro,
            dataset_arg,
            cli,
            venv_python or sys.executable,
            base_dir / "branch",
            metric,
            task_type,
            dataset_pickle=host_pickle,
            config_name=config_name,
            random_state=random_state,
        )

    # Duplication branch (`{"branch": [[A], [B], …]}`) + avg/mean fusion merge → ONE native run: each
    # branch is a full-data model node (NO fan-out / NO branch_view); dag-ml's native fusion merge handler
    # averages the branches' held-out Validation OOF per sample (leakage-safe) into one full-universe OOF.
    if detected_duplication is not None:
        branches, aggregate = detected_duplication
        return _run_duplication_branch(
            list(pipeline),
            branches,
            aggregate,
            spectro,
            dataset_arg,
            cli,
            venv_python or sys.executable,
            base_dir / "duplication",
            metric,
            task_type,
            dataset_pickle=host_pickle,
            config_name=config_name,
            random_state=random_state,
        )

    # by_source separation branch (`{"branch": {"by_source": True, "steps": [...model...]}}`) + avg/mean
    # fusion merge on a MULTI-source dataset → ONE native run: dag-ml fans the shared body into one
    # per-source model node (each bound to its source's block via metadata.source_index — LATE fusion
    # by source), and the native fusion merge handler averages the per-source held-out Validation OOF
    # per sample into one full-universe OOF. Each branch sees ALL samples but only ITS source's columns
    # (a feature-axis selection, not a sample partition like by_metadata).
    if detected_by_source is not None:
        by_source_body, by_source_aggregate = detected_by_source
        return _run_by_source_branch(
            list(pipeline),
            by_source_body,
            by_source_aggregate,
            spectro.features_sources(),
            spectro,
            dataset_arg,
            cli,
            venv_python or sys.executable,
            base_dir / "by_source",
            metric,
            task_type,
            dataset_pickle=host_pickle,
            config_name=config_name,
            random_state=random_state,
        )

    # by_source shared preprocessing + concat feature merge + one downstream model → ONE native run:
    # the downstream model's X-chain is applied independently per source, then hstacked, and the generic
    # projection is replicated per source to preserve legacy's branch-row bookkeeping.
    if detected_by_source_concat is not None:
        preproc_body, downstream_body = detected_by_source_concat
        return _run_by_source_concat_shared_preproc(list(pipeline), preproc_body, downstream_body, spectro.features_sources(), spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "by_source_concat", metric, task_type, dataset_pickle=host_pickle, config_name=config_name, random_state=random_state)

    # by_source per-source DICT preprocessing + concat feature merge + downstream model:
    # one native model node materializes source blocks, fits each source's transform chain
    # fold-locally, hstacks in source_layout order, then fits the downstream estimator.
    if detected_by_source_distinct_concat is not None:
        source_steps, downstream_body = detected_by_source_distinct_concat
        return _run_by_source_distinct_preproc_concat(
            list(pipeline),
            source_steps,
            downstream_body,
            spectro.features_sources(),
            spectro,
            dataset_arg,
            cli,
            venv_python or sys.executable,
            base_dir / "by_source_distinct_concat",
            metric,
            task_type,
            dataset_pickle=host_pickle,
            config_name=config_name,
            random_state=random_state,
        )

    # by_source per-source models + merge='predictions' + downstream Ridge is a legacy source-layout
    # replay, not ordinary 3-column OOF stacking: each source branch mutates its source in sequence, the
    # merge writes the cumulative source concat back to source 0, and the downstream model emits CV-only
    # rows because legacy skips the by_source stacking refit pass.
    if detected_by_source_stacking is not None:
        branch_body, meta_learner = detected_by_source_stacking
        return _run_by_source_stacking_branch(
            list(pipeline),
            branch_body,
            meta_learner,
            spectro.features_sources(),
            spectro,
            dataset_arg,
            cli,
            venv_python or sys.executable,
            base_dir / "by_source_stacking",
            metric,
            task_type,
            dataset_pickle=host_pickle,
            config_name=config_name,
            random_state=random_state,
        )

    # Top-level source concat (`X-transform* -> {"merge": {"sources": "concat"}} -> splitter -> model`) is
    # a native source-layout boundary: upstream X transforms run per source, then the transformed blocks are
    # concatenated for the downstream model. A plain early-fusion lowering would transform the already-concat
    # matrix and diverge for row-wise ops such as SNV.
    if detected_source_concat is not None:
        pre_merge_steps, post_merge_steps, source_indices = detected_source_concat
        return _run_source_concat_merge(
            pre_merge_steps,
            post_merge_steps,
            source_indices,
            spectro,
            dataset_arg,
            cli,
            venv_python or sys.executable,
            base_dir / "source_concat",
            metric,
            task_type,
            dataset_pickle=host_pickle,
            config_name=config_name,
            random_state=random_state,
        )

    # STACKING (backlog #10): a duplication branch (`{"branch": [[A], [B], …]}`) + `{"merge": "predictions"}`
    # + a downstream meta-model (`{"model": MetaModel(Ridge())}` or a plain `{"model": Ridge()}`) → ONE
    # native dag-ml run: each base branch model is FIT_CV on the full fold-train and predicts the full
    # fold-validation (held-out Validation OOF); the meta-node consumes those branches' Validation OOF
    # (via requires_oof+requires_fold_alignment edges, leakage-safe — train predictions are refused), fits
    # the meta-learner on the per-fold OOF meta-feature matrix and emits its own scored OOF.
    if detected_stacking is not None:
        branches, meta_learner = detected_stacking
        return _run_stacking_branch(
            list(pipeline),
            branches,
            meta_learner,
            spectro,
            dataset_arg,
            cli,
            venv_python or sys.executable,
            base_dir / "stacking",
            metric,
            task_type,
            dataset_pickle=host_pickle,
            config_name=config_name,
            random_state=random_state,
        )

    # Named duplication branches with a branch-local MetaModel, a structured per-branch best-by-RMSE
    # prediction merge into features, and one downstream estimator. Legacy emits a CV-only row table
    # (named-dict stacking skips refit), so this path projects that exact surface and no final rows.
    if detected_named_metamodel_stack is not None:
        branch_names, branches, meta_step, prediction_configs, downstream_step = detected_named_metamodel_stack
        return _run_named_metamodel_feature_stack(
            list(pipeline),
            branch_names,
            branches,
            meta_step,
            prediction_configs,
            downstream_step,
            spectro,
            metric,
            task_type,
            config_name=config_name,
        )

    # A STACKING merge that is NOT the handled shape above (a per-branch predictions config, a missing /
    # mis-ordered meta-model, a MetaModel carrying unhandled options) must fail LOUD here, naming #10,
    # rather than reach the bridge's generic raw-merge error — so the deferral stays explicit.
    if any(_is_stacking_merge_step(step) for step in pipeline) and any(_is_duplication_branch_step(step) for step in pipeline):
        raise NotImplementedError(
            "engine='dag-ml' supports STACKING only as a duplication branch + {'merge': 'predictions'} + "
            "a downstream meta-model ({'model': MetaModel(Ridge())} or {'model': Ridge()}) with default "
            "options; this richer stacking shape is not yet wired (backlog #10). Use {'merge': 'mean'} for "
            "an averaging (fusion) ensemble instead."
        )

    if (reason := _unsupported_fallback_reason(list(pipeline))) is not None:
        raise DagMlUnsupported(reason)

    # `sample_augmentation` → run nirs4all's REAL augmentation machinery to create the synthetic TRAIN
    # rows in the dataset, then run ONE native dag-ml CV+refit: base-grain folds (the synthetic children
    # never reach a holdout) + a CV-universe envelope carrying the children's origin/augmentation grain.
    # The model trains on base + its augmented children (host-side expansion); OOF is over base val only.
    # Detected on the ORIGINAL pipeline so it composes only with the supported transform+model+splitter
    # shape — a branch/exclude beside it is out of scope (the bridge fails loud below).
    #
    # Both leakage regimes run natively (`_run_augmentation` picks the path): a STATELESS augmenter is
    # augmented ONCE globally (#8, children shared across folds); a STATEFUL/SUPERVISED/BALANCED augmenter
    # is augmented FOLD-LOCALLY (#32, fit inside each fold's train only + a full-train refit pass), so it
    # never sees a fold's validation rows. A single augmentation step of either kind is supported here; an
    # unsupported richer shape still falls through to the bridge's raw `sample_augmentation` error.
    if augmentation_steps:
        return _run_augmentation(list(pipeline), spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "augment", metric, task_type, config_name=config_name, random_state=random_state)

    # Consume the `exclude` step (if any) BEFORE generator handling: run the SampleFilter operator(s)
    # in Python on the full CV train pool to get the excluded sample ints, then choose the CV universe
    # per the `keep_in_oof` mode. `cv_pool` is the sample-int universe the splitter runs over;
    # `excluded` is non-empty only in the opt-in (keep_in_oof=True) leakage-pure mode.
    pipeline, cv_pool, excluded = _resolve_exclude(list(pipeline), spectro)
    # Consume handled `tag` steps AFTER the CV universe is known: tags fit on that train pool and are
    # emitted onto relations, but do not remove samples from the splitter/model pool.
    pipeline, tags_by_sample = _resolve_tags(list(pipeline), spectro, cv_pool)

    # Param-level model sweeps (`_range_`/`_log_range_`/`_grid_` on a model step) run as ONE native
    # dag-ml run: the bridge lowers them to native `generators`, the compiler expands variants, and
    # dag-ml generates + scores + SELECTs + refits the best (no Python expand). Operator-level
    # generators (`_or_`/`_cartesian_`, multi-model) stay on the Python `expand_spec` path below.
    if _generation_kind(list(pipeline)) == "param_model":
        return _run_native_generation(
            list(pipeline),
            spectro,
            dataset_arg,
            cli,
            venv_python or sys.executable,
            base_dir / "native",
            metric,
            task_type,
            cv_pool,
            excluded,
            tags_by_sample,
            dataset_pickle=host_pickle,
            config_name=config_name,
            variant_config_names=variant_config_names,
            variant_model_params=variant_model_params,
            random_state=random_state,
        )

    # FLAT-SINGLE operator `_or_` (a bare-operator preprocessing sweep) → ONE native dag-ml operator-SELECT
    # run (#23 Phase 7): the bridge lowers the `_or_` to a compat Generator step, dag-ml compiles the
    # operator-variant models, and the in-process binding scores each choice by CV-OOF, refits the winner
    # only, and surfaces every variant's validation reports (each carrying a `variant_label` content
    # fingerprint). A richer `_or_` the lowering cannot handle (a non-flat-bare / non-finite / non-JSON
    # choice that slips the predicate) raises the DISTINCT `_OperatorLoweringUnsupported` sentinel from
    # `_run_native_operator_generation`'s narrow LOWERING guard, and we fall through to the Python
    # `expand_spec` path below (the INNER fallback, which STAYS on the dag-ml engine — it never bubbles to
    # legacy). Gated on `_generation_kind == "operator"` AND the conservative flat-single predicate so only
    # the canonical bare-`_or_` shape attempts native. ONLY the lowering sentinel is caught — a RUNTIME error
    # (incl. a runtime `DagMlUnsupported` from `_raise_run_failure` for a non-zero run classified unsupported,
    # OR a runtime NotImplementedError from compile / run / result-mapping) PROPAGATES, never silently
    # reclassified as a lowering gap and masked.
    # A CONSTRAINED operator generator (`_or_`-pick / `_cartesian_` with `_mutex_`/`_requires_`/`_exclude_`)
    # routes the SAME native operator-SELECT run (#1a + 1b-cartesian): the host expands the pruned survivor
    # set (`expand_spec`, the constraint source of truth both engines use) and lowers each survivor into ONE
    # model-terminated canonical Generator branch (`assemble_constrained_cv_refit_dsl`), so dag-ml scores the
    # SAME pruned set by CV-OOF, refits the winner, and stamps each per-variant report with the multi-op
    # `variant_label` the host recomputes byte-identically. dag-ml's own native constraint engine
    # (`prune_sequences_by_constraints`) produces the identical survivor set + labels, so the content-keyed
    # config map aligns. Both the flat-single and constrained shapes go through the SAME inner LOWERING guard:
    # a lowering refusal raises the DISTINCT `_OperatorLoweringUnsupported` sentinel and falls through to the
    # Python `expand_spec` path below (still dag-ml-native), while a RUNTIME error PROPAGATES.
    # An UNCONSTRAINED operator generator (`_or_`-pick/-arrange or a multi-stage `_cartesian_` with NO
    # `_mutex_`/`_requires_`/`_exclude_`) routes the SAME native operator-SELECT run (ADR-17 item 5 slice C):
    # its survivor set is simply ALL pick/arrange/cartesian combinations (no constraint prune), which dag-ml's
    # `expand_or_generator_sequences` / `expand_cartesian_generator_sequences` produce in legacy
    # `itertools.combinations`/`permutations` order, so the SAME `assemble_constrained_cv_refit_dsl` lowering
    # (with an EMPTY constraints set) + content-keyed config map align. arrange (ordered permutations) reaches
    # parity because dag-ml's permutation order matches legacy AND the config map is content-keyed; `then_*` /
    # `count` / `_seed_` / `_weights_` / oversize-pick / non-routable shapes are demoted by the predicate.
    if _generation_kind(list(pipeline)) == "operator" and (
        _is_flat_single_operator_generator(list(pipeline)) or _is_constrained_operator_generator(list(pipeline)) or _is_unconstrained_operator_generator(list(pipeline))
    ):
        try:
            return _run_native_operator_generation(
                list(pipeline),
                spectro,
                dataset_arg,
                cli,
                venv_python or sys.executable,
                base_dir / "native_op",
                metric,
                task_type,
                cv_pool,
                excluded,
                tags_by_sample,
                dataset_pickle=host_pickle,
                config_name=config_name,
                variant_config_names=variant_config_names,
                random_state=random_state,
            )
        except _OperatorLoweringUnsupported:
            pass  # lowering-unsupported generator → fall through to the Python expand path (stays on dag-ml)

    # Expand operator-level generators (_or_/_cartesian_/param-keyed _range_/_grid_/...) into concrete,
    # flat pipelines of live operator instances (nirs4all's own serialize → expand → deserialize +
    # flatten) and run each through the verified single-variant dag-ml path to get its native ScoreSet.
    # A single variant maps straight through; a sweep COMBINES every variant's ScoreSet into one
    # per-variant projection (legacy num_predictions parity) — selecting the best by CV (mirroring
    # nirs4all) and emitting the winner's refit rows only.
    variants = _expand_operator_generators(list(pipeline))
    variant_runs = [
        _run_concrete_scores(variant, spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / f"variant{index}", cv_pool, excluded, tags_by_sample, dataset_pickle=host_pickle, random_state=random_state)
        for index, variant in enumerate(variants)
    ]
    if len(variant_runs) == 1:
        # SINGLE concrete pipeline: thread the node results + minted identity into the projection so the
        # strict direct-block rows (per-fold val + refit final/test) carry real y_pred/y_true/sample_indices
        # (2a-i), plus the captured fitted REFIT estimators (2c-i) for native model-artifact persistence.
        # scores/skip_refit unchanged — num_predictions and scores are score-set-driven as before.
        scores, model_name, skip_refit, results, identity, refit_artifacts = variant_runs[0]
        return _scores_to_run_result(scores, spectro.name, model_name, metric, task_type, config_name=config_name, skip_refit=skip_refit, results=results, identity=identity, refit_artifacts=refit_artifacts)

    # Operator SWEEP (2a-ii): thread EACH variant's own node results + the (shared) identity into the
    # per-variant projection so every variant's direct-block rows carry ITS OWN y_pred/y_true/sample_indices
    # (winner: fold-val + refit final/test; losers: fold-val — every operator variant ran fully). The
    # projection re-keys the results by the synthetic variant tag it stamps on the reports, so a row's
    # arrays come from its own variant's blocks (NO cross-variant leakage). All variants ran on the same
    # `spectro`, so the identity is identical — take the first. The aggregated avg/w_avg rows stay
    # score-only (deferred to 2a-iii). scores/skip_refit unchanged — num_predictions stays score-set-driven.
    variant_scores = [(scores, model_name, skip_refit) for scores, model_name, skip_refit, _results, _identity, _artifacts in variant_runs]
    results_by_index = [results for _scores, _model_name, _skip_refit, results, _identity, _artifacts in variant_runs]
    refit_artifacts_by_index = [artifacts for _scores, _model_name, _skip_refit, _results, _identity, artifacts in variant_runs]
    identity = variant_runs[0][4]
    return _project_operator_sweep(
        variant_scores, spectro.name, metric, task_type, is_classification, variant_config_names, results_by_index=results_by_index, identity=identity, refit_artifacts_by_index=refit_artifacts_by_index
    )
