"""Run a nirs4all pipeline on the **dag-ml** engine and return a ``RunResult`` (ADR-17 backend).

This is the operational seam for ``engine="dag-ml"``: it assembles the executable compat DSL,
drives ``dag-ml-cli`` through the nirs4all process adapter, and maps dag-ml's **native**
``bundle.scores`` — per-fold validation RMSE/R², the cross-fold OOF average (``cv_best_score``)
and the final-test score (``best_rmse``), all computed in Rust — into an in-memory
:class:`~nirs4all.data.predictions.Predictions`, wrapped in a :class:`~nirs4all.api.result.RunResult`.

No workspace is created and no scoring happens Python-side: the numbers are dag-ml's. Supports the
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

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from nirs4all.api.result import RunResult

from .dataset import _dataset_inputs, _materialize_dataset
from .detect import (
    _detect_by_source_branch,
    _detect_duplication_branch,
    _detect_rep_fusion,
    _detect_separation_branch,
    _detect_stacking_branch,
    _fusion_merge_aggregate,
    _generation_kind,
    _is_augmentation_step,
    _is_duplication_branch_step,
    _is_exclude_step,
    _is_stacking_merge_step,
)
from .errors import DagMlUnsupported
from .exclude import _excluded_from_pool, _resolve_exclude, _resolve_tags
from .folds import _build_folds, _build_group_folds, _is_repetition_dataset, _repetition_groups_for_pool
from .run_paths import (
    _FUSION_MERGE_NODE_ID,
    _augmentation_is_leakage_free,
    _canonical_source_branch,
    _operator_is_stateless,
    _reshape_for_rep_fusion,
    _run_augmentation,
    _run_by_source_branch,
    _run_concrete,
    _run_duplication_branch,
    _run_native_generation,
    _run_rep_fusion,
    _run_repetition,
    _run_separation_branch,
    _run_stacking_branch,
)
from .steps import _expand_operator_generators

_DEFAULT_CLI = Path(__file__).resolve().parents[4] / "dag-ml" / "target" / "release" / "dag-ml-cli"

# Names re-exported for the stable ``nirs4all.pipeline.dagml.run_backend`` import surface (the parity
# suite and any caller import these private helpers directly from this module).
__all__ = [
    "DagMlUnsupported",
    "_FUSION_MERGE_NODE_ID",
    "_augmentation_is_leakage_free",
    "_build_folds",
    "_build_group_folds",
    "_canonical_source_branch",
    "_detect_by_source_branch",
    "_detect_duplication_branch",
    "_detect_rep_fusion",
    "_detect_separation_branch",
    "_detect_stacking_branch",
    "_excluded_from_pool",
    "_fusion_merge_aggregate",
    "_generation_kind",
    "_is_augmentation_step",
    "_is_stacking_merge_step",
    "_operator_is_stateless",
    "_repetition_groups_for_pool",
    "_reshape_for_rep_fusion",
    "_resolve_exclude",
    "_run_rep_fusion",
    "run_via_dagml",
]


# Residual ``run()`` ``**runner_kwargs`` keys the dag-ml path POSITIVELY honors (acts on). The dag-ml
# backend has its OWN (Rust) execution and never builds a ``PipelineRunner``, so NO ``PipelineRunner`` /
# ``runner.run()`` option (n_jobs, continue_on_error, max_generation_count, log_*, workspace_path, …)
# reaches it — the allowlist is EMPTY. Any key present therefore cannot be honored and must fall back to
# legacy (which DOES build a PipelineRunner from these), rather than be silently dropped. Kept as an
# explicit name so the rule is visible and a future genuinely-no-op key can be added with proof.
_HONORED_RUNNER_KWARGS: frozenset[str] = frozenset()

# Persistence keys that get a CLEARER message than the generic catch-all (they are instances of the same
# "not honored → fall back" rule, just with a specific reason). Order: these first, then the catch-all.
_PERSISTENCE_REJECT_MESSAGES: dict[str, str] = {
    "workspace_path": "engine='dag-ml' creates no workspace, so workspace_path is unsupported; falling back to legacy.",
    "store_run_id": "engine='dag-ml' creates no workspace store, so store_run_id is unsupported; falling back to legacy.",
    "keep_datasets": "engine='dag-ml' keeps no dataset snapshots, so keep_datasets is unsupported; falling back to legacy.",
}


def _reject_unsupported_run_options(*, refit: Any, project: str | None, session: Any, cache: Any, runner_kwargs: dict[str, Any]) -> None:
    """Raise :class:`DagMlUnsupported` for any run() option the scores-only dag-ml path cannot honor.

    Forwarding-not-dropping contract (P1b): every option the dag-ml engine genuinely cannot satisfy
    must raise a CATCHABLE error so ``run()`` falls back to legacy, never silently ignore the user's
    choice. Only NON-DEFAULT, un-honorable values raise — defaults run natively, so the common
    ``engine='dag-ml'`` run is untouched:

    * ``refit`` — the engine ALWAYS runs native CV+refit on the single CV winner; only the default
      ``True`` is honorable. ``False``/``None`` (disable) and a custom selection ``dict``/``list``
      (top_k/ranking) cannot be expressed, so they fall back.
    * ``project`` — tagging a run needs the workspace store the in-memory path never creates.
    * ``session`` — a Session carries a live runner with its OWN workspace + configuration that legacy
      reuses across runs; the scores-only path creates no runner/workspace and cannot share one, so a
      non-``None`` session falls back to legacy (which honors ``session.runner``).
    * ``cache`` — legacy installs the CacheConfig on the runner (→ the orchestrator's StepCache); the
      in-memory dag-ml path runs no nirs4all StepCache, so a non-``None`` cache falls back to legacy.
    * ``runner_kwargs`` — the residual ``**runner_kwargs`` are ``PipelineRunner`` / ``runner.run()``
      options (``n_jobs``, ``continue_on_error``, ``max_generation_count``, ``log_*``, ``workspace_path``,
      ``store_run_id``, ``keep_datasets``, …). The dag-ml path builds no ``PipelineRunner``, so it honors
      NONE of them (:data:`_HONORED_RUNNER_KWARGS` is empty). Rather than enumerate each (whack-a-mole),
      ANY key not in the allowlist falls back to legacy — guaranteeing no runner_kwarg is ever silently
      dropped. The well-known persistence keys get a clearer message; the rest fall back by the
      generic rule naming the key.
    """
    if refit is not True:
        raise DagMlUnsupported(
            f"engine='dag-ml' always runs native CV+refit on the single CV winner and cannot honor "
            f"refit={refit!r} (disable / custom top-k / ranking selection); falling back to legacy."
        )
    if project is not None:
        raise DagMlUnsupported(
            f"engine='dag-ml' returns scores in memory and creates no workspace store, so it cannot tag "
            f"the run with project={project!r}; falling back to legacy."
        )
    if session is not None:
        raise DagMlUnsupported(
            "engine='dag-ml' creates no runner/workspace and cannot share a Session's runner+workspace; "
            "falling back to legacy (which reuses session.runner)."
        )
    if cache is not None:
        raise DagMlUnsupported(
            "engine='dag-ml' runs no nirs4all StepCache, so it cannot honor a CacheConfig; falling back to legacy."
        )
    # General no-silent-drop rule: ANY runner_kwarg the dag-ml path does not POSITIVELY honor falls back.
    for key in runner_kwargs:
        if key in _HONORED_RUNNER_KWARGS:
            continue
        if key in _PERSISTENCE_REJECT_MESSAGES:
            raise DagMlUnsupported(_PERSISTENCE_REJECT_MESSAGES[key])
        raise DagMlUnsupported(
            f"engine='dag-ml' builds no PipelineRunner and does not honor the run() option {key!r}; "
            f"falling back to legacy (which does)."
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
) -> RunResult:
    """Execute ``pipeline`` on ``dataset`` via dag-ml-cli; return a RunResult of dag-ml's native scores.

    Args:
        pipeline: nirs4all pipeline (feature transforms, one ``{"model": ...}`` step, and a splitter).
        dataset: Anything :class:`~nirs4all.data.config.DatasetConfigs` accepts (path/config).
        name: Legacy ``run()`` pipeline label. Legacy does NOT use it raw — ``PipelineConfigs`` derives
            the ``config_name`` as ``"config_{hash}"`` (unnamed) / ``"{name}_p0_{hash}"`` (named), with a
            ``"_refit"`` suffix on the refit rows. HONORED here (not rejected) by DERIVING that exact
            canonical name (:func:`_derive_config_name`, reusing ``PipelineConfigs`` — no hand-rolled
            hash) and carrying it on the dag-ml RunResult predictions, so a single-variant run keeps the
            same ``config_name`` on both engines. A GENERATOR pipeline (winner-only projection, not
            cleanly variant-mappable) carries ``""`` rather than a wrong name (#55).
        random_state: Global random seed, applied exactly as legacy ``run()`` does (numpy / random /
            TF / torch / sklearn) so the stochastic paths (sample augmentation, randomized splitters,
            and any unseeded stochastic operator) are reproducible on the dag-ml engine too. Seeds the
            PARENT process (covers the in-process path) AND is THREADED to cli_runner, which seeds the
            subprocess worker via the PER-CALL child env (``N4A_RANDOM_STATE``) — no ``os.environ``
            mutation, so no stale-seed leak or concurrency hazard.
        refit: Legacy ``run()`` refit option. The dag-ml engine ALWAYS runs native CV+refit on the
            single CV winner, so only ``True`` (the default) is honored; any other value (``False`` /
            ``None`` to disable, or a custom top-k/ranking ``dict``/``list``) raises
            :class:`DagMlUnsupported` so ``run()`` falls back to legacy rather than silently ignore it.
        project: Legacy ``run()`` project tag. Tagging requires the workspace store, which the dag-ml
            path does not create; a non-``None`` project raises :class:`DagMlUnsupported` → legacy.
        session: Legacy ``run()`` Session. It carries a live runner + workspace legacy reuses across
            runs; the scores-only path creates none, so a non-``None`` session raises
            :class:`DagMlUnsupported` → legacy (which honors ``session.runner``).
        cache: Legacy ``run()`` CacheConfig. Legacy installs it on the runner's StepCache; the dag-ml
            path runs no StepCache, so a non-``None`` cache raises :class:`DagMlUnsupported` → legacy.
        runner_kwargs: The residual legacy ``run()`` ``**runner_kwargs``. Any key whose value requests
            on-disk persistence the in-memory dag-ml path cannot provide (``workspace_path``,
            ``store_run_id``, ``keep_datasets=True``) raises :class:`DagMlUnsupported` → legacy; the
            scores-only-irrelevant keys are accepted.
        dagml_cli: Path to the ``dag-ml-cli`` binary (defaults to the sibling ``dag-ml`` build).
        venv_python: Python interpreter the process adapter re-execs under (defaults to the current).
        workdir: Scratch directory for the run inputs/outputs (defaults to a temp dir).

    Returns:
        A :class:`~nirs4all.api.result.RunResult` whose ``best_rmse`` is the native final-test score
        and ``cv_best_score`` is the native cross-fold OOF average.
    """
    # Validate the run() options the scores-only in-memory path cannot honor BEFORE any work: a
    # non-honorable non-default raises DagMlUnsupported so run() falls back to legacy (never a silent
    # drop). Defaults pass through untouched, so a plain engine='dag-ml' run is unaffected.
    _reject_unsupported_run_options(refit=refit, project=project, session=session, cache=cache, runner_kwargs=runner_kwargs or {})

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

    cli = str(dagml_cli or _DEFAULT_CLI)
    if not Path(cli).exists():
        raise FileNotFoundError(f"dag-ml-cli binary not found at {cli}; build it (cargo build -p dag-ml-cli --release)")

    # Materialize the host dataset from ANY input legacy `run()` accepts (path / config /
    # DatasetConfigs / live SpectroDataset / (X, y) tuple / array) — DatasetConfigs alone silently
    # skips the in-memory ones, so `_materialize_dataset` wraps them with the legacy normalization.
    spectro = _materialize_dataset(dataset)
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
    try:
        return _dispatch_run(pipeline, spectro, base_dir, dataset_arg, host_pickle, cli, venv_python, name, random_state)
    finally:
        if workdir is None:
            shutil.rmtree(base_dir, ignore_errors=True)


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

    config_name = _derive_config_name(pipeline, name)

    is_classification = "classif" in str(detect_task_type(np.asarray(spectro.y({"partition": "train"}))))
    metric = "accuracy" if is_classification else "rmse"
    task_type = "classification" if is_classification else "regression"

    # Detect the special-composition steps UP FRONT so the repetition guard below can reject an
    # unsupported combination BEFORE any non-group dispatch path (branch/augmentation/exclude) runs.
    detected = _detect_separation_branch(list(pipeline))
    detected_duplication = _detect_duplication_branch(list(pipeline))
    detected_stacking = _detect_stacking_branch(list(pipeline))
    detected_by_source = _detect_by_source_branch(list(pipeline), spectro.features_sources())
    detected_rep_fusion = _detect_rep_fusion(list(pipeline))
    augmentation_steps = [step for step in pipeline if _is_augmentation_step(step)]

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
        return _run_rep_fusion(list(pipeline), detected_rep_fusion, spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "rep_fusion", metric, task_type, config_name=config_name, random_state=random_state)

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
        if augmentation_steps or detected is not None or detected_duplication is not None or detected_stacking is not None or detected_by_source is not None or any(_is_exclude_step(step) for step in pipeline):
            raise NotImplementedError(
                "engine='dag-ml' does not yet support a repetition dataset combined with "
                "exclude/branch/sample_augmentation (the group constraint would be lost); backlog #21."
            )
        return _run_repetition(list(pipeline), spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "repetition", metric, task_type, dataset_pickle=host_pickle, config_name=config_name, random_state=random_state)

    # Separation branch (by_metadata/by_tag) + concat merge → ONE native fan-out run: dag-ml fans the
    # branch into one model node per partition value (discovered from the envelope metadata/tags),
    # runs per-partition FIT_CV, and the native concat-merge handler reassembles a full-universe OOF.
    # Detected on the ORIGINAL pipeline (before exclude consumption) so an exclude step beside the
    # branch is still visible — exclude+branch is rejected (out of scope) rather than silently dropped.
    if detected is not None:
        branch_step, branch_body = detected
        return _run_separation_branch(list(pipeline), branch_step, branch_body, spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "branch", metric, task_type, dataset_pickle=host_pickle, config_name=config_name, random_state=random_state)

    # Duplication branch (`{"branch": [[A], [B], …]}`) + avg/mean fusion merge → ONE native run: each
    # branch is a full-data model node (NO fan-out / NO branch_view); dag-ml's native fusion merge handler
    # averages the branches' held-out Validation OOF per sample (leakage-safe) into one full-universe OOF.
    if detected_duplication is not None:
        branches, aggregate = detected_duplication
        return _run_duplication_branch(list(pipeline), branches, aggregate, spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "duplication", metric, task_type, dataset_pickle=host_pickle, config_name=config_name, random_state=random_state)

    # by_source separation branch (`{"branch": {"by_source": True, "steps": [...model...]}}`) + avg/mean
    # fusion merge on a MULTI-source dataset → ONE native run: dag-ml fans the shared body into one
    # per-source model node (each bound to its source's block via metadata.source_index — LATE fusion
    # by source), and the native fusion merge handler averages the per-source held-out Validation OOF
    # per sample into one full-universe OOF. Each branch sees ALL samples but only ITS source's columns
    # (a feature-axis selection, not a sample partition like by_metadata).
    if detected_by_source is not None:
        by_source_body, by_source_aggregate = detected_by_source
        return _run_by_source_branch(list(pipeline), by_source_body, by_source_aggregate, spectro.features_sources(), spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "by_source", metric, task_type, dataset_pickle=host_pickle, config_name=config_name, random_state=random_state)

    # STACKING (backlog #10): a duplication branch (`{"branch": [[A], [B], …]}`) + `{"merge": "predictions"}`
    # + a downstream meta-model (`{"model": MetaModel(Ridge())}` or a plain `{"model": Ridge()}`) → ONE
    # native dag-ml run: each base branch model is FIT_CV on the full fold-train and predicts the full
    # fold-validation (held-out Validation OOF); the meta-node consumes those branches' Validation OOF
    # (via requires_oof+requires_fold_alignment edges, leakage-safe — train predictions are refused), fits
    # the meta-learner on the per-fold OOF meta-feature matrix and emits its own scored OOF.
    if detected_stacking is not None:
        branches, meta_learner = detected_stacking
        return _run_stacking_branch(list(pipeline), branches, meta_learner, spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "stacking", metric, task_type, dataset_pickle=host_pickle, config_name=config_name, random_state=random_state)

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
            list(pipeline), spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "native", metric, task_type, cv_pool, excluded, tags_by_sample, dataset_pickle=host_pickle, config_name=config_name, random_state=random_state
        )

    # Expand operator-level generators (_or_/_cartesian_/param-keyed _range_/_grid_/...) into concrete,
    # flat pipelines of live operator instances (nirs4all's own serialize → expand → deserialize +
    # flatten), run each through the verified single-variant dag-ml path, then select the best by its
    # CV score — mirroring nirs4all selecting the best variant by its metric.
    variants = _expand_operator_generators(list(pipeline))
    results = [
        _run_concrete(variant, spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / f"variant{index}", metric, task_type, cv_pool, excluded, tags_by_sample, dataset_pickle=host_pickle, config_name=config_name, random_state=random_state)
        for index, variant in enumerate(variants)
    ]
    if len(results) == 1:
        return results[0]

    def _cv_rank(result: RunResult) -> float:
        score = result.cv_best_score
        if score != score:  # NaN (no CV score) ranks last
            return float("inf")
        return -score if is_classification else score  # maximize accuracy, minimize RMSE

    return min(results, key=_cv_rank)
