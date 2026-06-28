"""Native dag-ml run paths for the host backend.

One ``_run_*`` function per supported pipeline shape — single-variant CV+refit, native param-model
generation, repetition (group-aware), rep-fusion reshape, sample-augmentation (global / fold-local),
and the separation / by_source / duplication / stacking branch fan-outs. Each assembles the compat or
canonical DSL, drives dag-ml-cli through the process adapter, and maps the native ``bundle.scores``
into a :class:`~nirs4all.api.result.RunResult`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nirs4all.api.result import RunResult
from nirs4all.pipeline.dagml_bridge import controller_manifests

from .cli_runner import assemble_cv_refit_dsl
from .detect import _is_augmentation_step, _is_rep_fusion_step
from .envelope import build_envelope
from .errors import DagMlUnsupported, _raise_run_failure, _reject_multi_model
from .folds import _build_folds, _build_group_folds, _repetition_grain, _split_pool
from .identity import mint_identity
from .in_process_runner import run_cv_refit_bundle_router as run_cv_refit_bundle
from .result import _scores_to_run_result
from .steps import _apply_model_params, _apply_plain_model_params, _assert_supported_operators, _model_name, _split_pipeline, _supported_body_steps


def _run_native_generation(
    pipeline: list[Any],
    spectro: Any,
    dataset_arg: str,
    cli: str,
    venv_python: str,
    run_dir: Path,
    metric: str,
    task_type: str,
    cv_pool: list[int] | None = None,
    excluded: set[int] | None = None,
    tags_by_sample: dict[int, list[str]] | None = None,
    dataset_pickle: str | None = None,
    config_name: str = "",
    random_state: int | None = None,
) -> RunResult:
    """Run a param-level model sweep as ONE native dag-ml generation + SELECT + refit run.

    The model step keeps its generator dict so the bridge lowers it to native ``generators``; we
    apply only the plain (non-generator) sibling params to the model, never the sweep. dag-ml
    expands the variants, scores each by its cross-fold OOF ``metric``, and refits the winner —
    ``bundle.scores`` is the selected variant's, mapped to a RunResult exactly like the single path.

    ``cv_pool`` is the CV sample-int universe (the de-excluded pool in legacy mode, the full train
    in opt-in mode); ``excluded`` is marked in the envelope only in opt-in (``keep_in_oof=True``).
    """
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    _reject_multi_model(steps)
    _assert_supported_operators(steps)
    steps = _apply_plain_model_params(steps)

    identity = mint_identity(spectro)
    pool = list(cv_pool) if cv_pool is not None else spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, excluded or set())
    envelope = build_envelope(spectro, identity, sample_ints=pool, excluded_sample_ints=excluded or None, tags_by_sample=tags_by_sample)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle, dataset=spectro, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml engine run failed")

    return _scores_to_run_result(outcome["scores"], spectro.name, _model_name(steps), metric, task_type, config_name=config_name)


def _run_concrete(
    pipeline: Any,
    spectro: Any,
    dataset_arg: str,
    cli: str,
    venv_python: str,
    run_dir: Path,
    metric: str = "rmse",
    task_type: str = "regression",
    cv_pool: list[int] | None = None,
    excluded: set[int] | None = None,
    tags_by_sample: dict[int, list[str]] | None = None,
    dataset_pickle: str | None = None,
    config_name: str = "",
    random_state: int | None = None,
) -> RunResult:
    """Run one concrete (generator-free) pipeline through dag-ml-cli; map its native scores.

    ``cv_pool`` is the CV sample-int universe (de-excluded pool in legacy mode, full train in opt-in
    mode); ``excluded`` is marked in the envelope only in the opt-in (``keep_in_oof=True``) mode.
    """
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    _reject_multi_model(steps)
    _assert_supported_operators(steps)
    steps = _apply_model_params(steps)

    identity = mint_identity(spectro)
    pool = list(cv_pool) if cv_pool is not None else spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, excluded or set())
    envelope = build_envelope(spectro, identity, sample_ints=pool, excluded_sample_ints=excluded or None, tags_by_sample=tags_by_sample)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, dataset_pickle=dataset_pickle, dataset=spectro, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml engine run failed")

    return _scores_to_run_result(outcome["scores"], spectro.name, _model_name(steps), metric, task_type, config_name=config_name)


def _run_repetition(pipeline: list[Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None, config_name: str = "", random_state: int | None = None) -> RunResult:
    """Run a REPETITION (sample-grain grouped) pipeline as ONE native dag-ml CV+refit run.

    The CV universe is the repetition ROWS of the train partition (each stored row is its own
    sample int / sample_id — repetitions are NOT collapsed). Folds are GROUP-aware
    (:func:`_build_group_folds`): all replicates of a sample land on the same fold side, so every
    rep row is validated exactly once (a clean OOF partition) while a group is never split. The
    envelope carries each row's ``group_id`` (the repetition column value), and dag-ml-data's
    ``validate_fold_set_against_sample_relations`` refuses any fold that splits a group — the
    native group-leakage guarantee. Each rep row is scored individually (the repetition grain),
    which is exactly what nirs4all's ``cv_best_score``/``best_rmse`` report; the sample-level
    aggregation (the ``_agg`` twin) is a separate score nirs4all does not surface on RunResult,
    so no aggregation reducer is needed.

    Generators are expanded in Python (operator-level via ``expand_spec``; a param-level model
    sweep also goes through ``expand_spec`` here for simplicity) and each concrete variant runs
    through the group-aware path, selecting the best by its CV score — mirroring nirs4all.
    """
    from nirs4all.pipeline.config.generator import expand_spec

    variants = expand_spec(pipeline)
    results = [
        _run_repetition_concrete(variant, spectro, dataset_arg, cli, venv_python, run_dir / f"variant{index}", metric, task_type, dataset_pickle=dataset_pickle, config_name=config_name, random_state=random_state)
        for index, variant in enumerate(variants)
    ]
    if len(results) == 1:
        return results[0]
    maximize = metric in ("accuracy", "r2")

    def _cv_rank(result: RunResult) -> float:
        score = result.cv_best_score
        if score != score:  # NaN ranks last
            return float("inf")
        return -score if maximize else score

    return min(results, key=_cv_rank)


def _run_repetition_concrete(pipeline: Any, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None, config_name: str = "", random_state: int | None = None) -> RunResult:
    """One concrete repetition variant: group-aware folds + a ``group_id``-carrying envelope."""
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    _assert_supported_operators(steps)
    steps = _apply_model_params(steps)

    identity = mint_identity(spectro)
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_group_folds(splitter, spectro, pool)
    group_by_sample = _repetition_grain(spectro, pool)
    envelope = build_envelope(spectro, identity, sample_ints=pool, group_by_sample=group_by_sample)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle, dataset=spectro, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml repetition run failed")

    return _scores_to_run_result(outcome["scores"], spectro.name, _model_name(steps), metric, task_type, config_name=config_name)


def _reshape_for_rep_fusion(rep_step: dict[str, Any], spectro: Any) -> None:
    """Run nirs4all's REAL repetition reshape (``rep_to_sources`` / ``rep_to_pp``) in place.

    Reuses the production operator (:class:`RepetitionConfig`) + the dataset's own reshape methods
    (``reshape_reps_to_sources`` / ``reshape_reps_to_preprocessings``) — no reshape logic is
    reimplemented here. The reshape collapses the N replicate ROWS of a physical sample into either
    N sample-aligned feature SOURCES (``rep_to_sources``) or N stacked PROCESSING layers
    (``rep_to_pp``), reducing the dataset to ``n_unique`` physical samples.

    The ``repetition`` flag is cleared afterwards: it named the rep grouping of the ORIGINAL rows,
    which no longer exists once the replicates have become sources/processings — the reshaped
    dataset's unit of analysis is the physical sample, so the downstream folds must be sample-grain
    (the plain-repetition group-fold path, #21, must NOT fire on the reshaped dataset).
    """
    from nirs4all.operators.data.repetition import RepetitionConfig

    if "rep_to_sources" in rep_step:
        config = RepetitionConfig.from_step_value(rep_step["rep_to_sources"])
        spectro.reshape_reps_to_sources(config)
    else:
        config = RepetitionConfig.from_step_value(rep_step["rep_to_pp"])
        spectro.reshape_reps_to_preprocessings(config)
    spectro._repetition = None  # noqa: SLF001 - the rep grouping was consumed by the reshape


def _run_rep_fusion(pipeline: list[Any], rep_step: dict[str, Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, config_name: str = "", random_state: int | None = None) -> RunResult:
    """Run a ``rep_to_sources`` / ``rep_to_pp`` pipeline as ONE native dag-ml CV+refit run (S7).

    REP FUSION is a one-time host RESHAPE feeding an already-native multimodal path:

    * ``rep_to_sources`` — each replicate becomes a feature SOURCE, so the reshaped dataset is
      MULTI-SOURCE; the envelope auto-emits a ``feature_block_set`` (S3 early fusion), and an MB-PLS
      model takes the per-source block list (S5 intermediate fusion) — both already native.
    * ``rep_to_pp`` — each replicate becomes a PROCESSING layer, so the reshaped dataset is
      single-source with N stacked layers; the FLAT_2D materialization hstacks them by processing
      order (the feature-axis concat S6 already does), which the legacy ``tabular_numeric`` path runs.

    The reshaped dataset lives only in host memory (the on-disk dataset has no such structure), so it
    is PICKLED for the adapter (the same mechanism ``sample_augmentation`` uses) — the adapter resolves
    the exact reshaped sources/processings, identity-keyed by the physical sample_id.

    LEAKAGE: after the reshape the unit of analysis is the physical SAMPLE (N reps → N sources/layers
    of ONE sample), so folds/OOF are over SAMPLES — distinct from a plain repetition dataset (#21,
    rep-grain). A sample's N source-blocks (or N processing layers) all ride ONE row and therefore the
    SAME fold side by construction; per-source / per-layer preprocessing fits on fold-train only (the
    per-block X-chain, like the multi-source path). No cross-sample mixing.

    Generators are expanded in Python (operator-level via ``expand_spec``), each concrete variant runs
    through the reshaped early-fusion path, and the best is selected by its CV score — mirroring nirs4all.
    """
    import pickle

    from nirs4all.pipeline.config.generator import expand_spec

    body = [step for step in pipeline if not _is_rep_fusion_step(step)]
    variants = expand_spec(body)
    run_dir.mkdir(parents=True, exist_ok=True)
    results = [
        _run_rep_fusion_concrete(variant, rep_step, spectro, dataset_arg, cli, venv_python, run_dir / f"variant{index}", metric, task_type, pickle, config_name=config_name, random_state=random_state)
        for index, variant in enumerate(variants)
    ]
    if len(results) == 1:
        return results[0]
    maximize = metric in ("accuracy", "r2")

    def _cv_rank(result: RunResult) -> float:
        score = result.cv_best_score
        if score != score:  # NaN ranks last
            return float("inf")
        return -score if maximize else score

    return min(results, key=_cv_rank)


def _run_rep_fusion_concrete(body: Any, rep_step: dict[str, Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, pickle: Any, config_name: str = "", random_state: int | None = None) -> RunResult:
    """One concrete rep-fusion variant: reshape a fresh dataset copy, then the sample-grain CV+refit."""
    import copy

    steps, splitter = _split_pipeline(body)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    _assert_supported_operators(steps)
    steps = _apply_model_params(steps)

    # Reshape a FRESH copy per variant so each variant's pickled dataset is independent (and the
    # caller's spectro stays the original rep dataset, unmutated, for any later use).
    reshaped = copy.deepcopy(spectro)
    _reshape_for_rep_fusion(rep_step, reshaped)

    identity = mint_identity(reshaped)
    # The reshape leaves every physical sample in `partition: train` (the splitter follows the reshape),
    # so the CV universe is the full reshaped sample set. Folds are sample-grain — a sample's N
    # source-blocks / processing-layers ride ONE row, so they cannot split across the fold boundary.
    pool = reshaped.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, reshaped, pool, set())
    envelope = build_envelope(reshaped, identity, sample_ints=pool)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()

    run_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = run_dir / "reshaped_dataset.pkl"
    pickle_path.write_bytes(pickle.dumps(reshaped))

    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=str(pickle_path), dataset=reshaped, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml rep-fusion run failed")

    rep_key = "rep_to_sources" if "rep_to_sources" in rep_step else "rep_to_pp"
    return _scores_to_run_result(outcome["scores"], reshaped.name, f"{rep_key}_{_model_name(steps)}", metric, task_type, config_name=config_name)


def _apply_sample_augmentation(aug_step: dict[str, Any], spectro: Any) -> None:
    """Run nirs4all's REAL ``SampleAugmentationController`` to add synthetic train rows in place.

    Reuses the production machinery (the controller delegates to ``TransformerMixinController``, which
    fits the augmentation transformer on fold-train data and inserts each synthetic row via
    ``dataset.add_samples_batch`` with index ``{partition: train, origin: <base sample int>,
    augmentation: <op>}``) — no augmentation logic is reimplemented here. A minimal real ``StepRunner``
    + ``RuntimeContext`` drive it (no workspace/artifacts), exactly as the orchestrator would. The
    children land in ``partition: train`` with their ``origin`` set to the base sample, which is what
    the base→child fold expansion and the envelope's augmentation grain key off.
    """
    from nirs4all.controllers.data.sample_augmentation import SampleAugmentationController
    from nirs4all.pipeline.config.context import DataSelector, ExecutionContext, PipelineState, RuntimeContext, StepMetadata
    from nirs4all.pipeline.steps.parser import ParsedStep, StepType
    from nirs4all.pipeline.steps.step_runner import StepRunner

    step_info = ParsedStep(operator=None, keyword="sample_augmentation", step_type=StepType.DIRECT, original_step=aug_step, metadata={})
    context = ExecutionContext(selector=DataSelector(partition="train", processing=[["raw"]]), state=PipelineState(), metadata=StepMetadata())
    runtime_context = RuntimeContext()
    runtime_context.step_runner = StepRunner(verbose=0, mode="train")
    runtime_context.save_artifacts = False
    runtime_context.save_charts = False
    SampleAugmentationController().execute(step_info, spectro, context, runtime_context, mode="train")


def _augment_fold_train(aug_step: dict[str, Any], spectro: Any, fold_train: list[int]) -> list[tuple[int, np.ndarray]]:
    """Augment a fold's TRAIN only and return the synthetic children as ``[(origin_int, child_X(1,F)), ...]``.

    A FRESH copy of ``spectro`` is restricted so ``partition: train`` is exactly ``fold_train`` (the
    rest held out), then nirs4all's real augmentation machinery runs — so a STATEFUL/SUPERVISED/balanced
    augmenter fits inside this fold's train ONLY (its neighbors / global mean / class balance never see
    the fold's validation rows). The created children (origin in ``fold_train``) are read back as flat
    feature rows, in creation order — one tuple per child, so an origin augmented multiple times (``count``
    > 1 / a balancing factor) yields several. The caller inserts them into the master dataset and records
    the fold→children map; the fold copy is discarded. Leakage-safe by construction: each fold's children
    come only from its train.
    """
    import copy

    fold_ds = copy.deepcopy(spectro)
    fold_ds._indexer.update_by_filter({"partition": "train"}, {"partition": "hold"})  # noqa: SLF001
    fold_ds._indexer.update_by_indices(list(fold_train), {"partition": "train"})  # noqa: SLF001
    fold_ds._invalidate_content_hash()  # noqa: SLF001

    before = {int(s) for s in fold_ds.index_column("sample", {})}
    _apply_sample_augmentation(aug_step, fold_ds)
    samples = [int(s) for s in fold_ds.index_column("sample", {})]
    origins = [int(o) for o in fold_ds.index_column("origin", {})]
    children: list[tuple[int, np.ndarray]] = []
    for sample_int, origin_int in zip(samples, origins, strict=True):
        if sample_int not in before and sample_int != origin_int:
            children.append((origin_int, np.asarray(fold_ds.x_rows([sample_int], layout="2d"), dtype=float).reshape(1, -1)))
    return children


def _build_fold_local_children(aug_step: dict[str, Any], spectro: Any, base_folds: list[tuple[list[int], list[int]]], base_train: list[int]) -> tuple[dict[str, dict[int, list[int]]], dict[int, str]]:
    """Augment fold-by-fold + a full-train refit pass; insert all children into ``spectro`` in place.

    For each fold (key ``"fold{i}"``, matching :func:`build_fold_set`'s fold ids) and the full-train
    refit (key ``"refit"``) the augmenter is fit inside that train pool only (:func:`_augment_fold_train`),
    and the resulting children are appended to the master ``spectro`` as ``partition: train`` rows with
    their ``origin``. Returns ``(fold_children, augmentation_by_sample)`` where ``fold_children`` is
    ``{fold_label: {origin_int: [child_int, ...]}}`` (the resolver's fold-local expansion map) and
    ``augmentation_by_sample`` tags every inserted child for the envelope's augmentation metadata.

    All folds' children coexist in one dataset and one base-grain envelope (each child shares its
    origin's ``sample_id``, a base train id in the fold set → the origin-boundary contract holds), but
    they are kept fold-distinct host-side via the returned map — a fold's children only ever join that
    fold's fit-train (see :meth:`MaterializationResolver.expand_with_augmented_children`).
    """
    transform_label = _augmentation_label(aug_step)
    passes: list[tuple[str, list[int]]] = [(f"fold{index}", train_ints) for index, (train_ints, _val) in enumerate(base_folds)]
    passes.append(("refit", base_train))

    fold_children: dict[str, dict[int, list[int]]] = {}
    augmentation_by_sample: dict[int, str] = {}
    for fold_label, fold_train in passes:
        children = _augment_fold_train(aug_step, spectro, fold_train)
        if not children:
            fold_children[fold_label] = {}
            continue
        rows = np.stack([child_x for _origin, child_x in children])  # (n, 1, F), one row per child
        indexes = [{"partition": "train", "origin": origin_int, "augmentation": f"{transform_label}|{fold_label}"} for origin_int, _x in children]
        before = {int(s) for s in spectro.index_column("sample", {})}
        spectro.add_samples_batch(data=rows, indexes_list=indexes)
        samples = [int(s) for s in spectro.index_column("sample", {})]
        origins = [int(o) for o in spectro.index_column("origin", {})]
        by_origin: dict[int, list[int]] = {}
        for sample_int, origin_int in zip(samples, origins, strict=True):
            if sample_int not in before and sample_int != origin_int:
                by_origin.setdefault(origin_int, []).append(sample_int)
                augmentation_by_sample[sample_int] = transform_label
        fold_children[fold_label] = by_origin
    return fold_children, augmentation_by_sample


def _augmentation_grain(spectro: Any, transform_label: str) -> tuple[list[int], dict[int, str]]:
    """Post-augmentation grain: ``(augmented_ints, augmentation_by_sample)``.

    ``augmented_ints`` is every augmented child (``sample != origin``); ``augmentation_by_sample`` tags
    each with the augmentation transform id for the envelope's structured ``augmentation`` metadata. The
    fit-time base→child expansion is recomputed by the resolver from the minted identity, not here.
    """
    samples = [int(s) for s in spectro.index_column("sample", {})]
    origins = [int(o) for o in spectro.index_column("origin", {})]
    augmented_ints = [sample_int for sample_int, origin_int in zip(samples, origins, strict=True) if sample_int != origin_int]
    augmentation_by_sample = dict.fromkeys(augmented_ints, transform_label)
    return augmented_ints, augmentation_by_sample


def _augmentation_label(aug_step: dict[str, Any]) -> str:
    """A stable transform label for the augmentation step's structured envelope metadata."""
    transformers = _augmentation_transformers(aug_step)
    return "+".join(type(transformer).__name__ for transformer in transformers) or "sample_augmentation"


def _augmentation_transformers(aug_step: dict[str, Any]) -> list[Any]:
    """The deserialized transformer instances of an augmentation step (mirrors the controller).

    A transformer may be given bare or wrapped in a ``{"transformer": op, ...}`` dict (per-transformer
    variation_scope); both forms resolve to the instance via ``deserialize_component``, exactly as
    :meth:`SampleAugmentationController.execute` parses them.
    """
    from nirs4all.pipeline.config.component_serialization import deserialize_component

    raw = aug_step["sample_augmentation"].get("transformers", [])
    return [deserialize_component(t["transformer"] if isinstance(t, dict) and "transformer" in t else t) for t in raw]


def _operator_is_stateless(operator: Any) -> bool:
    """Whether an augmentation transformer learns NO data-dependent state in ``fit``.

    The first augmentation slice augments ONCE globally (before folds exist), so a transformer that
    fits on the whole train partition would see future fold-validation rows. That is leakage-free ONLY
    for STATELESS per-sample augmenters (Gaussian/multiplicative noise, scatter, baseline, spline,
    wavelength warps — ``fit`` only seeds an RNG); a STATEFUL/SUPERVISED one (mixup storing neighbors,
    a global-mean reference, a supervised transform) leaks. There is no declared marker on these
    operators, so the signal is twofold and conservative:

    * **supervised** — a ``requires_y`` tag (``_more_tags()``/``_tags``) means ``fit`` consumes y; reject.
    * **fit-learned data state** — fit the transformer on two differently-distributed dummy datasets and
      compare its post-fit attributes (sklearn fitted attrs ``*_`` + any ndarray, excluding the seeded
      RNG). A transformer whose state varies with the fit data carries learned state (``X_fit_``,
      ``global_mean_``, …) → stateful → reject. A clone is fit each time so no instance state is shared;
      any error during the probe is treated as NOT stateless (fail closed).
    """
    from nirs4all.controllers.transforms.transformer import TransformerMixinController

    if TransformerMixinController._requires_y(operator):  # noqa: SLF001 - reuse the supervised-tag check
        return False
    try:
        from sklearn.base import clone

        rng = np.random.default_rng(0)
        probe_a = rng.normal(size=(24, 32))
        probe_b = rng.normal(loc=8.0, scale=4.0, size=(24, 32))

        def _fit_state(data: np.ndarray) -> dict[str, Any]:
            fitted = clone(operator).fit(data)
            return {name: value for name, value in vars(fitted).items() if (name.endswith("_") or isinstance(value, np.ndarray)) and not name.startswith("_")}

        state_a, state_b = _fit_state(probe_a), _fit_state(probe_b)
        for name in set(state_a) | set(state_b):
            left, right = state_a.get(name), state_b.get(name)
            if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                if not (left is not None and right is not None and np.array_equal(np.asarray(left), np.asarray(right))):
                    return False
            elif left != right:
                return False
    except Exception:  # noqa: BLE001 - any probe failure ⇒ cannot prove stateless ⇒ fail closed
        return False
    return True


def _augmentation_is_leakage_free(aug_step: dict[str, Any]) -> bool:
    """Whether a ``sample_augmentation`` step is safe to run as ONE GLOBAL pre-fold augmentation.

    True (→ the GLOBAL path, #8) ONLY when neither leakage vector applies to global augmentation:

    * the **balanced/supervised mode** (a ``balance`` key) is NOT used — it fits class/bin targets on
      the whole train y, so the synthetic counts depend on the (future-fold-val-inclusive) label set;
    * EVERY transformer is stateless (:func:`_operator_is_stateless`).

    False (→ the FOLD-LOCAL path, #32) for a stateful/supervised/balanced augmentation: it is fit inside
    each fold's train only (and a full-train refit pass), so it never sees a fold's validation rows. Both
    paths are leakage-safe; this only routes which one :func:`_run_augmentation` uses.
    """
    config = aug_step["sample_augmentation"]
    if "balance" in config:
        return False
    transformers = _augmentation_transformers(aug_step)
    return bool(transformers) and all(_operator_is_stateless(transformer) for transformer in transformers)


def _run_augmentation(pipeline: list[Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, config_name: str = "", random_state: int | None = None) -> RunResult:
    """Run a ``sample_augmentation`` pipeline as ONE native dag-ml CV+refit on augmented train.

    Adds the synthetic train rows (real augmentation machinery), builds BASE-grain folds (each base
    val sample validated once — a clean OOF partition that dag-ml/dag-ml-data accept) and a CV-universe
    envelope (base + augmented children; observation-grain relations carrying each child's origin +
    augmentation, deduped to the origin's sample grain in the schema). The fold-train views stay
    base-grain + ``include_augmented_train`` so the host expands each base id to base + its children at
    fit time — the children TRAIN, the OOF/validation/test never see them. The augmented dataset is
    pickled for the adapter (augmentation is stochastic, not reproducible cross-process).

    Two augmentation regimes, picked by :func:`_augmentation_is_leakage_free`:

    * **GLOBAL** (stateless per-sample augmenter) — augment ONCE on the whole train; the children are
      shared across folds. Leakage-free because the augmenter learns no data state (#8).
    * **FOLD-LOCAL** (stateful / supervised / balanced augmenter) — augment inside EACH fold's train
      (plus a full-train refit pass) separately, so each fold has its own children fit only on that
      fold's train. A ``fold_children`` map keys the resolver's per-fold expansion; it is pickled with
      the dataset so the adapter expands the right children per fold (#32). This is what makes the
      stateful case (mixup neighbors, global-mean scatter, class balancing) leakage-safe.

    Only the supported ``transform* + sample_augmentation + splitter + model`` shape runs here; the
    remaining steps are lowered through the bridge (a raw ``sample_augmentation`` still raises, keeping
    the coverage boundary). A branch / exclude / generator beside it is out of scope and fails loud.
    """
    import pickle

    aug_steps = [step for step in pipeline if _is_augmentation_step(step)]
    if len(aug_steps) != 1:
        raise NotImplementedError("engine='dag-ml' supports exactly one sample_augmentation step")
    aug_step = aug_steps[0]
    rest = [step for step in pipeline if not _is_augmentation_step(step)]
    steps, splitter = _split_pipeline(rest)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    _assert_supported_operators(steps)
    steps = _apply_model_params(steps)

    base_train = [int(s) for s in spectro.index_column("sample", {"partition": "train"})]
    fold_local = not _augmentation_is_leakage_free(aug_step)

    # BASE-grain folds: split the base train pool only; train = base-train, val = base-val. The children
    # are NEVER listed in a fold (the FoldSet stays a clean base-grain OOF partition); they are pulled
    # into fit-train by the host expansion keyed on the origin's fold side. The split runs over the BASE
    # rows only (before any child exists), so the fold partition is identical for both augmentation paths.
    base_folds = [([base_train[i] for i in train_idx], [base_train[i] for i in val_idx]) for train_idx, val_idx in _split_pool(splitter, spectro, base_train)]

    # GLOBAL stateless augmentation (#8): fit once on the whole train (leakage-free only for stateless
    # per-sample augmenters), children shared across all folds (resolver discovers them from identity).
    # FOLD-LOCAL augmentation (#32): a STATEFUL/SUPERVISED/balanced augmenter is fit inside EACH fold's
    # train only, so each fold (+ the full-train refit) has its OWN children — leakage-safe for the
    # stateful case. `fold_children` keys the per-fold expansion; it is pickled for the adapter's resolver.
    fold_children: dict[str, dict[int, list[int]]] | None = None
    if fold_local:
        fold_children, augmentation_by_sample_int = _build_fold_local_children(aug_step, spectro, base_folds, base_train)
    else:
        _apply_sample_augmentation(aug_step, spectro)
        _augmented_ints, augmentation_by_sample_int = _augmentation_grain(spectro, _augmentation_label(aug_step))

    # Identity is minted on the AUGMENTED dataset so children get their own observation_id + the origin's
    # sample_id (augmented=True). The CV universe = base train + the augmented children.
    identity = mint_identity(spectro)
    samples = [int(s) for s in spectro.index_column("sample", {})]
    origins = [int(o) for o in spectro.index_column("origin", {})]
    augmented_ints = [sample_int for sample_int, origin_int in zip(samples, origins, strict=True) if sample_int != origin_int]
    cv_universe = base_train + augmented_ints

    envelope = build_envelope(spectro, identity, sample_ints=cv_universe, augmentation_by_sample=augmentation_by_sample_int)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, base_folds, dsl_id="nirs4all-augmentation", n_splits=len(base_folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()

    run_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = run_dir / "augmented_dataset.pkl"
    # Fold-local pickles the dataset + the fold→children map (the resolver's per-fold expansion); the
    # global path pickles the bare dataset (the resolver discovers dataset-global children from identity).
    pickle_path.write_bytes(pickle.dumps({"dataset": spectro, "fold_children": fold_children} if fold_local else spectro))

    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=str(pickle_path), dataset=spectro, fold_children=fold_children, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml augmentation run failed")

    return _scores_to_run_result(outcome["scores"], spectro.name, _model_name(steps), metric, task_type, config_name=config_name)


_MERGE_NODE_ID = "merge:concat"


def _run_separation_branch(pipeline: list[Any], branch_step: dict[str, Any], branch_body: list[Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None, config_name: str = "", random_state: int | None = None) -> RunResult:
    """Run a by_metadata/by_tag separation branch + concat merge as ONE native dag-ml fan-out run.

    Lowers the branch to an ``auto_separate`` template (one branch carrying the criterion + the
    model sub-pipeline) followed by a concat ``PredictionJoin`` merge, builds the envelope with the
    criterion column's per-sample metadata so the **native** ``fan_out_data_aware_branches`` discovers
    the partition values, compiles the fanned graph (one model node per value), and drives dag-ml-cli.
    The native concat-merge handler reassembles the per-partition OOF into one full-universe OOF
    attributed to the merge node — its cross-fold average is ``cv_best_score``.

    The criterion column's metadata is emitted onto the relations; the adapter honors each fanned
    model's ``branch_view`` selector (via the sample→metadata map) so each model fits/predicts only
    its partition. The legacy nirs4all concat-merge reassembly is broken here (MERGE-E003), so this
    native path is a correction — the parity baseline is a direct sklearn-per-partition OOF.
    """
    import dag_ml

    from nirs4all.pipeline.dagml.cli_runner import data_bindings_for_nodes, split_invocation_for

    criterion = branch_step["branch"]
    mode, key = ("by_metadata", criterion["by_metadata"]) if "by_metadata" in criterion else ("by_tag", criterion["by_tag"])

    # The splitter lives at the top level (before the branch); the branch body is the model
    # sub-pipeline applied per partition. Drop the splitter from the body if it slipped in.
    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    # Drop the splitter / None no-ops and reject wavelength-requiring or non-routable ops in the body.
    body_steps = _supported_body_steps([step for step in branch_body if not hasattr(step, "split")])

    identity = mint_identity(spectro)
    # The handled shape rejects any exclude step, so the CV universe is the full train pool.
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())

    # Per-sample criterion values: the first map seeds the envelope relations (native fan-out reads
    # partition values from it); the second is the adapter's sample_id→metadata map for branch_view.
    metadata_by_sample, sample_metadata = _branch_metadata(spectro, identity, mode, key)
    envelope = build_envelope(spectro, identity, sample_ints=pool, metadata_by_sample=metadata_by_sample)

    # Compat auto_separate template: ONE branch (the model sub-pipeline) carrying the criterion +
    # mode, marked auto_separate; the native fan-out expands it into N per-partition branches.
    template = {"id": "per_partition", "steps": [_branch_compat_step(step) for step in body_steps]}
    # Always by_metadata mode: the criterion (whether nirs4all by_metadata or by_tag) is emitted as a
    # metadata column on the relations, so the native fan-out discovers its values from there.
    compat_dsl = {
        "id": "nirs4all-separation-branch",
        "pipeline": [
            {"branch": {"branches": [template]}, "mode": "by_metadata", "selector": {"metadata_key": key}, "metadata": {"auto_separate": True}},
            {"merge": "concat", "output_as": "predictions", "id": _MERGE_NODE_ID},
        ],
    }

    # NATIVE fan-out (no Python suffix replication): dag-ml reads the partition values from the
    # envelope relations and expands one branch per sorted value, owning the node-id suffixing.
    fanned_dsl = dag_ml.fan_out_data_aware_branches(compat_dsl, envelope).to_dict()
    manifests = controller_manifests()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(fanned_dsl, manifests).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    if not model_ids:
        raise DagMlUnsupported("separation-branch fan-out produced no per-partition model nodes")

    # Per-partition data_bindings (one per fanned model node) + the materialized fold set. The CLI's
    # own fan-out is a no-op on the already-fanned DSL (the auto_separate marker is consumed).
    fanned_dsl["data_bindings"] = data_bindings_for_nodes(model_ids, envelope)
    fanned_dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))

    outcome = run_cv_refit_bundle(
        dsl=fanned_dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, sample_metadata=sample_metadata, dataset_pickle=dataset_pickle, dataset=spectro, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml separation-branch run failed")

    # The concat-merge producer's reports carry both the full-universe cross-fold OOF average
    # (`cv_best_score`) AND a reassembled `(test, fold_id=None)` block (`best_rmse`): dag-ml's native
    # off-fold merge handler reassembles each per-partition refit model's held-out TEST prediction
    # (the node runner emits it with `fold_id=None`) into one full-universe test block under the merge
    # node. Both scores are the separation branch's, surfaced by `_scores_to_run_result`.
    return _scores_to_run_result(outcome["scores"], spectro.name, _model_name(body_steps), metric, task_type, producer=_MERGE_NODE_ID, config_name=config_name)


def _branch_compat_step(step: Any) -> dict[str, Any]:
    """Lower one branch-body step (transform / {"model": M} / {"y_processing": op}) to compat DSL."""
    from nirs4all.pipeline.dagml_bridge import _step_to_dsl

    return _step_to_dsl(step)


def _branch_metadata(spectro: Any, identity: Any, mode: str, key: str) -> tuple[dict[str, dict[int, Any]], dict[str, dict[str, Any]]]:
    """Build the criterion's per-sample metadata: ``({col: {sample_int: value}}, {wire_id: {col: value}})``.

    The first map seeds the envelope relations (native fan-out reads partition values from it); the
    second is the adapter's ``sample_id → metadata`` map for honoring each branch's ``branch_view``.
    A ``by_tag`` criterion is represented as a metadata column (the tag value per sample), matching
    the envelope's metadata-carried relations.
    """
    sample_ints = [int(value) for value in spectro.index_column("sample", {})]
    values = spectro.metadata_column(key, {}) if mode == "by_metadata" else spectro.get_tag(key, {})
    by_int = {sample_int: (str(value) if value is not None else None) for sample_int, value in zip(sample_ints, values, strict=True)}
    metadata_by_sample = {key: dict(by_int)}
    sample_metadata = {identity.to_wire(sample_int): {key: value} for sample_int, value in by_int.items()}
    return metadata_by_sample, sample_metadata


_FUSION_MERGE_NODE_ID = "merge:fusion"


def _canonical_branch_step(step: Any, node_id: str) -> dict[str, Any]:
    """Lower one branch-body step to a CANONICAL pipeline-DSL step (``kind`` + ``operator.class``).

    The duplication-fusion path emits a *canonical* ``steps`` DSL (not the compat ``pipeline`` form):
    dag-ml's nirs4all-compat importer whitelists merge modes (``concat``/``predictions``/… only) and
    REFUSES ``fusion`` — but the canonical ``PipelineDslMergeStep.merge_mode`` is a free-form string the
    runtime reads verbatim, so the fusion merge must be expressed canonically. The branch bodies must
    therefore also be canonical. This reuses the verified compat lowering
    (:func:`~nirs4all.pipeline.dagml_bridge._step_to_dsl` — operator FQN + JSON-safe params +
    native param-generator entries) and re-keys it into the canonical ``{"kind", "id", "operator":
    {"class": …}, "params": …}`` shape per node kind (transform / y_transform / model), assigning the
    explicit ``node_id`` so each branch's nodes are uniquely named.
    """
    from nirs4all.pipeline.dagml_bridge import _step_to_dsl

    compat = _step_to_dsl(step)
    if "model" in compat:
        out: dict[str, Any] = {"kind": "model", "id": node_id, "operator": {"class": compat["model"]}, "params": compat.get("params", {})}
        if "generators" in compat:
            out["generators"] = compat["generators"]
        return out
    if "y_processing" in compat:
        inner = compat["y_processing"]
        return {"kind": "y_transform", "id": node_id, "operator": {"class": inner["class"]}, "params": inner.get("params", {})}
    # Bare transform: compat is {"class": FQN, "params": {...}}.
    return {"kind": "transform", "id": node_id, "operator": {"class": compat["class"]}, "params": compat.get("params", {})}


def _canonical_branch(branch_body: list[Any], branch_index: int) -> dict[str, Any]:
    """Lower one duplication sub-pipeline (a list of steps) to a canonical branch with unique node ids.

    ``None`` no-ops are dropped (never lowered to a ``builtins.NoneType`` node) and wavelength-requiring /
    non-routable ops in the body raise a catchable :class:`DagMlUnsupported` — same coverage guarantee
    the top-level path gives, applied here so duplication / by_source / stacking branch bodies are safe.
    """
    steps = _supported_body_steps([step for step in branch_body if not hasattr(step, "split")])
    return {
        "id": f"branch_{branch_index}",
        "steps": [_canonical_branch_step(step, f"branch:{branch_index}.node:{node_index}") for node_index, step in enumerate(steps)],
    }


def _canonical_source_branch(branch_body: list[Any], source_index: int) -> dict[str, Any]:
    """Lower the shared by_source body to a canonical branch BOUND to one source (S4).

    Same lowering as :func:`_canonical_branch` (the shared model sub-pipeline, unique node ids per
    source), but every MODEL node carries ``metadata.source_index`` so the node runner materializes
    ONLY that source's feature block — late fusion by source. The branch index IS the source index
    (one branch per source), so a fold view stays full-sample (all branches see all samples) while
    each branch's model sees a different source's columns.
    """
    branch = _canonical_branch(branch_body, source_index)
    for node in branch["steps"]:
        if node["kind"] == "model":
            node["metadata"] = {**node.get("metadata", {}), "source_index": source_index}
    return branch


def _run_by_source_branch(pipeline: list[Any], branch_body: list[Any], aggregate: str, n_sources: int, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None, config_name: str = "", random_state: int | None = None) -> RunResult:
    """Run a by_source separation branch + avg/mean fusion merge as ONE native dag-ml run (S4).

    LATE fusion BY SOURCE: fans the shared body into one canonical branch PER feature source
    (:func:`_canonical_source_branch`), each MODEL node bound to its source via ``metadata.source_index``
    so the node runner feeds it ONLY that source's block (all samples, that source's columns). The fold
    set, OOF, and merge are sample-keyed and identical to the duplication-fusion path — the ONLY
    difference from duplication is the feature-axis (per-source) restriction each branch's model sees.

    dag-ml runs ONE native CV+refit: each per-source branch model is FIT_CV on the full fold-train
    (its source's columns) and predicts the full fold-validation (held-out OOF); the native fusion
    merge handler averages the branches' per-sample Validation OOF (leakage-safe — train predictions
    never enter the average) into one full-universe OOF attributed to the merge node, whose cross-fold
    average is ``cv_best_score``. ``best_rmse`` (final test) is also native: each branch's REFIT predicts
    the held-out TEST set (its source's columns, ``fold_id=None``) and dag-ml's off-fold fusion handler
    averages those per sample into one scored ``(test, fold_id=None)`` block under the merge node.

    LEAKAGE: folds/OOF over SAMPLES (unchanged — all branches see all samples, just different source
    columns); a sample's blocks all land on the same fold side (the source restriction is a feature-axis
    selection, never a sample partition); per-source per-fold preprocessing fits on fold-train only
    (the per-branch X-chain is fit inside the fold's train materialization, like the single-block path);
    the fusion merge is OOF-safe. No cross-sample mixing.

    Classification (``fusion_proba_mean``) is NOT wired (identical to the duplication-fusion path): the
    node runner emits scalar value predictions, not per-class probability rows, so a probability-mean
    fusion has no proba blocks to average — it fails loud rather than silently running ``proba_mean`` as
    a value (regression) fusion (audit H-P0-1).
    """
    import dag_ml

    from nirs4all.pipeline.dagml.cli_runner import data_bindings_for_nodes, split_invocation_for

    if aggregate == "proba_mean":
        raise DagMlUnsupported(
            "engine='dag-ml' does not yet support proba-mean fusion (classification) for a by_source branch: "
            "the process adapter emits class-label predictions, not per-class probability rows; backlog #20-avg (proba)."
        )

    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")

    identity = mint_identity(spectro)
    # The handled shape rejects any exclude step, so the CV universe is the full train pool.
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())
    envelope = build_envelope(spectro, identity, sample_ints=pool)

    # Canonical DSL: one duplication-mode branch with N per-source sub-pipelines (each the shared body
    # bound to its source via metadata.source_index) + a fusion merge. The branch is `duplication` mode
    # because every branch sees the FULL fold sample view (no fan-out / no branch_view / no
    # sample_metadata) — the per-source restriction is applied host-side at materialization, not by a
    # sample-partition branch_view.
    canonical_dsl: dict[str, Any] = {
        "id": "nirs4all-by-source-fusion",
        "steps": [
            {"kind": "branch", "mode": "duplication", "branches": [_canonical_source_branch(branch_body, source_index) for source_index in range(n_sources)]},
            {"kind": "merge", "id": _FUSION_MERGE_NODE_ID, "merge_mode": "fusion", "output_as": "predictions"},
        ],
    }

    manifests = controller_manifests()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(canonical_dsl, manifests).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    if len(model_ids) != n_sources:
        raise DagMlUnsupported(f"by_source compile produced {len(model_ids)} model nodes, expected {n_sources}")

    # One data_binding per per-source model node (each binds its `x` to the full source set; the node
    # runner selects its own source's block by metadata.source_index) + the materialized fold set.
    canonical_dsl["data_bindings"] = data_bindings_for_nodes(model_ids, envelope)
    canonical_dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))

    outcome = run_cv_refit_bundle(
        dsl=canonical_dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle, dataset=spectro, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml by_source run failed")

    model_label = f"by_source_{_model_name(branch_body)}x{n_sources}"
    return _scores_to_run_result(outcome["scores"], spectro.name, model_label, metric, task_type, producer=_FUSION_MERGE_NODE_ID, config_name=config_name)


def _run_duplication_branch(pipeline: list[Any], branches: list[list[Any]], aggregate: str, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None, config_name: str = "", random_state: int | None = None) -> RunResult:
    """Run a duplication branch (``[[A], [B], …]``) + avg/mean fusion merge as ONE native dag-ml run.

    Lowers each inner sub-pipeline to a canonical branch (``mode: "duplication"`` — every branch model
    node gets the FULL fold data view: NO fan-out, NO ``auto_separate``, NO ``branch_view``,
    NO ``sample_metadata``) and a fusion merge node (``merge_mode: "fusion"`` for the value mean,
    ``output_as: "predictions"``). dag-ml runs ONE native CV+refit: each branch model is FIT_CV on the
    full fold-train and predicts the full fold-validation (held-out OOF); the native fusion merge handler
    averages the branches' per-sample Validation OOF (leakage-safe — train predictions never enter the
    average) into one full-universe OOF attributed to the merge node, whose cross-fold average is
    ``cv_best_score``.

    ``best_rmse`` (final test) is also native: each branch's REFIT predicts the held-out TEST set
    (the node runner emits it with ``fold_id=None``), and dag-ml's off-fold fusion handler
    (``reassemble_branch_merge_off_fold``) averages those base test predictions per sample into one
    scored ``(test, fold_id=None)`` block under the merge node — the same average as ``cv_best_score``
    but over the held-out test set.

    Classification (``fusion_proba_mean``) is NOT wired: the node runner emits scalar value predictions,
    not per-class probability rows, so a probability-mean fusion has no proba blocks to average — it
    fails loud rather than averaging class labels (which is not what ``proba_mean`` means).
    """
    import dag_ml

    from nirs4all.pipeline.dagml.cli_runner import data_bindings_for_nodes, split_invocation_for

    if aggregate == "proba_mean":
        raise NotImplementedError(
            "engine='dag-ml' does not yet support proba-mean fusion (classification): the process adapter "
            "emits class-label predictions, not per-class probability rows; backlog #20-avg (proba)."
        )
    merge_mode = "fusion"

    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")

    identity = mint_identity(spectro)
    # The handled shape rejects any exclude step, so the CV universe is the full train pool.
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())
    envelope = build_envelope(spectro, identity, sample_ints=pool)

    # Canonical DSL: one duplication branch with N sub-pipelines (each on the FULL data) + a fusion merge.
    canonical_dsl: dict[str, Any] = {
        "id": "nirs4all-duplication-fusion",
        "steps": [
            {"kind": "branch", "mode": "duplication", "branches": [_canonical_branch(branch, index) for index, branch in enumerate(branches)]},
            {"kind": "merge", "id": _FUSION_MERGE_NODE_ID, "merge_mode": merge_mode, "output_as": "predictions"},
        ],
    }

    manifests = controller_manifests()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(canonical_dsl, manifests).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    if len(model_ids) < 2:
        raise DagMlUnsupported("duplication-fusion compile produced fewer than two model nodes")

    # One data_binding per branch model node (each binds its `x` to the full source) + the materialized
    # fold set. Every model node sees the full fold view — no branch_view/sample_metadata (duplication).
    canonical_dsl["data_bindings"] = data_bindings_for_nodes(model_ids, envelope)
    canonical_dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))

    outcome = run_cv_refit_bundle(
        dsl=canonical_dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle, dataset=spectro, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml duplication-fusion run failed")

    # The fusion-merge producer's reports carry the full-universe cross-fold OOF average (the fused
    # ensemble's `cv_best_score`) AND a reassembled `(test, fold_id=None)` block (`best_rmse`, the
    # branches' test predictions averaged per sample). Both are surfaced by `_scores_to_run_result`.
    model_label = "+".join(_model_name(branch) for branch in branches)
    return _scores_to_run_result(outcome["scores"], spectro.name, model_label, metric, task_type, producer=_FUSION_MERGE_NODE_ID, config_name=config_name)


_META_NODE_ID = "merge:stack"


def _run_stacking_branch(pipeline: list[Any], branches: list[list[Any]], meta_learner: Any, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None, config_name: str = "", random_state: int | None = None) -> RunResult:
    """Run a duplication branch + ``{"merge": "predictions"}`` + meta-model as ONE native dag-ml run (#10).

    Lowers each inner sub-pipeline to a canonical duplication branch (``mode: "duplication"`` — each base
    model node gets the FULL fold data view) + a ``merge_model`` meta-node carrying the meta-learner (its
    FQN as ``operator.class`` so the node runner instantiates it) bound to ``controller:nirs4all.meta_model``
    (which declares ``consumes_oof_predictions``, so dag-ml's planner permits the base→meta ``requires_oof``
    edges). dag-ml runs ONE native CV+refit:

    * each base branch model is FIT_CV on the full fold-train and predicts the full fold-validation
      (held-out Validation OOF);
    * the meta-node receives the base branches' **Validation OOF** per fold (Option A: the runtime delivers
      ``prediction_inputs[*].values`` aligned by sample_id, sourced ONLY from Validation blocks — the
      ``requires_oof`` edge refuses any train block), builds the per-fold meta-feature matrix (columns in
      deterministic producer order), fits the meta-learner and emits its own scored Validation OOF.

    The meta producer's cross-fold OOF average is ``cv_best_score`` — the stacking ensemble's CV score.
    ``best_rmse`` (final test) is also native: in REFIT dag-ml delivers each base producer's held-out
    TEST prediction to the meta-node as a SEPARATE off-fold input keyed ``…oof:refit`` (partition Test,
    ``fold_id=None``), alongside the full Validation OOF the meta fits on. The refit meta-model predicts
    the test set from those base TEST meta-features and emits a scored ``(test, fold_id=None)`` block.

    LEAKAGE INVARIANT: the meta-learner is fit on Validation OOF ONLY (the ``requires_oof`` edge +
    ``collect_oof_prediction_input`` enforce Validation-only); the TEST meta-features come from the base
    models' TEST predictions (the ``:refit`` off-fold input, phase-gated to REFIT), never their OOF/train,
    and never enter the FIT_CV meta-features.
    """
    import dag_ml

    from nirs4all.pipeline.dagml.cli_runner import data_bindings_for_nodes, split_invocation_for
    from nirs4all.pipeline.dagml_bridge import _META_MODEL_CONTROLLER_ID, _META_MODEL_REF, _json_safe_params, _qualname

    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")

    identity = mint_identity(spectro)
    # The handled shape rejects any exclude step, so the CV universe is the full train pool.
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())
    envelope = build_envelope(spectro, identity, sample_ints=pool)

    # Canonical DSL: one duplication branch with N base sub-pipelines (each on the FULL data) + a
    # merge_model meta-node. The meta-node carries the bare sklearn meta-learner (FQN + params) and the
    # _META_MODEL_REF (so its dedicated manifest is not a generic model-kind catch-all) and binds to the
    # meta-model controller via metadata.controller_id.
    canonical_dsl: dict[str, Any] = {
        "id": "nirs4all-stacking",
        "steps": [
            {"kind": "branch", "mode": "duplication", "branches": [_canonical_branch(branch, index) for index, branch in enumerate(branches)]},
            {
                "kind": "merge_model",
                "id": _META_NODE_ID,
                "operator": {"class": _qualname(meta_learner), "ref": _META_MODEL_REF},
                "params": _json_safe_params(meta_learner),
                "metadata": {"controller_id": _META_MODEL_CONTROLLER_ID},
            },
        ],
    }

    manifests = controller_manifests()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(canonical_dsl, manifests).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    base_model_ids = [model_id for model_id in model_ids if model_id != _META_NODE_ID]
    if len(base_model_ids) < 2:
        raise DagMlUnsupported("stacking compile produced fewer than two base model nodes")
    if _META_NODE_ID not in model_ids:
        raise DagMlUnsupported("stacking compile produced no meta-model node")

    # One data_binding per BASE model node (each binds its `x` to the full source). The meta-node has NO
    # data binding: its features are the base branches' OOF, delivered as prediction_inputs (not data).
    canonical_dsl["data_bindings"] = data_bindings_for_nodes(base_model_ids, envelope)
    canonical_dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))

    outcome = run_cv_refit_bundle(
        dsl=canonical_dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle, dataset=spectro, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml stacking run failed")

    # The meta-node producer's reports carry the full-universe cross-fold OOF average (the stacking
    # ensemble's `cv_best_score`) AND a `(test, fold_id=None)` block (`best_rmse`): the refit meta-model
    # predicting the held-out test from the base producers' REFIT-test predictions (`…oof:refit`).
    model_label = f"MetaModel_{type(meta_learner).__name__}"
    return _scores_to_run_result(outcome["scores"], spectro.name, model_label, metric, task_type, producer=_META_NODE_ID, config_name=config_name)
