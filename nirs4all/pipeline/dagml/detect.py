"""Pipeline step-shape detection for the dag-ml backend.

The ``_is_*`` step predicates, the ``_detect_*`` composition detectors (separation / by_source /
duplication / stacking branches, rep-fusion), the fusion-merge / meta-learner parsers, and the
``_generation_kind`` classifier — the conservative, fail-loud recognizers that decide which native
dag-ml path a pipeline takes (and reject anything not exactly handled).
"""

from __future__ import annotations

from typing import Any

from .steps import _RESERVED_STEP_KEYS


def _is_exclude_step(step: Any) -> bool:
    return isinstance(step, dict) and "exclude" in step


def _is_augmentation_step(step: Any) -> bool:
    return isinstance(step, dict) and "sample_augmentation" in step


# `rep_to_sources` / `rep_to_pp` are one-time HOST dataset RESHAPES (RepToSourcesController /
# RepToPPController, priority 3 — applied BEFORE the CV splitter). `rep_to_sources` turns each
# replicate of a physical sample into a separate feature SOURCE (N reps → N sources × n_unique
# samples), and `rep_to_pp` stacks each replicate into the PROCESSING axis (N reps → n_pp×N
# processing layers × n_unique samples). After the reshape the unit of analysis is the physical
# SAMPLE (not the rep row), so folds/OOF are sample-grain — distinct from a PLAIN repetition
# dataset (#21, which keeps the rep rows and scores at the rep grain).
_REP_FUSION_KEYS = ("rep_to_sources", "rep_to_pp")


def _is_rep_fusion_step(step: Any) -> bool:
    return isinstance(step, dict) and any(key in step for key in _REP_FUSION_KEYS)


def _detect_rep_fusion(pipeline: list[Any]) -> dict[str, Any] | None:
    """The single ``rep_to_sources`` / ``rep_to_pp`` reshape step, else ``None`` (fail-loud elsewhere).

    Returns the reshape step only for the EXACTLY-supported shape — one reshape step plus the
    ordinary ``transform* + splitter + model`` body. More than one reshape, or a reshape combined
    with a branch / exclude / sample_augmentation (compositions the reshaped sample-grain folds
    cannot honor here), returns ``None`` so the bridge's generic fail-loud path names #31.
    """
    rep_steps: list[dict[str, Any]] = [step for step in pipeline if _is_rep_fusion_step(step)]
    if len(rep_steps) != 1:
        return None
    if any(_is_augmentation_step(step) or _is_exclude_step(step) or (isinstance(step, dict) and "branch" in step) for step in pipeline):
        return None
    return rep_steps[0]


# Step keywords whose presence forces the Python path even alongside a model param sweep: Optuna
# finetune / per-model train kwargs are not part of the native generation+SELECT contract, so a
# pipeline carrying them must NOT be mistaken for a clean param-sweep-only pipeline.
_FORCE_PYTHON_STEP_KEYS = frozenset({"finetune_params", "train_params"})


def _generation_kind(pipeline: list[Any]) -> str:
    """Classify a pipeline's generators: ``"none"``, ``"param_model"`` (native), or ``"operator"``.

    CONSERVATIVE by design — native (``"param_model"``) is returned ONLY when the pipeline is a clean
    model-param-sweep, i.e. ALL of:

    (a) at least one ``{"model": ...}`` step carries a natively-lowerable param sweep
        (:func:`~nirs4all.pipeline.dagml_bridge.is_param_generator_spec` — the exact ``_range_`` /
        ``_log_range_`` list forms), AND
    (b) NO other generator exists ANYWHERE — no generator keyword on a non-model step, no
        generator-valued model (multi-model ``{"model": {"_or_": ...}}``), no generator-shaped model
        sibling that is not natively lowerable (``_grid_``, dict-form, modifier-bearing), AND
    (c) NO step carries ``finetune_params`` or ``train_params``.

    Any other generator (or finetune/train_params) → ``"operator"`` (the correct Python ``expand_spec``
    path). ``"none"`` means no generators at all. When in doubt, this never returns ``"param_model"``.
    """
    from nirs4all.pipeline.config._generator.keywords import GENERATION_KEYWORDS, has_nested_generator_keywords
    from nirs4all.pipeline.dagml_bridge import is_param_generator_spec

    has_param_model = False
    has_other = False
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        if _FORCE_PYTHON_STEP_KEYS & set(step):
            has_other = True  # finetune/train_params are not in the native contract
        if "model" in step:
            # A generator-valued model (multi-model) is operator-level, not a clean param sweep.
            if has_nested_generator_keywords(step["model"]):
                has_other = True
            for key, value in step.items():
                if key in _RESERVED_STEP_KEYS:
                    continue
                if is_param_generator_spec(value):
                    has_param_model = True
                elif has_nested_generator_keywords(value):
                    # A generator-shaped sibling we cannot lower natively (e.g. `_grid_`, dict-form,
                    # or a modifier-bearing range) — Python expand owns it.
                    has_other = True
        elif GENERATION_KEYWORDS & set(step) or has_nested_generator_keywords(step):
            # Any generator on a non-model step (bare `_or_`/`_range_`/... or a nested one).
            has_other = True
    if has_other:
        return "operator"
    return "param_model" if has_param_model else "none"


def _is_separation_branch_step(step: Any) -> bool:
    """True for a separation branch by metadata/tag: ``{"branch": {"by_metadata"|"by_tag": ...}}``."""
    return isinstance(step, dict) and isinstance(step.get("branch"), dict) and bool({"by_metadata", "by_tag"} & set(step["branch"]))


def _is_concat_merge_step(step: Any) -> bool:
    return isinstance(step, dict) and step.get("merge") == "concat"


# Keys recognised inside a separation-branch criterion dict. `run_backend` honors ONLY the criterion
# (by_metadata/by_tag) + the shared `steps` body; `values` (explicit grouping), `min_samples`
# (cardinality drop) and per-branch selectors are NOT honored, so a branch carrying them must fall
# through to the loud bridge error rather than be silently run with default behavior.
_HANDLED_BRANCH_KEYS = frozenset({"by_metadata", "by_tag", "steps"})


def _detect_separation_branch(pipeline: list[Any]) -> tuple[dict[str, Any], list[Any]] | None:
    """Detect the EXACT handled shape, else return ``None`` (fail-loud via the bridge).

    Admits ONLY a pipeline that is exactly: the splitter + ONE by_metadata/by_tag separation branch
    (a single shared ``steps`` body containing the model) + ONE ``{"merge": "concat"}`` — nothing
    that ``_run_separation_branch`` does not actually honor. Returns ``(branch_step, branch_body)``
    when matched. ANY deviation returns ``None`` so the bridge's raw-branch ``NotImplementedError``
    fires (the coverage-boundary fail-loud guarantee), never a silent-wrong run. Specifically REJECTED:

    * a top-level operator/transform/``tag``/``y_processing`` step beside the branch (only the branch
      body is lowered, so a top-level step would be silently dropped) — out-of-scope follow-up;
    * an ``exclude`` step anywhere (the folds are built over the full pool with no excluded bit, so the
      exclusion would be silently lost) — exclude+branch is a follow-up slice;
    * an unhandled branch option (``values`` / ``min_samples`` / a per-branch ``selector`` / any key
      outside ``by_metadata``/``by_tag``/``steps``) — those grouping semantics are not honored;
    * a per-value dict ``steps`` (different sub-pipeline per partition), a missing model in the body,
      a model after the merge, a non-concat merge, or more than one branch/merge.
    """
    branch_steps = [step for step in pipeline if _is_separation_branch_step(step)]
    merge_steps = [step for step in pipeline if _is_concat_merge_step(step)]
    if len(branch_steps) != 1 or len(merge_steps) != 1:
        return None
    branch_step, merge_step = branch_steps[0], merge_steps[0]

    # The pipeline must be EXACTLY {splitter, branch, merge} — no other top-level steps. A top-level
    # transform / tag / y_processing / exclude / model would be silently ignored (only the branch body
    # is lowered), so its presence rejects the match → fail-loud.
    for step in pipeline:
        if step is branch_step or step is merge_step or hasattr(step, "split"):
            continue
        return None

    criterion = branch_step["branch"]
    # Only the criterion (by_metadata/by_tag) + the shared `steps` body are honored. Any other branch
    # option (values/min_samples/per-branch selector/...) is not → reject.
    if set(criterion) - _HANDLED_BRANCH_KEYS:
        return None

    body = criterion.get("steps")
    # Only the shared-body LIST form (one sub-pipeline applied per partition) with a model inside is
    # supported. The per-value dict form and a body without a model fall through to the bridge error.
    if not isinstance(body, list) or not any(isinstance(sub, dict) and "model" in sub for sub in body):
        return None
    return branch_step, body


# Keys recognised inside a by_source branch criterion dict. `run_backend` honors ONLY the
# `by_source` flag + the shared `steps` body (one sub-pipeline applied per source). A per-source
# dict body (`{"src0": [...], "src1": [...]}`), `values`/`min_samples`, or any other option falls
# through to the loud bridge error rather than being silently run with default behavior.
_HANDLED_BY_SOURCE_KEYS = frozenset({"by_source", "steps"})


def _is_by_source_branch_step(step: Any) -> bool:
    """True for a by_source separation branch: ``{"branch": {"by_source": True|"auto", ...}}``.

    LATE fusion BY SOURCE: a branch PER feature source, each branch's model fed ONLY that source's
    block (a feature-axis selection — all samples, one source's columns), distinct from by_metadata
    (a SAMPLE partition) and duplication (every branch sees the FULL data).
    """
    if not isinstance(step, dict) or not isinstance(step.get("branch"), dict):
        return False
    return step["branch"].get("by_source") in (True, "auto")


def _detect_by_source_branch(pipeline: list[Any], n_sources: int) -> tuple[list[Any], str] | None:
    """Detect the EXACT handled by_source shape, else ``None`` (fail-loud via the bridge).

    Admits ONLY: the splitter + ONE ``{"branch": {"by_source": True|"auto", "steps": [...model...]}}``
    (a single shared body LIST containing the model, applied per source) + ONE avg/mean fusion merge
    (:func:`_fusion_merge_aggregate`) on a MULTI-source dataset (``n_sources >= 2``). Returns
    ``(body, aggregate)`` when matched — the shared branch body (the model sub-pipeline) AND the fusion
    aggregate (``"mean"`` or ``"proba_mean"``), mirroring :func:`_detect_duplication_branch`. The
    aggregate is RETURNED (not dropped) so the runner can reject ``proba_mean`` fail-loud, exactly as
    duplication does — accepting ``proba_mean`` here while hardcoding value-fusion in the runner would
    silently run a probability-mean merge as a regression-fusion (audit H-P0-1). ANY deviation returns
    ``None`` so the bridge's raw-branch ``NotImplementedError`` fires — never a silent-wrong run.
    Specifically REJECTED:

    * a single-source dataset (by_source on one source is a no-op — there is nothing to fuse);
    * the per-source DICT body (``{"src0": [...], "src1": [...]}`` — different model per source) — a
      later slice; only the shared body is honored here;
    * an unhandled branch option (``values`` / ``min_samples`` / any key outside ``by_source``/``steps``);
    * a body without a model (late fusion averages MODEL predictions);
    * a non-fusion merge (``concat`` / ``predictions`` stacking), a top-level step beside the branch,
      a model after the merge, or more than one branch/merge.
    """
    branch_steps = [step for step in pipeline if _is_by_source_branch_step(step)]
    merge_aggregates = [(step, agg) for step in pipeline if (agg := _fusion_merge_aggregate(step)) is not None]
    if len(branch_steps) != 1 or len(merge_aggregates) != 1 or n_sources < 2:
        return None
    branch_step = branch_steps[0]
    merge_step, aggregate = merge_aggregates[0]

    # The pipeline must be EXACTLY {splitter, branch, merge} — no other top-level steps (a top-level
    # transform / tag / y_processing / exclude / model would be silently dropped, since only the branch
    # body is lowered per source).
    for step in pipeline:
        if step is branch_step or step is merge_step or hasattr(step, "split"):
            continue
        return None

    criterion = branch_step["branch"]
    if set(criterion) - _HANDLED_BY_SOURCE_KEYS:
        return None
    body = criterion.get("steps")
    # Only the shared-body LIST form (one model sub-pipeline applied per source) is supported here.
    if not isinstance(body, list) or not any(isinstance(sub, dict) and "model" in sub for sub in body):
        return None
    return body, aggregate


def _is_duplication_branch_step(step: Any) -> bool:
    """True for a DUPLICATION branch: ``{"branch": [[A], [B], ...]}`` (the list-of-lists form).

    Legacy nirs4all (``BranchController._detect_branch_mode``) treats *list* branch syntax as ALWAYS
    duplication — N parallel sub-pipelines, each seeing the FULL data (no sample partitioning). The
    dict form (``{"by_metadata": ...}``/named branches) is separation/other and is NOT matched here.
    Each inner element must itself be a list (a sub-pipeline of steps).
    """
    if not isinstance(step, dict):
        return False
    branch = step.get("branch")
    return isinstance(branch, list) and len(branch) >= 2 and all(isinstance(sub, list) for sub in branch)


# The cross-branch fusion (avg / proba-mean) merge tokens this backend maps to dag-ml's native fusion
# merge handler. Simple-string ``"mean"``/``"average"`` (a NEW token — legacy MergeConfigParser rejects
# it, so there is no collision) average the branches' held-out OOF per sample into ONE final prediction;
# the explicit-aggregation dict form reuses nirs4all's established aggregation vocabulary
# (``AggregationStrategy.MEAN``/``PROBA_MEAN``). A STACKING merge (``{"merge": "predictions"}`` →
# MetaModel, backlog #10) is deliberately NOT a fusion token and falls through to the loud bridge error.
_FUSION_MERGE_STRINGS = frozenset({"mean", "average"})


def _fusion_merge_aggregate(step: Any) -> str | None:
    """The fusion aggregation if ``step`` is a handled avg/mean fusion merge, else ``None``.

    Returns ``"mean"`` (value average → dag-ml ``merge_mode: "fusion"``) or ``"proba_mean"``
    (class-probability average → ``"fusion_proba_mean"``) for the two recognized spellings:

    * ``{"merge": "mean"}`` / ``{"merge": "average"}`` → ``"mean"``;
    * ``{"merge": {"predictions": "all", "aggregate": "mean"|"proba_mean"}}`` — the explicit
      aggregation-vocabulary form (``predictions`` collection + an ``aggregate`` reducer), with NO
      other keys (no per-branch ``select``/``metric``, no ``features``, no downstream model implied).

    Everything else (``"predictions"`` stacking, ``"concat"``, ``"features"``, a per-branch config,
    ``weighted_mean``, ``separate``) returns ``None`` so the bridge fails loud.
    """
    if not isinstance(step, dict) or "merge" not in step:
        return None
    spec = step["merge"]
    if isinstance(spec, str):
        return "mean" if spec in _FUSION_MERGE_STRINGS else None
    if isinstance(spec, dict):
        # Only the exact {"predictions": "all", "aggregate": <mean|proba_mean>} shape — nothing else.
        if set(spec) != {"predictions", "aggregate"} or spec.get("predictions") not in ("all", True):
            return None
        aggregate = spec.get("aggregate")
        return aggregate if aggregate in ("mean", "proba_mean") else None
    return None


def _is_stacking_merge_step(step: Any) -> bool:
    """True for a STACKING merge (``{"merge": "predictions"}`` or a per-branch predictions config).

    Stacking turns the branch OOF into meta-features for a downstream meta-model — a separate, larger
    subsystem (backlog #10). It is detected only to fail LOUD with a clear #10 message, never run.
    """
    if not isinstance(step, dict) or "merge" not in step:
        return False
    spec = step["merge"]
    if spec == "predictions":
        return True
    return isinstance(spec, dict) and ("predictions" in spec) and _fusion_merge_aggregate(step) is None


def _detect_duplication_branch(pipeline: list[Any]) -> tuple[list[list[Any]], str] | None:
    """Detect the EXACT duplication-branch + avg/mean fusion-merge shape, else ``None`` (fail-loud).

    Admits ONLY a pipeline that is exactly: the splitter + ONE duplication branch
    (``{"branch": [[A], [B], ...]}`` with N≥2 sub-pipelines, each containing a model) + ONE avg/mean
    fusion merge (:func:`_fusion_merge_aggregate`). Returns ``(branches, aggregate)`` when matched
    (``aggregate`` is ``"mean"`` or ``"proba_mean"``). ANY deviation returns ``None`` so the bridge's
    raw-branch / raw-merge ``NotImplementedError`` fires. Specifically REJECTED (fall through to loud):

    * a STACKING merge (``{"merge": "predictions"}`` / a per-branch predictions config → a meta-model,
      backlog #10) — raised loud naming #10 by the caller, never silently averaged;
    * a separation (dict-form) branch — handled by :func:`_detect_separation_branch`, not here;
    * a sub-pipeline without a model (fusion averages MODEL predictions);
    * a top-level operator/transform/``tag``/``y_processing``/``exclude`` beside the branch (only each
      branch body is lowered, so a top-level step would be silently dropped) — out-of-scope follow-up;
    * a model after the merge, more than one branch/merge, or any unrecognized merge spelling.
    """
    branch_steps = [step for step in pipeline if _is_duplication_branch_step(step)]
    merge_aggregates = [(step, agg) for step in pipeline if (agg := _fusion_merge_aggregate(step)) is not None]
    if len(branch_steps) != 1 or len(merge_aggregates) != 1:
        return None
    branch_step = branch_steps[0]
    merge_step, aggregate = merge_aggregates[0]

    # The pipeline must be EXACTLY {splitter, branch, merge} — no other top-level steps. A top-level
    # transform / tag / y_processing / exclude / model would be silently ignored (each branch body is
    # lowered, not the top level), so its presence rejects the match → fail-loud.
    for step in pipeline:
        if step is branch_step or step is merge_step or hasattr(step, "split"):
            continue
        return None

    branches = branch_step["branch"]
    # Every sub-pipeline must contain a model — fusion averages MODEL predictions; a modelless branch
    # (features only) is not the supported shape.
    if not all(any(isinstance(sub, dict) and "model" in sub for sub in branch) for branch in branches):
        return None
    return branches, aggregate


def _is_simple_predictions_merge_step(step: Any) -> bool:
    """True ONLY for the exact ``{"merge": "predictions"}`` stacking-merge string — nothing richer.

    A per-branch predictions config (``{"merge": {"predictions": [{"branch": 0, "select": "best"}]}}``)
    carries model-selection/aggregation semantics this slice does NOT honor, so it is rejected (it stays
    on the loud #10 path). Only the plain ``"predictions"`` collect-all merge is the supported stacking shape.
    """
    return isinstance(step, dict) and step.get("merge") == "predictions"


def _is_default_except_level(config: Any) -> bool:
    """True iff ``config`` is a ``StackingConfig`` equal to the default in EVERY field except ``level``.

    A MetaModel may carry only the stacking options this slice actually HONORS. ``level`` is the one
    permitted deviation (AUTO / LEVEL_1 → a single base→meta level, which the dag-ml lowering produces);
    every other field (``coverage_strategy``, ``test_aggregation``, ``branch_scope``, ``allow_no_cv``,
    ``min_coverage_ratio``, ``allow_meta_sources``, ``max_level``, ``relation_profile``) is SILENTLY
    IGNORED by the lowering — notably ``test_aggregation``, which has no effect because this slice cannot
    score test meta-features at all (best_rmse is NaN). So a non-default value of any of those must reject
    the stacking shape (fail loud) rather than run with the option dropped. Comparison is field-exhaustive
    by construction: clone the config with ``level`` reset to the default and compare to a fresh default,
    so any future ``StackingConfig`` field is covered without enumerating them here.
    """
    import dataclasses

    from nirs4all.operators.models.meta import StackingConfig

    if not isinstance(config, StackingConfig):
        return False
    normalized = dataclasses.replace(config, level=StackingConfig().level)
    return normalized == StackingConfig()


def _meta_learner(model_step: dict[str, Any]) -> Any | None:
    """The sklearn meta-learner estimator from a downstream ``{"model": …}`` stacking step, else ``None``.

    Two equivalent nirs4all spellings (per ``MergeController``'s own docstring): a ``MetaModel`` wrapper
    (``{"model": MetaModel(Ridge())}`` — the meta-learner is its wrapped ``.model``) or a plain sklearn
    estimator (``{"model": Ridge()}`` after ``{"merge": "predictions"}``). Either way we return the bare
    sklearn estimator that fits on the meta-feature matrix.

    Returns ``None`` (→ fail loud, never run wrong) for any MetaModel option this slice does not honor:
    a non-default ``source_models`` list, ``use_proba``, a custom ``selector``, a ``finetune_space``, a
    non-AUTO/non-1 stacking ``level``, OR any OTHER non-default ``stacking_config`` field
    (``test_aggregation``, ``coverage_strategy``, … — silently ignored by the lowering; see
    :func:`_is_default_except_level`).
    """
    from nirs4all.operators.models.meta import MetaModel, StackingLevel

    model = model_step.get("model")
    if isinstance(model, MetaModel):
        config = model.stacking_config
        if (
            model.source_models != "all"
            or model.use_proba
            or model.selector is not None
            or model.finetune_space is not None
            or config.level not in (StackingLevel.AUTO, StackingLevel.LEVEL_1)
            or not _is_default_except_level(config)
        ):
            return None
        return model.model
    # A plain sklearn estimator step (no other generator/reserved sibling that would change its meaning).
    if model is not None and hasattr(model, "fit") and hasattr(model, "predict"):
        return model
    return None


def _detect_stacking_branch(pipeline: list[Any]) -> tuple[list[list[Any]], Any] | None:
    """Detect the EXACT duplication-branch + ``{"merge": "predictions"}`` + meta-model shape, else ``None``.

    Admits ONLY: the splitter + ONE duplication branch (``{"branch": [[A], [B], …]}``, N≥2 sub-pipelines
    each with a model) + ONE ``{"merge": "predictions"}`` + ONE downstream ``{"model": M}`` whose M is a
    handled meta-learner (a default ``MetaModel`` wrapper or a plain sklearn estimator; see
    :func:`_meta_learner`). Returns ``(branches, meta_learner)`` (the bare sklearn estimator) when matched.
    ANY deviation returns ``None`` so the bridge / the loud #10 path fires — never a silent-wrong run.
    Specifically REJECTED (fall through to loud):

    * a fusion/avg merge or a concat merge (those are the duplication-fusion / separation paths);
    * a per-branch predictions config (model-selection/aggregation semantics not honored — stays #10);
    * a missing downstream model, more than one branch/merge/model, or a model BEFORE the merge;
    * a top-level operator/transform/``tag``/``y_processing``/``exclude`` beside the branch (only each
      branch body is lowered, so a top-level step would be silently dropped) — out-of-scope follow-up;
    * a sub-pipeline without a model (the base level needs a model to produce OOF);
    * a MetaModel carrying unhandled options (non-default source_models/use_proba/selector/finetune/config);
    * a meta-model step carrying a sibling param (``{"model": Ridge(), "alpha": 0.2}``) or a generator
      (``{"model": Ridge(), "alpha": {"_range_": [...]}}``): the meta-model node is lowered as a bare
      estimator, so ``_apply_model_params`` / native generation never run for it — the param/sweep would
      be silently ignored. A tuned/swept meta-model is a later slice.
    """
    from nirs4all.pipeline.dagml_bridge import is_param_generator_spec

    branch_steps = [step for step in pipeline if _is_duplication_branch_step(step)]
    merge_steps = [step for step in pipeline if _is_simple_predictions_merge_step(step)]
    model_steps = [step for step in pipeline if isinstance(step, dict) and "model" in step]
    if len(branch_steps) != 1 or len(merge_steps) != 1 or len(model_steps) != 1:
        return None
    branch_step, merge_step, model_step = branch_steps[0], merge_steps[0], model_steps[0]

    # The meta-model step must be a BARE {"model": <estimator>} (plus harmless reserved keys like name):
    # any extra non-reserved sibling param OR a param-generator on the meta step is silently dropped by the
    # bare-estimator lowering, so reject it (fail loud) rather than run the meta-model with the option lost.
    if any(key not in _RESERVED_STEP_KEYS or is_param_generator_spec(value) for key, value in model_step.items() if key != "model"):
        return None

    # The merge must come BEFORE the model (the model is the meta-learner over the merged OOF), and the
    # pipeline must be EXACTLY {splitter, branch, merge, model} — no other top-level steps (a top-level
    # transform / tag / y_processing / exclude would be silently dropped, since only branch bodies are
    # lowered). Order + membership are both enforced.
    order = [step for step in pipeline if step is branch_step or step is merge_step or step is model_step]
    if order != [branch_step, merge_step, model_step]:
        return None
    for step in pipeline:
        if step is branch_step or step is merge_step or step is model_step or hasattr(step, "split"):
            continue
        return None

    branches = branch_step["branch"]
    if not all(any(isinstance(sub, dict) and "model" in sub for sub in branch) for branch in branches):
        return None
    meta_learner = _meta_learner(model_step)
    if meta_learner is None:
        return None
    return branches, meta_learner
