"""Pipeline step-shape detection for the dag-ml backend.

The ``_is_*`` step predicates, the ``_detect_*`` composition detectors (separation / by_source /
duplication / stacking branches, rep-fusion), the fusion-merge / meta-learner parsers, and the
``_generation_kind`` classifier — the conservative, fail-loud recognizers that decide which native
dag-ml path a pipeline takes (and reject anything not exactly handled).
"""

from __future__ import annotations

from typing import Any

from .steps import _RESERVED_STEP_KEYS, _is_split_step


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


def _source_concat_indices(source_spec: Any, n_sources: int) -> list[int] | None:
    """Resolve a ``{"merge": {"sources": ...}}`` concat spec to source indices, else ``None``.

    This is a deliberately small parser for the native source-concat boundary: it accepts the
    production shorthand ``"concat"`` and the dict form when the strategy/mode is concat. Source names
    are the legacy/default ``source_<index>`` names used by ``MergeController`` for unnamed sources.
    """
    strategy: Any = "concat"
    sources: Any = "all"
    if isinstance(source_spec, str):
        strategy = source_spec
    elif isinstance(source_spec, dict):
        raw_strategy = source_spec.get("strategy", source_spec.get("mode", "concat"))
        if not isinstance(raw_strategy, str):
            return None
        strategy = raw_strategy
        sources = source_spec.get("sources", source_spec.get("select", "all"))
    else:
        return None

    if strategy != "concat":
        return None
    if sources == "all":
        return list(range(n_sources))
    if not isinstance(sources, list) or not sources:
        return None

    indices: list[int] = []
    for source in sources:
        if isinstance(source, int) and not isinstance(source, bool):
            index = source
        elif isinstance(source, str) and source.startswith("source_"):
            try:
                index = int(source.removeprefix("source_"))
            except ValueError:
                return None
        else:
            return None
        if not 0 <= index < n_sources:
            return None
        indices.append(index)
    return indices


def _is_stateless_x_transform(step: Any) -> bool:
    """Whether a pre-merge transform is safe to replay per fold/source for this native boundary."""
    if isinstance(step, dict) or _is_split_step(step):
        return False
    if not (hasattr(step, "fit") and hasattr(step, "transform")):
        return False
    tags = step._more_tags() if hasattr(step, "_more_tags") else {}
    return bool(tags.get("stateless"))


def _detect_source_concat_merge(pipeline: list[Any], n_sources: int) -> tuple[list[Any], list[Any], list[int]] | None:
    """Detect the exact top-level source-concat boundary this bridge can reproduce natively.

    Supported shape:

    ``X-transform* -> {"merge": {"sources": "concat"}} -> splitter -> model``

    The merge is a feature-layout boundary, not a model/fusion node: upstream X transforms must be
    applied independently to each selected source and only then hstacked for the downstream model. Richer
    compositions stay on the existing fallback path until their layout contracts are proven.
    """
    if n_sources <= 1:
        return None

    merge_hits: list[tuple[int, list[int]]] = []
    for index, step in enumerate(pipeline):
        if not isinstance(step, dict) or "merge" not in step:
            continue
        merge_config = step["merge"]
        if not isinstance(merge_config, dict) or "sources" not in merge_config:
            continue
        source_indices = _source_concat_indices(merge_config["sources"], n_sources)
        if source_indices is None or len(source_indices) < 2:
            return None
        if source_indices != list(range(n_sources)):
            return None
        merge_hits.append((index, source_indices))
    if len(merge_hits) != 1:
        return None

    merge_index, source_indices = merge_hits[0]
    pre_merge = pipeline[:merge_index]
    post_merge = pipeline[merge_index + 1 :]

    # Keep this native contract narrow: no branch/sample/exclude/rep/generator composition, and the
    # splitter must live after the source-layout boundary so the downstream model sees the merged matrix.
    if any(
        isinstance(step, dict)
        and (
            "branch" in step
            or "sample_augmentation" in step
            or "exclude" in step
            or "tag" in step
            or any(key in step for key in _REP_FUSION_KEYS)
        )
        for step in pipeline
    ):
        return None
    if any(_is_split_step(step) for step in pre_merge):
        return None
    if not all(_is_stateless_x_transform(step) for step in pre_merge):
        return None
    if not any(_is_split_step(step) for step in post_merge):
        return None
    if any(isinstance(step, dict) and ("model" in step or "y_processing" in step) for step in pre_merge):
        return None
    post_body = [step for step in post_merge if not _is_split_step(step)]
    if len(post_body) != 1 or not isinstance(post_body[0], dict) or "model" not in post_body[0]:
        return None
    if any(isinstance(step, dict) and "merge" in step for step in pre_merge + post_merge):
        return None

    return pre_merge, post_merge, source_indices


# Step keywords whose presence forces the Python path even alongside a model param sweep: Optuna
# finetune / per-model train kwargs are not part of the native generation+SELECT contract, so a
# pipeline carrying them must NOT be mistaken for a clean param-sweep-only pipeline.
_FORCE_PYTHON_STEP_KEYS = frozenset({"finetune_params", "train_params"})

# Sampling modifiers that DEMOTE a constrained operator generator to the Python expand path (ADR-17 item 5
# B MUST-FIX 1): legacy `expand_spec` SAMPLES the post-prune survivors (seeded subsample / weighted random),
# but dag-ml's native generator `count` TRUNCATES (a different set) and rejects `count == 0` (legacy = "no
# limit"). Keeping these Python-side avoids a silent survivor-set divergence / compile break.
_CONSTRAINED_SAMPLING_MODIFIER_KEYS = frozenset({"count", "_seed_", "_weights_"})


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
    from nirs4all.pipeline.dagml_bridge import is_param_generator_spec, step_has_native_grid

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
            # A step-level `_grid_` over model params the native dag-ml `Grid` generator can represent
            # (flat Cartesian product of plain, alphabetically-ordered param lists, no variant-changing
            # modifier sibling). A non-native `_grid_` (nested-generator / modifier-bearing / non-
            # alphabetical / non-JSON) is caught by the per-sibling generator-keyword check below.
            native_grid = step_has_native_grid(step)
            if native_grid:
                has_param_model = True
            for key, value in step.items():
                if key in _RESERVED_STEP_KEYS:
                    continue
                if key == "_grid_" and native_grid:
                    continue  # the native grid was already counted
                if is_param_generator_spec(value):
                    has_param_model = True
                elif has_nested_generator_keywords(value) or key in GENERATION_KEYWORDS:
                    # A generator-shaped sibling we cannot lower natively (a non-native `_grid_`, the
                    # dict-form range, a modifier-bearing range, or a nested-generator grid value) —
                    # Python expand owns it.
                    has_other = True
        elif GENERATION_KEYWORDS & set(step) or has_nested_generator_keywords(step):
            # Any generator on a non-model step (bare `_or_`/`_range_`/... or a nested one).
            has_other = True
    if has_other:
        return "operator"
    return "param_model" if has_param_model else "none"


def _is_flat_single_operator_generator(pipeline: list[Any]) -> bool:
    """True ONLY for the canonical FLAT-SINGLE operator ``_or_`` shape the native operator-SELECT lowers.

    CONSERVATIVE whitelist (#23 Phase 7 + ADR-17 item 5 slice D): native operator generation is taken ONLY
    when the pipeline has EXACTLY ONE generator step, that step is a bare ``_or_`` (NO selector / constraint)
    of single bare operators OR multi-step bare-operator sub-pipelines, and nothing forces the Python path.
    The native lowering (:func:`~nirs4all.pipeline.dagml_bridge._lower_operator_generator`) additionally
    re-validates the choice shape and raises for anything richer, so a slip here is caught by the inner
    fallback — but keeping the predicate tight avoids a wasted native attempt. ALL of:

    * exactly ONE step carries a top-level generation keyword, and that keyword set is exactly ``{"_or_"}``
      (NO ``_cartesian_``/``_grid_``/``_chain_``/``_zip_``/``_sample_``, NO second generator, NO multi-model
      ``{"model": {"_or_": …}}``); AND
    * every ``_or_`` choice is a single bare operator OR a MULTI-STEP list of bare operators (``[SNV,
      Detrend]``, slice D — each survivor is exactly that one choice's flat operator list, with NO
      pick/arrange recombination); a ``{"model": …}`` multi-model choice, a NESTED-generator dict choice, a
      ``None`` no-op, or a list nesting a list/dict element forces the Python path; AND
    * every leaf operator (the single op, or EACH operator of a multi-step choice) is a genuinely ROUTABLE
      X-transform — the SAME routability / FQN-importability / wavelength gate
      :func:`~nirs4all.pipeline.dagml.steps._assert_supported_operators` applies to every other native path
      (:func:`~nirs4all.pipeline.dagml.steps._check_x_operator`). A non-routable / non-reconstructible /
      wavelength-requiring choice would slip into native operator-SELECT and fail at fit (the native run
      SKIPS the ``_or_`` step in its own support check), so it forces the Python path; AND
    * the only sibling keys on the ``_or_`` step are inert annotations (``_tags_``/``_metadata_``/``name``) —
      any modifier/constraint (``pick``/``arrange``/``count``/``_mutex_``/``_requires_``/``_exclude_``/
      ``_weights_``/``_seed_``) forces the Python path (a selector over multi-step choices makes nested-list
      survivors the native fingerprint cannot key); AND
    * NO step carries ``finetune_params`` / ``train_params``.

    A non-``_or_`` operator generator, a constrained/modified ``_or_``, a multi-model ``_or_``, a
    nested-generator or non-routable choice, a cartesian/grid, or a finetune/train_params pipeline →
    ``False`` (stays on the proven Python expand path, still dag-ml-native via that route). When in doubt,
    this returns ``False``.
    """
    from nirs4all.pipeline.config._generator.keywords import GENERATION_KEYWORDS, has_nested_generator_keywords
    from nirs4all.pipeline.dagml_bridge import _INERT_GENERATOR_ANNOTATION_KEYS, _is_bare_operator_choice, _is_multistep_operator_choice, _operator_choice_operators

    generator_steps = [step for step in pipeline if isinstance(step, dict) and GENERATION_KEYWORDS & set(step)]
    if len(generator_steps) != 1:
        return False
    # No OTHER generator anywhere — a nested generator on a non-generator step (incl. a multi-model
    # `{"model": {"_or_": …}}`) or finetune/train_params forces the Python path.
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        if step is not generator_steps[0] and has_nested_generator_keywords(step):
            return False
        if _FORCE_PYTHON_STEP_KEYS & set(step):
            return False

    generator = generator_steps[0]
    if GENERATION_KEYWORDS & set(generator) != {"_or_"}:
        return False
    if set(generator) - {"_or_"} - _INERT_GENERATOR_ANNOTATION_KEYS:
        return False
    choices = generator["_or_"]
    # Every choice is a NON-None single bare operator OR a MULTI-STEP list of bare ops. A NESTED generator
    # anywhere inside a choice (a dict choice, or a list element that is a dict/list) makes neither
    # `_is_bare_operator_choice` nor `_is_multistep_operator_choice` true → demoted here. A `None` no-op
    # choice is bare-typed but NOT lowerable to a transform node (it is the "no preprocessing" idiom the
    # Python expand path DROPS), so it forces the Python path — excluded explicitly here (slice-D scope is
    # concrete bare ops only).
    if not (isinstance(choices, list) and bool(choices) and all((choice is not None and _is_bare_operator_choice(choice)) or _is_multistep_operator_choice(choice) for choice in choices)):
        return False
    # Every LEAF operator must be a genuinely routable X-transform (same gate as the other native paths) — a
    # non-routable / wavelength-requiring / non-reconstructible choice would otherwise slip into native
    # operator-SELECT (which skips the `_or_` in its own support check) and crash at fit. For a multi-step
    # choice, each of its operators is checked.
    return all(all(_choice_is_native_routable(operator) for operator in _operator_choice_operators(choice)) for choice in choices)


def _choice_is_native_routable(choice: Any) -> bool:
    """True when an ``_or_`` operator choice passes the native X-transform routability gate (boolean form).

    Wraps :func:`~nirs4all.pipeline.dagml.steps._check_x_operator` (the wavelength + sklearn-contract +
    FQN-importability + lossless-params gate every other native path enforces) so the flat-single predicate
    can keep a non-routable choice OFF the native path (→ Python expand) instead of dispatching it to native
    operator-SELECT where it would fail at fit.
    """
    from nirs4all.pipeline.dagml.errors import DagMlUnsupported
    from nirs4all.pipeline.dagml.steps import _check_x_operator

    try:
        _check_x_operator(choice)
    except DagMlUnsupported:
        return False
    return True


def _is_constrained_operator_generator(pipeline: list[Any]) -> bool:
    """True ONLY for a CONSTRAINED operator generator the native operator-SELECT lowers (ADR-17 item 1a + 1b-cartesian).

    The constrained sibling to :func:`_is_flat_single_operator_generator`: it admits the operator-content
    generators a bare ``_or_`` cannot — a ``pick``/``arrange``-combinatorial ``_or_`` with
    ``_mutex_``/``_requires_``/``_exclude_`` constraints, OR a constrained ``_cartesian_`` of ``_or_`` stages.
    Each survivor is a multi-operator SEQUENCE (not a single op). The host expands the survivor set with
    nirs4all's own ``expand_spec`` (the constraint source of truth both engines use) and lowers each survivor
    into ONE native operator-variant branch, so dag-ml scores the SAME pruned set by CV-OOF, refits the
    winner, and stamps each per-variant report with the multi-op ``variant_label`` content fingerprint the
    host recomputes byte-identically (:func:`~nirs4all.pipeline.dagml.result._native_operator_config_by_label`).

    CONSERVATIVE whitelist — native is taken ONLY when ALL of:

    * EXACTLY ONE generator step, and it is a PURE ``_or_`` (keys ⊆ ``PURE_OR_KEYS``) carrying a
      ``_mutex_``/``_requires_``/``_exclude_`` CONSTRAINT (item 1a, the ``pick``-combinatorial set the
      constraint prunes), OR a PURE ``_cartesian_`` (keys ⊆ ``PURE_CARTESIAN_KEYS``) carrying a CONSTRAINT
      (item 1b-cartesian); a NON-pure node (a generator merged with ``class``/``model``/other keys), a bare
      ``_or_``/``_cartesian_``, or an unconstrained pick/arrange/``_cartesian_`` forces the Python path; AND
    * no OTHER generator anywhere (no nested generator on a non-generator step, no multi-model
      ``{"model": {"_or_": …}}``), and no ``finetune_params`` / ``train_params``; AND
    * every leaf operator choice is a genuinely ROUTABLE bare X-transform (the SAME
      :func:`_choice_is_native_routable` gate the flat-single predicate enforces) — a ``None`` choice, a
      ``{"model": …}`` choice, a nested generator choice, or a non-routable / wavelength-requiring choice
      forces the Python path; AND
    * the pipeline carries exactly one downstream concrete model (the survivor sequence terminates in it).

    Anything richer (an UNCONSTRAINED pick/arrange/``then_*``/``count``/``_seed_``/``_weights_`` ``_or_`` or
    a pick-only ``_cartesian_`` — out of the constrained scope; a ``_grid_``/``_zip_``/``_chain_``/
    ``_sample_`` operator generator; a non-pure node; a non-routable choice) → ``False`` (stays on the proven
    Python ``expand_spec`` path, still dag-ml-native via that route). When in doubt, this returns ``False``.
    """
    from nirs4all.pipeline.config._generator.keywords import (
        CONSTRAINT_KEYWORDS,
        GENERATION_KEYWORDS,
        has_nested_generator_keywords,
        is_pure_cartesian_node,
        is_pure_or_node,
    )

    generator_steps = [step for step in pipeline if isinstance(step, dict) and GENERATION_KEYWORDS & set(step)]
    if len(generator_steps) != 1:
        return False
    generator = generator_steps[0]

    # No OTHER generator anywhere, and no finetune/train_params — same fail-loud guards as the flat path.
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        if step is not generator and has_nested_generator_keywords(step):
            return False
        if _FORCE_PYTHON_STEP_KEYS & set(step):
            return False

    # Exactly one concrete downstream model — the survivor sequence terminates in it (a multi-model or a
    # generator-valued model is excluded by the no-other-generator guard above; a missing model is not a
    # CV+refit shape).
    model_steps = [step for step in pipeline if isinstance(step, dict) and "model" in step]
    if len(model_steps) != 1:
        return False

    # --- FAIL-CLOSED ADMIT GATE (ADR-17 item 5 B, round-4) ------------------------------------------------
    # Native operator-SELECT is taken ONLY for the EXACT shapes proven legacy-equivalent below; ANY other
    # shape demotes to the proven Python `expand_spec` path (still dag-ml-native via that route). The root
    # pattern of every prior divergence is the SAME: legacy `OrStrategy`/`CartesianStrategy` are LENIENT (skip
    # constraints on single-op variants, no-op degenerate groups, skip oversize / zero selections, seeded
    # sample on `count`), while dag-ml's native sequence-build is STRICT (applies constraints, rejects
    # degenerate groups / zero / oversize). So we whitelist only the strict-==-lenient intersection.
    keys = set(generator)

    # (A) PURITY + STRUCTURE: a PURE `_or_` with an INTEGER `pick`/`arrange` in [1, n_options] and NO
    #     `then_pick`/`then_arrange`, OR a PURE `_cartesian_` of >=2 routable `_or_` stages with NO
    #     `pick`/`arrange`/`then_*`. A bare `_or_` (no pick/arrange) makes single-op variants legacy SKIPS
    #     constraints on (`OrStrategy._apply_constraints`: `not isinstance(results[0], list)` → returned
    #     unfiltered) while native applies them (EDGE A) — so a constrained `_or_` MUST carry a list-producing
    #     pick/arrange. A range/tuple `pick` (mixed cardinality), a `then_*` second-order selection (a
    #     different survivor structure), or a `_cartesian_` `pick`/`arrange` (pipeline-pair selection, not the
    #     multi-op-sequence shape) all demote.
    if "_cartesian_" in generator:
        if not is_pure_cartesian_node(generator):
            return False
        if "pick" in keys or "arrange" in keys:
            return False
        stages = generator["_cartesian_"]
        if not isinstance(stages, list) or len(stages) < 2:
            return False
    elif "_or_" in generator:
        if not is_pure_or_node(generator):
            return False
        if "then_pick" in keys or "then_arrange" in keys:
            return False
        options = generator["_or_"]
        if not isinstance(options, list) or not options:
            return False
        # EXACTLY ONE of pick / arrange, an INTEGER in [1, n_options] (EDGE A: not bare; EDGE C: not 0, not
        # oversize, not a range/tuple). A bare constrained `_or_` (neither) demotes.
        selector_keys = {"pick", "arrange"} & keys
        if len(selector_keys) != 1:
            return False
        size = generator[next(iter(selector_keys))]
        if not isinstance(size, int) or isinstance(size, bool) or not (1 <= size <= len(options)):
            return False
    else:
        return False

    # (B) CONSTRAINED scope: at least one `_mutex_`/`_requires_`/`_exclude_` (a bare pick/arrange `_or_` is the
    #     flat / unconstrained path).
    if not (CONSTRAINT_KEYWORDS & keys):
        return False

    # (C) SAMPLING modifiers (`count`/`_seed_`/`_weights_`) DEMOTE (MUST-FIX 1): legacy SAMPLES the post-prune
    #     survivors (seeded subsample; `count <= 0` means "no limit"), but dag-ml's native `count` TRUNCATES (a
    #     different set) and rejects `count == 0`. `_weights_` is weighted random with no native analogue.
    if _CONSTRAINED_SAMPLING_MODIFIER_KEYS & keys:
        return False

    # (D) Every leaf operator choice is a ROUTABLE bare X-transform (no None / model / nested-generator /
    #     non-routable / wavelength-requiring choice). Walk the `_or_` / `_cartesian_`→`_or_` stage choices.
    if not _constrained_choices_native_routable(generator):
        return False

    # (E) NO cross-stage/branch DUPLICATE option (MUST-FIX 2 + 5): the bridge resolves each constraint ref
    #     through ONE global `_normalize_item` identity map to the FIRST branch of that identity, but legacy
    #     set-membership matches the ref from ANY branch — so a repeated identity across the whole generator
    #     mis-resolves natively.
    if _constrained_generator_has_duplicate_options(generator):
        return False

    # (F) Every CONSTRAINT GROUP is WELL-FORMED + NON-DEGENERATE + its refs resolve to real options (EDGE B +
    #     MUST-FIX 3/6): legacy normalizes sets and silently no-ops a degenerate group (`_mutex_:[[A]]` /
    #     `[[A,A]]`, a self-pair, an empty group), and an empty `_requires_` group crashes the bridge, while
    #     native rejects mutex <2 distinct refs / repeated requires-exclude pairs. `_exclude_` additionally
    #     routes native ONLY at uniform survivor cardinality == its (pair) cardinality (native SUBSET ==
    #     legacy EXACT-COMBO only then).
    return _constrained_constraints_well_formed(generator) and not _constrained_exclude_diverges(generator)


def _constrained_choices_native_routable(generator: dict[str, Any]) -> bool:
    """True iff EVERY leaf operator choice of a constrained ``_or_`` / ``_cartesian_`` is a routable bare X-transform.

    Walks the choice leaves: a ``_cartesian_`` of ``{"_or_": [...]}`` stages flattens to every stage's
    choices; a constrained ``_or_`` is its own choice list. Each leaf must be a single bare operator (NOT a
    ``None`` no-op, a ``{"model": …}``, a multi-step list, or a nested generator dict) AND pass the native
    routability gate (:func:`_choice_is_native_routable`) — otherwise the whole generator forces the Python
    expand path (where the survivor enumeration + per-variant native run cannot run safely).
    """
    from nirs4all.pipeline.dagml_bridge import _is_bare_operator_choice

    if "_cartesian_" in generator:
        stages = generator["_cartesian_"]
        if not isinstance(stages, list) or not stages:
            return False
        choice_lists = []
        for stage in stages:
            if not (isinstance(stage, dict) and set(stage) == {"_or_"} and isinstance(stage["_or_"], list) and stage["_or_"]):
                return False
            choice_lists.append(stage["_or_"])
    else:
        choices = generator["_or_"]
        if not (isinstance(choices, list) and choices):
            return False
        choice_lists = [choices]

    for choices in choice_lists:
        for choice in choices:
            if choice is None or not _is_bare_operator_choice(choice) or not _choice_is_native_routable(choice):
                return False
    return True


def _constrained_generator_all_options(generator: dict[str, Any]) -> list[Any]:
    """Every operator option of a constrained generator across the WHOLE generator (all ``_cartesian_`` stages, or the ``_or_``).

    The union — the SAME global identity space the bridge resolves constraint refs in (``dagml_bridge``
    `_native_constrained_generator` uses ONE global `identity.setdefault` keyed by ``_normalize_item``). The
    duplicate-option check operates on this union so a repeated identity ACROSS stages/branches (not just
    within one) is caught.
    """
    if "_cartesian_" in generator:
        return [option for stage in generator["_cartesian_"] for option in stage["_or_"]]
    return list(generator["_or_"])


def _constrained_generator_has_duplicate_options(generator: dict[str, Any]) -> bool:
    """True iff an operator identity appears in MORE THAN ONE branch/stage option of the constrained generator (MUST-FIX 2 + 5).

    A repeated ``_normalize_item`` identity — within ONE stage (MUST-FIX 2) OR across DIFFERENT
    ``_cartesian_`` stages / branches (MUST-FIX 5) — makes the native lowering diverge from legacy: the bridge
    resolves a constraint ref through ONE GLOBAL ``identity.setdefault`` to the FIRST branch carrying that
    identity, but legacy builds a normalized SET from the selected items, so the ref should match ANY branch
    of that identity. The native per-branch member rule then mis-resolves the ref (e.g. ``_cartesian_
    [[SNV,MSC],[SNV,Detrend]]`` with ``_requires_[[SNV,Detrend]]``: native pins SNV to stage 0 only). So a
    generator with ANY operator identity in two or more options DEMOTES to the Python expand path. Identity is
    nirs4all's own ``_normalize_item`` (the constraint-match key) so the check mirrors how a ref resolves.
    """
    from nirs4all.pipeline.config._generator.constraints import _normalize_item

    seen: set[Any] = set()
    for option in _constrained_generator_all_options(generator):
        key = _normalize_item(option)
        if key in seen:
            return True
        seen.add(key)
    return False


def _constrained_uniform_survivor_cardinality(generator: dict[str, Any]) -> int | None:
    """The number of operators EVERY survivor of the constrained generator carries, or ``None`` if it is not uniform/known.

    `_exclude_` parity depends on this: dag-ml native ``exclude`` forbids ANY co-occurrence of the pair
    (subset), while legacy ``_exclude_`` forbids the EXACT combo equal to the group — these agree ONLY when a
    survivor's cardinality equals the exclude-group cardinality. The cardinality is uniform and known only for:

    * ``_or_`` with a single-int ``pick`` (== that int) and NO ``arrange`` / ``then_pick`` / ``then_arrange``
      (a range pick, arrange, or second-order selector yields MIXED or permuted cardinalities); a bare ``_or_``
      with no selector picks ONE option (cardinality 1);
    * ``_cartesian_`` with NO ``pick`` / ``arrange`` (which would subsample) — its cardinality is the #stages.

    Anything else returns ``None`` (treated as a parity risk by the ``_exclude_`` guard).
    """
    if "_cartesian_" in generator:
        if "pick" in generator or "arrange" in generator:
            return None
        return len(generator["_cartesian_"])
    # `_or_`
    if "arrange" in generator or "then_pick" in generator or "then_arrange" in generator:
        return None
    pick = generator.get("pick")
    if pick is None:
        return 1  # a bare `_or_` selects exactly one option
    if isinstance(pick, int) and not isinstance(pick, bool):
        return pick
    return None  # a range/tuple pick yields mixed cardinalities


def _constrained_exclude_diverges(generator: dict[str, Any]) -> bool:
    """True iff an ``_exclude_`` would prune DIFFERENTLY natively than legacy — so the generator DEMOTES (MUST-FIX 3 + 6).

    Legacy ``_exclude_`` is EXACT-COMBO matching (``apply_exclude_constraint``,
    ``constraints.py``: ``frozenset(combo) not in exclude_normalized`` — a survivor is dropped ONLY when its
    full operator set EQUALS the exclude group), whereas dag-ml's native ``exclude`` is a SUBSET rule
    (``generation.rs`` ``constraints_satisfied``: ``present(left) && present(right)`` — drops ANY survivor
    where both co-occur). These agree ONLY when every exclude group is an exact 2-operator pair AND the
    survivor cardinality equals that pair size (2): then "the pair co-occurs" == "the survivor IS exactly that
    pair". So demote when ANY of:

    * an exclude group is not a 2-operator pair (no native pair form at all — MUST-FIX 3), OR
    * the survivor cardinality is not a known uniform 2 (e.g. ``_or_`` ``pick`` 3, or a ``_cartesian_`` of 3
      stages, where the legacy exact-2-combo never matches a 3-operator survivor but native subset would prune
      every survivor containing both — MUST-FIX 6).

    (``_mutex_`` = ``issubset`` "not all co-occur" MATCHES legacy at any cardinality, and a multi-``_requires_``
    splits into independent pairs, so only ``_exclude_`` needs this cardinality guard.)
    """
    exclude_groups = generator.get("_exclude_", [])
    if not exclude_groups:
        return False
    if any(len(group) != 2 for group in exclude_groups):
        return True
    return _constrained_uniform_survivor_cardinality(generator) != 2


def _constrained_constraints_well_formed(generator: dict[str, Any]) -> bool:
    """True iff EVERY constraint group is well-formed, NON-DEGENERATE, and refs resolve to real options (EDGE B).

    Legacy ``apply_all_constraints`` is LENIENT — it normalizes refs into SETS and silently no-ops a
    degenerate group (``_mutex_:[[A]]`` reduces to ``{A}.issubset(combo)``; ``[[A,A]]`` dedups to ``{A}``; an
    empty ``_requires_`` pair hits ``len < 2: continue``), and an UNKNOWN ref simply never matches — whereas
    dag-ml's native sequence-build is STRICT: it REJECTS a mutex group with <2 distinct refs, a
    requires/exclude pair with equal refs, and an unknown ref, and the bridge lowering crashes on an EMPTY
    ``_requires_`` group (``group[0]``). So a generator whose constraints are degenerate / repeated / empty /
    dangling routes the Python expand path. Each constraint type:

    * ``_mutex_``: every group has >=2 refs that are DISTINCT by ``_normalize_item`` identity;
    * ``_requires_``: every group has >=2 refs (the ``[trigger, each-subsequent]`` split the bridge emits),
      the trigger DISTINCT from each required ref (no self-require pair);
    * ``_exclude_``: every group is a 2-ref pair with DISTINCT refs (cardinality handled separately by
      :func:`_constrained_exclude_diverges`);
    * EVERY ref (across all three) resolves to a real option of the generator (by ``_normalize_item``).
    """
    from nirs4all.pipeline.config._generator.constraints import _normalize_item

    option_identities = {_normalize_item(option) for option in _constrained_generator_all_options(generator)}

    def refs_known(group: list[Any]) -> bool:
        return all(_normalize_item(ref) in option_identities for ref in group)

    for group in generator.get("_mutex_", []):
        if not isinstance(group, list) or not refs_known(group):
            return False
        distinct = {_normalize_item(ref) for ref in group}
        # Native rejects a mutex group with <2 DISTINCT refs AND one that REPEATS any ref (generation.rs
        # `distinct.len() != lowered.len()`), so `[[A, B, B]]` (2 distinct, 3 refs) must DEMOTE, not crash.
        if len(distinct) < 2 or len(distinct) != len(group):
            return False
    for group in generator.get("_requires_", []):
        if not isinstance(group, list) or len(group) < 2 or not refs_known(group):
            return False
        trigger = _normalize_item(group[0])
        if any(_normalize_item(req) == trigger for req in group[1:]):  # self-require → native rejects the pair
            return False
    for group in generator.get("_exclude_", []):
        if not isinstance(group, list) or len(group) != 2 or not refs_known(group):
            return False
        if _normalize_item(group[0]) == _normalize_item(group[1]):  # self-pair → native rejects
            return False
    return True


def _is_unconstrained_operator_generator(pipeline: list[Any]) -> bool:
    """True ONLY for an UNCONSTRAINED operator generator the native operator-SELECT lowers (ADR-17 item 5 slice C).

    The sibling to :func:`_is_constrained_operator_generator` for the NO-constraint operator-content shapes
    a bare ``_or_`` cannot express: a ``pick``/``arrange``-combinatorial ``_or_`` (each survivor a multi-op
    SEQUENCE) or a multi-stage ``_cartesian_`` of ``_or_`` stages — both with NO
    ``_mutex_``/``_requires_``/``_exclude_``. For the unconstrained case the survivor set is simply ALL
    pick/arrange/cartesian combinations (no constraint pruning), and dag-ml's
    ``expand_or_generator_sequences`` / ``expand_cartesian_generator_sequences`` produce EXACTLY that set in
    legacy ``itertools.combinations``/``permutations`` order — so
    :func:`~nirs4all.pipeline.dagml_bridge._native_constrained_generator` lowers them with an EMPTY
    constraints set, dag-ml scores the SAME survivor set by CV-OOF, refits the winner, and stamps each
    per-variant report with the multi-op ``variant_label`` content fingerprint the host recomputes
    byte-identically (the ``{variant_label → config_name}`` map aligns by CONTENT, not position).

    CONSERVATIVE whitelist — native is taken ONLY when ALL of (the SAME fail-closed gates as the constrained
    predicate, MINUS the constraint requirement, PLUS the no-constraint requirement):

    * EXACTLY ONE generator step, and it is a PURE ``_or_`` (keys ⊆ ``PURE_OR_KEYS``) with EXACTLY ONE
      INTEGER ``pick``/``arrange`` in ``[1, n_options]`` and NO ``then_pick``/``then_arrange``, OR a PURE
      ``_cartesian_`` (keys ⊆ ``PURE_CARTESIAN_KEYS``) of >=2 ``_or_`` stages with NO ``pick``/``arrange``;
      a bare ``_or_`` (no selector — that is the FLAT-SINGLE single-op path), a ``then_*`` second-order
      selection (a different survivor structure), an oversize/zero/range ``pick``, or a ``_cartesian_``
      ``pick``/``arrange`` (pipeline-PAIR selection, which dag-ml's cartesian mode refuses) all demote; AND
    * the generator carries NO ``_mutex_``/``_requires_``/``_exclude_`` (a constrained shape is the
      constrained predicate's job — this admits ONLY the unconstrained survivor enumeration); AND
    * no OTHER generator anywhere (no nested generator on a non-generator step, no multi-model
      ``{"model": {"_or_": …}}``), and no ``finetune_params`` / ``train_params``; AND
    * no ``count`` / ``_seed_`` / ``_weights_`` SAMPLING modifier (the Python expander samples the survivors —
      seeded / weighted / count-capped — but dag-ml's native ``count`` TRUNCATES a different set and has no
      weighted sampling analogue); AND
    * every leaf operator choice is a genuinely ROUTABLE bare X-transform (the SAME
      :func:`_constrained_choices_native_routable` gate) — a ``None`` choice, a multi-step list choice (the
      ``generator_or_multistep_branch`` shape), a ``{"model": …}`` choice, a nested-generator choice, or a
      non-routable / wavelength-requiring choice forces the Python path; AND
    * no DUPLICATE operator option across the whole generator (an equal-content option would make two
      survivors fingerprint-collide, mis-zipping the content-keyed ``{variant_label → config_name}`` map);
      AND
    * the pipeline carries exactly one downstream concrete model (the survivor sequence terminates in it).

    Anything richer (a ``then_*``/``count``/``_seed_``/``_weights_`` / oversize-pick ``_or_``, a
    ``pick``/``arrange``-bearing ``_cartesian_``, a non-pure node, a constrained node, a non-routable / dup
    option) → ``False`` (stays on the proven Python ``expand_spec`` path, still dag-ml-native via that
    route). When in doubt, this returns ``False``.
    """
    from nirs4all.pipeline.config._generator.keywords import (
        CONSTRAINT_KEYWORDS,
        GENERATION_KEYWORDS,
        has_nested_generator_keywords,
        is_pure_cartesian_node,
        is_pure_or_node,
    )

    generator_steps = [step for step in pipeline if isinstance(step, dict) and GENERATION_KEYWORDS & set(step)]
    if len(generator_steps) != 1:
        return False
    generator = generator_steps[0]

    # No OTHER generator anywhere, and no finetune/train_params — same fail-loud guards as the constrained path.
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        if step is not generator and has_nested_generator_keywords(step):
            return False
        if _FORCE_PYTHON_STEP_KEYS & set(step):
            return False

    # Exactly one concrete downstream model — the survivor sequence terminates in it.
    model_steps = [step for step in pipeline if isinstance(step, dict) and "model" in step]
    if len(model_steps) != 1:
        return False

    keys = set(generator)

    # (A) PURITY + STRUCTURE — IDENTICAL to the constrained admit gate (the survivor shape is the same; only
    #     the constraint prune differs). A PURE `_or_` with one INTEGER `pick`/`arrange` in [1, n] and no
    #     `then_*`, OR a PURE `_cartesian_` of >=2 `_or_` stages with no `pick`/`arrange`. ARRANGE-ORDER: dag-ml
    #     `build_permutations` enumerates in the SAME index order as legacy `itertools.permutations`, and the
    #     config map is content-keyed, so ordered survivors align (verified e2e); arrange therefore ADMITS.
    if "_cartesian_" in generator:
        if not is_pure_cartesian_node(generator):
            return False
        if "pick" in keys or "arrange" in keys:
            return False
        stages = generator["_cartesian_"]
        if not isinstance(stages, list) or len(stages) < 2:
            return False
    elif "_or_" in generator:
        if not is_pure_or_node(generator):
            return False
        if "then_pick" in keys or "then_arrange" in keys:
            return False
        options = generator["_or_"]
        if not isinstance(options, list) or not options:
            return False
        selector_keys = {"pick", "arrange"} & keys
        if len(selector_keys) != 1:
            return False
        size = generator[next(iter(selector_keys))]
        if not isinstance(size, int) or isinstance(size, bool) or not (1 <= size <= len(options)):
            return False
    else:
        return False

    # (B) UNCONSTRAINED scope: NO `_mutex_`/`_requires_`/`_exclude_` (a constrained node is the constrained
    #     predicate's domain). This is the ONLY gate that differs from `_is_constrained_operator_generator` —
    #     here the constraint set must be EMPTY, there it must be NON-empty.
    if CONSTRAINT_KEYWORDS & keys:
        return False

    # (C) SAMPLING modifiers (`count`/`_seed_`/`_weights_`) DEMOTE: the Python expander SAMPLES the
    #     survivors deterministically when `_seed_` is present; dag-ml's native `count` TRUNCATES a
    #     different set and has no weighted sampling analogue.
    if _CONSTRAINED_SAMPLING_MODIFIER_KEYS & keys:
        return False

    # (D) Every leaf operator choice is a ROUTABLE bare X-transform (no None / model / nested-generator /
    #     multi-step / non-routable / wavelength-requiring choice). Same walk as the constrained path.
    if not _constrained_choices_native_routable(generator):
        return False

    # (E) NO DUPLICATE operator option across the whole generator: two equal-content options would make two
    #     survivors fingerprint-collide, mis-zipping the content-keyed `{variant_label → config_name}` map.
    return not _constrained_generator_has_duplicate_options(generator)


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
        if step is branch_step or step is merge_step or _is_split_step(step):
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


def _detect_separation_preproc_concat(pipeline: list[Any]) -> tuple[dict[str, Any], list[Any], list[Any]] | None:
    """Detect by_metadata preprocessing + concat features + one downstream model.

    This is the single-source sibling of the by_source shared-preprocessing concat
    detector. It admits the narrow parity shape:

    * one ``by_metadata`` separation branch with a shared LIST ``steps`` body
      containing only stateless X transforms;
    * one ``{"merge": "concat"}`` immediately before one top-level model;
    * no other top-level operator/keyword beside the splitter, branch, merge,
      and model.

    The existing :func:`_detect_separation_branch` owns model-in-branch fan-out.
    This detector owns feature reassembly before a downstream model.
    """
    branch_steps = [step for step in pipeline if _is_separation_branch_step(step)]
    merge_steps = [step for step in pipeline if _is_concat_merge_step(step)]
    model_steps = [step for step in pipeline if isinstance(step, dict) and "model" in step]
    if len(branch_steps) != 1 or len(merge_steps) != 1 or len(model_steps) != 1:
        return None
    branch_step, merge_step, model_step = branch_steps[0], merge_steps[0], model_steps[0]

    order = [step for step in pipeline if step is branch_step or step is merge_step or step is model_step]
    if order != [branch_step, merge_step, model_step]:
        return None

    for step in pipeline:
        if step is branch_step or step is merge_step or step is model_step or _is_split_step(step):
            continue
        return None

    criterion = branch_step["branch"]
    if "by_metadata" not in criterion or set(criterion) - _HANDLED_BRANCH_KEYS:
        return None
    body = criterion.get("steps")
    if not isinstance(body, list) or not body:
        return None
    if any(isinstance(sub, dict) and "model" in sub for sub in body):
        return None
    if not all(_is_stateless_x_transform(sub) for sub in body):
        return None
    return branch_step, body, [model_step]


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
        if step is branch_step or step is merge_step or _is_split_step(step):
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


def _detect_by_source_concat_shared_preproc(pipeline: list[Any], n_sources: int) -> tuple[list[Any], list[Any]] | None:
    """Detect shared by_source preprocessing + concat features + one downstream model.

    Admits ONLY the narrow legacy shape used by
    ``multi_source_by_source_branch_shared_preproc``:

    * a multi-source dataset (``n_sources >= 2``);
    * one by_source branch with a shared LIST ``steps`` body that contains X preprocessing only
      (no model, no per-source dict, no extra branch options);
    * one ``{"merge": "concat"}`` immediately before one top-level downstream model;
    * no other top-level operator/keyword beside the splitter, branch, merge, and model.

    The existing :func:`_detect_by_source_branch` owns by_source MODEL fusion
    (``merge: mean``/``average``). This detector owns feature-axis concat reassembly and returns
    ``(preproc_body, downstream_body)`` for the dedicated runner.
    """
    branch_steps = [step for step in pipeline if _is_by_source_branch_step(step)]
    merge_steps = [step for step in pipeline if _is_concat_merge_step(step)]
    model_steps = [step for step in pipeline if isinstance(step, dict) and "model" in step]
    if len(branch_steps) != 1 or len(merge_steps) != 1 or len(model_steps) != 1 or n_sources < 2:
        return None
    branch_step, merge_step, model_step = branch_steps[0], merge_steps[0], model_steps[0]

    order = [step for step in pipeline if step is branch_step or step is merge_step or step is model_step]
    if order != [branch_step, merge_step, model_step]:
        return None

    for step in pipeline:
        if step is branch_step or step is merge_step or step is model_step or _is_split_step(step):
            continue
        return None

    criterion = branch_step["branch"]
    if set(criterion) - _HANDLED_BY_SOURCE_KEYS:
        return None
    body = criterion.get("steps")
    if not isinstance(body, list) or any(isinstance(sub, dict) and "model" in sub for sub in body):
        return None
    if not body:
        return None
    return body, [model_step]


def _detect_by_source_distinct_preproc_concat(pipeline: list[Any], n_sources: int) -> tuple[dict[str, list[Any]], list[Any]] | None:
    """Detect per-source by_source preprocessing + concat features + one downstream model.

    Admits ONLY the target feature-concat shape:

    * a multi-source dataset (``n_sources >= 2``);
    * one by_source branch whose ``steps`` body is a DICT of per-source preprocessing lists;
    * no model/y_processing inside any per-source body;
    * one ``{"merge": "concat"}`` followed immediately by one downstream model step;
    * no other top-level step except the splitter.

    Source-key validation is intentionally left to the runner, which consumes the explicit
    envelope ``source_layout.source_order`` contract and rejects any dict that does not match
    those legacy keys exactly.
    """
    branch_steps = [step for step in pipeline if _is_by_source_branch_step(step)]
    merge_steps = [step for step in pipeline if _is_concat_merge_step(step)]
    model_steps = [step for step in pipeline if isinstance(step, dict) and "model" in step]
    if len(branch_steps) != 1 or len(merge_steps) != 1 or len(model_steps) != 1 or n_sources < 2:
        return None
    branch_step, merge_step, model_step = branch_steps[0], merge_steps[0], model_steps[0]

    ordered_special = [step for step in pipeline if step is branch_step or step is merge_step or step is model_step]
    if ordered_special != [branch_step, merge_step, model_step]:
        return None
    for step in pipeline:
        if step is branch_step or step is merge_step or step is model_step or _is_split_step(step):
            continue
        return None

    criterion = branch_step["branch"]
    if set(criterion) - _HANDLED_BY_SOURCE_KEYS:
        return None
    body = criterion.get("steps")
    if not isinstance(body, dict) or len(body) != n_sources or not all(isinstance(key, str) for key in body):
        return None

    normalized: dict[str, list[Any]] = {}
    for key, value in body.items():
        if not isinstance(value, list):
            return None
        if any(isinstance(substep, dict) and ("model" in substep or "y_processing" in substep) for substep in value):
            return None
        normalized[key] = value
    return normalized, [model_step]


_DUPLICATION_BRANCH_CONFIG_KEYS = frozenset({"parallel", "n_jobs"})


def _duplication_branch_bodies(step: Any) -> list[list[Any]] | None:
    """The ordered duplication branch bodies for list or named-dict syntax, else ``None``.

    Legacy ``BranchController._detect_branch_mode`` treats list syntax and dict syntax without any
    ``by_*`` separation key as duplication. Dict insertion order is the branch order legacy executes and
    later merge configs address by integer index, so preserve it exactly while dropping only branch-level
    config/internal keys that ``BranchController._parse_branch_definitions`` skips.
    """
    if not isinstance(step, dict):
        return None
    branch = step.get("branch")
    if isinstance(branch, list) and len(branch) >= 2 and all(isinstance(sub, list) for sub in branch):
        return branch
    if isinstance(branch, dict):
        if any(key in branch for key in ("by_source", "by_tag", "by_metadata", "by_filter")):
            return None
        bodies = [
            steps if isinstance(steps, list) else [steps]
            for name, steps in branch.items()
            if isinstance(name, str) and not name.startswith("_") and name not in _DUPLICATION_BRANCH_CONFIG_KEYS
        ]
        if len(bodies) >= 2:
            return bodies
    return None


def _is_duplication_branch_step(step: Any) -> bool:
    """True for a DUPLICATION branch in legacy list or named-dict syntax."""
    return _duplication_branch_bodies(step) is not None


# The cross-branch fusion (avg / proba-mean) merge tokens this backend maps to dag-ml's native fusion
# merge handler. Simple-string ``"mean"``/``"average"`` (a NEW token — legacy MergeConfigParser rejects
# it, so there is no collision) average the branches' held-out OOF per sample into ONE final prediction;
# the explicit-aggregation dict form reuses nirs4all's established aggregation vocabulary
# (``AggregationStrategy.MEAN``/``PROBA_MEAN``). A STACKING merge (``{"merge": "predictions"}`` →
# MetaModel) is deliberately NOT a fusion token and is handled by the stacking detector.
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


def _simple_duplication_merge_mode(step: Any) -> str | None:
    """Return a simple legacy duplication merge mode this slice handles, else ``None``."""
    if not isinstance(step, dict) or "merge" not in step:
        return None
    spec = step["merge"]
    return spec if spec in ("features", "all") else None


def _model_step_is_plain_estimator(model_step: dict[str, Any]) -> bool:
    """True for a downstream bare sklearn-style model step, excluding MetaModel-owned W33 shapes."""
    from nirs4all.operators.models.meta import MetaModel
    from nirs4all.pipeline.dagml_bridge import is_param_generator_spec

    model = model_step.get("model")
    if isinstance(model, MetaModel):
        return False
    if model is None or not (hasattr(model, "fit") and hasattr(model, "predict")):
        return False
    return not any(key not in _RESERVED_STEP_KEYS or is_param_generator_spec(value) for key, value in model_step.items() if key != "model")


def _detect_duplication_branch(pipeline: list[Any]) -> tuple[list[list[Any]], str] | None:
    """Detect the handled duplication-branch merge shapes, else ``None`` (fail-loud).

    Admits:

    * splitter + one duplication branch + avg/mean fusion merge (legacy list or named-dict branch syntax),
      with every branch containing a model; and
    * splitter + one duplication branch + ``merge="features"`` + one downstream plain estimator.
      The branch bodies must be feature-only; and
    * splitter + one duplication branch + ``merge="all"`` + one downstream plain estimator.
      Every branch body must contain a branch-local model, and the native transformer concatenates
      branch feature snapshots followed by branch prediction columns, matching the legacy collector's
      feature-then-prediction ordering.

    ANY deviation returns ``None`` so the bridge's raw-branch / raw-merge guard fires. Specifically
    REJECTED (fall through to loud):

    * a STACKING merge (``{"merge": "predictions"}`` / a per-branch predictions config → a meta-model)
      — handled by :func:`_detect_stacking_branch` or raised loud naming #10 by the caller, never silently
      averaged;
    * a separation (``by_*`` dict-form) branch — handled by :func:`_detect_separation_branch`, not here;
    * a MetaModel or structured/per-branch merge config (left to W33);
    * a top-level operator/transform/``tag``/``y_processing``/``exclude`` beside the branch (only each
      branch body is lowered, so a top-level step would be silently dropped) — out-of-scope follow-up;
    * more than one branch/merge/model, a model in the wrong place, or any unrecognized merge spelling.
    """
    branch_steps = [step for step in pipeline if _is_duplication_branch_step(step)]
    merge_aggregates = [(step, agg) for step in pipeline if (agg := _fusion_merge_aggregate(step)) is not None]
    simple_merges = [(step, mode) for step in pipeline if (mode := _simple_duplication_merge_mode(step)) is not None]
    if len(branch_steps) != 1 or len(merge_aggregates) + len(simple_merges) != 1:
        return None
    branch_step = branch_steps[0]
    branches = _duplication_branch_bodies(branch_step)
    if branches is None:
        return None

    if merge_aggregates:
        merge_step, aggregate = merge_aggregates[0]
        # The pipeline must be EXACTLY {splitter, branch, merge} — no other top-level steps. A top-level
        # transform / tag / y_processing / exclude / model would be silently ignored (each branch body is
        # lowered, not the top level), so its presence rejects the match → fail-loud.
        for step in pipeline:
            if step is branch_step or step is merge_step or _is_split_step(step):
                continue
            return None

        # Every sub-pipeline must contain a model — fusion averages MODEL predictions; a modelless branch
        # (features only) is not the supported shape.
        if not all(any(isinstance(sub, dict) and "model" in sub for sub in branch) for branch in branches):
            return None
        return branches, aggregate

    merge_step, merge_mode = simple_merges[0]
    model_steps = [step for step in pipeline if isinstance(step, dict) and "model" in step]
    if len(model_steps) != 1:
        return None
    model_step = model_steps[0]
    if not _model_step_is_plain_estimator(model_step):
        return None
    order = [step for step in pipeline if step is branch_step or step is merge_step or step is model_step]
    if order != [branch_step, merge_step, model_step]:
        return None
    for step in pipeline:
        if step is branch_step or step is merge_step or step is model_step or _is_split_step(step):
            continue
        return None

    branch_has_model = [any(isinstance(sub, dict) and "model" in sub for sub in branch) for branch in branches]
    if merge_mode == "features" and any(branch_has_model):
        return None
    if merge_mode == "all" and not all(branch_has_model):
        return None
    return branches, merge_mode


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


def _branch_local_meta_model_step(model_step: dict[str, Any]) -> dict[str, Any] | None:
    """Return a handled branch-local ``MetaModel`` step for named prediction-feature stacking.

    This is deliberately separate from :func:`_meta_learner`: the target shape runs ``MetaModel``
    inside each named duplication branch context, then feeds a later structured prediction merge. Its
    non-default ``coverage_strategy`` / ``min_coverage_ratio`` are harmless when every branch model has
    complete OOF coverage, but all options that would change source selection, probabilities, test
    aggregation, branch scope, or generated meta-learner params still reject the shape.
    """
    import dataclasses

    from nirs4all.operators.models.meta import BranchScope, MetaModel, StackingConfig, StackingLevel, TestAggregation
    from nirs4all.pipeline.dagml_bridge import is_param_generator_spec

    if any(key not in _RESERVED_STEP_KEYS or is_param_generator_spec(value) for key, value in model_step.items() if key != "model"):
        return None

    model = model_step.get("model")
    if not isinstance(model, MetaModel):
        return None
    config = model.stacking_config
    default_config = StackingConfig()
    normalized = dataclasses.replace(
        config,
        coverage_strategy=default_config.coverage_strategy,
        min_coverage_ratio=default_config.min_coverage_ratio,
        level=default_config.level,
    )
    if (
        model.source_models != "all"
        or model.use_proba
        or model.selector is not None
        or model.finetune_space is not None
        or normalized != default_config
        or config.level not in (StackingLevel.AUTO, StackingLevel.LEVEL_1)
        or config.branch_scope != BranchScope.CURRENT_ONLY
        or config.test_aggregation != TestAggregation.MEAN
        or config.allow_no_cv
        or config.relation_profile
        or model.model is None
        or not (hasattr(model.model, "fit") and hasattr(model.model, "predict"))
    ):
        return None
    return model_step


def _structured_prediction_feature_merge(step: Any) -> list[dict[str, Any]] | None:
    """Return the handled per-branch best-prediction merge config, else ``None``."""
    if not isinstance(step, dict) or "merge" not in step:
        return None
    spec = step["merge"]
    if not isinstance(spec, dict) or set(spec) != {"predictions", "output_as"} or spec.get("output_as") != "features":
        return None
    prediction_specs = spec.get("predictions")
    if not isinstance(prediction_specs, list) or not prediction_specs:
        return None
    out: list[dict[str, Any]] = []
    for item in prediction_specs:
        if not isinstance(item, dict):
            return None
        if set(item) != {"branch", "select", "metric"}:
            return None
        branch = item.get("branch")
        if not isinstance(branch, int) or isinstance(branch, bool):
            return None
        if item.get("select") != "best" or item.get("metric") != "rmse":
            return None
        out.append(dict(item))
    return out


def _named_duplication_branch_parts(step: dict[str, Any]) -> tuple[list[str], list[list[Any]]] | None:
    """Ordered branch names + bodies for legacy named-dict duplication syntax."""
    branch = step.get("branch")
    if not isinstance(branch, dict) or any(key in branch for key in ("by_source", "by_tag", "by_metadata", "by_filter")):
        return None
    names: list[str] = []
    bodies: list[list[Any]] = []
    for name, body in branch.items():
        if not isinstance(name, str) or name.startswith("_") or name in _DUPLICATION_BRANCH_CONFIG_KEYS:
            continue
        names.append(name)
        bodies.append(body if isinstance(body, list) else [body])
    if len(bodies) < 2:
        return None
    return names, bodies


def _detect_named_metamodel_feature_stack(pipeline: list[Any]) -> tuple[list[str], list[list[Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]] | None:
    """Detect named branch-local ``MetaModel`` + structured prediction merge → downstream model.

    The handled shape is intentionally narrow and mirrors the legacy W73 contract:

    ``splitter + named duplication branch + {"model": MetaModel(...)} +``
    ``{"merge": {"predictions": [{"branch": i, "select": "best", "metric": "rmse"}, ...], "output_as": "features"}} +``
    ``{"model": Ridge-like estimator}``

    Legacy emits CV-only rows for branch-local base models, branch-local ``MetaModel`` rows, and the
    downstream estimator; its refit skips named-dict stacking, so there must be no native final rows.
    Anything outside that contract returns ``None`` and stays on the explicit fallback boundary.
    """
    branch_steps = [step for step in pipeline if _is_duplication_branch_step(step)]
    merge_steps = [(step, config) for step in pipeline if (config := _structured_prediction_feature_merge(step)) is not None]
    model_steps = [step for step in pipeline if isinstance(step, dict) and "model" in step]
    if len(branch_steps) != 1 or len(merge_steps) != 1 or len(model_steps) != 2:
        return None

    branch_step = branch_steps[0]
    merge_step, prediction_configs = merge_steps[0]
    named_parts = _named_duplication_branch_parts(branch_step)
    if named_parts is None:
        return None
    branch_names, branches = named_parts

    meta_step = _branch_local_meta_model_step(model_steps[0])
    downstream_step = model_steps[1]
    if meta_step is None or not _model_step_is_plain_estimator(downstream_step):
        return None

    order = [step for step in pipeline if step is branch_step or step is meta_step or step is merge_step or step is downstream_step]
    if order != [branch_step, meta_step, merge_step, downstream_step]:
        return None
    for step in pipeline:
        if step is branch_step or step is meta_step or step is merge_step or step is downstream_step or _is_split_step(step):
            continue
        return None

    if not all(0 <= config["branch"] < len(branches) for config in prediction_configs):
        return None
    if not all(any(isinstance(sub, dict) and "model" in sub for sub in branch) for branch in branches):
        return None
    return branch_names, branches, meta_step, prediction_configs, downstream_step


def _detect_stacking_branch(pipeline: list[Any]) -> tuple[list[list[Any]], Any] | None:
    """Detect the EXACT duplication-branch + ``{"merge": "predictions"}`` + meta-model shape, else ``None``.

    Admits ONLY: the splitter + ONE duplication branch (legacy list-of-lists or named-dict syntax, N≥2
    sub-pipelines each with a model) + ONE ``{"merge": "predictions"}`` + ONE downstream ``{"model": M}``
    whose M is a handled meta-learner (a default ``MetaModel`` wrapper or a plain sklearn estimator; see
    :func:`_meta_learner`). Returns ``(branches, meta_learner)`` (the bare sklearn estimator) when matched.
    ANY deviation returns ``None`` so the bridge / the loud #10 path fires — never a silent-wrong run.
    Specifically REJECTED (fall through to loud):

    * a fusion/avg merge or a concat merge (those are the duplication-fusion / separation paths);
    * a per-branch predictions config (model-selection/aggregation semantics not honored — stays #10);
    * a missing downstream model, more than one branch/merge/model, or a model BEFORE the merge;
    * a top-level operator/transform/``tag``/``y_processing``/``exclude`` beside the branch (only each
      branch body is lowered, so a top-level step would be silently dropped) — out-of-scope follow-up;
    * named-dict duplication branches are admitted only for this default stacking shape; the runner requests
      dag-ml's CV-only stacking policy and projects the legacy no-refit row surface;
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
        if step is branch_step or step is merge_step or step is model_step or _is_split_step(step):
            continue
        return None

    branches = _duplication_branch_bodies(branch_step)
    if branches is None:
        return None
    if not all(any(isinstance(sub, dict) and "model" in sub for sub in branch) for branch in branches):
        return None
    meta_learner = _meta_learner(model_step)
    if meta_learner is None:
        return None
    return branches, meta_learner


def _detect_by_source_stacking_branch(pipeline: list[Any], n_sources: int) -> tuple[list[Any], Any] | None:
    """Detect the legacy by_source-model stacking shape that uses source-layout replay.

    Admits ONLY:

    ``splitter + {"branch": {"by_source": True, "steps": [X-transform*, {"model": Base}]}}
    + {"merge": "predictions"} + {"model": Meta}``

    This is not ordinary OOF stacking. In source-branch mode legacy's merge controller collects the
    cumulatively-mutated source feature layout for ``{"merge": "predictions"}``, writes that concat back
    to source 0, leaves the remaining sources in place, and then the downstream estimator trains on that
    post-merge source layout. The native runner has a dedicated replay for that exact contract, so any
    richer form stays on the loud fallback-boundary path rather than being run as 3-column OOF stacking.
    """
    from nirs4all.pipeline.dagml_bridge import is_param_generator_spec

    if n_sources < 2:
        return None
    branch_steps = [step for step in pipeline if _is_by_source_branch_step(step)]
    merge_steps = [step for step in pipeline if _is_simple_predictions_merge_step(step)]
    model_steps = [step for step in pipeline if isinstance(step, dict) and "model" in step]
    if len(branch_steps) != 1 or len(merge_steps) != 1 or len(model_steps) != 1:
        return None
    branch_step, merge_step, model_step = branch_steps[0], merge_steps[0], model_steps[0]

    if any(key not in _RESERVED_STEP_KEYS or is_param_generator_spec(value) for key, value in model_step.items() if key != "model"):
        return None
    order = [step for step in pipeline if step is branch_step or step is merge_step or step is model_step]
    if order != [branch_step, merge_step, model_step]:
        return None
    for step in pipeline:
        if step is branch_step or step is merge_step or step is model_step or _is_split_step(step):
            continue
        return None

    criterion = branch_step["branch"]
    if set(criterion) - _HANDLED_BY_SOURCE_KEYS:
        return None
    body = criterion.get("steps")
    if not isinstance(body, list):
        return None
    model_positions = [index for index, substep in enumerate(body) if isinstance(substep, dict) and "model" in substep]
    if len(model_positions) != 1 or model_positions[0] != len(body) - 1:
        return None
    if any(isinstance(substep, dict) for substep in body[: model_positions[0]]):
        return None
    branch_model_step = body[model_positions[0]]
    if any(key not in _RESERVED_STEP_KEYS or is_param_generator_spec(value) for key, value in branch_model_step.items() if key != "model"):
        return None

    meta_learner = _meta_learner(model_step)
    if meta_learner is None:
        return None
    return body, meta_learner
