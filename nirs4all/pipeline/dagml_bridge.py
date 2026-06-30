"""nirs4all pipeline → dag-ml DSL frontend (migration spike, compile-only).

First slice of the #1 migration gap (``dag-ml/docs/migration-nirs4all/``): dag-ml's
compiler ingests serialized nirs4all-style *compat* DSL JSON, but nirs4all has no
serializer from a *live* pipeline (operator instances) to that DSL. This module
provides it for the linear ``transform → y_processing → splitter → model`` shape —
the parity gate-zero slice (``baseline_vertical_slice``).

Scope (deliberately narrow for the spike):

- **Supported:** bare transformer/splitter instances → ``{"class", "params"}``
  (dag-ml infers transform-vs-splitter from the class; splitters become campaign
  controller calls, not graph nodes); ``{"y_processing": op}`` → ``y_transform``;
  ``{"model": op}`` → ``model``.
- **Not yet:** branch / merge / tag / exclude / generators / augmentation /
  multi-source — these raise ``NotImplementedError`` naming the offending step, to
  be filled in against ``dag-ml/docs/design/DSL_NIRS4ALL_PARITY.md``.

dag-ml is a CORE dependency since the ADR-17 cutover (the default engine); the import is
still guarded so this module imports cleanly even if a broken wheel lacks the native backend.
This is **compile-only** (DSL lowering); execution via host controllers is a
later migration phase.
"""

from __future__ import annotations

import json
from typing import Any

from nirs4all import __version__ as _NIRS4ALL_VERSION
from nirs4all.pipeline.dagml.errors import DagMlUnsupported

# Stacking meta-model wiring (backlog #10). The meta-node is a `model`-kind node bound to a dedicated
# controller (via `metadata.controller_id`) that declares `consumes_oof_predictions`. The ref is the
# operator-selector token that keeps the meta-model manifest out of the generic model-kind catch-all.
_META_MODEL_CONTROLLER_ID = "controller:nirs4all.meta_model"
_META_MODEL_REF = "nirs4all.meta_model"

# Every nirs4all generation keyword (mirrors config._generator.keywords.GENERATION_KEYWORDS). Used
# to detect a generator-shaped model sibling that this bridge does NOT lower natively, so it can fail
# loud instead of silently demoting it to a plain param.
_GENERATION_KEYWORDS = frozenset({"_or_", "_range_", "_log_range_", "_grid_", "_zip_", "_chain_", "_sample_", "_cartesian_"})

# Inert generator-annotation keys (`_tags_` / `_metadata_`) that do NOT change the variant set — they
# annotate a generator node only (see config._generator). A flat-single operator `_or_` carrying just
# these still lowers natively (the variant set is the bare `_or_`); any OTHER sibling key forces the
# Python expand path (see `_lower_operator_generator`).
_INERT_GENERATOR_ANNOTATION_KEYS = frozenset({"_tags_", "_metadata_", "name"})

# Step keywords recognised by nirs4all but not yet lowered by this spike.
_UNSUPPORTED_STEP_KEYS = frozenset({
    "branch",
    "merge",
    "tag",
    "exclude",
    "sample_augmentation",
    "rep_to_sources",
    "rep_to_pp",
    "finetune_params",
    "train_params",
})

# feature_augmentation action modes (mirrors FeatureAugmentationController.VALID_ACTIONS). The default
# is "add" (FeatureAugmentationController.execute), which for a single base "raw" processing keeps the
# raw layer beside the new ones — same column set as "extend"; "replace" drops the raw layer.
_FEATURE_AUGMENTATION_ACTIONS = frozenset({"extend", "add", "replace"})

# `FeatureConcat` is the single-source replace-mode lowering of `concat_transform` (backlog #27):
# nirs4all hstacks several sub-transformers' outputs into one wider 2D feature matrix, which is exactly
# what this transformer does, so the model node runs it as an ordinary X-chain transform node.
_FEATURE_CONCAT_CLASS = "nirs4all.operators.transforms.concat.FeatureConcat"

# Keys on a model step that are NOT a swept hyperparameter (mirrors run_backend._RESERVED_STEP_KEYS,
# itself StepParser.RESERVED_KEYWORDS). Any other sibling is a model hyperparameter — a plain value
# goes to ``params``; a natively-lowerable param-level generator dict (``_range_``/``_log_range_``)
# lowers to a native dag-ml ``generators`` entry so the compiler expands variants and dag-ml selects
# natively (``_grid_``/dict-form/modifier sweeps stay on the Python expand path).
_RESERVED_MODEL_KEYS = frozenset({
    "model",
    "params",
    "metadata",
    "steps",
    "name",
    "finetune_params",
    "train_params",
    "refit_params",
    "fit_on_all",
    "force_layout",
    "na_policy",
    "fill_value",
    "y_processing",
})


def _qualname(obj: Any) -> str:
    """Fully-qualified ``module.QualName`` of an instance or class."""
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _json_safe_params(obj: Any) -> dict[str, Any]:
    """sklearn-style ``get_params()`` coerced to JSON-native values.

    Compile lowers the DSL structurally and never instantiates the operator, so
    a lossy ``repr`` fallback for exotic values is acceptable here.
    """
    if isinstance(obj, type) or not hasattr(obj, "get_params"):
        return {}
    params: dict[str, Any] = json.loads(json.dumps(obj.get_params(), default=repr))
    return params


def _param_generator(param: str, spec: Any) -> dict[str, Any]:
    """Lower one nirs4all param-level generator sibling to a dag-ml ``generators`` entry.

    Only the ``_range_`` and ``_log_range_`` list forms are native (see
    :func:`is_param_generator_spec`); the dict-form ranges and modifier-bearing sweeps stay on the
    Python path, so this never receives them (a step-level ``_grid_`` is lowered by
    :func:`_grid_param_generator`, not here). Field names verified against
    ``examples/pipeline_dsl_compact_generation.json`` and ``dsl.rs``:

    * ``{"_range_": [a, b, s]}`` → ``{"kind": "range", "param", "start": a, "stop": b, "step": s}``
    * ``{"_log_range_": [a, b, n]}`` → ``{"kind": "log_range", "param", "start": a, "stop": b, "count": n}``

    dag-ml's ``range`` is end-inclusive (``inclusive`` defaults to true), matching nirs4all
    ``_range_`` (``range(a, b + 1, s)``). dag-ml's ``log_range`` generates ``count`` base-10
    geometric points end-inclusive (``base.powf(start_log + (stop_log - start_log) * i / (count-1))``),
    matching nirs4all's ``_log_range_`` ``[from, to, num]`` expansion exactly.
    """
    if "_log_range_" in spec:
        start, stop, count = spec["_log_range_"]
        return {"kind": "log_range", "param": param, "start": start, "stop": stop, "count": int(count)}
    start, stop, step = spec["_range_"]
    return {"kind": "range", "param": param, "start": start, "stop": stop, "step": step}


def is_grid_param_generator_spec(grid_map: Any) -> bool:
    """True ONLY for a ``_grid_`` over MODEL params the native dag-ml ``Grid`` generator can lower at parity.

    A step-level ``{"_grid_": {param: [values], …}, "model": M}`` lowers to one native dag-ml ``Grid``
    param generator (:func:`_grid_param_generator`). dag-ml's ``Grid`` is a flat Cartesian product of
    each param's PLAIN value list — its ``PipelineDslGeneratorValue`` is an *untagged literal*, so it
    does NOT recursively expand a nested generator value. nirs4all's ``GridStrategy`` DOES expand a
    nested-generator value (``n_components: {"_range_": …}``) BEFORE the product, so only a grid whose
    every value is a list (or bare scalar) of PLAIN, finite, JSON-native scalars — with NO nested dict
    (a nested generator / object value) and NO nested-generator list element — is representable natively
    and produces the identical variant set. ALL of:

    * a non-empty dict of ``str → values`` (the bare ``_grid_`` param-map, NO ``count``/``_seed_``
      modifier — those reach here only when the caller already split the keyword off, and a modifier
      changes the variant set vs. ``expand_spec``, so it is excluded by the caller); AND
    * every value is a list of plain scalars (or a single plain scalar), each scalar finite and
      JSON-native (``bool``/``int``/``float``/``str``/``None``) — NO dict, NO list-of-dicts (a nested
      generator the native ``Grid`` cannot expand), NO numpy / non-finite / non-JSON value; AND
    * the param insertion order is already alphabetical (``list(grid_map) == sorted(grid_map)``):
      dag-ml's ``Grid`` nests params in ``BTreeMap`` (sorted) order while nirs4all nests in INSERTION
      order, so a non-alphabetical grid would produce a different variant ENUMERATION order than
      ``expand_spec`` — keep it on the Python path so the per-variant config map stays aligned.

    Any other shape (a nested-generator value, a modifier-bearing or non-alphabetical grid, a non-list
    non-scalar value, a non-JSON / non-finite scalar) → ``False`` (stays on the correct Python
    ``expand_spec`` path, still dag-ml-native via that route). When in doubt, this returns ``False``.
    """
    import math

    if not isinstance(grid_map, dict) or not grid_map:
        return False
    keys = list(grid_map)
    # Validate ALL keys are strings BEFORE any ordering comparison — `sorted()` on mixed str/non-str keys
    # would raise TypeError; the gate must fail CLOSED (demote to Python-expand) on a non-str key, never crash.
    if not all(isinstance(key, str) for key in keys):
        return False
    if keys != sorted(keys):
        return False
    for value in grid_map.values():
        values = value if isinstance(value, list) else [value]
        if not values:
            return False
        for item in values:
            if isinstance(item, bool) or item is None or isinstance(item, (str, int)):
                continue
            if isinstance(item, float) and math.isfinite(item):
                continue
            return False
    return True


# Variant-set-changing modifier siblings a `_grid_` model step can carry (a `count` subsample or a
# `_seed_`). When present beside `_grid_`, the native dag-ml `Grid` does NOT reproduce nirs4all's seeded
# subsample, so the step stays on the Python expand path. `_tags_`/`_metadata_` are inert (they do not
# change the variant set), so they do NOT block the native path.
_GRID_VARIANT_MODIFIER_KEYS = frozenset({"count", "_seed_"})


def step_has_native_grid(step: Any) -> bool:
    """True iff a model ``step`` carries a step-level ``_grid_`` the native dag-ml ``Grid`` can lower at parity.

    A clean native grid is ``{"_grid_": <native param-map>, "model": M[, plain params]}``: a step-level
    ``_grid_`` whose param-map passes :func:`is_grid_param_generator_spec` AND NO variant-set-changing
    grid modifier sibling (``count`` / ``_seed_``) — those would subsample/reseed the grid, which the
    native ``Grid`` does not reproduce, so a modifier-bearing grid stays on the Python expand path. The
    single recognition point shared by the bridge lowering (:func:`_step_to_dsl`), the generation-kind
    classifier, and the plain-param application, so the three never disagree on which grids go native.
    """
    if not isinstance(step, dict) or "model" not in step or "_grid_" not in step:
        return False
    if _GRID_VARIANT_MODIFIER_KEYS & set(step):
        return False
    return is_grid_param_generator_spec(step["_grid_"])


def _grid_param_generator(grid_map: dict[str, Any]) -> dict[str, Any]:
    """Lower a native-representable step-level ``_grid_`` param-map to a dag-ml ``Grid`` generator entry.

    ``{"n_components": [5, 8, 11, 14], "scale": [True, False, True]}`` →
    ``{"kind": "grid", "params": {"n_components": [5, 8, 11, 14], "scale": [True, False, True]}}``. The
    caller (:func:`is_grid_param_generator_spec`) has already proven every value is a list (or bare
    scalar) of plain finite JSON scalars, so each value is normalized to a list and emitted verbatim as
    the ``Grid`` param's value vector (dag-ml's ``PipelineDslGeneratorValue`` reads a bare scalar as a
    literal value). The ``Grid`` is the flat Cartesian product of these vectors in ``BTreeMap`` (sorted)
    param order — which equals nirs4all's insertion order because the predicate requires the param keys
    already be alphabetical, so the variant set AND enumeration order match ``expand_spec``.
    """
    params = {key: (value if isinstance(value, list) else [value]) for key, value in grid_map.items()}
    return {"kind": "grid", "params": params}


def is_param_generator_spec(spec: Any) -> bool:
    """True ONLY for the exact ``_range_`` / ``_log_range_`` list forms lowered natively at proven parity.

    Conservative by design: a single key (``_range_`` or ``_log_range_``) whose value is a list of
    exactly three numbers — ``{"_range_": [a, b, s]}`` or ``{"_log_range_": [a, b, n]}``. dag-ml's
    native log_range now round-trips through ``build_execution_plan`` (the float-label fingerprint
    drift is fixed by ``canonical_generator_number`` at value generation, dag-ml ``2a77a7f``), and its
    base-10 geometric expansion matches nirs4all's ``_log_range_`` exactly, so the list form is native.
    Everything else still falls back to the correct Python ``expand_spec`` path:

    * ``_grid_`` — value-level lowering is not proven equivalent to step-level grid expansion;
    * the dict ``{"from"/"to"/...}`` form, a wrong-length list, or any modifier key (``count``/``_seed_``)
      — would change the variant set versus ``expand_spec``.

    Other keys in the dict (e.g. a ``model`` sibling) are handled by the caller, not here.
    """
    if not isinstance(spec, dict) or len(spec) != 1:
        return False
    key, value = next(iter(spec.items()))
    if key not in ("_range_", "_log_range_"):
        return False
    return isinstance(value, list) and len(value) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)


def _is_bare_operator_choice(choice: Any) -> bool:
    """True when an ``_or_`` choice is a SINGLE bare operator (a transform instance/class), not a list/dict.

    A flat operator generator the native path lowers is an ``_or_`` whose every choice is one bare
    operator (``SNV`` / ``MSC()``). A multi-step LIST choice (``[SNV, FirstDerivative]``), a ``{"model":
    …}`` (multi-model) or any other dict (a nested generator / param spec) is NOT flat-bare — those stay
    on the Python ``expand_spec`` path (:func:`_lower_operator_generator` raises for them).
    """
    return not isinstance(choice, (list, dict))


def _lower_operator_generator(step: dict[str, Any]) -> dict[str, Any]:
    """Lower a FLAT-SINGLE operator ``_or_`` of bare operators to a compat ``{"_or_": [<op>, …]}`` step.

    The dag-ml compat importer lowers ``{"_or_": [op0, op1, …]}`` to a ``PipelineDslStep::Generator``
    (one branch per choice, each branch a single transform step), which ``compile_operator_variant_models``
    expands into the operator-variant models the in-process binding scores by CV-OOF + refits the winner
    of (native operator-SELECT). This emits exactly that compat shape — each ``_or_`` value lowered to the
    bare-operator ``{"class": FQN, "params": {…}}`` dict :func:`_step_to_dsl` already produces.

    ONLY the flat bare-operator ``_or_`` is accepted (so the variant set is the bare ``_or_`` and each
    choice lowers to ONE transform step). Anything richer raises :class:`NotImplementedError` so
    ``run_via_dagml`` falls back to the Python ``expand_spec`` path (which stays on the dag-ml engine):

    * a non-``_or_`` keyword (``_cartesian_`` / ``_grid_`` / ``_chain_`` / ``_zip_`` / ``_sample_``);
    * a ``pick`` / ``arrange`` / ``then_pick`` / ``then_arrange`` / ``count`` modifier, a ``_weights_`` /
      ``_seed_`` sampler, or a ``_mutex_`` / ``_requires_`` / ``_exclude_`` constraint;
    * a multi-step LIST choice or a ``{"model": …}`` (multi-model) choice — only single bare operators.

    Inert annotation keys (``_tags_`` / ``_metadata_`` / ``name``) are no-ops that do not change the
    variant set, so they are tolerated and dropped (the lowered ``_or_`` carries the choices only).
    """
    keywords = _GENERATION_KEYWORDS & set(step)
    if keywords != {"_or_"}:
        raise NotImplementedError(
            f"dag-ml bridge lowers only a flat single `_or_` operator generator natively; "
            f"{sorted(keywords)} stays on the Python expand path"
        )
    extra = set(step) - {"_or_"} - _INERT_GENERATOR_ANNOTATION_KEYS
    if extra:
        raise NotImplementedError(
            f"dag-ml bridge does not lower `_or_` with modifier/constraint key(s) {sorted(extra)} natively; "
            f"the Python expand path owns pick/arrange/count/_mutex_/_requires_/_exclude_/_weights_/_seed_"
        )
    choices = step["_or_"]
    if not isinstance(choices, list) or not choices or not all(_is_bare_operator_choice(choice) for choice in choices):
        raise NotImplementedError(
            "dag-ml bridge lowers `_or_` natively only when every choice is a single bare operator "
            "(not a multi-step list, a {'model': …} multi-model, or a nested generator)"
        )
    return {"_or_": [_operator_choice_dsl(choice) for choice in choices]}


def _operator_choice_dsl(choice: Any) -> dict[str, Any]:
    """One ``_or_`` operator choice lowered to the bare-operator ``{"class": FQN, "params": {…}}`` dict.

    A ``None`` choice (the "no preprocessing" idiom) cannot lower to a transform node, so it forces the
    Python expand path (where it is dropped); every other choice reuses the bare-operator lowering.
    """
    if choice is None:
        raise NotImplementedError("dag-ml bridge does not lower a `None` (no-op) `_or_` choice natively")
    return {"class": _qualname(choice), "params": _json_safe_params(choice)}


def constrained_operator_survivor_sequences(generator_node: dict[str, Any]) -> list[list[Any]]:
    """The PRUNED survivor SEQUENCES of a constrained operator generator (ADR-17 1a + 1b-cartesian).

    Expands the constrained ``_or_``-pick / ``_cartesian_`` node through nirs4all's own ``expand_spec`` —
    the SAME constraint source of truth both engines use (and the engine-agnostic survivor lock in
    ``test_generators_conformance_extra``) — and normalizes each variant to a LIST of operator instances
    (a single-op variant becomes a one-element list). dag-ml's native operator-SELECT later compiles ONE
    survivor branch per sequence and, applying pick + ``_mutex_``/``_requires_``/``_exclude_`` during its
    own sequence-build (``prune_sequences_by_constraints``), produces the byte-identical pruned set, so the
    per-variant ``variant_label`` fingerprints agree by content (see
    :func:`operator_choice_variant_label`). The survivor ORDER follows ``expand_spec`` (legacy
    ``OrStrategy``/``CartesianStrategy`` input order), which matches dag-ml's stable enumerate-retain order
    — so the ``{variant_label → config_name}`` map (built in expand order) and dag-ml's reports align by
    content, not position.
    """
    from nirs4all.pipeline.config.generator import expand_spec

    sequences: list[list[Any]] = []
    for variant in expand_spec(generator_node):
        sequences.append(list(variant) if isinstance(variant, list) else [variant])
    return sequences


def _operator_option_step(operator: Any, option_id: str) -> dict[str, Any]:
    """One operator option lowered to a canonical single-transform branch (``{"id", "steps":[transform]}``).

    Each ``_or_`` choice (or ``_cartesian_`` stage option) becomes ONE generator branch carrying ONE
    transform step. The branch ``id`` is the operator-content identity (``option_id``) the constraint refs
    resolve against — dag-ml's prune keys its member set + refs on ``sanitize_generation_label(branch.id)``,
    so equal-identity operators share a branch id and a ``_mutex_``/``_requires_``/``_exclude_`` ref lands on
    the SAME id. The transform itself carries the real ``{"class": FQN, "params": …}`` (a stray non-JSON
    param fails loud via :func:`_json_safe_params` → Python expand), so the per-survivor ``variant_label``
    (computed by dag-ml over operator CONTENT, NOT the branch id) is byte-identical to the host map.
    """
    return {"id": option_id, "steps": [{"kind": "transform", "id": f"t:{option_id}", "operator": {"class": _qualname(operator)}, "params": _json_safe_params(operator)}]}


def _resolve_constraint_ref(operator: Any, identity: dict[Any, str]) -> str:
    """Resolve a constraint-ref operator to its option branch id (nirs4all ``_normalize_item`` identity).

    A ref that names no option is a malformed constraint — nirs4all's own ``validate_constraints`` rejects
    it, so demote to the Python expand path rather than emit a dangling ref dag-ml would also reject.
    """
    branch_id = identity.get(_normalize_item_key(operator))
    if branch_id is None:
        raise NotImplementedError(f"dag-ml bridge: constraint references operator {operator!r} that is not one of the generator options")
    return branch_id


def _lower_generator_constraints(node: dict[str, Any], identity: dict[Any, str]) -> dict[str, Any]:
    """Lower the nirs4all ``_mutex_``/``_requires_``/``_exclude_`` refs to dag-ml ``constraints`` over branch ids.

    Keys are passed through unchanged (dag-ml's ``PipelineDslGeneratorConstraints`` reads ``_mutex_`` /
    ``_requires_`` / ``_exclude_`` via serde aliases); only the operator REFS are rewritten to the option
    branch ids assigned in :func:`_native_constrained_generator`, so the dag-ml prune matches the same
    survivor set ``expand_spec`` produces.
    """
    constraints: dict[str, Any] = {}
    # `_mutex_` is any-cardinality `issubset` natively (dag-ml `Vec<Vec<String>>`), so each group passes
    # through unchanged.
    if node.get("_mutex_"):
        constraints["_mutex_"] = [[_resolve_constraint_ref(op, identity) for op in group] for group in node["_mutex_"]]
    # A multi-`_requires_` `[A, B, C, …]` is legacy "first requires ALL subsequent" (constraints.py
    # `_satisfies_requires`), but dag-ml's native `requires` is a `[trigger, required]` PAIR only — so SPLIT it
    # into one pair per subsequent operator (`[A, B], [A, C], …`), which is the EXACT same logical constraint
    # (each pair independently demands the required operator when the trigger is present). A plain 2-`_requires_`
    # yields a single pair, byte-identical to before (MUST-FIX 3).
    if node.get("_requires_"):
        requires_pairs = []
        for group in node["_requires_"]:
            trigger = _resolve_constraint_ref(group[0], identity)
            for required in group[1:]:
                requires_pairs.append([trigger, _resolve_constraint_ref(required, identity)])
        constraints["_requires_"] = requires_pairs
    # `_exclude_` is a native `[String; 2]` PAIR with SUBSET semantics; the predicate
    # (`_constrained_exclude_diverges`) demotes upstream unless every group is a 2-operator pair AND the
    # survivor cardinality is exactly 2 (so native subset == legacy exact-combo), so every group reaching here
    # is a 2-operator pair the dag-ml prune matches byte-identically.
    if node.get("_exclude_"):
        constraints["_exclude_"] = [[_resolve_constraint_ref(op, identity) for op in pair] for pair in node["_exclude_"]]
    return constraints


# Selection modifiers a constrained `_or_` / `_cartesian_` carries verbatim onto the canonical generator (the
# SAME keys dag-ml's `PipelineDslGeneratorStep` reads). nirs4all `pick`/`arrange` use the identical `int` /
# `[lo, hi]` selection-size form dag-ml's `PipelineDslSelectionSpec` accepts, so they pass through. `count`
# is DELIBERATELY excluded: legacy SAMPLES post-prune survivors while dag-ml's `count` TRUNCATES (a different
# set), so a `count`-bearing constrained generator is demoted to Python-expand by the predicate and never
# reaches here (MUST-FIX 1).
_SELECTION_PASSTHROUGH_KEYS = ("pick", "arrange", "then_pick", "then_arrange")


def _native_constrained_generator(generator_node: dict[str, Any], tail: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the ONE native canonical Generator step for a constrained ``_or_``-pick / ``_cartesian_`` (ADR-17 item 5 B).

    Emits a single ``{"kind": "generator", …}`` step that carries the operator-content options as branches
    (``_or_``) or stages (``_cartesian_``), the ``pick``/``arrange``/``then_*`` selection modifiers verbatim
    (``count``/``_seed_``/``_weights_`` are sampling modifiers DEMOTED upstream by the predicate, so they never
    reach here — MUST-FIX 1), the lowered operator-content ``constraints``, and the downstream model (+ any
    ``y_processing``) as the ``tail`` — and lets dag-ml prune. dag-ml's ``expand_*_generator_sequences`` applies pick + the constraint
    prune to the SAME survivor set ``expand_spec`` produces (proven byte-identical in ADR-17 1a), appends the
    ``tail`` to each survivor so every choice terminates in the model, and fingerprints each over
    ``[<survivor ops>, <tail>]`` — the EXACT sub-sequence :func:`operator_sequence_variant_label` recomputes
    host-side. This REPLACES the old ``expand_spec`` survivor pre-expansion: the constraints + pick now reach
    dag-ml's native prune (the CATCH-22 fix), instead of the host pre-pruning into a constraint-free ``_or_``.
    """
    generator_dsl: dict[str, Any] = {"kind": "generator", "id": "generator:preproc"}
    # One branch PER option (1:1 with the nirs4all option list, so dag-ml enumerates the same options in the
    # same order), each with a POSITIONAL branch id (unique even if two options were equal). The constraint
    # refs resolve to the FIRST option matching their `_normalize_item` identity — exactly how nirs4all's set
    # membership matches a ref against the option list — so a `_mutex_`/`_requires_`/`_exclude_` ref lands on
    # the right branch id and dag-ml prunes the byte-identical survivor set.
    identity: dict[Any, str] = {}
    if "_cartesian_" in generator_node:
        stages = []
        for stage_index, stage in enumerate(generator_node["_cartesian_"]):
            if not (isinstance(stage, dict) and set(stage) == {"_or_"}):
                raise NotImplementedError("dag-ml bridge: a constrained `_cartesian_` stage must be a bare `{'_or_': [...]}` of operator options")
            branches = []
            for option_index, operator in enumerate(stage["_or_"]):
                branch_id = f"s{stage_index}op{option_index}"
                identity.setdefault(_normalize_item_key(operator), branch_id)
                branches.append(_operator_option_step(operator, branch_id))
            stages.append({"id": f"stage{stage_index}", "branches": branches})
        generator_dsl["mode"] = "cartesian"
        generator_dsl["stages"] = stages
    else:
        branches = []
        for option_index, operator in enumerate(generator_node["_or_"]):
            branch_id = f"op{option_index}"
            identity.setdefault(_normalize_item_key(operator), branch_id)
            branches.append(_operator_option_step(operator, branch_id))
        generator_dsl["mode"] = "or"
        generator_dsl["branches"] = branches

    constraints = _lower_generator_constraints(generator_node, identity)
    if constraints:
        generator_dsl["constraints"] = constraints
    for key in _SELECTION_PASSTHROUGH_KEYS:
        if key in generator_node:
            generator_dsl[key] = generator_node[key]
    if tail:
        generator_dsl["tail"] = tail
    return generator_dsl


def _normalize_item_key(operator: Any) -> Any:
    """nirs4all ``_normalize_item`` identity of an operator (the constraint-match key)."""
    from nirs4all.pipeline.config._generator.constraints import _normalize_item

    return _normalize_item(operator)


def lower_constrained_operator_pipeline(steps: list[Any], dsl_id: str = "nirs4all-pipeline") -> dict[str, Any]:
    """Lower a CONSTRAINED operator-generator pipeline to ONE NATIVE Generator DSL step (ADR-17 item 5 slice B).

    ``steps`` is the splitter-free pipeline ``[<constrained generator>, …downstream…]`` (the predicate
    :func:`~nirs4all.pipeline.dagml.detect._is_constrained_operator_generator` admits exactly one constrained
    ``_or_``-pick / ``_cartesian_`` generator plus one concrete downstream model, optionally a
    ``y_processing``). The generator lowers to ONE canonical ``{"kind": "generator", …}`` step that carries
    its operator OPTIONS as branches/stages, its ``pick``/``arrange``/``then_*``/``count`` modifiers, its
    lowered operator-content ``constraints``, and the downstream model (+ any ``y_processing``) as the
    ``tail`` — and dag-ml PRUNES (the CATCH-22 fix). dag-ml applies pick + the
    ``_mutex_``/``_requires_``/``_exclude_`` prune to the SAME survivor set ``expand_spec`` produced (proven
    byte-identical in ADR-17 1a), appends the ``tail`` so every survivor terminates in the model, and stamps
    each per-variant report with the multi-op ``variant_label`` fingerprint of ``[<survivor ops>, <tail>]`` —
    the EXACT sub-sequence :func:`operator_sequence_variant_label` recomputes host-side.

    This REPLACES the previous survivor PRE-EXPANSION (``constrained_operator_survivor_sequences`` →
    ``expand_spec`` → one ``_or_`` branch per survivor): with the native ``tail`` field on
    ``PipelineDslGeneratorStep`` and ``compile_operator_variant_models`` reaching the prune for a
    model-terminated generator, the host no longer pre-prunes — it hands the constraints + pick to dag-ml.
    The ``{variant_label → config_name}`` map (built in :mod:`...dagml.result` from ``expand_spec``) still
    aligns because dag-ml prunes the SAME survivor set, in the SAME legacy order. A non-bare / ``None`` /
    non-routable choice is rejected upstream by the predicate; an option still lowers each transform via the
    strict bare-operator path so a stray non-JSON param fails loud (→ Python expand), and a constraint ref
    that names no option (a malformed constraint) likewise demotes.
    """
    generator_index = next((index for index, step in enumerate(steps) if isinstance(step, dict) and _GENERATION_KEYWORDS & set(step)), None)
    if generator_index is None:
        raise NotImplementedError("dag-ml bridge: no operator generator step to lower as a constrained native generator")
    generator_node = steps[generator_index]
    prefix = steps[:generator_index]
    downstream = steps[generator_index + 1 :]
    if not any(isinstance(step, dict) and "model" in step for step in downstream):
        raise NotImplementedError("dag-ml bridge: a constrained operator generator must be followed by a concrete model step")

    # Any concrete steps BEFORE the generator (a preprocessing prefix, e.g. `[FirstDerivative(), _or_pick, model]`)
    # lower to SHARED upstream nodes placed verbatim before the Generator step — applied ONCE on fold-train and
    # flowing into every survivor branch — exactly as the FLAT-SINGLE path lowers a prefix (`pipeline_to_dsl`
    # emits the prefix transform as one upstream node, NOT duplicated per branch). The prefix is OUTSIDE the
    # generator, so it is NOT part of any survivor's `variant_label` (the host map fingerprints only
    # `[<survivor transforms>, downstream]`, matching dag-ml's `choice.steps`, which likewise excludes upstream
    # nodes) — so byte-identity with the dag-ml report label holds for a prefixed pipeline too. Each prefix
    # step is lowered to its CANONICAL tagged form (every step carries `kind`).
    prefix_steps = [_canonical_branch_step(_step_to_dsl(step), f"prefix{prefix_index}") for prefix_index, step in enumerate(prefix)]

    # The downstream tail (model, any y_processing) lowered ONCE and carried in the generator `tail` — dag-ml
    # appends it to each pruned survivor so every choice terminates in the model. Reuse _step_to_dsl (the same
    # operator/param split the flat-single path uses); a generator-bearing downstream is impossible here (the
    # predicate forbids a second generator).
    downstream_dsl = [_step_to_dsl(step) for step in downstream]
    tail = [_canonical_branch_step(dsl_step, f"down{down_index}") for down_index, dsl_step in enumerate(downstream_dsl)]

    generator_dsl = _native_constrained_generator(generator_node, tail)
    return {"id": dsl_id, "pipeline": [*prefix_steps, generator_dsl]}


def _canonical_branch_step(dsl_step: dict[str, Any], step_id: str) -> dict[str, Any]:
    """Convert a compat ``_step_to_dsl`` step (``{"model"|"y_processing"|"class": …}``) to a canonical tagged step.

    A canonical prefix step / generator ``tail`` step carries canonical tagged form
    (``{"kind", "id", "operator", "params"}``), so a ``{"model": FQN, "params": …}`` becomes
    ``{"kind": "model", "operator": FQN, "params": …}``, a ``{"y_processing": {"class": …}}`` becomes a
    ``y_transform`` whose object operator carries its params, and a bare ``{"class": FQN, "params": …}``
    transform becomes a canonical ``transform`` — mirroring the operator/params split dag-ml's compat lowerer
    itself applies, so the embedded downstream is byte-shape identical to what a separate downstream step
    would lower to.
    """
    if "model" in dsl_step:
        params = dict(dsl_step.get("params", {}))
        if "generators" in dsl_step:
            raise NotImplementedError("dag-ml bridge does not lower a param-generator model inside a constrained generator tail")
        return {"kind": "model", "id": step_id, "operator": dsl_step["model"], "params": params}
    if "y_processing" in dsl_step:
        return {"kind": "y_transform", "id": step_id, "operator": dsl_step["y_processing"], "params": {}}
    return {"kind": "transform", "id": step_id, "operator": {"class": dsl_step["class"]}, "params": dict(dsl_step.get("params", {}))}


def operator_choice_variant_label(choice: Any, downstream_steps: list[Any]) -> str:
    """The AUTHORITATIVE ``variant_label`` (hex sha256) for ONE bare-operator ``_or_`` choice + its tail.

    Maps a per-variant dag-ml report back to its operator-choice config (#23 Phase 7). The label is a
    cross-language CONTENT fingerprint dag-ml stamps on every operator-SELECT report; the host MUST call
    the dag-ml PyO3 helper (``dag_ml.canonical_operator_variant_label``) — NOT a pure-Python ``json.dumps``
    + sha256 — because Python's float formatting diverges from Rust's (``1e-05`` / ``1e-7`` / 1-ULP shortest
    decimals, all common NIRS params), which would SILENTLY mis-key the map. Sharing the one Rust codepath
    makes the host label byte-identical to the report label by construction.

    The fingerprint is computed over the FULL LOWERED choice sub-sequence — dag-ml's
    ``lower_operator_variant_model`` activates the chosen ``_or_`` branch PLUS every downstream step the
    branch flows into (the concrete model, any ``y_processing``), so ``choice.steps`` is the choice's
    transform step FOLLOWED by the downstream steps. The host therefore builds ``steps_json`` =
    ``[<choice transform>] + [<lowered downstream steps>]`` in the ``PipelineDslStep`` tagged shape
    (``{"kind", "id", "operator", "params"}``) — lowering each downstream step from its RAW form via
    :func:`_label_step_from_raw` (NOT through :func:`_step_to_dsl`'s ``default=repr`` params, which would
    stringify a non-JSON downstream param BEFORE strict checking). The ``id`` is ignored by the
    canonicalization (it reads only ``kind`` / operator class / params), but the OPERATOR SHAPE matters: a
    bare-operator transform choice keeps the object operator ``{"class": FQN}``, while a model keeps the
    BARE-STRING operator FQN — exactly what dag-ml's lowering produces. ``downstream_steps`` is the lowered
    pipeline tail after the generator (no splitter, no ``None`` — already split off by the caller).

    The ENTIRE label payload is STRICTLY sanitized (chosen branch + downstream model + y_processing): every
    step's operator + params is recursively coerced (numpy ``.item()`` / ``.tolist()``) and any NaN / Inf /
    non-str-key / non-JSON value is REJECTED via :func:`_strict_json_safe` (NOT ``default=repr``, which
    would emit ``"np.int64(5)"`` strings that break byte-identity, and NOT ``allow_nan=True``, which would
    pass a non-finite the PyO3 helper then rejects mid-run). ANY failure to construct or fingerprint the
    label — a non-JSON / non-finite param OR the PyO3 helper raising — is converted to
    :class:`DagMlUnsupported` so the inner Python-expand fallback fires instead of crashing.
    """
    return operator_sequence_variant_label([choice], downstream_steps)


def operator_sequence_variant_label(operators: list[Any], downstream_steps: list[Any]) -> str:
    """The AUTHORITATIVE ``variant_label`` (hex sha256) for a MULTI-OPERATOR survivor sequence + its tail.

    The constrained-generator generalization of :func:`operator_choice_variant_label`: a constrained
    ``_or_``-pick / ``_cartesian_`` survivor is a SEQUENCE of operators (``[SNV, Detrend]``), not a single
    choice. dag-ml lowers each survivor branch to ``[<transform op0>, …, <transform opN>] + downstream`` and
    fingerprints THAT whole sub-sequence (``operator_variant_label(choice.steps)``), so the host builds the
    SAME ``steps_json`` = ``[<each transform>, <lowered downstream>]`` and fingerprints it via the SAME Rust
    canonicalization (``dag_ml.canonical_operator_variant_label``) — byte-identical to the report label by
    construction. Each operator's params are STRICTLY sanitized (:func:`_canonical_label_params`,
    NaN/Inf/non-JSON/non-str-key REJECTED → :class:`DagMlUnsupported` → the inner Python-expand fallback);
    the downstream tail is lowered from its RAW form (:func:`_label_step_from_raw`). A single-element
    ``operators`` reproduces the flat-single label exactly.
    """
    import importlib
    import json

    transform_steps = [
        {"kind": "transform", "id": f"operator_choice{index}", "operator": {"class": _qualname(operator)}, "params": _canonical_label_params(operator)}
        for index, operator in enumerate(operators)
    ]
    steps = [*transform_steps, *(_label_step_from_raw(step, index) for index, step in enumerate(downstream_steps))]
    try:
        # allow_nan=False is belt-and-braces: _strict_json_safe already rejected non-finite numbers, so a NaN
        # never reaches here; this guarantees the host never silently emits `NaN` text the helper would reject.
        steps_json = json.dumps(steps, allow_nan=False)
        # Call the helper on the compiled `dag_ml._dag_ml` C extension directly (as the in-process runner does
        # for run_cv_refit_in_process): the facade `.pyi` stub does not declare the native re-exports, so the
        # dynamic import keeps mypy from flagging the attribute (no `type: ignore` to drift) while binding the
        # SAME Rust canonicalization codepath dag-ml stamps reports with.
        dag_ml_ext = importlib.import_module("dag_ml._dag_ml")
        return str(dag_ml_ext.canonical_operator_variant_label(steps_json))
    except DagMlUnsupported:
        raise
    except Exception as exc:  # noqa: BLE001 - any helper/serialization failure → lowering-unsupported fallback
        raise DagMlUnsupported(f"dag-ml bridge could not compute the operator variant_label ({type(exc).__name__}: {exc}); the Python expand path owns it") from exc


def _label_step_from_raw(step: Any, index: int) -> dict[str, Any]:
    """Lower one RAW downstream pipeline step to its ``PipelineDslStep`` label shape with STRICT params.

    The ``variant_label`` fingerprint is over ``PipelineDslStep`` objects (``{"kind", "operator",
    "params"}``). This mirrors :func:`_step_to_dsl`'s operator/params split EXACTLY — a MODEL keeps its
    bare-string operator FQN, a ``y_processing`` becomes a ``y_transform`` whose object operator carries
    its params INSIDE (``{"class": FQN, "params": {…}}``, step params ``{}``), and a bare transform keeps
    the object operator ``{"class": FQN}`` — BUT lowers params via :func:`_canonical_label_params` (STRICT:
    numpy coerced, NaN / Inf / non-JSON / non-str-key REJECTED → :class:`DagMlUnsupported`) instead of
    :func:`_step_to_dsl` → :func:`_json_safe_params` (``default=repr``). Lowering from the RAW step is what
    keeps a non-JSON downstream param from being silently stringified to a ``repr`` BEFORE strict checking.

    Only the step kinds a flat single ``_or_`` sub-sequence can carry — a concrete ``{"model": op[, plain
    siblings]}`` (no param-generator sibling; the predicate excludes a second generator) or a
    ``{"y_processing": op}`` — are handled here. A bare transform step (rare in this tail) lowers as an
    object-operator transform with its strict params.
    """
    if isinstance(step, dict) and "model" in step:
        op = step["model"]
        params = _canonical_label_params(op)
        # Plain sibling hyperparameters extend params exactly as _step_to_dsl does — strict-sanitized too,
        # INCLUDING the sibling KEY: a non-string step-level key (e.g. `{"model": op, 1: "x"}`) breaks JSON
        # validity / byte-identity, so reject it rather than insert a non-str key into the params dict.
        for key, value in step.items():
            if key in _RESERVED_MODEL_KEYS:
                continue
            if not isinstance(key, str):
                raise DagMlUnsupported(f"dag-ml bridge cannot fingerprint a non-string model sibling key `{key!r}` (type {type(key).__name__}) at `downstream{index}` for variant_label; the Python expand path owns it")
            params[key] = _strict_json_safe(value, f"downstream{index}.{key}")
        return {"kind": "model", "id": f"downstream{index}", "operator": _qualname(op), "params": params}
    if isinstance(step, dict) and "y_processing" in step:
        op = step["y_processing"]
        # dag-ml's y_processing lowering carries the WHOLE operator object (params INSIDE it) and leaves the
        # step params empty, so the canonical `class` is the compact JSON of `{"class": FQN, "params": {…}}`.
        return {"kind": "y_transform", "id": f"downstream{index}", "operator": {"class": _qualname(op), "params": _canonical_label_params(op)}, "params": {}}
    # A bare downstream transform (operator object + strict params), mirroring _step_to_dsl's bare form.
    return {"kind": "transform", "id": f"downstream{index}", "operator": {"class": _qualname(step)}, "params": _canonical_label_params(step)}


def _canonical_label_params(operator: Any) -> dict[str, Any]:
    """``get_params()`` STRICTLY sanitized to JSON-native values for the ``variant_label`` fingerprint.

    Unlike :func:`_json_safe_params` (which uses ``default=repr`` — acceptable for the structural compile
    that never instantiates the operator), the fingerprint must be byte-identical to dag-ml's, so every
    param value must be a true, FINITE JSON scalar/container. The sanitization (:func:`_strict_json_safe`)
    recursively unwraps numpy scalars (``.item()``) / arrays (``.tolist()``), rejects NaN/Inf, and rejects
    any remaining non-JSON value — raising :class:`DagMlUnsupported` (→ the inner Python-expand fallback)
    rather than emitting a ``repr`` string or letting ``allow_nan`` pass a non-finite that dag-ml then
    rejects mid-run. A bare CLASS carries no instance params (``{}``).
    """
    if isinstance(operator, type) or not hasattr(operator, "get_params"):
        return {}
    # Run the WHOLE params dict through _strict_json_safe (not value-by-value), so the TOP-LEVEL keys are
    # validated too: a non-string top-level key (e.g. a model sibling param `{1: "x"}`) breaks JSON validity
    # / byte-identity, so it must be REJECTED → DagMlUnsupported, never silently stringified.
    coerced: dict[str, Any] = _strict_json_safe(operator.get_params(), type(operator).__name__)
    return coerced


def _strict_json_safe(value: Any, label: str) -> Any:
    """Recursively coerce ``value`` to a FINITE JSON-native value, else raise :class:`DagMlUnsupported`.

    The cross-language ``variant_label`` is a byte-identity contract, so the host's label payload must
    contain ONLY values dag-ml can fingerprint: finite numbers, bools, strings, ``None``, and JSON
    arrays/objects of the same. This walks containers recursively, unwraps numpy scalars (``.item()``) /
    arrays (``.tolist()``), and REJECTS (→ ``DagMlUnsupported``, caught as lowering-unsupported):

    * a NaN / Inf float (``json.dumps(allow_nan=True)`` would silently pass it, then dag-ml's
      ``reject_non_finite`` raises a ``DagMlValidationError`` mid-run — not caught by the inner fallback);
    * any value with no JSON-native representation (a callable, a fitted object, an arbitrary object) —
      ``default=repr`` would emit a ``"<object …>"`` / ``"np.int64(5)"`` string that diverges from the
      report label.
    """
    import math

    import numpy as np

    if isinstance(value, np.generic):
        value = value.item()
    elif isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DagMlUnsupported(f"dag-ml bridge cannot fingerprint a non-finite param `{label}`={value!r} for variant_label; the Python expand path owns it")
        return value
    if isinstance(value, (list, tuple)):
        return [_strict_json_safe(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        coerced: dict[str, Any] = {}
        for key, item in value.items():
            # JSON objects allow ONLY string keys. Stringifying a non-str key (an int / tuple / object) would
            # silently diverge from dag-ml's canonical form (and ``json.dumps`` would itself coerce/clash), so
            # reject it as lowering-unsupported rather than fabricate a key.
            if not isinstance(key, str):
                raise DagMlUnsupported(f"dag-ml bridge cannot fingerprint a non-string dict key `{key!r}` (type {type(key).__name__}) at `{label}` for variant_label; the Python expand path owns it")
            coerced[key] = _strict_json_safe(item, f"{label}.{key}")
        return coerced
    raise DagMlUnsupported(f"dag-ml bridge cannot fingerprint a non-JSON param `{label}` of type {type(value).__name__} for variant_label; the Python expand path owns it")


def _concat_operation_spec(operation: Any) -> Any:
    """Serialize one ``concat_transform`` sub-operation to ``FeatureConcat``'s JSON ``operations`` form.

    A single transformer instance → ``{"class": FQN, "params": {...}}``; a chain (a list of
    instances) → a list of those dicts (applied sequentially). The 3D shapes that grow the
    processing axis fail loud in :func:`_lower_concat_transform`, so this only sees flat sub-ops.
    """
    if isinstance(operation, list):
        return [_concat_operation_spec(item) for item in operation]
    if isinstance(operation, dict):
        # A nested {"concat_transform": ...} (concat-of-concat) is a multi-block 3D construct, not a
        # flat single-matrix concat — defer to the data-plane rather than mis-lower it.
        raise NotImplementedError(
            "dag-ml bridge does not yet lower a nested `concat_transform` (concat-of-concat builds a "
            "multi-block feature representation); needs the multi-source/fusion data-plane (backlog #29/#31)"
        )
    if operation is None:
        # A pass-through "raw" channel keeps the un-transformed processing alongside the new ones —
        # that is a 3D processing-axis growth, not a flat single-matrix concat.
        raise NotImplementedError(
            "dag-ml bridge does not yet lower a `concat_transform` with a pass-through (None) channel "
            "(it preserves the raw processing as a parallel block); needs the data-plane (backlog #29/#31)"
        )
    return {"class": _qualname(operation), "params": _json_safe_params(operation)}


def _lower_concat_transform(step: dict[str, Any]) -> dict[str, Any]:
    """Lower a supported (single-source, replace-mode) ``concat_transform`` step to a transform node.

    Supported host-only shape: the list form ``{"concat_transform": [op, [chain...], ...]}`` (or the
    equivalent ``{"concat_transform": {"operations": [...]}}`` dict form) of transformer instances /
    chains, with NO ``name`` / ``source_processing`` (those name a per-processing 3D output) and NO
    generator (``_or_``) syntax (expanded upstream). It becomes one ``FeatureConcat`` transform node
    that hstacks the sub-transformers' fold-train-fit outputs — the model node's X-chain runs it like
    any other column-changing X-transform.

    The 3D shapes (a ``name``/``source_processing`` selector, a nested concat, a pass-through channel,
    or use inside ``feature_augmentation``'s *add* mode) raise ``NotImplementedError`` naming the
    multi-source/fusion data-plane (backlog #29/#31).
    """
    config = step["concat_transform"]
    if isinstance(config, dict):
        if "_or_" in config:
            raise NotImplementedError(
                "dag-ml bridge does not lower an unexpanded `_or_` generator inside `concat_transform`; "
                "the Python expand path handles operator-level generators"
            )
        if config.get("name") or config.get("source_processing"):
            raise NotImplementedError(
                "dag-ml bridge does not yet lower a `concat_transform` with a `name`/`source_processing` "
                "selector (it targets a named 3D processing layer); needs the data-plane (backlog #29/#31)"
            )
        operations = config.get("operations")
        if operations is None:
            raise NotImplementedError(
                "dag-ml bridge does not yet lower this `concat_transform` dict form; needs the "
                "multi-source/fusion data-plane (backlog #29/#31)"
            )
    elif isinstance(config, list):
        operations = config
    else:
        raise NotImplementedError(
            f"dag-ml bridge does not yet lower a `concat_transform` of type {type(config).__name__}; "
            f"needs the multi-source/fusion data-plane (backlog #29/#31)"
        )
    if not operations:
        raise NotImplementedError(
            "dag-ml bridge does not lower an empty `concat_transform` (no operations to concatenate)"
        )
    return {"class": _FEATURE_CONCAT_CLASS, "params": {"operations": [_concat_operation_spec(op) for op in operations]}}


def _lower_feature_augmentation(step: dict[str, Any]) -> dict[str, Any]:
    """Lower a supported (single-source, 2D-model) ``feature_augmentation`` step to a ``FeatureConcat`` node.

    ``feature_augmentation`` grows the dataset's processing axis: each operation runs on the base
    ("raw") processing, producing one new parallel preprocessing layer ``op(raw)``
    (``FeatureAugmentationController._execute_*_mode`` → ``add_features``). For a 2D model that axis is
    materialized by the ``FLAT_2D`` layout, an ``np.hstack`` of the layers in processing order
    (``layout_transformer.py``) — so the model sees the SAME matrix as a ``FeatureUnion`` over the
    layers. The action mode selects which layers survive:

    * **extend / add** — keep the raw layer beside the new ones: ``[raw, op1(raw), …, opN(raw)]`` →
      ``FeatureConcat([None, op1, …, opN])`` (the ``None`` pass-through is the raw layer).
    * **replace** — drop the raw layer: ``[op1(raw), …, opN(raw)]`` → ``FeatureConcat([op1, …, opN])``
      (identical to a ``concat_transform``).

    The ``FeatureConcat`` node lives in the model's upstream X-chain, so each augmentation
    sub-transformer is fit on fold-train only (leakage-safe) and re-applied to fold-val/test, exactly
    like ``concat_transform``. The processing axis is a FEATURE axis, not a sample axis — no new
    SAMPLE rows are created (distinct from ``sample_augmentation``), so sample-keying is preserved.

    Out of scope here (fail loud, needs the 3D data-plane / a DL operator — backlog #29/#31): operations
    that are a nested ``concat_transform`` or a dict (a multi-block construct), and the generator dict
    form (``{"_or_": …, "pick": …}``) which the Python ``expand_spec`` path must expand upstream. The
    single-source 2D model is what the host resolver materializes; a 3D/CNN model that genuinely needs
    the parallel processing channels is a Python-only DL slice, not this host-concat lowering.
    """
    operations = step["feature_augmentation"]
    action = step.get("action", "add")
    if action not in _FEATURE_AUGMENTATION_ACTIONS:
        raise ValueError(f"invalid feature_augmentation action {action!r}; must be one of {sorted(_FEATURE_AUGMENTATION_ACTIONS)}")
    if isinstance(operations, dict):
        # The generator dict form ({"_or_": [...], "pick": n, "count": m}) builds the operation set by
        # combinatorial selection — the Python expand_spec path owns it (it is expanded BEFORE this
        # bridge runs); a dict reaching here is an unexpanded generator, never a flat operation list.
        raise NotImplementedError(
            "dag-ml bridge does not lower a generator-form `feature_augmentation` (the `{_or_, pick, count}` "
            "spec); the Python expand path expands operator-level generators before lowering"
        )
    if not isinstance(operations, list):
        # A single transformer instance (e.g. {"feature_augmentation": SNV()}) is one augmentation layer.
        operations = [operations]
    layers = [op for op in operations if op is not None]
    if not layers:
        raise NotImplementedError(
            "dag-ml bridge does not lower an empty `feature_augmentation` (no operations to add)"
        )
    specs = [_concat_operation_spec(op) for op in layers]
    if action in ("extend", "add"):
        # Prepend the raw pass-through layer (FeatureConcat lowers None → sklearn "passthrough").
        specs = [None, *specs]
    return {"class": _FEATURE_CONCAT_CLASS, "params": {"operations": specs}}


def _step_to_dsl(step: Any) -> dict[str, Any]:
    """Lower one nirs4all pipeline step to a compat-DSL step object."""
    if isinstance(step, dict):
        if "model" in step:
            op = step["model"]
            # The model id is the fully-qualified class (like transforms), so any sklearn-style
            # estimator — regressor or classifier — resolves by import, not a hardcoded table.
            dsl_step: dict[str, Any] = {"model": _qualname(op), "params": _json_safe_params(op)}
            # Non-reserved siblings are model hyperparameters: plain values extend ``params``;
            # param-level generator dicts lower to native dag-ml ``generators`` so the compiler
            # expands variants and dag-ml runs generation + SELECT + refit natively (no Python expand).
            generators: list[dict[str, Any]] = []
            # A step-level `{"_grid_": {param: [vals], …}, "model": M}` over model params lowers to ONE
            # native dag-ml `Grid` generator (flat Cartesian product) — dag-ml expands + scores + SELECTs +
            # refits natively. A nested-generator / modifier-bearing / non-alphabetical / non-JSON grid the
            # native `Grid` cannot represent fails the predicate, so its `_grid_` key hits the
            # generator-keyword guard below → the Python expand path.
            native_grid = step_has_native_grid(step)
            if native_grid:
                generators.append(_grid_param_generator(step["_grid_"]))
            for key, value in step.items():
                if key in _RESERVED_MODEL_KEYS:
                    continue
                if key == "_grid_" and native_grid:
                    continue  # already lowered to a native Grid generator above
                if is_param_generator_spec(value):
                    generators.append(_param_generator(key, value))
                elif (key in _GENERATION_KEYWORDS) or (isinstance(value, dict) and _GENERATION_KEYWORDS & set(value)):
                    # A generator-shaped sibling the bridge does NOT lower natively (a non-native `_grid_`,
                    # the dict range form, or a modifier-bearing sweep). Fail loud rather than silently
                    # treat it as a plain param — run_via_dagml routes these to the Python expand path.
                    offending = sorted(value) if isinstance(value, dict) else [key]
                    raise NotImplementedError(
                        f"dag-ml bridge does not lower model param generator {offending} on `{key}`; "
                        f"use the Python expand path (operator-level / nested-grid / modifier sweeps)"
                    )
                else:
                    dsl_step["params"][key] = value
            if generators:
                dsl_step["generators"] = generators
            return dsl_step
        if "y_processing" in step:
            op = step["y_processing"]
            return {"y_processing": {"class": _qualname(op), "params": _json_safe_params(op)}}
        if "concat_transform" in step:
            # Single-source replace-mode `concat_transform` lowers to one `FeatureConcat` X-transform
            # node (hstack of sub-transformers); the 3D processing-axis shapes fail loud naming #29/#31.
            return _lower_concat_transform(step)
        if "feature_augmentation" in step:
            # Single-source 2D-model `feature_augmentation` (extend/add/replace) lowers to one
            # `FeatureConcat` X-transform node — the augmented processing layers hstacked onto the
            # feature axis (the FLAT_2D materialization a 2D model already sees). The 3D/multi-source
            # shapes (parallel channels to a DL model) fail loud naming the data-plane (#29/#31).
            return _lower_feature_augmentation(step)
        if _GENERATION_KEYWORDS & set(step):
            # A flat single `_or_` of bare operators lowers to a compat `{"_or_": [<op>, …]}` Generator
            # step — dag-ml compiles it to operator-variant models and the in-process binding scores +
            # SELECTs natively (#23 Phase 7). Any richer operator generator (`_cartesian_`/`_grid_`/
            # modifier/constraint/multi-step/multi-model) raises NotImplementedError → the Python expand
            # path (which stays on the dag-ml engine) owns it.
            return _lower_operator_generator(step)
        offending = sorted(set(step) & _UNSUPPORTED_STEP_KEYS) or sorted(step)
        raise NotImplementedError(
            f"dag-ml bridge spike does not yet serialize step keyword(s) {offending}; "
            f"see dag-ml/docs/design/DSL_NIRS4ALL_PARITY.md"
        )
    # Bare operator instance: transform or splitter. dag-ml infers the kind from
    # the class (splitters lower to campaign controller calls, not graph nodes).
    return {"class": _qualname(step), "params": _json_safe_params(step)}


def _is_x_node_step(step: Any) -> bool:
    """True if ``step`` becomes an X-side graph node that re-reads the (flattened) feature matrix.

    A bare transformer instance, a ``concat_transform``, or a ``feature_augmentation`` all operate on
    the X feature matrix. After a ``feature_augmentation`` has grown the processing axis, the LEGACY
    path applies such a step PER processing layer (``TransformerMixinController`` loops over the 3D
    processing axis), whereas the flat ``FeatureConcat`` lowering would apply it to the already-hstacked
    wide matrix — a different result. Splitters (campaign-level, no node), ``y_processing`` (operates on
    y), and ``model`` (consumes the flat 2D) are NOT per-processing X ops, so they compose correctly.
    """
    if isinstance(step, dict):
        return "concat_transform" in step or "feature_augmentation" in step
    # A bare splitter has a ``split`` method and lowers to a campaign controller call, not an X node.
    return not hasattr(step, "split")


def pipeline_to_dsl(pipeline: list[Any], dsl_id: str = "nirs4all-pipeline") -> dict[str, Any]:
    """Serialize a live nirs4all pipeline list into dag-ml compat DSL JSON.

    The single-source ``feature_augmentation`` lowering (S6) hstacks the augmented processing layers
    into one flat 2D matrix (the ``FLAT_2D`` materialization a 2D model already sees). That flattening
    is only behaviour-preserving when no LATER step must still see the per-layer processing axis — a
    bare X-transform, a ``concat_transform``, or a second ``feature_augmentation`` after it would be
    applied per-layer by the legacy path but to the hstacked matrix by the flat lowering. Such a stack
    needs the 3D data-plane (parallel processing channels), so it fails loud here naming #29/#31; a
    ``feature_augmentation`` feeding directly into the model (the canonical case) lowers cleanly.

    Raises:
        NotImplementedError: if a step uses a construct the spike does not yet cover.
    """
    for index, step in enumerate(pipeline):
        if isinstance(step, dict) and "feature_augmentation" in step and any(_is_x_node_step(later) for later in pipeline[index + 1 :]):
            raise NotImplementedError(
                "dag-ml bridge does not lower a `feature_augmentation` followed by another X-side step "
                "(a bare transform / `concat_transform` / second `feature_augmentation`): the grown "
                "processing axis must stay per-layer for that step, which needs the 3D data-plane "
                "(parallel processing channels); see backlog #29/#31. A feature_augmentation feeding "
                "directly into the model lowers as a flat feature-axis concat."
            )
    return {"id": dsl_id, "pipeline": [_step_to_dsl(step) for step in pipeline]}


def controller_manifests() -> list[dict[str, Any]]:
    """The host-controller manifests for the vertical-slice node kinds.

    One manifest per ``operator_kind`` — a dag-ml manifest serves exactly one node
    kind, so ``transform`` / ``y_transform`` / ``model`` each need their own. These
    are **control-plane declarations only**: no process-adapter command lives here
    (that is a runtime concern of the later execution phase).

    Binding is **by node kind**, mirroring nirs4all's one-controller-per-role
    dispatch (``TransformerMixinController`` / ``YTransformerMixinController`` /
    ``SklearnModelController``). Each manifest leaves ``operator_selectors`` empty,
    which dag-ml treats as a kind-level catch-all that matches any operator of that
    node kind. Class-name selectors are deliberately avoided: a generic scaler
    (``MinMaxScaler``, ``StandardScaler``, …) is an X-transform or a y-transform
    purely by its **DSL position** (bare step vs ``{"y_processing": …}`` wrapper),
    not by its class — so a ``y_transform`` selector claiming those class names
    would wrongly re-type a bare X-scaler as a target transform.
    """
    return [
        {
            "controller_id": "controller:nirs4all.transform",
            "controller_version": _NIRS4ALL_VERSION,
            "operator_kind": "transform",
            "priority": 20,
            "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
            "input_ports": [{"name": "x", "kind": "data", "representation": "tabular_numeric", "cardinality": "one"}],
            "output_ports": [{"name": "x_out", "kind": "data", "representation": "tabular_numeric", "cardinality": "one"}],
            "data_requirements": None,
            "capabilities": ["deterministic", "thread_safe", "process_safe", "uses_core_rng"],
            "operator_selectors": [],  # empty => bind any transform-kind node
            "fit_scope": "fold_train",
            "rng_policy": "uses_core_seed",
            "artifact_policy": "serializable",
        },
        {
            "controller_id": "controller:nirs4all.y_transform",
            "controller_version": _NIRS4ALL_VERSION,
            "operator_kind": "y_transform",
            "priority": 20,
            "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
            "input_ports": [{"name": "y", "kind": "target", "representation": "tabular_numeric", "cardinality": "one"}],
            "output_ports": [{"name": "y_out", "kind": "target", "representation": "tabular_numeric", "cardinality": "one"}],
            "data_requirements": None,
            "capabilities": ["deterministic", "thread_safe", "process_safe", "uses_core_rng"],
            "operator_selectors": [],  # empty => bind any y_transform-kind node (the {"y_processing": …} wrapper, not the class)
            "fit_scope": "fold_train",
            "rng_policy": "uses_core_seed",
            "artifact_policy": "serializable",
        },
        {
            "controller_id": "controller:nirs4all.model",
            "controller_version": _NIRS4ALL_VERSION,
            "operator_kind": "model",
            "priority": 20,
            "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
            "input_ports": [{"name": "x", "kind": "data", "representation": "tabular_numeric", "cardinality": "one"}],
            "output_ports": [
                {"name": "y_hat", "kind": "prediction", "representation": None, "cardinality": "one"},
                {"name": "model", "kind": "artifact", "representation": None, "cardinality": "one"},
            ],
            "data_requirements": None,
            # A prediction output port requires emits_predictions; an artifact port requires
            # emits_artifacts (dag-ml ControllerManifest::validate). No consumes_oof_predictions:
            # the vertical slice has no stacking/meta-model that would consume OOF.
            "capabilities": ["deterministic", "thread_safe", "process_safe", "uses_core_rng", "emits_predictions", "emits_artifacts", "stateful"],
            "operator_selectors": [],  # empty => bind any model-kind node
            "fit_scope": "fold_train",
            "rng_policy": "uses_core_seed",
            "artifact_policy": "serializable",
        },
        {
            # Separation-branch concat merge. The merge node is a PredictionJoin handled NATIVELY by
            # the dag-ml runtime (it reassembles the per-partition OOF blocks into one full-universe
            # OOF), but the PLAN phase still requires a controller manifest for the node kind — this
            # is that plan-time declaration. No process-adapter command runs for it: the runtime
            # intercepts the PredictionJoin(merge_mode=concat) node before the controller path.
            "controller_id": "controller:nirs4all.merge_concat",
            "controller_version": _NIRS4ALL_VERSION,
            "operator_kind": "prediction_join",
            "priority": 20,
            "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
            "input_ports": [{"name": "oof", "kind": "prediction", "representation": None, "cardinality": "many"}],
            "output_ports": [{"name": "oof", "kind": "prediction", "representation": None, "cardinality": "one"}],
            "data_requirements": None,
            "capabilities": ["deterministic", "thread_safe", "process_safe", "consumes_oof_predictions", "emits_predictions"],
            "operator_selectors": [],  # empty => bind any prediction_join-kind node
            "fit_scope": "fold_train",
            "rng_policy": "uses_core_seed",
            "artifact_policy": "serializable",
        },
        {
            # Stacking meta-model (backlog #10). The meta-node compiles to a `model`-kind node (it fits
            # a real estimator) but is distinguished from a base model by `metadata.controller_id` set to
            # this id (dag-ml's `requested_controller` binds it directly). It declares
            # `consumes_oof_predictions` so the dag-ml planner permits the base→meta `requires_oof` edges
            # (a base model lacks it, so a stray OOF edge into a base model is still refused — fail-loud).
            # The node runner reads the meta-node's `prediction_inputs[*].values` (the base branches'
            # Validation OOF, Option A) → meta-feature matrix → fits the MetaModel → emits its own OOF.
            "controller_id": "controller:nirs4all.meta_model",
            "controller_version": _NIRS4ALL_VERSION,
            "operator_kind": "model",
            "priority": 20,
            "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
            "input_ports": [{"name": "oof", "kind": "prediction", "representation": None, "cardinality": "many"}],
            "output_ports": [
                {"name": "y_hat", "kind": "prediction", "representation": None, "cardinality": "one"},
                {"name": "model", "kind": "artifact", "representation": None, "cardinality": "one"},
            ],
            "data_requirements": None,
            "capabilities": ["deterministic", "thread_safe", "process_safe", "uses_core_rng", "consumes_oof_predictions", "emits_predictions", "emits_artifacts", "stateful"],
            # A NON-EMPTY selector keeps this manifest OUT of the generic model-kind catch-all (else a
            # base model node would match BOTH this and `controller:nirs4all.model` → ambiguous). The
            # meta-node binds via `metadata.controller_id` (requested), which bypasses selectors anyway;
            # the selector's only job is to never be a generic candidate for ordinary model nodes.
            "operator_selectors": [{"refs": [_META_MODEL_REF]}],
            "fit_scope": "fold_train",
            "rng_policy": "uses_core_seed",
            "artifact_policy": "serializable",
        },
    ]


def build_dagml_plan(
    pipeline: list[Any],
    plan_id: str = "plan:nirs4all-pipeline",
    dsl_id: str = "nirs4all-pipeline",
) -> Any:
    """Lower → compile-with-controllers → build the dag-ml ``ExecutionPlan``.

    The canonical compile→plan bridge: dag-ml's ``build_execution_plan`` takes the
    ``campaign`` as a separate argument and does **not** auto-extract
    ``campaign_template`` from the compiled artifact, so the bridge reads
    ``artifact.graph`` + ``artifact.campaign_template`` and passes them explicitly,
    alongside the same controller-manifest array used to compile.

    Control-plane only — this builds the validated plan (PLAN phase); no host
    controller is executed and no feature matrix is touched.

    Raises:
        ImportError: if the dag-ml core dependency is somehow missing (``pip install nirs4all``).
        NotImplementedError: if the pipeline uses an unsupported construct.
    """
    try:
        import dag_ml
    except ImportError as exc:  # pragma: no cover - exercised only without dag-ml
        raise ImportError("dag-ml is not installed; it is a core dependency — reinstall with `pip install nirs4all`") from exc
    manifests = controller_manifests()
    artifact = dag_ml.compile_pipeline_dsl_artifact_with_controllers(pipeline_to_dsl(pipeline, dsl_id), manifests)
    return dag_ml.build_execution_plan(plan_id, artifact.graph, artifact.campaign_template, manifests)


def compile_with_dagml(pipeline: list[Any], dsl_id: str = "nirs4all-pipeline") -> Any:
    """Lower a nirs4all pipeline and compile it to a dag-ml ``CompiledPipelineArtifact``.

    Raises:
        ImportError: if the dag-ml core dependency is somehow missing (``pip install nirs4all``).
        NotImplementedError: if the pipeline uses an unsupported construct.
    """
    try:
        import dag_ml
    except ImportError as exc:  # pragma: no cover - exercised only without dag-ml
        raise ImportError("dag-ml is not installed; it is a core dependency — reinstall with `pip install nirs4all`") from exc
    return dag_ml.compile_pipeline_dsl_artifact(pipeline_to_dsl(pipeline, dsl_id))
