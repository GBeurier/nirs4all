"""Fail-closed semantic migration checks before DAG-ML routing.

This module is deliberately narrower than a second pipeline parser.  It only proves two legacy
semantics that may not use ``allow_legacy_fallback``: refitting with held-out partitions, and a
stateful ``concat_transform`` materialized before cross-validation.  All ordinary unsupported
shapes continue through the existing capability fallback path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import yaml

from .errors import (
    DagMlMigrationRequired,
    DagMlPipelinePreflightRequired,
    DagMlRefitParamsMigrationRequired,
    DagMlStatefulConcatTransformMigrationRequired,
)

_MAX_PREFLIGHT_VARIANTS = 10_000
_SERIALIZED_COMPONENT_KEYS = frozenset({"class", "function", "module", "object", "instance"})
_SEPARATION_BRANCH_KEYS = frozenset({"by_tag", "by_metadata", "by_filter", "by_source"})
_BRANCH_OPTION_KEYS = frozenset({"parallel", "n_jobs", "metadata", "params"})
_RANDOM_COUNT_GENERATOR_KEYS = frozenset({"_or_", "_grid_", "_zip_", "_cartesian_"})
_GENERATOR_KEYS = frozenset({"_or_", "_range_", "_log_range_", "_grid_", "_zip_", "_chain_", "_sample_", "_cartesian_"})


@dataclass(frozen=True)
class _Opaque:
    """Preserve a payload's identity while keeping generator expansion out of it."""

    value: Any


def _raise_uninspectable(context: str, cause: BaseException | None = None) -> NoReturn:
    """Refuse an active DSL shape whose semantic boundary cannot be proven safely."""

    error = DagMlPipelinePreflightRequired(
        f"engine='dag-ml' cannot inspect {context} before semantic migration preflight. "
        "Automatic legacy fallback is disabled for this active pipeline form; provide a documented "
        "list/dict/YAML-or-JSON/Path/PipelineConfigs pipeline, or select engine='legacy' explicitly."
    )
    if cause is None:
        raise error
    raise error from cause


def _is_serialized_component(value: Mapping[str, Any]) -> bool:
    """Whether a mapping is an atomic serialized operator rather than a workflow step."""

    return bool(_SERIALIZED_COMPONENT_KEYS & value.keys())


def _unwrap(value: Any) -> Any:
    return value.value if isinstance(value, _Opaque) else value


def _minimal_step_config(value: Any) -> Any:
    """Unwrap trace replay steps without treating arbitrary payload objects as DSL nodes."""

    try:
        from nirs4all.pipeline.trace.extractor import MinimalPipelineStep
    except Exception:  # pragma: no cover - import is part of the installed package contract.
        return value
    return value.step_config if isinstance(value, MinimalPipelineStep) else value


def _effective_seed(generator: Mapping[str, Any], root_seed: int | None) -> Any:
    """Match generator-core's per-node seed lookup without inventing seed propagation.

    A mixed ``_or_`` is expanded by ``_expand_mixed_or_node`` rather than ``OrStrategy``. That helper
    leaves ``_seed_`` in the base payload, so only its caller seed affects selection. Treating the
    payload key as effective here would let a duplication branch hide an unsafe candidate.
    """

    if "_or_" in generator:
        from nirs4all.pipeline.config._generator.keywords import PURE_OR_KEYS

        if not set(generator).issubset(PURE_OR_KEYS):
            return root_seed

    return generator.get("_seed_", root_seed)


def _random_count_is_unseeded(value: Mapping[str, Any], root_seed: int | None) -> bool:
    """Whether a generator's positive count can randomly hide an otherwise active variant."""

    count = value.get("count")
    return type(count) is int and count > 0 and bool(_RANDOM_COUNT_GENERATOR_KEYS & value.keys()) and _effective_seed(value, root_seed) is None


def _unseeded_choice_sample(value: Mapping[str, Any], root_seed: int | None) -> list[Any] | None:
    """Return every possible value of an unseeded choice sample, or ``None`` when not applicable."""

    sample = value.get("_sample_")
    if _effective_seed(value, root_seed) is not None or not isinstance(sample, Mapping):
        return None
    if sample.get("distribution", "uniform") != "choice":
        return None
    choices = sample.get("values", [])
    num = sample.get("num", 10)
    count = value.get("count")
    if not isinstance(choices, list) or type(num) is not int or num < 0 or (count is not None and type(count) is not int):
        _raise_uninspectable("an unseeded _sample_ generator")
    emitted = min(num, count) if count is not None and count > 0 else num
    return choices if emitted > 0 else []


def _has_unprojectable_unseeded_sample(value: Any, root_seed: int | None, active: set[int] | None = None) -> bool:
    """Whether a relevant ZipStrategy column retains unbounded random sampling.

    An unseeded choice sample is rewritten exhaustively by ``_project_for_expansion``. Other sample
    distributions produce fresh values on every expansion and cannot prove a semantic boundary from
    one draw, so their relevant column must use the generic preflight refusal instead.
    """

    if isinstance(value, _Opaque) or not isinstance(value, (Mapping, list, tuple)):
        return False
    if active is None:
        active = set()
    marker = id(value)
    if marker in active:
        return True
    active.add(marker)
    try:
        if isinstance(value, Mapping):
            if "_sample_" in value and _effective_seed(value, root_seed) is None:
                sample = value.get("_sample_")
                if not isinstance(sample, Mapping) or sample.get("distribution", "uniform") != "choice":
                    return True
            return any(_has_unprojectable_unseeded_sample(item, root_seed, active) for item in value.values())
        return any(_has_unprojectable_unseeded_sample(item, root_seed, active) for item in value)
    finally:
        active.remove(marker)


def _opaque_with_shape(value: Any) -> Any:
    """Keep ZipStrategy's list cardinality while leaving payloads inactive."""

    # ``ZipStrategy`` treats only a list as a value column.  A tuple is one scalar value, so
    # projecting it as several opaque values would fabricate an alignment that legacy never has.
    if isinstance(value, list):
        return [_Opaque(item) for item in value]
    return _Opaque(value)


def _zip_column_values(value: Any, root_seed: int | None) -> list[Any] | None:
    """Return all values a relevant ZipStrategy column can emit safely.

    ``None`` means the column's active result set is too large or malformed to prove.  It is used
    only to decide whether an opaque generated sibling can hide a tracked migration; it never
    substitutes an invented cardinality into the projected generator. Unseeded random choices are
    first projected exhaustively, matching the main pipeline scan rather than sampling once.
    """

    if isinstance(value, list):
        # ZipStrategy deliberately does not recursively expand list members. Duplication branches do
        # perform a later nested-generator phase, though, so inspect every possible deferred member
        # here instead of allowing that later phase to introduce a tracked migration unseen.
        try:
            from nirs4all.pipeline.config._generator.keywords import has_nested_generator_keywords

            values: list[Any] = []
            for item in value:
                if not isinstance(item, dict) or not has_nested_generator_keywords(item):
                    values.append(item)
                    continue
                expanded_item = _zip_column_values(item, root_seed)
                if expanded_item is None:
                    return None
                values.extend(expanded_item)
            return values
        except Exception:  # noqa: BLE001 - a later branch expansion cannot be sampled once here.
            return None
    if not isinstance(value, dict):
        return [value]
    if _has_unprojectable_unseeded_sample(value, root_seed):
        return None
    try:
        from nirs4all.pipeline.config.generator import count_combinations, expand_spec

        projected = _project_for_expansion(value, root_seed)
        count = count_combinations(projected)
        if type(count) is not int or count > _MAX_PREFLIGHT_VARIANTS:
            return None
        values = expand_spec(projected, seed=root_seed)
    except Exception:  # noqa: BLE001 - an unprovable relevant column must not hide a migration.
        return None
    if not isinstance(values, list) or len(values) > _MAX_PREFLIGHT_VARIANTS:
        return None
    return values


def _zip_refit_column_may_require_migration(value: Any, root_seed: int | None) -> bool:
    """Whether a ZipStrategy refit column can emit the held-out-refit override."""

    values = _zip_column_values(value, root_seed)
    if values is None:
        return True
    return any(isinstance(item, Mapping) and item.get("use_all_partitions") is True for item in values)


def _zip_concat_column_may_require_migration(value: Any, root_seed: int | None) -> bool:
    """Whether a ZipStrategy concat column can emit a stateful configuration."""

    values = _zip_column_values(value, root_seed)
    if values is None:
        return True
    try:
        return any(_concat_requires_migration(item) for item in values)
    except Exception:  # noqa: BLE001 - only the generic preflight boundary is sound here.
        return True


def _has_opaque_generated_mapping(value: Mapping[str, Any]) -> bool:
    """Whether an opaque payload mapping is itself active generator input."""

    try:
        from nirs4all.pipeline.config._generator.keywords import has_nested_generator_keywords

        return any(isinstance(item, dict) and has_nested_generator_keywords(item) for key, item in value.items() if key in {"model", "metadata", "params"})
    except Exception:  # noqa: BLE001 - only relevant unknown columns are fail-closed below.
        return any(isinstance(item, dict) for key, item in value.items() if key in {"model", "metadata", "params"})


def _generator_columns_may_require_migration(value: Mapping[str, Any], root_seed: int | None, pre_cv_possible: bool) -> bool:
    """Whether one grid/zip column map can produce either tracked semantic shape."""

    return ("model" in value and "refit_params" in value and _zip_refit_column_may_require_migration(value["refit_params"], root_seed)) or (
        pre_cv_possible and "concat_transform" in value and _zip_concat_column_may_require_migration(value["concat_transform"], root_seed)
    )


def _generator_columns_need_preflight(
    value: Mapping[str, Any],
    root_seed: int | None,
    pre_cv_possible: bool,
    generator_key: str,
    selection_context: bool,
) -> bool:
    """Fail closed when opacity can alter a selected grid/zip candidate population."""

    if not _has_opaque_generated_mapping(value):
        return False
    # Zip truncates by alignment even without ``count``. Grid is otherwise exhaustive unless a
    # current or enclosing selector retains only part of its candidate population.
    if generator_key != "_zip_" and not selection_context:
        return False
    return _generator_columns_may_require_migration(value, root_seed, pre_cv_possible)


def _step_refit_may_require_migration(value: Any, root_seed: int | None) -> bool:
    """Whether a direct step-level refit configuration can enable held-out refitting."""

    values = _zip_column_values(value, root_seed)
    if values is None:
        return True
    return any(isinstance(item, Mapping) and item.get("use_all_partitions") is True for item in values)


def _step_concat_may_require_migration(value: Any, root_seed: int | None) -> bool:
    """Whether a direct step-level concat configuration can be stateful."""

    if not isinstance(value, dict):
        try:
            return _concat_requires_migration(value)
        except Exception:  # noqa: BLE001 - a relevant unresolved concat is uninspectable.
            return True
    values = _zip_column_values(value, root_seed)
    if values is None:
        return True
    try:
        return any(_concat_requires_migration(item) for item in values)
    except Exception:  # noqa: BLE001 - a relevant unresolved concat is uninspectable.
        return True


def _mapping_may_require_migration(value: Mapping[str, Any], root_seed: int | None, pre_cv_possible: bool) -> bool:
    """Inspect direct and mixed-OR step candidates without interpreting unrelated payloads."""

    def candidate_may_require_migration(candidate: Mapping[str, Any]) -> bool:
        return ("model" in candidate and "refit_params" in candidate and _step_refit_may_require_migration(candidate["refit_params"], root_seed)) or (
            pre_cv_possible and "concat_transform" in candidate and _step_concat_may_require_migration(candidate["concat_transform"], root_seed)
        )

    if candidate_may_require_migration(value):
        return True
    choices = value.get("_or_")
    if not isinstance(choices, list):
        return False
    from nirs4all.pipeline.config._generator.keywords import PURE_OR_KEYS

    base = {key: item for key, item in value.items() if key not in PURE_OR_KEYS}
    return any(candidate_may_require_migration({**base, **choice}) for choice in choices if isinstance(choice, Mapping))


def _selection_can_change_population(value: Mapping[str, Any], root_seed: int | None) -> bool:
    """Whether this generator may retain only an order/cardinality-sensitive subset."""

    if "_zip_" in value:
        return True
    count = value.get("count")
    if type(count) is not int or count <= 0:
        return False
    if "_chain_" in value:
        # Unseeded chains take the first N items; seeded chains sample N items.
        return True
    return bool({"_or_", "_grid_", "_cartesian_"} & value.keys()) and _effective_seed(value, root_seed) is not None


def _selector_population_has_opaque_generator(value: Any, active: set[int] | None = None) -> bool:
    """Whether one selected population contains an opaque mapping that expands its size/order."""

    if not isinstance(value, (Mapping, list, tuple)):
        return False
    if active is None:
        active = set()
    marker = id(value)
    if marker in active:
        return True
    active.add(marker)
    try:
        if isinstance(value, Mapping):
            if _is_serialized_component(value) or _has_opaque_generated_mapping(value):
                return _has_opaque_generated_mapping(value)
            return any(_selector_population_has_opaque_generator(item, active) for key, item in value.items() if key not in {"model", "metadata", "params"})
        return any(_selector_population_has_opaque_generator(item, active) for item in value)
    finally:
        active.remove(marker)


def _selector_population_may_require_migration(
    value: Any,
    root_seed: int | None,
    pre_cv_possible: bool,
    active: set[int] | None = None,
) -> bool:
    """Whether any selected candidate can carry one of the two semantic migrations."""

    if not isinstance(value, (Mapping, list, tuple)):
        return False
    if active is None:
        active = set()
    marker = id(value)
    if marker in active:
        return True
    active.add(marker)
    try:
        if isinstance(value, Mapping):
            if _is_serialized_component(value):
                return False
            if _mapping_may_require_migration(value, root_seed, pre_cv_possible):
                return True
            return any(_selector_population_may_require_migration(item, root_seed, pre_cv_possible, active) for key, item in value.items() if key not in {"model", "metadata", "params"})
        return any(_selector_population_may_require_migration(item, root_seed, pre_cv_possible, active) for item in value)
    finally:
        active.remove(marker)


def _selected_population_needs_preflight(value: Mapping[str, Any], root_seed: int | None, pre_cv_possible: bool) -> bool:
    """Whether an opaque sibling can change which migration-bearing candidate selection retains."""

    return _selection_can_change_population(value, root_seed) and _selector_population_has_opaque_generator(value) and _selector_population_may_require_migration(value, root_seed, pre_cv_possible)


def _project_generator_columns(
    value: Mapping[str, Any],
    root_seed: int | None,
    active: set[int],
    block_inherited_seed: bool,
    generator_key: str,
    pre_cv_possible: bool,
    selection_context: bool,
) -> dict[str, Any]:
    """Preserve map-generator alignment while making model and parameter payloads opaque."""

    marker = id(value)
    if marker in active:
        _raise_uninspectable("a cyclic active generator")
    active.add(marker)
    try:
        if _generator_columns_need_preflight(
            value,
            root_seed,
            pre_cv_possible,
            generator_key,
            selection_context,
        ):
            _raise_uninspectable("a selected generator whose opaque generated mapping can change a migration-bearing candidate")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"model", "metadata", "params"}:
                result[key] = _opaque_with_shape(item)
            else:
                result[key] = _project_for_expansion(
                    item,
                    root_seed,
                    active,
                    block_inherited_seed,
                    pre_cv_possible,
                    selection_context,
                )
        return result
    finally:
        active.remove(marker)


def _project_branch_for_expansion(
    value: Any,
    root_seed: int | None,
    active: set[int],
    pre_cv_possible: bool,
    selection_context: bool,
) -> Any:
    """Project a branch with the seed contract of the controller that expands it."""

    if isinstance(value, Mapping) and _SEPARATION_BRANCH_KEYS & value.keys():
        # PipelineConfigs expands separation-branch generators at its top level, carrying the root seed.
        return _project_for_expansion(
            value,
            root_seed,
            active,
            pre_cv_possible=pre_cv_possible,
            selection_context=selection_context,
        )
    # Duplication branches are expanded later by BranchController, which calls expand_spec() without
    # PipelineConfigs.random_state. A ``_seed_: None`` sentinel must be projected onto each of their
    # generator nodes, otherwise the OUTER expand_spec(..., seed=root_seed) would re-inherit that seed.
    # Their local explicit _seed_ remains meaningful.
    return _project_for_expansion(
        value,
        None,
        active,
        block_inherited_seed=True,
        pre_cv_possible=pre_cv_possible,
        selection_context=selection_context,
    )


def _project_for_expansion(
    value: Any,
    root_seed: int | None,
    active: set[int] | None = None,
    block_inherited_seed: bool = False,
    pre_cv_possible: bool = True,
    selection_context: bool = False,
) -> Any:
    """Copy only active DSL structure for generator expansion.

    ``metadata`` and component ``params`` are opaque payloads: their keys must never create a semantic
    refusal.  Keeping them opaque also means a cycle in metadata cannot poison the active-DSL check.
    """

    if active is None:
        active = set()
    value = _minimal_step_config(value)
    if isinstance(value, _Opaque):
        return value
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            _raise_uninspectable("a cyclic active pipeline sequence")
        active.add(marker)
        try:
            return [
                _project_for_expansion(
                    item,
                    root_seed,
                    active,
                    block_inherited_seed,
                    pre_cv_possible,
                    selection_context,
                )
                for item in value
            ]
        finally:
            active.remove(marker)
    if not isinstance(value, Mapping):
        return value
    if _is_serialized_component(value):
        return _Opaque(value)

    marker = id(value)
    if marker in active:
        _raise_uninspectable("a cyclic active pipeline mapping")
    active.add(marker)
    try:
        is_selector = _selection_can_change_population(value, root_seed)
        if _selected_population_needs_preflight(value, root_seed, pre_cv_possible):
            _raise_uninspectable("a selected generator whose opaque sibling can change a migration-bearing candidate")
        selection_context = selection_context or is_selector
        if selection_context and _has_opaque_generated_mapping(value) and _mapping_may_require_migration(value, root_seed, pre_cv_possible):
            _raise_uninspectable("a selected generator whose opaque generated mapping can change a migration-bearing candidate")
        sampled_choices = _unseeded_choice_sample(value, root_seed)
        if sampled_choices is not None:
            # A raw pipeline is normalized again by the legacy runner.  An unseeded choice can change
            # between this preflight and that later normalization, so inspect every possible choice.
            sampled_result: dict[str, Any] = {
                "_or_": [
                    _project_for_expansion(
                        choice,
                        root_seed,
                        active,
                        block_inherited_seed,
                        pre_cv_possible,
                        selection_context,
                    )
                    for choice in sampled_choices
                ]
            }
            if block_inherited_seed:
                sampled_result["_seed_"] = None
            return sampled_result

        remove_random_count = _random_count_is_unseeded(value, root_seed)
        if remove_random_count and "_or_" in value and "_weights_" in value:
            # A weighted selection can make a syntactically unsafe option unreachable (for example a
            # zero-weight candidate). Reproducing its support exactly across nested expansion is a
            # separate grammar problem, so fail closed as *uninspectable* rather than falsely call that
            # candidate an active semantic migration.
            _raise_uninspectable("an unseeded weighted _or_ generator")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key == "count" and remove_random_count:
                # ``_or_``, grid, zip, and cartesian sample a positive count at random when unseeded.
                # Enumerating their full legal output is the only stable proof before a later legacy run.
                continue
            if key in {"metadata", "params", "model"}:
                result[key] = _Opaque(item)
            elif key == "branch":
                result[key] = _project_branch_for_expansion(
                    item,
                    root_seed,
                    active,
                    pre_cv_possible,
                    selection_context,
                )
            elif key in {"_grid_", "_zip_"} and isinstance(item, Mapping):
                result[key] = _project_generator_columns(
                    item,
                    root_seed,
                    active,
                    block_inherited_seed,
                    generator_key=key,
                    pre_cv_possible=pre_cv_possible,
                    selection_context=selection_context,
                )
            else:
                result[key] = _project_for_expansion(
                    item,
                    root_seed,
                    active,
                    block_inherited_seed,
                    pre_cv_possible,
                    selection_context,
                )
        if block_inherited_seed and "_seed_" not in result and _GENERATOR_KEYS & value.keys():
            result["_seed_"] = None
        return result
    finally:
        active.remove(marker)


def _project_pipeline_sequence(steps: list[Any], root_seed: int | None) -> list[Any]:
    """Project top-level steps while retaining the definite pre-CV prefix boundary.

    A generated splitter is not proof that every output has entered CV, so only an already concrete
    splitter closes the prefix. This deliberately keeps uncertain nested control flow fail-closed.
    """

    active: set[int] = set()
    marker = id(steps)
    active.add(marker)
    try:
        projected: list[Any] = []
        pre_cv_possible = True
        for step in steps:
            projected.append(_project_for_expansion(step, root_seed, active, pre_cv_possible=pre_cv_possible))
            pre_cv_possible = pre_cv_possible and not _is_splitter(_minimal_step_config(_unwrap(step)))
        return projected
    finally:
        active.remove(marker)


def _load_string_definition(value: str) -> Any:
    """Load the documented YAML/JSON path or literal form while retaining root ``random_state``."""

    path = Path(value)
    suffix = path.suffix.lower()
    if suffix in {".json", ".yaml", ".yml"}:
        if not path.is_file():
            _raise_uninspectable(f"pipeline configuration path {value!r}")
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if suffix == ".json" else yaml.safe_load(text)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return yaml.safe_load(value)


def _raw_steps_and_seed(pipeline: Any) -> tuple[list[Any], int | None]:
    """Normalize one public definition just far enough to inspect its active template."""

    if isinstance(pipeline, Path):
        pipeline = str(pipeline)
    try:
        definition = _load_string_definition(pipeline) if isinstance(pipeline, str) else pipeline
        if isinstance(definition, list):
            return list(definition), None
        if not isinstance(definition, Mapping):
            _raise_uninspectable("pipeline")
        key = "pipeline" if "pipeline" in definition else "steps" if "steps" in definition else None
        if key is None or not isinstance(definition[key], list):
            _raise_uninspectable("pipeline wrapper")
        seed = definition.get("random_state")
        return list(definition[key]), seed if type(seed) is int else None
    except DagMlMigrationRequired:
        raise
    except Exception as error:  # noqa: BLE001 - this is the fail-closed public boundary.
        _raise_uninspectable("pipeline configuration", error)


def _looks_like_pipeline_step(value: Any) -> bool:
    """Mirror the public list/batch disambiguation without importing the public API module."""

    if value is None or isinstance(value, (dict, type)):
        return True
    if hasattr(value, "fit") or hasattr(value, "transform") or hasattr(value, "predict") or hasattr(value, "split"):
        return True
    return bool(hasattr(value, "__class__") and value.__class__.__module__.startswith("nirs4all"))


def _batch_status(value: list[Any], pipeline_configs_type: type[Any]) -> tuple[bool, int | None]:
    """Return ``(is_batch, opaque_member_index)`` for the public outer-list grammar."""

    saw_pipeline_spec = False
    saw_step = False
    opaque_member: int | None = None
    for index, item in enumerate(value):
        if isinstance(item, (pipeline_configs_type, str, Path, list)) or (isinstance(item, Mapping) and any(key in item for key in ("pipeline", "steps"))):
            saw_pipeline_spec = True
        elif _looks_like_pipeline_step(item):
            saw_step = True
        elif opaque_member is None:
            opaque_member = index
    if saw_pipeline_spec and opaque_member is not None:
        return False, opaque_member
    return saw_pipeline_spec and not saw_step, None


def _public_variants(pipeline: Any, active_batches: set[int] | None = None) -> list[tuple[list[Any], int | None]]:
    """Return every inspectable public pipeline member and its generator root seed."""

    from nirs4all.pipeline.config.pipeline_config import PipelineConfigs

    if isinstance(pipeline, PipelineConfigs):
        steps = pipeline.steps
        if not isinstance(steps, list) or any(not isinstance(item, list) for item in steps):
            _raise_uninspectable("PipelineConfigs.steps")
        seed = pipeline.random_state if type(pipeline.random_state) is int else None
        return [(list(item), seed) for item in steps]
    if not isinstance(pipeline, list):
        return [_raw_steps_and_seed(pipeline)]

    is_batch, opaque_member = _batch_status(pipeline, PipelineConfigs)
    if opaque_member is not None:
        _raise_uninspectable(f"pipeline batch item {opaque_member}")
    if not is_batch:
        return [(list(pipeline), None)]

    if active_batches is None:
        active_batches = set()
    marker = id(pipeline)
    if marker in active_batches:
        _raise_uninspectable("a cyclic public pipeline batch")
    active_batches.add(marker)
    try:
        variants: list[tuple[list[Any], int | None]] = []
        for item in pipeline:
            variants.extend(_public_variants(item, active_batches))
        return variants
    finally:
        active_batches.remove(marker)


def _expanded_variants(steps: list[Any], root_seed: int | None) -> list[list[Any]]:
    """Expand the same generator grammar legacy uses, with exhaustive unseeded choices."""

    from nirs4all.pipeline.config.generator import count_combinations, expand_spec

    projected = _project_pipeline_sequence(steps, root_seed)
    try:
        count = count_combinations(projected)
        if type(count) is not int or count > _MAX_PREFLIGHT_VARIANTS:
            _raise_uninspectable("a generator set larger than the inspectable preflight bound")
        expanded = expand_spec(projected, seed=root_seed)
    except DagMlMigrationRequired:
        raise
    except Exception as error:  # noqa: BLE001 - an opaque active generator is not a fallback path.
        _raise_uninspectable("an active pipeline generator", error)
    if len(expanded) > _MAX_PREFLIGHT_VARIANTS or any(not isinstance(item, list) for item in expanded):
        _raise_uninspectable("expanded pipeline variants")
    return expanded


def _is_splitter(value: Any) -> bool:
    """Recognize raw and serialized splitter steps without descending into their params."""

    value = _unwrap(value)
    if hasattr(value, "split"):
        return True
    if not isinstance(value, (str, Mapping)):
        return False
    try:
        from nirs4all.pipeline.config.component_serialization import deserialize_component

        return hasattr(deserialize_component(value), "split")
    except Exception:  # noqa: BLE001 - an unknown component cannot prove that CV already began.
        return False


def _operator_is_stateless(value: Any) -> bool:
    """Resolve serialized concat leaves before reusing the existing conservative stateless probe."""

    value = _unwrap(value)
    if value is None:
        return True
    if isinstance(value, (str, Mapping)):
        try:
            from nirs4all.pipeline.config.component_serialization import deserialize_component

            value = deserialize_component(value)
        except Exception:  # noqa: BLE001 - unresolved leaves are not proven stateless.
            return False
    try:
        from .run_paths import _operator_is_stateless as probe

        return probe(value)
    except Exception:  # noqa: BLE001 - the probe itself is intentionally fail-closed.
        return False


def _concat_requires_migration(config: Any, active: set[int] | None = None) -> bool:
    """Whether one active concat configuration contains a learned or opaque leaf."""

    if active is None:
        active = set()
    config = _unwrap(config)
    if isinstance(config, (list, tuple, Mapping)):
        marker = id(config)
        if marker in active:
            _raise_uninspectable("a cyclic active concat_transform")
        active.add(marker)
    else:
        marker = None
    try:
        if isinstance(config, Mapping) and "concat_transform" in config:
            return _concat_requires_migration(config["concat_transform"], active)
        if isinstance(config, Mapping):
            # Legacy's concat controller also accepts a named-operation map such as
            # {"snv": StandardNormalVariate()}; without an explicit ``operations`` key, every value
            # is an operation rather than an opaque configuration payload.
            operations = config.get("operations") if "operations" in config else list(config.values())
        else:
            operations = config
        if not isinstance(operations, (list, tuple)):
            return True
        for operation in operations:
            operation = _unwrap(operation)
            if isinstance(operation, (list, tuple)):
                if _concat_requires_migration(operation, active):
                    return True
            elif isinstance(operation, Mapping) and "concat_transform" in operation:
                if _concat_requires_migration(operation["concat_transform"], active):
                    return True
            elif not _operator_is_stateless(operation):
                return True
        return False
    finally:
        if marker is not None:
            active.remove(marker)


def _raise_refit_migration() -> None:
    raise DagMlRefitParamsMigrationRequired(
        "engine='dag-ml' refuses refit_params.use_all_partitions=True: legacy may refit on held-out "
        "test partitions, while DAG-ML fixes REFIT FullTrain to FoldSet sample IDs. This is an explicit "
        "migration requirement, not a legacy-fallback boundary; remove the override or select "
        "engine='legacy' explicitly."
    )


def _raise_concat_migration() -> None:
    raise DagMlStatefulConcatTransformMigrationRequired(
        "engine='dag-ml' refuses a stateful concat_transform before CV: legacy materializes learned "
        "features globally, while native FeatureConcat fits them fold-locally. This is an explicit "
        "migration requirement, not a legacy-fallback boundary; use a native-equivalent pipeline or "
        "select engine='legacy' explicitly."
    )


def _scan_branch(branch: Any, seen_splitter: bool, active: set[int]) -> None:
    """Inspect only executable branch bodies, never branch metadata or selectors."""

    branch = _unwrap(branch)
    if isinstance(branch, list):
        for entry in branch:
            entry = _unwrap(entry)
            if isinstance(entry, Mapping) and "steps" in entry:
                _scan_sequence(entry["steps"], seen_splitter, active)
            else:
                _scan_step(entry, seen_splitter, active)
        return
    if not isinstance(branch, Mapping):
        _scan_step(branch, seen_splitter, active)
        return
    if _SEPARATION_BRANCH_KEYS & branch.keys():
        body = branch.get("steps")
        if isinstance(body, Mapping):
            for item in body.values():
                _scan_step(item, seen_splitter, active)
        elif body is not None:
            _scan_step(body, seen_splitter, active)
        return
    for name, body in branch.items():
        if not isinstance(name, str) or name.startswith("_") or name in _BRANCH_OPTION_KEYS:
            continue
        _scan_step(body, seen_splitter, active)


def _scan_step(value: Any, seen_splitter: bool, active: set[int]) -> bool:
    """Inspect one expanded step and return whether its containing sequence has reached CV."""

    value = _minimal_step_config(_unwrap(value))
    if isinstance(value, list):
        return _scan_sequence(value, seen_splitter, active)
    if not isinstance(value, Mapping):
        return seen_splitter or _is_splitter(value)
    if _is_serialized_component(value):
        return seen_splitter or _is_splitter(value)

    marker = id(value)
    if marker in active:
        _raise_uninspectable("a cyclic active pipeline mapping")
    active.add(marker)
    try:
        if "model" in value:
            refit = _unwrap(value.get("refit_params"))
            if isinstance(refit, Mapping) and refit.get("use_all_partitions") is True:
                _raise_refit_migration()
        if "concat_transform" in value and not seen_splitter and _concat_requires_migration(value["concat_transform"]):
            _raise_concat_migration()
        if "branch" in value:
            _scan_branch(value["branch"], seen_splitter, active)
        if "feature_augmentation" in value:
            _scan_step(value["feature_augmentation"], seen_splitter, active)
        return seen_splitter or _is_splitter(value)
    finally:
        active.remove(marker)


def _scan_sequence(steps: Any, seen_splitter: bool, active: set[int] | None = None) -> bool:
    """Scan a concrete pipeline sequence in execution order to identify the pre-CV prefix."""

    steps = _unwrap(steps)
    if not isinstance(steps, (list, tuple)):
        return _scan_step(steps, seen_splitter, active or set())
    if active is None:
        active = set()
    marker = id(steps)
    if marker in active:
        _raise_uninspectable("a cyclic active pipeline sequence")
    active.add(marker)
    try:
        for step in steps:
            seen_splitter = _scan_step(step, seen_splitter, active)
        return seen_splitter
    finally:
        active.remove(marker)


def preflight_dagml_pipeline_migration(pipeline: Any) -> None:
    """Refuse semantic migrations before backend availability, datasets, or legacy fallback can run.

    Raw definitions use the library's generator expander.  Deterministic generators are inspected in
    their exact legacy expansion; unseeded random selections are expanded conservatively across every
    possible candidate, since a later legacy normalization may choose a different result.
    """

    for steps, root_seed in _public_variants(pipeline):
        for variant in _expanded_variants(steps, root_seed):
            _scan_sequence(variant, seen_splitter=False)
