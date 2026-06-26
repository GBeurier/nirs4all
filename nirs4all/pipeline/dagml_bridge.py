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

dag-ml is an optional dependency (``nirs4all[dagml]``); the import is guarded.
This is **compile-only** (DSL lowering); execution via host controllers is a
later migration phase.
"""

from __future__ import annotations

import json
from typing import Any

from nirs4all import __version__ as _NIRS4ALL_VERSION

# Stacking meta-model wiring (backlog #10). The meta-node is a `model`-kind node bound to a dedicated
# controller (via `metadata.controller_id`) that declares `consumes_oof_predictions`. The ref is the
# operator-selector token that keeps the meta-model manifest out of the generic model-kind catch-all.
_META_MODEL_CONTROLLER_ID = "controller:nirs4all.meta_model"
_META_MODEL_REF = "nirs4all.meta_model"

# Every nirs4all generation keyword (mirrors config._generator.keywords.GENERATION_KEYWORDS). Used
# to detect a generator-shaped model sibling that this bridge does NOT lower natively, so it can fail
# loud instead of silently demoting it to a plain param.
_GENERATION_KEYWORDS = frozenset({"_or_", "_range_", "_log_range_", "_grid_", "_zip_", "_chain_", "_sample_", "_cartesian_"})

# Step keywords recognised by nirs4all but not yet lowered by this spike.
_UNSUPPORTED_STEP_KEYS = frozenset({
    "branch",
    "merge",
    "tag",
    "exclude",
    "sample_augmentation",
    "feature_augmentation",
    "rep_to_sources",
    "rep_to_pp",
    "finetune_params",
    "train_params",
})

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
    :func:`is_param_generator_spec`); ``_grid_``, dict-form ranges, and modifier-bearing sweeps stay
    on the Python path, so this never receives them. Field names verified against
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
            for key, value in step.items():
                if key in _RESERVED_MODEL_KEYS:
                    continue
                if is_param_generator_spec(value):
                    generators.append(_param_generator(key, value))
                elif isinstance(value, dict) and _GENERATION_KEYWORDS & set(value):
                    # A generator-shaped sibling the bridge does NOT lower natively (e.g. `_grid_`,
                    # the dict range form, or a modifier-bearing sweep). Fail loud rather than silently
                    # treat it as a plain param — run_via_dagml routes these to the Python expand path.
                    raise NotImplementedError(
                        f"dag-ml bridge does not lower model param generator {sorted(value)} on `{key}`; "
                        f"use the Python expand path (operator-level / _grid_ / modifier sweeps)"
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
        offending = sorted(set(step) & _UNSUPPORTED_STEP_KEYS) or sorted(step)
        raise NotImplementedError(
            f"dag-ml bridge spike does not yet serialize step keyword(s) {offending}; "
            f"see dag-ml/docs/design/DSL_NIRS4ALL_PARITY.md"
        )
    # Bare operator instance: transform or splitter. dag-ml infers the kind from
    # the class (splitters lower to campaign controller calls, not graph nodes).
    return {"class": _qualname(step), "params": _json_safe_params(step)}


def pipeline_to_dsl(pipeline: list[Any], dsl_id: str = "nirs4all-pipeline") -> dict[str, Any]:
    """Serialize a live nirs4all pipeline list into dag-ml compat DSL JSON.

    Raises:
        NotImplementedError: if a step uses a construct the spike does not yet cover.
    """
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
        ImportError: if dag-ml is not installed (``pip install nirs4all[dagml]``).
        NotImplementedError: if the pipeline uses an unsupported construct.
    """
    try:
        import dag_ml
    except ImportError as exc:  # pragma: no cover - exercised only without dag-ml
        raise ImportError("dag-ml is not installed; install with `pip install nirs4all[dagml]`") from exc
    manifests = controller_manifests()
    artifact = dag_ml.compile_pipeline_dsl_artifact_with_controllers(pipeline_to_dsl(pipeline, dsl_id), manifests)
    return dag_ml.build_execution_plan(plan_id, artifact.graph, artifact.campaign_template, manifests)


def compile_with_dagml(pipeline: list[Any], dsl_id: str = "nirs4all-pipeline") -> Any:
    """Lower a nirs4all pipeline and compile it to a dag-ml ``CompiledPipelineArtifact``.

    Raises:
        ImportError: if dag-ml is not installed (``pip install nirs4all[dagml]``).
        NotImplementedError: if the pipeline uses an unsupported construct.
    """
    try:
        import dag_ml
    except ImportError as exc:  # pragma: no cover - exercised only without dag-ml
        raise ImportError("dag-ml is not installed; install with `pip install nirs4all[dagml]`") from exc
    return dag_ml.compile_pipeline_dsl_artifact(pipeline_to_dsl(pipeline, dsl_id))
