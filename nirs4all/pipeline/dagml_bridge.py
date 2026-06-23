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

# Step keywords recognised by nirs4all but not yet lowered by this spike.
_UNSUPPORTED_STEP_KEYS = frozenset({
    "branch",
    "merge",
    "tag",
    "exclude",
    "sample_augmentation",
    "feature_augmentation",
    "concat_transform",
    "rep_to_sources",
    "rep_to_pp",
    "finetune_params",
    "train_params",
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


def _model_name(obj: Any) -> str:
    """Short class name dag-ml uses as the opaque model operator id."""
    return obj.__name__ if isinstance(obj, type) else type(obj).__name__


def _step_to_dsl(step: Any) -> dict[str, Any]:
    """Lower one nirs4all pipeline step to a compat-DSL step object."""
    if isinstance(step, dict):
        if "model" in step:
            op = step["model"]
            return {"model": _model_name(op), "params": _json_safe_params(op)}
        if "y_processing" in step:
            op = step["y_processing"]
            return {"y_processing": {"class": _qualname(op), "params": _json_safe_params(op)}}
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
