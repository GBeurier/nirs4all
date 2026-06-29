"""Contract snapshot: the stable 0.9.x public Python API surface.

This file locks the part of the 0.9.x stable contract that downstream code
(notably nirs4all-studio) binds to directly: the module-level entry points,
their exact call signatures, the package ``__all__`` exports, and the public
shape of the result objects.

Each assertion compares the *live* library against a frozen snapshot captured
from the current code. A failure means one of:

  * a public function signature changed (param added/removed/renamed, default
    changed, kw-only-ness changed) — ``test_<fn>_signature_frozen``;
  * an export was added to or removed from ``nirs4all.__all__`` /
    ``nirs4all.api.__all__`` — ``test_package_all_frozen`` /
    ``test_api_all_frozen``;
  * a public attribute/method disappeared from a result class —
    ``test_<Result>_public_surface``.

Signatures are compared as the full ``str(inspect.signature(fn))`` string
(this is the strictest contract: it catches default-value and annotation drift,
not just name changes). The ``__all__`` lists are compared exactly. Result
classes are checked with a *subset* assertion (the frozen names must remain a
subset of the live public surface) so that adding new helpers does not break
the test, while removing or renaming a documented one does.

Snapshots were captured from nirs4all 0.9.1. Updating them requires a
deliberate decision because these are stable contracts within the 0.9.x line.
"""

from __future__ import annotations

import dataclasses
import inspect

import nirs4all
import nirs4all.api as nirs4all_api

# ---------------------------------------------------------------------------
# Frozen signature snapshots (captured from nirs4all 0.9.1).
#
# Captured via: str(inspect.signature(getattr(nirs4all, name)))
# These are the exact, full signature strings including annotations and
# defaults. Any drift (rename, reorder, default change, kw-only change) fails.
# ---------------------------------------------------------------------------

EXPECTED_SIGNATURES: dict[str, str] = {
    "run": (
        "(pipeline: list[typing.Any] | dict[str, typing.Any] | str | pathlib.Path | "
        "nirs4all.pipeline.config.pipeline_config.PipelineConfigs | list[list[typing.Any] | "
        "dict[str, typing.Any] | str | pathlib.Path | "
        "nirs4all.pipeline.config.pipeline_config.PipelineConfigs], "
        "dataset: str | pathlib.Path | numpy.ndarray | tuple[numpy.ndarray, ...] | "
        "dict[str, typing.Any] | nirs4all.data.dataset.SpectroDataset | "
        "nirs4all.data.config.DatasetConfigs | list[str | pathlib.Path | numpy.ndarray | "
        "tuple[numpy.ndarray, ...] | dict[str, typing.Any] | "
        "nirs4all.data.dataset.SpectroDataset | nirs4all.data.config.DatasetConfigs], *, "
        "name: str = '', session: nirs4all.api.session.Session | None = None, "
        "verbose: int = 1, save_artifacts: bool = True, save_charts: bool = True, "
        "plots_visible: bool = False, random_state: int | None = None, "
        "refit: bool | dict[str, typing.Any] | list[dict[str, typing.Any]] | None = True, "
        "cache: typing.Any | None = None, project: str | None = None, "
        "report_naming: str = 'nirs', engine: str | None = None, "
        "results_path: str | pathlib.Path | None = None, **runner_kwargs: Any) -> "
        "nirs4all.api.result.RunResult"
    ),
    "predict": (
        "(model: 'ModelSpec | None' = None, data: 'DataSpec | None' = None, *, "
        "chain_id: 'str | None' = None, workspace_path: 'str | Path | None' = None, "
        "name: 'str' = 'prediction_dataset', all_predictions: 'bool' = False, "
        "session: 'Session | None' = None, verbose: 'int' = 0, "
        "**runner_kwargs: 'Any') -> 'PredictResult'"
    ),
    "explain": (
        "(model: dict[str, typing.Any] | str | pathlib.Path, "
        "data: str | pathlib.Path | numpy.ndarray | dict[str, typing.Any] | "
        "nirs4all.data.dataset.SpectroDataset | nirs4all.data.config.DatasetConfigs, *, "
        "name: str = 'explain_dataset', "
        "session: nirs4all.api.session.Session | None = None, verbose: int = 1, "
        "plots_visible: bool = True, n_samples: int | None = None, "
        "explainer_type: str = 'auto', **shap_params: Any) -> "
        "nirs4all.api.result.ExplainResult"
    ),
    "retrain": (
        "(source: dict[str, typing.Any] | str | pathlib.Path, "
        "data: str | pathlib.Path | numpy.ndarray | tuple[numpy.ndarray, ...] | "
        "dict[str, typing.Any] | nirs4all.data.dataset.SpectroDataset | "
        "nirs4all.data.config.DatasetConfigs, *, mode: str = 'full', "
        "name: str = 'retrain_dataset', new_model: typing.Any | None = None, "
        "epochs: int | None = None, "
        "session: nirs4all.api.session.Session | None = None, verbose: int = 1, "
        "save_artifacts: bool = True, **kwargs: Any) -> nirs4all.api.result.RunResult"
    ),
    "session": (
        "(pipeline: list[typing.Any] | None = None, name: str = '', **kwargs: Any) -> "
        "collections.abc.Generator[nirs4all.api.session.Session, None, None]"
    ),
    "load_session": (
        "(path: str | pathlib.Path) -> nirs4all.api.session.Session"
    ),
    "generate": (
        "(n_samples: 'int' = 1000, *, random_state: 'int | None' = None, "
        "complexity: \"Literal['simple', 'realistic', 'complex']\" = 'simple', "
        "wavelength_range: 'tuple[float, float] | None' = None, "
        "components: 'list[str] | None' = None, "
        "target_range: 'tuple[float, float] | None' = None, train_ratio: 'float' = 0.8, "
        "as_dataset: 'bool' = True, name: 'str' = 'synthetic_nirs', **kwargs: 'Any') -> "
        "'SpectroDataset | tuple[np.ndarray, np.ndarray]'"
    ),
}

# ---------------------------------------------------------------------------
# Frozen ``__all__`` snapshots (sorted; captured from nirs4all 0.9.1).
# ---------------------------------------------------------------------------

EXPECTED_PACKAGE_ALL: list[str] = [
    "CONTROLLER_REGISTRY",
    "ExplainResult",
    "PipelineConfigs",
    "PipelineRunner",
    "PredictResult",
    "Run",
    "RunConfig",
    "RunResult",
    "RunStatus",
    "Session",
    "explain",
    "framework",
    "generate",
    "generate_run_id",
    "is_gpu_available",
    "is_tensorflow_available",
    "load_session",
    "predict",
    "register_controller",
    "retrain",
    "run",
    "session",
]

EXPECTED_API_ALL: list[str] = [
    "ExplainResult",
    "LazyModelRefitResult",
    "ModelRefitResult",
    "PredictResult",
    "RunResult",
    "Session",
    "explain",
    "generate",
    "load_session",
    "predict",
    "retrain",
    "run",
    "session",
]

# ---------------------------------------------------------------------------
# Frozen public surface of the result classes (captured from nirs4all 0.9.1).
#
# Subset contract: these names MUST remain present in the live class' public
# surface. Adding new public members is allowed and does not fail; removing or
# renaming one of these documented members does fail.
#
# The result classes are dataclasses, so the "live" public surface is
# ``set(dir(cls)) | {f.name for f in dataclasses.fields(cls)}``. ``dir(cls)``
# alone misses public INSTANCE fields that have no class-level default (e.g.
# ``RunResult.predictions`` / ``per_dataset``, ``PredictResult.y_pred``,
# ``ExplainResult.shap_values``); the dataclass-fields union pulls those in so
# they are guarded too.
# ---------------------------------------------------------------------------

EXPECTED_RUNRESULT_MEMBERS: frozenset[str] = frozenset(
    {
        "artifacts_path",
        "best",
        "best_accuracy",
        "best_final",
        "best_r2",
        "best_rmse",
        "best_score",
        "close",
        "cv_best",
        "cv_best_score",
        "detach",
        "export",
        "export_model",
        "filter",
        "final",
        "final_score",
        "get_datasets",
        "get_models",
        "models",
        "num_predictions",
        # dataclass instance fields (missed by dir(cls), recovered via fields())
        "per_dataset",
        "predictions",
        "summary",
        "top",
        "validate",
    }
)

EXPECTED_PREDICTRESULT_MEMBERS: frozenset[str] = frozenset(
    {
        "flatten",
        "explanation_level",
        "feature_lineage",
        "get_feature_lineage",
        "is_multioutput",
        "lineage_warning",
        # dataclass instance fields (missed by dir(cls), recovered via fields())
        "metadata",
        "model_name",
        "preprocessing_steps",
        "relation_materialization_manifest",
        "relation_replay_manifest",
        "sample_indices",
        "shape",
        "to_dataframe",
        "to_list",
        "to_numpy",
        "values",
        "y_pred",
    }
)

EXPECTED_EXPLAINRESULT_MEMBERS: frozenset[str] = frozenset(
    {
        "base_value",
        "explainer_type",
        "explanation_level",
        "feature_names",
        "feature_lineage",
        "get_feature_lineage",
        "get_feature_importance",
        "get_sample_explanation",
        "lineage_warning",
        "mean_abs_shap",
        "model_name",
        "n_samples",
        "shape",
        # dataclass instance fields (missed by dir(cls), recovered via fields())
        "shap_values",
        "to_dataframe",
        "top_features",
        "values",
        "visualizations",
    }
)


def _public_members(cls: type) -> set[str]:
    """Public (non-underscore) attribute/method/field names exposed by a class.

    Unions ``dir(cls)`` with the dataclass field names so public instance
    fields that have no class-level default (and therefore never appear in
    ``dir(cls)``) are included in the live surface.
    """
    names = set(dir(cls))
    if dataclasses.is_dataclass(cls):
        names |= {f.name for f in dataclasses.fields(cls)}
    return {name for name in names if not name.startswith("_")}


# ---------------------------------------------------------------------------
# Signature contracts.
# ---------------------------------------------------------------------------


def test_public_functions_are_importable() -> None:
    """All seven documented entry points are importable from the top package."""
    for name in EXPECTED_SIGNATURES:
        assert hasattr(nirs4all, name), f"nirs4all.{name} is missing"
        assert callable(getattr(nirs4all, name)), f"nirs4all.{name} is not callable"


def _sig(fn: object) -> str:
    """Render a signature for comparison, normalized across interpreter builds.

    Python 3.13 moved ``Path`` into ``pathlib._local``, and some interpreter
    builds leak that internal module in annotation reprs. The public contract is
    the ``pathlib.Path`` spelling, so normalize before comparing.
    """
    return str(inspect.signature(fn)).replace("pathlib._local.Path", "pathlib.Path")  # type: ignore[arg-type]


def test_run_signature_frozen() -> None:
    assert _sig(nirs4all.run) == EXPECTED_SIGNATURES["run"]


def test_predict_signature_frozen() -> None:
    assert _sig(nirs4all.predict) == EXPECTED_SIGNATURES["predict"]


def test_explain_signature_frozen() -> None:
    assert _sig(nirs4all.explain) == EXPECTED_SIGNATURES["explain"]


def test_retrain_signature_frozen() -> None:
    assert _sig(nirs4all.retrain) == EXPECTED_SIGNATURES["retrain"]


def test_session_signature_frozen() -> None:
    assert _sig(nirs4all.session) == EXPECTED_SIGNATURES["session"]


def test_load_session_signature_frozen() -> None:
    assert _sig(nirs4all.load_session) == EXPECTED_SIGNATURES["load_session"]


def test_generate_signature_frozen() -> None:
    assert _sig(nirs4all.generate) == EXPECTED_SIGNATURES["generate"]


# ---------------------------------------------------------------------------
# ``__all__`` contracts.
# ---------------------------------------------------------------------------


def test_package_all_frozen() -> None:
    """``nirs4all.__all__`` matches the frozen export set exactly."""
    assert sorted(nirs4all.__all__) == EXPECTED_PACKAGE_ALL


def test_api_all_frozen() -> None:
    """``nirs4all.api.__all__`` matches the frozen export set exactly."""
    assert sorted(nirs4all_api.__all__) == EXPECTED_API_ALL


# ---------------------------------------------------------------------------
# Result-class public surface contracts (subset / superset semantics).
# ---------------------------------------------------------------------------


def test_runresult_public_surface() -> None:
    """RunResult still exposes every frozen public member."""
    live = _public_members(nirs4all.RunResult)
    missing = EXPECTED_RUNRESULT_MEMBERS - live
    assert not missing, f"RunResult dropped public members: {sorted(missing)}"


def test_predictresult_public_surface() -> None:
    """PredictResult still exposes every frozen public member."""
    live = _public_members(nirs4all.PredictResult)
    missing = EXPECTED_PREDICTRESULT_MEMBERS - live
    assert not missing, f"PredictResult dropped public members: {sorted(missing)}"


def test_explainresult_public_surface() -> None:
    """ExplainResult still exposes every frozen public member."""
    live = _public_members(nirs4all.ExplainResult)
    missing = EXPECTED_EXPLAINRESULT_MEMBERS - live
    assert not missing, f"ExplainResult dropped public members: {sorted(missing)}"
