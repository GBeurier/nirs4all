"""Explicit public entry point for the first fully-native training lane.

The broad :func:`nirs4all.run` compatibility surface still contains legacy
workflow features that are not expressible by the portable runtime.  This
module deliberately exposes only the proven raw-array lane instead of silently
re-running that workflow through :class:`PipelineRunner` during export.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from nirs4all.pipeline.dagml.estimator import DagMLPipelineEstimator
from nirs4all.pipeline.dagml.methods_runtime import resolve_methods_library_path
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError
from nirs4all.pipeline.dagml.raw_replay_lowerer import (
    RawArrayMethodsReplayCompiler,
    RawArrayMethodsReplayError,
    validate_native_methods_package,
)
from nirs4all.pipeline.dagml.raw_training_lowerer import RawArrayDagMLTrainingCompiler

from .native_result import NativeMethodsRunResult


def fit_native_pipeline(
    pipeline: list[Any],
    X: Any,
    y: Any,
    *,
    sample_ids: Sequence[str],
    groups: Sequence[Any] | None = None,
    metadata: Mapping[str, Sequence[Any]] | None = None,
    selection_metric: str = "rmse",
    selection_objective: str = "minimize",
    package_id: str | None = None,
    dagml_module: str = "dag_ml",
    native_client: Any = None,
    training_losses: tuple[Mapping[str, Any], ...] = (),
    local_implementations: Any = None,
    methods_library_path: str | None = None,
    seed: int = 12345,
) -> DagMLPipelineEstimator:
    """Fit the supported raw-array pipeline entirely through DAG-ML.

    This is an intentionally narrow native API: a linear list pipeline with
    one splitter and one supported model, finite 2-D ``X``, and finite targets.
    ``sample_ids`` are mandatory because the resulting Package V2 and Archive
    V2 require stable identities for every later PREDICT cohort.

    The installed DAG-ML runtime must produce a ``portable_required`` Methods
    Package V2 with one durable N4MM refit artifact.  Host-sidecar/joblib
    results are refused before this function returns: they cannot be replayed
    or exported as a native archive.  A successful estimator retains the exact
    native ``TrainingResult``, ``TrainingOutcome`` and Package V2.  Its
    :meth:`export_native_archive` method writes Archive V2 directly; it never
    invokes the legacy runner or fits the model a second time.  With
    ``nirs4all[native]``, the bundled ``nirs4all-methods`` runtime is discovered
    automatically; ``methods_library_path`` remains an explicit deployment
    override.
    """

    if not isinstance(pipeline, list):
        raise TypeError("fit_native_pipeline requires a list pipeline")
    if sample_ids is None:
        raise ValueError("fit_native_pipeline requires explicit sample_ids")
    if not isinstance(dagml_module, str) or not dagml_module:
        raise ValueError("dagml_module must be a non-empty string")

    # Fail before assembling the bridge when a caller passed an obviously
    # non-portable matrix.  The lowerer repeats contract-level checks and
    # DAG-ML remains authoritative for execution semantics.
    features = np.asarray(X)
    targets = np.asarray(y)
    if features.ndim != 2 or targets.ndim not in (1, 2):
        raise ValueError("fit_native_pipeline requires 2-D X and 1-D or 2-D y")
    if features.shape[0] == 0 or features.shape[0] != targets.shape[0]:
        raise ValueError("fit_native_pipeline requires aligned non-empty X and y")
    if len(sample_ids) != features.shape[0]:
        raise ValueError("fit_native_pipeline requires sample_ids length to match X")
    if not np.issubdtype(features.dtype, np.number) or not np.issubdtype(targets.dtype, np.number):
        raise TypeError("fit_native_pipeline requires numeric X and y")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("fit_native_pipeline requires finite X and y")
    resolved_methods_library_path = resolve_methods_library_path(methods_library_path)

    estimator = DagMLPipelineEstimator(
        pipeline=pipeline,
        selection_output_id="output:prediction",
        package_id=package_id,
        dagml_module=dagml_module,
        native_client=native_client,
        training_compiler=RawArrayDagMLTrainingCompiler(
            selection_metric=selection_metric,
            selection_objective=selection_objective,
            dagml_module=dagml_module,
            training_losses=training_losses,
            local_implementations=local_implementations,
            methods_library_path=resolved_methods_library_path,
            seed=seed,
        ),
        require_explicit_sample_ids=True,
    )
    estimator.fit(X, y, sample_ids=sample_ids, groups=groups, metadata=metadata)
    if estimator.predictor_package_ is None:
        raise DagMLNativeCoverageError("native DAG-ML training did not return an exportable Package V2")
    try:
        validate_native_methods_package(estimator.predictor_package_)
    except RawArrayMethodsReplayError as error:
        raise DagMLNativeCoverageError(
            "native DAG-ML training did not return a replayable portable Methods Package V2"
        ) from error
    estimator.prediction_compiler = RawArrayMethodsReplayCompiler(
        estimator.predictor_package_,
        dagml_module=dagml_module,
        methods_library_path=resolved_methods_library_path,
    )
    estimator.prediction_identity_decoder = _decode_raw_methods_prediction
    return estimator


def run_native_methods(
    pipeline: list[Any],
    dataset: Mapping[str, Any],
    *,
    name: str = "",
    session: Any = None,
    save_artifacts: bool = True,
    save_charts: bool = False,
    plots_visible: bool = False,
    random_state: int | None = None,
    refit: bool | None = True,
    cache: Any = None,
    project: str | None = None,
    report_naming: str = "nirs",
    results_path: Any = None,
    runner_kwargs: Mapping[str, Any] | None = None,
) -> NativeMethodsRunResult:
    """Run the verified public Methods training subset without a legacy runner.

    The accepted dataset is exactly ``{X, y, sample_ids}`` with optional
    ``groups`` and ``metadata``.  Charts, workspaces, sessions, cache, tuning,
    and non-native refit policies remain explicit capability refusals rather
    than becoming ignored legacy arguments.
    """

    if not isinstance(dataset, Mapping):
        raise TypeError("engine='native' requires dataset={'X': matrix, 'y': targets, 'sample_ids': explicit_ids}")
    unknown = set(dataset) - {"X", "y", "sample_ids", "groups", "metadata"}
    if unknown:
        raise ValueError(f"engine='native' dataset has unsupported keys: {sorted(unknown)}")
    missing = {"X", "y", "sample_ids"} - set(dataset)
    if missing:
        raise ValueError(f"engine='native' dataset is missing required keys: {sorted(missing)}")
    if session is not None:
        raise NotImplementedError("engine='native' training sessions are not available yet; use the stateless native run subset")
    if not save_artifacts:
        raise ValueError("engine='native' requires save_artifacts=True to retain its portable N4MM artifact")
    if save_charts:
        raise NotImplementedError("engine='native' does not yet produce legacy charts; pass save_charts=False")
    if plots_visible:
        raise NotImplementedError("engine='native' does not yet produce interactive legacy plots")
    if refit is not True:
        raise NotImplementedError("engine='native' currently requires refit=True")
    if cache is not None or project is not None or results_path is not None:
        raise NotImplementedError("engine='native' does not yet support cache, project, or results_path")
    if report_naming != "nirs":
        raise NotImplementedError("engine='native' currently supports only report_naming='nirs'")
    if runner_kwargs:
        raise NotImplementedError(f"engine='native' does not accept legacy runner kwargs: {sorted(runner_kwargs)}")
    if random_state is not None and (isinstance(random_state, bool) or not isinstance(random_state, int)):
        raise TypeError("engine='native' random_state must be an integer or None")

    estimator = fit_native_pipeline(
        pipeline,
        dataset["X"],
        dataset["y"],
        sample_ids=dataset["sample_ids"],
        groups=dataset.get("groups"),
        metadata=dataset.get("metadata"),
        seed=12345 if random_state is None else random_state,
    )
    return NativeMethodsRunResult.from_estimator(
        estimator,
        dataset_name=name or "native",
        model_name=_native_model_name(pipeline),
    )


def _native_model_name(pipeline: list[Any]) -> str:
    """Return a display name without changing native model selection semantics."""

    if pipeline:
        final = pipeline[-1]
        if isinstance(final, Mapping) and set(final) == {"model"}:
            final = final["model"]
        name = getattr(final, "__name__", None) or type(final).__name__
        if isinstance(name, str) and name:
            return name
    return "MethodsModel"


def _decode_raw_methods_prediction(outcome: Any, identity_frame: Any) -> np.ndarray:
    """Decode a replay only after checking its exact current sample ordering."""

    document = outcome.to_dict() if hasattr(outcome, "to_dict") else outcome
    if not isinstance(document, Mapping):
        raise DagMLNativeCoverageError("native replay did not return an outcome object")
    outputs = document.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], Mapping):
        raise DagMLNativeCoverageError("native raw replay requires exactly one output")
    blocks = outputs[0].get("predictions")
    if not isinstance(blocks, list) or len(blocks) != 1 or not isinstance(blocks[0], Mapping):
        raise DagMLNativeCoverageError("native raw replay requires exactly one prediction block")
    block = blocks[0]
    if block.get("sample_ids") != list(identity_frame.sample_ids):
        raise DagMLNativeCoverageError("native replay prediction identities do not exactly match the current cohort")
    try:
        values = np.asarray(block.get("values"), dtype=float)
    except (TypeError, ValueError) as error:
        raise DagMLNativeCoverageError("native replay prediction values are not numeric") from error
    if values.ndim != 2 or values.shape[0] != identity_frame.n_samples or not np.isfinite(values).all():
        raise DagMLNativeCoverageError("native replay prediction values are not a finite aligned matrix")
    return cast(np.ndarray, values)


__all__ = ["fit_native_pipeline", "run_native_methods"]
