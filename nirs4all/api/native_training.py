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
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError
from nirs4all.pipeline.dagml.raw_replay_lowerer import (
    RawArrayMethodsReplayCompiler,
    RawArrayMethodsReplayError,
    validate_native_methods_package,
)
from nirs4all.pipeline.dagml.raw_training_lowerer import RawArrayDagMLTrainingCompiler


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
    invokes the legacy runner or fits the model a second time.
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
    if not np.issubdtype(features.dtype, np.number) or not np.issubdtype(targets.dtype, np.number):
        raise TypeError("fit_native_pipeline requires numeric X and y")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("fit_native_pipeline requires finite X and y")

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
            methods_library_path=methods_library_path,
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
        methods_library_path=methods_library_path,
    )
    estimator.prediction_identity_decoder = _decode_raw_methods_prediction
    return estimator


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


__all__ = ["fit_native_pipeline"]
