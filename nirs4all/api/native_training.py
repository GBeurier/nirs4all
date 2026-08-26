"""Explicit public entry point for the first fully-native training lane.

The broad :func:`nirs4all.run` compatibility surface still contains legacy
workflow features that are not expressible by the portable runtime.  This
module deliberately exposes only the proven raw-array lane instead of silently
re-running that workflow through :class:`PipelineRunner` during export.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from nirs4all.pipeline.dagml.estimator import DagMLPipelineEstimator
from nirs4all.pipeline.dagml.methods_runtime import resolve_methods_library_path
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError
from nirs4all.pipeline.dagml.native_conformal_calibration import (
    NativeConformalCalibrationError,
    compile_methods_conformal_calibration_replay,
)
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
    methods_hpo_operation: Mapping[str, Any] | None = None,
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
            methods_hpo_operation=methods_hpo_operation,
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
        raise DagMLNativeCoverageError("native DAG-ML training did not return a replayable portable Methods Package V2") from error
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
    tuning: Any = None,
    calibration: Any = None,
) -> NativeMethodsRunResult:
    """Run the verified public Methods training subset without a legacy runner.

    The accepted dataset is exactly ``{X, y, sample_ids}`` with optional
    ``groups`` and ``metadata``.  Charts, workspaces, sessions, cache, tuning,
    and non-native refit policies remain explicit capability refusals rather
    than becoming ignored legacy arguments.

    ``tuning`` is deliberately a separate, strict native operation rather
    than the older Python objective adapter.  Its V1 shape is
    ``{"engine": "methods-hpo", "trials": N}``, optionally selecting the
    attested ``random``/``tpe`` sampler and ``none``/``median`` pruner.  DAG-ML
    owns all trial execution, SELECT, native incumbent attestation, and the
    single selected rerun/refit.
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
    methods_hpo_operation = _native_methods_hpo_operation(
        tuning,
        seed=12345 if random_state is None else random_state,
    )
    calibration_operation = _native_conformal_calibration_operation(calibration)

    fit_kwargs: dict[str, Any] = {
        "sample_ids": dataset["sample_ids"],
        "groups": dataset.get("groups"),
        "metadata": dataset.get("metadata"),
        "seed": 12345 if random_state is None else random_state,
    }
    if methods_hpo_operation is not None:
        fit_kwargs["methods_hpo_operation"] = methods_hpo_operation
    estimator = fit_native_pipeline(pipeline, dataset["X"], dataset["y"], **fit_kwargs)
    if calibration_operation is not None:
        _attach_native_conformal_calibration(estimator, calibration_operation)
    return NativeMethodsRunResult.from_estimator(
        estimator,
        dataset_name=name or "native",
        model_name=_native_model_name(pipeline),
    )


_NATIVE_METHODS_HPO_SAMPLERS = frozenset({"random", "tpe"})
_NATIVE_METHODS_HPO_PRUNERS = frozenset({"none", "median"})
_NATIVE_CONFORMAL_POLICIES = frozenset({"marginal", "joint_max"})
_NATIVE_CONFORMAL_SMALL_SAMPLE_POLICIES = frozenset({"error", "unbounded"})


def _native_conformal_calibration_operation(calibration: Any) -> dict[str, Any] | None:
    """Parse the explicit raw-array conformal cohort accepted by native run.

    This does not accept the broad legacy/tuning calibration aliases.  The
    caller supplies the actual calibration measurements and stable sample
    identities; the native DAG-ML coordinator derives every fingerprint,
    residual and quantile from the resulting PREDICT replay.
    """

    if calibration is None:
        return None
    if not isinstance(calibration, Mapping):
        raise TypeError("engine='native' calibration must be a mapping")
    payload = dict(calibration)
    allowed = {
        "X",
        "y",
        "sample_ids",
        "groups",
        "metadata",
        "coverages",
        "multi_target_policy",
        "small_sample_policy",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"engine='native' calibration has unsupported keys: {sorted(unknown)}")
    missing = {"X", "y", "sample_ids", "coverages"} - set(payload)
    if missing:
        raise ValueError(f"engine='native' calibration is missing required keys: {sorted(missing)}")
    coverages = payload["coverages"]
    if isinstance(coverages, (str, bytes)) or not isinstance(coverages, Sequence):
        raise TypeError("engine='native' calibration coverages must be a non-empty sequence")
    normalized_coverages: list[float] = []
    for coverage in coverages:
        if isinstance(coverage, bool) or not isinstance(coverage, int | float):
            raise TypeError("engine='native' calibration coverages must be finite numbers")
        numeric_coverage = float(coverage)
        if not np.isfinite(numeric_coverage) or not 0.0 < numeric_coverage < 1.0:
            raise ValueError("engine='native' calibration coverages must lie strictly between zero and one")
        normalized_coverages.append(numeric_coverage)
    if not normalized_coverages or len(set(normalized_coverages)) != len(normalized_coverages):
        raise ValueError("engine='native' calibration coverages must be non-empty and unique")
    multi_target_policy = payload.get("multi_target_policy", "marginal")
    if multi_target_policy not in _NATIVE_CONFORMAL_POLICIES:
        raise ValueError(f"engine='native' calibration multi_target_policy is unsupported: {multi_target_policy!r}")
    small_sample_policy = payload.get("small_sample_policy", "error")
    if small_sample_policy not in _NATIVE_CONFORMAL_SMALL_SAMPLE_POLICIES:
        raise ValueError(f"engine='native' calibration small_sample_policy is unsupported: {small_sample_policy!r}")
    return {
        "X": payload["X"],
        "y": payload["y"],
        "sample_ids": payload["sample_ids"],
        "groups": payload.get("groups"),
        "metadata": payload.get("metadata"),
        "coverages": normalized_coverages,
        "multi_target_policy": multi_target_policy,
        "small_sample_policy": small_sample_policy,
    }


def _attach_native_conformal_calibration(
    estimator: DagMLPipelineEstimator,
    calibration: Mapping[str, Any],
) -> None:
    """Attach a native calibration replay and refresh the portable package.

    The replay is compiled from the package emitted by the just-completed fit.
    Its target-free PREDICT result and separately identity-bound truth cross
    the Python boundary unchanged.  Only DAG-ML may derive calibration
    provenance, residuals or interval quantiles.
    """

    package = estimator.predictor_package_
    training_result = getattr(estimator, "training_result_", None)
    if package is None or training_result is None:
        raise DagMLNativeCoverageError("native conformal calibration requires a fitted portable Methods Package V2")
    try:
        replay = compile_methods_conformal_calibration_replay(
            package,
            calibration["X"],
            calibration["y"],
            sample_ids=calibration["sample_ids"],
            groups=calibration.get("groups"),
            metadata=calibration.get("metadata"),
            methods_library_path=getattr(getattr(estimator, "prediction_compiler", None), "methods_library_path", None),
            dagml_module=estimator.dagml_module,
        )
    except (KeyError, NativeConformalCalibrationError, TypeError, ValueError) as error:
        raise DagMLNativeCoverageError("native conformal calibration cohort is not replayable") from error
    replay_outcome = estimator.execute_compiled_replay(replay.execution)
    attach = getattr(training_result, "attach_conformal_calibration", None)
    export = getattr(training_result, "export_portable_predictor_package", None)
    if not callable(attach) or not callable(export):
        raise DagMLNativeCoverageError("installed DAG-ML lacks native conformal calibration attachment")
    attach(
        replay_outcome,
        binding_id=replay.binding_id,
        calibration_relations=replay.calibration_relations,
        truth=replay.truth,
        coverages=calibration["coverages"],
        # The released DAG-ML facade accepts object values directly but treats
        # scalar strings as pre-serialized TCV1 JSON contracts.
        multi_target_policy=json.dumps(calibration["multi_target_policy"]),
        small_sample_policy=json.dumps(calibration["small_sample_policy"]),
    )
    package_id = _portable_package_id(package)
    estimator.training_outcome_ = getattr(training_result, "outcome", None)
    estimator.predictor_package_ = export(
        package_id,
        fitted_artifact_mode="portable_required",
        artifact_load_mode="native_portable",
    )
    try:
        validate_native_methods_package(estimator.predictor_package_)
    except RawArrayMethodsReplayError as error:
        raise DagMLNativeCoverageError("native conformal calibration did not re-export a portable Methods Package V2") from error


def _portable_package_id(package: Any) -> str:
    """Read the durable Package V2 id without fabricating an export identity."""

    document = package.to_dict() if hasattr(package, "to_dict") else package
    if not isinstance(document, Mapping):
        raise DagMLNativeCoverageError("native conformal calibration package is not a structured contract")
    package_id = document.get("package_id")
    if not isinstance(package_id, str) or not package_id:
        raise DagMLNativeCoverageError("native conformal calibration package has no stable package_id")
    return package_id


def _native_methods_hpo_operation(tuning: Any, *, seed: int) -> dict[str, Any] | None:
    """Parse the public, deliberately narrow Methods scheduler operation.

    This parser does not accept generic nirs4all tuning settings such as
    ``space``, ``score_data``, callbacks, or workspace resume fields: those
    belong to the historical Python objective path.  The V1 native search
    space is attested by DAG-ML and intentionally fixes PLS ``n_components``
    to the portable 1..=3 integer domain.  TPE and the Median pruner remain
    controller-owned: the public payload cannot provide an objective,
    intermediate scores, callbacks, or optimiser-specific knobs.  A two-trial
    startup budget is fixed whenever either needs historical observations.
    """

    if tuning is None:
        return None
    if not isinstance(tuning, Mapping):
        raise TypeError("engine='native' tuning must be a mapping")
    payload = dict(tuning)
    allowed = {"engine", "trials", "sampler", "pruner"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"engine='native' tuning has unsupported keys: {sorted(unknown)}")
    if payload.get("engine") != "methods-hpo":
        raise ValueError("engine='native' tuning requires engine='methods-hpo'")
    trials = payload.get("trials")
    if isinstance(trials, bool) or not isinstance(trials, int) or not 1 <= trials <= 64:
        raise ValueError("engine='native' Methods HPO trials must be an integer in 1..64")
    sampler = payload.get("sampler", "random")
    pruner = payload.get("pruner", "none")
    if sampler not in _NATIVE_METHODS_HPO_SAMPLERS:
        raise ValueError(f"engine='native' Methods HPO sampler is unsupported: {sampler!r}")
    if pruner not in _NATIVE_METHODS_HPO_PRUNERS:
        raise ValueError(f"engine='native' Methods HPO pruner is unsupported: {pruner!r}")
    n_startup_trials = 2 if sampler == "tpe" or pruner == "median" else 0
    return {
        "operation_id": "hpo:methods",
        "study": {
            "controller_id": "controller:methods.hpo",
            "study_id": "study:nirs4all.native.pls",
            "methods_abi": "n4m-abi-2.2",
            "search_space": {
                "parameters": [
                    {
                        "kind": "int",
                        "name": "n_components",
                        "low": 1,
                        "high": 3,
                        "step": 1,
                        "log": False,
                    }
                ]
            },
            "optimizer": {
                "sampler": sampler,
                "pruner": pruner,
                "direction": "minimize",
                "metric": "rmse",
                "seed": seed,
                "n_startup_trials": n_startup_trials,
                "max_resource": 0,
                "reduction_factor": 0,
            },
        },
        "trials": trials,
        "parameter_paths": {"n_components": "n_components"},
    }


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
