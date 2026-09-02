"""Callback-free Methods training that closes directly into Core Archive V2.

This module is intentionally limited to the portable V1 minimum: explicit raw
arrays, one KFold-like splitter and one sklearn ``PLSRegression`` declaration.
Single- and multi-target regression are supported when target identities are
explicit.
DAG-ML owns scheduling and archive-member assembly, Methods owns fit/predict and
N4MM bytes, and Core alone writes and validates the ``.n4a`` container.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from nirs4all.pipeline.dagml.core_archive_replay import _resolve_methods_library_path
from nirs4all.pipeline.dagml.fit_identity import normalize_fit_identity
from nirs4all.pipeline.dagml.raw_training_lowerer import lower_raw_array_training_contracts
from nirs4all.pipeline.dagml.result import _scores_to_run_result

from .result import RunResult


class NativeArchiveTrainingError(RuntimeError):
    """The installed native stack cannot produce a portable Archive V2."""


class NativeMethodsHpoCapabilityError(NativeArchiveTrainingError, ValueError):
    """The public HPO request exceeds the selected portable Methods slice."""


@dataclass(frozen=True)
class _PortableMethodsHpo:
    """Closed public projection onto DAG-ML's controller-owned Methods HPO."""

    trials: int
    seed: int
    low: int
    high: int
    step: int


class NativeMethodsArchiveRunResult(RunResult):
    """A normal score projection retaining only portable Methods contracts."""

    def __init__(
        self,
        projected: RunResult,
        *,
        dag_ml: Any,
        core: Any,
        training_result: Any,
        outcome: Mapping[str, Any],
        package: Mapping[str, Any],
        outcome_contract: Any,
        package_contract: Any,
        archive_id: str,
    ) -> None:
        super().__init__(
            predictions=projected.predictions,
            per_dataset={dataset_name: {**info, "engine": "native"} for dataset_name, info in projected.per_dataset.items()},
        )
        self._dagml_score_set = dict(outcome["score_set"])
        self._native_dag_ml = dag_ml
        self._native_core = core
        self._native_training_result = training_result
        self._native_outcome = dict(outcome)
        self._native_package = dict(package)
        self._native_outcome_contract = outcome_contract
        self._native_package_contract = package_contract
        self._native_archive_id = archive_id
        self._native_archive_reference: dict[str, str] | None = None

    @property
    def native_archive_reference(self) -> dict[str, str] | None:
        """Return the Core-issued id/checksum of the last successful save."""

        if self._native_archive_reference is None:
            return None
        return dict(self._native_archive_reference)

    @property
    def native_execution_is_live(self) -> bool:
        """Whether the native training resources have not yet been detached."""

        return bool(getattr(self._native_training_result, "is_attached", False))

    @property
    def tuning_best_params(self) -> dict[str, Any]:
        """Return the native Methods incumbent without consulting Python HPO."""

        state = self._native_outcome.get("methods_hpo_resume_state")
        if not isinstance(state, Mapping):
            fallback: dict[str, Any] = super().tuning_best_params
            return fallback
        incumbent = state.get("incumbent")
        terminals = state.get("terminal_trials")
        if not isinstance(incumbent, Mapping) or not isinstance(terminals, Sequence):
            raise NativeArchiveTrainingError("native Methods HPO outcome omitted incumbent evidence")
        trial_id = incumbent.get("trial_id")
        for terminal in terminals:
            if not isinstance(terminal, Mapping) or not isinstance(terminal.get("trial"), Mapping):
                continue
            trial = terminal["trial"]
            if trial.get("id") != trial_id:
                continue
            parameters = trial.get("parameters")
            if not isinstance(parameters, Mapping) or set(parameters) != {"n_components"}:
                break
            parameter = parameters["n_components"]
            if not isinstance(parameter, Mapping):
                break
            value = parameter.get("value")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value).is_integer():
                return {"model.n_components": int(value)}
            break
        raise NativeArchiveTrainingError("native Methods HPO outcome cannot identify its incumbent parameters")

    @property
    def tuning_best_value(self) -> float | None:
        """Return the scheduler-checked native Methods incumbent score."""

        state = self._native_outcome.get("methods_hpo_resume_state")
        if not isinstance(state, Mapping):
            fallback: float | None = super().tuning_best_value
            return fallback
        incumbent = state.get("incumbent")
        score = incumbent.get("score") if isinstance(incumbent, Mapping) else None
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not np.isfinite(score):
            raise NativeArchiveTrainingError("native Methods HPO outcome omitted a finite incumbent score")
        return float(score)

    def export(
        self,
        output_path: str | Path,
        format: str = "n4a",
        source: dict[str, Any] | None = None,
        chain_id: str | None = None,
        *,
        compatibility: str | None = None,
    ) -> Path:
        """Write the captured Package V2/N4MM without refitting or host models."""

        if format != "n4a":
            raise ValueError("native Methods export supports only format='n4a' (Core Archive V2)")
        if source is not None or chain_id is not None or compatibility is not None:
            raise NotImplementedError("native Methods Archive V2 export rejects legacy selectors and compatibility refits")
        path = Path(output_path)
        if path.suffix.lower() != ".n4a":
            raise ValueError("native Methods export requires a .n4a path")
        try:
            manifest, members = self._native_dag_ml.build_archive_v2_native_portable_payloads(
                self._native_archive_id,
                self._native_outcome_contract,
                self._native_package_contract,
            )
            reference = self._native_core.write_archive_v2_from_native_payloads(
                path,
                manifest,
                members,
            )
        except Exception as error:
            raise NativeArchiveTrainingError("DAG-ML/Core refused native Archive V2 publication") from error
        if not isinstance(reference, Mapping):
            raise NativeArchiveTrainingError("Core Archive V2 writer returned an invalid reference")
        archive_id = reference.get("archive_id")
        archive_sha256 = reference.get("archive_sha256")
        if not isinstance(archive_id, str) or not isinstance(archive_sha256, str):
            raise NativeArchiveTrainingError("Core Archive V2 writer omitted id/checksum evidence")
        self._native_archive_reference = {
            "archive_id": archive_id,
            "archive_sha256": archive_sha256,
        }
        return path

    def close(self) -> None:
        """Release native handles while retaining immutable package/archive bytes."""

        if getattr(self._native_training_result, "is_attached", False):
            detach = getattr(self._native_training_result, "detach", None)
            if callable(detach):
                detach()


def run_native_methods_archive(
    pipeline: Any,
    dataset: Any,
    *,
    name: str = "",
    verbose: int = 1,
    save_artifacts: bool = True,
    save_charts: bool = False,
    plots_visible: bool = False,
    random_state: int | None = None,
    refit: Any = True,
    cache: Any = None,
    project: str | None = None,
    report_naming: str = "nirs",
    results_path: Any = None,
    session: Any = None,
    runner_kwargs: Mapping[str, Any] | None = None,
) -> NativeMethodsArchiveRunResult:
    """Run the frozen portable Methods subset without ``PipelineRunner``."""

    if not isinstance(pipeline, list):
        raise TypeError("engine='native' requires a list pipeline")
    if not isinstance(dataset, Mapping):
        raise TypeError("engine='native' requires dataset={'X': matrix, 'y': targets, 'sample_ids': ids, 'target_names': optional_names}")
    unknown = set(dataset) - {
        "X",
        "y",
        "sample_ids",
        "target_names",
        "groups",
        "metadata",
    }
    missing = {"X", "y", "sample_ids"} - set(dataset)
    if unknown:
        raise ValueError(f"engine='native' dataset has unsupported keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"engine='native' dataset is missing required keys: {sorted(missing)}")
    if not save_artifacts:
        raise ValueError("engine='native' requires save_artifacts=True for its N4MM artifact")
    if verbose != 1 or save_charts or plots_visible:
        raise NotImplementedError("engine='native' does not expose legacy progress or chart controls")
    if refit is not True:
        raise NotImplementedError("engine='native' requires the native refit")
    if cache is not None or project is not None or results_path is not None:
        raise NotImplementedError("engine='native' does not use legacy cache, project, or results_path")
    native_session = None
    if session is not None:
        from nirs4all.api.session import Session

        if not isinstance(session, Session):
            raise TypeError("engine='native' session must be a nirs4all.Session")
        native_session = session
    if report_naming != "nirs":
        raise NotImplementedError("engine='native' supports report_naming='nirs' only")
    native_options = dict(runner_kwargs or {})
    methods_library_override = native_options.pop("methods_library_path", None)
    if native_options:
        raise NotImplementedError(f"engine='native' rejects legacy runner options: {sorted(native_options)}")
    if random_state is not None and (isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0):
        raise TypeError("engine='native' random_state must be a non-negative integer or None")

    seed = 12345 if random_state is None else random_state
    portable_pipeline, hpo = _extract_portable_methods_hpo(
        pipeline,
        seed=seed,
    )
    features, targets, sample_ids, target_names = _normalize_training_arrays(dataset)
    if hpo is not None and features.shape[1] < 3:
        raise NativeMethodsHpoCapabilityError("engine='native' portable Methods HPO v1 evaluates n_components=1..3 and therefore requires X to contain at least 3 features")
    identity = normalize_fit_identity(
        features,
        targets,
        sample_ids=sample_ids,
        groups=dataset.get("groups"),
        metadata=dataset.get("metadata"),
        require_explicit_sample_ids=True,
    )
    if native_session is not None:
        native_session._prepare_native_run()
    dag_ml, core = _require_archive_runtime()
    methods_library_path = _resolve_methods_library_path(methods_library_override)
    contracts = lower_raw_array_training_contracts(
        portable_pipeline,
        features,
        targets,
        identity_frame=identity,
        seed=seed,
        portable_methods=True,
        target_names=target_names,
    )
    if hpo is not None:
        contracts = _attach_portable_methods_hpo(contracts, hpo)
    prepared = contracts.to_prepared()
    requirement_keys = sorted(prepared.data_envelopes)
    if len(requirement_keys) != 1:
        raise NativeArchiveTrainingError("portable Methods PLS requires exactly one signed data requirement")
    target_matrix = targets.reshape(-1, 1) if targets.ndim == 1 else targets
    methods_inputs = {
        requirement_keys[0]: {
            "sample_ids": list(identity.sample_ids),
            "x": features.tolist(),
            "y": target_matrix.tolist(),
            "target_names": list(target_names),
        }
    }
    request = dag_ml.sign_training_request(prepared.request)
    training_result: Any | None = None
    try:
        training_result = dag_ml.execute_methods_training(
            request,
            prepared.data_envelopes,
            prepared.relations,
            prepared.training_influence,
            methods_inputs,
            methods_library_path=methods_library_path,
            outcome_id=prepared.outcome_id,
            run_id=prepared.run_id,
            bundle_id=prepared.bundle_id,
            warnings=prepared.warnings,
            diagnostics={
                **dict(prepared.diagnostics or {}),
                "nirs4all_execution": ("methods_controller_owned_hpo_archive_v2" if hpo is not None else "methods_callback_free_archive_v2"),
            },
        )
        package_object = training_result.export_portable_predictor_package(
            f"package:{prepared.outcome_id}",
            fitted_artifact_mode="portable_required",
            artifact_load_mode="native_portable",
        )
        outcome_object = training_result.outcome
        outcome = _to_mapping(outcome_object, "TrainingOutcome")
        package = _to_mapping(package_object, "Package V2")
        score_set = outcome.get("score_set")
        fingerprint = outcome.get("outcome_fingerprint")
        if not isinstance(score_set, Mapping) or not isinstance(fingerprint, str) or not fingerprint:
            raise NativeArchiveTrainingError("native Methods training omitted score/archive identity evidence")
        projected = _scores_to_run_result(
            dict(score_set),
            name or "native",
            _native_model_name(pipeline),
            "rmse",
            "regression",
        )
        result = NativeMethodsArchiveRunResult(
            projected,
            dag_ml=dag_ml,
            core=core,
            training_result=training_result,
            outcome=outcome,
            package=package,
            outcome_contract=outcome_object,
            package_contract=package_object,
            archive_id=f"archive:{fingerprint}",
        )
        if native_session is not None:
            native_session._adopt_native_result(result, dataset)
        return result
    except BaseException:
        if training_result is not None and getattr(training_result, "is_attached", False):
            training_result.detach()
        raise


def _require_archive_runtime() -> tuple[Any, Any]:
    try:
        dag_ml = importlib.import_module("dag_ml")
        core = importlib.import_module("nirs4all_core")
    except ImportError as error:
        raise NativeArchiveTrainingError("engine='native' requires matching dag-ml and nirs4all-core native wheels") from error
    required_dagml = (
        "compile_pipeline_dsl_artifact_with_controllers",
        "sign_training_request",
        "execute_methods_training",
        "build_archive_v2_native_portable_payloads",
    )
    missing = [name for name in required_dagml if not callable(getattr(dag_ml, name, None))]
    if missing:
        raise NativeArchiveTrainingError("installed dag-ml lacks native Archive V2 producer capabilities: " + ", ".join(missing))
    if not callable(getattr(core, "write_archive_v2_from_native_payloads", None)):
        raise NativeArchiveTrainingError("installed nirs4all-core lacks the Archive V2 writer")
    return dag_ml, core


def _extract_portable_methods_hpo(
    pipeline: list[Any],
    *,
    seed: int,
) -> tuple[list[Any], _PortableMethodsHpo | None]:
    """Extract the first executable API-001 HPO slice from a Methods pipeline.

    Python only validates and translates the public request. DAG-ML owns folds,
    OOF scoring, selection and refit; the official Methods controller owns the
    optimizer state machine.  This first public slice deliberately does not
    expose N4MOPT checkpoint/resume and accepts only the exact search space
    implemented by the selected portable Methods HPO v1 runtime.
    """

    model_steps = [(index, step) for index, step in enumerate(pipeline) if isinstance(step, Mapping) and "finetune_params" in step]
    if not model_steps:
        return list(pipeline), None
    if len(model_steps) != 1:
        raise ValueError("native Methods HPO requires exactly one finetune_params model step")
    index, step = model_steps[0]
    if set(step) != {"model", "finetune_params"}:
        raise ValueError("native Methods HPO model step supports only model and finetune_params")
    params = step["finetune_params"]
    if not isinstance(params, Mapping):
        raise TypeError("native Methods HPO finetune_params must be a mapping")
    allowed = {
        "approach",
        "direction",
        "engine",
        "metric",
        "model_params",
        "n_trials",
        "pruner",
        "sampler",
        "seed",
    }
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"native Methods HPO does not support finetune_params keys {unknown}")
    if str(params.get("engine", "n4m")).strip().lower() not in {"n4m", "methods", "libn4m"}:
        raise ValueError("native Methods HPO requires finetune_params.engine='n4m'")
    if params.get("approach", "grouped") != "grouped":
        raise ValueError("native Methods HPO supports only approach='grouped'")
    if str(params.get("metric", "rmse")).strip().lower() != "rmse":
        raise ValueError("native Methods HPO currently selects only the native OOF RMSE")
    if str(params.get("direction", "minimize")).strip().lower() != "minimize":
        raise ValueError("native Methods HPO RMSE requires direction='minimize'")
    if str(params.get("sampler", "random")).strip().lower() != "random":
        raise ValueError("native Methods HPO first slice supports only sampler='random'")
    if str(params.get("pruner", "none")).strip().lower() != "none":
        raise ValueError("native Methods HPO first slice supports only pruner='none'")

    trials = params.get("n_trials", 20)
    hpo_seed = params.get("seed", seed)
    if not isinstance(trials, int) or isinstance(trials, bool) or not 1 <= trials <= 256:
        raise ValueError("native Methods HPO n_trials must be an integer in 1..256")
    if not isinstance(hpo_seed, int) or isinstance(hpo_seed, bool) or hpo_seed < 0:
        raise ValueError("native Methods HPO seed must be a non-negative integer")
    space = params.get("model_params")
    if not isinstance(space, Mapping) or set(space) != {"n_components"}:
        raise ValueError("native Methods HPO first slice requires exactly model_params.n_components")
    low, high, step_size = _normalize_n_components_space(space["n_components"])
    if (low, high, step_size) != (1, 3, 1):
        raise NativeMethodsHpoCapabilityError(
            "engine='native' portable Methods HPO v1 requires "
            "model_params.n_components exactly ['int', 1, 3, 1] "
            f"(received low={low}, high={high}, step={step_size}); "
            "broader spaces and N4MOPT checkpoint/resume are not exposed by this public slice"
        )

    stripped = list(pipeline)
    stripped[index] = {"model": step["model"]}
    return stripped, _PortableMethodsHpo(
        trials=trials,
        seed=hpo_seed,
        low=low,
        high=high,
        step=step_size,
    )


def _normalize_n_components_space(value: Any) -> tuple[int, int, int]:
    """Normalize only public range spellings; runtime support is checked next."""

    if isinstance(value, Mapping):
        keys = set(value)
        unknown = keys - {"type", "min", "max", "low", "high", "step", "log"}
        bound_keys = keys - {"type", "step", "log"}
        if unknown or bound_keys not in ({"low", "high"}, {"min", "max"}):
            raise NativeMethodsHpoCapabilityError(
                "engine='native' portable Methods HPO v1 requires exactly one complete n_components bound pair: type='int' with low/high or min/max; mixed, duplicated, partial, or unknown aliases are refused"
            )
        if value.get("type") != "int" or value.get("log", False) is not False:
            raise NativeMethodsHpoCapabilityError("engine='native' portable Methods HPO v1 requires one linear integer n_components range")
        if bound_keys == {"low", "high"}:
            low, high = value["low"], value["high"]
        else:
            low, high = value["min"], value["max"]
        step = value.get("step", 1)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) not in (3, 4) or value[0] != "int":
            raise NativeMethodsHpoCapabilityError("engine='native' portable Methods HPO v1 requires n_components=['int', 1, 3, 1]")
        low, high = value[1], value[2]
        step = value[3] if len(value) == 4 else 1
    else:
        raise NativeMethodsHpoCapabilityError("engine='native' portable Methods HPO v1 requires n_components=['int', 1, 3, 1]")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in (low, high, step)):
        raise NativeMethodsHpoCapabilityError("engine='native' portable Methods HPO v1 requires integer n_components bounds and step")
    if low < 1 or high < low or step < 1:
        raise NativeMethodsHpoCapabilityError("engine='native' portable Methods HPO v1 requires n_components bounds 1..=3 with step=1")
    return low, high, step


def _attach_portable_methods_hpo(contracts: Any, hpo: _PortableMethodsHpo) -> Any:
    """Attach the public request to Dag-ML's signed scheduler HPO operation."""

    spec = contracts.request_spec
    graph = copy.deepcopy(dict(spec.graph))
    model_nodes = [node for node in graph.get("nodes", []) if node.get("kind") == "model"]
    if len(model_nodes) != 1:
        raise NativeArchiveTrainingError("native Methods HPO requires exactly one model node")
    target = model_nodes[0]
    target["operator"] = "pls"

    manifests = [copy.deepcopy(dict(manifest)) for manifest in spec.controller_manifests]
    model_manifests = [manifest for manifest in manifests if manifest.get("controller_id") == "controller:methods.pls"]
    if len(model_manifests) != 1:
        raise NativeArchiveTrainingError("native Methods HPO requires the Methods PLS controller")
    tuner = copy.deepcopy(model_manifests[0])
    tuner.update(
        {
            "controller_id": "controller:tuner.methods",
            "operator_kind": "tuner",
            "input_ports": [],
            "output_ports": [],
        }
    )
    manifests.append(tuner)

    campaign = copy.deepcopy(dict(spec.campaign))
    generation = campaign.get("generation")
    if not isinstance(generation, dict) or generation.get("dimensions"):
        raise NativeArchiveTrainingError("native Methods HPO requires one unexpanded base variant")
    metadata = campaign.setdefault("metadata", {})
    if not isinstance(metadata, dict) or "methods_hpo_operation" in metadata:
        raise NativeArchiveTrainingError("native Methods HPO campaign metadata is ambiguous")
    metadata["methods_hpo_operation"] = {
        "operation_id": "hpo:nirs4all.native.methods",
        "study": {
            "controller_id": "controller:tuner.methods",
            "study_id": "study:nirs4all.native.methods",
            "methods_abi": "n4m-abi-2.2",
            "search_space": {
                "parameters": [
                    {
                        "kind": "int",
                        "name": "n_components",
                        "low": hpo.low,
                        "high": hpo.high,
                        "step": hpo.step,
                        "log": False,
                    }
                ]
            },
            "optimizer": {
                "sampler": "random",
                "pruner": "none",
                "direction": "minimize",
                "metric": "rmse",
                "seed": hpo.seed,
                "n_startup_trials": 0,
                "max_resource": 0,
                "reduction_factor": 0,
            },
        },
        "trials": hpo.trials,
        "target_node_id": target["id"],
        "parameter_paths": {"n_components": "n_components"},
    }
    return replace(
        contracts,
        request_spec=replace(
            spec,
            graph=graph,
            campaign=campaign,
            controller_manifests=manifests,
        ),
        diagnostics={
            **dict(contracts.diagnostics or {}),
            "nirs4all_execution": "methods_controller_owned_hpo_archive_v2",
            "nirs4all_methods_hpo_resume": "not_exposed_by_public_api_v1",
        },
    )


def _normalize_training_arrays(
    dataset: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, Sequence[Any], tuple[str, ...]]:
    features = np.ascontiguousarray(np.asarray(dataset["X"], dtype=np.dtype("<f8")))
    targets = np.ascontiguousarray(np.asarray(dataset["y"], dtype=np.dtype("<f8")))
    if features.ndim != 2 or targets.ndim not in (1, 2):
        raise ValueError("engine='native' requires 2-D X and 1-D or 2-D y")
    if features.shape[0] == 0 or features.shape[0] != targets.shape[0]:
        raise ValueError("engine='native' requires aligned non-empty X and y")
    target_width = 1 if targets.ndim == 1 else targets.shape[1]
    if target_width == 0:
        raise ValueError("engine='native' requires at least one target column")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("engine='native' requires finite X and y")
    sample_ids = dataset["sample_ids"]
    if not isinstance(sample_ids, Sequence) or isinstance(sample_ids, (str, bytes)):
        raise TypeError("engine='native' sample_ids must be a sequence")
    target_names = _normalize_target_names(dataset.get("target_names"), target_width)
    return features, targets, sample_ids, target_names


def _normalize_target_names(value: Any, target_width: int) -> tuple[str, ...]:
    if value is None:
        if target_width == 1:
            return ("y",)
        raise ValueError("engine='native' multi-target y requires explicit target_names")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("engine='native' target_names must be a sequence of strings")
    if len(value) != target_width:
        raise ValueError(f"engine='native' target_names length must match y width {target_width}")
    names = tuple(value)
    if not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("engine='native' target_names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("engine='native' target_names must be unique")
    return names


def _to_mapping(value: Any, label: str) -> Mapping[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    if not isinstance(value, Mapping):
        raise NativeArchiveTrainingError(f"native {label} is not a mapping")
    return value


def _native_model_name(pipeline: list[Any]) -> str:
    final = pipeline[-1] if pipeline else None
    if isinstance(final, Mapping) and "model" in final:
        final = final["model"]
    name = getattr(final, "__name__", None) or type(final).__name__
    return name if isinstance(name, str) and name else "MethodsPLS"


__all__ = [
    "NativeArchiveTrainingError",
    "NativeMethodsHpoCapabilityError",
    "NativeMethodsArchiveRunResult",
    "run_native_methods_archive",
]
