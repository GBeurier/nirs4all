"""Optimizer adapters for the native DAG-ML ``PipelineObjective``.

P5 starts here: optimizers drive the same objective instead of embedding their
own evaluation logic. The Optuna adapter is functional for the portable subset
of the public ``tuning.space`` contract. The n4m adapter uses the native
ask/tell wrapper for the same objective where the wrapper exposes equivalent
search-space primitives.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from .pipeline_objective import PipelineObjective
from .tuning_contracts import DagMLTuningSpec, TrialResult, TuningResult, tcv1_sha256

_N4M_CHECKPOINT_FORMAT = "nirs4all.n4m.optimizer_checkpoint"
_N4M_CHECKPOINT_VERSION = 1
_OPTUNA_STUDY_FORMAT = "nirs4all.optuna.pipeline_objective"
_OPTUNA_STUDY_VERSION = 1
_OPTUNA_STUDY_ATTR_FORMAT = "nirs4all_format"
_OPTUNA_STUDY_ATTR_VERSION = "nirs4all_schema_version"
_OPTUNA_STUDY_ATTR_CONTRACT_FINGERPRINT = "nirs4all_optimizer_contract_fingerprint"
_OPTUNA_STUDY_ATTR_SEARCH_SPACE_FINGERPRINT = "nirs4all_search_space_fingerprint"


class TuningAdapterUnavailable(NotImplementedError):
    """Raised when a requested optimizer adapter is not wired to P4 yet."""


@dataclass(frozen=True)
class ObjectiveTuningRunResult:
    """Internal result for optimize-then-optional-refit over ``PipelineObjective``."""

    tuning_result: TuningResult
    refit_estimator: Any | None
    tuning_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tuning_result, TuningResult):
            raise ValueError("ObjectiveTuningRunResult.tuning_result must be a TuningResult")
        if self.tuning_id is not None:
            if not isinstance(self.tuning_id, str) or not self.tuning_id.strip() or self.tuning_id != self.tuning_id.strip():
                raise ValueError("ObjectiveTuningRunResult.tuning_id must be a canonical non-empty string")
            if "\x00" in self.tuning_id:
                raise ValueError("ObjectiveTuningRunResult.tuning_id must not contain NUL characters")


@dataclass(frozen=True)
class _CategoricalCodec:
    """Internal optimizer-safe representation for one categorical parameter."""

    choices: tuple[Any, ...]
    decoder: Mapping[Any, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.choices, tuple) or not self.choices:
            raise ValueError("_CategoricalCodec.choices must be a non-empty tuple")
        seen_choices: list[Any] = []
        for choice in self.choices:
            if any(choice == seen for seen in seen_choices):
                raise ValueError("_CategoricalCodec.choices must not contain duplicates")
            seen_choices.append(choice)
            try:
                tcv1_sha256(choice)
            except (TypeError, ValueError) as exc:
                raise ValueError("_CategoricalCodec.choices must contain TCV1-compatible JSON-native values") from exc
        if self.decoder is None:
            for choice in self.choices:
                if not _is_optimizer_native_categorical_value(choice):
                    raise ValueError("_CategoricalCodec.choices without a decoder must be optimizer-native categorical values")
            return
        if not isinstance(self.decoder, Mapping) or not self.decoder:
            raise ValueError("_CategoricalCodec.decoder must be a non-empty mapping")
        decoder = dict(self.decoder)
        if tuple(decoder) != self.choices:
            raise ValueError("_CategoricalCodec.decoder keys must match choices in order")
        seen_decoded: list[Any] = []
        for value in decoder.values():
            if any(value == seen for seen in seen_decoded):
                raise ValueError("_CategoricalCodec.decoder values must not contain duplicates")
            seen_decoded.append(value)
            try:
                tcv1_sha256(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("_CategoricalCodec.decoder values must contain TCV1-compatible JSON-native values") from exc
        object.__setattr__(self, "decoder", decoder)

    def decode(self, value: Any) -> Any:
        if self.decoder is None:
            if value in self.choices:
                return value
            raise KeyError(value)
        return self.decoder[value]

    def encode(self, value: Any) -> Any:
        if self.decoder is None:
            if value in self.choices:
                return value
            raise ValueError(f"force_params value {value!r} is not one of categorical choices {list(self.choices)!r}")
        for encoded, decoded in self.decoder.items():
            if decoded == value:
                return encoded
        raise ValueError(f"force_params value {value!r} is not one of categorical choices {list(self.decoder.values())!r}")


def run_pipeline_objective_tuning(
    objective: PipelineObjective,
    X: Any,
    y: Any,
    *,
    sample_ids: Any = None,
    groups: Any = None,
    metadata: Any = None,
    refit: bool = True,
    workspace_path: str | Path | None = None,
    workspace_name: str = "",
    workspace_tuning_id: str | None = None,
    workspace_metadata: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    pipeline_id: str | None = None,
    chain_id: str | None = None,
) -> ObjectiveTuningRunResult:
    """Run optimizer-driving and the terminal winner refit as separate phases.

    When ``workspace_path`` is provided, the completed ``TuningResult`` is
    persisted immediately after optimizer-driving and before terminal refit.
    That preserves the HPO evidence tape even if a later refit fails.
    """

    tuning_result = optimize_pipeline_objective(
        objective,
        X,
        y,
        sample_ids=sample_ids,
        groups=groups,
        metadata=metadata,
    )
    tuning_id = None
    if workspace_path is not None:
        tuning_id = _save_objective_tuning_result_to_workspace(
            workspace_path,
            tuning_result,
            name=workspace_name,
            tuning_id=workspace_tuning_id,
            metadata=workspace_metadata,
            run_id=run_id,
            pipeline_id=pipeline_id,
            chain_id=chain_id,
        )
    refit_estimator = None
    if refit and _has_completed_trial(tuning_result):
        refit_estimator = objective.refit_best(
            tuning_result,
            X,
            y,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
        )
    return ObjectiveTuningRunResult(tuning_result=tuning_result, refit_estimator=refit_estimator, tuning_id=tuning_id)


def _save_objective_tuning_result_to_workspace(
    workspace_path: str | Path,
    tuning_result: TuningResult,
    *,
    name: str = "",
    tuning_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    pipeline_id: str | None = None,
    chain_id: str | None = None,
) -> str:
    """Persist an internal objective tuning result through the workspace store."""

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    store = WorkspaceStore(Path(workspace_path))
    try:
        return store.save_tuning_result(
            tuning_result,
            name=name,
            tuning_id=tuning_id,
            metadata=metadata,
            run_id=run_id,
            pipeline_id=pipeline_id,
            chain_id=chain_id,
        )
    finally:
        store.close()


def optimize_pipeline_objective(
    objective: PipelineObjective,
    X: Any,
    y: Any,
    *,
    sample_ids: Any = None,
    groups: Any = None,
    metadata: Any = None,
) -> TuningResult:
    """Dispatch the objective to the optimizer selected by its tuning spec."""

    if objective.tuning.engine == "optuna":
        return OptunaPipelineObjectiveAdapter().optimize(
            objective,
            X,
            y,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
        )
    return N4MPipelineObjectiveAdapter().optimize(
        objective,
        X,
        y,
        sample_ids=sample_ids,
        groups=groups,
        metadata=metadata,
    )


class OptunaPipelineObjectiveAdapter:
    """Drive a ``PipelineObjective`` with Optuna without duplicating scoring."""

    def optimize(
        self,
        objective: PipelineObjective,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
    ) -> TuningResult:
        """Run Optuna over the objective's typed tuning space."""

        optuna = _import_optuna()
        categorical_codecs = _categorical_codecs(objective.tuning.space)
        trial_diagnostics: dict[int, Mapping[str, Any]] = {}

        def evaluate_trial(trial: Any) -> float:
            params = {path: _suggest_optuna_value(trial, path, spec, categorical_codecs) for path, spec in sorted(objective.tuning.space.items())}
            try:
                result = objective.evaluate(
                    params,
                    X,
                    y,
                    trial_index=trial.number,
                    sample_ids=sample_ids,
                    groups=groups,
                    metadata=metadata,
                )
            except optuna.exceptions.TrialPruned as exc:
                trial_diagnostics[int(trial.number)] = _pruned_trial_diagnostics(objective, exc)
                raise
            except Exception as exc:
                trial_diagnostics[int(trial.number)] = _failed_trial_diagnostics(objective, exc)
                raise
            else:
                trial_diagnostics[int(trial.number)] = dict(result.diagnostics)
                return result.score

        _reject_ambiguous_optuna_resume_controls(objective.tuning)
        with _quiet_optuna_expected_failures(optuna):
            sampler = _make_optuna_sampler(optuna, objective.tuning, categorical_codecs)
            study = optuna.create_study(
                direction=objective.tuning.direction,
                pruner=_make_optuna_pruner(optuna, objective.tuning),
                sampler=sampler,
                study_name=objective.tuning.study_name,
                storage=objective.tuning.storage,
                load_if_exists=objective.tuning.resume,
            )
            _validate_optuna_resume_trial_contract(study, objective.tuning, categorical_codecs)
            _enqueue_optuna_force_params(study, objective.tuning, categorical_codecs)
            _sync_optuna_study_contract_attrs(study, objective.tuning)
            remaining_trials = _optuna_remaining_trials(study, objective.tuning)
            if remaining_trials > 0:
                study.optimize(evaluate_trial, n_trials=remaining_trials, catch=(Exception,))
        trials = tuple(
            TrialResult(
                number=int(trial.number),
                params=_decode_categorical_params(trial.params, categorical_codecs),
                value=_optuna_trial_result_value(trial),
                state=str(trial.state.name),
                diagnostics=_trial_diagnostics(trial_diagnostics, trial, objective),
            )
            for trial in study.trials
        )
        complete_trials = [trial for trial in trials if trial.state == "COMPLETE"]
        if not complete_trials:
            return TuningResult(
                tuning=objective.tuning,
                best_params={},
                best_value=_no_success_best_value(objective.tuning),
                trials=trials,
                optimizer="optuna",
            )
        return TuningResult(
            tuning=objective.tuning,
            best_params=_decode_categorical_params(study.best_params, categorical_codecs),
            best_value=float(study.best_value),
            trials=trials,
            optimizer="optuna",
        )


class N4MPipelineObjectiveAdapter:
    """Drive a ``PipelineObjective`` with the native n4m ask/tell optimizer."""

    def __init__(self, optimizer_api: Any | None = None) -> None:
        self.optimizer_api = optimizer_api

    def optimize(
        self,
        objective: PipelineObjective,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
    ) -> TuningResult:
        """Run n4m over the objective's typed tuning space."""

        api = self.optimizer_api if self.optimizer_api is not None else _import_n4m_optimizer()
        space, slots = _make_n4m_space(api, objective.tuning.space)
        optimizer_kwargs = {
            "sampler": _n4m_enum(api.Sampler, _n4m_sampler_name(objective.tuning.sampler)),
            "direction": _n4m_enum(api.Direction, objective.tuning.direction),
            "seed": 0 if objective.tuning.seed is None else int(objective.tuning.seed),
        }
        n4m_pruner = _n4m_pruner(objective.tuning, api)
        if n4m_pruner is not None:
            optimizer_kwargs["pruner"] = n4m_pruner
        checkpoint_path = _n4m_checkpoint_path(objective.tuning)
        if objective.tuning.resume:
            optimizer = _load_n4m_optimizer_checkpoint(api, objective.tuning, checkpoint_path)
        else:
            _reject_existing_n4m_checkpoint_without_resume(objective.tuning, checkpoint_path)
            optimizer = api.Optimizer(space, **optimizer_kwargs)
            _enqueue_n4m_force_params(optimizer, objective.tuning, _slot_categorical_codecs(slots))
        try:
            existing_trials = _n4m_trial_results_from_optimizer(optimizer, objective, slots)
            remaining_trials = max(objective.tuning.n_trials - len(existing_trials), 0)
            trials: list[TrialResult] = []
            for _ in range(remaining_trials):
                trial = optimizer.ask()
                params = _n4m_trial_params(trial, slots)
                try:
                    result = objective.evaluate(
                        params,
                        X,
                        y,
                        trial_index=int(trial.id),
                        sample_ids=sample_ids,
                        groups=groups,
                        metadata=metadata,
                    )
                except Exception as exc:
                    if _is_pruned_exception(exc):
                        _tell_n4m_trial_pruned(api, optimizer, trial.id)
                        trials.append(
                            TrialResult(
                                number=int(trial.id),
                                params=params,
                                value=None,
                                state="PRUNED",
                                diagnostics=_pruned_trial_diagnostics(objective, exc),
                            )
                        )
                        _save_n4m_optimizer_checkpoint(optimizer, objective.tuning, checkpoint_path)
                        continue
                    _tell_n4m_trial_failed(api, optimizer, trial.id, objective.tuning, exc)
                    trials.append(
                        TrialResult(
                            number=int(trial.id),
                            params=params,
                            value=None,
                            state="FAIL",
                            diagnostics=_failed_trial_diagnostics(objective, exc),
                        )
                    )
                    _save_n4m_optimizer_checkpoint(optimizer, objective.tuning, checkpoint_path)
                else:
                    optimizer.tell(trial.id, result.score)
                    trials.append(
                        TrialResult(
                            number=int(trial.id),
                            params=params,
                            value=result.score,
                            state="COMPLETE",
                            diagnostics=dict(result.diagnostics),
                        )
                    )
                    _save_n4m_optimizer_checkpoint(optimizer, objective.tuning, checkpoint_path)
            all_trials = (*existing_trials, *trials)
            if not any(trial.state == "COMPLETE" for trial in all_trials):
                return TuningResult(
                    tuning=objective.tuning,
                    best_params={},
                    best_value=_no_success_best_value(objective.tuning),
                    trials=tuple(all_trials),
                    optimizer="n4m",
                )
            best_trial, best_value = optimizer.best()
            return TuningResult(
                tuning=objective.tuning,
                best_params=_n4m_trial_params(best_trial, slots),
                best_value=float(best_value),
                trials=tuple(all_trials),
                optimizer="n4m",
            )
        finally:
            close = getattr(optimizer, "close", None)
            if callable(close):
                close()


def _trial_diagnostics(
    trial_diagnostics: Mapping[int, Mapping[str, Any]],
    trial: Any,
    objective: PipelineObjective,
) -> Mapping[str, Any]:
    trial_number = int(trial.number)
    diagnostics = trial_diagnostics.get(trial_number)
    if diagnostics is not None:
        return dict(diagnostics)
    state = str(trial.state.name)
    score_extractor = "optuna_storage"
    error_type: str | None = None
    if state == "FAIL":
        score_extractor = "failed"
        error_type = "OptunaFailedTrial"
    elif state == "PRUNED":
        score_extractor = "pruned"
        error_type = "TrialPruned"
    elif state not in {"COMPLETE", "WAITING", "RUNNING"}:
        score_extractor = f"optuna_{state.lower()}"
    restored = {
        "direction": objective.tuning.direction,
        "engine": objective.tuning.engine,
        "metric": objective.tuning.metric,
        "score_family": "conformal" if objective.tuning.metric.startswith("conformal_") else "objective",
        "score_extractor": score_extractor,
        "search_space_fingerprint": objective.tuning.ordered_search_space.fingerprint,
        "tuning_fingerprint": objective.tuning.fingerprint,
    }
    if error_type is not None:
        restored["error_type"] = error_type
    return restored


def _is_pruned_exception(exc: Exception) -> bool:
    """Return true for optimizer-level prune exceptions without importing Optuna."""

    return type(exc).__name__ == "TrialPruned"


def _failed_trial_diagnostics(objective: PipelineObjective, exc: Exception) -> dict[str, Any]:
    return {
        "direction": objective.tuning.direction,
        "engine": objective.tuning.engine,
        "error": str(exc)[:500],
        "error_type": type(exc).__name__,
        "metric": objective.tuning.metric,
        "score_family": "conformal" if objective.tuning.metric.startswith("conformal_") else "objective",
        "score_extractor": "failed",
        "search_space_fingerprint": objective.tuning.ordered_search_space.fingerprint,
        "tuning_fingerprint": objective.tuning.fingerprint,
    }


def _pruned_trial_diagnostics(objective: PipelineObjective, exc: Exception) -> dict[str, Any]:
    return {
        "direction": objective.tuning.direction,
        "engine": objective.tuning.engine,
        "error": str(exc)[:500],
        "error_type": type(exc).__name__,
        "metric": objective.tuning.metric,
        "score_family": "conformal" if objective.tuning.metric.startswith("conformal_") else "objective",
        "score_extractor": "pruned",
        "search_space_fingerprint": objective.tuning.ordered_search_space.fingerprint,
        "tuning_fingerprint": objective.tuning.fingerprint,
    }


def _has_completed_trial(tuning_result: TuningResult) -> bool:
    return any(trial.state == "COMPLETE" for trial in tuning_result.trials)


def _no_success_best_value(tuning: DagMLTuningSpec) -> float:
    return _failure_score(tuning)


def _import_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise TuningAdapterUnavailable("Optuna is required for run(tuning={engine: 'optuna'})") from exc
    return optuna


@contextmanager
def _quiet_optuna_expected_failures(optuna: Any) -> Iterator[None]:
    """Temporarily suppress Optuna's expected per-trial failure warnings.

    The adapter preserves failure evidence in ``TrialResult.diagnostics``; the
    global Optuna logger should not emit a traceback for each caught candidate.
    """

    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    try:
        yield
    finally:
        optuna.logging.set_verbosity(previous)


def _import_n4m_optimizer() -> Any:
    try:
        from n4m.model_selection import optimizer as optimizer_api
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise TuningAdapterUnavailable("n4m optimizer bindings are required for run(tuning={engine: 'n4m'})") from exc
    return optimizer_api


def _make_optuna_sampler(optuna: Any, tuning: DagMLTuningSpec, categorical_codecs: Mapping[str, _CategoricalCodec]) -> Any:
    if tuning.sampler in {None, "auto", "tpe"}:
        return optuna.samplers.TPESampler(seed=tuning.seed)
    if tuning.sampler == "random":
        return optuna.samplers.RandomSampler(seed=tuning.seed)
    if tuning.sampler == "grid":
        return optuna.samplers.GridSampler(_grid_search_space(tuning.space, categorical_codecs), seed=tuning.seed)
    raise ValueError(f"Optuna PipelineObjective adapter does not support sampler={tuning.sampler!r} yet")


def _make_optuna_pruner(optuna: Any, tuning: DagMLTuningSpec) -> Any:
    if tuning.pruner in {None, "none"}:
        return optuna.pruners.NopPruner()
    if tuning.pruner == "median":
        return optuna.pruners.MedianPruner()
    if tuning.pruner in {"successive_halving", "asha"}:
        return optuna.pruners.SuccessiveHalvingPruner()
    if tuning.pruner == "hyperband":
        return optuna.pruners.HyperbandPruner()
    raise ValueError(f"Optuna PipelineObjective adapter does not support pruner={tuning.pruner!r} yet")


def _enqueue_optuna_force_params(
    study: Any,
    tuning: DagMLTuningSpec,
    categorical_codecs: Mapping[str, _CategoricalCodec],
) -> None:
    params = _encode_force_params(tuning, categorical_codecs)
    if params is None:
        return
    existing_trials = tuple(getattr(study, "trials", ()) or ())
    if tuning.resume and existing_trials:
        first_materialized_params = next(
            (dict(trial.params) for trial in existing_trials if getattr(trial, "params", None)),
            None,
        )
        if first_materialized_params is not None and first_materialized_params != params:
            raise ValueError(
                "Optuna PipelineObjective adapter cannot change tuning.force_params when resume=True "
                "and the study already has materialized trials; use the same public warm-start assignment "
                "or create a new study_name/storage"
            )
        return
    study.enqueue_trial(params)


def _enqueue_n4m_force_params(
    optimizer: Any,
    tuning: DagMLTuningSpec,
    categorical_codecs: Mapping[str, _CategoricalCodec],
) -> None:
    params = _encode_force_params(tuning, categorical_codecs)
    if params is None:
        return
    enqueue = getattr(optimizer, "enqueue", None)
    if not callable(enqueue):
        raise ValueError("n4m PipelineObjective adapter requires optimizer.enqueue(...) when tuning.force_params is supplied; upgrade n4m bindings or remove force_params")
    enqueue(params)


def _encode_force_params(
    tuning: DagMLTuningSpec,
    categorical_codecs: Mapping[str, _CategoricalCodec],
) -> dict[str, Any] | None:
    if tuning.force_params is None:
        return None
    unknown = sorted(set(tuning.force_params) - set(tuning.space))
    if unknown:
        raise ValueError(f"tuning.force_params keys must be present in tuning.space; unknown keys {unknown}")
    encoded: dict[str, Any] = {}
    for path, value in tuning.force_params.items():
        codec = categorical_codecs.get(str(path))
        encoded[str(path)] = codec.encode(value) if codec is not None else value
    return encoded


def _slot_categorical_codecs(slots: list[tuple[str, str, _CategoricalCodec | None]]) -> dict[str, _CategoricalCodec]:
    return {path: codec for path, _kind, codec in slots if codec is not None}


def _grid_search_space(space: Mapping[str, Any], categorical_codecs: Mapping[str, _CategoricalCodec]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    for path, spec in space.items():
        choices = _categorical_choices(spec)
        if choices is None:
            raise ValueError("Optuna grid sampler requires categorical/list choices for every tuning.space entry")
        grid[path] = list(categorical_codecs[path].choices)
    return grid


def _suggest_optuna_value(trial: Any, path: str, spec: Any, categorical_codecs: Mapping[str, _CategoricalCodec]) -> Any:
    choices = _categorical_choices(spec)
    if choices is not None:
        codec = categorical_codecs[path]
        selected = trial.suggest_categorical(path, list(codec.choices))
        return codec.decode(selected)
    if isinstance(spec, tuple) and len(spec) == 2:
        low, high = spec
        if isinstance(low, int) and isinstance(high, int):
            return trial.suggest_int(path, int(low), int(high))
        return trial.suggest_float(path, float(low), float(high))
    if isinstance(spec, tuple) and len(spec) == 3:
        ptype, low, high = spec
        return _suggest_typed_range(trial, path, str(ptype), low, high, step=None, log=None)
    if isinstance(spec, Mapping):
        ptype = str(spec.get("type", "")).lower()
        low = spec.get("low", spec.get("min"))
        high = spec.get("high", spec.get("max"))
        if low is not None and high is not None:
            return _suggest_typed_range(
                trial,
                path,
                ptype or ("int" if isinstance(low, int) and isinstance(high, int) else "float"),
                low,
                high,
                step=spec.get("step"),
                log=spec.get("log"),
            )
    raise ValueError(f"unsupported tuning.space spec for {path!r}: {spec!r}")


def _suggest_typed_range(
    trial: Any,
    path: str,
    ptype: str,
    low: Any,
    high: Any,
    *,
    step: Any,
    log: Any,
) -> Any:
    use_log = bool(log) or ptype in {"int_log", "float_log", "log_int", "log_float"}
    if ptype in {"int", "int_log", "log_int"}:
        return trial.suggest_int(path, int(low), int(high), step=1 if step is None else int(step), log=use_log)
    if ptype in {"float", "float_log", "log_float"}:
        return trial.suggest_float(path, float(low), float(high), step=None if step is None else float(step), log=use_log)
    raise ValueError(f"unsupported range type {ptype!r} for tuning.space path {path!r}")


def _make_n4m_space(api: Any, space_spec: Mapping[str, Any]) -> tuple[Any, list[tuple[str, str, _CategoricalCodec | None]]]:
    space = api.SearchSpace()
    slots: list[tuple[str, str, _CategoricalCodec | None]] = []
    categorical_codecs = _categorical_codecs(space_spec)
    for path, spec in sorted(space_spec.items()):
        choices = _categorical_choices(spec)
        if choices is not None:
            codec = categorical_codecs[path]
            space.add_categorical(path, list(codec.choices))
            slots.append((path, "categorical", codec))
            continue
        if isinstance(spec, tuple) and len(spec) == 2:
            low, high = spec
            if isinstance(low, int) and isinstance(high, int):
                space.add_int(path, int(low), int(high))
                slots.append((path, "int", None))
            else:
                space.add_float(path, float(low), float(high))
                slots.append((path, "float", None))
            continue
        if isinstance(spec, tuple) and len(spec) == 3:
            ptype, low, high = spec
            _add_n4m_range(space, path, str(ptype), low, high, step=None, log=None)
            slots.append((path, _slot_kind(str(ptype)), None))
            continue
        if isinstance(spec, Mapping):
            ptype = str(spec.get("type", "")).lower()
            low = spec.get("low", spec.get("min"))
            high = spec.get("high", spec.get("max"))
            if low is not None and high is not None:
                inferred = ptype or ("int" if isinstance(low, int) and isinstance(high, int) else "float")
                _add_n4m_range(space, path, inferred, low, high, step=spec.get("step"), log=spec.get("log"))
                slots.append((path, _slot_kind(inferred), None))
                continue
        raise ValueError(f"unsupported n4m tuning.space spec for {path!r}: {spec!r}")
    return space, slots


def _add_n4m_range(space: Any, path: str, ptype: str, low: Any, high: Any, *, step: Any, log: Any) -> None:
    use_log = bool(log) or ptype in {"int_log", "float_log", "log_int", "log_float"}
    if ptype in {"int", "int_log", "log_int"}:
        space.add_int(path, int(low), int(high), 1 if step is None else int(step), log=use_log)
        return
    if ptype in {"float", "float_log", "log_float"}:
        space.add_float(path, float(low), float(high), 0.0 if step is None else float(step), log=use_log)
        return
    raise ValueError(f"unsupported n4m range type {ptype!r} for tuning.space path {path!r}")


def _slot_kind(ptype: str) -> str:
    if ptype in {"int", "int_log", "log_int"}:
        return "int"
    if ptype in {"float", "float_log", "log_float"}:
        return "float"
    raise ValueError(f"unsupported n4m range type {ptype!r}")


def _n4m_trial_params(trial: Any, slots: list[tuple[str, str, _CategoricalCodec | None]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for path, kind, codec in slots:
        if kind == "int":
            params[path] = trial.get_int(path)
        elif kind == "float":
            params[path] = trial.get_float(path)
        else:
            index, label = trial.get_category(path)
            if codec is None:
                params[path] = label
            else:
                try:
                    params[path] = codec.decode(label)
                except KeyError:
                    params[path] = codec.decode(codec.choices[int(index)])
    return params


def _categorical_codecs(space: Mapping[str, Any]) -> dict[str, _CategoricalCodec]:
    codecs: dict[str, _CategoricalCodec] = {}
    for path, spec in space.items():
        codec = _categorical_codec_for_spec(spec)
        if codec is not None:
            codecs[path] = codec
    return codecs


def _categorical_codec_for_spec(spec: Any) -> _CategoricalCodec | None:
    if isinstance(spec, Mapping):
        ptype = str(spec.get("type", "")).lower()
        if ptype == "categorical" and isinstance(spec.get("options"), Mapping):
            raw_options = spec["options"]
            if not raw_options:
                raise ValueError("categorical tuning.space choices must be non-empty")
            labels = list(raw_options)
            codec = _categorical_codec(labels)
            decoder: dict[Any, Any] = {}
            for label in labels:
                optimizer_label = label
                if codec.decoder is not None:
                    optimizer_label = next(encoded for encoded in codec.choices if codec.decode(encoded) == label)
                decoder[optimizer_label] = raw_options[label]
            return _CategoricalCodec(codec.choices, decoder)
    choices = _categorical_choices(spec)
    if choices is None:
        return None
    return _categorical_codec(choices)


def _categorical_codec(choices: list[Any]) -> _CategoricalCodec:
    if all(_is_optimizer_native_categorical_value(choice) for choice in choices):
        return _CategoricalCodec(tuple(choices))
    encoded: list[str] = []
    decoder: dict[str, Any] = {}
    for index, choice in enumerate(choices):
        token = f"__nirs4all_choice_{index:06d}__"
        encoded.append(token)
        decoder[token] = choice
    return _CategoricalCodec(tuple(encoded), decoder)


def _is_optimizer_native_categorical_value(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _decode_categorical_params(params: Mapping[str, Any], categorical_codecs: Mapping[str, _CategoricalCodec]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for path, value in params.items():
        codec = categorical_codecs.get(str(path))
        decoded[str(path)] = codec.decode(value) if codec is not None else value
    return decoded


def _n4m_sampler_name(sampler: str | None) -> str:
    if sampler in {None, "auto", "tpe"}:
        return "tpe"
    if sampler == "random":
        return "random"
    if sampler == "grid":
        return "grid"
    raise ValueError(f"n4m PipelineObjective adapter does not support sampler={sampler!r} yet")


def _n4m_checkpoint_path(tuning: DagMLTuningSpec) -> Path | None:
    if tuning.storage is None:
        if tuning.resume or tuning.study_name is not None:
            raise ValueError("n4m PipelineObjective adapter requires storage='file:///...' and study_name for optimizer-state resume")
        return None
    if tuning.study_name is None:
        raise ValueError("n4m PipelineObjective adapter requires study_name when storage is configured")
    study_name = tuning.study_name.strip()
    if not study_name or "/" in study_name or "\\" in study_name:
        raise ValueError("n4m PipelineObjective adapter study_name must be a non-empty filename-safe name")
    parsed = urlparse(tuning.storage)
    if parsed.scheme != "file" or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("n4m PipelineObjective adapter only supports storage='file:///directory' for native N4MOPT checkpoints")
    if parsed.netloc not in ("", "localhost"):
        raise ValueError("n4m PipelineObjective adapter file:// storage must not name a remote host")
    directory = Path(url2pathname(parsed.path))
    if not directory.is_absolute():
        raise ValueError("n4m PipelineObjective adapter file:// storage must resolve to an absolute directory")
    return directory / f"{study_name}.n4mopt.json"


def _reject_existing_n4m_checkpoint_without_resume(tuning: DagMLTuningSpec, checkpoint_path: Path | None) -> None:
    if checkpoint_path is not None and checkpoint_path.exists():
        raise ValueError("n4m PipelineObjective adapter found an existing optimizer checkpoint; set resume=True or choose a new study_name/storage")
    if tuning.resume:
        raise ValueError("n4m PipelineObjective adapter resume=True requires an existing optimizer checkpoint")


def _n4m_checkpoint_manifest(tuning: DagMLTuningSpec, checkpoint: bytes) -> dict[str, Any]:
    checkpoint_b64 = base64.b64encode(checkpoint).decode("ascii")
    return {
        "checkpoint_b64": checkpoint_b64,
        "checkpoint_fingerprint": tcv1_sha256({"checkpoint_b64": checkpoint_b64}),
        "format": _N4M_CHECKPOINT_FORMAT,
        "optimizer_contract_fingerprint": _n4m_optimizer_contract_fingerprint(tuning),
        "schema_version": _N4M_CHECKPOINT_VERSION,
        "search_space_fingerprint": tuning.ordered_search_space.fingerprint,
        "study_name": tuning.study_name,
        "tuning_fingerprint": tuning.fingerprint,
    }


def _save_n4m_optimizer_checkpoint(optimizer: Any, tuning: DagMLTuningSpec, checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return
    save = getattr(optimizer, "save", None)
    if not callable(save):
        raise ValueError("n4m PipelineObjective adapter requires optimizer.save() for storage-backed native resume; upgrade n4m bindings or remove storage")
    checkpoint = save()
    if not isinstance(checkpoint, (bytes, bytearray, memoryview)):
        raise ValueError("n4m optimizer.save() must return N4MOPT bytes")
    manifest = _n4m_checkpoint_manifest(tuning, bytes(checkpoint))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(checkpoint_path)


def _load_n4m_optimizer_checkpoint(api: Any, tuning: DagMLTuningSpec, checkpoint_path: Path | None) -> Any:
    if checkpoint_path is None or not checkpoint_path.exists():
        raise ValueError("n4m PipelineObjective adapter resume=True requires an existing optimizer checkpoint")
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("n4m optimizer checkpoint manifest must be a mapping")
    expected = {
        "format": _N4M_CHECKPOINT_FORMAT,
        "optimizer_contract_fingerprint": _n4m_optimizer_contract_fingerprint(tuning),
        "schema_version": _N4M_CHECKPOINT_VERSION,
        "study_name": tuning.study_name,
        "search_space_fingerprint": tuning.ordered_search_space.fingerprint,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"n4m optimizer checkpoint mismatch for {key}; use the same tuning contract or a new study_name/storage")
    checkpoint_b64 = payload.get("checkpoint_b64")
    if not isinstance(checkpoint_b64, str) or not checkpoint_b64:
        raise ValueError("n4m optimizer checkpoint manifest is missing checkpoint_b64")
    if payload.get("checkpoint_fingerprint") != tcv1_sha256({"checkpoint_b64": checkpoint_b64}):
        raise ValueError("n4m optimizer checkpoint fingerprint mismatch")
    try:
        checkpoint = base64.b64decode(checkpoint_b64.encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError("n4m optimizer checkpoint payload is not valid base64") from exc
    load = getattr(api.Optimizer, "load", None)
    if not callable(load):
        raise ValueError("n4m PipelineObjective adapter requires Optimizer.load(...) for resume=True; upgrade n4m bindings or remove resume")
    return load(checkpoint)


def _n4m_optimizer_contract_fingerprint(tuning: DagMLTuningSpec) -> str:
    payload = tuning.to_dict()
    for key in ("n_trials", "resume", "storage", "study_name"):
        payload.pop(key, None)
    return tcv1_sha256(payload)


def _n4m_trial_results_from_optimizer(
    optimizer: Any,
    objective: PipelineObjective,
    slots: list[tuple[str, str, _CategoricalCodec | None]],
) -> tuple[TrialResult, ...]:
    get_trials = getattr(optimizer, "get_trials", None)
    if not callable(get_trials):
        return ()
    records = get_trials()
    if not records:
        return ()
    ordered_records = sorted(records, key=_n4m_record_trial_id)
    trial_ids = [_n4m_record_trial_id(record) for record in ordered_records]
    duplicate_ids = sorted({trial_id for trial_id in trial_ids if trial_ids.count(trial_id) > 1})
    if duplicate_ids:
        raise ValueError(f"n4m optimizer checkpoint contains duplicate trial ids {duplicate_ids!r}; resume requires one terminal record per trial id")
    return tuple(_n4m_trial_result_from_record(record, objective, slots) for record in ordered_records)


def _n4m_record_trial_id(record: Any) -> int:
    try:
        return int(record.id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"n4m optimizer checkpoint trial id {getattr(record, 'id', None)!r} is not an integer; resume requires canonical numeric trial ids") from exc


def _n4m_trial_result_from_record(
    record: Any,
    objective: PipelineObjective,
    slots: list[tuple[str, str, _CategoricalCodec | None]],
) -> TrialResult:
    status = _n4m_trial_state(getattr(record, "status", ""))
    if status not in {"COMPLETE", "FAIL", "PRUNED", "CANCELLED"}:
        raise ValueError(f"n4m optimizer checkpoint contains a non-terminal or unsupported trial status {status!r}; resume requires terminal trial records")
    score = getattr(record, "score", None)
    if status == "COMPLETE":
        value = _n4m_complete_trial_score(record, score)
    else:
        _n4m_non_complete_trial_score(record, status, score)
        value = None
    params = getattr(record, "params", {})
    if not isinstance(params, Mapping):
        raise ValueError("n4m optimizer checkpoint trial params must be a mapping")
    _validate_n4m_resume_record_params(record, params, objective, slots)
    diagnostics = {
        "direction": objective.tuning.direction,
        "engine": objective.tuning.engine,
        "metric": objective.tuning.metric,
        "score_family": "conformal" if objective.tuning.metric.startswith("conformal_") else "objective",
        "score_extractor": _n4m_checkpoint_score_extractor(status),
        "search_space_fingerprint": objective.tuning.ordered_search_space.fingerprint,
        "tuning_fingerprint": objective.tuning.fingerprint,
    }
    error = getattr(record, "error", None)
    if error is not None:
        diagnostics["error_type"] = getattr(error, "code", type(error).__name__)
    return TrialResult(
        number=_n4m_record_trial_id(record),
        params=_decode_n4m_record_params(params, slots),
        value=value,
        state=status,
        diagnostics=diagnostics,
    )


def _n4m_complete_trial_score(record: Any, score: Any) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"n4m optimizer checkpoint COMPLETE trial {_n4m_record_trial_id(record)} is missing a numeric finite score")
    value = float(score)
    if not math.isfinite(value):
        raise ValueError(f"n4m optimizer checkpoint COMPLETE trial {_n4m_record_trial_id(record)} has a non-finite score")
    return value


def _n4m_non_complete_trial_score(record: Any, status: str, score: Any) -> None:
    if score is not None:
        raise ValueError(f"n4m optimizer checkpoint {status} trial {_n4m_record_trial_id(record)} carries a final score; only COMPLETE trials may carry scores")
    return None


def _validate_n4m_resume_record_params(
    record: Any,
    params: Mapping[str, Any],
    objective: PipelineObjective,
    slots: list[tuple[str, str, _CategoricalCodec | None]],
) -> None:
    expected_keys = {path for path, _kind, _codec in slots}
    actual_keys = {str(path) for path in params}
    if actual_keys != expected_keys:
        raise ValueError(
            "n4m optimizer checkpoint contains trial params that do not match tuning.space keys; "
            f"trial {int(record.id)} has {sorted(actual_keys)!r}, expected {sorted(expected_keys)!r}. "
            "Use the same search space or choose a new study_name/storage."
        )
    slot_by_path = {path: (kind, codec) for path, kind, codec in slots}
    for path, value in params.items():
        path_str = str(path)
        _kind, codec = slot_by_path[path_str]
        if codec is not None:
            try:
                codec.decode(value)
            except KeyError as exc:
                raise ValueError(
                    "n4m optimizer checkpoint contains a trial categorical value "
                    f"{value!r} for {path_str!r} that is not present in the current tuning.space choices; "
                    "use the same search space or choose a new study_name/storage."
                ) from exc
            continue
        if not _optuna_resume_value_matches_space_spec(value, objective.tuning.space[path_str]):
            raise ValueError(f"n4m optimizer checkpoint contains a trial value {value!r} for {path_str!r} outside the current tuning.space range; use the same search space or choose a new study_name/storage.")


def _n4m_checkpoint_score_extractor(status: str) -> str:
    if status == "FAIL":
        return "failed"
    if status == "PRUNED":
        return "pruned"
    if status == "CANCELLED":
        return "cancelled"
    return "native_checkpoint"


def _decode_n4m_record_params(
    params: Mapping[str, Any],
    slots: list[tuple[str, str, _CategoricalCodec | None]],
) -> dict[str, Any]:
    codecs = _slot_categorical_codecs(slots)
    decoded: dict[str, Any] = {}
    for path, value in params.items():
        codec = codecs.get(str(path))
        decoded[str(path)] = codec.decode(value) if codec is not None else value
    return decoded


def _n4m_trial_state(status: Any) -> str:
    name = getattr(status, "name", str(status)).upper()
    if name in {"COMPLETE", "COMPLETED"}:
        return "COMPLETE"
    if name in {"FAIL", "FAILED"}:
        return "FAIL"
    if name == "PRUNED":
        return "PRUNED"
    if name == "CANCELLED":
        return "CANCELLED"
    if name == "RUNNING":
        return "RUNNING"
    return name


def _reject_ambiguous_optuna_resume_controls(tuning: DagMLTuningSpec) -> None:
    if not tuning.resume:
        return
    missing = []
    if tuning.storage is None:
        missing.append("storage")
    if tuning.study_name is None:
        missing.append("study_name")
    if missing:
        raise ValueError(f"Optuna PipelineObjective adapter requires explicit storage and study_name when resume=True; otherwise Optuna would create or load an ambiguous in-memory/anonymous study. Missing: {missing}")


def _optuna_remaining_trials(study: Any, tuning: DagMLTuningSpec) -> int:
    if not tuning.resume:
        return int(tuning.n_trials)
    terminal_states = {"COMPLETE", "FAIL", "PRUNED"}
    existing_terminal = sum(1 for trial in tuple(getattr(study, "trials", ()) or ()) if str(getattr(getattr(trial, "state", None), "name", "")) in terminal_states)
    return max(int(tuning.n_trials) - existing_terminal, 0)


def _sync_optuna_study_contract_attrs(study: Any, tuning: DagMLTuningSpec) -> None:
    expected = _optuna_study_contract_attrs(tuning)
    attrs = dict(getattr(study, "user_attrs", {}) or {})
    has_trials = bool(tuple(getattr(study, "trials", ()) or ()))
    for key, expected_value in expected.items():
        actual = attrs.get(key)
        if actual is None:
            if tuning.resume and has_trials:
                raise ValueError("Optuna PipelineObjective adapter cannot resume a study without nirs4all optimizer contract fingerprints; use a study created by this adapter or choose a new study_name/storage.")
            continue
        if actual != expected_value:
            raise ValueError(f"Optuna PipelineObjective adapter study contract mismatch for {key}; use the same tuning contract or choose a new study_name/storage.")
    for key, expected_value in expected.items():
        study.set_user_attr(key, expected_value)


def _optuna_study_contract_attrs(tuning: DagMLTuningSpec) -> dict[str, Any]:
    return {
        _OPTUNA_STUDY_ATTR_CONTRACT_FINGERPRINT: _optuna_optimizer_contract_fingerprint(tuning),
        _OPTUNA_STUDY_ATTR_FORMAT: _OPTUNA_STUDY_FORMAT,
        _OPTUNA_STUDY_ATTR_SEARCH_SPACE_FINGERPRINT: tuning.ordered_search_space.fingerprint,
        _OPTUNA_STUDY_ATTR_VERSION: _OPTUNA_STUDY_VERSION,
    }


def _optuna_optimizer_contract_fingerprint(tuning: DagMLTuningSpec) -> str:
    payload = tuning.to_dict()
    for key in ("n_trials", "resume", "storage", "study_name"):
        payload.pop(key, None)
    return tcv1_sha256(payload)


def _optuna_trial_result_value(trial: Any) -> float | None:
    state = str(getattr(getattr(trial, "state", None), "name", ""))
    score = getattr(trial, "value", None)
    if state != "COMPLETE":
        if score is not None:
            _raise_optuna_non_complete_trial_value(trial, state)
        return None
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"Optuna PipelineObjective adapter storage COMPLETE trial {int(trial.number)} is missing a numeric finite value")
    value = float(score)
    if not math.isfinite(value):
        raise ValueError(f"Optuna PipelineObjective adapter storage COMPLETE trial {int(trial.number)} has a non-finite value")
    return value


def _raise_optuna_non_complete_trial_value(trial: Any, state: str) -> None:
    raise ValueError(f"Optuna PipelineObjective adapter storage {state or 'UNKNOWN'} trial {int(trial.number)} carries a final value; only COMPLETE trials may carry values")


def _validate_optuna_resume_trial_contract(
    study: Any,
    tuning: DagMLTuningSpec,
    categorical_codecs: Mapping[str, _CategoricalCodec],
) -> None:
    if not tuning.resume:
        return
    expected_keys = set(tuning.space)
    trials = _optuna_resume_study_trials(study)
    seen_numbers: set[int] = set()
    for trial in trials:
        trial_number = _optuna_resume_trial_number(trial)
        if trial_number in seen_numbers:
            raise ValueError(
                f"Optuna PipelineObjective adapter cannot resume a study with duplicate trial number {trial_number}; resume requires canonical unique trial numbers. Clean the study or choose a new study_name/storage."
            )
        seen_numbers.add(trial_number)
        state = str(getattr(getattr(trial, "state", None), "name", ""))
        if state in {"WAITING", "RUNNING"}:
            if getattr(trial, "value", None) is not None:
                _raise_optuna_non_complete_trial_value(trial, state)
            if state == "RUNNING":
                raise ValueError(
                    "Optuna PipelineObjective adapter cannot resume a study with RUNNING trials; "
                    f"trial {int(trial.number)} was active when the previous run stopped. "
                    "Finish or clean the study, or choose a new study_name/storage."
                )
            params = dict(getattr(trial, "params", {}) or {})
            fixed_params = _optuna_resume_trial_fixed_params(trial)
            if fixed_params is not None:
                _validate_optuna_resume_trial_params(trial, fixed_params, tuning, categorical_codecs, expected_keys)
                if params and params != fixed_params:
                    raise ValueError(
                        "Optuna PipelineObjective adapter cannot resume a queued trial whose "
                        f"materialized params differ from fixed_params; trial {int(trial.number)} "
                        "has inconsistent storage rows. Clean the study or choose a new study_name/storage."
                    )
            if params:
                _validate_optuna_resume_trial_params(trial, params, tuning, categorical_codecs, expected_keys)
            continue
        params = dict(getattr(trial, "params", {}) or {})
        _validate_optuna_resume_trial_params(trial, params, tuning, categorical_codecs, expected_keys)


def _optuna_resume_study_trials(study: Any) -> tuple[Any, ...]:
    try:
        return tuple(getattr(study, "trials", ()) or ())
    except TypeError as exc:
        raise ValueError("Optuna PipelineObjective adapter cannot resume a study whose trial numbers are not canonical integers; clean the study or choose a new study_name/storage.") from exc


def _optuna_resume_trial_number(trial: Any) -> int:
    number = getattr(trial, "number", None)
    if isinstance(number, bool) or not isinstance(number, int):
        raise ValueError(f"Optuna PipelineObjective adapter cannot resume a study whose trial numbers are not canonical integers; trial number {number!r} is invalid. Clean the study or choose a new study_name/storage.")
    return number


def _optuna_resume_trial_fixed_params(trial: Any) -> dict[str, Any] | None:
    system_attrs = getattr(trial, "system_attrs", {}) or {}
    fixed_params = system_attrs.get("fixed_params")
    if fixed_params is None:
        return None
    if not isinstance(fixed_params, Mapping):
        raise ValueError(
            "Optuna PipelineObjective adapter cannot resume a queued trial whose fixed_params "
            f"are not a mapping; trial {int(trial.number)} has {type(fixed_params).__name__}. "
            "Clean the study or choose a new study_name/storage."
        )
    return {str(path): value for path, value in fixed_params.items()}


def _validate_optuna_resume_trial_params(
    trial: Any,
    params: Mapping[str, Any],
    tuning: DagMLTuningSpec,
    categorical_codecs: Mapping[str, _CategoricalCodec],
    expected_keys: set[str],
) -> None:
    actual_keys = set(params)
    if actual_keys != expected_keys:
        raise ValueError(
            "Optuna PipelineObjective adapter cannot resume a study whose materialized trial "
            f"params do not match tuning.space keys; trial {int(trial.number)} has "
            f"{sorted(actual_keys)!r}, expected {sorted(expected_keys)!r}. "
            "Use the same search space or choose a new study_name/storage."
        )
    for path, value in params.items():
        codec = categorical_codecs.get(str(path))
        if codec is not None:
            try:
                codec.decode(value)
            except KeyError as exc:
                raise ValueError(
                    "Optuna PipelineObjective adapter cannot resume a study whose materialized "
                    f"categorical value {value!r} for {str(path)!r} is not present in the current "
                    "tuning.space choices; use the same search space or choose a new study_name/storage."
                ) from exc
            continue
        if not _optuna_resume_value_matches_space_spec(value, tuning.space[str(path)]):
            raise ValueError(
                "Optuna PipelineObjective adapter cannot resume a study whose materialized "
                f"value {value!r} for {str(path)!r} is outside the current tuning.space range; "
                "use the same search space or choose a new study_name/storage."
            )


def _optuna_resume_value_matches_space_spec(value: Any, spec: Any) -> bool:
    if isinstance(spec, tuple) and len(spec) == 2:
        low, high = spec
        ptype = "int" if isinstance(low, int) and isinstance(high, int) else "float"
        return _value_matches_numeric_range(value, ptype, low, high)
    if isinstance(spec, tuple) and len(spec) == 3:
        ptype, low, high = spec
        return _value_matches_numeric_range(value, str(ptype), low, high)
    if isinstance(spec, Mapping):
        low = spec.get("low", spec.get("min"))
        high = spec.get("high", spec.get("max"))
        if low is None or high is None:
            return True
        ptype = str(spec.get("type", "")).lower() or ("int" if isinstance(low, int) and isinstance(high, int) else "float")
        return _value_matches_numeric_range(value, ptype, low, high, step=spec.get("step"))
    return True


def _value_matches_numeric_range(
    value: Any,
    ptype: str,
    low: Any,
    high: Any,
    *,
    step: Any = None,
) -> bool:
    if isinstance(value, bool):
        return False
    if "int" in ptype:
        if not isinstance(value, int):
            return False
        numeric = int(value)
        if not int(low) <= numeric <= int(high):
            return False
        if step is None:
            return True
        return (numeric - int(low)) % int(step) == 0
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False
    numeric_float = float(value)
    if not float(low) <= numeric_float <= float(high):
        return False
    if step is None:
        return True
    offset = (numeric_float - float(low)) / float(step)
    return math.isclose(offset, round(offset), rel_tol=1.0e-12, abs_tol=1.0e-12)


def _tell_n4m_trial_failed(api: Any, optimizer: Any, trial_id: Any, tuning: DagMLTuningSpec, exc: Exception) -> None:
    tell_result = getattr(optimizer, "tell_result", None)
    if callable(tell_result) and hasattr(api, "TrialStatus"):
        try:
            tell_result(trial_id, api.TrialStatus.FAILED, error=str(exc)[:200])
            return
        except TypeError:
            tell_result(trial_id, api.TrialStatus.FAILED)
            return
    tell = getattr(optimizer, "tell", None)
    if callable(tell):
        tell(trial_id, _failure_score(tuning))
        return
    raise ValueError("n4m PipelineObjective adapter requires optimizer.tell_result(...) or optimizer.tell(...) to record failed trials; upgrade n4m bindings or use engine='optuna' for failure-tolerant candidate tapes")


def _tell_n4m_trial_pruned(api: Any, optimizer: Any, trial_id: Any) -> None:
    tell_result = getattr(optimizer, "tell_result", None)
    trial_status = getattr(api, "TrialStatus", None)
    pruned_status = getattr(trial_status, "PRUNED", None) if trial_status is not None else None
    if callable(tell_result) and pruned_status is not None:
        tell_result(trial_id, pruned_status)
        return
    raise ValueError("n4m PipelineObjective adapter requires optimizer.tell_result(..., TrialStatus.PRUNED) to record pruned trials; upgrade n4m bindings or use engine='optuna' for prune-aware candidate tapes")


def _failure_score(tuning: DagMLTuningSpec) -> float:
    return -1.0e308 if tuning.direction == "maximize" else 1.0e308


def _n4m_pruner(tuning: DagMLTuningSpec, api: Any) -> Any | None:
    if tuning.pruner in {None, "none"}:
        return None
    if not hasattr(api, "Pruner"):
        raise ValueError("n4m PipelineObjective adapter received pruner but optimizer API exposes no Pruner enum")
    return _n4m_enum(api.Pruner, _n4m_pruner_name(tuning.pruner))


def _n4m_pruner_name(pruner: str | None) -> str:
    if pruner == "median":
        return "median"
    if pruner in {"successive_halving", "asha"}:
        return "asha"
    if pruner == "hyperband":
        return "hyperband"
    if pruner == "racing":
        return "racing"
    raise ValueError(f"n4m PipelineObjective adapter does not support pruner={pruner!r} yet")


def _n4m_enum(enum_type: Any, name: str) -> Any:
    enum_name = name.upper()
    try:
        return getattr(enum_type, enum_name)
    except AttributeError as exc:
        raise ValueError(f"n4m PipelineObjective adapter requires optimizer enum {enum_name!r}; upgrade n4m bindings or choose another supported sampler/pruner") from exc


def _categorical_choices(spec: Any) -> list[Any] | None:
    if isinstance(spec, list):
        if not spec:
            raise ValueError("categorical tuning.space lists must be non-empty")
        return list(spec)
    if isinstance(spec, tuple) and len(spec) == 2 and str(spec[0]).lower() in {"categorical", "bool"}:
        choices = list(spec[1])
        if not choices:
            raise ValueError("categorical tuning.space choices must be non-empty")
        return choices
    if isinstance(spec, Mapping):
        ptype = str(spec.get("type", "")).lower()
        if ptype == "categorical":
            raw = spec.get("choices", spec.get("values", spec.get("options")))
            if isinstance(raw, Mapping):
                choices = list(raw)
            else:
                choices = list(raw or [])
            if not choices:
                raise ValueError("categorical tuning.space choices must be non-empty")
            return choices
        if ptype == "bool":
            return [False, True]
    return None


__all__ = [
    "N4MPipelineObjectiveAdapter",
    "ObjectiveTuningRunResult",
    "OptunaPipelineObjectiveAdapter",
    "TuningAdapterUnavailable",
    "optimize_pipeline_objective",
    "run_pipeline_objective_tuning",
]
