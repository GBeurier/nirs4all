"""Public helpers for native DAG-ML tuning results."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Literal

from nirs4all.api.result import PredictResult, RunResult
from nirs4all.optimization.n4m_engine import _PRUNER_MAP as _N4M_PRUNER_MAP
from nirs4all.optimization.n4m_engine import _SAMPLER_MAP as _N4M_SAMPLER_MAP
from nirs4all.optimization.optuna import (
    VALID_APPROACHES,
    VALID_EVAL_MODES,
    VALID_PRUNERS,
    VALID_SAMPLERS,
)
from nirs4all.pipeline.dagml.conformal_contracts import (
    SUPPORTED_CONFORMAL_METHODS,
    SUPPORTED_CONFORMAL_UNITS,
    CalibratedRunResult,
)
from nirs4all.pipeline.dagml.finetune_lowering import (
    DETERMINISTIC_FINETUNE_ENGINES,
    PUBLIC_DAGML_SELECTION_METRICS,
    SUPPORTED_FINETUNE_META_KEYS,
)
from nirs4all.pipeline.dagml.tuning_contracts import (
    SUPPORTED_TUNING_DIRECTIONS,
    SUPPORTED_TUNING_ENGINES,
    SUPPORTED_TUNING_KEYS,
    TUNING_SPACE_SCHEMA_ID,
    TUNING_SUMMARY_SCHEMA_ID,
    DagMLTuningSpec,
    OrderedSearchSpaceSpec,
    ParameterPatch,
    SearchSpaceParameter,
    TrialResult,
    TuningDirection,
    TuningEngine,
    TuningResult,
    get_tuning_space_schema,
    get_tuning_summary_schema,
    parse_tuning_spec,
    tuning_space_schema_json,
    tuning_summary_schema_json,
)

FinetuneEngine = Literal["optuna", "n4m", "dag-ml", "grid"]
FinetuneSampler = Literal["auto", "grid", "tpe", "random", "cmaes", "binary", "sobol", "lhs", "ternary", "ga", "pso", "gp_ei"]
FinetunePruner = Literal["none", "median", "successive_halving", "hyperband", "asha", "racing"]
FinetuneApproach = Literal["single", "grouped", "individual"]
FinetuneEvalMode = Literal["best", "mean", "robust_best"]

FINETUNE_ENGINES: tuple[FinetuneEngine, ...] = ("optuna", "n4m", "dag-ml", "grid")
FINETUNE_ENGINE_ALIASES: tuple[tuple[str, FinetuneEngine], ...] = (
    ("dagml", "dag-ml"),
    ("native", "dag-ml"),
    ("methods", "n4m"),
    ("libn4m", "n4m"),
)
FINETUNE_SAMPLER_KEY_ALIASES: tuple[tuple[str, str], ...] = (("sample", "sampler"),)
FINETUNE_EVAL_MODE_ALIASES: tuple[tuple[str, FinetuneEvalMode], ...] = (("avg", "mean"),)
FINETUNE_APPROACHES: tuple[FinetuneApproach, ...] = ("single", "grouped", "individual")
FINETUNE_EVAL_MODES: tuple[FinetuneEvalMode, ...] = ("best", "mean", "robust_best")
FINETUNE_OPTUNA_SAMPLERS: tuple[FinetuneSampler, ...] = ("auto", "grid", "tpe", "random", "cmaes", "binary")
FINETUNE_OPTUNA_PRUNERS: tuple[FinetunePruner, ...] = ("none", "median", "successive_halving", "hyperband")
FINETUNE_N4M_SAMPLERS: tuple[FinetuneSampler, ...] = (
    "auto",
    "grid",
    "binary",
    "random",
    "sobol",
    "lhs",
    "ternary",
    "ga",
    "pso",
    "cmaes",
    "tpe",
    "gp_ei",
)
FINETUNE_N4M_PRUNERS: tuple[FinetunePruner, ...] = ("none", "median", "successive_halving", "asha", "hyperband", "racing")
FINETUNE_DAGML_DETERMINISTIC_ENGINES: tuple[FinetuneEngine, ...] = ("dag-ml", "grid")
FINETUNE_DAGML_META_KEYS: tuple[str, ...] = ("approach", "direction", "engine", "eval_mode", "metric", "model_params")
FINETUNE_DAGML_SELECTION_METRICS: tuple[str, ...] = ("rmse", "accuracy", "balanced_accuracy")
FINETUNE_DAGML_APPROACHES: tuple[FinetuneApproach, ...] = ("grouped",)
FINETUNE_DAGML_EVAL_MODES: tuple[FinetuneEvalMode, ...] = ("mean", "best")

TUNING_ENGINES: tuple[TuningEngine, ...] = ("optuna", "n4m")
TUNING_DIRECTIONS: tuple[TuningDirection, ...] = ("minimize", "maximize")
TUNING_CONTRACT_KEYS: tuple[str, ...] = (
    "direction",
    "engine",
    "force_params",
    "metric",
    "n_trials",
    "pruner",
    "resume",
    "sampler",
    "seed",
    "space",
    "storage",
    "study_name",
)
TUNING_OPTIMIZER_PERSISTENCE_KEYS: tuple[str, ...] = ("storage", "study_name")
TUNING_RUNTIME_KEYS: tuple[str, ...] = (
    "calibration",
    "score_data",
    "tuning_id",
    "winner",
    "workspace_metadata",
    "workspace_tuning_id",
)
CONFORMAL_TUNING_SCORE_METRICS: tuple[str, ...] = (
    "conformal_abs_coverage_gap",
    "conformal_interval_score",
    "conformal_mean_interval_score",
    "conformal_mean_width",
    "conformal_median_width",
    "conformal_missed_rate",
    "conformal_observed_coverage",
)
_CONFORMAL_TUNING_SCORE_METRICS = frozenset(CONFORMAL_TUNING_SCORE_METRICS)
_TUNING_CALIBRATION_RESERVED_EXTRA_KEYS = frozenset(
    {
        "as_predict_result",
        "calibration_data",
        "coverage",
        "method",
        "prediction_sample_ids",
        "unit",
        "workspace_conformal_id",
        "workspace_metadata",
        "y_pred",
    }
)

if set(TUNING_ENGINES) != set(SUPPORTED_TUNING_ENGINES):
    raise RuntimeError("public TUNING_ENGINES drifted from DagMLTuningSpec validation")
if set(TUNING_DIRECTIONS) != set(SUPPORTED_TUNING_DIRECTIONS):
    raise RuntimeError("public TUNING_DIRECTIONS drifted from DagMLTuningSpec validation")
if set(TUNING_CONTRACT_KEYS) != set(SUPPORTED_TUNING_KEYS):
    raise RuntimeError("public TUNING_CONTRACT_KEYS drifted from DagMLTuningSpec validation")
if not set(TUNING_OPTIMIZER_PERSISTENCE_KEYS) <= set(SUPPORTED_TUNING_KEYS):
    raise RuntimeError("public TUNING_OPTIMIZER_PERSISTENCE_KEYS drifted from DagMLTuningSpec validation")
if set(FINETUNE_OPTUNA_SAMPLERS) != set(VALID_SAMPLERS):
    raise RuntimeError("public FINETUNE_OPTUNA_SAMPLERS drifted from OptunaManager validation")
if set(FINETUNE_OPTUNA_PRUNERS) != set(VALID_PRUNERS):
    raise RuntimeError("public FINETUNE_OPTUNA_PRUNERS drifted from OptunaManager validation")
if set(FINETUNE_APPROACHES) != set(VALID_APPROACHES):
    raise RuntimeError("public FINETUNE_APPROACHES drifted from OptunaManager validation")
if set(FINETUNE_EVAL_MODES) != set(VALID_EVAL_MODES):
    raise RuntimeError("public FINETUNE_EVAL_MODES drifted from OptunaManager validation")
if set(FINETUNE_N4M_SAMPLERS) != set(_N4M_SAMPLER_MAP) - {"sample"}:
    raise RuntimeError("public FINETUNE_N4M_SAMPLERS drifted from N4MFinetuneManager validation")
if set(FINETUNE_N4M_PRUNERS) != set(_N4M_PRUNER_MAP):
    raise RuntimeError("public FINETUNE_N4M_PRUNERS drifted from N4MFinetuneManager validation")
if set(FINETUNE_DAGML_DETERMINISTIC_ENGINES) | {"", "dagml", "native"} != set(DETERMINISTIC_FINETUNE_ENGINES):
    raise RuntimeError("public FINETUNE_DAGML_DETERMINISTIC_ENGINES drifted from DAG-ML lowering")
if set(FINETUNE_DAGML_META_KEYS) != set(SUPPORTED_FINETUNE_META_KEYS):
    raise RuntimeError("public FINETUNE_DAGML_META_KEYS drifted from DAG-ML lowering")
if set(FINETUNE_DAGML_SELECTION_METRICS) != set(PUBLIC_DAGML_SELECTION_METRICS):
    raise RuntimeError("public FINETUNE_DAGML_SELECTION_METRICS drifted from DAG-ML lowering")


@dataclass(frozen=True)
class TuningConformalScoreCalibration:
    """Typed development calibration cohort for conformal-aware tuning scores.

    This cohort is used only inside ``run(tuning=...).score_data`` to rank
    candidates with a temporary split conformal calibrator. It is not the final
    ``run(..., calibration=...)`` calibration payload.
    """

    X: Any | None = None
    X_calibration: Any = None
    features: Any = None
    y_true: Any | None = None
    y: Any | None = None
    y_calibration: Any = None
    target: Any = None
    targets: Any = None
    sample_ids: Any = None
    calibration_sample_ids: Any = None
    physical_sample_ids: Any = None
    groups: Any = None
    calibration_groups: Any = None
    metadata: Any = None
    calibration_metadata: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Return the mapping consumed by ``score_data.conformal_calibration``."""

        X = _single_alias_value(
            "TuningConformalScoreCalibration features",
            X=self.X,
            X_calibration=self.X_calibration,
            features=self.features,
        )
        if X is None:
            raise ValueError("TuningConformalScoreCalibration requires X, X_calibration, or features")
        y_true = _single_alias_value(
            "TuningConformalScoreCalibration target",
            y_true=self.y_true,
            y=self.y,
            y_calibration=self.y_calibration,
            target=self.target,
            targets=self.targets,
        )
        if y_true is None:
            raise ValueError("TuningConformalScoreCalibration requires y_true, y, y_calibration, target, or targets")
        payload = {"X": X, "y_true": y_true}
        _set_if_not_none(
            payload,
            "sample_ids",
            _single_alias_value(
                "TuningConformalScoreCalibration sample ids",
                sample_ids=self.sample_ids,
                calibration_sample_ids=self.calibration_sample_ids,
                physical_sample_ids=self.physical_sample_ids,
            ),
        )
        _set_if_not_none(
            payload,
            "groups",
            _single_alias_value(
                "TuningConformalScoreCalibration groups",
                groups=self.groups,
                calibration_groups=self.calibration_groups,
            ),
        )
        metadata = _single_alias_value(
            "TuningConformalScoreCalibration metadata",
            metadata=self.metadata,
            calibration_metadata=self.calibration_metadata,
        )
        _set_if_not_none(
            payload,
            "metadata",
            None if metadata is None else _canonical_metadata_payload(metadata, "TuningConformalScoreCalibration.metadata"),
        )
        return payload


@dataclass(frozen=True)
class TuningScoreData:
    """Typed public score cohort for ``run(tuning=...)``.

    Use either explicit ``X``/``y`` arrays, a positional ``tuple`` at the call
    site, or an explicit dataset-backed source via ``dataset`` + ``selector``.
    The object serializes to the mapping form consumed by the existing runtime.
    """

    X: Any | None = None
    X_score: Any | None = None
    y: Any | None = None
    y_score: Any | None = None
    metric: str | None = None
    score_metric: str | None = None
    sample_ids: Any = None
    score_sample_ids: Any = None
    prediction_sample_ids: Any = None
    physical_sample_ids: Any = None
    groups: Any = None
    score_groups: Any = None
    metadata: Any = None
    score_metadata: Any = None
    dataset: Any | None = None
    selector: Mapping[str, Any] | None = None
    sample_id_column: str | None = None
    group_column: str | None = None
    metadata_columns: str | Sequence[str] | None = None
    include_augmented: bool = False
    conformal_calibration: TuningConformalScoreCalibration | Mapping[str, Any] | None = None
    conformal_coverage: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the public mapping consumed by ``run(tuning=...)``."""

        payload = _dataset_backed_payload(self)
        if payload is None:
            explicit_xy = self.X is not None or self.y is not None
            explicit_score_xy = self.X_score is not None or self.y_score is not None
            if explicit_xy and explicit_score_xy:
                raise ValueError("TuningScoreData received multiple aliases (X/y, X_score/y_score); provide exactly one")
            if explicit_xy:
                if self.X is None or self.y is None:
                    raise ValueError("TuningScoreData requires both X and y")
                payload = {"X": self.X, "y": self.y}
            elif explicit_score_xy:
                if self.X_score is None or self.y_score is None:
                    raise ValueError("TuningScoreData requires both X_score and y_score")
                payload = {"X": self.X_score, "y": self.y_score}
            else:
                raise ValueError("TuningScoreData requires X/y, X_score/y_score, or dataset/selector")
        _set_if_not_none(
            payload,
            "metric",
            _public_text_string(
                metric,
                "TuningScoreData.metric",
                lowercase=True,
            )
            if (metric := _single_alias_value("TuningScoreData metric", metric=self.metric, score_metric=self.score_metric)) is not None
            else None,
        )
        _set_if_not_none(
            payload,
            "sample_ids",
            _single_alias_value(
                "TuningScoreData sample ids",
                sample_ids=self.sample_ids,
                score_sample_ids=self.score_sample_ids,
                prediction_sample_ids=self.prediction_sample_ids,
                physical_sample_ids=self.physical_sample_ids,
            ),
        )
        _set_if_not_none(payload, "groups", _single_alias_value("TuningScoreData groups", groups=self.groups, score_groups=self.score_groups))
        metadata = _single_alias_value("TuningScoreData metadata", metadata=self.metadata, score_metadata=self.score_metadata)
        _set_if_not_none(
            payload,
            "metadata",
            None if metadata is None else _canonical_metadata_payload(metadata, "TuningScoreData.metadata"),
        )
        _set_if_not_none(payload, "conformal_calibration", _coerce_public_payload(self.conformal_calibration))
        _set_if_not_none(
            payload,
            "conformal_coverage",
            None if self.conformal_coverage is None else _coverage_scalar_payload(self.conformal_coverage, "TuningScoreData.conformal_coverage"),
        )
        return payload


@dataclass(frozen=True)
class TuningWinner:
    """Typed public winner projection cohort for ``run(tuning=...)``."""

    X: Any | None = None
    y_true: Any | None = None
    score: float | None = None
    metric: str | None = None
    sample_ids: Any = None
    winner_sample_ids: Any = None
    prediction_sample_ids: Any = None
    physical_sample_ids: Any = None
    dataset_name: str = "tuning_winner"
    model_name: str | None = None
    task_type: str = "regression"
    metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None
    dataset: Any | None = None
    selector: Mapping[str, Any] | None = None
    sample_id_column: str | None = None
    group_column: str | None = None
    metadata_columns: str | Sequence[str] | None = None
    include_augmented: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the public mapping consumed by ``run(tuning=...).winner``."""

        payload = _dataset_backed_payload(self)
        if payload is None:
            if self.X is None or self.y_true is None:
                raise ValueError("TuningWinner requires X/y_true or dataset/selector")
            payload = {"X": self.X, "y_true": self.y_true}
        _set_if_not_none(payload, "score", None if self.score is None else _finite_number(self.score, "TuningWinner.score"))
        _set_if_not_none(
            payload,
            "metric",
            None if self.metric is None else _public_text_string(self.metric, "TuningWinner.metric", lowercase=True),
        )
        _set_if_not_none(
            payload,
            "sample_ids",
            _single_alias_value(
                "TuningWinner sample ids",
                sample_ids=self.sample_ids,
                winner_sample_ids=self.winner_sample_ids,
                prediction_sample_ids=self.prediction_sample_ids,
                physical_sample_ids=self.physical_sample_ids,
            ),
        )
        _set_if_not_none(payload, "dataset_name", _public_text_string(self.dataset_name, "TuningWinner.dataset_name"))
        _set_if_not_none(payload, "model_name", None if self.model_name is None else _public_text_string(self.model_name, "TuningWinner.model_name"))
        _set_if_not_none(payload, "task_type", _public_text_string(self.task_type, "TuningWinner.task_type", lowercase=True))
        _set_if_not_none(payload, "metadata", None if self.metadata is None else _canonical_metadata_payload(self.metadata, "TuningWinner.metadata"))
        return payload


@dataclass(frozen=True)
class TuningCalibration:
    """Typed public conformal calibration payload nested under ``tuning``."""

    y_pred: Any
    prediction_sample_ids: Any
    coverage: float | Sequence[float] = 0.9
    method: str = "split_absolute_residual"
    unit: str = "physical_sample"
    workspace_conformal_id: str | None = None
    workspace_metadata: Mapping[str, Any] | None = None
    as_predict_result: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the public mapping consumed by ``run(tuning=...).calibration``."""

        method = _supported_lower_string(self.method, "TuningCalibration.method", SUPPORTED_CONFORMAL_METHODS)
        unit = _supported_lower_string(self.unit, "TuningCalibration.unit", SUPPORTED_CONFORMAL_UNITS)
        if not isinstance(self.as_predict_result, bool):
            raise ValueError("TuningCalibration.as_predict_result must be a boolean")
        payload = {
            "as_predict_result": self.as_predict_result,
            "coverage": _coverage_payload(self.coverage, "TuningCalibration.coverage"),
            "method": method,
            "prediction_sample_ids": self.prediction_sample_ids,
            "unit": unit,
            "y_pred": self.y_pred,
        }
        _set_if_not_none(
            payload,
            "workspace_conformal_id",
            None
            if self.workspace_conformal_id is None
            else _canonical_optional_string(
                self.workspace_conformal_id,
                "TuningCalibration.workspace_conformal_id",
            ),
        )
        _set_if_not_none(payload, "workspace_metadata", None if self.workspace_metadata is None else _canonical_json_mapping(self.workspace_metadata, "TuningCalibration.workspace_metadata"))
        extra = _canonical_json_mapping(self.extra, "TuningCalibration.extra")
        reserved = sorted(set(extra) & _TUNING_CALIBRATION_RESERVED_EXTRA_KEYS)
        if reserved:
            raise ValueError(f"TuningCalibration.extra must not override reserved keys {reserved}; run(tuning=...) derives calibration_data from winner")
        payload.update(extra)
        return payload


@dataclass(frozen=True)
class TuningPassthrough:
    """Typed JSON-native marker for an optional non-final preprocessing step.

    Use this in ``NativeTuning.space`` when a named preprocessing step should be
    searchable as a no-op without relying on the raw ``"passthrough"`` string.
    """

    def to_dict(self) -> dict[str, str]:
        """Return the public JSON-native marker consumed by ``run(tuning=...)``."""

        return {"kind": "passthrough"}


@dataclass(frozen=True)
class NativeTuning:
    """Typed public configuration for the current native ``run(tuning=...)`` lane."""

    space: Mapping[str, Any]
    force_params: Mapping[str, Any] | None = None
    engine: str = "optuna"
    metric: str = "rmse"
    direction: str = "minimize"
    n_trials: int = 50
    sampler: str | None = None
    pruner: str | None = None
    seed: int | None = None
    resume: bool = False
    storage: str | None = None
    study_name: str | None = None
    score_data: TuningScoreData | Mapping[str, Any] | tuple[Any, ...] | list[Any] | None = None
    winner: TuningWinner | Mapping[str, Any] | None = None
    calibration: TuningCalibration | Mapping[str, Any] | None = None
    workspace_tuning_id: str | None = None
    workspace_metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the public mapping consumed by ``run(tuning=...)``."""

        core_payload: dict[str, Any] = {
            "direction": self.direction,
            "engine": self.engine,
            "metric": self.metric,
            "n_trials": self.n_trials,
            "resume": self.resume,
            "space": {key: _coerce_space_value(value) for key, value in _canonical_path_mapping(self.space, "NativeTuning.space").items()},
        }
        _set_if_not_none(
            core_payload,
            "force_params",
            None if self.force_params is None else {key: _coerce_public_payload(value) for key, value in _canonical_path_mapping(self.force_params, "NativeTuning.force_params").items()},
        )
        _set_if_not_none(core_payload, "sampler", self.sampler)
        _set_if_not_none(core_payload, "pruner", self.pruner)
        _set_if_not_none(core_payload, "seed", self.seed)
        _set_if_not_none(core_payload, "storage", self.storage)
        _set_if_not_none(core_payload, "study_name", self.study_name)
        payload = parse_tuning_spec(core_payload, context="NativeTuning").to_dict()
        _set_if_not_none(payload, "score_data", _coerce_public_payload(self.score_data))
        _set_if_not_none(payload, "winner", _coerce_public_payload(self.winner))
        _set_if_not_none(payload, "calibration", _coerce_public_payload(self.calibration))
        _set_if_not_none(
            payload,
            "workspace_tuning_id",
            None
            if self.workspace_tuning_id is None
            else _canonical_optional_string(
                self.workspace_tuning_id,
                "NativeTuning.workspace_tuning_id",
            ),
        )
        _set_if_not_none(payload, "workspace_metadata", None if self.workspace_metadata is None else _canonical_json_mapping(self.workspace_metadata, "NativeTuning.workspace_metadata"))
        _validate_conformal_score_payload(payload)
        _validate_winner_payload(payload)
        _normalize_tuning_calibration_payload(payload)
        return payload

    def to_tuning_spec(self) -> DagMLTuningSpec:
        """Return the normalized deterministic tuning contract."""

        return parse_tuning_spec(_core_tuning_payload(self.to_dict()))

    def inspect_space(self) -> dict[str, Any]:
        """Return the public ordered ``tuning.space`` inspection artifact."""

        return inspect_tuning_space(self)


def inspect_tuning_space(tuning: DagMLTuningSpec | NativeTuning | Mapping[str, Any]) -> dict[str, Any]:
    """Inspect and fingerprint the ordered public tuning search space.

    The returned mapping is JSON-native and intended for docs, CLIs, Studio and
    bindings. It shows the canonical dotted patch paths, deterministic parameter
    order, optional ``force_params`` as decoded ``ParameterPatch`` records, the
    search-space fingerprint and the enclosing tuning-contract fingerprint.
    """

    tuning_payload = _coerce_public_payload(tuning)
    tuning_spec = tuning if isinstance(tuning, DagMLTuningSpec) else parse_tuning_spec(_core_tuning_payload(tuning_payload))
    ordered = tuning_spec.ordered_search_space
    artifact = ordered.to_dict()
    artifact["fingerprint"] = ordered.fingerprint
    artifact["tuning_fingerprint"] = tuning_spec.fingerprint
    artifact["force_params"] = [patch.to_dict() for patch in tuning_spec.parameter_patches(tuning_spec.force_params or {}, context="force_params")]
    return artifact


@dataclass(frozen=True)
class TunedSingleEstimatorConformalResult:
    """Result bundle returned by ``tune_single_estimator(..., calibration=...)``."""

    run: RunResult
    calibrated: CalibratedRunResult | PredictResult

    @property
    def tuning_result(self) -> TuningResult | None:
        """Native tuning result attached to the underlying run."""

        return self.run.tuning_result

    @property
    def tuning_id(self) -> str | None:
        """Workspace tuning id attached to the underlying run, if persisted."""

        return self.run.tuning_id

    @property
    def tuning_best_params(self) -> dict[str, Any]:
        """Best native tuning parameters attached to the underlying run."""

        return self.run.tuning_best_params

    @property
    def tuning_best_value(self) -> float | None:
        """Best native tuning objective value attached to the underlying run."""

        return self.run.tuning_best_value

    @property
    def interval_coverages(self) -> tuple[float, ...]:
        """Materialized conformal interval coverages on the calibrated result."""

        coverages = getattr(self.calibrated, "interval_coverages", None)
        if coverages is not None:
            return tuple(float(coverage) for coverage in coverages)
        prediction = getattr(self.calibrated, "prediction", None)
        prediction_coverages = getattr(prediction, "coverages", None)
        if prediction_coverages is not None:
            return tuple(float(coverage) for coverage in prediction_coverages)
        return ()

    @property
    def conformal_guarantee_status(self) -> dict[str, Any] | None:
        """Guarantee metadata exposed by the calibrated result, when present."""

        status = getattr(self.calibrated, "conformal_guarantee_status", None)
        if isinstance(status, Mapping):
            return dict(status)
        return status

    def interval(self, coverage: float) -> Any:
        """Return the materialized conformal interval for ``coverage``."""

        interval = getattr(self.calibrated, "interval", None)
        if callable(interval):
            return interval(coverage)
        prediction = getattr(self.calibrated, "prediction", None)
        prediction_interval = getattr(prediction, "interval", None)
        if callable(prediction_interval):
            return prediction_interval(coverage)
        raise TypeError("calibrated result does not expose materialized conformal intervals")

    def metrics(self, y_true: Any) -> dict[float, Any]:
        """Evaluate materialized conformal intervals against observed targets."""

        from nirs4all.api.calibrate import conformal_metrics

        return conformal_metrics(self.calibrated, y_true=y_true)

    def robustness(
        self,
        *,
        y_true: Any,
        X: Any | None = None,
        predictor: Any | None = None,
        mode: str = "clean_frozen",
        scenarios: Sequence[Any] | None = None,
        slice_by: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
        workspace_path: str | Path | None = None,
        workspace_name: str = "",
        workspace_robustness_id: str | None = None,
        workspace_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Compute a robustness report from the calibrated prediction result."""

        return self.calibrated.robustness(
            y_true=y_true,
            X=X,
            predictor=predictor,
            mode=mode,
            scenarios=scenarios,
            slice_by=slice_by,
            metadata=metadata,
            seed=seed,
            workspace_path=workspace_path,
            workspace_name=workspace_name,
            workspace_robustness_id=workspace_robustness_id,
            workspace_metadata=workspace_metadata,
        )


def tune_single_estimator(
    pipeline: Any,
    X: Any,
    y: Any,
    tuning: DagMLTuningSpec | NativeTuning | Mapping[str, Any],
    *,
    score_extractor: Callable[[Any], float] | None = None,
    X_score: Any | None = None,
    y_score: Any | None = None,
    score_metric: str | None = None,
    score_sample_ids: Any = None,
    score_groups: Any = None,
    score_metadata: Any = None,
    sample_ids: Any = None,
    groups: Any = None,
    metadata: Any = None,
    clone_estimator: bool = True,
    refit: bool = True,
    workspace_path: str | Path | None = None,
    workspace_name: str = "",
    workspace_tuning_id: str | None = None,
    workspace_metadata: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    pipeline_id: str | None = None,
    chain_id: str | None = None,
    resume_tuning_result: TuningResult | None = None,
    resume_tuning_id: str | None = None,
    per_dataset: Mapping[str, Any] | None = None,
    winner_x: Any | None = None,
    winner_y_true: Any | None = None,
    winner_score: float | None = None,
    winner_metric: str | None = None,
    winner_sample_ids: Sequence[Any] | None = None,
    winner_dataset_name: str = "tuning_winner",
    winner_model_name: str | None = None,
    winner_task_type: str = "regression",
    winner_metadata: Mapping[str, Any] | None = None,
    calibration: TuningCalibration | Mapping[str, Any] | None = None,
) -> RunResult | TunedSingleEstimatorConformalResult:
    """Tune the explicit native linear lane and return a ``RunResult``.

    This is the narrow Python surface for the currently implemented native DAG-ML
    tuning seams. It accepts only the single-estimator and linear
    transformer→estimator forms supported by ``compile_pipeline_objective`` and
    is not a replacement for the planned full ``run(tuning=...)`` pipeline
    compiler.

    Scoring is explicit. Provide exactly one of:

    * a custom ``score_extractor`` callable;
    * explicit ``X_score``/``y_score`` arguments;
    * ``NativeTuning(score_data=...)`` or a mapping ``tuning["score_data"]``.

    When a cohort is supplied, the scorer uses ``score_metric``, the metric
    nested in ``score_data``, or the metric from ``tuning``. Optional scoring
    identities are validated for row alignment and forwarded to ``predict()``
    when the fitted estimator accepts those keyword arguments.
    """

    tuning_payload = _coerce_public_payload(tuning)
    tuning_spec = tuning if isinstance(tuning, DagMLTuningSpec) else parse_tuning_spec(_core_tuning_payload(tuning_payload))
    tuning_score_data = _tuning_score_data_payload(tuning_payload)
    tuning_winner = _tuning_winner_payload(tuning_payload)
    tuning_calibration = _tuning_calibration_payload(tuning_payload)
    explicit_score_args = X_score is not None or y_score is not None or score_metric is not None or score_sample_ids is not None or score_groups is not None or score_metadata is not None
    if score_extractor is not None and (explicit_score_args or tuning_score_data is not None):
        raise ValueError("provide exactly one of score_extractor, X_score/y_score scoring arguments, or tuning.score_data")
    if tuning_score_data is not None and explicit_score_args:
        raise ValueError("provide either tuning.score_data or X_score/y_score scoring arguments, not both")
    if tuning_winner is not None:
        if _has_explicit_winner_args(
            winner_x=winner_x,
            winner_y_true=winner_y_true,
            winner_score=winner_score,
            winner_metric=winner_metric,
            winner_sample_ids=winner_sample_ids,
            winner_dataset_name=winner_dataset_name,
            winner_model_name=winner_model_name,
            winner_task_type=winner_task_type,
            winner_metadata=winner_metadata,
        ):
            raise ValueError("provide either tuning.winner or winner_* arguments, not both")
        (
            winner_x,
            winner_y_true,
            winner_score,
            winner_metric,
            winner_sample_ids,
            winner_dataset_name,
            winner_model_name,
            winner_task_type,
            winner_metadata,
        ) = _winner_args_from_tuning_winner(tuning_winner)
    if tuning_calibration is not None:
        if calibration is not None:
            raise ValueError("provide conformal calibration either as calibration=... or tuning.calibration, not both")
        calibration = tuning_calibration
    if score_extractor is None:
        if tuning_score_data is not None:
            score_extractor = _make_score_extractor_from_score_data(tuning_score_data, tuning_spec.metric)
        else:
            if X_score is None or y_score is None:
                raise ValueError("tune_single_estimator requires score_extractor, tuning.score_data, or both X_score and y_score")
            from nirs4all.pipeline.dagml.pipeline_objective import make_prediction_score_extractor

            score_extractor = make_prediction_score_extractor(
                score_metric or tuning_spec.metric,
                X_score,
                y_score,
                sample_ids=score_sample_ids,
                groups=score_groups,
                metadata=score_metadata,
            )
    if resume_tuning_result is not None and _tuning_fingerprint_without_resume(resume_tuning_result.tuning) != _tuning_fingerprint_without_resume(tuning_spec):
        raise ValueError("resume_tuning_result does not match the requested tuning contract")

    from nirs4all.pipeline.dagml.pipeline_objective_compiler import run_single_estimator_tuning_to_run_result

    run_result = run_single_estimator_tuning_to_run_result(
        pipeline,
        X,
        y,
        tuning_spec,
        score_extractor=score_extractor,
        sample_ids=sample_ids,
        groups=groups,
        metadata=metadata,
        clone_estimator=clone_estimator,
        refit=refit,
        workspace_path=workspace_path,
        workspace_name=workspace_name,
        workspace_tuning_id=workspace_tuning_id,
        workspace_metadata=workspace_metadata,
        run_id=run_id,
        pipeline_id=pipeline_id,
        chain_id=chain_id,
        resume_tuning_result=resume_tuning_result,
        resume_tuning_id=resume_tuning_id,
        per_dataset=per_dataset,
        winner_x=winner_x,
        winner_y_true=winner_y_true,
        winner_score=winner_score,
        winner_metric=winner_metric,
        winner_sample_ids=winner_sample_ids,
        winner_dataset_name=winner_dataset_name,
        winner_model_name=winner_model_name,
        winner_task_type=winner_task_type,
        winner_metadata=winner_metadata,
    )
    if calibration is None:
        return run_result
    calibration_payload = _coerce_public_payload(calibration)
    if not isinstance(calibration_payload, Mapping):
        raise TypeError("calibration must be a TuningCalibration helper or a mapping of nirs4all.calibrate keyword arguments")
    if "calibration_data" in calibration_payload:
        raise ValueError("calibration payload must not include calibration_data; tune_single_estimator uses the projected winner entry")
    if not run_result.best:
        raise ValueError("calibration requires a projected winner entry; provide winner_x, winner_y_true, winner_score, winner_metric and winner_sample_ids")

    from nirs4all.api.calibrate import calibrate

    calibration_kwargs = dict(calibration_payload)
    calibration_kwargs.setdefault("as_predict_result", True)
    result_metadata = calibration_kwargs.get("result_metadata")
    if result_metadata is None:
        result_metadata_payload: dict[str, Any] = {}
    elif isinstance(result_metadata, Mapping):
        result_metadata_payload = dict(result_metadata)
    else:
        raise TypeError("calibration.result_metadata must be a mapping")
    result_metadata_payload.setdefault(
        "tuning_calibration_source",
        {
            "source": "tuning.winner",
            "score_data_role": "hpo_objective_only",
            "score_data_used": False,
        },
    )
    calibration_kwargs["result_metadata"] = result_metadata_payload
    calibrated = calibrate(
        calibration_data=run_result.best,
        **calibration_kwargs,
    )
    return TunedSingleEstimatorConformalResult(run=run_result, calibrated=calibrated)


def save_workspace_tuning_result(
    workspace_path: str | Path,
    result: TuningResult,
    *,
    name: str = "",
    tuning_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    pipeline_id: str | None = None,
    chain_id: str | None = None,
) -> str:
    """Persist a native DAG-ML tuning result in a nirs4all workspace."""

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    store = WorkspaceStore(Path(workspace_path))
    try:
        return store.save_tuning_result(
            result,
            name=name,
            tuning_id=tuning_id,
            metadata=metadata,
            run_id=run_id,
            pipeline_id=pipeline_id,
            chain_id=chain_id,
        )
    finally:
        store.close()


def load_workspace_tuning_result(workspace_path: str | Path, tuning_id: str) -> TuningResult:
    """Load a verified native DAG-ML tuning result from a nirs4all workspace."""

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    store = WorkspaceStore(Path(workspace_path))
    try:
        result = store.load_tuning_result(tuning_id)
    finally:
        store.close()
    if not isinstance(result, TuningResult):
        raise TypeError("workspace did not return a TuningResult")
    return result


__all__ = [
    "CONFORMAL_TUNING_SCORE_METRICS",
    "FINETUNE_APPROACHES",
    "FINETUNE_DAGML_APPROACHES",
    "FINETUNE_DAGML_DETERMINISTIC_ENGINES",
    "FINETUNE_DAGML_EVAL_MODES",
    "FINETUNE_DAGML_META_KEYS",
    "FINETUNE_DAGML_SELECTION_METRICS",
    "FINETUNE_ENGINES",
    "FINETUNE_ENGINE_ALIASES",
    "FINETUNE_EVAL_MODE_ALIASES",
    "FINETUNE_EVAL_MODES",
    "FINETUNE_N4M_PRUNERS",
    "FINETUNE_N4M_SAMPLERS",
    "FINETUNE_OPTUNA_PRUNERS",
    "FINETUNE_OPTUNA_SAMPLERS",
    "FINETUNE_SAMPLER_KEY_ALIASES",
    "FinetuneApproach",
    "FinetuneEngine",
    "FinetuneEvalMode",
    "FinetunePruner",
    "FinetuneSampler",
    "inspect_tuning_space",
    "NativeTuning",
    "OrderedSearchSpaceSpec",
    "ParameterPatch",
    "SearchSpaceParameter",
    "TUNING_SPACE_SCHEMA_ID",
    "TuningCalibration",
    "TuningConformalScoreCalibration",
    "TuningDirection",
    "TuningEngine",
    "TUNING_CONTRACT_KEYS",
    "TUNING_DIRECTIONS",
    "TUNING_ENGINES",
    "TUNING_OPTIMIZER_PERSISTENCE_KEYS",
    "TuningPassthrough",
    "TUNING_RUNTIME_KEYS",
    "TuningScoreData",
    "TuningWinner",
    "TunedSingleEstimatorConformalResult",
    "get_tuning_space_schema",
    "load_workspace_tuning_result",
    "save_workspace_tuning_result",
    "tune_single_estimator",
    "tuning_space_schema_json",
]


def _tuning_fingerprint_without_resume(tuning: DagMLTuningSpec) -> str:
    """Return the contract fingerprint with the operational resume bit cleared."""

    payload = tuning.to_dict()
    payload["resume"] = False
    return parse_tuning_spec(payload).fingerprint


def _coerce_public_payload(value: Any) -> Any:
    if value is None:
        return None
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return value


def _coerce_space_value(value: Any) -> Any:
    value = _coerce_public_payload(value)
    if isinstance(value, Mapping):
        return {key: _coerce_space_value(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_coerce_space_value(nested) for nested in value]
    if isinstance(value, tuple):
        return tuple(_coerce_space_value(nested) for nested in value)
    return value


def _core_tuning_payload(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, DagMLTuningSpec):
        return payload.to_dict()
    if not isinstance(payload, Mapping):
        raise TypeError("tuning must be a mapping, DagMLTuningSpec, or NativeTuning")
    return {
        key: payload[key]
        for key in (
            "direction",
            "engine",
            "force_params",
            "metric",
            "n_trials",
            "pruner",
            "resume",
            "sampler",
            "seed",
            "space",
            "storage",
            "study_name",
        )
        if key in payload
    }


def _tuning_score_data_payload(tuning_payload: Any) -> Any | None:
    if isinstance(tuning_payload, Mapping):
        return _coerce_public_payload(tuning_payload.get("score_data"))
    return None


def _tuning_winner_payload(tuning_payload: Any) -> Any | None:
    if isinstance(tuning_payload, Mapping):
        return _coerce_public_payload(tuning_payload.get("winner"))
    return None


def _tuning_calibration_payload(tuning_payload: Any) -> Any | None:
    if isinstance(tuning_payload, Mapping):
        return _coerce_public_payload(tuning_payload.get("calibration"))
    return None


def _has_explicit_winner_args(
    *,
    winner_x: Any | None,
    winner_y_true: Any | None,
    winner_score: float | None,
    winner_metric: str | None,
    winner_sample_ids: Sequence[Any] | None,
    winner_dataset_name: str,
    winner_model_name: str | None,
    winner_task_type: str,
    winner_metadata: Mapping[str, Any] | None,
) -> bool:
    return (
        winner_x is not None
        or winner_y_true is not None
        or winner_score is not None
        or winner_metric is not None
        or winner_sample_ids is not None
        or winner_dataset_name != "tuning_winner"
        or winner_model_name is not None
        or winner_task_type != "regression"
        or winner_metadata is not None
    )


def _winner_args_from_tuning_winner(
    winner: Any,
) -> tuple[Any | None, Any | None, float | None, str | None, Any, str, str | None, str, Mapping[str, Any] | None]:
    if not isinstance(winner, Mapping):
        raise TypeError("tuning.winner must be a TuningWinner helper or mapping")
    if "dataset" in winner or "spectro_dataset" in winner:
        raise ValueError("tune_single_estimator tuning.winner supports explicit X/y_true arrays; use run(tuning=...) for dataset-backed winner")
    winner_x = _mapping_single_alias_value(winner, "tuning.winner features", "X", "x", "winner_x")
    winner_y_true = _mapping_single_alias_value(winner, "tuning.winner target", "y_true", "winner_y_true")
    winner_score = _mapping_single_alias_value(winner, "tuning.winner score", "score", "winner_score")
    winner_metric = _mapping_single_alias_value(winner, "tuning.winner metric", "metric", "winner_metric")
    winner_sample_ids = _mapping_single_alias_value(winner, "tuning.winner sample_ids", "sample_ids", "winner_sample_ids", "prediction_sample_ids", "physical_sample_ids")
    winner_dataset_name = _mapping_single_alias_value(winner, "tuning.winner dataset_name", "dataset_name", "winner_dataset_name") or "tuning_winner"
    winner_model_name = _mapping_single_alias_value(winner, "tuning.winner model_name", "model_name", "winner_model_name")
    winner_task_type = _mapping_single_alias_value(winner, "tuning.winner task_type", "task_type", "winner_task_type") or "regression"
    winner_metadata = _mapping_single_alias_value(winner, "tuning.winner metadata", "metadata", "winner_metadata")
    if winner_metadata is not None and not isinstance(winner_metadata, Mapping):
        raise ValueError("tuning.winner metadata must be a mapping")
    return (
        winner_x,
        winner_y_true,
        winner_score,
        None if winner_metric is None else _public_text_string(winner_metric, "tuning.winner.metric", lowercase=True),
        winner_sample_ids,
        _public_text_string(winner_dataset_name, "tuning.winner.dataset_name"),
        None if winner_model_name is None else _public_text_string(winner_model_name, "tuning.winner.model_name"),
        _public_text_string(winner_task_type, "tuning.winner.task_type", lowercase=True),
        winner_metadata,
    )


def _make_score_extractor_from_score_data(score_data: Any, tuning_metric: str) -> Callable[[Any], float]:
    if isinstance(score_data, tuple | list):
        X_score, y_score, score_sample_ids, score_groups, score_metadata = _score_data_tuple_parts(score_data)
        from nirs4all.pipeline.dagml.pipeline_objective import make_prediction_score_extractor

        return make_prediction_score_extractor(
            tuning_metric,
            X_score,
            y_score,
            sample_ids=score_sample_ids,
            groups=score_groups,
            metadata=score_metadata,
        )
    if not isinstance(score_data, Mapping):
        raise ValueError("tuning.score_data must be a mapping or tuple/list")
    if "dataset" in score_data or "spectro_dataset" in score_data:
        raise ValueError("tune_single_estimator tuning.score_data supports explicit X/y arrays or tuple/list forms; use run(tuning=...) for dataset-backed score_data")

    X_score = _mapping_single_alias_value(score_data, "score_data features", "X", "X_score")
    y_score = _mapping_single_alias_value(score_data, "score_data target", "y", "y_score")
    if X_score is None or y_score is None:
        raise ValueError("tuning.score_data requires X/y or X_score/y_score")
    raw_score_metric = _mapping_single_alias_value(score_data, "score_data metric", "metric", "score_metric")
    score_metric = _public_text_string(raw_score_metric, "score_data metric", lowercase=True) if raw_score_metric is not None else _public_text_string(tuning_metric, "tuning.metric", lowercase=True)
    score_sample_ids = _mapping_single_alias_value(score_data, "score_data sample_ids", "sample_ids", "score_sample_ids", "prediction_sample_ids", "physical_sample_ids")
    score_groups = _mapping_single_alias_value(score_data, "score_data groups", "groups", "score_groups")
    score_metadata = _mapping_single_alias_value(score_data, "score_data metadata", "metadata", "score_metadata")
    conformal_calibration = _mapping_single_alias_value(
        score_data,
        "score_data conformal_calibration",
        "conformal_calibration",
        "conformal_score_calibration",
    )
    if conformal_calibration is not None:
        from nirs4all.pipeline.dagml.pipeline_objective import make_conformal_prediction_score_extractor

        _validate_conformal_score_calibration_mapping(conformal_calibration)
        coverage = _mapping_first(score_data, "conformal_coverage", "coverage")
        return make_conformal_prediction_score_extractor(
            score_metric,
            X_score,
            y_score,
            conformal_calibration,
            coverage=0.9 if coverage is None else _coverage_scalar_payload(coverage, "score_data.conformal_coverage"),
            sample_ids=score_sample_ids,
            groups=score_groups,
            metadata=score_metadata,
        )

    from nirs4all.pipeline.dagml.pipeline_objective import make_prediction_score_extractor

    return make_prediction_score_extractor(
        score_metric,
        X_score,
        y_score,
        sample_ids=score_sample_ids,
        groups=score_groups,
        metadata=score_metadata,
    )


def _score_data_tuple_parts(score_data: tuple[Any, ...] | list[Any]) -> tuple[Any, Any, Any, Any, Any]:
    normalized = _canonical_score_data_sequence(score_data, "tuning.score_data")
    return (
        normalized[0],
        normalized[1],
        normalized[2] if len(normalized) > 2 else None,
        normalized[3] if len(normalized) > 3 else None,
        normalized[4] if len(normalized) > 4 else None,
    )


def _mapping_first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return _coerce_public_payload(mapping[key])
    return None


def _mapping_single_alias_value(mapping: Mapping[str, Any], label: str, *keys: str) -> Any:
    provided = [(key, _coerce_public_payload(mapping[key])) for key in keys if key in mapping and mapping[key] is not None]
    if not provided:
        return None
    if len(provided) > 1:
        names = ", ".join(key for key, _value in provided)
        raise ValueError(f"{label} received multiple aliases ({names}); provide exactly one")
    return provided[0][1]


def _validate_conformal_score_payload(payload: MutableMapping[str, Any]) -> None:
    score_data = payload.get("score_data")
    tuning_metric = _normalized_metric(payload.get("metric"))
    if score_data is None:
        return
    if isinstance(score_data, tuple | list):
        if tuning_metric in _CONFORMAL_TUNING_SCORE_METRICS:
            raise ValueError("NativeTuning with a conformal metric requires mapping score_data.conformal_calibration")
        if isinstance(payload, dict):
            payload["score_data"] = _canonical_score_data_sequence(score_data, "NativeTuning.score_data")
        return
    if not isinstance(score_data, Mapping):
        if tuning_metric in _CONFORMAL_TUNING_SCORE_METRICS:
            raise ValueError("NativeTuning with a conformal metric requires mapping score_data.conformal_calibration")
        raise TypeError("NativeTuning.score_data must be a mapping, tuple, or list")
    score_data = dict(score_data)
    payload["score_data"] = score_data
    _validate_dataset_backed_public_mapping(score_data, "NativeTuning.score_data")

    conformal_calibration_present = "conformal_calibration" in score_data or "conformal_score_calibration" in score_data
    coverage_present = "conformal_coverage" in score_data or "coverage" in score_data
    _validate_optional_mapping_alias_group(score_data, "score_data metric", ("metric", "score_metric"))
    _validate_optional_mapping_alias_group(
        score_data,
        "score_data sample_ids",
        ("sample_ids", "score_sample_ids", "prediction_sample_ids", "physical_sample_ids"),
    )
    _validate_optional_mapping_alias_group(score_data, "score_data groups", ("groups", "score_groups"))
    score_metadata = _mapping_single_alias_value(score_data, "score_data metadata", "metadata", "score_metadata")
    if score_metadata is not None:
        _canonical_metadata_payload(score_metadata, "NativeTuning.score_data.metadata")
    if "dataset" not in score_data and "spectro_dataset" not in score_data:
        _validate_optional_mapping_alias_group(score_data, "score_data features", ("X", "X_score"))
        _validate_optional_mapping_alias_group(score_data, "score_data target", ("y", "y_score"))
    raw_score_metric = _mapping_single_alias_value(score_data, "score_data metric", "metric", "score_metric")
    score_metric = None
    if raw_score_metric is not None:
        score_metric = _public_text_string(raw_score_metric, "NativeTuning.score_data.metric", lowercase=True)
        _set_canonical_alias_value(score_data, ("metric", "score_metric"), "metric", score_metric)
    effective_metric = score_metric or tuning_metric
    if conformal_calibration_present:
        conformal_calibration = _mapping_single_alias_value(
            score_data,
            "score_data conformal_calibration",
            "conformal_calibration",
            "conformal_score_calibration",
        )
        _validate_conformal_score_calibration_mapping(conformal_calibration)
        if coverage_present:
            coverage = score_data["conformal_coverage"] if "conformal_coverage" in score_data else score_data["coverage"]
            _coverage_scalar_payload(coverage, "score_data.conformal_coverage")
        if effective_metric not in _CONFORMAL_TUNING_SCORE_METRICS:
            raise ValueError(f"score_data.conformal_calibration requires a conformal score metric ({', '.join(sorted(_CONFORMAL_TUNING_SCORE_METRICS))})")
        return
    if coverage_present:
        raise ValueError("score_data.conformal_coverage requires score_data.conformal_calibration")
    if effective_metric in _CONFORMAL_TUNING_SCORE_METRICS:
        raise ValueError("conformal tuning score metrics require score_data.conformal_calibration")


def _validate_conformal_score_calibration_mapping(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("score_data.conformal_calibration must be a mapping")
    _validate_required_mapping_alias_group(
        value,
        "score_data.conformal_calibration features",
        ("X", "X_calibration", "features"),
    )
    _validate_required_mapping_alias_group(
        value,
        "score_data.conformal_calibration target",
        ("y_true", "y", "y_calibration", "target", "targets"),
    )
    _validate_optional_mapping_alias_group(
        value,
        "score_data.conformal_calibration sample_ids",
        ("sample_ids", "calibration_sample_ids", "physical_sample_ids"),
    )
    _validate_optional_mapping_alias_group(value, "score_data.conformal_calibration groups", ("groups", "calibration_groups"))
    metadata = _mapping_single_alias_value(
        value,
        "score_data.conformal_calibration metadata",
        "metadata",
        "calibration_metadata",
    )
    if metadata is not None:
        _canonical_metadata_payload(metadata, "score_data.conformal_calibration.metadata")


def _validate_winner_payload(payload: MutableMapping[str, Any]) -> None:
    winner = payload.get("winner")
    if winner is None:
        return
    if not isinstance(winner, Mapping):
        raise TypeError("NativeTuning.winner must be a mapping")
    winner = dict(winner)
    payload["winner"] = winner
    dataset_backed = "dataset" in winner or "spectro_dataset" in winner
    _validate_dataset_backed_public_mapping(winner, "NativeTuning.winner")
    _validate_optional_mapping_alias_group(winner, "winner dataset", ("dataset", "spectro_dataset"))
    _validate_optional_mapping_alias_group(winner, "winner features", ("X", "x", "winner_x"))
    _validate_optional_mapping_alias_group(winner, "winner target", ("y_true", "winner_y_true"))
    _validate_optional_mapping_alias_group(winner, "winner score", ("score", "winner_score"))
    _validate_optional_mapping_alias_group(winner, "winner metric", ("metric", "winner_metric"))
    _validate_optional_mapping_alias_group(
        winner,
        "winner sample_ids",
        ("sample_ids", "winner_sample_ids", "prediction_sample_ids", "physical_sample_ids"),
    )
    _validate_optional_mapping_alias_group(winner, "winner dataset_name", ("dataset_name", "winner_dataset_name"))
    _validate_optional_mapping_alias_group(winner, "winner model_name", ("model_name", "winner_model_name"))
    _validate_optional_mapping_alias_group(winner, "winner task_type", ("task_type", "winner_task_type"))
    winner_metadata = _mapping_single_alias_value(winner, "winner metadata", "metadata", "winner_metadata")
    if winner_metadata is not None:
        _canonical_metadata_payload(winner_metadata, "NativeTuning.winner.metadata")
    winner_score = _mapping_single_alias_value(winner, "winner score", "score", "winner_score")
    if winner_score is not None:
        _finite_number(winner_score, "NativeTuning.winner.score")
    winner_metric = _mapping_single_alias_value(winner, "winner metric", "metric", "winner_metric")
    if winner_metric is not None:
        _set_canonical_alias_value(
            winner,
            ("metric", "winner_metric"),
            "metric",
            _public_text_string(winner_metric, "NativeTuning.winner.metric", lowercase=True),
        )
    winner_dataset_name = _mapping_single_alias_value(winner, "winner dataset_name", "dataset_name", "winner_dataset_name")
    if winner_dataset_name is not None:
        _set_canonical_alias_value(
            winner,
            ("dataset_name", "winner_dataset_name"),
            "dataset_name",
            _public_text_string(winner_dataset_name, "NativeTuning.winner.dataset_name"),
        )
    winner_model_name = _mapping_single_alias_value(winner, "winner model_name", "model_name", "winner_model_name")
    if winner_model_name is not None:
        _set_canonical_alias_value(
            winner,
            ("model_name", "winner_model_name"),
            "model_name",
            _public_text_string(winner_model_name, "NativeTuning.winner.model_name"),
        )
    winner_task_type = _mapping_single_alias_value(winner, "winner task_type", "task_type", "winner_task_type")
    if winner_task_type is not None:
        _set_canonical_alias_value(
            winner,
            ("task_type", "winner_task_type"),
            "task_type",
            _public_text_string(winner_task_type, "NativeTuning.winner.task_type", lowercase=True),
        )
    if dataset_backed and any(key in winner for key in ("X", "x", "winner_x", "y_true", "winner_y_true")):
        raise ValueError("NativeTuning.winner dataset-backed form must not also provide X/y_true arrays")


def _normalize_tuning_calibration_payload(payload: dict[str, Any]) -> None:
    calibration = payload.get("calibration")
    if calibration is None:
        return
    if not isinstance(calibration, Mapping):
        raise TypeError("NativeTuning.calibration must be a mapping")
    normalized = _canonical_mapping(calibration, "NativeTuning.calibration")
    if "calibration_data" in calibration:
        raise ValueError("NativeTuning.calibration must not include calibration_data; run(tuning=...) derives calibration_data from winner")
    if "coverage" in normalized:
        normalized["coverage"] = _coverage_payload(normalized["coverage"], "NativeTuning.calibration.coverage")
    if "method" in normalized:
        normalized["method"] = _supported_lower_string(normalized["method"], "NativeTuning.calibration.method", SUPPORTED_CONFORMAL_METHODS)
    if "unit" in normalized:
        normalized["unit"] = _supported_lower_string(normalized["unit"], "NativeTuning.calibration.unit", SUPPORTED_CONFORMAL_UNITS)
    if "workspace_metadata" in normalized and normalized["workspace_metadata"] is not None:
        normalized["workspace_metadata"] = _canonical_json_mapping(normalized["workspace_metadata"], "NativeTuning.calibration.workspace_metadata")
    if "workspace_conformal_id" in normalized and normalized["workspace_conformal_id"] is not None:
        normalized["workspace_conformal_id"] = _canonical_optional_string(
            normalized["workspace_conformal_id"],
            "NativeTuning.calibration.workspace_conformal_id",
        )
    if "as_predict_result" in normalized and not isinstance(normalized["as_predict_result"], bool):
        raise ValueError("NativeTuning.calibration.as_predict_result must be a boolean")
    payload["calibration"] = normalized


def _validate_dataset_backed_public_mapping(mapping: Mapping[str, Any], label: str) -> None:
    dataset_backed = "dataset" in mapping or "spectro_dataset" in mapping
    if not dataset_backed:
        return
    _validate_optional_mapping_alias_group(mapping, f"{label} dataset", ("dataset", "spectro_dataset"))
    if "selector" not in mapping or mapping["selector"] is None:
        raise ValueError(f"{label} dataset-backed form requires selector")
    _canonical_json_mapping(mapping["selector"], f"{label}.selector")
    if "include_augmented" in mapping and not isinstance(mapping["include_augmented"], bool):
        raise ValueError(f"{label}.include_augmented must be a boolean")
    if "sample_id_column" in mapping and mapping["sample_id_column"] is not None:
        _set_mapping_value_if_mutable(
            mapping,
            "sample_id_column",
            _canonical_optional_string(mapping["sample_id_column"], f"{label}.sample_id_column"),
        )
    if "group_column" in mapping and mapping["group_column"] is not None:
        _set_mapping_value_if_mutable(
            mapping,
            "group_column",
            _canonical_optional_string(mapping["group_column"], f"{label}.group_column"),
        )
    if "metadata_columns" in mapping and mapping["metadata_columns"] is not None:
        _set_mapping_value_if_mutable(
            mapping,
            "metadata_columns",
            _canonical_metadata_columns(mapping["metadata_columns"], f"{label}.metadata_columns"),
        )


def _validate_optional_mapping_alias_group(mapping: Mapping[str, Any], label: str, keys: tuple[str, ...]) -> None:
    _mapping_single_alias_value(mapping, label, *keys)


def _validate_required_mapping_alias_group(mapping: Mapping[str, Any], label: str, keys: tuple[str, ...]) -> None:
    if _mapping_single_alias_value(mapping, label, *keys) is None:
        raise ValueError(f"{label} requires one of {list(keys)}")


def _normalized_metric(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _dataset_backed_payload(value: TuningScoreData | TuningWinner) -> dict[str, Any] | None:
    if value.dataset is None:
        return None
    if isinstance(value, TuningScoreData) and (value.X is not None or value.y is not None or value.X_score is not None or value.y_score is not None):
        raise ValueError("TuningScoreData dataset-backed form must not also provide X/y or X_score/y_score")
    if isinstance(value, TuningWinner) and (value.X is not None or value.y_true is not None):
        raise ValueError("TuningWinner dataset-backed form must not also provide X/y_true")
    if value.selector is None:
        raise ValueError(f"{value.__class__.__name__} dataset-backed form requires selector")
    if not isinstance(value.include_augmented, bool):
        raise ValueError(f"{value.__class__.__name__}.include_augmented must be a boolean")
    payload: dict[str, Any] = {
        "dataset": value.dataset,
        "include_augmented": value.include_augmented,
        "selector": _canonical_json_mapping(value.selector, f"{value.__class__.__name__}.selector"),
    }
    label = value.__class__.__name__
    _set_if_not_none(
        payload,
        "sample_id_column",
        None if value.sample_id_column is None else _canonical_optional_string(value.sample_id_column, f"{label}.sample_id_column"),
    )
    _set_if_not_none(
        payload,
        "group_column",
        None if value.group_column is None else _canonical_optional_string(value.group_column, f"{label}.group_column"),
    )
    _set_if_not_none(
        payload,
        "metadata_columns",
        None if value.metadata_columns is None else _canonical_metadata_columns(value.metadata_columns, f"{label}.metadata_columns"),
    )
    return payload


def _set_if_not_none(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


def _set_mapping_value_if_mutable(mapping: Mapping[str, Any], key: str, value: Any) -> None:
    if isinstance(mapping, dict):
        mapping[key] = value


def _single_alias_value(label: str, **values: Any) -> Any:
    provided = [(key, value) for key, value in values.items() if value is not None]
    if not provided:
        return None
    if len(provided) > 1:
        keys = ", ".join(key for key, _value in provided)
        raise ValueError(f"{label} received multiple aliases ({keys}); provide exactly one")
    return provided[0][1]


def _canonical_mapping(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip() or key != key.strip() or "\x00" in key:
            raise ValueError(f"{label} keys must be canonical non-empty strings")
        if key in normalized:
            raise ValueError(f"{label} contains duplicate keys")
        normalized[key] = value
    return normalized


def _canonical_json_mapping(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    normalized = _canonical_mapping(payload, label)
    return {key: _strict_json_native_value(value, f"{label}[{key}]") for key, value in normalized.items()}


def _canonical_metadata_payload(payload: Any, label: str) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        return _canonical_json_mapping(payload, label)
    if isinstance(payload, str | bytes):
        raise ValueError(f"{label} must be a mapping or a sequence of mappings")
    if not isinstance(payload, Sequence):
        raise ValueError(f"{label} must be a mapping or a sequence of mappings")
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"{label}[{index}] must be a mapping")
        normalized_rows.append(_canonical_json_mapping(row, f"{label}[{index}]"))
    return normalized_rows


def _canonical_path_mapping(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{label} keys must be non-empty strings")
        canonical = key.strip()
        if canonical in normalized:
            raise ValueError(f"{label} contains duplicate keys after canonicalization")
        normalized[canonical] = value
    return normalized


def _strict_json_native_value(value: Any, label: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int | float):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{label} must be JSON-native and finite")
        return value
    if isinstance(value, bytes | tuple | set | frozenset):
        raise ValueError(f"{label} must be JSON-native")
    if isinstance(value, Mapping):
        return _canonical_json_mapping(value, label)
    if isinstance(value, list):
        return [_strict_json_native_value(item, f"{label}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{label} must be JSON-native")


def _canonical_optional_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a canonical non-empty string")
    return value


def _canonical_metadata_columns(value: Any, label: str) -> str | list[str]:
    if isinstance(value, str):
        return _canonical_optional_string(value, label)
    if isinstance(value, bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a canonical string or a sequence of canonical strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for index, column in enumerate(value):
        canonical = _canonical_optional_string(column, f"{label}[{index}]")
        if canonical in seen:
            raise ValueError(f"{label} contains duplicate column names")
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


def _canonical_score_data_sequence(value: tuple[Any, ...] | list[Any], label: str) -> list[Any]:
    if len(value) < 2:
        raise ValueError(f"{label} tuple/list must contain (X_score, y_score)")
    if len(value) > 5:
        raise ValueError(f"{label} tuple/list supports at most (X_score, y_score, sample_ids, groups, metadata)")
    normalized = list(value)
    if len(normalized) > 4 and normalized[4] is not None:
        normalized[4] = _canonical_metadata_payload(normalized[4], f"{label}.metadata")
    return normalized


def _public_text_string(value: Any, label: str, *, lowercase: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a non-empty string")
    normalized = value.strip()
    if not normalized or "\x00" in normalized:
        raise ValueError(f"{label} must be a non-empty string")
    return normalized.lower() if lowercase else normalized


def _set_canonical_alias_value(mapping: dict[str, Any], keys: tuple[str, ...], canonical_key: str, value: Any) -> None:
    for key in keys:
        if key != canonical_key:
            mapping.pop(key, None)
    mapping[canonical_key] = value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number")
    return numeric


def _supported_lower_string(value: Any, label: str, supported: frozenset[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be one of {sorted(supported)}")
    normalized = value.strip().lower()
    if normalized not in supported:
        raise ValueError(f"{label} must be one of {sorted(supported)}")
    return normalized


def _coverage_payload(value: Any, label: str) -> float | list[float]:
    if isinstance(value, bool):
        raise ValueError(f"{label} values must be finite floats in the open interval (0, 1)")
    if isinstance(value, int | float):
        return _coverage_float(value, label)
    if isinstance(value, str | bytes):
        raise ValueError(f"{label} must be a float or a sequence of floats")
    try:
        raw_values = list(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a float or a sequence of floats") from exc
    if not raw_values:
        raise ValueError(f"{label} requires at least one value")
    values = [_coverage_float(item, label) for item in raw_values]
    if len(set(values)) != len(values):
        raise ValueError(f"{label} list values must be unique")
    return values


def _coverage_scalar_payload(value: Any, label: str) -> float:
    if isinstance(value, bool | str | bytes):
        raise ValueError(f"{label} must be a finite float in the open interval (0, 1)")
    return _coverage_float(value, label)


def _coverage_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} values must be finite floats in the open interval (0, 1)")
    coverage = float(value)
    if not math.isfinite(coverage) or coverage <= 0.0 or coverage >= 1.0:
        raise ValueError(f"{label} values must be finite floats in the open interval (0, 1)")
    return coverage
