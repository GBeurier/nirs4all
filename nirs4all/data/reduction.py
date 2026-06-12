"""Prediction levels, unit ids and the full ``ReductionPlan`` contract.

This module is the roadmap *reduction socle* (N0 minimal adapter, completed in
N4): a single, typed entry point that the existing reduction mechanisms
(``Predictions.aggregate``, ``Predictions.top(by_repetition=...)`` and
``TestAggregation``) become adapters to, **without changing any legacy output**.
It also introduces internal prediction-level / prediction-unit metadata and the
typed ``fold_id`` accessor that later phases use to de-overload ``fold_id``.

The N4 completion adds, on top of the N0 adapter:

* the full role / axis / level / method taxonomy
  (``role=score|persist|fold_ensemble|meta_feature|final_output``,
  ``axis=unit|fold|model|metric``,
  ``method=mean|median|vote|weighted_mean|robust|custom``);
* ``weight_source`` and ``task_compatibility`` metadata;
* deterministic ``reduction_id`` / :meth:`ReductionPlan.fingerprint` /
  :meth:`ReductionPlan.to_dict` / :meth:`ReductionPlan.from_dict` manifest
  support;
* a serialisable fit/replay state contract
  (:class:`FitScope`, :class:`ReducerState`) for any reducer / transform that
  estimates parameters, with a hard leakage guard refusing state fitted on a
  validation / test partition.

Only **leakage-free, group-stateless** methods are executable in this phase
(``mean`` / ``median`` / ``vote`` / ``robust`` and ``weighted_mean`` when the
weights are supplied as inputs). Fitable reducers carry a contract and a state
holder; executing one requires an explicit, non-leaked fitted state. Everything
else (``custom`` without a reducer, the non-``unit`` axes) fails loud. None of
this is wired into the persisted prediction format: the additive columns and
public kwargs stay inert metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Prediction levels and scopes
# ---------------------------------------------------------------------------


class PredictionLevel(StrEnum):
    """Typed level a prediction row describes.

    The main user-facing output is always :attr:`SAMPLE`; the other levels are
    intermediate / diagnostic / meta-feature levels.

    Attributes:
        ROW: A raw materialised row (e.g. one cartesian combo before reduction).
        OBSERVATION: One ``(sample, source, rep)`` measurement.
        SOURCE: One ``(sample, source)`` aggregate.
        COMBO: One derived cartesian unit.
        SAMPLE: One physical sample (the deliverable level).
    """

    ROW = "row"
    OBSERVATION = "observation"
    SOURCE = "source"
    COMBO = "combo"
    SAMPLE = "sample"


class PredictionScope(StrEnum):
    """Role of a prediction entry, derived from the legacy ``fold_id``.

    Attributes:
        OOF: An out-of-fold cross-validation prediction (or an aggregate of OOF
            predictions such as the ``avg`` / ``w_avg`` twins).
        REFIT: A prediction produced by a final refit model (``fold_id="final"``).
        TEST: A held-out test-partition prediction.
    """

    OOF = "oof"
    REFIT = "refit"
    TEST = "test"


class EvaluationScope(StrEnum):
    """Unit level at which metrics and model selection are evaluated."""

    ROW = "row"
    OBSERVATION = "observation"
    COMBO = "combo"
    SAMPLE = "sample"


@dataclass(frozen=True)
class PredictionUnitId:
    """Typed key for a prediction / reduction unit.

    A light, hashable identity used to reason about *what* a prediction row
    represents, independent of the legacy ``fold_id`` overloading.

    Attributes:
        level: The :class:`PredictionLevel` of this unit.
        physical_sample_id: The owning physical sample, if applicable.
        source_id: The source, for observation / source levels.
        rep_id: The repetition index, for observation level.
        derived_unit_id: The combo id, for combo level.
    """

    level: PredictionLevel
    physical_sample_id: str | None = None
    source_id: str | None = None
    rep_id: int | None = None
    derived_unit_id: str | None = None

    def as_key(self) -> str:
        """Return a canonical string key for this unit."""
        if self.level is PredictionLevel.SAMPLE:
            return f"sample:{self.physical_sample_id}"
        if self.level is PredictionLevel.SOURCE:
            return f"source_sample:{self.source_id}:{self.physical_sample_id}"
        if self.level is PredictionLevel.OBSERVATION:
            return f"source_observation:{self.source_id}:{self.physical_sample_id}:{self.rep_id}"
        if self.level is PredictionLevel.COMBO:
            return f"derived_combo:{self.derived_unit_id}"
        return f"row:{self.physical_sample_id}:{self.source_id}:{self.rep_id}"

    def __str__(self) -> str:
        return self.as_key()


# ---------------------------------------------------------------------------
# Typed fold_id accessor (de-overloading fold_id additively)
# ---------------------------------------------------------------------------

#: Legacy ``fold_id`` values that are NOT real CV folds.
#:
#: Real CV fold ids are stringified non-negative integers (``"0"``, ``"1"`` ...).
#: The values below are pseudo-predictions overloaded onto ``fold_id`` today:
#:
#: * ``"final"`` / ``"final_agg"`` -- final refit model (and its aggregated twin);
#: * ``"avg"`` -- RMSECV averaged fold built from concatenated OOF predictions;
#: * ``"w_avg"`` -- weighted-average fold;
#: * ``"ensemble"`` / ``"all"`` -- ensemble / all-fold pseudo entries;
#: * any value ending in ``"_agg"`` -- a repetition-aggregated twin.
PSEUDO_FOLD_IDS: frozenset[str] = frozenset({"final", "final_agg", "avg", "w_avg", "ensemble", "all"})

#: Legacy ``fold_id`` values that denote a final refit entry.
REFIT_FOLD_IDS: frozenset[str] = frozenset({"final", "final_agg"})


def is_real_cv_fold(fold_id: Any) -> bool:
    """Return ``True`` if ``fold_id`` denotes a real cross-validation fold.

    Real CV folds are stringified non-negative integers. Pseudo-folds
    (``"final"``, ``"avg"``, ``"w_avg"``, ``"ensemble"``, ``"all"`` and any
    ``"*_agg"`` twin) return ``False``.

    Args:
        fold_id: The legacy ``fold_id`` value (``str``, ``int`` or ``None``).

    Returns:
        Whether the id is a genuine CV fold.
    """
    if fold_id is None:
        return False
    fid = str(fold_id).strip()
    if not fid or fid in PSEUDO_FOLD_IDS or fid.endswith("_agg"):
        return False
    try:
        return int(fid) >= 0
    except ValueError:
        return False


def prediction_scope_from_legacy(
    fold_id: Any,
    *,
    partition: str | None = None,
    is_refit: bool | None = None,
) -> PredictionScope:
    """Derive a typed :class:`PredictionScope` from legacy fields.

    This is an additive accessor: it does not change ``fold_id`` values, it only
    interprets them. Refit entries (``fold_id="final"`` / ``"final_agg"`` or an
    explicit ``is_refit``) map to :attr:`PredictionScope.REFIT`; otherwise a
    test partition maps to :attr:`PredictionScope.TEST`; everything else (CV fold
    predictions and their OOF aggregates) maps to :attr:`PredictionScope.OOF`.

    Args:
        fold_id: The legacy ``fold_id`` value.
        partition: The partition of the prediction (``"train"`` / ``"val"`` /
            ``"test"``), if known.
        is_refit: Explicit refit marker overriding the ``fold_id`` heuristic.

    Returns:
        The interpreted scope.
    """
    fid = "" if fold_id is None else str(fold_id).strip()
    if is_refit or fid in REFIT_FOLD_IDS or fid.startswith("final"):
        return PredictionScope.REFIT
    if partition is not None and str(partition).lower() == "test":
        return PredictionScope.TEST
    return PredictionScope.OOF


def normalize_evaluation_scope(value: Any) -> EvaluationScope | None:
    """Return a typed evaluation scope from user-facing strings."""
    if value is None:
        return None
    raw = str(getattr(value, "value", value)).strip().lower()
    aliases = {
        "rows": "row",
        "raw": "row",
        "obs": "observation",
        "observations": "observation",
        "combo_level": "combo",
        "combos": "combo",
        "sample_level": "sample",
        "samples": "sample",
    }
    raw = aliases.get(raw, raw)
    try:
        return EvaluationScope(raw)
    except ValueError as exc:
        allowed = ", ".join(scope.value for scope in EvaluationScope)
        raise ReductionError(f"Unknown evaluation_scope={value!r}; expected one of: {allowed}.") from exc


# ---------------------------------------------------------------------------
# ReductionPlan taxonomy
# ---------------------------------------------------------------------------


class ReductionRole(StrEnum):
    """Why a reduction is performed (one impl, several roles)."""

    SCORE = "score"
    PERSIST = "persist"
    FOLD_ENSEMBLE = "fold_ensemble"
    META_FEATURE = "meta_feature"
    FINAL_OUTPUT = "final_output"


class ReductionAxis(StrEnum):
    """Which axis a reduction collapses."""

    UNIT = "unit"
    FOLD = "fold"
    MODEL = "model"
    METRIC = "metric"


class ReductionMethod(StrEnum):
    """Reduction method.

    ``mean`` / ``median`` / ``vote`` mirror :meth:`Predictions.aggregate` exactly.
    ``robust`` is a MAD-outlier-excluded mean (leakage-free, per group).
    ``weighted_mean`` requires the weights to be supplied as inputs. ``custom``
    is declared-but-not-executable without an explicit reducer.
    """

    MEAN = "mean"
    MEDIAN = "median"
    VOTE = "vote"
    WEIGHTED_MEAN = "weighted_mean"
    ROBUST = "robust"
    CUSTOM = "custom"


class TaskCompatibility(StrEnum):
    """Task families a reduction method is meaningful for."""

    ANY = "any"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class FitScope(StrEnum):
    """Whether a reducer / transform estimates parameters, and from where.

    Attributes:
        STATELESS: No parameters are learned; the reduction is computed within
            each group at evaluation time and is leakage-free by construction.
        FOLD_TRAIN: Parameters are estimated on a CV fold's *train* partition and
            replayed on that fold's validation rows.
        FULL_TRAIN_REFIT: Parameters are estimated on the full training set during
            the final refit, producing a state separate from the per-fold states.
    """

    STATELESS = "stateless"
    FOLD_TRAIN = "fold_train"
    FULL_TRAIN_REFIT = "full_train_refit"


#: Partitions a fitable reducer must never be *fit* on (only replayed on).
_LEAKY_FIT_PARTITIONS: frozenset[str] = frozenset({"val", "valid", "validation", "test", "holdout"})

#: Default task compatibility per method.
_DEFAULT_TASK_COMPATIBILITY: dict[ReductionMethod, TaskCompatibility] = {
    ReductionMethod.MEAN: TaskCompatibility.ANY,
    ReductionMethod.MEDIAN: TaskCompatibility.REGRESSION,
    ReductionMethod.VOTE: TaskCompatibility.CLASSIFICATION,
    ReductionMethod.WEIGHTED_MEAN: TaskCompatibility.REGRESSION,
    ReductionMethod.ROBUST: TaskCompatibility.REGRESSION,
    ReductionMethod.CUSTOM: TaskCompatibility.ANY,
}

# Mapping from legacy ``TestAggregation`` values to a (method, weight_source,
# custom_name) fold-axis descriptor. ``weighted`` / ``best`` are fold-ensemble
# axes keyed by the per-fold ``val_score``; ``best`` selects a single fold and
# is therefore a non-averaging ``custom`` strategy, not one of the safe means.
_TEST_AGGREGATION_DESCRIPTOR: dict[str, tuple[ReductionMethod, str | None, str | None]] = {
    "mean": (ReductionMethod.MEAN, None, None),
    "weighted": (ReductionMethod.WEIGHTED_MEAN, "val_score", None),
    "best": (ReductionMethod.CUSTOM, "val_score", "best_fold"),
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReductionError(ValueError):
    """Raised when a reduction contract is misused or not executable.

    Subclasses :class:`ValueError` so existing boundary handlers treat it
    uniformly. Carries an optional machine-readable ``code`` in the
    ``REDUCE-Exxx`` style used across the relational modules.

    Attributes:
        code: Short actionable error code, e.g. ``"REDUCE-E001"``.
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message if code is None else f"{message} [Error: {code}]")
        self.code = code


class LeakedReducerStateError(ReductionError):
    """Raised when a fitable reducer carries a state estimated out-of-fold.

    Parameters of QC / outlier / trimmed-robust / imputer / calibration reducers
    must be fitted on a training partition and replayed on validation / test.
    A state fitted on a validation / test partition is a leakage path and is
    refused.
    """


# ---------------------------------------------------------------------------
# Fit / replay state contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReducerState:
    """Serialisable fitted-state holder for a parameter-estimating reducer.

    A pure data contract: it records *where* the parameters were fitted and the
    learned parameters themselves, so the state can be persisted and replayed.
    Validation (the leakage guard) is explicit via :meth:`validate` rather than
    enforced at construction, so a leaked state can be represented and then
    rejected by the contract.

    Attributes:
        fit_scope: How the parameters were estimated (see :class:`FitScope`).
        fit_partition: The partition the parameters were *fitted* on (e.g.
            ``"train"``). Validation / test partitions are leakage and rejected.
        fold_id: The CV fold the fold-train state belongs to (``None`` for a
            full-train refit state).
        parameters: The learned parameters (JSON-serialisable).
        state_id: Stable identifier; derived from :meth:`fingerprint` when empty.
    """

    fit_scope: FitScope = FitScope.STATELESS
    fit_partition: str | None = None
    fold_id: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    state_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "fit_scope", FitScope(self.fit_scope))
        if not self.state_id:
            object.__setattr__(self, "state_id", f"state-{self.fingerprint()[:16]}")

    def validate(self) -> None:
        """Refuse leaked or structurally inconsistent fitted state.

        Raises:
            LeakedReducerStateError: If the state was fitted on a validation /
                test partition.
            ReductionError: If the fold/scope combination is inconsistent (a
                full-train refit state pinned to a real CV fold, or a fold-train
                state pinned to a pseudo fold).
        """
        if self.fit_scope is FitScope.STATELESS:
            return
        if self.fit_partition is not None and str(self.fit_partition).strip().lower() in _LEAKY_FIT_PARTITIONS:
            raise LeakedReducerStateError(
                f"Reducer state was fitted on partition {self.fit_partition!r}; fitable reducers must be "
                "fitted on a training partition and replayed on validation/test, never fitted out-of-fold.",
                code="REDUCE-E010",
            )
        if self.fit_scope is FitScope.FULL_TRAIN_REFIT and is_real_cv_fold(self.fold_id):
            raise ReductionError(
                f"full_train_refit state must not be pinned to a real CV fold (fold_id={self.fold_id!r}); "
                "the refit state is a separate full-train state.",
                code="REDUCE-E011",
            )
        if self.fit_scope is FitScope.FOLD_TRAIN and self.fold_id is not None and not is_real_cv_fold(self.fold_id):
            raise ReductionError(
                f"fold_train state declares fold_id={self.fold_id!r}, which is not a real CV fold; a "
                "fold-train state must reference the fold it was fitted on.",
                code="REDUCE-E012",
            )

    def fingerprint(self) -> str:
        """Deterministic SHA-256 over the canonical state descriptor."""
        payload = json.dumps(
            {
                "fit_scope": FitScope(self.fit_scope).value,
                "fit_partition": self.fit_partition,
                "fold_id": None if self.fold_id is None else str(self.fold_id),
                "parameters": _canonical_mapping(self.parameters),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the state for manifests / lineage."""
        return {
            "fit_scope": FitScope(self.fit_scope).value,
            "fit_partition": self.fit_partition,
            "fold_id": None if self.fold_id is None else str(self.fold_id),
            "parameters": dict(self.parameters),
            "state_id": self.state_id,
            "parameter_fingerprint": self.fingerprint(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReducerState:
        """Reconstruct a :class:`ReducerState` from :meth:`to_dict` output."""
        return cls(
            fit_scope=FitScope(data.get("fit_scope", FitScope.STATELESS)),
            fit_partition=data.get("fit_partition"),
            fold_id=data.get("fold_id"),
            parameters=dict(data.get("parameters", {})),
            state_id=str(data.get("state_id", "")),
        )


# ---------------------------------------------------------------------------
# ReductionPlan
# ---------------------------------------------------------------------------


@dataclass
class ReductionPlan:
    """Canonical reduction contract that generalises the existing mechanisms.

    Wraps the existing reduction semantics so scoring, ranking, refit and
    reporting can converge on one object. The ``unit`` axis is executable and,
    for the safe leakage-free methods, byte-for-byte identical to the legacy
    :meth:`Predictions.aggregate` path. Other axes are declared metadata until
    their phase lands.

    Attributes:
        role: Why the reduction is applied.
        axis: Which axis it collapses (only ``unit`` is executable here).
        method: ``mean`` / ``median`` / ``vote`` / ``weighted_mean`` / ``robust``
            / ``custom``.
        input_level: The level the reduction consumes.
        output_level: The level it produces.
        exclude_outliers: Whether to drop per-group outliers before reducing.
        outlier_threshold: Confidence level for outlier detection.
        weight_source: Column / source of weights (e.g. ``"val_score"`` for the
            fold-ensemble axis, ``"sample_influence_weight"`` for the unit axis).
        task_compatibility: Task family this reduction is meaningful for. Derived
            from ``method`` when left as ``None``.
        fit_scope: Whether this reduction estimates parameters and from where.
        fit_partition: The partition a fitable reduction is fitted on.
        fold_id: The fold a fold-train reduction belongs to.
        custom_name: Identifier of a ``custom`` strategy (e.g. ``"best_fold"``).
        reduction_id: Stable identifier; derived from :meth:`fingerprint` when
            left empty.
    """

    role: ReductionRole = ReductionRole.SCORE
    axis: ReductionAxis = ReductionAxis.UNIT
    method: ReductionMethod = ReductionMethod.MEAN
    input_level: PredictionLevel = PredictionLevel.OBSERVATION
    output_level: PredictionLevel = PredictionLevel.SAMPLE
    exclude_outliers: bool = False
    outlier_threshold: float = 0.95
    weight_source: str | None = None
    task_compatibility: TaskCompatibility | None = None
    fit_scope: FitScope = FitScope.STATELESS
    fit_partition: str | None = None
    fold_id: str | None = None
    custom_name: str | None = None
    reduction_id: str = ""

    def __post_init__(self) -> None:
        self.role = ReductionRole(self.role)
        self.axis = ReductionAxis(self.axis)
        self.method = ReductionMethod(self.method)
        self.input_level = PredictionLevel(self.input_level)
        self.output_level = PredictionLevel(self.output_level)
        self.fit_scope = FitScope(self.fit_scope)
        if self.task_compatibility is None:
            self.task_compatibility = _DEFAULT_TASK_COMPATIBILITY[self.method]
        else:
            self.task_compatibility = TaskCompatibility(self.task_compatibility)
        if not self.reduction_id:
            self.reduction_id = f"reduce-{self.fingerprint()[:16]}"

    # -- construction adapters --------------------------------------------

    @classmethod
    def from_legacy_aggregate(
        cls,
        method: str = "mean",
        *,
        exclude_outliers: bool = False,
        outlier_threshold: float = 0.95,
        role: ReductionRole | str = ReductionRole.SCORE,
    ) -> ReductionPlan:
        """Build a plan equivalent to a legacy ``Predictions.aggregate`` call."""
        return cls(
            role=ReductionRole(role),
            axis=ReductionAxis.UNIT,
            method=ReductionMethod(method),
            exclude_outliers=exclude_outliers,
            outlier_threshold=outlier_threshold,
        )

    @classmethod
    def from_by_repetition(
        cls,
        method: str | None = "mean",
        *,
        exclude_outliers: bool = False,
    ) -> ReductionPlan:
        """Build a plan equivalent to ``top(by_repetition=..., method=...)``."""
        return cls.from_legacy_aggregate(method or "mean", exclude_outliers=exclude_outliers)

    @classmethod
    def from_test_aggregation(cls, test_aggregation: Any) -> ReductionPlan:
        """Build a fold-axis plan from a legacy ``TestAggregation`` value.

        Captures the fold-ensemble role/method/weight metadata faithfully. The
        ``fold`` axis is not executable in this phase, so the plan is metadata
        only: ``weighted`` keeps ``method=weighted_mean`` with
        ``weight_source="val_score"`` and ``best`` is the non-averaging
        ``custom`` best-fold selection (also keyed by ``val_score``).

        Args:
            test_aggregation: A ``TestAggregation`` enum or its string value
                (``"mean"`` / ``"weighted"`` / ``"best"``).

        Returns:
            A :class:`ReductionPlan` on the ``fold`` axis.
        """
        value = getattr(test_aggregation, "value", test_aggregation)
        method, weight_source, custom_name = _TEST_AGGREGATION_DESCRIPTOR.get(
            str(value), (ReductionMethod.MEAN, None, None)
        )
        return cls(
            role=ReductionRole.FOLD_ENSEMBLE,
            axis=ReductionAxis.FOLD,
            method=method,
            input_level=PredictionLevel.SAMPLE,
            output_level=PredictionLevel.SAMPLE,
            weight_source=weight_source,
            custom_name=custom_name,
        )

    # -- task compatibility -----------------------------------------------

    def is_task_compatible(self, task_type: Any) -> bool:
        """Whether this reduction is meaningful for ``task_type``.

        Unknown task types are not blocked (returns ``True``); only an explicit
        regression/classification mismatch is reported.
        """
        assert self.task_compatibility is not None  # set in __post_init__
        if self.task_compatibility is TaskCompatibility.ANY:
            return True
        norm = _normalize_task(task_type)
        if norm is None:
            return True
        return norm == self.task_compatibility.value

    def validate_task(self, task_type: Any) -> None:
        """Raise if this reduction is incompatible with ``task_type``."""
        if not self.is_task_compatible(task_type):
            raise ReductionError(
                f"ReductionPlan.method={self.method.value!r} is declared for "
                f"task_compatibility={self.task_compatibility.value!r} but was applied to task "  # type: ignore[union-attr]
                f"{task_type!r}.",
                code="REDUCE-E001",
            )

    # -- legacy aggregation bridge ----------------------------------------

    def as_repetition_aggregation(self) -> tuple[str, bool]:
        """Return the ``(method, exclude_outliers)`` for the by_repetition path.

        Maps the plan onto the legacy ``Predictions.top(by_repetition=...)``
        aggregation knobs. Only the leakage-free, group-stateless methods are
        expressible there. ``robust`` becomes an outlier-excluded mean;
        ``weighted_mean`` / ``custom`` cannot be expressed on the legacy path and
        fail loud.

        Raises:
            ReductionError: If the method cannot drive by_repetition aggregation.
        """
        if self.method in (ReductionMethod.MEAN, ReductionMethod.MEDIAN, ReductionMethod.VOTE):
            return self.method.value, self.exclude_outliers
        if self.method is ReductionMethod.ROBUST:
            return "mean", True
        raise ReductionError(
            f"ReductionPlan.method={self.method.value!r} cannot drive by_repetition aggregation; only "
            "mean/median/vote/robust are expressible on the legacy aggregation path.",
            code="REDUCE-E003",
        )

    # -- execution ---------------------------------------------------------

    def reduce(
        self,
        y_pred: np.ndarray,
        group_ids: np.ndarray,
        *,
        y_proba: np.ndarray | None = None,
        y_true: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        task_type: Any | None = None,
        state: ReducerState | None = None,
    ) -> dict[str, Any]:
        """Reduce predictions by group.

        For the safe leakage-free methods on the ``unit`` axis, the output is
        identical to the legacy :meth:`Predictions.aggregate` call. Fitable plans
        (``fit_scope != stateless``) require an explicit, non-leaked
        :class:`ReducerState`; unsupported axes and ``custom`` fail loud.

        Args:
            y_pred: Predicted values ``(n_samples,)``.
            group_ids: Group identifiers ``(n_samples,)``.
            y_proba: Optional class probabilities.
            y_true: Optional true values.
            weights: Optional per-row weights (required for ``weighted_mean``).
            task_type: Optional task type; validated against
                ``task_compatibility`` when provided.
            state: Optional fitted state for a fitable plan.

        Returns:
            The aggregation dict (``y_pred``, ``group_ids``, ``group_sizes`` and
            optional ``y_proba`` / ``y_true`` / ``outliers_excluded``).

        Raises:
            NotImplementedError: If the axis is not yet executable.
            ReductionError: For an unsupported method or a missing fitted state.
            LeakedReducerStateError: If the supplied state leaks.
        """
        if self.axis is not ReductionAxis.UNIT:
            raise NotImplementedError(
                f"ReductionPlan axis={self.axis.value!r} is declared but only the 'unit' axis is "
                "executable in this phase."
            )
        if task_type is not None:
            self.validate_task(task_type)
        self._require_valid_state(state)

        # Lazy import to avoid a heavy import cycle with predictions.py.
        from nirs4all.data.predictions import Predictions

        if self.method in (ReductionMethod.MEAN, ReductionMethod.MEDIAN, ReductionMethod.VOTE):
            aggregated: dict[str, Any] = Predictions.aggregate(
                y_pred=y_pred,
                group_ids=group_ids,
                y_proba=y_proba,
                y_true=y_true,
                method=self.method.value,
                exclude_outliers=self.exclude_outliers,
                outlier_threshold=self.outlier_threshold,
            )
            return aggregated
        if self.method is ReductionMethod.ROBUST:
            # MAD-outlier-excluded mean: leakage-free, computed within each group.
            robust: dict[str, Any] = Predictions.aggregate(
                y_pred=y_pred,
                group_ids=group_ids,
                y_proba=y_proba,
                y_true=y_true,
                method="mean",
                exclude_outliers=True,
                outlier_threshold=self.outlier_threshold,
            )
            return robust
        if self.method is ReductionMethod.WEIGHTED_MEAN:
            if weights is None:
                raise ReductionError(
                    "ReductionPlan.method='weighted_mean' requires per-row weights; pass weights= or set "
                    "a weight_source the caller can resolve.",
                    code="REDUCE-E004",
                )
            return _weighted_reduce(y_pred, group_ids, weights, y_proba=y_proba, y_true=y_true)
        # CUSTOM and any future non-safe method.
        raise ReductionError(
            f"ReductionPlan.method={self.method.value!r} (custom_name={self.custom_name!r}) is not "
            "executable without an explicit reducer; custom reductions fail loud in this phase.",
            code="REDUCE-E002",
        )

    def _require_valid_state(self, state: ReducerState | None) -> None:
        """Enforce the fit/replay contract for fitable plans.

        Stateless plans ignore any state. Fitable plans (``fold_train`` /
        ``full_train_refit``) must replay an explicit, non-leaked state.
        """
        if self.fit_scope is FitScope.STATELESS:
            return
        if state is None:
            raise ReductionError(
                f"ReductionPlan declares fit_scope={self.fit_scope.value!r} but no fitted ReducerState was "
                "provided; a fitable reducer must replay an explicit train-fitted state.",
                code="REDUCE-E005",
            )
        state.validate()

    # -- serialisation -----------------------------------------------------

    def _descriptor(self) -> dict[str, Any]:
        """Canonical, reduction_id-independent descriptor for fingerprinting."""
        assert self.task_compatibility is not None  # set in __post_init__
        return {
            "role": ReductionRole(self.role).value,
            "axis": ReductionAxis(self.axis).value,
            "method": ReductionMethod(self.method).value,
            "input_level": PredictionLevel(self.input_level).value,
            "output_level": PredictionLevel(self.output_level).value,
            "exclude_outliers": bool(self.exclude_outliers),
            "outlier_threshold": _fingerprint_value(self.outlier_threshold),
            "weight_source": self.weight_source,
            "task_compatibility": TaskCompatibility(self.task_compatibility).value,
            "fit_scope": FitScope(self.fit_scope).value,
            "fit_partition": self.fit_partition,
            "fold_id": None if self.fold_id is None else str(self.fold_id),
            "custom_name": self.custom_name,
        }

    def fingerprint(self) -> str:
        """Deterministic SHA-256 over the canonical plan descriptor.

        The volatile ``reduction_id`` is excluded so the digest is a stable
        function of the plan's semantics.
        """
        payload = json.dumps(self._descriptor(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the plan for manifests / lineage."""
        out = self._descriptor()
        out["reduction_id"] = self.reduction_id
        out["fingerprint"] = self.fingerprint()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReductionPlan:
        """Reconstruct a :class:`ReductionPlan` from :meth:`to_dict` output.

        Tolerates the N0 dict shape (without the N4 fields) by relying on the
        dataclass defaults; the stored ``fingerprint`` is informational and not
        required for reconstruction.
        """
        return cls(
            role=ReductionRole(data.get("role", ReductionRole.SCORE)),
            axis=ReductionAxis(data.get("axis", ReductionAxis.UNIT)),
            method=ReductionMethod(data.get("method", ReductionMethod.MEAN)),
            input_level=PredictionLevel(data.get("input_level", PredictionLevel.OBSERVATION)),
            output_level=PredictionLevel(data.get("output_level", PredictionLevel.SAMPLE)),
            exclude_outliers=bool(data.get("exclude_outliers", False)),
            outlier_threshold=float(data.get("outlier_threshold", 0.95)),
            weight_source=data.get("weight_source"),
            task_compatibility=(
                TaskCompatibility(data["task_compatibility"])
                if data.get("task_compatibility") is not None
                else None
            ),
            fit_scope=FitScope(data.get("fit_scope", FitScope.STATELESS)),
            fit_partition=data.get("fit_partition"),
            fold_id=data.get("fold_id"),
            custom_name=data.get("custom_name"),
            reduction_id=str(data.get("reduction_id", "")),
        )

    def prediction_metadata(
        self,
        *,
        fold_id: Any | None = None,
        partition: str | None = None,
        evaluation_scope: str | None = None,
        is_refit: bool | None = None,
    ) -> dict[str, Any]:
        """Build additive (non-persisted) prediction metadata for this reduction.

        Ties the reduction to the typed :func:`prediction_scope_from_legacy`
        accessor introduced in N0. These keys are internal helpers; they are not
        forced into the persisted workspace schema.

        Args:
            fold_id: The legacy ``fold_id`` of the reduced entry. Defaults to the
                plan's own ``fold_id``.
            partition: The partition of the reduced entry, if known.
            evaluation_scope: The (reserved) evaluation scope label.
            is_refit: Explicit refit marker forwarded to the scope accessor.

        Returns:
            A dict with ``prediction_level``, ``prediction_scope``,
            ``evaluation_scope``, ``reduction_role`` and ``reduction_id``.
        """
        effective_fold = self.fold_id if fold_id is None else fold_id
        scope = prediction_scope_from_legacy(effective_fold, partition=partition, is_refit=is_refit)
        return {
            "prediction_level": PredictionLevel(self.output_level).value,
            "prediction_scope": scope.value,
            "evaluation_scope": evaluation_scope,
            "reduction_role": ReductionRole(self.role).value,
            "reduction_id": self.reduction_id,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _weighted_reduce(
    y_pred: np.ndarray,
    group_ids: np.ndarray,
    weights: np.ndarray,
    *,
    y_proba: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-group weighted mean of predictions (and optional proba / true).

    Weights are *inputs* (e.g. ``sample_influence_weight``), never learned, so
    this reduction is leakage-free. A group whose weights sum to zero falls back
    to the unweighted mean. The output mirrors :meth:`Predictions.aggregate`.
    """
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    group_ids = np.asarray(group_ids).flatten()
    weights = np.asarray(weights, dtype=float).flatten()
    if len(weights) != len(y_pred):
        raise ReductionError(
            f"weights length ({len(weights)}) must match y_pred length ({len(y_pred)}).",
            code="REDUCE-E004",
        )
    if len(y_pred) != len(group_ids):
        raise ReductionError(
            f"Length mismatch: y_pred ({len(y_pred)}) != group_ids ({len(group_ids)}).",
            code="REDUCE-E004",
        )
    if np.any(weights < 0):
        raise ReductionError("weights must be non-negative for a weighted mean.", code="REDUCE-E004")

    unique_groups, inverse = np.unique(group_ids, return_inverse=True)
    n_groups = len(unique_groups)
    group_sizes = np.zeros(n_groups, dtype=int)
    for g in inverse:
        group_sizes[g] += 1

    def _weighted_per_group(values: np.ndarray) -> np.ndarray:
        acc: np.ndarray = np.zeros(n_groups)
        wsum: np.ndarray = np.zeros(n_groups)
        for idx, val, w in zip(inverse, values, weights, strict=True):
            acc[idx] += val * w
            wsum[idx] += w
        for g in range(n_groups):
            if wsum[g] > 0:
                acc[g] /= wsum[g]
            elif group_sizes[g] > 0:
                acc[g] = float(np.mean(values[inverse == g]))
        return acc

    result: dict[str, Any] = {"y_pred": _weighted_per_group(y_pred), "group_ids": unique_groups, "group_sizes": group_sizes}

    if y_proba is not None and np.asarray(y_proba).size > 0:
        proba = np.asarray(y_proba, dtype=float)
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        agg_proba = np.zeros((n_groups, proba.shape[1]))
        wsum = np.zeros(n_groups)
        for idx, pr, w in zip(inverse, proba, weights, strict=True):
            agg_proba[idx] += pr * w
            wsum[idx] += w
        for g in range(n_groups):
            if wsum[g] > 0:
                agg_proba[g] /= wsum[g]
        result["y_proba"] = agg_proba
        result["y_pred"] = np.argmax(agg_proba, axis=1).astype(float)

    if y_true is not None:
        result["y_true"] = _weighted_per_group(np.asarray(y_true, dtype=float).flatten())

    return result


def _normalize_task(task_type: Any) -> str | None:
    """Map a task-type label onto ``"regression"`` / ``"classification"``.

    Returns ``None`` for unknown / unset labels so callers do not over-block.
    """
    if task_type is None:
        return None
    raw = str(getattr(task_type, "value", task_type)).strip().lower()
    if raw in ("regression", "reg", "regressor"):
        return "regression"
    if raw in ("classification", "classifier", "clf", "binary", "multiclass", "multi-class", "multilabel"):
        return "classification"
    return None


def _fingerprint_value(value: Any) -> Any:
    """Normalise a value for stable fingerprinting (floats -> rounded repr)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return repr(round(value, 12))
    if value is None or isinstance(value, (int, str)):
        return value
    return str(value)


def _canonical_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalise a parameter mapping for deterministic fingerprinting."""
    return {str(k): _fingerprint_value(v) for k, v in sorted(mapping.items(), key=lambda kv: str(kv[0]))}


__all__ = [
    "PredictionLevel",
    "PredictionScope",
    "EvaluationScope",
    "PredictionUnitId",
    "PSEUDO_FOLD_IDS",
    "REFIT_FOLD_IDS",
    "is_real_cv_fold",
    "prediction_scope_from_legacy",
    "normalize_evaluation_scope",
    "ReductionRole",
    "ReductionAxis",
    "ReductionMethod",
    "TaskCompatibility",
    "FitScope",
    "ReductionError",
    "LeakedReducerStateError",
    "ReducerState",
    "ReductionPlan",
]
