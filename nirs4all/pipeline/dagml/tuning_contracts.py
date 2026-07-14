"""Typed public tuning contracts for the native DAG-ML tuning lane.

This module is intentionally execution-free. It gives ``run(tuning=...)`` a
strict, deterministic contract shape before the Optuna/n4m adapters are wired
to a shared DAG objective. The parser rejects ambiguous or lossy inputs now so
the later execution lane can consume a stable spec instead of an opaque dict.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from .training_contracts import tcv1_sha256

TuningEngine = Literal["optuna", "n4m"]
TuningDirection = Literal["minimize", "maximize"]
TUNING_SUMMARY_SCHEMA_ID = "https://nirs4all.org/schemas/tuning-summary/v1"
TUNING_SPACE_SCHEMA_ID = "https://nirs4all.org/schemas/tuning-ordered-search-space/v1"

SUPPORTED_TUNING_ENGINES = frozenset({"optuna", "n4m"})
SUPPORTED_TUNING_DIRECTIONS = frozenset({"minimize", "maximize"})
SUPPORTED_TUNING_KEYS = frozenset(
    {
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
    }
)

_TUNING_SUMMARY_SCHEMA: dict[str, Any] = {
    "$id": TUNING_SUMMARY_SCHEMA_ID,
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "additionalProperties": False,
    "properties": {
        "best_params": {"type": "object"},
        "best_value": {"type": "number"},
        "direction": {"enum": sorted(SUPPORTED_TUNING_DIRECTIONS)},
        "engine": {"enum": sorted(SUPPORTED_TUNING_ENGINES)},
        "fingerprint": {"type": "string"},
        "format": {"const": "nirs4all.tuning.summary"},
        "metric": {"type": "string"},
        "n_trials": {"minimum": 0, "type": "integer"},
        "optimizer": {"type": "string"},
        "persistence": {
            "additionalProperties": False,
            "properties": {
                "optimizer_state_resume_supported": {"type": "boolean"},
                "resume": {"type": "boolean"},
                "storage_configured": {"type": "boolean"},
                "study_name": {"type": ["string", "null"]},
            },
            "required": [
                "optimizer_state_resume_supported",
                "resume",
                "storage_configured",
                "study_name",
            ],
            "type": "object",
        },
        "pruner": {"type": ["string", "null"]},
        "sampler": {"type": ["string", "null"]},
        "schema_version": {"const": 1},
        "seed": {"type": ["integer", "null"]},
        "trial_states": {
            "additionalProperties": {"minimum": 0, "type": "integer"},
            "type": "object",
        },
        "trials": {
            "items": {
                "additionalProperties": False,
                "properties": {
                    "diagnostics": {
                        "additionalProperties": {
                            "type": ["boolean", "integer", "number", "string", "null"],
                        },
                        "type": "object",
                    },
                    "number": {"minimum": 0, "type": "integer"},
                    "state": {"type": "string"},
                    "value": {"type": ["number", "null"]},
                },
                "required": ["diagnostics", "number", "state", "value"],
                "type": "object",
            },
            "type": "array",
        },
        "version": {"const": 1},
    },
    "required": [
        "best_params",
        "best_value",
        "direction",
        "engine",
        "fingerprint",
        "format",
        "metric",
        "n_trials",
        "optimizer",
        "persistence",
        "schema_version",
        "trial_states",
        "trials",
        "version",
    ],
    "title": "NIRS4All tuning summary artifact",
    "type": "object",
}

_JSON_NATIVE_VALUE_SCHEMA: dict[str, Any] = {
    "description": "Any TCV1 JSON-native value; NaN/Infinity, bytes and opaque Python objects are rejected by runtime validation.",
}

_TUNING_SPACE_SCHEMA: dict[str, Any] = {
    "$id": TUNING_SPACE_SCHEMA_ID,
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "additionalProperties": False,
    "properties": {
        "fingerprint": {"pattern": "^[0-9a-f]{64}$", "type": "string"},
        "force_params": {
            "items": {
                "additionalProperties": False,
                "properties": {
                    "path": {"minLength": 1, "type": "string"},
                    "segments": {"items": {"minLength": 1, "type": "string"}, "minItems": 1, "type": "array"},
                    "value": _JSON_NATIVE_VALUE_SCHEMA,
                },
                "required": ["path", "segments", "value"],
                "type": "object",
            },
            "type": "array",
        },
        "format": {"const": "nirs4all.tuning.ordered_search_space"},
        "parameters": {
            "items": {
                "additionalProperties": False,
                "properties": {
                    "index": {"minimum": 0, "type": "integer"},
                    "path": {"minLength": 1, "type": "string"},
                    "segments": {"items": {"minLength": 1, "type": "string"}, "minItems": 1, "type": "array"},
                    "spec": _JSON_NATIVE_VALUE_SCHEMA,
                },
                "required": ["index", "path", "segments", "spec"],
                "type": "object",
            },
            "type": "array",
        },
        "schema_version": {"const": 1},
        "tuning_fingerprint": {"pattern": "^[0-9a-f]{64}$", "type": "string"},
    },
    "required": [
        "fingerprint",
        "force_params",
        "format",
        "parameters",
        "schema_version",
        "tuning_fingerprint",
    ],
    "title": "NIRS4All ordered tuning search-space artifact",
    "type": "object",
}


def get_tuning_summary_schema() -> dict[str, Any]:
    """Return the JSON Schema for lightweight tuning summary artifacts."""

    return cast(dict[str, Any], json.loads(json.dumps(_TUNING_SUMMARY_SCHEMA, sort_keys=True)))


def tuning_summary_schema_json(*, indent: int | None = 2) -> str:
    """Serialize the tuning summary JSON Schema deterministically."""

    return (
        json.dumps(
            get_tuning_summary_schema(),
            indent=indent,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
        )
        + "\n"
    )


def get_tuning_space_schema() -> dict[str, Any]:
    """Return the JSON Schema for ordered tuning search-space artifacts."""

    return cast(dict[str, Any], json.loads(json.dumps(_TUNING_SPACE_SCHEMA, sort_keys=True)))


def tuning_space_schema_json(*, indent: int | None = 2) -> str:
    """Serialize the ordered tuning search-space JSON Schema deterministically."""

    return (
        json.dumps(
            get_tuning_space_schema(),
            indent=indent,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
        )
        + "\n"
    )


@dataclass(frozen=True)
class SearchSpaceParameter:
    """One ordered, canonical tuning-space parameter.

    ``path`` is the public patch path consumed by ``PipelineObjective``. The
    parser accepts both dotted public spelling (``ridge.alpha``) and sklearn
    double-underscore spelling (``ridge__alpha``), then canonicalizes to dotted
    spelling so Optuna, n4m and native adapters see the same ordered contract.
    """

    index: int
    path: str
    segments: tuple[str, ...]
    spec: Any

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0:
            raise ValueError("SearchSpaceParameter.index must be a non-negative integer")
        path, segments = normalize_parameter_path(self.path, label="SearchSpaceParameter.path")
        if self.segments != segments:
            raise ValueError("SearchSpaceParameter.segments must match path")
        try:
            tcv1_sha256(self.spec)
        except (TypeError, ValueError) as exc:
            raise ValueError("SearchSpaceParameter.spec must contain TCV1-compatible JSON-native values") from exc
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "segments", segments)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-like parameter form."""

        return {
            "index": self.index,
            "path": self.path,
            "segments": list(self.segments),
            "spec": self.spec,
        }


@dataclass(frozen=True)
class ParameterPatch:
    """One canonical candidate or force-parameter patch."""

    path: str
    segments: tuple[str, ...]
    value: Any

    def __post_init__(self) -> None:
        path, segments = normalize_parameter_path(self.path, label="ParameterPatch.path")
        if self.segments != segments:
            raise ValueError("ParameterPatch.segments must match path")
        try:
            tcv1_sha256(self.value)
        except (TypeError, ValueError) as exc:
            raise ValueError("ParameterPatch.value must contain TCV1-compatible JSON-native values") from exc
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "segments", segments)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-like patch form."""

        return {
            "path": self.path,
            "segments": list(self.segments),
            "value": self.value,
        }


@dataclass(frozen=True)
class OrderedSearchSpaceSpec:
    """Canonical ordered tuning search space.

    This is the P3 bridge between public ``tuning.space`` syntax and optimizer
    adapters. It is intentionally JSON-native and execution-free: adapters can
    walk ``parameters`` in a stable order and produce ``ParameterPatch`` records
    without reparsing public dict syntax.
    """

    parameters: tuple[SearchSpaceParameter, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, tuple) or not self.parameters:
            raise ValueError("OrderedSearchSpaceSpec.parameters must be a non-empty tuple")
        paths: set[str] = set()
        canonical_parameters: list[SearchSpaceParameter] = []
        for position, parameter in enumerate(self.parameters):
            if not isinstance(parameter, SearchSpaceParameter):
                raise ValueError("OrderedSearchSpaceSpec.parameters must contain SearchSpaceParameter entries")
            if parameter.index != position:
                raise ValueError("OrderedSearchSpaceSpec.parameters indexes must be contiguous from zero")
            if parameter.path in paths:
                raise ValueError("OrderedSearchSpaceSpec.parameters contains duplicate paths")
            paths.add(parameter.path)
            canonical_parameters.append(parameter)
        if tuple(parameter.path for parameter in canonical_parameters) != tuple(sorted(paths)):
            raise ValueError("OrderedSearchSpaceSpec.parameters must be sorted by canonical path")
        object.__setattr__(self, "parameters", tuple(canonical_parameters))

    @classmethod
    def from_mapping(cls, space: Mapping[str, Any], *, context: str = "run(tuning=...).space") -> OrderedSearchSpaceSpec:
        """Normalize a public mapping into deterministic ordered parameters."""

        if not isinstance(space, Mapping) or not space:
            raise ValueError(f"{context} must be a non-empty mapping of patch paths to parameter specs")
        entries: dict[str, tuple[tuple[str, ...], Any]] = {}
        raw_by_canonical: dict[str, str] = {}
        for raw_key, spec in space.items():
            path, segments = normalize_parameter_path(raw_key, label=f"{context} key")
            previous = raw_by_canonical.get(path)
            if previous is not None:
                raise ValueError(f"{context} contains duplicate patch path {path!r} after canonicalization from {previous!r} and {raw_key!r}")
            try:
                tcv1_sha256(spec)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context}[{path!r}] must contain TCV1-compatible JSON-native candidate specs") from exc
            raw_by_canonical[path] = str(raw_key)
            entries[path] = (segments, spec)
        return cls(tuple(SearchSpaceParameter(index=index, path=path, segments=entries[path][0], spec=entries[path][1]) for index, path in enumerate(sorted(entries))))

    @classmethod
    def from_tuning_spec(cls, tuning: DagMLTuningSpec) -> OrderedSearchSpaceSpec:
        """Build the ordered space from a parsed tuning spec."""

        return cls.from_mapping(tuning.space, context="DagMLTuningSpec.space")

    @property
    def paths(self) -> tuple[str, ...]:
        """Canonical parameter paths in optimizer iteration order."""

        return tuple(parameter.path for parameter in self.parameters)

    def to_space_mapping(self) -> dict[str, Any]:
        """Return the canonical ``DagMLTuningSpec.space`` mapping."""

        return {parameter.path: parameter.spec for parameter in self.parameters}

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-like ordered search-space form."""

        return {
            "format": "nirs4all.tuning.ordered_search_space",
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "schema_version": 1,
        }

    @property
    def fingerprint(self) -> str:
        """TCV1 fingerprint of the ordered search-space contract."""

        return tcv1_sha256(self.to_dict())

    def patch(self, path: str, value: Any, *, context: str = "tuning parameter patch") -> ParameterPatch:
        """Create a validated patch for one known path."""

        canonical, segments = normalize_parameter_path(path, label=f"{context} path")
        if canonical not in self.paths:
            raise ValueError(f"{context} path {canonical!r} is not present in tuning.space")
        try:
            tcv1_sha256(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}[{canonical!r}] must contain a TCV1-compatible JSON-native value") from exc
        return ParameterPatch(path=canonical, segments=segments, value=value)

    def patches_from_mapping(self, params: Mapping[str, Any], *, context: str = "trial params") -> tuple[ParameterPatch, ...]:
        """Normalize a candidate mapping into ordered ``ParameterPatch`` records."""

        if not isinstance(params, Mapping):
            raise TypeError(f"{context} must be a mapping")
        normalized: dict[str, ParameterPatch] = {}
        raw_by_canonical: dict[str, str] = {}
        for raw_key, value in params.items():
            patch = self.patch(raw_key, value, context=context)
            previous = raw_by_canonical.get(patch.path)
            if previous is not None:
                raise ValueError(f"{context} contains duplicate patch path {patch.path!r} after canonicalization from {previous!r} and {raw_key!r}")
            raw_by_canonical[patch.path] = str(raw_key)
            normalized[patch.path] = patch
        return tuple(normalized[path] for path in self.paths if path in normalized)


def normalize_parameter_path(path: Any, *, label: str = "parameter path") -> tuple[str, tuple[str, ...]]:
    """Normalize public dotted/sklearn tuning paths to canonical dotted form."""

    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{label} must be a non-empty string")
    raw = path.strip()
    if "\x00" in raw:
        raise ValueError(f"{label} must not contain NUL characters")
    canonical = raw.replace("__", ".")
    segments = tuple(part.strip() for part in canonical.split("."))
    if any(not part for part in segments):
        raise ValueError(f"{label} {path!r} contains an empty path segment")
    if any("__" in part for part in segments):
        raise ValueError(f"{label} {path!r} contains an invalid double-underscore segment")
    return ".".join(segments), segments


@dataclass(frozen=True)
class DagMLTuningSpec:
    """Normalized fixed-topology full-DAG tuning request.

    ``space`` keys are user-facing patch paths. Values remain intentionally
    opaque at this layer because Optuna/n4m adapters own the concrete sampler
    translation; the contract only freezes shape, defaults and determinism.
    """

    engine: TuningEngine
    space: Mapping[str, Any]
    force_params: Mapping[str, Any] | None = None
    metric: str = "rmse"
    direction: TuningDirection = "minimize"
    n_trials: int = 50
    sampler: str | None = None
    pruner: str | None = None
    seed: int | None = None
    resume: bool = False
    storage: str | None = None
    study_name: str | None = None

    def __post_init__(self) -> None:
        engine = _normalize_non_empty_string(self.engine, "DagMLTuningSpec.engine").lower()
        if engine not in SUPPORTED_TUNING_ENGINES:
            raise ValueError(f"DagMLTuningSpec.engine must be one of {sorted(SUPPORTED_TUNING_ENGINES)}")
        ordered_space = OrderedSearchSpaceSpec.from_mapping(self.space, context="DagMLTuningSpec.space")
        normalized_space = ordered_space.to_space_mapping()
        direction = _normalize_non_empty_string(self.direction, "DagMLTuningSpec.direction").lower()
        if direction not in SUPPORTED_TUNING_DIRECTIONS:
            raise ValueError("DagMLTuningSpec.direction must be 'minimize' or 'maximize'")
        metric = _normalize_non_empty_string(self.metric, "DagMLTuningSpec.metric").lower()
        force_params = _optional_force_params({"force_params": self.force_params}, normalized_space, "DagMLTuningSpec")
        sampler = None if self.sampler is None else _normalize_non_empty_string(self.sampler, "DagMLTuningSpec.sampler").lower()
        pruner = None if self.pruner is None else _normalize_non_empty_string(self.pruner, "DagMLTuningSpec.pruner").lower()
        n_trials = _positive_int(self.n_trials, "DagMLTuningSpec.n_trials")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ValueError("DagMLTuningSpec.seed must be an integer")
        if not isinstance(self.resume, bool):
            raise ValueError("DagMLTuningSpec.resume must be a boolean")
        storage = _optional_storage_uri({"storage": self.storage}, "storage", "DagMLTuningSpec")
        study_name = _optional_nul_free_string({"study_name": self.study_name}, "study_name", "DagMLTuningSpec")
        object.__setattr__(self, "engine", engine)
        object.__setattr__(self, "space", normalized_space)
        object.__setattr__(self, "force_params", force_params)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "n_trials", n_trials)
        object.__setattr__(self, "sampler", sampler)
        object.__setattr__(self, "pruner", pruner)
        object.__setattr__(self, "storage", storage)
        object.__setattr__(self, "study_name", study_name)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-like contract form."""

        return {
            "direction": self.direction,
            "engine": self.engine,
            "force_params": None if self.force_params is None else {key: self.force_params[key] for key in sorted(self.force_params)},
            "metric": self.metric,
            "n_trials": self.n_trials,
            "pruner": self.pruner,
            "resume": self.resume,
            "sampler": self.sampler,
            "seed": self.seed,
            "space": {key: self.space[key] for key in sorted(self.space)},
            "storage": self.storage,
            "study_name": self.study_name,
        }

    @property
    def fingerprint(self) -> str:
        """TCV1 fingerprint of the normalized tuning contract."""

        return tcv1_sha256(self.to_dict())

    @property
    def ordered_search_space(self) -> OrderedSearchSpaceSpec:
        """Return the P3 ordered search-space contract for this tuning spec."""

        return OrderedSearchSpaceSpec.from_tuning_spec(self)

    def parameter_patches(self, params: Mapping[str, Any], *, context: str = "trial params") -> tuple[ParameterPatch, ...]:
        """Normalize candidate params into ordered patch records."""

        return self.ordered_search_space.patches_from_mapping(params, context=context)


@dataclass(frozen=True)
class TrialResult:
    """Portable result for one optimizer trial against a ``PipelineObjective``."""

    number: int
    params: Mapping[str, Any]
    value: float | None
    state: str
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.number, int) or isinstance(self.number, bool) or self.number < 0:
            raise ValueError("TrialResult.number must be a non-negative integer")
        if not isinstance(self.params, Mapping):
            raise ValueError("TrialResult.params must be a mapping")
        params = _normalize_parameter_value_mapping(self.params, "TrialResult.params")
        value = None if self.value is None else _finite_float(self.value, "TrialResult.value")
        state = _normalize_non_empty_string(self.state, "TrialResult.state").upper()
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("TrialResult.diagnostics must be a mapping")
        diagnostics = _normalize_str_key_json_mapping(self.diagnostics, "TrialResult.diagnostics")
        object.__setattr__(self, "number", self.number)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "diagnostics", diagnostics)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-like trial form."""

        return {
            "diagnostics": dict(self.diagnostics),
            "number": self.number,
            "params": {key: self.params[key] for key in sorted(self.params)},
            "state": self.state,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrialResult:
        """Parse one serialized trial result."""

        if not isinstance(payload, Mapping):
            raise TypeError("TrialResult payload must be a mapping")
        required = {"diagnostics", "number", "params", "state", "value"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"TrialResult payload is missing keys {missing}")
        if not isinstance(payload["number"], int) or isinstance(payload["number"], bool):
            raise ValueError("TrialResult.number must be an integer")
        if not isinstance(payload["params"], Mapping):
            raise ValueError("TrialResult.params must be a mapping")
        if not isinstance(payload["diagnostics"], Mapping):
            raise ValueError("TrialResult.diagnostics must be a mapping")
        if not isinstance(payload["state"], str) or not payload["state"]:
            raise ValueError("TrialResult.state must be a non-empty string")
        value = payload["value"]
        if value is not None:
            value = _finite_float(value, "TrialResult.value")
        return cls(
            number=payload["number"],
            params=dict(payload["params"]),
            value=value,
            state=payload["state"],
            diagnostics=dict(payload["diagnostics"]),
        )


@dataclass(frozen=True)
class TuningResult:
    """Typed result for the native DAG-ML tuning lane."""

    tuning: DagMLTuningSpec
    best_params: Mapping[str, Any]
    best_value: float
    trials: tuple[TrialResult, ...]
    optimizer: str

    def __post_init__(self) -> None:
        if not isinstance(self.tuning, DagMLTuningSpec):
            raise ValueError("TuningResult.tuning must be a DagMLTuningSpec")
        best_params = _normalize_parameter_value_mapping(self.best_params, "TuningResult.best_params")
        if best_params:
            self.tuning.ordered_search_space.patches_from_mapping(best_params, context="TuningResult.best_params")
        best_value = _finite_float(self.best_value, "TuningResult.best_value")
        if not isinstance(self.trials, tuple):
            raise ValueError("TuningResult.trials must be a tuple")
        seen_numbers: set[int] = set()
        trials: list[TrialResult] = []
        for trial in self.trials:
            if not isinstance(trial, TrialResult):
                raise ValueError("TuningResult.trials must contain TrialResult entries")
            if trial.number in seen_numbers:
                raise ValueError("TuningResult.trials contains duplicate trial numbers")
            seen_numbers.add(trial.number)
            if trial.params:
                self.tuning.ordered_search_space.patches_from_mapping(
                    trial.params,
                    context=f"TuningResult.trials[{trial.number}].params",
                )
            trials.append(trial)
        optimizer = _normalize_non_empty_string(self.optimizer, "TuningResult.optimizer")
        object.__setattr__(self, "best_params", best_params)
        object.__setattr__(self, "best_value", best_value)
        object.__setattr__(self, "trials", tuple(sorted(trials, key=lambda trial: trial.number)))
        object.__setattr__(self, "optimizer", optimizer)

    @property
    def n_trials(self) -> int:
        """Number of terminal trial records."""

        return len(self.trials)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-like result form."""

        return {
            "best_params": {key: self.best_params[key] for key in sorted(self.best_params)},
            "best_value": self.best_value,
            "fingerprint": self.fingerprint,
            "optimizer": self.optimizer,
            "trials": [trial.to_dict() for trial in self.trials],
            "tuning": self.tuning.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return deterministic JSON with a trailing newline."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, separators=None if indent is not None else (",", ":")) + "\n"

    def save_json(self, path: str | Path) -> Path:
        """Persist the tuning result as deterministic verified JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    def summary_artifact(self) -> dict[str, Any]:
        """Return the lightweight deterministic HPO summary payload."""

        trial_states: dict[str, int] = {}
        for trial in self.trials:
            trial_states[trial.state] = trial_states.get(trial.state, 0) + 1
        return {
            "best_params": {key: self.best_params[key] for key in sorted(self.best_params)},
            "best_value": self.best_value,
            "direction": self.tuning.direction,
            "engine": self.tuning.engine,
            "fingerprint": self.fingerprint,
            "format": "nirs4all.tuning.summary",
            "metric": self.tuning.metric,
            "n_trials": self.n_trials,
            "optimizer": self.optimizer,
            "persistence": {
                "optimizer_state_resume_supported": self.tuning.engine in {"optuna", "n4m"},
                "resume": self.tuning.resume,
                "storage_configured": self.tuning.storage is not None,
                "study_name": self.tuning.study_name,
            },
            "pruner": self.tuning.pruner,
            "sampler": self.tuning.sampler,
            "schema_version": 1,
            "seed": self.tuning.seed,
            "trial_states": {key: trial_states[key] for key in sorted(trial_states)},
            "trials": [
                {
                    "diagnostics": _summary_trial_diagnostics(trial.diagnostics),
                    "number": trial.number,
                    "state": trial.state,
                    "value": trial.value,
                }
                for trial in self.trials
            ],
            "version": 1,
        }

    def to_summary_json(self, *, indent: int | None = 2) -> str:
        """Return deterministic lightweight summary JSON with a trailing newline."""

        return (
            json.dumps(
                self.summary_artifact(),
                indent=indent,
                sort_keys=True,
                separators=None if indent is not None else (",", ":"),
            )
            + "\n"
        )

    def save_summary(self, path: str | Path) -> Path:
        """Persist the lightweight deterministic tuning summary artifact."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_summary_json(), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TuningResult:
        """Parse a serialized tuning result and verify its fingerprint when present."""

        if not isinstance(payload, Mapping):
            raise TypeError("TuningResult payload must be a mapping")
        required = {"best_params", "best_value", "optimizer", "trials", "tuning"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"TuningResult payload is missing keys {missing}")
        if not isinstance(payload["best_params"], Mapping):
            raise ValueError("TuningResult.best_params must be a mapping")
        if not isinstance(payload["optimizer"], str) or not payload["optimizer"]:
            raise ValueError("TuningResult.optimizer must be a non-empty string")
        if not isinstance(payload["trials"], list):
            raise ValueError("TuningResult.trials must be a list")
        result = cls(
            tuning=parse_tuning_spec(payload["tuning"], context="TuningResult.tuning"),
            best_params=dict(payload["best_params"]),
            best_value=_finite_float(payload["best_value"], "TuningResult.best_value"),
            trials=tuple(TrialResult.from_dict(trial) for trial in payload["trials"]),
            optimizer=payload["optimizer"],
        )
        expected = payload.get("fingerprint")
        if expected is not None and expected != result.fingerprint:
            raise ValueError("TuningResult fingerprint mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str) -> TuningResult:
        """Parse a JSON tuning result and verify its fingerprint."""

        return cls.from_dict(json.loads(payload))

    @classmethod
    def load_json(cls, path: str | Path) -> TuningResult:
        """Load a persisted tuning result and verify its fingerprint."""

        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    @property
    def fingerprint(self) -> str:
        """TCV1 fingerprint of the result summary."""

        return tcv1_sha256(
            {
                "best_params": {key: self.best_params[key] for key in sorted(self.best_params)},
                "best_value": self.best_value,
                "optimizer": self.optimizer,
                "trials": [trial.to_dict() for trial in self.trials],
                "tuning": self.tuning.to_dict(),
            }
        )


_SUMMARY_DIAGNOSTIC_KEYS = frozenset(
    {
        "direction",
        "engine",
        "error_type",
        "final_calibration_scope",
        "metric",
        "score_extractor",
        "score_family",
        "search_space_fingerprint",
        "tuning_fingerprint",
    }
)


def _summary_trial_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, bool | int | float | str | None]:
    """Return bounded JSON-safe diagnostics for lightweight summary cards.

    The full ``TrialResult`` remains the authoritative HPO tape. The summary
    card intentionally exposes only stable, scalar routing/audit fields so
    CLI/UI/bindings can explain failed or conformal-aware trials without
    parsing full candidate params or raw exception messages.
    """

    summary: dict[str, bool | int | float | str | None] = {}
    for key in sorted(_SUMMARY_DIAGNOSTIC_KEYS):
        if key not in diagnostics:
            continue
        value = diagnostics[key]
        if value is None or isinstance(value, (bool, int, float, str)):
            summary[key] = value
        else:
            raise ValueError(f"summary diagnostics.{key} must be a scalar JSON-native value")
    return summary


class DagMLTuningNotImplementedError(NotImplementedError):
    """Structured public boundary for reserved ``run(tuning=...)`` execution."""

    def __init__(
        self,
        tuning_spec: DagMLTuningSpec,
        *,
        available_internal: tuple[str, ...] = (
            "DagMLTuningSpec",
            "PipelineObjective",
            "prediction score extractor for PipelineObjective",
            "single-estimator and linear transformer→estimator PipelineObjective compiler",
            "OptunaPipelineObjectiveAdapter",
            "N4MPipelineObjectiveAdapter",
            "TuningResult",
            "TuningResult JSON persistence",
            "WorkspaceStore TuningResult persistence",
            "RunResult tuning evidence projection",
            "explicit winner prediction projection into RunResult",
            "linear compile→tune→refit→RunResult orchestration",
            "public run(tuning=...) single-estimator/linear array subset",
        ),
        missing_gates: tuple[str, ...] = (
            "full nirs4all pipeline-to-DAG-ML objective compiler for splitters/branches/dataset loaders",
            "public run(tuning=...) resume integration for broader pipeline shapes",
            "public documentation/examples for executable run(tuning=...)",
        ),
    ) -> None:
        self.tuning_spec = tuning_spec
        self.available_internal = available_internal
        self.missing_gates = missing_gates
        super().__init__(
            "run(tuning=...) is reserved for the native DAG-ML tuning API and is not executable yet; "
            f"validated tuning contract fingerprint: {tuning_spec.fingerprint}; "
            f"available internal seams: {', '.join(available_internal)}; "
            f"missing gates: {', '.join(missing_gates)}."
        )


def parse_tuning_spec(tuning: Mapping[str, Any], *, context: str = "run(tuning=...)") -> DagMLTuningSpec:
    """Validate and normalize a public full-DAG tuning mapping."""

    if not isinstance(tuning, Mapping):
        raise TypeError(f"{context} must be a mapping")
    unknown = sorted(set(tuning) - SUPPORTED_TUNING_KEYS)
    if unknown:
        raise ValueError(f"{context} does not support keys {unknown}; supported keys are {sorted(SUPPORTED_TUNING_KEYS)}")

    engine = _required_lower_string(tuning, "engine", context)
    if engine not in SUPPORTED_TUNING_ENGINES:
        raise ValueError(f"{context}.engine must be one of {sorted(SUPPORTED_TUNING_ENGINES)}")

    raw_space = tuning.get("space")
    if not isinstance(raw_space, Mapping):
        raise TypeError(f"{context}.space must be a mapping")
    ordered_space = OrderedSearchSpaceSpec.from_mapping(raw_space, context=f"{context}.space")
    normalized_space = ordered_space.to_space_mapping()

    metric = _optional_lower_string(tuning, "metric", default="rmse")
    direction = _optional_lower_string(tuning, "direction", default="minimize")
    if direction not in SUPPORTED_TUNING_DIRECTIONS:
        raise ValueError(f"{context}.direction must be 'minimize' or 'maximize'")

    return DagMLTuningSpec(
        engine=engine,  # type: ignore[arg-type]
        space=normalized_space,
        force_params=_optional_force_params(tuning, normalized_space, context),
        metric=metric,
        direction=direction,  # type: ignore[arg-type]
        n_trials=_positive_int(tuning.get("n_trials", 50), f"{context}.n_trials"),
        sampler=_optional_lower_string_or_none(tuning, "sampler"),
        pruner=_optional_lower_string_or_none(tuning, "pruner"),
        seed=_optional_int(tuning, "seed", context),
        resume=_optional_bool(tuning, "resume", default=False, context=context),
        storage=_optional_storage_uri(tuning, "storage", context),
        study_name=_optional_nul_free_string(tuning, "study_name", context),
    )


def _required_lower_string(mapping: Mapping[str, Any], key: str, context: str) -> str:
    if key not in mapping:
        raise ValueError(f"{context}.{key} is required")
    return _normalize_non_empty_string(mapping[key], f"{context}.{key}").lower()


def _optional_lower_string(mapping: Mapping[str, Any], key: str, *, default: str) -> str:
    if key not in mapping or mapping[key] is None:
        return default
    return _normalize_non_empty_string(mapping[key], key).lower()


def _optional_string(mapping: Mapping[str, Any], key: str) -> str | None:
    if key not in mapping or mapping[key] is None:
        return None
    return _normalize_non_empty_string(mapping[key], key)


def _optional_lower_string_or_none(mapping: Mapping[str, Any], key: str) -> str | None:
    value = _optional_string(mapping, key)
    if value is None:
        return None
    return value.lower()


def _optional_storage_uri(mapping: Mapping[str, Any], key: str, context: str) -> str | None:
    value = _optional_nul_free_string(mapping, key, context)
    if value is None:
        return None
    if not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
        raise ValueError(f"{context}.{key} must be a URI with an explicit scheme, such as sqlite:///study.db")
    return value


def _optional_nul_free_string(mapping: Mapping[str, Any], key: str, context: str) -> str | None:
    value = _optional_string(mapping, key)
    if value is None:
        return None
    if "\x00" in value:
        raise ValueError(f"{context}.{key} must not contain NUL characters")
    return value


def _normalize_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_force_params(
    mapping: Mapping[str, Any],
    normalized_space: Mapping[str, Any],
    context: str,
) -> dict[str, Any] | None:
    if "force_params" not in mapping or mapping["force_params"] is None:
        return None
    raw = mapping["force_params"]
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{context}.force_params must be a non-empty mapping of tuning.space paths to warm-start values")
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        normalized_key, _segments = normalize_parameter_path(key, label=f"{context}.force_params key")
        if normalized_key not in normalized_space:
            raise ValueError(f"{context}.force_params keys must be a subset of tuning.space; unknown key {normalized_key!r}")
        try:
            tcv1_sha256(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}.force_params[{normalized_key!r}] must contain TCV1-compatible JSON-native values") from exc
        normalized[normalized_key] = value
    if len(normalized) != len(raw):
        raise ValueError(f"{context}.force_params contains duplicate keys after canonicalization")
    return normalized


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _optional_int(mapping: Mapping[str, Any], key: str, context: str) -> int | None:
    if key not in mapping or mapping[key] is None:
        return None
    value = mapping[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be an integer")
    return value


def _optional_bool(mapping: Mapping[str, Any], key: str, *, default: bool, context: str) -> bool:
    if key not in mapping or mapping[key] is None:
        return default
    value = mapping[key]
    if not isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a boolean")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number")
    return numeric


def _normalize_str_key_json_mapping(mapping: Mapping[str, Any], label: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key.strip() or key != key.strip():
            raise ValueError(f"{label} keys must be canonical non-empty strings")
        if key in normalized:
            raise ValueError(f"{label} contains duplicate keys")
        normalized[key] = value
    try:
        tcv1_sha256(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain TCV1-compatible JSON-native values") from exc
    return normalized


def _normalize_parameter_value_mapping(mapping: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{label} must be a mapping")
    normalized: dict[str, Any] = {}
    raw_by_canonical: dict[str, str] = {}
    for raw_key, value in mapping.items():
        path, _segments = normalize_parameter_path(raw_key, label=f"{label} key")
        previous = raw_by_canonical.get(path)
        if previous is not None:
            raise ValueError(f"{label} contains duplicate patch path {path!r} after canonicalization from {previous!r} and {raw_key!r}")
        try:
            tcv1_sha256(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}[{path!r}] must contain TCV1-compatible JSON-native values") from exc
        raw_by_canonical[path] = str(raw_key)
        normalized[path] = value
    return normalized


__all__ = [
    "DagMLTuningNotImplementedError",
    "DagMLTuningSpec",
    "OrderedSearchSpaceSpec",
    "ParameterPatch",
    "SUPPORTED_TUNING_DIRECTIONS",
    "SUPPORTED_TUNING_ENGINES",
    "SUPPORTED_TUNING_KEYS",
    "SearchSpaceParameter",
    "TUNING_SPACE_SCHEMA_ID",
    "TUNING_SUMMARY_SCHEMA_ID",
    "TrialResult",
    "TuningResult",
    "get_tuning_space_schema",
    "get_tuning_summary_schema",
    "normalize_parameter_path",
    "parse_tuning_spec",
    "tuning_space_schema_json",
    "tuning_summary_schema_json",
]
