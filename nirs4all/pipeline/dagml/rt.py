"""Runtime envelopes for the dag-ml backend — ``RtResult`` / ``RtRunRequest`` / ``RtError`` (B-018 / L10).

These are the **thin, pure-projection wrappers** the runtime API (`LOCK-RT`, `DEC-RT-001`) puts over
surfaces that ALREADY exist on the dag-ml backend — they add **zero new dag-ml fields** and never
recompute a score. The four inputs all live next to this module in ``pipeline/dagml/``:

* the native results triple (``manifest.json`` + ``score_set.json`` + ``predictions.parquet``) written /
  read by :mod:`nirs4all.pipeline.dagml.native_results`;
* the raw native ScoreSet stashed on a dag-ml ``RunResult`` (``RunResult._dagml_score_set``,
  :func:`nirs4all.pipeline.dagml.result._scores_to_run_result`);
* the dag-ml error taxonomy (:class:`~nirs4all.pipeline.dagml.errors.DagMlUnsupported` /
  :class:`~nirs4all.pipeline.dagml.errors.DagMlUnavailable`) the ``run()`` fallback catches.

Anchored on the dag-ml **ScoreSet** (``score_set.schema.json``): ``RtResult.reports`` is the VERBATIM
``score_set.reports[]`` — the ``partition / level / fold_id / variant_id / target`` join key both the
Studio ``ChainSummary`` pivot and the Web ``RunResult`` nest are deterministic group-bys of. The wire
schema lives in ``nirs4all-ecosystem/docs/contracts/runtime/*.v1.schema.json`` (it ``$ref``s the dag-ml
contracts; the cause vocabulary is owned by ``CAP-004``). This module is the Python projection; nothing
here mutates ``RunResult``, the ``.n4a`` bundle, or the native results format (all 0.9.x-stable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from nirs4all.api.result import RunResult
    from nirs4all.data.predictions import Predictions

# The RtResult envelope's OWN schema version (the NET-NEW wrapper), independent of the dag-ml ScoreSet
# schema (owned by dag-ml, carried verbatim) and the native-results manifest schema.
RT_RESULT_SCHEMA_VERSION = 1

# RtError.cause vocabulary (RT-003). The COARSE wire causes carried across REST / WASM / CLI; the finer
# controller-manifest cause codes + mitigations are owned by CAP-004 and ride in ``unsupported_capability``.
RT_ERROR_CAUSES: frozenset[str] = frozenset(
    {"unsupported_shape", "unsupported_capability", "unavailable_backend", "invalid_request", "runtime_error"}
)

# The 8 runtime verbs (RT-001 / DEC-RT-001) — the product surface an RtError/RtRunRequest is scoped to.
RT_VERBS: frozenset[str] = frozenset({"inspect", "validate", "plan", "run", "predict", "replay", "explain", "export"})


class RtError(Exception):
    """One unified runtime error envelope (RT-003) — raisable AND serializable.

    Converges the three divergent failure shapes (Python ``DagMl*`` warn+fallback, Studio preflight
    ``issues[]`` / driver ``metadata.reason``, Web silent catch) onto a single wire shape
    ``{ verb, cause, message, mitigation, unsupported_capability?, portable_level? }``. A subclass of
    ``Exception`` (NOT of :class:`~nirs4all.pipeline.dagml.errors.DagMlUnsupported` /
    ``NotImplementedError``) so raising it at the RT boundary (``run(..., allow_fallback=False)``) is NOT
    re-caught by the legacy-fallback ``except`` — it propagates to the caller as the explicit "no silent
    fallback" signal ``B-018`` asks for.

    The ``cause`` vocabulary and ``mitigation`` text are **referenced** from ``CAP-004`` (RT-003 owns only
    this envelope, not the vocabulary). ``unsupported_capability`` carries the finer CAP capability token
    when one applies; ``portable_level`` is the CAP-002 classifier (opaque here, ``None`` in V1).
    """

    def __init__(
        self,
        verb: str,
        cause: str,
        message: str,
        *,
        mitigation: str | None = None,
        unsupported_capability: str | None = None,
        portable_level: str | None = None,
    ) -> None:
        if cause not in RT_ERROR_CAUSES:
            raise ValueError(f"unknown RtError cause {cause!r}; valid causes: {sorted(RT_ERROR_CAUSES)}")
        self.verb = verb
        self.cause = cause
        self.message = message
        self.mitigation = mitigation
        self.unsupported_capability = unsupported_capability
        self.portable_level = portable_level
        super().__init__(f"[{verb}/{cause}] {message}")

    def to_dict(self) -> dict[str, Any]:
        """The wire dict (matches ``rt_error.v1.schema.json``); ``None`` optionals are omitted."""
        payload: dict[str, Any] = {"verb": self.verb, "cause": self.cause, "message": self.message}
        if self.mitigation is not None:
            payload["mitigation"] = self.mitigation
        if self.unsupported_capability is not None:
            payload["unsupported_capability"] = self.unsupported_capability
        if self.portable_level is not None:
            payload["portable_level"] = self.portable_level
        return payload

    @classmethod
    def invalid_request(
        cls,
        exc_or_message: BaseException | str,
        *,
        verb: str,
        mitigation: str | None = None,
        unsupported_capability: str | None = None,
        portable_level: str | None = None,
    ) -> RtError:
        """Build an ``invalid_request`` envelope for bad runtime inputs, selectors, or workspace refs.

        This is the shared mapping for boundary validation failures (bad dataset/spec, impossible export
        selector, detached legacy result with no workspace). It intentionally does NOT replace the original
        exception type at the public API boundary; callers that expose the RT contract can catch the legacy
        exception and serialize this envelope without parsing ad hoc message strings.
        """
        message = str(exc_or_message)
        return cls(
            verb,
            "invalid_request",
            f"invalid {verb} request: {message}",
            mitigation=mitigation or "fix the request inputs/selectors and retry.",
            unsupported_capability=unsupported_capability,
            portable_level=portable_level,
        )

    @classmethod
    def runtime_error(
        cls,
        exc_or_message: BaseException | str,
        *,
        verb: str,
        mitigation: str | None = None,
    ) -> RtError:
        """Build a ``runtime_error`` envelope for genuine execution failures that must not fall back."""
        message = str(exc_or_message)
        return cls(
            verb,
            "runtime_error",
            f"{verb} failed at runtime: {message}",
            mitigation=mitigation or "inspect the runtime error and fix the failing operator, data, or environment.",
        )

    @classmethod
    def from_dagml_error(cls, exc: BaseException, *, verb: str = "run") -> RtError:
        """Classify a caught dag-ml backend exception into an ``RtError`` (the RT-003 migration table).

        * :class:`~nirs4all.pipeline.dagml.errors.DagMlUnavailable` → ``cause="unavailable_backend"``
          (neither dag-ml execution mechanism is installed — the wheel is missing the native backend).
        * :class:`~nirs4all.pipeline.dagml.errors.DagMlUnsupported` / any other ``NotImplementedError``
          → ``cause="unsupported_shape"`` (a pipeline shape the dag-ml path does not yet cover).

        Each carries a derived ``mitigation`` (the remedy the original error documents). ANY other
        exception is a genuine bug that must NOT be masked as an RtError — callers catch only the two
        fallback signals, so this method is only ever handed one of them.
        """
        # Local import keeps this module importable without loading the backend error module eagerly.
        from nirs4all.pipeline.dagml.errors import DagMlUnavailable

        if isinstance(exc, DagMlUnavailable):
            return cls(
                verb,
                "unavailable_backend",
                f"the dag-ml backend is not available: {exc}",
                mitigation="install the dag-ml native backend (the in-process extension or the dag-ml-cli binary), or run engine='legacy'.",
            )
        # DagMlUnsupported is a NotImplementedError subclass; both land here as an unsupported SHAPE.
        return cls(
            verb,
            "unsupported_shape",
            f"engine='dag-ml' does not support this pipeline shape: {exc}",
            mitigation="run this shape on engine='legacy', or rewrite it into a dag-ml-covered shape (see the dag-ml coverage matrix).",
        )


def _to_jsonable_array(value: Any) -> list[Any] | None:
    """A per-sample array as nested python lists (preserving 2D multi-target shape), ``None`` if empty.

    Pure projection: ``np.asarray(value).tolist()`` — no reshape, no recompute. A 1D array becomes a flat
    list, a 2D (multi-target) array becomes a list of rows, so the wire form round-trips the target width.
    """
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    values: list[Any] = arr.tolist()
    return values


def _project_predictions(predictions: Predictions) -> list[dict[str, Any]]:
    """Project an in-memory :class:`Predictions` into the RtResult ``predictions[]`` block rows (pure).

    One row per native prediction BLOCK (the ``(variant, partition, fold)`` granularity the native parquet
    stores), carrying the identity columns + the per-sample arrays as nested JSON lists. NEVER recomputes a
    score — the per-row ``scores`` are read verbatim. The score_set ``reports[]`` remain the metric source
    of truth; these rows are the drill-down arrays a view joins by ``sample_indices``.
    """
    rows: list[dict[str, Any]] = []
    for entry in predictions.filter_predictions(load_arrays=True):
        fold_id = entry.get("fold_id")
        config_name = entry.get("config_name")
        sample_indices = entry.get("sample_indices")
        if isinstance(sample_indices, np.ndarray):
            sample_indices = sample_indices.tolist()
        rows.append(
            {
                "partition": entry.get("partition", ""),
                "fold_id": str(fold_id) if fold_id is not None else None,
                # No dedicated native variant_id column — a variant's identity is its config_name (the
                # native parquet projection mirrors config_name into variant_id the same way).
                "variant_id": str(config_name) if config_name else None,
                "model_name": entry.get("model_name", ""),
                "sample_indices": [int(i) for i in (sample_indices or [])],
                "y_true": _to_jsonable_array(entry.get("y_true")),
                "y_pred": _to_jsonable_array(entry.get("y_pred")),
                "y_proba": _to_jsonable_array(entry.get("y_proba")),
                "scores": entry.get("scores") or {},
                "metric": entry.get("metric", ""),
                "task_type": entry.get("task_type", ""),
            }
        )
    return rows


@dataclass
class RtResult:
    """The unified runtime result envelope (RT-002) — a pure projection of the dag-ml native triple.

    Anchored on the dag-ml ScoreSet: ``reports`` is VERBATIM ``score_set.reports[]``. The Studio
    ``ChainSummary`` (pivot ``group_by(run_id, variant)``) and the Web ``RunResult`` (nest ``group_by(kind)``)
    are both deterministic VIEWS of this envelope — neither is the contract. Constructed by
    :meth:`from_native_dir` (wrapping :func:`~nirs4all.pipeline.dagml.native_results.read_native_results`)
    or :meth:`from_run_result` (reading ``RunResult._dagml_score_set`` + projecting the in-memory
    predictions); both are pure projections with no recompute. ``manifest`` mirrors the native
    ``manifest.json`` header; ``diagnostics`` carries any :class:`RtError` (e.g. "ran legacy because <cause>").
    """

    schema_version: int
    run_id: str | None
    plan_id: str | None
    selection: dict[str, Any] | None
    reports: list[dict[str, Any]]
    predictions: list[dict[str, Any]]
    manifest: dict[str, Any]
    artifacts: list[dict[str, Any]] | None = None
    diagnostics: list[RtError] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The wire dict (matches ``rt_result.v1.schema.json``). ``diagnostics`` serialize via RtError.to_dict."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "selection": self.selection,
            "reports": self.reports,
            "predictions": self.predictions,
            "manifest": self.manifest,
        }
        if self.artifacts is not None:
            payload["artifacts"] = self.artifacts
        if self.diagnostics:
            payload["diagnostics"] = [d.to_dict() for d in self.diagnostics]
        return payload

    @staticmethod
    def _manifest_view(engine: str, *, fingerprints: dict[str, Any], capabilities: dict[str, Any], files: dict[str, Any]) -> dict[str, Any]:
        """The RtResult ``manifest`` sub-object (RT-002 shape) — ``portable_level`` is the CAP-002 slot (``None`` in V1)."""
        return {
            "engine": engine,
            "fingerprints": fingerprints,
            "capabilities": capabilities,
            # portable_level is the CAP-002 classifier — carried as an opaque referenced field, not computed here.
            "portable_level": None,
            "files": files,
        }

    @classmethod
    def from_native_dir(cls, run_dir: str | Path, *, diagnostics: list[RtError] | None = None) -> RtResult:
        """Project a native results directory (``manifest`` + ``score_set`` + ``predictions``) into an RtResult.

        Wraps :func:`~nirs4all.pipeline.dagml.native_results.read_native_results` (which hash-validates the
        ScoreSet and verify-then-loads any model artifacts), then projects — ``reports`` are the verbatim
        ScoreSet reports, ``predictions`` are the round-tripped blocks, ``manifest`` mirrors the native
        header, ``artifacts`` are the recorded ArtifactRefs. No recompute.
        """
        # Local import: native_results pulls polars/joblib; reading a native dir inherently needs them.
        from nirs4all.pipeline.dagml.native_results import read_native_results

        native = read_native_results(run_dir)
        manifest = native["manifest"]
        score_set = native["score_set"]
        selected_variant = manifest.get("selected_variant")
        fingerprints = {
            "score_set_hash": manifest.get("score_set_hash"),
            "plan_id": manifest.get("plan_id"),
            "bundle_id": manifest.get("bundle_id"),
        }
        return cls(
            schema_version=RT_RESULT_SCHEMA_VERSION,
            run_id=manifest.get("run_id"),
            plan_id=score_set.get("plan_id"),
            selection={"selected_variant": selected_variant} if selected_variant else None,
            reports=list(score_set.get("reports", [])),
            predictions=_project_predictions(native["predictions"]),
            manifest=cls._manifest_view(
                str(manifest.get("engine", "dag-ml")),
                fingerprints=fingerprints,
                capabilities=dict(manifest.get("capabilities", {})),
                files=dict(manifest.get("files", {})),
            ),
            artifacts=list(manifest.get("artifacts", [])) or None,
            diagnostics=list(diagnostics or []),
        )

    @classmethod
    def from_run_result(cls, result: RunResult) -> RtResult:
        """Project a :class:`~nirs4all.api.result.RunResult` into an RtResult (pure; works for both engines).

        For a dag-ml result the ScoreSet is read from ``result._dagml_score_set`` (``reports`` verbatim,
        ``engine="dag-ml"``). For a LEGACY result (no native ScoreSet — e.g. the transparent fallback) the
        envelope is sparse (``reports=[]``, ``engine`` from ``per_dataset``) but still carries the
        predictions projection and any attached :class:`RtError` diagnostics, so a caller always sees a
        uniform "ran <engine> because <cause>" envelope. No recompute — scores come from the ScoreSet rows.
        """
        score_set = result._dagml_score_set  # noqa: SLF001 — the captured raw native ScoreSet (None for legacy)
        diagnostics: list[RtError] = list(getattr(result, "_rt_diagnostics", []) or [])
        best = result.best if result.num_predictions else {}
        selected_variant = str(best.get("config_name") or "") if isinstance(best, dict) else ""

        if score_set is not None:
            from nirs4all.pipeline.dagml.native_results import _score_set_hash  # noqa: PLC0415 — shared hash for fingerprint parity

            fingerprints = {"score_set_hash": _score_set_hash(score_set), "plan_id": score_set.get("plan_id"), "bundle_id": score_set.get("bundle_id")}
            capabilities = {"has_model_artifacts": bool(result._dagml_refit_artifacts), "has_aggregate_predictions": False}  # noqa: SLF001
            engine = "dag-ml"
            reports = list(score_set.get("reports", []))
            plan_id = score_set.get("plan_id")
        else:
            fingerprints = {"score_set_hash": None, "plan_id": None, "bundle_id": None}
            capabilities = {"has_model_artifacts": False, "has_aggregate_predictions": False}
            engine = next((str(info.get("engine")) for info in result.per_dataset.values() if isinstance(info, dict) and info.get("engine")), "legacy")
            reports = []
            plan_id = None

        return cls(
            schema_version=RT_RESULT_SCHEMA_VERSION,
            run_id=None,
            plan_id=plan_id,
            selection={"selected_variant": selected_variant} if selected_variant else None,
            reports=reports,
            predictions=_project_predictions(result.predictions),
            manifest=cls._manifest_view(engine, fingerprints=fingerprints, capabilities=capabilities, files={}),
            artifacts=None,
            diagnostics=diagnostics,
        )


@dataclass
class RtRunRequest:
    """The unified runtime run request envelope (RT-002) — the ``run`` verb's input shape.

    ``{ pipeline_dsl, dataset_ref, cv, execution_backend, options }`` — the union of the ``run_via_dagml``
    field set, the ``ExecutionDriverCapability`` ``execution_backend`` taxonomy
    (``local-python | wasm-local | cluster``), and the NET-NEW ``options.allow_fallback`` strict-mode flag.
    A thin descriptor the Studio/Web runtimes fill and serialize; nirs4all's ``run()`` still executes the
    pipeline (V1 does not re-route execution through this envelope). ``execution_backend`` (the environment)
    stays orthogonal to the ML ``engine`` (which rides in ``options``).
    """

    pipeline_dsl: Any
    dataset_ref: Any
    cv: dict[str, Any] = field(default_factory=dict)
    execution_backend: str = "local-python"
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The wire dict (matches ``rt_run_request.v1.schema.json``)."""
        return {
            "pipeline_dsl": self.pipeline_dsl,
            "dataset_ref": self.dataset_ref,
            "cv": self.cv,
            "execution_backend": self.execution_backend,
            "options": self.options,
        }
