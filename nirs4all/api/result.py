"""
Result classes for nirs4all API.

These dataclasses wrap the outputs from pipeline execution, prediction,
and explanation operations, providing convenient accessor methods.

Classes:
    RunResult: Result from nirs4all.run()
    PredictResult: Result from nirs4all.predict()
    ExplainResult: Result from nirs4all.explain()

Phase 1 Implementation (v0.6.0):
    - RunResult: Full implementation with best, best_score, top(), export()
    - PredictResult: Full implementation with values, to_dataframe()
    - ExplainResult: Full implementation with values, feature attributions
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from nirs4all.core.logging import get_logger

if TYPE_CHECKING:
    from nirs4all.data.predictions import Predictions
    from nirs4all.pipeline import PipelineRunner
    from nirs4all.pipeline.execution.refit.config_extractor import RefitConfig
    from nirs4all.pipeline.execution.refit.model_selector import PerModelSelection

logger = get_logger(__name__)


def _plain_mapping(value: Any) -> dict[str, Any] | None:
    """Return a shallow dict copy when *value* is mapping-like."""
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _metadata_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _materialization_manifest(value: Any) -> dict[str, Any] | None:
    """Extract the materialization manifest from a relation replay payload."""
    manifest = _plain_mapping(value)
    if manifest is None:
        return None
    inner = manifest.get("materialization_manifest")
    if isinstance(inner, Mapping):
        return dict(inner)
    if manifest.get("representation") is not None and any(key in manifest for key in ("headers", "model_headers", "shape", "fingerprint")):
        return manifest
    return None


def _relation_manifest_from_metadata(metadata: Any) -> dict[str, Any] | None:
    meta = _metadata_mapping(metadata)
    for key in ("relation_replay_manifest", "relation_materialization_manifest"):
        manifest = _plain_mapping(meta.get(key))
        if manifest is not None:
            return manifest
    return None


def _derive_relation_lineage(
    manifest: Any,
    *,
    feature_names: Sequence[str] | None = None,
    n_features: int | None = None,
) -> Any | None:
    payload = _plain_mapping(manifest)
    if payload is None:
        return None
    from nirs4all.pipeline.explain_lineage import derive_relation_explain_lineage

    return derive_relation_explain_lineage(
        payload,
        feature_names=feature_names,
        n_features=n_features,
    )


def _lineage_by_feature(feature_lineage: Mapping[str, Any], feature: str | int) -> dict[str, Any]:
    if isinstance(feature, int):
        for lineage in feature_lineage.values():
            if isinstance(lineage, Mapping) and lineage.get("feature_index") == feature:
                return dict(lineage)
        names = list(feature_lineage)
        if 0 <= feature < len(names):
            lineage = feature_lineage.get(names[feature])
            return dict(lineage) if isinstance(lineage, Mapping) else {}
        return {}

    lineage = feature_lineage.get(feature)
    return dict(lineage) if isinstance(lineage, Mapping) else {}


# The on-disk extensions that the legacy ``export_model`` format-inference maps to the joblib serialization
# backend: a ``.joblib`` (or any UNRECOGNIZED extension, whose ``format_map.get(ext, 'joblib')`` default is
# joblib). The non-joblib extensions (``.pkl``/``.pickle`` → cloudpickle, ``.h5``/``.hdf5`` → keras_h5,
# ``.keras`` → tensorflow_keras, ``.pt``/``.pth`` → pytorch_state_dict) are the ones the joblib-only native
# helper must decline; the default dag-ml path then refuses unless explicit legacy-refit compatibility was
# requested before the helper is attempted.
_NON_JOBLIB_EXTENSIONS = frozenset({".pkl", ".pickle", ".h5", ".hdf5", ".keras", ".pt", ".pth"})
_DAGML_FUSION_PRODUCER_NODE = "merge:fusion"
_DAGML_STACKING_PRODUCER_NODE = "merge:stack"
_DAGML_BRANCH_ARTIFACT_PREFIX = "artifact:branch:"
_DAGML_BY_SOURCE_MODEL_PREFIX = "by_source_"
_DAGML_LEGACY_REFIT_COMPATIBILITY = "legacy-refit"
_DAGML_EXPORT_UNSUPPORTED_CAPABILITY = "dagml_native_export"


def _request_is_joblib(output_path: str | Path, format: str | None) -> bool:
    """Whether an ``export_model`` request resolves to the joblib backend (so the native helper may fire).

    Mirrors the legacy ``export_model`` format contract EXACTLY: an EXPLICIT ``format`` is honored verbatim
    (joblib only when it is the literal ``"joblib"``); with ``format=None`` the extension decides (a
    ``.joblib`` or any unrecognized extension defaults to joblib via ``format_map.get(ext, 'joblib')``, while
    the framework extensions in :data:`_NON_JOBLIB_EXTENSIONS` resolve to cloudpickle / keras / torch). Any
    non-joblib request returns ``False`` so :meth:`RunResult._dagml_native_export_model` refuses unless the
    caller explicitly opts into legacy-refit compatibility (no silent joblib-under-foreign-extension).
    """
    if format is not None:
        return format == "joblib"
    return Path(output_path).suffix.lower() not in _NON_JOBLIB_EXTENSIONS


class _DagmlExportedModel:
    """A predict-capable wrapper over a captured dag-ml REFIT model (P3 Slice 2c-ii, Codex D6).

    The native ``export_model`` exports a model that LOADS + ``.predict()``s (a user-facing model), NOT the
    raw ``{estimator, y_transform}`` dict the native results reader returns. This wrapper holds the captured
    sklearn ``Pipeline`` (the X-transform chain + the fitted model, fit on raw ``X``) and the OPTIONAL fitted
    ``y_transform`` (the y-processing inverse), and reproduces the dag-ml run's predict path EXACTLY: it
    applies the estimator, then — when a ``y_transform`` was captured — the inverse y-transform, so
    ``predict`` returns values in the ORIGINAL target space (mirroring the node runner's predict, which
    inverse-transforms before scoring). The exported model's ``predict`` therefore equals the dag-ml run's
    scored REFIT model's predictions — it IS that model.

    Module-level (not a closure / nested class) so joblib can pickle + reload it by its stable import path.
    When ``y_transform`` is ``None`` the wrapper is a pass-through over the estimator (single-target /
    multi-target shape is preserved by the estimator itself).
    """

    def __init__(self, estimator: Any, y_transform: Any) -> None:
        self.estimator = estimator
        self.y_transform = y_transform

    def predict(self, X: Any) -> np.ndarray:
        """Predict in the ORIGINAL target space: estimator, then inverse y-transform when present."""
        pred = np.asarray(self.estimator.predict(X), dtype=float)
        if self.y_transform is None:
            return pred
        return np.asarray(self.y_transform.inverse_transform(pred.reshape(len(pred), -1)), dtype=float)


class _DagmlNativeFusionModel:
    """Predict-capable wrapper for a native branch-fusion run's captured REFIT branch models.

    Native duplication-fusion computes its final prediction by averaging each branch model's REFIT
    prediction in original target space. Each member is therefore the same `_DagmlExportedModel` used by
    the single-artifact native export; this wrapper only normalizes shapes and averages the member outputs.
    """

    def __init__(self, members: Sequence[_DagmlExportedModel]) -> None:
        if not members:
            raise ValueError("native fusion export requires at least one member model")
        self.members = list(members)

    def predict(self, X: Any) -> np.ndarray:
        member_preds: list[np.ndarray] = []
        for member in self.members:
            pred = np.asarray(member.predict(X), dtype=float)
            member_preds.append(pred.reshape(len(pred), -1))
        shapes = {pred.shape for pred in member_preds}
        if len(shapes) != 1:
            raise ValueError(f"native fusion member predictions have incompatible shapes: {sorted(shapes)!r}")
        return cast(np.ndarray, np.mean(np.stack(member_preds, axis=0), axis=0))


class _DagmlNativeBySourceFusionModel:
    """Predict-capable wrapper for native by_source mean-fusion artifacts.

    Each captured member model was fit on one source block, not the full concatenated feature matrix. Public
    bundle prediction currently hands the exported model the legacy raw concat matrix, so this wrapper splits
    that matrix back into source blocks using the fitted estimators' feature widths before averaging member
    predictions in original target space.
    """

    def __init__(self, members: Sequence[tuple[int, _DagmlExportedModel]]) -> None:
        if not members:
            raise ValueError("native by_source fusion export requires at least one member model")
        ordered = sorted(members, key=lambda item: item[0])
        source_indices = [index for index, _member in ordered]
        if source_indices != list(range(len(source_indices))):
            raise ValueError(f"native by_source fusion export requires contiguous source indices from 0: {source_indices!r}")
        self.source_indices = source_indices
        self.members = [member for _index, member in ordered]
        self.source_widths = [_estimator_feature_width(member.estimator) for member in self.members]

    def predict(self, X: Any) -> np.ndarray:
        blocks = self._source_blocks(X)
        member_preds: list[np.ndarray] = []
        for source_index, member in enumerate(self.members):
            pred = np.asarray(member.predict(blocks[source_index]), dtype=float)
            member_preds.append(pred.reshape(len(pred), -1))
        shapes = {pred.shape for pred in member_preds}
        if len(shapes) != 1:
            raise ValueError(f"native by_source fusion member predictions have incompatible shapes: {sorted(shapes)!r}")
        return cast(np.ndarray, np.mean(np.stack(member_preds, axis=0), axis=0))

    def _source_blocks(self, X: Any) -> list[np.ndarray]:
        if isinstance(X, (list, tuple)) and not isinstance(X, (str, bytes)):
            blocks = [np.asarray(block) for block in X]
            if len(blocks) == len(self.members):
                return blocks

        arr = np.asarray(X)
        if arr.ndim == 3:
            if arr.shape[0] == len(self.members):
                return [np.asarray(arr[index]) for index in range(len(self.members))]
            if arr.shape[1] == len(self.members):
                return [np.asarray(arr[:, index, :]) for index in range(len(self.members))]
        if arr.ndim == 2:
            if any(width is None for width in self.source_widths):
                raise ValueError("native by_source fusion export cannot split a concatenated feature matrix because at least one source width is unknown")
            widths = [int(width) for width in self.source_widths if width is not None]
            expected = sum(widths)
            if arr.shape[1] != expected:
                raise ValueError(f"native by_source fusion export expected {expected} concatenated features from source widths {widths!r}, got {arr.shape[1]}")
            stops = np.cumsum(widths)[:-1]
            return [np.asarray(block) for block in np.split(arr, stops, axis=1)]

        raise ValueError(
            "native by_source fusion export expects either a concatenated 2D feature matrix or one 2D matrix per source"
        )


class _DagmlNativeStackingModel:
    """Predict-capable wrapper for native branch stacking artifacts.

    The base members are captured REFIT models that predict in original target space. The meta member is
    the captured REFIT meta-model, fit by dag-ml on base prediction columns ordered by the native
    ``stacking_replay`` manifest. Prediction rebuilds that same meta-feature matrix from raw X.
    """

    def __init__(self, base_members: Sequence[_DagmlExportedModel], meta_member: _DagmlExportedModel) -> None:
        if len(base_members) < 2:
            raise ValueError("native stacking export requires at least two base member models")
        self.base_members = list(base_members)
        self.meta_member = meta_member

    def predict(self, X: Any) -> np.ndarray:
        base_blocks: list[np.ndarray] = []
        expected_rows: int | None = None
        for member in self.base_members:
            pred = np.asarray(member.predict(X), dtype=float)
            rows = len(pred)
            if expected_rows is None:
                expected_rows = rows
            elif rows != expected_rows:
                raise ValueError(f"native stacking base predictions have incompatible row counts: expected {expected_rows}, got {rows}")
            base_blocks.append(pred.reshape(rows, -1))
        x_meta = np.column_stack(base_blocks)
        return np.asarray(self.meta_member.predict(x_meta), dtype=float)


def _native_manifest_strings(manifest: Mapping[str, Any], key: str) -> set[str]:
    value = manifest.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    return {str(item) for item in value}


def _score_set_final_producers(score_set: Mapping[str, Any] | None) -> set[str]:
    if score_set is None:
        return set()
    reports = score_set.get("reports")
    if not isinstance(reports, Sequence):
        return set()
    producers: set[str] = set()
    for report in reports:
        if not isinstance(report, Mapping):
            continue
        if report.get("fold_id") is None and report.get("partition") in {"final", "test"}:
            producer = report.get("producer_node")
            if producer is not None:
                producers.add(str(producer))
    return producers


def _native_final_producers(native: Mapping[str, Any]) -> set[str]:
    manifest = native.get("manifest")
    producers = _native_manifest_strings(manifest, "final_producer_nodes") if isinstance(manifest, Mapping) else set()
    return producers or _score_set_final_producers(cast(Mapping[str, Any] | None, native.get("score_set")))


def _native_model_names(native_manifest: Mapping[str, Any]) -> list[str]:
    value = native_manifest.get("model_names")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _native_manifest_is_by_source(native_manifest: Mapping[str, Any]) -> bool:
    model_names = _native_model_names(native_manifest)
    return len(model_names) == 1 and model_names[0].startswith(_DAGML_BY_SOURCE_MODEL_PREFIX)


def _artifact_branch_index(artifact: Mapping[str, Any]) -> int | None:
    value = artifact.get("branch_index")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    artifact_id = artifact.get("artifact_id")
    if not isinstance(artifact_id, str):
        return None
    parts = artifact_id.split(":", 3)
    if len(parts) < 3 or parts[0] != "artifact" or parts[1] != "branch":
        return None
    try:
        return int(parts[2].split(".", 1)[0])
    except ValueError:
        return None


def _indexed_branch_artifacts(artifacts: Sequence[Mapping[str, Any]]) -> list[tuple[int, Mapping[str, Any]]] | None:
    indexed: list[tuple[int, Mapping[str, Any]]] = []
    for artifact in artifacts:
        index = _artifact_branch_index(artifact)
        if index is None:
            return None
        indexed.append((index, artifact))
    indexed.sort(key=lambda item: item[0])
    indices = [index for index, _artifact in indexed]
    if indices != list(range(len(indices))):
        return None
    return indexed


def _estimator_feature_width(estimator: Any) -> int | None:
    """Best-effort fitted input width for a sklearn estimator or Pipeline."""
    value = getattr(estimator, "n_features_in_", None)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    steps = getattr(estimator, "steps", None)
    if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
        for step in reversed(steps):
            if not isinstance(step, Sequence) or isinstance(step, (str, bytes)) or len(step) < 2:
                continue
            width = getattr(step[1], "n_features_in_", None)
            if width is not None:
                try:
                    return int(width)
                except (TypeError, ValueError):
                    continue
    return None


def _dagml_native_bundle_provenance(
    native_manifest: Mapping[str, Any],
    *,
    export_path: str,
    artifact_count: int,
    export_shape: str | None = None,
) -> dict[str, Any]:
    provenance = {
        "source_type": "dagml_native",
        "export_path": export_path,
        "dagml_run_id": native_manifest.get("run_id"),
        "dagml_plan_id": native_manifest.get("plan_id"),
        "dagml_bundle_id": native_manifest.get("bundle_id"),
        "dagml_selected_variant": native_manifest.get("selected_variant"),
        "dagml_artifact_count": artifact_count,
    }
    if export_shape is not None:
        provenance["dagml_native_export_shape"] = export_shape
    return provenance


def _is_native_branch_fusion_bundle(native: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]]) -> bool:
    """Whether a multi-artifact native result is the narrow duplication-fusion shape export can replay."""
    if len(artifacts) < 2 or _DAGML_FUSION_PRODUCER_NODE not in _native_final_producers(native):
        return False
    if not all(str(artifact.get("artifact_id") or "").startswith(_DAGML_BRANCH_ARTIFACT_PREFIX) for artifact in artifacts):
        return False

    manifest = native.get("manifest")
    # By-source fusion also uses branch model nodes, but each captured estimator expects one source block,
    # so it is handled by the separate source-splitting wrapper below rather than this full-X wrapper.
    return not (isinstance(manifest, Mapping) and _native_manifest_is_by_source(manifest))


def _is_native_by_source_fusion_bundle(native: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]]) -> bool:
    """Whether a native by_source mean-fusion result has enough branch metadata to replay."""
    if len(artifacts) < 2 or _DAGML_FUSION_PRODUCER_NODE not in _native_final_producers(native):
        return False
    manifest = native.get("manifest")
    if not isinstance(manifest, Mapping) or not _native_manifest_is_by_source(manifest):
        return False
    if not all(str(artifact.get("artifact_id") or "").startswith(_DAGML_BRANCH_ARTIFACT_PREFIX) for artifact in artifacts):
        return False
    return _indexed_branch_artifacts(artifacts) is not None


def _native_stacking_artifacts(native_manifest: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]] | None:
    """Return ``(base_artifacts_in_meta_feature_order, meta_artifact)`` from ``stacking_replay``."""
    replay = native_manifest.get("stacking_replay")
    if not isinstance(replay, Mapping) or replay.get("producer_node") != _DAGML_STACKING_PRODUCER_NODE:
        return None
    construction = replay.get("meta_feature_construction")
    if not isinstance(construction, Mapping) or construction.get("kind") != "base_prediction_column_stack":
        return None

    by_id = {str(artifact.get("artifact_id")): artifact for artifact in artifacts if artifact.get("artifact_id") is not None}
    meta_artifact_id = replay.get("meta_artifact_id")
    if meta_artifact_id is None or str(meta_artifact_id) not in by_id:
        return None
    base_producers = replay.get("base_producers")
    if not isinstance(base_producers, Sequence) or isinstance(base_producers, (str, bytes)) or len(base_producers) < 2:
        return None

    base_artifacts: list[Mapping[str, Any]] = []
    for producer in base_producers:
        if not isinstance(producer, Mapping):
            return None
        artifact_id = producer.get("artifact_id")
        if artifact_id is None or str(artifact_id) not in by_id:
            return None
        base_artifacts.append(by_id[str(artifact_id)])
    return base_artifacts, by_id[str(meta_artifact_id)]


@dataclass
class ModelRefitResult:
    """Per-model refit result metadata.

    Captures the refit outcome for a single model node in the pipeline.
    For Phase 2 (non-stacking), there is only one model node, so
    ``RunResult.models`` will contain exactly one entry.

    Attributes:
        model_name: Name of the model (e.g. ``"PLSRegression"``).
        final_entry: The refit prediction entry (``fold_id="final"``).
        cv_entry: The best CV prediction entry for this model.
        final_score: Test score from the refit model.
        cv_score: Best validation score from CV.
        metric: Metric name used for evaluation.
    """

    model_name: str = ""
    final_entry: dict[str, Any] = field(default_factory=dict)
    cv_entry: dict[str, Any] = field(default_factory=dict)
    final_score: float | None = None
    cv_score: float | None = None
    metric: str = ""

class LazyModelRefitResult:
    """Lazy per-model refit result that triggers refit on first access.

    Wraps a :class:`PerModelSelection` and defers the actual refit
    execution until a property requiring the result (e.g. ``.score``,
    ``.final_entry``, ``.export()``) is accessed.

    After the first access, the result is cached so subsequent accesses
    return instantly.

    If the underlying resources (artifact registry, step cache) have
    been destroyed by the time of access, the lazy refit re-executes
    from scratch.

    Attributes:
        model_name: Name of the model.
        selection: The PerModelSelection metadata from the CV phase.
    """

    def __init__(
        self,
        model_name: str,
        selection: PerModelSelection,
        refit_config: RefitConfig,
        dataset: Any,
        context: Any,
        runtime_context: Any,
        artifact_registry: Any,
        executor: Any,
        prediction_store: Any,
    ) -> None:
        self.model_name = model_name
        self.selection = selection
        self._refit_config = refit_config
        self._dataset = dataset
        self._context = context
        self._runtime_context = runtime_context
        self._artifact_registry = artifact_registry
        self._executor = executor
        self._prediction_store = prediction_store
        self._result: ModelRefitResult | None = None
        self._lock = threading.Lock()

    def _execute_refit(self) -> ModelRefitResult:
        """Execute the refit and return a ModelRefitResult.

        Thread-safe: uses a lock to prevent concurrent refits.
        """
        with self._lock:
            if self._result is not None:
                return self._result

            # Build a RefitConfig from the selection metadata
            from nirs4all.pipeline.execution.refit.config_extractor import RefitConfig
            from nirs4all.pipeline.execution.refit.executor import execute_simple_refit

            refit_config = RefitConfig(
                expanded_steps=self.selection.expanded_steps,
                best_params=self.selection.best_params,
                variant_index=self.selection.variant_index,
                metric=self._refit_config.metric,
                selection_score=self.selection.best_score,
            )

            # Create a fresh prediction store for capturing refit predictions
            from nirs4all.data.predictions import Predictions

            refit_predictions = Predictions()

            try:
                refit_result = execute_simple_refit(
                    refit_config=refit_config,
                    dataset=self._dataset,
                    context=self._context,
                    runtime_context=self._runtime_context,
                    artifact_registry=self._artifact_registry,
                    executor=self._executor,
                    prediction_store=refit_predictions,
                )
            except Exception:
                logger.warning(
                    f"Lazy refit for model '{self.model_name}' failed. "
                    f"Resources may have been destroyed."
                )
                # Return a minimal result with just the CV info
                self._result = ModelRefitResult(
                    model_name=self.model_name,
                    cv_score=self.selection.best_score,
                    metric=self._refit_config.metric,
                )
                return self._result

            # Build ModelRefitResult from the refit output
            final_entries = refit_predictions.iter_entries(fold_id="final")
            final_entry = final_entries[0] if final_entries else {}

            self._result = ModelRefitResult(
                model_name=self.model_name,
                final_entry=final_entry,
                cv_entry={},
                final_score=refit_result.test_score,
                cv_score=self.selection.best_score,
                metric=refit_result.metric,
            )
            return self._result

    def _ensure_result(self) -> ModelRefitResult:
        """Ensure the refit has been executed and return the result."""
        if self._result is None:
            return self._execute_refit()
        return self._result

    @property
    def score(self) -> float | None:
        """Get the refit model's test score (triggers refit on first access)."""
        return self._ensure_result().final_score

    @property
    def final_score(self) -> float | None:
        """Get the refit model's test score (triggers refit on first access)."""
        return self._ensure_result().final_score

    @property
    def final_entry(self) -> dict[str, Any]:
        """Get the refit prediction entry (triggers refit on first access)."""
        return self._ensure_result().final_entry

    @property
    def cv_entry(self) -> dict[str, Any]:
        """Get the best CV prediction entry."""
        return self._ensure_result().cv_entry

    @property
    def cv_score(self) -> float | None:
        """Get the best CV validation score (does not trigger refit)."""
        return self.selection.best_score

    @property
    def metric(self) -> str:
        """Get the metric name (does not trigger refit)."""
        return self._refit_config.metric

    @property
    def is_resolved(self) -> bool:
        """Check if the refit has already been executed."""
        return self._result is not None

    def __repr__(self) -> str:
        status = "resolved" if self.is_resolved else "pending"
        return f"LazyModelRefitResult(model='{self.model_name}', status={status})"

@dataclass
class RunResult:
    """Result from nirs4all.run().

    Provides convenient access to predictions, best model, and artifacts.
    Wraps the raw (predictions, per_dataset) tuple returned by PipelineRunner.run().

    Attributes:
        predictions: Predictions object containing all pipeline results.
        per_dataset: Dictionary with per-dataset execution details.

    Properties:
        best: Best prediction entry by default ranking.
        best_score: Best model's primary test score (CV selection metric).
        best_rmse: Best model's RMSE (regression).
        best_r2: Best model's R² (regression).
        best_accuracy: Best model's accuracy (classification).
        final: Refit model prediction entry (``fold_id="final"``), or ``None``.
        final_score: Refit model test score, or ``None`` if no refit.
        cv_best: Best CV prediction entry.
        cv_best_score: Best CV validation score.
        models: Per-model refit results (for Phase 2, one entry).
        artifacts_path: Path to run artifacts directory.
        num_predictions: Total number of predictions stored.

    Key Operations:
        top(n): Get top N predictions by ranking.
        export(path): Export best model to .n4a bundle.
        filter(**kwargs): Filter predictions by criteria.
        get_datasets(): Get list of unique dataset names.
        get_models(): Get list of unique model names.

    Example:
        >>> result = nirs4all.run(pipeline, dataset)
        >>> print(f"Best RMSE: {result.best_rmse:.4f}")
        >>> print(f"Best R²: {result.best_r2:.4f}")
        >>> result.export("exports/best_model.n4a")
    """

    predictions: Predictions
    per_dataset: dict[str, Any]
    _runner: PipelineRunner | None = field(default=None, repr=False)
    _owns_runner: bool = field(default=True, repr=False)
    _workspace_path: Path | None = field(default=None, repr=False)

    # Lazy refit dependencies (set by the orchestrator when per-model
    # selections are available so that ``models`` returns lazy results)
    _per_model_selections: dict[str, PerModelSelection] | None = field(default=None, repr=False)
    _refit_config: RefitConfig | None = field(default=None, repr=False)
    _refit_dataset: Any = field(default=None, repr=False)
    _refit_context: Any = field(default=None, repr=False)
    _refit_runtime_context: Any = field(default=None, repr=False)
    _refit_artifact_registry: Any = field(default=None, repr=False)
    _refit_executor: Any = field(default=None, repr=False)

    # dag-ml legacy-refit compatibility inputs: the dag-ml backend returns scores in memory with NO legacy
    # workspace, so the default V1 export path must use captured native artifacts or refuse. This spec
    # FREEZES the run inputs (a deepcopy of the pipeline + the dataset deepcopied for in-memory forms / kept
    # as a stable path/config ref otherwise, plus name/random_state) at dag-ml run time solely for the
    # explicit ``compatibility="legacy-refit"`` opt-in. That opt-in re-runs the SAME frozen pipeline through
    # the LEGACY engine (save_artifacts=True), producing the workspace + chain + artifacts the legacy export
    # path needs. ``_dagml_legacy_result`` caches that materialized legacy RunResult (kept alive so its
    # workspace store stays open for the export, closed on close()).
    #
    # PARITY SCOPE (explicit compatibility): for a FULLY-SEEDED deterministic run the legacy refit can
    # reproduce the dag-ml-scored model (engine numerical parity); otherwise the export is BEST-EFFORT. A
    # per-run WARNING fires on the two ``_dagml_export_stochastic`` signals — (a) CERTAIN: a
    # sample_augmentation step (re-augmentation is non-reproducible across processes and the augmenter's
    # own RNG is not covered by run(random_state)); (b) CONSERVATIVE: run(random_state) is None (nothing
    # globally seeded — this may over-warn a fully-deterministic pipeline, the safe direction for a "may
    # differ" caveat). A per-estimator "unseeded-stochastic?" probe is NOT attempted: random_state use is
    # solver/config-conditional (Ridge() / PCA(svd_solver="full") carry a DORMANT random_state=None yet are
    # deterministic → false alarm; MLPRegressor(shuffle=False) is stochastic via weight init; wrapped
    # estimators hide theirs), so any static heuristic both over- and under-warns. The uncertain middle (a
    # seeded run whose individual component left random_state=None) is documented in the export()/
    # export_model() docstrings (general caveat), not warned. P3 (native fitted-model capture) removes the
    # limitation by exporting the actual scored artifacts. Reloadable on-disk-PATH datasets must be
    # unchanged at explicit compatibility export time (only non-reloadable / in-memory dataset forms are
    # snapshotted; a path/file config is replayed from disk).
    _dagml_export_spec: dict[str, Any] | None = field(default=None, repr=False)
    _dagml_legacy_result: RunResult | None = field(default=None, repr=False)
    _dagml_export_stochastic: bool = field(default=False, repr=False)

    # The RAW native dag-ml ScoreSet (the canonical object the projection consumed), captured at
    # projection time so the native-results writer (P3 Slice 2b-i, OFF by default) can persist it
    # VERBATIM. In-memory metadata only; ``None`` for a legacy result.
    _dagml_score_set: dict[str, Any] | None = field(default=None, repr=False)

    # The fitted REFIT estimators the dag-ml run produced (P3 Slice 2c-i), captured host-side from the
    # in-process model store at projection time so the native-results writer can joblib-persist them as
    # loadable model artifacts (retiring the P1c legacy-refit export bridge). Each entry is
    # ``{artifact_id, estimator, y_transform, kind, controller_id, backend}``. A LIST (D3: a branch /
    # stacking / operator-expanded run emits several REFIT artifacts). Empty for the subprocess mechanism
    # (its child-process models are unreachable) and for a legacy result. In-memory metadata only, OFF by
    # default (the writer fires solely when native results are enabled).
    _dagml_refit_artifacts: list[dict[str, Any]] = field(default_factory=list, repr=False)

    # The on-disk native results directory the 2b-i writer produced for this dag-ml run (recorded by
    # ``run_via_dagml`` when native results were enabled; ``None`` for an in-memory-only dag-ml run or a
    # legacy result). It holds ``manifest.json`` + ``score_set.json`` + ``predictions.parquet`` + the
    # ``artifacts/`` model tree. P3 Slice 2c-ii uses it for native exports: a dag-ml run with EXACTLY ONE
    # concrete model artifact exports that captured (verify-then-load) estimator DIRECTLY, and `.n4a`
    # additionally supports narrow multi-artifact branch mean-fusion wrappers, including by_source fusion
    # when branch/source order is recoverable. Other unsupported shapes refuse on the default V1 path unless
    # the caller explicitly requests ``compatibility="legacy-refit"``.
    _dagml_results_dir: Path | None = field(default=None, repr=False)

    # --- Lifecycle ---

    def detach(self) -> None:
        """Detach from the runner by closing its store and dropping the reference.

        After detaching, the RunResult operates in detached mode: export
        operations re-open the store on demand.  This releases the DB
        connection so other processes can access the workspace.

        Called automatically by :func:`nirs4all.run` for non-session runs.
        """
        if self._runner is not None and self._owns_runner:
            if self._workspace_path is None:
                self._workspace_path = getattr(self._runner, 'workspace_path', None)
            self._runner.close()
            self._runner = None

    def close(self) -> None:
        """Close the underlying WorkspaceStore to release DB resources.

        Safe to call multiple times.  For detached results this is a no-op.
        Session-owned runners are closed by the session. A dag-ml export bridge's
        materialized legacy result (if any) is closed too, releasing its workspace store.
        """
        if self._runner is not None and self._owns_runner:
            self._runner.close()
        if self._dagml_legacy_result is not None:
            self._dagml_legacy_result.close()

    def __enter__(self) -> RunResult:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        """Safety net: close store if caller forgot."""
        with contextlib.suppress(Exception):
            self.close()

    # --- Primary accessors ---

    @property
    def best(self) -> dict[str, Any]:
        """Get the best prediction entry, preferring refit (final) models.

        When refit entries exist, returns the best final entry.
        Otherwise falls back to the best CV entry.

        Returns:
            Dictionary containing best model's metrics, name, and configuration.
            Empty dict if no predictions available.
        """
        final = self.best_final
        if final:
            return final
        return self.cv_best

    def _selected_metric(self, metric: str, *test_score_aliases: str) -> float:
        """Read ``metric`` from the SELECTED model — the one ``best``/``best_score`` describe.

        Shared anchor for ``best_rmse`` / ``best_r2`` / ``best_accuracy`` so every scalar
        shortcut reports the SAME model (the selection-metric winner: the refit/``best_final``
        entry, or the selected CV entry for a no-refit run), never a per-metric-reranked row.

        Lookup tiers on that one entry: a flat ``metric`` key (from ``display_metrics``), then the
        per-partition ``scores['test'][metric]`` block, then ``test_score`` when the entry's own
        selection ``metric`` is one of ``test_score_aliases`` (so e.g. ``best_rmse`` still returns
        the test score of an rmse-selected entry that carries no expanded ``scores`` dict).

        Re-ranking each shortcut independently via ``get_best(metric=X)`` (the prior behaviour) is
        a BUG: it ranks rows by their VALIDATION ``X`` and, under cross-validation, the X-best row
        is a different *fold* model than the selection winner — so ``best_r2`` returned a CV fold's
        test R² (e.g. a ShuffleSplit fold's 0.5426 instead of the selected model's 0.5499) and
        ``best_accuracy`` returned a different fold's plain accuracy than the balanced-accuracy-
        selected model's. Anchoring all of them on ``best`` makes the trio self-consistent.
        """
        best = self.best
        if not best:
            return float('nan')

        # Flat key first (from display_metrics)
        if metric in best and best[metric] is not None:
            return float(best[metric])

        # Nested per-partition scores dict
        scores = best.get('scores', {})
        if isinstance(scores, dict):
            test_scores = scores.get('test', {})
            if metric in test_scores and test_scores[metric] is not None:
                return float(test_scores[metric])

        # Fall back to test_score when the selection metric IS this metric (or an alias)
        if test_score_aliases and best.get('metric', '') in test_score_aliases:
            test_score = best.get('test_score')
            if test_score is not None:
                return float(test_score)

        return float('nan')

    @property
    def best_score(self) -> float:
        """Get the selected model's primary test score (the selection-metric value).

        Returns:
            The test_score value from best prediction, or NaN if unavailable.
        """
        score = self.best.get('test_score')
        return float(score) if score is not None else float('nan')

    @property
    def best_rmse(self) -> float:
        """Get the SELECTED model's RMSE (the same model ``best_score``/``best_r2`` describe).

        Reads RMSE from :attr:`best` — the selection-metric winner — so the scalar shortcuts are
        mutually consistent (for an rmse-selected single model this equals ``best_score``). See
        :meth:`_selected_metric` for why per-shortcut ``get_best(metric=...)`` re-ranking was wrong.

        Returns:
            RMSE value or NaN if unavailable.
        """
        return self._selected_metric('rmse', 'rmse', 'mse')

    @property
    def best_r2(self) -> float:
        """Get the SELECTED model's R² (the same model ``best_score``/``best_rmse`` describe).

        Reads R² from :attr:`best` — the selection-metric winner — instead of re-ranking by R²,
        which under CV surfaced a different fold model's test R². See :meth:`_selected_metric`.

        Returns:
            R² value or NaN if unavailable.
        """
        return self._selected_metric('r2', 'r2')

    @property
    def best_accuracy(self) -> float:
        """Get the SELECTED model's accuracy (the same model ``best_score`` describes).

        Reads plain ``accuracy`` from :attr:`best` — the selection-metric winner (selection uses
        ``balanced_accuracy``) — instead of re-ranking by accuracy, which surfaced a different
        fold model's accuracy than the selected model's. See :meth:`_selected_metric`.

        Returns:
            Accuracy value or NaN if unavailable.
        """
        return self._selected_metric('accuracy', 'accuracy')

    # --- Refit accessors ---

    @property
    def best_final(self) -> dict[str, Any]:
        """Get the best refit entry across all models.

        Filters predictions to ``fold_id="final"`` entries and ranks them
        by their selection score (``selection_score``).

        Returns:
            Best refit prediction dict, or empty dict if no refit entries.
        """
        results = self.predictions.top(n=1, score_scope="refit")
        top = cast(list, results)
        return top[0] if top else {}

    @property
    def final(self) -> dict[str, Any] | None:
        """Get the refit model prediction entry (``fold_id="final"``).

        Searches the per-dataset prediction stores where refit entries
        are stored (they are not merged into the global predictions
        buffer to avoid polluting CV-centric ranking).

        Returns:
            Prediction dict for the refit model, or ``None`` if refit
            was not performed or no refit entries exist.
        """
        # Search per-dataset prediction stores (refit entries live here)
        for ds_info in self.per_dataset.values():
            ds_preds = ds_info.get("run_predictions")
            if ds_preds is None:
                continue
            entries = ds_preds.filter_predictions(fold_id="final")
            for entry in entries:
                if str(entry.get("fold_id")) == "final":
                    return dict(entry)
        # Fallback: check global predictions (when refit entries were merged there)
        entries = self.predictions.filter_predictions(fold_id="final")
        for entry in entries:
            if str(entry.get("fold_id")) == "final":
                return dict(entry)
        return None

    @property
    def final_score(self) -> float | None:
        """Get the refit model's test score.

        Returns:
            Test score from the refit entry, or ``None`` if refit was
            not performed.
        """
        entry = self.final
        if entry is None:
            return None
        score = entry.get("test_score")
        if score is not None:
            return float(score)
        return None

    @property
    def cv_best(self) -> dict[str, Any]:
        """Get the best CV prediction entry (excludes refit entries).

        This is the prediction entry that won the cross-validation
        selection phase.  Refit entries (``fold_id="final"``) are
        excluded from ranking.

        Returns:
            Best CV prediction dict, or empty dict if no CV predictions.
        """
        results = self.predictions.top(n=1, score_scope="folds", fold_id="avg")
        top = cast(list[dict[str, Any]], results)
        if top:
            return top[0]

        results = self.predictions.top(n=1, score_scope="folds")
        top = cast(list[dict[str, Any]], results)
        return top[0] if top else {}

    @property
    def cv_best_score(self) -> float:
        """Get the best CV validation score.

        Returns:
            The val_score from the best CV entry, or NaN if unavailable.
        """
        entry = self.cv_best
        if not entry:
            return float("nan")
        score = entry.get("val_score")
        if score is not None:
            return float(score)
        return float("nan")

    @property
    def models(self) -> dict[str, ModelRefitResult | LazyModelRefitResult]:
        """Get per-model refit results.

        When per-model selections are available (set by the orchestrator),
        returns :class:`LazyModelRefitResult` instances that defer the
        actual refit until a property requiring the result is accessed.

        When per-model selections are not available, falls back to
        the eager approach using already-executed refit entries.

        Returns:
            Dictionary mapping model name to :class:`ModelRefitResult`
            or :class:`LazyModelRefitResult`.
        """
        # Lazy path: per-model selections were stored by the orchestrator
        if self._per_model_selections is not None and self._refit_config is not None:
            result: dict[str, ModelRefitResult | LazyModelRefitResult] = {}
            for model_name, selection in self._per_model_selections.items():
                result[model_name] = LazyModelRefitResult(
                    model_name=model_name,
                    selection=selection,
                    refit_config=self._refit_config,
                    dataset=self._refit_dataset,
                    context=self._refit_context,
                    runtime_context=self._refit_runtime_context,
                    artifact_registry=self._refit_artifact_registry,
                    executor=self._refit_executor,
                    prediction_store=self.predictions,
                )
            return result

        # Eager fallback: use already-executed refit entries
        final_entry = self.final
        if final_entry is None:
            return {}

        model_name = final_entry.get("model_name", "unknown")
        cv_entry = self.cv_best
        metric = final_entry.get("metric", "")

        final_score_val = final_entry.get("test_score")
        cv_score_val = cv_entry.get("val_score") if cv_entry else None

        return {
            model_name: ModelRefitResult(
                model_name=model_name,
                final_entry=final_entry,
                cv_entry=cv_entry,
                final_score=float(final_score_val) if final_score_val is not None else None,
                cv_score=float(cv_score_val) if cv_score_val is not None else None,
                metric=metric,
            )
        }

    # --- Metadata accessors ---

    @property
    def artifacts_path(self) -> Path | None:
        """Get path to workspace artifacts directory.

        Returns:
            Path to the workspace directory, or None if not available.
        """
        if self._workspace_path is not None:
            return self._workspace_path
        if self._runner and hasattr(self._runner, 'workspace_path'):
            return self._runner.workspace_path
        return None

    @property
    def num_predictions(self) -> int:
        """Get total number of predictions stored.

        Returns:
            Number of prediction entries.
        """
        return self.predictions.num_predictions

    # --- Query methods ---

    def top(self, n: int = 5, **kwargs) -> Any:
        """Get top N predictions by ranking.

        Args:
            n: Number of top predictions to return. When group_by is used,
               this means top N **per group** (e.g., top 3 per dataset).
            **kwargs: Additional arguments passed to predictions.top().
                Supported kwargs include:
                - rank_metric: Metric to rank by (default: uses record's metric)
                - rank_partition: Partition to rank on (default: "val")
                - display_partition: Partition for display metrics (default: "test")
                - aggregate_partitions: If True, include train/val/test data
                - ascending: Sort order (None = infer from metric)
                - group_by: Group predictions by column(s). Returns top N per group.
                  Each result includes 'group_key' for easy filtering.
                - return_grouped: If True with group_by, return dict of group->results
                  instead of flat list. Default: False.

        Returns:
            - If return_grouped=False (default): List of prediction dicts,
              ranked by score. With group_by, returns top N per group as flat list.
            - If return_grouped=True: Dict mapping group keys to lists of predictions.

        Examples:
            >>> # Top 5 overall
            >>> result.top(5)
            >>>
            >>> # Top 3 per dataset (flat list)
            >>> top_per_ds = result.top(3, group_by='dataset_name')
            >>> ds1 = [r for r in top_per_ds if r['group_key'] == ('my_dataset',)]
            >>>
            >>> # Top 3 per dataset (grouped dict)
            >>> grouped = result.top(3, group_by='dataset_name', return_grouped=True)
            >>> for key, results in grouped.items():
            ...     print(f"{key}: {len(results)} results")
            >>>
            >>> # Multi-column grouping: top 2 per (dataset, model) combination
            >>> top_per_combo = result.top(2, group_by=['dataset_name', 'model_name'])
            >>> # Group keys are tuples: ('wheat', 'PLSRegression'), ('corn', 'RandomForest')
            >>> for r in top_per_combo:
            ...     dataset, model = r['group_key']
            ...     print(f"{dataset}/{model}: {r['test_score']:.4f}")
        """
        return self.predictions.top(n=n, **kwargs)

    def filter(self, **kwargs) -> list[dict[str, Any]]:
        """Filter predictions by criteria.

        Args:
            **kwargs: Filter criteria passed to predictions.filter_predictions().
                Supported kwargs include:
                - dataset_name: Filter by dataset name
                - model_name: Filter by model name
                - partition: Filter by partition ('train', 'val', 'test')
                - fold_id: Filter by fold ID
                - step_idx: Filter by pipeline step index
                - branch_id: Filter by branch ID
                - load_arrays: If True, load actual arrays (default: True)

        Returns:
            List of matching prediction dictionaries.
        """
        return self.predictions.filter_predictions(**kwargs)

    def get_datasets(self) -> list[str]:
        """Get list of unique dataset names.

        Returns:
            List of dataset names in predictions.
        """
        return self.predictions.get_datasets()

    def get_models(self) -> list[str]:
        """Get list of unique model names.

        Returns:
            List of model names in predictions.
        """
        return self.predictions.get_models()

    def to_rt_result(self) -> Any:
        """Project this result into the runtime ``RtResult`` envelope (``LOCK-RT``); a pure, additive view.

        Returns a :class:`nirs4all.pipeline.dagml.rt.RtResult` — the unified runtime result envelope anchored
        on the dag-ml ScoreSet. For a ``run(engine="dag-ml")`` result, ``reports`` is the VERBATIM native
        ``score_set.reports[]`` (read from the captured ``_dagml_score_set``); for a legacy result the
        envelope is sparse (no native ScoreSet) but still carries the predictions projection and any attached
        :class:`~nirs4all.pipeline.dagml.rt.RtError` diagnostics (e.g. "ran legacy because <cause>" from the
        ``run()`` fallback). Pure projection — nothing is recomputed and this ``RunResult`` is not mutated.
        Studio's ``ChainSummary`` pivot and the Web ``RunResult`` nest are both deterministic views of it.

        Additive 0.9.x-safe seam: it does NOT touch ``RunResult``'s frozen fields, the ``.n4a`` bundle, or
        the native results format.
        """
        from nirs4all.pipeline.dagml.rt import RtResult

        return RtResult.from_run_result(self)

    @property
    def relation_replay_manifests(self) -> dict[str, dict[str, Any]]:
        """Return relation replay manifests keyed by chain or prediction id."""
        manifests: dict[str, dict[str, Any]] = {}
        chain_ids: list[str] = []

        for row in self.predictions.filter_predictions(load_arrays=False):
            row_key = str(row.get("chain_id") or row.get("prediction_id") or row.get("id") or len(manifests))
            row_manifest = _relation_manifest_from_metadata(row.get("metadata"))
            if row_manifest is not None:
                manifests.setdefault(row_key, row_manifest)

            chain_id = row.get("chain_id")
            if chain_id is not None and str(chain_id):
                chain_ids.append(str(chain_id))

        missing_chain_ids = [chain_id for chain_id in dict.fromkeys(chain_ids) if chain_id not in manifests]
        if not missing_chain_ids:
            return manifests

        try:
            with self._open_store_for_export() as store:
                if store is None:
                    return manifests
                for chain_id in missing_chain_ids:
                    chain = store.get_chain(chain_id)
                    if not isinstance(chain, Mapping):
                        continue
                    manifest = _plain_mapping(chain.get("relation_replay_manifest"))
                    if manifest is not None:
                        manifests[chain_id] = manifest
        except Exception as exc:
            logger.debug("Could not load relation replay manifests for RunResult: %s", exc)

        return manifests

    @property
    def relation_replay_manifest(self) -> dict[str, Any] | None:
        """Return the relation replay manifest for the best/only chain, if any."""
        manifests = self.relation_replay_manifests
        if not manifests:
            return None

        best = self.best
        best_chain_id = best.get("chain_id") if isinstance(best, Mapping) else None
        if best_chain_id is not None and str(best_chain_id) in manifests:
            return dict(manifests[str(best_chain_id)])

        return dict(next(iter(manifests.values())))

    @property
    def relation_materialization_manifest(self) -> dict[str, Any] | None:
        """Return materialization provenance for the best relation replay manifest."""
        return _materialization_manifest(self.relation_replay_manifest)

    @property
    def feature_lineage(self) -> dict[str, Any]:
        """Feature provenance derived from the relation manifest, when available."""
        lineage = _derive_relation_lineage(
            self.relation_replay_manifest,
            n_features=self._best_n_features(),
        )
        if lineage is None or not getattr(lineage, "feature_lineage", None):
            return {}
        return {str(name): dict(payload) for name, payload in lineage.feature_lineage.items()}

    @property
    def lineage_warning(self) -> str | None:
        """Warning describing derived relation features, when applicable."""
        lineage = _derive_relation_lineage(
            self.relation_replay_manifest,
            n_features=self._best_n_features(),
        )
        if lineage is None:
            return None
        warning = lineage.lineage_warning
        return str(warning) if warning is not None else None

    @property
    def explanation_level(self) -> str | None:
        """Feature explanation level inferred from relation provenance."""
        lineage = _derive_relation_lineage(
            self.relation_replay_manifest,
            n_features=self._best_n_features(),
        )
        if lineage is None:
            return None
        level = lineage.explanation_level
        return str(level) if level is not None else None

    def get_feature_lineage(self, feature: str | int) -> dict[str, Any]:
        """Get relation lineage for a feature name or zero-based feature index."""
        return _lineage_by_feature(self.feature_lineage, feature)

    def _best_n_features(self) -> int | None:
        best = self.best
        if not isinstance(best, Mapping):
            return None
        n_features = best.get("n_features")
        try:
            return int(n_features) if n_features is not None else None
        except (TypeError, ValueError):
            return None

    # --- Export methods ---

    def _is_dagml_engine(self) -> bool:
        """True when this result came from the dag-ml backend (``run(engine="dag-ml")``).

        The dag-ml backend builds an in-memory :class:`Predictions` with native scores and NO workspace
        (no SQLite store, no artifacts dir), tagging ``per_dataset[name]["engine"] == "dag-ml"``. Export
        entry points use this to route through native dag-ml artifacts when present, or to raise a stable
        runtime export refusal instead of silently building a legacy workspace.
        """
        return any(isinstance(info, dict) and info.get("engine") == "dag-ml" for info in self.per_dataset.values())

    def _no_workspace_export_error(self) -> Exception:
        """The right fail-loud error when an export has no workspace path.

        A dag-ml result without workspace/native export material raises a stable :class:`RtError` refusal; a
        genuinely detached/misused legacy result keeps the original :class:`RuntimeError` (a real misuse).
        """
        if self._is_dagml_engine():
            return self._dagml_export_refusal(
                "export",
                "the result has no captured native export artifacts and no legacy workspace to read",
            )
        return RuntimeError("Cannot export: no workspace path available (result was not created from a workspace run)")

    def _dagml_export_refusal(self, operation: str, reason: str) -> Exception:
        """Build the stable V1 dag-ml export refusal envelope."""
        from nirs4all.pipeline.dagml.rt import RtError

        return RtError(
            "export",
            "unsupported_capability",
            f"engine='dag-ml' {operation} requires captured native artifacts; {reason}. "
            "The default V1 export path does not rerun the pipeline with engine='legacy'.",
            mitigation=(
                "Use nirs4all-tools to convert existing legacy workspaces/artifacts, or pass "
                f"compatibility='{_DAGML_LEGACY_REFIT_COMPATIBILITY}' to explicitly rerun the frozen "
                "pipeline with engine='legacy'."
            ),
            unsupported_capability=_DAGML_EXPORT_UNSUPPORTED_CAPABILITY,
        )

    def _legacy_refit_compatibility_requested(self, compatibility: str | None) -> bool:
        """Validate the explicit dag-ml export compatibility opt-in."""
        if compatibility is None:
            return False
        if compatibility == _DAGML_LEGACY_REFIT_COMPATIBILITY:
            return True
        raise ValueError(
            "unsupported dag-ml export compatibility opt-in "
            f"{compatibility!r}; expected {_DAGML_LEGACY_REFIT_COMPATIBILITY!r}"
        )

    def _dagml_export_delegate(self) -> RunResult | None:
        """Materialize (once) the explicit legacy-refit compatibility RunResult; ``None`` if unavailable.

        The default V1 dag-ml export path never calls this method. It is used only after the caller passes
        ``compatibility="legacy-refit"`` to request the old compatibility behavior deliberately. The method
        re-runs the SAME FROZEN pipeline (the deepcopy captured at run time) through the legacy engine with
        ``save_artifacts=True``; that legacy run owns a real workspace + chain + artifacts the existing
        export path consumes. The legacy result is cached and kept alive (its store stays open for the export
        and is closed by :meth:`close`), so repeated explicit compatibility exports do not refit.

        PARITY (compatibility): this re-fit is EXACT for a fully-seeded deterministic run (the dag-ml and
        legacy engines are at numerical parity, so the exported model's ``predict()`` may match the dag-ml
        run within tolerance), but only BEST-EFFORT otherwise — see the export()/export_model() docstrings for
        the general caveat. A per-run WARNING fires on the two ``_dagml_export_stochastic`` signals — a
        ``sample_augmentation`` step (CERTAIN), or ``run(random_state) is None`` (CONSERVATIVE — may
        over-warn a fully-deterministic pipeline, the safe direction); the uncertain middle (a seeded run
        whose individual component left ``random_state=None``) is documented, not warned. P3 captures the
        real fitted artifacts natively.
        """
        if self._dagml_export_spec is None:
            return None
        if self._dagml_legacy_result is None:
            from nirs4all.api.run import run as _run

            spec = self._dagml_export_spec
            if self._dagml_export_stochastic:
                import warnings

                warnings.warn(
                    "engine='dag-ml' export re-fits this pipeline on the legacy engine, and this run may be "
                    "nondeterministic (sample_augmentation, or run(random_state) is None), so the dag-ml-scored "
                    "model and the on-export legacy refit may differ. For an EXACT export set "
                    "run(random_state=...) AND seed every stochastic component's random_state; P3 will "
                    "capture the fitted model natively.",
                    stacklevel=3,
                )
            # Re-run the export-bound pipeline on the legacy engine: save_artifacts=True persists the
            # workspace + chain the export path needs; the same name/random_state keep the explicit
            # compatibility refit aligned with the dag-ml run. verbose=0 keeps the refit quiet.
            self._dagml_legacy_result = _run(
                spec["pipeline"],
                spec["dataset"],
                name=spec.get("name", ""),
                random_state=spec.get("random_state"),
                save_artifacts=True,
                save_charts=False,
                verbose=0,
                engine="legacy",
            )
        return self._dagml_legacy_result

    def _open_store_for_export(self):
        """Open a temporary WorkspaceStore for export operations.

        Returns a context-managed store that is closed after use.
        Works in both attached (runner alive) and detached modes.
        """
        # Attached mode: use runner's store directly
        if self._runner is not None:
            return contextlib.nullcontext(self._runner.store)

        # Detached mode: open a fresh store
        if self._workspace_path is None:
            raise self._no_workspace_export_error()

        from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

        class _TempStore:
            def __init__(self, path: Path):
                self._store = WorkspaceStore(path)

            def __enter__(self):
                return self._store

            def __exit__(self, *exc: object):
                self._store.close()

        return _TempStore(self._workspace_path)

    def export(
        self,
        output_path: str | Path,
        format: str = "n4a",
        source: dict[str, Any] | None = None,
        chain_id: str | None = None,
        *,
        compatibility: str | None = None,
    ) -> Path:
        """Export a model to bundle.

        Two export paths are supported:

        **Store-based** (preferred) -- pass ``chain_id`` to export
        directly from the workspace store:

        >>> result.export("model.n4a", chain_id="abc123")

        **Resolver-based** -- exports via ``BundleGenerator``:

        >>> result.export("model.n4a")  # uses best prediction

        Works in both attached (runner alive) and detached modes.
        In detached mode, a temporary store is opened for the export
        and closed immediately after.

        **dag-ml runs (P3 — NATIVE ``.n4a``):** when the run captured EXACTLY ONE concrete model artifact in
        its native results dir (``run(engine="dag-ml", results_path=...)`` or ``$N4A_NATIVE_RESULTS``), this
        builds the ``.n4a`` DIRECTLY from that captured REFIT model (rehydrated via the verify-then-load
        native reader). Narrow multi-artifact duplication mean-fusion, by_source mean-fusion, and branch
        stacking runs also export natively when the native manifest carries enough replay metadata. These
        native paths avoid legacy refit and stochastic warnings. A :class:`BundleLoader` reload-predict of
        the bundle reproduces the dag-ml run's scored REFIT model/composite exactly.

        **dag-ml refusal / compatibility:** without replayable native artifacts, for unsupported native
        shapes, or for the ``n4a.py`` portable-script format, the default V1 path raises a structured
        ``RtError`` that points to ``nirs4all-tools`` conversion or the explicit compatibility opt-in. It
        never re-runs the pipeline through ``engine="legacy"`` implicitly. To deliberately request the old
        refit bridge, pass ``compatibility="legacy-refit"``; this re-runs the frozen pipeline through the
        legacy engine and is best-effort for stochastic pipelines. ``source`` / ``chain_id`` are not
        supported for a dag-ml run (they reference its non-existent workspace).

        Args:
            output_path: Path for the exported bundle file.
            format: Export format ('n4a' or 'n4a.py').
            source: Prediction dict to export. If None, exports best model.
            chain_id: Chain identifier for store-based export.
                When provided, ``source`` is ignored and the chain is
                exported directly from the workspace store.
            compatibility: Optional dag-ml compatibility opt-in. The only supported value is
                ``"legacy-refit"``, which explicitly reruns the frozen dag-ml pipeline on
                ``engine="legacy"`` for export. The default is ``None``.

        Returns:
            Path to the exported bundle file.

        Raises:
            RuntimeError: If no workspace path available.
            ValueError: If no predictions available and source not provided.
            NotImplementedError: For a dag-ml run, if ``source``/``chain_id`` is given.
        """
        legacy_refit_compatibility = self._legacy_refit_compatibility_requested(compatibility)

        # dag-ml exports use captured native artifacts by default. The legacy refit bridge is available
        # only through the explicit compatibility opt-in above.
        if self._is_dagml_engine():
            if source is not None or chain_id is not None:
                raise NotImplementedError(
                    "engine='dag-ml' export does not support an explicit source=/chain_id= (they reference "
                    "the dag-ml run's non-existent workspace); export the run's best model with "
                    "result.export(path) (no source/chain_id)."
                )
            if legacy_refit_compatibility:
                delegate = self._dagml_export_delegate()
                if delegate is None:
                    raise self._dagml_export_refusal(
                        "export",
                        "legacy-refit compatibility was requested, but no frozen pipeline/dataset inputs were captured",
                    )
                return delegate.export(output_path, format=format)
            # NATIVE export (P3): when this dag-ml run captured a replayable native artifact shape in its
            # native results dir AND the request is the default ``n4a`` ZIP bundle, build the ``.n4a``
            # DIRECTLY from those captured (verify-then-load) REFIT artifacts — no legacy refit, no
            # stochastic warning. A ``None`` return means not applicable (no native dir / ``n4a.py`` format /
            # unsupported multi-artifact shape / unreadable native dir) → refuse on the default V1 path.
            native = self._dagml_native_export_bundle(output_path, format)
            if native is not None:
                return native
            raise self._dagml_export_refusal(
                "export",
                "no replayable native .n4a bundle artifacts were available for this result and requested format",
            )

        if legacy_refit_compatibility:
            raise ValueError("compatibility='legacy-refit' is only valid for engine='dag-ml' export results")

        # Store-based export path
        if chain_id is not None:
            with self._open_store_for_export() as store:
                if store is None:
                    raise RuntimeError("Cannot export from chain_id: no WorkspaceStore available")
                return Path(store.export_chain(chain_id, Path(output_path), format=format))

        # Resolver-based path
        if source is None:
            source = self.final or self.best
            if not source:
                raise ValueError("No predictions available to export")

        # Attached mode: delegate to runner
        if self._runner is not None:
            return Path(self._runner.export(source=source, output_path=output_path, format=format))

        # Detached mode: create BundleGenerator with temporary store
        workspace_path = self._workspace_path
        if workspace_path is None:
            raise self._no_workspace_export_error()

        from nirs4all.pipeline.bundle import BundleGenerator

        with self._open_store_for_export() as store:
            generator = BundleGenerator(workspace_path=workspace_path, verbose=0, store=store)
            return generator.export(source=source, output_path=output_path, format=format)

    def _dagml_native_export_model(self, output_path: str | Path, format: str | None) -> Path | None:
        """Export the CAPTURED native REFIT model directly when EXACTLY ONE concrete artifact exists (2c-ii).

        The promised replacement of the P1c legacy-refit export bridge for the SINGLE-model case: when this
        dag-ml run has a native results dir (P3 2b-i) holding EXACTLY ONE model artifact AND the requested
        export is joblib-compatible, rehydrate the artifact via
        :func:`~nirs4all.pipeline.dagml.native_results.read_native_results` (the existing verify-then-load
        reader — backend + content-fingerprint checked before any ``joblib.load``) and export a PREDICT-CAPABLE
        model (the captured ``Pipeline`` + a :class:`_DagmlExportedModel` applying the y inverse when a
        ``y_transform`` was captured) — NO legacy refit, NO stochastic warning. The exported model's
        ``predict`` reproduces the dag-ml run's scored REFIT model EXACTLY (it IS that model).

        Returns the written path on success, or ``None`` to signal that the native export is not applicable.
        The default caller raises a structured refusal; only ``compatibility="legacy-refit"`` can choose the
        legacy bridge before this helper is attempted.

        * no native dir;
        * the requested export is NOT joblib (an explicit non-joblib ``format`` such as ``cloudpickle`` /
          ``keras_h5``, or a non-joblib extension such as ``.pkl`` / ``.keras`` / ``.pt`` — the native helper
          only writes joblib bytes, so the default path refuses rather than silently writing joblib under a
          foreign extension);
        * ≠1 model artifact (multi-model / branch / stacking);
        * ANY native-read/rehydrate failure — not only the verify-then-load guards' ``ValueError`` /
          ``FileNotFoundError`` / ``KeyError`` but also a fingerprint-valid yet UNLOADABLE artifact (e.g.
          ``EOFError`` / ``UnpicklingError`` / ``ModuleNotFoundError`` / ``ImportError`` / ``AttributeError``
          from ``joblib.load``, or a parquet read error). The broad ``except Exception`` is intentional HERE:
          the default caller must turn any native-read failure into the stable V1 refusal. It is scoped to
          ONLY the read+rehydrate attempt so a genuine bug in the export write below is never swallowed.

        A plain ``y_transform``-less model exports a wrapper that is a pass-through over the estimator.
        """
        if self._dagml_results_dir is None:
            return None
        # FORMAT GATE: the native helper writes ONLY joblib bytes, so it may fire only when the request
        # resolves to joblib. Otherwise the default caller refuses unless explicit compatibility was chosen
        # before this helper was attempted.
        if not _request_is_joblib(output_path, format):
            return None
        from nirs4all.pipeline.dagml.native_results import read_native_results

        try:
            artifacts = read_native_results(self._dagml_results_dir)["artifacts"]
        except Exception as exc:  # noqa: BLE001 -- default contract: ANY native-read failure → stable refusal
            # A tampered/edited manifest (verify-then-load ValueError), a missing/malformed native dir
            # (FileNotFoundError/KeyError/parquet error), OR a fingerprint-valid but UNLOADABLE artifact
            # (EOFError/UnpicklingError/ModuleNotFoundError/ImportError/AttributeError from joblib.load —
            # bytes that hash correctly but cannot be unpickled/imported in this environment) → refuse.
            # The broad catch is SCOPED to the read+rehydrate only, so a real bug in the write below escapes.
            logger.debug("native dag-ml export_model is unavailable: %s", exc)
            return None
        # EXACTLY ONE concrete artifact only (D4): a multi-model / branch / stacking run captures several
        # REFIT artifacts and is NOT cleanly a single exportable model on this lightweight model-only path.
        if len(artifacts) != 1:
            return None

        artifact = artifacts[0]
        model = _DagmlExportedModel(artifact["estimator"], artifact["y_transform"])

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Joblib-serialize the predict-capable wrapper (it holds a sklearn Pipeline + an optional sklearn
        # y-transform — all joblib-friendly). The captured artifact's backend is "joblib" (the reader only
        # loads joblib payloads), and the format gate above guaranteed a joblib-compatible request.
        import joblib

        joblib.dump(model, output_path, compress=3)
        return output_path

    def _dagml_replayable_train_steps(self) -> list[Any] | None:
        """Serialize the FROZEN run pipeline into replayable training steps for the native ``.n4a``.

        The native bundle's ``pipeline.json`` carries a single COSMETIC model step (the predict loader
        keys on the step index, not the label), so on its own the bundle is predict-only:
        ``retrain(mode="full")`` cannot re-train from it. This helper turns the run inputs that
        :func:`~nirs4all.pipeline.dagml.run_backend._attach_export_spec` froze at run time into the
        legacy-schema serialized step list (via
        :func:`~nirs4all.pipeline.config.component_serialization.serialize_component` — the SAME form
        the step parser deserializes), written as the bundle's additive ``train_pipeline.json``.

        Fail-closed scope — returns ``None`` (bundle stays predict-only) when:

        * no export spec was captured (not a dag-ml ``run()`` result);
        * the pipeline is a GENERATOR / SWEEP of any kind (routing-aligned detection, exactly like
          :func:`~nirs4all.pipeline.dagml.run_backend._derive_config_name`): the frozen pipeline is the
          WHOLE sweep, while the exported artifact is the winner-only projection — retraining the sweep
          would silently run a different (much larger) job than "retrain this model";
        * any step fails to serialize to plain JSON (the ``json`` round-trip doubles as the
          serializability guard) — a wrong training spec must never be written.
        """
        spec = self._dagml_export_spec
        if spec is None:
            return None
        try:
            import json

            from nirs4all.pipeline.config._generator.keywords import has_nested_generator_keywords
            from nirs4all.pipeline.config.component_serialization import serialize_component
            from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
            from nirs4all.pipeline.dagml.detect import _generation_kind

            pipeline = spec.get("pipeline")
            if pipeline is None or isinstance(pipeline, PipelineConfigs):
                return None  # No frozen steps, or a pre-built (possibly multi-variant) PipelineConfigs: not derivable here.
            steps = PipelineConfigs._load_steps(pipeline)  # noqa: SLF001 - reuse the exact legacy step-loading.
            if _generation_kind(steps) != "none" or has_nested_generator_keywords(steps):
                return None
            serialized = [serialize_component(step) for step in steps]
            return cast(list[Any], json.loads(json.dumps(serialized)))
        except Exception as exc:  # noqa: BLE001 -- fail-closed: drop the training spec rather than write a wrong one
            logger.debug("native dag-ml train_pipeline.json is unavailable: %s", exc)
            return None

    def _dagml_native_export_bundle(self, output_path: str | Path, format: str) -> Path | None:
        """Export a NATIVE ``.n4a`` bundle from captured dag-ml refit artifacts when safely replayable.

        The single-artifact path is the ``.n4a`` bundle counterpart of
        :meth:`_dagml_native_export_model`: when this dag-ml run has a native results dir (P3 2b-i) holding
        EXACTLY ONE model artifact AND the request is the default ``n4a`` ZIP bundle, rehydrate the artifact via
        :func:`~nirs4all.pipeline.dagml.native_results.read_native_results` (verify-then-load — backend +
        content-fingerprint checked before any ``joblib.load``), wrap it in the same predict-capable
        :class:`_DagmlExportedModel` the native ``export_model`` uses, and package it into a single-model
        ``.n4a`` via :func:`~nirs4all.pipeline.bundle.write_single_model_bundle`. NO legacy refit, NO
        stochastic warning: a :class:`BundleLoader` reload-predict of the bundle reproduces the dag-ml run's
        scored REFIT model EXACTLY (it IS that model).

        The narrow multi-artifact paths admit duplication branch + mean-fusion and by_source mean-fusion:
        native manifest metadata must identify the final producer as ``merge:fusion`` and every captured
        artifact must be a branch model. The exported bundle embeds a small fusion wrapper that averages
        those captured branch refit models in original target space, matching dag-ml's native fusion merge.

        Branch stacking is admitted only when native results v3 persisted a ``stacking_replay`` manifest
        naming the meta artifact and the base artifacts in dag-ml's meta-feature column order. The exported
        bundle embeds a wrapper that predicts every base model from raw X, column-stacks those predictions,
        and feeds the captured meta REFIT model.

        Returns the written path on success, or ``None`` to signal that the native bundle is not applicable.
        The default caller raises a structured refusal; only ``compatibility="legacy-refit"`` can choose the
        legacy bridge before this helper is attempted.

        * no native dir;
        * a non-``n4a`` format (the ``n4a.py`` PORTABLE SCRIPT embeds artifacts through the legacy
          generator's template path and is out of this native writer's scope; the default path refuses
          rather than silently substituting a ZIP bundle);
        * a multi-artifact shape other than the supported branch / by_source mean-fusion or stacking replay;
        * ANY native-read/rehydrate failure — a tampered/edited manifest (verify-then-load ``ValueError``), a
          missing/malformed native dir (``FileNotFoundError`` / ``KeyError`` / parquet error), OR a
          fingerprint-valid but UNLOADABLE artifact (``EOFError`` / ``UnpicklingError`` / ``ModuleNotFoundError``
          / ``ImportError`` / ``AttributeError`` from ``joblib.load``). The broad ``except Exception`` is
          intentional HERE: the default caller must turn any native-read failure into the stable V1 refusal.
          It is scoped to ONLY the read+rehydrate, so a genuine bug in the bundle write below is never
          swallowed.
        """
        if self._dagml_results_dir is None:
            return None
        # FORMAT GATE: the native writer produces the ``.n4a`` ZIP bundle only. A ``n4a.py`` portable-script
        # request refuses by default unless explicit compatibility was chosen before this helper was
        # attempted. ``BundleFormat`` is a ``StrEnum`` so ``== "n4a"`` matches both the bare string and the
        # enum member.
        if format != "n4a":
            return None
        from nirs4all.pipeline.dagml.native_results import read_native_results

        try:
            native = read_native_results(self._dagml_results_dir)
            artifacts = native["artifacts"]
        except Exception as exc:  # noqa: BLE001 -- default contract: ANY native-read failure → stable refusal
            logger.debug("native dag-ml .n4a export is unavailable: %s", exc)
            return None
        native_manifest = cast(Mapping[str, Any], native["manifest"])
        model_names = _native_model_names(native_manifest)
        from nirs4all.pipeline.bundle import write_single_model_bundle

        # Replayable ORIGINAL training steps (train_pipeline.json) so retrain(mode="full") works from the
        # exported bundle; None (predict-only bundle) for generator sweeps / unserializable pipelines.
        train_steps = self._dagml_replayable_train_steps()

        if len(artifacts) == 1:
            artifact = artifacts[0]
            model = _DagmlExportedModel(artifact["estimator"], artifact["y_transform"])
            model_label = model_names[0] if model_names else type(artifact["estimator"]).__name__
            return write_single_model_bundle(
                model,
                output_path,
                model_label=model_label,
                pipeline_uid=str(native_manifest.get("run_id") or ""),
                provenance=_dagml_native_bundle_provenance(native_manifest, export_path="dagml_native", artifact_count=1),
                train_steps=train_steps,
            )

        stacking = _native_stacking_artifacts(native_manifest, artifacts)
        if stacking is not None:
            base_artifacts, meta_artifact = stacking
            base_members = [_DagmlExportedModel(artifact["estimator"], artifact["y_transform"]) for artifact in base_artifacts]
            meta_member = _DagmlExportedModel(meta_artifact["estimator"], meta_artifact["y_transform"])
            model_label = model_names[0] if model_names else "dagml_native_stacking"
            provenance = _dagml_native_bundle_provenance(
                native_manifest,
                export_path="dagml_native_stacking",
                artifact_count=len(artifacts),
                export_shape="branch_stacking_predictions",
            )
            provenance["dagml_stacking_base_count"] = len(base_artifacts)
            provenance["dagml_stacking_meta_artifact_id"] = meta_artifact.get("artifact_id")
            return write_single_model_bundle(
                _DagmlNativeStackingModel(base_members, meta_member),
                output_path,
                model_label=model_label,
                pipeline_uid=str(native_manifest.get("run_id") or ""),
                provenance=provenance,
                train_steps=train_steps,
            )

        if _is_native_branch_fusion_bundle(native, artifacts):
            members = [_DagmlExportedModel(artifact["estimator"], artifact["y_transform"]) for artifact in artifacts]
            model_label = model_names[0] if model_names else "dagml_native_branch_fusion"
            return write_single_model_bundle(
                _DagmlNativeFusionModel(members),
                output_path,
                model_label=model_label,
                pipeline_uid=str(native_manifest.get("run_id") or ""),
                provenance=_dagml_native_bundle_provenance(
                    native_manifest,
                    export_path="dagml_native_fusion",
                    artifact_count=len(artifacts),
                    export_shape="branch_fusion_mean",
                ),
                train_steps=train_steps,
            )

        if _is_native_by_source_fusion_bundle(native, artifacts):
            indexed = _indexed_branch_artifacts(artifacts)
            assert indexed is not None  # predicate above guarantees branch/source indices are recoverable
            source_members = [(source_index, _DagmlExportedModel(artifact["estimator"], artifact["y_transform"])) for source_index, artifact in indexed]
            source_model = _DagmlNativeBySourceFusionModel(source_members)
            model_label = model_names[0] if model_names else "dagml_native_by_source_fusion"
            provenance = _dagml_native_bundle_provenance(
                native_manifest,
                export_path="dagml_native_by_source_fusion",
                artifact_count=len(artifacts),
                export_shape="by_source_fusion_mean",
            )
            provenance["dagml_source_count"] = len(source_model.source_indices)
            provenance["dagml_source_widths"] = list(source_model.source_widths)
            return write_single_model_bundle(
                source_model,
                output_path,
                model_label=model_label,
                pipeline_uid=str(native_manifest.get("run_id") or ""),
                provenance=provenance,
                train_steps=train_steps,
            )

        return None

    def export_model(
        self,
        output_path: str | Path,
        source: dict[str, Any] | None = None,
        format: str | None = None,
        fold: int | None = None,
        *,
        compatibility: str | None = None,
    ) -> Path:
        """Export only the model artifact (lightweight).

        Unlike export() which creates a full bundle, this exports just the model.
        Works in both attached (runner alive) and detached modes.

        **dag-ml runs (P3 Slice 2c-ii — NATIVE single-model export):** when the run captured EXACTLY ONE
        concrete model artifact in its native results dir (``run(engine="dag-ml", results_path=...)`` or
        ``$N4A_NATIVE_RESULTS``), this exports that captured REFIT model DIRECTLY — rehydrated via the
        verify-then-load native reader, wrapped as a PREDICT-CAPABLE model (the estimator + the y inverse
        when a ``y_transform`` was captured) — with NO legacy refit and NO stochastic warning. The exported
        model's ``predict`` reproduces the dag-ml run's scored REFIT model EXACTLY (it IS that model).

        **dag-ml refusal / compatibility:** without a replayable native single-artifact model, for non-joblib
        formats, or when ``fold`` is given, the default V1 path raises a structured ``RtError`` that points
        to ``nirs4all-tools`` conversion or the explicit compatibility opt-in. It never re-runs the pipeline
        through ``engine="legacy"`` implicitly. To deliberately request the old refit bridge, pass
        ``compatibility="legacy-refit"``; this re-runs the frozen pipeline through the legacy engine and is
        best-effort for stochastic pipelines. ``source`` is not supported for a dag-ml run (it references its
        non-existent workspace).

        Args:
            output_path: Path for the output model file.
            source: Prediction dict to export. If None, exports best model.
            format: Model format (inferred from extension if None).
            fold: Fold index to export (default: fold 0).
            compatibility: Optional dag-ml compatibility opt-in. The only supported value is
                ``"legacy-refit"``, which explicitly reruns the frozen dag-ml pipeline on
                ``engine="legacy"`` for export. The default is ``None``.

        Returns:
            Path to the exported model file.

        Raises:
            RuntimeError: If no workspace path available.
        """
        legacy_refit_compatibility = self._legacy_refit_compatibility_requested(compatibility)

        # dag-ml exports use captured native artifacts by default. The legacy refit bridge is available
        # only through the explicit compatibility opt-in above.
        if self._is_dagml_engine():
            if source is not None:
                raise NotImplementedError(
                    "engine='dag-ml' export_model does not support an explicit source= (it references the "
                    "dag-ml run's non-existent workspace); export the run's best model with "
                    "result.export_model(path[, fold=...]) (no source)."
                )
            if legacy_refit_compatibility:
                delegate = self._dagml_export_delegate()
                if delegate is None:
                    raise self._dagml_export_refusal(
                        "export_model",
                        "legacy-refit compatibility was requested, but no frozen pipeline/dataset inputs were captured",
                    )
                return delegate.export_model(output_path, format=format, fold=fold)
            # NATIVE export (P3 Slice 2c-ii): when this dag-ml run captured EXACTLY ONE concrete model
            # artifact in its native results dir AND the request is joblib-compatible, export that captured
            # (verify-then-load) REFIT model DIRECTLY — no legacy refit, no stochastic warning. ``fold`` is
            # incompatible with the native single-artifact export (it would select a fold's model, which
            # lives only in the legacy workspace), so the native path is attempted only for the default
            # whole-model export. A ``None`` return means not applicable (no native dir / non-joblib format /
            # ≠1 artifact / unloadable artifact) → refuse on the default V1 path.
            if fold is None:
                native = self._dagml_native_export_model(output_path, format)
                if native is not None:
                    return native
            raise self._dagml_export_refusal(
                "export_model",
                "no single replayable native joblib model artifact was available for this result, requested format, and fold selector",
            )

        if legacy_refit_compatibility:
            raise ValueError("compatibility='legacy-refit' is only valid for engine='dag-ml' export results")

        if source is None:
            source = self.best
            if not source:
                raise ValueError("No predictions available to export")

        # Attached mode: delegate to runner
        if self._runner is not None:
            return self._runner.export_model(source=source, output_path=output_path, format=format, fold=fold)

        # Detached mode: replicate runner.export_model logic with temporary store
        workspace_path = self._workspace_path
        if workspace_path is None:
            raise self._no_workspace_export_error()

        from nirs4all.pipeline.resolver import PredictionResolver
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import to_bytes

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with self._open_store_for_export() as store:
            resolver = PredictionResolver(workspace_path=workspace_path, runs_dir=workspace_path, store=store)
            resolved = resolver.resolve(source, verbose=0)

            if resolved.model_step_index is None:
                raise ValueError("No model step found in the resolved prediction")
            if resolved.artifact_provider is None:
                raise ValueError("No artifact provider available for this source")

            artifacts = resolved.artifact_provider.get_artifacts_for_step(resolved.model_step_index)
            if not artifacts:
                raise ValueError(f"No model artifacts found at step {resolved.model_step_index}")

            if fold is not None:
                model = None
                for artifact_id, artifact in artifacts:
                    if f":{fold}" in str(artifact_id) or artifact_id.endswith(f"_{fold}"):
                        model = artifact
                        break
                if model is None:
                    raise ValueError(f"No artifact found for fold {fold}")
            else:
                _, model = artifacts[0]

        if format is None:
            ext = output_path.suffix.lower()
            format_map = {'.joblib': 'joblib', '.pkl': 'cloudpickle', '.pickle': 'cloudpickle', '.h5': 'keras_h5', '.hdf5': 'keras_h5', '.keras': 'tensorflow_keras', '.pt': 'pytorch_state_dict', '.pth': 'pytorch_state_dict'}
            format = format_map.get(ext, 'joblib')

        data, _actual_format = to_bytes(model, format)
        with open(output_path, 'wb') as f:
            f.write(data)

        return output_path

    # --- Utility methods ---

    def summary(self) -> str:
        """Get a summary string of the run result.

        Returns:
            Multi-line summary string with key metrics.
        """
        lines = []
        lines.append(f"RunResult: {self.num_predictions} predictions")

        if self.artifacts_path:
            lines.append(f"  Artifacts: {self.artifacts_path}")

        datasets = self.get_datasets()
        if datasets:
            lines.append(f"  Datasets: {', '.join(datasets)}")

        models = self.get_models()
        if models:
            lines.append(f"  Models: {', '.join(models[:5])}" +
                        (f" (+{len(models)-5} more)" if len(models) > 5 else ""))

        best = self.best
        if best:
            lines.append(f"  Best: {best.get('model_name', 'unknown')}")
            lines.append(f"    test_score: {self.best_score:.4f}")
            if not np.isnan(self.best_rmse):
                lines.append(f"    rmse: {self.best_rmse:.4f}")
            if not np.isnan(self.best_r2):
                lines.append(f"    r2: {self.best_r2:.4f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"RunResult(predictions={self.num_predictions}, best_score={self.best_score:.4f})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.summary()

    def validate(
        self,
        check_nan_metrics: bool = True,
        check_empty: bool = True,
        raise_on_failure: bool = True,
        nan_threshold: float = 0.0
    ) -> dict[str, Any]:
        """Validate the run result for common issues.

        Checks for NaN values in metrics, empty predictions, and other issues
        that might indicate problems with the pipeline execution.

        Args:
            check_nan_metrics: If True, check for NaN values in metrics.
            check_empty: If True, check for empty predictions.
            raise_on_failure: If True, raise ValueError on validation failure.
            nan_threshold: Maximum allowed ratio of predictions with NaN metrics (0.0 = none allowed).

        Returns:
            Dictionary with validation results:
                - valid: True if all checks passed.
                - issues: List of issue descriptions.
                - nan_count: Number of predictions with NaN metrics.
                - total_count: Total number of predictions.

        Raises:
            ValueError: If raise_on_failure=True and validation fails.

        Example:
            >>> result = nirs4all.run(pipeline, dataset)
            >>> result.validate()  # Raises if issues found
            >>> # Or check without raising
            >>> report = result.validate(raise_on_failure=False)
            >>> if not report['valid']:
            ...     print(f"Issues: {report['issues']}")
        """
        issues = []
        nan_count = 0
        total_count = self.num_predictions

        # Check for empty predictions
        if check_empty and total_count == 0:
            issues.append("No predictions found")

        # Check for NaN metrics
        if check_nan_metrics and total_count > 0:
            all_preds = self.predictions.filter_predictions(load_arrays=False)
            for pred in all_preds:
                has_nan = False
                # Check common metrics
                for metric in ['rmse', 'r2', 'accuracy', 'mse', 'mae']:
                    value = pred.get(metric)
                    if value is not None and np.isnan(value):
                        has_nan = True
                        break

                # Check scores dict
                if not has_nan:
                    scores = pred.get('scores', {})
                    if isinstance(scores, dict):
                        for partition_scores in scores.values():
                            if isinstance(partition_scores, dict):
                                for val in partition_scores.values():
                                    if isinstance(val, (int, float)) and np.isnan(val):
                                        has_nan = True
                                        break

                # Check score fields
                if not has_nan:
                    for score_key in ('test_score', 'val_score', 'train_score'):
                        score_val = pred.get(score_key)
                        if score_val is not None and isinstance(score_val, float) and np.isnan(score_val):
                            has_nan = True
                            break

                if has_nan:
                    nan_count += 1

            # Check threshold
            if nan_count > 0:
                nan_ratio = nan_count / total_count if total_count > 0 else 0
                if nan_ratio > nan_threshold:
                    issues.append(
                        f"NaN ratio ({nan_ratio:.1%}) exceeds threshold ({nan_threshold:.1%}): "
                        f"{nan_count} of {total_count} predictions have NaN metrics"
                    )

        valid = len(issues) == 0

        report = {
            'valid': valid,
            'issues': issues,
            'nan_count': nan_count,
            'total_count': total_count,
        }

        if raise_on_failure and not valid:
            raise ValueError(
                "RunResult validation failed:\n" +
                "\n".join(f"  - {issue}" for issue in issues)
            )

        return report

@dataclass
class PredictResult:
    """Result from nirs4all.predict().

    Wraps prediction outputs with convenient accessors and conversion methods.

    Attributes:
        y_pred: Predicted values array (n_samples,) or (n_samples, n_outputs).
        metadata: Additional prediction metadata (uncertainty, timing, etc.).
        sample_indices: Optional indices of predicted samples.
        model_name: Name of the model used for prediction.
        preprocessing_steps: List of preprocessing steps applied.

    Properties:
        values: Alias for y_pred (for consistency).
        shape: Shape of prediction array.
        is_multioutput: True if predictions have multiple outputs.

    Key Operations:
        to_numpy(): Get predictions as numpy array.
        to_list(): Get predictions as Python list.
        to_dataframe(): Get predictions as pandas DataFrame.
        flatten(): Get flattened 1D predictions.

    Example:
        >>> result = nirs4all.predict(model, X_new)
        >>> print(f"Predictions shape: {result.shape}")
        >>> df = result.to_dataframe()
    """

    y_pred: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_indices: np.ndarray | None = None
    model_name: str = ""
    preprocessing_steps: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure y_pred is a numpy array."""
        if self.y_pred is not None and not isinstance(self.y_pred, np.ndarray):
            self.y_pred = np.asarray(self.y_pred)
        if self.metadata is None:
            self.metadata = {}

    @property
    def values(self) -> np.ndarray:
        """Get prediction values (alias for y_pred)."""
        return self.y_pred

    @property
    def relation_replay_manifest(self) -> dict[str, Any] | None:
        """Relation replay manifest used to materialize heterogeneous inputs."""
        return _plain_mapping(_metadata_mapping(self.metadata).get("relation_replay_manifest"))

    @property
    def relation_materialization_manifest(self) -> dict[str, Any] | None:
        """Materialization provenance embedded in the relation replay manifest."""
        metadata = _metadata_mapping(self.metadata)
        explicit = _plain_mapping(metadata.get("relation_materialization_manifest"))
        if explicit is not None:
            return explicit
        return _materialization_manifest(self.relation_replay_manifest)

    @property
    def feature_lineage(self) -> dict[str, Any]:
        """Feature provenance derived from prediction relation metadata."""
        metadata = _metadata_mapping(self.metadata)
        explicit = _plain_mapping(metadata.get("feature_lineage"))
        if explicit is not None:
            return explicit
        lineage = _derive_relation_lineage(self.relation_replay_manifest or self.relation_materialization_manifest)
        if lineage is None or not getattr(lineage, "feature_lineage", None):
            return {}
        return {str(name): dict(payload) for name, payload in lineage.feature_lineage.items()}

    @property
    def lineage_warning(self) -> str | None:
        """Warning describing derived relation features, when applicable."""
        metadata = _metadata_mapping(self.metadata)
        warning = metadata.get("lineage_warning")
        if warning is not None:
            return str(warning)
        lineage = _derive_relation_lineage(self.relation_replay_manifest or self.relation_materialization_manifest)
        if lineage is None:
            return None
        warning = lineage.lineage_warning
        return str(warning) if warning is not None else None

    @property
    def explanation_level(self) -> str | None:
        """Feature explanation level inferred from relation provenance."""
        metadata = _metadata_mapping(self.metadata)
        level = metadata.get("explanation_level")
        if level is not None:
            return str(level)
        lineage = _derive_relation_lineage(self.relation_replay_manifest or self.relation_materialization_manifest)
        if lineage is None:
            return None
        level = lineage.explanation_level
        return str(level) if level is not None else None

    def get_feature_lineage(self, feature: str | int) -> dict[str, Any]:
        """Get relation lineage for a feature name or zero-based feature index."""
        return _lineage_by_feature(self.feature_lineage, feature)

    @property
    def shape(self) -> tuple:
        """Get shape of prediction array."""
        if self.y_pred is None:
            return (0,)
        return self.y_pred.shape

    @property
    def is_multioutput(self) -> bool:
        """Check if predictions have multiple outputs."""
        return len(self.shape) > 1 and self.shape[1] > 1

    def __len__(self) -> int:
        """Return number of predictions."""
        if self.y_pred is None:
            return 0
        return len(self.y_pred)

    def to_numpy(self) -> np.ndarray:
        """Get predictions as numpy array.

        Returns:
            Numpy array of predictions.
        """
        return self.y_pred

    def to_list(self) -> list[float]:
        """Get predictions as Python list.

        Returns:
            List of prediction values (flattened if 2D).
        """
        if self.y_pred is None:
            return []
        return self.y_pred.flatten().tolist()

    def to_dataframe(self, include_indices: bool = True):
        """Get predictions as pandas DataFrame.

        Args:
            include_indices: If True and sample_indices available, include as column.

        Returns:
            pandas DataFrame with predictions.

        Raises:
            ImportError: If pandas is not available.
        """
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError("pandas is required for to_dataframe()") from err

        data = {}

        if include_indices and self.sample_indices is not None:
            data['sample_index'] = self.sample_indices

        if self.is_multioutput:
            for i in range(self.shape[1]):
                data[f'y_pred_{i}'] = self.y_pred[:, i]
        else:
            data['y_pred'] = self.y_pred.flatten()

        return pd.DataFrame(data)

    def flatten(self) -> np.ndarray:
        """Get flattened 1D predictions.

        Returns:
            1D numpy array of predictions.
        """
        if self.y_pred is None:
            return np.array([])
        return self.y_pred.flatten()

    def __repr__(self) -> str:
        """String representation."""
        return f"PredictResult(shape={self.shape}, model='{self.model_name}')"

    def __str__(self) -> str:
        """User-friendly string representation."""
        lines = [f"PredictResult: {len(self)} predictions"]
        if self.model_name:
            lines.append(f"  Model: {self.model_name}")
        if self.preprocessing_steps:
            lines.append(f"  Preprocessing: {' -> '.join(self.preprocessing_steps)}")
        if self.relation_replay_manifest is not None or self.relation_materialization_manifest is not None:
            lines.append("  Relation provenance: available")
        lines.append(f"  Shape: {self.shape}")
        return "\n".join(lines)

@dataclass
class ExplainResult:
    """Result from nirs4all.explain().

    Wraps SHAP explanation outputs with visualization helpers and accessors.

    Attributes:
        shap_values: SHAP values array or Explanation object.
        feature_names: Names/labels of features explained.
        base_value: Expected value (baseline prediction).
        visualizations: Paths to generated visualization files.
        explainer_type: Type of SHAP explainer used.
        model_name: Name of the explained model.
        n_samples: Number of samples explained.
        explanation_level: Unit level explained, such as raw_observation,
            source_aggregate, sample_aggregate, combo, or stack.
        feature_lineage: Mapping from explained feature names to relation
            lineage/provenance payloads.
        lineage_warning: Optional warning when explanations are for derived or
            aggregated features rather than raw observations.

    Properties:
        values: Raw SHAP values array.
        shape: Shape of SHAP values array.
        mean_abs_shap: Mean absolute SHAP values per feature.
        top_features: Feature names sorted by importance.

    Key Operations:
        get_feature_importance(): Get feature importance ranking.
        get_sample_explanation(idx): Get explanation for a single sample.
        to_dataframe(): Get SHAP values as DataFrame.

    Example:
        >>> result = nirs4all.explain(model, X_test)
        >>> print(f"Top features: {result.top_features[:5]}")
        >>> importance = result.get_feature_importance()
    """

    shap_values: Any  # shap.Explanation or np.ndarray
    feature_names: list[str] | None = None
    base_value: float | np.ndarray | None = None
    visualizations: dict[str, Path] = field(default_factory=dict)
    explainer_type: str = "auto"
    model_name: str = ""
    n_samples: int = 0
    explanation_level: str | None = None
    feature_lineage: dict[str, Any] = field(default_factory=dict)
    lineage_warning: str | None = None

    def __post_init__(self):
        """Extract metadata from shap_values if available."""
        if hasattr(self.shap_values, 'values'):
            # It's a shap.Explanation object
            if self.feature_names is None and hasattr(self.shap_values, 'feature_names'):
                self.feature_names = list(self.shap_values.feature_names)
            if self.base_value is None and hasattr(self.shap_values, 'base_values'):
                self.base_value = self.shap_values.base_values
            if self.n_samples == 0:
                self.n_samples = len(self.shap_values.values)

    @property
    def values(self) -> np.ndarray:
        """Get raw SHAP values array.

        Returns:
            Numpy array of SHAP values (n_samples, n_features).
        """
        if hasattr(self.shap_values, 'values'):
            return np.asarray(self.shap_values.values)
        return np.asarray(self.shap_values)

    @property
    def shape(self) -> tuple:
        """Get shape of SHAP values array."""
        return self.values.shape

    @property
    def mean_abs_shap(self) -> np.ndarray:
        """Get mean absolute SHAP values per feature.

        Returns:
            1D array of mean |SHAP| values, one per feature.
        """
        vals = self.values
        if vals.ndim == 1:
            return np.asarray(np.abs(vals))
        return np.asarray(np.mean(np.abs(vals), axis=0))

    @property
    def top_features(self) -> list[str]:
        """Get feature names sorted by importance (descending).

        Returns:
            List of feature names, most important first.
            Returns indices as strings if feature_names not available.
        """
        importance = self.mean_abs_shap
        sorted_indices = np.argsort(importance)[::-1]

        if self.feature_names:
            return [self.feature_names[i] for i in sorted_indices]
        return [str(i) for i in sorted_indices]

    def get_feature_importance(
        self,
        top_n: int | None = None,
        normalize: bool = False
    ) -> dict[str, float]:
        """Get feature importance ranking.

        Args:
            top_n: If provided, return only top N features.
            normalize: If True, normalize values to sum to 1.

        Returns:
            Dictionary mapping feature names to importance values.
        """
        importance = self.mean_abs_shap

        if normalize and importance.sum() > 0:
            importance = importance / importance.sum()

        sorted_indices = np.argsort(importance)[::-1]

        if top_n:
            sorted_indices = sorted_indices[:top_n]

        result = {}
        for idx in sorted_indices:
            name = self.feature_names[idx] if self.feature_names else str(idx)
            result[name] = float(importance[idx])

        return result

    def get_sample_explanation(
        self,
        idx: int
    ) -> dict[str, float]:
        """Get SHAP explanation for a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary mapping feature names to SHAP values for that sample.
        """
        vals = self.values
        if idx >= len(vals):
            raise IndexError(f"Sample index {idx} out of range (n_samples={len(vals)})")

        sample_shap = vals[idx] if vals.ndim > 1 else vals

        result = {}
        for i, val in enumerate(sample_shap):
            name = self.feature_names[i] if self.feature_names else str(i)
            result[name] = float(val)

        return result

    def get_feature_lineage(self, feature: str | int) -> dict[str, Any]:
        """Get relation lineage for an explained feature.

        Args:
            feature: Feature name or positional feature index.

        Returns:
            Lineage payload for the feature, or an empty dictionary when absent.
        """
        if isinstance(feature, int):
            if self.feature_names and 0 <= feature < len(self.feature_names):
                feature_name = self.feature_names[feature]
            else:
                feature_name = str(feature)
        else:
            feature_name = feature
        lineage = self.feature_lineage.get(feature_name, {})
        return dict(lineage) if isinstance(lineage, dict) else {}

    def to_dataframe(self, include_feature_names: bool = True):
        """Get SHAP values as pandas DataFrame.

        Args:
            include_feature_names: If True, use feature names as columns.

        Returns:
            pandas DataFrame with SHAP values.

        Raises:
            ImportError: If pandas is not available.
        """
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError("pandas is required for to_dataframe()") from err

        vals = self.values

        columns = self.feature_names if include_feature_names and self.feature_names else [f"feature_{i}" for i in range(vals.shape[-1])]

        if vals.ndim == 1:
            vals = vals.reshape(1, -1)

        return pd.DataFrame(vals, columns=columns)

    def __repr__(self) -> str:
        """String representation."""
        return f"ExplainResult(shape={self.shape}, explainer='{self.explainer_type}')"

    def __str__(self) -> str:
        """User-friendly string representation."""
        lines = [f"ExplainResult: {self.n_samples} samples explained"]
        if self.model_name:
            lines.append(f"  Model: {self.model_name}")
        lines.append(f"  Explainer: {self.explainer_type}")
        lines.append(f"  Shape: {self.shape}")
        if self.feature_names:
            lines.append(f"  Features: {len(self.feature_names)}")
        if self.explanation_level:
            lines.append(f"  Explanation level: {self.explanation_level}")
        if self.feature_lineage:
            lines.append("  Feature lineage: available")
        if self.lineage_warning:
            lines.append(f"  Lineage warning: {self.lineage_warning}")

        # Show top 5 features
        top = self.top_features[:5]
        if top:
            lines.append(f"  Top features: {', '.join(top)}")

        if self.visualizations:
            lines.append(f"  Visualizations: {list(self.visualizations.keys())}")

        return "\n".join(lines)
