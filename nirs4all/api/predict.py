"""
Module-level predict() function for nirs4all.

This module provides a simple interface for making predictions with trained
nirs4all models. It wraps PipelineRunner.predict() with ergonomic defaults.

Prediction paths supported by the public API:

1. **Store-based** (preferred): ``nirs4all.predict(chain_id="abc", data=X)``
   replays a stored chain directly from the workspace store.

2. **Model-based**: ``nirs4all.predict(model="model.n4a", data=X)``
   resolves via PredictionResolver / BundleLoader.

3. **Calibrated replayed-array**: ``nirs4all.predict(model=calibrated, data={"y_pred": y, "sample_ids": ids})``
   applies an existing conformal calibrator to already computed point predictions.

4. **Attached conformal model bundle**: ``nirs4all.predict(model="model.calibrated.n4a", data={"X": X, "sample_ids": ids}, coverage=0.9)``
   replays the model bundle and selects already materialized conformal intervals.
   ``all_predictions=True`` remains fail-closed for conformal sidecars until
   every returned prediction entry carries calibrated identity mapping.

Example:
    >>> import nirs4all
    >>> result = nirs4all.predict(
    ...     model="exports/best_model.n4a",
    ...     data=X_new,
    ...     verbose=1
    ... )
    >>> print(f"Predictions shape: {result.shape}")
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline import PipelineRunner
from nirs4all.pipeline.engine import require_legacy_engine

from .result import PredictResult
from .session import Session

# Type aliases for clarity
ModelSpec: TypeAlias = (
    dict[str, Any]  # Prediction dict from previous run
    | str  # Path to bundle (.n4a) or config
    | Path  # Path to bundle or config
)

DataSpec: TypeAlias = (
    str  # Path to data folder
    | Path  # Path to data folder
    | np.ndarray  # X array
    | tuple[np.ndarray, ...]  # (X,) or (X, y)
    | dict[str, Any]  # Dict with X key
    | SpectroDataset  # Direct SpectroDataset instance
    | DatasetConfigs  # Backward compat
)


def predict(
    model: ModelSpec | None = None,
    data: DataSpec | None = None,
    *,
    chain_id: str | None = None,
    workspace_path: str | Path | None = None,
    name: str = "prediction_dataset",
    all_predictions: bool = False,
    session: Session | None = None,
    verbose: int = 0,
    coverage: float | list[float] | tuple[float, ...] | None = None,
    save_to_workspace: bool = False,
    workspace_metadata: Mapping[str, Any] | None = None,
    workspace_result_metadata: Mapping[str, Any] | None = None,
    **runner_kwargs: Any,
) -> PredictResult:
    """Make predictions with a trained model on new data.

    This function provides a simple interface for running inference with
    trained nirs4all pipelines.

    Main prediction paths are supported:

    **Store-based** (preferred) -- pass ``chain_id`` together with a
    raw numpy array for ``data``:

    >>> result = nirs4all.predict(chain_id="abc123", data=X_new)

    **Model-based** -- pass ``model`` together with ``data``:

    >>> result = nirs4all.predict(model="exports/model.n4a", data=X_new)

    **Calibrated replayed-array** -- pass a calibrated result or conformal
    result archive together with already computed point predictions:

    >>> result = nirs4all.predict(model=calibrated, data={"y_pred": y_pred, "sample_ids": ids}, coverage=0.9)

    **Attached conformal model bundle** -- pass a model ``.n4a`` carrying a
    conformal sidecar and explicit prediction sample ids:

    >>> result = nirs4all.predict(model="exports/model.calibrated.n4a", data={"X": X_new, "sample_ids": ids}, coverage=0.9)

    Args:
        model: Trained model specification. Can be:
            - Prediction dict from ``result.best`` or ``result.top()``
            - Path to exported bundle: ``"exports/model.n4a"``
            - Path to pipeline config directory
            Mutually exclusive with ``chain_id``.

        data: Data to predict on. Can be:
            - Path to data folder: ``"new_data/"``
            - Numpy array: ``X_new`` (n_samples, n_features)
            - Tuple: ``(X,)`` or ``(X, y)`` for evaluation
            - Dict: ``{"X": X, "metadata": meta}``
            - SpectroDataset instance

        chain_id: Chain identifier in the workspace store.
            When provided, uses the fast store-based replay path.
            Mutually exclusive with ``model``.

        workspace_path: Workspace root directory.  Required when using
            ``chain_id`` outside a session.  Ignored when a ``session``
            is provided (the session's workspace is used instead).

        name: Name for the prediction dataset (for logging).
            Default: "prediction_dataset"

        all_predictions: If True, return predictions from all folds/entries for
            non-conformal predictions. With a conformal sidecar and
            ``coverage=...``, only ``all_predictions=False`` is currently
            supported.

        coverage: Optional conformal interval coverage selector. This is
            supported when ``model`` is either a calibrated replayed-array
            result/store/bundle with ``data={"y_pred": ..., "sample_ids": ...}``
            or a model ``.n4a`` bundle carrying an attached conformal sidecar
            with ``data={"X": ..., "sample_ids": ...}``. The selector must
            reference already materialized coverages; it never recalibrates or
            creates a new guarantee. If a model bundle contains a ``conformal/``
            sidecar, the sidecar is validated strictly; invalid sidecars fail
            validation instead of falling back to uncalibrated prediction.

        save_to_workspace: If ``True``, publish the returned ``PredictResult``
            through ``save_workspace_predict_result(...)`` and add
            ``workspace_prediction_id`` to ``result.metadata``. This publishes
            prediction/evidence rows only; conformal artifacts remain owned by
            ``save_workspace_calibrated_result(...)``.

        workspace_metadata: Optional sample-level metadata forwarded to the
            workspace prediction sidecar when ``save_to_workspace=True``.

        workspace_result_metadata: Optional result-level metadata merged over
            ``result.metadata`` when publishing the workspace prediction row.

        session: Optional Session for resource reuse.
            If provided, uses the session's runner.

        verbose: Verbosity level (0=quiet, 1=info, 2=debug).
            Default: 0

        **runner_kwargs: Additional PipelineRunner parameters.
            Common options: plots_visible. ``engine`` is accepted here for the
            transition release; only ``"legacy"`` is supported by this helper
            until native prediction replay is implemented.

    Returns:
        PredictResult containing:
            - y_pred: Predicted values array (n_samples,)
            - metadata: Additional prediction metadata
            - model_name: Name of the model used
            - preprocessing_steps: List of preprocessing steps applied

        Use ``result.to_dataframe()`` for pandas DataFrame output.

    Raises:
        ValueError: If neither ``model`` nor ``chain_id`` is provided,
            or if both are provided.
        FileNotFoundError: If model bundle or data path doesn't exist.

    Examples:
        Predict from a stored chain (preferred):

        >>> import nirs4all
        >>> result = nirs4all.predict(chain_id="abc123", data=X_new)

        Predict from an exported bundle:

        >>> result = nirs4all.predict(
        ...     model="exports/wheat_model.n4a",
        ...     data=X_new
        ... )

        Predict using a result from a previous run:

        >>> train_result = nirs4all.run(pipeline, train_data)
        >>> pred_result = nirs4all.predict(
        ...     model=train_result.best,
        ...     data=X_test
        ... )

    See Also:
        - :func:`nirs4all.run`: Train a pipeline
        - :func:`nirs4all.explain`: Generate SHAP explanations
        - :class:`nirs4all.api.result.PredictResult`: Result class
    """
    # ---- Validate mutually exclusive arguments ----
    if model is not None and chain_id is not None:
        raise ValueError("Provide either 'model' or 'chain_id', not both.")
    if model is None and chain_id is None:
        raise ValueError("Provide either 'model' or 'chain_id'.")
    if data is None:
        raise ValueError("'data' is required.")

    engine = runner_kwargs.pop("engine", None)

    if _is_calibrated_replayed_prediction_request(model, data):
        result = _predict_from_calibrated_replayed_arrays(
            model=model,
            data=data,
            coverage=coverage,
        )
        return _maybe_publish_predict_result(
            result,
            data=data,
            name=name,
            save_to_workspace=save_to_workspace,
            workspace_path=workspace_path,
            session=session,
            workspace_metadata=workspace_metadata,
            workspace_result_metadata=workspace_result_metadata,
        )

    if coverage is not None and _is_conformal_attached_bundle_request(model):
        require_legacy_engine("predict", engine)
        result = _predict_from_conformal_attached_model_bundle(
            model=model,
            data=data,
            name=name,
            all_predictions=all_predictions,
            session=session,
            verbose=verbose,
            workspace_path=workspace_path,
            coverage=coverage,
            **runner_kwargs,
        )
        return _maybe_publish_predict_result(
            result,
            data=data,
            name=name,
            save_to_workspace=save_to_workspace,
            workspace_path=workspace_path,
            session=session,
            workspace_metadata=workspace_metadata,
            workspace_result_metadata=workspace_result_metadata,
        )

    if coverage is not None:
        raise NotImplementedError(
            "predict(..., coverage=...) currently requires a calibrated replayed-array result/store/bundle "
            "as 'model' with data={'y_pred': ..., 'sample_ids': ...}, or a model .n4a bundle with an attached "
            "conformal sidecar and data={'X': ..., 'sample_ids': ...}."
        )

    require_legacy_engine("predict", engine)

    # ---- Store-based path (chain_id) ----
    if chain_id is not None:
        result = _predict_from_chain(
            chain_id=chain_id,
            data=data,
            workspace_path=workspace_path,
            session=session,
            verbose=verbose,
            **runner_kwargs,
        )
        return _maybe_publish_predict_result(
            result,
            data=data,
            name=name,
            save_to_workspace=save_to_workspace,
            workspace_path=workspace_path,
            session=session,
            workspace_metadata=workspace_metadata,
            workspace_result_metadata=workspace_result_metadata,
            chain_id=chain_id,
        )

    # ---- Model-based path ----
    assert model is not None
    result = _predict_from_model(
        model=model,
        data=data,
        name=name,
        all_predictions=all_predictions,
        session=session,
        verbose=verbose,
        workspace_path=workspace_path,
        **runner_kwargs,
    )
    return _maybe_publish_predict_result(
        result,
        data=data,
        name=name,
        save_to_workspace=save_to_workspace,
        workspace_path=workspace_path,
        session=session,
        workspace_metadata=workspace_metadata,
        workspace_result_metadata=workspace_result_metadata,
    )


# -----------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------


def _maybe_publish_predict_result(
    result: PredictResult,
    *,
    data: DataSpec,
    name: str,
    save_to_workspace: bool,
    workspace_path: str | Path | None,
    session: Session | None,
    workspace_metadata: Mapping[str, Any] | None,
    workspace_result_metadata: Mapping[str, Any] | None,
    chain_id: str | None = None,
) -> PredictResult:
    """Persist a prediction result when the public ``predict`` call asks for it."""

    if not save_to_workspace:
        return result

    from .result import save_workspace_predict_result

    publish_path = _resolve_publish_workspace_path(workspace_path, session)
    result_metadata = {"publisher": "nirs4all.predict"}
    if workspace_result_metadata is not None:
        result_metadata.update(dict(workspace_result_metadata))
    prediction_id = save_workspace_predict_result(
        publish_path,
        result,
        dataset_name=name,
        metadata=workspace_metadata,
        result_metadata=result_metadata,
        X=_extract_optional_replay_X(data),
        model_name=result.model_name,
        pipeline_id=_pipeline_id_for_chain(publish_path, chain_id) if chain_id is not None else None,
        chain_id=chain_id,
        preprocessings=result.preprocessing_steps,
    )
    result.metadata["workspace_prediction_id"] = prediction_id
    result.metadata["workspace_path"] = str(publish_path)
    result.metadata["workspace_prediction_published"] = True
    return result


def _resolve_publish_workspace_path(workspace_path: str | Path | None, session: Session | None) -> Path:
    """Return the workspace path used for explicit prediction publication."""

    if session is not None:
        runner = getattr(session, "runner", None)
        candidate = getattr(runner, "workspace_path", None)
        if candidate is not None:
            return Path(candidate)
    if workspace_path is not None:
        return Path(workspace_path)
    return Path("workspace")


def _pipeline_id_for_chain(workspace_path: Path, chain_id: str | None) -> str | None:
    """Resolve the owning pipeline for an existing chain when publishing."""

    if chain_id is None:
        return None
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    store = WorkspaceStore(workspace_path)
    try:
        chain = store.get_chain(chain_id)
    finally:
        store.close()
    if isinstance(chain, Mapping):
        pipeline_id = chain.get("pipeline_id")
        return str(pipeline_id) if pipeline_id else None
    return None


def _extract_optional_replay_X(data: DataSpec) -> np.ndarray | None:
    """Best-effort extraction of row-aligned X for workspace evidence publishing."""

    try:
        return _extract_X(data)
    except (TypeError, ValueError, AttributeError):
        return None


def _is_calibrated_replayed_prediction_request(model: ModelSpec | None, data: DataSpec) -> bool:
    """Whether predict() should route to the narrow conformal replayed-array lane."""

    from nirs4all.pipeline.dagml.conformal_contracts import CalibratedRunResult

    if isinstance(model, CalibratedRunResult):
        return True
    return isinstance(data, Mapping) and "y_pred" in data


def _is_conformal_attached_bundle_request(model: ModelSpec | None) -> bool:
    """Whether *model* is a path to a ``.n4a`` bundle with a conformal sidecar."""

    if not isinstance(model, (str, Path)):
        return False
    path = Path(model)
    if path.suffix.lower() != ".n4a" or not path.is_file():
        return False
    import zipfile

    from nirs4all.pipeline.dagml.conformal_store import BUNDLE_ROOT, load_conformal_result_archive

    try:
        with zipfile.ZipFile(path, "r") as archive:
            has_conformal_sidecar = any(name.startswith(BUNDLE_ROOT) for name in archive.namelist())
    except zipfile.BadZipFile:
        return False
    if not has_conformal_sidecar:
        return False

    # If a conformal sidecar is present, fail closed on corrupt, incomplete,
    # duplicated, or unexpected members instead of silently falling back to the
    # generic model path.
    load_conformal_result_archive(path)
    return True


def _predict_from_calibrated_replayed_arrays(
    *,
    model: ModelSpec | None,
    data: DataSpec,
    coverage: float | list[float] | tuple[float, ...] | None,
) -> PredictResult:
    """Apply a stored conformal calibrator to already computed point predictions."""

    if model is None:
        raise ValueError("'model' must be a calibrated result, store path, or conformal .n4a bundle")
    if not isinstance(data, Mapping):
        raise TypeError("calibrated replayed-array prediction requires data={'y_pred': ..., 'sample_ids': ...}; full predictor replay with coverage is still planned.")

    from nirs4all.api.calibrate import predict_calibrated
    from nirs4all.pipeline.dagml.conformal_contracts import CalibratedRunResult

    if not isinstance(model, (CalibratedRunResult, str, Path)):
        raise TypeError("calibrated replayed-array prediction requires 'model' to be a CalibratedRunResult, a conformal result store path, or a conformal .n4a bundle path")

    sample_ids = data.get("prediction_sample_ids", data.get("sample_ids"))
    result_metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else None
    result = predict_calibrated(
        model,
        y_pred=data["y_pred"],
        prediction_sample_ids=sample_ids,
        result_metadata=result_metadata,
        as_predict_result=True,
    )
    assert isinstance(result, PredictResult)
    selected_coverages = _normalize_requested_coverages(coverage)
    if selected_coverages is None:
        return result

    result.intervals = _select_materialized_intervals(result, selected_coverages)
    result.metadata["selected_interval_coverages"] = list(selected_coverages)
    _record_selected_conformal_coverages(result.metadata, selected_coverages)
    return result


def _predict_from_conformal_attached_model_bundle(
    *,
    model: ModelSpec | None,
    data: DataSpec,
    name: str,
    all_predictions: bool,
    session: Session | None,
    verbose: int,
    workspace_path: str | Path | None,
    coverage: float | list[float] | tuple[float, ...],
    **runner_kwargs: Any,
) -> PredictResult:
    """Run a model bundle prediction and apply its attached conformal sidecar."""

    if not isinstance(model, (str, Path)):
        raise TypeError("conformal attached bundle prediction requires a model .n4a path")
    if all_predictions:
        raise NotImplementedError("conformal interval application currently requires all_predictions=False")

    from nirs4all.api.calibrate import predict_calibrated
    from nirs4all.pipeline.dagml.conformal_store import load_conformal_result_archive

    calibrated = load_conformal_result_archive(model)
    raw_result = _predict_from_model(
        model=model,
        data=data,
        name=name,
        all_predictions=all_predictions,
        session=session,
        verbose=verbose,
        workspace_path=workspace_path,
        **runner_kwargs,
    )
    result = predict_calibrated(
        calibrated,
        y_pred=np.asarray(raw_result.y_pred).reshape(-1),
        prediction_sample_ids=_extract_prediction_sample_ids(data),
        result_metadata={
            **dict(raw_result.metadata),
            "conformal_source_bundle": str(model),
            "point_prediction_model_name": raw_result.model_name,
        },
        as_predict_result=True,
    )
    assert isinstance(result, PredictResult)
    result.preprocessing_steps = list(raw_result.preprocessing_steps)

    selected_coverages = _normalize_requested_coverages(coverage)
    assert selected_coverages is not None
    result.intervals = _select_materialized_intervals(result, selected_coverages)
    result.metadata["selected_interval_coverages"] = list(selected_coverages)
    _record_selected_conformal_coverages(result.metadata, selected_coverages)
    return result


def _select_materialized_intervals(result: PredictResult, selected_coverages: tuple[float, ...]) -> dict[float, Any]:
    intervals: dict[float, Any] = {}
    for selected in selected_coverages:
        try:
            intervals[selected] = result.interval(selected)
        except KeyError as exc:
            available = ", ".join(str(value) for value in result.interval_coverages) or "none"
            raise ValueError(f"coverage {selected} was not materialized; available coverages: {available}") from exc
    return intervals


def _normalize_requested_coverages(coverage: float | list[float] | tuple[float, ...] | None) -> tuple[float, ...] | None:
    if coverage is None:
        return None
    if isinstance(coverage, (str, bytes)):
        raise TypeError("coverage must be a float or sequence of floats")
    if isinstance(coverage, (list, tuple)):
        values = tuple(float(value) for value in coverage)
    else:
        values = (float(coverage),)
    if not values:
        raise ValueError("coverage must not be empty")
    if any(not np.isfinite(value) or value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("coverage values must be finite floats strictly between 0 and 1")
    if len(set(values)) != len(values):
        raise ValueError("coverage values must be unique")
    return values


def _record_selected_conformal_coverages(metadata: dict[str, Any], selected_coverages: tuple[float, ...]) -> None:
    status = metadata.get("conformal_guarantee_status")
    if not isinstance(status, Mapping):
        return
    updated = dict(status)
    updated["coverage"] = list(selected_coverages)
    updated["selected_coverages"] = list(selected_coverages)
    metadata["conformal_guarantee_status"] = updated


def _extract_prediction_sample_ids(data: DataSpec) -> Any:
    """Extract explicit physical sample ids for conformal prediction output."""

    if isinstance(data, Mapping):
        sample_ids = data.get("prediction_sample_ids", data.get("sample_ids"))
        if sample_ids is not None:
            return sample_ids
    raise ValueError("conformal bundle prediction requires explicit physical sample ids: pass data={'X': X_new, 'sample_ids': [...]} or data={'y_pred': ..., 'sample_ids': [...]}")


def _predict_from_chain(
    chain_id: str,
    data: DataSpec,
    workspace_path: str | Path | None,
    session: Session | None,
    verbose: int,
    **runner_kwargs: Any,
) -> PredictResult:
    """Replay a stored chain on new data via WorkspaceStore."""
    from nirs4all.pipeline.storage.chain_replay import replay_chain
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    # Resolve workspace path
    if session is not None:
        ws_path = session.runner.workspace_path
    elif workspace_path is not None:
        ws_path = Path(workspace_path)
    else:
        ws_path = Path("workspace")

    # Get X from data
    X = _extract_X(data)

    # Open store, replay, and fetch chain metadata in one session
    store = WorkspaceStore(ws_path)
    try:
        y_pred = replay_chain(store, chain_id=chain_id, X=X)
        chain_info = store.get_chain(chain_id)
        model_name = chain_info.get("model_class", "") if chain_info else ""
        relation_replay_manifest = chain_info.get("relation_replay_manifest") if isinstance(chain_info, Mapping) else None
    finally:
        store.close()

    metadata: dict[str, Any] = {"chain_id": chain_id}
    if isinstance(relation_replay_manifest, Mapping):
        metadata["relation_replay_manifest"] = dict(relation_replay_manifest)

    return PredictResult(
        y_pred=y_pred,
        metadata=metadata,
        model_name=model_name,
        preprocessing_steps=[],
    )


def _predict_from_model(
    model: ModelSpec,
    data: DataSpec,
    name: str,
    all_predictions: bool,
    session: Session | None,
    verbose: int,
    workspace_path: str | Path | None = None,
    **runner_kwargs: Any,
) -> PredictResult:
    """Predict via the model/resolver path."""
    # Use session runner if provided, otherwise create new
    owns_runner = session is None
    if session is not None:
        runner = session.runner
    else:
        all_kwargs: dict[str, Any] = {
            "mode": "predict",
            "verbose": verbose,
            **runner_kwargs,
        }
        if workspace_path is not None:
            all_kwargs["workspace_path"] = workspace_path
        runner = PipelineRunner(**all_kwargs)

    try:
        # Convert Path to str for compatibility with type hints
        model_arg = str(model) if isinstance(model, Path) else model
        data_arg = str(data) if isinstance(data, Path) else data

        # Call the runner's predict method
        y_pred, predictions = runner.predict(
            prediction_obj=model_arg,
            dataset=data_arg,
            dataset_name=name,
            all_predictions=all_predictions,
            verbose=verbose,
        )

        # Extract model info for the result
        model_name = ""
        preprocessing_steps: list[str] = []

        if isinstance(model, dict):
            model_name = model.get("model_name", "")
            raw_pp = model.get("preprocessings", [])
            preprocessing_steps = [raw_pp] if isinstance(raw_pp, str) else list(raw_pp)

        # Handle array output
        if isinstance(y_pred, dict):
            first_key = next(iter(y_pred.keys()), None)
            y_array = y_pred[first_key] if first_key and isinstance(y_pred[first_key], np.ndarray) else np.array([])
            metadata: dict[str, Any] = {"all_folds": y_pred}
        else:
            y_array = y_pred if isinstance(y_pred, np.ndarray) else np.asarray(y_pred)
            metadata = {}

        metadata.update(_relation_metadata_from_predictions(predictions))
        metadata.update(_relation_metadata_from_runner(runner))

        return PredictResult(
            y_pred=y_array,
            metadata=metadata,
            model_name=model_name,
            preprocessing_steps=preprocessing_steps,
        )
    finally:
        if owns_runner:
            runner.close()


def _relation_metadata_from_predictions(predictions: Any) -> dict[str, Any]:
    """Extract relation provenance attached to prediction-row metadata."""
    metadata: dict[str, Any] = {}
    if not hasattr(predictions, "filter_predictions"):
        return metadata
    try:
        rows = predictions.filter_predictions(load_arrays=False)
    except Exception:
        return metadata
    for row in rows:
        row_meta = row.get("metadata") if isinstance(row, Mapping) else None
        if not isinstance(row_meta, Mapping):
            continue
        for key in ("relation_replay_manifest", "relation_materialization_manifest", "feature_lineage", "lineage_warning", "explanation_level"):
            value = row_meta.get(key)
            if isinstance(value, Mapping):
                metadata.setdefault(key, dict(value))
            elif value is not None:
                metadata.setdefault(key, value)
    return metadata


def _relation_metadata_from_runner(runner: PipelineRunner) -> dict[str, Any]:
    """Extract relation replay metadata from the predictor's resolved source."""
    predictor = getattr(runner, "predictor", None)
    resolved = getattr(predictor, "_resolved", None)
    manifest = getattr(resolved, "manifest", None)
    if not isinstance(manifest, Mapping):
        return {}

    relation = manifest.get("relation_replay_manifest")
    if isinstance(relation, Mapping):
        return {"relation_replay_manifest": dict(relation)}
    return {}


def _extract_X(data: DataSpec) -> np.ndarray:
    """Extract feature matrix X from various data formats.

    Args:
        data: Data in any supported format.

    Returns:
        Feature matrix as numpy array.

    Raises:
        TypeError: If data format is not supported for chain replay.
    """
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, tuple):
        return np.asarray(data[0])
    if isinstance(data, dict):
        if "X" in data:
            return np.asarray(data["X"])
        raise TypeError("Dict data must contain an 'X' key for chain replay.")
    if isinstance(data, SpectroDataset):
        return np.asarray(data.x({}))
    raise TypeError(f"Unsupported data type for chain replay: {type(data).__name__}. Pass a numpy array, tuple, dict with 'X' key, or SpectroDataset.")
