"""
Module-level retrain() function for nirs4all.

This module provides a simple interface for retraining nirs4all pipelines
on new data.  Full retrain of a concrete DAG-ML bundle replays its native
training contract; the historical PipelineRunner path requires an explicit
``engine="legacy"`` selection.

Example:
    >>> import nirs4all
    >>> # Full retrain on new data
    >>> result = nirs4all.retrain(
    ...     source="exports/model.n4a",
    ...     data=new_data,
    ...     mode="full"
    ... )
    >>> print(f"New RMSE: {result.best_rmse:.4f}")
"""

import hashlib
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np

from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.dagml.rt import RtError

from .result import RunResult
from .retrain_capabilities import (
    RetrainCapabilityDecision,
    preflight_retrain,
    require_dagml_retrain_backend,
)
from .session import Session

_TRAIN_PIPELINE_MEMBER = "train_pipeline.json"
_MAX_TRAIN_PIPELINE_BYTES = 1_048_576
_NATIVE_FULL_OPTIONS = frozenset(
    {
        "cache",
        "plots_visible",
        "project",
        "random_state",
        "refit",
        "report_naming",
        "results_path",
        "save_charts",
    }
)

# Type aliases for clarity
SourceSpec: TypeAlias = (
    dict[str, Any]               # Prediction dict from previous run
    | str                          # Path to bundle (.n4a) or config
    | Path                          # Path to bundle or config
)

DataSpec: TypeAlias = (
    str                          # Path to data folder
    | Path                         # Path to data folder
    | np.ndarray                   # X array
    | tuple[np.ndarray, ...]       # (X,) or (X, y)
    | dict[str, Any]               # Dict with X, y keys
    | SpectroDataset               # Direct SpectroDataset instance
    | DatasetConfigs                # Backward compat
)


def retrain_preflight(
    mode: str = "full",
    *,
    engine: str | None = None,
    plugin: str | None = None,
    allow_fallback: bool = False,
    session_present: bool = False,
) -> RetrainCapabilityDecision:
    """Return the side-effect-free API-004 decision for ``retrain``."""
    return preflight_retrain(
        mode,
        engine=engine,
        plugin=plugin,
        allow_fallback=allow_fallback,
        session_present=session_present,
    )


def _native_retrain_request_error(
    message: str,
    *,
    capability: str = "dagml_full_retrain",
) -> RtError:
    return RtError.invalid_request(
        message,
        verb="run",
        unsupported_capability=capability,
        mitigation=(
            "provide a concrete DAG-ML .n4a bundle containing exactly one bounded "
            "train_pipeline.json specification, or select engine='legacy' explicitly"
        ),
    )


def _bundle_training_spec(source: SourceSpec) -> tuple[Path, list[Any], str, str]:
    """Read the bounded native training spec after the backend preflight."""
    if not isinstance(source, (str, Path)):
        raise _native_retrain_request_error(
            "native full retrain requires a .n4a bundle path; in-memory prediction dictionaries have no native training contract",
            capability="dagml_full_retrain_source",
        )

    source_path = Path(source)
    if source_path.is_symlink():
        raise _native_retrain_request_error(
            "native full retrain refuses a symlink source bundle",
            capability="dagml_full_retrain_source",
        )
    try:
        resolved_path = source_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise _native_retrain_request_error(
            f"source bundle does not exist: {source_path}",
            capability="dagml_full_retrain_source",
        ) from exc
    if not resolved_path.is_file() or not zipfile.is_zipfile(resolved_path):
        raise _native_retrain_request_error(
            f"source is not a regular .n4a ZIP bundle: {source_path}",
            capability="dagml_full_retrain_source",
        )

    try:
        with zipfile.ZipFile(resolved_path) as archive:
            members = [info for info in archive.infolist() if info.filename == _TRAIN_PIPELINE_MEMBER]
            if len(members) != 1:
                raise _native_retrain_request_error(
                    f"source bundle must contain exactly one {_TRAIN_PIPELINE_MEMBER}; found {len(members)}",
                    capability="dagml_full_retrain_training_spec",
                )
            member = members[0]
            if member.file_size > _MAX_TRAIN_PIPELINE_BYTES:
                raise _native_retrain_request_error(
                    f"{_TRAIN_PIPELINE_MEMBER} exceeds the {_MAX_TRAIN_PIPELINE_BYTES}-byte limit",
                    capability="dagml_full_retrain_training_spec",
                )
            with archive.open(member) as stream:
                payload_bytes = stream.read(_MAX_TRAIN_PIPELINE_BYTES + 1)
            if len(payload_bytes) > _MAX_TRAIN_PIPELINE_BYTES:
                raise _native_retrain_request_error(
                    f"{_TRAIN_PIPELINE_MEMBER} exceeds the {_MAX_TRAIN_PIPELINE_BYTES}-byte limit",
                    capability="dagml_full_retrain_training_spec",
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise _native_retrain_request_error(
            f"source bundle cannot be read safely: {exc}",
            capability="dagml_full_retrain_source",
        ) from exc

    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _native_retrain_request_error(
            f"{_TRAIN_PIPELINE_MEMBER} is not valid UTF-8 JSON: {exc}",
            capability="dagml_full_retrain_training_spec",
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != {"steps"}:
        raise _native_retrain_request_error(
            f"{_TRAIN_PIPELINE_MEMBER} must be an object containing only 'steps'",
            capability="dagml_full_retrain_training_spec",
        )
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise _native_retrain_request_error(
            f"{_TRAIN_PIPELINE_MEMBER}.steps must be a non-empty list",
            capability="dagml_full_retrain_training_spec",
        )

    bundle_hasher = hashlib.sha256()
    with resolved_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            bundle_hasher.update(chunk)
    return (
        resolved_path,
        steps,
        bundle_hasher.hexdigest(),
        hashlib.sha256(payload_bytes).hexdigest(),
    )


def _native_full_retrain(
    source: SourceSpec,
    data: DataSpec,
    *,
    name: str,
    verbose: int,
    save_artifacts: bool,
    options: dict[str, Any],
) -> RunResult:
    """Replay a concrete bundle training spec through the real DAG-ML run."""
    unknown = sorted(set(options) - _NATIVE_FULL_OPTIONS)
    if unknown:
        raise _native_retrain_request_error(
            f"native full retrain does not support options {unknown}",
            capability="dagml_full_retrain_option",
        )
    if options.get("refit", True) is not True:
        raise _native_retrain_request_error(
            "native full retrain requires refit=True so the execution produces a new artifact",
            capability="dagml_full_retrain_option",
        )
    for option_name in ("cache", "project"):
        if options.get(option_name) is not None:
            raise _native_retrain_request_error(
                f"native full retrain cannot honor {option_name}={options[option_name]!r}",
                capability="dagml_full_retrain_option",
            )

    require_dagml_retrain_backend()
    source_path, steps, source_sha256, spec_sha256 = _bundle_training_spec(source)

    from .run import run

    result = cast(
        RunResult,
        run(
            pipeline=steps,
            dataset=data,
            name=name,
            verbose=verbose,
            save_artifacts=save_artifacts,
            engine="dag-ml",
            allow_fallback=False,
            **options,
        ),
    )
    artifacts = getattr(result, "_dagml_refit_artifacts", None)
    if not isinstance(artifacts, list) or not artifacts:
        result.close()
        raise RtError.runtime_error(
            "native full retrain completed without producing a new refit artifact",
            verb="run",
        )

    lineage: dict[str, Any] = {
        "schema_version": 1,
        "operation": "retrain",
        "mode": "full",
        "engine": "dag-ml",
        "source_kind": "n4a_bundle",
        "source_bundle": source_path.name,
        "source_bundle_sha256": source_sha256,
        "source_training_spec_sha256": spec_sha256,
        "new_artifact_count": len(artifacts),
    }
    setattr(result, "_retrain_lineage", lineage)  # noqa: B010 - additive API-004 evidence
    for dataset_result in result.per_dataset.values():
        if isinstance(dataset_result, dict):
            dataset_result["retrain_lineage"] = dict(lineage)
    return result


def retrain(
    source: SourceSpec,
    data: DataSpec,
    *,
    mode: str = "full",
    name: str = "retrain_dataset",
    new_model: Any | None = None,
    epochs: int | None = None,
    session: Session | None = None,
    verbose: int = 1,
    save_artifacts: bool = True,
    **kwargs: Any
) -> RunResult:
    """Retrain a pipeline on new data.

    This function enables retraining trained pipelines with various modes,
    allowing for full retraining, transfer learning, or fine-tuning.

    Args:
        source: Pipeline source to retrain from. Can be:
            - Prediction dict from ``result.best`` or ``result.top()``
            - Path to exported bundle: ``"exports/model.n4a"``
            - Path to pipeline config directory

        data: New dataset to train on. Can be:
            - Path to data folder: ``"new_data/"``
            - Numpy arrays: ``(X, y)``
            - Dict: ``{"X": X, "y": y}``
            - SpectroDataset instance

        mode: Retrain mode. Options:
            - "full": Train everything from scratch (same pipeline structure)
            - "transfer": Use existing preprocessing, train new model
            - "finetune": Continue training existing model
            Default: "full"

        name: Name for the retrain dataset (for logging).
            Default: "retrain_dataset"

        new_model: Optional new model for transfer mode.
            Replaces the original model while keeping preprocessing.

        epochs: Optional number of epochs for fine-tuning neural networks.

        session: Optional Session for resource reuse.
            If provided, uses the session's runner.

        verbose: Verbosity level (0=quiet, 1=info, 2=debug).
            Default: 1

        save_artifacts: Whether to save retrained artifacts.
            Default: True

        **kwargs: Additional retraining parameters and the frozen API-004
            controls ``engine``, ``plugin``, and ``allow_fallback``.  The
            Python retrainer requires ``engine="legacy"`` explicitly.
            - learning_rate: Learning rate for fine-tuning
            - freeze_layers: List of layers to freeze during fine-tuning
            - step_modes: Per-step mode overrides (advanced)

    Returns:
        RunResult containing:
            - predictions: Predictions from the retrained pipeline
            - per_dataset: Per-dataset execution details
            - best: Best prediction entry
            - best_score: Best model's primary test score

    Raises:
        ValueError: If mode is invalid or source cannot be resolved.
        FileNotFoundError: If source references files that don't exist.

    Examples:
        Full retrain on new data:

        >>> import nirs4all
        >>>
        >>> # Original training
        >>> original = nirs4all.run(pipeline, train_data)
        >>>
        >>> # Retrain on new data with same pipeline
        >>> retrained = nirs4all.retrain(
        ...     source=original.best,
        ...     data=new_train_data,
        ...     mode="full",
        ...     engine="legacy"
        ... )
        >>> print(f"Original: {original.best_rmse:.4f}")
        >>> print(f"Retrained: {retrained.best_rmse:.4f}")

        Transfer learning with new model:

        >>> from sklearn.ensemble import RandomForestRegressor
        >>>
        >>> result = nirs4all.retrain(
        ...     source="exports/pls_model.n4a",
        ...     data=new_data,
        ...     mode="transfer",
        ...     new_model=RandomForestRegressor(n_estimators=100),
        ...     engine="legacy"
        ... )

        Fine-tune a neural network:

        >>> result = nirs4all.retrain(
        ...     source="exports/nn_model.n4a",
        ...     data=new_data,
        ...     mode="finetune",
        ...     epochs=10,
        ...     learning_rate=0.0001,
        ...     engine="legacy"
        ... )

        Retrain from an exported bundle:

        >>> result = nirs4all.retrain(
        ...     source="exports/wheat_model.n4a",
        ...     data="new_wheat_data/",
        ...     mode="full",
        ...     verbose=2
        ... )
        >>> result.export("exports/retrained_model.n4a")

    See Also:
        - :func:`nirs4all.run`: Train a pipeline from scratch
        - :func:`nirs4all.predict`: Make predictions
        - :class:`nirs4all.pipeline.RetrainMode`: Retrain mode enum
    """
    options = dict(kwargs)
    requested_engine = options.pop("engine", None)
    requested_plugin = options.pop("plugin", None)
    allow_fallback = options.pop("allow_fallback", False)
    decision = retrain_preflight(
        mode,
        engine=requested_engine,
        plugin=requested_plugin,
        allow_fallback=allow_fallback,
        session_present=session is not None,
    ).require()

    if decision.lane == "legacy" and isinstance(session, Session):
        session._prepare_legacy_access("run")

    if decision.lane == "dag-ml":
        if new_model is not None or epochs is not None:
            raise _native_retrain_request_error(
                "native full retrain does not accept new_model or epochs",
                capability="dagml_full_retrain_option",
            )
        return _native_full_retrain(
            source,
            data,
            name=name,
            verbose=verbose,
            save_artifacts=save_artifacts,
            options=options,
        )

    # Only the explicitly selected ADR-24 rollback lane reaches PipelineRunner.
    if decision.lane != "legacy":
        raise AssertionError(f"unexpected executable retrain lane: {decision.lane}")
    if session is not None:
        runner = session.runner
    else:
        from nirs4all.pipeline import PipelineRunner

        runner = PipelineRunner(verbose=verbose, save_artifacts=save_artifacts)

    # Convert Path to str for compatibility with type hints
    source_arg = str(source) if isinstance(source, Path) else source
    data_arg = str(data) if isinstance(data, Path) else data

    # Call the runner's retrain method
    predictions, per_dataset = runner.retrain(
        source=source_arg,
        dataset=data_arg,
        mode=mode,
        dataset_name=name,
        new_model=new_model,
        epochs=epochs,
        verbose=verbose,
        **options
    )

    return RunResult(
        predictions=predictions,
        per_dataset=per_dataset,
        _runner=runner,
        _owns_runner=session is None,
    )


# Additive discovery surface without changing the frozen public signature.
retrain.preflight = retrain_preflight  # type: ignore[attr-defined]
