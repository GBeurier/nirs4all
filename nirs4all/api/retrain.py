"""
Module-level retrain() function for nirs4all.

This module provides a simple interface for retraining nirs4all pipelines
on new data.  Full retrain of a concrete DAG-ML bundle replays its native
training contract; the historical PipelineRunner path requires an explicit
``engine="legacy"`` selection. A canonical Core ``.n4a`` container (or the
:class:`~nirs4all.api.native_n4a.CanonicalN4aRunResult` that produced it) is
retrained only with ``engine="native"``: Core re-runs cross-validation, selection
and refit from the recipe stored in the container on a new IO record.

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

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np

from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.bundle.loader import BundleLoader as HeldBundleLoader
from nirs4all.pipeline.dagml.host_container import N4aSource, open_n4a_source
from nirs4all.pipeline.dagml.rt import RtError

from .result import RunResult
from .retrain_capabilities import (
    PYTHON_LIBRARY_RETRAIN_PLUGIN,
    RetrainCapabilityDecision,
    preflight_retrain,
    require_dagml_retrain_backend,
)
from .session import Session

if TYPE_CHECKING:
    from nirs4all.pipeline.bundle import BundleLoader

_TRAIN_PIPELINE_MEMBER = "train_pipeline.json"
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
            "provide a concrete DAG-ML .n4a bundle containing exactly one "
            "train_pipeline.json specification, or select engine='legacy' explicitly"
        ),
    )


def _bundle_training_spec(source: SourceSpec | BundleLoader, *, opened_source: N4aSource | None = None) -> tuple[Path, list[Any], str, str]:
    """Read the verified native training spec after the backend preflight."""
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
    with open_n4a_source(resolved_path) if opened_source is None else nullcontext(opened_source) as opened:
        if opened.path.resolve() != resolved_path:
            raise ValueError("training source does not match its path")
        container = opened.require_host()
        members = [member for member in container.namelist() if member == _TRAIN_PIPELINE_MEMBER]
        if len(members) != 1:
            raise _native_retrain_request_error(
                f"source bundle must contain exactly one {_TRAIN_PIPELINE_MEMBER}; found {len(members)}",
                capability="dagml_full_retrain_training_spec",
            )
        payload_bytes = container.read(_TRAIN_PIPELINE_MEMBER)
        source_sha256 = container.sha256.removeprefix("sha256:")

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

    return resolved_path, steps, source_sha256, hashlib.sha256(payload_bytes).hexdigest()



def _native_full_retrain(
    source: SourceSpec | BundleLoader,
    data: DataSpec,
    *,
    name: str,
    verbose: int,
    save_artifacts: bool,
    options: dict[str, Any],
    mode: str = "full",
    new_model: Any = None,
    opened_source: N4aSource | None = None,
) -> RunResult:
    """Train a concrete spec or captured winner through real DAG-ML."""
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

    from .general_retrain import captured_training_spec

    captured = captured_training_spec(source, mode=mode, new_model=new_model, opened_source=opened_source)
    if captured is not None:
        steps, lineage = captured
    else:
        if mode == "transfer":
            raise _native_retrain_request_error(
                "transfer requires a recorded DAG workspace predictor or captured-host archive",
                capability="dagml_transfer_retrain_source",
            )
        source_path, steps, source_sha256, spec_sha256 = _bundle_training_spec(source, opened_source=opened_source)
        lineage = {
            "schema_version": 1, "operation": "retrain", "mode": "full", "engine": "dag-ml",
            "source_kind": "n4a_bundle", "source_bundle": source_path.name,
            "source_bundle_sha256": source_sha256, "source_training_spec_sha256": spec_sha256,
        }

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

    lineage["new_artifact_count"] = len(artifacts)
    setattr(result, "_retrain_lineage", lineage)  # noqa: B010 - additive API-004 evidence
    for dataset_result in result.per_dataset.values():
        if isinstance(dataset_result, dict):
            dataset_result["retrain_lineage"] = dict(lineage)
    return result


def retrain(
    source: SourceSpec | BundleLoader,
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
            - Path to a canonical Core ``.n4a`` container, or the
              ``CanonicalN4aRunResult`` returned by
              ``run(..., native_profile="nirs4all.n4a.container.v1")``
              (requires ``engine="native"``)

        data: New dataset to train on. Can be:
            - Path to data folder: ``"new_data/"``
            - Numpy arrays: ``(X, y)``
            - Dict: ``{"X": X, "y": y}``
            - SpectroDataset instance
            - For a canonical ``.n4a`` source: the exact new IO record as a
              path or bytes, or ``{"record": <path or bytes>}``

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
            controls ``engine``, ``plugin``, and ``allow_fallback``. Transfer
            through the bounded Python-library adapter requires
            ``plugin="nirs4all-python-library"`` explicitly. Historical
            rollback execution requires ``engine="legacy"`` explicitly.
            - learning_rate: Learning rate for fine-tuning
            - freeze_layers: List of layers to freeze during fine-tuning
            - step_modes: Per-step mode overrides (advanced)
            - methods_library_path: Methods library for a canonical ``.n4a``
              retrain (the only option that retrain accepts). Other options,
              ``new_model``, ``epochs``, sessions and non-``full`` modes are
              refused before any fit.

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
        >>> # Retrain the training contract stored in a native bundle
        >>> retrained = nirs4all.retrain(
        ...     source="exports/original_model.n4a",
        ...     data=new_train_data,
        ...     mode="full"
        ... )
        >>> print(f"Retrained: {retrained.best_rmse:.4f}")

        Transfer learning with new model:

        >>> from sklearn.ensemble import RandomForestRegressor
        >>>
        >>> result = nirs4all.retrain(
        ...     source="exports/pls_model.n4a",
        ...     data=new_data,
        ...     mode="transfer",
        ...     new_model=RandomForestRegressor(n_estimators=100),
        ...     plugin="nirs4all-python-library"
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

        Retrain a canonical Core ``.n4a`` on a new IO record (fresh full
        retrain from the stored recipe; the source container is unchanged):

        >>> result = nirs4all.retrain(
        ...     "exports/canonical.n4a",
        ...     "new_train_record.json",
        ...     engine="native",
        ... )
        >>> result.export("exports/canonical_retrained.n4a")

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
    # A held loader (for example a loaded Session's) is normalized once to its path and the
    # already open source it borrows, so no lane reopens the container or sees a loader object.
    held_source: N4aSource | None = None
    if isinstance(source, HeldBundleLoader):
        held_source = source._source
        source = source.bundle_path
    decision = retrain_preflight(
        mode,
        engine=requested_engine,
        plugin=requested_plugin,
        allow_fallback=allow_fallback,
        session_present=session is not None,
    ).require()

    if held_source is not None:
        source_context: Any = nullcontext(held_source)
    elif isinstance(source, (str, Path)):
        source_context = open_n4a_source(Path(source))
    else:
        source_context = nullcontext(None)
    with source_context as opened_source:
        from .native_n4a import CanonicalN4aRunResult

        if isinstance(source, CanonicalN4aRunResult) or (opened_source is not None and opened_source.kind == "canonical"):
            # Canonical container: Core owns the single fresh fit. No bundle, V3, runner or
            # legacy fallback exists for it; an unselected or other lane is refused.
            if decision.lane != "native":
                from nirs4all.pipeline.dagml.core_archive_replay import CoreArchiveReplayError

                raise CoreArchiveReplayError("a canonical Core .n4a is retrained only with engine='native'")
            from .native_n4a import retrain_canonical_n4a

            return cast(
                RunResult,
                retrain_canonical_n4a(
                    source, data, name=name, new_model=new_model, epochs=epochs,
                    verbose=verbose, save_artifacts=save_artifacts, options=options, opened_source=opened_source,
                ),
            )

        if (
            opened_source is not None
            and decision.lane in ("legacy", "plugin")
            and (
                opened_source.kind == "native"
                or (opened_source.kind == "none" and opened_source.path.is_file() and opened_source.path.suffix.lower() == ".n4a")
            )
        ):
            raise ValueError("host retrain requires a current host container; re-export old SDK ZIP bundles with current nirs4all")

        if decision.lane == "legacy" and isinstance(session, Session):
            session._prepare_legacy_access("run")

        if decision.lane == "dag-ml":
            if (mode == "full" and new_model is not None) or epochs is not None:
                raise _native_retrain_request_error(
                    "full retrain does not accept new_model; full/transfer retrain do not accept epochs",
                    capability="dagml_full_retrain_option",
                )
            return _native_full_retrain(
                source,
                data,
                name=name,
                verbose=verbose,
                save_artifacts=save_artifacts,
                options=options,
                mode=mode,
                new_model=new_model,
                opened_source=opened_source,
            )

        if decision.lane == "native":
            from .native_archive_training import NativeMethodsArchiveRunResult
            from .native_refit_result import NativeMethodsRefitResult
            from .native_result import NativeMethodsRunResult
            from .native_training import refit_native_methods

            if mode != "full" or new_model is not None or epochs is not None or options:
                raise _native_retrain_request_error(
                    "native Methods V3 full refit accepts only the parent result and new dataset",
                    capability="core_archive_v3_retrain_option",
                )
            if not save_artifacts:
                raise _native_retrain_request_error(
                    "native Methods V3 full refit requires save_artifacts=True",
                    capability="core_archive_v3_retrain_option",
                )
            if not isinstance(source, (NativeMethodsRunResult, NativeMethodsArchiveRunResult)):
                raise _native_retrain_request_error(
                    "native Methods V3 full refit requires a native Methods run result parent",
                    capability="core_archive_v3_retrain_source",
                )
            if not isinstance(data, dict):
                raise _native_retrain_request_error(
                    "native Methods V3 full refit requires a dataset mapping with X, y, and sample_ids",
                    capability="core_archive_v3_retrain_data",
                )
            _ = verbose
            result: NativeMethodsRefitResult = refit_native_methods(source, data, name=name)
            # Preserve the published retrain() signature; the native V3 result is
            # a separate runtime result type exposed through the same entry point.
            return cast(RunResult, result)

        # PipelineRunner is reachable only through the explicit Python-library
        # transfer plugin or the explicitly selected ADR-24 rollback lane.
        if decision.lane not in ("legacy", "plugin"):
            raise AssertionError(f"unexpected executable retrain lane: {decision.lane}")
        if decision.lane == "plugin":
            if decision.plugin != PYTHON_LIBRARY_RETRAIN_PLUGIN or mode != "transfer":
                raise AssertionError(f"unexpected retrain plugin decision: {decision!r}")
            if session is not None:
                raise AssertionError("plugin retrain preflight accepted a Session")
            from nirs4all.pipeline import PipelineRunner

            runner = PipelineRunner(verbose=verbose, save_artifacts=save_artifacts)
        elif session is not None:
            runner = session.runner
        else:
            from nirs4all.pipeline import PipelineRunner

            runner = PipelineRunner(verbose=verbose, save_artifacts=save_artifacts)

        # Convert Path to str for compatibility with type hints
        source_arg: dict[str, Any] | str | BundleLoader = str(source) if isinstance(source, Path) else source
        if opened_source is not None and opened_source.kind == "host":
            from nirs4all.pipeline.bundle import BundleLoader

            assert isinstance(source, (str, Path))
            source_arg = BundleLoader(source, source=opened_source)
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
