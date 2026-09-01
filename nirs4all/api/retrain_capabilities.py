"""API-004 capability decisions for public retraining.

The executable native subset is deliberately narrow: a concrete DAG-ML
``.n4a`` bundle carrying ``train_pipeline.json`` can replay that training spec
through the real ``run(engine="dag-ml")`` adapter.  Core Archive V2 is
prediction-only, Archive V3 is not exposed here, and the native HPO controller
does not implement transfer learning or continuation of an existing artifact.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

from nirs4all.pipeline.dagml.rt import RtError
from nirs4all.pipeline.engine import resolve_engine

RetrainModeName = Literal["full", "transfer", "finetune"]
RetrainLane = Literal["dag-ml", "native", "plugin", "legacy", "refused"]

RETRAIN_CAPABILITIES_V1: dict[str, dict[str, dict[str, Any]]] = {
    "full": {
        "dag-ml": {
            "executable": True,
            "contract": "nirs4all.bundle.train_pipeline.v1+dag-ml.run",
            "capability": "dagml_full_retrain",
        },
        "native": {
            "executable": False,
            "contract": None,
            "capability": "core_archive_v3_retrain",
        },
        "plugin": {
            "executable": False,
            "contract": None,
            "capability": "retrain_plugin",
        },
        "legacy": {
            "executable": True,
            "contract": "nirs4all.python.pipeline_runner.retrain",
            "capability": "legacy_retrain",
        },
    },
    "transfer": {
        "dag-ml": {
            "executable": False,
            "contract": None,
            "capability": "native_transfer_retrain",
        },
        "native": {
            "executable": False,
            "contract": None,
            "capability": "native_transfer_retrain",
        },
        "plugin": {
            "executable": False,
            "contract": None,
            "capability": "retrain_plugin",
        },
        "legacy": {
            "executable": True,
            "contract": "nirs4all.python.pipeline_runner.retrain",
            "capability": "legacy_retrain",
        },
    },
    "finetune": {
        "dag-ml": {
            "executable": False,
            "contract": None,
            "capability": "native_finetune_retrain",
        },
        "native": {
            "executable": False,
            "contract": None,
            "capability": "native_finetune_retrain",
        },
        "plugin": {
            "executable": False,
            "contract": None,
            "capability": "retrain_plugin",
        },
        "legacy": {
            "executable": True,
            "contract": "nirs4all.python.pipeline_runner.retrain",
            "capability": "legacy_retrain",
        },
    },
}


@dataclass(frozen=True)
class RetrainCapabilityDecision:
    """Side-effect-free API-004 routing decision."""

    mode: RetrainModeName
    requested_engine: str | None
    lane: RetrainLane
    executable: bool
    contract: str | None
    plugin: str | None
    unsupported_capability: str | None
    reason: str | None
    mitigation: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-native decision record."""
        return asdict(self)

    def require(self) -> RetrainCapabilityDecision:
        """Return an executable decision or raise the shared runtime error."""
        if self.executable:
            return self
        # Retrain is a specialized training run, so the RT-003 wire verb is
        # ``run`` (the runtime verb vocabulary has no separate retrain token).
        raise RtError(
            "run",
            "unsupported_capability",
            self.reason or f"retrain(mode={self.mode!r}) is unavailable",
            mitigation=self.mitigation,
            unsupported_capability=self.unsupported_capability,
        )


def retrain_capability_ledger() -> dict[str, dict[str, dict[str, Any]]]:
    """Return a detached copy of the API-004 capability ledger."""
    return {
        mode: {lane: dict(record) for lane, record in lanes.items()}
        for mode, lanes in RETRAIN_CAPABILITIES_V1.items()
    }


def preflight_retrain(
    mode: str = "full",
    *,
    engine: str | None = None,
    plugin: str | None = None,
    allow_fallback: bool = False,
    session_present: bool = False,
) -> RetrainCapabilityDecision:
    """Decide retrain routing without inspecting source, data, or Session."""
    if mode not in RETRAIN_CAPABILITIES_V1:
        raise ValueError(
            f"Invalid mode {mode!r}. Must be one of: {set(RETRAIN_CAPABILITIES_V1)}"
        )
    if engine is not None and not isinstance(engine, str):
        raise TypeError("engine must be a string or None")
    if plugin is not None and not isinstance(plugin, str):
        raise TypeError("plugin must be a string or None")
    if not isinstance(allow_fallback, bool):
        raise TypeError("allow_fallback must be a boolean")
    if not isinstance(session_present, bool):
        raise TypeError("session_present must be a boolean")

    mode_name = cast(RetrainModeName, mode)
    plugin_name = plugin if plugin is not None else os.environ.get("N4A_RETRAIN_PLUGIN")
    if plugin_name is not None:
        plugin_name = plugin_name.strip()
        if not plugin_name:
            raise ValueError("plugin must be a non-empty name")

    if allow_fallback:
        return RetrainCapabilityDecision(
            mode=mode_name,
            requested_engine=engine,
            lane="refused",
            executable=False,
            contract=None,
            plugin=plugin_name,
            unsupported_capability="implicit_legacy_fallback",
            reason="retrain does not permit allow_fallback=True",
            mitigation="select engine='legacy' explicitly during the ADR-24 rollback window",
        )

    if plugin_name is not None:
        if engine is not None:
            return RetrainCapabilityDecision(
                mode=mode_name,
                requested_engine=engine,
                lane="refused",
                executable=False,
                contract=None,
                plugin=plugin_name,
                unsupported_capability="retrain_selector_conflict",
                reason=f"retrain received both engine={engine!r} and plugin={plugin_name!r}",
                mitigation="select exactly one execution lane",
            )
        return RetrainCapabilityDecision(
            mode=mode_name,
            requested_engine=None,
            lane="plugin",
            executable=False,
            contract=None,
            plugin=plugin_name,
            unsupported_capability="retrain_plugin",
            reason=f"the explicitly selected retrain plugin {plugin_name!r} has no callable V1 adapter",
            mitigation="install and wire a plugin implementing the retrain V1 contract, or select engine='legacy' explicitly",
        )

    selected_engine = resolve_engine(engine)
    if selected_engine == "legacy":
        record = RETRAIN_CAPABILITIES_V1[mode_name]["legacy"]
        return RetrainCapabilityDecision(
            mode=mode_name,
            requested_engine=selected_engine,
            lane="legacy",
            executable=True,
            contract=str(record["contract"]),
            plugin=None,
            unsupported_capability=None,
            reason=None,
            mitigation=None,
        )

    lane: RetrainLane = "dag-ml" if selected_engine == "dag-ml" else "native"
    record = RETRAIN_CAPABILITIES_V1[mode_name][lane]
    if bool(record["executable"]) and session_present:
        return RetrainCapabilityDecision(
            mode=mode_name,
            requested_engine=selected_engine,
            lane="refused",
            executable=False,
            contract=None,
            plugin=None,
            unsupported_capability="native_retrain_session",
            reason="native full retrain cannot share a legacy PipelineRunner Session",
            mitigation="omit session for the native bundle replay, or select engine='legacy' explicitly",
        )
    if bool(record["executable"]):
        return RetrainCapabilityDecision(
            mode=mode_name,
            requested_engine=selected_engine,
            lane=lane,
            executable=True,
            contract=str(record["contract"]),
            plugin=None,
            unsupported_capability=None,
            reason=None,
            mitigation=None,
        )

    if mode_name == "finetune":
        detail = "the native HPO controller searches/refits a new estimator; it cannot continue an existing artifact"
    elif mode_name == "transfer":
        detail = "no native controller hydrates and freezes existing preprocessing artifacts for transfer"
    else:
        detail = "Core Archive V3 retrain is not exposed by the Python API"
    return RetrainCapabilityDecision(
        mode=mode_name,
        requested_engine=selected_engine,
        lane=lane,
        executable=False,
        contract=None,
        plugin=None,
        unsupported_capability=str(record["capability"]),
        reason=f"engine={selected_engine!r} cannot execute retrain(mode={mode_name!r}): {detail}",
        mitigation="select a callable retrain plugin when available, or select engine='legacy' explicitly",
    )


def require_dagml_retrain_backend() -> None:
    """Probe the real DAG-ML mechanism before reading a source bundle or data."""
    from nirs4all.pipeline.dagml.errors import DagMlUnavailable
    from nirs4all.pipeline.dagml.run_backend import (
        _default_dagml_cli,
        preflight_dagml_backend,
    )

    try:
        preflight_dagml_backend(str(_default_dagml_cli()))
    except DagMlUnavailable as exc:
        raise RtError.from_dagml_error(exc, verb="run") from exc


__all__ = [
    "RETRAIN_CAPABILITIES_V1",
    "RetrainCapabilityDecision",
    "preflight_retrain",
    "require_dagml_retrain_backend",
    "retrain_capability_ledger",
]
