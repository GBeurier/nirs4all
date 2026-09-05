"""Fail-closed capability decisions for the advanced public API verbs.

Explicit native profiles remain strict. General synthesis selects the built-in
scientific library host before execution: it does not use PipelineRunner or a
legacy ML coordinator. Unknown plugins and execution-error retries are refused.
Explanations use the captured complete predictor without training it again.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

from nirs4all.pipeline.dagml.rt import RtError
from nirs4all.pipeline.engine import ENGINE_ENV_VAR, resolve_engine

AdvancedApiVerb = Literal["explain", "generate"]
AdvancedApiLane = Literal["native", "plugin", "legacy", "refused"]

_PLUGIN_ENV_VARS: dict[AdvancedApiVerb, str] = {
    "explain": "N4A_EXPLAIN_PLUGIN",
    "generate": "N4A_GENERATE_PLUGIN",
}

_BUILTIN_LIBRARY_PLUGINS: dict[AdvancedApiVerb, str] = {
    "generate": "nirs4all.python.synthesis.v1",
    "explain": "nirs4all.python.shap.v1",
}

# API-005 capability ledger.  This describes executable adapters in this
# package, not capabilities inferred from an installed wheel.  A native lane
# becomes available only when nirs4all has a concrete callable adapter to the
# corresponding Core/DAG-ML contract.
ADVANCED_API_CAPABILITIES_V1: dict[str, dict[str, dict[str, Any]]] = {
    "explain": {
        "native": {
            "executable": False,
            "contract": None,
            "capability": "native_explain",
        },
        "plugin": {
            "executable": True,
            "contract": "nirs4all.python.shap.v1",
            "capability": "explain_plugin",
        },
        "legacy": {
            "executable": True,
            "contract": "nirs4all.python.pipeline_runner.shap",
            "capability": "legacy_explain",
        },
    },
    "generate": {
        "native": {
            "executable": False,
            "contract": None,
            "capability": "native_generate",
        },
        "plugin": {
            "executable": True,
            "contract": "nirs4all.python.synthesis.v1",
            "capability": "generate_plugin",
        },
        "legacy": {
            "executable": True,
            "contract": "nirs4all.python.synthesis",
            "capability": "legacy_generate",
        },
    },
}


@dataclass(frozen=True)
class AdvancedApiCapabilityDecision:
    """Side-effect-free API-005 routing decision."""

    verb: AdvancedApiVerb
    requested_engine: str | None
    lane: AdvancedApiLane
    executable: bool
    contract: str | None
    plugin: str | None
    unsupported_capability: str | None
    reason: str | None
    mitigation: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-native capability record for callers and Studio."""
        return asdict(self)

    def require(self) -> AdvancedApiCapabilityDecision:
        """Return this decision when executable, otherwise raise ``RtError``."""
        if self.executable:
            return self
        raise RtError(
            self.verb,
            "unsupported_capability",
            self.reason or f"{self.verb} is not executable on the selected lane",
            mitigation=self.mitigation,
            unsupported_capability=self.unsupported_capability,
        )


def advanced_api_capability_ledger() -> dict[str, dict[str, dict[str, Any]]]:
    """Return a detached copy of the API-005 capability ledger."""
    return {verb: {lane: dict(record) for lane, record in lanes.items()} for verb, lanes in ADVANCED_API_CAPABILITIES_V1.items()}


def preflight_advanced_api(
    verb: AdvancedApiVerb,
    *,
    engine: str | None = None,
    plugin: str | None = None,
    allow_fallback: bool = False,
) -> AdvancedApiCapabilityDecision:
    """Decide one advanced verb without importing its Python implementation.

    Only a named built-in adapter is executable. With no selector, general
    synthesis and explanation choose their installed adapters before touching inputs.
    Explicit native/legacy selectors remain unchanged; errors never retry.
    """
    if verb not in ("explain", "generate"):
        raise ValueError(f"unknown advanced API verb {verb!r}")
    if engine is not None and not isinstance(engine, str):
        raise TypeError("engine must be a string or None")
    if plugin is not None and not isinstance(plugin, str):
        raise TypeError("plugin must be a string or None")
    if not isinstance(allow_fallback, bool):
        raise TypeError("allow_fallback must be a boolean")

    plugin_name = plugin if plugin is not None else os.environ.get(_PLUGIN_ENV_VARS[verb])
    if plugin_name is not None:
        plugin_name = plugin_name.strip()
        if not plugin_name:
            raise ValueError("plugin must be a non-empty name")

    if allow_fallback:
        return AdvancedApiCapabilityDecision(
            verb=verb,
            requested_engine=engine,
            lane="refused",
            executable=False,
            contract=None,
            plugin=plugin_name,
            unsupported_capability="implicit_legacy_fallback",
            reason=f"{verb} does not permit allow_fallback=True; fallback must never select Python implicitly",
            mitigation="select engine='legacy' explicitly during the V1 rollback window",
        )

    if plugin_name is None and engine is None and not os.environ.get(ENGINE_ENV_VAR, "").strip():
        plugin_name = _BUILTIN_LIBRARY_PLUGINS.get(verb)

    if plugin_name is not None:
        if engine is not None:
            return AdvancedApiCapabilityDecision(
                verb=verb,
                requested_engine=engine,
                lane="refused",
                executable=False,
                contract=None,
                plugin=plugin_name,
                unsupported_capability=f"{verb}_selector_conflict",
                reason=f"{verb} received both engine={engine!r} and plugin={plugin_name!r}",
                mitigation="select exactly one execution lane",
            )
        # Even an explicitly named host cannot bypass an ambient forbidden
        # legacy/dual selector at a product boundary.
        resolve_engine(None)
        if plugin_name == _BUILTIN_LIBRARY_PLUGINS.get(verb):
            return AdvancedApiCapabilityDecision(
                verb=verb, requested_engine=None, lane="plugin", executable=True,
                contract=plugin_name, plugin=plugin_name, unsupported_capability=None,
                reason=None, mitigation=None,
            )
        capability = ADVANCED_API_CAPABILITIES_V1[verb]["plugin"]["capability"]
        return AdvancedApiCapabilityDecision(
            verb=verb,
            requested_engine=None,
            lane="plugin",
            executable=False,
            contract=None,
            plugin=plugin_name,
            unsupported_capability=str(capability),
            reason=f"the explicitly selected {verb} plugin {plugin_name!r} has no callable V1 adapter",
            mitigation="install and wire a plugin implementing the published V1 contract, or select engine='legacy' explicitly",
        )

    selected_engine = resolve_engine(engine)
    if selected_engine == "legacy":
        record = ADVANCED_API_CAPABILITIES_V1[verb]["legacy"]
        return AdvancedApiCapabilityDecision(
            verb=verb,
            requested_engine=selected_engine,
            lane="legacy",
            executable=True,
            contract=str(record["contract"]),
            plugin=None,
            unsupported_capability=None,
            reason=None,
            mitigation=None,
        )

    record = ADVANCED_API_CAPABILITIES_V1[verb]["native"]
    return AdvancedApiCapabilityDecision(
        verb=verb,
        requested_engine=selected_engine,
        lane="native",
        executable=False,
        contract=None,
        plugin=None,
        unsupported_capability=str(record["capability"]),
        reason=(f"engine={selected_engine!r} has no callable {verb} contract in the installed nirs4all Core/DAG-ML adapter"),
        mitigation=f"select a supported {verb} plugin when one is installed, or select engine='legacy' explicitly",
    )


__all__ = [
    "ADVANCED_API_CAPABILITIES_V1",
    "AdvancedApiCapabilityDecision",
    "advanced_api_capability_ledger",
    "preflight_advanced_api",
]
