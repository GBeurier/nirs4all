"""Run-local DAG-ML execution resource declarations.

The value is stored in a :class:`contextvars.ContextVar` so concurrent runs do
not communicate through process-global environment variables.  Both the
in-process binding and the subprocess CLI read the same normalized contract.
"""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DagMLExecutionResources:
    """Resources declared for every host task in one DAG-ML run."""

    cpu_threads: int = 1
    gpu_devices: tuple[str, ...] = ()

    def to_contract(self) -> dict[str, Any]:
        """Return the closed DAG-ML ``TrainingResourceLimits`` payload."""

        return {
            "cpu_threads": self.cpu_threads,
            "gpu_devices": list(self.gpu_devices),
        }


_CURRENT_RESOURCES: ContextVar[DagMLExecutionResources | None] = ContextVar(
    "nirs4all_dagml_execution_resources",
    default=None,
)


def normalize_execution_resources(
    cpu_threads: Any = 1,
    gpu_devices: Any = (),
) -> DagMLExecutionResources:
    """Validate and canonicalize public DAG-ML resource options."""

    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    if gpu_devices is None:
        devices: tuple[str, ...] = ()
    elif isinstance(gpu_devices, str):
        devices = (gpu_devices,)
    else:
        try:
            devices = tuple(gpu_devices)
        except TypeError as exc:
            raise TypeError("gpu_devices must be a string or a sequence of strings") from exc
    if any(not isinstance(device, str) or not device.strip() for device in devices):
        raise ValueError("gpu_devices must contain non-empty strings")
    devices = tuple(device.strip() for device in devices)
    if any(re.fullmatch(r"cuda:\d+", device) is None for device in devices):
        raise ValueError("gpu_devices currently supports canonical CUDA identifiers such as 'cuda:0'")
    if len(set(devices)) != len(devices):
        raise ValueError("gpu_devices must not contain duplicates")
    if devices != tuple(sorted(devices)):
        raise ValueError("gpu_devices must be sorted canonically")
    return DagMLExecutionResources(cpu_threads=cpu_threads, gpu_devices=devices)


def bind_execution_resources(resources: DagMLExecutionResources) -> Token[DagMLExecutionResources | None]:
    """Bind resources to the current run context and return its reset token."""

    return _CURRENT_RESOURCES.set(resources)


def reset_execution_resources(token: Token[DagMLExecutionResources | None]) -> None:
    """Restore the prior run-local resource declaration."""

    _CURRENT_RESOURCES.reset(token)


def current_execution_resources() -> DagMLExecutionResources:
    """Return the resources bound to the current DAG-ML run."""

    return _CURRENT_RESOURCES.get() or DagMLExecutionResources()


__all__ = [
    "DagMLExecutionResources",
    "bind_execution_resources",
    "current_execution_resources",
    "normalize_execution_resources",
    "reset_execution_resources",
]
