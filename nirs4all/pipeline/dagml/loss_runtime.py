"""Process-local execution handoff for DAG-ML training losses."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any


class DagMLTrainingLossExecution:
    """Bind one native ``NodeTask`` loss requirement to its local registry.

    The object is deliberately process-local. It keeps executable state out of
    DAG-ML contracts while allowing backend controllers to invoke the exact
    callback selected for one ``FIT_CV`` or ``REFIT`` task.
    """

    __slots__ = (
        "_attestation",
        "_invocation_count",
        "_invoke",
        "_phase",
        "_required_attestation",
    )

    def __init__(
        self,
        task: Mapping[str, Any],
        registry: Any,
        *,
        role_index: int = 0,
    ) -> None:
        if not isinstance(task, Mapping):
            raise TypeError("DAG-ML training loss task must be a mapping")
        phase = task.get("phase")
        if phase not in {"FIT_CV", "REFIT"}:
            raise ValueError("DAG-ML training losses can execute only in FIT_CV or REFIT")
        bind_training_loss = getattr(registry, "bind_training_loss", None)
        if not callable(bind_training_loss):
            raise TypeError("DAG-ML loss registry must expose bind_training_loss()")
        if isinstance(role_index, bool) or not isinstance(role_index, int) or role_index < 0:
            raise ValueError("DAG-ML training loss role_index must be a non-negative integer")
        expected_attestations = task.get("required_loss_attestations") or ()
        expected_attestation: dict[str, Any] | None = None
        if expected_attestations:
            if not isinstance(expected_attestations, Sequence) or isinstance(
                expected_attestations,
                (str, bytes),
            ):
                raise TypeError("DAG-ML required_loss_attestations must be a sequence")
            if role_index >= len(expected_attestations):
                raise ValueError("DAG-ML training loss role_index is outside required_loss_attestations")
            expected = expected_attestations[role_index]
            if not isinstance(expected, Mapping):
                raise TypeError("DAG-ML required loss attestation must be a mapping")
            expected_attestation = copy.deepcopy(dict(expected))

        binding = bind_training_loss(
            copy.deepcopy(dict(task)),
            role_index=role_index,
        )
        if not isinstance(binding, Mapping):
            raise TypeError("DAG-ML loss registry returned a non-mapping binding")
        invoke = binding.get("invoke")
        required_attestation = binding.get("required_attestation")
        if not callable(invoke) or not isinstance(required_attestation, Mapping):
            raise ValueError("DAG-ML loss binding lacks invoke or required_attestation")
        required_attestation_dict = copy.deepcopy(dict(required_attestation))
        if required_attestation_dict.get("phase") != phase:
            raise ValueError("DAG-ML loss binding attestation phase does not match its task")
        if expected_attestation is not None and required_attestation_dict != expected_attestation:
            raise ValueError("DAG-ML loss binding attestation does not match the NodeTask requirement")

        self._invoke = invoke
        self._phase = str(phase)
        self._required_attestation = required_attestation_dict
        self._attestation: dict[str, Any] | None = None
        self._invocation_count = 0

    @property
    def phase(self) -> str:
        """Return the task phase bound to this execution."""

        return self._phase

    @property
    def invocation_count(self) -> int:
        """Return the number of successful local callback invocations."""

        return self._invocation_count

    @property
    def loss_attestations(self) -> list[dict[str, Any]]:
        """Return the attestation only after at least one successful invocation."""

        if self._attestation is None:
            return []
        return [copy.deepcopy(self._attestation)]

    def __call__(self, target: Any, prediction: Any, *args: Any, **kwargs: Any) -> Any:
        """Invoke the semantic ``(target, prediction)`` DAG-ML loss callback."""

        value = self._invoke(target, prediction, *args, **kwargs)
        self._attestation = copy.deepcopy(self._required_attestation)
        self._invocation_count += 1
        return value

    def invoke_prediction_target(self, prediction: Any, target: Any) -> Any:
        """Adapt a backend ``(prediction, target)`` call to semantic ordering."""

        return self(target, prediction)

    def invoke_target_prediction(self, target: Any, prediction: Any) -> Any:
        """Expose the semantic ordering expected by TensorFlow/Keras."""

        return self(target, prediction)

    def __reduce__(self) -> Any:
        raise TypeError("DAG-ML training loss executions cannot be serialized")


__all__ = ["DagMLTrainingLossExecution"]
