"""Invocation-scoped cooperative cancellation at scientific task boundaries."""

from collections.abc import Callable
from contextvars import ContextVar


class DagRunCancelled(RuntimeError):
    """The caller cancelled a DAG run before its next scientific task."""


SHOULD_STOP: ContextVar[Callable[[], bool] | None] = ContextVar("nirs4all_dag_should_stop", default=None)


def check_cancellation() -> None:
    """Never retry or publish a cancelled run as completed."""
    callback = SHOULD_STOP.get()
    if callback is not None and callback():
        raise DagRunCancelled("DAG run cancelled by caller")
