"""Invocation-scoped observation of real DAG operator execution.

Observers receive only bounded execution metadata, never spectra or targets.
Progress measures completed cross-validation fits, not elapsed fit time. The unit
is one FIT_CV task of one model node on one fold of one pipeline variant, so a
dispatch owes ``variants x folds x model nodes`` fits. Tuning trials run inside a
model fit and are not counted.

A sequence of model checkpoints runs as one dispatch per model. The caller
announces the dispatch count with :meth:`ExecutionProgress.plan_dispatches`;
without that announcement only the first dispatch has a trustworthy
denominator. Bookkeeping failures degrade the percentage to "unknown" and never
reach the scientific code.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from threading import RLock
from typing import Any


def _telemetry_safe(fallback: Callable[[Any], Any] = lambda _observer: None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Keep observer bookkeeping from ever interrupting the scientific execution."""

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(method)
        def guarded(self: ExecutionProgress, *args: Any, **kwargs: Any) -> Any:
            try:
                return method(self, *args, **kwargs)
            except Exception:
                # Telemetry must never change the scientific outcome; the counters are no longer trustworthy.
                self.reliable_total = False
                return fallback(self)

        return guarded

    return decorate


@dataclass
class ExecutionProgress:
    callback: Callable[[dict[str, Any]], None]
    dataset: str = ""
    total_variants: int = 1
    completed: set[tuple[Any, ...]] = field(default_factory=set)
    variants: dict[tuple[Any, ...], int] = field(default_factory=dict)
    total_folds: int = 0
    model_nodes: int = 0
    bundle: int = 0
    planned_dispatches: int = 0
    dispatches: int = 0
    skipped_dispatches: int = 0
    settled_dispatches: int = 0
    settled_fits: int = 0
    settled_total: int = 0
    peak: float = 0.0
    reliable_total: bool = True
    lock: RLock = field(default_factory=RLock)

    @property
    def dispatch_total(self) -> int:
        """CV fits owed by the current dispatch (0 until its first bundle is opened)."""
        return self.total_variants * self.total_folds * self.model_nodes

    @property
    def total_fits(self) -> int:
        """CV fits owed by the whole invocation; announced dispatches not yet sized are assumed average."""
        sized = self.settled_dispatches + (1 if self.dispatch_total else 0)
        if not sized:
            return 0
        known = self.settled_total + self.dispatch_total
        pending = max(0, self.planned_dispatches - self.skipped_dispatches - sized)
        return known + -(-known * pending // sized)

    @_telemetry_safe()
    def emit(self, message: str, **fields: Any) -> None:
        with self.lock:
            total = self.total_fits
            done = self.settled_fits + len(self.completed)
            progress = None
            if total and self.reliable_total:
                # A bar that moves backwards when a later dispatch turns out larger would read as a defect.
                progress = self.peak = max(self.peak, min(99.0, 100 * done / total))
            snapshot = {"completed_cv_fits": done, "total_cv_fits": total if self.reliable_total else None}
        text = " ".join(str(message).split())[:1200]
        with suppress(Exception):  # a failing observer must never change the scientific outcome
            self.callback({"message": text, "progress": progress, **snapshot, **fields})

    @_telemetry_safe()
    def plan_dispatches(self, count: int) -> None:
        """Announce the single-model dispatches the invocation is about to run.

        An announcement made while an earlier one is still being consumed is a
        nested expansion of work already counted, so it is ignored.
        """
        with self.lock:
            if self.dispatches >= self.planned_dispatches:
                self.planned_dispatches = self.dispatches + max(1, int(count))

    def _settle_dispatch(self) -> None:
        self.settled_fits += len(self.completed)
        if self.dispatch_total:
            self.settled_total += self.dispatch_total
            self.settled_dispatches += 1
        elif self.dispatches:
            # A dispatch that never sized itself (for example the native Methods lane) owes no countable fits.
            self.skipped_dispatches += 1

    @_telemetry_safe()
    def configure(self, variants: int, dataset: str) -> None:
        with self.lock:
            self._settle_dispatch()
            self.completed.clear()
            self.variants.clear()
            self.total_folds = 0
            self.model_nodes = 0
            self.dispatches += 1
            if self.dispatches > self.planned_dispatches:
                # Nobody announced this dispatch, so the invocation-wide denominator is unknown.
                self.planned_dispatches = self.dispatches
                self.reliable_total = self.reliable_total and self.dispatches == 1
            self.dataset = str(dataset)[:200]
            self.total_variants = max(1, variants)
            self.emit(f"{self.dataset}: preparing {self.total_variants} pipeline variants")

    @_telemetry_safe(lambda observer: (observer.bundle, {}))
    def open_bundle(self, dsl: dict[str, Any], graph: dict[str, Any]) -> tuple[int, dict[str, int]]:
        with self.lock:
            self.bundle += 1
            fold_set = (dsl.get("split_invocation") or {}).get("fold_set") or {}
            folds = fold_set.get("folds") or []
            labels = {str(fold["fold_id"]): index + 1 for index, fold in enumerate(folds)}
            if self.total_folds and self.total_folds != len(labels):
                self.reliable_total = False
            self.total_folds = len(labels)
            nodes = graph.get("nodes", [])
            models = sum(1 for node in nodes if node.get("kind") == "model")
            if self.model_nodes and self.model_nodes != models:
                self.reliable_total = False
            self.model_nodes = models
            if any(node.get("kind") == "tuner" for node in nodes):
                self.reliable_total = False
            self.emit(f"{self.dataset}: cross-validation, {self.total_variants} variants, {len(labels)} folds")
            return self.bundle, labels

    @_telemetry_safe()
    def task(self, bundle: int, folds: dict[str, int], task: dict[str, Any], node: dict[str, Any], *, completed: bool) -> None:
        with self.lock:
            phase = str(task.get("phase", ""))
            variant_key = (bundle, str(task.get("variant_id") or "base"))
            variant = self.variants.setdefault(variant_key, len(self.variants) + 1)
            node_id = str(task["node_plan"]["node_id"])
            operator = node.get("operator", node_id)
            if isinstance(operator, dict):
                operator = operator.get("class", node_id)
            operator = str(operator).rsplit(".", 1)[-1][:150]
            fold_id = str(task.get("fold_id") or "")
            model = task["node_plan"].get("kind") == "model"
            if model and phase == "FIT_CV":
                if len(self.completed) > 100_000:
                    self.reliable_total = False
                elif completed:
                    self.completed.add((bundle, node_id, variant_key[1], fold_id))
            if len(self.completed) > self.dispatch_total:
                self.reliable_total = False
            total = self.total_fits
            done = self.settled_fits + len(self.completed)
            context = f"{self.dataset} · "
            if self.planned_dispatches > 1 and self.reliable_total:
                context += f"Model {self.dispatches}/{self.planned_dispatches} · "
            context += f"Variant {variant}/{self.total_variants}"
            if phase == "FIT_CV":
                context += f" · Fold {folds.get(fold_id, fold_id)}/{len(folds)}"
            action = "Refitting selected model" if phase == "REFIT" else "Predicting" if phase == "PREDICT" else "Fitting" if model else "Preprocessing"
            suffix = "completed" if completed else "running"
            counter = f" · CV fits {done}/{total}" if total and self.reliable_total else f" · {done} CV fits completed"
            self.emit(f"{context} · {action}: {operator} ({suffix}){counter}", phase=phase,
                      current_variant=variant, total_variants=self.total_variants, current_fold=folds.get(fold_id),
                      total_folds=len(folds), operator=operator)


_PROGRESS: ContextVar[ExecutionProgress | None] = ContextVar("nirs4all_execution_progress", default=None)


def execution_progress() -> ExecutionProgress | None:
    """Return the current invocation observer, if explicitly installed."""
    return _PROGRESS.get()


@contextmanager
def observe_progress(callback: Callable[[dict[str, Any]], None]) -> Iterator[ExecutionProgress]:
    """Observe the actual synchronous execution without changing scientific inputs."""
    if not callable(callback):
        raise TypeError("progress observer must be callable")
    observer = ExecutionProgress(callback)
    token = _PROGRESS.set(observer)
    try:
        yield observer
    finally:
        _PROGRESS.reset(token)
