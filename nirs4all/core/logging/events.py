"""Event types and structured logging for nirs4all.

This module defines event categories and structured event types used
for consistent logging throughout the pipeline execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class Phase(str, Enum):
    """Major workflow phases for high-level tracking."""

    INIT = "init"
    DATA = "data"
    SPLIT = "split"
    GENERATE = "generate"
    BRANCH = "branch"
    SOURCE = "source"
    EVALUATE = "evaluate"
    STACK = "stack"
    TRAIN = "train"
    PREDICT = "predict"
    EXPORT = "export"
    COMPLETE = "complete"


class EventType(str, Enum):
    """Event types within phases."""

    # General events
    START = "start"
    COMPLETE = "complete"
    PROGRESS = "progress"
    ERROR = "error"
    WARNING = "warning"
    SKIP = "skip"

    # Data events
    LOAD = "load"
    VALIDATE = "validate"
    TRANSFORM = "transform"

    # Split events
    BUILD = "build"
    LEAKAGE_CHECK = "leakage_check"

    # Generate events
    EXPAND = "expand"
    PRUNE = "prune"
    DEDUPLICATE = "deduplicate"

    # Branch events
    BRANCH_ENTER = "branch_enter"
    BRANCH_EXIT = "branch_exit"
    BRANCH_COMPARE = "branch_compare"
    STEP = "step"

    # Source events
    SOURCE_PROCESS = "source_process"
    SOURCE_CONCAT = "source_concat"

    # Stack events
    COLLECT = "collect"
    TRAIN_META = "train_meta"

    # Export events
    MODEL = "model"
    REPORT = "report"
    PREDICTIONS = "predictions"
    ARTIFACT = "artifact"

    # Metrics
    METRIC = "metric"

    # Config
    CONFIG_LOAD = "config_load"
    ENVIRONMENT = "environment"


class Status(str, Enum):
    """Status indicators for log messages."""

    STARTING = "starting"
    SUCCESS = "success"
    IN_PROGRESS = "in_progress"
    SKIPPED = "skipped"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class LogEvent:
    """Structured log event for machine-readable logging.

    Attributes:
        timestamp: Event timestamp.
        level: Log level (INFO, DEBUG, WARNING, ERROR).
        phase: Current workflow phase.
        event_type: Type of event within the phase.
        message: Human-readable message.
        run_id: Unique run identifier.
        status: Event status indicator.
        branch_name: Current branch name (if in branch context).
        branch_path: Full branch path for nested branches.
        branch_index: Index of current branch (0-based).
        source_index: Index of current source (for multi-source).
        source_name: Name of current source.
        duration_ms: Duration in milliseconds (for completion events).
        metrics: Dict of metric name to value.
        extra: Additional context fields.
    """

    timestamp: datetime
    level: str
    phase: Optional[Phase]
    event_type: EventType
    message: str
    run_id: Optional[str] = None
    status: Optional[Status] = None
    branch_name: Optional[str] = None
    branch_path: Optional[list[str]] = None
    branch_index: Optional[int] = None
    source_index: Optional[int] = None
    source_name: Optional[str] = None
    duration_ms: Optional[float] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        result = {
            "ts": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
        }

        if self.run_id:
            result["run_id"] = self.run_id
        if self.phase:
            result["phase"] = self.phase.value
        if self.event_type:
            result["event"] = self.event_type.value
        if self.status:
            result["status"] = self.status.value
        if self.branch_name:
            result["branch_name"] = self.branch_name
        if self.branch_path:
            result["branch_path"] = self.branch_path
        if self.branch_index is not None:
            result["branch_index"] = self.branch_index
        if self.source_index is not None:
            result["source_index"] = self.source_index
        if self.source_name:
            result["source_name"] = self.source_name
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.metrics:
            result["metrics"] = self.metrics
        if self.extra:
            result.update(self.extra)

        return result

    def to_json(self) -> str:
        """Convert event to JSON string."""
        import json

        return json.dumps(self.to_dict())
