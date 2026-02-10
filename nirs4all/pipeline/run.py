"""
Run entity for nirs4all pipeline execution.

A Run represents a complete experiment session that combines:
- One or more Pipeline Templates (or concrete Pipelines)
- One or more Datasets

The Run generates Results for every combination of expanded pipeline
configurations and datasets.

Formula:
    Run = [Pipeline Templates] × [Datasets]
        = [Σ Expanded Pipelines from all Templates] × [All Datasets]
        = Results
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class RunStatus(Enum):
    """Run execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


# Valid state transitions for the run state machine
VALID_TRANSITIONS = {
    RunStatus.QUEUED: [RunStatus.RUNNING, RunStatus.CANCELLED],
    RunStatus.RUNNING: [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.PAUSED],
    RunStatus.PAUSED: [RunStatus.RUNNING, RunStatus.CANCELLED],
    RunStatus.FAILED: [RunStatus.QUEUED],  # retry
    RunStatus.COMPLETED: [],  # terminal
    RunStatus.CANCELLED: [],  # terminal
}


# Metric metadata for proper score comparison
METRIC_METADATA = {
    # Regression metrics
    "r2": {"higher_is_better": True, "optimal": 1.0, "range": (-float('inf'), 1.0)},
    "rmse": {"higher_is_better": False, "optimal": 0.0, "range": (0.0, float('inf'))},
    "rmsecv": {"higher_is_better": False, "optimal": 0.0, "range": (0.0, float('inf'))},
    "rmsep": {"higher_is_better": False, "optimal": 0.0, "range": (0.0, float('inf'))},
    "mae": {"higher_is_better": False, "optimal": 0.0, "range": (0.0, float('inf'))},
    "mse": {"higher_is_better": False, "optimal": 0.0, "range": (0.0, float('inf'))},
    "mape": {"higher_is_better": False, "optimal": 0.0, "range": (0.0, float('inf'))},
    "rpd": {"higher_is_better": True, "optimal": float('inf'), "range": (0.0, float('inf'))},
    "bias": {"higher_is_better": False, "optimal": 0.0, "range": (-float('inf'), float('inf'))},
    "sep": {"higher_is_better": False, "optimal": 0.0, "range": (0.0, float('inf'))},

    # Classification metrics
    "accuracy": {"higher_is_better": True, "optimal": 1.0, "range": (0.0, 1.0)},
    "precision": {"higher_is_better": True, "optimal": 1.0, "range": (0.0, 1.0)},
    "recall": {"higher_is_better": True, "optimal": 1.0, "range": (0.0, 1.0)},
    "f1": {"higher_is_better": True, "optimal": 1.0, "range": (0.0, 1.0)},
    "f1_score": {"higher_is_better": True, "optimal": 1.0, "range": (0.0, 1.0)},
    "auc": {"higher_is_better": True, "optimal": 1.0, "range": (0.0, 1.0)},
    "roc_auc": {"higher_is_better": True, "optimal": 1.0, "range": (0.0, 1.0)},

    # Default for unknown metrics
    "default": {"higher_is_better": True, "optimal": 1.0, "range": (-float('inf'), float('inf'))},
}


def get_metric_info(metric_name: str) -> Dict[str, Any]:
    """
    Get metadata for a metric.

    Args:
        metric_name: Name of the metric (e.g., 'r2', 'rmse', 'accuracy')

    Returns:
        Dict with 'higher_is_better', 'optimal', and 'range' keys
    """
    metric_lower = metric_name.lower()
    return METRIC_METADATA.get(metric_lower, METRIC_METADATA["default"])


def is_better_score(score: float, best_score: float, metric: str) -> bool:
    """
    Compare two scores and determine if the new score is better.

    Args:
        score: New score to compare
        best_score: Current best score
        metric: Metric name to determine comparison direction

    Returns:
        True if score is better than best_score
    """
    info = get_metric_info(metric)
    if info["higher_is_better"]:
        return score > best_score
    else:
        return score < best_score


@dataclass
class TemplateInfo:
    """Information about a pipeline template in a run."""
    id: str
    name: str
    file_path: Optional[str] = None
    expansion_count: int = 1
    description: Optional[str] = None


@dataclass
class DatasetInfo:
    """Information about a dataset used in a run."""
    name: str
    path: str
    hash: Optional[str] = None
    file_size: Optional[int] = None
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    task_type: Optional[str] = None
    y_columns: Optional[List[str]] = None
    y_stats: Optional[Dict[str, Dict[str, float]]] = None
    wavelength_range: Optional[List[float]] = None
    wavelength_unit: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    version: Optional[str] = None


@dataclass
class RunConfig:
    """Configuration for a run."""
    cv_folds: int = 5
    cv_strategy: str = "kfold"
    random_state: Optional[int] = 42
    metric: str = "r2"
    save_predictions: bool = True
    save_models: bool = True
    project: Optional[str] = None


@dataclass
class RunSummary:
    """Summary of run results."""
    total_results: int = 0
    completed_results: int = 0
    failed_results: int = 0
    best_result: Optional[Dict[str, Any]] = None


@dataclass
class Run:
    """
    Represents a complete experiment session.

    A Run combines pipeline templates with datasets and generates results
    for every combination of expanded pipeline configurations and datasets.

    Attributes:
        id: Unique identifier for the run
        name: Human-readable name
        templates: List of pipeline templates
        datasets: List of datasets
        status: Current execution status
        config: Run configuration
        created_at: Creation timestamp
        started_at: Execution start timestamp
        completed_at: Completion timestamp
        summary: Post-execution summary
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    templates: List[TemplateInfo] = field(default_factory=list)
    datasets: List[DatasetInfo] = field(default_factory=list)
    status: RunStatus = RunStatus.QUEUED
    config: RunConfig = field(default_factory=RunConfig)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    summary: RunSummary = field(default_factory=RunSummary)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_pipeline_configs(self) -> int:
        """Total number of expanded pipeline configurations."""
        return sum(t.expansion_count for t in self.templates)

    @property
    def total_results_expected(self) -> int:
        """Expected number of results (configs × datasets)."""
        return self.total_pipeline_configs * len(self.datasets)

    def can_transition_to(self, new_status: RunStatus) -> bool:
        """Check if transition to new status is valid."""
        return new_status in VALID_TRANSITIONS.get(self.status, [])

    def transition_to(self, new_status: RunStatus) -> None:
        """
        Transition to a new status.

        Raises:
            ValueError: If transition is not valid
        """
        if not self.can_transition_to(new_status):
            raise ValueError(
                f"Invalid transition from {self.status.value} to {new_status.value}. "
                f"Valid transitions: {[s.value for s in VALID_TRANSITIONS.get(self.status, [])]}"
            )

        self.status = new_status

        if new_status == RunStatus.RUNNING and self.started_at is None:
            self.started_at = datetime.now(timezone.utc).isoformat()
        elif new_status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            self.completed_at = datetime.now(timezone.utc).isoformat()

    def add_checkpoint(self, result_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a completed result as a checkpoint."""
        checkpoint = {
            "result_id": result_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            checkpoint.update(metadata)
        self.checkpoints.append(checkpoint)

    def to_dict(self) -> Dict[str, Any]:
        """Convert run to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "templates": [
                {
                    "id": t.id,
                    "name": t.name,
                    "file_path": t.file_path,
                    "expansion_count": t.expansion_count,
                    "description": t.description,
                }
                for t in self.templates
            ],
            "datasets": [
                {
                    "name": d.name,
                    "path": d.path,
                    "hash": d.hash,
                    "file_size": d.file_size,
                    "n_samples": d.n_samples,
                    "n_features": d.n_features,
                    "task_type": d.task_type,
                    "y_columns": d.y_columns,
                    "y_stats": d.y_stats,
                    "wavelength_range": d.wavelength_range,
                    "wavelength_unit": d.wavelength_unit,
                    "metadata": d.metadata,
                    "version": d.version,
                }
                for d in self.datasets
            ],
            "status": self.status.value,
            "config": {
                "cv_folds": self.config.cv_folds,
                "cv_strategy": self.config.cv_strategy,
                "random_state": self.config.random_state,
                "metric": self.config.metric,
                "save_predictions": self.config.save_predictions,
                "save_models": self.config.save_models,
            },
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_pipeline_configs": self.total_pipeline_configs,
            "summary": {
                "total_results": self.summary.total_results,
                "completed_results": self.summary.completed_results,
                "failed_results": self.summary.failed_results,
                "best_result": self.summary.best_result,
            },
            "checkpoints": self.checkpoints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Run":
        """Create run from dictionary."""
        run = cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data.get("name", ""),
            status=RunStatus(data.get("status", "queued")),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            checkpoints=data.get("checkpoints", []),
        )

        # Parse templates
        for t in data.get("templates", []):
            run.templates.append(TemplateInfo(
                id=t.get("id", ""),
                name=t.get("name", ""),
                file_path=t.get("file_path"),
                expansion_count=t.get("expansion_count", 1),
                description=t.get("description"),
            ))

        # Parse datasets
        for d in data.get("datasets", []):
            run.datasets.append(DatasetInfo(
                name=d.get("name", ""),
                path=d.get("path", ""),
                hash=d.get("hash"),
                file_size=d.get("file_size"),
                n_samples=d.get("n_samples"),
                n_features=d.get("n_features"),
                task_type=d.get("task_type"),
                y_columns=d.get("y_columns"),
                y_stats=d.get("y_stats"),
                wavelength_range=d.get("wavelength_range"),
                wavelength_unit=d.get("wavelength_unit"),
                metadata=d.get("metadata"),
                version=d.get("version"),
            ))

        # Parse config
        config_data = data.get("config", {})
        run.config = RunConfig(
            cv_folds=config_data.get("cv_folds", 5),
            cv_strategy=config_data.get("cv_strategy", "kfold"),
            random_state=config_data.get("random_state", 42),
            metric=config_data.get("metric", "r2"),
            save_predictions=config_data.get("save_predictions", True),
            save_models=config_data.get("save_models", True),
        )

        # Parse summary
        summary_data = data.get("summary", {})
        run.summary = RunSummary(
            total_results=summary_data.get("total_results", 0),
            completed_results=summary_data.get("completed_results", 0),
            failed_results=summary_data.get("failed_results", 0),
            best_result=summary_data.get("best_result"),
        )

        return run


def generate_run_id(name: str = "") -> str:
    """
    Generate a unique run ID.

    Format: YYYY-MM-DD_<Name>_<hash>

    Args:
        name: Optional descriptive name

    Returns:
        Unique run ID string
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    hash_str = str(uuid.uuid4())[:6]

    if name:
        # Sanitize name for use in ID
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        safe_name = safe_name[:30]  # Limit length
        return f"{date_str}_{safe_name}_{hash_str}"
    else:
        return f"{date_str}_{hash_str}"
