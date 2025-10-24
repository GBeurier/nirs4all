"""
Configuration schemas for workspace architecture.

Pydantic models for run configuration and summary data.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class PipelineEntry(BaseModel):
    """Single pipeline in run (NNNN_hash format)."""
    pipeline_id: str  # e.g., "0001_a1b2c3" or "0001_baseline_a1b2c3"
    config_file: str
    status: str = "pending"
    custom_name: Optional[str] = None  # Custom pipeline name if provided


class RunConfig(BaseModel):
    """Configuration for a run (date_dataset or date_dataset_runname)."""
    dataset_name: str
    created_at: datetime
    created_by: Optional[str] = None
    nirs4all_version: str
    python_version: str
    dataset_source: str
    task_type: str  # 'regression' or 'classification'
    pipelines: List[PipelineEntry]
    global_params: Dict = {}
    description: Optional[str] = None
    custom_run_name: Optional[str] = None  # Custom run name if provided


class RunSummary(BaseModel):
    """Summary generated after run completes."""
    dataset_name: str
    run_date: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    total_pipelines: int
    successful_pipelines: int
    failed_pipelines: int
    best_pipeline: Optional[Dict] = None  # {pipeline_id, metric_name, metric_value}
    statistics: Dict
    errors: List[Dict] = []
