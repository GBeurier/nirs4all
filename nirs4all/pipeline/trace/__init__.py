"""
Execution Trace module for nirs4all pipeline.

This module provides data structures and utilities for recording the exact
execution path through a pipeline, enabling deterministic prediction replay.

Key Components:
    - ExecutionTrace: Complete trace of a pipeline execution path
    - ExecutionStep: Record of a single step's execution
    - StepArtifacts: Artifacts produced by a single step
    - TraceRecorder: Records traces during pipeline execution
    - TraceBasedExtractor: Extracts minimal pipeline from trace (Phase 5)
    - MinimalPipeline: Minimal pipeline ready for prediction replay
    - MinimalPipelineStep: A single step in the minimal pipeline

Design Principles:
    1. Controller-Agnostic: Works with any controller type without hardcoding
    2. Deterministic: Same trace -> same execution path
    3. Minimal: Only records what's needed for replay
    4. Composable: Same infrastructure for predict, retrain, transfer, export

Usage:
    >>> from nirs4all.pipeline.trace import TraceRecorder, ExecutionTrace
    >>>
    >>> # During training
    >>> recorder = TraceRecorder(pipeline_uid="0001_pls_abc123")
    >>> recorder.start_step(step_index=1, operator_type="transform", operator_class="SNV")
    >>> recorder.record_artifact("0001:1:all")
    >>> recorder.end_step()
    >>> trace = recorder.finalize()
    >>>
    >>> # During prediction (loading trace from manifest)
    >>> trace = ExecutionTrace.from_dict(manifest["execution_traces"]["trace_id"])
    >>> artifacts = trace.get_artifacts_by_step(1)
    >>>
    >>> # Extract minimal pipeline for efficient prediction (Phase 5)
    >>> from nirs4all.pipeline.trace import TraceBasedExtractor
    >>> extractor = TraceBasedExtractor()
    >>> minimal = extractor.extract(trace, full_pipeline_steps)
    >>> print(f"Minimal: {minimal.get_step_count()} steps")
"""

from nirs4all.pipeline.trace.execution_trace import (
    ExecutionTrace,
    ExecutionStep,
    StepArtifacts,
    StepExecutionMode,
)
from nirs4all.pipeline.trace.recorder import TraceRecorder
from nirs4all.pipeline.trace.extractor import (
    TraceBasedExtractor,
    MinimalPipeline,
    MinimalPipelineStep,
)


__all__ = [
    "ExecutionTrace",
    "ExecutionStep",
    "StepArtifacts",
    "StepExecutionMode",
    "TraceRecorder",
    "TraceBasedExtractor",
    "MinimalPipeline",
    "MinimalPipelineStep",
]
